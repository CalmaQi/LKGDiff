import os
import math
import pprint
import ast
import json
import torch
import random
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data as torch_data
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ultra import util
from ultra.stage12 import VQpretrain

from ultra.diffnew1 import TokenDiffusionMR as TokenDiffusion, EMA, SimpleDiscriminator
from torch_geometric.data import Data


def _get_cfg_value(cfg, section_name: str, key: str, default=None):
    section = getattr(cfg, section_name, None)
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def _resolve_relation_map_path(cfg):
    # Allow explicit specification of the mapping file path in the config
    for key in ["relation_map_path", "rel_map_path", "wiki_relation_map_path"]:
        value = getattr(cfg, key, None)
        if isinstance(value, str) and value:
            return value

    for section in ["train", "diffusion", "dataset", "data"]:
        for key in ["relation_map_path", "rel_map_path", "wiki_relation_map_path"]:
            value = _get_cfg_value(cfg, section, key, None)
            if isinstance(value, str) and value:
                return value

    # Common default paths (infer workspace root directory relative to the current script)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates = [
        os.path.join(base_dir, "exp1", "Graph-DiT", "wiki143", "0.8_r.txt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _load_relation_correspondence(path: str):
    """
    Read 0.8_r.txt, format example:
      4\t[4, 89]
    Returns: Dict[int, Set[int]]
    """
    mapping = {}
    if not path or (not os.path.exists(path)):
        return mapping

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            k_str, v_str = parts[0].strip(), parts[1].strip()
            try:
                k = int(k_str)
                v = ast.literal_eval(v_str)
                if isinstance(v, (list, tuple, set)):
                    rels = {int(x) for x in v}
                else:
                    rels = {int(v)}
                rels.add(k)
                mapping[k] = rels
            except Exception:
                continue
    return mapping


def _relations_correspond(r_exist: int, r_replace: int, rel_map: dict) -> bool:
    if r_exist == r_replace:
        return True
    s_exist = rel_map.get(r_exist, {r_exist})
    s_replace = rel_map.get(r_replace, {r_replace})
    return (r_replace in s_exist) or (r_exist in s_replace)


def _build_triple_set_and_pair_rel_map(data: Data):
    """
    Build based on target_* edges (including original and inverse edges):
      - triples: {(h,r,t), (t,r_rev,h)}
      - ht2rels: {(h,t): {r1,r2,...}, (t,h): {r_rev1,r_rev2,...}}
    Inverse relation rule:
      r_rev = r + rel_offset, rel_offset = max(R) + 1
    """
    ei = getattr(data, "target_edge_index", None)
    et = getattr(data, "target_edge_type", None)
    triples = set()
    ht2rels = {}
    if ei is None:
        return triples, ht2rels

    H = ei[0].detach().cpu().tolist()
    T = ei[1].detach().cpu().tolist()
    if et is None:
        R = [0] * len(H)
    else:
        R = et.detach().cpu().tolist()

    rel_offset = (max(R) + 1) if len(R) > 0 else 1

    for h, r, t in zip(H, R, T):
        h = int(h); r = int(r); t = int(t)

        # Original edge
        triples.add((h, r, t))
        key_fwd = (h, t)
        if key_fwd not in ht2rels:
            ht2rels[key_fwd] = set()
        ht2rels[key_fwd].add(r)

        # Inverse edge
        r_rev = r + rel_offset
        triples.add((t, r_rev, h))
        key_rev = (t, h)
        if key_rev not in ht2rels:
            ht2rels[key_rev] = set()
        ht2rels[key_rev].add(r_rev)

    return triples, ht2rels

def build_joint_target_graph(train_data: Data, test_data: Data, device=None, reverse=True) -> Data:
    """
    Construct joint graph (only target_* edges):
    - First concatenate train/test target edges;
    - Optionally append corresponding inverse edges (relation id increases by offset).
    """
    device = device or getattr(train_data.edge_index, "device", None)

    def _get_target_edges(data: Data):
        ei = getattr(data, "target_edge_index", None)
        et = getattr(data, "target_edge_type", None)
        return ei, et

    train_ei, train_et = _get_target_edges(train_data)
    test_ei, test_et = _get_target_edges(test_data)

    base_edge_index = torch.cat([train_ei, test_ei], dim=1)
    if train_et is not None and test_et is not None:
        base_edge_type = torch.cat([train_et, test_et], dim=0)
    else:
        base_edge_type = None

    if reverse:
        reverse_edge_index = torch.stack([base_edge_index[1], base_edge_index[0]], dim=0)
        if base_edge_type is not None:
            rel_offset = int(base_edge_type.max().item()) + 1
            reverse_edge_type = base_edge_type + rel_offset
            edge_type = torch.cat([base_edge_type, reverse_edge_type], dim=0)
        else:
            edge_type = None
        edge_index = torch.cat([base_edge_index, reverse_edge_index], dim=1)
    else:
        edge_index = base_edge_index
        edge_type = base_edge_type

    num_nodes = max(int(train_data.num_nodes), int(test_data.num_nodes))
    joint = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_nodes,
                 target_edge_index=base_edge_index, target_edge_type=base_edge_type)
    if edge_type is not None:
        joint.num_relations = int(edge_type.max().item()) + 1
    return joint


def _get_target_edges(data: Data):
    ei = getattr(data, "target_edge_index", None)
    et = getattr(data, "target_edge_type", None)
    return ei, et


def _build_graph_from_indices(
    template: Data,
    keep_idx: torch.Tensor,
) -> Data:
    """
    Select keep_idx from template's target_* edges to construct graph; optionally add inverse edges.
    reverse_mode:
    """
    ei_all, et_all = _get_target_edges(template)
    ei = ei_all[:, keep_idx]
    et = et_all[keep_idx] if et_all is not None else None

    rev_ei = torch.stack([ei[1], ei[0]], dim=0)
    base_num_rel =  int(et_all.max().item()) + 1
    rev_et = et + base_num_rel
        
    ei = torch.cat([ei, rev_ei], dim=1)
    et = torch.cat([et, rev_et], dim=0) if et is not None else None

    g = Data(edge_index=ei, edge_type=et, num_nodes=template.num_nodes)
    g.num_relations = base_num_rel * 2
    return g


def split_uncomplete_complete(train_data: Data, ratio_target_range=(0.7, 1.0),
                              ratio_observed: float = 0.8, device=None):
    """
    Sample a target graph and then an observed subset, stratified by relation:
      - completekg: retain rho_tar of each relation, rho_tar~Uniform(ratio_target_range)
      - uncompletekg: retain rho_obs of the target triples
    Returns: uncompletekg, completekg (Data)
    """
    device = device or getattr(train_data.edge_index, "device", None)
    ei, et = _get_target_edges(train_data)
    E = ei.size(1)
    ratio_target = random.uniform(*ratio_target_range)

    keep_u_list, keep_c_list = [], []
    uniq = torch.unique(et)
    for r in uniq.tolist():
        idx = (et == r).nonzero(as_tuple=False).view(-1)
        nr = idx.numel()
        if nr == 0:
            continue
        k_c = max(0, min(nr, int(math.floor(nr * ratio_target))))
        perm = idx[torch.randperm(nr, device=idx.device)]
        sel_c = perm[:k_c]
        k_u = max(0, min(k_c, int(math.floor(k_c * ratio_observed))))
        sel_u = sel_c[:k_u]
        keep_u_list.append(sel_u)
        keep_c_list.append(sel_c)

    keep_u = torch.unique(torch.cat(keep_u_list, dim=0)) if keep_u_list else torch.arange(0, E, device=device)
    keep_c = torch.unique(torch.cat(keep_c_list, dim=0)) if keep_c_list else keep_u
    uncompletekg = _build_graph_from_indices(train_data, keep_u)
    completekg   = _build_graph_from_indices(train_data, keep_c)
    return uncompletekg, completekg


@torch.no_grad()
def analyze_chunk_embeddings(
    x0_chunk: torch.Tensor,
    known_x0_chunk: torch.Tensor,
    completekg: Data,
    uncompletekg: Data,
    head_start: int,
    logger,
    num_samples: int = 200,
):
    """Analyze differences between x0_chunk and known_x0_chunk on (h,r) dimension for denoising network design.

    x0_chunk / known_x0_chunk: (M,R,D), corresponding to global head index range [head_start, head_start+M)
    completekg / uncompletekg: complete / incomplete graph for current epoch (edge_index / edge_type).
    """
    M, num_rel, dim = x0_chunk.shape
    head_end = head_start + M

    def _build_hr2tails(g: Data, lo: int, hi: int):
        ei = g.edge_index
        et = g.edge_type
        mask = (ei[0] >= lo) & (ei[0] < hi)
        if mask.sum() == 0:
            return {}
        idx = mask.nonzero(as_tuple=False).view(-1)
        h = ei[0][idx].long().cpu().tolist()
        r = et[idx].long().cpu().tolist()
        t = ei[1][idx].long().cpu().tolist()
        hr2tails = {}
        for h_i, r_i, t_i in zip(h, r, t):
            key = (int(h_i), int(r_i))
            if key not in hr2tails:
                hr2tails[key] = set()
            hr2tails[key].add(int(t_i))
        return hr2tails

    hr2tails_comp = _build_hr2tails(completekg, head_start, head_end)
    hr2tails_un   = _build_hr2tails(uncompletekg, head_start, head_end)

    hr_comp = set(hr2tails_comp.keys())
    hr_un   = set(hr2tails_un.keys())

    common_pairs = list(hr_comp & hr_un)
    dropped_pairs = list(hr_comp - hr_un)
    un_only_pairs = list(hr_un - hr_comp)

    def _sample_and_stats(pairs, label: str):
        if not pairs:
            logger.warning(f"[emb_analyze] {label}: no (h,r) pairs in this chunk.")
            return

        k = min(num_samples, len(pairs))
        sample_pairs = random.sample(pairs, k)

        l2_diffs = []
        cos_sims  = []

        for h, r in sample_pairs:
            local_h = h - head_start
            if not (0 <= local_h < M):
                continue
            if not (0 <= r < num_rel):
                continue

            v_full = x0_chunk[local_h, r].detach()
            v_incomp = known_x0_chunk[local_h, r].detach()
            diff_vec = v_full - v_incomp
            l2 = diff_vec.norm(p=2).item()
            l2_diffs.append(l2)

            n1 = v_full.norm(p=2) + 1e-8
            n2 = v_incomp.norm(p=2) + 1e-8
            cos = torch.dot(v_full, v_incomp) / (n1 * n2)
            cos_sims.append(float(cos.item()))

        # Print differences in t for specific (h,r) pairs to help understand structural changes
        max_print = min(10, len(sample_pairs))
        for i in range(max_print):
            h, r = sample_pairs[i]
            tails_c = sorted(list(hr2tails_comp.get((h, r), set())))
            tails_u = sorted(list(hr2tails_un.get((h, r), set())))
            only_c = sorted(list(set(tails_c) - set(tails_u)))
            only_u = sorted(list(set(tails_u) - set(tails_c)))

            def _fmt(ts, max_len=10):
                if not ts:
                    return "[]"
                if len(ts) > max_len:
                    return f"{ts[:max_len]}...(+{len(ts)-max_len} more)"
                return str(ts)

            logger.warning(
                f"[emb_analyze] {label} sample#{i}: h={h}, r={r}, "
                f"t_complete={_fmt(tails_c)}, t_uncomplete={_fmt(tails_u)}, "
                f"only_in_complete={_fmt(only_c)}, only_in_uncomplete={_fmt(only_u)}"
            )

        if l2_diffs:
            mean_l2 = float(np.mean(l2_diffs))
            std_l2  = float(np.std(l2_diffs))
            mean_cos = float(np.mean(cos_sims))
            std_cos  = float(np.std(cos_sims))
            logger.warning(
                f"[emb_analyze] {label}: n={len(l2_diffs)}  "
                f"L2 mean={mean_l2:.4f} std={std_l2:.4f}  "
                f"cos mean={mean_cos:.4f} std={std_cos:.4f}"
            )
        else:
            logger.warning(f"[emb_analyze] {label}: sampled pairs but no valid indices.")

    _sample_and_stats(common_pairs,  "common(h,r) in both graphs")
    _sample_and_stats(dropped_pairs, "dropped(h,r): only in completekg")
    _sample_and_stats(un_only_pairs, "only_in_uncomplete(h,r)")


@torch.no_grad()
def build_labels_for_heads(data: Data, batch_heads: torch.Tensor, num_rel: int, num_nodes: int, has_empty: bool, device):
    B = batch_heads.numel()
    C = num_nodes + 1 if has_empty else num_nodes
    labels = torch.zeros((B, num_rel, C), device=device)
    H_all = data.edge_index[0].to(device)
    T_all = data.edge_index[1].to(device)
    R_all = getattr(data, "edge_type", None)
    if R_all is not None:
        R_all = R_all.to(device)
    head2b = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
    head2b[batch_heads] = torch.arange(B, device=device, dtype=torch.long)
    mask = head2b[H_all] >= 0
    if mask.any():
        b_idx = head2b[H_all[mask]]
        t_idx = T_all[mask]
        r_idx = R_all[mask] if R_all is not None else torch.zeros_like(t_idx)
        labels[b_idx, r_idx, t_idx] = 1.0
    if has_empty:
        has_pos = labels[:, :, :num_nodes].any(dim=-1)
        labels[:, :, num_nodes] = (~has_pos).float()
    return labels


@torch.no_grad()
def pool_graph_ctx(rel_tokens: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    # rel_tokens: (B,R,D) -> (B,D)
    if mode == "mean":
        return rel_tokens.mean(dim=1)
    elif mode == "sum":
        return rel_tokens.sum(dim=1)
    else:
        return rel_tokens.mean(dim=1)


@torch.no_grad()
def get_tokens(vq_model: VQpretrain, graph: Data, batch: torch.Tensor, use_soft_vq: bool = True) -> torch.Tensor:
    # Return (B,R,rel_dim) tokens; softVQ always exists, so go through quantizer first
    rel_vec = vq_model.context_embed(graph, batch)  # (B,R,D)
    return rel_vec

@torch.no_grad()
def _build_forbidden_mask_for_heads(data: Data, valid_data, batch_heads: torch.Tensor, num_rel: int,
                                    num_nodes: int, device) -> torch.Tensor:
    """Mask known triples, optionally including an additional validation graph."""
    B = batch_heads.numel()
    mask = torch.zeros((B, num_rel, num_nodes), device=device, dtype=torch.bool)

    joint_data = data if valid_data is None else build_joint_target_graph(data, valid_data, device=device)

    H_all = joint_data.edge_index[0].to(device)
    T_all = joint_data.edge_index[1].to(device)
    R_all = getattr(joint_data, "edge_type", None)
    if R_all is not None:
        R_all = R_all.to(device)
    else:
        R_all = torch.zeros_like(T_all, device=device)

    head2b = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
    head2b[batch_heads] = torch.arange(B, device=device)
    sel = head2b[H_all] >= 0
    if sel.any():
        b = head2b[H_all[sel]]
        r = R_all[sel].clamp_min(0).clamp_max(num_rel - 1)
        t = T_all[sel].clamp_min(0).clamp_max(num_nodes - 1)
        mask[b, r, t] = True
    return mask


def _merge_pair_rel_maps(*maps):
    merged = {}
    for pair_map in maps:
        for key, rels in pair_map.items():
            merged.setdefault(key, set()).update(rels)
    return merged


def _count_powa_negative(h, r, t, pair_rel_map, rel_map):
    count = 0
    for h_idx, r_pred, t_idx in zip(h, r, t):
        rels_exist = pair_rel_map.get((int(h_idx), int(t_idx)))
        if rels_exist and any(
            not _relations_correspond(r_exist, int(r_pred), rel_map) for r_exist in rels_exist
        ):
            count += 1
    return count


def _compute_tsp_metrics(true_positive, pred_total, label_total, powa_negative=None):
    cwa_precision = true_positive / max(1, pred_total)
    if powa_negative is None:
        jprecision = cwa_precision
    else:
        powa_precision = true_positive / max(1, true_positive + powa_negative)
        jprecision = 0.5 * (cwa_precision + powa_precision)
    st_recall = (true_positive / max(1, label_total)) ** 0.5
    f_tsp = 2 * jprecision * st_recall / max(1e-8, jprecision + st_recall)
    return jprecision, st_recall, f_tsp

@torch.no_grad()
def precompute_tokens_chunked(
    vq_model: VQpretrain,
    graph: Data,
    device,
    num_nodes: int,
    num_rel: int,
    rel_dim: int,
    chunk_size: int = 2048,
    start: int = 0,
    end: int = None,
) -> torch.Tensor:
    """
    Compute whole graph head representation in chunks, returns (M,R,D), M=end-start.
    Only forward VQ, avoid memory peak from putting all heads at once.
    """
    end = num_nodes if end is None else end
    M = max(0, end - start)
    out = torch.empty((M, num_rel, rel_dim), device=device)
    heads_all = torch.arange(start, end, device=device)
    for s in range(0, M, chunk_size):
        e = min(M, s + chunk_size)
        h = heads_all[s:e]
        out[s:e] = vq_model.context_embed(graph, h)  # (b,R,D)
    return out



@torch.no_grad()
def evaluate(cfg, vq_model: VQpretrain, diff_model: TokenDiffusion, data_incomplete: Data, data_complete: Data,
             device, logger, test_data,valid_data, writer: SummaryWriter = None, step: int = 0, guidance_scale: float = 0,
             use_soft_vq: bool = True,tb_tag: str = "difftest", return_details: bool = False):
    """Select a threshold on the configured split; report the source with test metrics."""
    vq_model.eval(); diff_model.eval()
    num_nodes = int(data_complete.num_nodes)
    num_rel = int(vq_model.vq_vae.num_rel)
    rel_dim = int(getattr(vq_model.vq_vae, "rel_dim", 64))
    has_empty = False
    vq_chunk = int(getattr(cfg.train, "vq_chunk", 2048))

    assumption = str(_get_cfg_value(cfg, "evaluation", "assumption", "cwa")).lower()
    if assumption not in ("cwa", "rs_powa"):
        raise ValueError(f"Unsupported evaluation assumption: {assumption}")
    use_powa = assumption == "rs_powa"
    threshold_split = str(_get_cfg_value(cfg, "evaluation", "threshold_split", "validation")).lower()
    if threshold_split == "valid":
        threshold_split = "validation"
    if threshold_split not in ("validation", "test"):
        raise ValueError(f"Unsupported threshold_split: {threshold_split}")

    def _make_label_graph(data):
        reverse_edge_index = torch.stack([data.target_edge_index[1], data.target_edge_index[0]], dim=0)
        reverse_edge_type = data.target_edge_type + int(data.target_edge_type.max().item()) + 1
        label_edge_index = torch.cat([data.target_edge_index, reverse_edge_index], dim=1)
        label_edge_type = torch.cat([data.target_edge_type, reverse_edge_type], dim=0)
        graph = Data(edge_index=label_edge_index, edge_type=label_edge_type, num_nodes=data.num_nodes)
        graph.num_relations = getattr(data, "num_relations", getattr(data_incomplete, "num_relations", 0))
        return graph

    valid_label_graph = _make_label_graph(valid_data)
    label_graph = _make_label_graph(test_data)
    threshold_label_graph = label_graph if threshold_split == "test" else valid_label_graph
    threshold_known_data = valid_data if threshold_split == "test" else None

    test_triples, test_ht2rels = _build_triple_set_and_pair_rel_map(test_data)
    _, train_ht2rels = _build_triple_set_and_pair_rel_map(data_incomplete)
    _, valid_ht2rels = _build_triple_set_and_pair_rel_map(valid_data)
    valid_pair_rel_map = _merge_pair_rel_maps(train_ht2rels, valid_ht2rels)
    test_pair_rel_map = _merge_pair_rel_maps(train_ht2rels, test_ht2rels)

    rel_map_path = _resolve_relation_map_path(cfg) if use_powa else None
    rel_map = _load_relation_correspondence(rel_map_path)
    if use_powa:
        if rel_map_path is None:
            logger.warning(f"[{tb_tag}] relation map file not found; fallback to exact-relation matching only.")
        else:
            logger.warning(f"[{tb_tag}] relation map loaded: {rel_map_path}")


    # First accumulate known_x0_all / known_mask_all in chunks
    known_x0_all = precompute_tokens_chunked(
        vq_model, data_incomplete, device, num_nodes, num_rel, rel_dim, chunk_size=vq_chunk
    )  # (N,R,D)
    
    # Diffusion samples tokens by entity
    z_tokens_all = torch.empty((1, num_nodes, num_rel, rel_dim), device=device)
    known_x0_batched = known_x0_all.unsqueeze(0)  # (1,N,R,D)
    for s in range(0, num_nodes, vq_chunk):
        e = min(num_nodes, s + vq_chunk)
        heads = torch.arange(s, e, device=device)          # (H,)
        z_tokens_all[:, heads] = diff_model.sample(known_x0=known_x0_batched,
                                                   incomplete_graph=data_incomplete, heads=heads, 
                                                   soft_vq=getattr(vq_model.vq_vae, "rvq", None) if use_soft_vq else None,)
        
    thr_grid = torch.linspace(0.2, 0.9999999, steps=int(_get_cfg_value(cfg, "evaluation", "threshold_steps", 10)), device=device)
    sweep_tp = torch.zeros_like(thr_grid, dtype=torch.long)
    sweep_fp = torch.zeros_like(thr_grid, dtype=torch.long)
    sweep_fn = torch.zeros_like(thr_grid, dtype=torch.long)
    sweep_powa_negative = torch.zeros_like(thr_grid, dtype=torch.long)
    cached_tail_prob_cpu = []
    cached_ranges = []

    logits_cpu = torch.empty((num_nodes, num_rel, num_nodes), dtype=torch.float32, device="cpu")
    for s in range(0, num_nodes, vq_chunk):
        e = min(num_nodes, s + vq_chunk)
        heads = torch.arange(s, e, device=device)
        logits_chunk = vq_model.logits_from_head(z_tokens_all.squeeze(0)[s:e])
        logits_cpu[heads.detach().cpu()] = logits_chunk.detach().float().cpu()

    tail_prob_cpu = torch.sigmoid(logits_cpu[..., :num_nodes])

    rel_offset = num_rel // 2
    if rel_offset > 0 and num_rel >= rel_offset * 2:
        r0 = torch.arange(rel_offset)
        r1 = r0 + rel_offset
        A = tail_prob_cpu[:, r0, :]
        B = tail_prob_cpu[:, r1, :]
        B_t = B.transpose(0, 2)
        avg = 0.5 * (A + B_t)
        tail_prob_cpu[:, r0, :] = avg
        tail_prob_cpu[:, r1, :] = avg.transpose(0, 2)

    for s in range(0, num_nodes, vq_chunk):
        e = min(num_nodes, s + vq_chunk)
        heads = torch.arange(s, e, device=torch.device("cpu"))
        valid_labels = build_labels_for_heads(
            threshold_label_graph, heads, num_rel, num_nodes, has_empty, device=torch.device("cpu")
        )
        valid_forbid = _build_forbidden_mask_for_heads(
            data_incomplete, threshold_known_data, heads, num_rel, num_nodes, device=device
        )
        valid_prob = tail_prob_cpu[heads, :num_nodes].to(device).masked_fill(valid_forbid, 0.0)
        valid_gt = valid_labels[..., :num_nodes].bool().to(device)

        for i, thr in enumerate(thr_grid):
            preds = valid_prob > thr
            sweep_tp[i] += (preds & valid_gt).sum()
            sweep_fp[i] += (preds & (~valid_gt)).sum()
            sweep_fn[i] += ((~preds) & valid_gt).sum()
            if use_powa:
                wrong = (preds & (~valid_gt)).nonzero(as_tuple=False).detach().cpu()
                if wrong.numel() > 0:
                    h = (wrong[:, 0] + s).numpy()
                    r = wrong[:, 1].numpy()
                    t = wrong[:, 2].numpy()
                    sweep_powa_negative[i] += _count_powa_negative(h, r, t, test_pair_rel_map if threshold_split == "test" else valid_pair_rel_map, rel_map)

        test_forbid = _build_forbidden_mask_for_heads(
            data_incomplete, valid_data, heads, num_rel, num_nodes, device=device
        )
        test_prob = tail_prob_cpu[heads, :num_nodes].to(device).masked_fill(test_forbid, 0.0)
        cached_tail_prob_cpu.append(test_prob.detach().to(device="cpu", dtype=torch.float32))
        cached_ranges.append((s, e))

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sweep_tp, op=dist.ReduceOp.SUM)
        dist.all_reduce(sweep_fp, op=dist.ReduceOp.SUM)
        dist.all_reduce(sweep_fn, op=dist.ReduceOp.SUM)
        dist.all_reduce(sweep_powa_negative, op=dist.ReduceOp.SUM)

    val_f1s, val_ps, val_rs = [], [], []
    for i in range(thr_grid.numel()):
        tp = int(sweep_tp[i].item())
        fp = int(sweep_fp[i].item())
        fn = int(sweep_fn[i].item())
        powa_negative = int(sweep_powa_negative[i].item()) if use_powa else None
        p, r_, f1 = _compute_tsp_metrics(tp, tp + fp, tp + fn, powa_negative)
        val_f1s.append(f1)
        val_ps.append(p)
        val_rs.append(r_)
    best_i = int(torch.tensor(val_f1s).argmax().item())
    best_thr = float(thr_grid[best_i].item())

    def _edges_to_sorted_keys(edge_index: torch.Tensor, edge_type: torch.Tensor,
                              num_nodes: int, num_rel: int) -> np.ndarray:
        h = edge_index[0].detach().to("cpu", torch.int64).numpy()
        t = edge_index[1].detach().to("cpu", torch.int64).numpy()
        r = edge_type.detach().to("cpu", torch.int64).numpy()
        keys = (h * np.int64(num_rel) + r) * np.int64(num_nodes) + t
        keys = np.unique(keys)
        keys.sort()
        return keys
    
    # test keys: test + reverse (already built with label_graph)
    test_keys_sorted = _edges_to_sorted_keys(
        label_graph.edge_index, label_graph.edge_type, num_nodes, num_rel
    )
    
    # forbid keys: train + valid + reverse (for explicit exclusion)
    
    def _isin_sorted(keys: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
        if sorted_set.size == 0 or keys.size == 0:
            return np.zeros((keys.size,), dtype=bool)
        pos = np.searchsorted(sorted_set, keys)
        hit = np.zeros(keys.shape, dtype=bool)
        valid = pos < sorted_set.size
        hit[valid] = sorted_set[pos[valid]] == keys[valid]
        return hit
    
    pred_total = 0
    correct_total_in_test = 0
    wrong_open_replace_count = 0
    rs_scores = []
    rs_correct = []
    rs_keys = []

    for idx, (s, e) in enumerate(cached_ranges):
        probs = cached_tail_prob_cpu[idx]
        preds = probs > best_thr
        k = int(preds.sum().item())
        pred_total += k
        if k == 0:
            continue

        nz = preds.nonzero(as_tuple=False)
        b = nz[:, 0].to(torch.int64)
        r = nz[:, 1].to(torch.int64)
        t = nz[:, 2].to(torch.int64)
        h = b + int(s)

        scores = probs[preds].to(torch.float32).numpy()
        keys = ((h * num_rel + r) * num_nodes + t).to(torch.int64).numpy()
        h = h.numpy()
        r = r.numpy()
        t = t.numpy()

        is_correct = _isin_sorted(keys, test_keys_sorted)
        correct_total_in_test += int(is_correct.sum())
        rs_scores.append(scores.astype(np.float32, copy=False))
        rs_correct.append(is_correct)
        rs_keys.append(keys)

        if use_powa:
            wrong_idx = np.nonzero(~is_correct)[0]
            wrong_open_replace_count += _count_powa_negative(
                h[wrong_idx], r[wrong_idx], t[wrong_idx], test_pair_rel_map, rel_map
            )

    if len(rs_scores) == 0:
        rs_tsp = 0.0
        rs_tsp_norm = 0.0
        rs_K = 0
    else:
        scores_all = np.concatenate(rs_scores, axis=0)
        correct_all = np.concatenate(rs_correct, axis=0)
        keys_all = np.concatenate(rs_keys, axis=0)
        rs_K = int(scores_all.shape[0])
        order = np.lexsort((keys_all, -scores_all))
        sign = np.where(correct_all, 1.0, -1.0).astype(np.float32, copy=False)
        w = 1.0 / (np.arange(rs_K, dtype=np.float32) + 1.0)
        rs_tsp = float(np.sum(sign[order] * w))
        rs_tsp_norm = float(rs_tsp / float(np.sum(w)))

    test_total = len(test_triples)
    test_total_with_reverse = int(label_graph.edge_index.size(1))
    powa_negative = wrong_open_replace_count if use_powa else None
    test_p, test_r, test_f1 = _compute_tsp_metrics(
        correct_total_in_test, pred_total, test_total_with_reverse, powa_negative
    )
    test_cwa_p = correct_total_in_test / max(1, pred_total)
    test_powa_p = (
        correct_total_in_test / max(1, correct_total_in_test + wrong_open_replace_count)
        if use_powa else test_cwa_p
    )

    logger.warning(
        f"[{tb_tag}] {threshold_split} threshold best_thr={best_thr:.7f} "
        f"best_F_TSP={val_f1s[best_i]:.4f} best_JPrecision={val_ps[best_i]:.4f} "
        f"best_STRecall={val_rs[best_i]:.4f} assumption={assumption}"
    )
    logger.warning(
        f"[{tb_tag}] test JPrecision={test_p:.4f} STRecall={test_r:.4f} F_TSP={test_f1:.4f} "
        f"CWA_Precision={test_cwa_p:.4f} POWA_Precision={test_powa_p:.4f} "
        f"pred_total={pred_total} correct_in_test={correct_total_in_test} "
        f"test_total={test_total} test_total_with_reverse={test_total_with_reverse} "
        f"wrong_open_replace_count={wrong_open_replace_count} "
        f"RS_TSP={rs_tsp:.6f} RS_TSP_norm={rs_tsp_norm:.6f} K={rs_K}"
    )
    if writer is not None:
        writer.add_scalar(f"{tb_tag}/best_thr", best_thr, step)
        writer.add_scalar(f"{tb_tag}/{'val' if threshold_split == 'validation' else 'test'}_best_F_TSP", val_f1s[best_i], step)
        writer.add_scalar(f"{tb_tag}/{'val' if threshold_split == 'validation' else 'test'}_best_JPrecision", val_ps[best_i], step)
        writer.add_scalar(f"{tb_tag}/{'val' if threshold_split == 'validation' else 'test'}_best_STRecall", val_rs[best_i], step)
        writer.add_scalar(f"{tb_tag}/JPrecision", test_p, step)
        writer.add_scalar(f"{tb_tag}/STRecall", test_r, step)
        writer.add_scalar(f"{tb_tag}/F_TSP", test_f1, step)
        writer.add_scalar(f"{tb_tag}/CWA_Precision", test_cwa_p, step)
        writer.add_scalar(f"{tb_tag}/POWA_Precision", test_powa_p, step)
        writer.add_scalar(f"{tb_tag}/pred_total", pred_total, step)
        writer.add_scalar(f"{tb_tag}/correct_in_test", correct_total_in_test, step)
        writer.add_scalar(f"{tb_tag}/test_total", test_total, step)
        writer.add_scalar(f"{tb_tag}/test_total_with_reverse", test_total_with_reverse, step)
        writer.add_scalar(f"{tb_tag}/wrong_open_replace_count", wrong_open_replace_count, step)
        writer.add_scalar(f"{tb_tag}/RS_TSP", rs_tsp, step)
        writer.add_scalar(f"{tb_tag}/RS_TSP_norm", rs_tsp_norm, step)
        writer.add_scalar(f"{tb_tag}/RS_TSP_K", rs_K, step)

    if return_details:
        return {"validation_f_tsp": val_f1s[best_i] if threshold_split == "validation" else None,
                "selection_f_tsp": val_f1s[best_i], "threshold_selection_split": threshold_split,
                "threshold": best_thr, "test_jprecision": test_p,
                "test_strecall": test_r, "test_f_tsp": test_f1}
    return test_p, test_r, test_f1, best_thr


def build_edge_drop_scheduler(cfg, total_steps: int, base_drop: float):
    """
    Optional: edge drop schedule (currently unused, reserved for future extension).
    """
    aug = getattr(cfg, "augment", None)
    sched = None
    if isinstance(aug, dict):
        sched = aug.get("edge_drop_schedule", None)
    elif aug is not None and hasattr(aug, "edge_drop_schedule"):
        sched = aug.edge_drop_schedule

    if not sched:
        return lambda step: base_drop

    mode = str(sched.get("mode", "linear")).lower()
    start = float(sched.get("start", base_drop))
    end = float(sched.get("end", base_drop))
    start = max(0.0, min(1.0, start))
    end = max(0.0, min(1.0, end))

    def ratio_at(step):
        if total_steps <= 0:
            return end
        t = max(0.0, min(1.0, step / total_steps))
        if mode == "cosine":
            w = 0.5 * (1.0 - math.cos(math.pi * t))
            return (1 - w) * start + w * end
        else:
            return (1 - t) * start + t * end

    return ratio_at

def save_diffusion_checkpoint(path, model, optimizer, scaler, cfg, epoch, metrics):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scaler": scaler.state_dict(), "config": dict(cfg), "epoch": epoch,
             "metrics": metrics, "stage1_checkpoint": cfg.checkpoint,
             "selection_metric": "validation_f_tsp" if metrics["threshold_selection_split"] == "validation" else "test_threshold_f_tsp"}
    temporary = path + ".tmp"
    torch.save(state, temporary)
    os.replace(temporary, path)


def main():
    args, vars_ = util.parse_args()
    cfg = util.load_config(args.config, context=vars_)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())
    logger = util.get_root_logger()
    device = util.get_device(cfg)

    if util.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), "runs", "vqdiff_tokens"))
        logger.warning(pprint.pformat(cfg))
    else:
        writer = None

    dataset = util.build_dataset(cfg)
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device); valid_data = valid_data.to(device); test_data = test_data.to(device)

    train_data.num_relations = cfg.model.relation_model.num_relation
    valid_data.num_relations = cfg.model.relation_model.num_relation
    test_data.num_relations = cfg.model.relation_model.num_relation

    # Load trained VQpretrain and freeze
    vq_model = VQpretrain(rel_model_cfg=cfg.model.relation_model, vq_model_cfg=cfg.model.vq_model).to(device)
    state = torch.load(cfg.checkpoint, map_location="cpu")
    vq_model.load_state_dict(state["model"],strict=True)
    for p in vq_model.parameters():
        p.requires_grad_(False)
    vq_model.eval()

    # Get token dimensions and length
    tokens = int(vq_model.vq_vae.num_rel)

    # Build diffusion model
    num_rel = int(vq_model.vq_vae.num_rel)
    rel_dim = int(getattr(vq_model.vq_vae, "rel_dim",64))
    diff = TokenDiffusion(
        z_dim=rel_dim,
        tokens=num_rel,
        T=getattr(cfg, "diffusion", {}).get("T", 1000),
        hidden=getattr(cfg, "diffusion", {}).get("hidden", 128),
        layers=getattr(cfg, "diffusion", {}).get("layers", getattr(cfg, "diffusion", {}).get("num_layers", 3)),
        time_dim=getattr(cfg, "diffusion", {}).get("time_dim", 64),
        dropout=getattr(cfg, "diffusion", {}).get("dropout", 0.1),
        relconfig=cfg.model.relation_model,
        use_ou=cfg.diffusion.get("use_ou", False),
        kappa=cfg.diffusion.get("kappa", 4.0),
        sigma_inf=cfg.diffusion.get("sigma_inf", 1.0),
        eta=cfg.diffusion.get("eta", 0.9),
        use_ffn=cfg.diffusion.get("use_ffn", True),
        use_relation_attn=cfg.diffusion.get("use_relation_attn", True),
        use_node_attn=cfg.diffusion.get("use_node_attn", True),
        mixer=cfg.diffusion.get("mixer", "linear_attention"),
        node_mixing_size=cfg.diffusion.get("node_mixing_size", 512),
    ).to(device)

    # LinearAttention.promlp is never called by forward. Exclude this inactive
    # branch from optimization/checkpoints while preserving initialization order.
    for block in list(diff.net.relation_attn) + list(diff.net.node_attn):
        if hasattr(block, "promlp"):
            del block.promlp

    opt_cfg = dict(cfg.optimizer)
    opt_cls = opt_cfg.pop("class", "AdamW")
    if opt_cls not in ("Adam", "AdamW"):
        raise ValueError(f"Unsupported optimizer: {opt_cls}. Choose Adam or AdamW.")
    opt = getattr(optim, opt_cls)(diff.parameters(), **opt_cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    world_size = util.get_world_size()
    rank = util.get_rank()
    if world_size > 1:
        diff = nn.parallel.DistributedDataParallel(diff, device_ids=[device])

   
    # Training: VQ/label chunk accumulation -> diffusion
    epochs = int(cfg.train.num_epoch)
    log_interval = int(cfg.train.log_interval)
    repeat_per_epoch = int(getattr(cfg.train, "steps_per_epoch", 5))
    vq_chunk = int(getattr(cfg.train, "vq_chunk", 1500))
    net_chuck = int(getattr(cfg.train, "net_chunk", 512)) 

    use_soft_vq = True
    soft_vq = getattr(vq_model.vq_vae, "vq", None) if use_soft_vq else None

    world_size = util.get_world_size()

    global_step = 0
    best_validation_f1 = float("-inf")
    last_metrics = None
    for epoch in range(epochs):
        diff.train()
        losses = []

        # Construct observed/target graphs with Tobs subset of Ttar subset of Ttrain
        ratio_target_min = float(getattr(cfg, "augment", {}).get("ratio_target_min", 0.7))
        ratio_target_max = float(getattr(cfg, "augment", {}).get("ratio_target_max", 1.0))
        ratio_observed = float(getattr(cfg, "augment", {}).get("ratio_observed", 0.8))
        uncompletekg, completekg = split_uncomplete_complete(
            train_data,
            ratio_target_range=(ratio_target_min, ratio_target_max),
            ratio_observed=ratio_observed,
            device=device,
        )
        uncompletekg = uncompletekg.to(device)
        completekg = completekg.to(device)

        N = int(uncompletekg.num_nodes)
        num_rel = int(vq_model.vq_vae.num_rel)
        if world_size > 1:
            s = (N * rank) // world_size
            e = (N * (rank + 1)) // world_size
        else:
            s, e = 0, N

        ##### Accumulate x0 / known_x0 / known_mask for this rank in chunks
        with torch.no_grad():
            x0_chunk = precompute_tokens_chunked(
                vq_model, completekg, device, N, num_rel, rel_dim, chunk_size=vq_chunk, start=s, end=e
            )  # (M,R,D)
            known_x0_chunk = precompute_tokens_chunked(
                vq_model, uncompletekg, device, N, num_rel, rel_dim, chunk_size=vq_chunk, start=s, end=e
            )  # (M,R,D)

        
        T_total = diff.module.T if isinstance(diff, nn.parallel.DistributedDataParallel) else diff.T

        for _ in range(repeat_per_epoch):
            model_ref = diff.module if isinstance(diff, nn.parallel.DistributedDataParallel) else diff
           
            t = torch.randint(0, T_total, (1,), device=device, dtype=torch.long)

            # Treat x0 / known_x0 of the whole rank as (1, N, R, D)
            x0_full = x0_chunk.unsqueeze(0)         # (1, N, R, D)
            known_full = known_x0_chunk.unsqueeze(0)
            opt.zero_grad(set_to_none=True)
            for s_h in range(0, N, net_chuck):
                opt.zero_grad(set_to_none=True)
                e_h = min(N, s_h + net_chuck)
                h_index = torch.arange(s_h, e_h, device=device)
                with torch.cuda.amp.autocast(enabled=True):
                    loss, loss_mse, loss_bce = model_ref.train_step(
                        x0=x0_full,
                        t_idx=t,
                        known_x0=known_full,
                        incomplete_graph=uncompletekg,
                        complete_graph=completekg,
                        vq_model=vq_model,
                        heads=h_index,  
                        )
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
           
            model_ref = diff.module if isinstance(diff, nn.parallel.DistributedDataParallel) else diff

            if rank == 0 and (global_step % log_interval == 0):
                logger.warning(f"Epoch {epoch+1}/{epochs} Step {global_step}: loss={loss.item():.4f}")
                if writer:
                    writer.add_scalar("diff/train_loss", float(loss.item()), global_step)
            losses.append(float(loss.item()))
            global_step += 1

        if rank == 0:
            avg_loss = sum(losses) / max(1, len(losses))
            if writer:
                writer.add_scalar("diff/avg_loss", avg_loss, epoch)

        util.synchronize()

        if rank == 0 and ((epoch + 1) % cfg.train.eval_interval == 0 or (epoch + 1) == epochs):
            model_ref = diff
            bare = model_ref.module if isinstance(model_ref, nn.parallel.DistributedDataParallel) else model_ref
            backup = {n: p.detach().clone() for n, p in bare.named_parameters()}
            try:
                joint_graph = build_joint_target_graph(train_data, test_data, device=device, reverse=True).to(device)
                last_metrics = evaluate(
                    cfg, vq_model, bare, train_data, joint_graph, device, logger,valid_data=valid_data,
                    writer=writer, test_data=test_data, step=epoch, guidance_scale=0.5, use_soft_vq=True, return_details=True
                )
            finally:
                for n, p in bare.named_parameters():
                    p.data.copy_(backup[n])
            with open("metrics.jsonl", "a") as fout:
                fout.write(json.dumps({"epoch": epoch + 1, **last_metrics}) + "\n")
            if math.isfinite(last_metrics["selection_f_tsp"]) and last_metrics["selection_f_tsp"] > best_validation_f1:
                best_validation_f1 = last_metrics["selection_f_tsp"]
                save_diffusion_checkpoint("diffusion_best.pth", bare, opt, scaler, cfg,
                                          epoch + 1, last_metrics)
                logger.warning(f"[Checkpoint] diffusion_best.pth epoch={epoch + 1} selection_F_TSP={best_validation_f1:.6f} split={last_metrics['threshold_selection_split']}")

    if rank == 0:
        bare = diff.module if isinstance(diff, nn.parallel.DistributedDataParallel) else diff
        save_diffusion_checkpoint("diffusion_last.pth", bare, opt, scaler, cfg, epochs, last_metrics)
    if writer:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    
    main()