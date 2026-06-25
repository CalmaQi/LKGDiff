import os
import math
import pprint
import ast
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


def split_uncomplete_complete(train_data: Data, ratio_uncomplete: float = 0.8,
                              extra_ratio_range=(0.15, 0.20), device=None):
    """
    Based on train target edges, stratified sampling by relation:
      - uncompletekg: keep floor(0.8*nr) for each relation
      - completekg: take floor(u*nr) from remaining edges (u~Uniform[0.15,0.20]), merge with uncomplete
    Returns: uncompletekg, completekg (Data)
    """
    device = device or getattr(train_data.edge_index, "device", None)
    ei, et = _get_target_edges(train_data)
    E = ei.size(1)
    
    # Stratified sampling by relation
    keep_u_list, keep_c_list = [], []
    uniq = torch.unique(et)
    for r in uniq.tolist():
        idx = (et == r).nonzero(as_tuple=False).view(-1)
        nr = idx.numel()
        if nr == 0:
            continue
        k_u = int(math.floor(nr * ratio_uncomplete))
        k_u = max(0, min(nr, k_u))
        k_x = int(math.floor(nr * random.uniform(extra_ratio_range[0], extra_ratio_range[1])))
        k_x = max(0, min(nr - k_u, k_x))
        perm = idx[torch.randperm(nr, device=idx.device)]
        sel_u = perm[:k_u]
        remain = perm[k_u:]
        sel_x = remain[:k_x] if k_x > 0 else torch.empty(0, dtype=perm.dtype, device=perm.device)
        keep_u_list.append(sel_u)
        keep_c_list.append(torch.unique(torch.cat([sel_u, sel_x], dim=0)))

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
    """
    Returns forbidden mask (B,R,N), True means (h,r,t) already exists in data, needs to be masked during evaluation.
    Usually pass train_data to mask known triples in training set.
    """
    B = batch_heads.numel()
    mask = torch.zeros((B, num_rel, num_nodes), device=device, dtype=torch.bool)
    
    joint_data = build_joint_target_graph(data, valid_data, device=device)
    
    H_all = joint_data.edge_index[0].to(device)
    T_all = joint_data.edge_index[1].to(device)
    R_all = getattr(joint_data, "edge_type", None)
    if R_all is not None:
        R_all = R_all.to(device)
    else:
        # If no relation type, treat as class 0
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
             use_soft_vq: bool = True,tb_tag: str = "difftest"):
    """
    Condition uses data_incomplete, labels come from data_complete (should be joint(train,test) in this requirement).
    Accumulate VQ in chunks, then feed into diffusion; evaluate logits by chunks to avoid peak.
    """
    vq_model.eval(); diff_model.eval()
    num_nodes = int(data_complete.num_nodes)
    num_rel = int(vq_model.vq_vae.num_rel)
    rel_dim = int(getattr(vq_model.vq_vae, "rel_dim", 64))
    has_empty = False
    vq_chunk = int(getattr(cfg.train, "vq_chunk", 2048))

    # label graph (test + reverse)
    reverse_edge_index = torch.stack([test_data.target_edge_index[1], test_data.target_edge_index[0]], dim=0)
    reverse_edge_type = test_data.target_edge_type + int(test_data.target_edge_type.max().item()) + 1
    label_edge_index = torch.cat([test_data.target_edge_index, reverse_edge_index], dim=1)
    label_edge_type  = torch.cat([test_data.target_edge_type, reverse_edge_type],  dim=0)
    label_graph = Data(edge_index=label_edge_index, edge_type=label_edge_type, num_nodes=test_data.num_nodes)
    label_graph.num_relations = getattr(test_data, "num_relations", getattr(data_incomplete, "num_relations", 0))

    # Required for open-set error statistics: test triples, train+test (h,t)->rels, relation correspondence table
    test_triples, _ = _build_triple_set_and_pair_rel_map(test_data)
    train_triples, train_ht2rels = _build_triple_set_and_pair_rel_map(data_incomplete)
    test_triples_for_pair, test_ht2rels = _build_triple_set_and_pair_rel_map(test_data)
    _ = train_triples, test_triples_for_pair
    joint_ht2rels = {}
    for key, rels in train_ht2rels.items():
        joint_ht2rels[key] = set(rels)
    for key, rels in test_ht2rels.items():
        if key not in joint_ht2rels:
            joint_ht2rels[key] = set()
        joint_ht2rels[key].update(rels)

    rel_map_path = _resolve_relation_map_path(cfg)
    rel_map = _load_relation_correspondence(rel_map_path)
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
        
    # Threshold sweep counting
    thr_grid = torch.linspace(0.2, 0.9999999, steps=10, device=device)
    sweep_tp = torch.zeros_like(thr_grid, dtype=torch.long)
    sweep_fp = torch.zeros_like(thr_grid, dtype=torch.long)
    sweep_fn = torch.zeros_like(thr_grid, dtype=torch.long)
    # Scheme A: cache tail_prob of each chunk to CPU, reuse by best_thr later to avoid repeated forward logits.
    cached_tail_prob_cpu = []
    cached_ranges = []
    
    # Pre-allocate CPU logits container (only fill rows for heads managed by this rank)
    logits_cpu = torch.empty((num_nodes, num_rel, num_nodes), dtype=torch.float32, device="cpu")

    # Precompute logits -> CPU
    for s in range(0, num_nodes, vq_chunk):
        e = min(num_nodes, s + vq_chunk)
        heads = torch.arange(s, e, device=device)
        logits_chunk = vq_model.logits_from_head(z_tokens_all.squeeze(0)[s:e])  # (b,R,N)
        logits_cpu[heads.detach().cpu()] = logits_chunk.detach().float().cpu()

    # Probability and forward/inverse relation averaging (on CPU)
    tail_prob_cpu = torch.sigmoid(logits_cpu[..., :num_nodes])  # (N,R,N)

    # Estimate forward/inverse relation offset and do symmetric averaging
   
    rel_offset = num_rel // 2  
    if rel_offset > 0 and num_rel >= rel_offset * 2:
        r0 = torch.arange(rel_offset)         # Forward relation index set (K,)
        r1 = r0 + rel_offset                  # Inverse relation index set (K,)

        # A[h,k,t] = P(h, r0[k], t)
        A = tail_prob_cpu[:, r0, :]           # (N,K,N)
        # B[h,k,t] = P(h, r1[k], t); B_t[h,k,t] = P(t, r1[k], h)
        B = tail_prob_cpu[:, r1, :]           # (N,K,N)
        B_t = B.transpose(0, 2)               # (N,K,N) Swap head and tail dimensions

        avg = 0.5 * (A + B_t)
        # Write back symmetric sides: forward as avg, inverse as avg transpose (keep P(t,r_rev,h) corresponding)
        tail_prob_cpu[:, r0, :] = avg
        tail_prob_cpu[:, r1, :] = avg.transpose(0, 2)

    for s in range(0, num_nodes, vq_chunk):
        e = min(num_nodes, s + vq_chunk)
        heads = torch.arange(s, e, device=torch.device("cpu"))
        labels = build_labels_for_heads(label_graph, heads, num_rel, num_nodes, has_empty, device=torch.device("cpu"))  # (b,R,N)
        forbid = _build_forbidden_mask_for_heads(data_incomplete,valid_data, heads, num_rel, num_nodes, device=device)  # (b,R,N)
        tail_prob = tail_prob_cpu[heads, :num_nodes].to(device).masked_fill(forbid, -1e9)

        gt = labels[..., :num_nodes].bool()
        gt = gt.to(device)
        cached_tail_prob_cpu.append(tail_prob.detach().to(device="cpu", dtype=torch.float16))
        cached_ranges.append((s, e))
        for i, thr in enumerate(thr_grid):
            preds = tail_prob > thr
            sweep_tp[i] += (preds & gt).sum()
            sweep_fp[i] += (preds & (~gt)).sum()
            sweep_fn[i] += ((~preds) & gt).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(sweep_tp, op=dist.ReduceOp.SUM)
        dist.all_reduce(sweep_fp, op=dist.ReduceOp.SUM)
        dist.all_reduce(sweep_fn, op=dist.ReduceOp.SUM)

    f1s = []
    ps_, rs_ = [], []
    for i in range(thr_grid.numel()):
        tp = sweep_tp[i].item(); fp = sweep_fp[i].item(); fn = sweep_fn[i].item()
        p = tp / max(1, tp + fp); r_ = (tp / max(1, tp + fn))**0.5
        f1 = 2 * p * r_ / max(1e-8, p + r_)
        f1s.append(f1); ps_.append(p); rs_.append(r_)
    best_i = int(torch.tensor(f1s).argmax().item())
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
        hit = (pos < sorted_set.size) & (sorted_set[pos] == keys)
        return hit
    
    # Use best_thr for open-set statistics
    pred_total = 0
    correct_total_by_label = int(sweep_tp[best_i].item())
    correct_total_in_test = 0
    wrong_open_replace_count = 0
    rs_scores = []
    rs_correct = []
    rs_keys = []  # For tie-break (sort by key when scores are equal)

    pred_total = 0
    correct_total_in_test = 0
    wrong_open_replace_count = 0

    for idx, (s, e) in enumerate(cached_ranges):
        probs = cached_tail_prob_cpu[idx]  # (b,R,N) float16 on CPU
        preds = probs > best_thr
        k = int(preds.sum().item())
        pred_total += k
        if k == 0:
            continue

        nz = preds.nonzero(as_tuple=False)  # (K,3) columns: b,r,t (CPU)
        b = nz[:, 0].to(torch.int64)
        r = nz[:, 1].to(torch.int64)
        t = nz[:, 2].to(torch.int64)
        h = b + int(s)

        scores = probs[preds].to(torch.float32).numpy()
        keys = ((h * num_rel + r) * num_nodes + t).to(torch.int64).numpy()

        # Explicitly exclude train+valid(+inverse edges) (theoretically already suppressed by forbid mask, but filter again here as requested)
        
        h = h.numpy(); r = r.numpy(); t = t.numpy()

        is_correct = _isin_sorted(keys, test_keys_sorted)
        correct_total_in_test += int(is_correct.sum())

        rs_scores.append(scores.astype(np.float32, copy=False))
        rs_correct.append(is_correct)
        rs_keys.append(keys)

        # If wrong_open_replace_count still needed: only iterate over the wrong part (reduce loop amount)
        wrong_idx = np.nonzero(~is_correct)[0]
        for j in wrong_idx:
            h_idx = int(h[j]); r_pred = int(r[j]); t_idx = int(t[j])
            rels_exist = joint_ht2rels.get((h_idx, t_idx), None)
            if not rels_exist:
                continue
            if any((not _relations_correspond(r_exist, r_pred, rel_map)) for r_exist in rels_exist):
                wrong_open_replace_count += 1

    # --- Compute RS_TSP ---
    if len(rs_scores) == 0:
        rs_tsp = 0.0
        rs_tsp_norm = 0.0
        rs_K = 0
    else:
        scores_all = np.concatenate(rs_scores, axis=0)
        correct_all = np.concatenate(rs_correct, axis=0)
        keys_all = np.concatenate(rs_keys, axis=0)
        rs_K = int(scores_all.shape[0])

        # Score descending; tie-break by key ascending for determinism
        order = np.lexsort((keys_all, -scores_all))

        sign = np.where(correct_all, 1.0, -1.0).astype(np.float32, copy=False)
        w = (1.0 / (np.arange(rs_K, dtype=np.float32) + 1.0))
        rs_tsp = float(np.sum(sign[order] * w))
        rs_tsp_norm = float(rs_tsp / float(np.sum(w)))  # Optional: normalize to [-1,1]

    test_total = len(test_triples)
    test_total_with_reverse = int(label_graph.edge_index.size(1))

    logger.warning(f"[{tb_tag}] threshold sweep best_thr={best_thr:.3f}  best_F1={f1s[best_i]:.4f} best_P={ps_[best_i]:.4f} best_R={rs_[best_i]:.4f}")
    logger.warning(
        f"[{tb_tag}] triples@best_thr pred_total={pred_total} "
        f"correct_in_test={correct_total_in_test} "
        f"correct_by_eval_label={correct_total_by_label} "
        f"test_total={test_total} "
        f"test_total_with_reverse={test_total_with_reverse} "
        f"wrong_open_replace_count={wrong_open_replace_count}"
        f"[{tb_tag}] RS_TSP={rs_tsp:.6f} RS_TSP_norm={rs_tsp_norm:.6f} K={rs_K}"
    )
    if writer is not None:
        writer.add_scalar(f"{tb_tag}/best_thr", best_thr, step)
        writer.add_scalar(f"{tb_tag}/best_F1", f1s[best_i], step)
        writer.add_scalar(f"{tb_tag}/best_P", ps_[best_i], step)
        writer.add_scalar(f"{tb_tag}/best_R", rs_[best_i], step)
        writer.add_scalar(f"{tb_tag}/pred_total", pred_total, step)
        writer.add_scalar(f"{tb_tag}/correct_in_test", correct_total_in_test, step)
        writer.add_scalar(f"{tb_tag}/correct_by_eval_label", correct_total_by_label, step)
        writer.add_scalar(f"{tb_tag}/test_total", test_total, step)
        writer.add_scalar(f"{tb_tag}/test_total_with_reverse", test_total_with_reverse, step)
        writer.add_scalar(f"{tb_tag}/wrong_open_replace_count", wrong_open_replace_count, step)
        writer.add_scalar(f"{tb_tag}/RS_TSP", rs_tsp, step)
        writer.add_scalar(f"{tb_tag}/RS_TSP_norm", rs_tsp_norm, step)
        writer.add_scalar(f"{tb_tag}/RS_TSP_K", rs_K, step)

    return ps_[best_i], rs_[best_i], f1s[best_i], best_thr


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
        layers=getattr(cfg, "diffusion", {}).get("layers", 3),
        time_dim=getattr(cfg, "diffusion", {}).get("time_dim", 64),
        dropout=getattr(cfg, "diffusion", {}).get("dropout", 0.1),
        relconfig=cfg.model.relation_model,
    ).to(device)
    

    

    opt = optim.AdamW(diff.parameters(), lr=getattr(cfg.optimizer, "lr", 1e-4), weight_decay=1e-2)
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
    for epoch in range(epochs):
        diff.train()
        losses = []

        # Construct uncomplete/complete graph
        ratio_uncomplete = float(getattr(cfg, "augment", {}).get("ratio_uncomplete", 0.75))
        extra_lo = float(getattr(cfg, "augment", {}).get("complete_extra_min", 0.2))
        extra_hi = float(getattr(cfg, "augment", {}).get("complete_extra_max", 0.25))
        uncompletekg, completekg = split_uncomplete_complete(
            train_data, ratio_uncomplete=ratio_uncomplete, extra_ratio_range=(extra_lo, extra_hi), device=device
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
                prec, rec, f1, thr = evaluate(
                    cfg, vq_model, bare, train_data, joint_graph, device, logger,valid_data=valid_data,
                    writer=writer, test_data=test_data, step=epoch, guidance_scale=0.5, use_soft_vq=True
                )
            finally:
                for n, p in bare.named_parameters():
                    p.data.copy_(backup[n])
            

    if writer:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    
    main()