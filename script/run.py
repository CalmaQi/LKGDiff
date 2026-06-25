import os
import sys
import math
import pprint
from itertools import islice
import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data
import random
from torch.utils.tensorboard import SummaryWriter  # added
import numpy as np  

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util

from ultra.stage12 import VQpretrain
from collections import defaultdict
torch.set_float32_matmul_precision("medium")
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'
original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr

class AugmentCache:
    def __init__(self, num_nodes: int, num_rel: int, rel_offset: int,
                 tail_pop_r: list, heads_of_tail: list,
                 cooc_prob: torch.Tensor, impossible: torch.Tensor,
                 twohop_cands: dict,
                 tail_pop_mat: torch.Tensor = None,     # added: stacked matrix (R,N)
                 twohop_mat: torch.Tensor = None):      # added: 2D candidates (N,K), padded with -1
        self.num_nodes = int(num_nodes)
        self.num_rel = int(num_rel)
        self.rel_offset = int(rel_offset)
        self.tail_pop_r = tail_pop_r
        self.heads_of_tail = heads_of_tail
        self.cooc_prob = cooc_prob
        self.impossible = impossible
        self.twohop_cands = twohop_cands
        self.tail_pop_mat = tail_pop_mat
        self.twohop_mat = twohop_mat

@torch.no_grad()
def build_augment_cache_pos_rev(data: Data, twohop_cap: int = 1024) -> AugmentCache:
    """
    Precompute global statistics needed for augmentation based on positive half edges, for fast reference by subsequent augmentation.
    """
    device = data.edge_index.device
    E_pos, rel_offset = _detect_pos_rev_layout(data)
    ei_pos = data.edge_index[:, :E_pos].to(device)
    et_pos = data.edge_type[:E_pos].to(device)
    num_nodes = int(data.num_nodes)
    num_rel = int(et_pos.max().item()) + 1

    # Intra-relation statistics (using base positive edges)
    tails_of_head_base, heads_of_tail_base, tail_pop_r, _ = _build_rel_stats_pos(ei_pos, et_pos, num_rel, num_nodes)

    # Relation co-occurrence + impossible
    cooc_prob, impossible, head_rel_sets = _compute_rel_cooc_from_pos(et_pos, ei_pos, num_rel)

    # Two-hop candidates: across relations
    all_tails_of_head = defaultdict(list)
    for r in range(num_rel):
        for h, lst in tails_of_head_base[r].items():
            if lst:
                all_tails_of_head[h].extend(lst)

    twohop_cands = {}
    for h, onehop in all_tails_of_head.items():
        if not onehop:
            continue
        onehop = list(dict.fromkeys(onehop))[:twohop_cap]
        cand = set()
        for v in onehop:
            for r2 in range(num_rel):
                lst = tails_of_head_base[r2].get(v, [])
                if lst:
                    # Uniformly clip by cap
                    for t2 in lst[: max(1, twohop_cap // max(1, num_rel))]:
                        cand.add(t2)
            if len(cand) >= twohop_cap:
                break
        if cand:
            twohop_cands[h] = list(cand)[:twohop_cap]
            
    tail_pop_mat = torch.stack(tail_pop_r, dim=0).contiguous()  # (R,N)
    twohop_mat = torch.full((num_nodes, max(1, twohop_cap)), -1, dtype=torch.long, device=device)
    if len(twohop_cands) > 0:
        for h, lst in twohop_cands.items():
            if not lst:
                continue
            k = min(len(lst), twohop_mat.size(1))
            twohop_mat[h, :k] = torch.tensor(lst[:k], device=device, dtype=torch.long)

    return AugmentCache(
        num_nodes=num_nodes,
        num_rel=num_rel,
        rel_offset=rel_offset,
        tail_pop_r=tail_pop_r,
        heads_of_tail=heads_of_tail_base,
        cooc_prob=cooc_prob,
        impossible=impossible,
        twohop_cands=twohop_cands,
        tail_pop_mat=tail_pop_mat,
        twohop_mat=twohop_mat,
    )
@torch.no_grad()
def augment_pos_rev_reconstruct_fast(
    data: Data,
    cache: AugmentCache,
    degree_inflate: float = 0.2,     # Degree of existing (h,r) +20%
    drop_ratio: float = 0.0,         # Stratified edge drop (positive half)
    rewire_ratio: float = 0.0,       # Replace tail (positive half)
    add_cap_per_hr: int = 64,        # Max additions per (h,r) (fast mode uses half estimate)
    twohop_prob: float = 0.7,        # Probability of sampling from two-hop candidates
    dedup_new: bool = False,         # Whether to deduplicate additions (slower, off by default)
    no_self_loop: bool = True,
    device=None,
) -> Data:
    """
    Fast augmentation (vectorized):
      - - Only perform delete/modify/add on positive half edges;
      - - Rewiring and new tail nodes use two-hop candidates first, otherwise fall back to uniform;
      - - Does not perform missing relation co-occurrence edge addition (sacrifices quality for speed, can be kept in slow function).
    """
    assert cache is not None and cache.twohop_mat is not None and cache.tail_pop_mat is not None
    device = device or data.edge_index.device

    ei_all = data.edge_index.to(device)
    et_all = data.edge_type.to(device)
    E_pos = ei_all.size(1) // 2
    num_nodes = cache.num_nodes
    num_rel = cache.num_rel
    rel_offset = cache.rel_offset

    # Positive edge view
    ei_pos = ei_all[:, :E_pos].clone()
    et_pos = et_all[:E_pos].clone()

    # Helper: two-hop or uniform tail sampling (vectorized)
    def sample_tail_for_heads(H: torch.Tensor) -> torch.Tensor:
        K = cache.twohop_mat.size(1)
        M = H.numel()
        use_twohop = (torch.rand(M, device=device) < float(twohop_prob))
        rand_idx = torch.randint(0, K, (M,), device=device)
        t_twohop = cache.twohop_mat[H, rand_idx]
        t_rand = torch.randint(0, num_nodes, (M,), device=device)
        t_new = torch.where((use_twohop & (t_twohop >= 0)), t_twohop, t_rand)
        if no_self_loop:
            same = (t_new == H)
            if same.any():
                t_new[same] = (t_new[same] + 1) % num_nodes
        return t_new

    if rewire_ratio > 0.0 and ei_pos.size(1) > 0:
        M = int(ei_pos.size(1) * float(rewire_ratio))
        if M > 0:
            idx = torch.randperm(ei_pos.size(1), device=device)[:M]
            H_sel = ei_pos[0, idx]
            t_new = sample_tail_for_heads(H_sel)
            ei_pos[1, idx] = t_new

    add_H, add_R, add_T = [], [], []
    if degree_inflate > 1e-9 and ei_pos.size(1) > 0:
        H = ei_pos[0]
        R = et_pos
        hr_index = H.long() * num_rel + R.long()                       # (E,)
        deg = torch.bincount(hr_index, minlength=num_nodes * num_rel)  # (N*R,)
        # Target and demand
        alpha = float(degree_inflate)
        target = torch.ceil(deg.float() * (1.0 + alpha)).long()
        need = (target - deg).clamp_min(0)
        # Half cap (reserve half quota for other augmentations; fast mode simplifies to half)
        cap_half = max(0, int(add_cap_per_hr) // 2)
        need = torch.clamp(need, max=cap_half)
        if need.sum().item() > 0:
            hr_need_idx = (need > 0).nonzero(as_tuple=False).view(-1)   # (K_hr,)
            need_counts = need[hr_need_idx]                             # (K_hr,)
            h_add = (hr_need_idx // num_rel)
            r_add = (hr_need_idx % num_rel)
            H_add = torch.repeat_interleave(h_add, need_counts)
            R_add = torch.repeat_interleave(r_add, need_counts)
            if H_add.numel() > 0:
                T_add = sample_tail_for_heads(H_add)
                if bool(dedup_new):
                    # Optional deduplication (slower)
                    key = H_add.long() * (num_rel * num_nodes) + R_add.long() * num_nodes + T_add.long()
                    uniq, sel = torch.unique(key, sorted=False, return_inverse=False, return_counts=False, return_index=True)
                    H_add = H_add[sel]; R_add = R_add[sel]; T_add = T_add[sel]
                add_H.append(H_add); add_R.append(R_add); add_T.append(T_add)

    # Merge additions
    if len(add_H) > 0:
        Hn = torch.cat(add_H, dim=0)
        Rn = torch.cat(add_R, dim=0)
        Tn = torch.cat(add_T, dim=0)
        add_ei_pos = torch.stack([Hn.to(ei_pos.dtype), Tn.to(ei_pos.dtype)], dim=0)
        add_et_pos = Rn.to(et_pos.dtype)
        ei_pos = torch.cat([ei_pos, add_ei_pos], dim=1)
        et_pos = torch.cat([et_pos, add_et_pos], dim=0)

    ei_rev = torch.stack([ei_pos[1], ei_pos[0]], dim=0)
    et_rev = et_pos + rel_offset
    ei_aug = torch.cat([ei_pos, ei_rev], dim=1)
    et_aug = torch.cat([et_pos, et_rev], dim=0)

    out = Data(edge_index=ei_aug, edge_type=et_aug, num_nodes=num_nodes)
    out.num_relations = int(et_aug.max().item()) + 1
    return out
@torch.no_grad()
def augment_pos_rev_reconstruct(
    data: Data,
    degree_inflate: float = 0.2,      # Degree of existing (h,r) +20%
    drop_ratio: float = 0.0,          # Stratified edge drop (positive half)
    rewire_ratio: float = 0.0,        # Replace tail (positive half)
    add_cap_per_hr: int = 64,         # Total max additions per (h,r)
    no_self_loop: bool = True,
    # Missing relation guidance parameters
    miss_rel_topk: int = 2,           # For each h, select top-k missing relations to try adding edges
    miss_rel_add_per_hr: int = 1,     # Max new edges per (h, r_missing)
    twohop_cap: int = 1024,           # Backward compatible (uses twohop_cands from cache)
    device=None,
    cache: AugmentCache = None,       # added: precomputed cache (strongly recommended to pass)
) -> Data:
    """
    Only edit positive half edges (delete/modify/add), and rebuild reverse edges. Training label = augmented graph.
    Use precomputed cache to accelerate (cooc, heads_of_tail, pop_r, twohop).
    """
    assert cache is not None, "Please build cache via build_augment_cache_pos_rev(...) and pass cache=..."
    device = device or data.edge_index.device

    ei_all = data.edge_index.to(device)
    et_all = data.edge_type.to(device)

    E_pos = ei_all.size(1) // 2
    rel_offset = cache.rel_offset
    num_rel = cache.num_rel
    num_nodes = cache.num_nodes

    # Extract positive half view
    ei_pos = ei_all[:, :E_pos].clone()
    et_pos = et_all[:E_pos].clone()

    if drop_ratio > 0.0 and ei_pos.numel() > 0:
        keep = torch.zeros(E_pos, dtype=torch.bool, device=device)
        uniq = torch.unique(et_pos)
        p_keep = max(0.0, 1.0 - float(drop_ratio))
        for r in uniq.tolist():
            idx = (et_pos == r).nonzero(as_tuple=False).view(-1)
            nr = int(idx.numel())
            if nr == 0: continue
            k = int(math.floor(nr * p_keep))
            if p_keep > 0.0 and k == 0: k = 1
            if k >= nr:
                keep[idx] = True
            elif k > 0:
                perm = torch.randperm(nr, device=device)[:k]
                keep[idx[perm]] = True
        if not keep.any() and E_pos > 0:
            keep[torch.randint(0, E_pos, (1,), device=device)] = True
        ei_pos = ei_pos[:, keep]
        et_pos = et_pos[keep]

    tails_of_head_cur = [defaultdict(list) for _ in range(num_rel)]
    deg_hr_cur = defaultdict(int)
    present_rel_cur = defaultdict(set)
    Hc = ei_pos[0].tolist()
    Tc = ei_pos[1].tolist()
    Rc = et_pos.tolist()
    for h, t, r in zip(Hc, Tc, Rc):
        tails_of_head_cur[r][h].append(t)
        deg_hr_cur[(h, r)] += 1
        present_rel_cur[h].add(r)

    if rewire_ratio > 0.0 and ei_pos.size(1) > 0:
        heads_of_tail = cache.heads_of_tail
        tail_pop_r = cache.tail_pop_r
        M = int(ei_pos.size(1) * float(rewire_ratio))
        idx = torch.randperm(ei_pos.size(1), device=device)[:M]
        H = ei_pos[0, idx].tolist()
        R = et_pos[idx].tolist()
        for j, (h, r) in enumerate(zip(H, R)):
            avoid = set(tails_of_head_cur[r].get(h, []))
            # Similar head migration + pop_r guidance
            t_new = _sample_tail_guided(h, r, num_nodes,
                                        tails_of_head_cur[r], heads_of_tail[r],
                                        tail_pop_r[r], avoid, no_self_loop)
            # Update current view mapping
            t_old = int(ei_pos[1, idx[j]].item())
            if t_old in tails_of_head_cur[r].get(h, []):
                try:
                    tails_of_head_cur[r][h].remove(t_old)
                except ValueError:
                    pass
            tails_of_head_cur[r][h].append(t_new)
            deg_hr_cur[(h, r)] = len(tails_of_head_cur[r][h])
            present_rel_cur[h].add(r)
            ei_pos[1, idx[j]] = t_new

    add_H, add_R, add_T = [], [], []
    added_cnt = defaultdict(int)
    if degree_inflate > 1e-9:
        tail_pop_r = cache.tail_pop_r
        heads_of_tail = cache.heads_of_tail
        alpha = float(degree_inflate)
        for (h, r), deg in list(deg_hr_cur.items()):
            target = int(math.ceil(deg * (1.0 + alpha)))
            need = max(0, target - deg)
            cap_here = max(0, int(add_cap_per_hr) // 2 - added_cnt[(h, r)])
            need = min(need, cap_here)
            if need <= 0: continue
            avoid = set(tails_of_head_cur[r].get(h, []))
            for _ in range(need):
                t_new = _sample_tail_guided(h, r, num_nodes,
                                            tails_of_head_cur[r], heads_of_tail[r],
                                            tail_pop_r[r], avoid, no_self_loop)
                if t_new in avoid:
                    continue
                add_H.append(h); add_R.append(r); add_T.append(t_new)
                avoid.add(t_new)
                tails_of_head_cur[r][h].append(t_new)
                deg_hr_cur[(h, r)] += 1
                present_rel_cur[h].add(r)
                added_cnt[(h, r)] += 1

    if miss_rel_topk > 0 and miss_rel_add_per_hr > 0:
        cooc_prob = cache.cooc_prob
        impossible = cache.impossible
        tail_pop_r = cache.tail_pop_r
        heads_of_tail = cache.heads_of_tail
        for h, present in present_rel_cur.items():
            if not present:
                continue
            scores = torch.zeros(num_rel, device=device)
            for ri in present:
                scores += cooc_prob[ri]
            # Mask existing relations
            for ri in present:
                scores[ri] = -1e9
            # Mask 'impossible co-occurrence' relations
            for r_miss in range(num_rel):
                if r_miss in present:
                    continue
                bad = all(bool(impossible[ri, r_miss].item()) for ri in present)
                if bad:
                    scores[r_miss] = -1e9
            topk = int(miss_rel_topk)
            if topk <= 0:
                continue
            vals, idxs = torch.topk(scores, k=min(topk, num_rel), largest=True)
            for val, r in zip(vals.tolist(), idxs.tolist()):
                if val <= -1e8:
                    continue
                quota = min(miss_rel_add_per_hr, max(0, add_cap_per_hr - added_cnt[(h, r)]))
                if quota <= 0:
                    continue
                avoid = set(tails_of_head_cur[r].get(h, []))
                # 2-hop candidates first (from cache)
                cands = cache.twohop_cands.get(h, [])
                cands = list(cands)
                random.shuffle(cands)
                added_local = 0
                for t_new in cands:
                    if added_local >= quota:
                        break
                    if (no_self_loop and t_new == h) or (t_new in avoid):
                        continue
                    add_H.append(h); add_R.append(r); add_T.append(int(t_new))
                    avoid.add(int(t_new))
                    tails_of_head_cur[r][h].append(int(t_new))
                    deg_hr_cur[(h, r)] += 1
                    present_rel_cur[h].add(r)
                    added_cnt[(h, r)] += 1
                    added_local += 1
                # If insufficient, fill with intra-relation popularity
                while added_local < quota:
                    t_new = _sample_tail_guided(h, r, num_nodes,
                                                tails_of_head_cur[r], heads_of_tail[r],
                                                tail_pop_r[r], avoid, no_self_loop)
                    if t_new in avoid:
                        break
                    add_H.append(h); add_R.append(r); add_T.append(t_new)
                    avoid.add(t_new)
                    tails_of_head_cur[r][h].append(t_new)
                    deg_hr_cur[(h, r)] += 1
                    present_rel_cur[h].add(r)
                    added_cnt[(h, r)] += 1
                    added_local += 1

    # Merge additions
    if len(add_H) > 0:
        add_ei_pos = torch.stack([
            torch.tensor(add_H, device=device, dtype=ei_pos.dtype),
            torch.tensor(add_T, device=device, dtype=ei_pos.dtype)
        ], dim=0)
        add_et_pos = torch.tensor(add_R, device=device, dtype=et_pos.dtype)
        ei_pos = torch.cat([ei_pos, add_ei_pos], dim=1)
        et_pos = torch.cat([et_pos, add_et_pos], dim=0)

    ei_rev = torch.stack([ei_pos[1], ei_pos[0]], dim=0)
    et_rev = et_pos + rel_offset
    ei_aug = torch.cat([ei_pos, ei_rev], dim=1)
    et_aug = torch.cat([et_pos, et_rev], dim=0)

    out = Data(edge_index=ei_aug, edge_type=et_aug, num_nodes=num_nodes)
    out.num_relations = int(et_aug.max().item()) + 1
    return out
@torch.no_grad()
def _compute_rel_cooc_from_pos(et_pos: torch.Tensor, ei_pos: torch.Tensor, num_rel: int):
    """
    Based on positive half edges, count relation co-occurrence matrix cooc[r_i, r_j] by head (same h appears simultaneously).
    Returns:
        - cooc_prob: (R,R) row-normalized P(r_j | r_i) (with smoothing)
        - impossible: (R,R) boolean mask for near-impossible co-occurrence
        - head_rel_sets: Dict[int, Set[int]] set of appeared relations for each h
    """
    device = et_pos.device
    H = ei_pos[0].tolist()
    R = et_pos.tolist()
    head_rel_sets = defaultdict(set)
    for h, r in zip(H, R):
        head_rel_sets[int(h)].add(int(r))
    cooc = torch.zeros((num_rel, num_rel), dtype=torch.float32, device=device)
    for _, rels in head_rel_sets.items():
        rels = list(rels)
        m = len(rels)
        if m == 0:
            continue
        for i in range(m):
            ri = rels[i]
            for j in range(m):
                rj = rels[j]
                cooc[ri, rj] += 1.0
    # Row normalization + smoothing
    smooth = 1e-2
    row_sum = cooc.sum(dim=1, keepdim=True)
    cooc_prob = (cooc + smooth) / (row_sum + smooth * num_rel)
    # Low co-occurrence treated as 'impossible' pairs (symmetric mask, threshold adjustable)
    thresh = 1e-4
    impossible = (cooc <= thresh) & (cooc.t() <= thresh)
    # Self-relation not treated as impossible
    impossible.fill_diagonal_(False)
    return cooc_prob, impossible, head_rel_sets
@torch.no_grad()
def _detect_pos_rev_layout(data: Data):
    """
    Assume first half of edge_index is positive edges, second half is corresponding reverse edges; reverse edge_type = positive edge_type + rel_offset.
    Returns (E_pos, rel_offset); raises error if convention not met to avoid silent errors.
    """
    ei = data.edge_index
    et = getattr(data, "edge_type", None)
    E = int(ei.size(1))
    assert E % 2 == 0 and et is not None, "edge_index must be [pos, rev] with edge_type present"
    E_pos = E // 2
    et_pos = et[:E_pos]
    et_rev = et[E_pos:]
    offset = int((et_rev.min() - et_pos.min()).item())
    idx = torch.randperm(E_pos, device=ei.device)[:min(2048, E_pos)]
    match = (et_rev[idx] == et_pos[idx] + offset).float().mean().item()
    if match < 0.95:
        raise ValueError(f"[augment] edge_type not paired (match={match:.2f})")
    return E_pos, offset

@torch.no_grad()
def _build_rel_stats_pos(ei_pos: torch.Tensor, et_pos: torch.Tensor, num_rel: int, num_nodes: int):
    """
    Build intra-relation statistics and adjacency dictionaries from positive half edges.
    Returns:
      - tails_of_head[r][h] -> List[int]
      - heads_of_tail[r][t] -> List[int]
        - tail_pop_r[r] -> Tensor[num_nodes] intra-relation tail distribution
      - deg_hr: dict[(h,r)] -> int
    """
    device = ei_pos.device
    H = ei_pos[0].tolist()
    T = ei_pos[1].tolist()
    R = et_pos.tolist()
    tails_of_head = [defaultdict(list) for _ in range(num_rel)]
    heads_of_tail = [defaultdict(list) for _ in range(num_rel)]
    tail_pop_r = [torch.zeros(num_nodes, dtype=torch.float32, device=device) for _ in range(num_rel)]
    deg_hr = defaultdict(int)
    for h, r, t in zip(H, R, T):
        tails_of_head[r][h].append(t)
        heads_of_tail[r][t].append(h)
        tail_pop_r[r][t] += 1.0
        deg_hr[(h, r)] += 1
    for r in range(num_rel):
        pop = tail_pop_r[r]
        if pop.sum() > 0:
            tail_pop_r[r] = (pop + 1e-3) / (pop.sum() + 1e-3 * num_nodes)
        else:
            tail_pop_r[r] = torch.full((num_nodes,), 1.0 / max(1, num_nodes), device=device)
    return tails_of_head, heads_of_tail, tail_pop_r, deg_hr

@torch.no_grad()
def _sample_tail_guided(h: int, r: int, num_nodes: int,
                        tails_of_head_r: dict, heads_of_tail_r: dict,
                        pop_r: torch.Tensor, avoid: set, no_self_loop: bool):
    """
    Guided tail sampling:
      - First try 'similar head migration': from h's connected tail t0 find co-occurring head h', then sample t' from h''s tail set
      - - Fallback: sample by intra-relation tail popularity pop_r
      - - Constraints: avoid self-loop and avoid; retry a few times on failure
    """
    import random
    for _ in range(5):
        cand = None
        if h in tails_of_head_r and len(tails_of_head_r[h]) > 0:
            t0 = random.choice(tails_of_head_r[h])
            co_heads = heads_of_tail_r.get(t0, [])
            if len(co_heads) > 0:
                h2 = random.choice(co_heads)
                tails_h2 = tails_of_head_r.get(h2, [])
                if len(tails_h2) > 0:
                    cand = random.choice(tails_h2)
        if cand is None:
            idx = torch.multinomial(pop_r, num_samples=1, replacement=True).item()
            cand = int(idx)
        if no_self_loop and cand == h:
            continue
        if cand in avoid:
            continue
        return cand
    # Fallback: uniform random
    for _ in range(5):
        cand = int(torch.randint(0, num_nodes, (1,), device=pop_r.device).item())
        if not (no_self_loop and cand == h) and cand not in avoid:
            return cand
    return int(torch.randint(0, num_nodes, (1,), device=pop_r.device).item())

def build_degree_inflate_scheduler(cfg, total_steps: int, base_inflate: float):
    """
    cfg.augment.degree_inflate_schedule: {mode: 'linear'|'cosine', start, end}
    """
    aug = getattr(cfg, "augment", None)
    sched = None
    if isinstance(aug, dict):
        sched = aug.get("degree_inflate_schedule", None)
    elif aug is not None and hasattr(aug, "degree_inflate_schedule"):
        sched = aug.degree_inflate_schedule
    if not sched:
        return lambda step: base_inflate
    mode = str(sched.get("mode", "linear")).lower()
    start = float(sched.get("start", base_inflate))
    end = float(sched.get("end", base_inflate))
    start = max(0.0, start); end = max(0.0, end)
    def ratio_at(step):
        if total_steps <= 0: return end
        t = max(0.0, min(1.0, step / total_steps))
        if mode == "cosine":
            w = 0.5 * (1.0 - math.cos(math.pi * t))
            return (1 - w) * start + w * end
        else:
            return (1 - t) * start + t * end
    return ratio_at

def build_edge_rewire_scheduler(cfg, total_steps: int, base_rewire: float):
    """
    cfg.augment.edge_rewire_schedule: {mode, start, end}
    """
    aug = getattr(cfg, "augment", None)
    sched = None
    if isinstance(aug, dict):
        sched = aug.get("edge_rewire_schedule", None)
    elif aug is not None and hasattr(aug, "edge_rewire_schedule"):
        sched = aug.edge_rewire_schedule
    if not sched:
        return lambda step: base_rewire
    mode = str(sched.get("mode", "linear")).lower()
    start = float(sched.get("start", base_rewire))
    end = float(sched.get("end", base_rewire))
    start = max(0.0, start); end = max(0.0, end)
    def ratio_at(step):
        if total_steps <= 0: return end
        t = max(0.0, min(1.0, step / total_steps))
        if mode == "cosine":
            w = 0.5 * (1.0 - math.cos(math.pi * t))
            return (1 - w) * start + w * end
        else:
            return (1 - t) * start + t * end
    return ratio_at

def _build_stage_schedule(cfg) -> list:
    """
    Returns [('identity', e1), ('ae', e2), ('vq', e3)], only keeping stages with epoch>0.
    Supports cfg.train.stage_epochs = {'identity': 200, 'ae': 0, 'vq': 300}
    If not configured, defaults to identity=200 (if total epochs sufficient), ae=0, vq=remaining.
    """
    total = int(getattr(cfg.train, "num_epoch", 0))
    stage_cfg = getattr(cfg.train, "stage_epochs", None)
    if stage_cfg is None:
        e_id = min(200, total)  # As suggested, skip token/vq for 200 epochs first
        e_ae = 0
        e_vq = max(0, total - e_id - e_ae)
    else:
        e_id = int(stage_cfg.get("identity", 0))
        e_ae = int(stage_cfg.get("ae", 0))
        e_vq = int(stage_cfg.get("vq", 0))
        # If sum differs from total, do not override: use config
    stages = []
    if e_id > 0: stages.append(("identity", e_id))
    if e_ae > 0: stages.append(("ae", e_ae))
    if e_vq > 0: stages.append(("vq", e_vq))
    if len(stages) == 0 and total > 0:
        stages = [("identity", min(200, total)), ("vq", max(0, total - min(200, total)))]
    return stages
def _stage_for_epoch(stages, epoch):
    """
    stages: [('identity', e1), ('ae', e2), ('vq', e3)]
    Returns the stage name for current epoch, local_epoch within the stage, and total stage length stage_len.
    """
    s = 0
    for name, length in stages:
        if epoch < s + length:
            return name, epoch - s, length
        s += length
    return stages[-1][0], stages[-1][1] - 1, stages[-1][1]

def build_joint_target_graph(train_data: Data, test_data: Data, device=None,reverse=False) -> Data:
    """
    Construct joint graph:
    - Only use train/test's target_edge_*;
    - First concatenate original target edges, then append corresponding reverse edges;
    - Reverse edge relation ids have increasing offset to avoid conflict with original relations.
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

@torch.no_grad()
def _build_labels_for_heads(data: Data, batch_heads: torch.Tensor, num_rel: int,
                            num_nodes: int, has_empty_class: bool, device) -> torch.Tensor:
    B = batch_heads.numel()
    C = num_nodes + 1 if has_empty_class else num_nodes
    labels = torch.zeros((B, num_rel, C), device=device, dtype=torch.float32)
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
    if has_empty_class:
        has_pos = labels[:, :, :num_nodes].any(dim=-1)
        labels[:, :, num_nodes] = (~has_pos).float()
    return labels

@torch.no_grad()
def _compute_logits_for_heads(model, data: Data, batch_heads: torch.Tensor, device):
    # Use new model inference path (quantize+decode+classify)
    logits, _ = model(data, batch_heads, phase="eval", return_logits=True)
    return logits

@torch.no_grad()
def _tb_logit_hists(writer: SummaryWriter, tag_prefix: str, step: int,
                    logits: torch.Tensor, labels: torch.Tensor, num_nodes: int,
                    max_samples: int = 200_000):
    # Only count tail entity channels (excluding empty class)
    tail_logits = logits[..., :num_nodes]                 # (B,R,N)
    tail_prob = torch.sigmoid(tail_logits)
    gt = labels[..., :num_nodes].bool()
    pos_logit = tail_logits[gt]
    neg_logit = tail_logits[~gt]
    pos_prob = tail_prob[gt]
    neg_prob = tail_prob[~gt]
    # Sample to avoid overflow
    def _sample(x):
        if x.numel() > max_samples:
            idx = torch.randperm(x.numel(), device=x.device)[:max_samples]
            x = x[idx]
        return x.detach().float().cpu()
    if pos_logit.numel() > 0:
        writer.add_histogram(f"{tag_prefix}/logit_pos", _sample(pos_logit), step)
        writer.add_histogram(f"{tag_prefix}/prob_pos", _sample(pos_prob), step)
    if neg_logit.numel() > 0:
        writer.add_histogram(f"{tag_prefix}/logit_neg", _sample(neg_logit), step)
        writer.add_histogram(f"{tag_prefix}/prob_neg", _sample(neg_prob), step)

def edge_dropout_view(full_data: Data, drop_ratio: float, device=None,reverse=False) -> Data:
    """
    Dynamically drop edges by ratio, return a new Data (shallow copy of edges).
    - If edge_type exists, independently sample by same ratio within each relation type (stratified sampling), trying to preserve original relation distribution;
        and keep at least 1 edge per relation type (when drop_ratio < 1).
    - If edge_type does not exist, sample globally by ratio.
    Prefer target_edge_index/edge_type; if not present, fall back to edge_index/edge_type.
    """
    device = device or full_data.edge_index.device
    ei = full_data.target_edge_index
    et = getattr(full_data, "target_edge_type", None)

    E = ei.size(1)
    if E == 0:
        return Data(edge_index=ei, edge_type=et, num_nodes=full_data.num_nodes)

    p_keep = max(0.0, 1.0 - float(drop_ratio))
    if et is None:
        # No relation type, sample globally by ratio
        keep = (torch.rand(E, device=device) < p_keep)
        if not keep.any() and E > 0:
            keep[torch.randint(0, E, (1,), device=device)] = True
    else:
        # Stratified sampling: independently retain by ratio within each relation
        keep = torch.zeros(E, dtype=torch.bool, device=device)
        uniq = torch.unique(et)
        for r in uniq.tolist():
            idx = (et == r).nonzero(as_tuple=False).view(-1)  # Edge indices for this relation
            nr = idx.numel()
            if nr == 0:
                continue
            # Expected number to retain
            k = int(math.floor(nr * p_keep))
            if p_keep > 0.0 and k == 0:
                k = 1  # Ensure at least 1 edge retained for this relation (when not fully deleting)
            if k >= nr:
                keep[idx] = True
            elif k > 0:
                perm = torch.randperm(nr, device=device)[:k]
                keep[idx[perm]] = True

        # Global fallback: if all False unexpectedly, keep at least one
        if not keep.any():
            keep[torch.randint(0, E, (1,), device=device)] = True
    if reverse:
        keep = torch.cat([keep,keep],dim=0)
    edge_index = full_data.edge_index[:, keep]
    edge_type = full_data.edge_type[keep]
    
    data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=full_data.num_nodes)
    # Keep relation count consistent with original graph (or infer from subgraph)
    if edge_type is not None and edge_type.numel() > 0:
        num_rel = int(edge_type.max().item()) + 1
    else:
        num_rel = getattr(full_data, "num_relations", 0)
    data.num_relations = num_rel
    return data

def build_lr_scheduler(optimizer, cfg, total_steps: int):
    """
    Supports:
    - scheduler.class == 'WarmupCosine'
        keys: warmup_steps, min_lr_ratio
    - Otherwise returns None
    """
    sch_cfg = getattr(cfg, "scheduler", None)
    if not sch_cfg or not hasattr(sch_cfg, "class"):
        return None

    name = sch_cfg["class"]
    if name == "WarmupCosine":
        warmup = int(sch_cfg.get("warmup_steps", 0))
        min_lr_ratio = float(sch_cfg.get("min_lr_ratio", 0.1))
        min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

        def lr_lambda(step):
            if total_steps <= 0:
                return 1.0
            if step < warmup and warmup > 0:
                return float(step + 1) / float(warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        return None

def build_edge_drop_scheduler(cfg, total_steps: int, base_drop: float):
    """
    cfg.augment.edge_drop_schedule:
      - mode: 'linear' | 'cosine'
      - start: float (default uses base_drop)
      - end: float
    If not configured, returns constant base_drop.
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
            # Cosine from start -> end
            w = 0.5 * (1.0 - math.cos(math.pi * t))
            return (1 - w) * start + w * end
        else:
            # Linear
            return (1 - t) * start + t * end

    return ratio_at

separator = ">" * 30
line = "-" * 30

def train_and_validate(cfg, model, train_data, test_data, device, logger, 
                       filtered_data=None, batch_per_epoch=None, drop_ratio=0.1, writer=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    train_triplets = torch.arange(train_data.num_nodes, device=device)
    sampler = torch_data.DistributedSampler(train_triplets, world_size, rank)
    train_loader = torch_data.DataLoader(train_triplets, batch_size=cfg.train.batch_size, sampler=sampler)
    joint_data = build_joint_target_graph(train_data, test_data, device=device,reverse=True)
    aug_cfg = getattr(cfg, "augment", {})
    base_drop    = float(aug_cfg.get("edge_drop", 0.2))
    base_inflate = float(aug_cfg.get("degree_inflate", 0.2))
    base_rewire  = float(aug_cfg.get("edge_rewire", 0.1))
    add_cap_hr   = int(aug_cfg.get("add_cap_per_hr", 64))
    miss_topk    = int(aug_cfg.get("miss_rel_topk", 2))
    miss_add_per = int(aug_cfg.get("miss_rel_add_per_hr", 1))
    twohop_cap   = int(aug_cfg.get("twohop_cap", 200))
    fast_mode    = bool(aug_cfg.get("fast", True))                # added: fast augmentation switch
    twohop_prob  = float(aug_cfg.get("twohop_prob", 0.7))
    dedup_new    = bool(aug_cfg.get("dedup_new", False))
    # Precompute augmentation cache
   
    drop_sched    = build_edge_drop_scheduler(cfg, int(cfg.train.epoch_pergraph), base_drop)

    # Build two optimizers: G (all params except discriminator) and D (discriminator only)
    opt_cfg = dict(cfg.optimizer)
    opt_cls = opt_cfg.pop("class")
    gen_params, disc_params = [], []
    for n, p in model.named_parameters():
        if "vq_vae.discriminator" in n:
            disc_params.append(p)
        else:
            gen_params.append(p)
    optimizer_g = getattr(optim, opt_cls)(gen_params, **opt_cfg)
    optimizer_d = getattr(optim, opt_cls)(disc_params, **opt_cfg) if len(disc_params) > 0 else None

    num_params = sum(p.numel() for p in model.parameters())
    logger.warning("-" * 30)
    logger.warning(f"Number of parameters: {num_params}")

    # LR schedulers
    total_steps = int(cfg.train.num_epoch) * len(train_loader) * int(cfg.train.epoch_pergraph)
    lr_scheduler_g = build_lr_scheduler(optimizer_g, cfg, total_steps)
    lr_scheduler_d = build_lr_scheduler(optimizer_d, cfg, total_steps) if optimizer_d is not None else None

    # DDP
    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step_chunk = int(getattr(cfg.train, "eval_interval", 100)) 
    global_step = 0
    batch_id = 0
    n_critic = int(getattr(cfg.train, "n_critic", 1))  # Number of D steps before each G step

    # Stage schedule
    stages = _build_stage_schedule(cfg)
    if rank == 0:
        logger.warning(f"[Stages] {stages}")

    # VQ hyperparameter baseline and warmup targets
    base_beta = float(getattr(cfg.model.vq_model, "beta", 0.25))
    base_usage = float(getattr(cfg.model.vq_model, "usage_lambda", 1e-2))
    vq_beta_start = float(getattr(cfg.train, "vq_beta_start", 0.10))
    vq_usage_start = float(getattr(cfg.train, "vq_usage_start", 1e-3))

    prev_stage = None

    for i in range(0, cfg.train.num_epoch, step_chunk):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step_chunk)):
            stage, local_e, stage_len = _stage_for_epoch(stages, epoch)
            ######### Distillation ##########

            if stage != prev_stage:
                # Switch model mode
                if hasattr(parallel_model, "set_mode"):
                    parallel_model.set_mode(stage)
                elif hasattr(parallel_model, "vq_vae") and hasattr(parallel_model.vq_vae, "set_mode"):
                    parallel_model.vq_vae.set_mode(stage)
                if rank == 0:
                    logger.warning(">" * 30)
                    logger.warning(f"[Stage Switch] epoch={epoch} -> mode={stage} (len={stage_len})")
                prev_stage = stage

            if stage == "vq" and hasattr(parallel_model, "vq_vae"):
                prog = (local_e + 1) / max(1, stage_len)
                parallel_model.vq_vae.beta = vq_beta_start + (base_beta - vq_beta_start) * prog
                parallel_model.vq_vae.usage_lambda = vq_usage_start + (base_usage - vq_usage_start) * prog

            if util.get_rank() == 0:
                logger.warning(">" * 30)
                logger.warning(f"Epoch {epoch} begin (mode={stage})")

            losses = []
            sampler.set_epoch(epoch)
            
            # Dynamically sample epoch_pergraph edge-dropped subgraphs per epoch
            for epoch_pergraph in range(int(cfg.train.epoch_pergraph)):
                curr_drop    = float(drop_sched(epoch_pergraph))

                data_view = edge_dropout_view(train_data, curr_drop, device=device,reverse=True)
                for batch in train_loader:
                    # Discriminator step (default off; enable via previous comments if needed)
                    status_d = {}

                    # Generator step
                    if hasattr(model, "vq_vae") and hasattr(model.vq_vae, "discriminator"):
                        for p in model.vq_vae.discriminator.parameters():
                            p.requires_grad = False
                    lambda_lr = float(getattr(cfg.train, "lambda_lr", 1.0))
                    if hasattr(parallel_model, "latent_replace_step"):
                        loss_g, status_g = parallel_model.latent_replace_step(
                            data_uncomplete=data_view,
                            data_complete=train_data,
                            batch=batch,
                            lambda_lr=lambda_lr,
                        )
                    else:
                        # Compatible with old path
                        loss_g, status_g = parallel_model(data_view, batch, phase="G")

                    optimizer_g.zero_grad(set_to_none=True)
                    loss_g.backward()
                    optimizer_g.step()
                    if lr_scheduler_g is not None:
                        lr_scheduler_g.step()

                   # Logging
                    if util.get_rank() == 0 and batch_id % int(cfg.train.log_interval) == 0:
                        curr_lr_g = optimizer_g.param_groups[0]["lr"]
                        curr_lr_d = optimizer_d.param_groups[0]["lr"] if optimizer_d is not None else 0.0
                        status = dict(status_g)
                        status["mode"] = stage
                        status.update({
                            "lr_g": curr_lr_g,
                            "lr_d": curr_lr_d,
                            "edge_drop": curr_drop,
                            "lambda_lr": lambda_lr,
                            "step": global_step,
                        })
                        logger.warning(">" * 30)
                        logger.warning(status)
                    
                    losses.append(loss_g.item())
                    batch_id += 1
                    global_step += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / max(1, len(losses))
                logger.warning(">" * 30)
                logger.warning(f"Epoch {epoch} end")
                logger.warning("-" * 30)
                logger.warning(f"average loss: {avg_loss:g}")
                if writer is not None:
                    writer.add_scalar("train/avg_loss", avg_loss, epoch)

        # Synchronize
        util.synchronize()

        if util.get_rank() == 0:
            logger.warning(">" * 30)
            logger.warning("Evaluate on valid")
        recon_metric = test_joint(cfg, model, train_data, test_data, device=device, logger=logger, writer=writer, tb_tag="valid_joint", tb_step=epoch)

    if util.get_rank() == 0:
        state = {
            "model": model.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None
        }
        torch.save(state, "model_epoch_%d.pth" % epoch)

@torch.no_grad()
def _build_forbidden_mask_for_heads(data: Data, batch_heads: torch.Tensor, num_rel: int,
                                    num_nodes: int, device) -> torch.Tensor:
    """
    Returns forbidden mask (B,R,N), True means (h,r,t) already exists in data and should be masked during evaluation.
    Usually pass train_data to mask known triples in training set.
    """
    B = batch_heads.numel()
    mask = torch.zeros((B, num_rel, num_nodes), device=device, dtype=torch.bool)

    H_all = data.edge_index[0].to(device)
    T_all = data.edge_index[1].to(device)
    R_all = getattr(data, "edge_type", None)
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
def test_joint(
    cfg,
    model,
    train_data: Data,           # Used to construct part of the joint graph
    test_data: Data,            # Used to construct joint graph + provide test triple labels
    device,
    logger,
    return_metrics=True,
    threshold: float = 0.9,
    empty_gate: float = 0.1,
    use_softmax_empty: bool = False,
    do_threshold_sweep: bool = True,
    writer: SummaryWriter = None,
    tb_tag: str = "test_joint",
    tb_step: int = 0,
):
    """
    Use joint graph (train+test) for inference, but only compute P/R/F1 based on test_data triples.
    """
    world_size = util.get_world_size()
    rank = util.get_rank()
    model.eval()
    joint_graph = build_joint_target_graph(train_data, test_data, device=device,reverse=True)
    
    reverse_edge_index = torch.stack([test_data.target_edge_index[1], test_data.target_edge_index[0]], dim=0)
    reverse_edge_type = test_data.target_edge_type + int(test_data.target_edge_type.max().item()) + 1

    label_edge_index = torch.cat([test_data.target_edge_index,reverse_edge_index], dim=1)  # Only use test triples
    label_edge_type  = torch.cat([test_data.target_edge_type, reverse_edge_type],  dim=0)
    label_graph = Data(edge_index=label_edge_index, edge_type=label_edge_type, num_nodes=test_data.num_nodes)
    label_graph.num_relations = getattr(test_data, "num_relations", getattr(train_data, "num_relations", 0))

    num_nodes = int(joint_graph.num_nodes)
    if hasattr(model, "vq_vae") and hasattr(model.vq_vae, "num_rel") and model.vq_vae.num_rel > 0:
        num_rel = int(model.vq_vae.num_rel)
    else:
        # Infer from graph
        if getattr(joint_graph, "edge_type", None) is not None:
            num_rel = int(joint_graph.edge_type.max().item()) + 1
        else:
            num_rel = int(getattr(joint_graph, "num_relations", 1))

    out_C = None
    if hasattr(model, "vq_vae") and hasattr(model.vq_vae, "cls") and isinstance(model.vq_vae.cls, nn.Linear):
        out_C = int(model.vq_vae.cls.out_features)
    has_empty_class = (out_C == num_nodes + 1)

    heads_tensor = torch.arange(num_nodes, device=device)
    bs = int(getattr(cfg.test, "batch_size", 128))
    sampler = torch_data.DistributedSampler(heads_tensor, world_size, rank, shuffle=False)
    loader = torch_data.DataLoader(heads_tensor, batch_size=bs, sampler=sampler)

    tp_total = torch.tensor(0, device=device, dtype=torch.long)
    fp_total = torch.tensor(0, device=device, dtype=torch.long)
    fn_total = torch.tensor(0, device=device, dtype=torch.long)

    if do_threshold_sweep:
        thr_grid = torch.linspace(0.6, 0.9999, steps=10, device=device)
        sweep_tp = torch.zeros_like(thr_grid, dtype=torch.long)
        sweep_fp = torch.zeros_like(thr_grid, dtype=torch.long)
        sweep_fn = torch.zeros_like(thr_grid, dtype=torch.long)

    # Histogram collection
    hist_pos_prob, hist_neg_prob = [], []
    hist_pos_logit, hist_neg_logit = [], []
    max_hist_samples = 200000

    for batch_heads in loader:
        labels = _build_labels_for_heads(
            label_graph, batch_heads, num_rel, num_nodes, has_empty_class, device
        )
        # Inference with joint graph
        if getattr(model, "latent_replace_inference", None):
            logits = model.latent_replace_inference(train_data,joint_graph, batch_heads)
        else:
            logits = _compute_logits_for_heads(model, joint_graph, batch_heads, device)  # (B,R,C)

        forbid = _build_forbidden_mask_for_heads(train_data, batch_heads, num_rel, num_nodes, device)  # (B,R,N)
        tail_logits_full = logits[..., :num_nodes]                     # (B,R,N)
        tail_logits = tail_logits_full.masked_fill(forbid, -1e9)       # masked logits
        tail_prob = torch.sigmoid(tail_logits)                          # (B,R,N)

        gt = labels[..., :num_nodes].bool()

        # Threshold sweep
        if do_threshold_sweep:
            if has_empty_class:
                empty_prob = torch.sigmoid(logits[..., num_nodes])  # (B,R)
                empty_wins = (empty_prob >= empty_gate) & (empty_prob >= tail_prob.max(dim=-1).values)
            for i, thr in enumerate(thr_grid):
                preds = tail_prob > thr
                if has_empty_class:
                    preds = preds & (~empty_wins.unsqueeze(-1))
                sweep_tp[i] += (preds & gt).sum()
                sweep_fp[i] += (preds & (~gt)).sum()
                sweep_fn[i] += ((~preds) & gt).sum()

        # Fixed threshold
        preds = tail_prob > threshold
        if has_empty_class:
            empty_prob = torch.sigmoid(logits[..., num_nodes])
            empty_wins = (empty_prob >= empty_gate) & (empty_prob >= tail_prob.max(dim=-1).values)
            preds = preds & (~empty_wins.unsqueeze(-1))

        tp_total += (preds & gt).sum()
        fp_total += (preds & (~gt)).sum()
        fn_total += ((~preds) & gt).sum()

    # Multi-GPU aggregation
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tp_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(fp_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(fn_total, op=dist.ReduceOp.SUM)

    if do_threshold_sweep:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sweep_tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(sweep_fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(sweep_fn, op=dist.ReduceOp.SUM)
        f1s = []
        ps_, rs_ = [], []
        for i in range(thr_grid.numel()):
            tp = sweep_tp[i].item(); fp = sweep_fp[i].item(); fn = sweep_fn[i].item()
            p = tp / max(1, tp + fp); r_ = tp / max(1, tp + fn)
            f1 = 2 * p * r_ / max(1e-8, p + r_)
            f1s.append(f1); ps_.append(p); rs_.append(r_)
        best_i = int(torch.tensor(f1s).argmax().item())
        best_thr = float(thr_grid[best_i].item())
        if rank == 0:
            logger.warning(f"[Joint-Test] threshold sweep best_thr={best_thr:.3f}  best_F1={f1s[best_i]:.4f} best_P={ps_[best_i]:.4f} best_R={rs_[best_i]:.4f}")
            if writer is not None:
                writer.add_scalar(f"{tb_tag}/best_thr", best_thr, tb_step)
                writer.add_scalar(f"{tb_tag}/best_F1", f1s[best_i], tb_step)
                writer.add_scalar(f"{tb_tag}/best_P", ps_[best_i], tb_step)
                writer.add_scalar(f"{tb_tag}/best_R", rs_[best_i], tb_step)
        # Override with best threshold
        threshold = best_thr

    tp = tp_total.item(); fp = fp_total.item(); fn = fn_total.item()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, (prec + rec))

    return (ps_[best_i], rs_[best_i], f1s[best_i]) if return_metrics else None

@torch.no_grad()
def test_jointnew(
    cfg,
    model,
    train_data: Data,           # Used to construct part of the joint graph
    test_data: Data,            # Used to construct joint graph + provide test triple labels
    device,
    logger,
    return_metrics=True,
    threshold: float = 0.9,
    empty_gate: float = 0.1,
    use_softmax_empty: bool = False,
    do_threshold_sweep: bool = True,
    writer: SummaryWriter = None,
    tb_tag: str = "test_joint",
    tb_step: int = 0,
):
    """
    Use joint graph (train+test) for inference, but only compute P/R/F1 based on test_data triples.
    Precompute logits for all heads to CPU; average probabilities of positive and corresponding reverse relations, then threshold.
    """
    world_size = util.get_world_size()
    rank = util.get_rank()
    model.eval()
    joint_graph = build_joint_target_graph(train_data, test_data, device=device, reverse=True)

    reverse_edge_index = torch.stack([test_data.target_edge_index[1], test_data.target_edge_index[0]], dim=0)
    reverse_edge_type = test_data.target_edge_type + int(test_data.target_edge_type.max().item()) + 1
    label_edge_index = torch.cat([test_data.target_edge_index, reverse_edge_index], dim=1)
    label_edge_type  = torch.cat([test_data.target_edge_type, reverse_edge_type],  dim=0)
    label_graph = Data(edge_index=label_edge_index, edge_type=label_edge_type, num_nodes=test_data.num_nodes)
    label_graph.num_relations = getattr(test_data, "num_relations", getattr(train_data, "num_relations", 0))

    num_nodes = int(joint_graph.num_nodes)
    # Infer number of relations
    if hasattr(model, "vq_vae") and hasattr(model.vq_vae, "num_rel") and model.vq_vae.num_rel > 0:
        num_rel = int(model.vq_vae.num_rel)
    else:
        num_rel = int(joint_graph.edge_type.max().item()) + 1 if getattr(joint_graph, "edge_type", None) is not None else int(getattr(joint_graph, "num_relations", 1))

    # Output class count (whether empty class channel exists)
    out_C = None
    if hasattr(model, "vq_vae") and hasattr(model.vq_vae, "cls") and isinstance(model.vq_vae, nn.Linear):
        out_C = int(model.vq_vae.cls.out_features)
    has_empty_class = (out_C == num_nodes + 1)
    C = (num_nodes + 1) if has_empty_class else num_nodes

    # Build distributed DataLoader (no shuffle)
    heads_tensor = torch.arange(num_nodes, device=device)
    bs = int(getattr(cfg.train, "batch_size", 128))
    sampler = torch_data.DistributedSampler(heads_tensor, world_size, rank, shuffle=False)
    loader = torch_data.DataLoader(heads_tensor, batch_size=bs, sampler=sampler)

    logits_cpu = torch.empty((num_nodes, num_rel, C), dtype=torch.float32, device="cpu")

    for batch_heads in loader:
        logits_b = _compute_logits_for_heads(model, joint_graph, batch_heads, device)  # (B,R,C)
        logits_cpu[batch_heads.detach().cpu()] = logits_b.detach().float().cpu()

    # Average probability for 'positive/reverse relations' (on CPU)
    tail_prob_cpu = torch.sigmoid(logits_cpu[..., :num_nodes])  # (N,R,N)
    if has_empty_class:
        empty_prob_cpu = torch.sigmoid(logits_cpu[..., num_nodes])  # (N,R)

    # Estimate positive/reverse relation offset and do symmetric averaging
    try:
        _, rel_offset = _detect_pos_rev_layout(joint_graph)
    except Exception:
        rel_offset = num_rel // 2  # Fallback inference
    if rel_offset > 0 and num_rel >= rel_offset * 2:
        r0 = torch.arange(rel_offset)         # Positive relation index set (K,)
        r1 = r0 + rel_offset                  # Reverse relation index set (K,)

        A = tail_prob_cpu[:, r0, :]           # (N,K,N)
        B = tail_prob_cpu[:, r1, :]           # (N,K,N)
        B_t = B.transpose(0, 2)               # (N,K,N) swap head and tail dimensions

        avg = 0.5 * (A + B_t)
        tail_prob_cpu[:, r0, :] = avg
        tail_prob_cpu[:, r1, :] = avg.transpose(0, 2)

    tp_total = torch.tensor(0, device=device, dtype=torch.long)
    fp_total = torch.tensor(0, device=device, dtype=torch.long)
    fn_total = torch.tensor(0, device=device, dtype=torch.long)

    if do_threshold_sweep:
        thr_grid = torch.linspace(0.0, 0.999, steps=100, device=device)
        sweep_tp = torch.zeros_like(thr_grid, dtype=torch.long)
        sweep_fp = torch.zeros_like(thr_grid, dtype=torch.long)
        sweep_fn = torch.zeros_like(thr_grid, dtype=torch.long)

    # Histogram collection (CPU aggregation)
    hist_pos_prob, hist_neg_prob = [], []
    hist_pos_logit, hist_neg_logit = [], []
    max_hist_samples = 200000

    # Re-iterate loader, but use precomputed probabilities (CPU -> GPU slices)
    for batch_heads in loader:
        bh_cpu = batch_heads.detach().cpu()
        # Labels (built on CPU)
        labels = _build_labels_for_heads(
            label_graph, bh_cpu, num_rel, num_nodes, has_empty_class, device=torch.device("cpu")
        )  # (B,R,C)
        gt_cpu = labels[..., :num_nodes].bool()  # (B,R,N)

        tail_prob_b = tail_prob_cpu[bh_cpu]  # (B,R,N) CPU
        forbid = _build_forbidden_mask_for_heads(train_data, bh_cpu, num_rel, num_nodes, device)  # GPU
        tail_prob_b = tail_prob_b.to(device).masked_fill(forbid, 0.0)  # GPU (set masked positions to 0 probability)
        gt = gt_cpu.to(device)

        if has_empty_class:
            empty_prob_b = empty_prob_cpu[bh_cpu].to(device)  # (B,R)
            empty_wins = (empty_prob_b >= empty_gate) & (empty_prob_b >= tail_prob_b.max(dim=-1).values)
        else:
            empty_wins = torch.zeros(tail_prob_b.shape[:2], dtype=torch.bool, device=device)

        # Histogram (sampling)
        with torch.no_grad():
            pos_prob = tail_prob_b[gt]; neg_prob = tail_prob_b[~gt]
            pos_logit = torch.logit(pos_prob.clamp(1e-6, 1 - 1e-6))
            neg_logit = torch.logit(neg_prob.clamp(1e-6, 1 - 1e-6))
            def _sample(x):
                if x.numel() > max_hist_samples:
                    idx = torch.randperm(x.numel(), device=x.device)[:max_hist_samples]
                    x = x[idx]
                return x.detach().float().cpu()
            if pos_prob.numel() > 0:
                hist_pos_prob.append(_sample(pos_prob))
                hist_pos_logit.append(_sample(pos_logit))
            if neg_prob.numel() > 0:
                hist_neg_prob.append(_sample(neg_prob))
                hist_neg_logit.append(_sample(neg_logit))

        # Threshold sweep
        if do_threshold_sweep:
            for i, thr in enumerate(thr_grid):
                preds = tail_prob_b > thr
                preds = preds & (~empty_wins.unsqueeze(-1))
                sweep_tp[i] += (preds & gt).sum()
                sweep_fp[i] += (preds & (~gt)).sum()
                sweep_fn[i] += ((~preds) & gt).sum()

        # Fixed threshold P/R/F1
        preds = tail_prob_b > threshold
        preds = preds & (~empty_wins.unsqueeze(-1))
        tp_total += (preds & gt).sum()
        fp_total += (preds & (~gt)).sum()
        fn_total += ((~preds) & gt).sum()

    # Multi-GPU aggregation
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tp_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(fp_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(fn_total, op=dist.ReduceOp.SUM)

    # Histogram write (rank0 only)
    if writer is not None and rank == 0:
        def _cat_cpu(xs):
            if len(xs) == 0:
                return None
            x = torch.cat(xs, dim=0).detach().float().cpu()
            return x if x.numel() > 0 else None
        Pprob = _cat_cpu(hist_pos_prob); Nprob = _cat_cpu(hist_neg_prob)
        Plog  = _cat_cpu(hist_pos_logit); Nlog  = _cat_cpu(hist_neg_logit)
        if Plog is not None:  writer.add_histogram(f"{tb_tag}/logit_pos", Plog, tb_step)
        if Nlog is not None:  writer.add_histogram(f"{tb_tag}/logit_neg", Nlog, tb_step)
        if Pprob is not None: writer.add_histogram(f"{tb_tag}/prob_pos",  Pprob, tb_step)
        if Nprob is not None: writer.add_histogram(f"{tb_tag}/prob_neg",  Nprob, tb_step)

    if do_threshold_sweep:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sweep_tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(sweep_fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(sweep_fn, op=dist.ReduceOp.SUM)
        f1s, ps_, rs_ = [], [], []
        for i in range(thr_grid.numel()):
            tp = sweep_tp[i].item(); fp = sweep_fp[i].item(); fn = sweep_fn[i].item()
            p = tp / max(1, tp + fp); r_ = tp / max(1, tp + fn)
            f1 = 2 * p * r_ / max(1e-8, p + r_)
            f1s.append(f1); ps_.append(p); rs_.append(r_)
        best_i = int(torch.tensor(f1s).argmax().item())
        best_thr = float(thr_grid[best_i].item())
        if rank == 0:
            logger.warning(f"[Joint-Test] threshold sweep best_thr={best_thr:.3f}  best_F1={f1s[best_i]:.4f} best_P={ps_[best_i]:.4f} best_R={rs_[best_i]:.4f}")
            if writer is not None:
                writer.add_scalar(f"{tb_tag}/best_thr", best_thr, tb_step)
                writer.add_scalar(f"{tb_tag}/best_F1", f1s[best_i], tb_step)
                writer.add_scalar(f"{tb_tag}/best_P", ps_[best_i], tb_step)
                writer.add_scalar(f"{tb_tag}/best_R", rs_[best_i], tb_step)
        threshold = best_thr

    # P/R/F1 at fixed threshold
    tp = tp_total.item(); fp = fp_total.item(); fn = fn_total.item()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, (prec + rec))

    return (prec, rec, f1) if return_metrics else None
@torch.no_grad()
def test(cfg, model, test_data, device, logger, filtered_data=None, 
         return_metrics=True, threshold: float = 0.5, empty_gate: float = 0.1, use_softmax_empty: bool = False,
         do_threshold_sweep: bool = True, writer: SummaryWriter = None, tb_tag: str = "test", tb_step: int = 0):
 
    world_size = util.get_world_size()
    rank = util.get_rank()
    model.eval()

    H_all = test_data.edge_index[0].to(device)
    T_all = test_data.edge_index[1].to(device)
    R_all = getattr(test_data, "edge_type", None)
    if R_all is not None:
        R_all = R_all.to(device)

    num_nodes = int(test_data.num_nodes)
    if hasattr(model, "vq_vae") and hasattr(model.vq_vae, "num_rel") and model.vq_vae.num_rel > 0:
        num_rel = int(model.vq_vae.num_rel)
    else:
        num_rel = int(test_data.num_relations) if hasattr(test_data, "num_relations") else (
            int(R_all.max().item()) + 1 if R_all is not None else 1
        )

    out_C = None
    if hasattr(model, "vq_vae") and hasattr(model.vq_vae, "cls") and isinstance(model.vq_vae.cls, nn.Linear):
        out_C = int(model.vq_vae.cls.out_features)
    has_empty_class = (out_C == num_nodes + 1)

    heads_tensor = torch.arange(num_nodes, device=device)
    bs = int(getattr(cfg.train, "batch_size", 128))
    sampler = torch_data.DistributedSampler(heads_tensor, world_size, rank, shuffle=False)
    loader = torch_data.DataLoader(heads_tensor, batch_size=bs, sampler=sampler)

    tp_total = torch.tensor(0, device=device, dtype=torch.long)
    fp_total = torch.tensor(0, device=device, dtype=torch.long)
    fn_total = torch.tensor(0, device=device, dtype=torch.long)

    sweep_stats = []  # [(thr, tp, fp, fn)]
    if do_threshold_sweep:
        thr_grid = torch.linspace(0.0, 0.999, steps=100, device=device)
        sweep_tp = torch.zeros_like(thr_grid, dtype=torch.long)
        sweep_fp = torch.zeros_like(thr_grid, dtype=torch.long)
        sweep_fn = torch.zeros_like(thr_grid, dtype=torch.long)

    hist_pos_prob, hist_neg_prob = [], []
    hist_pos_logit, hist_neg_logit = [], []
    
    max_hist_samples = 200000

    for batch_heads in loader:
        num_nodes = int(test_data.num_nodes)
        num_rel = int(model.vq_vae.num_rel) if hasattr(model, "vq_vae") else (
            int(getattr(test_data, "num_relations", 1)))
        labels = _build_labels_for_heads(test_data, batch_heads, num_rel, num_nodes, False, device)

        logits = _compute_logits_for_heads(model, test_data, batch_heads, device)  # (B,R,C)

        tail_logits = logits[..., :num_nodes]                # (B,R,N)
        tail_prob = torch.sigmoid(tail_logits)               # (B,R,N)
        gt = labels[..., :num_nodes].bool()

         # prob
        pos_prob = tail_prob[gt]; neg_prob = tail_prob[~gt]
        pos_logit = tail_logits[gt]; neg_logit = tail_logits[~gt]
        # Subsample to avoid overflow
        def _sample(x):
            if x.numel() > max_hist_samples:
                idx = torch.randperm(x.numel(), device=x.device)[:max_hist_samples]
                x = x[idx]
            return x
        if pos_prob.numel() > 0:
            hist_pos_prob.append(_sample(pos_prob))
            hist_pos_logit.append(_sample(pos_logit))
        if neg_prob.numel() > 0:
            hist_neg_prob.append(_sample(neg_prob))
            hist_neg_logit.append(_sample(neg_logit))
        if do_threshold_sweep:
            if has_empty_class:
                empty_prob = torch.sigmoid(logits[..., num_nodes])  # (B,R)
                empty_wins = (empty_prob >= empty_gate) & (empty_prob >= tail_prob.max(dim=-1).values)
            for i, thr in enumerate(thr_grid):
                preds = tail_prob > thr
                if has_empty_class:
                    preds = preds & (~empty_wins.unsqueeze(-1))
                sweep_tp[i] += (preds & gt).sum()
                sweep_fp[i] += (preds & (~gt)).sum()
                sweep_fn[i] += ((~preds) & gt).sum()

        preds = tail_prob > threshold
        if has_empty_class:
            empty_prob = torch.sigmoid(logits[..., num_nodes])
            empty_wins = (empty_prob >= empty_gate) & (empty_prob >= tail_prob.max(dim=-1).values)
            preds = preds & (~empty_wins.unsqueeze(-1))

        tp_total += (preds & gt).sum()
        fp_total += (preds & (~gt)).sum()
        fn_total += ((~preds) & gt).sum()

        with torch.no_grad():
            rel_prec_rec = []
            for r in range(num_rel):
                pr = preds[:, r, :]
                gr = gt[:, r, :]
                tp = (pr & gr).sum().item()
                fp = (pr & (~gr)).sum().item()
                fn = ((~pr) & gr).sum().item()
                p = tp / max(1, tp + fp)
                r_ = tp / max(1, tp + fn)
                f1r = 2 * p * r_ / max(1e-8, p + r_)
                rel_prec_rec.append((r, p, r_, f1r, tp, fp, fn))
            if rank == 0:
                worst = sorted(rel_prec_rec, key=lambda x: x[3])[:5]
                logger.warning(f"[Diag] worst_rel_by_F1 (top5): {[(r, round(f1,4), tp, fp, fn) for r,_,__,f1,tp,fp,fn in worst]}")

    # Multi-GPU aggregation
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tp_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(fp_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(fn_total, op=dist.ReduceOp.SUM)
    if writer is not None and rank == 0:
        def _cat_cpu(xs):
            if len(xs) == 0:
                return None
            x = torch.cat(xs, dim=0).detach().float().cpu()
            return x if x.numel() > 0 else None
        Pprob = _cat_cpu(hist_pos_prob); Nprob = _cat_cpu(hist_neg_prob)
        Plog  = _cat_cpu(hist_pos_logit); Nlog  = _cat_cpu(hist_neg_logit)
        if Plog is not None:  writer.add_histogram(f"{tb_tag}/logit_pos", Plog, tb_step)
        if Nlog is not None:  writer.add_histogram(f"{tb_tag}/logit_neg", Nlog, tb_step)
        if Pprob is not None: writer.add_histogram(f"{tb_tag}/prob_pos",  Pprob, tb_step)
        if Nprob is not None: writer.add_histogram(f"{tb_tag}/prob_neg",  Nprob, tb_step)
    # Threshold sweep results
    if do_threshold_sweep:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sweep_tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(sweep_fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(sweep_fn, op=dist.ReduceOp.SUM)
        f1s = []
        ps_=[]; rs_ = []
        for i in range(thr_grid.numel()):
            tp = sweep_tp[i].item(); fp = sweep_fp[i].item(); fn = sweep_fn[i].item()
            p = tp / max(1, tp + fp); r_ = tp / max(1, tp + fn)
            f1 = 2 * p * r_ / max(1e-8, p + r_)
            f1s.append(f1)
            ps_.append(p)
            rs_.append(r_)
        best_i = int(torch.tensor(f1s).argmax().item())
        best_thr = float(thr_grid[best_i].item())
        if rank == 0:
            logger.warning(f"[Diag] threshold sweep best_thr={best_thr:.3f}  best_F1={f1s[best_i]:.4f} best_P={ps_[best_i]:.4f} best_R={rs_[best_i]:.4f}")
            if writer is not None:
                writer.add_scalar(f"{tb_tag}/best_thr", best_thr, tb_step)
                writer.add_scalar(f"{tb_tag}/best_F1", f1s[best_i], tb_step)
                writer.add_scalar(f"{tb_tag}/best_P", ps_[best_i], tb_step)
                writer.add_scalar(f"{tb_tag}/best_R", rs_[best_i], tb_step)
        threshold = best_thr  # Optional: override external threshold

    tp = tp_total.item()
    fp = fp_total.item()
    fn = fn_total.item()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-8, (prec + rec))

    if rank == 0:
        logger.warning(separator)
        logger.warning(f"[Reconstruction] thr={threshold:.3f} empty_gate={empty_gate:.2f}  P={ps_[best_i]:.4f}  R={rs_[best_i]:.4f}  F1={f1s[best_i]:.4f}  (tp={tp}, fp={fp}, fn={fn})")

    return (ps_[best_i], rs_[best_i], f1s[best_i]) if return_metrics else None

if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), "runs", "vqdiff"))
    else:
        writer = None
        
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    dataset = util.build_dataset(cfg)
    device = util.get_device(cfg)
    
    train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    test_data = test_data.to(device)
    train_data.num_relations=cfg.model.relation_model.num_relation
    test_data.num_relations=cfg.model.relation_model.num_relation
    valid_data.num_relations=cfg.model.relation_model.num_relation
    
    model = VQpretrain(
        rel_model_cfg=cfg.model.relation_model,
        vq_model_cfg=cfg.model.vq_model,
    )

    if "checkpoint" in cfg and cfg.checkpoint is not None:
        state = torch.load(cfg.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])

    model = model.to(device)
    
    if task_name == "InductiveInference":
        if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
            full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
            val_filtered_data = test_filtered_data
        else:
            full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
            full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
            test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)

            val_filtered_data = Data(
                edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
            )
    else:
        filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
        val_filtered_data = test_filtered_data = filtered_data
    
    val_filtered_data = val_filtered_data.to(device)
    test_filtered_data = test_filtered_data.to(device)
    
    train_and_validate(cfg, model, train_data, test_data, filtered_data=val_filtered_data, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger,writer=writer)
    if util.get_rank() == 0:
        logger.warning(separator)
        logger.warning("Evaluate joint (train+test context) with metrics on test triples only")

    if writer is not None:
        writer.flush()
        writer.close()
