import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch import distributed as ddpist
from torch_geometric.nn import RGCNConv



class HeadAdjacencyEncoder(nn.Module):
    """
    Encode the head-entity adjacency slice (relation x tail) into a dense vector.
    - Input: data(edge_index, edge_type), batch=(B,)
    - Output: head representation (B, num_rel, out_dim)
    Enhancements:
      - Optional adjacency feature dropout (edge_dropout) for robustness
      - MLP + LayerNorm + Dropout structure for stable training
    """
    def __init__(
        self,
        num_nodes: int,
        num_rel: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        edge_dropout: float = 0.0,
        # New: relation-attention branch config
        attn_enabled: bool = True,
        tail_embed_dim: int = 64,      # Map N-dim tail multi-hot to this dimension
        attn_dim: int = 128,           # Channel dim for relation tokens
        attn_heads: int = 4,
        attn_layers: int = 1,
        attn_dropout: float = 0.1,
        # New: reasoning branch config
        reasoning_enabled: bool = False,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_rel = int(num_rel)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else max(self.out_dim * 2, 512)
        self.edge_dropout = float(edge_dropout)

        # MLP now processes each relation independently (N -> out_dim), not flattened (R*N)
        self.projector = nn.Sequential(
            nn.Linear(self.num_nodes, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )

        # -------- New: reasoning branch (RGCN) --------
        self.reasoning_enabled = reasoning_enabled
        if self.reasoning_enabled:
            # Node initial embedding: each node gets a learnable vector
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.out_dim))
            nn.init.xavier_uniform_(self.node_emb)

            # RGCN layer: aggregate neighbor info on the graph
            # num_bases=4 reduces parameters via basis decomposition to prevent overfitting
            self.rgcn = RGCNConv(self.out_dim, self.out_dim, self.num_rel, num_bases=4)

            # Relation embedding: used to expand node features into (Head, Relation) features
            self.rel_emb_reasoning = nn.Parameter(torch.empty(self.num_rel, self.out_dim))
            nn.init.xavier_uniform_(self.rel_emb_reasoning)

            self.reasoning_norm = nn.LayerNorm(self.out_dim)

        # -------- Relation-dim self-attention branch (can be disabled) --------
        self.attn_enabled = bool(attn_enabled)
        if self.attn_enabled:
            self.tail_embed_dim = int(tail_embed_dim)
            self.attn_dim = int(attn_dim)
            self.attn_heads = int(attn_heads)
            self.attn_layers = int(attn_layers)
            self.attn_dropout = float(attn_dropout)

            # Learnable tail basis: compress (B,R,N) multi-hot to (B,R,tail_embed_dim)
            self.tail_basis = nn.Parameter(torch.empty(self.num_nodes, self.tail_embed_dim))
            nn.init.normal_(self.tail_basis, std=1.0 / math.sqrt(max(1, self.tail_embed_dim)))

            # Map tail_embed to attention dimension
            self.rel_token_proj = nn.Sequential(
                nn.Linear(self.tail_embed_dim, self.attn_dim),
                nn.LayerNorm(self.attn_dim),
                nn.SiLU(),
            )

            # Stack of relation-dim Transformer blocks (self-attention over R tokens)
            self.rel_attn_blocks = nn.ModuleList()
            for _ in range(self.attn_layers):
                self.rel_attn_blocks.append(_RelSelfAttnBlock(self.attn_dim, self.attn_heads, self.attn_dropout))

            # Map attention branch output to out_dim
            # No pooling, keep (B, R, out_dim)
            self.attn_out = nn.Sequential(
                nn.Linear(self.attn_dim, self.out_dim),
                nn.LayerNorm(self.out_dim),
            )

            # Residual gating (learnable scalar, weight after sigmoid)
            self._fuse_logit = nn.Parameter(torch.tensor(-1.0))  # Small initial weight, gradually learns fusion


    def _select_edges(self, data, device):
        edge_index = data.edge_index
        edge_type = getattr(data, "edge_type", None)
        if edge_index.device != device:
            edge_index = edge_index.to(device)
        if edge_type is not None and edge_type.device != device:
            edge_type = edge_type.to(device)
        return edge_index.long(), edge_type.long()

    def forward(self, data, batch: torch.Tensor) -> torch.Tensor:
        device = batch.device
        batch = batch.long()
        B = batch.numel()
        edge_index, edge_type = self._select_edges(data, device)
        head_index = edge_index[0]
        tail_index = edge_index[1]

        # Map heads in batch to [0..B-1]
        head2batch = torch.full((self.num_nodes,), -1, dtype=torch.long, device=device)
        head2batch[batch] = torch.arange(B, device=device, dtype=torch.long)

        assign = head2batch[head_index]
        valid = assign >= 0

        # Build per-head (B, R, N) sparse indicator vector
        adj = torch.zeros(B, self.num_rel, self.num_nodes, device=device, dtype=torch.float32)
        if valid.any():
            b_idx = assign[valid]
            rel_idx = edge_type[valid] % self.num_rel
            rel_idx = rel_idx.clamp_min(0).clamp_max(self.num_rel - 1)
            tail_idx = tail_index[valid].clamp_min(0).clamp_max(self.num_nodes - 1)
            # Directly fill 3D tensor
            adj[b_idx, rel_idx, tail_idx] = 1.0

        # Adjacency feature dropout (training only)
        if self.edge_dropout > 0.0 and self.training:
            adj = F.dropout(adj, p=self.edge_dropout)

        # Main MLP path
        # Input (B, R, N) -> Flatten (B*R, N) -> MLP -> (B*R, out_dim) -> (B, R, out_dim)
        adj_flat = adj.view(B * self.num_rel, self.num_nodes)
        mlp_out = self.projector(adj_flat)
        mlp_out = mlp_out.view(B, self.num_rel, self.out_dim)

        # -------- New: reasoning branch (RGCN) --------
        if self.reasoning_enabled:
            # 1. Full-graph RGCN update: each node aggregates neighbor info
            # x: (Num_Nodes, Out_Dim)
            x = self.node_emb
            if edge_type is not None:
                x = self.rgcn(x, edge_index, edge_type)

            # 2. Extract representations for head nodes in current batch
            # h_vec: (B, Out_Dim)
            h_vec = x[batch]

            # 3. Combine (Head, Relation) -> Head + Relation
            # r_vec: (Num_Rel, Out_Dim)
            r_vec = self.rel_emb_reasoning

            # Broadcasting: (B, 1, D) + (1, R, D) -> (B, R, D)
            # Generate reasoning features for each head over all relations
            reasoning_feat = h_vec.unsqueeze(1) + r_vec.unsqueeze(0)
            reasoning_feat = self.reasoning_norm(reasoning_feat)

            # 4. Fusion: add reasoning features to MLP features
            mlp_out = mlp_out + reasoning_feat

        # Relation-dim attention branch (optional)
        if self.attn_enabled:
            # (B, R, N)
            adj3 = adj  # Already (B, R, N)
            # Map each relation's tail multi-hot to low-dim token (sqrt-normalized by degree to avoid high-degree heads dominating)
            # (B,R,N) @ (N,tail_embed_dim) -> (B,R,tail_embed_dim)
            rel_token = adj3 @ self.tail_basis
            deg = adj3.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B,R,1)
            rel_token = rel_token / deg.sqrt()
            rel_token = self.rel_token_proj(rel_token)  # (B,R,attn_dim)

            # Self-attention stack (only over R dimension)
            for blk in self.rel_attn_blocks:
                rel_token = blk(rel_token)

            # No pooling, keep per-relation token
            # rel_token: (B, R, attn_dim)
            attn_out = self.attn_out(rel_token)  # (B, R, out_dim)

            fuse_alpha = torch.sigmoid(self._fuse_logit)  # Scalar
            out = mlp_out + fuse_alpha * attn_out
            return out  # (B, R, out_dim)

        return mlp_out

class _RelSelfAttnBlock(nn.Module):
    """
    Lightweight Transformer block acting on relation-dim tokens:
      x -> LN -> MHA -> Drop -> Residual -> LN -> FFN -> Drop -> Residual
    Uses batch_first=True, input (B, R, D)
    """
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(max(1, heads))
        self.dropout = float(dropout)

        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = nn.MultiheadAttention(self.dim, self.heads, dropout=self.dropout, batch_first=True)
        self.drop1 = nn.Dropout(self.dropout)

        self.norm2 = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(4 * self.dim, self.dim),
        )
        self.drop2 = nn.Dropout(self.dropout)

        # Small initialization for stable training
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop1(attn_out)
        y = self.norm2(x)
        y = self.ffn(y)
        x = x + self.drop2(y)
        return x

class InverseAdjacencyDecoder(nn.Module):
    """
    Map head vectors back to per-relation x per-node tail entity logits.
    Combines two paths for better generalization:
      1) MLP path: head -> (R*N) -> (R, N)
      2) Factorized bilinear path: head -> (R, k), multiplied with shared tail basis E_tail(N, k)
         logits_factorized[b, r, n] = <A[b, r, :], E_tail[n, :]>
    Final logits = mix_alpha * logits_mlp + (1 - mix_alpha) * logits_fact + biases
    """
    def __init__(
        self,
        num_nodes: int,
        num_rel: int,
        head_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        has_empty_class: bool = False,   # Kept for compatibility, actually ignored
        factor_rank: int = 64,
        mix_init: float = 0.5,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_rel = int(num_rel)
        self.head_dim = int(head_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else max(self.head_dim * 2, 512)

        # 1) MLP decode path
        # Input (D) -> Output (N) (per relation)
        self.deprojector = nn.Sequential(
            nn.Linear(self.head_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_nodes),
        )

        # 2) Factorized decode path
        self.factor_rank = int(max(4, factor_rank))
        self.tail_basis = nn.Parameter(torch.empty(self.num_nodes, self.factor_rank))
        nn.init.normal_(self.tail_basis, std=1.0 / math.sqrt(self.factor_rank))

        # Input (D) -> Output (k) (per relation)
        self.rel_factor = nn.Linear(self.head_dim, self.factor_rank)
        nn.init.xavier_uniform_(self.rel_factor.weight)
        nn.init.zeros_(self.rel_factor.bias)

        # Learnable mixing coefficient (sigmoid)
        self._mix_logit = nn.Parameter(torch.tensor(float(mix_init)).logit())

        # Lightweight bias (for calibration)
        # Input (D) -> Output (1) (per relation scalar bias)
        self.rel_bias_mlp = nn.Linear(self.head_dim, 1)
        nn.init.zeros_(self.rel_bias_mlp.weight)
        nn.init.zeros_(self.rel_bias_mlp.bias)

        self.tail_bias = nn.Parameter(torch.zeros(self.num_nodes))
        # Compatibility flag, empty class not actually used
        self.has_empty_class = False

    def forward(self, head_vec: torch.Tensor) -> torch.Tensor:
        """
        Input: head_vec (B, R, D)
        Output: logits (B, R, N)
        """
        B, R, D = head_vec.shape

        # MLP path
        # (B, R, D) -> (B, R, N)
        logits_mlp = self.deprojector(head_vec)

        # Factorized path
        # (B, R, D) -> (B, R, k)
        rel_factors = self.rel_factor(head_vec)
        # (B,R,k) @ (N,k).T -> (B,R,N)
        logits_fact = torch.einsum("brk,nk->brn", rel_factors, self.tail_basis)

        # Mix
        mix_alpha = torch.sigmoid(self._mix_logit)  # Scalar
        logits = mix_alpha * logits_mlp + (1.0 - mix_alpha) * logits_fact

        # Add lightweight bias
        rb = self.rel_bias_mlp(head_vec)  # (B,R,1)
        tb = self.tail_bias.view(1, 1, self.num_nodes)  # (1,1,N)
        logits = logits + rb + tb

        return logits

    def forward_sparse(self, head_vec: torch.Tensor, b_idx: torch.Tensor, r_idx: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        """
        Sparse evaluation: compute logits only on specified (b_idx, r_idx, t_idx) triples to avoid huge B*R*N memory cost.
        Input:
          head_vec: (B, R, D)
          b_idx, r_idx, t_idx: (E,) 1D index tensors
        Output:
          logits_sparse: (E,)
        """
        # --- MLP path ---
        h = head_vec[b_idx, r_idx]  # (E, D)
        # Derive through deprojector up to second-last layer
        for layer in self.deprojector[:-1]:
            h = layer(h)
        # Last layer is nn.Linear(hidden_dim, num_nodes)
        last_layer = self.deprojector[-1]
        weight = last_layer.weight[t_idx]  # (E, hidden_dim)
        bias = last_layer.bias[t_idx]      # (E,)
        logits_mlp = (h * weight).sum(dim=-1) + bias  # (E,)

        # --- Factorized path ---
        rel_factors = self.rel_factor(head_vec)   # (B, R, k)
        rf_e = rel_factors[b_idx, r_idx]          # (E, k)
        tb_e = self.tail_basis[t_idx]             # (E, k)
        logits_fact = (rf_e * tb_e).sum(dim=-1)   # (E,)

        # --- Mix ---
        mix_alpha = torch.sigmoid(self._mix_logit)
        logits_sparse = mix_alpha * logits_mlp + (1.0 - mix_alpha) * logits_fact

        # --- Lightweight Bias ---
        rb = self.rel_bias_mlp(head_vec).squeeze(-1)  # (B, R)
        rb_e = rb[b_idx, r_idx]                       # (E,)
        tb_e_bias = self.tail_bias[t_idx]             # (E,)

        logits_sparse = logits_sparse + rb_e + tb_e_bias

        return logits_sparse



class VQCodebook(nn.Module):
    """
    Vector Quantization codebook (VQ): input/output shape same (..., d), returns indices (...,).
    Supports EMA update of codebook.
    """
    def __init__(self, K: int, d: int, ema: bool = False, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.K = int(K)
        self.d = int(d)
        self.ema = bool(ema)
        self.decay = float(decay)
        self.eps = float(eps)

        self.embed = nn.Parameter(torch.randn(self.K, self.d))
        nn.init.normal_(self.embed, std=1.0 / math.sqrt(self.d))

        if self.ema:
            self.register_buffer("embed_avg", self.embed.detach().clone())
            self.register_buffer("cluster_size", torch.zeros(self.K, dtype=self.embed.dtype))

        self.register_buffer("ever_used", torch.zeros(self.K, dtype=torch.bool))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig = z.shape
        d = orig[-1]
        z_flat = z.reshape(-1, d)  # (M, d)

        with torch.no_grad():
            # L2 distance
            dist = (
                z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.embed.t()
                + self.embed.pow(2).sum(1, keepdim=True).t()
            )  # (M, K)
            indices = dist.argmin(dim=1)  # (M,)

            # Mark usage
            step_used = torch.unique(indices)
            used_mask = torch.zeros(self.K, device=indices.device, dtype=torch.uint8)
            used_mask[step_used] = 1
            if ddpist.is_available() and ddpist.is_initialized():
                ddpist.all_reduce(used_mask, op=ddpist.ReduceOp.MAX)
            self.ever_used |= used_mask.bool()

        z_q_flat = self.embed[indices]  # (M, d)

        # EMA update
        if self.ema and self.training:
            with torch.no_grad():
                counts = torch.bincount(indices, minlength=self.K).to(dtype=self.embed.dtype, device=z_flat.device)
                sums = torch.zeros(self.K, self.d, device=z_flat.device, dtype=self.embed.dtype)
                sums.index_add_(0, indices, z_flat)

                if ddpist.is_available() and ddpist.is_initialized():
                    ddpist.all_reduce(counts, op=ddpist.ReduceOp.SUM)
                    ddpist.all_reduce(sums, op=ddpist.ReduceOp.SUM)

                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(sums, alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cluster_size = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
                new_embed = self.embed_avg / cluster_size.unsqueeze(1)
                self.embed.data.copy_(new_embed)

        z_q = z_q_flat.view(*orig)
        # Straight-Through
        z_st = z + (z_q - z).detach()
        commit = F.mse_loss(z, z_q.detach()) + F.mse_loss(z.detach(), z_q)
        idx = indices.view(*orig[:-1])
        return z_st, commit, idx

# New: Conditional residual token block (ResNet style), applies FiLM + token-mixing to tokens using ent_embed
class CondResTokenBlock(nn.Module):
    def __init__(self, z_dim: int, cond_dim: int, tokens: int, dropout: float = 0.0):
        super().__init__()
        self.z_dim = int(z_dim)
        self.cond_dim = int(cond_dim)
        self.tokens = int(tokens)

        self.norm = nn.LayerNorm(self.z_dim)
        # FiLM: generate per-channel gamma/beta from ent_embed (shared across all tokens)
        self.film = nn.Linear(self.cond_dim, 2 * self.z_dim)
        # Residual gating (global scalar, stable fusion)
        self.gate = nn.Linear(self.cond_dim, 1)

        # Token internal MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.z_dim, 4 * self.z_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.z_dim, self.z_dim),
        )
        # Lightweight token-mixing (depthwise separable conv over token dimension)
        self.dwconv = nn.Conv1d(self.z_dim, self.z_dim, kernel_size=3, padding=1, groups=self.z_dim)
        self.pwconv = nn.Conv1d(self.z_dim, self.z_dim, kernel_size=1)

        # Small initialization to avoid disrupting early tokenizer outputs
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.zeros_(self.dwconv.weight)
        nn.init.zeros_(self.dwconv.bias)
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)  # Early gate≈0.12, gradually learns larger fusion

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B,T,z), cond: (B,cond_dim)
        B, T, D = x.shape
        y = self.norm(x)
        gamma, beta = self.film(cond).chunk(2, dim=-1)  # (B,z),(B,z)
        y = y * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)  # FiLM
        y = self.mlp(y)  # (B,T,z)

        # Token-mixing: (B,T,z) -> (B,z,T) -> conv -> (B,T,z)
        y_mix = y.transpose(1, 2)
        y_mix = self.pwconv(self.dwconv(y_mix))
        y_mix = y_mix.transpose(1, 2)
        y = y + y_mix

        gate = torch.sigmoid(self.gate(cond)).view(B, 1, 1)
        return x + gate * y

class ResidualVQ(nn.Module):
    """
    Multi-level residual vector quantization (RVQ): z dimension is small, serial multi-codebooks.
    Returns: z_st, total_commit_loss, indices(L, ...)
    """
    def __init__(self, num_quantizers: int, K: int, d: int, ema: bool = True, decay: float = 0.99):
        super().__init__()
        self.L = int(num_quantizers)
        self.codebooks = nn.ModuleList([VQCodebook(K, d, ema=ema, decay=decay) for _ in range(self.L)])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        zq_sum = torch.zeros_like(z)
        total_loss = 0.0
        all_idx = []
        for cb in self.codebooks:
            q, loss, idx = cb(residual)
            zq_sum = zq_sum + q
            residual = residual - q.detach()
            total_loss = total_loss + loss
            all_idx.append(idx)
        z_st = z + (zq_sum - z).detach()
        indices = torch.stack(all_idx, dim=0)  # (L, ...)
        return z_st, total_loss, indices


class VQGANFCGraph(nn.Module):
    """
    Pure VQ-VAE reconstruction (empty class removed):
      encoder (tiny tokens) -> RVQ -> FC decoder -> logits(B,R,N)
      Loss: BCEWithLogits + beta*commit + usage_KL(vq)
    Enhancements:
      - Relation Applicability Gate: predict per-relation applicability prior and inject as logit bias
      - Decoder adds low-rank factorized path for better generalization
      - Tokenizer adds positional encoding and token dropout
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        rel_diag=None,
        num_rel: int = 0,
        num_nodes: int = 0,
        has_empty_class: bool = False,   # Kept but not used
        # Key compression and quantization hyperparameters
        tokens: int = 16,
        z_dim: int = 16,
        num_quantizers: int = 4,
        ema: bool = True,
        decay: float = 0.99,
        beta: float = 0.25,
        usage_lambda: float = 1e-2,
        temp: float = 1.0,
        # Training/loss hyperparameters
        label_smoothing: float = 0.00,
        pos_weight_max: float = 10000.0,
        # GAN discriminator (default off)
        adv_lambda: float = 0.0,
        disc_hidden: int = None,
        # Mode
        mode: str = "ae",               # 'identity' | 'ae' | 'vq'
        use_ema_pos_weight: bool = False,
        ema_pos_decay: float = 0.98,
        # Relation prior gating
        apply_rel_gate: bool = True,
        rel_gate_tau: float = 1.0,
        # Decoder factor rank
        factor_rank: int = 64,
        token_dropout: float = 0.05,
        # -------- Sparse prior regularization (only affects loss during training, does not change forward) --------
        sparsity_lambda: float = 1e-1,         # >0 enables; suggested 1e-4 ~ 1e-2
        sparsity_mode: str = "kl",            # 'kl' | 'mse' | 'l1'
        sparsity_target: Optional[float] = None,  # If given, use this global target density; otherwise use EMA
        sparsity_decay: float = 0.99,

        # Conditional residual block count
        token_resblocks: int = 2,
        # New: decoder-side residual block count
        token_dec_resblocks: int = 1,
        # -------- Generalization enhancement --------
        mixup_p: float = 0.0,                 # Latent MixUp probability
        mixup_alpha: float = 0.0,             # Beta(alpha,alpha)
        neg_weight: float = 0.0,              # Negative sample weight scaling (PU style)
        degree_loss_lambda: float = 0.05,     # Degree alignment loss coefficient
        degree_scale_min: float = 1.0,        # Target degree scaling range (simulate denser)
        degree_scale_max: float = 1.2,
        num_neg_samples: int = 0,             # New: negative sampling multiplier, 0 means no sampling (full)

    ):
        super().__init__()
        self.dim = int(dim)
        self.num_rel = int(num_rel)
        self.num_nodes = int(num_nodes)
        self.has_empty_class = False  # Hard disable empty class
        self.tokens = int(tokens)
        self.z_dim = int(z_dim)
        self.beta = float(beta)
        self.usage_lambda = float(usage_lambda)
        self.temp = float(temp)
        self.label_smoothing = float(label_smoothing)
        self.pos_weight_max = float(pos_weight_max)
        self.adv_lambda = float(adv_lambda)
        self.mode = mode
        self.use_ema_pos_weight = bool(use_ema_pos_weight)
        self.ema_pos_decay = float(ema_pos_decay)
        self.apply_rel_gate = bool(apply_rel_gate)
        self.rel_gate_tau = float(max(1e-6, rel_gate_tau))
        self.token_dropout = float(token_dropout)
        # Sparse prior regularization config
        self.sparsity_lambda = float(sparsity_lambda)
        self.sparsity_mode = str(sparsity_mode)
        self.sparsity_target = None if sparsity_target is None else float(sparsity_target)
        self.sparsity_decay = float(sparsity_decay)
        # New: generalization enhancement hyperparameters
        self.mixup_p = float(mixup_p)
        self.mixup_alpha = float(mixup_alpha)
        self.neg_weight = float(neg_weight)
        self.degree_loss_lambda = float(degree_loss_lambda)
        self.deg_scale_min = float(degree_scale_min)
        self.deg_scale_max = float(degree_scale_max)

        # Negative sampling parameter
        self.num_neg_samples = int(num_neg_samples)


        # Statistics buffer (keep interface)
        self.register_buffer("pos_counts_ema", torch.zeros(self.num_nodes))
        self.register_buffer("total_pairs_ema", torch.tensor(1.0))
        # Per-relation positive density EMA prior (initialized to very sparse)
        init_prior = 1.0 / max(64, self.num_nodes)
        self.register_buffer("rel_density_ema", torch.full((self.num_rel,), init_prior))

        # Decoder (from head vector to (B,R,N))
        self.inversedecoder = InverseAdjacencyDecoder(
            num_nodes=self.num_nodes,
            num_rel=self.num_rel,
            head_dim=self.dim,
            factor_rank=factor_rank,
        )

        # Tokenizer: (B, dim) -> (B, T, z)
        hidden = max(128, self.dim*2)
        self.tokenizer = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.tokens * self.z_dim),
        )
        self.pre_vq_ln = nn.LayerNorm(self.z_dim)
        # Learnable token positional encoding
        self.token_pos = nn.Parameter(torch.zeros(self.tokens, self.z_dim))
        nn.init.normal_(self.token_pos, std=0.02)

        # Conditional residual token block stack (ResNet style)
        self.token_resblocks = nn.ModuleList(
            [CondResTokenBlock(self.z_dim, self.dim, self.tokens, dropout=self.token_dropout) for _ in range(int(token_resblocks))]
        )
        # RVQ
        self.rvq = ResidualVQ(num_quantizers=num_quantizers, K=codebook_size, d=self.z_dim, ema=ema, decay=decay)

        # Decode tokens back to head vector
        dec_in = self.tokens * self.z_dim
        dec_hidden = max(dec_in, self.dim*2)
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, dec_hidden),
            nn.SiLU(),
            nn.Linear(dec_hidden, self.dim),
        )
        self.post_dec_ln = nn.LayerNorm(self.dim)
        # Decoder-side conditional residual blocks (act on (B,T,z) token space)
        self.dec_token_blocks = nn.ModuleList(
            [CondResTokenBlock(self.z_dim, self.dim, self.tokens, dropout=self.token_dropout) for _ in range(int(token_dec_resblocks))]
        )
        # Linear residual from flattened tokens to head
        self.dec_skip = nn.Linear(dec_in, self.dim)
        nn.init.zeros_(self.dec_skip.weight)
        nn.init.zeros_(self.dec_skip.bias)

        # Relation applicability gate: from head predict per-relation prior (as full-tail bias)
        # Input (D) -> Output (1) (per relation score)
        self.rel_gate = nn.Sequential(
            nn.Linear(self.dim, max(64, self.dim // 2)),
            nn.SiLU(),
            nn.Linear(max(64, self.dim // 2), 1),
        )

        # Keep skip-related statistics key
        self.register_buffer("_last_gate", torch.tensor(1.0))

    def set_mode(self, mode: str):
        self.mode = mode
    # -------- New: sparse prior computation and regularization --------
    def _batch_rel_density(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Current batch's true relation-level density estimate: p_r = mean_{B,N} label[b,r,n]
        labels: (B,R,N) -> (R,)
        """
        return labels.float().mean(dim=(0, 2))

    def _update_rel_prior(self, labels: torch.Tensor):
        """
        Update EMA prior with true labels; only called during training, does not build graph.
        """
        if self.sparsity_target is not None:
            return  # Use fixed target, do not update EMA
        with torch.no_grad():
            batch_density = self._batch_rel_density(labels)
            self.rel_density_ema.mul_(self.sparsity_decay).add_(batch_density, alpha=1.0 - self.sparsity_decay)

    def _get_rel_prior(self, device, dtype) -> torch.Tensor:
        """
        Return per-relation target density prior q_r, from fixed target or EMA.
        """
        if self.sparsity_target is not None:
            q = torch.full((self.num_rel,), float(self.sparsity_target), device=device, dtype=dtype)
        else:
            q = self.rel_density_ema.to(device=device, dtype=dtype)
        q = q.clamp(1e-6, 1.0 - 1e-6)
        return q

    def _sparsity_regularizer(self, probs: torch.Tensor, prior_q: torch.Tensor) -> torch.Tensor:
        """
        probs: (B,R,N) in (0,1), prior_q: (R,)
        Return scalar loss. Default KL(mean prob || q).
        """
        p_hat = probs.mean(dim=(0, 2)).clamp(1e-6, 1.0 - 1e-6)  # (R,)
        if self.sparsity_mode == "kl":
            kl = p_hat * (p_hat / prior_q).log() + (1.0 - p_hat) * ((1.0 - p_hat) / (1.0 - prior_q)).log()
            return kl.mean()
        elif self.sparsity_mode == "mse":
            return F.mse_loss(p_hat, prior_q)
        elif self.sparsity_mode == "l1":
            # L1 toward target density (more stable than directly penalizing mean(probs))
            return (p_hat - prior_q).abs().mean()
        else:
            return torch.zeros((), device=probs.device, dtype=probs.dtype)

    # --------- Sub-functions ---------
    def encode_tokens(self, ent_embed: torch.Tensor) -> torch.Tensor:
        # ent_embed: (B, dim) -> (B, T, z)
        B, _ = ent_embed.shape
        tokens = self.tokenizer(ent_embed).view(B, self.tokens, self.z_dim)
        tokens = self.pre_vq_ln(tokens)
        tokens = tokens + self.token_pos.view(1, self.tokens, self.z_dim)


        # Conditional residual blocks: ResNet style, condition tokens using ent_embed
        for blk in self.token_resblocks:
            tokens = blk(tokens, ent_embed)

        if self.token_dropout > 0 and self.training:
            tokens = F.dropout(tokens, p=self.token_dropout)
        return tokens

    def quantize_tokens(self, tokens: torch.Tensor):
        return self.rvq(tokens)

    def decode_tokens(self, z_q: torch.Tensor = None, tokens: torch.Tensor = None, ent_embed: torch.Tensor = None) -> torch.Tensor:
        """
        - z_q: (B,T,z) quantized tokens
        - tokens/ent_embed: compatible with identity/ae
        Output head vector (B, dim)
        """
        if z_q is not None:
            B = z_q.size(0)
            x = z_q  # (B,T,z)
            # Decoder-side conditional residual blocks (degenerate to unconditional residual if no ent_embed)
            cond = ent_embed if ent_embed is not None else torch.zeros(B, self.dim, device=z_q.device, dtype=z_q.dtype)
            for blk in self.dec_token_blocks:
                x = blk(x, cond)
            flat = x.view(B, self.tokens * self.z_dim)
            core = self.decoder(flat)
            out = core + self.dec_skip(flat)  # Residual fusion
            out = self.post_dec_ln(out)
            return out
        assert ent_embed is not None, "decode_tokens requires z_q or ent_embed"
        return ent_embed

    def compute_sparse_logits_and_labels(self, head_vec: torch.Tensor, batch: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, B: int, device):
        H, T = edge_index[0], edge_index[1]
        R = edge_type
        head2b = torch.full((self.num_nodes,), -1, device=device, dtype=torch.long)
        head2b[batch] = torch.arange(B, device=device, dtype=torch.long)
        b_idx = head2b[H]
        valid = b_idx >= 0

        pos_b = b_idx[valid]
        pos_r = R[valid]
        pos_t = T[valid]
        num_pos = pos_b.size(0)

        num_neg_target = int(num_pos * self.num_neg_samples)

        if num_neg_target > 0:
            # Random negative sampling
            neg_b = pos_b.repeat(self.num_neg_samples)
            neg_r = pos_r.repeat(self.num_neg_samples)
            neg_t = torch.randint(0, self.num_nodes, (num_neg_target,), device=device, dtype=torch.long)

            eval_b = torch.cat([pos_b, neg_b])
            eval_r = torch.cat([pos_r, neg_r])
            eval_t = torch.cat([pos_t, neg_t])
            labels = torch.cat([torch.ones(num_pos, device=device), torch.zeros(num_neg_target, device=device)])
        else:
            eval_b = pos_b
            eval_r = pos_r
            eval_t = pos_t
            labels = torch.ones(num_pos, device=device)

        # Call sparse forward to get (E,) sized logits
        logits_sparse = self.inversedecoder.forward_sparse(head_vec, eval_b, eval_r, eval_t) / float(max(self.temp, 1e-6))

        # Handle relation prior gate bias (if enabled)
        if self.apply_rel_gate:
            rel_prior = torch.sigmoid(self.rel_gate(head_vec).squeeze(-1) / self.rel_gate_tau)  # (B, R)
            prior_bias = torch.log(rel_prior.clamp_min(1e-6)) - torch.log((1.0 - rel_prior).clamp_min(1e-6))
            logits_sparse = logits_sparse + prior_bias[eval_b, eval_r]
            self._last_gate = rel_prior.mean()
        else:
            self._last_gate = torch.tensor(1.0, device=head_vec.device)

        if self.label_smoothing > 0.0:
            labels = (1.0 - self.label_smoothing) * labels + 1.0 / float(self.num_nodes)

        return logits_sparse, labels

    def compute_logits(self, head_vec: torch.Tensor) -> torch.Tensor:
        # head_vec: (B, R, D)
        # Base logits
        logits = self.inversedecoder(head_vec) / float(max(self.temp, 1e-6))  # (B,R,N)
        # Relation applicability prior (logit bias), does not change loss form
        if self.apply_rel_gate:
            # (B, R, D) -> (B, R, 1) -> (B, R)
            rel_prior = torch.sigmoid(self.rel_gate(head_vec).squeeze(-1) / self.rel_gate_tau)
            prior_bias = torch.log(rel_prior.clamp_min(1e-6)) - torch.log((1.0 - rel_prior).clamp_min(1e-6))
            logits = logits + prior_bias.unsqueeze(-1)  # (B,R,1) broadcast to N
            # Record average gate (for monitoring)
            self._last_gate = rel_prior.mean()
        else:
            self._last_gate = torch.tensor(1.0, device=head_vec.device)
        return logits

    def build_labels(self, batch: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, B: int, device):
        # Multi-label only (B, R, N), no empty class
        labels = torch.zeros((B, self.num_rel, self.num_nodes), device=device)

        H, T = edge_index[0], edge_index[1]
        R = edge_type
        head2b = torch.full((self.num_nodes,), -1, device=device, dtype=torch.long)
        head2b[batch] = torch.arange(B, device=device, dtype=torch.long)
        b_idx = head2b[H]
        valid = b_idx >= 0
        if valid.any():
            labels[b_idx[valid], R[valid], T[valid]] = 1.0

        # Label smoothing (does not change dimension)
        if self.label_smoothing > 0.0:
            total_classes = self.num_nodes
            labels = (1.0 - self.label_smoothing) * labels + 1.0 / float(total_classes)
        return labels

    def compute_pos_weight(self, labels: torch.Tensor, B: int, dtype, device, pos_weight=None):
        if pos_weight is None:
            total_pairs = torch.tensor(B * self.num_rel, device=device, dtype=dtype)
            pos_counts = labels.sum(dim=(0, 1))
            neg_counts = (total_pairs - pos_counts).clamp_min(1.0)
            pos_counts = pos_counts.clamp_min(1.0)
            pos_weight_vec = (neg_counts / pos_counts).detach()
            pos_weight_vec = torch.clamp(pos_weight_vec, max=self.pos_weight_max)
        else:
            pos_weight_vec = pos_weight
            C = labels.size(-1)
            if isinstance(pos_weight_vec, float):
                pos_weight_vec = torch.full((C,), float(pos_weight_vec), device=device, dtype=dtype)
            elif isinstance(pos_weight_vec, torch.Tensor) and pos_weight_vec.numel() == 1:
                pos_weight_vec = pos_weight_vec.expand(C)
        return pos_weight_vec

    def recon_loss(self, logits: torch.Tensor, labels: torch.Tensor, pos_weight_vec: torch.Tensor):
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_vec)
        return bce(logits, labels)

    def _usage_kl(self, indices: torch.Tensor, K: int, device) -> Tuple[torch.Tensor, dict]:
        """
        Compute KL between RVQ per-level usage distribution and uniform distribution, plus diagnostics.
        indices: (L, B, T)
        """
        L = indices.size(0)
        kl_total = 0.0
        per_stats = {}
        for l in range(L):
            idx_l = indices[l].reshape(-1)  # (B*T,)
            counts = torch.bincount(idx_l, minlength=K).to(device=device, dtype=torch.float)
            if ddpist.is_available() and ddpist.is_initialized():
                ddpist.all_reduce(counts, op=ddpist.ReduceOp.SUM)
            total = counts.sum().clamp_min(1.0)
            p = (counts / total).clamp_min(1e-12)
            logK = math.log(K)
            kl = (p * p.log()).sum() + logK
            kl_total = kl_total + kl
            entropy = -(p * p.log()).sum()
            perplex = torch.exp(entropy)
            used = int((counts > 0).sum().item())
            per_stats[f"q{l+1}_used"] = used
            per_stats[f"q{l+1}_perplex"] = float(perplex.item())
        return kl_total, per_stats

    def usage_kl(self, indices: torch.Tensor, K: int, device):
        return self._usage_kl(indices, K, device)

    def gan_losses(self, head_real: torch.Tensor, head_fake: torch.Tensor):
        # Default disabled (adv_lambda=0), keep placeholder
        d_real = torch.zeros_like(head_real[:, :1]).mean()
        d_fake = torch.zeros_like(head_fake[:, :1]).mean()
        d_loss = (d_real + d_fake) * 0.0
        g_adv = d_loss
        return g_adv, d_loss

    # --------- Main flow ---------
    def forward(
        self,
        batch,
        ent_embed: torch.Tensor,          # (B, R, dim)
        edge_index: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
        pos_weight: Optional[float] = None,
        head_chunk: int = 4096,
        tail_chunk: int = 100000,
        rel_chunk: int = 512,
        phase: str = "G",                 # 'G' | 'D' | 'eval'
        return_logits: bool = True,
        topk: Optional[int] = None
    ) -> Tuple[torch.Tensor, dict]:
        assert phase in ("G", "D", "eval"), "phase must be 'G' | 'D' | 'eval'"
        device = ent_embed.device

        B, R, D = ent_embed.shape

        per_stats = {}
        # ------- Latent MixUp (does not create fake edges) -------
        mix_active = False

        # Flatten Batch and Relation dimensions for VQ processing
        # (B, R, D) -> (B*R, D)
        ent_embed_flat = ent_embed.view(B * R, D)

        # Encode tokens
        # (B*R, D) -> (B*R, T, z)
        tokens = self.encode_tokens(ent_embed_flat)


        z_q, commit_loss, all_idx = self.quantize_tokens(tokens)
        head_fake_flat = self.decode_tokens(z_q=z_q, tokens=None, ent_embed=None)
        K = self.rvq.codebooks[0].K
        usage_kl, per_stats = self.usage_kl(all_idx, K=K, device=device)

        # Restore shape (B, R, D)
        head_fake = head_fake_flat.view(B, R, D)

        # Evaluation only: directly return logits and statistics
        if phase == "eval":
            logits = self.compute_logits(head_fake)
            stats = {
                "mode": self.mode,
                "logits_mean": float(logits.detach().mean().item()),
                "logits_std": float(logits.detach().std().item()),
                "loss_commit": float(commit_loss.detach()),
                "loss_usage_kl": float((usage_kl if self.mode == "vq" else torch.tensor(0.0, device=device)).detach()),
            }
            stats.update(per_stats)
            return logits, stats


        # Training: build labels and reconstruction loss
        assert edge_type is not None, "edge_type is required for training"

        # [NEW] Check if negative sampling is enabled to enter fully sparse loss stage, greatly optimizing memory
        is_sparse_mode = (hasattr(self, 'num_neg_samples') and self.num_neg_samples > 0)

        if is_sparse_mode:
            # In this mode, do not create B*R*N prediction tensor; compute logits only for sampled positives and negatives
            sparse_logits, sparse_labels = self.compute_sparse_logits_and_labels(
                head_fake, batch, edge_index, edge_type, B, device
            )
            loss_recon = F.binary_cross_entropy_with_logits(sparse_logits, sparse_labels, reduction='mean')

            # For subsequent statistics, infer full logits under no_grad with very low overhead
            with torch.no_grad():
                logits = self.compute_logits(head_fake)
                labels = self.build_labels(batch, edge_index, edge_type, B, device)
                pos_weight_vec = torch.tensor([1.0], device=device)
        else:
            # Full branch B*R*N
            labels = self.build_labels(batch, edge_index, edge_type, B, device)
            logits = self.compute_logits(head_fake)
            pos_weight_vec = self.compute_pos_weight(labels, B, logits.dtype, device, pos_weight)
            loss_recon = self.recon_loss(logits, labels, pos_weight_vec)

        K = self.rvq.codebooks[0].K
        usage_kl, per_stats = self.usage_kl(all_idx, K=K, device=device)

        # Discriminator path (0 when disabled)
        if phase == "D":
            with torch.no_grad():
                head_real_flat = self.decode_tokens(z_q=None, tokens=tokens, ent_embed=ent_embed_flat)
                head_real = head_real_flat.view(B, R, D)
            _, d_loss = self.gan_losses(head_real=head_real, head_fake=head_fake.detach())
            g_adv_loss = torch.zeros((), device=device)
        else:
            g_adv_loss = torch.zeros((), device=device)
            d_loss = torch.zeros((), device=device)

        # Statistics
        with torch.no_grad():
            probs = torch.sigmoid(logits)  # (B,R,N)
            deg_true = labels.sum(dim=-1).float().mean()
            deg_pred = probs.sum(dim=-1).float().mean()
            empty_prob = torch.tensor(0.0, device=device)  # Compatibility key, empty class already removed


        if phase == "D":
            return d_loss, stats
        # -------- Degree alignment (density-controllable soft constraint) --------
        loss_degree = torch.zeros((), device=device, dtype=loss_recon.dtype)
        if self.training and self.degree_loss_lambda > 0.0 and not is_sparse_mode:
            probs = torch.sigmoid(logits)
            deg_true_br = labels.sum(dim=-1)          # (B,R)
            deg_pred_br = probs.sum(dim=-1)           # (B,R)
            scale = torch.empty((), device=device).uniform_(self.deg_scale_min, self.deg_scale_max)
            deg_target = (deg_true_br * scale).clamp_max(float(self.num_nodes))
            # Relative error MSE to avoid bias from different degree magnitudes
            loss_degree = F.mse_loss(deg_pred_br, deg_target) / max(1.0, float(self.num_nodes))
            loss_degree = loss_degree * self.degree_loss_lambda

        # -------- Sparse prior regularization (only affects loss) --------
        loss_sparse = torch.zeros((), device=device, dtype=loss_recon.dtype)
        if self.training and self.sparsity_lambda > 0.0 and not is_sparse_mode:
            # Update EMA prior
            self._update_rel_prior(labels)
            prior_q = self._get_rel_prior(device=device, dtype=loss_recon.dtype)
            loss_sparse = self._sparsity_regularizer(probs, prior_q) * self.sparsity_lambda



        # Total loss (does not change original main term, only adds regularization)
        loss_g = (
            loss_recon
            + self.beta * commit_loss
            + (self.usage_lambda * usage_kl if self.mode == "vq" else 0.0)
            + loss_sparse
            + loss_degree if self.degree_loss_lambda > 0.0 else 0.0
        )

        if self.adv_lambda > 0:
            loss_g = loss_g + self.adv_lambda * g_adv_loss
        # Statistics summary
        stats = dict(
                loss_recon=float(loss_recon.detach()),
                loss_commit=float(commit_loss.detach()),
                loss_usage_kl=float((usage_kl if self.mode == "vq" else torch.tensor(0.0, device=device)).detach()),
                loss_g_adv=float(g_adv_loss.detach()),
                gate_mean=float(self._last_gate.detach()),
                deg_true=float(deg_true.item()),
                deg_pred=float(deg_pred.item()),
                empty_prob=float(empty_prob.item()),
                logits_mean=float(logits.detach().mean().item()),
                logits_std=float(logits.detach().std().item()),
                head_norm=float(head_fake.detach().norm(dim=1).mean().item()),
                token_norm=float(tokens.detach().norm(dim=2).mean().item()),
                posw_mean=float(pos_weight_vec.mean().item()),
                posw_max=float(pos_weight_vec.max().item()),
                mode=self.mode,
            )
        if self.mode == "vq":
            stats.update(per_stats)

        return loss_g, stats

    # --------- Helpers ---------
    def _collect_stats(self, commit_loss: torch.Tensor, all_idx: torch.Tensor):
        with torch.no_grad():
            never_used = []
            per_stats = {}
            for i, cb in enumerate(self.rvq.codebooks):
                never_used.append(int((~cb.ever_used).sum().item()))
            per_stats["never_used_sum"] = int(sum(never_used))
        return {
            "loss_commit": float(commit_loss.detach()),
            **per_stats,
        }

    def _topk_from_logits(self, logits: torch.Tensor, k: int):
        scores = torch.sigmoid(logits)
        topk_scores, topk_idx = scores.topk(k, dim=-1)  # (B,R,k)
        return {"topk_idx": topk_idx, "topk_scores": topk_scores}

    @torch.no_grad()
    def infer(self, ent_embed: torch.Tensor, topk: Optional[int] = None):
        return

    def discriminator_step(
        self,
        batch,
        ent_embed: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ):
        # Discriminator optimization step (default no-op)
        loss_d, stats = self.forward(
            batch=batch,
            ent_embed=ent_embed,
            edge_index=edge_index,
            edge_type=edge_type,
            phase="D"
        )
        return loss_d, stats


class VQpretrain(nn.Module):
    """
    Wrapper: adjacency data -> HeadAdjacencyEncoder -> VQ-VAE -> reconstruction (B,R,N)
    """
    def __init__(self, rel_model_cfg, vq_model_cfg):
        super().__init__()
        rel_cfg = dict(rel_model_cfg)
        self.num_nodes = int(rel_cfg["num_nodes"])
        self.num_rel = int(rel_cfg["num_relation"])

        vq_cfg = dict(vq_model_cfg)
        encoder_hidden = vq_cfg.pop("encoder_hidden_dim", None)
        encoder_dropout = vq_cfg.pop("encoder_dropout", 0.1)
        edge_dropout = vq_cfg.pop("encoder_edge_dropout", 0.0)
        # Pop params not accepted by VQGANFCGraph to keep config compatibility
        vq_cfg.pop("rel_dim", None)
        vq_cfg.pop("use_soft_vq", None)
        vq_cfg.pop("use_vq", None)

        # New: read reasoning switch from config, default False to preserve original logic
        reasoning_enabled = vq_cfg.pop("reasoning_enabled", False)

        # Pure VQ-VAE (has_empty_class parameter kept but not enabled)
        self.vq_vae = VQGANFCGraph(
            num_nodes=self.num_nodes,
            num_rel=self.num_rel,
            **vq_cfg,
        )
        self.adj_encoder = HeadAdjacencyEncoder(
            num_nodes=self.num_nodes,
            num_rel=self.num_rel,
            out_dim=self.vq_vae.dim,
            hidden_dim=encoder_hidden,
            dropout=encoder_dropout,
            edge_dropout=edge_dropout,
            reasoning_enabled=reasoning_enabled,  # New: reasoning branch config
        )

    def set_mode(self, mode: str):
        self.vq_vae.set_mode(mode)

    def forward(self, data, batch, phase: str = "G", return_logits: bool = False, topk: int = None):
        ent_representation = self.adj_encoder(data, batch)
        return self.vq_vae(
            batch=batch,
            ent_embed=ent_representation,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
            phase=phase,
            return_logits=return_logits,
            topk=topk,
        )

    def d_step(self, data, batch):
        ent_representation = self.adj_encoder(data, batch)
        return self.vq_vae.discriminator_step(
            batch=batch,
            ent_embed=ent_representation,
            edge_index=data.edge_index,
            edge_type=data.edge_type,
        )

    @torch.no_grad()
    def encode_tokens(self, data, batch):
        ent_representation = self.adj_encoder(data, batch)
        # Adapt (B, R, D) -> (B*R, D)
        B, R, D = ent_representation.shape
        ent_flat = ent_representation.view(B * R, D)
        return self.vq_vae.encode_tokens(ent_flat)

    @torch.no_grad()
    def context_embed(self, data, batch):
        # Return quantized latent codes
        ent_representation = self.adj_encoder(data, batch)
        B, R, D = ent_representation.shape
        ent_flat = ent_representation.view(B * R, D)

        # Encode -> Quantize
        tokens = self.vq_vae.encode_tokens(ent_flat)
        z_q, _, _ = self.vq_vae.quantize_tokens(tokens)  # (B*R, T, z)

        # Flatten latent codes: (B*R, T*z) -> (B, R, T*z)
        z_flat = z_q.view(B, R, -1)
        return z_flat

    def decode_tokens_to_head(self, z_tokens):
        # z_tokens: (B*R, T, z) or (B, R, T, z) ?
        # Assume input is already flattened or needs flattening
        if z_tokens.dim() == 4:  # (B, R, T, z)
            B, R, T, Z = z_tokens.shape
            z_tokens = z_tokens.view(B*R, T, Z)
        return self.vq_vae.decode_tokens(z_tokens)

    def logits_from_head(self, head_vec: torch.Tensor) -> torch.Tensor:
        """
        Decode logits (B, R, N) from latent tokens (B, R, T*z).
        """
        # Input is latent codes, need to decode back to head space first
        B, R, _ = head_vec.shape
        T = self.vq_vae.tokens
        z_dim = self.vq_vae.z_dim

        # Reshape to (B*R, T, z)
        z_tokens = head_vec.view(B * R, T, z_dim)

        # Decode -> Head Vector
        head_recon_flat = self.vq_vae.decode_tokens(z_q=z_tokens)
        head_recon = head_recon_flat.view(B, R, -1)

        return self.vq_vae.compute_logits(head_recon)
