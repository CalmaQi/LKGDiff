import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t_cont: torch.Tensor, dim: int) -> torch.Tensor:
    """
    t_cont: (B,) in [0,1]
    returns: (B, dim)
    """
    half = dim // 2
    device = t_cont.device
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / max(1, half))
    args = t_cont.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(2 * math.pi * args), torch.sin(2 * math.pi * args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def mr_ou_schedule(T: int, kappa: float = 4.0, sigma_inf: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mean-reverting OU-like schedule (continuous-time normalized τ∈[0,1]):
      α_t = exp(-κ τ), σ_t = σ_inf * sqrt(1 - α_t^2)
    Returns:
      alphas: (T,)
      sigmas: (T,)
    """
    tau = torch.linspace(0.0, 1.0, steps=T, dtype=torch.float32)
    alphas = torch.exp(-kappa * tau).clamp(1e-6, 1.0)
    sigmas = sigma_inf * torch.sqrt(torch.clamp(1.0 - alphas ** 2, min=0.0))
    return alphas.float(), sigmas.float()


class FiLMTime(nn.Module):
    """
    Modulate channels with time embedding: h = h * (1+scale) + shift
    """
    def __init__(self, hidden: int, time_dim: int, cond_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, hidden * 2),
        )

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # h: (B,N,T,C); t_emb: (N,time_dim)
        B, N, T, C = h.shape
        shift, scale = self.mlp(t_emb).chunk(2, dim=-1)  # (B, N, C), (B, N, C)
        shift = shift.view(B, 1, 1, C)
        scale = scale.view(B, 1, 1, C)
        return h * (1.0 + scale) + shift


class LinearAttention(nn.Module):
    """
    Linear attention (elu+1 kernel):
      - axis="tokens": for each node, apply attention along the T sequence. Input (B,N,T,C) -> output same shape
      - axis="nodes":  for each token, apply attention along the N sequence. Input (B,N,T,C) -> output same shape
    """
    def __init__(self, dim: int, tlen=24, heads: int = 8, dim_head: int = 64, axis: str = "tokens"):
        super().__init__()
        assert axis in ("tokens", "nodes")
        self.axis = axis
        self.heads = heads
        self.dim_head = dim_head
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim, bias=False))
        self.promlp = nn.Sequential(
            nn.Linear(tlen*dim, tlen*dim*2),
            nn.ReLU(),
            nn.Linear(tlen*dim*2, tlen*dim),
        )
       
    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,T,C)
        B, N, T, C = x.shape
        H, Dh = self.heads, self.dim_head
        scale = 1.0 / math.sqrt(Dh)
        eps = 1e-6
        dtype = x.dtype

        if self.axis == "tokens":
            # merge B,N -> Bn, seq=T
            xn = x.reshape(B * N, T,C)
            
            q = self.to_q(xn).reshape(B * N, T, H, Dh)
            k = self.to_k(xn).reshape(B * N, T, H, Dh)
            v = self.to_v(xn).reshape(B * N, T, H, Dh)
            # Numerical stabilization: float32 computation, phi, scaling and normalization
            q = self.phi(q.float()) * scale
            k = self.phi(k.float()) * scale
            v = v.float()
            q = F.normalize(q, dim=-1, eps=eps)
            k = F.normalize(k, dim=-1, eps=eps)
            # KV: (Bn, H, Dh, Dh)
            kv = torch.einsum("bthd,bthm->bhdm", k, v)
            # Z: (Bn, T, H, 1)
            k_sum = k.sum(dim=1)  # (Bn,H,Dh)
            denom = torch.einsum("bthd,bhd->bth", q, k_sum)  # (Bn,T,H)
            z = (1.0 / torch.clamp(denom, min=eps)).unsqueeze(-1)
            # out: (Bn, T, H, Dh)
            out = torch.einsum("bthd,bhdm->bthm", q, kv) * z
            out = out.reshape(B * N, T, H * Dh)
            out = self.to_out(out.to(dtype)).reshape(B, N, T, C)
            return out
        else:
            # axis == "nodes": merge B,T -> Bt, seq=N
            xt = x.permute(0, 2, 1, 3).contiguous()  # (B,T,N,C)
            xt = xt.reshape(B * T, N, C)
            q = self.to_q(xt).reshape(B * T, N, H, Dh)
            k = self.to_k(xt).reshape(B * T, N, H, Dh)
            v = self.to_v(xt).reshape(B * T, N, H, Dh)
            q = self.phi(q.float()) * scale
            k = self.phi(k.float()) * scale
            v = v.float()
            q = F.normalize(q, dim=-1, eps=eps)
            k = F.normalize(k, dim=-1, eps=eps)
            kv = torch.einsum("bnhd,bnhm->bhdm", k, v)
            k_sum = k.sum(dim=1)  # (Bt,H,Dh)
            denom = torch.einsum("bnhd,bhd->bnh", q, k_sum)  # (Bt,N,H)
            z = (1.0 / torch.clamp(denom, min=eps)).unsqueeze(-1)
            out = torch.einsum("bnhd,bhdm->bnhm", q, kv) * z
            out = out.reshape(B * T, N, H * Dh)
            out = self.to_out(out.to(dtype)).reshape(B, T, N, C).permute(0, 2, 1, 3).contiguous()
            return out
            
class LinearAttentionNodesFlat(nn.Module):
    """
    Flatten (B,N,T,C) along (T,C) into F=T*C, then apply linear attention along N:
      - First LayerNorm(F), then project to heads*dim_head
      - Computation in float32 with scaling and normalization to reduce z->0 risk
      - Output reshaped back to (B,N,T,C) for residual connection
    """
    def __init__(self, tokens: int, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.tokens = int(tokens)
        self.dim = int(dim)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.in_dim = self.tokens * self.dim           # F = T*C
        inner = self.heads * self.dim_head
        self.ln = nn.LayerNorm(self.in_dim)
        self.to_q = nn.Linear(self.in_dim, inner, bias=False)
        self.to_k = nn.Linear(self.in_dim, inner, bias=False)
        self.to_v = nn.Linear(self.in_dim, inner, bias=False)
        self.to_out = nn.Linear(inner, self.in_dim, bias=False)

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,T,C)
        B, N, T, C = x.shape
        assert T * C == self.in_dim
        H, Dh = self.heads, self.dim_head
        scale = 1.0 / math.sqrt(Dh)
        eps = 1e-6
        dtype = x.dtype

        xf = x.reshape(B, N, T * C)
        xf = self.ln(xf)  # (B,N,F)

        q = self.to_q(xf).reshape(B, N, H, Dh)
        k = self.to_k(xf).reshape(B, N, H, Dh)
        v = self.to_v(xf).reshape(B, N, H, Dh)

        q = self.phi(q.float()) * scale
        k = self.phi(k.float()) * scale
        v = v.float()
        q = F.normalize(q, dim=-1, eps=eps)
        k = F.normalize(k, dim=-1, eps=eps)

        # KV: (B,H,Dh,Dh)
        kv = torch.einsum("bnhd,bnhm->bhdm", k, v)
        # Z: (B,N,H,1)
        k_sum = k.sum(dim=1)  # (B,H,Dh)
        denom = torch.einsum("bnhd,bhd->bnh", q, k_sum)
        z = (1.0 / torch.clamp(denom, min=eps)).unsqueeze(-1)

        out = torch.einsum("bnhd,bhdm->bnhm", q, kv) * z  # (B,N,H,Dh)
        out = out.reshape(B, N, H * Dh)
        out = self.to_out(out.to(dtype))  # (B,N,F)
        out = out.reshape(B, N, T, C)     # back to (B,N,T,C)
        return out
    
class PreNormResidual(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)
    
class ChannelMLP(nn.Module):
    def __init__(self, dim, mult=2, dropout=0.0):
        super().__init__()
        hidden = dim // mult
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class TokenMixMLP(nn.Module):
    def __init__(self, tokens, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(tokens),
            nn.Linear(tokens, tokens * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(tokens * 2, tokens),
        )

    def forward(self, x):
        B, N, T, C = x.shape
        h = x.permute(0, 1, 3, 2).reshape(B * N * C, T)
        h = h + self.net(h)
        return h.reshape(B, N, C, T).permute(0, 1, 3, 2).contiguous()   
class MergeDenoiser(nn.Module):
    """
    Fusion module:
      - Input xt, known_x0 -> cat([xt, known_x0]) for deep interaction
      - Currently runs on head sub-blocks instead of the whole graph by default
      - Time FiLM modulation
      - Output x0_hat with the same shape as input x_t
    """
    def __init__(
        self,
        dim: int,
        tokens: int,
        relconfig=None,
        hidden: int = 128,
        layers: int = 6,
        time_dim: int = 64,
        heads: int = 4,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.tokens = int(tokens)
        self.time_emb_dim = int(time_dim)

        in_dim = 2 * dim
        self.proj_in = nn.Linear(in_dim, hidden)
        self.time_film = FiLMTime(hidden, time_dim=time_dim, cond_dim=hidden)

        blocks = []
        for _ in range(layers):
            channel_mlp = ChannelMLP(hidden, mult=2, dropout=dropout)
            token_channel_mlp = TokenMixMLP(tokens=tokens, dropout=dropout)
            blocks.append(nn.ModuleList([token_channel_mlp, channel_mlp]))
        
        self.blocks = nn.ModuleList(blocks)

        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t_cont: torch.Tensor,
        known_x0: torch.Tensor,
        data=None,
        h_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x_t:      (B,N,T,D) or (B,H,T,D)
        t_cont:   (B,) scalar in [0,1]
        known_x0: (B,N,T,D) or (B,H,T,D)

        If h_index is provided and the current input is still the whole graph tensor, it is first cropped into head sub-blocks internally.
        """
        if h_index is not None:
            h_index = h_index.to(x_t.device)
            if x_t.size(1) != h_index.numel():
                x_t = x_t[:, h_index]
                known_x0 = known_x0[:, h_index]

        B, N, T, D = x_t.shape
        assert D == self.dim and T == self.tokens

        def _forward_impl():
            merge = torch.cat([x_t, known_x0], dim=-1)  # (B,N,T,2D)
            h = self.proj_in(merge)  # (B,N,T,H)

            t_emb = timestep_embedding(t_cont.clamp(0, 1), self.time_emb_dim)  # (B,time_dim)
            t_emb = self.time_mlp(t_emb)
            h = self.time_film(h, t_emb)

            for attn_t, ffn in self.blocks:
                h = attn_t(h)
                h = ffn(h)

            x0_hat = self.proj_out(h)  # (B,N,T,D)
            return x0_hat

        if x_t.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=True):
                return _forward_impl()
        return _forward_impl()


class TokenDiffusionMR(nn.Module):
    """
    Mean-Reverting SDE (OU-like):
      Forward: x_t = α_t x0 + (1-α_t) m + σ_t ε, where m = known_x0

    Head-based training / inference:
      - The outside still passes the whole graph x0 / known_x0
      - Internally only extracts head sub-blocks for q_sample, denoise, and sample
      - No longer reconstructs the whole graph at every step
    """
    def __init__(
        self,
        z_dim: int,
        tokens: int,
        T: int = 1000,
        kappa: float = 4.0,
        sigma_inf: float = 1.0,
        eta: float = 0.9,  # extra noise weight during sampling; 0.9 helps performance without introducing obvious randomness
        hidden: int = 128,
        layers: int = 2,
        time_dim: int = 64,
        heads: int = 2,
        dim_head: int = 64,
        dropout: float = 0.0,
        relconfig=None,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.tokens = int(tokens)
        self.T = int(T)
        self.kappa = float(kappa)
        self.sigma_inf = float(sigma_inf)
        self.eta = float(eta)

        alphas, sigmas = mr_ou_schedule(self.T, kappa=self.kappa, sigma_inf=self.sigma_inf)
        self.register_buffer("alphas", alphas)
        self.register_buffer("sigmas", sigmas)

        self.net = MergeDenoiser(
            dim=z_dim,
            tokens=tokens,
            hidden=hidden,
            layers=layers,
            time_dim=time_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            relconfig=relconfig,
        )

    @staticmethod
    def _build_labels_and_mask_for_heads(
        complete_graph,
        incomplete_graph,
        heads: torch.Tensor,
        num_rel: int,
        num_nodes: int,
        device: torch.device,
    ):
        """
        Build:
          labels: (H, R, num_nodes) complete graph adjacency
          mask:   (H, R, num_nodes) = True means "newly added edge / unobserved position"
        """
        num_heads = heads.numel()
        num_rel = int(num_rel)

        labels = torch.zeros((num_heads, num_rel, num_nodes), device=device)
        ci = complete_graph.edge_index.to(device)
        ct = complete_graph.edge_type.to(device)
        h_all, t_all, r_all = ci[0], ci[1], ct

        head2local = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
        head2local[heads.long()] = torch.arange(num_heads, device=device)

        local_head = head2local[h_all]
        valid = local_head >= 0
        if valid.any():
            h_loc = local_head[valid]
            r_loc = r_all[valid].clamp_(0, num_rel - 1)
            t_loc = t_all[valid].clamp_(0, num_nodes - 1)
            labels[h_loc, r_loc, t_loc] = 1.0

        known = torch.zeros((num_heads, num_rel, num_nodes), device=device, dtype=torch.bool)
        ii = incomplete_graph.edge_index.to(device)
        it = incomplete_graph.edge_type.to(device)
        h_i, t_i, r_i = ii[0], ii[1], it

        local_head_i = head2local[h_i]
        valid_i = local_head_i >= 0
        if valid_i.any():
            h_loc_i = local_head_i[valid_i]
            r_loc_i = r_i[valid_i].clamp_(0, num_rel - 1)
            t_loc_i = t_i[valid_i].clamp_(0, num_nodes - 1)
            known[h_loc_i, r_loc_i, t_loc_i] = True

        mask = ~known
        return labels, mask

    @torch.no_grad()
    def _alpha_sigma(self, t_idx: torch.Tensor):
        idx = t_idx.long().view(-1)
        a_t = self.alphas.index_select(0, idx)[0]
        s_t = self.sigmas.index_select(0, idx)[0]
        return a_t, s_t

    def q_sample(self, x0: torch.Tensor, t_idx: torch.Tensor, known_x0: torch.Tensor):
        """
        x0:       (B,H,T,D) or (B,N,T,D)
        t_idx:    scalar or shape (1,)
        known_x0: same shape as x0
        """
        a_t, s_t = self._alpha_sigma(t_idx)
        while a_t.dim() < x0.dim():
            a_t = a_t.unsqueeze(0)
            s_t = s_t.unsqueeze(0)

        noise = torch.randn_like(x0)
        x_t = a_t * x0 + (1.0 - a_t) * known_x0 + s_t * noise
        return x_t, noise

    def predict_residual(
        self,
        x_t: torch.Tensor,
        t_idx: torch.Tensor,
        known_x0: torch.Tensor,
        incomplete_graph=None,
        heads: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        When heads is provided and x_t / known_x0 are still the whole graph, they are first cropped into head sub-blocks internally.
        Returns:
          - heads is None: (B,N,T,D) or (B,H,T,D), depending on input
          - heads is not None: (B,H,T,D)
        """
        if heads is not None:
            heads = heads.to(x_t.device)
            if x_t.size(1) != heads.numel():
                x_t = x_t[:, heads]
                known_x0 = known_x0[:, heads]

        if t_idx.dim() == 0:
            t_cont = t_idx.float().unsqueeze(0) / max(1, self.T - 1)
        else:
            t_cont = t_idx.float() / max(1, self.T - 1)

        return self.net(
            x_t,
            t_cont,
            known_x0=known_x0,
            data=incomplete_graph,
            h_index=heads,
        )

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t_idx: torch.Tensor,
        known_x0: torch.Tensor,
        incomplete_graph=None,
        heads: Optional[torch.Tensor] = None,
        complete_graph=None,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Directly returns head sub-block prediction; no longer defaults to stitching back into the whole graph.
        """
        return self.predict_residual(
            x_t,
            t_idx,
            known_x0,
            incomplete_graph=incomplete_graph,
            heads=heads,
        )

    @torch.no_grad()
    def _apply_soft_vq_chunked(
        self,
        x: torch.Tensor,
        soft_vq: nn.Module,
        chunk_size: int = 2048,
    ) -> torch.Tensor:
        """
        Call soft_vq in chunks along the N dimension to avoid memory spikes from one-shot quantization.
        x: (B,N,T,D)
        """
        B, N, T, D = x.shape
        out = torch.empty_like(x)
        for bs in range(B):
            x_b = x[bs]
            for s in range(0, N, chunk_size):
                e = min(N, s + chunk_size)
                x_chunk = x_b[s:e]
                q_chunk = soft_vq(x_chunk)[0]
                out[bs, s:e] = q_chunk.to(dtype=x.dtype)
        return out

    @torch.no_grad()
    def sample(
        self,
        known_x0: torch.Tensor,          # (1,N,R,D)
        incomplete_graph=None,
        heads: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        complete_graph=None,
        num_nodes: Optional[int] = None,
        soft_vq: Optional[nn.Module] = None,
        snap_to_codebook: bool = True,
        data=None,
    ) -> torch.Tensor:
        """
        Head-based MR-DDIM/ancestral sampling:
          - Only samples head sub-blocks
          - No longer maintains the whole graph x internally
        Returns:
          (H,R,D)
        """
        device = known_x0.device
        B, N, R, D = known_x0.shape
        assert B == 1 and R == self.tokens and D == self.z_dim

        if heads is None:
            heads = torch.arange(N, device=device)
        else:
            heads = heads.to(device)

        known_head = known_x0[:, heads]  # (1,H,R,D)

        T_steps = self.T if steps is None else steps
        t_seq = torch.linspace(self.T - 1, 0, steps=T_steps, device=device).long()

        a_T = self.alphas[t_seq[0]]
        s_T = self.sigmas[t_seq[0]]

        x = (1.0 - a_T) * known_head + s_T * torch.randn_like(known_head)

        for ti in t_seq:
            x0_hat_head = self.predict_x0(
                x,
                ti,
                known_head,
                incomplete_graph=incomplete_graph if incomplete_graph is not None else data,
                heads=None,
                complete_graph=complete_graph,
                num_nodes=num_nodes,
            )

            t_prev = torch.clamp(ti - 1, min=0)
            a_prev = self.alphas[t_prev]
            s_prev = self.sigmas[t_prev]
            mean_prev = a_prev * x0_hat_head + (1.0 - a_prev) * known_head

            if (ti.item() > 0) and (self.eta > 0.0):
                noise = torch.randn_like(x)
                x = mean_prev + (self.eta * s_prev) * noise
            else:
                x = mean_prev

        out = x[0]  # (H,R,D)
       
        return out

    def train_step(
        self,
        x0: torch.Tensor,               # (1,N,R,D)
        t_idx: torch.Tensor,
        known_x0: torch.Tensor,         # (1,N,R,D)
        incomplete_graph=None,
        heads: Optional[torch.Tensor] = None,
        vq_model: Optional[nn.Module] = None,
        complete_graph=None,
        lambda_bce: Optional[float] = None,
        num_rel: Optional[int] = None,
        num_nodes: Optional[int] = None,
        data=None,
        soft_vq: Optional[nn.Module] = None,
    ):
        """
        Head-based training:
          - Only applies q_sample and denoise to head sub-blocks
          - MSE / BCE supervision also only applies to heads
        """
        device = x0.device
        B, N, R, D = x0.shape
        assert B == 1

        if heads is None:
            heads = torch.arange(N, device=device)
        else:
            heads = heads.to(device)

        if lambda_bce is None:
            lambda_bce = 0.0
 

        x0_head = x0[:, heads]          # (1,H,R,D)
        known_head = known_x0[:, heads] # (1,H,R,D)

        x_t_head, _ = self.q_sample(
            x0_head,
            t_idx,
            known_x0=known_head,
        )

        x0_hat_head = self.predict_x0(
            x_t_head,
            t_idx,
            known_head,
            incomplete_graph=incomplete_graph if incomplete_graph is not None else data,
            heads=None,
            complete_graph=complete_graph,
            num_nodes=num_nodes,
        )

        pred = x0_hat_head[0]   # (H,R,D)
        target = x0_head[0]     # (H,R,D)

        loss_bce = x0.new_zeros(())

        loss_mse = F.mse_loss(pred, target)

        loss = loss_mse
        


        return loss, loss, loss_bce

    def train_stepnew(
        self,
        x0: torch.Tensor,
        t_idx: torch.Tensor,
        known_x0: torch.Tensor,
        vq_model: Optional[nn.Module] = None,
        complete_graph=None,
        incomplete_graph=None,
        heads: Optional[torch.Tensor] = None,
        lambda_bce: float = 1.0,
        num_rel: Optional[int] = None,
        num_nodes: Optional[int] = None,
        bce_chunk_size: int = 512,
        data=None,
    ):
        loss, _, _ = self.train_step(
            x0=x0,
            t_idx=t_idx,
            known_x0=known_x0,
            incomplete_graph=incomplete_graph,
            heads=heads,
            vq_model=vq_model,
            complete_graph=complete_graph,
            lambda_bce=lambda_bce,
            num_rel=num_rel,
            num_nodes=num_nodes,
            data=data,
        )
        return loss
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name] = self.shadow[name] * self.decay + (1.0 - self.decay) * p.detach()

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)



class SimpleDiscriminator(nn.Module):
    """
    Discriminate based on a single head entity:
    Input x: (..., R, D), where the last two dimensions are (num_relations, embedding_dim)
    First flatten and concat all R relation embeddings for this head, then pass through MLP to output one logit per head.
    """
    def __init__(self, num_rel: int, dim: int, hidden: int = 256):
        super().__init__()
        self.num_rel = int(num_rel)
        self.dim = int(dim)
        self.in_dim = self.num_rel * self.dim
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., R, D)
        *head_shape, R, D = x.shape
        assert R == self.num_rel and D == self.dim, f"expect (...,{self.num_rel},{self.dim}), got {x.shape}"
        x_flat = x.reshape(-1, self.in_dim)        # (B*, R*D)
        logits = self.net(x_flat).squeeze(-1)      # (B*,)
        return logits.view(*head_shape)            # output one scalar per head
    
    

def sample_neighbors(edge_index, edge_type, nodes, num_samples):
    """
    Randomly sample a fixed number of neighbors for each node.
    """
    if edge_index.numel() == 0 or nodes.numel() == 0:
        return edge_index, edge_type

    row, col = edge_index
    # Find edges whose source nodes are in the nodes list
    mask = torch.isin(row, nodes)
    row, col = row[mask], col[mask]
    etype = edge_type[mask]

    if row.numel() == 0:
        return edge_index.new_zeros((2, 0)), edge_type.new_zeros(0)

    # Generate random weights for each edge for sampling
    perm = torch.randperm(row.size(0), device=row.device)
    row, col, etype = row[perm], col[perm], etype[perm]

    # Sort by source node
    row, sort_idx = row.sort(stable=True)
    col = col[sort_idx]
    etype = etype[sort_idx]

    # Compute edge indices per node and keep only the first num_samples
    # Use cumsum and unique_consecutive to find the start position of each node
    _, counts = torch.unique_consecutive(row, return_counts=True)
    
    # Generate internal indices within each node (0, 1, 2... count-1)
    # e.g., counts=[3, 2] -> [0, 1, 2, 0, 1]
    idx_in_node = torch.arange(row.size(0), device=row.device) - \
                  torch.repeat_interleave(row.new_tensor([0] + counts[:-1].tolist()).cumsum(0), counts)
    
    final_mask = idx_in_node < num_samples
    
    return torch.stack([row[final_mask], col[final_mask]], dim=0), etype[final_mask]
