"""CADR-U components for dice-rl-local R3L.

PyTorch port of the JAX implementation under examples/r3l_cadr/utils/.
Adds two critic-decoupled reflective signals on top of the existing
best-of-N action selection:

  Chunk Anchor (CA):   probabilistic forward model on (h, a_chunk) → N(μ, σ²).
                       Provides ρ_dyn (predictive σ + lag-1 W2² post-correction).
  RND:                 frozen target + trained student MLP on (h, a_chunk).
                       Provides ρ_unc (input-distribution OOD signal).

The base class R3LResidualRLImgModel uses these to compute
  ρ_final = ρ_ens · ρ_dyn · ρ_unc
during best-of-N action selection. ρ_ens is derived from the existing
critic ensemble (lower-quartile advantage + per-head agreement, mirroring
the JAX chunk_gate v2 logic).
"""
from __future__ import annotations

from typing import Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_pos_emb(seq_len: int, d_model: int, device) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
    inv_freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, d_model, 2, device=device).float()
        / d_model
    )
    angles = pos * inv_freq.unsqueeze(0)
    pe = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return pe[:, :d_model]


class _CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, ffn_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_ratio),
            nn.GELU(),
            nn.Linear(d_model * ffn_ratio, d_model),
        )
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_n = self.ln1(x)
        a, _ = self.attn(x_n, x_n, x_n, attn_mask=attn_mask, need_weights=False)
        x = x + a
        x_n = self.ln2(x)
        y = self.ffn(x_n)
        return self.ln3(x + y)


class ChunkAnchor(nn.Module):
    """Predicts the next-chunk latent as N(μ, σ²) given (h, a_chunk)."""

    def __init__(
        self,
        h_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ffn_ratio: int = 4,
        log_sigma_clip_min: float = -5.0,
        log_sigma_clip_max: float = 2.0,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.log_sigma_clip_min = log_sigma_clip_min
        self.log_sigma_clip_max = log_sigma_clip_max

        self.h_proj = nn.Linear(h_dim, d_model)
        self.a_proj = nn.Linear(action_dim, d_model)
        self.pred_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            _CausalTransformerBlock(d_model, n_heads, ffn_ratio)
            for _ in range(n_layers)
        ])
        self.mu_head = nn.Linear(d_model, h_dim)
        self.log_sigma_head = nn.Linear(d_model, h_dim)

    def forward(
        self,
        h: torch.Tensor,           # (B, h_dim)
        a_chunk: torch.Tensor,     # (B, chunk_size, action_dim) or (B, chunk_size*action_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = h.shape[0]
        device = h.device

        if a_chunk.dim() == 2:
            a_chunk = a_chunk.view(B, self.chunk_size, self.action_dim)

        h_tok = self.h_proj(h).unsqueeze(1)                  # (B, 1, d)
        a_tok = self.a_proj(a_chunk)                         # (B, C, d)
        pred_tok = self.pred_token.expand(B, 1, -1)          # (B, 1, d)
        seq = torch.cat([h_tok, a_tok, pred_tok], dim=1)     # (B, C+2, d)

        L = seq.shape[1]
        seq = seq + _sinusoidal_pos_emb(L, self.d_model, device).unsqueeze(0)

        # Causal mask — `attn_mask=True` blocks attention. Float -inf form.
        mask = torch.triu(
            torch.full((L, L), float('-inf'), device=device), diagonal=1,
        )

        x = seq
        for blk in self.blocks:
            x = blk(x, attn_mask=mask)

        pred_repr = x[:, -1, :]                              # (B, d)
        mu = self.mu_head(pred_repr)
        log_sigma = self.log_sigma_head(pred_repr).clamp(
            self.log_sigma_clip_min, self.log_sigma_clip_max,
        )
        return mu, log_sigma


def gaussian_nll(
    mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    sigma2 = torch.exp(2.0 * log_sigma)
    nll = 0.5 * (
        math.log(2.0 * math.pi)
        + 2.0 * log_sigma
        + ((target - mu) ** 2) / (sigma2 + 1e-8)
    )
    return nll.sum(dim=-1).mean()


def w2_squared_to_point(
    mu: torch.Tensor, log_sigma: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """W2²(N(μ, σ²), δ_target) = ||target - μ||² + Σ σ²."""
    sigma = torch.exp(log_sigma)
    return ((target - mu) ** 2).sum(-1).mean() + (sigma ** 2).sum(-1).mean()


class RNDTarget(nn.Module):
    """Frozen random target. Never updated."""
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDStudent(nn.Module):
    """Trained student that learns to mimic the target on visited inputs."""
    def __init__(self, in_dim: int, hidden_dims=(256, 128), out_dim: int = 64):
        super().__init__()
        layers = []
        prev = in_dim
        for d in hidden_dims:
            layers += [nn.Linear(prev, d), nn.GELU()]
            prev = d
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CADRUHead(nn.Module):
    """Container holding ChunkAnchor + RND target/student.

    Owns its own Adam optimizer (separate from actor/critic optimizers) so
    the surrounding agent only needs to call `step(state, action, next_state)`.
    """

    def __init__(
        self,
        h_dim: int,
        action_dim: int,
        chunk_size: int,
        anchor_d_model: int = 256,
        anchor_n_layers: int = 4,
        anchor_n_heads: int = 4,
        anchor_ffn_ratio: int = 4,
        anchor_log_sigma_clip_min: float = -5.0,
        anchor_log_sigma_clip_max: float = 2.0,
        rnd_target_dim: int = 64,
        rnd_student_hidden=(256, 128),
        lr: float = 3e-4,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.h_dim = h_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.ema_decay = ema_decay

        self.anchor = ChunkAnchor(
            h_dim=h_dim, action_dim=action_dim, chunk_size=chunk_size,
            d_model=anchor_d_model, n_layers=anchor_n_layers,
            n_heads=anchor_n_heads, ffn_ratio=anchor_ffn_ratio,
            log_sigma_clip_min=anchor_log_sigma_clip_min,
            log_sigma_clip_max=anchor_log_sigma_clip_max,
        )

        rnd_in = h_dim + chunk_size * action_dim
        self.rnd_target = RNDTarget(rnd_in, out_dim=rnd_target_dim)
        for p in self.rnd_target.parameters():
            p.requires_grad = False
        self.rnd_student = RNDStudent(
            rnd_in, hidden_dims=rnd_student_hidden, out_dim=rnd_target_dim,
        )

        self.optimizer = torch.optim.Adam(
            list(self.anchor.parameters()) + list(self.rnd_student.parameters()),
            lr=lr,
        )

        # EMA tracking buffers (persistent across snapshots)
        self.register_buffer('sigma_norm_ema', torch.tensor(float('nan')))
        self.register_buffer('u_rnd_ema', torch.tensor(float('nan')))
        self.register_buffer('w2_ema', torch.tensor(float('nan')))

    @staticmethod
    def _ema_update(ema_buf: torch.Tensor, value: torch.Tensor, decay: float):
        if torch.isnan(ema_buf):
            ema_buf.copy_(value.detach())
        else:
            ema_buf.copy_(decay * ema_buf + (1.0 - decay) * value.detach())

    def step(
        self,
        h: torch.Tensor,
        a_chunk: torch.Tensor,         # (B, chunk_size * action_dim) or (B, C, A)
        h_next: torch.Tensor,
    ) -> dict:
        """One gradient step on CA + RND. Returns dict of float metrics."""
        if a_chunk.dim() == 3:
            a_flat = a_chunk.reshape(a_chunk.shape[0], -1)
        else:
            a_flat = a_chunk

        # CA loss = NLL + 0.1 * W2-to-point (σ-trace regularizer prevents σ→∞).
        mu, log_sigma = self.anchor(h, a_flat)
        target = h_next.detach()
        l_nll = gaussian_nll(mu, log_sigma, target)
        l_w2 = w2_squared_to_point(mu, log_sigma, target)
        ca_loss = l_nll + 0.1 * l_w2

        # RND loss
        rnd_in = torch.cat([h, a_flat], dim=-1)
        with torch.no_grad():
            t_out = self.rnd_target(rnd_in)
        s_out = self.rnd_student(rnd_in)
        rnd_loss = ((s_out - t_out) ** 2).sum(-1).mean()

        loss = ca_loss + rnd_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        sigma_mean = torch.exp(log_sigma).mean()
        self._ema_update(self.sigma_norm_ema, sigma_mean, self.ema_decay)
        self._ema_update(self.u_rnd_ema, rnd_loss, self.ema_decay)

        return {
            'cadr/ca_nll': l_nll.item(),
            'cadr/ca_w2': l_w2.item(),
            'cadr/ca_loss': ca_loss.item(),
            'cadr/rnd_loss': rnd_loss.item(),
            'cadr/sigma_mean': sigma_mean.item(),
            'cadr/sigma_norm_ema': float(self.sigma_norm_ema),
            'cadr/u_rnd_ema': float(self.u_rnd_ema),
        }

    @torch.no_grad()
    def predict(
        self, h: torch.Tensor, a_chunk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if a_chunk.dim() == 3:
            a_chunk = a_chunk.reshape(a_chunk.shape[0], -1)
        return self.anchor(h, a_chunk)

    @torch.no_grad()
    def rnd_score(
        self, h: torch.Tensor, a_chunk: torch.Tensor,
    ) -> torch.Tensor:
        if a_chunk.dim() == 3:
            a_chunk = a_chunk.reshape(a_chunk.shape[0], -1)
        rnd_in = torch.cat([h, a_chunk], dim=-1)
        s = self.rnd_student(rnd_in)
        t = self.rnd_target(rnd_in)
        return ((s - t) ** 2).sum(-1)


# ---------------------------------------------------------------------------
# Three-rho fusion helpers
# ---------------------------------------------------------------------------

def fuse_three_rho(
    rho_ens: torch.Tensor,
    rho_dyn: torch.Tensor,
    rho_unc: torch.Tensor,
    mode: str = 'and',
) -> torch.Tensor:
    rho_ens = rho_ens.clamp(0.0, 1.0)
    rho_dyn = rho_dyn.clamp(0.0, 1.0)
    rho_unc = rho_unc.clamp(0.0, 1.0)
    if mode == 'min':
        return torch.minimum(rho_ens, torch.minimum(rho_dyn, rho_unc))
    if mode == 'geometric_mean':
        return (rho_ens * rho_dyn * rho_unc).clamp_min(1e-12).pow(1.0 / 3.0)
    return rho_ens * rho_dyn * rho_unc  # 'and' (product)


def compute_rho_ens_from_critic_ensemble(
    q_all: list,                          # list of (N*B, 1) tensors, len = ensemble_size
    num_samples: int,
    batch_size: int,
    advantage_margin: float = 0.10,
    temp: float = 0.20,
    min_agreement: float = 0.60,
    strict_agreement: float = 0.80,
    quantile: float = 0.25,
    base_idx: Optional[int] = None,
) -> torch.Tensor:
    """Compute per-(sample, batch) ρ_ens from the critic ensemble.

    q_all is the list returned by `critic(..., return_all=True)`. We treat
    one of the N samples as the base candidate. If base_idx is None, the
    last sample (index N-1) is used as base.

    Returns ρ_ens of shape (N, B). For the base index, ρ_ens = 0.
    """
    base_idx = num_samples - 1 if base_idx is None else base_idx
    # (E, N*B, 1) -> (E, N, B, 1)
    q_stacked = torch.stack(q_all, dim=0)
    E = q_stacked.shape[0]
    q_stacked = q_stacked.view(E, num_samples, batch_size, 1)

    q_base = q_stacked[:, base_idx:base_idx + 1, :, :]    # (E, 1, B, 1)
    adv_heads = q_stacked - q_base                         # (E, N, B, 1)
    adv_heads = adv_heads.squeeze(-1)                      # (E, N, B)

    agreement = (adv_heads > advantage_margin).float().mean(dim=0)  # (N, B)
    lower_adv = torch.quantile(adv_heads, quantile, dim=0)          # (N, B)

    soft_lower = torch.sigmoid((lower_adv - advantage_margin) / max(temp, 1e-6))
    strict_agree = max(strict_agreement, min_agreement + 1e-6)
    agree_rho = ((agreement - min_agreement) / (strict_agree - min_agreement)).clamp(0.0, 1.0)
    rho = (soft_lower * agree_rho).clamp(0.0, 1.0)
    rho[base_idx] = 0.0  # base candidate is the reference; never selected via ρ_ens
    return rho


def compute_rho_dyn_unc_from_signals(
    sigma_score: torch.Tensor,         # (N, B) — mean σ per candidate
    u_rnd: torch.Tensor,               # (N, B)
    sigma_norm_ema: float,
    u_rnd_ema: float,
    w2_ema: float,
    last_w2_observed: Optional[float],
    tau_dyn: float = 1.0,
    tau_dyn_post: float = 1.5,
    T_dyn: float = 0.3,
    tau_unc: float = 0.0,
    T_unc: float = 0.5,
    alpha_rnd: float = 1.0,
    beta_sigma: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (rho_dyn, rho_unc, rho_dyn_prior, rho_dyn_post) shaped (N, B)."""
    eps = 1e-6
    sigma_norm = sigma_score / max(sigma_norm_ema, eps)
    rho_dyn_prior = torch.sigmoid((tau_dyn - sigma_norm) / max(T_dyn, eps))

    if last_w2_observed is None:
        rho_dyn_post = torch.ones_like(sigma_score)
    else:
        w2_norm = last_w2_observed / max(w2_ema, eps)
        post_scalar = float(
            1.0 / (1.0 + math.exp(-(tau_dyn_post - w2_norm) / max(T_dyn, eps)))
        )
        rho_dyn_post = torch.full_like(sigma_score, post_scalar)
    rho_dyn = rho_dyn_prior * rho_dyn_post

    log_u_rnd = torch.log((u_rnd / max(u_rnd_ema, eps)).clamp_min(1e-8))
    log_sigma = torch.log(sigma_norm.clamp_min(1e-8))
    rho_unc = torch.sigmoid(
        (tau_unc - alpha_rnd * log_u_rnd - beta_sigma * log_sigma) / max(T_unc, eps)
    )
    return rho_dyn, rho_unc, rho_dyn_prior, rho_dyn_post
