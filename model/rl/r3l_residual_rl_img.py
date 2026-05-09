"""R3L-aligned residual head for DICE-RL image Robomimic post-training.

The rest of the training stack deliberately stays on the DICE-RL path
(RLPD, replay, n-step returns, TD target, optimizer scheduling).  This module
only changes the residual policy surface:
  - frozen base policy plus smooth bounded residual correction
  - optional final action clipping to the normalized action range
  - delayed best-of-N q-chunk action selection using the current critic
  - (CADR-U, optional) Chunk Anchor + RND signals fused with critic-ensemble
    agreement into ρ_final = ρ_ens · ρ_dyn · ρ_unc; gates best-of-N selection.
"""

from typing import Optional, Tuple

import torch

from model.rl.distill_residual_rl_img import DistillResidualRLImgModel
from model.rl.cadr_u_modules import (
    CADRUHead,
    fuse_three_rho,
    compute_rho_ens_from_critic_ensemble,
    compute_rho_dyn_unc_from_signals,
)


class R3LResidualRLImgModel(DistillResidualRLImgModel):
    def __init__(
        self,
        *args,
        max_correction: float = 0.15,
        clip_final_action: bool = True,
        action_clip_min: float = -1.0,
        action_clip_max: float = 1.0,
        q_chunk_num_samples: int = 4,
        q_chunk_critic_reduction: str = "min",
        q_chunk_warmup_steps: int = 50000,
        # --- CADR-U ---
        use_cadr_u: bool = False,
        cadr_anchor_d_model: int = 256,
        cadr_anchor_n_layers: int = 4,
        cadr_anchor_n_heads: int = 4,
        cadr_anchor_log_sigma_clip_min: float = -5.0,
        cadr_anchor_log_sigma_clip_max: float = 2.0,
        cadr_rnd_target_dim: int = 64,
        cadr_rnd_student_hidden: tuple = (256, 128),
        cadr_lr: float = 3e-4,
        cadr_warmup_steps: int = 0,        # SAC step before CADR-U gate activates
        rho_fusion_mode: str = "and",
        chunk_gate_advantage_margin: float = 0.10,
        chunk_gate_temp: float = 0.20,
        chunk_gate_min_agreement: float = 0.60,
        chunk_gate_strict_agreement: float = 0.80,
        chunk_gate_quantile: float = 0.25,
        chunk_gate_min_scale: float = 0.70,
        chunk_gate_include_base_candidate: bool = True,
        chunk_gate_base_margin: float = 0.0,
        tau_dyn: float = 1.0,
        tau_dyn_post: float = 1.5,
        T_dyn: float = 0.3,
        tau_unc: float = 0.0,
        T_unc: float = 0.5,
        alpha_rnd: float = 1.0,
        beta_sigma: float = 0.5,
        cadr_signal_ema_decay: float = 0.99,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_correction = max_correction
        self.clip_final_action = clip_final_action
        self.action_clip_min = action_clip_min
        self.action_clip_max = action_clip_max
        self.q_chunk_num_samples = q_chunk_num_samples
        self.q_chunk_critic_reduction = q_chunk_critic_reduction
        self.q_chunk_warmup_steps = q_chunk_warmup_steps

        # CADR-U flags
        self.use_cadr_u = bool(use_cadr_u)
        self.cadr_warmup_steps = int(cadr_warmup_steps)
        self.rho_fusion_mode = rho_fusion_mode
        self.chunk_gate_advantage_margin = chunk_gate_advantage_margin
        self.chunk_gate_temp = chunk_gate_temp
        self.chunk_gate_min_agreement = chunk_gate_min_agreement
        self.chunk_gate_strict_agreement = chunk_gate_strict_agreement
        self.chunk_gate_quantile = chunk_gate_quantile
        self.chunk_gate_min_scale = chunk_gate_min_scale
        self.chunk_gate_include_base_candidate = chunk_gate_include_base_candidate
        self.chunk_gate_base_margin = chunk_gate_base_margin
        self.tau_dyn = tau_dyn
        self.tau_dyn_post = tau_dyn_post
        self.T_dyn = T_dyn
        self.tau_unc = tau_unc
        self.T_unc = T_unc
        self.alpha_rnd = alpha_rnd
        self.beta_sigma = beta_sigma

        # CADR-U head — uses flattened (cond_steps × augmented_obs_dim) as h.
        # Anchor predicts h_next at the next chunk boundary.
        if self.use_cadr_u:
            cond_steps = getattr(self, 'cond_steps', 1)
            self._h_dim = int(cond_steps * self.obs_dim)
            self.cadr_u = CADRUHead(
                h_dim=self._h_dim,
                action_dim=self.action_dim,
                chunk_size=self.horizon_steps,
                anchor_d_model=cadr_anchor_d_model,
                anchor_n_layers=cadr_anchor_n_layers,
                anchor_n_heads=cadr_anchor_n_heads,
                anchor_log_sigma_clip_min=cadr_anchor_log_sigma_clip_min,
                anchor_log_sigma_clip_max=cadr_anchor_log_sigma_clip_max,
                rnd_target_dim=cadr_rnd_target_dim,
                rnd_student_hidden=tuple(cadr_rnd_student_hidden),
                lr=cadr_lr,
                ema_decay=cadr_signal_ema_decay,
            )
        else:
            self.cadr_u = None

        # Lag-1 W2 cache for ρ_dyn_post
        self._last_anchor_mu = None        # (h_dim,)
        self._last_anchor_log_sigma = None
        self._last_w2_observed: Optional[float] = None

        # Rolling buffer of (ρ_ens, ρ_dyn, ρ_unc) per-candidate vectors —
        # used for the non-redundancy correlation diagnostic. Capped to
        # ~5000 paired samples; agent should pull + clear via
        # `pop_cadr_corr_diag()` and log per training step.
        self._cadr_diag_buf_max = 5000
        self._cadr_diag_ens: list = []
        self._cadr_diag_dyn: list = []
        self._cadr_diag_unc: list = []

    # ------------------------------------------------------------------
    # Action selection (unchanged signature)
    # ------------------------------------------------------------------

    def get_action(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        return_pretrained_actions: bool = False,
    ):
        with torch.no_grad():
            output = self.pretrained_flow_policy.forward_from_features(
                features=state,
                init_noise=noise,
            )
            pretrained_actions = output.trajectories.detach()

        if self.condition_residual_on_base_action:
            raw_residual_actions = self.actor(state, pretrained_actions)
        else:
            raw_residual_actions = self.actor(state, noise)

        residual_actions = torch.tanh(raw_residual_actions) * self.max_correction
        total_actions = pretrained_actions + residual_actions
        if self.clip_final_action:
            total_actions = torch.clamp(
                total_actions,
                min=self.action_clip_min,
                max=self.action_clip_max,
            )

        if return_pretrained_actions:
            return total_actions, pretrained_actions
        return total_actions

    def get_exploration_action(
        self,
        state: torch.Tensor,
        num_samples: int = 4,
        exploration_strategy: str = "r3l_q_chunk",
        training_step: int = 0,
        replay_flow_model=None,
        replay_flow_config=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exploration_strategy != "r3l_q_chunk":
            return super().get_exploration_action(
                state=state,
                num_samples=num_samples,
                exploration_strategy=exploration_strategy,
                training_step=training_step,
                replay_flow_model=replay_flow_model,
                replay_flow_config=replay_flow_config,
            )

        batch_size = state.shape[0]
        device = state.device
        num_samples = max(1, int(num_samples or self.q_chunk_num_samples))

        # Pre-warmup or single-sample: identical to the simple residual path.
        if training_step <= self.q_chunk_warmup_steps or num_samples == 1:
            noise = torch.randn(batch_size, self.horizon_steps, self.action_dim, device=device)
            action = self.get_action(state, noise)
            return action, noise

        # Sample N candidate actions.
        noise_samples = torch.randn(
            num_samples, batch_size, self.horizon_steps, self.action_dim,
            device=device,
        )
        state_flat = (
            state.unsqueeze(0)
            .expand(num_samples, -1, -1, -1)
            .reshape(num_samples * batch_size, *state.shape[1:])
        )
        noise_flat = noise_samples.reshape(
            num_samples * batch_size, self.horizon_steps, self.action_dim,
        )

        with torch.no_grad():
            actions_flat = self.get_action(state_flat, noise_flat)
            q_all = self.critic(state_flat, noise_flat, actions_flat, return_all=True)

            cadr_active = (
                self.use_cadr_u
                and self.cadr_u is not None
                and training_step >= self.cadr_warmup_steps
            )

            if not cadr_active:
                # Original min-Q best-of-N argmax (unchanged behavior).
                q_stacked = torch.stack(q_all, dim=0)
                q_stacked = q_stacked.view(len(q_all), num_samples, batch_size, 1)
                if self.q_chunk_critic_reduction == "min":
                    q_scores = q_stacked.min(dim=0).values
                elif self.q_chunk_critic_reduction == "mean":
                    q_scores = q_stacked.mean(dim=0)
                else:
                    raise ValueError(
                        f"Unknown q_chunk_critic_reduction: {self.q_chunk_critic_reduction}"
                    )
                selected = q_scores.squeeze(-1).argmax(dim=0)
            else:
                # CADR-U 3-rho fusion
                # ρ_ens — append a base candidate (δ=0 ≈ flow policy actions) as
                # the reference, then re-derive per-candidate agreement signals.
                with torch.no_grad():
                    base_actions_flat = self.pretrained_flow_policy.forward_from_features(
                        features=state_flat,
                        init_noise=noise_flat,
                    ).trajectories.detach()
                # Concat (N candidates) + (1 base) along sample axis.
                # We re-evaluate Q on the base too.
                base_q_all = self.critic(state_flat, noise_flat, base_actions_flat, return_all=True)
                # Stack into one big list of length E with shape ((N+1)*B, 1)
                fused_q_all = []
                for q_cand, q_base in zip(q_all, base_q_all):
                    q_cand_v = q_cand.view(num_samples, batch_size, 1)
                    q_base_v = q_base.view(num_samples, batch_size, 1).mean(dim=0, keepdim=True)
                    fused = torch.cat([q_cand_v, q_base_v], dim=0)             # (N+1, B, 1)
                    fused_q_all.append(fused.reshape((num_samples + 1) * batch_size, 1))
                rho_ens_per = compute_rho_ens_from_critic_ensemble(
                    fused_q_all,
                    num_samples=num_samples + 1,
                    batch_size=batch_size,
                    advantage_margin=self.chunk_gate_advantage_margin,
                    temp=self.chunk_gate_temp,
                    min_agreement=self.chunk_gate_min_agreement,
                    strict_agreement=self.chunk_gate_strict_agreement,
                    quantile=self.chunk_gate_quantile,
                    base_idx=num_samples,                                     # last is base
                )                                                              # (N+1, B)
                rho_ens_per = rho_ens_per[:num_samples]                        # (N, B)

                # ρ_dyn / ρ_unc — flatten state to h, score each candidate.
                h = state_flat.view(num_samples * batch_size, -1)              # (N*B, h_dim)
                a_flat = actions_flat.view(num_samples * batch_size, -1)
                mu, log_sigma = self.cadr_u.predict(h, a_flat)
                sigma = torch.exp(log_sigma)
                sigma_score = sigma.mean(dim=-1).view(num_samples, batch_size)
                u_rnd = self.cadr_u.rnd_score(h, a_flat).view(num_samples, batch_size)

                rho_dyn, rho_unc, _, _ = compute_rho_dyn_unc_from_signals(
                    sigma_score=sigma_score,
                    u_rnd=u_rnd,
                    sigma_norm_ema=float(self.cadr_u.sigma_norm_ema),
                    u_rnd_ema=float(self.cadr_u.u_rnd_ema),
                    w2_ema=float(self.cadr_u.w2_ema),
                    last_w2_observed=self._last_w2_observed,
                    tau_dyn=self.tau_dyn,
                    tau_dyn_post=self.tau_dyn_post,
                    T_dyn=self.T_dyn,
                    tau_unc=self.tau_unc,
                    T_unc=self.T_unc,
                    alpha_rnd=self.alpha_rnd,
                    beta_sigma=self.beta_sigma,
                )

                rho_final = fuse_three_rho(rho_ens_per, rho_dyn, rho_unc, mode=self.rho_fusion_mode)

                # Buffer the per-candidate vectors for correlation diagnostic.
                # rho_ens_per / rho_dyn / rho_unc are (N, B) tensors — flatten
                # to (N*B,) and append. We keep CPU numpy to avoid keeping
                # GPU references around between train steps.
                self._cadr_diag_ens.append(rho_ens_per.detach().flatten().cpu().numpy())
                self._cadr_diag_dyn.append(rho_dyn.detach().flatten().cpu().numpy())
                self._cadr_diag_unc.append(rho_unc.detach().flatten().cpu().numpy())
                while sum(a.size for a in self._cadr_diag_ens) > self._cadr_diag_buf_max:
                    self._cadr_diag_ens.pop(0)
                    self._cadr_diag_dyn.pop(0)
                    self._cadr_diag_unc.pop(0)

                # Score candidates by ρ_final · Q_min (or Q_mean).
                q_stacked = torch.stack(q_all, dim=0)
                q_stacked = q_stacked.view(len(q_all), num_samples, batch_size, 1)
                if self.q_chunk_critic_reduction == "min":
                    q_scores = q_stacked.min(dim=0).values.squeeze(-1)         # (N, B)
                else:
                    q_scores = q_stacked.mean(dim=0).squeeze(-1)               # (N, B)
                selected = (rho_final * q_scores).argmax(dim=0)                # (B,)

                # Cache anchor prediction for the selected candidate so we can
                # compute lag-1 W2² on the next chunk.
                idx_in_flat = selected + torch.arange(batch_size, device=device) * 0  # (B,)
                # Use first batch element only for the global cache.
                b0 = 0
                gather_idx = selected[b0] * batch_size + b0
                self._last_anchor_mu = mu[gather_idx].detach().clone()
                self._last_anchor_log_sigma = log_sigma[gather_idx].detach().clone()

            actions = actions_flat.view(
                num_samples, batch_size, self.horizon_steps, self.action_dim,
            )
            selected_actions = torch.stack(
                [actions[selected[b], b] for b in range(batch_size)]
            )
            selected_noise = torch.stack(
                [noise_samples[selected[b], b] for b in range(batch_size)]
            )

        return selected_actions, selected_noise

    # ------------------------------------------------------------------
    # CADR-U training hook (called from the agent's update loop)
    # ------------------------------------------------------------------

    def cadr_u_step(
        self,
        state: torch.Tensor,           # (B, cond_steps, augmented_obs_dim)
        action: torch.Tensor,          # (B, horizon_steps, action_dim)
        next_state: torch.Tensor,
    ) -> dict:
        """One CA + RND gradient step. Returns dict of float metrics."""
        if not self.use_cadr_u or self.cadr_u is None:
            return {}
        h = state.reshape(state.shape[0], -1).detach()
        h_next = next_state.reshape(next_state.shape[0], -1).detach()
        a_flat = action.reshape(action.shape[0], -1).detach()
        return self.cadr_u.step(h, a_flat, h_next)

    def pop_cadr_corr_diag(self) -> dict:
        """Compute and clear (ρ_ens, ρ_dyn, ρ_unc) correlations.

        Returns a dict with keys 'cadr/corr_ens_dyn', 'cadr/corr_ens_unc',
        'cadr/corr_dyn_unc' or empty dict if buffer is empty / no variance.
        Called by the agent on its log interval.
        """
        if not self._cadr_diag_ens:
            return {}
        import numpy as np
        try:
            e = np.concatenate(self._cadr_diag_ens).astype(np.float64)
            d = np.concatenate(self._cadr_diag_dyn).astype(np.float64)
            u = np.concatenate(self._cadr_diag_unc).astype(np.float64)
        finally:
            # Clear regardless to bound memory
            self._cadr_diag_ens.clear()
            self._cadr_diag_dyn.clear()
            self._cadr_diag_unc.clear()

        def _corr(a, b):
            if a.std() < 1e-6 or b.std() < 1e-6:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        return {
            'cadr/corr_ens_dyn': _corr(e, d),
            'cadr/corr_ens_unc': _corr(e, u),
            'cadr/corr_dyn_unc': _corr(d, u),
            'cadr/corr_n_samples': int(e.size),
        }

    def update_anchor_post_observation(self, h_actual: torch.Tensor) -> None:
        """Update lag-1 W2² cache after a chunk has executed.

        h_actual: (h_dim,) — the actually-observed h at the chunk boundary.
        """
        if (not self.use_cadr_u or self.cadr_u is None
                or self._last_anchor_mu is None
                or self._last_anchor_log_sigma is None):
            return
        h_actual = h_actual.reshape(-1).detach()
        sigma = torch.exp(self._last_anchor_log_sigma)
        w2 = float(((h_actual - self._last_anchor_mu) ** 2).sum() + (sigma ** 2).sum())
        self._last_w2_observed = w2
        # Update the head's W2 EMA as well.
        if torch.isnan(self.cadr_u.w2_ema):
            self.cadr_u.w2_ema.copy_(torch.tensor(w2))
        else:
            self.cadr_u.w2_ema.mul_(0.99).add_(torch.tensor(w2 * 0.01))
