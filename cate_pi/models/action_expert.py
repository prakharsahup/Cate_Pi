"""
Tactile-Aware Action Expert for CATE-π (§4.6).

This is architecturally π0's "Expert Gemma" — a second transformer with
narrower hidden width (~300M params) — retargeted to output the augmented
[position, force] action vector via Conditional Flow Matching (CFM).

Key design decisions:
1. Forward pass (single denoising step):
   a_τ = τ·a + (1-τ)·ε  (linear interpolation, ε ~ N(0,I))
   v_θ = ExpertTransformer(prefix_kv=KV(S_t), suffix=[state, a_τ], time=τ)

2. Training objective (CFM):
   L_CFM = E[|| v_θ(a_τ, τ, o) - (a - ε) ||²]
   τ sampled from Beta distribution skewed toward low (noisy) values.

3. CRITICAL (from §4.6): Position (meters) and force (Newtons) live on
   wildly different numeric scales. Must quantile-normalize per-channel
   to [-1, 1] before computing the loss, or force channels get ignored.

4. Inference: integrate the learned ODE from τ=0 to τ=1 with ~10 Euler steps.

5. Attention structure inside suffix (§5):
   - State token: attends to full prefix + earlier state tokens (causal)
   - Action tokens: attend to full prefix + state + each other
     (bidirectional within action block, NOT causal to each other)
   - Force and position are jointly denoised within each chunk step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

from cate_pi.models.vlm_backbone import AdaRMSNorm, MultiHeadAttention


class ActionExpertBlock(nn.Module):
    """
    Single transformer block for the action expert.
    Includes AdaRMSNorm for flow-matching timestep conditioning.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads, head_dim, dropout)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads, head_dim, dropout)
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.norm3 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.ada_norm = AdaRMSNorm(hidden_dim, cond_dim=1)  # timestep conditioning
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        prefix_kv: torch.Tensor,
        tau: torch.Tensor,
        suffix_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T_suffix, D) suffix tokens (state + action).
            prefix_kv: (B, T_prefix, D) encoded prefix hidden states.
            tau: (B,) or (B, 1) flow-matching timestep.
            suffix_mask: (B, T_suffix, T_suffix) mask for suffix self-attention.
            prefix_mask: (B, 1, T_prefix) mask for cross-attention to prefix.
        """
        # AdaRMSNorm conditioned on τ
        residual = x
        x = self.ada_norm(x, tau)
        x = self.self_attn(x, attention_mask=suffix_mask)
        x = residual + x

        # Cross-attention to prefix (with contact-gating mask)
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, kv=prefix_kv, attention_mask=prefix_mask)
        x = residual + x

        # FFN
        residual = x
        x = self.norm3(x)
        x = self.mlp(x)
        x = residual + x

        return x


class TactileAwareActionExpert(nn.Module):
    """
    Tactile-Aware Action Expert implementing Conditional Flow Matching (CFM)
    over the augmented [position, force] action space.

    This is the core policy head: it takes encoded prefix states and produces
    action chunks via iterative denoising.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        head_dim: int = 64,
        intermediate_dim: int = 2048,
        action_dim: int = 14,
        padded_action_dim: int = 32,
        chunk_horizon: int = 50,
        prefix_dim: int = 2048,
        num_euler_steps: int = 10,
        beta_alpha: float = 1.0,
        beta_beta: float = 1.5,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim: Expert transformer hidden dimension (narrower than backbone).
            num_layers: Number of expert transformer layers.
            num_heads: Number of attention heads.
            head_dim: Per-head dimension.
            intermediate_dim: FFN intermediate dimension.
            action_dim: True action dimensions (pos + orient + gripper + force).
            padded_action_dim: Padded action dimension (32 per π0 convention).
            chunk_horizon: H — number of action steps per chunk.
            prefix_dim: Backbone hidden dim (for projection).
            num_euler_steps: ODE integration steps at inference.
            beta_alpha: Beta distribution α for τ sampling.
            beta_beta: Beta distribution β for τ sampling.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.padded_action_dim = padded_action_dim
        self.chunk_horizon = chunk_horizon
        self.num_euler_steps = num_euler_steps
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta

        # Project prefix from backbone dim to expert dim
        self.prefix_projection = nn.Linear(prefix_dim, hidden_dim)

        # Action token embedding: project noisy action vector to hidden dim
        self.action_embed = nn.Sequential(
            nn.Linear(padded_action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # State token projection (suffix copy of state)
        self.state_suffix_embed = nn.Linear(prefix_dim, hidden_dim)

        # Timestep embedding (sinusoidal + linear projection)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # scalar conditioning for AdaRMSNorm
        )

        # Expert transformer blocks
        self.blocks = nn.ModuleList([
            ActionExpertBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(hidden_dim, eps=1e-6)

        # Output head: project back to action space
        self.output_head = nn.Linear(hidden_dim, padded_action_dim)

        # Action dimension mask (for padded channels)
        self.register_buffer(
            "action_mask",
            self._build_action_mask(action_dim, padded_action_dim),
        )

        # Learnable positional embeddings for chunk positions
        self.chunk_pos_embed = nn.Parameter(
            torch.randn(1, chunk_horizon, hidden_dim) * 0.02
        )

    @staticmethod
    def _build_action_mask(action_dim: int, padded_dim: int) -> torch.Tensor:
        """Build a binary mask for active action dimensions."""
        mask = torch.zeros(padded_dim)
        mask[:action_dim] = 1.0
        return mask

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: (B,) timestep values in [0, 1].
        Returns:
            emb: (B, hidden_dim) timestep embeddings.
        """
        half_dim = self.hidden_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, hidden_dim)
        return emb

    def _interpolate_noisy_action(
        self,
        action: torch.Tensor,
        noise: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Linear interpolation for flow matching: a_τ = τ·a + (1-τ)·ε

        Args:
            action: (B, H, padded_dim) clean action targets.
            noise: (B, H, padded_dim) Gaussian noise.
            tau: (B, 1, 1) flow-matching timestep.
        Returns:
            noisy_action: (B, H, padded_dim) interpolated.
        """
        return tau * action + (1.0 - tau) * noise

    def forward_single_step(
        self,
        noisy_action: torch.Tensor,
        tau: torch.Tensor,
        prefix_hidden: torch.Tensor,
        state_hidden: torch.Tensor,
        suffix_mask: Optional[torch.Tensor] = None,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single denoising step of the action expert.

        Args:
            noisy_action: (B, H, padded_dim) noisy action tokens.
            tau: (B,) flow-matching timestep.
            prefix_hidden: (B, N_prefix, prefix_dim) encoded prefix from backbone.
            state_hidden: (B, 1, prefix_dim) state token from prefix (for suffix copy).
            suffix_mask: Optional suffix self-attention mask.

        Returns:
            v_theta: (B, H, padded_dim) predicted vector field.
        """
        B = noisy_action.shape[0]

        # Project prefix to expert dim
        prefix_kv = self.prefix_projection(prefix_hidden)  # (B, N_prefix, hidden_dim)

        # Embed noisy actions
        action_tokens = self.action_embed(noisy_action)  # (B, H, hidden_dim)
        action_tokens = action_tokens + self.chunk_pos_embed[:, :action_tokens.shape[1], :]

        # State suffix token
        state_suffix = self.state_suffix_embed(state_hidden)  # (B, 1, hidden_dim)

        # Build suffix: [state_token, action_tokens]
        suffix = torch.cat([state_suffix, action_tokens], dim=1)  # (B, 1+H, hidden_dim)

        # Timestep conditioning
        time_emb = self._sinusoidal_embedding(tau)  # (B, hidden_dim)
        tau_cond = self.time_embed(time_emb)  # (B, 1)

        # Forward through expert transformer
        x = suffix
        for block in self.blocks:
            x = block(
                x,
                prefix_kv=prefix_kv,
                tau=tau_cond,
                suffix_mask=suffix_mask,
                prefix_mask=prefix_mask
            )

        x = self.final_norm(x)

        # Extract action tokens (skip the state token)
        action_out = x[:, 1:, :]  # (B, H, hidden_dim)

        # Project to action space
        v_theta = self.output_head(action_out)  # (B, H, padded_dim)

        # Mask padded dimensions
        v_theta = v_theta * self.action_mask.unsqueeze(0).unsqueeze(0)

        return v_theta

    def compute_cfm_loss(
        self,
        action: torch.Tensor,
        prefix_hidden: torch.Tensor,
        state_hidden: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
        suffix_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the Conditional Flow Matching training loss.

        L_CFM = E[|| v_θ(a_τ, τ, o) - (a - ε) ||²]

        CRITICAL (§4.6): This loss must operate on per-channel normalized
        actions to prevent position channels from dominating force channels.

        Args:
            action: (B, H, padded_dim) target action chunk (already normalized).
            prefix_hidden: (B, N_prefix, prefix_dim) encoded prefix.
            state_hidden: (B, 1, prefix_dim) state token.

        Returns:
            loss: Scalar CFM loss.
            metrics: Dict with per-channel loss breakdown for monitoring.
        """
        B = action.shape[0]
        device = action.device

        # Sample τ from Beta distribution (skewed toward low/noisy values)
        tau = torch.distributions.Beta(self.beta_alpha, self.beta_beta).sample((B,)).to(device)

        # Sample noise
        noise = torch.randn_like(action)

        # Interpolate: a_τ = τ·a + (1-τ)·ε
        tau_expanded = tau.view(B, 1, 1)
        noisy_action = self._interpolate_noisy_action(action, noise, tau_expanded)

        # Predict vector field
        v_theta = self.forward_single_step(
            noisy_action=noisy_action,
            tau=tau,
            prefix_hidden=prefix_hidden,
            state_hidden=state_hidden,
            prefix_mask=prefix_mask,
            suffix_mask=suffix_mask,
        )

        # Target: (a - ε)
        target = action - noise

        # CFM loss (MSE)
        loss_per_dim = F.mse_loss(v_theta, target, reduction="none")  # (B, H, padded_dim)

        # Apply action mask (zero out padded dims)
        loss_per_dim = loss_per_dim * self.action_mask.unsqueeze(0).unsqueeze(0)

        # Aggregate: normalize by active channels, not padded dimensions
        loss = loss_per_dim.sum() / (B * self.chunk_horizon * self.action_dim)

        # Per-channel metrics for monitoring (§12 — detect ignored force channels)
        with torch.no_grad():
            per_channel_loss = loss_per_dim.mean(dim=(0, 1))  # (padded_dim,)
            metrics = {
                "cfm_loss": loss.item(),
                "position_loss": per_channel_loss[:3].mean().item(),
                "orientation_loss": per_channel_loss[3:9].mean().item(),
                "gripper_loss": per_channel_loss[9].item(),
                "force_ext_normal_loss": per_channel_loss[10].item() if self.action_dim > 10 else 0.0,
                "force_ext_shear_loss": per_channel_loss[11:13].mean().item() if self.action_dim > 11 else 0.0,
                "force_grasp_loss": per_channel_loss[13].item() if self.action_dim > 13 else 0.0,
            }

        return loss, metrics

    @torch.no_grad()
    def generate_actions(
        self,
        prefix_hidden: torch.Tensor,
        state_hidden: torch.Tensor,
        prefix_mask: Optional[torch.Tensor] = None,
        suffix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate an action chunk via Euler ODE integration (inference).

        Integrates from τ=0 (pure noise) to τ=1 (clean action) in
        num_euler_steps steps.

        Args:
            prefix_hidden: (B, N_prefix, prefix_dim) encoded prefix.
            state_hidden: (B, 1, prefix_dim) state token.

        Returns:
            actions: (B, H, padded_dim) predicted action chunk.
        """
        B = prefix_hidden.shape[0]
        device = prefix_hidden.device

        # Start from pure noise
        a_t = torch.randn(B, self.chunk_horizon, self.padded_action_dim, device=device)

        # Euler integration
        dt = 1.0 / self.num_euler_steps
        for step in range(self.num_euler_steps):
            tau_val = step * dt
            tau = torch.full((B,), tau_val, device=device)

            v_theta = self.forward_single_step(
                noisy_action=a_t,
                tau=tau,
                prefix_hidden=prefix_hidden,
                state_hidden=state_hidden,
                prefix_mask=prefix_mask,
                suffix_mask=suffix_mask,
            )

            a_t = a_t + v_theta * dt

        # Mask padded dimensions
        a_t = a_t * self.action_mask.unsqueeze(0).unsqueeze(0)

        return a_t

    def split_action_vector(
        self,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Split the padded action vector into named components per §6.

        Action layout (14 active dims):
          [0:3]   position (Δx, Δy, Δz)
          [3:9]   orientation (6D rotation)
          [9:10]  gripper width
          [10:11] F_ext_normal — external contact normal force (EE → environment)
          [11:13] F_ext_shear — external shear force (x, y)
          [13:14] F_grasp — internal grasp force (fingers → object)

        The external and grasp force channels are DECOUPLED (§7).
        F_ext controls how hard the end-effector presses on the environment.
        F_grasp controls how hard the fingers squeeze the object.

        Args:
            action: (B, H, padded_dim) or (B, padded_dim) action vector.

        Returns:
            Dict with named force channels explicitly separated.
        """
        if action.dim() == 2:
            action = action.unsqueeze(1)  # (B, 1, D)

        return {
            "position": action[..., :3],             # Δx, Δy, Δz
            "orientation": action[..., 3:9],         # 6D rotation
            "gripper": action[..., 9:10],            # gripper width
            "force_ext_normal": action[..., 10:11],  # external contact normal force
            "force_ext_shear": action[..., 11:13],   # external shear force (x, y)
            "force_grasp": action[..., 13:14],       # internal grasp force
        }

    @classmethod
    def from_config(cls, config: dict) -> "TactileAwareActionExpert":
        """Create from config dicts."""
        expert_cfg = config["action_expert"]
        action_cfg = config["action_space"]["gripper"]
        return cls(
            hidden_dim=expert_cfg["hidden_dim"],
            num_layers=expert_cfg["num_layers"],
            num_heads=expert_cfg["num_heads"],
            head_dim=expert_cfg["head_dim"],
            intermediate_dim=expert_cfg["intermediate_dim"],
            action_dim=action_cfg["total_dim"],
            padded_action_dim=action_cfg["padded_dim"],
            chunk_horizon=expert_cfg["chunk_horizon"],
            prefix_dim=config["backbone"]["gemma"]["hidden_dim"],
            num_euler_steps=expert_cfg["flow_matching"]["num_euler_steps"],
            beta_alpha=expert_cfg["flow_matching"]["beta_dist_alpha"],
            beta_beta=expert_cfg["flow_matching"]["beta_dist_beta"],
        )
