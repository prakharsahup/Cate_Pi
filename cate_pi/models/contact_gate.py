"""
Contact Detector and Gate for CATE-π (§4.4).

Implements two gating mechanisms:

1. Hard gate (paper-faithful, TacVLA):
   c_t = 1 if count(taxels where pressure > p_thresh) > n_thresh else 0
   Applied at TWO levels (both required per TacVLA):
   - Attention mask gate: excludes tactile tokens from attention when c_t = 0
   - Embedding gate: zeroes the token values themselves when c_t = 0
   This double application matters: masking alone still lets positional/embedding
   offsets leak in; zeroing alone still lets attention attend to all-zero tokens.

2. Soft learnable gate (CATE-π extension, recommended for "perform well" build):
   g_t = sigmoid(MLP(mean_pool(z_tac)))
   L_gate = BCE(g_t, c_t)  # heuristic c_t as weak label
   Gradients from main CFM loss flow through g_t, so the network can learn to
   keep tactile influence partially active in ambiguous near-contact frames.
   This directly addresses TacVLA's own stated limitation about hard thresholds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HardContactGate(nn.Module):
    """
    Paper-faithful hard contact gate (TacVLA's design).

    Binary threshold on raw tactile pressure readings.
    No learnable parameters — thresholds are calibrated from data.
    """

    def __init__(
        self,
        pressure_threshold: float = 0.05,
        count_threshold: int = 5,
    ):
        """
        Args:
            pressure_threshold: p_thresh — minimum pressure per taxel to count as active.
            count_threshold: n_thresh — minimum active taxels to declare contact.
        """
        super().__init__()
        self.pressure_threshold = pressure_threshold
        self.count_threshold = count_threshold

    def forward(
        self,
        raw_tactile: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute binary contact flag from raw tactile readings.

        Args:
            raw_tactile: (B, rows, cols) or (B, rows*cols) raw pressure readings
                        (current timestep only — this is a per-frame detector).

        Returns:
            contact_flag: (B,) binary flags, 1.0 = contact, 0.0 = no contact.
        """
        if raw_tactile.dim() == 4:
            raw_tactile = raw_tactile[..., 0]  # (B, rows, cols, 3) -> normal channel
            
        if raw_tactile.dim() == 3:
            raw_tactile = raw_tactile.flatten(1)  # (B, rows*cols)

        # Count taxels above pressure threshold
        active_count = (raw_tactile > self.pressure_threshold).float().sum(dim=-1)  # (B,)

        # Binary contact decision
        contact_flag = (active_count > self.count_threshold).float()  # (B,)

        return contact_flag


class SoftContactGate(nn.Module):
    """
    Learnable soft contact gate (CATE-π extension).

    Predicts a continuous gate value g_t ∈ [0, 1] from tactile token embeddings.
    Trained with auxiliary BCE loss against the hard gate's heuristic labels,
    but also receives gradients from the main CFM loss.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        """
        Args:
            input_dim: Dimension of tactile token embeddings.
            hidden_dim: Hidden dimension of the gate MLP.
            num_layers: Number of MLP layers.
        """
        super().__init__()

        layers = []
        in_dim = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))  # scalar output (pre-sigmoid)

        self.gate_mlp = nn.Sequential(*layers)

    def forward(
        self,
        tactile_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict soft gate value from tactile token embeddings.

        Args:
            tactile_tokens: (B, num_tokens, dim) encoded tactile tokens.

        Returns:
            gate_value: (B,) continuous gate values in [0, 1].
        """
        # Mean pool across tokens
        pooled = tactile_tokens.mean(dim=1)  # (B, dim)

        # MLP → scalar → sigmoid
        logit = self.gate_mlp(pooled).squeeze(-1)  # (B,)
        gate_value = torch.sigmoid(logit)

        return gate_value


class ContactGate(nn.Module):
    """
    Unified contact gate module for CATE-π.

    Supports both hard (paper-faithful) and soft (learnable) gating modes,
    and applies double gating: attention mask + embedding zeroing.

    In soft mode, also computes the auxiliary BCE loss for training.
    """

    def __init__(
        self,
        mode: str = "soft",
        # Hard gate params
        pressure_threshold: float = 0.05,
        count_threshold: int = 5,
        # Soft gate params
        token_dim: int = 2048,
        mlp_hidden_dim: int = 64,
        mlp_layers: int = 2,
        bce_loss_weight: float = 0.1,
        # Double gating
        double_gate: bool = True,
    ):
        """
        Args:
            mode: "hard" for binary threshold, "soft" for learnable gate.
            pressure_threshold: For hard gate calibration.
            count_threshold: For hard gate calibration.
            token_dim: Tactile token embedding dimension (for soft gate).
            mlp_hidden_dim: Soft gate MLP hidden dimension.
            mlp_layers: Soft gate MLP depth.
            bce_loss_weight: Weight for auxiliary contact BCE loss.
            double_gate: Apply both mask and embedding gating (recommended).
        """
        super().__init__()
        self.mode = mode
        self.bce_loss_weight = bce_loss_weight
        self.double_gate = double_gate

        # Always have the hard gate (used as weak labels for soft gate too)
        self.hard_gate = HardContactGate(
            pressure_threshold=pressure_threshold,
            count_threshold=count_threshold,
        )

        # Soft gate (only created if needed)
        self.soft_gate = None
        if mode == "soft":
            self.soft_gate = SoftContactGate(
                input_dim=token_dim,
                hidden_dim=mlp_hidden_dim,
                num_layers=mlp_layers,
            )

    def forward(
        self,
        tactile_tokens: torch.Tensor,
        raw_tactile: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply contact gating to tactile tokens.

        Implements the double gating from TacVLA:
        1. Compute gate value (hard or soft)
        2. Zero tactile token embeddings: z̃_tac = gate * z_tac
        3. Return gate value for attention mask construction

        Args:
            tactile_tokens: (B, num_tokens, dim) encoded tactile token embeddings.
            raw_tactile: (B, rows, cols) raw tactile reading for contact detection.

        Returns:
            gated_tokens: (B, num_tokens, dim) gated tactile tokens
                         (zeroed when no contact).
            gate_value: (B,) gate values — binary for hard, continuous for soft.
                       Used by attention_masks.py to build the mask.
            aux_loss: Scalar BCE loss for soft gate training, or None for hard gate.
        """
        # Always compute the hard contact flag (used as label for soft gate)
        hard_flag = self.hard_gate(raw_tactile)  # (B,)

        if self.mode == "hard":
            gate_value = hard_flag
            aux_loss = None
        elif self.mode == "soft":
            gate_value = self.soft_gate(tactile_tokens)  # (B,) ∈ [0, 1]
            # BCE loss: train soft gate against heuristic hard labels
            aux_loss = self.bce_loss_weight * F.binary_cross_entropy(
                gate_value, hard_flag.detach()
            )
        else:
            raise ValueError(f"Unknown gate mode: {self.mode}")

        # === Double gating (§4.4) ===
        # Embedding gate: z̃_tac = gate * z_tac
        if self.double_gate:
            gate_expanded = gate_value.view(-1, 1, 1)  # (B, 1, 1)
            gated_tokens = tactile_tokens * gate_expanded
        else:
            gated_tokens = tactile_tokens

        # The attention mask gate is applied externally via gate_value
        # passed to attention_masks.build_prefix_mask()

        return gated_tokens, gate_value, aux_loss

    @classmethod
    def from_config(cls, config: dict) -> "ContactGate":
        """Create from the tactile sensor config dict."""
        gate_cfg = config["contact_gate"]
        return cls(
            mode=gate_cfg["mode"],
            pressure_threshold=gate_cfg["hard"]["pressure_threshold"],
            count_threshold=gate_cfg["hard"]["count_threshold"],
            token_dim=config["tactile_encoder"]["token_dim"],
            mlp_hidden_dim=gate_cfg["soft"]["mlp_hidden_dim"],
            mlp_layers=gate_cfg["soft"]["mlp_layers"],
            bce_loss_weight=gate_cfg["soft"]["bce_loss_weight"],
            double_gate=gate_cfg["double_gate"],
        )
