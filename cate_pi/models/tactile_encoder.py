"""
Spatio-Temporal Tactile Encoder for CATE-π (§4.3).

Combines the strengths of two source approaches:
- Tactile-VLA's MLP: collapses temporal window into one token (dynamics, loses spatial)
- TacVLA's MLP: 36 tokens with 2D sin-cos positions (spatial, loses dynamics)

This encoder does BOTH:
1. Per-taxel embedding over a temporal window via shared MLP
2. 1D temporal conv (or GRU) to capture dynamics
3. Output tokens with 2D sin-cos positional embeddings (spatial structure)
4. Optional mean-pool summary token

Architecture (from §4.3):
    for each timestep i in [t-K+1, ..., t]:
        e_i = shared_MLP(taxel_grid_i)              # per-taxel embed
    proj = 1D temporal conv over {e_i}
    tactile_tokens = flatten(proj) + sincos_2d_pos   # spatial tokens
    tactile_summary = mean_pool(tactile_tokens)       # single-token digest
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def sincos_2d_positional_embedding(
    rows: int,
    cols: int,
    embed_dim: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate 2D sin-cos positional embeddings for a spatial grid.

    Following TacVLA's convention for spatially-structured tactile tokens.

    Args:
        rows: Number of rows in the spatial grid.
        cols: Number of columns in the spatial grid.
        embed_dim: Embedding dimension (split equally between row and col).
        device: Target device.

    Returns:
        pos_embed: (rows * cols, embed_dim) positional embeddings.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sin-cos"

    half_dim = embed_dim // 4
    omega = torch.arange(half_dim, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / half_dim))

    row_pos = torch.arange(rows, device=device, dtype=torch.float32).unsqueeze(1)  # (rows, 1)
    col_pos = torch.arange(cols, device=device, dtype=torch.float32).unsqueeze(1)  # (cols, 1)

    # Row embeddings
    row_sin = torch.sin(row_pos * omega)  # (rows, half_dim)
    row_cos = torch.cos(row_pos * omega)  # (rows, half_dim)

    # Col embeddings
    col_sin = torch.sin(col_pos * omega)  # (cols, half_dim)
    col_cos = torch.cos(col_pos * omega)  # (cols, half_dim)

    # Combine: for each (r, c) pair, concat [row_sin, row_cos, col_sin, col_cos]
    pos_embed = torch.zeros(rows, cols, embed_dim, device=device)
    for r in range(rows):
        for c in range(cols):
            pos_embed[r, c] = torch.cat([
                row_sin[r], row_cos[r], col_sin[c], col_cos[c]
            ])

    return pos_embed.reshape(rows * cols, embed_dim)


class TaxelEmbedder(nn.Module):
    """
    Per-taxel (or per-patch) MLP that embeds raw pressure readings.
    Shared across all spatial positions and temporal steps.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., input_dim) raw taxel readings.
        Returns:
            (..., output_dim) embedded taxels.
        """
        return self.mlp(x)


class TemporalAggregator(nn.Module):
    """
    Aggregates embedded taxel readings across the temporal window.
    Supports 1D temporal convolution or GRU.
    """

    def __init__(
        self,
        input_dim: int = 64,
        output_dim: int = 64,
        temporal_window: int = 4,
        mode: str = "conv",
        conv_kernel_size: int = 3,
    ):
        """
        Args:
            input_dim: Per-taxel embedding dimension.
            output_dim: Output dimension per spatial position.
            temporal_window: K — number of past timesteps.
            mode: "conv" for 1D temporal conv, "gru" for GRU.
            conv_kernel_size: Kernel size for temporal conv.
        """
        super().__init__()
        self.mode = mode
        self.temporal_window = temporal_window

        if mode == "conv":
            # 1D conv over temporal axis, per spatial position
            padding = (conv_kernel_size - 1) // 2
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=conv_kernel_size,
                    padding=padding,
                ),
                nn.GELU(),
                nn.Conv1d(
                    in_channels=output_dim,
                    out_channels=output_dim,
                    kernel_size=conv_kernel_size,
                    padding=padding,
                ),
            )
        elif mode == "gru":
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=output_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown temporal aggregation mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, num_taxels, K, embed_dim) — embedded taxel readings
               across K temporal steps.

        Returns:
            out: (B, num_taxels, output_dim) — temporally aggregated features.
        """
        B, N, K, D = x.shape

        if self.mode == "conv":
            # Reshape for Conv1d: (B*N, D, K)
            x = x.reshape(B * N, K, D).transpose(1, 2)
            out = self.temporal_conv(x)  # (B*N, output_dim, K)
            # Take the last temporal step's output
            out = out[:, :, -1]  # (B*N, output_dim)
            out = out.reshape(B, N, -1)

        elif self.mode == "gru":
            # Reshape for GRU: (B*N, K, D)
            x = x.reshape(B * N, K, D)
            _, h_n = self.gru(x)  # h_n: (1, B*N, output_dim)
            out = h_n.squeeze(0).reshape(B, N, -1)

        return out


class SpatioTemporalTactileEncoder(nn.Module):
    """
    Spatio-temporal tactile encoder combining Tactile-VLA's temporal dynamics
    with TacVLA's spatially-structured tokens (§4.3).

    Produces a fixed set of tactile tokens with 2D positional embeddings,
    ready for gating (§4.4) and injection into the VLM prefix.
    """

    def __init__(
        self,
        sensor_rows: int = 15,
        sensor_cols: int = 8,
        temporal_window: int = 4,
        taxel_input_dim: int = 3,
        taxel_embed_dim: int = 64,
        taxel_hidden_dim: int = 128,
        temporal_mode: str = "conv",
        temporal_conv_kernel: int = 3,
        num_output_tokens: int = 36,
        output_token_dim: int = 2048,
        use_summary_token: bool = True,
    ):
        """
        Args:
            sensor_rows: Spatial rows in tactile sensor array.
            sensor_cols: Spatial columns in tactile sensor array.
            temporal_window: K — number of past timesteps to aggregate.
            taxel_input_dim: Raw input dimension per taxel (1 for pressure).
            taxel_embed_dim: Intermediate embedding per taxel.
            taxel_hidden_dim: Hidden dim for taxel MLP.
            temporal_mode: "conv" or "gru" for temporal aggregation.
            temporal_conv_kernel: Kernel size for temporal conv.
            num_output_tokens: Final number of tactile tokens (36 per TacVLA).
            output_token_dim: Final token dimension (match backbone hidden_dim).
            use_summary_token: Whether to produce a mean-pool summary token.
        """
        super().__init__()
        self.sensor_rows = sensor_rows
        self.sensor_cols = sensor_cols
        self.num_taxels = sensor_rows * sensor_cols
        self.temporal_window = temporal_window
        self.num_output_tokens = num_output_tokens
        self.output_token_dim = output_token_dim
        self.use_summary_token = use_summary_token

        # Step 1: Per-taxel shared MLP
        self.taxel_embedder = TaxelEmbedder(
            input_dim=taxel_input_dim,
            hidden_dim=taxel_hidden_dim,
            output_dim=taxel_embed_dim,
        )

        # Step 2: Temporal aggregation
        self.temporal_aggregator = TemporalAggregator(
            input_dim=taxel_embed_dim,
            output_dim=taxel_embed_dim,
            temporal_window=temporal_window,
            mode=temporal_mode,
            conv_kernel_size=temporal_conv_kernel,
        )

        # Step 3: Spatial token pooling
        # Pool from num_taxels (e.g. 120) down to num_output_tokens (e.g. 36)
        # using a learned linear projection across spatial dimension
        self.spatial_pool = nn.Linear(self.num_taxels, num_output_tokens, bias=True)

        # Step 4: Project to backbone dimension
        self.output_projection = nn.Sequential(
            nn.Linear(taxel_embed_dim, output_token_dim),
            nn.GELU(),
            nn.LayerNorm(output_token_dim),
        )

        # 2D sin-cos positional embeddings
        # Compute a grid for the output tokens (approximate layout)
        out_rows = int(math.ceil(math.sqrt(num_output_tokens)))
        out_cols = int(math.ceil(num_output_tokens / out_rows))
        pos_embed_full = sincos_2d_positional_embedding(
            out_rows, out_cols, output_token_dim
        )
        # Register as buffer (not a parameter — fixed)
        self.register_buffer(
            "pos_embed",
            pos_embed_full[:num_output_tokens],  # trim to exact count
        )

    def forward(
        self,
        tactile_history: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode a temporal window of tactile sensor readings into tokens.

        Args:
            tactile_history: (B, K, rows, cols) or (B, K, rows, cols, C)
                            Raw tactile sensor readings over the last K steps.
                            If 4D, C=1 is assumed (pressure only).

        Returns:
            tactile_tokens: (B, num_output_tokens, output_token_dim)
                           Spatially-structured tactile tokens with positional encoding.
            summary_token: (B, 1, output_token_dim) or None
                          Mean-pool summary token if use_summary_token=True.
        """
        # Ensure 5D input: (B, K, rows, cols, C)
        if tactile_history.dim() == 4:
            tactile_history = tactile_history.unsqueeze(-1)

        B, K, R, C_s, C_in = tactile_history.shape
        assert K == self.temporal_window, (
            f"Expected temporal window {self.temporal_window}, got {K}"
        )
        assert R == self.sensor_rows and C_s == self.sensor_cols, (
            f"Expected sensor ({self.sensor_rows}, {self.sensor_cols}), "
            f"got ({R}, {C_s})"
        )

        # Flatten spatial dims: (B, K, num_taxels, C_in)
        x = tactile_history.reshape(B, K, self.num_taxels, C_in)

        # Step 1: Per-taxel embedding across all timesteps
        # (B, K, num_taxels, C_in) -> (B, K, num_taxels, taxel_embed_dim)
        x = self.taxel_embedder(x)

        # Reshape for temporal aggregation: (B, num_taxels, K, embed_dim)
        x = x.permute(0, 2, 1, 3)

        # Step 2: Temporal aggregation
        # (B, num_taxels, K, embed_dim) -> (B, num_taxels, embed_dim)
        x = self.temporal_aggregator(x)

        # Step 3: Spatial pooling to reduce token count
        # (B, num_taxels, embed_dim) -> transpose -> (B, embed_dim, num_taxels)
        x = x.transpose(1, 2)
        # Pool: (B, embed_dim, num_taxels) -> (B, embed_dim, num_output_tokens)
        x = self.spatial_pool(x)
        # Back to token shape: (B, num_output_tokens, embed_dim)
        x = x.transpose(1, 2)

        # Step 4: Project to backbone dim + add positional embeddings
        x = self.output_projection(x)  # (B, num_output_tokens, output_token_dim)
        x = x + self.pos_embed.unsqueeze(0)  # add 2D sin-cos positional encoding

        tactile_tokens = x

        # Optional summary token
        summary_token = None
        if self.use_summary_token:
            summary_token = tactile_tokens.mean(dim=1, keepdim=True)  # (B, 1, dim)

        return tactile_tokens, summary_token

    @classmethod
    def from_config(cls, config: dict) -> "SpatioTemporalTactileEncoder":
        """Create from the tactile sensor config dict."""
        sensor_cfg = config["tactile_sensor"]
        enc_cfg = config["tactile_encoder"]
        return cls(
            sensor_rows=sensor_cfg["rows"],
            sensor_cols=sensor_cfg["cols"],
            temporal_window=enc_cfg["temporal_window"],
            taxel_input_dim=enc_cfg.get("taxel_input_dim", 3),
            taxel_embed_dim=enc_cfg["per_taxel_embed_dim"],
            taxel_hidden_dim=enc_cfg["mlp_hidden_dim"],
            temporal_mode="gru" if enc_cfg.get("use_gru", False) else "conv",
            temporal_conv_kernel=enc_cfg.get("temporal_conv_kernel", 3),
            num_output_tokens=enc_cfg["num_tactile_tokens"],
            output_token_dim=enc_cfg["token_dim"],
            use_summary_token=enc_cfg.get("use_summary_token", True),
        )
