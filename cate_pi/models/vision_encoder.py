"""
Vision Encoder for CATE-π (§4.1).

Wraps SigLIP (the vision half of PaliGemma) exactly as π0 does.
- Patch size 14, resolution 224×224 → 256 tokens/image
- Two cameras (front + wrist) → 512 vision tokens
- No new vision tower — reuses SigLIP pretrained weights.

Design note from §4.1: If compute-constrained, reduce resolution before
reducing cameras — wrist-camera visibility is critical for contact-rich tasks
when the gripper occludes the front view.
"""

import torch
import torch.nn as nn
from typing import Optional, List


class VisionEncoder(nn.Module):
    """
    SigLIP-based vision encoder for CATE-π.

    Encodes multi-camera RGB images into a flat sequence of vision tokens
    ready for the VLM prefix.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        hidden_dim: int = 1152,
        output_dim: int = 2048,
        num_cameras: int = 2,
        pretrained_model_name: str = "google/siglip-so400m-patch14-384",
    ):
        """
        Args:
            image_size: Input image resolution (square). Default 224 per §4.1.
            patch_size: ViT patch size. Default 14 for SigLIP.
            hidden_dim: SigLIP hidden dimension (1152 for SO400M).
            output_dim: Target dimension to project into (backbone hidden_dim).
            num_cameras: Number of camera views (2 = front + wrist).
            pretrained_model_name: HuggingFace model identifier for SigLIP.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_cameras = num_cameras
        self.tokens_per_image = (image_size // patch_size) ** 2  # 256 for 224/14

        # SigLIP vision transformer backbone
        # In practice, this would be loaded from a pretrained PaliGemma checkpoint.
        # Here we define the architecture for standalone use / testing.
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        # Learned position embeddings for patches
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.tokens_per_image, hidden_dim) * 0.02
        )

        # Minimal transformer blocks for standalone testing
        # (In full deployment, these are replaced by pretrained SigLIP layers)
        self.encoder_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=16,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=12,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Project from SigLIP hidden dim to backbone hidden dim
        # (needed when SigLIP hidden != Gemma hidden)
        self.projection = nn.Linear(hidden_dim, output_dim, bias=True) if hidden_dim != output_dim else nn.Identity()

        # Per-camera learned embedding to distinguish front vs. wrist
        self.camera_embed = nn.Embedding(num_cameras, output_dim)

    def encode_single_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image into patch tokens.

        Args:
            image: (B, 3, H, W) normalized RGB image.

        Returns:
            tokens: (B, num_patches, output_dim) vision tokens.
        """
        # Patch embedding: (B, 3, H, W) -> (B, hidden_dim, H/P, W/P)
        x = self.patch_embed(image)

        # Reshape to sequence: (B, hidden_dim, grid, grid) -> (B, num_patches, hidden_dim)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_dim)

        # Add positional embeddings
        x = x + self.pos_embed[:, :x.shape[1], :]

        # Transformer encoder
        x = self.encoder_layers(x)
        x = self.layer_norm(x)

        # Project to backbone dimension
        x = self.projection(x)

        return x

    def forward(
        self,
        images: torch.Tensor,
        camera_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode multi-camera images into a flat sequence of vision tokens.

        Args:
            images: (B, num_cameras, 3, H, W) multi-camera RGB images.
            camera_ids: (num_cameras,) integer IDs for camera type embedding.
                        Default: [0, 1, ..., num_cameras-1].

        Returns:
            vision_tokens: (B, num_cameras * tokens_per_image, output_dim)
                          Flat sequence of all vision tokens, ready for prefix.
        """
        B, N_cam, C, H, W = images.shape
        assert N_cam == self.num_cameras, (
            f"Expected {self.num_cameras} cameras, got {N_cam}"
        )

        if camera_ids is None:
            camera_ids = torch.arange(N_cam, device=images.device)

        all_tokens = []
        for cam_idx in range(N_cam):
            # Encode each camera view
            cam_image = images[:, cam_idx]  # (B, 3, H, W)
            tokens = self.encode_single_image(cam_image)  # (B, tokens_per_image, output_dim)

            # Add camera-specific embedding
            cam_emb = self.camera_embed(camera_ids[cam_idx].unsqueeze(0))  # (1, output_dim)
            tokens = tokens + cam_emb.unsqueeze(1)  # broadcast over token dim

            all_tokens.append(tokens)

        # Concatenate all camera tokens: (B, num_cameras * tokens_per_image, output_dim)
        vision_tokens = torch.cat(all_tokens, dim=1)

        return vision_tokens

    @property
    def total_tokens(self) -> int:
        """Total number of vision tokens produced (across all cameras)."""
        return self.num_cameras * self.tokens_per_image

    @classmethod
    def from_config(cls, config: dict) -> "VisionEncoder":
        """Create from the backbone config dict."""
        vision_cfg = config["backbone"]["vision"]
        return cls(
            image_size=vision_cfg["image_size"],
            patch_size=vision_cfg["patch_size"],
            hidden_dim=vision_cfg["hidden_dim"],
            output_dim=config["backbone"]["gemma"]["hidden_dim"],
            num_cameras=vision_cfg["num_cameras"],
        )
