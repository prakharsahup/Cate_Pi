"""
CATE-π Training Phase 2: End-to-end CFM Fine-tuning (§8, Phase 2).

Unfreeze tactile encoder, gate, and action expert.
LoRA-adapt the shared VLM backbone.
Train on Conditional Flow Matching loss over demonstration data
with the augmented [position, force] action space.

This phase gives Tactile-VLA's core capability:
force-grounded, tactile-conditioned action generation.

CRITICAL (§4.6, §12):
- Monitor per-channel loss contribution to catch force channel being ignored.
- Track gate distribution to catch collapse.
- Verify per-channel normalization before long runs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml
import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

from cate_pi.models.vision_encoder import VisionEncoder
from cate_pi.models.tactile_encoder import SpatioTemporalTactileEncoder
from cate_pi.models.contact_gate import ContactGate
from cate_pi.models.vlm_backbone import VLMBackbone
from cate_pi.models.action_expert import TactileAwareActionExpert
from cate_pi.models.attention_masks import CATEPiAttentionMaskBuilder
from cate_pi.data.dataset import create_dataloader
from cate_pi.data.action_normalization import ActionNormalizer


class CATEPiModel(nn.Module):
    """
    Full CATE-π model for Phase 2 training.

    Combines all modules into a single forward pass for
    end-to-end CFM training.
    """

    def __init__(
        self,
        vision_encoder: VisionEncoder,
        tactile_encoder: SpatioTemporalTactileEncoder,
        contact_gate: ContactGate,
        vlm_backbone: VLMBackbone,
        action_expert: TactileAwareActionExpert,
        mask_builder: CATEPiAttentionMaskBuilder,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.tactile_encoder = tactile_encoder
        self.contact_gate = contact_gate
        self.vlm_backbone = vlm_backbone
        self.action_expert = action_expert
        self.mask_builder = mask_builder

    def forward(
        self,
        images: torch.Tensor,
        language_ids: torch.Tensor,
        state: torch.Tensor,
        tactile_history: torch.Tensor,
        raw_tactile: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float], Optional[torch.Tensor]]:
        """
        Full forward pass for CFM training.

        Args:
            images: (B, N_cam, 3, H, W) multi-camera images.
            language_ids: (B, L) language token IDs.
            state: (B, state_dim) proprioceptive state.
            tactile_history: (B, K, rows, cols) tactile history.
            raw_tactile: (B, rows, cols) current tactile reading.
            action_chunk: (B, H, padded_dim) target action chunk.

        Returns:
            cfm_loss: Scalar CFM loss.
            metrics: Dict with per-channel loss, gate stats, etc.
            gate_aux_loss: Optional auxiliary gate loss.
        """
        # 1. Vision encoding
        vision_tokens = self.vision_encoder(images)  # (B, N_vis, D)

        # 2. Tactile encoding
        tactile_tokens, summary_token = self.tactile_encoder(tactile_history)

        # 3. Contact gating
        gated_tactile, gate_value, gate_aux_loss = self.contact_gate(
            tactile_tokens, raw_tactile
        )

        # 4. Build attention mask
        prefix_mask = self.mask_builder(gate_value, mode="action")

        # 5. VLM backbone prefix encoding
        prefix_hidden = self.vlm_backbone(
            vision_tokens=vision_tokens,
            language_ids=language_ids,
            state=state,
            tactile_tokens=gated_tactile,
            prefix_attention_mask=None,  # simplified: use full mask in mask_builder
        )

        # 6. Extract state token from prefix (for action expert suffix)
        # State token is at position: N_vis + N_lang + 0 (first state token)
        n_vis = vision_tokens.shape[1]
        n_lang = language_ids.shape[1]
        state_idx = n_vis + n_lang
        state_hidden = prefix_hidden[:, state_idx:state_idx + 1, :]

        # 7. Action expert: compute CFM loss
        cfm_loss, metrics = self.action_expert.compute_cfm_loss(
            action=action_chunk,
            prefix_hidden=prefix_hidden,
            state_hidden=state_hidden,
        )

        # Add gate stats to metrics
        with torch.no_grad():
            metrics["gate_mean"] = gate_value.mean().item()
            metrics["gate_std"] = gate_value.std().item()
            metrics["gate_fraction_active"] = (gate_value > 0.5).float().mean().item()

        return cfm_loss, metrics, gate_aux_loss


class Phase2Trainer:
    """
    Phase 2: End-to-end CFM fine-tuning trainer.
    """

    def __init__(
        self,
        model: CATEPiModel,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        lr: float = 2.5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        gradient_clip: float = 1.0,
        gradient_accumulation: int = 4,
        gate_loss_weight: float = 0.1,
        log_every: int = 100,
        eval_every: int = 2000,
        save_every: int = 5000,
        save_dir: str = "checkpoints/phase2",
        device: str = "cuda",
        mixed_precision: bool = True,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.gradient_clip = gradient_clip
        self.gradient_accumulation = gradient_accumulation
        self.gate_loss_weight = gate_loss_weight
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.total_steps = total_steps
        self.mixed_precision = mixed_precision

        # Freeze vision encoder (per §8)
        for param in model.vision_encoder.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps
        )
        self.warmup_steps = warmup_steps

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=mixed_precision)

        # Tracking
        self.step_count = 0
        self.best_eval_loss = float("inf")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step."""
        self.model.train()

        images = batch["images"].to(self.device)
        language_ids = batch["language_ids"].to(self.device)
        state = batch["state"].to(self.device)
        tactile_history = batch["tactile_history"].to(self.device)
        raw_tactile = batch["raw_tactile"].to(self.device)
        action_chunk = batch["action_chunk"].to(self.device)

        with autocast(device_type=self.device.type, enabled=self.mixed_precision):
            cfm_loss, metrics, gate_aux_loss = self.model(
                images=images,
                language_ids=language_ids,
                state=state,
                tactile_history=tactile_history,
                raw_tactile=raw_tactile,
                action_chunk=action_chunk,
            )

            total_loss = cfm_loss
            if gate_aux_loss is not None:
                total_loss = total_loss + self.gate_loss_weight * gate_aux_loss
                metrics["gate_aux_loss"] = gate_aux_loss.item()

            # Scale for gradient accumulation
            total_loss = total_loss / self.gradient_accumulation

        self.scaler.scale(total_loss).backward()

        if (self.step_count + 1) % self.gradient_accumulation == 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.gradient_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if self.step_count >= self.warmup_steps:
                self.scheduler.step()

            metrics["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

        metrics["total_loss"] = total_loss.item() * self.gradient_accumulation
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]

        return metrics

    def save_checkpoint(self, name: str = "best"):
        """Save model checkpoint."""
        path = self.save_dir / f"phase2_{name}.pt"
        torch.save({
            "step": self.step_count,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_eval_loss": self.best_eval_loss,
        }, path)
        print(f"Saved checkpoint: {path}")

    def train(self):
        """Run the full Phase 2 training loop."""
        print("=" * 60)
        print("CATE-π Phase 2: End-to-end CFM Fine-tuning")
        print("=" * 60)

        data_iter = iter(self.train_loader)

        for step in range(self.total_steps):
            self.step_count = step

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            metrics = self.train_step(batch)

            if step % self.log_every == 0:
                # §12: Log per-channel loss to detect ignored force channels
                print(f"[Step {step:6d}/{self.total_steps}] "
                      f"loss={metrics.get('total_loss', 0):.4f} "
                      f"pos={metrics.get('position_loss', 0):.4f} "
                      f"ori={metrics.get('orientation_loss', 0):.4f} "
                      f"grip={metrics.get('gripper_loss', 0):.4f} "
                      f"f_n={metrics.get('force_normal_loss', 0):.4f} "
                      f"f_s={metrics.get('force_shear_loss', 0):.4f} "
                      f"gate_μ={metrics.get('gate_mean', 0):.3f}")

            if step % self.save_every == 0 and step > 0:
                self.save_checkpoint(f"step_{step}")

        self.save_checkpoint("final")
        print("\nPhase 2 complete.")


def main():
    parser = argparse.ArgumentParser(description="CATE-π Phase 2 Training")
    parser.add_argument("--backbone-config", default="cate_pi/configs/backbone_pi0_base.yaml")
    parser.add_argument("--tactile-config", default="cate_pi/configs/tactile_sensor.yaml")
    parser.add_argument("--training-config", default="cate_pi/configs/training_phase2.yaml")
    parser.add_argument("--data-dir", default="data/demonstrations")
    parser.add_argument("--phase1-ckpt", default="checkpoints/phase1/phase1_best.pt")
    parser.add_argument("--save-dir", default="checkpoints/phase2")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load configs
    with open(args.backbone_config) as f:
        backbone_cfg = yaml.safe_load(f)
    with open(args.tactile_config) as f:
        tactile_cfg = yaml.safe_load(f)

    # Create all modules
    vision_encoder = VisionEncoder.from_config(backbone_cfg)
    tactile_encoder = SpatioTemporalTactileEncoder.from_config(tactile_cfg)
    contact_gate = ContactGate.from_config(tactile_cfg)
    vlm_backbone = VLMBackbone.from_config(backbone_cfg)
    action_expert = TactileAwareActionExpert.from_config(backbone_cfg)
    mask_builder = CATEPiAttentionMaskBuilder()

    # Load Phase 1 checkpoint for tactile encoder + gate
    if Path(args.phase1_ckpt).exists():
        ckpt = torch.load(args.phase1_ckpt, map_location="cpu")
        tactile_encoder.load_state_dict(ckpt["tactile_encoder"])
        contact_gate.load_state_dict(ckpt["contact_gate"])
        print(f"Loaded Phase 1 checkpoint from {args.phase1_ckpt}")

    # Assemble model
    model = CATEPiModel(
        vision_encoder=vision_encoder,
        tactile_encoder=tactile_encoder,
        contact_gate=contact_gate,
        vlm_backbone=vlm_backbone,
        action_expert=action_expert,
        mask_builder=mask_builder,
    )

    # Create data loader
    train_loader = create_dataloader(args.data_dir, batch_size=16, split="train")

    # Create trainer
    trainer = Phase2Trainer(
        model=model,
        train_loader=train_loader,
        save_dir=args.save_dir,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
