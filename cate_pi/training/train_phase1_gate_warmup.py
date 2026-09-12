"""
CATE-π Training Phase 1: Tactile/Gate Warm-up (§8, Phase 1).

Freeze VLM backbone + action expert entirely.
Train ONLY the tactile encoder + contact detector/gate on an
auxiliary contact-classification task.

This is a practical scaffolding step (not in any source paper) that:
- Validates the tactile pipeline and simulator's signal quality
- Is cheap, fast to iterate on
- Catches tactile encoding bugs before they're buried in Phase 2

Trains: predict c_t from X_t, supervised by calibrated heuristic labels.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import time
from pathlib import Path
from typing import Dict, Optional

from cate_pi.models.tactile_encoder import SpatioTemporalTactileEncoder
from cate_pi.models.contact_gate import ContactGate
from cate_pi.data.dataset import CATEPiDataset, create_dataloader


class Phase1Trainer:
    """
    Phase 1: Contact-classification warm-up trainer.

    Trains the tactile encoder and contact gate in isolation
    to validate tactile signal quality before joint optimization.
    """

    def __init__(
        self,
        tactile_encoder: SpatioTemporalTactileEncoder,
        contact_gate: ContactGate,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        total_steps: int = 10000,
        gradient_clip: float = 1.0,
        log_every: int = 50,
        eval_every: int = 500,
        save_dir: str = "checkpoints/phase1",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tactile_encoder = tactile_encoder.to(self.device)
        self.contact_gate = contact_gate.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.gradient_clip = gradient_clip
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.total_steps = total_steps

        # Optimizer (only tactile encoder + gate parameters)
        params = list(tactile_encoder.parameters()) + list(contact_gate.parameters())
        self.optimizer = optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
        )
        self.warmup_steps = warmup_steps

        # Metrics
        self.best_f1 = 0.0
        self.step_count = 0

    def _warmup_lr(self):
        """Linear warmup for learning rate."""
        if self.step_count < self.warmup_steps:
            factor = self.step_count / max(1, self.warmup_steps)
            for pg in self.optimizer.param_groups:
                pg["lr"] = pg["lr"] * factor

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step."""
        self.tactile_encoder.train()
        self.contact_gate.train()

        tactile_history = batch["tactile_history"].to(self.device)
        raw_tactile = batch["raw_tactile"].to(self.device)
        contact_label = batch["contact_flag"].to(self.device)

        # Forward through tactile encoder
        tactile_tokens, summary_token = self.tactile_encoder(tactile_history)

        # Forward through contact gate
        gated_tokens, gate_value, aux_loss = self.contact_gate(
            tactile_tokens, raw_tactile
        )

        # Primary loss: binary cross-entropy for contact classification
        bce_loss = nn.functional.binary_cross_entropy(
            gate_value, contact_label
        )

        # Total loss
        total_loss = bce_loss
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(
            list(self.tactile_encoder.parameters()) + list(self.contact_gate.parameters()),
            self.gradient_clip,
        )

        self.optimizer.step()
        self._warmup_lr()
        if self.step_count >= self.warmup_steps:
            self.scheduler.step()

        # Metrics
        with torch.no_grad():
            predictions = (gate_value > 0.5).float()
            accuracy = (predictions == contact_label).float().mean().item()
            tp = ((predictions == 1) & (contact_label == 1)).float().sum().item()
            fp = ((predictions == 1) & (contact_label == 0)).float().sum().item()
            fn = ((predictions == 0) & (contact_label == 1)).float().sum().item()
            precision = tp / max(tp + fp, 1e-8)
            recall = tp / max(tp + fn, 1e-8)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            # Gate distribution stats (§12: detect always-on/always-off collapse)
            gate_mean = gate_value.mean().item()
            gate_std = gate_value.std().item()

        return {
            "loss": total_loss.item(),
            "bce_loss": bce_loss.item(),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "gate_mean": gate_mean,
            "gate_std": gate_std,
            "grad_norm": total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        if self.eval_loader is None:
            return {}

        self.tactile_encoder.eval()
        self.contact_gate.eval()

        total_correct = 0
        total_samples = 0
        total_tp, total_fp, total_fn = 0, 0, 0
        gate_values = []

        for batch in self.eval_loader:
            tactile_history = batch["tactile_history"].to(self.device)
            raw_tactile = batch["raw_tactile"].to(self.device)
            contact_label = batch["contact_flag"].to(self.device)

            tactile_tokens, _ = self.tactile_encoder(tactile_history)
            _, gate_value, _ = self.contact_gate(tactile_tokens, raw_tactile)

            predictions = (gate_value > 0.5).float()
            total_correct += (predictions == contact_label).float().sum().item()
            total_samples += len(contact_label)

            total_tp += ((predictions == 1) & (contact_label == 1)).float().sum().item()
            total_fp += ((predictions == 1) & (contact_label == 0)).float().sum().item()
            total_fn += ((predictions == 0) & (contact_label == 1)).float().sum().item()
            gate_values.extend(gate_value.cpu().tolist())

        precision = total_tp / max(total_tp + total_fp, 1e-8)
        recall = total_tp / max(total_tp + total_fn, 1e-8)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        import numpy as np
        gate_arr = np.array(gate_values)

        return {
            "eval_accuracy": total_correct / max(total_samples, 1),
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1,
            "eval_gate_mean": float(gate_arr.mean()),
            "eval_gate_std": float(gate_arr.std()),
        }

    def save_checkpoint(self, name: str = "best"):
        """Save model checkpoint."""
        path = self.save_dir / f"phase1_{name}.pt"
        torch.save({
            "step": self.step_count,
            "tactile_encoder": self.tactile_encoder.state_dict(),
            "contact_gate": self.contact_gate.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_f1": self.best_f1,
        }, path)
        print(f"Saved checkpoint: {path}")

    def train(self):
        """Run the full Phase 1 training loop."""
        print("=" * 60)
        print("CATE-π Phase 1: Tactile/Gate Warm-up")
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
                print(f"[Step {step:6d}/{self.total_steps}] "
                      f"loss={metrics['loss']:.4f} "
                      f"acc={metrics['accuracy']:.3f} "
                      f"F1={metrics['f1']:.3f} "
                      f"gate_μ={metrics['gate_mean']:.3f} "
                      f"gate_σ={metrics['gate_std']:.3f} "
                      f"lr={metrics['lr']:.6f}")

            if step % self.eval_every == 0 and step > 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    print(f"  [EVAL] F1={eval_metrics['eval_f1']:.3f} "
                          f"acc={eval_metrics['eval_accuracy']:.3f}")

                    if eval_metrics["eval_f1"] > self.best_f1:
                        self.best_f1 = eval_metrics["eval_f1"]
                        self.save_checkpoint("best")

        self.save_checkpoint("final")
        print(f"\nPhase 1 complete. Best F1: {self.best_f1:.3f}")


def main():
    parser = argparse.ArgumentParser(description="CATE-π Phase 1 Training")
    parser.add_argument("--config", type=str, default="cate_pi/configs/training_phase1.yaml")
    parser.add_argument("--tactile-config", type=str, default="cate_pi/configs/tactile_sensor.yaml")
    parser.add_argument("--data-dir", type=str, default="data/demonstrations")
    parser.add_argument("--save-dir", type=str, default="checkpoints/phase1")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load configs
    with open(args.tactile_config) as f:
        tactile_cfg = yaml.safe_load(f)

    # Create models
    tactile_encoder = SpatioTemporalTactileEncoder.from_config(tactile_cfg)
    contact_gate = ContactGate.from_config(tactile_cfg)

    # Create data loaders
    train_loader = create_dataloader(args.data_dir, batch_size=64, split="train")
    eval_loader = create_dataloader(args.data_dir, batch_size=64, split="eval")

    # Create trainer
    trainer = Phase1Trainer(
        tactile_encoder=tactile_encoder,
        contact_gate=contact_gate,
        train_loader=train_loader,
        eval_loader=eval_loader,
        save_dir=args.save_dir,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
