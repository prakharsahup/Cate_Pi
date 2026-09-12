"""
CATE-π Training Phase 3: CoT Fine-tuning (§8, Phase 3).

Freeze the action expert (protect control skill from Phase 2).
LoRA-adapt the language decoder on a small failure/diagnosis dataset.
Mix in general instruction-following data to guard against
catastrophic forgetting of language ability.

This phase adds Tactile-VLA-CoT's reasoning capability:
the VLM can diagnose tactile failures and emit corrective instructions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np

from cate_pi.models.vlm_backbone import VLMBackbone
from cate_pi.models.cot_head import ChainOfThoughtHead


class CoTDataset(Dataset):
    """
    Dataset for CoT fine-tuning.

    Combines failure-diagnosis data with general instruction-following data.
    Each sample contains:
    - prefix_context: pre-encoded prefix (or raw inputs to encode)
    - tactile_buffer: raw tactile history for the buffer
    - target_ids: ground-truth CoT token sequence
    - sample_type: "failure_diagnosis" or "general_instruction"
    """

    def __init__(
        self,
        failure_dir: str,
        general_dir: str,
        failure_ratio: float = 0.7,
        max_seq_len: int = 128,
        split: str = "train",
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.failure_ratio = failure_ratio

        # Load failure-diagnosis pairs
        self.failure_samples = self._load_samples(failure_dir, "failure")
        self.general_samples = self._load_samples(general_dir, "general")

        # Build interleaved index
        self.samples = self._build_mixed_index()

        print(f"CoT dataset: {len(self.failure_samples)} failure, "
              f"{len(self.general_samples)} general, "
              f"{len(self.samples)} total mixed samples")

    def _load_samples(self, data_dir: str, sample_type: str) -> List[Dict]:
        """Load samples from a directory."""
        samples = []
        data_path = Path(data_dir)

        if data_path.exists():
            for f in sorted(data_path.glob("*.json")):
                import json
                with open(f) as fp:
                    sample = json.load(fp)
                    sample["type"] = sample_type
                    samples.append(sample)

        if not samples:
            # Synthetic data for development
            rng = np.random.RandomState(42)
            for i in range(50 if sample_type == "failure" else 20):
                if sample_type == "failure":
                    text = (
                        "Task not completed. The normal force was too low during "
                        "the grasp phase. Increase grasp force and retry."
                    )
                else:
                    text = "Pick up the red cube and place it on the green target."

                samples.append({
                    "type": sample_type,
                    "text": text,
                    "tactile_buffer": rng.rand(50, 3).tolist(),
                })

        return samples

    def _build_mixed_index(self) -> List[Dict]:
        """Build a mixed sample list with the specified ratio."""
        total_failure = int(len(self.failure_samples) / self.failure_ratio)
        total_general = total_failure - len(self.failure_samples)
        total_general = min(total_general, len(self.general_samples))

        mixed = self.failure_samples.copy()

        # Repeat general samples to match ratio
        rng = np.random.RandomState(42)
        if self.general_samples:
            general_indices = rng.choice(
                len(self.general_samples),
                size=max(total_general, 1),
                replace=True,
            )
            for idx in general_indices:
                mixed.append(self.general_samples[idx])

        rng.shuffle(mixed)
        return mixed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample.get("text", "")

        # Simple tokenization (placeholder — real impl uses PaliGemma tokenizer)
        token_ids = torch.zeros(self.max_seq_len, dtype=torch.long)
        for i, char in enumerate(text[:self.max_seq_len]):
            token_ids[i] = ord(char) % 256

        # Tactile buffer
        buffer_data = sample.get("tactile_buffer", np.zeros((50, 3)).tolist())
        buffer_tensor = torch.tensor(buffer_data, dtype=torch.float32)
        if buffer_tensor.shape[0] < 50:
            pad = torch.zeros(50 - buffer_tensor.shape[0], 3)
            buffer_tensor = torch.cat([pad, buffer_tensor])

        return {
            "target_ids": token_ids,
            "tactile_buffer": buffer_tensor,
            "sample_type": 1.0 if sample["type"] == "failure" else 0.0,
        }


class Phase3Trainer:
    """
    Phase 3: CoT fine-tuning trainer.

    Trains the CoT head's language generation capability while
    protecting the action expert (frozen) and mixing in general
    instruction data to prevent catastrophic forgetting.
    """

    def __init__(
        self,
        vlm_backbone: VLMBackbone,
        cot_head: ChainOfThoughtHead,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        total_steps: int = 5000,
        gradient_clip: float = 1.0,
        log_every: int = 50,
        eval_every: int = 500,
        save_dir: str = "checkpoints/phase3",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.vlm_backbone = vlm_backbone.to(self.device)
        self.cot_head = cot_head.to(self.device)
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.gradient_clip = gradient_clip
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

        # Only train CoT head parameters + LoRA on backbone
        trainable_params = list(cot_head.parameters())
        # In full impl: add LoRA parameters from backbone

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps
        )

        self.step_count = 0
        self.best_loss = float("inf")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one CoT training step."""
        self.vlm_backbone.train()
        self.cot_head.train()

        target_ids = batch["target_ids"].to(self.device)
        tactile_buffer = batch["tactile_buffer"].to(self.device)
        sample_type = batch["sample_type"].to(self.device)

        B = target_ids.shape[0]

        # Initialize the CoT buffer with the provided data
        self.cot_head.raw_buffer.reset(B, self.device)
        self.cot_head.raw_buffer._buffer = tactile_buffer

        # Create a dummy prefix (in full impl: encode actual observations)
        prefix_hidden = torch.randn(
            B, 100, self.vlm_backbone.hidden_dim, device=self.device
        ) * 0.01

        # Compute CoT loss
        loss = self.cot_head.compute_cot_loss(
            prefix_hidden=prefix_hidden,
            target_ids=target_ids,
            backbone_layers=self.vlm_backbone.layers,
            backbone_norm=self.vlm_backbone.final_norm,
            token_embedding=self.vlm_backbone.token_embedding,
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.cot_head.parameters(),
            self.gradient_clip,
        )

        self.optimizer.step()
        if self.step_count >= self.warmup_steps:
            self.scheduler.step()

        # Compute per-type loss for monitoring (§12: detect reasoning degradation)
        with torch.no_grad():
            failure_mask = sample_type > 0.5
            if failure_mask.any():
                failure_loss = loss.item()  # simplified; real: compute per-sample
            else:
                failure_loss = 0.0

        return {
            "cot_loss": loss.item(),
            "failure_loss": failure_loss,
            "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def save_checkpoint(self, name: str = "best"):
        """Save checkpoint."""
        path = self.save_dir / f"phase3_{name}.pt"
        torch.save({
            "step": self.step_count,
            "cot_head": self.cot_head.state_dict(),
            "vlm_backbone": self.vlm_backbone.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }, path)
        print(f"Saved checkpoint: {path}")

    def train(self):
        """Run the full Phase 3 training loop."""
        print("=" * 60)
        print("CATE-π Phase 3: CoT Fine-tuning")
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
                      f"cot_loss={metrics['cot_loss']:.4f} "
                      f"lr={metrics['lr']:.6f}")

            if step % self.eval_every == 0 and step > 0:
                if metrics["cot_loss"] < self.best_loss:
                    self.best_loss = metrics["cot_loss"]
                    self.save_checkpoint("best")

        self.save_checkpoint("final")
        print(f"\nPhase 3 complete. Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="CATE-π Phase 3 Training")
    parser.add_argument("--backbone-config", default="cate_pi/configs/backbone_pi0_base.yaml")
    parser.add_argument("--failure-dir", default="data/cot_failures")
    parser.add_argument("--general-dir", default="data/general_instructions")
    parser.add_argument("--phase2-ckpt", default="checkpoints/phase2/phase2_best.pt")
    parser.add_argument("--save-dir", default="checkpoints/phase3")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.backbone_config) as f:
        backbone_cfg = yaml.safe_load(f)

    vlm_backbone = VLMBackbone.from_config(backbone_cfg)
    cot_head = ChainOfThoughtHead.from_config({**backbone_cfg, "cot_buffer": {}})

    # Load Phase 2 checkpoint
    if Path(args.phase2_ckpt).exists():
        ckpt = torch.load(args.phase2_ckpt, map_location="cpu")
        vlm_backbone.load_state_dict(
            {k.replace("vlm_backbone.", ""): v
             for k, v in ckpt["model"].items() if "vlm_backbone" in k},
            strict=False,
        )
        print(f"Loaded Phase 2 backbone from {args.phase2_ckpt}")

    # Data
    cot_dataset = CoTDataset(args.failure_dir, args.general_dir)
    train_loader = DataLoader(cot_dataset, batch_size=8, shuffle=True)

    # Train
    trainer = Phase3Trainer(
        vlm_backbone=vlm_backbone,
        cot_head=cot_head,
        train_loader=train_loader,
        save_dir=args.save_dir,
        device=args.device,
    )

    trainer.train()


if __name__ == "__main__":
    main()
