"""
Per-channel Action Normalization for CATE-π (§4.6).

CRITICAL: Position (meters) and force (Newtons) live on wildly different
numeric scales. Without per-channel normalization, position channels
dominate the gradient and the model effectively ignores force.

Uses quantile normalization to map each action dimension independently
to [-1, 1], following π0-FAST's normalization approach.

The normalization statistics (per-dimension quantiles) must be computed
from the demonstration dataset and stored alongside the checkpoint.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict, Union


class ActionNormalizer:
    """
    Per-channel quantile normalization for the augmented action space.

    Maps each action dimension independently to [-1, 1] using quantile
    statistics computed from the training data.
    """

    def __init__(
        self,
        action_dim: int = 13,
        padded_dim: int = 32,
        target_range: tuple = (-1.0, 1.0),
        quantile_low: float = 0.01,
        quantile_high: float = 0.99,
    ):
        """
        Args:
            action_dim: Number of active action dimensions.
            padded_dim: Padded action dimension (zero-mask unused).
            target_range: (min, max) of normalized range.
            quantile_low: Lower quantile for clipping.
            quantile_high: Upper quantile for clipping.
        """
        self.action_dim = action_dim
        self.padded_dim = padded_dim
        self.target_min, self.target_max = target_range
        self.quantile_low = quantile_low
        self.quantile_high = quantile_high

        # Per-channel statistics (computed from data)
        self.q_low: Optional[np.ndarray] = None   # (action_dim,) lower quantiles
        self.q_high: Optional[np.ndarray] = None   # (action_dim,) upper quantiles
        self.mean: Optional[np.ndarray] = None      # (action_dim,) mean
        self.std: Optional[np.ndarray] = None        # (action_dim,) std

        # Channel names for logging
        self.channel_names = self._default_channel_names()

    @staticmethod
    def _default_channel_names() -> Dict[int, str]:
        """Default channel name mapping for parallel-jaw gripper."""
        return {
            0: "delta_x", 1: "delta_y", 2: "delta_z",
            3: "rot_0", 4: "rot_1", 5: "rot_2",
            6: "rot_3", 7: "rot_4", 8: "rot_5",
            9: "gripper_width",
            10: "force_ext_normal",
            11: "force_ext_shear_x", 12: "force_ext_shear_y",
            13: "force_grasp",
        }

    def fit(self, actions: np.ndarray):
        """
        Compute normalization statistics from training data.

        Args:
            actions: (N, action_dim) or (N, H, action_dim) action data.
                    N = number of samples, H = chunk horizon (optional).
        """
        if actions.ndim == 3:
            # Flatten chunk horizon: (N, H, D) -> (N*H, D)
            actions = actions.reshape(-1, actions.shape[-1])

        # Only use active dimensions
        active = actions[:, :self.action_dim]

        self.q_low = np.quantile(active, self.quantile_low, axis=0)
        self.q_high = np.quantile(active, self.quantile_high, axis=0)
        self.mean = np.mean(active, axis=0)
        self.std = np.std(active, axis=0) + 1e-8

        # Log statistics for debugging (§12: verify per-channel stats early)
        print("=" * 60)
        print("Action Normalization Statistics:")
        print("=" * 60)
        for i in range(self.action_dim):
            name = self.channel_names.get(i, f"dim_{i}")
            print(f"  [{i}] {name:20s}: "
                  f"q_low={self.q_low[i]:8.4f}, "
                  f"q_high={self.q_high[i]:8.4f}, "
                  f"mean={self.mean[i]:8.4f}, "
                  f"std={self.std[i]:8.4f}, "
                  f"range={self.q_high[i] - self.q_low[i]:8.4f}")
        print("=" * 60)

    def normalize(self, actions: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize actions to [-1, 1] per channel.

        Args:
            actions: (..., padded_dim) action vectors.

        Returns:
            normalized: (..., padded_dim) normalized actions.
        """
        assert self.q_low is not None, "Must call fit() before normalize()"

        is_torch = isinstance(actions, torch.Tensor)
        if is_torch:
            device = actions.device
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions.copy()

        # Normalize active dimensions
        active = actions_np[..., :self.action_dim]

        # Quantile-based normalization: clip to [q_low, q_high], then scale to [-1, 1]
        q_range = self.q_high - self.q_low
        q_range = np.maximum(q_range, 1e-8)  # prevent division by zero

        clipped = np.clip(active, self.q_low, self.q_high)
        normalized = (clipped - self.q_low) / q_range  # [0, 1]
        normalized = normalized * (self.target_max - self.target_min) + self.target_min  # [-1, 1]

        result = actions_np.copy()
        result[..., :self.action_dim] = normalized

        if is_torch:
            return torch.from_numpy(result).to(device)
        return result

    def denormalize(self, actions: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Inverse normalization: [-1, 1] back to raw action units.

        Args:
            actions: (..., padded_dim) normalized actions.

        Returns:
            raw: (..., padded_dim) denormalized actions in original units.
        """
        assert self.q_low is not None, "Must call fit() before denormalize()"

        is_torch = isinstance(actions, torch.Tensor)
        if is_torch:
            device = actions.device
            actions_np = actions.detach().cpu().numpy()
        else:
            actions_np = actions.copy()

        active = actions_np[..., :self.action_dim]

        # Inverse: [-1, 1] -> [0, 1] -> [q_low, q_high]
        q_range = self.q_high - self.q_low
        unscaled = (active - self.target_min) / (self.target_max - self.target_min)  # [0, 1]
        raw = unscaled * q_range + self.q_low

        result = actions_np.copy()
        result[..., :self.action_dim] = raw

        if is_torch:
            return torch.from_numpy(result).to(device)
        return result

    def save(self, path: Union[str, Path]):
        """Save normalization statistics to JSON."""
        path = Path(path)
        stats = {
            "action_dim": self.action_dim,
            "padded_dim": self.padded_dim,
            "target_range": [self.target_min, self.target_max],
            "quantile_low": self.quantile_low,
            "quantile_high": self.quantile_high,
            "q_low": self.q_low.tolist(),
            "q_high": self.q_high.tolist(),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved normalization stats to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ActionNormalizer":
        """Load normalization statistics from JSON."""
        with open(path) as f:
            stats = json.load(f)

        normalizer = cls(
            action_dim=stats["action_dim"],
            padded_dim=stats["padded_dim"],
            target_range=tuple(stats["target_range"]),
            quantile_low=stats["quantile_low"],
            quantile_high=stats["quantile_high"],
        )
        normalizer.q_low = np.array(stats["q_low"])
        normalizer.q_high = np.array(stats["q_high"])
        normalizer.mean = np.array(stats["mean"])
        normalizer.std = np.array(stats["std"])

        return normalizer
