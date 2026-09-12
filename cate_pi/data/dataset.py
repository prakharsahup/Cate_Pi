"""
Dataset for CATE-π.

Loads demonstration data with multi-modal observations:
- RGB images (multi-camera)
- Language instructions
- Proprioceptive state
- Tactile sensor readings
- Augmented action vectors (position + force)

Handles:
- Action chunk construction (H-step windows)
- Temporal tactile history windows (K steps)
- Per-channel action normalization
- Contact flag labeling for gate supervision
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from cate_pi.data.action_normalization import ActionNormalizer


class CATEPiDataset(Dataset):
    """
    Multi-modal demonstration dataset for CATE-π training.

    Each sample contains:
    - images: (num_cameras, 3, H, W) RGB images
    - language_ids: (L,) tokenized instruction
    - state: (state_dim,) proprioceptive state
    - tactile_history: (K, rows, cols) recent tactile readings
    - raw_tactile: (rows, cols) current tactile reading (for contact detection)
    - action_chunk: (H, padded_dim) target action chunk
    - contact_flag: (1,) binary contact label
    """

    def __init__(
        self,
        data_dir: str,
        chunk_horizon: int = 50,
        tactile_window: int = 4,
        image_size: int = 224,
        num_cameras: int = 2,
        action_dim: int = 13,
        padded_action_dim: int = 32,
        sensor_rows: int = 15,
        sensor_cols: int = 8,
        max_language_tokens: int = 128,
        normalizer: Optional[ActionNormalizer] = None,
        split: str = "train",
    ):
        """
        Args:
            data_dir: Root directory containing demonstration data.
            chunk_horizon: H — action chunk length.
            tactile_window: K — tactile temporal history length.
            image_size: Target image resolution (square).
            num_cameras: Number of camera views.
            action_dim: Active action dimensions.
            padded_action_dim: Padded action vector size.
            sensor_rows: Tactile sensor rows.
            sensor_cols: Tactile sensor cols.
            max_language_tokens: Max language sequence length.
            normalizer: Pre-fitted ActionNormalizer for actions.
            split: "train" or "eval".
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.chunk_horizon = chunk_horizon
        self.tactile_window = tactile_window
        self.image_size = image_size
        self.num_cameras = num_cameras
        self.action_dim = action_dim
        self.padded_action_dim = padded_action_dim
        self.sensor_rows = sensor_rows
        self.sensor_cols = sensor_cols
        self.max_language_tokens = max_language_tokens
        self.normalizer = normalizer
        self.split = split

        # Load trajectory index
        self.trajectories = self._load_trajectory_index()
        # Build sample index: (trajectory_idx, timestep)
        self.samples = self._build_sample_index()

    def _load_trajectory_index(self) -> List[Dict]:
        """
        Load the list of available trajectories.

        Expected directory structure:
        data_dir/
          trajectories/
            traj_0000/
              images/    (front_0000.png, wrist_0000.png, ...)
              actions.npy
              states.npy
              tactile.npy
              forces.npy
              instruction.txt
              metadata.json
        """
        traj_dir = self.data_dir / "trajectories"
        trajectories = []

        if traj_dir.exists():
            for traj_path in sorted(traj_dir.iterdir()):
                if traj_path.is_dir():
                    meta = {
                        "path": traj_path,
                        "length": self._get_trajectory_length(traj_path),
                    }
                    trajectories.append(meta)

        if not trajectories:
            # Create synthetic placeholder data for development
            print(f"Warning: No trajectories found in {traj_dir}. "
                  f"Using synthetic data for development.")
            trajectories = self._create_synthetic_trajectories()

        print(f"Loaded {len(trajectories)} trajectories for split={self.split}")
        return trajectories

    def _get_trajectory_length(self, traj_path: Path) -> int:
        """Get the number of timesteps in a trajectory."""
        actions_path = traj_path / "actions.npy"
        if actions_path.exists():
            return len(np.load(actions_path))
        return 200  # default

    def _create_synthetic_trajectories(self, num_trajs: int = 100) -> List[Dict]:
        """Create synthetic trajectory metadata for development."""
        return [
            {"path": None, "length": 200, "synthetic": True}
            for _ in range(num_trajs)
        ]

    def _build_sample_index(self) -> List[Tuple[int, int]]:
        """
        Build a flat index of (trajectory_idx, timestep) pairs.
        Ensures each sample has enough history (K steps) and
        future (H steps) for the tactile window and action chunk.
        """
        samples = []
        for traj_idx, traj in enumerate(self.trajectories):
            length = traj["length"]
            # Valid timesteps: need K past steps and H future steps
            start = self.tactile_window
            end = length - self.chunk_horizon
            for t in range(start, max(start, end)):
                samples.append((traj_idx, t))
        return samples

    def _load_real_sample(self, traj: Dict, t: int) -> Dict:
        """Load a real sample from disk."""
        path = traj["path"]

        # Load images (current timestep, all cameras)
        images = np.zeros((self.num_cameras, 3, self.image_size, self.image_size), dtype=np.float32)
        # In practice: load from path/images/front_{t:04d}.png, etc.

        # Load actions (chunk from t to t+H)
        actions = np.load(path / "actions.npy")[t:t + self.chunk_horizon]

        # Load states
        state = np.load(path / "states.npy")[t]

        # Load tactile history (t-K+1 to t)
        tactile = np.load(path / "tactile.npy")[t - self.tactile_window + 1:t + 1]

        # Load instruction
        with open(path / "instruction.txt") as f:
            instruction = f.read().strip()

        return {
            "images": images,
            "actions": actions,
            "state": state,
            "tactile_history": tactile,
            "instruction": instruction,
        }

    def _generate_synthetic_sample(self, t: int) -> Dict:
        """Generate a synthetic sample for development/testing."""
        rng = np.random.RandomState(t)

        images = rng.randn(
            self.num_cameras, 3, self.image_size, self.image_size
        ).astype(np.float32) * 0.1

        # Synthetic actions with both position and force channels
        actions = np.zeros((self.chunk_horizon, self.padded_action_dim), dtype=np.float32)
        actions[:, :3] = rng.randn(self.chunk_horizon, 3) * 0.01  # position deltas
        actions[:, 3:9] = rng.randn(self.chunk_horizon, 6) * 0.1   # orientation
        actions[:, 9] = rng.rand(self.chunk_horizon) * 0.08         # gripper
        actions[:, 10] = rng.rand(self.chunk_horizon) * 5.0         # ext normal force
        actions[:, 11:13] = rng.randn(self.chunk_horizon, 2) * 0.5  # ext shear
        actions[:, 13] = rng.rand(self.chunk_horizon) * 3.0         # grasp force

        state = rng.randn(32).astype(np.float32) * 0.1

        tactile = rng.rand(
            self.tactile_window, self.sensor_rows, self.sensor_cols
        ).astype(np.float32) * 0.2

        return {
            "images": images,
            "actions": actions,
            "state": state,
            "tactile_history": tactile,
            "instruction": "Pick up the cube gently.",
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_idx, t = self.samples[idx]
        traj = self.trajectories[traj_idx]

        # Load data
        if traj.get("synthetic", False):
            raw = self._generate_synthetic_sample(t + traj_idx * 1000)
        else:
            raw = self._load_real_sample(traj, t)

        # Normalize actions
        actions = raw["actions"]
        if actions.shape[0] < self.chunk_horizon:
            # Pad short chunks with last action
            pad = np.repeat(actions[-1:], self.chunk_horizon - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        if self.normalizer is not None:
            actions = self.normalizer.normalize(actions)

        # Current tactile for contact detection
        current_tactile = raw["tactile_history"][-1]  # (rows, cols)

        # Simple contact flag from raw tactile (for supervised gate training)
        contact_flag = float(
            (current_tactile > 0.05).sum() > 5
        )

        # Placeholder language token IDs (would use PaliGemma tokenizer)
        language_ids = np.zeros(self.max_language_tokens, dtype=np.int64)
        # Simple hash-based placeholder encoding
        for i, char in enumerate(raw["instruction"][:self.max_language_tokens]):
            language_ids[i] = ord(char) % 256

        return {
            "images": torch.from_numpy(raw["images"]),
            "language_ids": torch.from_numpy(language_ids),
            "state": torch.from_numpy(raw["state"]),
            "tactile_history": torch.from_numpy(raw["tactile_history"]),
            "raw_tactile": torch.from_numpy(current_tactile),
            "action_chunk": torch.from_numpy(actions).float(),
            "contact_flag": torch.tensor(contact_flag, dtype=torch.float32),
        }


def create_dataloader(
    data_dir: str,
    batch_size: int = 16,
    split: str = "train",
    normalizer: Optional[ActionNormalizer] = None,
    num_workers: int = 4,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader for CATE-π training."""
    dataset = CATEPiDataset(
        data_dir=data_dir,
        normalizer=normalizer,
        split=split,
        **kwargs,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
