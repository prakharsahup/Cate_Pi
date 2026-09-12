"""
Force-Adverb Labeler for CATE-π (§9).

Auto-labels demonstration trajectories with force-descriptive language
using ground-truth force measurements from simulation.

This solves the "how do I get force-language labels without manual annotation"
problem by binning measured peak contact force into an adverb vocabulary
and synthesizing instruction strings from templates.

Reproduces Tactile-VLA's Table 2 experiment (learned-word generalization
to unseen adverbs and unseen tasks) without any manual force-language labeling.

Example:
    <1N  → "gently"
    1-3N → "normally"
    >3N  → "firmly" / "hard"
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ForceAdverbBin:
    """A single force range → adverb mapping."""
    min_force: float
    max_force: float
    adverbs: List[str]  # synonyms within this bin
    description: str    # for logging


# Default force bins (calibrate to your sim's force scale)
DEFAULT_FORCE_BINS = [
    ForceAdverbBin(
        min_force=0.0,
        max_force=1.0,
        adverbs=["gently", "softly", "lightly", "delicately"],
        description="light contact",
    ),
    ForceAdverbBin(
        min_force=1.0,
        max_force=3.0,
        adverbs=["normally", "moderately", "with moderate force"],
        description="normal contact",
    ),
    ForceAdverbBin(
        min_force=3.0,
        max_force=7.0,
        adverbs=["firmly", "hard", "with strong force", "forcefully"],
        description="strong contact",
    ),
    ForceAdverbBin(
        min_force=7.0,
        max_force=float("inf"),
        adverbs=["very hard", "with maximum force", "as hard as possible"],
        description="maximum force contact",
    ),
]


# Instruction templates with {adverb} placeholder
DEFAULT_TEMPLATES = [
    "Pick up the {object} {adverb}.",
    "Grasp the {object} {adverb} and place it on the target.",
    "{adverb_cap} pick up the {object}.",
    "Use a {adverb} grip to grasp the {object}.",
    "Wipe the surface {adverb}.",
    "{adverb_cap} wipe the {object}.",
    "Press the {object} {adverb}.",
    "{adverb_cap} push the {object} to the target location.",
]


class ForceAdverbLabeler:
    """
    Auto-labels trajectories with force-descriptive language instructions.

    Given a trajectory with ground-truth force measurements (from sim),
    produces a natural-language instruction that includes the appropriate
    force adverb, ready for VLM training data.
    """

    def __init__(
        self,
        force_bins: Optional[List[ForceAdverbBin]] = None,
        templates: Optional[List[str]] = None,
        # Holdout config for generalization testing
        holdout_adverbs: Optional[List[str]] = None,
        holdout_fraction: float = 0.2,
        seed: int = 42,
    ):
        """
        Args:
            force_bins: List of ForceAdverbBin defining force→adverb mapping.
            templates: Instruction template strings with {adverb} placeholder.
            holdout_adverbs: Adverbs to hold out from training for eval.
            holdout_fraction: Fraction of data to reserve for held-out adverbs.
            seed: Random seed for reproducibility.
        """
        self.force_bins = force_bins or DEFAULT_FORCE_BINS
        self.templates = templates or DEFAULT_TEMPLATES
        self.holdout_adverbs = set(holdout_adverbs or [])
        self.holdout_fraction = holdout_fraction
        self.rng = random.Random(seed)

        # Build quick lookup
        self._validate_bins()

    def _validate_bins(self):
        """Verify bins are contiguous and non-overlapping."""
        sorted_bins = sorted(self.force_bins, key=lambda b: b.min_force)
        for i in range(len(sorted_bins) - 1):
            assert sorted_bins[i].max_force == sorted_bins[i + 1].min_force, (
                f"Gap between bins: [{sorted_bins[i].max_force}, {sorted_bins[i+1].min_force}]"
            )

    def get_bin(self, peak_force: float) -> ForceAdverbBin:
        """Find the bin for a given peak force value."""
        for b in self.force_bins:
            if b.min_force <= peak_force < b.max_force:
                return b
        # Fallback: highest bin
        return self.force_bins[-1]

    def sample_adverb(
        self,
        peak_force: float,
        allow_holdout: bool = False,
    ) -> str:
        """
        Sample an adverb for a given force level.

        Args:
            peak_force: Peak contact force during the trajectory (Newtons).
            allow_holdout: If True, can sample from held-out adverbs.

        Returns:
            adverb: Sampled force-descriptive adverb string.
        """
        force_bin = self.get_bin(peak_force)
        available = [a for a in force_bin.adverbs
                     if allow_holdout or a not in self.holdout_adverbs]
        if not available:
            available = force_bin.adverbs  # fallback if all held out
        return self.rng.choice(available)

    def generate_instruction(
        self,
        peak_force: float,
        object_name: str = "object",
        task_name: Optional[str] = None,
        allow_holdout: bool = False,
    ) -> Dict[str, str]:
        """
        Generate a full instruction string for a trajectory.

        Args:
            peak_force: Peak contact force during the trajectory.
            object_name: Name of the manipulated object.
            task_name: Optional task identifier for template selection.
            allow_holdout: Whether to use held-out adverbs.

        Returns:
            Dict with:
                - "instruction": Full instruction string.
                - "adverb": The sampled adverb.
                - "force_bin": Description of the force bin.
                - "peak_force": The input peak force value.
                - "is_holdout": Whether the adverb is from the holdout set.
        """
        adverb = self.sample_adverb(peak_force, allow_holdout)
        template = self.rng.choice(self.templates)

        instruction = template.format(
            adverb=adverb,
            adverb_cap=adverb.capitalize(),
            object=object_name,
        )

        force_bin = self.get_bin(peak_force)

        return {
            "instruction": instruction,
            "adverb": adverb,
            "force_bin": force_bin.description,
            "peak_force": peak_force,
            "is_holdout": adverb in self.holdout_adverbs,
        }

    def label_trajectory(
        self,
        force_trajectory: np.ndarray,
        object_name: str = "object",
        allow_holdout: bool = False,
    ) -> Dict:
        """
        Label a full trajectory with force-descriptive language.

        Args:
            force_trajectory: (T,) or (T, 3) force measurements over time.
                            If 3D: [normal, shear_x, shear_y]. Uses normal ([:,0]).
            object_name: Name of the manipulated object.
            allow_holdout: Whether to use held-out adverbs.

        Returns:
            Label dict with instruction, adverb, force stats.
        """
        if force_trajectory.ndim == 2:
            normal_forces = force_trajectory[:, 0]
        else:
            normal_forces = force_trajectory

        peak_force = float(np.max(np.abs(normal_forces)))
        mean_force = float(np.mean(np.abs(normal_forces)))

        label = self.generate_instruction(
            peak_force=peak_force,
            object_name=object_name,
            allow_holdout=allow_holdout,
        )

        label["mean_force"] = mean_force
        label["force_trajectory_length"] = len(normal_forces)

        return label

    def label_dataset(
        self,
        trajectories: List[Dict],
        split_holdout: bool = True,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Label an entire dataset of trajectories.

        Args:
            trajectories: List of trajectory dicts, each with:
                - "forces": (T,) or (T,3) force array
                - "object": object name string
            split_holdout: Whether to split into train/eval by holdout adverbs.

        Returns:
            train_labeled: Trajectories with non-holdout adverb labels.
            eval_labeled: Trajectories with holdout adverb labels (for generalization eval).
        """
        train_labeled = []
        eval_labeled = []

        for traj in trajectories:
            forces = np.asarray(traj["forces"])
            obj_name = traj.get("object", "object")

            # Generate both train and eval labels
            train_label = self.label_trajectory(forces, obj_name, allow_holdout=False)
            eval_label = self.label_trajectory(forces, obj_name, allow_holdout=True)

            train_entry = {**traj, **train_label}
            eval_entry = {**traj, **eval_label}

            train_labeled.append(train_entry)
            if eval_label["is_holdout"]:
                eval_labeled.append(eval_entry)

        return train_labeled, eval_labeled

    def get_statistics(self) -> Dict:
        """Get labeler configuration statistics for logging."""
        return {
            "num_bins": len(self.force_bins),
            "bins": [
                {
                    "range": f"[{b.min_force}, {b.max_force})",
                    "adverbs": b.adverbs,
                    "description": b.description,
                }
                for b in self.force_bins
            ],
            "num_templates": len(self.templates),
            "holdout_adverbs": list(self.holdout_adverbs),
        }

# --- Isaac Lab integration ---

FORCE_BINS = {
    "gently":   (0.0,  1.0),   # Newtons — calibrate against your sensor output
    "normally": (1.0,  3.0),
    "firmly":   (3.0,  float("inf")),
}

INSTRUCTION_TEMPLATES = {
    "gently":   "Pick up the {object} gently.",
    "normally": "Pick up the {object}.",
    "firmly":   "Grasp the {object} firmly.",
}

def label_episode_force(force_mag_log: list[float], object_name: str = "object") -> str:
    """
    Bins peak contact force during an episode into an adverb and
    synthesises the instruction string.
    Uses ground-truth Isaac Lab force data; no manual annotation needed.
    """
    peak = max(force_mag_log)
    for adverb, (lo, hi) in FORCE_BINS.items():
        if lo <= peak < hi:
            return INSTRUCTION_TEMPLATES[adverb].format(object=object_name)
    return INSTRUCTION_TEMPLATES["firmly"].format(object=object_name)
