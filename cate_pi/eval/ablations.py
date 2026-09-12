"""
Ablation Harness for CATE-π (§11).

Runs the ablation table from the implementation plan, isolating
the contribution of each component:

| Variant                                     | What it isolates                          |
|---------------------------------------------|-------------------------------------------|
| Full CATE-π                                 | —                                         |
| − contact gating (tactile always on)         | value of §4.4                             |
| hard gate vs. soft learnable gate            | value of soft gate extension              |
| − CoT head                                  | value of §4.7's reasoning loop            |
| fixed-interval CoT vs. event-triggered CoT  | value of event-trigger extension          |
| − force channels (position-only)             | value of the tactile-aware action expert  |
| gated CoT buffer vs. raw always-on buffer    | value of the raw-buffer design decision   |

This is also the strongest evidence for a paper — mirrors TacVLA's
own ablation structure.
"""

import torch
import copy
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path

from cate_pi.eval.eval_suite import CATEPiEvaluator, TaskMetrics


@dataclass
class AblationConfig:
    """Configuration for a single ablation variant."""
    name: str
    description: str
    modifications: Dict  # what to change from the full model


# Define all ablation variants
ABLATION_VARIANTS = [
    AblationConfig(
        name="full_cate_pi",
        description="Full CATE-π (baseline)",
        modifications={},
    ),
    AblationConfig(
        name="no_contact_gating",
        description="Tactile always on (no contact gating)",
        modifications={"gate_mode": "always_on"},
    ),
    AblationConfig(
        name="hard_gate_only",
        description="Hard threshold gate (no learnable gate)",
        modifications={"gate_mode": "hard"},
    ),
    AblationConfig(
        name="soft_gate",
        description="Soft learnable gate (CATE-π extension)",
        modifications={"gate_mode": "soft"},
    ),
    AblationConfig(
        name="no_cot",
        description="No Chain-of-Thought reasoning head",
        modifications={"cot_enabled": False},
    ),
    AblationConfig(
        name="fixed_interval_cot",
        description="Fixed-interval CoT trigger only (no event trigger)",
        modifications={"cot_event_trigger": False},
    ),
    AblationConfig(
        name="event_triggered_cot",
        description="Event-triggered CoT (CATE-π extension)",
        modifications={"cot_event_trigger": True},
    ),
    AblationConfig(
        name="no_force_channels",
        description="Position-only action space (no force channels)",
        modifications={"force_channels": False},
    ),
    AblationConfig(
        name="gated_cot_buffer",
        description="CoT buffer also gated (loses non-contact history)",
        modifications={"cot_buffer_gated": True},
    ),
    AblationConfig(
        name="raw_cot_buffer",
        description="Raw always-on CoT buffer (CATE-π design)",
        modifications={"cot_buffer_gated": False},
    ),
]


class AblationRunner:
    """
    Runs the full ablation table and produces comparison results.
    """

    def __init__(
        self,
        base_model,
        controller,
        env,
        normalizer=None,
        tasks: Optional[List[Dict]] = None,
        num_episodes: int = 20,
        device: str = "cuda",
        save_dir: str = "results/ablations",
    ):
        """
        Args:
            base_model: Fully trained CATE-π model (the "full" variant).
            controller: HybridPositionForceController.
            env: Simulation environment.
            normalizer: ActionNormalizer.
            tasks: Task configurations for evaluation.
            num_episodes: Episodes per task per variant.
            device: Compute device.
            save_dir: Directory to save results.
        """
        self.base_model = base_model
        self.controller = controller
        self.env = env
        self.normalizer = normalizer
        self.tasks = tasks
        self.num_episodes = num_episodes
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def create_variant(self, config: AblationConfig):
        """
        Create a model variant by modifying the base model.

        Returns a modified copy of the model with the ablation applied.
        """
        model = copy.deepcopy(self.base_model)
        mods = config.modifications

        if mods.get("gate_mode") == "always_on":
            # Disable gating: set gate to always return 1.0
            if hasattr(model, "contact_gate"):
                model.contact_gate.mode = "hard"
                model.contact_gate.hard_gate.pressure_threshold = -1.0  # always active
                model.contact_gate.hard_gate.count_threshold = -1

        elif mods.get("gate_mode") == "hard":
            if hasattr(model, "contact_gate"):
                model.contact_gate.mode = "hard"

        elif mods.get("gate_mode") == "soft":
            if hasattr(model, "contact_gate"):
                model.contact_gate.mode = "soft"

        if mods.get("cot_enabled") is False:
            # Disable CoT: never trigger reasoning
            if hasattr(model, "cot_head"):
                model.cot_head = None

        if "cot_event_trigger" in mods:
            if hasattr(model, "cot_head") and model.cot_head is not None:
                model.cot_head.trigger_policy.event_trigger_enabled = mods["cot_event_trigger"]

        if mods.get("force_channels") is False:
            # Zero out force dimensions in action expert
            if hasattr(model, "action_expert"):
                model.action_expert.action_mask[10:] = 0.0

        return model

    def run_variant(
        self,
        config: AblationConfig,
    ) -> Dict[str, TaskMetrics]:
        """
        Run evaluation for a single ablation variant.

        Args:
            config: Ablation configuration.

        Returns:
            Dict mapping task names to TaskMetrics.
        """
        print(f"\n{'='*60}")
        print(f"Ablation: {config.name}")
        print(f"  {config.description}")
        print(f"{'='*60}")

        model = self.create_variant(config)

        evaluator = CATEPiEvaluator(
            model=model,
            controller=self.controller,
            env=self.env,
            normalizer=self.normalizer,
            num_episodes_per_task=self.num_episodes,
            device=self.device,
        )

        return evaluator.run_full_evaluation(tasks=self.tasks)

    def run_all_ablations(
        self,
        variants: Optional[List[AblationConfig]] = None,
    ) -> Dict[str, Dict[str, TaskMetrics]]:
        """
        Run the complete ablation table.

        Args:
            variants: List of ablation configs to run. Default: all.

        Returns:
            Nested dict: {variant_name: {task_name: TaskMetrics}}
        """
        if variants is None:
            variants = ABLATION_VARIANTS

        all_results = {}

        for config in variants:
            results = self.run_variant(config)
            all_results[config.name] = results

        # Print comparison table
        self._print_comparison_table(all_results)

        # Save results
        self._save_results(all_results)

        return all_results

    def _print_comparison_table(
        self,
        all_results: Dict[str, Dict[str, TaskMetrics]],
    ):
        """Print a formatted comparison table across all variants and tasks."""
        print("\n" + "=" * 90)
        print("ABLATION COMPARISON TABLE")
        print("=" * 90)

        # Get all tasks
        all_tasks = set()
        for results in all_results.values():
            all_tasks.update(results.keys())
        all_tasks = sorted(all_tasks)

        # Header
        header = f"{'Variant':<35}"
        for task in all_tasks:
            header += f" {task[:15]:>15}"
        header += f" {'Mean':>10}"
        print(header)
        print("-" * len(header))

        # Rows
        for variant_name, results in all_results.items():
            row = f"{variant_name:<35}"
            rates = []
            for task in all_tasks:
                if task in results:
                    sr = results[task].success_rate
                    rates.append(sr)
                    row += f" {sr:>14.1%}"
                else:
                    row += f" {'N/A':>15}"
            if rates:
                row += f" {sum(rates)/len(rates):>9.1%}"
            print(row)

        print("=" * 90)

    def _save_results(self, all_results: Dict):
        """Save ablation results to file."""
        import json

        results_dict = {}
        for variant, tasks in all_results.items():
            results_dict[variant] = {}
            for task_name, metrics in tasks.items():
                results_dict[variant][task_name] = {
                    "success_rate": metrics.success_rate,
                    "force_accuracy": metrics.force_accuracy,
                    "mean_latency_ms": metrics.mean_latency_ms,
                    "gate_activation_rate": metrics.gate_activation_rate,
                    "num_episodes": metrics.num_episodes,
                }

        output_path = self.save_dir / "ablation_results.json"
        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {output_path}")
