# CATE-π — Isaac Lab Implementation Plan

**Architecture:** Contact-Aware Tactile-Expert π-VLA  
**Simulator:** NVIDIA Isaac Lab (Isaac Sim backend)  
**Robot:** Open-source arm + parallel gripper in USD/URDF format  
**Goal:** Generate synthetic tactile demonstration data in Isaac Lab and train the full CATE-π pipeline

---

## Table of Contents

1. [Architecture Recap](#1-architecture-recap)
2. [Prerequisites and Environment Setup](#2-prerequisites-and-environment-setup)
3. [Robot Asset Selection and Setup](#3-robot-asset-selection-and-setup)
4. [Tactile Sensor Integration](#4-tactile-sensor-integration)
5. [Isaac Lab Environment Implementation](#5-isaac-lab-environment-implementation)
6. [Demonstration Data Collection](#6-demonstration-data-collection)
7. [Data Pipeline: HDF5 → LeRobot Format](#7-data-pipeline-hdf5--lerobot-format)
8. [Model Architecture Changes](#8-model-architecture-changes)
9. [Training Curriculum](#9-training-curriculum)
10. [Evaluation Protocol](#10-evaluation-protocol)
11. [Repository Layout](#11-repository-layout)
12. [Implementation Roadmap](#12-implementation-roadmap)
13. [Known Pitfalls and Mitigations](#13-known-pitfalls-and-mitigations)

---

## 1. Architecture Recap

CATE-π merges three papers into one VLA:

| Component | Source Paper | Role |
|---|---|---|
| PaliGemma backbone + flow-matching action expert | π0 / LeRobot | Core VLM + action generation chassis |
| Tactile-Aware Action Expert + CoT reasoning | Tactile-VLA (arXiv:2507.09160) | `[position, force]` action vector + failure recovery |
| Contact-aware gating | TacVLA (arXiv:2603.12665) | Gate tactile tokens out of attention when no contact |

Two original extensions beyond the papers:

- **Soft learnable gate** — a sigmoid MLP trained with BCE against the heuristic contact flag, replacing TacVLA's hard threshold
- **Raw ungated tactile history buffer** — feeds the CoT reasoning head only, so the action pathway's gating efficiency is preserved without blinding the reasoning pathway to contact→non-contact transitions

This plan covers only the **data layer and simulation environment**. The model code (`models/`, `controllers/`, `training/`) is simulator-agnostic and unchanged from the original plan; what changes are the two `sim/` files, `data/dataset.py`, the sensor config, and the force-labelling data source.

---

## 2. Prerequisites and Environment Setup

### 2.1 Software stack

| Package | Version | Notes |
|---|---|---|
| Isaac Sim | ≥ 4.5 / 5.0 | Omniverse-based; install via NVIDIA Omniverse Launcher or pip |
| Isaac Lab | ≥ 2.0 | `pip install isaaclab` or clone + `./isaaclab.sh --install` |
| isaaclab-contrib | current main | Contains `VisuoTactileSensorCfg` (TacSL backend) |
| Python | 3.10 | Isaac Sim ships its own interpreter; use it |
| PyTorch | ≥ 2.3 (CUDA 12.x) | Must match Isaac Sim's bundled CUDA |
| LeRobot | current main | For π0 checkpoint loading and dataset format |

### 2.2 Installation steps

```bash
# 1. Install Isaac Sim (headless-capable)
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics \
    isaacsim-extscache-kit isaacsim-extscache-kit-sdk --extra-index-url \
    https://pypi.nvidia.com

# 2. Install Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && ./isaaclab.sh --install

# 3. Install isaac-lab-contrib (contains TacSL visuo-tactile sensor)
pip install isaaclab-contrib

# 4. Install LeRobot (for π0 checkpoint + dataset utilities)
git clone https://github.com/huggingface/lerobot.git
pip install -e ./lerobot

# 5. Install CATE-π repo
git clone <your-repo> Pi-VLA
pip install -e Pi-VLA
```

### 2.3 GPU requirements

- Minimum: 1× RTX 3090 (24 GB) — single-env data collection
- Recommended: 1× A100 (80 GB) or 4090 — 16+ parallel envs for throughput
- Isaac Lab's GPU-parallelised tactile rendering (TacSL) runs all `num_envs` in a single CUDA kernel; throughput scales near-linearly with GPU memory up to ~256 envs

---

## 3. Robot Asset Selection and Setup

### 3.1 Recommended arms

Two open-source options, both with Isaac Lab support:

**Option A — Franka Panda + Robotiq 2F-85 (recommended for fastest start)**

- Both are natively in Isaac Sim's asset library as USD files; zero import work
- Franka is the most-cited arm in imitation learning (500+ papers in the last 3 years); results are directly comparable to prior work
- `FRANKA_PANDA_CFG` is importable directly from `isaaclab_assets`

```python
from isaaclab_assets import FRANKA_PANDA_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG
```

**Option B — SO-ARM100 / SO-ARM101 (recommended if you want real-hardware deployment later)**

- Fully open-source 6-DOF arm; 3D-printable; designed for imitation learning research
- `isaac_so_arm101` external project implements Isaac Lab tasks for it
- LeIsaac (Lightwheel) demonstrates a complete pipeline: Isaac Lab sim → HDF5 → LeRobot → GR00T fine-tuning → real hardware deployment
- URDF importable from LeRobot repo: `lerobot/SO-ARM100/URDF/.../SO_5DOF_ARM100.urdf`

```python
# SO-ARM100 via URDF import
from isaaclab.sim import UrdfFileCfg
from isaaclab.scene import ArticulationCfg

robot_cfg = ArticulationCfg(
    spawn=UrdfFileCfg(
        asset_path="lerobot/SO-ARM100/URDF/SO_5DOF_ARM100_8j_URDF.SLDASM/urdf/"
                   "SO_5DOF_ARM100_8j_URDF.SLDASM.urdf",
        fix_base=True,
        import_inertia_tensor=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(joint_names_expr=["joint.*"], stiffness=80.0, damping=4.0),
        "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper.*"], stiffness=40.0, damping=2.0),
    },
)
```

### 3.2 USD asset preparation

For the Franka path, skip this section — assets are pre-loaded.

For the SO-ARM100 path:

1. Open Isaac Sim GUI → File → Import → select the URDF file
2. In the URDF importer dialog: set `fix_base=True`, enable `import_inertia_tensor`
3. In the Stage panel: find `root_joint` → Properties → Physics → Articulation Root → delete it
4. Right-click on `SO100` → Add → Physics → Articulation Root (re-add at the correct level)
5. Export as USD: File → Save As → `assets/so_arm100.usd`
6. Add the elastomer prim for the tactile sensor at the fingertip(s) — see §4.2

### 3.3 Gripper tactile mounting points

The tactile sensor elastomer must be a separate USD prim parented to each fingertip link. For the Robotiq 2F-85:

```
/Robot/robotiq_2f_85/left_inner_finger_pad   ← elastomer prim here
/Robot/robotiq_2f_85/right_inner_finger_pad  ← elastomer prim here
```

For SO-ARM100, the gripper fingers are at:

```
/Robot/gripper_link/left_finger   ← elastomer prim here
/Robot/gripper_link/right_finger  ← elastomer prim here
```

The elastomer prim is a thin flat cylinder (matching your physical sensor geometry) with a PhysicsMaterial set to high stiffness. Isaac Lab's TacSL sensor attaches to this prim path.

---

## 4. Tactile Sensor Integration

### 4.1 Sensor choice rationale

Isaac Lab offers two tactile paths:

| Path | Class | Output | Best for |
|---|---|---|---|
| **TacSL visuo-tactile** (recommended) | `VisuoTactileSensorCfg` | RGB tactile image + distributed force field `(rows, cols, 3)` | CATE-π — gives both `X_t` for the tactile encoder and contact forces for the gate |
| Contact sensor (supplementary) | `ContactSensorCfg` | Net contact force per rigid body | Hard gate calibration (§4.4); cheaper to read every step |

Use TacSL for the primary `X_t` stream and `ContactSensor` for the per-step hard gate signal.

### 4.2 TacSL visuo-tactile sensor config

```python
# sim/sensor_configs.py

from isaaclab.sensors import TiledCameraCfg
from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

TACTILE_SENSOR_CFG = VisuoTactileSensorCfg(
    # Prim path — one sensor per fingertip; use regex for both fingers
    prim_path="{ENV_REGEX_NS}/Robot/gripper/.*_finger/elastomer",

    # Sensor output modes
    enable_camera_tactile=True,   # visuotactile RGB image → optional, for richer X_t
    enable_force_field=True,      # distributed normal + shear → primary X_t for CATE-π

    # Tactile array resolution — determines spatial token count in §4.3
    # 20×25 = 500 taxels per fingertip; comparable to TacVLA's 36-token setup
    # (your tactile encoder will pool this down to ~36 tokens)
    tactile_array_size=(20, 25),
    tactile_margin=0.003,         # elastomer boundary margin in metres

    # Contact object — randomised objects in the scene
    contact_object_prim_path_expr="{ENV_REGEX_NS}/object",

    # Force field physics parameters — tune against your sensor spec
    normal_contact_stiffness=1.0,
    friction_coefficient=2.0,
    tangential_stiffness=0.1,

    # Optional: camera for visuotactile RGB (can disable if only using force field)
    camera_cfg=TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/.*_finger/elastomer_cam",
        update_period=1 / 60,     # 60 Hz
        height=320,
        width=240,
        data_types=["distance_to_image_plane"],
        spawn=None,               # camera already spawned in USD
    ),
)
```

### 4.3 Contact sensor config (for hard gate calibration)

```python
from isaaclab.sensors import ContactSensorCfg

CONTACT_SENSOR_CFG = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper/.*_fingertip",
    track_air_time=False,
    max_contact_data_count=10,    # only need aggregate force, not full contact report
)
```

### 4.4 Reading sensor data in the environment step

```python
def _get_tactile_obs(self) -> dict:
    # Force field: (num_envs, rows, cols, 3) — channels are [normal_z, shear_x, shear_y]
    ff = self.tactile_sensor.data.force_field_data

    # Split channels — these map directly to CATE-π's action space §6
    normal  = ff[..., 0]    # (num_envs, 20, 25)   → X_t normal pressure map
    shear_x = ff[..., 1]    # (num_envs, 20, 25)   → X_t shear x
    shear_y = ff[..., 2]    # (num_envs, 20, 25)   → X_t shear y

    # Hard contact flag from ContactSensor (cheap, every step)
    net_force = self.contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    force_mag  = net_force.norm(dim=-1).max(dim=-1).values  # (num_envs,)
    c_t = (force_mag > self.p_thresh).float()               # binary (num_envs,)

    return {
        "tactile_normal":  normal,          # → tactile encoder input
        "tactile_shear":   torch.stack([shear_x, shear_y], dim=-1),
        "contact_flag":    c_t,             # → hard gate input
        "net_force_mag":   force_mag,       # → for force adverb auto-labelling
    }
```

### 4.5 Hard gate calibration

Before training, run a short calibration episode:

```python
# data/calibrate_gate.py

def calibrate_contact_thresholds(env, num_episodes=50):
    """
    Collect net_force_mag distributions from known-contact and
    known-no-contact frames. Fit p_thresh and n_thresh from the
    resulting empirical CDFs.
    """
    contact_forces    = []
    no_contact_forces = []

    for ep in range(num_episodes):
        obs = env.reset()
        for step in range(env.max_steps):
            # scripted policy: close gripper on object (known contact)
            # or hold arm in air (known no-contact)
            action = scripted_policy(obs, step)
            obs, _, done, info = env.step(action)

            phase = info["contact_phase"]   # "contact" | "free"
            f = obs["net_force_mag"]

            if phase == "contact":
                contact_forces.extend(f.cpu().tolist())
            else:
                no_contact_forces.extend(f.cpu().tolist())

    p_thresh = np.percentile(no_contact_forces, 95)   # 95th pctile of no-contact
    n_thresh = 3   # require at least 3 taxels above threshold (tune from data)

    print(f"p_thresh={p_thresh:.4f}  n_thresh={n_thresh}")
    return p_thresh, n_thresh
```

Write `p_thresh` and `n_thresh` to `configs/tactile_sensor.yaml` before any training run. Do not hand-pick these.

---

## 5. Isaac Lab Environment Implementation

### 5.1 Environment class structure

Create `sim/isaaclab_env.py`. This replaces both the old `pybullet_tacto_env.py` and `mujoco_shadowhand_env.py`.

```python
# sim/isaaclab_env.py

from __future__ import annotations
import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import (
    ObservationGroupCfg, ObservationTermCfg,
    ActionTermCfg, EventTermCfg, RewardTermCfg, TerminationTermCfg,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg

from .sensor_configs import TACTILE_SENSOR_CFG, CONTACT_SENSOR_CFG
from .robot_configs  import ROBOT_CFG       # Franka or SO-ARM100


@configclass
class CATEPiSceneCfg(InteractiveSceneCfg):
    """Scene: robot arm + fingertip tactile sensor + object + two cameras."""

    robot         = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    tactile_sensor = TACTILE_SENSOR_CFG
    contact_sensor = CONTACT_SENSOR_CFG

    # Wrist RGB camera (matches TacVLA/Tactile-VLA setup)
    wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_link/wrist_cam",
        update_period=1 / 30,
        height=224, width=224,
        data_types=["rgb"],
        spawn=None,
    )

    # Front-facing RGB camera
    front_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/front_cam",
        update_period=1 / 30,
        height=224, width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0),
    )

    # Manipulable object with randomisable physics
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(usd_path="assets/objects/cube.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )


@configclass
class CATEPiEnvCfg(ManagerBasedRLEnvCfg):
    scene       = CATEPiSceneCfg(num_envs=16, env_spacing=2.0)
    observations = CATEPiObsCfg()
    actions     = CATEPiActionCfg()
    events      = CATEPiEventCfg()     # domain randomisation
    rewards     = CATEPiRewardCfg()
    terminations = CATEPiTerminationCfg()

    # Control
    decimation        = 2              # policy runs at sim_hz / decimation
    sim_dt            = 1.0 / 100.0   # 100 Hz physics
    episode_length_s  = 10.0


class CATEPiEnv(ManagerBasedRLEnv):
    """
    Isaac Lab environment for CATE-π data collection.

    Observation dict (returned every control step):
        rgb_wrist:        (B, 3, 224, 224)
        rgb_front:        (B, 3, 224, 224)
        proprioception:   (B, D_prop)
        tactile_normal:   (B, 20, 25)
        tactile_shear:    (B, 20, 25, 2)
        contact_flag:     (B,)
        net_force_mag:    (B,)      ← for force adverb labelling only

    Action dict (sent to hybrid controller each step):
        joint_pos_targets:    (B, N_joints)
        gripper_pos_target:   (B, 1)
        force_normal_target:  (B, 1)      ← from action expert §6
        force_shear_target:   (B, 2)      ← from action expert §6
    """

    cfg: CATEPiEnvCfg

    def _setup_scene(self):
        super()._setup_scene()
        # Grab sensor handles after scene build
        self.tactile_sensor  = self.scene["tactile_sensor"]
        self.contact_sensor  = self.scene["contact_sensor"]
        self.wrist_cam       = self.scene["wrist_cam"]
        self.front_cam       = self.scene["front_cam"]
        self.robot           = self.scene["robot"]

        # Load calibrated thresholds from config
        self.p_thresh = self.cfg.tactile_p_thresh   # from configs/tactile_sensor.yaml
        self.n_thresh = self.cfg.tactile_n_thresh
```

### 5.2 Observation manager

```python
@configclass
class CATEPiObsCfg:
    @configclass
    class PolicyObs(ObservationGroupCfg):
        rgb_wrist       = ObservationTermCfg(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam")})
        rgb_front       = ObservationTermCfg(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front_cam")})
        joint_pos       = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel       = ObservationTermCfg(func=mdp.joint_vel_rel)
        tactile_ff      = ObservationTermCfg(func=obs_tactile_ff)    # custom term → see below
        contact_flag    = ObservationTermCfg(func=obs_contact_flag)  # custom term

    policy = PolicyObs()


def obs_tactile_ff(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns (B, 20, 25, 3) — normal + shear XY force field."""
    return env.tactile_sensor.data.force_field_data


def obs_contact_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns (B,) binary contact flag from ContactSensor."""
    net_force = env.contact_sensor.data.net_forces_w.norm(dim=-1).max(dim=-1).values
    return (net_force > env.p_thresh).float()
```

### 5.3 Action manager

The policy outputs a CATE-π action chunk; the environment executes the hybrid position-force controller (§7 of the original plan) which is unchanged:

```python
@configclass
class CATEPiActionCfg:
    arm_action     = JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"])
    gripper_action = GripperActionCfg(asset_name="robot", joint_names=["panda_finger.*"])
    # Force targets are consumed by the hybrid controller; they are not directly
    # sent to the articulation — the controller reads them and adjusts joint positions
```

The hybrid controller in `controllers/hybrid_position_force.py` stays logically identical but updates its joint command calls from the old PyBullet API to Isaac Lab's articulation API:

```python
# OLD (PyBullet):
# p.setJointMotorControl2(robot_id, joint_id, p.POSITION_CONTROL, target)

# NEW (Isaac Lab):
self.robot.set_joint_position_target(
    positions  = adjusted_positions,    # (B, N_joints)
    joint_ids  = self.arm_joint_ids,
)
self.robot.set_joint_effort_target(
    efforts    = grasp_torques,         # (B, N_gripper_joints)
    joint_ids  = self.gripper_joint_ids,
)
```

### 5.4 Domain randomisation

This is the primary advantage over the original PyBullet plan — randomisation runs natively in parallel across all `num_envs`:

```python
@configclass
class CATEPiEventCfg:
    # Randomise object physical properties every episode reset
    object_physics = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range":  (0.3, 1.5),
            "dynamic_friction_range": (0.2, 1.2),
            "restitution_range":      (0.0, 0.3),
        },
    )

    # Randomise object mass
    object_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.05, 0.5),  # kg
        },
    )

    # Randomise object initial position (for OOD generalisation eval)
    object_pose = EventTermCfg(
        func=mdp.randomize_rigid_body_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.35, 0.65),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
            },
        },
    )

    # Additive Gaussian noise on tactile readings (tactile sim-to-real gap)
    tactile_noise = EventTermCfg(
        func=add_tactile_noise,    # custom term
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={"noise_std": 0.02},
    )
```

### 5.5 Force adverb auto-labelling

Run this at the end of each episode before writing to HDF5, using the `net_force_mag` log:

```python
# data/force_adverb_labeler.py

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
```

---

## 6. Demonstration Data Collection

### 6.1 Collection methods

Isaac Lab supports three collection modes. Use them in this order:

**Mode 1 — Scripted state-machine policy (primary)**

Implement a heuristic policy in `data/scripted_policy.py`. This is the fastest way to generate large volumes of force-varied data without a pre-trained model:

```
Phase 1: Move arm above object (no contact)
Phase 2: Descend to grasp height (approach)
Phase 3: Close gripper to target_force (CONTACT)
Phase 4: Lift
Phase 5: Place / release
```

Vary `target_force` per episode: sample from `[0.1, 5.0]` N uniformly. This directly generates the force vocabulary distribution needed for Tactile-VLA's Table 2 experiment without any manual labelling.

**Mode 2 — Isaac Lab Mimic (data augmentation)**

After collecting 20–50 scripted demonstrations, annotate subtasks and use Isaac Lab Mimic to generate thousands more:

```bash
# Annotate subtasks in recorded HDF5
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
    --input_file  datasets/catephi_raw.hdf5 \
    --output_file datasets/catephi_annotated.hdf5 \
    --task        CATEPi-Grasp-v0-Mimic

# Generate augmented dataset (scales to hundreds of episodes)
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
    --input_file  datasets/catephi_annotated.hdf5 \
    --output_file datasets/catephi_augmented.hdf5 \
    --num_demos   500
```

**Mode 3 — Failure injection for CoT dataset (Phase 3 training only)**

Scripted policy that deliberately under-applies force: clamp `target_force` to 10% of the successful threshold. These failed trajectories, paired with auto-generated diagnostic text from the force log, form the CoT fine-tuning dataset (§4.7 of the original plan):

```python
def failure_policy(obs, target_force_fraction=0.1):
    """Same as scripted_policy but intentionally uses too little force."""
    ...
```

### 6.2 Data recording script

Isaac Lab's built-in recorder writes to RoboMimic-compatible HDF5. Extend it to capture the two extra CATE-π fields:

```bash
./isaaclab.sh -p sim/collect_demos.py \
    --task     CATEPi-Grasp-v0 \
    --num_demos 200 \
    --num_envs  16 \
    --output    datasets/catephi_raw.hdf5
```

The custom `collect_demos.py` extends Isaac Lab's `record_demos.py` to write:

```python
# Per-step additions to the HDF5 episode group
hdf5_episode["obs/tactile_ff"]    = tactile_ff_log     # (T, 20, 25, 3)
hdf5_episode["obs/contact_flag"]  = contact_flag_log   # (T,)
hdf5_episode["obs/net_force_mag"] = net_force_log      # (T,) — for labelling only

# Per-episode language instruction (from force adverb labeller)
hdf5_episode.attrs["language_instruction"] = label_episode_force(net_force_log)
```

### 6.3 Replay verification

Always replay a subset of collected demos before running training. Isaac Lab's non-deterministic reset means some replays will fail even if collection succeeded — filter on replay success, not collection:

```bash
./isaaclab.sh -p scripts/tools/replay_demos.py \
    --task         CATEPi-Grasp-v0 \
    --dataset_file datasets/catephi_raw.hdf5 \
    --num_demos    20
```

---

## 7. Data Pipeline: HDF5 → LeRobot Format

### 7.1 Why this step

π0 / CATE-π training loads data in the LeRobot format (columnar Parquet for time-series, MP4 for videos). Isaac Lab produces RoboMimic-schema HDF5. The conversion is provided by Isaac Lab and by LeIsaac, but both need extension to carry the tactile fields.

### 7.2 Conversion config

```yaml
# data/hdf5_to_lerobot_config.yaml

data_root:  datasets/
hdf5_name:  catephi_augmented.hdf5
output_dir: datasets/catephi_lerobot/

# Standard field mappings
state_name_sim:    robot_joint_pos
action_name_sim:   processed_actions
pov_cam_name_sim:  rgb_wrist

# NEW — CATE-π tactile fields (not in vanilla Isaac Lab converter)
extra_fields:
  - source:      obs/tactile_ff       # (T, 20, 25, 3) per episode
    target:      observation.tactile_ff
    dtype:       float32
  - source:      obs/contact_flag     # (T,) binary
    target:      observation.contact_flag
    dtype:       float32

# Language instruction (per-episode, from force adverb labeller)
language_instruction_key: language_instruction

fps:         50
chunk_size:  1000
```

### 7.3 Conversion script

Fork LeIsaac's `convert_hdf5_to_lerobot.py` and extend the field mapping loop:

```python
# data/convert_to_lerobot.py

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import yaml, h5py, numpy as np

def convert(config_path: str):
    cfg = yaml.safe_load(open(config_path))
    h5 = h5py.File(f"{cfg['data_root']}/{cfg['hdf5_name']}", "r")
    dataset = LeRobotDataset.create(
        repo_id   = "local/catephi",
        fps       = cfg["fps"],
        root      = cfg["output_dir"],
    )

    for ep_key in h5["data"].keys():
        ep = h5["data"][ep_key]
        T  = ep["obs/robot_joint_pos"].shape[0]

        frame = {
            # Standard fields
            "observation.state":         ep["obs/robot_joint_pos"][:],
            "action":                    ep["actions"][:],
            "observation.image.wrist":   ep["obs/rgb_wrist"][:],

            # CATE-π tactile fields
            "observation.tactile_ff":    ep["obs/tactile_ff"][:],     # (T, 20, 25, 3)
            "observation.contact_flag":  ep["obs/contact_flag"][:],   # (T,)

            # Language instruction (same string repeated for all frames)
            "language_instruction": [ep.attrs["language_instruction"]] * T,
        }
        dataset.add_episode(frame)

    dataset.consolidate()
    print(f"Saved LeRobot dataset to {cfg['output_dir']}")
```

Run it:

```bash
python data/convert_to_lerobot.py \
    --config data/hdf5_to_lerobot_config.yaml
```

### 7.4 Verifying tactile fields in the converted dataset

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/catephi", root="datasets/catephi_lerobot")
sample = ds[0]

assert "observation.tactile_ff"   in sample, "tactile_ff missing"
assert "observation.contact_flag" in sample, "contact_flag missing"
assert sample["observation.tactile_ff"].shape == (20, 25, 3), \
    f"Expected (20, 25, 3), got {sample['observation.tactile_ff'].shape}"

print("Tactile field check passed.")
```

---

## 8. Model Architecture Changes

The model code is **simulator-agnostic** — no changes to `models/`, the training losses, or attention masks. The only change is how the tactile tensor `X_t` is shaped when it enters the tactile encoder.

### 8.1 Tactile encoder input shape update

The original plan assumed TACTO/Taxim output `(rows, cols)` single-channel pressure. Isaac Lab's TacSL force field outputs `(rows, cols, 3)` — normal + shear XY.

Update `models/tactile_encoder.py` to accept 3-channel input:

```python
# BEFORE (TACTO/Taxim — single pressure channel)
# input: (B, K, rows, cols)

# AFTER (Isaac Lab TacSL — normal + shear XY)
# input: (B, K, rows, cols, 3)

class TactileEncoder(nn.Module):
    def __init__(self, rows=20, cols=25, in_channels=3, ...):
        # Project 3-channel taxel reading to embedding
        self.taxel_proj = nn.Linear(in_channels, hidden_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, K, rows, cols, 3)
        B, K, R, C, Ch = X.shape
        x = X.reshape(B * K, R * C, Ch)
        x = self.taxel_proj(x)          # (B*K, R*C, hidden_dim)
        # ... rest of temporal conv / GRU unchanged
```

### 8.2 Contact gate input update

`contact_gate.py` currently reads a single-channel pressure proxy for the soft gate. With TacSL you now have a direct normal force magnitude per taxel — use the zero-th channel of the force field directly:

```python
# In contact_gate.py → HardContactGate.forward()
# BEFORE: pressure = X_t.mean(dim=(-2, -1))   (single-channel proxy)
# AFTER:
normal_force = X_t[..., 0]                      # (B, rows, cols) — normal channel
pressure = normal_force.mean(dim=(-2, -1))       # (B,) — mean normal pressure
n_active = (normal_force > self.p_thresh).sum(dim=(-2, -1))  # (B,)
c_t = (n_active >= self.n_thresh).float()
```

### 8.3 Action space — fix the force channel gap

This was flagged in the code review: `hybrid_position_force.py` reuses a single `force_normal` for both external and grasp force because the action vector only had one normal force channel. Add a second channel:

```yaml
# configs/backbone_pi0_base.yaml → action_space.gripper

total_dim:  14     # was 13; added F_grasp_normal
padded_dim: 32

# Channel layout:
#   0:2   Δx, Δy, Δz          (Cartesian delta)
#   3:8   6D rotation
#   9     gripper_width
#   10    F_ext_normal         # how hard end-effector presses environment
#   11:12 F_ext_shear_x/y      # shear on environment contact
#   13    F_grasp_normal       # NEW — how hard fingers squeeze object (internal)
```

Update `action_expert.py`'s `split_action_vector()` to expose `force_grasp_normal` as a separate key, and update `hybrid_position_force.py`'s `step()` to read two distinct channels:

```python
def step(self, action_dict):
    ext_force_target   = action_dict["force_normal"]    # channel 10
    grasp_force_target = action_dict["force_grasp_normal"]  # channel 13 — NEW
    ...
    adjusted_pos = self.compute_hybrid_target(
        pos_target=action_dict["position"],
        force_normal_target=ext_force_target,           # external only
        force_normal_measured=self.measured_ext_force,
    )
    adjusted_gripper = self.compute_grasp_target(
        gripper_target=action_dict["gripper"],
        grasp_force_target=grasp_force_target,          # internal grasp only
        grasp_force_measured=self.measured_grasp_force,
    )
```

This fixes the conflation bug identified in the code review — the controller can now decouple external pressing force from internal squeeze force, which is physically required for tasks like "hold gently while moving" vs. "squeeze firmly while stationary."

---

## 9. Training Curriculum

No changes to the training curriculum structure — but Phase 2 now benefits from Isaac Lab's GPU-parallel data collection:

| Phase | What trains | What freezes | Data source | Duration |
|---|---|---|---|---|
| **0** | — | Everything | Pre-trained π0/π0.5 checkpoint from LeRobot Hub | 0 training; verify forward pass |
| **1** | Tactile encoder + contact gate only | VLM backbone + action expert | Isaac Lab: scripted contact/no-contact episodes | 1–2 days, cheap |
| **2** | Tactile encoder + gate + Tactile-Aware Action Expert; LoRA on backbone | — | Isaac Lab: full grasp demos with force labels | 3–5 days |
| **3** | CoT language head only; LoRA on decoder | Action expert | Isaac Lab: failure injection episodes + ~10% general VQA data | 1–2 days |
| **4** (optional) | All (LoRA); on-policy | — | Isaac Lab: closed-loop rollouts with DAgger | 2–4 days |

### Phase 1 data requirements

- 500–1000 contact episodes + 500–1000 free-space episodes
- Each episode: 100 steps, record `(X_t, c_t)` pairs
- Goal: contact classification accuracy ≥ 90% before proceeding

### Phase 2 data requirements

- 2000–5000 successful grasp demonstrations (use Isaac Lab Mimic to scale from 50 human/scripted demos)
- Force vocabulary: ~33% "gently", ~33% "normally", ~33% "firmly" (enforce during scripted policy)
- Action normalisation: compute quantile stats per channel across the full dataset before training; write to `data/normalization_stats.json`

```python
# data/action_normalization.py
# Run once on the full converted LeRobot dataset

def compute_normalization_stats(dataset: LeRobotDataset) -> dict:
    all_actions = []
    for i in range(len(dataset)):
        all_actions.append(dataset[i]["action"])
    actions = torch.stack(all_actions)   # (N, padded_dim)

    stats = {}
    for ch in range(actions.shape[-1]):
        q01 = torch.quantile(actions[:, ch], 0.01)
        q99 = torch.quantile(actions[:, ch], 0.99)
        stats[f"ch{ch}_min"] = q01.item()
        stats[f"ch{ch}_max"] = q99.item()
    return stats
```

### Phase 3 data requirements

- 200–500 failure trajectories (from failure injection scripted policy)
- Each failure paired with auto-generated text: `"The grasp force was too low. Apply more force."`
- Mixed with 10% general instruction-following data (from original π0 training mix) to prevent catastrophic forgetting

---

## 10. Evaluation Protocol

All evaluation runs inside Isaac Lab; no separate eval environment needed.

### 10.1 Core metrics per task

| Metric | Implementation |
|---|---|
| Success rate (%) | `termination_manager.episode_done & reward > threshold` |
| Force accuracy | Pearson correlation between language adverb bucket and `peak(net_force_mag)` per episode |
| ID/OOD generalisation | Train on `[friction 0.5, mass 0.1 kg]`; eval on `[friction 1.5, mass 0.4 kg]` |
| Camera occlusion robustness | Zero out `rgb_wrist` obs at eval time; report success rate delta |
| Tactile dropout robustness | Zero out `tactile_ff` obs at eval time; report success rate delta |

### 10.2 Ablation table

| Variant | Isolates |
|---|---|
| Full CATE-π | — |
| − contact gating (tactile always-on) | Value of TacVLA gating |
| Hard gate vs. soft learnable gate | Value of learnable gate extension |
| − CoT head | Value of Tactile-VLA-CoT recovery |
| Fixed-interval CoT vs. event-triggered CoT | Value of event-trigger extension |
| − force channels in action space (position-only) | Value of Tactile-Aware Action Expert |
| Gated CoT buffer vs. raw always-on buffer | Value of §4.7 raw-buffer design |
| Single force channel vs. split ext/grasp channels | Value of the force-channel fix |

Run each ablation variant on the same 100-episode eval set. Record `(success_rate, force_accuracy)` for each.

---

## 11. Repository Layout

```
Pi-VLA/
├── assets/
│   ├── robots/
│   │   ├── franka_panda.usd          # or symlink to Isaac Sim asset library
│   │   └── so_arm100.usd             # exported from URDF import
│   ├── objects/
│   │   ├── cube.usd
│   │   ├── sphere.usd
│   │   └── ycb/                      # YCB objects for OOD eval
│   └── sensors/
│       └── gelsight_r15.usd          # elastomer prim for TacSL
│
├── configs/
│   ├── backbone_pi0_base.yaml        # action_dim updated to 14
│   ├── tactile_sensor.yaml           # p_thresh, n_thresh, array_size
│   └── training_phase{0,1,2,3}.yaml
│
├── sim/                              # ← primary change from original plan
│   ├── isaaclab_env.py               # NEW — replaces pybullet + mujoco files
│   ├── sensor_configs.py             # NEW — TACTILE_SENSOR_CFG, CONTACT_SENSOR_CFG
│   ├── robot_configs.py              # NEW — ROBOT_CFG (Franka or SO-ARM100)
│   └── collect_demos.py             # NEW — extends Isaac Lab record_demos.py
│
├── data/
│   ├── dataset.py                    # MODIFIED — adds tactile_ff, contact_flag fields
│   ├── action_normalization.py       # unchanged logic; runs on new dataset
│   ├── force_adverb_labeler.py       # MODIFIED — reads from ContactSensor, not PyBullet
│   ├── calibrate_gate.py             # NEW — calibrates p_thresh, n_thresh from Isaac Lab data
│   ├── convert_to_lerobot.py         # NEW — extends LeIsaac converter with tactile fields
│   └── hdf5_to_lerobot_config.yaml   # NEW — conversion config
│
├── models/
│   ├── tactile_encoder.py            # MODIFIED — input channels: 1→3 (normal+shear XY)
│   ├── contact_gate.py               # MODIFIED — reads normal channel directly
│   ├── action_expert.py              # MODIFIED — action_dim 13→14, split force channels
│   ├── vlm_backbone.py               # unchanged
│   ├── attention_masks.py            # unchanged (NaN bug fix applied separately)
│   └── cot_head.py                   # unchanged (crash fix applied separately)
│
├── controllers/
│   └── hybrid_position_force.py      # MODIFIED — Isaac Lab joint API + split force channels
│
├── training/
│   ├── train_phase1_gate_warmup.py   # unchanged
│   ├── train_phase2_cfm_finetune.py  # unchanged
│   └── train_phase3_cot_finetune.py  # unchanged
│
└── eval/
    ├── eval_suite.py                 # unchanged — runs inside Isaac Lab env
    └── ablations.py                  # add: "split_force_channels" ablation arm
```

---

## 12. Implementation Roadmap

### Week 1 — Isaac Lab environment skeleton

- [ ] Install Isaac Sim + Isaac Lab + isaaclab-contrib
- [ ] Choose robot: Franka (faster start) or SO-ARM100 (better deployment path)
- [ ] Import robot asset; verify forward pass with zero-action agent in Isaac Sim GUI
- [ ] Implement `sim/sensor_configs.py` — `TACTILE_SENSOR_CFG` and `CONTACT_SENSOR_CFG`
- [ ] Implement `sim/isaaclab_env.py` skeleton (scene + obs manager, no tactile yet)
- [ ] Verify RGB cameras render in both `rgb_wrist` and `rgb_front` slots
- [ ] Milestone: `CATEPiEnv` instantiates with 4 envs; obs dict shape-checks pass

### Week 2 — Tactile sensor integration and gate calibration

- [ ] Add TacSL sensor to scene; verify `force_field_data` shape is `(num_envs, 20, 25, 3)`
- [ ] Implement `obs_tactile_ff` and `obs_contact_flag` observation terms
- [ ] Implement scripted contact/no-contact policy for gate calibration
- [ ] Run `data/calibrate_gate.py`; write `p_thresh`, `n_thresh` to `configs/tactile_sensor.yaml`
- [ ] Update `models/tactile_encoder.py` input channels 1→3
- [ ] Update `models/contact_gate.py` to read normal channel directly
- [ ] Milestone: contact classification accuracy ≥ 90% on held-out calibration episodes (Phase 1 training)

### Week 3 — Scripted policy and data collection

- [ ] Implement `data/scripted_policy.py` with force-varied grasps (`target_force` ∈ `[0.1, 5.0]` N)
- [ ] Implement failure injection variant for Phase 3
- [ ] Implement `sim/collect_demos.py` (extends Isaac Lab recorder with tactile fields)
- [ ] Collect 50 scripted demos; run `scripts/tools/replay_demos.py` to verify HDF5 quality
- [ ] Implement `data/force_adverb_labeler.py` with Isaac Lab data source
- [ ] Milestone: 50 verified demonstrations in HDF5 with correct tactile fields and language labels

### Week 4 — Isaac Lab Mimic and data pipeline

- [ ] Annotate subtasks in collected demos
- [ ] Run Isaac Lab Mimic to generate 500+ augmented demonstrations
- [ ] Implement `data/convert_to_lerobot.py` with tactile field extension
- [ ] Run conversion; verify LeRobot dataset has `observation.tactile_ff` and `observation.contact_flag`
- [ ] Compute and save action normalisation stats per channel (critical before Phase 2)
- [ ] Milestone: 500+ episode LeRobot dataset ready; action normalisation stats verified

### Week 5–6 — Phase 2 training

- [ ] Fix action_dim: update configs from 13→14 (add `F_grasp_normal`)
- [ ] Fix Beta distribution skew in `action_expert.py`: flip `alpha=1.5, beta=1.0` → `alpha=1.0, beta=1.5`
- [ ] Fix NaN bug in `attention_masks.py`: remove row-zeroing, keep column-zeroing only
- [ ] Fix crash in `cot_head.py`: rename `F` → `n_feat` in `TactileBufferTokenizer.forward`
- [ ] Update `hybrid_position_force.py`: split ext/grasp force channels; update joint command API
- [ ] Run Phase 2 CFM fine-tuning; monitor per-channel loss (confirm force channels not ignored)
- [ ] Milestone: model achieves >60% grasp success rate on ID eval set

### Week 7 — Phase 3 (CoT) and ablations

- [ ] Generate 200 failure episodes with Isaac Lab failure injection policy
- [ ] Fix CoT prompt injection in `cot_head.py`: tokenise and prepend `_cot_prompt_template`
- [ ] Run Phase 3 CoT fine-tuning
- [ ] Run ablation table (§10.2) — each variant is a config toggle, no code changes
- [ ] Milestone: ablation table complete; CoT head generates coherent corrective instructions

### Week 8 — Robustness evaluation and cleanup

- [ ] Run camera occlusion and tactile dropout eval variants
- [ ] Port domain randomisation ranges from ID → OOD (wider friction, mass ranges)
- [ ] Optional Phase 4: on-policy fine-tuning with DAgger
- [ ] Clean up repo; update `README.md` with Isaac Lab setup instructions
- [ ] Milestone: full eval suite complete with publishable ablation table

---

## 13. Known Pitfalls and Mitigations

### Isaac Lab specific

**Non-deterministic episode reset.**
Isaac Lab's `env.reset()` is not deterministically reproducible when using GPU parallelism. Demos that succeed during collection may fail on replay. Always replay 10–20% of collected demos and use only episodes that succeed on replay for training.

**TacSL sensor baseline calibration.**
The TacSL sensor must be in a no-contact state when its baseline is initialised. If the sensor touches any object during `reset()`, the baseline will be wrong, causing erroneous force readings for that episode. Enforce a brief arm-retract phase at the start of every episode before the sensor baseline is set.

**Elastomer prim USD setup.**
The TacSL sensor attaches to a specific elastomer prim path in the USD. If this prim is missing or incorrectly positioned, the sensor initialises silently but returns all-zero force fields. Verify with the Isaac Sim GUI's physics debug overlay before running data collection.

**Multi-env tactile rendering cost.**
Each TacSL sensor renders tactile images via tiled cameras. At 16 envs with two fingertip sensors each, you have 32 tiled cameras active per physics step. Benchmark your GPU's frame budget before committing to `num_envs`; if you hit memory limits, disable `enable_camera_tactile` and use only `enable_force_field` (which is cheaper and is what CATE-π actually uses for `X_t`).

### Model bugs already identified (fix before training)

| Bug | File | Fix |
|---|---|---|
| NaN gradients in gated attention | `attention_masks.py` | Remove row-zeroing; keep only column masking |
| Variable shadowing crash | `cot_head.py` L165 | Rename `F` → `n_feat` in `TactileBufferTokenizer.forward` |
| CoT prompt never injected | `cot_head.py` L401 | Tokenise `_cot_prompt_template`; prepend to generation |
| Beta distribution skew backwards | `action_expert.py` L123 | Swap `alpha=1.5, beta=1.0` → `alpha=1.0, beta=1.5` |
| Single force channel reused for ext + grasp | `action_expert.py` §6 | Add `F_grasp_normal` channel; update `action_dim` 13→14 |
| Cross-attention ignores gate mask | `action_expert.py` L90 | Pass gated prefix mask to `self.cross_attn` |
| CoT local mask ignores gate | `cot_head.py` L423 | Multiply prefix block of local mask by `contact_gate` |
| `vlm_backbone.py` mask consumption unverified | `vlm_backbone.py` | Review: confirm gated mask from `attention_masks.py` is consumed |

All of these must be fixed before Phase 2 training. Phase 1 (gate warmup) only trains the tactile encoder and contact gate, so it is unaffected by the attention mask and action expert bugs.

### Training stability

**Force channels ignored (most common failure mode in Phase 2).**
If `force_normal_loss` and `force_shear_loss` in your per-channel metrics plateau near zero while `position_loss` decreases, the model is ignoring force. The cause is always unnormalised action channels — position in metres and force in Newtons have a ~10–100× scale difference. Verify normalisation stats are applied before the CFM loss, not after.

**Soft gate collapses.**
If `g_t` saturates at 0 or 1 for all inputs during Phase 2, the BCE auxiliary loss weight is out of balance. Log the `g_t` distribution histogram every 100 steps. If it collapses, try reducing the BCE loss weight by 5× and ensure your demonstration data has a roughly equal balance of contact and no-contact frames.

---

## Key References

- Black, K. et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* arXiv:2410.24164, 2024.
- Huang, J. et al. *Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization.* arXiv:2507.09160, 2025.
- Zhang, K. et al. *TacVLA: Contact-Aware Tactile Fusion for Robust Vision-Language-Action Manipulation.* arXiv:2603.12665, 2026.
- Akinola, I. et al. *TacSL: A Library for Visuotactile Sensor Simulation and Learning.* IEEE Transactions on Robotics, 2025.
- NVIDIA. *Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning.* arXiv:2511.04831, 2025.
- Lightwheel. *LeIsaac: Teleoperation and Policy Training in Isaac Lab.* GitHub, 2026.
- LeRobot `PI0Policy` implementation and π0/π0-FAST integration — Hugging Face.
