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
import isaaclab.envs.mdp as mdp
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from .sensor_configs import TACTILE_SENSOR_CFG, CONTACT_SENSOR_CFG
from .robot_configs import ROBOT_CFG
from isaaclab.envs.mdp import JointPositionActionCfg


def add_tactile_noise(env: ManagerBasedRLEnv, noise_std: float = 0.02):
    # Dummy function for event term
    pass


@configclass
class CATEPiSceneCfg(InteractiveSceneCfg):
    """Scene: robot arm + fingertip tactile sensor + object + two cameras."""
    
    robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    tactile_sensor = TACTILE_SENSOR_CFG
    contact_sensor = CONTACT_SENSOR_CFG

    # Wrist RGB camera (matches TacVLA/Tactile-VLA setup)
    wrist_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/wrist_link/wrist_cam", # Example path
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


def obs_tactile_ff(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns (B, 20, 25, 3) — normal + shear XY force field."""
    return env.tactile_sensor.data.force_field_data


def obs_contact_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns (B,) binary contact flag from ContactSensor."""
    net_force = env.contact_sensor.data.net_forces_w.norm(dim=-1).max(dim=-1).values
    return (net_force > env.p_thresh).float()


@configclass
class CATEPiObsCfg:
    @configclass
    class PolicyObs(ObservationGroupCfg):
        rgb_wrist       = ObservationTermCfg(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist_cam")})
        rgb_front       = ObservationTermCfg(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front_cam")})
        joint_pos       = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel       = ObservationTermCfg(func=mdp.joint_vel_rel)
        tactile_ff      = ObservationTermCfg(func=obs_tactile_ff)
        contact_flag    = ObservationTermCfg(func=obs_contact_flag)

    policy = PolicyObs()


@configclass
class CATEPiActionCfg:
    arm_action     = JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"])
    # Gripper action logic handled in hybrid controller.


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
        func=add_tactile_noise,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={"noise_std": 0.02},
    )


@configclass
class CATEPiRewardCfg:
    pass

@configclass
class CATEPiTerminationCfg:
    pass


@configclass
class CATEPiEnvCfg(ManagerBasedRLEnvCfg):
    scene       = CATEPiSceneCfg(num_envs=16, env_spacing=2.0)
    observations = CATEPiObsCfg()
    actions     = CATEPiActionCfg()
    events      = CATEPiEventCfg()     # domain randomisation
    rewards     = CATEPiRewardCfg()
    terminations = CATEPiTerminationCfg()

    # thresholds to be updated from calibration
    tactile_p_thresh = 0.5
    tactile_n_thresh = 3

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
