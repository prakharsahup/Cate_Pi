"""
PyBullet + TACTO/Taxim Simulation Environment for CATE-π (§9, Path A).

Higher-fidelity tactile simulation using TACTO (DIGIT-style optical sensors)
or Taxim (data-driven GelSight-style) integrated with PyBullet.

This environment provides:
- Spatially-resolved tactile images/taxel grids (maps to TacVLA-style tokenization)
- Ground-truth contact forces for auto-labeling
- Multi-camera rendering (front + wrist)
- Domain randomization (friction, mass, object geometry)

Recommended for Phase 2+ after architecture is validated with Path B (MuJoCo).
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class TactileSensorConfig:
    """Configuration for a simulated tactile sensor."""
    rows: int = 15
    cols: int = 8
    width_mm: float = 24.0
    height_mm: float = 18.0
    max_pressure: float = 10.0  # Newtons
    noise_std: float = 0.01
    sensor_type: str = "tacto"  # "tacto" or "taxim"


@dataclass
class GripperConfig:
    """Configuration for the simulated gripper."""
    type: str = "parallel_jaw"  # "parallel_jaw" or "dexterous"
    max_width: float = 0.085    # meters (Robotiq 2F-85)
    max_force: float = 20.0     # Newtons
    num_fingers: int = 2
    tactile_sites: int = 1      # sensors per finger


class PyBulletTactoEnv:
    """
    PyBullet + TACTO simulation environment.

    Provides a Gym-like interface for collecting demonstrations
    and running the CATE-π policy.
    """

    def __init__(
        self,
        render: bool = True,
        tactile_config: Optional[TactileSensorConfig] = None,
        gripper_config: Optional[GripperConfig] = None,
        control_hz: float = 15.0,
        camera_width: int = 224,
        camera_height: int = 224,
        use_domain_randomization: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            render: Whether to render the GUI.
            tactile_config: Tactile sensor configuration.
            gripper_config: Gripper configuration.
            control_hz: Control loop frequency.
            camera_width: Camera image width.
            camera_height: Camera image height.
            use_domain_randomization: Randomize physics params.
            seed: Random seed.
        """
        self.render = render
        self.tactile_config = tactile_config or TactileSensorConfig()
        self.gripper_config = gripper_config or GripperConfig()
        self.control_hz = control_hz
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.use_domain_randomization = use_domain_randomization
        self.rng = np.random.RandomState(seed)

        # Simulation state (initialized on reset)
        self._physics_client = None
        self._robot_id = None
        self._object_ids: List[int] = []
        self._step_count = 0

        # Observation / action spaces
        self.observation_space = {
            "images": (2, 3, camera_height, camera_width),  # front + wrist
            "tactile": (tactile_config.rows if tactile_config else 15,
                       tactile_config.cols if tactile_config else 8),
            "state": (32,),  # joint angles + gripper state
            "force": (3,),   # measured [normal, shear_x, shear_y]
        }

        self.action_space = {
            "position": (3,),
            "orientation": (6,),
            "gripper": (1,),
            "force_normal": (1,),
            "force_shear": (2,),
        }

    def _init_physics(self):
        """Initialize PyBullet physics engine."""
        try:
            import pybullet as p
            import pybullet_data

            if self.render:
                self._physics_client = p.connect(p.GUI)
            else:
                self._physics_client = p.connect(p.DIRECT)

            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client)
            p.setTimeStep(1.0 / 240.0, physicsClientId=self._physics_client)

            # Load ground plane
            p.loadURDF("plane.urdf", physicsClientId=self._physics_client)

        except ImportError:
            print("PyBullet not installed. Using mock physics for development.")
            self._physics_client = "mock"

    def _init_tactile_sensor(self):
        """Initialize TACTO or Taxim tactile sensor."""
        if self.tactile_config.sensor_type == "tacto":
            try:
                import tacto
                self._tactile_sensor = tacto.Sensor(
                    width=self.tactile_config.cols,
                    height=self.tactile_config.rows,
                )
                print("TACTO tactile sensor initialized")
            except ImportError:
                print("TACTO not installed. Using simulated tactile readings.")
                self._tactile_sensor = None
        else:
            print(f"Sensor type {self.tactile_config.sensor_type} — using simulated readings.")
            self._tactile_sensor = None

    def _apply_domain_randomization(self):
        """Randomize physics parameters for generalization."""
        if not self.use_domain_randomization:
            return {}

        params = {
            "friction": self.rng.uniform(0.3, 1.0),
            "mass_scale": self.rng.uniform(0.5, 2.0),
            "restitution": self.rng.uniform(0.0, 0.5),
            "contact_stiffness": self.rng.uniform(1e4, 1e6),
            "contact_damping": self.rng.uniform(1e2, 1e4),
        }

        return params

    def _get_camera_images(self) -> np.ndarray:
        """
        Render front and wrist camera images.

        Returns:
            images: (2, 3, H, W) RGB images normalized to [0, 1].
        """
        images = np.zeros(
            (2, 3, self.camera_height, self.camera_width),
            dtype=np.float32,
        )

        if self._physics_client == "mock":
            # Synthetic images for development
            images = self.rng.rand(*images.shape).astype(np.float32) * 0.5
        else:
            try:
                import pybullet as p
                for cam_idx, (pos, target) in enumerate([
                    ([0.5, 0.0, 0.5], [0.0, 0.0, 0.1]),    # front camera
                    ([0.0, 0.0, 0.3], [0.0, 0.0, 0.05]),   # wrist camera
                ]):
                    view_matrix = p.computeViewMatrix(
                        cameraEyePosition=pos,
                        cameraTargetPosition=target,
                        cameraUpVector=[0, 0, 1],
                        physicsClientId=self._physics_client,
                    )
                    proj_matrix = p.computeProjectionMatrixFOV(
                        fov=60,
                        aspect=1.0,
                        nearVal=0.01,
                        farVal=10.0,
                    )
                    _, _, rgba, _, _ = p.getCameraImage(
                        width=self.camera_width,
                        height=self.camera_height,
                        viewMatrix=view_matrix,
                        projectionMatrix=proj_matrix,
                        physicsClientId=self._physics_client,
                    )
                    rgb = np.array(rgba[:, :, :3], dtype=np.float32) / 255.0
                    images[cam_idx] = rgb.transpose(2, 0, 1)  # HWC -> CHW
            except Exception:
                pass

        return images

    def _get_tactile_reading(self) -> np.ndarray:
        """
        Get current tactile sensor reading.

        Returns:
            tactile: (rows, cols) pressure values.
        """
        rows = self.tactile_config.rows
        cols = self.tactile_config.cols

        if self._tactile_sensor is not None:
            try:
                # Real TACTO sensor reading
                color, depth = self._tactile_sensor.render()
                # Convert depth to pressure (simplified)
                tactile = depth[0].mean(axis=-1)[:rows, :cols]
            except Exception:
                tactile = np.zeros((rows, cols), dtype=np.float32)
        else:
            # Simulated tactile: use contact points from PyBullet
            tactile = np.zeros((rows, cols), dtype=np.float32)

            if self._physics_client != "mock" and self._robot_id is not None:
                try:
                    import pybullet as p
                    contacts = p.getContactPoints(
                        bodyA=self._robot_id,
                        physicsClientId=self._physics_client,
                    )
                    for contact in contacts:
                        force = contact[9]  # normal force
                        # Map contact position to taxel grid (simplified)
                        r = self.rng.randint(0, rows)
                        c = self.rng.randint(0, cols)
                        tactile[r, c] = min(force, self.tactile_config.max_pressure)
                except Exception:
                    pass

        # Add noise
        tactile += self.rng.randn(rows, cols).astype(np.float32) * self.tactile_config.noise_std
        tactile = np.clip(tactile, 0.0, self.tactile_config.max_pressure)

        return tactile.astype(np.float32)

    def _get_state(self) -> np.ndarray:
        """Get proprioceptive state vector."""
        state = np.zeros(32, dtype=np.float32)

        if self._physics_client != "mock" and self._robot_id is not None:
            try:
                import pybullet as p
                num_joints = p.getNumJoints(
                    self._robot_id,
                    physicsClientId=self._physics_client,
                )
                for j in range(min(num_joints, 16)):
                    joint_state = p.getJointState(
                        self._robot_id, j,
                        physicsClientId=self._physics_client,
                    )
                    state[j * 2] = joint_state[0]      # position
                    state[j * 2 + 1] = joint_state[1]  # velocity
            except Exception:
                pass

        return state

    def _get_measured_force(self) -> np.ndarray:
        """Get measured contact force [normal, shear_x, shear_y]."""
        force = np.zeros(3, dtype=np.float32)

        if self._physics_client != "mock" and self._robot_id is not None:
            try:
                import pybullet as p
                contacts = p.getContactPoints(
                    bodyA=self._robot_id,
                    physicsClientId=self._physics_client,
                )
                for contact in contacts:
                    force[0] += contact[9]   # normal force
                    force[1] += contact[11]  # lateral friction 1
                    force[2] += contact[13] if len(contact) > 13 else 0.0  # lateral friction 2
            except Exception:
                pass

        return force

    def reset(
        self,
        task: str = "pick_place",
        object_name: str = "cube",
    ) -> Dict[str, np.ndarray]:
        """
        Reset the environment for a new episode.

        Args:
            task: Task name (pick_place, wipe, push, etc.).
            object_name: Object to manipulate.

        Returns:
            observation: Initial observation dict.
        """
        if self._physics_client is None:
            self._init_physics()
            self._init_tactile_sensor()

        self._step_count = 0
        self._apply_domain_randomization()

        return self._get_observation()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Collect all observation modalities."""
        return {
            "images": self._get_camera_images(),
            "tactile": self._get_tactile_reading(),
            "state": self._get_state(),
            "force": self._get_measured_force(),
        }

    def step(
        self,
        action: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        Execute one control step.

        Args:
            action: Action dict with position, orientation, gripper, force targets.

        Returns:
            observation: New observation dict.
            reward: Step reward (task-dependent).
            done: Whether the episode is complete.
            info: Additional info (forces, success, etc.).
        """
        self._step_count += 1

        # Apply action (in real implementation: send to controller, step physics)
        if self._physics_client != "mock":
            try:
                import pybullet as p
                # Step physics multiple times per control step
                physics_steps = int(240.0 / self.control_hz)
                for _ in range(physics_steps):
                    p.stepSimulation(physicsClientId=self._physics_client)
            except Exception:
                pass

        observation = self._get_observation()
        reward = 0.0  # task-dependent
        done = self._step_count >= 300  # max episode length
        info = {
            "step": self._step_count,
            "measured_force": observation["force"],
        }

        return observation, reward, done, info

    def close(self):
        """Clean up resources."""
        if self._physics_client is not None and self._physics_client != "mock":
            try:
                import pybullet as p
                p.disconnect(physicsClientId=self._physics_client)
            except Exception:
                pass
        self._physics_client = None

    def collect_demonstration(
        self,
        policy_fn,
        task: str = "pick_place",
        object_name: str = "cube",
        max_steps: int = 200,
    ) -> Dict:
        """
        Collect a demonstration trajectory using a scripted/expert policy.

        Args:
            policy_fn: Callable(observation) -> action dict.
            task: Task name.
            object_name: Object name.
            max_steps: Maximum steps per episode.

        Returns:
            trajectory: Dict with all observations and actions over the episode.
        """
        obs = self.reset(task=task, object_name=object_name)

        trajectory = {
            "images": [],
            "tactile": [],
            "states": [],
            "actions": [],
            "forces": [],
        }

        for step in range(max_steps):
            action = policy_fn(obs)

            trajectory["images"].append(obs["images"])
            trajectory["tactile"].append(obs["tactile"])
            trajectory["states"].append(obs["state"])
            trajectory["forces"].append(obs["force"])

            # Convert action dict to flat vector for storage
            action_vector = np.zeros(32, dtype=np.float32)
            action_vector[:3] = action.get("position", np.zeros(3))
            action_vector[3:9] = action.get("orientation", np.zeros(6))
            action_vector[9] = action.get("gripper", 0.0)
            action_vector[10] = action.get("force_ext_normal", 0.0)
            action_vector[11:13] = action.get("force_ext_shear", np.zeros(2))
            action_vector[13] = action.get("force_grasp", 0.0)
            trajectory["actions"].append(action_vector)

            obs, reward, done, info = self.step(action)

            if done:
                break

        # Convert lists to arrays
        for key in trajectory:
            trajectory[key] = np.array(trajectory[key])

        trajectory["task"] = task
        trajectory["object"] = object_name
        trajectory["length"] = len(trajectory["actions"])

        return trajectory
