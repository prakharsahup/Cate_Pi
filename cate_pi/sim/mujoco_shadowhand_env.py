"""
MuJoCo + Shadow Dexterous Hand Environment for CATE-π (§9, Path B).

Turn-key prototyping environment using Gymnasium-Robotics.
HandManipulateBlockTouchSensors provides 92 built-in touch sensors
distributed across palm and finger phalanges.

Lower spatial fidelity than PyBullet+TACTO, but zero sensor-integration
work and a real dexterous hand model out of the box.

Recommended for FIRST pass: validate architecture and training pipeline
before investing in Path A's higher-fidelity tactile sim.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ShadowHandConfig:
    """Configuration for the Shadow Hand environment."""
    # Touch sensor layout: map 92 MuJoCo sensors to grid format
    num_touch_sensors: int = 92
    mapped_rows: int = 15   # map to CATE-π's expected grid
    mapped_cols: int = 8
    # Task
    task_name: str = "HandManipulateBlockTouchSensors-v1"
    reward_type: str = "dense"
    # Control
    control_hz: float = 25.0  # MuJoCo default
    # Rendering
    camera_width: int = 224
    camera_height: int = 224


class MuJoCoShadowHandEnv:
    """
    MuJoCo Shadow Dexterous Hand environment for CATE-π.

    Wraps Gymnasium-Robotics hand manipulation environments,
    adapting their interface to match CATE-π's expected I/O contract.
    """

    def __init__(
        self,
        config: Optional[ShadowHandConfig] = None,
        render_mode: str = "rgb_array",
        seed: int = 42,
    ):
        """
        Args:
            config: Environment configuration.
            render_mode: "rgb_array" for headless, "human" for GUI.
            seed: Random seed.
        """
        self.config = config or ShadowHandConfig()
        self.render_mode = render_mode
        self.rng = np.random.RandomState(seed)
        self._env = None
        self._step_count = 0

    def _init_gym_env(self):
        """Initialize the Gymnasium-Robotics environment."""
        try:
            import gymnasium as gym
            import gymnasium_robotics

            gym.register_envs(gymnasium_robotics)

            self._env = gym.make(
                self.config.task_name,
                render_mode=self.render_mode,
            )
            print(f"Initialized MuJoCo environment: {self.config.task_name}")

        except ImportError:
            print("gymnasium-robotics not installed. Using mock environment.")
            self._env = None

        except Exception as e:
            print(f"Failed to create environment: {e}. Using mock.")
            self._env = None

    def _map_touch_to_grid(self, touch_sensors: np.ndarray) -> np.ndarray:
        """
        Map MuJoCo's 92 touch sensor readings to a 2D grid format
        compatible with CATE-π's tactile encoder.

        The 92 sensors are distributed across:
        - 5 fingertips (distal phalanges)
        - 5 middle phalanges
        - 5 proximal phalanges
        - Palm sensors

        We map them to a (rows, cols) grid by spatial approximation.

        Args:
            touch_sensors: (92,) or (N_sensors,) raw touch readings.

        Returns:
            grid: (rows, cols) mapped tactile grid.
        """
        rows = self.config.mapped_rows
        cols = self.config.mapped_cols
        grid = np.zeros((rows, cols), dtype=np.float32)

        if touch_sensors is None or len(touch_sensors) == 0:
            return grid

        n_sensors = len(touch_sensors)

        # Simple spatial mapping: distribute sensors across grid
        # Row 0-2: fingertip sensors
        # Row 3-6: middle phalanges
        # Row 7-10: proximal phalanges
        # Row 11-14: palm sensors
        sensors_per_region = n_sensors // 4

        for i, val in enumerate(touch_sensors):
            region = i // sensors_per_region if sensors_per_region > 0 else 0
            region = min(region, 3)
            r_start = region * (rows // 4)
            r = r_start + (i % (rows // 4))
            c = i % cols
            if r < rows and c < cols:
                grid[r, c] = val

        return grid

    def _get_camera_images(self) -> np.ndarray:
        """
        Render camera images from MuJoCo.

        Returns:
            images: (2, 3, H, W) front + wrist camera images.
        """
        images = np.zeros(
            (2, 3, self.config.camera_height, self.config.camera_width),
            dtype=np.float32,
        )

        if self._env is not None:
            try:
                # Render from the environment
                rgb = self._env.render()
                if rgb is not None:
                    from PIL import Image as PILImage
                    img = PILImage.fromarray(rgb).resize(
                        (self.config.camera_width, self.config.camera_height)
                    )
                    img_np = np.array(img, dtype=np.float32) / 255.0
                    images[0] = img_np[:, :, :3].transpose(2, 0, 1)
                    images[1] = images[0]  # duplicate as wrist cam placeholder
            except Exception:
                pass
        else:
            images = self.rng.rand(*images.shape).astype(np.float32) * 0.3

        return images

    def reset(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Reset the environment.

        Returns:
            observation: Initial observation dict.
        """
        if self._env is None:
            self._init_gym_env()

        self._step_count = 0

        if self._env is not None:
            obs, info = self._env.reset()
            return self._convert_observation(obs)
        else:
            return self._get_mock_observation()

    def _convert_observation(self, gym_obs: Dict) -> Dict[str, np.ndarray]:
        """
        Convert Gymnasium observation dict to CATE-π format.

        Gymnasium-Robotics hand envs provide:
        - observation: robot state (joint positions, velocities, etc.)
        - achieved_goal: current object pose
        - desired_goal: target object pose
        - (with TouchSensors): touch sensor readings
        """
        # Extract touch sensors if available
        obs_flat = gym_obs.get("observation", np.zeros(63))
        touch_raw = None

        # Touch sensor variants append sensor readings to observation
        if len(obs_flat) > 63:
            touch_raw = obs_flat[63:]  # remaining entries are touch sensors
            state_raw = obs_flat[:63]
        else:
            state_raw = obs_flat

        # Map touch to grid
        tactile_grid = self._map_touch_to_grid(
            touch_raw if touch_raw is not None else np.array([])
        )

        # Build state vector (pad to 32)
        state = np.zeros(32, dtype=np.float32)
        state[:min(len(state_raw), 32)] = state_raw[:32]

        # Approximate force from touch sensor sum
        force = np.zeros(3, dtype=np.float32)
        if touch_raw is not None:
            force[0] = np.sum(touch_raw)  # approximate normal force
            force[1] = np.std(touch_raw)  # approximate shear (proxy)

        return {
            "images": self._get_camera_images(),
            "tactile": tactile_grid,
            "state": state,
            "force": force,
        }

    def _get_mock_observation(self) -> Dict[str, np.ndarray]:
        """Generate mock observation for development."""
        return {
            "images": self.rng.rand(2, 3, self.config.camera_height,
                                    self.config.camera_width).astype(np.float32),
            "tactile": (self.rng.rand(self.config.mapped_rows,
                                      self.config.mapped_cols) * 0.1).astype(np.float32),
            "state": self.rng.randn(32).astype(np.float32) * 0.1,
            "force": np.zeros(3, dtype=np.float32),
        }

    def step(
        self,
        action: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute one control step.

        Args:
            action: CATE-π action dict.

        Returns:
            observation, reward, terminated, truncated, info
        """
        self._step_count += 1

        if self._env is not None:
            # Convert CATE-π action to Gymnasium format
            # Shadow hand expects 20 joint position targets
            gym_action = np.zeros(20, dtype=np.float32)

            # Map position + gripper to joint targets (simplified)
            pos = action.get("position", np.zeros(3))
            gripper = action.get("gripper", np.array([0.0]))

            # Simple mapping: scale position to joint range
            gym_action[:3] = pos * 0.1  # wrist joints
            gym_action[3:] = float(gripper[0]) * 0.5  # finger joints

            gym_obs, reward, terminated, truncated, info = self._env.step(gym_action)
            obs = self._convert_observation(gym_obs)

            return obs, reward, terminated, truncated, info
        else:
            obs = self._get_mock_observation()
            done = self._step_count >= 200
            return obs, 0.0, done, False, {"step": self._step_count}

    def close(self):
        """Clean up."""
        if self._env is not None:
            self._env.close()

    def collect_demonstration(
        self,
        policy_fn,
        max_steps: int = 200,
    ) -> Dict:
        """
        Collect a demonstration trajectory.

        Args:
            policy_fn: Callable(observation) -> action dict.
            max_steps: Maximum episode length.

        Returns:
            trajectory: Dict with observations and actions.
        """
        obs = self.reset()

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

            # Flatten action
            action_vector = np.zeros(32, dtype=np.float32)
            action_vector[:3] = action.get("position", np.zeros(3))
            action_vector[3:9] = action.get("orientation", np.zeros(6))
            action_vector[9] = action.get("gripper", np.array([0.0]))[0]
            action_vector[10] = action.get("force_ext_normal", np.array([0.0]))[0]
            action_vector[11:13] = action.get("force_ext_shear", np.zeros(2))
            action_vector[13] = action.get("force_grasp", np.array([0.0]))[0]
            trajectory["actions"].append(action_vector)

            obs, reward, terminated, truncated, info = self.step(action)
            if terminated or truncated:
                break

        for key in trajectory:
            trajectory[key] = np.array(trajectory[key])

        trajectory["length"] = len(trajectory["actions"])
        return trajectory
