"""
Hybrid Position-Force Controller for CATE-π (§7).

This is a CLASSICAL controller, NOT a learned module. It converts the
action expert's (P_target, F_target) output into actual joint commands.

Core equation:
    ΔF = F_target - F_measured
    P_hybrid = P_target + K·ΔF    if ||ΔF|| > τ_threshold
             = P_target            otherwise

Key design points:
- K is a gain matrix (tune per-axis; conservative in sim to avoid oscillation)
- τ_threshold prevents chattering from measurement noise
- PID loop drives joints to P_hybrid
- CRITICAL (§7): Decouple TWO force channels:
  1. End-effector position → regulates NET EXTERNAL force (pressing on environment)
  2. Gripper width / finger closure → regulates INTERNAL GRASP force (squeezing object)
  These are physically distinct and must NOT be conflated.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PIDState:
    """State for a PID controller."""
    integral: np.ndarray = field(default_factory=lambda: np.zeros(3))
    prev_error: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dt: float = 0.01


class PIDController:
    """Simple PID controller for joint-level control."""

    def __init__(
        self,
        kp: np.ndarray,
        ki: np.ndarray,
        kd: np.ndarray,
        integral_limit: float = 1.0,
    ):
        """
        Args:
            kp: Proportional gains, shape (N_dof,).
            ki: Integral gains, shape (N_dof,).
            kd: Derivative gains, shape (N_dof,).
            integral_limit: Anti-windup clamp for integral term.
        """
        self.kp = np.asarray(kp, dtype=np.float64)
        self.ki = np.asarray(ki, dtype=np.float64)
        self.kd = np.asarray(kd, dtype=np.float64)
        self.integral_limit = integral_limit

        self._integral = np.zeros_like(self.kp)
        self._prev_error = np.zeros_like(self.kp)

    def reset(self):
        """Reset PID state."""
        self._integral = np.zeros_like(self.kp)
        self._prev_error = np.zeros_like(self.kp)

    def compute(
        self,
        error: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Compute PID control output.

        Args:
            error: (N_dof,) current error (target - measured).
            dt: Timestep.

        Returns:
            output: (N_dof,) control command.
        """
        # Proportional
        p_term = self.kp * error

        # Integral (with anti-windup)
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self._integral

        # Derivative
        d_term = self.kd * (error - self._prev_error) / max(dt, 1e-8)
        self._prev_error = error.copy()

        return p_term + i_term + d_term


class HybridPositionForceController:
    """
    Hybrid Position-Force Controller (§7).

    Converts the action expert's (P_target, F_target) outputs into
    joint-level commands by:
    1. Computing force error ΔF = F_target - F_measured
    2. Adjusting position target: P_hybrid = P_target + K·ΔF (if above threshold)
    3. PID control to drive joints to P_hybrid

    Two DECOUPLED force channels:
    - External force: controlled via end-effector Cartesian position
    - Grasp force: controlled via gripper width / finger closure
    """

    def __init__(
        self,
        # Force correction gain matrix (3×3 for 3D position adjustment)
        force_gain: np.ndarray = None,
        # Chattering prevention threshold
        force_threshold: float = 0.1,
        # Separate grasp force gain (scalar for parallel jaw)
        grasp_force_gain: float = 0.01,
        grasp_force_threshold: float = 0.05,
        # PID for Cartesian position
        position_pid_kp: np.ndarray = None,
        position_pid_ki: np.ndarray = None,
        position_pid_kd: np.ndarray = None,
        # PID for gripper
        gripper_pid_kp: float = 10.0,
        gripper_pid_ki: float = 0.5,
        gripper_pid_kd: float = 1.0,
        # Control rate
        control_dt: float = 1.0 / 15.0,
        # Safety limits
        max_position_delta: float = 0.05,  # meters per step
        max_force_correction: float = 0.02,  # meters of force-induced position shift
    ):
        """
        Args:
            force_gain: K matrix (3×3) for force→position correction.
            force_threshold: τ_threshold — minimum ||ΔF|| to apply correction.
            grasp_force_gain: Gain for grasp force → gripper width adjustment.
            grasp_force_threshold: Min grasp ΔF to apply correction.
            position_pid_kp/ki/kd: PID gains for Cartesian position control.
            gripper_pid_kp/ki/kd: PID gains for gripper control.
            control_dt: Control loop timestep.
            max_position_delta: Safety clamp on position change per step.
            max_force_correction: Safety clamp on force-induced correction.
        """
        # Force correction gains
        if force_gain is None:
            # Conservative diagonal gain (tune per-axis)
            force_gain = np.diag([0.005, 0.005, 0.01])  # z-axis more responsive
        self.force_gain = np.asarray(force_gain, dtype=np.float64)
        self.force_threshold = force_threshold
        self.grasp_force_gain = grasp_force_gain
        self.grasp_force_threshold = grasp_force_threshold
        self.max_force_correction = max_force_correction
        self.max_position_delta = max_position_delta
        self.control_dt = control_dt

        # PID controllers
        if position_pid_kp is None:
            position_pid_kp = np.array([50.0, 50.0, 50.0])
        if position_pid_ki is None:
            position_pid_ki = np.array([5.0, 5.0, 5.0])
        if position_pid_kd is None:
            position_pid_kd = np.array([10.0, 10.0, 10.0])

        self.position_pid = PIDController(
            kp=position_pid_kp,
            ki=position_pid_ki,
            kd=position_pid_kd,
        )

        self.gripper_pid = PIDController(
            kp=np.array([gripper_pid_kp]),
            ki=np.array([gripper_pid_ki]),
            kd=np.array([gripper_pid_kd]),
        )

    def reset(self):
        """Reset controller state for a new episode."""
        self.position_pid.reset()
        self.gripper_pid.reset()

    def compute_hybrid_target(
        self,
        position_target: np.ndarray,
        force_target: np.ndarray,
        force_measured: np.ndarray,
    ) -> np.ndarray:
        """
        Compute hybrid position target with force correction.

        P_hybrid = P_target + K·ΔF  if ||ΔF|| > threshold
                 = P_target          otherwise

        This handles the EXTERNAL force channel (end-effector → environment).

        Args:
            position_target: (3,) target Cartesian position from action expert.
            force_target: (3,) target contact force [F_normal, F_shear_x, F_shear_y].
            force_measured: (3,) measured contact force from sensor/sim.

        Returns:
            hybrid_target: (3,) corrected position target.
        """
        delta_f = force_target - force_measured  # (3,)

        if np.linalg.norm(delta_f) > self.force_threshold:
            # Apply force correction
            correction = self.force_gain @ delta_f

            # Safety clamp
            correction_norm = np.linalg.norm(correction)
            if correction_norm > self.max_force_correction:
                correction = correction * (self.max_force_correction / correction_norm)

            hybrid_target = position_target + correction
        else:
            hybrid_target = position_target

        return hybrid_target

    def compute_grasp_target(
        self,
        gripper_target: float,
        grasp_force_target: float,
        grasp_force_measured: float,
    ) -> float:
        """
        Compute gripper width target with grasp force correction.

        This handles the INTERNAL GRASP force channel (gripper → object).
        Decoupled from external force per §7's critical design note.

        Args:
            gripper_target: Target gripper width from action expert.
            grasp_force_target: Target internal grasp force.
            grasp_force_measured: Measured internal grasp force.

        Returns:
            adjusted_gripper: Force-corrected gripper width target.
        """
        delta_grasp = grasp_force_target - grasp_force_measured

        if abs(delta_grasp) > self.grasp_force_threshold:
            # Positive delta_grasp → need more force → close gripper more (decrease width)
            correction = -self.grasp_force_gain * delta_grasp
            adjusted = gripper_target + correction
            adjusted = np.clip(adjusted, 0.0, 1.0)  # gripper width ∈ [0, 1]
        else:
            adjusted = gripper_target

        return float(adjusted)

    def step(
        self,
        action: Dict[str, np.ndarray],
        measured_position: np.ndarray,
        measured_force: np.ndarray,
        measured_gripper: float,
        measured_grasp_force: float,
    ) -> Dict[str, np.ndarray]:
        """
        Execute one control step.

        Takes the action expert's output and measured state, returns
        joint-level commands.

        Args:
            action: Dict from action_expert.split_action_vector():
                - "position": (3,) Δx, Δy, Δz
                - "orientation": (6,) 6D rotation
                - "gripper": (1,) gripper width
                - "force_ext_normal": (1,) target external contact normal force
                - "force_ext_shear": (2,) target external shear force (x, y)
                - "force_grasp": (1,) target internal grasp force
            measured_position: (3,) current end-effector position.
            measured_force: (3,) measured force [normal, shear_x, shear_y].
            measured_gripper: Current gripper width.
            measured_grasp_force: Measured internal grasp force.

        Returns:
            command: Dict with:
                - "position": (3,) target position for position controller
                - "orientation": (6,) target orientation (pass-through)
                - "gripper": (1,) target gripper width
                - "force_correction_applied": bool
                - "grasp_correction_applied": bool
        """
        # Parse action vector
        pos_target = np.asarray(action["position"]).flatten()[:3]
        orientation = np.asarray(action["orientation"]).flatten()
        gripper_target = float(np.asarray(action["gripper"]).flatten()[0])
        force_ext_normal = float(np.asarray(action["force_ext_normal"]).flatten()[0])
        force_ext_shear = np.asarray(action["force_ext_shear"]).flatten()[:2]
        force_grasp = float(np.asarray(action["force_grasp"]).flatten()[0])

        # Construct EXTERNAL force target vector: [normal, shear_x, shear_y]
        # This is the force the end-effector exerts on the environment.
        force_ext_target = np.array([force_ext_normal, force_ext_shear[0], force_ext_shear[1]])

        # === External force channel ===
        # Hybrid position target with external force correction
        hybrid_pos = self.compute_hybrid_target(
            position_target=pos_target,
            force_target=force_ext_target,
            force_measured=measured_force,
        )

        # Safety clamp on position delta
        pos_delta = hybrid_pos - measured_position
        delta_norm = np.linalg.norm(pos_delta)
        if delta_norm > self.max_position_delta:
            pos_delta = pos_delta * (self.max_position_delta / delta_norm)
            hybrid_pos = measured_position + pos_delta

        # PID for position
        pos_error = hybrid_pos - measured_position
        pos_command = self.position_pid.compute(pos_error, self.control_dt)

        # === Internal grasp force channel (DECOUPLED per §7) ===
        # force_grasp is a SEPARATE, independently-predicted channel from the
        # action expert — NOT a reuse of force_ext_normal. This is the force
        # the fingers exert on the grasped object, physically distinct from
        # the force the end-effector exerts on the environment.
        adjusted_gripper = self.compute_grasp_target(
            gripper_target=gripper_target,
            grasp_force_target=force_grasp,
            grasp_force_measured=measured_grasp_force,
        )

        # PID for gripper
        gripper_error = np.array([adjusted_gripper - measured_gripper])
        gripper_command = self.gripper_pid.compute(gripper_error, self.control_dt)

        return {
            "position_command": pos_command,
            "position_target": hybrid_pos,
            "orientation": orientation,
            "gripper_command": float(gripper_command[0]),
            "gripper_target": adjusted_gripper,
            "force_correction_applied": float(np.linalg.norm(force_ext_target - measured_force)) > self.force_threshold,
            "grasp_correction_applied": abs(force_grasp - measured_grasp_force) > self.grasp_force_threshold,
        }
