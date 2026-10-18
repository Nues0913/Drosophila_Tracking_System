"""Privileged teacher. Its output is an info label, never an observation."""

import math

import numpy as np
from flytrack_core.contracts import ActionLimits, SensorContract
from flytrack_core.world.trajectories import wrap_angle


def _odor_turn(observation: dict[str, np.ndarray], gain: float) -> float:
    """Use the student's latest, corrupted raw ORN input, never plume truth."""
    odor = np.asarray(observation["odor"], dtype=np.float64)
    valid = np.asarray(observation["valid"])
    if odor.shape != (2,) or valid.shape != (3,):
        raise ValueError("odor steering requires odor[2] and valid[3]")
    if not np.isfinite(valid[2]) or valid[2] <= 0 or not np.isfinite(odor).all():
        return 0.0
    # Match the nonnegative concentration contract. Scaling avoids overflow and
    # does not impose a concentration threshold that would hide faint cues.
    odor = np.maximum(odor, 0.0)
    scale = float(odor.max())
    if scale == 0:
        return 0.0
    left, right = odor / scale
    return float(gain * (left - right) / (left + right))


def teacher_action(
    agent_pose: np.ndarray,
    target_pose: np.ndarray,
    previous_action: np.ndarray,
    contract: SensorContract,
    following_distance_m: float = 1.0,
    turn_gain: float = 2.0,
    distance_gain: float = 0.7,
    target_velocity: np.ndarray | None = None,
    evasion_distance_m: float = 1.5,
    evasion_clearance_m: float = 0.6,
    *,
    occluded: bool = False,
    observation: dict[str, np.ndarray] | None = None,
    occlusion_odor_gain: float | None = None,
) -> np.ndarray:
    limits = ActionLimits.from_contract(contract)
    displacement = np.asarray(target_pose[:2]) - np.asarray(agent_pose[:2])
    distance = np.linalg.norm(displacement)
    bearing = float(wrap_angle(np.arctan2(displacement[1], displacement[0]) - agent_pose[2]))
    # No reverse command: slow down while turning and keep a safe stand-off.
    velocity = max(0, distance_gain * (distance - following_distance_m)) * max(0, np.cos(bearing))
    requested = np.asarray([velocity, turn_gain * bearing], dtype=np.float32)
    speed = 0.0 if target_velocity is None else float(np.linalg.norm(target_velocity[:2]))
    if speed > 0 and distance < evasion_distance_m:
        # A target that turns back drives into the stand-off, and the agent cannot
        # reverse: step sideways off its path until it would pass at the clearance.
        heading = np.asarray(target_velocity[:2]) / speed
        ahead = float(-displacement @ heading)
        lateral = float(displacement[0] * heading[1] - displacement[1] * heading[0])  # left > 0
        if ahead > 0 and abs(lateral) < evasion_clearance_m:
            side = 1.0 if lateral >= 0 else -1.0
            escape = np.arctan2(heading[1], heading[0]) + side * np.pi / 2
            turn = float(wrap_angle(escape - agent_pose[2]))
            requested = np.asarray(
                [limits.v_max * max(0, np.cos(turn)), turn_gain * turn], dtype=np.float32
            )
    if occluded and occlusion_odor_gain is not None:
        if (
            isinstance(occlusion_odor_gain, (str, bool))
            or not math.isfinite(occlusion_odor_gain)
            or occlusion_odor_gain < 0
        ):
            raise ValueError("occlusion odor gain must be finite and nonnegative")
        if observation is None:
            raise ValueError("occlusion odor steering requires the student's observation")
        omega = _odor_turn(observation, occlusion_odor_gain)
        # Preserve the privileged forward-speed rule, but replace requested turn
        # before applying the normal velocity and acceleration limits once.
        requested[1] = np.clip(omega, -limits.omega_max, limits.omega_max)
    return limits.clip(requested, previous_action, contract.control_dt)
