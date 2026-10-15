"""Privileged teacher. Its output is an info label, never an observation."""

import numpy as np
from flytrack_core.contracts import ActionLimits, SensorContract
from flytrack_core.world.trajectories import wrap_angle


def teacher_action(
    agent_pose: np.ndarray,
    target_pose: np.ndarray,
    previous_action: np.ndarray,
    contract: SensorContract,
    following_distance_m: float = 1.0,
    turn_gain: float = 2.0,
    distance_gain: float = 0.7,
) -> np.ndarray:
    displacement = np.asarray(target_pose[:2]) - np.asarray(agent_pose[:2])
    distance = np.linalg.norm(displacement)
    bearing = float(wrap_angle(np.arctan2(displacement[1], displacement[0]) - agent_pose[2]))
    # No reverse command: slow down while turning and keep a safe stand-off.
    velocity = max(0, distance_gain * (distance - following_distance_m)) * max(0, np.cos(bearing))
    requested = np.asarray([velocity, turn_gain * bearing], dtype=np.float32)
    return ActionLimits.from_contract(contract).clip(
        requested, previous_action, contract.control_dt
    )
