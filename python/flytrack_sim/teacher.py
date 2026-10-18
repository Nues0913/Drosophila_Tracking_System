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
    target_velocity: np.ndarray | None = None,
    evasion_distance_m: float = 1.5,
    evasion_clearance_m: float = 0.6,
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
    return limits.clip(requested, previous_action, contract.control_dt)
