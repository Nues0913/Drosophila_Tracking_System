"""v2 sensor contract, observation layout and success rule (plan section 5.3, frozen at M0).

Changing anything in this module after M0 needs a PM decision and both reviewers. Shape checks
are free; value checks synchronise with the GPU, so hot loops call validate_obs with
check_values=False and tests or debugging turn the value checks on.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from flytrack_core.contracts import SensorContract, canonical_json
from torch import Tensor

from flytrack_v2 import SCHEMA_VERSION, SPEC_VERSION

# The policy sees exactly these keys. Anything else (truth, teacher labels) is refused.
OBS_KEYS = ("vision", "odor", "wind", "valid")
# valid[:, i] is 1 when channel i holds a fresh, usable reading.
VALID_CHANNELS = ("left_eye", "right_eye", "odor", "wind")
# world.privileged() keys: teacher, critic and evaluation only, never the actor.
PRIVILEGED_KEYS = (
    "agent_pose",  # [B,3] x, y (m), yaw (rad)
    "agent_velocity",  # [B,2] applied v (m/s), omega (rad/s)
    "target_pose",  # [B,3] x, y (m), heading (rad)
    "target_velocity",  # [B,2] vx, vy (m/s), world frame
    "occluded",  # [B] bool, scenario occlusion active
    "target_visible",  # [B] bool, inside either eye's field of view and not occluded
    "wind_velocity",  # [B,2] vx, vy (m/s), world frame; zeros in the v1compat world
    "decoy_active",  # [B] bool, conflict scenario decoy present
    "decoy_position",  # [B,2] x, y (m); zeros when inactive
    "time_s",  # [B] episode time (s)
    "episode_seed",  # [B] int64
)
WORLDS = ("v1compat", "wind")
SCENARIOS = ("normal", "occlusion", "noise", "conflict")
MOTIONS = ("curved", "wander")


@dataclass(frozen=True)
class SuccessRule:
    """v1's tracking success (protocol schema 3), kept unchanged so G3 stays comparable."""

    distance_m: float = 2.0
    bearing_deg: float = 45.0
    fraction: float = 0.8
    collision_invalidates: bool = True


SUCCESS_RULE = SuccessRule()


@dataclass(frozen=True)
class SensorContractV2(SensorContract):
    """v1's camera, eye, timing and action limits plus the two-antenna wind sense.

    Antenna model (WP2b implements it in the wind world): with theta the direction the wind
    comes from, relative to the heading (left positive), and alpha = antenna_angle_deg,
    left = (U / wind_reference_m_s) * cos(theta - alpha) and
    right = (U / wind_reference_m_s) * cos(theta + alpha), plus the world's sensor noise.
    left - right is proportional to sin(theta): linear in azimuth near the midline.
    """

    wind_hz: float = 50.0
    antenna_angle_deg: float = 45.0
    wind_reference_m_s: float = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0 < self.antenna_angle_deg < 90:
            raise ValueError("antenna_angle_deg must be between 0 and 90 degrees")

    def sha256(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "spec_version": SPEC_VERSION,
            "hex_geometry_version": 1,
            **self.to_dict(),
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def obs_shapes(contract: SensorContractV2, batch: int) -> dict[str, tuple[int, ...]]:
    return {
        "vision": (batch, 2, contract.n_ommatidia),
        "odor": (batch, 2),
        "wind": (batch, 2),
        "valid": (batch, len(VALID_CHANNELS)),
    }


def validate_obs(
    obs: Mapping[str, Tensor], contract: SensorContractV2, check_values: bool = False
) -> int:
    """Check keys, dtypes, devices and shapes; return the batch size.

    check_values also checks ranges and finiteness, which needs a host sync: keep it out of
    training loops (plan section 0.2).
    """
    keys = set(obs)
    if keys != set(OBS_KEYS):
        leaked = sorted(keys & set(PRIVILEGED_KEYS))
        if leaked:
            raise ValueError(f"Privileged keys must never reach the policy: {leaked}")
        raise ValueError(f"Observation keys must be exactly {OBS_KEYS}, got {sorted(keys)}")
    tensors = [obs[name] for name in OBS_KEYS]
    if not all(isinstance(value, Tensor) for value in tensors):
        raise TypeError("Observation values must be torch tensors")
    if any(value.dtype != torch.float32 for value in tensors):
        raise TypeError("Observation tensors must be float32")
    if len({value.device for value in tensors}) != 1:
        raise ValueError("Observation tensors must share one device")
    batch = int(obs["vision"].shape[0]) if obs["vision"].ndim == 3 else -1
    for name, shape in obs_shapes(contract, batch).items():
        if tuple(obs[name].shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(obs[name].shape)}")
    if check_values:
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("Observation tensors must be finite; mark missing data in valid")
        vision, odor, valid = obs["vision"], obs["odor"], obs["valid"]
        if bool(((vision < 0) | (vision > 1)).any()) or bool((odor < 0).any()):
            raise ValueError("Vision must be in [0, 1] and odor nonnegative")
        if bool(((valid != 0) & (valid != 1)).any()):
            raise ValueError("valid must be binary")
    return batch
