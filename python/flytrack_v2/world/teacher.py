"""v1's privileged teacher, batched (#132, interfaces.Teacher).

The same rule as ``flytrack_sim.teacher.teacher_action``, on [B] tensors:

- forward speed gain * (distance - following distance), only when the target is ahead
  (times max(0, cos bearing)), never reversing;
- turn gain * bearing;
- a target that turns back into the stand-off is evaded sideways (v1's evasion rule);
- optionally (v1's #33 option, ``occlusion_odor_gain``), during occlusion the turn comes from
  the student's own raw odor reading, never from plume truth;
- the label is clipped to the contract's speed and acceleration limits around the previous
  applied action, as v1 does.

The teacher keeps no state (width 0). Labels are training targets, never observations.
"""

from __future__ import annotations

import math

import torch
from flytrack_core.contracts import ActionLimits, SensorContract
from torch import Tensor


def _wrap(angle: Tensor) -> Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


class PrivilegedTeacher:
    def __init__(
        self,
        contract: SensorContract,
        following_distance_m: float = 1.0,
        turn_gain: float = 2.0,
        distance_gain: float = 0.7,
        evasion_distance_m: float = 1.5,
        evasion_clearance_m: float = 0.6,
        occlusion_odor_gain: float | None = None,
    ):
        if occlusion_odor_gain is not None and not (
            math.isfinite(occlusion_odor_gain) and occlusion_odor_gain >= 0
        ):
            raise ValueError("occlusion odor gain must be finite and nonnegative")
        self.contract = contract
        self.limits = ActionLimits.from_contract(contract)
        self.following = following_distance_m
        self.turn_gain = turn_gain
        self.distance_gain = distance_gain
        self.evasion_distance = evasion_distance_m
        self.evasion_clearance = evasion_clearance_m
        self.odor_gain = occlusion_odor_gain

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(batch_size, 0, device=device)

    def act(
        self,
        privileged: dict[str, Tensor],
        obs: dict[str, Tensor],
        previous_action: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        f64 = torch.float64
        agent = privileged["agent_pose"].to(f64)
        target = privileged["target_pose"].to(f64)
        velocity = privileged["target_velocity"].to(f64)
        displacement = target[:, :2] - agent[:, :2]
        distance = displacement.norm(dim=-1)
        bearing = _wrap(torch.atan2(displacement[:, 1], displacement[:, 0]) - agent[:, 2])
        forward = (self.distance_gain * (distance - self.following)).clamp(min=0)
        speed_cmd = forward * torch.cos(bearing).clamp(min=0)
        turn_cmd = self.turn_gain * bearing
        # Evasion: the target heads back at the agent inside the stand-off.
        speed = velocity.norm(dim=-1)
        heading = velocity / torch.where(speed > 0, speed, 1.0)[:, None]
        ahead = -(displacement * heading).sum(-1)
        lateral = displacement[:, 0] * heading[:, 1] - displacement[:, 1] * heading[:, 0]
        evade = (
            (speed > 0)
            & (distance < self.evasion_distance)
            & (ahead > 0)
            & (lateral.abs() < self.evasion_clearance)
        )
        side = torch.where(lateral >= 0, 1.0, -1.0).to(f64)
        escape = torch.atan2(heading[:, 1], heading[:, 0]) + side * math.pi / 2
        turn = _wrap(escape - agent[:, 2])
        speed_cmd = torch.where(evade, self.limits.v_max * torch.cos(turn).clamp(min=0), speed_cmd)
        turn_cmd = torch.where(evade, self.turn_gain * turn, turn_cmd)
        # v1 builds the request as float32 before the odor override and the limits.
        requested = torch.stack((speed_cmd, turn_cmd), -1).to(torch.float32)
        if self.odor_gain is not None:
            omega = self._odor_turn(obs).clamp(-self.limits.omega_max, self.limits.omega_max)
            occluded = privileged["occluded"].to(torch.bool)
            requested[:, 1] = torch.where(occluded, omega.to(torch.float32), requested[:, 1])
        return self._clip(requested, previous_action), state

    def _odor_turn(self, obs: dict[str, Tensor]) -> Tensor:
        """gain * (L - R) / (L + R) from the student's raw odor; 0 without a valid reading."""
        odor = obs["odor"].to(torch.float64).clamp(min=0)
        valid = obs["valid"][:, 2].to(torch.float64)
        scale = odor.max(-1).values
        usable = torch.isfinite(valid) & (valid > 0) & torch.isfinite(odor).all(-1) & (scale > 0)
        left, right = (odor / torch.where(scale > 0, scale, 1.0)[:, None]).unbind(-1)
        turn = self.odor_gain * (left - right) / torch.where(usable, left + right, 1.0)
        return torch.where(usable, turn, 0.0)

    def _clip(self, requested: Tensor, previous: Tensor) -> Tensor:
        """ActionLimits.clip (bounds in float32, the acceleration limit in float64)."""
        lim, dt = self.limits, self.contract.control_dt
        device = requested.device
        low = torch.tensor([0.0, -lim.omega_max], device=device)
        high = torch.tensor([lim.v_max, lim.omega_max], device=device)
        delta = dt * torch.tensor(
            [lim.linear_accel_max, lim.angular_accel_max], dtype=torch.float64, device=device
        )
        value = torch.minimum(torch.maximum(requested, low), high).to(torch.float64)
        previous64 = previous.to(torch.float64)
        value = torch.minimum(torch.maximum(value, previous64 - delta), previous64 + delta)
        return torch.minimum(torch.maximum(value, low.double()), high.double()).to(torch.float32)
