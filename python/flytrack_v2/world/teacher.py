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


class OcclusionTeacher:
    """DAgger labels with an observation-only rule during occlusion (#133; #161 D4).

    Outside occlusion: the privileged teacher. While the scenario occludes the target the
    label comes from the student's own observation and the teacher's state only (never the
    target's hidden position):

    - odor present (valid and above ``detect``): surge upwind (the antenna's wind angle,
      ``upwind_gain``) plus the bilateral odor turn ``odor_gain`` * (L - R) / (L + R);
    - odor lost: with ``memory``, keep the turn rate applied just before the occlusion (dead
      reckoning); otherwise cast crosswind (``cast_gain``), switching sides every
      ``cast_period_s``;
    - without a valid wind reading (the v1compat world) the upwind and cast terms are 0;
    - forward speed: the applied speed held from just before the occlusion, times
      ``lost_speed_scale`` while the odor is lost and scaled down linearly to 0 as the mean
      odor approaches ``near`` (the source is close: slow down rather than hit it).

    With ``walls``, a look-ahead point that leaves the arena's safe box turns the agent toward
    the centre (the agent's own pose: self-knowledge, not the target).

    State [B,4]: held speed, held turn rate, seconds since odor was last present, cast side.
    The occlusion flag is the scenario's timing, a label-side switch, never an observation.
    """

    def __init__(
        self,
        privileged: PrivilegedTeacher,
        upwind_gain: float = 0.5,
        odor_gain: float = 4.0,
        cast_gain: float = 1.5,
        cast_period_s: float = 1.0,
        detect: float = 1e-3,
        lost_speed_scale: float = 1.0,
        memory: bool = True,
        near: float = 2.0,
        walls: tuple[float, float, float, float] | None = None,
        wall_lookahead_m: float = 0.8,
        wall_margin_m: float = 0.5,
        wall_gain: float = 2.0,
    ):
        """Defaults: tuned on seeds 1,630,000,000+ (not the audit's) in the wind world's
        occlusion scenario, U = 0.5 m/s, 100 episodes per motion (#133)."""
        self.privileged = privileged
        self.contract = privileged.contract
        self.upwind_gain, self.odor_gain, self.cast_gain = upwind_gain, odor_gain, cast_gain
        self.cast_period_s, self.detect = cast_period_s, detect
        self.lost_speed_scale, self.memory, self.near = lost_speed_scale, memory, near
        self.walls, self.wall_lookahead_m = walls, wall_lookahead_m
        self.wall_margin_m, self.wall_gain = wall_margin_m, wall_gain

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        state = torch.zeros(batch_size, 4, device=device)
        state[:, 3] = 1.0
        return state

    def act(
        self,
        privileged: dict[str, Tensor],
        obs: dict[str, Tensor],
        previous_action: Tensor,
        state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        from flytrack_v2.world.wind import upwind_angle

        label, _ = self.privileged.act(privileged, obs, previous_action, state[:, :0])
        occluded = privileged["occluded"].to(torch.bool)
        dt = self.contract.control_dt
        f64 = torch.float64
        held_v, held_w = state[:, 0].to(f64), state[:, 1].to(f64)
        lost, side = state[:, 2].to(f64), state[:, 3].to(f64)
        held_v = torch.where(occluded, held_v, previous_action[:, 0].to(f64))
        held_w = torch.where(occluded, held_w, previous_action[:, 1].to(f64))
        odor = obs["odor"].to(f64).clamp(min=0)
        present = (obs["valid"][:, 2] > 0) & (odor.max(-1).values > self.detect)
        left, right = odor.unbind(-1)
        bilateral = torch.where(present, (left - right) / (left + right).clamp(min=1e-12), 0.0)
        wind_valid = obs["valid"][:, 3] > 0
        upwind = torch.where(
            wind_valid, upwind_angle(obs["wind"].to(f64), self.contract.antenna_angle_deg), 0.0
        )
        lost = torch.where(present | ~occluded, 0.0, lost + dt)
        flip = (lost > 0) & (
            torch.floor(lost / self.cast_period_s)
            != torch.floor((lost - dt).clamp(min=0) / self.cast_period_s)
        )
        side = torch.where(occluded & flip, -side, side)
        if self.memory:
            lost_turn = held_w
        else:
            cast = self.cast_gain * _wrap(upwind + side * math.pi / 2)
            lost_turn = torch.where(wind_valid, cast, 0.0)
        found_turn = torch.where(wind_valid, self.upwind_gain * upwind, 0.0)
        turn = torch.where(present, found_turn + self.odor_gain * bilateral, lost_turn)
        if self.walls is not None:
            # The agent's own pose only (never the target's): if a point ahead leaves the
            # safe box, turn toward the arena centre.
            pose = privileged["agent_pose"].to(f64)
            ahead = pose[:, :2] + self.wall_lookahead_m * torch.stack(
                (torch.cos(pose[:, 2]), torch.sin(pose[:, 2])), -1
            )
            xmin, ymin, xmax, ymax = self.walls
            m = self.wall_margin_m
            out = (
                (ahead[:, 0] < xmin + m) | (ahead[:, 0] > xmax - m)
                | (ahead[:, 1] < ymin + m) | (ahead[:, 1] > ymax - m)
            )  # fmt: skip
            centre = torch.atan2((ymin + ymax) / 2 - pose[:, 1], (xmin + xmax) / 2 - pose[:, 0])
            turn = torch.where(out, self.wall_gain * _wrap(centre - pose[:, 2]), turn)
        # A strong odor means the source is close: slow down rather than run into it.
        strength = odor.mean(-1)
        approach = held_v * (1 - strength / self.near).clamp(0, 1)
        speed = torch.where(present, approach, self.lost_speed_scale * held_v)
        requested = torch.stack((speed, turn), -1).to(torch.float32)
        odor_label = self.privileged._clip(requested, previous_action)
        action = torch.where(occluded[:, None], odor_label, label)
        state = torch.stack((held_v, held_w, lost, side), -1).to(state.dtype)
        return action, state
