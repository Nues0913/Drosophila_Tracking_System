"""#138 exploratory candidate (expert proposal 11-21): VO2 plus an observation-triggered
odor/upwind controller.

The wrapped VO2 runs every step (its recurrent state keeps updating). The controller takes over
only when the eyes have lost the target and no wall is ahead; it reads only the student's own
observation (both eyes, both antennae, the wind sense) and an efference copy of its own commands.
It never reads the occlusion clock, a pose, the target's bearing or the teacher.

- visible: the mean of the 3 brightest ommatidia over both eyes >= VISIBLE (targets are drawn at
  0.75-1.0, walls at 0.25-0.65). The controller engages after DARK_STEPS consecutive steps below.
- wall ahead: the mean luminance of the ommatidia within +-WALL_YAW of the heading and at
  WALL_PITCH elevation exceeds WALL_LUM; VO2 then acts (its own wall avoidance).
- odor present (valid and max > DETECT): turn = UPWIND_GAIN * upwind angle + ODOR_GAIN *
  (L - R) / (L + R); speed = held speed * clamp(1 - mean odor / NEAR, 0, 1).
- odor lost: hold the turn rate and speed commanded on the last visible step.
The command goes through the world's bounds and acceleration limit (ActionLimits) against the
efference copy, as OcclusionTeacher's rule does.
"""

from __future__ import annotations

import math

import torch
from flytrack_core.sensing.hex_eye import compute_directions
from flytrack_v2.world.teacher import PrivilegedTeacher
from flytrack_v2.world.wind import upwind_angle

PARAMS = {
    "visible": 0.65,
    "dark_steps": 3,
    "wall_yaw_deg": 40.0,
    "wall_pitch_deg": (25.0, 36.0),
    "wall_lum": 0.26,
    "detect": 1e-3,
    "upwind_gain": 0.5,
    "odor_gain": 4.0,
    "near": 2.0,
}


class ChemController:
    def __init__(self, actor, contract, params=None, *, use_chem=True):
        self.actor, self.contract = actor, contract
        self.p = {**PARAMS, **(params or {})}
        self.use_chem = use_chem  # False: the hold-only control (odor and wind never used)
        self.limits = PrivilegedTeacher(contract)
        directions = torch.as_tensor(compute_directions(contract), dtype=torch.float64)  # [2,N,2]
        yaw, pitch = directions[..., 0], directions[..., 1]
        lo, hi = (math.radians(v) for v in self.p["wall_pitch_deg"])
        self.wall_mask = (yaw.abs() <= math.radians(self.p["wall_yaw_deg"])) & (pitch >= lo) & (
            pitch <= hi
        )
        if not bool(self.wall_mask.any()):
            raise ValueError("the wall window selects no ommatidium")
        self.stats = None

    def reset(self, batch, device):
        self.actor.reset(batch, device)
        z = lambda: torch.zeros(batch, device=device)  # noqa: E731
        self.dark = torch.zeros(batch, dtype=torch.int64, device=device)
        self.held = torch.zeros(batch, 2, device=device)
        self.prev = torch.zeros(batch, 2, device=device)  # efference copy of the applied command
        self.mask = self.wall_mask.to(device)
        self.stats = {"steps": z(), "active": z(), "chem": z(), "wall": z()}
        self.alive = torch.ones(batch, dtype=torch.bool, device=device)  # first episode only

    @torch.no_grad()
    def act(self, obs, reset_mask):
        p = self.p
        if reset_mask is not None:
            r = reset_mask
            self.alive &= ~r
            self.dark = torch.where(r, torch.zeros_like(self.dark), self.dark)
            self.held = torch.where(r[:, None], torch.zeros_like(self.held), self.held)
            self.prev = torch.where(r[:, None], torch.zeros_like(self.prev), self.prev)
        base = self.actor.act(obs, reset_mask)
        vision = obs["vision"]
        top3 = vision.flatten(1).topk(3, dim=-1).values.mean(-1)
        visible = top3 >= p["visible"]
        self.dark = torch.where(visible, torch.zeros_like(self.dark), self.dark + 1)
        active = self.dark >= p["dark_steps"]
        wall = vision[:, self.mask].mean(-1) > p["wall_lum"]
        f64 = torch.float64
        odor = obs["odor"].to(f64).clamp(min=0)
        present = (obs["valid"][:, 2] > 0) & (odor.max(-1).values > p["detect"])
        if not self.use_chem:
            present = torch.zeros_like(present)
        left, right = odor.unbind(-1)
        bilateral = torch.where(present, (left - right) / (left + right).clamp(min=1e-12), 0.0)
        wind_valid = obs["valid"][:, 3] > 0
        upwind = torch.where(
            wind_valid, upwind_angle(obs["wind"].to(f64), self.contract.antenna_angle_deg), 0.0
        )
        held_v, held_w = self.held[:, 0].to(f64), self.held[:, 1].to(f64)
        turn = torch.where(present, p["upwind_gain"] * upwind + p["odor_gain"] * bilateral, held_w)
        approach = held_v * (1 - odor.mean(-1) / p["near"]).clamp(0, 1)
        speed = torch.where(present, approach, held_v)
        own = self.limits._clip(torch.stack((speed, turn), -1).float(), self.prev)
        use = active & ~wall
        action = torch.where(use[:, None], own, base)
        self.prev = self.limits._clip(action, self.prev)
        self.held = torch.where(visible[:, None], self.prev, self.held)
        s, live = self.stats, self.alive.float()
        s["steps"] += live
        s["active"] += live * use.float()
        s["chem"] += live * (use & present).float()
        s["wall"] += live * (active & wall).float()
        return action
