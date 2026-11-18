"""The ``wind`` world's wind field, filament plume and antenna sense (WP2b, #133; plan 3.3).

- Wind: a uniform field of speed U whose direction is drawn per episode and meanders as an
  Ornstein-Uhlenbeck process (stationary std ``direction_std_deg``, correlation time
  ``direction_tau_s``), updated every plume tick.
- Plume: the target (and the conflict decoy) releases one filament every 1 / emission_hz s.
  Filaments are advected by the wind, take a turbulent random walk, widen as
  sigma^2 = sigma0^2 + 2 D age, live ``lifetime_s`` and at most ``capacity`` exist per
  environment (the oldest is dropped). Concentration is the sum of their Gaussians
  (Farrell et al. 2002, in 2D), the same form as v1's puff plume.
- Antennae (interfaces.md section 1): with theta the direction the wind comes from relative
  to the heading (left positive) and alpha the antenna angle,
  left = (U / U_ref) cos(theta - alpha), right = (U / U_ref) cos(theta + alpha), plus noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class WindConfig:
    speed_m_s: float = 0.5  # U; pilot candidates 0.3 and 0.5 (plan 3.10)
    direction_std_deg: float = 20.0
    direction_tau_s: float = 10.0
    emission_hz: float = 10.0
    lifetime_s: float = 20.0
    capacity: int = 256
    initial_sigma_m: float = 0.10
    diffusion_m2_s: float = 0.025  # filament widening
    turbulent_diffusion_m2_s: float = 0.015  # filament centre random walk
    puff_mass: float = 0.03
    decay_tau_s: float = math.inf  # no decay: a filament keeps its mass for its lifetime
    antenna_noise_std: float = 0.05

    def __post_init__(self) -> None:
        if not (self.speed_m_s > 0 and self.emission_hz > 0 and self.lifetime_s > 0):
            raise ValueError("wind speed, emission rate and lifetime must be positive")
        if self.direction_std_deg < 0 or self.direction_tau_s <= 0 or self.antenna_noise_std < 0:
            raise ValueError("wind meander and antenna noise must be nonnegative")
        if self.capacity < 1 or self.initial_sigma_m <= 0 or self.decay_tau_s <= 0:
            raise ValueError("capacity, initial width and decay time must be positive")

    @classmethod
    def from_dict(cls, values: dict | None) -> WindConfig:
        return cls(**dict(values or {}))


def antenna(wind: Tensor, heading: Tensor, angle_deg: float, reference_m_s: float) -> Tensor:
    """[B,2] noiseless left/right antenna readings for world wind velocities [B,2] (the
    direction the wind blows to) and headings [B]."""
    speed = wind.norm(dim=-1)
    coming_from = torch.atan2(-wind[:, 1], -wind[:, 0])
    theta = torch.remainder(coming_from - heading + math.pi, 2 * math.pi) - math.pi
    alpha = math.radians(angle_deg)
    scale = speed / reference_m_s
    return torch.stack((scale * torch.cos(theta - alpha), scale * torch.cos(theta + alpha)), -1)


def upwind_angle(reading: Tensor, angle_deg: float) -> Tensor:
    """theta (where the wind comes from, relative to the heading, left positive) recovered
    from a [B,2] antenna reading: left + right = 2k cos(theta) cos(alpha) and
    left - right = 2k sin(theta) sin(alpha)."""
    alpha = math.radians(angle_deg)
    left, right = reading.unbind(-1)
    return torch.atan2((left - right) / math.sin(alpha), (left + right) / math.cos(alpha))
