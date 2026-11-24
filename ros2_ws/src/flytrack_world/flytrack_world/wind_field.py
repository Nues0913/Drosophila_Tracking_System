"""The v2 ``wind`` world's wind, filament plume and antenna sense for one Gazebo episode (#140).

The same model as the 2D wind world (``flytrack_v2.world.wind`` and ``v1compat.py``,
``_plume_tick`` / ``_wind_tick`` / ``_apply_wind``), in numpy for a single environment:

- Wind: speed U; the direction is drawn uniformly per episode and meanders as an
  Ornstein-Uhlenbeck process (stationary std ``direction_std_deg``, correlation time
  ``direction_tau_s``). Its initial offset is drawn from the stationary distribution.
- Plume tick (every ``integration_dt``): the wind takes its OU step first, then v1's puff order:
  age, advect by the current wind, turbulent walk, drop puffs past their lifetime, emit.
- Antennae: left = (U / U_ref) cos(theta - alpha), right = (U / U_ref) cos(theta + alpha), with
  theta the direction the wind comes from relative to the heading (left positive), plus
  ``antenna_noise_std`` Gaussian noise (always on, like the 2D world).

The random draws come from numpy generators seeded by the episode seed, not from the 2D world's
counter-based hash streams, so a Gazebo episode and a 2D episode with the same seed see
different wind directions: G1 compares distributions, as it does for v1.
"""

from __future__ import annotations

import math

import numpy as np
from flytrack_core.world import PlumeConfig, PuffPlume

# The 2D world's PlumeConfig needs a finite decay; exp(-age / 1e30) is exactly 1 in float64.
NO_DECAY_TAU_S = 1e30


def antenna(wind: np.ndarray, heading: float, angle_deg: float, reference_m_s: float) -> np.ndarray:
    """Noiseless [left, right] antenna reading for a world wind velocity (where it blows to)."""
    wind = np.asarray(wind, dtype=np.float64)
    speed = float(np.hypot(wind[0], wind[1]))
    coming_from = math.atan2(-wind[1], -wind[0])
    theta = (coming_from - heading + math.pi) % (2 * math.pi) - math.pi
    alpha = math.radians(angle_deg)
    scale = speed / reference_m_s
    return np.asarray([scale * math.cos(theta - alpha), scale * math.cos(theta + alpha)])


def plume_config(wind_config, sources: int, base: PlumeConfig | None = None) -> PlumeConfig:
    """The 2D wind world's plume: v1's puff settings with the wind config's filament values."""
    w = wind_config
    base = base or PlumeConfig()
    # Per source, as the 2D world (#192 R6): the conflict decoy does not shorten the target's
    # trail. PuffPlume emits the sources in turn and keeps the newest max_puffs filaments.
    per_source = min(int(w.lifetime_s * w.emission_hz) + 3, w.capacity)
    return PlumeConfig(
        wind_x=0.0,
        wind_y=0.0,
        emission_hz=w.emission_hz,
        initial_sigma_m=w.initial_sigma_m,
        diffusion_m2_s=w.diffusion_m2_s,
        turbulent_diffusion_m2_s=w.turbulent_diffusion_m2_s,
        puff_mass=w.puff_mass,
        lifetime_s=w.lifetime_s,
        decay_tau_s=w.decay_tau_s if math.isfinite(w.decay_tau_s) else NO_DECAY_TAU_S,
        integration_dt=base.integration_dt,
        max_puffs=sources * per_source,
        background=base.background,
    )


class WindPlume(PuffPlume):
    """A PuffPlume whose wind meanders (OU) and is stepped at every plume tick."""

    def __init__(self, wind_config, sources: int = 1, seed: int = 0):
        self.wind_config = wind_config
        super().__init__(plume_config(wind_config, sources), seed)

    def reset(self, seed: int = 0, time_s: float = 0.0):
        super().reset(seed, time_s)
        wind_rng, self.antenna_rng = (
            np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)
        )
        self.wind_rng = wind_rng
        self.direction = float(wind_rng.uniform(-math.pi, math.pi))
        self.eta = math.radians(self.wind_config.direction_std_deg) * float(wind_rng.normal())
        self._wind = self._vector()

    @property
    def wind(self) -> np.ndarray:
        return self._wind

    def _vector(self) -> np.ndarray:
        angle = self.direction + self.eta
        return self.wind_config.speed_m_s * np.asarray([math.cos(angle), math.sin(angle)])

    def _wind_step(self) -> None:
        w, dt = self.wind_config, self.config.integration_dt
        decay = math.exp(-dt / w.direction_tau_s)
        sigma = math.radians(w.direction_std_deg) * math.sqrt(1 - decay**2)
        self.eta = decay * self.eta + sigma * float(self.wind_rng.normal())
        self._wind = self._vector()

    def advance(self, time_s: float, source_positions):
        """v1's PuffPlume ticks, each preceded by the wind's OU step (the 2D order)."""
        if time_s < self.time_s - 1e-9:
            raise ValueError("plume time cannot run backwards; reset on a new episode")
        dt = self.config.integration_dt
        desired_tick = int(np.floor((time_s - self._origin_time_s + 1e-9) / dt))
        while self._tick < desired_tick:
            self._wind_step()
            # One parent tick with the current wind; it advances self._tick by one.
            super().advance(self._origin_time_s + (self._tick + 1) * dt, source_positions)
        self.time_s = float(time_s)

    def antenna_reading(self, heading: float, angle_deg: float, reference_m_s: float) -> np.ndarray:
        """The antenna reading at the current wind, with the always-on sensor noise."""
        clean = antenna(self._wind, heading, angle_deg, reference_m_s)
        noise = self.wind_config.antenna_noise_std * self.antenna_rng.normal(size=2)
        return (clean + noise).astype(np.float32)


WIND_RNG_STREAM = 3  # SensorCorruptor's own streams are children 0-2 (two eyes, odor)


def wind_rng(seed: int) -> np.random.Generator:
    """The corruptor's wind-dropout stream, independent of its eye and odor streams."""
    return np.random.default_rng(np.random.SeedSequence(seed).spawn(WIND_RNG_STREAM + 1)[-1])


def wind_dropped(rng: np.random.Generator, probability: float) -> bool:
    """One dropout draw per sample, only when the scenario has dropout (as SensorCorruptor)."""
    return probability > 0 and rng.random() < probability
