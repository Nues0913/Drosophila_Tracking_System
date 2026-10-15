"""Seed-reproducible Gaussian curriculum and filament (puff) concentration."""

from collections.abc import Callable
from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class PlumeConfig:
    wind_x: float = 0.15
    wind_y: float = 0.0
    emission_hz: float = 8.0
    initial_sigma_m: float = 0.10
    diffusion_m2_s: float = 0.025
    turbulent_diffusion_m2_s: float = 0.015
    puff_mass: float = 0.03
    lifetime_s: float = 20.0
    decay_tau_s: float = 10.0
    integration_dt: float = 0.01
    max_puffs: int = 5000
    background: float = 0.0

    def __post_init__(self):
        if any(not np.isfinite(getattr(self, field.name)) for field in fields(self)):
            raise ValueError("plume parameters must be finite")
        if (
            min(
                self.emission_hz,
                self.initial_sigma_m,
                self.puff_mass,
                self.lifetime_s,
                self.decay_tau_s,
                self.integration_dt,
                self.max_puffs,
            )
            <= 0
        ):
            raise ValueError("emission, scale, mass, lifetime, step and capacity must be positive")
        if min(self.diffusion_m2_s, self.turbulent_diffusion_m2_s, self.background) < 0:
            raise ValueError("diffusion and background must be nonnegative")
        if isinstance(self.max_puffs, bool) or not isinstance(self.max_puffs, (int, np.integer)):
            raise ValueError("max_puffs must be an integer")

    @classmethod
    def from_dict(cls, values: dict):
        names = {field.name for field in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"unknown plume parameters: {sorted(unknown)}")
        return cls(**values)


def _sources(value) -> np.ndarray:
    positions = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    if not np.all(np.isfinite(positions)):
        raise ValueError("odor source positions must be finite xy")
    return positions


class PuffPlume:
    def __init__(self, config: PlumeConfig | None = None, seed: int = 0):
        self.config = config or PlumeConfig()
        self.reset(seed)

    @property
    def wind(self) -> np.ndarray:
        return np.asarray([self.config.wind_x, self.config.wind_y], dtype=np.float64)

    def reset(self, seed: int = 0, time_s: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.centers = np.empty((0, 2), dtype=np.float64)
        self.ages = np.empty(0, dtype=np.float64)
        self.time_s = float(time_s)
        self._origin_time_s = float(time_s)
        self._tick = 0
        self._emission_credit = 0.0

    def advance(self, time_s: float, source_positions: np.ndarray | Callable[[float], np.ndarray]):
        """Advance fixed internal ticks; concentration queries never consume RNG.

        Supply a time→xy callback for exact moving-source sampling in offline
        simulation. ROS may instead supply its latest measured target pose.
        """
        if time_s < self.time_s - 1e-9:
            raise ValueError("plume time cannot run backwards; reset on a new episode")
        dt = self.config.integration_dt
        desired_tick = int(np.floor((time_s - self._origin_time_s + 1e-9) / dt))
        while self._tick < desired_tick:
            self._tick += 1
            tick_time = self._origin_time_s + self._tick * dt
            self.ages += dt
            self.centers += self.wind * dt
            if len(self.centers) and self.config.turbulent_diffusion_m2_s:
                self.centers += self.rng.normal(
                    0, np.sqrt(2 * self.config.turbulent_diffusion_m2_s * dt), self.centers.shape
                )
            alive = self.ages <= self.config.lifetime_s
            self.centers, self.ages = self.centers[alive], self.ages[alive]
            self._emission_credit += self.config.emission_hz * dt
            emitted = int(self._emission_credit + 1e-12)
            if emitted:
                self._emission_credit -= emitted
                sources = _sources(
                    source_positions(tick_time) if callable(source_positions) else source_positions
                )
                positions = np.tile(sources, (emitted, 1))
                self.centers = np.concatenate((self.centers, positions), axis=0)[
                    -self.config.max_puffs :
                ]
                self.ages = np.concatenate((self.ages, np.zeros(len(positions))))[
                    -self.config.max_puffs :
                ]
        self.time_s = float(time_s)

    def concentration(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1:] != (2,):
            raise ValueError("sample positions must have last dimension xy")
        result = np.full(points.shape[:-1], self.config.background, dtype=np.float64)
        if len(self.centers):
            variance = self.config.initial_sigma_m**2 + 2 * self.config.diffusion_m2_s * self.ages
            flattened = points.reshape(-1, 2)
            # Chunk samples so a large vectorized world need not allocate B×puffs×xy.
            for start in range(0, len(flattened), 256):
                difference = flattened[start : start + 256, None, :] - self.centers[None, :, :]
                squared = np.sum(difference**2, axis=-1)
                density = (
                    self.config.puff_mass
                    * np.exp(-self.ages / self.config.decay_tau_s)
                    / (2 * np.pi * variance)
                )
                result.reshape(-1)[start : start + 256] += (
                    np.exp(-squared / (2 * variance)) * density
                ).sum(axis=-1)
        return result.astype(np.float32)


class GaussianPlume:
    """Steady downstream Gaussian; moving source updates do not leave a trail."""

    def __init__(self, config: PlumeConfig | None = None, seed: int = 0):
        self.config = config or PlumeConfig()
        self.reset(seed)

    @property
    def wind(self) -> np.ndarray:
        return np.asarray([self.config.wind_x, self.config.wind_y], dtype=np.float64)

    def reset(self, seed: int = 0, time_s: float = 0.0):
        self.time_s = float(time_s)
        self.sources = np.empty((0, 2))

    def advance(self, time_s: float, source_positions):
        if time_s < self.time_s - 1e-9:
            raise ValueError("plume time cannot run backwards")
        self.sources = _sources(
            source_positions(time_s) if callable(source_positions) else source_positions
        )
        self.time_s = float(time_s)

    def concentration(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        wind_speed = np.linalg.norm(self.wind)
        direction = self.wind / max(wind_speed, 1e-9) if wind_speed else np.asarray([1.0, 0.0])
        lateral_direction = np.asarray([-direction[1], direction[0]])
        difference = points[..., None, :] - self.sources
        downstream = difference @ direction
        crosswind = difference @ lateral_direction
        sigma2 = self.config.initial_sigma_m**2 + 2 * self.config.diffusion_m2_s * np.maximum(
            downstream, 0
        ) / max(wind_speed, 0.1)
        # Weak upstream concentration keeps this curriculum smooth around a source.
        along_scale = np.where(downstream >= 0, 3.0, 0.35)
        density = (
            self.config.puff_mass
            * self.config.emission_hz
            * np.exp(-0.5 * crosswind**2 / sigma2 - np.abs(downstream) / along_scale)
            / np.sqrt(2 * np.pi * sigma2)
        )
        return (self.config.background + density.sum(axis=-1)).astype(np.float32)


def make_plume(model: str = "puff", config: PlumeConfig | dict | None = None, seed: int = 0):
    if isinstance(config, dict):
        config = PlumeConfig.from_dict(config)
    constructors = {"puff": PuffPlume, "gaussian": GaussianPlume}
    if model not in constructors:
        raise ValueError(f"unknown plume model: {model}")
    return constructors[model](config, seed)
