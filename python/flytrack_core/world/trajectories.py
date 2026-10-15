"""Analytic bounded target trajectories with reproducible episode phases."""

import math
from dataclasses import dataclass, fields

import numpy as np


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class TrajectoryConfig:
    motion: str = "curved"
    speed_m_s: float = 0.20
    center_x: float = 0.0
    center_y: float = 0.0
    radius_m: float = 2.0
    initial_x: float = 0.0
    initial_y: float = 0.0
    heading_rad: float = 0.0
    xmin: float = -4.5
    ymin: float = -4.5
    xmax: float = 4.5
    ymax: float = 4.5

    def __post_init__(self):
        if self.motion not in ("stationary", "linear", "curved"):
            raise ValueError(f"unknown target motion: {self.motion}")
        if any(
            not np.isfinite(getattr(self, field.name))
            for field in fields(self)
            if field.name != "motion"
        ):
            raise ValueError("trajectory parameters must be finite")
        if self.speed_m_s < 0 or self.radius_m <= 0:
            raise ValueError("trajectory speed must be nonnegative and radius positive")
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("trajectory bounds must be ordered")
        if not (
            self.xmin <= self.initial_x <= self.xmax and self.ymin <= self.initial_y <= self.ymax
        ):
            raise ValueError("initial target pose must fit inside trajectory bounds")
        if self.motion == "curved" and not (
            self.xmin <= self.center_x - self.radius_m
            and self.center_x + self.radius_m <= self.xmax
            and self.ymin <= self.center_y - self.radius_m
            and self.center_y + self.radius_m <= self.ymax
        ):
            raise ValueError("curved target trajectory must fit inside bounds")

    @classmethod
    def from_dict(cls, values: dict):
        names = {field.name for field in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"unknown trajectory parameters: {sorted(unknown)}")
        return cls(**values)


class TargetTrajectory:
    def __init__(self, config: TrajectoryConfig | dict | None = None, seed: int = 0):
        self.config = (
            TrajectoryConfig.from_dict(config)
            if isinstance(config, dict)
            else (config or TrajectoryConfig())
        )
        self.phase = np.random.default_rng(seed).uniform(-np.pi, np.pi)

    def state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return pose[x,y,yaw] and velocity[vx,vy]; no hidden mutable state."""
        if time_s < 0:
            raise ValueError("trajectory time must be nonnegative")
        c = self.config
        if c.motion == "stationary":
            xy = np.asarray([c.initial_x, c.initial_y])
            velocity = np.zeros(2)
            heading = c.heading_rad
        elif c.motion == "curved":
            angle = self.phase + c.speed_m_s * time_s / c.radius_m
            xy = np.asarray(
                [
                    c.center_x + c.radius_m * math.cos(angle),
                    c.center_y + c.radius_m * math.sin(angle),
                ]
            )
            velocity = c.speed_m_s * np.asarray([-math.sin(angle), math.cos(angle)])
            heading = angle + np.pi / 2
        else:
            raw_velocity = c.speed_m_s * np.asarray(
                [math.cos(c.heading_rad), math.sin(c.heading_rad)]
            )
            lower, upper = np.asarray([c.xmin, c.ymin]), np.asarray([c.xmax, c.ymax])
            span = upper - lower
            raw = np.asarray([c.initial_x, c.initial_y]) - lower + raw_velocity * time_s
            phase = np.mod(raw, 2 * span)
            xy = lower + np.where(phase < span, phase, 2 * span - phase)
            velocity = raw_velocity * np.where(phase < span, 1, -1)
            heading = (
                math.atan2(velocity[1], velocity[0]) if np.linalg.norm(velocity) else c.heading_rad
            )
        return np.asarray([*xy, float(wrap_angle(heading))], dtype=np.float64), velocity.astype(
            np.float64
        )
