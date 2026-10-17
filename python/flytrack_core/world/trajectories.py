"""Bounded target trajectories that are pure functions of (config, seed, time).

``stationary``, ``linear`` and ``curved`` are analytic. ``wander`` is a smooth
random walk (Ornstein-Uhlenbeck turn rate with soft wall avoidance) integrated
on a fixed grid from a seed-derived stream, so ROS and the 2D world replay the
same path for the same seed.
"""

import math
from dataclasses import dataclass, fields

import numpy as np


def wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


MOTIONS = ("stationary", "linear", "curved", "wander")
WANDER_DT_S = 0.01
_WANDER_CHUNK_S = 10.0


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
    # Curved motion only: +1 counter-clockwise (default), -1 clockwise.
    direction: float = 1.0
    # Wander motion only: turn rate is an OU process with this stationary std and
    # correlation time, clipped to the maximum; walls repel within the margin.
    wander_turn_rate_std: float = 0.15
    wander_turn_tau_s: float = 2.0
    wander_max_turn_rate: float = 0.4
    wander_wall_margin_m: float = 1.0
    wander_wall_gain: float = 1.0
    xmin: float = -4.5
    ymin: float = -4.5
    xmax: float = 4.5
    ymax: float = 4.5

    def __post_init__(self):
        if self.motion not in MOTIONS:
            raise ValueError(f"unknown target motion: {self.motion}")
        if any(
            not np.isfinite(getattr(self, field.name))
            for field in fields(self)
            if field.name != "motion"
        ):
            raise ValueError("trajectory parameters must be finite")
        if self.direction not in (1.0, -1.0):
            raise ValueError("curved trajectory direction must be +1 or -1")
        if self.speed_m_s < 0 or self.radius_m <= 0:
            raise ValueError("trajectory speed must be nonnegative and radius positive")
        if (
            min(self.wander_turn_rate_std, self.wander_wall_gain) < 0
            or min(self.wander_turn_tau_s, self.wander_max_turn_rate, self.wander_wall_margin_m)
            <= 0
        ):
            raise ValueError("wander parameters must be positive (std and wall gain nonnegative)")
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
        if self.config.motion == "wander":
            c = self.config
            self._wander_rng = np.random.default_rng([int(seed), 0x5EED])
            self._wander_x = [float(c.initial_x)]
            self._wander_y = [float(c.initial_y)]
            self._wander_heading = [float(c.heading_rad)]
            self._wander_rate = 0.0

    def _extend_wander(self, time_s: float) -> None:
        """Integrate the wander path forward in fixed chunks; draws are sequential."""
        c = self.config
        needed = int(math.ceil(time_s / WANDER_DT_S)) + 2
        decay = math.exp(-WANDER_DT_S / c.wander_turn_tau_s)
        noise = c.wander_turn_rate_std * math.sqrt(1 - decay**2)
        center = np.asarray([(c.xmin + c.xmax) / 2, (c.ymin + c.ymax) / 2])
        chunk = int(round(_WANDER_CHUNK_S / WANDER_DT_S))
        while len(self._wander_x) < needed:
            for draw in self._wander_rng.standard_normal(chunk):
                x, y, heading = self._wander_x[-1], self._wander_y[-1], self._wander_heading[-1]
                self._wander_rate = float(
                    np.clip(
                        decay * self._wander_rate + noise * draw,
                        -c.wander_max_turn_rate,
                        c.wander_max_turn_rate,
                    )
                )
                wall = min(x - c.xmin, c.xmax - x, y - c.ymin, c.ymax - y)
                proximity = min(max((c.wander_wall_margin_m - wall) / c.wander_wall_margin_m, 0), 1)
                inward = float(wrap_angle(math.atan2(center[1] - y, center[0] - x) - heading))
                rate = self._wander_rate + c.wander_wall_gain * proximity * inward
                heading = float(wrap_angle(heading + rate * WANDER_DT_S))
                x += c.speed_m_s * math.cos(heading) * WANDER_DT_S
                y += c.speed_m_s * math.sin(heading) * WANDER_DT_S
                if not (c.xmin <= x <= c.xmax and c.ymin <= y <= c.ymax):
                    # Hard fallback only: stay inside and head back toward the centre.
                    x, y = float(np.clip(x, c.xmin, c.xmax)), float(np.clip(y, c.ymin, c.ymax))
                    heading = math.atan2(center[1] - y, center[0] - x)
                self._wander_x.append(x)
                self._wander_y.append(y)
                self._wander_heading.append(heading)

    def state(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return pose[x,y,yaw] and velocity[vx,vy] as a pure function of (config, seed, time).

        Wander only memoizes its deterministic, seed-derived path.
        """
        if time_s < 0:
            raise ValueError("trajectory time must be nonnegative")
        c = self.config
        if c.motion == "stationary":
            xy = np.asarray([c.initial_x, c.initial_y])
            velocity = np.zeros(2)
            heading = c.heading_rad
        elif c.motion == "wander":
            self._extend_wander(time_s)
            position = time_s / WANDER_DT_S
            index = int(position)
            fraction = position - index
            xy = np.asarray(
                [
                    (1 - fraction) * self._wander_x[index] + fraction * self._wander_x[index + 1],
                    (1 - fraction) * self._wander_y[index] + fraction * self._wander_y[index + 1],
                ]
            )
            start, end = self._wander_heading[index], self._wander_heading[index + 1]
            heading = float(wrap_angle(start + fraction * float(wrap_angle(end - start))))
            velocity = c.speed_m_s * np.asarray([math.cos(heading), math.sin(heading)])
        elif c.motion == "curved":
            angle = self.phase + c.direction * c.speed_m_s * time_s / c.radius_m
            xy = np.asarray(
                [
                    c.center_x + c.radius_m * math.cos(angle),
                    c.center_y + c.radius_m * math.sin(angle),
                ]
            )
            velocity = c.direction * c.speed_m_s * np.asarray([-math.sin(angle), math.cos(angle)])
            heading = angle + c.direction * np.pi / 2
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
