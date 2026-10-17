"""Per-episode target/agent layout, shared by the 2D world and the ROS scenario manager.

The original fixed layout (counter-clockwise circle, agent directly behind the
target) let a blind constant left turn succeed in every scenario. The defaults
here randomize direction, geometry and start pose so that only policies that
use their senses can track; ``LayoutConfig.legacy()`` reproduces the old layout
for re-running historical diagnostics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from .trajectories import MOTIONS, TargetTrajectory, TrajectoryConfig, wrap_angle

# Separate stream so layout draws never shift any other per-episode random draw.
LAYOUT_STREAM = 0x1A70


@dataclass(frozen=True)
class LayoutConfig:
    randomize_target_direction: bool = True
    start_distance_m: float = 1.5
    start_lateral_offset_m: float = 0.5
    start_heading_offset_deg: float = 30.0
    # Curved radius, linear heading/start and stationary/wander start positions.
    randomize_target_layout: bool = True
    curve_radius_min_m: float = 1.5
    curve_radius_max_m: float = 3.5
    # Half-width of the centred square that holds linear/stationary/wander starts.
    start_region_half_m: float = 2.5

    def __post_init__(self):
        numbers = [
            getattr(self, field.name)
            for field in fields(self)
            if not field.name.startswith("randomize")
        ]
        if any(not np.isfinite(value) for value in numbers):
            raise ValueError("layout parameters must be finite")
        if (
            self.start_distance_m <= 0
            or self.start_lateral_offset_m < 0
            or not 0 <= self.start_heading_offset_deg <= 180
            or not 0 < self.curve_radius_min_m <= self.curve_radius_max_m
            or self.start_region_half_m < 0
        ):
            raise ValueError(
                "layout needs a positive start distance, nonnegative offsets (heading <= 180 deg), "
                "0 < curve_radius_min_m <= curve_radius_max_m and a nonnegative start region"
            )

    @classmethod
    def legacy(cls) -> LayoutConfig:
        """Layout used before 2026-10-17 (fixed CCW circle, agent directly behind)."""
        return cls(
            randomize_target_direction=False,
            start_lateral_offset_m=0.0,
            start_heading_offset_deg=0.0,
            randomize_target_layout=False,
        )

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> LayoutConfig:
        values = dict(values or {})
        if values.pop("preset", None) == "legacy":
            return cls(**{**asdict(cls.legacy()), **values})
        names = {field.name for field in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"unknown layout parameters: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def motion_choices(motion: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    choices = (motion,) if isinstance(motion, str) else tuple(motion)
    if not choices or any(choice not in MOTIONS for choice in choices):
        raise ValueError(f"target_motion must be one of {MOTIONS} or a nonempty list of them")
    return choices


def sample_episode_layout(
    episode_seed: int,
    motion: str | list[str] | tuple[str, ...],
    arena_defaults: dict[str, Any],
    explicit: dict[str, Any],
    layout: LayoutConfig,
    walls: np.ndarray,
    agent_radius_m: float,
    agent_pose: Any = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return (trajectory config, initial agent pose [x, y, yaw]) for one episode.

    An explicit ``agent_pose`` is used as given (the caller validates its shape);
    the unused sampled start is then neither computed nor checked against the
    arena, as before layouts were sampled.

    Precedence: ``arena_defaults`` (centre, bounds, speed) < sampled values <
    ``explicit`` user trajectory options, which are deliberate choices (e.g. a
    fixed radius) and are never overridden. Draw order is fixed: direction,
    lateral, heading, motion, then geometry; disabled options consume no draw
    after the first three, so older layouts reproduce exactly.
    """
    rng = np.random.default_rng([int(episode_seed), LAYOUT_STREAM])
    direction = float(rng.choice((-1.0, 1.0))) if layout.randomize_target_direction else 1.0
    lateral = rng.uniform(-1, 1) * layout.start_lateral_offset_m
    yaw_offset = np.deg2rad(rng.uniform(-1, 1) * layout.start_heading_offset_deg)
    choices = motion_choices(motion)
    chosen = choices[int(rng.integers(len(choices)))] if len(choices) > 1 else choices[0]
    explicit = {key: value for key, value in explicit.items() if key != "motion"}
    base = {**arena_defaults, **explicit}
    sampled: dict[str, Any] = {"direction": direction}
    if layout.randomize_target_layout:
        center = np.asarray([base.get("center_x", 0.0), base.get("center_y", 0.0)])
        half = layout.start_region_half_m
        start = center + rng.uniform(-half, half, size=2)
        heading = float(rng.uniform(-np.pi, np.pi))
        if chosen == "curved":
            sampled["radius_m"] = float(
                rng.uniform(layout.curve_radius_min_m, layout.curve_radius_max_m)
            )
        elif chosen in ("linear", "stationary", "wander"):
            sampled.update(
                initial_x=float(start[0]), initial_y=float(start[1]), heading_rad=heading
            )
    config = {**arena_defaults, **sampled, **explicit, "motion": chosen}
    trajectory = TrajectoryConfig.from_dict(config)
    if agent_pose is not None:
        return config, np.asarray(agent_pose, dtype=np.float64).copy()
    target, _ = TargetTrajectory(trajectory, int(episode_seed)).state(0.0)
    heading = target[2]
    # Behind the target, shifted sideways by `lateral` (positive = left of its heading).
    agent = np.asarray(
        [
            target[0] - layout.start_distance_m * np.cos(heading) - lateral * np.sin(heading),
            target[1] - layout.start_distance_m * np.sin(heading) + lateral * np.cos(heading),
            float(wrap_angle(heading + yaw_offset)) if yaw_offset else heading,
        ],
        dtype=np.float64,
    )
    walls = np.asarray(walls, dtype=np.float64)
    if np.any(agent[:2] < walls[:2] + agent_radius_m) or np.any(
        agent[:2] > walls[2:] - agent_radius_m
    ):
        raise ValueError(
            "sampled agent start lies outside the arena; shrink start_region_half_m, "
            "curve_radius_max_m or start offsets"
        )
    return config, agent
