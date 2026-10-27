"""Scenario time windows shared by Gymnasium and ROS event consumers."""

from dataclasses import asdict, dataclass, fields

import numpy as np

from flytrack_core.sensing.corruption import CorruptionConfig


@dataclass(frozen=True)
class ScenarioSpec:
    # Defaults equal configs/scenarios/*.yaml, the source every trainer and evaluator
    # must load (flytrack_train.worlds.make_env refuses to fall back to these).
    name: str = "normal"
    duration_s: float = 30.0
    occlusion_start_s: float = 8.0
    occlusion_duration_s: float = 30.0
    image_noise_std: float = 0.06
    brightness_std: float = 0.15000000000000002
    image_dropout: float = 0.037500000000000006
    odor_noise_std: float = 0.0225
    odor_dropout: float = 0.07500000000000001
    decoy_x: float = -2.0
    decoy_y: float = 0.0

    def __post_init__(self):
        if self.name not in ("normal", "occlusion", "noise", "conflict"):
            raise ValueError(f"unknown scenario: {self.name}")
        if self.duration_s <= 0 or min(self.occlusion_start_s, self.occlusion_duration_s) < 0:
            raise ValueError("scenario duration must be positive and event times nonnegative")
        if any(
            not np.isfinite(getattr(self, field.name))
            for field in fields(self)
            if field.name != "name"
        ):
            raise ValueError("scenario parameters must be finite")
        CorruptionConfig(
            image_noise_std=self.image_noise_std,
            brightness_std=self.brightness_std,
            image_dropout=self.image_dropout,
            odor_noise_std=self.odor_noise_std,
            odor_dropout=self.odor_dropout,
        )

    def occluded_at(self, time_s: float) -> bool:
        return (
            self.name == "occlusion"
            and self.occlusion_start_s
            <= time_s
            < self.occlusion_start_s + self.occlusion_duration_s
        )

    def corruption_at(self, time_s: float) -> CorruptionConfig:
        if self.name == "noise":
            return CorruptionConfig(
                image_noise_std=self.image_noise_std,
                brightness_std=self.brightness_std,
                image_dropout=self.image_dropout,
                odor_noise_std=self.odor_noise_std,
                odor_dropout=self.odor_dropout,
            )
        return CorruptionConfig(blackout=self.occluded_at(time_s))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict, name: str | None = None):
        names = {field.name for field in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"unknown scenario parameters: {sorted(unknown)}")
        selected = dict(values)
        if name is not None:
            if selected.get("name", name) != name:
                raise ValueError("scenario name does not match the selected scenario")
            selected["name"] = name
        return cls(**selected)
