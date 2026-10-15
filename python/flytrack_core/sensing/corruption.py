"""Seeded corruption functions; simulator and ROS use the same implementation."""

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True)
class CorruptionConfig:
    image_noise_std: float = 0.0
    brightness_std: float = 0.0
    image_dropout: float = 0.0
    odor_noise_std: float = 0.0
    odor_dropout: float = 0.0
    blackout: bool = False

    def __post_init__(self):
        for name in ("image_noise_std", "brightness_std", "odor_noise_std"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("image_dropout", "odor_dropout"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")

    @classmethod
    def from_dict(cls, values: dict):
        return cls(
            **{field.name: values[field.name] for field in fields(cls) if field.name in values}
        )


class SensorCorruptor:
    """Independent camera and odor streams avoid callback ordering dependence."""

    def __init__(self, seed: int = 0):
        self.reset(seed)

    def reset(self, seed: int = 0):
        children = np.random.SeedSequence(seed).spawn(3)
        self.image_rng = [np.random.default_rng(children[0]), np.random.default_rng(children[1])]
        self.odor_rng = np.random.default_rng(children[2])

    def image(
        self, image: np.ndarray, config: CorruptionConfig, eye: int = 0
    ) -> tuple[np.ndarray, bool]:
        rng = self.image_rng[eye]
        values = np.asarray(image, dtype=np.float32).copy()
        if config.blackout:
            # A covered camera is a valid black image; transport dropout is missing.
            return np.zeros_like(values), True
        if config.image_dropout > 0 and rng.random() < config.image_dropout:
            return np.zeros_like(values), False
        if config.brightness_std:
            values *= max(0, 1 + rng.normal(0, config.brightness_std))
        if config.image_noise_std:
            values += rng.normal(0, config.image_noise_std, values.shape)
        return np.clip(values, 0, 1).astype(np.float32), True

    def odor(self, odor: np.ndarray, config: CorruptionConfig) -> tuple[np.ndarray, bool]:
        values = np.asarray(odor, dtype=np.float32).copy()
        if config.odor_dropout > 0 and self.odor_rng.random() < config.odor_dropout:
            return np.zeros_like(values), False
        if config.odor_noise_std:
            values += self.odor_rng.normal(0, config.odor_noise_std, values.shape)
        return np.maximum(values, 0).astype(np.float32), True
