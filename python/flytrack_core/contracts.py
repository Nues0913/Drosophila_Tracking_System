"""Shared units and shapes. This module depends only on numpy and the standard library."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping

import numpy as np

SCHEMA_VERSION = 1
GROUP_VISUAL = 1
GROUP_OLFACTORY = 2
GROUP_CONVERGENCE = 4
GROUP_TURN = 8
GROUP_FORWARD = 16
GROUP_WIND = 32
VARIANTS = ("V", "O", "VO", "VO-late", "GRU", "RS")
SCENARIOS = ("normal", "occlusion", "noise", "conflict")


def canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SensorContract:
    control_dt: float = 0.05
    neural_dt: float = 0.01
    history_steps: int = 10
    eye_radius: int = 15
    horizontal_fov_deg: float = 120.0
    vertical_fov_deg: float = 104.82182106205009
    eye_yaw_deg: float = 45.0
    acceptance_angle_deg: float = 5.0
    camera_width: int = 160
    camera_height: int = 120
    odor_baseline_m: float = 0.12
    vision_hz: float = 30.0
    odor_hz: float = 50.0
    watchdog_s: float = 0.2
    v_max: float = 0.5
    omega_max: float = 1.5
    linear_accel_max: float = 1.0
    angular_accel_max: float = 3.0

    def __post_init__(self) -> None:
        numeric = asdict(self)
        if any(not np.isfinite(value) or value <= 0 for value in numeric.values()):
            raise ValueError("All sensor-contract quantities must be finite and positive")
        for name in ("history_steps", "eye_radius", "camera_width", "camera_height"):
            if isinstance(numeric[name], (bool, np.bool_)) or not isinstance(
                numeric[name], (int, np.integer)
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(numeric[name]))
        ratio = self.control_dt / self.neural_dt
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt must be an integer multiple of neural_dt")
        if not 0 < self.horizontal_fov_deg < 180 or not 0 < self.vertical_fov_deg < 180:
            raise ValueError("Pinhole camera fields of view must be between 0 and 180 degrees")
        if self.watchdog_s < self.control_dt:
            raise ValueError("watchdog_s must be at least one control interval")

    @property
    def n_ommatidia(self) -> int:
        return 1 + 3 * self.eye_radius * (self.eye_radius + 1)

    @property
    def neural_substeps(self) -> int:
        return round(self.control_dt / self.neural_dt)

    def to_dict(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        # The geometry algorithm version is part of the contract: radius alone is insufficient.
        payload = {"schema_version": SCHEMA_VERSION, "hex_geometry_version": 1, **self.to_dict()}
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping) -> "SensorContract":
        known = {field.name for field in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"Unknown sensor contract fields: {sorted(unknown)}")
        return cls(**dict(value))

    @classmethod
    def from_json(cls, path: str | Path) -> "SensorContract":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class SensorObservation:
    vision: np.ndarray  # [2, n_ommatidia], grayscale in [0, 1]
    odor: np.ndarray  # [2], nonnegative arbitrary concentration units
    valid: np.ndarray  # [3]: left image, right image, paired odor reading
    timestamp_s: float  # simulator time; never target pose

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"vision": self.vision, "odor": self.odor, "valid": self.valid}

    def validate(self, contract: SensorContract) -> None:
        expected = {"vision": (2, contract.n_ommatidia), "odor": (2,), "valid": (3,)}
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid {name}: expected finite array with shape {shape}")
        if np.any((self.vision < 0) | (self.vision > 1)) or np.any(self.odor < 0):
            raise ValueError("Vision must be in [0, 1] and odor concentration nonnegative")
        if np.any((self.valid != 0) & (self.valid != 1)) or not np.isfinite(self.timestamp_s):
            raise ValueError("Validity channels must be binary and timestamp must be finite")


@dataclass(frozen=True)
class ActionLimits:
    v_max: float = 0.5
    omega_max: float = 1.5
    linear_accel_max: float = 1.0
    angular_accel_max: float = 3.0

    @classmethod
    def from_contract(cls, contract: SensorContract) -> "ActionLimits":
        return cls(
            contract.v_max,
            contract.omega_max,
            contract.linear_accel_max,
            contract.angular_accel_max,
        )

    def clip(
        self, action: np.ndarray, previous: np.ndarray | None = None, dt: float = 0.05
    ) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape[-1] != 2 or not np.all(np.isfinite(value)):
            raise ValueError("Action must contain finite [v_m_per_s, omega_rad_per_s]")
        value = np.clip(value, [0.0, -self.omega_max], [self.v_max, self.omega_max])
        if previous is not None:
            if dt <= 0:
                raise ValueError("Action integration interval must be positive")
            delta = np.array([self.linear_accel_max, self.angular_accel_max]) * dt
            value = np.clip(value, np.asarray(previous) - delta, np.asarray(previous) + delta)
            value = np.clip(value, [0.0, -self.omega_max], [self.v_max, self.omega_max])
        return value.astype(np.float32)
