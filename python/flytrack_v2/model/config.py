"""Configuration for the v2 policies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ModelConfigV2:
    graph_path: str = "data/subgraphs/v2_main/graph.npz"
    frontend_path: str = "artifacts/frontend_v2/frontend.npz"
    visual_bins: int = 9
    activation_beta: float = 10.0
    tau_min_s: float = 0.01
    tau_max_s: float = 10.0
    tau_init_range_s: tuple[float, float] = (0.02, 2.0)
    gain_parameterization: str = "softplus"
    gain_init: float = 0.9
    gain_max: float = 0.97
    bias_lr_scale: float = 0.1
    # 計劃書 0.2 不採用 checkpoint，僅在記憶體不足且經 PM 同意時開啟。
    activation_checkpoint: bool = False
    readout_init_weight: float = 1.0
    initial_speed_m_s: float = 0.2
    rest_settle_s: float = 8.0
    rest_refresh_s: float = 1.0
    initial_state: str = "rest"
    steer_types: tuple[str, ...] | None = None
    speed_types: tuple[str, ...] | str | None = None
    log_std_init: float = -0.5
    odor_K: tuple[float, ...] = (0.3,)
    odor_hill_n: float = 2.0
    odor_adaptation_tau_s: float = 2.0
    odor_shared_adaptation: bool = True
    norm_std_floor: float = 1e-3
    init_seed: int = 0
    structure_seed: int | None = None
    budget_target: int | None = None

    def __post_init__(self) -> None:
        for key in ("graph_path", "frontend_path"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value or Path(value).is_absolute():
                raise ValueError(f"{key} must be a nonempty repo-relative path")
        for key in ("visual_bins", "init_seed", "structure_seed", "budget_target"):
            value = getattr(self, key)
            if value is None and key in ("structure_seed", "budget_target"):
                continue
            minimum = int(key in ("visual_bins", "budget_target"))
            if type(value) is not int or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        for key in ("activation_checkpoint", "odor_shared_adaptation"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        for key in ("tau_init_range_s", "odor_K", "steer_types", "speed_types"):
            value = getattr(self, key)
            if value is None and key in ("steer_types", "speed_types"):
                continue
            if key == "speed_types" and isinstance(value, str) and value == "bilateral":
                continue
            if not isinstance(value, (tuple, list)):
                options = " or 'bilateral'" if key == "speed_types" else ""
                raise ValueError(f"{key} must be a list or tuple{options}")
            object.__setattr__(self, key, tuple(value))
        for key in ("steer_types", "speed_types"):
            value = getattr(self, key)
            if isinstance(value, tuple) and (
                any(not isinstance(name, str) or not name for name in value)
                or len(set(value)) != len(value)
            ):
                raise ValueError(f"{key} must contain unique, nonempty type names")
        for key in (
            "activation_beta",
            "tau_min_s",
            "tau_max_s",
            "gain_init",
            "gain_max",
            "initial_speed_m_s",
            "odor_hill_n",
            "odor_adaptation_tau_s",
            "norm_std_floor",
        ):
            value = getattr(self, key)
            if not _is_finite_number(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if not _is_finite_number(self.rest_settle_s) or self.rest_settle_s <= 0:
            raise ValueError("rest_settle_s must be positive and finite")
        if (
            not _is_finite_number(self.rest_refresh_s)
            or not 0 < self.rest_refresh_s <= self.rest_settle_s
        ):
            raise ValueError("rest_refresh_s must be positive, finite, and at most rest_settle_s")
        if self.initial_state not in ("rest", "bias"):
            raise ValueError("initial_state must be rest or bias")
        if (
            len(self.tau_init_range_s) != 2
            or not all(map(_is_finite_number, self.tau_init_range_s))
            or not self.tau_min_s
            < self.tau_init_range_s[0]
            <= self.tau_init_range_s[1]
            <= self.tau_max_s
        ):
            raise ValueError("tau_init_range_s must lie in (tau_min_s, tau_max_s]")
        if not self.odor_K or any(not _is_finite_number(k) or k <= 0 for k in self.odor_K):
            raise ValueError("odor_K must contain positive, finite baselines")
        if self.gain_parameterization not in ("softplus", "bounded"):
            raise ValueError("gain_parameterization must be softplus or bounded")
        if self.gain_parameterization == "bounded" and self.gain_init >= self.gain_max:
            raise ValueError("bounded gain requires gain_init < gain_max")
        if not _is_finite_number(self.bias_lr_scale) or self.bias_lr_scale < 0:
            raise ValueError("bias_lr_scale must be nonnegative and finite")
        for key in ("readout_init_weight", "log_std_init"):
            if not _is_finite_number(getattr(self, key)):
                raise ValueError(f"{key} must be finite")

    @classmethod
    def from_dict(cls, values: Mapping) -> ModelConfigV2:
        unknown = values.keys() - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model config keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


def _is_finite_number(value) -> bool:
    return type(value) in (float, int) and math.isfinite(value)
