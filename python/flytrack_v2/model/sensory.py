"""Frozen sensory dynamics and the trainable projection from cached codes to seeds."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from flytrack_core.contracts import GROUP_OLFACTORY, GROUP_VISUAL, GROUP_WIND
from torch import Tensor, nn

from flytrack_v2.contracts import VALID_CHANNELS, SensorContractV2, validate_obs
from flytrack_v2.frontend import NodeMajorFlyvis
from flytrack_v2.interfaces import reset_rows

from .config import ModelConfigV2
from .core import inverse_softplus
from .graph import GraphV2


@dataclass(frozen=True)
class CodeLayout:
    """Left eye, right eye, left/right ORNs, left/right wind, then the four valid flags."""

    features_per_eye: int = 144
    odor_channels: int = 1

    def __post_init__(self) -> None:
        for value in (self.features_per_eye, self.odor_channels):
            if type(value) is not int or value <= 0:
                raise ValueError("Code layout widths must be positive integers")

    @property
    def widths(self) -> tuple[int, int, int, int]:
        return self.features_per_eye, self.features_per_eye, 2 * self.odor_channels, 2

    @property
    def code_size(self) -> int:
        return sum(self.widths) + len(VALID_CHANNELS)


class SensoryState(NamedTuple):
    frontend: Tensor
    odor_K: Tensor


class SensoryEncoderV2(nn.Module):
    """Run only the frozen front end, ORN adaptation and observation masks."""

    def __init__(
        self,
        frontend: NodeMajorFlyvis,
        contract: SensorContractV2,
        config: ModelConfigV2,
        variant: str,
    ):
        super().__init__()
        self.frontend = frontend
        self.contract = contract
        self.layout = CodeLayout(frontend.features_per_eye, len(config.odor_K))
        self.hill_n = config.odor_hill_n
        self.adaptation_ratio = min(contract.control_dt / config.odor_adaptation_tau_s, 1)
        self.shared_adaptation = config.odor_shared_adaptation
        self.register_buffer(
            "odor_baseline", torch.tensor(config.odor_K, dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "enabled",
            torch.tensor((variant != "O2", variant != "O2", variant != "V2", variant != "V2")),
            persistent=False,
        )

    def initial_state(self, batch_size: int, device: torch.device | str) -> SensoryState:
        return SensoryState(
            self.frontend.initial_state(batch_size, device),
            self.odor_baseline.to(device).expand(batch_size, 2, -1).clone(),
        )

    @torch.no_grad()
    def reset(self, state: SensoryState, mask: Tensor) -> SensoryState:
        return SensoryState(
            self.frontend.reset(state.frontend, mask),
            reset_rows(state.odor_K, self.odor_baseline, mask),
        )

    @torch.no_grad()
    def sense(self, obs: dict[str, Tensor], state: SensoryState) -> tuple[Tensor, SensoryState]:
        batch = validate_obs(obs, self.contract)
        vision, odor, wind, flags = (obs[key] for key in ("vision", "odor", "wind", "valid"))
        if state.odor_K.shape != (batch, 2, self.layout.odor_channels):
            raise ValueError("Incorrect ORN state shape")
        finite = torch.cat(
            (
                vision.isfinite().all(-1),
                odor.isfinite().all(-1, keepdim=True),
                wind.isfinite().all(-1, keepdim=True),
            ),
            dim=1,
        )
        valid = (flags > 0.5) & flags.isfinite() & finite & self.enabled
        vision = torch.where(valid[:, :2, None], vision, 0.0)
        visual, frontend_state = self.frontend(vision, state.frontend, valid[:, :2])

        odor_valid = valid[:, 2, None, None]
        concentration = torch.where(valid[:, 2, None], odor.clamp_min(0), 0.0).unsqueeze(-1)
        k = state.odor_K.clamp_min(1e-4)
        # The log-domain form of C^n / (C^n + K^n) stays finite at large C.
        response = torch.sigmoid(self.hill_n * (concentration.log() - k.log()))
        target = (
            (concentration * 0.5).sum(1, keepdim=True) if self.shared_adaptation else concentration
        )
        evolved_k = (k + self.adaptation_ratio * (target - k)).clamp_min(1e-4)
        new_k = torch.where(odor_valid, evolved_k, state.odor_K)

        wind = torch.where(valid[:, 3, None], wind, 0.0)
        code = torch.cat(
            (visual.flatten(1), response.flatten(1), wind, valid.to(vision.dtype)), dim=1
        )
        return code, SensoryState(frontend_state, new_k)


class InputNorm(nn.Module):
    """Fixed population mean and standard deviation for each non-flag code column."""

    def __init__(
        self,
        layout: CodeLayout | None = None,
        std_floor: float = 1e-3,
        *,
        mean: Tensor | None = None,
        std: Tensor | None = None,
        fitted: bool = False,
        n_samples: int = 0,
    ):
        super().__init__()
        layout = CodeLayout() if layout is None else layout
        self.layout = layout
        self.std_floor = std_floor
        self.fitted = fitted
        self.n_samples = n_samples
        size = sum(layout.widths)
        self.register_buffer(
            "mean", torch.zeros(size) if mean is None else mean.detach().clone(), persistent=False
        )
        self.register_buffer(
            "std", torch.ones(size) if std is None else std.detach().clone(), persistent=False
        )
        if self.mean.shape != (size,) or self.std.shape != (size,):
            raise ValueError("Input norm statistics do not match code_layout")
        if not (self.mean.isfinite().all() and self.std.isfinite().all() and (self.std >= 0).all()):
            raise ValueError("Input norm statistics must be finite with nonnegative std")
        if not np.isfinite(std_floor) or std_floor <= 0:
            raise ValueError("std_floor must be positive and finite")
        if type(n_samples) is not int or n_samples < 0 or (fitted and n_samples == 0):
            raise ValueError("Invalid input norm sample count")
        self.register_buffer(
            "validity_index",
            torch.arange(len(layout.widths)).repeat_interleave(torch.tensor(layout.widths)),
            persistent=False,
        )

    def neutral_code(self) -> Tensor:
        """Return one raw code with zero normalized features and all channels valid."""
        return torch.cat((self.mean, self.mean.new_ones(len(VALID_CHANNELS)))).unsqueeze(0)

    def forward(self, code: Tensor, enabled: Tensor | None = None) -> Tensor:
        if code.ndim != 2 or code.shape[1] != self.layout.code_size:
            raise ValueError(f"Expected sensory codes with shape [B, {self.layout.code_size}]")
        raw, flags = code.split((sum(self.layout.widths), len(VALID_CHANNELS)), dim=1)
        if enabled is not None:
            flags = torch.where(enabled, flags, 0.0)
        valid = ((flags > 0.5) & flags.isfinite()).index_select(1, self.validity_index)
        normalized = torch.where(valid, (raw - self.mean) / self.std.clamp_min(self.std_floor), 0.0)
        return torch.cat((normalized, flags), dim=1)

    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as stream:
            np.savez(
                stream,
                mean=self.mean.cpu().numpy(),
                std=self.std.cpu().numpy(),
                code_layout=json.dumps(asdict(self.layout)),
                fitted=self.fitted,
                n_samples=self.n_samples,
            )

    @classmethod
    def load(cls, path: str | Path, std_floor: float = 1e-3) -> InputNorm:
        with np.load(path, allow_pickle=False) as archive:
            layout = CodeLayout(**json.loads(archive["code_layout"].item()))
            return cls(
                layout,
                std_floor,
                mean=torch.from_numpy(archive["mean"]),
                std=torch.from_numpy(archive["std"]),
                fitted=bool(archive["fitted"].item()),
                n_samples=int(archive["n_samples"].item()),
            )


@torch.no_grad()
def fit_input_norm(
    codes: Tensor,
    layout: CodeLayout | None = None,
    std_floor: float = 1e-3,
) -> InputNorm:
    """Fit population statistics on the valid rows of each modality independently.

    A channel with zero valid rows retains identity statistics. With the default frontend,
    the number of ORN channels can be inferred from the code width.
    """
    if codes.ndim != 2 or not len(codes) or not codes.is_floating_point():
        raise ValueError("codes must be a nonempty floating point [T, code_size] tensor")
    if layout is None:
        features = CodeLayout().features_per_eye
        odor_width = codes.shape[1] - 2 * features - 2 - len(VALID_CHANNELS)
        if odor_width < 2 or odor_width % 2:
            raise ValueError("Cannot infer the code layout; supply layout explicitly")
        layout = CodeLayout(features, odor_width // 2)
    if codes.shape[1] != layout.code_size:
        raise ValueError(f"Expected sensory codes with shape [T, {layout.code_size}]")
    *modalities, flags = codes.split((*layout.widths, len(VALID_CHANNELS)), dim=1)
    mean, std = [], []
    for channel, values in enumerate(modalities):
        flag = flags[:, channel]
        valid = (flag > 0.5) & flag.isfinite()
        samples = values[valid]
        if len(samples):
            variance, center = torch.var_mean(samples, dim=0, correction=0)
            mean.append(center)
            std.append(variance.sqrt())
        else:
            mean.append(values.new_zeros(values.shape[1]))
            std.append(values.new_ones(values.shape[1]))
    return InputNorm(
        layout,
        std_floor,
        mean=torch.cat(mean),
        std=torch.cat(std),
        fitted=True,
        n_samples=len(codes),
    ).to(codes.device)


class SeedProjection(nn.Module):
    """Pool per-eye features into type/side groups, then send currents only to sensory seeds."""

    def __init__(self, graph: GraphV2, layout: CodeLayout, enable_vision: bool = True):
        super().__init__()
        self.n_nodes = graph.n_nodes
        self.layout = layout
        self.enable_vision = enable_vision
        modalities = graph.input_mask & (GROUP_VISUAL | GROUP_OLFACTORY | GROUP_WIND)
        nodes = np.flatnonzero(modalities)
        seed_types = graph.type_index[nodes]
        _, seed_type_slot = np.unique(seed_types, return_inverse=True)
        sides = graph.side[nodes]
        seed_modalities = modalities[nodes]
        if not np.isin(seed_modalities, (GROUP_VISUAL, GROUP_OLFACTORY, GROUP_WIND)).all():
            raise ValueError("Each sensory seed must have one modality")
        visual = seed_modalities == GROUP_VISUAL
        odor = seed_modalities == GROUP_OLFACTORY
        wind = seed_modalities == GROUP_WIND
        eye = (sides == 1).astype(np.int64)
        if (visual & (sides == 0)).any():
            raise ValueError("Visual seeds require a left or right side")
        rf_pairs, rf_slot = np.unique(
            np.column_stack((seed_types[visual], eye[visual])),
            axis=0,
            return_inverse=True,
        )
        rf_count = len(rf_pairs)
        self.lc_rf = (
            nn.Parameter(
                inverse_softplus(
                    torch.full((rf_count, layout.features_per_eye), 1.0 / layout.features_per_eye)
                )
            )
            if rf_count
            else None
        )
        self.missing_bias = nn.Parameter(torch.zeros(len(np.unique(seed_types))))

        # Sources are [LC groups, left ORNs, right ORNs, midline ORNs, wind L/R/mean].
        c = layout.odor_channels
        _, odor_type_slot = np.unique(seed_types[odor], return_inverse=True)
        side_slot = np.where(sides == 0, 2, eye)
        source = np.empty(len(nodes), dtype=np.int64)
        source[visual] = rf_slot
        source[odor] = rf_count + c * side_slot[odor] + odor_type_slot % c
        source[wind] = rf_count + 3 * c + side_slot[wind]
        validity = np.empty(len(nodes), dtype=np.int64)
        validity[visual], validity[odor], validity[wind] = eye[visual], 2, 3
        for name, values in (
            ("seed_indices", nodes),
            ("seed_types", seed_type_slot),
            ("seed_validity", validity),
            ("source_index", source),
            ("rf_eye", rf_pairs[:, 1]),
        ):
            self.register_buffer(name, torch.as_tensor(values), persistent=False)

    def forward(self, code: Tensor) -> Tensor:
        eye_width = self.layout.features_per_eye
        vision, odor, wind, valid = code.split(
            (2 * eye_width, 2 * self.layout.odor_channels, 2, 4), dim=1
        )
        odor_midline = odor.reshape(len(code), 2, self.layout.odor_channels).mean(dim=1)
        sources = [odor, odor_midline, wind, wind.mean(dim=1, keepdim=True)]
        if self.lc_rf is not None:
            if self.enable_vision:
                pooled = vision.reshape(len(code), 2, eye_width) @ F.softplus(self.lc_rf).T
                groups = torch.arange(len(self.rf_eye), device=code.device)
                visual_drive = pooled[:, self.rf_eye, groups]
            else:
                visual_drive = code.new_zeros(len(code), len(self.rf_eye))
            sources.insert(0, visual_drive)
        missing = valid.index_select(1, self.seed_validity)
        missing = ~((missing > 0.5) & missing.isfinite())
        seed_drive = torch.cat(sources, dim=1).index_select(1, self.source_index)
        seed_drive = seed_drive + missing * self.missing_bias[self.seed_types]
        return code.new_zeros(len(code), self.n_nodes).index_copy(1, self.seed_indices, seed_drive)
