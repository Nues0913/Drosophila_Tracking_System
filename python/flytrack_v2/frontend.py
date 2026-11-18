"""Frozen flyvis front end with a node-major state (WP3b, #135; plan 3.4).

v1's FrozenFlyvis keeps its state batch-major [B, 2N] and transposes to node-major [N, 2B]
and back on every control step, because CSR products want a contiguous node-major operand.
On the 2060 those two transposes took 19 of 67 ms per step at batch 256. Here the state stays
node-major between steps (the policy owns its sensory state, docs/v2/interfaces.md section 5),
and on CUDA the per-substep update runs as one fused kernel (torch.compile). Together: 1.9x at
batch 256 and 1024, within 2e-6 of v1. Column j of the state is environment j // 2, eye j % 2
(left 0, right 1), the same order as v1's reshape(batch * 2, ...).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from flytrack_core.contracts import SensorContract
from flytrack_core.model.frontend import FrozenFlyvis
from torch import Tensor, nn


def _update(voltage, recurrent, current, ratio, bias, columns):
    evolved = (bias - voltage + recurrent + current) * ratio + voltage
    voltage = torch.where(columns, evolved, voltage)
    return voltage, torch.relu(voltage)


_fused_update = torch.compile(_update, dynamic=False)
# Why the fused update is unavailable on this host, once it has failed (#158 review: the
# Triton backend needs a C compiler, which 4070's WSL lacks). None until a failure.
_FUSED_FAILURE: str | None = None


def _run_update(fused: bool, *args):
    """The fused update, or the eager one after warning once that compilation failed."""
    global _FUSED_FAILURE
    if fused and _FUSED_FAILURE is None:
        try:
            return _fused_update(*args)
        except Exception as error:  # noqa: BLE001 - any compiler failure falls back
            _FUSED_FAILURE = f"{type(error).__name__}: {error}"[:300]
            warnings.warn(
                "Fused frontend update unavailable, falling back to eager (about half the "
                f"speed): {_FUSED_FAILURE}",
                RuntimeWarning,
                stacklevel=3,
            )
    return _update(*args)


class NodeMajorFlyvis(nn.Module):
    """The validated flyvis export (FrozenFlyvis checks provenance and parity) run node-major."""

    def __init__(
        self,
        path: str | Path,
        contract: SensorContract,
        visual_bins: int,
        fused: bool | None = None,
    ):
        super().__init__()
        self.frozen = FrozenFlyvis(path, contract, visual_bins)
        self.n_nodes = self.frozen.n_nodes
        self.n_ommatidia = self.frozen.n_ommatidia
        self.output_types = self.frozen.output_types
        self.features_per_eye = self.frozen.features_per_eye
        self.substeps = self.frozen.substeps
        self.metadata = self.frozen.metadata
        # None: fuse on CUDA, eager on CPU (no compiler toolchain needed there).
        self.fused = fused
        ratio = self.frozen.neural_dt / self.frozen.tau.clamp_min(self.frozen.neural_dt)
        self.register_buffer("ratio", ratio.unsqueeze(1), persistent=False)

    def update_path(self, device: torch.device | str) -> str:
        """fused or eager: which update this front end runs on device (log it per run)."""
        wants = torch.device(device).type == "cuda" if self.fused is None else self.fused
        return "fused" if wants and _FUSED_FAILURE is None else "eager"

    def initial_state(self, batch_size: int, device: torch.device | str) -> Tensor:
        return self.frozen.bias.to(device).unsqueeze(1).expand(-1, 2 * batch_size).clone()

    def reset(self, state: Tensor, reset_mask: Tensor) -> Tensor:
        """Columns of the environments in reset_mask [B] restart from the resting state."""
        columns = reset_mask.repeat_interleave(2).unsqueeze(0)
        return torch.where(columns, self.frozen.bias.unsqueeze(1), state)

    @torch.no_grad()
    def forward(self, vision: Tensor, state: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """vision [B,2,n_ommatidia], state [N,2B], valid [B,2] -> features [B,2,F], state."""
        batch = len(vision)
        if vision.shape != (batch, 2, self.n_ommatidia) or state.shape != (
            self.n_nodes,
            2 * batch,
        ):
            raise ValueError("Incorrect frontend observation/state shape")
        frozen = self.frozen
        available = (valid > 0).reshape(batch * 2, 1)
        image = torch.where(available, vision.reshape(batch * 2, -1), 0.0)
        current = state.new_zeros(state.shape)
        current.index_add_(0, frozen.input_node, image[:, frozen.input_ommatidium].T)
        columns = available.reshape(1, batch * 2)
        bias = frozen.bias.unsqueeze(1)
        fused = state.is_cuda if self.fused is None else self.fused
        voltage, rates = state, torch.relu(state)
        for _ in range(self.substeps):
            recurrent = torch.sparse.mm(frozen.weight, rates)
            voltage, rates = _run_update(
                fused, voltage, recurrent, current, self.ratio, bias, columns
            )
        features = torch.sparse.mm(frozen.pooling, rates).T
        features = torch.where(available, features, 0.0)
        return features.reshape(batch, 2, -1), voltage
