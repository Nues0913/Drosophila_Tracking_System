"""Training monitor for v2 (plan 4.4, #131): one JSONL line and one TensorBoard step per update.

Per-step values accumulate on their device; log_update() moves everything to the host once
per update, so the hot loop never synchronises (plan 0.2). A tracked value that is not finite,
or a watched value above its threshold for too many consecutive updates, calls the trainer's
on_failure hook (save the batch and a checkpoint there) and raises EngineeringFailure, whose
exit code 3 marks an engineering failure for the experiment runner.

Two duties stay with the trainer (#150 review):
- check that the gradient norm is finite before optimizer.step() and skip the step if it is
  not; the monitor sees the update afterwards, so on_failure must save the last good
  checkpoint, never the parameters a non-finite step just wrote;
- on resume, pass the checkpoint's env step count as start_env_steps. Watch streaks are not
  checkpointed and restart from 0 after a resume.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

ENGINEERING_FAILURE_EXIT_CODE = 3


class EngineeringFailure(RuntimeError):
    exit_code = ENGINEERING_FAILURE_EXIT_CODE


@dataclass
class _Watch:
    threshold: float
    patience: int
    streak: int = 0


class RunMonitor:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        finite_keys: tuple[str, ...] = ("loss", "grad_norm"),
        on_failure: Callable[[str, dict], None] | None = None,
        tensorboard: bool = True,
        start_env_steps: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.finite_keys = finite_keys
        self.on_failure = on_failure
        self._jsonl = (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8")
        self._writer = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        self._sums: dict[str, Tensor] = {}
        self._counts: dict[str, Tensor] = {}
        self._watches: dict[str, _Watch] = {}
        self._start = self._last_time = time.perf_counter()
        # A resumed run starts counting from its checkpoint, or the first throughput would
        # divide every step so far by the seconds since this restart.
        self._last_steps = int(start_env_steps)

    def accumulate(self, **values: Tensor) -> None:
        """Add per-step values of any shape; log_update() reports their mean. No host sync."""
        for name, value in values.items():
            value = value.detach().float()
            total, count = value.sum(), value.new_tensor(float(value.numel()))
            if name in self._sums:
                self._sums[name] = self._sums[name] + total
                self._counts[name] = self._counts[name] + count
            else:
                self._sums[name], self._counts[name] = total, count

    def watch(self, name: str, threshold: float, patience: int) -> None:
        """Stop when ``name`` exceeds ``threshold`` for ``patience`` consecutive updates
        (plan 4.4: approx KL above 4 x target three times in a row)."""
        if patience < 1 or not math.isfinite(threshold):
            raise ValueError("watch needs a finite threshold and patience >= 1")
        self._watches[name] = _Watch(float(threshold), int(patience))

    def log_update(
        self, update: int, env_steps: int, scalars: Mapping[str, Tensor | float] | None = None
    ) -> dict:
        names: list[str] = []
        tensors: list[Tensor] = []
        record: dict = {}
        for name, value in (scalars or {}).items():
            if isinstance(value, Tensor):
                names.append(name)
                tensors.append(value.detach().float().reshape(()))
            else:
                record[name] = float(value)
        for name, total in self._sums.items():
            names.append(name)
            tensors.append(total / self._counts[name])
        self._sums, self._counts = {}, {}
        record.update(self._to_host(names, tensors))
        now = time.perf_counter()
        steps_per_s = (env_steps - self._last_steps) / max(now - self._last_time, 1e-9)
        self._last_time, self._last_steps = now, env_steps
        record = {
            "update": int(update),
            "env_steps": int(env_steps),
            "wall_s": now - self._start,
            "env_steps_per_s": steps_per_s,
            **record,
        }
        self._write(record)
        self._check(record)
        return record

    @staticmethod
    def _to_host(names: list[str], tensors: list[Tensor]) -> dict[str, float]:
        """One transfer per device, however many values were logged."""
        values: dict[str, float] = {}
        by_device: dict[torch.device, list[int]] = {}
        for index, tensor in enumerate(tensors):
            by_device.setdefault(tensor.device, []).append(index)
        for indices in by_device.values():
            host = torch.stack([tensors[index] for index in indices]).cpu().tolist()
            values.update({names[index]: value for index, value in zip(indices, host, strict=True)})
        return values

    def _write(self, record: dict) -> None:
        line = json.dumps(record, allow_nan=True)
        self._jsonl.write(line + "\n")
        self._jsonl.flush()
        if self._writer is not None:
            for name, value in record.items():
                if name not in ("update", "env_steps") and math.isfinite(value):
                    self._writer.add_scalar(name, value, record["env_steps"])

    def _fail(self, reason: str, record: dict) -> None:
        if self.on_failure is not None:
            self.on_failure(reason, record)
        self.close()
        raise EngineeringFailure(reason)

    def _check(self, record: dict) -> None:
        bad = [name for name in self.finite_keys if name in record]
        bad = [name for name in bad if not math.isfinite(record[name])]
        if bad:
            self._fail(f"non-finite {', '.join(bad)} at update {record['update']}", record)
        for name, watch in self._watches.items():
            value = record.get(name)
            watch.streak = watch.streak + 1 if value is not None and value > watch.threshold else 0
            if watch.streak >= watch.patience:
                self._fail(
                    f"{name} above {watch.threshold} for {watch.patience} updates "
                    f"(update {record['update']})",
                    record,
                )

    def close(self) -> None:
        if not self._jsonl.closed:
            self._jsonl.close()
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> RunMonitor:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
