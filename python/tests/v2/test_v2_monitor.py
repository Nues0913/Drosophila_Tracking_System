"""The v2 training monitor: one record per update and fast failure (#131, plan 4.4)."""

import json

import pytest
import torch
from flytrack_v2.monitor import ENGINEERING_FAILURE_EXIT_CODE, EngineeringFailure, RunMonitor


def _records(path):
    return [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]


def test_update_reports_means_of_accumulated_steps_and_throughput(tmp_path):
    with RunMonitor(tmp_path, tensorboard=True) as monitor:
        monitor.accumulate(reward=torch.tensor([1.0, 3.0]), dn_rate=torch.ones(2, 3))
        monitor.accumulate(reward=torch.tensor([5.0, 7.0]))
        record = monitor.log_update(1, 512, {"loss": torch.tensor(0.25), "lr": 3e-4})
        monitor.accumulate(reward=torch.tensor([2.0]))
        second = monitor.log_update(2, 1024)
    assert record["reward"] == pytest.approx(4.0) and record["dn_rate"] == pytest.approx(1.0)
    assert record["loss"] == pytest.approx(0.25) and record["lr"] == pytest.approx(3e-4)
    assert record["env_steps_per_s"] > 0 and second["reward"] == pytest.approx(2.0)
    assert [r["update"] for r in _records(tmp_path)] == [1, 2]
    assert any((tmp_path / "tensorboard").iterdir())


def test_non_finite_loss_dumps_then_fails_with_exit_code_3(tmp_path):
    dumped = []
    monitor = RunMonitor(
        tmp_path, tensorboard=False, on_failure=lambda reason, record: dumped.append(reason)
    )
    with pytest.raises(EngineeringFailure, match="non-finite loss") as error:
        monitor.log_update(7, 100, {"loss": torch.tensor(float("nan"))})
    assert error.value.exit_code == ENGINEERING_FAILURE_EXIT_CODE == 3
    assert dumped and "update 7" in dumped[0]
    assert _records(tmp_path)[0]["update"] == 7  # the failing update is still on disk


def test_watch_stops_only_after_consecutive_breaches(tmp_path):
    monitor = RunMonitor(tmp_path, tensorboard=False)
    monitor.watch("approx_kl", threshold=0.08, patience=3)
    for update, kl in enumerate((0.1, 0.1, 0.01, 0.1, 0.1), start=1):
        monitor.log_update(update, update * 10, {"approx_kl": torch.tensor(kl)})
    with pytest.raises(EngineeringFailure, match="approx_kl above 0.08 for 3 updates"):
        monitor.log_update(6, 60, {"approx_kl": torch.tensor(0.2)})


def test_watch_rejects_bad_settings(tmp_path):
    monitor = RunMonitor(tmp_path, tensorboard=False)
    with pytest.raises(ValueError):
        monitor.watch("approx_kl", threshold=float("inf"), patience=3)
    with pytest.raises(ValueError):
        monitor.watch("approx_kl", threshold=0.1, patience=0)
    monitor.close()
