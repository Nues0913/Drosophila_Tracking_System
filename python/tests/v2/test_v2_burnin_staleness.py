"""CPU coverage of the burn-in analysis and report files."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from flytrack_v2.model import ModelConfigV2, build_policy, export_policy
from flytrack_v2.train.config import STATE_STRIDE
from flytrack_v2.train.data import ReplayShard, StepData, _env_major
from test_v2_model_core import small_graph

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/v2/burnin_staleness.py"
_spec = importlib.util.spec_from_file_location("burnin_staleness", SCRIPT)
burnin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(burnin)
CPU = torch.device("cpu")


def make_shard(code, reset, states, first):
    ticks, rows = reset.shape
    return ReplayShard(
        StepData(
            _env_major(code, torch.float16),
            torch.zeros(rows, ticks, 2),
            _env_major(reset, torch.bool),
            torch.zeros(rows, ticks, dtype=torch.bool),
        ),
        _env_major(states, torch.float16),
        torch.zeros(rows, 3, dtype=torch.uint8),
        torch.arange(rows) == 0,
        first,
    )


@pytest.fixture
@torch.no_grad()
def synthetic_run(tmp_path):
    policy = build_policy(
        "VO2", ModelConfigV2(rest_settle_s=0.2, rest_refresh_s=0.1), graph=small_graph()
    )
    policy.readout.rate_mean.fill_(0.06)
    policy.readout.rate_scale.copy_(torch.linspace(0.02, 0.05, len(policy.readout.rate_mean)))
    policy.readout.rate_norm_fitted.fill_(True)

    ticks, rows, first = 46, 3, 3
    rng = torch.Generator().manual_seed(13)
    # Replay code is exactly representable in float16, isolating snapshot rounding.
    code = torch.randint(-128, 129, (ticks, rows, policy.code_size), generator=rng) / 128
    code[..., -4:] = 1
    reset = torch.zeros(ticks, rows, dtype=torch.bool)
    reset[1, 0] = reset[14, 1] = reset[20, 2] = True
    initial = policy.initial_central_state(rows, CPU)
    state = initial.clone()
    states = []
    for tick in range(ticks):
        if tick in range(first, ticks, STATE_STRIDE):
            states.append(state.clone())
        state = policy.forward_code(code[tick], state).central_state
        state[reset[tick]] = initial[reset[tick]]
    shard = make_shard(code, reset, torch.stack(states), first)

    run = tmp_path / "run"
    for number in (1, 2):
        directory = run / f"round_{number}"
        directory.mkdir(parents=True)
        torch.save(shard.checkpoint(), directory / "replay.pt")
    config = json.dumps({"variant": "VO2", "sequence": {"burn_in": 2}})
    (run / "config.json").write_text(config, encoding="utf-8")
    config_sha256 = hashlib.sha256(config.encode()).hexdigest()
    export_policy(
        policy,
        run / "final",
        provenance={"git_sha": None, "dirty": None, "config_sha256": config_sha256, "stages": []},
    )
    torch.save(
        {
            "config_sha256": config_sha256,
            "policy": policy.state_dict(),
            "rest_state": initial[:1],
        },
        run / "round_1" / "checkpoint.pt",
    )
    return run, policy, shard


def test_age_bins_and_reset_window():
    ticks, rows, first, window = 1060, 4, 2, 5
    reset = torch.zeros(ticks, rows, dtype=torch.bool)
    reset[1, 0] = reset[22, 0] = reset[1, 2] = True
    reset[23, 1] = True
    states = torch.zeros(len(range(first, ticks, STATE_STRIDE)), rows, 1)
    shard = make_shard(torch.zeros(ticks, rows, 1), reset, states, first)
    bins, counts = burnin.snapshot_bins(shard, window, first_round=False)
    sample = burnin.Snapshot
    for group, ages in enumerate(((0, 16), (32, 64), (80, 256), (272, 1024), (1040,))):
        for age in ages:
            assert sample(2, 2 + age, age) in bins[group]

    bins_first, _ = burnin.snapshot_bins(shard, window, first_round=True)
    assert sample(0, 18, 16) not in bins[0]  # Reset at t + window - 1.
    assert sample(0, 18, 16) in burnin.snapshot_bins(shard, window - 1, first_round=False)[0][0]
    assert sample(1, 18, 18) in bins_first[1]  # Reset at t + window is outside.
    assert sample(1, 34, 10) in bins[0]
    assert counts["excluded_reset"] == 1
    assert counts["excluded_short_window"] == rows
    assert counts["excluded_cross_round"] == 2 + len(range(first, ticks - window + 1, STATE_STRIDE))
    assert sum(map(len, bins)) == counts["snapshots"] - sum(
        value for key, value in counts.items() if key.startswith("excluded_")
    )


@torch.no_grad()
def test_reference_starts_after_reset_and_stops_before_snapshot_sense(synthetic_run):
    _, policy, shard = synthetic_run
    # Reset at 1, episode starts at 2, and the snapshot precedes sense at 3.
    snapshot = burnin.Snapshot(0, 3, 1)
    policy.core.bias.add_(0.075)
    initial = policy.initial_central_state(1, CPU)
    reference = policy.forward_code(shard.steps.code[0, 2:3].float(), initial).central_state
    stale = shard.states[0, 0:1].float()
    expected_0 = (stale - reference).square().mean().sqrt().item()
    for tick in (3, 4):
        code = shard.steps.code[0, tick : tick + 1].float()
        reference = policy.forward_code(code, reference).central_state
        stale = policy.forward_code(code, stale).central_state
    expected_2 = (stale - reference).square().mean().sqrt().item()

    actual, control = burnin.measure_cell(shard, [snapshot], policy, None, [2], CPU)
    assert actual[0]["state_rmse"] == pytest.approx(expected_0, rel=1e-5)
    assert actual[2]["state_rmse"] == pytest.approx(expected_2, rel=1e-5)
    assert control is None


@pytest.mark.parametrize("weights", ["final", "round_1/checkpoint.pt"])
def test_same_weights_cli_reports_float16_error(synthetic_run, tmp_path, weights):
    run, _, _ = synthetic_run
    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--run",
            str(run),
            "--weights",
            str(run / weights),
            "--device",
            "cpu",
            "--per-cell",
            "2",
            "--steps",
            "2",
            "4",
            "--out",
            str(out),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    report = json.loads((out / "burnin.json").read_text())
    assert report["steps"] == [2, 4]
    assert report["required_reset_free_steps"] == 6
    assert report["dn_normalization_fitted"] is True
    assert report["pm_trigger"] == {
        "step": 16,
        "threshold_sigma": 0.3,
        "samples": 0,
        "mean_dn_rmse_sigma": None,
        "exceeds_threshold": None,
    }
    first, second = report["rounds"]
    assert first["collection_checkpoint"] is None
    assert second["collection_checkpoint"] == str(run / "round_1" / "checkpoint.pt")
    assert second["excluded_cross_round"] > 0
    for round_result in (first, second):
        for cell in round_result["cells"]:
            assert cell["samples"] == min(cell["eligible"], 2)
            metrics = cell["measurements"]
            assert list(metrics) == ["0", "2", "4"]
            assert set(metrics["2"]) == {
                "samples",
                "state_rmse",
                "state_rmse_fraction_of_step_0",
                "dn_rmse_sigma",
                "dn_mae_sigma",
                "speed_mode_mae_m_s",
                "turn_mode_mae_rad_s",
            }
            if not cell["samples"]:
                assert metrics["2"]["state_rmse"] is None
                continue
            before, after_2, after_4 = [metrics[str(n)]["state_rmse"] for n in (0, 2, 4)]
            assert 0 < before < 1e-4
            assert 0 < after_4 <= after_2 <= before
            assert metrics["4"]["state_rmse_fraction_of_step_0"] == pytest.approx(after_4 / before)
            if round_result is second:
                assert cell["collection_state_rmse"] == pytest.approx(before, abs=1e-8)
    markdown = (out / "burnin.md").read_text()
    assert "**Mean DN readout RMSE after 16 steps across all samples: unavailable " in markdown
    assert "Turn MAE (rad/s)" in markdown


def test_pm_trigger_weights_each_snapshot_equally(synthetic_run):
    run, _, shard = synthetic_run
    shard.states[0].add_(2)
    torch.save(shard.checkpoint(), run / "round_2" / "replay.pt")
    report = burnin.probe(
        Namespace(run=run, weights=run / "final", device="cpu", seed=0, per_cell=2, steps=[16])
    )
    measurements = [
        cell["measurements"][16]
        for row in report["rounds"]
        for cell in row["cells"]
        if cell["samples"]
    ]
    count = sum(values["samples"] for values in measurements)
    mean = sum(values["dn_rmse_sigma"] * values["samples"] for values in measurements) / count
    pm = report["pm_trigger"]
    assert {values["samples"] for values in measurements} == {1, 2}
    assert pm["samples"] == count
    assert pm["mean_dn_rmse_sigma"] == pytest.approx(mean)
    assert pm["exceeds_threshold"] is True
