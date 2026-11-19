import json

import pytest
import torch
from flytrack_v2.artifacts import VARIANTS
from flytrack_v2.model import ModelConfigV2, build_policy, load_policy, parameter_groups
from flytrack_v2.monitor import RunMonitor
from flytrack_v2.train import dagger
from flytrack_v2.train.dagger import _normalize_device, build_training_policy, main
from flytrack_v2.train.data import SequenceBatch
from flytrack_v2.train.update import (
    LossSettings,
    batch_metrics,
    burn_in_state,
    make_scheduler,
    sequence_loss,
    train_update,
)
from test_v2_model_core import small_graph

SETTINGS = LossSettings(burn_in=2, activity_coef=1e-4, label_margin=1e-3)
OPTIMIZER = {"lr": 0.01, "lr_final": 0.001}


def test_normalize_cuda_device_without_a_gpu(monkeypatch):
    current_device_calls = []

    def current_device():
        current_device_calls.append(True)
        return 2

    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    assert _normalize_device("cuda") == torch.device("cuda:2")
    assert _normalize_device("cuda:0") == torch.device("cuda:0")
    assert _normalize_device("cpu") == torch.device("cpu")
    assert len(current_device_calls) == 1


def _policy(variant):
    cfg = ModelConfigV2(
        initial_state="bias",
        structure_seed=3,
        budget_target=12_000 if variant == "GRU2" else None,
    )
    return build_policy(variant, cfg, graph=small_graph())


def _batch(policy, *, dtype=torch.float32):
    rng = torch.Generator().manual_seed(12)
    ticks, rows = 6, 4
    code = 0.5 * torch.randn(ticks, rows, policy.code_size, generator=rng, dtype=dtype)
    code[..., -4:] = 1
    code[:, 0, -4:] = 0  # missing-sensor biases get supervision too
    action = torch.rand(ticks, rows, 2, generator=rng, dtype=dtype) * 0.35 + 0.1
    action[..., 1] -= 0.7
    reset = torch.zeros(ticks, rows, dtype=torch.bool)
    reset[1, 0] = True
    reset[3, 1] = True
    return SequenceBatch(
        code,
        action,
        reset,
        policy.initial_central_state(rows, "cpu"),
        torch.zeros(rows, 3, dtype=torch.uint8),
        torch.zeros_like(reset),
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_each_trainable_group_has_nonzero_gradient(variant):
    policy = _policy(variant)
    parameters = {id(p): name for name, p in policy.named_parameters()}
    batch_metrics(policy, _batch(policy), SETTINGS, 2, backward=True)
    for group in parameter_groups(policy, 1.0):
        for param in group["params"]:
            name = parameters[id(param)]
            if variant == "O2" and name == "projection.lc_rf":
                # O2's LC receptive field is counted by parameter_table but not output-reachable.
                assert param.grad is None
            else:
                assert param.grad is not None, name
                assert param.grad.isfinite().all() and param.grad.count_nonzero() > 0, name


def test_burn_in_does_not_create_an_autograd_path():
    policy = _policy("VO2")
    batch = _batch(policy)
    code = batch.code.requires_grad_()
    stored = batch.initial.requires_grad_()
    initial = policy.initial_central_state(1, "cpu")
    warmed = burn_in_state(policy, code[:2], batch.reset[:2], stored, initial)
    assert warmed.grad_fn is None and not warmed.requires_grad
    assert not torch.equal(
        warmed, burn_in_state(policy, torch.zeros_like(code[:2]), batch.reset[:2], stored, initial)
    )
    sequence_loss(policy, batch, initial, SETTINGS)["loss"].backward()
    assert stored.grad is None
    assert code.grad[:2].count_nonzero() == 0
    assert code.grad[2:].count_nonzero() > 0


def test_two_micro_batches_have_equivalent_gradients():
    policy = _policy("VO2").double()
    batch = _batch(policy, dtype=torch.float64)
    batch_metrics(policy, batch, SETTINGS, 4, backward=True)
    full = {name: p.grad.clone() for name, p in policy.named_parameters() if p.grad is not None}
    policy.zero_grad(set_to_none=True)
    batch_metrics(policy, batch, SETTINGS, 2, backward=True)
    errors = []
    for name, param in policy.named_parameters():
        torch.testing.assert_close(param.grad, full[name], atol=1e-12, rtol=1e-10)
        errors.append(float((param.grad - full[name]).abs().max()))
    print(f"maximum absolute gradient difference = {max(errors):.6g}")


def test_can_overfit_one_batch():
    policy = _policy("GRU2")
    batch = _batch(policy)
    optimizer = torch.optim.Adam(parameter_groups(policy, OPTIMIZER["lr"]))
    updates = 70
    scheduler = make_scheduler(optimizer, OPTIMIZER, updates)
    with torch.no_grad():
        before = batch_metrics(policy, batch, SETTINGS, 2, backward=False)["nll"].item()
    for _ in range(updates):
        metrics = train_update(policy, optimizer, scheduler, batch, SETTINGS, 2, 1.0)
    with torch.no_grad():
        after = batch_metrics(policy, batch, SETTINGS, 2, backward=False)["nll"].item()
    print(f"overfit NLL = {before:.6f} -> {after:.6f}")
    assert metrics["lr"] == pytest.approx(OPTIMIZER["lr_final"])
    assert after < 0.5 * before


def test_nonfinite_loss_skips_optimizer_and_scheduler():
    policy = _policy("GRU2")
    batch = _batch(policy)
    batch.action[2, 0, 0] = float("nan")
    optimizer = torch.optim.Adam(parameter_groups(policy, OPTIMIZER["lr"]))
    scheduler = make_scheduler(optimizer, OPTIMIZER, 2)
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    result = train_update(policy, optimizer, scheduler, batch, SETTINGS, 2, 1.0)
    assert not result["grad_norm"].isfinite()
    assert not optimizer.state
    assert scheduler.last_epoch == 0
    for name, value in policy.state_dict().items():
        assert torch.equal(before[name], value)


@pytest.mark.slow
@torch.no_grad()
def test_bundled_fitted_input_norm_calibrates_resting_speed():
    policy = build_training_policy("VO2", 0, activation_checkpoint=False)
    assert policy.input_norm.fitted and policy.input_norm.n_samples > 0
    state = policy.initial_central_state(1, "cpu")
    speed = policy._distribution(policy._readout_mean(state)).mode()[0, 0]
    assert speed.item() == pytest.approx(0.20, abs=1e-5)
    code = policy.input_norm.neutral_code()
    maximum_drift = 0.0
    for _ in range(round(8 / policy.contract.control_dt)):
        step = policy.forward_code(code, state)
        maximum_drift = max(maximum_drift, abs(step.dist.mode()[0, 0].item() - 0.20))
        state = step.central_state
    print(f"rest speed = {speed.item():.8f}, max drift = {maximum_drift:.6g} m/s")
    assert maximum_drift <= 0.01


@pytest.mark.slow
def test_stop_mid_round_resume_matches_uninterrupted_run_and_loads_export(tmp_path, monkeypatch):
    uninterrupted, resumed = tmp_path / "full", tmp_path / "resumed"
    args = [
        "--variant",
        "GRU2",
        "--seed",
        "0",
        "--host",
        "4060",
        "--device",
        "cpu",
        "--override",
        "rounds.beta=[1.0,0.5]",
        "envs_per_instance=2",
        "episode_seconds=0.8",
        "desync_first_episode=true",
        "rounds.env_steps_per_round=72",
        "replay_ratio=12",
        "batch_sequences=4",
        "sequence.length=4",
        "sequence.burn_in=2",
        "held_out.sequences=4",
        "held_out.every_updates=10",
    ]
    assert main([*args, "--out", str(uninterrupted)]) == 0
    log_update = RunMonitor.log_update

    def stop_after_50(monitor, update, *args, **kwargs):
        record = log_update(monitor, update, *args, **kwargs)
        if update == 50:
            (resumed / "STOP").touch()
        return record

    with monkeypatch.context() as patch:
        patch.setattr(RunMonitor, "log_update", stop_after_50)
        assert main([*args, "--out", str(resumed)]) == 0

    partial = torch.load(resumed / "round_1/progress.pt", weights_only=True)
    assert partial["progress"]["round_updates"] == 50
    assert partial["progress"]["round_index"] == 0
    schedule = partial["collector"]["first_episode_resets"]
    assert schedule["forced"].any()
    assert (schedule["steps"] > schedule["ticks"]).any()
    assert not (resumed / "final").exists()
    (resumed / "STOP").unlink()
    assert main([*args, "--out", str(resumed), "--resume"]) == 0

    full = torch.load(uninterrupted / "round_2/checkpoint.pt", weights_only=True)
    continued = torch.load(resumed / "round_2/checkpoint.pt", weights_only=True)
    assert full["progress"] == continued["progress"]
    assert full["scheduler"] == continued["scheduler"]
    for key in full["policy"]:
        assert torch.equal(full["policy"][key], continued["policy"][key]), key
    for key in ("train_rng", "held_rng", "torch_rng"):
        assert torch.equal(full[key], continued[key]), key
    torch.testing.assert_close(
        full["collector"], continued["collector"], rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(full["optimizer"], continued["optimizer"], rtol=0, atol=0)
    final_schedule = continued["collector"]["first_episode_resets"]
    assert torch.equal(final_schedule["steps"], schedule["steps"])
    assert torch.equal(final_schedule["forced"], schedule["steps"] > 0)
    records = [
        row
        for row in map(json.loads, (resumed / "monitor.jsonl").read_text().splitlines())
        if "loss" in row
    ]
    assert [row["update"] for row in records] == list(range(1, full["progress"]["updates"] + 1))
    assert records[49]["env_steps"] == records[50]["env_steps"]
    assert records[50]["env_steps_per_s"] == 0
    assert any("held_out_nll" in row for row in records)
    phase_records = [row for row in records if "episode_time_mean_s" in row]
    assert [row["round"] for row in phase_records] == [1, 2]
    ages = torch.zeros_like(schedule["steps"]).flatten()
    for row in phase_records:
        name = f"round_{int(row['round'])}/replay.pt"
        replay = torch.load(resumed / name, weights_only=True)
        baseline = torch.load(uninterrupted / name, weights_only=True)
        torch.testing.assert_close(replay, baseline, rtol=0, atol=0)
        assert "episode_steps" not in replay["steps"]
        samples = []
        for reset in replay["steps"]["reset"]:
            samples.append(ages)
            ages = torch.where(reset, 0, ages + 1)
        std, mean = torch.std_mean(torch.stack(samples).double(), correction=0)
        assert row["episode_time_mean_s"] == pytest.approx(mean.item() * 0.05)
        assert row["episode_time_std_s"] == pytest.approx(std.item() * 0.05)
    loaded = load_policy(resumed / "final")
    assert loaded.input_norm.fitted
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, continued["policy"][key]), key
    card = json.loads((resumed / "final/model_card.json").read_text())
    (stage,) = card["provenance"]["stages"]
    assert stage["name"] == "dagger"
    assert stage["micro_batch"] == 4
    assert stage["host"] == "4060"
    assert stage["activation_checkpoint"] is False
    assert stage["desync_first_episode"] is True


@pytest.mark.parametrize("silent_round", [None, 1, 4])
def test_round_health_warning_and_stage_failure(tmp_path, monkeypatch, silent_round):
    probes = 0

    def health_metrics(*args):
        nonlocal probes
        probes += 1
        silent = probes == silent_round
        return {
            "health_speed_std_m_s": torch.tensor(0.03 if probes == 1 or silent else 0.02),
            "health_odor_swap_delta_omega": torch.tensor(0.1),
            "health_dn_min_mean_rate": torch.tensor(0.0 if silent else 0.1),
            "health_dn_silent": torch.tensor(int(silent)),
        }

    monkeypatch.setattr(dagger, "health_metrics", health_metrics)
    args = [
        "--variant",
        "GRU2",
        "--seed",
        "0",
        "--host",
        "4060",
        "--device",
        "cpu",
        "--out",
        str(tmp_path),
        "--override",
        "rounds.beta=[1.0,0.5,0.25,0.0]",
        "envs_per_instance=2",
        "rounds.env_steps_per_round=72",
        "replay_ratio=0.01",
        "batch_sequences=4",
        "sequence.length=4",
        "sequence.burn_in=2",
        "held_out.sequences=4",
    ]
    if silent_round == 1:
        assert main(args) == 3
    else:
        with pytest.warns(UserWarning, match=r"round 2.*speed std") as caught:
            assert main(args) == 3
        assert len(caught) == 1

    records = [json.loads(line) for line in (tmp_path / "monitor.jsonl").read_text().splitlines()]
    round_ends = [row for row in records if row.get("round_end")]
    updates = [row for row in records if "loss" in row]
    expected_rounds = [1] if silent_round == 1 else [1, 2, 3, 4]
    assert [row["round"] for row in round_ends] == expected_rounds
    assert [row["round"] for row in updates] == expected_rounds
    assert [row["update"] for row in round_ends] == [row["update"] for row in updates]
    assert all("health_odor_swap_delta_omega" in row for row in round_ends)
    assert [row["health_failure"] for row in round_ends] == [0] * (len(expected_rounds) - 1) + [1]

    failure = json.loads((tmp_path / "failure/reason.json").read_text())
    assert failure["metrics"] == round_ends[-1]
    assert (tmp_path / "failure/checkpoint.pt").is_file()
    assert (tmp_path / "failure/batch.pt").is_file()
    if silent_round is not None:
        assert "silent readout DNs" in failure["reason"]
    if silent_round == 1:
        assert not (tmp_path / "final").exists()
        return

    if silent_round is None:
        assert "speed std is <= 0.02 m/s" in failure["reason"]
    assert "Exported to final/. Stage 1 health check failed." in failure["reason"]
    checkpoint = torch.load(tmp_path / "round_4/checkpoint.pt", weights_only=True)
    assert checkpoint["progress"]["round_index"] == 4
    assert checkpoint["progress"]["updates"] == 4
    loaded = load_policy(tmp_path / "final")
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, checkpoint["policy"][key]), key

    assert main([*args, "--resume"]) == 3
    assert probes == 4
    assert json.loads((tmp_path / "failure/reason.json").read_text()) == failure
