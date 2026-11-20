import json

import pytest
import torch
from flytrack_core.contracts import file_sha256
from flytrack_v2.artifacts import VARIANTS
from flytrack_v2.interfaces import ACTION_EPS, ActionDistribution, reset_rows, unsquash
from flytrack_v2.model import (
    ConnectomePolicyV2,
    ModelConfigV2,
    build_policy,
    load_policy,
    parameter_groups,
)
from flytrack_v2.monitor import RunMonitor
from flytrack_v2.train import dagger
from flytrack_v2.train.config import CONFIG_DIR, ROOT, load_config, read_yaml
from flytrack_v2.train.dagger import _normalize_device, build_training_policy, main
from flytrack_v2.train.data import SequenceBatch
from flytrack_v2.train.update import (
    LossSettings,
    _latent_log_prob_by_action,
    batch_metrics,
    burn_in_state,
    label_margins,
    make_scheduler,
    sequence_loss,
    shrink_labels,
    train_update,
)
from test_v2_model_core import small_graph

SETTINGS = LossSettings(burn_in=2, activity_coef=1e-4, label_margin=1e-3)
OPTIMIZER = {"lr": 0.01, "lr_final": 0.001}


def test_shrink_labels_respects_asymmetric_limits_and_preserves_interior():
    v_max, omega_max, margin = 0.4, 2.0, SETTINGS.label_margin
    action = torch.tensor(
        [[0, -omega_max], [v_max, omega_max], [0.3, -1.1], [-0.1, 0.1]],
        dtype=torch.float64,
    )
    shrunk = shrink_labels(action, v_max, omega_max, margin)
    latent = unsquash(shrunk, v_max, omega_max)
    expected = torch.tensor(
        [
            [margin, -(1 - margin)],
            [1 - margin, 1 - margin],
        ],
        dtype=action.dtype,
    )
    expected[:, 0] = expected[:, 0].logit()
    expected[:, 1] = expected[:, 1].atanh()
    assert latent.isfinite().all()
    torch.testing.assert_close(latent[:2], expected, rtol=0, atol=1e-12)
    assert (latent[:, 0] >= expected[0, 0]).all()
    assert torch.equal(shrunk[2], action[2])


def test_latent_log_prob_by_action_sums_to_joint_log_prob():
    rng = torch.Generator().manual_seed(12)
    mean = torch.randn(3, 4, 2, generator=rng, dtype=torch.float64)
    latent = torch.randn(3, 4, 2, generator=rng, dtype=torch.float64)
    log_std = torch.tensor([-0.3, 0.2], dtype=torch.float64)
    dist = ActionDistribution(mean, log_std, v_max=0.4, omega_max=2.0)

    per_action = _latent_log_prob_by_action(dist, latent)
    assert per_action.shape == latent.shape
    torch.testing.assert_close(per_action.sum(-1), dist.log_prob(latent), atol=1e-12, rtol=0)


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


def test_saturation_counts_both_speed_limits_and_joint_nll_once():
    policy = _policy("GRU2")
    batch = _batch(policy)
    v_max, omega_max = policy.contract.v_max, policy.contract.omega_max
    batch.action[: SETTINGS.burn_in].zero_()
    batch.action[SETTINGS.burn_in :] = batch.action.new_tensor(
        [
            [ACTION_EPS * v_max, 0],
            [(1 - ACTION_EPS) * v_max, omega_max],
            [0.5 * v_max, -omega_max],
            [2 * ACTION_EPS * v_max, 0],
        ]
    )
    initial = policy.initial_central_state(1, "cpu")
    interior = SequenceBatch(
        batch.code[:, 3:],
        batch.action[:, 3:],
        batch.reset[:, 3:],
        batch.initial[3:],
        batch.tags[3:],
        batch.occluded[:, 3:],
    )
    with torch.no_grad():
        interior_nll = sequence_loss(policy, interior, initial, SETTINGS)["nll"]
    optimizer = torch.optim.Adam(parameter_groups(policy, OPTIMIZER["lr"]))
    scheduler = make_scheduler(optimizer, OPTIMIZER, 1)
    metrics = train_update(policy, optimizer, scheduler, batch, SETTINGS, 2, 1.0)
    assert metrics["saturation_fraction"] == 0.75
    assert metrics["saturation_fraction_speed_low"] == 0.25
    assert metrics["saturation_fraction_speed_high"] == 0.25
    assert metrics["saturation_fraction_turn"] == 0.5
    torch.testing.assert_close(
        metrics["saturated_nll_fraction"], 1 - interior_nll / (4 * metrics["nll"])
    )


@pytest.mark.slow
@torch.no_grad()
def test_bundled_fitted_input_norm_calibrates_resting_speed():
    policy = build_training_policy("VO2", 0, activation_checkpoint=False)
    empty = build_training_policy("VO2", 0, activation_checkpoint=False, model_overrides={})
    assert empty.config == policy.config
    assert empty.readout.speed_types == policy.readout.speed_types == ("DNp09",)
    torch.testing.assert_close(empty.state_dict(), policy.state_dict(), atol=0, rtol=0)
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


@pytest.mark.parametrize("key", ["init_seed", "structure_seed", "activation_checkpoint", "unknown"])
def test_model_override_rejects_invalid_keys(tmp_path, key):
    message = "Unknown model config key" if key == "unknown" else f"model.{key} is controlled"
    with pytest.raises(ValueError, match=message):
        load_config(CONFIG_DIR / "dagger.yaml", "4060", [f"model.{key}=1"])
    with pytest.raises(ValueError, match=message):
        build_training_policy("VO2", 0, activation_checkpoint=False, model_overrides={key: 1})

    config = tmp_path / "dagger.yaml"
    defaults = (CONFIG_DIR / "dagger.yaml").read_text()
    config.write_text(defaults.replace("model: {}", f"model: {{{key}: 1}}"))
    with pytest.raises(ValueError, match=message):
        load_config(config, "4060")


def test_model_yaml_overrides_are_applied_before_cli(tmp_path):
    config = tmp_path / "dagger.yaml"
    defaults = (CONFIG_DIR / "dagger.yaml").read_text()
    config.write_text(defaults.replace("model: {}", "model: {gain_parameterization: bounded}"))
    settings = load_config(config, "4060", ["model.gain_parameterization=softplus"])
    assert settings["model"] == {"gain_parameterization": "softplus"}
    config.write_text(defaults.replace("model: {}", "model: {gain_parameterization: invalid}"))
    settings = load_config(config, "4060")
    with pytest.raises(ValueError, match="gain_parameterization"):
        build_training_policy(
            "VO2", 0, activation_checkpoint=False, model_overrides=settings["model"]
        )


@pytest.mark.slow
def test_bilateral_model_override_resumes_and_exports_resolved_readout(
    tmp_path, monkeypatch, capsys
):
    args = [
        "--variant",
        "VO2",
        "--seed",
        "0",
        "--host",
        "4060",
        "--device",
        "cpu",
        "--out",
        str(tmp_path),
        "--override",
        "rounds.beta=[1.0]",
        "envs_per_instance=2",
        "rounds.env_steps_per_round=72",
        "replay_ratio=0.25",
        "batch_sequences=4",
        "sequence.length=4",
        "sequence.burn_in=2",
        "held_out.sequences=4",
    ]
    overrides = {"speed_types": "bilateral", "gain_parameterization": "bounded"}
    spec = read_yaml(ROOT / "configs/v2/model/readout_bilateral_dn.yaml")
    expected_ids = {
        row["type"]: {"left": [row["left_body_id"]], "right": [row["right_body_id"]]}
        for row in spec["types"]
    }
    start_round = dagger.DAggerRun._start_round

    def stop_after_collection(run, monitor, index):
        assert run.policy.readout.speed_types == tuple(expected_ids)
        assert run.policy.readout.speed_body_ids == expected_ids
        assert run.policy.config.gain_parameterization == "bounded"
        start_round(run, monitor, index)
        (tmp_path / "STOP").touch()

    with monkeypatch.context() as patch:
        patch.setattr(dagger.DAggerRun, "_start_round", stop_after_collection)
        assert main([*args, *(f"model.{key}={value}" for key, value in overrides.items())]) == 0
    assert (tmp_path / "round_1/progress.pt").is_file()
    (tmp_path / "STOP").unlink()
    config_path = tmp_path / "config.json"
    saved_config = config_path.read_text()
    assert json.loads(saved_config)["model"] == overrides

    with pytest.raises(SystemExit, match="2"):
        main([*args, "model.speed_types=[DNp09]", "--resume"])
    assert "Resume requires the same model overrides" in capsys.readouterr().err
    assert main([*args, "--resume"]) in (0, 3)
    assert config_path.read_text() == saved_config

    card = json.loads((tmp_path / "final/model_card.json").read_text())
    (stage,) = card["provenance"]["stages"]
    assert stage["model"] == overrides
    assert card["model_config"]["gain_parameterization"] == "bounded"
    assert card["model_config"]["speed_types"] == list(expected_ids)
    assert card["model_config"]["readout_body_ids"]["speed"] == expected_ids
    assert load_policy(tmp_path / "final").readout.speed_body_ids == expected_ids


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
        assert replay.pop("layout") == baseline.pop("layout") == "env_major_f16"
        torch.testing.assert_close(replay, baseline, rtol=0, atol=0)
        assert "episode_steps" not in replay["steps"]
        samples = []
        for reset in replay["steps"]["reset"].T:
            samples.append(ages)
            ages = torch.where(reset, 0, ages + 1)
        std, mean = torch.std_mean(torch.stack(samples).double(), correction=0)
        assert row["episode_time_mean_s"] == pytest.approx(mean.item() * 0.05)
        assert row["episode_time_std_s"] == pytest.approx(std.item() * 0.05)
    loaded = load_policy(resumed / "final")
    assert loaded.input_norm.fitted
    assert not (resumed / "round_1/readout_norm.json").exists()
    assert not any(name.startswith("readout.rate_") for name in loaded.state_dict())
    diagnostics = json.loads((resumed / "diagnostics.json").read_text())
    assert "first_batch_core_gradients" not in diagnostics
    assert not diagnostics["readout_drift"]
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, continued["policy"][key]), key
    card = json.loads((resumed / "final/model_card.json").read_text())
    (stage,) = card["provenance"]["stages"]
    assert stage["name"] == "dagger"
    assert stage["micro_batch"] == 4
    assert stage["host"] == "4060"
    assert stage["activation_checkpoint"] is False
    assert stage["desync_first_episode"] is True


def test_readout_fit_checkpoint_resume_and_diagnostics(tmp_path, monkeypatch):
    def tiny_policy(variant, seed, *, activation_checkpoint, model_overrides):
        config = ModelConfigV2(
            init_seed=seed,
            rest_settle_s=0.5,
            rest_refresh_s=0.05,
            activation_checkpoint=activation_checkpoint,
            **model_overrides,
        )
        return build_policy(variant, config, graph=small_graph())

    monkeypatch.setattr(dagger, "build_training_policy", tiny_policy)
    create_optimizer = dagger.DAggerRun._create_optimizer

    def create_after_fit(run):
        assert run.policy.readout.rate_norm_fitted
        create_optimizer(run)

    monkeypatch.setattr(dagger.DAggerRun, "_create_optimizer", create_after_fit)
    args = [
        "--variant",
        "VO2",
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
        "rounds.env_steps_per_round=72",
        "replay_ratio=0.25",
        "batch_sequences=4",
        "sequence.length=4",
        "sequence.burn_in=2",
        "held_out.sequences=4",
        "held_out.every_updates=1",
        "activation_checkpoint=true",
    ]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    rollout_step = dagger.RolloutCollector.step
    rollout_codes = []

    def capture_code(collector, *args, **kwargs):
        data = rollout_step(collector, *args, **kwargs)
        rollout_codes.append(data.code.clone())
        return data

    with monkeypatch.context() as patch:
        patch.setattr(dagger.RolloutCollector, "step", capture_code)
        full_status = main([*args, "--out", str(full)])
    assert full_status in (0, 3)
    assert (full / "final").is_dir()

    save = dagger.DAggerRun.save
    resume_args = [*args, "--out", str(resumed)]

    def unexpected_collect(*_args, **_kwargs):
        pytest.fail("Pre-fit checkpoint must reuse the collected replay")

    for fitted in (False, True):
        with monkeypatch.context() as patch:

            def interrupt_after_save(run, path, *, expected_fitted=fitted):
                save(run, path)
                if (
                    path.name == "progress.pt"
                    and bool(run.policy.readout.rate_norm_fitted) == expected_fitted
                ):
                    raise InterruptedError("test checkpoint interruption")

            patch.setattr(dagger.DAggerRun, "save", interrupt_after_save)
            if fitted:
                patch.setattr(dagger.RolloutCollector, "collect", unexpected_collect)
            with pytest.raises(SystemExit, match="2"):
                main(resume_args)
        checkpoint = torch.load(resumed / "round_1/progress.pt", weights_only=True)
        assert checkpoint["progress"]["updates"] == 0
        assert bool(checkpoint["policy"]["readout.rate_norm_fitted"]) is fitted
        assert (checkpoint["optimizer"] is not None) is fitted
        resume_args = [*args, "--out", str(resumed), "--resume"]

    def unexpected_fit(*_args):
        pytest.fail("Fitted checkpoint must skip normalization fitting")

    monkeypatch.setattr(ConnectomePolicyV2, "fit_readout_norm", unexpected_fit)
    assert main(resume_args) == full_status

    for name in ("round_1/readout_norm.json", "diagnostics.json"):
        assert (full / name).read_bytes() == (resumed / name).read_bytes()
    baseline = torch.load(full / "round_2/checkpoint.pt", weights_only=True)
    continued = torch.load(resumed / "round_2/checkpoint.pt", weights_only=True)
    for key in (
        "progress",
        "policy",
        "optimizer",
        "scheduler",
        "collector",
        "train_rng",
        "held_rng",
    ):
        torch.testing.assert_close(baseline[key], continued[key], rtol=0, atol=0, equal_nan=True)

    # Independently replay the float32 rollout to recover exactly the fit samples.
    replay = torch.load(full / "round_1/replay.pt", weights_only=True)
    policy = tiny_policy("VO2", 0, activation_checkpoint=False, model_overrides={})
    resets = replay["steps"]["reset"].T
    codes = torch.stack(rollout_codes[: len(resets)])
    initial = policy.initial_central_state(resets.size(1), "cpu")
    assert torch.equal(replay["steps"]["code"], codes.transpose(0, 1).half())
    assert torch.equal(replay["states"][:, 0], initial.half())
    state, sampled = initial, []
    with torch.no_grad():
        for tick, (code, reset) in enumerate(zip(codes, resets, strict=True)):
            step = policy.forward_code(code, state)
            if tick % 4 == 0:
                sampled.append(step.aux["dn_rates"][~replay["held_out"]])
            state = reset_rows(step.central_state, initial, reset)
    rates = torch.cat(sampled)
    assert torch.equal(replay["readout_rates"], rates)
    norm = json.loads((full / "round_1/readout_norm.json").read_text())
    assert norm["n_samples"] == len(rates)
    torch.testing.assert_close(torch.tensor(norm["rate_mean"]), rates.mean(0))

    diagnostics = json.loads((full / "diagnostics.json").read_text())
    gradients = diagnostics["first_batch_core_gradients"]
    assert set(gradients) == {"normalized", "unnormalized"}
    for condition in gradients.values():
        assert set(condition) == {"g", "tau", "bias", "lc_rf", "missing_bias", "total"}
        total = condition["total"]
        assert total["speed"] > 0 and total["turn"] > 0
        assert total["speed_to_turn"] == pytest.approx(total["speed"] / total["turn"])
    assert gradients["normalized"] != gradients["unnormalized"]
    assert [row["round"] for row in diagnostics["readout_drift"]] == [1, 2]
    for row in diagnostics["readout_drift"]:
        assert len(row["max_abs_per_dn"]) == len(norm["body_ids"])
        assert row["max_abs"] == max(row["max_abs_per_dn"].values())
        assert 0 <= row["mean_abs"] <= row["max_abs"]
    for row in diagnostics["nll"]:
        for prefix in ("", "held_out_"):
            assert row[prefix + "nll"] == pytest.approx(
                row[prefix + "nll_speed"] + row[prefix + "nll_turn"], abs=1e-6
            )
    records = [json.loads(line) for line in (resumed / "monitor.jsonl").read_text().splitlines()]
    events = [row for row in records if row.get("readout_norm_fitted")]
    assert len(events) == 1 and events[0]["update"] == 0
    first_update = next(row for row in records if row.get("update") == 1)
    assert (
        first_update["first_batch_core_grad_normalized_total_speed_to_turn"]
        == (gradients["normalized"]["total"]["speed_to_turn"])
    )


@pytest.mark.parametrize("failure_case", ["low_speed", "early_silent", "final_silent", "nonfinite"])
def test_round_health_warning_and_stage_failure(tmp_path, monkeypatch, failure_case):
    probes = 0
    silent_round = {"early_silent": 1, "final_silent": 4}.get(failure_case)

    def health_metrics(*args):
        nonlocal probes
        probes += 1
        silent = probes == silent_round
        nonfinite = probes == 4 and failure_case == "nonfinite"
        return {
            "health_speed_std_m_s": torch.tensor(
                0.03 if probes == 1 or silent or nonfinite else 0.02
            ),
            "health_odor_swap_delta_omega": torch.tensor(float("inf") if nonfinite else 0.1),
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

    contract = dagger.SensorContractV2.from_json(dagger.ROOT / "configs/v2/sensor_contract.json")
    v_max, omega_max = contract.v_max, contract.omega_max
    resolved = json.loads((tmp_path / "config.json").read_text())
    for record in round_ends:
        replay = torch.load(tmp_path / f"round_{int(record['round'])}/replay.pt", weights_only=True)
        action = replay["steps"]["action"].reshape(-1, 2)
        labels = shrink_labels(action, v_max, omega_max, label_margins(resolved))
        latent = unsquash(labels, v_max, omega_max).double()
        quantiles = latent.new_tensor([0.001, 0.01, 0.5, 0.99, 0.999])
        expected = torch.quantile(latent, quantiles, dim=0).T.tolist()
        for name, values in zip(("speed", "turn"), expected, strict=True):
            for suffix, value in zip(("001", "010", "500", "990", "999"), values, strict=True):
                assert record[f"label_latent_{name}_p{suffix}"] == pytest.approx(value)

    failure = json.loads((tmp_path / "failure/reason.json").read_text())
    assert failure["metrics"] == round_ends[-1]
    assert (tmp_path / "failure/batch.pt").is_file()
    checkpoint_path = tmp_path / "failure" / failure["checkpoint"]
    assert checkpoint_path.is_file()
    assert failure["checkpoint_sha256"] == file_sha256(checkpoint_path)
    if silent_round is not None:
        assert "silent readout DNs" in failure["reason"]
    if failure_case == "nonfinite":
        assert "non-finite health metrics" in failure["reason"]
    if failure_case in ("early_silent", "nonfinite"):
        assert failure["checkpoint"] == "checkpoint.pt"
        assert not (tmp_path / f"round_{expected_rounds[-1]}/checkpoint.pt").exists()
        assert not (tmp_path / "final").exists()
        return

    if failure_case == "low_speed":
        assert "speed std is <= 0.02 m/s" in failure["reason"]
    assert "Exported to final/. Stage 1 health check failed." in failure["reason"]
    assert failure["checkpoint"] == "../round_4/checkpoint.pt"
    assert not (tmp_path / "failure/checkpoint.pt").exists()
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["progress"]["round_index"] == 4
    assert checkpoint["progress"]["updates"] == 4
    loaded = load_policy(tmp_path / "final")
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, checkpoint["policy"][key]), key

    assert main([*args, "--resume"]) == 3
    assert probes == 4
    assert json.loads((tmp_path / "failure/reason.json").read_text()) == failure
