"""#73: final-only BC centering, exact post-hoc equivalence and recovery on CPU."""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from flytrack_core.model import load_policy_artifact
from flytrack_core.model.artifacts import file_sha256
from flytrack_train import imitation, turn_bias
from flytrack_train.batches import Episode, SequenceModel
from flytrack_train.config import prepare_policy, read_config
from flytrack_train.recovery import EpisodeStore, held_out_episodes
from flytrack_train.turn_bias import TURN_BIAS_TOLERANCE, center_turn_bias, solve_turn_bias

ROOT = Path(__file__).resolve().parents[2]
CPU = torch.device("cpu")


def _config(kind="structured"):
    config = copy.deepcopy(read_config(ROOT / "configs/train/smoke.yaml"))
    config["variant"] = "GRU" if kind == "gru" else "VO"
    config["model"].update(readout="linear" if kind == "linear" else "structured", hidden_size=4)
    config["env"]["episode_seconds"] = 0.2
    config["train"].update(
        teacher_steps=24,
        dagger_steps_per_round=8,
        dagger_rounds=1,
        teacher_episodes=6,
        dagger_episodes_per_round=2,
        epochs=1,
        validation_every=3,
        batch_episodes=2,
        sequence_length=2,
    )
    return config


def _episodes():
    rng = np.random.default_rng(73)
    return [
        Episode(
            {
                "vision": rng.random((steps, 2, 721), dtype=np.float32),
                "odor": rng.random((steps, 2), dtype=np.float32),
                "valid": np.ones((steps, 3), dtype=np.float32),
            },
            {
                "action": np.c_[np.full(steps, 0.2), np.linspace(-0.3, 0.6, steps)].astype(
                    np.float32
                )
            },
        )
        for steps in (2, 5, 3, 4, 6, 2)
    ]


def _script():
    spec = importlib.util.spec_from_file_location(
        "center_turn_bias_script", ROOT / "scripts/center_turn_bias.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(path):
    return load_policy_artifact(path)[0].state_dict()


def _assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for name, value in left.items():
        assert torch.equal(value, right[name]), name


def _hashes(path):
    return {str(p.relative_to(path)): file_sha256(p) for p in path.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "kind,parameter",
    [
        ("structured", "readout.turn_bias"),
        ("linear", "readout.linear.bias[1]"),
        ("gru", "readout.bias[1]"),
    ],
)
def test_centering_uses_correct_parameter_and_step_weighted_training_mean(
    monkeypatch, kind, parameter
):
    monkeypatch.chdir(ROOT)
    config = _config(kind)
    policy, _ = prepare_policy(config, CPU)
    policy.train()
    policy.encoder.eval()  # Mixed submodule modes must also survive the replay.
    modes = [module.training for module in policy.modules()]
    initial = copy.deepcopy(policy.state_dict())
    episodes = _episodes()
    model = SequenceModel(policy, CPU)
    original_step = model.step

    def checked_step(*args):
        assert not torch.is_grad_enabled()
        assert not any(module.training for module in policy.modules())
        return original_step(*args)

    monkeypatch.setattr(model, "step", checked_step)
    report = center_turn_bias(policy, model, episodes, config["train"])
    assert [module.training for module in policy.modules()] == modes
    assert report["parameter"] == parameter
    held = set(held_out_episodes(len(episodes), config["train"]["validation_every"]))
    fitted = [e for index, e in enumerate(episodes) if index not in held]
    assert report["steps"] == sum(map(len, fitted))
    assert report["episodes"] == len(fitted) and report["held_out_excluded"] == len(held)
    assert (
        report["teacher_mean_omega"]
        == np.concatenate([e.labels["action"][:, 1] for e in fitted]).astype(np.float64).mean()
    )
    assert abs(report["train_mean_error_after"]) <= TURN_BIAS_TOLERANCE
    assert 1 <= report["iterations"] <= 100
    changed = [
        name for name, value in policy.state_dict().items() if not torch.equal(value, initial[name])
    ]
    assert changed == [parameter.replace("[1]", "")]
    if "[1]" in parameter:
        key = parameter.replace("[1]", "")
        assert torch.equal(policy.state_dict()[key][0], initial[key][0])
    # Independently replay the full policy on the recorded observations, carrying
    # sensory and central states continuously, and recompute the physical error.
    predicted = []
    with torch.no_grad():
        for episode in fitted:
            state = policy.initial_state(1, CPU)
            for step in range(len(episode)):
                obs = {
                    key: torch.from_numpy(values[step : step + 1])
                    for key, values in episode.observations.items()
                }
                action, state, _ = policy(obs, state)
                predicted.append(float(action[0, 1]))
    assert abs(np.mean(predicted) - report["teacher_mean_omega"]) <= TURN_BIAS_TOLERANCE
    assert not policy.readout._forward_hooks
    if kind == "linear":
        assert not policy.readout.linear._forward_hooks


def test_held_out_observations_and_labels_are_never_accessed(monkeypatch):
    monkeypatch.chdir(ROOT)
    config = _config()
    policy, _ = prepare_policy(config, CPU)
    other, _ = prepare_policy(config, CPU)
    other.load_state_dict(policy.state_dict())
    episodes = _episodes()
    first = center_turn_bias(policy, SequenceModel(policy, CPU), episodes, config["train"])

    class Unreadable:
        def __getitem__(self, key):
            raise AssertionError("Held-out data was accessed")

    for index in held_out_episodes(len(episodes), config["train"]["validation_every"]):
        episodes[index] = Episode(Unreadable(), Unreadable())
    second = center_turn_bias(other, SequenceModel(other, CPU), episodes, config["train"])
    assert first == second
    _assert_state_equal(policy.state_dict(), other.state_dict())


def test_disabled_flag_matches_missing_flag_bitwise(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)

    def reject(*args):
        raise AssertionError("Disabled centering must not execute")

    monkeypatch.setattr(imitation, "center_turn_bias", reject)
    config = _config()
    imitation.train_bc(config, tmp_path / "missing", 73, CPU)
    config["train"]["center_turn_bias"] = False
    imitation.train_bc(config, tmp_path / "false", 73, CPU)
    _assert_state_equal(_state(tmp_path / "missing/actor"), _state(tmp_path / "false/actor"))
    assert (tmp_path / "missing/training_metrics.json").read_bytes() == (
        tmp_path / "false/training_metrics.json"
    ).read_bytes()
    metadata = json.loads((tmp_path / "false/actor/model_card.json").read_text())["metadata"]
    assert "turn_bias_centering" not in metadata


@pytest.mark.parametrize("kind", ["structured", "linear", "gru"])
def test_post_hoc_and_training_are_bitwise_equivalent(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(ROOT)
    config = _config(kind)
    config["train"]["center_turn_bias"] = False
    source, enabled, post = tmp_path / "source", tmp_path / "enabled", tmp_path / "post/actor"
    imitation.train_bc(config, source, 73, CPU)
    before = _hashes(source)
    config["train"]["center_turn_bias"] = True
    imitation.train_bc(config, enabled, 73, CPU)
    script = _script()
    # Exercise the actual CLI parser and the default card-config path.
    monkeypatch.setattr(sys, "argv", ["center_turn_bias.py", str(source), "--output", str(post)])
    script.main()
    _assert_state_equal(_state(enabled / "actor"), _state(post))
    assert _hashes(source) == before
    original = json.loads((source / "actor/model_card.json").read_text())["metadata"]
    centered = json.loads((enabled / "actor/model_card.json").read_text())["metadata"]
    metadata = json.loads((post / "model_card.json").read_text())["metadata"]
    assert metadata["turn_bias_centering"] == centered["turn_bias_centering"]
    assert metadata["applied_post_hoc"] is True
    assert metadata["source_artifact"]["files"] == _hashes(source / "actor")
    assert metadata["source_artifact"]["training_metrics_sha256"] == before["training_metrics.json"]
    assert metadata["stage_budget_ledger"] == original["stage_budget_ledger"]
    assert metadata["total_environment_steps"] == original["total_environment_steps"]
    rounds = json.loads((enabled / "training_metrics.json").read_text())
    assert len(rounds) == 2 and "turn_bias_centering" not in rounds[0]
    assert rounds[-1]["turn_bias_centering"] == metadata["turn_bias_centering"]
    episodes = EpisodeStore(source / "dataset").load(centered["dataset_episodes"])
    policy = load_policy_artifact(enabled / "actor")[0]
    check = center_turn_bias(policy, SequenceModel(policy, CPU), episodes, config["train"])
    assert abs(check["train_mean_error_before"]) <= TURN_BIAS_TOLERANCE
    with pytest.raises(FileExistsError):
        script.center_artifact(source, post)
    (post / "stale.txt").write_text("remove on overwrite")
    script.center_artifact(source, post, config=config, overwrite=True)
    assert not (post / "stale.txt").exists()
    _assert_state_equal(_state(enabled / "actor"), _state(post))
    for forbidden in (source, source / "actor", source / "new", source.parent):
        with pytest.raises(ValueError, match="overlap"):
            script.center_artifact(source, forbidden, overwrite=True)
    assert _hashes(source) == before


@pytest.mark.parametrize("different_drive", [False, True])
def test_post_hoc_card_has_no_absolute_paths(tmp_path, monkeypatch, different_drive):
    monkeypatch.chdir(ROOT)
    config = _config()
    source, post = tmp_path / "source", tmp_path / "post/actor"
    imitation.train_bc(config, source, 73, CPU)
    before = _hashes(source)
    script = _script()
    # The bundled files must work even when the recorded repo paths are unavailable.
    monkeypatch.chdir(tmp_path)
    if different_drive:

        def cannot_relativize(*args):
            raise ValueError("different drives")

        monkeypatch.setattr(script.os.path, "relpath", cannot_relativize)
    # Config overrides must not replace the source card's recorded artifact paths.
    override = copy.deepcopy(config)
    for key in ("graph_path", "frontend_path"):
        override[key] = str(tmp_path / "unavailable" / key)
    script.center_artifact(source, post, config=override)
    card_text = (post / "model_card.json").read_text()
    assert str(tmp_path) not in card_text
    card = json.loads(card_text)

    def assert_no_absolute_paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert not key.startswith("/")
                assert_no_absolute_paths(item)
        elif isinstance(value, list):
            for item in value:
                assert_no_absolute_paths(item)
        elif isinstance(value, str):
            assert not value.startswith("/")

    assert_no_absolute_paths(card)
    metadata = card["metadata"]
    provenance = metadata["source_artifact"]
    if different_drive:
        assert provenance["path"] == "actor"
        assert provenance["path_is_relative"] is False
    else:
        assert provenance["path"] == os.path.relpath(source / "actor")
        assert (Path.cwd() / provenance["path"]).resolve() == source / "actor"
    for key in ("graph_path", "frontend_path"):
        assert metadata["config"][key] == config[key]
    for filename in ("graph.npz", "frontend.npz"):
        assert file_sha256(post / filename) == before[f"actor/{filename}"]
    load_policy_artifact(post)
    assert _hashes(source) == before
    assert override["graph_path"] == str(tmp_path / "unavailable/graph_path")


def test_completed_round_checkpoint_resumes_centering_idempotently(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    config = _config()
    config["train"]["center_turn_bias"] = True
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    original_center = imitation.center_turn_bias
    calls = []

    def counted_center(*args):
        calls.append(len(args[2]))
        return original_center(*args)

    monkeypatch.setattr(imitation, "center_turn_bias", counted_center)
    imitation.train_bc(config, full, 73, CPU)
    assert len(calls) == 1  # Exactly once, despite two fit rounds.
    final = _state(full / "actor")
    checkpoint = torch.load(full / "training_checkpoint.pt", weights_only=True)
    assert "turn_bias_centering" not in checkpoint["reports"][-1]
    assert not torch.equal(checkpoint["actor"]["readout.turn_bias"], final["readout.turn_bias"])
    original_save = imitation.Recovery.save

    def stop_before_centering(self, rng, **state):
        original_save(self, rng, **state)
        if state["round_index"] == 2:
            raise InterruptedError("after final fit, before centering")

    monkeypatch.setattr(imitation.Recovery, "save", stop_before_centering)
    with pytest.raises(InterruptedError):
        imitation.train_bc(config, interrupted, 73, CPU)
    assert len(calls) == 1
    monkeypatch.setattr(imitation.Recovery, "save", original_save)

    def no_fit(*args, **kwargs):
        raise AssertionError("A completed-round checkpoint must skip fitting")

    monkeypatch.setattr(imitation, "fit_behavior", no_fit)
    checkpoint_bytes = (interrupted / "training_checkpoint.pt").read_bytes()
    for _ in range(2):
        imitation.train_bc({**config, "resume": True}, interrupted, 73, CPU)
        _assert_state_equal(final, _state(interrupted / "actor"))
        assert (full / "training_metrics.json").read_bytes() == (
            interrupted / "training_metrics.json"
        ).read_bytes()
        assert (interrupted / "training_checkpoint.pt").read_bytes() == checkpoint_bytes
    assert len(calls) == 3


@pytest.mark.parametrize("kind", ["structured", "linear", "gru"])
def test_missing_bias_is_an_error_without_mode_changes(monkeypatch, kind):
    monkeypatch.chdir(ROOT)
    config = _config(kind)
    if kind == "structured":
        config["model"]["readout_turn_bias"] = False
    policy, _ = prepare_policy(config, CPU)
    if kind == "linear":
        policy.readout.linear.bias = None
    elif kind == "gru":
        policy.readout.bias = None
    policy.train()
    with pytest.raises(ValueError, match="bias"):
        center_turn_bias(policy, SequenceModel(policy, CPU), _episodes(), config["train"])
    assert policy.training


@pytest.mark.parametrize("variant", ["VO", "VO-late"])
def test_structured_latent_matches_forward_with_extra_rows(monkeypatch, variant):
    monkeypatch.chdir(ROOT)
    config = _config()
    config["variant"] = variant
    policy, contract = prepare_policy(config, CPU)
    policy.set_readout_extra_turns(["DNb05"], [0.2], [1.4], 0.1)
    readout = policy.readout
    with torch.no_grad():
        readout.extra_turn_weight.fill_(0.3)
        readout.turn_bias.fill_(0.11)
        activity = [torch.rand(2, policy.n_nodes) for _ in range(policy.branches)]
        expected = readout(activity)[:, 1]
        actual = contract.omega_max * torch.tanh(readout.omega_latent(activity))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("target", [-1.500001, -1.5, 1.5, 1.500001])
def test_bisection_rejects_unattainable_boundary_means(target):
    with pytest.raises(ValueError, match="strictly inside"):
        solve_turn_bias(np.array([-1.0, 0.2, 2.0]), np.full(3, target), 1.5)


@pytest.mark.parametrize("target", [-1.49999, 0.17, 1.49999])
def test_bisection_converges_including_near_boundaries(target):
    latent = np.array([-8.0, -1.0, 0.2, 3.0])
    teacher = np.full(4, target)
    delta, iterations = solve_turn_bias(latent, teacher, 1.5)
    assert abs(np.mean(1.5 * np.tanh(latent + delta)) - target) <= TURN_BIAS_TOLERANCE
    assert iterations <= 100


def test_bisection_fails_explicitly_on_nonconvergence(monkeypatch):
    monkeypatch.setattr(turn_bias, "TURN_BIAS_MAX_ITERATIONS", 1)
    with pytest.raises(RuntimeError, match="did not converge"):
        solve_turn_bias(np.array([-2.0, 0.0, 0.1]), np.full(3, 0.7), 1.5)


@pytest.mark.parametrize(
    "latent,teacher", [([], []), ([np.nan], [0.1]), ([0.1], [np.inf]), ([0.1, 0.2], [0.1])]
)
def test_bisection_rejects_invalid_data(latent, teacher):
    with pytest.raises(ValueError, match="finite"):
        solve_turn_bias(np.array(latent), np.array(teacher), 1.5)


def test_replay_failure_restores_modes_and_removes_hook(monkeypatch):
    monkeypatch.chdir(ROOT)
    config = _config()
    policy, _ = prepare_policy(config, CPU)
    policy.train()
    policy.encoder.eval()
    modes = [module.training for module in policy.modules()]
    model = SequenceModel(policy, CPU)

    def failed_encode(*args):
        raise InterruptedError("encoding failed")

    monkeypatch.setattr(model, "encode", failed_encode)
    with pytest.raises(InterruptedError):
        center_turn_bias(policy, model, _episodes(), config["train"])
    assert [module.training for module in policy.modules()] == modes
    assert not policy.readout._forward_hooks
