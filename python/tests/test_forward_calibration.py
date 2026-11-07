"""#95 S1: opt-in centering and scaling of the forward (speed) readout inputs."""

import copy
import json
import math
from pathlib import Path

import pytest
import torch
from flytrack_core.model import build_policy, load_policy_artifact, save_policy_artifact
from flytrack_core.model.policy import count_trainable_parameters
from flytrack_train import imitation
from flytrack_train.config import contract_from_config, prepare_policy, read_config

ROOT = Path(__file__).resolve().parents[2]


def _smoke(variant="VO-late"):
    config = copy.deepcopy(read_config(ROOT / "configs/train/smoke.yaml"))
    config.pop("init_artifact", None)
    config["variant"] = variant
    for key in ("graph_path", "frontend_path", "sensor_contract_path"):
        config[key] = str(ROOT / config[key])
    for name, path in config["scenario_configs"].items():
        config["scenario_configs"][name] = str(ROOT / path)
    return config


def _policy(variant="VO-late", **model):
    config = _smoke(variant)
    contract = contract_from_config(config)
    policy = build_policy(
        variant,
        config["graph_path"],
        contract,
        config["frontend_path"],
        **{**config["model"], **model},
    )
    return policy, contract, config


def _activities(policy, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [
        0.05 + 0.1 * torch.rand((4, policy.n_nodes), generator=generator)
        for _ in range(policy.branches)
    ]


def test_identity_calibration_is_exactly_the_uncalibrated_readout():
    policy, _, _ = _policy()
    assert not policy.readout.forward_calibrated
    assert policy.model_config["readout_forward_center"] == []
    columns = len(policy.readout.forward_weight)
    assert columns == policy.branches * len(policy.readout.forward_types)
    activities = _activities(policy)
    with torch.no_grad():
        plain = policy.readout(activities)
    count, keys = count_trainable_parameters(policy), set(policy.state_dict())
    policy.set_readout_forward_calibration([0.0] * columns, [1.0] * columns)
    with torch.no_grad():
        identity = policy.readout(activities)
    assert torch.equal(plain, identity)
    # Buffers, not parameters: the budget and the saved state are unchanged.
    assert count_trainable_parameters(policy) == count
    assert set(policy.state_dict()) == keys


def test_calibration_centers_and_scales_each_branch_column():
    policy, _, _ = _policy()
    readout = policy.readout
    activities = _activities(policy, seed=1)
    raw = torch.cat([a @ readout.forward_pool.T for a in activities], 1)
    columns = raw.shape[1]
    center = [0.07 + 0.01 * i for i in range(columns)]
    scale = [100.0 * (i + 1) for i in range(columns)]
    policy.set_readout_forward_calibration(center, scale)
    expected = (raw - torch.tensor(center)) * torch.tensor(scale)
    assert torch.allclose(readout.forward_inputs(activities), expected)
    with torch.no_grad():
        v = readout(activities)[:, 0]
    latent = expected @ readout.forward_weight + readout.forward_bias
    assert torch.allclose(v, readout.v_max * torch.sigmoid(latent))
    assert policy.model_config["readout_forward_scale"] == scale


def test_calibration_rejects_bad_constants_and_removal():
    policy, _, _ = _policy()
    columns = len(policy.readout.forward_weight)
    with pytest.raises(ValueError, match="one center and scale"):
        policy.set_readout_forward_calibration([0.0] * (columns + 1), [1.0] * (columns + 1))
    with pytest.raises(ValueError, match="scales positive"):
        policy.set_readout_forward_calibration([0.0] * columns, [0.0] * columns)
    with pytest.raises(ValueError, match="centers must be finite"):
        policy.set_readout_forward_calibration([math.nan] * columns, [1.0] * columns)
    policy.set_readout_forward_calibration([0.0] * columns, [1.0] * columns)
    with pytest.raises(ValueError, match="cannot be removed"):
        policy.set_readout_forward_calibration([], [])
    with pytest.raises(ValueError, match="connectome policy"):
        _policy("GRU", readout_forward_center=[0.0], readout_forward_scale=[1.0])
    with pytest.raises(ValueError, match="structured readout"):
        _policy("VO", readout="linear", readout_forward_center=[0.0], readout_forward_scale=[1.0])


def test_card_round_trip_and_legacy_cards(tmp_path):
    policy, contract, config = _policy()
    columns = len(policy.readout.forward_weight)
    center, scale = [0.069] * columns, [2.5e4] * columns
    policy.set_readout_forward_calibration(center, scale)
    source = tmp_path / "calibrated"
    save_policy_artifact(
        source, policy, policy.model_config, contract, config["graph_path"], config["frontend_path"]
    )
    card = json.loads((source / "model_card.json").read_text())
    assert card["model_config"]["readout_forward_center"] == center
    assert "forward_readout" in card
    loaded = load_policy_artifact(source)[0]
    assert loaded.readout.forward_calibrated
    activities = _activities(policy, seed=2)
    with torch.no_grad():
        assert torch.equal(loaded.readout(activities), policy.readout(activities))
    # A card from before #95 has no forward entries and keeps the uncalibrated readout.
    plain, _, _ = _policy()
    legacy = tmp_path / "legacy"
    save_policy_artifact(
        legacy, plain, plain.model_config, contract, config["graph_path"], config["frontend_path"]
    )
    card_path = legacy / "model_card.json"
    card = json.loads(card_path.read_text())
    for key in ("readout_forward_center", "readout_forward_scale"):
        del card["model_config"][key]
    card_path.write_text(json.dumps(card))
    old = load_policy_artifact(legacy)[0]
    assert not old.readout.forward_calibrated
    assert "forward_readout" not in card
    with torch.no_grad():
        assert torch.equal(old.readout(activities), plain.readout(activities))


def _bc_config(**train):
    config = _smoke("VO-late")
    config["env"]["episode_seconds"] = 5
    config["train"].update(teacher_steps=1000, epochs=1, **train)
    return config


def test_bc_calibrates_forward_on_round_zero_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    config = _bc_config(dagger_rounds=1, dagger_steps_per_round=200, calibrate_forward=True)
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    imitation.train_bc(config, full, 0, torch.device("cpu"))
    report = json.loads((full / "forward_calibration.json").read_text())
    card = json.loads((full / "actor/model_card.json").read_text())
    trained = load_policy_artifact(full / "actor")[0]
    columns = len(trained.readout.forward_weight)
    assert len(report["columns"]) == columns == 2 * len(trained.readout.forward_types)
    for key, value in report["model"].items():
        assert card["model_config"][key] == value
        assert len(value) == columns
    assert all(x > 0 for x in report["model"]["readout_forward_scale"])
    assert trained.readout.forward_calibrated
    # The smoke world never drives the odor branch's forward DN: it stays exactly at
    # softplus(0) and is kept at scale 1; the visual branch is amplified.
    assert report["active"] == [True, False]
    for active, std, scale in zip(
        report["active"], report["raw_std"], report["model"]["readout_forward_scale"], strict=True
    ):
        assert scale == (pytest.approx(1 / std) if active else 1.0)
    # BC's own validation episodes stay out of the constants.
    assert report["held_out_episodes"] and report["closed_form"]["held_out_corr"] is not None
    original_save = imitation.Recovery.save

    def stop_in_round_one(self, rng, **state):
        original_save(self, rng, **state)
        if state["round_index"] == 1:
            raise InterruptedError("simulated interruption after the calibrated round 0")

    monkeypatch.setattr(imitation.Recovery, "save", stop_in_round_one)
    with pytest.raises(InterruptedError):
        imitation.train_bc(config, interrupted, 0, torch.device("cpu"))
    monkeypatch.setattr(imitation.Recovery, "save", original_save)
    saved = torch.load(interrupted / "training_checkpoint.pt", weights_only=True)
    assert saved["forward_calibration"] == report["model"]

    def no_recalibration(*args):
        raise AssertionError("a resumed BC must reuse the saved constants")

    monkeypatch.setattr(imitation, "calibrate_forward", no_recalibration)
    imitation.train_bc({**config, "resume": True}, interrupted, 0, torch.device("cpu"))
    resumed = json.loads((interrupted / "actor/model_card.json").read_text())
    assert resumed["model_config"] == card["model_config"]
    expected = trained.state_dict()
    for name, value in load_policy_artifact(interrupted / "actor")[0].state_dict().items():
        assert torch.equal(value, expected[name])


def test_bc_forward_calibration_preconditions(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    with pytest.raises(ValueError, match="must be bool"):
        imitation.train_bc(
            _bc_config(calibrate_forward="yes"), tmp_path / "bad", 0, torch.device("cpu")
        )
    with pytest.raises(ValueError, match="connectome policy"):
        imitation.train_bc(
            {**_bc_config(calibrate_forward=True), "variant": "GRU"},
            tmp_path / "gru",
            0,
            torch.device("cpu"),
        )
    config = _bc_config(calibrate_forward=True, dagger_rounds=0)
    imitation.train_bc(config, tmp_path / "first", 0, torch.device("cpu"))
    warm = {**config, "init_artifact": str(tmp_path / "first/actor")}
    with pytest.raises(ValueError, match="without a forward calibration"):
        imitation.train_bc(warm, tmp_path / "second", 0, torch.device("cpu"))
    # The constants cannot be changed on a warm start, only kept.
    card = json.loads((tmp_path / "first/actor/model_card.json").read_text())
    same = {
        **config["model"],
        "readout_forward_center": card["model_config"]["readout_forward_center"],
    }
    policy, _ = prepare_policy({**warm, "model": same}, torch.device("cpu"))
    assert policy.readout.forward_calibrated
    changed = {
        **config["model"],
        "readout_forward_center": [0.0] * len(same["readout_forward_center"]),
    }
    with pytest.raises(ValueError, match="forward readout calibration differs"):
        prepare_policy({**warm, "model": changed}, torch.device("cpu"))
    # An uncalibrated warm start is untouched when no forward entries are configured.
    plain = _bc_config(dagger_rounds=0)
    imitation.train_bc(plain, tmp_path / "plain", 0, torch.device("cpu"))
    policy, _ = prepare_policy(
        {**plain, "init_artifact": str(tmp_path / "plain/actor")}, torch.device("cpu")
    )
    assert not policy.readout.forward_calibrated
