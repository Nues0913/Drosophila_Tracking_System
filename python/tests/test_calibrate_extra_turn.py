"""#4 B3: calibrate extra turn rows on a BC dataset, then warm-start BC with them."""

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from flytrack_core.model import build_policy, load_policy_artifact, save_policy_artifact
from flytrack_train import imitation
from flytrack_train.config import contract_from_config, prepare_policy, read_config

# v1 legacy-flag compatibility: runs in the full workflow only (#131).
pytestmark = pytest.mark.legacy

ROOT = Path(__file__).resolve().parents[2]


def test_calibration_feeds_a_bc_warm_start_that_keeps_the_actor(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    config = copy.deepcopy(read_config("configs/train/smoke.yaml"))
    config.pop("init_artifact", None)
    config["variant"] = "O"
    # DNb05 stays silent for the first ~3 s, so episodes must outlast that warm-up.
    config["env"]["episode_seconds"] = 5
    config["train"].update(teacher_steps=1000, dagger_rounds=0, epochs=1)
    imitation.train_bc(config, tmp_path / "bc", 0, torch.device("cpu"))
    count = len(list((tmp_path / "bc/dataset").glob("episode_*.npz")))
    assert count >= 10
    path = ROOT / "scripts/diagnostics/calibrate_extra_turn.py"
    spec = importlib.util.spec_from_file_location("calibrate_extra_turn", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "calibration.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calibrate_extra_turn.py",
            str(out),
            str(tmp_path / "bc/dataset"),
            str(count),
            str(tmp_path / "bc/actor"),
            "DNb05",
            # The smoke world drives DNb05 to only ~0.01 (4-R6's start actor: median 1.98).
            "--eps",
            "0.001",
        ],
    )
    module.main()
    result = json.loads(out.read_text())
    model = result["model"]
    assert model["readout_extra_turn_types"] == ["DNb05"]
    assert math.isfinite(model["readout_extra_turn_center"][0])
    assert model["readout_extra_turn_scale"][0] > 0
    assert result["held_out_episodes"] and result["training_steps"] > 0
    assert 0 <= result["excluded_fraction"][0] < 1 and result["device"] == "cpu"
    assert model["readout_extra_turn_eps"] == 0.001
    assert all(fit["held_out_corr"] is not None for fit in result["closed_form"].values())
    loaded = load_policy_artifact(tmp_path / "bc/actor")[0]
    warm, _ = prepare_policy(
        {
            **config,
            "init_artifact": str(tmp_path / "bc/actor"),
            "model": {**config["model"], **model},
        },
        torch.device("cpu"),
    )
    assert warm.readout.extra_turn_types == ["DNb05"]
    assert warm.model_config["readout_extra_turn_scale"] == model["readout_extra_turn_scale"]
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, warm.state_dict()[name])
    assert float(warm.readout.extra_turn_weight.detach()) == 0.0
    assert float(warm.readout.extra_eps) == pytest.approx(0.001)
    assert result["odor_shared_adaptation"] is False
    shared_out = tmp_path / "calibration_shared.json"
    argv = sys.argv[:]
    argv[1] = str(shared_out)
    monkeypatch.setattr(sys, "argv", [argv[0], "--shared-adaptation", *argv[1:]])
    module.main()
    shared = json.loads(shared_out.read_text())
    assert shared["odor_shared_adaptation"] is True
    assert math.isfinite(shared["model"]["readout_extra_turn_center"][0])


def _bc_config(**train):
    config = copy.deepcopy(read_config("configs/train/smoke.yaml"))
    config.pop("init_artifact", None)
    config["variant"] = "O"
    config["env"]["episode_seconds"] = 5  # Outlasts DNb05's silent warm-up.
    config["train"].update(teacher_steps=1000, epochs=1, **train)
    return config


def test_bc_calibrates_rows_on_round_zero_and_resumes_them(tmp_path, monkeypatch):
    """Formal spec (#4, 2026-10-24): BC adds and calibrates the rows itself."""
    monkeypatch.chdir(ROOT)
    config = _bc_config(
        dagger_rounds=1,
        dagger_steps_per_round=200,
        # The smoke world drives DNb05 to only ~0.01 (4-R6's start actor: median 1.98).
        calibrate_extra_turn={"types": ["DNb05"], "eps": 0.001},
    )
    full, interrupted = tmp_path / "full", tmp_path / "interrupted"
    imitation.train_bc(config, full, 0, torch.device("cpu"))
    report = json.loads((full / "extra_turn_calibration.json").read_text())
    card = json.loads((full / "actor/model_card.json").read_text())
    assert report["model"]["readout_extra_turn_types"] == ["DNb05"]
    assert report["model"]["readout_extra_turn_eps"] == 0.001
    for key, value in report["model"].items():
        assert card["model_config"][key] == value
    # BC's own validation episodes stay out of the constants.
    episodes = len(list((full / "dataset").glob("episode_*.npz")))
    assert report["held_out_episodes"] and report["training_steps"] < episodes * 100
    assert float(load_policy_artifact(full / "actor")[0].readout.extra_turn_weight.detach()) != 0
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
    assert saved["extra_turn_calibration"] == report["model"]

    def no_recalibration(*args):
        raise AssertionError("a resumed BC must reuse the saved constants")

    monkeypatch.setattr(imitation, "calibrate_extra_turns", no_recalibration)
    imitation.train_bc({**config, "resume": True}, interrupted, 0, torch.device("cpu"))
    resumed = json.loads((interrupted / "actor/model_card.json").read_text())
    assert resumed["model_config"] == card["model_config"]
    expected = load_policy_artifact(full / "actor")[0].state_dict()
    for name, value in load_policy_artifact(interrupted / "actor")[0].state_dict().items():
        assert torch.equal(value, expected[name])


def test_bc_calibration_needs_a_connectome_actor_without_rows(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    rows = {"types": ["DNb05"], "eps": 0.001}
    with pytest.raises(ValueError, match="connectome policy"):
        imitation.train_bc(
            {**_bc_config(calibrate_extra_turn=rows), "variant": "GRU"},
            tmp_path / "gru",
            0,
            torch.device("cpu"),
        )
    config = _bc_config(calibrate_extra_turn=rows, dagger_rounds=0)
    imitation.train_bc(config, tmp_path / "first", 0, torch.device("cpu"))
    with pytest.raises(ValueError, match="without extra turn rows"):
        imitation.train_bc(
            {**config, "init_artifact": str(tmp_path / "first/actor")},
            tmp_path / "second",
            0,
            torch.device("cpu"),
        )


def test_bc_calibration_uses_the_fixed_center_form_on_an_old_card(tmp_path, monkeypatch):
    """#67 review: a card from before the fix loads the legacy form; new rows must not."""
    monkeypatch.chdir(ROOT)
    config = _bc_config(dagger_rounds=0, calibrate_extra_turn={"types": ["DNb05"], "eps": 0.001})
    contract = contract_from_config(config)
    old = build_policy(
        "O", config["graph_path"], contract, config["frontend_path"], **config["model"]
    )
    source = tmp_path / "old"
    save_policy_artifact(
        source, old, old.model_config, contract, config["graph_path"], config["frontend_path"]
    )
    card_path = source / "model_card.json"
    card = json.loads(card_path.read_text())
    del card["model_config"]["readout_extra_turn_center_ramp"]
    card_path.write_text(json.dumps(card))
    assert not load_policy_artifact(source)[0].readout.extra_center_ramp
    imitation.train_bc(
        {**config, "init_artifact": str(source)}, tmp_path / "bc", 0, torch.device("cpu")
    )
    report = json.loads((tmp_path / "bc/extra_turn_calibration.json").read_text())
    assert report["readout_extra_turn_center_ramp"] is True
    trained = load_policy_artifact(tmp_path / "bc/actor")[0]
    assert trained.readout.extra_center_ramp
    assert trained.model_config["readout_extra_turn_center_ramp"] is True
