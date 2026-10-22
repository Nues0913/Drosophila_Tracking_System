"""#4 B3: calibrate extra turn rows on a BC dataset, then warm-start BC with them."""

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
from flytrack_core.model import load_policy_artifact
from flytrack_train import imitation
from flytrack_train.config import prepare_policy, read_config

ROOT = Path(__file__).resolve().parents[2]


def test_calibration_feeds_a_bc_warm_start_that_keeps_the_actor(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    config = copy.deepcopy(read_config("configs/train/smoke.yaml"))
    config.pop("init_artifact", None)
    config["variant"] = "O"
    config["train"].update(teacher_steps=400, dagger_rounds=0, epochs=1)
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
        ],
    )
    module.main()
    result = json.loads(out.read_text())
    model = result["model"]
    assert model["readout_extra_turn_types"] == ["DNb05"]
    assert math.isfinite(model["readout_extra_turn_center"][0])
    assert model["readout_extra_turn_scale"][0] > 0
    assert result["held_out_episodes"] and result["training_steps"] > 0
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
    assert result["odor_shared_adaptation"] is False
    shared_out = tmp_path / "calibration_shared.json"
    argv = sys.argv[:]
    argv[1] = str(shared_out)
    monkeypatch.setattr(sys, "argv", [argv[0], "--shared-adaptation", *argv[1:]])
    module.main()
    shared = json.loads(shared_out.read_text())
    assert shared["odor_shared_adaptation"] is True
    assert math.isfinite(shared["model"]["readout_extra_turn_center"][0])
