"""#104 D3: calibrated, detached forward-speed rows on the structured readout."""

import copy
import json
import math
from pathlib import Path

import pytest
import torch
from flytrack_core.model import load_policy_artifact, save_policy_artifact
from flytrack_core.model.policy import count_trainable_parameters
from flytrack_train import imitation
from flytrack_train.config import prepare_policy, read_config

ROOT = Path(__file__).resolve().parents[2]
TYPES = ["DNp09", "DNa01", "DNa02"]


def smoke_policy(monkeypatch, variant):
    monkeypatch.chdir(ROOT)
    config = copy.deepcopy(read_config("configs/train/smoke.yaml"))
    config.pop("init_artifact", None)
    config["variant"] = variant
    torch.manual_seed(0)
    policy, _ = prepare_policy(config, torch.device("cpu"))
    return config, policy.eval()


def rollout(policy, steps=8, seed=1):
    """Actions and per-step readout inputs for a fixed random observation sequence."""
    generator = torch.Generator().manual_seed(seed)
    contract = policy.contract
    state = policy.initial_state(2, torch.device("cpu"))
    actions = []
    for _ in range(steps):
        obs = {
            "vision": torch.rand((2, 2, contract.n_ommatidia), generator=generator),
            "odor": torch.rand((2, 2), generator=generator),
            "valid": torch.ones((2, 3)),
        }
        action, state, _ = policy(obs, state)
        actions.append(action)
    return torch.stack(actions)


@pytest.mark.parametrize("variant", ["V", "VO-late"])
def test_zero_weight_rows_keep_actions_and_count_parameters(monkeypatch, variant):
    _, policy = smoke_policy(monkeypatch, variant)
    assert policy.model_config["readout_forward_row_types"] == []
    before = rollout(policy)
    count = count_trainable_parameters(policy)
    width = len(TYPES) * policy.branches
    policy.set_readout_forward_rows(TYPES, [0.0] * width, [1.0] * width)
    assert torch.equal(rollout(policy), before)
    assert count_trainable_parameters(policy) == count + width
    assert policy.model_config["readout_forward_row_types"] == TYPES
    # A nonzero weight moves only v; omega is untouched.
    with torch.no_grad():
        policy.readout.forward_row_weight.fill_(0.5)
        policy.readout.forward_row_center.fill_(0.0)
        policy.readout.forward_row_scale.fill_(50.0)
    after = rollout(policy)
    assert not torch.equal(after[..., 0], before[..., 0])
    assert torch.equal(after[..., 1], before[..., 1])


def test_rows_send_no_gradient_upstream(monkeypatch):
    _, policy = smoke_policy(monkeypatch, "VO-late")
    width = len(TYPES) * policy.branches
    policy.set_readout_forward_rows(TYPES, [0.0] * width, [100.0] * width)
    with torch.no_grad():
        policy.readout.forward_row_weight.fill_(1.0)
    upstream = [p for name, p in policy.named_parameters() if not name.startswith("readout.")]
    activities = [
        torch.rand((3, policy.readout._graph.n_nodes), requires_grad=True) for _ in range(2)
    ]
    term = policy.readout.forward_row_features(activities) @ policy.readout.forward_row_weight
    grads = torch.autograd.grad(
        term.sum(), [*activities, *upstream, policy.readout.forward_row_weight], allow_unused=True
    )
    assert all(g is None for g in grads[: len(activities) + len(upstream)])
    assert grads[-1] is not None and torch.count_nonzero(grads[-1]) > 0
    # The existing forward pool still trains the core, exactly as before.
    latent = policy.readout.forward_latent(activities)
    assert torch.autograd.grad(latent.sum(), activities[0])[0].abs().sum() > 0


def test_set_forward_rows_validates(monkeypatch):
    _, policy = smoke_policy(monkeypatch, "V")
    readout = policy.readout
    with pytest.raises(ValueError, match="one center and scale"):
        readout.set_forward_rows(TYPES, [0.0], [1.0])
    with pytest.raises(ValueError, match="positive"):
        readout.set_forward_rows(TYPES, [0.0] * 3, [1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="not in the graph"):
        readout.set_forward_rows(["DNx99"], [0.0], [1.0])
    with pytest.raises(ValueError, match="unique"):
        readout.set_forward_rows(["DNp09", "DNp09"], [0.0] * 2, [1.0] * 2)
    readout.set_forward_rows(TYPES, [0.0] * 3, [1.0] * 3)
    readout.set_forward_rows(TYPES, [0.1] * 3, [2.0] * 3)  # recalibration is allowed
    assert torch.allclose(readout.forward_row_scale, torch.full((3,), 2.0))
    with pytest.raises(ValueError, match="cannot change"):
        readout.set_forward_rows(["DNp09"], [0.0], [1.0])
    with pytest.raises(ValueError, match="cannot be removed"):
        readout.set_forward_rows([], [], [])


def test_rows_survive_the_artifact_round_trip(monkeypatch, tmp_path):
    config, policy = smoke_policy(monkeypatch, "VO-late")
    width = len(TYPES) * policy.branches
    policy.set_readout_forward_rows(TYPES, [0.01 * i for i in range(width)], [10.0] * width)
    with torch.no_grad():
        policy.readout.forward_row_weight.copy_(torch.linspace(-0.3, 0.3, width))
    expected = rollout(policy)
    save_policy_artifact(
        tmp_path / "actor",
        policy,
        policy.model_config,
        policy.contract,
        config["graph_path"],
        config.get("frontend_path"),
    )
    loaded = load_policy_artifact(tmp_path / "actor")[0].eval()
    assert loaded.model_config["readout_forward_row_center"] == pytest.approx(
        [0.01 * i for i in range(width)]
    )
    assert torch.equal(rollout(loaded), expected)


def test_bc_calibrates_forward_rows(monkeypatch, tmp_path):
    monkeypatch.chdir(ROOT)
    config = copy.deepcopy(read_config("configs/train/smoke.yaml"))
    config.pop("init_artifact", None)
    config["variant"] = "V"
    # The calibration skips a 3 s burn-in, so episodes must outlast it.
    config["env"]["episode_seconds"] = 5
    config["train"].update(
        teacher_steps=1000,
        dagger_rounds=0,
        epochs=1,
        calibrate_forward_rows={"types": TYPES, "burn_in_steps": 60},
    )
    imitation.train_bc(config, tmp_path / "bc", 0, torch.device("cpu"))
    report = json.loads((tmp_path / "bc/forward_row_calibration.json").read_text())
    model = report["model"]
    assert model["readout_forward_row_types"] == TYPES
    assert len(model["readout_forward_row_center"]) == len(TYPES)
    assert all(math.isfinite(x) for x in model["readout_forward_row_center"])
    assert all(x > 0 for x in model["readout_forward_row_scale"])
    assert report["burn_in_steps"] == 60 and report["training_steps"] > 0
    assert set(report["closed_form"]) == {"pool_only", "pool_plus_rows"}
    actor = load_policy_artifact(tmp_path / "bc/actor")[0]
    assert actor.model_config["readout_forward_row_types"] == TYPES
    for key in ("readout_forward_row_center", "readout_forward_row_scale"):
        assert actor.model_config[key] == pytest.approx(model[key])
    assert actor.readout.forward_row_weight.shape == (len(TYPES),)
