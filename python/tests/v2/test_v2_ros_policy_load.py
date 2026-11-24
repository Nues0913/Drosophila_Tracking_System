"""The ROS policy node loads schema 4 artifacts through load_policy only (#140)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from flytrack_core.contracts import SensorContract
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.interfaces import policy_step
from flytrack_v2.model import build_policy, export_policy, fit_input_norm, load_policy

POLICY_PACKAGE = Path(__file__).resolve().parents[3] / "ros2_ws/src/flytrack_policy/flytrack_policy"
PROVENANCE = {"git_sha": "a" * 40, "dirty": False, "config_sha256": "b" * 64, "stages": []}


def module(name):
    spec = importlib.util.spec_from_file_location(
        f"ros_policy_{name}", POLICY_PACKAGE / f"{name}.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


artifact_module, runtime_module = module("artifact"), module("runtime")


def check_contract(path, artifact_contract):
    # contract.py is a package module; its one dependency is flytrack_core.
    return module("contract").check_deployment_contract(path, artifact_contract)


@pytest.fixture(scope="module", params=["GRU2", "VO2"])
def artifact(request, tmp_path_factory):
    torch.manual_seed(140)
    policy = build_policy(request.param, {}, contract=SensorContractV2())
    generator = torch.Generator().manual_seed(140)
    with torch.no_grad():
        for param in policy.parameters():
            if param.requires_grad:
                param.add_(0.01 * torch.randn(param.shape, generator=generator))
        codes = torch.rand(12, policy.code_size, generator=generator)
        codes[:, -4:] = 1.0
        policy.input_norm = fit_input_norm(codes, policy.encoder.layout)
    return export_policy(policy, tmp_path_factory.mktemp(request.param), provenance=PROVENANCE)


def observations(steps, seed=0):
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        yield {
            "vision": torch.rand(1, 2, 721, generator=generator),
            "odor": torch.rand(1, 2, generator=generator),
            "valid": torch.ones(1, 3),
        }


@torch.no_grad()
def test_node_loader_steps_like_the_v2_api(artifact):
    policy, contract, card, spec = artifact_module.load_deployment_policy(artifact)
    assert spec == "v2" and card["spec_version"] == "v2"
    assert isinstance(contract, SensorContractV2)
    reference = load_policy(artifact)
    sensory = reference.initial_sensory_state(1, "cpu")
    central = reference.initial_central_state(1, "cpu")
    state = policy.initial_state(1, "cpu")
    for obs in observations(20):
        action, state, activity = policy(obs, state)
        v2_obs = {
            "vision": obs["vision"],
            "odor": obs["odor"],
            "wind": torch.zeros(1, 2),
            "valid": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        }
        step, sensory, _ = policy_step(reference, v2_obs, sensory, central)
        central = step.central_state
        torch.testing.assert_close(action, step.dist.mode(), rtol=0, atol=0)
        torch.testing.assert_close(state[1], central, rtol=0, atol=0)
        assert activity.shape == (1, len(policy.node_ids))


def test_runtime_arms_and_commands_within_limits(artifact):
    policy, contract, _card, _spec = artifact_module.load_deployment_policy(artifact)
    runtime = runtime_module.PolicyRuntime(policy, contract)
    rng = np.random.default_rng(0)
    commands = []
    for tick in range(10):
        now = 0.05 * (tick + 1)
        for eye in (0, 1):
            assert runtime.update_vision(eye, rng.random(721, dtype=np.float32), now)
        assert runtime.update_odor(rng.random(2, dtype=np.float32), now)
        command, activity, valid = runtime.tick(now)
        assert valid.all() and activity is not None
        commands.append(command)
    commands = np.array(commands)
    assert np.isfinite(commands).all()
    assert (commands[:, 0] >= 0).all() and (commands[:, 0] <= contract.v_max).all()
    assert (np.abs(commands[:, 1]) <= contract.omega_max).all()


def feed(runtime, rng, now, wind=True):
    for eye in (0, 1):
        assert runtime.update_vision(eye, rng.random(721, dtype=np.float32), now)
    assert runtime.update_odor(rng.random(2, dtype=np.float32), now)
    if wind:
        assert runtime.update_wind(rng.normal(size=2).astype(np.float32), now)


def test_wind_runtime_arms_only_with_the_antennae(artifact):
    policy, contract, _card, _spec = artifact_module.load_deployment_policy(artifact)
    runtime = runtime_module.PolicyRuntime(policy, contract, wind=True)
    rng = np.random.default_rng(1)
    feed(runtime, rng, 0.05, wind=False)
    command, activity, valid = runtime.tick(0.05)
    assert activity is None and not command.any() and valid.tolist() == [True] * 3 + [False]
    feed(runtime, rng, 0.10)
    _command, activity, valid = runtime.tick(0.10)
    assert activity is not None and valid.all()


@torch.no_grad()
def test_wind_runtime_feeds_the_antennae_like_the_v2_api(artifact):
    """The runtime's 4-sensor observation drives the policy as policy_step does, wind included;
    a missing antenna sample turns valid[3] off and zeroes the wind (wind * valid[:, 3])."""
    policy, contract, _card, _spec = artifact_module.load_deployment_policy(artifact)
    runtime = runtime_module.PolicyRuntime(policy, contract, wind=True)
    reference = load_policy(artifact)
    sensory = reference.initial_sensory_state(1, "cpu")
    central = reference.initial_central_state(1, "cpu")
    rng = np.random.default_rng(2)
    for tick in range(12):
        now = 0.05 * (tick + 1)
        dropped = tick in (6, 7, 8, 9, 10, 11)  # older than the watchdog from tick 10 on
        feed(runtime, rng, now, wind=not dropped)
        wind_before = runtime.wind.copy()
        command, activity, valid = runtime.tick(now)
        assert activity is not None
        expected_valid = [1.0, 1.0, 1.0, float(valid[3])]
        obs = runtime.history[-1]
        v2_obs = {
            "vision": torch.as_tensor(obs["vision"])[None],
            "odor": torch.as_tensor(obs["odor"])[None],
            "wind": torch.as_tensor(wind_before * np.float32(valid[3]))[None],
            "valid": torch.tensor([expected_valid]),
        }
        np.testing.assert_array_equal(obs["valid"], expected_valid)
        np.testing.assert_array_equal(obs["wind"], v2_obs["wind"][0].numpy())
        step, sensory, _ = policy_step(reference, v2_obs, sensory, central)
        central = step.central_state
        torch.testing.assert_close(torch.as_tensor(runtime.state[1]), central, rtol=0, atol=0)
        if tick >= 10:
            assert not valid[3] and not obs["wind"].any()


def test_three_sensor_runtime_refuses_wind(artifact):
    policy, contract, _card, _spec = artifact_module.load_deployment_policy(artifact)
    runtime = runtime_module.PolicyRuntime(policy, contract)
    assert runtime.sensor_count == 3
    with pytest.raises(ValueError, match="no wind sensor"):
        runtime.update_wind(np.zeros(2, dtype=np.float32), 0.05)


def write(tmp_path, contract, name):
    path = tmp_path / name
    path.write_text(json.dumps(contract.to_dict()), encoding="utf-8")
    return str(path)


def test_v1_and_v2_contracts_are_not_mixed(tmp_path):
    v1, v2 = SensorContract(), SensorContractV2()
    assert check_contract(write(tmp_path, v2, "v2.json"), v2).sha256() == v2.sha256()
    with pytest.raises(ValueError, match="differs from model artifact"):
        check_contract(write(tmp_path, v1, "v1.json"), v2)
    with pytest.raises(ValueError, match="differs from model artifact"):
        check_contract(write(tmp_path, v2, "v2.json"), v1)


def test_card_spec_selects_the_loader(tmp_path):
    card = tmp_path / "model_card.json"
    card.write_text(json.dumps({"variant": "VO"}), encoding="utf-8")
    assert artifact_module.artifact_spec(tmp_path) == "v1"
    card.write_text(json.dumps({"spec_version": "v3"}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported model card spec_version"):
        artifact_module.artifact_spec(tmp_path)


def test_tampered_v2_artifact_is_refused(artifact, tmp_path):
    copy = tmp_path / "copy"
    copy.mkdir()
    for file in artifact.iterdir():
        (copy / file.name).write_bytes(file.read_bytes())
    weights = bytearray((copy / "weights.pt").read_bytes())
    weights[-1] ^= 1
    (copy / "weights.pt").write_bytes(bytes(weights))
    with pytest.raises(ValueError, match="checksum mismatch"):
        artifact_module.load_deployment_policy(copy)
