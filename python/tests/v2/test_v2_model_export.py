"""Portable artifacts preserve the learned v2 policy and its actual graph."""

import json
from dataclasses import fields
from time import perf_counter

import numpy as np
import pytest
import torch
from flytrack_core.contracts import file_sha256
from flytrack_v2.artifacts import VARIANTS, required_files
from flytrack_v2.contracts import SensorContractV2, obs_shapes
from flytrack_v2.interfaces import policy_step
from flytrack_v2.model import (
    ConnectomePolicyV2,
    GraphV2,
    ModelConfigV2,
    build_policy,
    export_policy,
    fit_input_norm,
    load_policy,
)
from flytrack_v2.model import policy as policy_module
from flytrack_v2.model import readout as readout_module

PROVENANCE = {"git_sha": "a" * 40, "dirty": False, "config_sha256": "b" * 64, "stages": []}


@pytest.mark.slow
@pytest.mark.parametrize(
    ("variant", "speed_types", "fit_readout"),
    [
        *((variant, None, False) for variant in VARIANTS),
        ("VO2", "bilateral", False),
        ("VO2", "bilateral", True),
    ],
    ids=[*VARIANTS, "VO2-bilateral", "VO2-bilateral-fitted"],
)
@torch.no_grad()
def test_policy_export_round_trip(tmp_path, monkeypatch, variant, speed_types, fit_readout):
    start = perf_counter()
    contract = SensorContractV2()
    policy = build_policy(
        variant,
        {"speed_types": speed_types, "visual_offset_init": 0.28},
        contract=contract,
        structure_seed=3 if variant in ("RW2", "ER2") else None,
    )
    generator = torch.Generator().manual_seed(161)
    for param in policy.parameters():
        if param.requires_grad:
            param.add_(0.01 * torch.randn(param.shape, generator=generator))
    fit_codes = torch.rand(12, policy.code_size, generator=generator)
    fit_codes[:, -4:] = 1.0
    policy.input_norm = fit_input_norm(fit_codes, policy.encoder.layout)
    if fit_readout:
        rates = torch.rand(64, len(policy.readout.node_indices), generator=generator)
        policy.fit_readout_norm(0.05 + 0.01 * rates)
    frontend_source = policy.encoder.frontend.source_path
    artifact = export_policy(policy, tmp_path / "export", provenance=PROVENANCE)
    assert set(required_files(variant)) == {file.name for file in artifact.iterdir()}
    assert (artifact / "frontend.npz").read_bytes() == frontend_source.read_bytes()
    card = json.loads((artifact / "model_card.json").read_text(encoding="utf-8"))
    config = card["model_config"]
    assert {field.name for field in fields(ModelConfigV2)} <= config.keys()
    assert config["visual_offset_init"] == 0.28
    assert "bilateral" not in json.dumps(config)
    assert card["provenance"] == PROVENANCE

    def unexpected(*_args, **_kwargs):
        pytest.fail("Loading must use the exported structure and readout weights")

    monkeypatch.setattr(policy_module, "_ROOT", tmp_path / "unavailable_repo")
    monkeypatch.setattr(readout_module, "_bilateral_speed_groups", unexpected)
    monkeypatch.setattr(policy_module._PolicyV2, "_calibrate_readout", unexpected)
    monkeypatch.setattr(policy_module, "rewire_rw2", unexpected)
    monkeypatch.setattr(policy_module, "erdos_renyi_er2", unexpected)
    restored = load_policy(artifact)

    if isinstance(policy, ConnectomePolicyV2):
        assert isinstance(restored, ConnectomePolicyV2)
        assert not torch.equal(
            policy.projection.visual_offset, torch.full_like(policy.projection.visual_offset, 0.28)
        )
        assert bool(restored.readout.rate_norm_fitted) is fit_readout
        for field in fields(GraphV2):
            if field.name == "source_sha256":
                continue
            actual, expected = (
                getattr(restored.graph, field.name),
                getattr(policy.graph, field.name),
            )
            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual, expected, strict=True)
            else:
                assert actual == expected
        assert restored.graph.source_sha256 == card["files"]["graph.npz"]
        assert card["graph_origin_sha256"] == policy.graph.source_sha256
        assert restored.graph.source_sha256 != restored.graph.origin_sha256
        assert config["readout_body_ids"] == {
            "steer": policy.readout.steer_body_ids,
            "speed": policy.readout.speed_body_ids,
        }
        if speed_types == "bilateral":
            assert len(config["speed_types"]) == 31
    else:
        assert config["budget_target"] == policy.config.budget_target
        assert config["hidden_size"] == policy.central_state_size
    assert restored.input_norm.fitted and restored.input_norm.n_samples == len(fit_codes)
    for name, tensor in policy.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, atol=0, rtol=0)
    assert restored.state_dict().keys() == policy.state_dict().keys()

    batch = 3
    sensory = [actor.initial_sensory_state(batch, "cpu") for actor in (policy, restored)]
    central = [actor.initial_central_state(batch, "cpu") for actor in (policy, restored)]
    max_mean_error = max_state_error = 0.0
    for step in range(1, 31):
        obs = {
            name: torch.rand(shape, generator=generator)
            for name, shape in obs_shapes(contract, batch).items()
        }
        obs["valid"] = (obs["valid"] > 0.25).float()
        reset = torch.tensor([True, False, False]) if step == 15 else None
        expected, sensory[0], expected_code = policy_step(
            policy, obs, sensory[0], central[0], reset
        )
        actual, sensory[1], actual_code = policy_step(restored, obs, sensory[1], central[1], reset)
        central = [expected.central_state, actual.central_state]
        for a, b in (
            (actual_code, expected_code),
            (actual.dist.mean, expected.dist.mean),
            (actual.dist.log_std, expected.dist.log_std),
            (actual.central_state, expected.central_state),
        ):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        if fit_readout:
            torch.testing.assert_close(actual.dist.mode(), expected.dist.mode(), atol=0, rtol=0)
        max_mean_error = max(
            max_mean_error, (actual.dist.mean - expected.dist.mean).abs().max().item()
        )
        max_state_error = max(max_state_error, (central[1] - central[0]).abs().max().item())
    print(
        f"{variant} ({speed_types or 'default'}): "
        f"mean={max_mean_error:.9g}, central_state={max_state_error:.9g}, "
        f"elapsed={perf_counter() - start:.3f}s"
    )


@pytest.fixture(scope="module")
def vo2_artifact(tmp_path_factory):
    policy = build_policy("VO2", {"rest_settle_s": 0.05, "rest_refresh_s": 0.05})
    return export_policy(policy, tmp_path_factory.mktemp("vo2_artifact"), provenance=PROVENANCE)


@pytest.fixture
def artifact_copy(vo2_artifact, tmp_path):
    for path in vo2_artifact.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    return tmp_path


def test_export_detects_modified_weights(artifact_copy):
    weights = artifact_copy / "weights.pt"
    with weights.open("r+b") as stream:
        first_byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first_byte[0] ^ 1]))
    with pytest.raises(ValueError, match="SHA256SUMS checksum mismatch: weights.pt"):
        load_policy(artifact_copy)


def _write_vo2_card(artifact, card):
    (artifact / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
    (artifact / "SHA256SUMS").write_text(
        "".join(
            f"{file_sha256(artifact / name)}  {name}\n"
            for name in required_files("VO2")
            if name != "SHA256SUMS"
        ),
        encoding="ascii",
    )


@torch.no_grad()
def test_load_legacy_card_without_visual_offset(artifact_copy):
    reference = load_policy(artifact_copy)
    card = json.loads((artifact_copy / "model_card.json").read_text(encoding="utf-8"))
    assert card["model_config"].pop("visual_offset_init") is None
    _write_vo2_card(artifact_copy, card)
    restored = load_policy(artifact_copy)
    assert restored.config.visual_offset_init is None
    assert restored.projection.visual_offset is None
    assert "projection.visual_offset" not in restored.state_dict()
    torch.testing.assert_close(restored.state_dict(), reference.state_dict(), atol=0, rtol=0)
    state = restored.initial_central_state(1, "cpu")
    torch.testing.assert_close(state, reference.initial_central_state(1, "cpu"), atol=0, rtol=0)
    code = torch.rand(1, reference.code_size, generator=torch.Generator().manual_seed(184))
    code[:, -4:] = 1
    actual = restored.forward_code(code, state)
    expected = reference.forward_code(code, state)
    torch.testing.assert_close(actual.central_state, expected.central_state, atol=0, rtol=0)
    torch.testing.assert_close(actual.dist.mean, expected.dist.mean, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("variant", "model_config.variant"),
        ("missing_body_id", "is missing from the graph"),
        ("wrong_side", "must be on the left side"),
        ("files", "Model card files"),
        ("sensor_contract_sha256", "does not match the sensor contract"),
        ("v1", "Not a v2 model card"),
    ],
)
def test_export_detects_modified_card(artifact_copy, change, message):
    card_path = artifact_copy / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if change == "variant":
        card["variant"] = "V2"
    elif change in ("missing_body_id", "wrong_side"):
        group = card["model_config"]["readout_body_ids"]["speed"]["DNp09"]
        group["left"] = [-1] if change == "missing_body_id" else group["right"]
    elif change == "files":
        card["files"]["weights.pt"] = "0" * 64
    elif change == "sensor_contract_sha256":
        card["sensor_contract_sha256"] = "0" * 64
    else:
        del card["spec_version"]
        card["schema_version"] = 1
    # Recompute the manifest so these checks reach the model card and reconstructed policy.
    _write_vo2_card(artifact_copy, card)
    with pytest.raises(ValueError, match=message):
        load_policy(artifact_copy)


@torch.no_grad()
def test_load_card_before_partial_odor_adaptation(artifact_copy):
    """#138 A2 eighth version: a card written before odor_adaptation_fraction loads with 1.0."""
    reference = load_policy(artifact_copy)
    card = json.loads((artifact_copy / "model_card.json").read_text(encoding="utf-8"))
    assert card["model_config"].pop("odor_adaptation_fraction") == 1.0
    _write_vo2_card(artifact_copy, card)
    restored = load_policy(artifact_copy)
    assert restored.config.odor_adaptation_fraction == 1.0
    torch.testing.assert_close(restored.state_dict(), reference.state_dict(), atol=0, rtol=0)
