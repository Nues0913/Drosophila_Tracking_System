"""Formal matrix configs: #55 no-KC graph, #4 P2 encoder and B3 rows (2026-10-24 spec);
the chain ends at BC/DAgger (#35 Decision 6382764327)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml
from flytrack_core.contracts import SensorContract
from flytrack_core.model import (
    assert_parameter_budget,
    build_policy,
    count_trainable_parameters,
    load_policy_artifact,
)
from flytrack_train.config import save_actor
from flytrack_train.parameters import count_effective_parameters

ROOT = Path(__file__).resolve().parents[2]


def _prepare(tmp_path, monkeypatch, *extra):
    monkeypatch.chdir(ROOT)
    path = ROOT / "scripts/prepare_experiments.py"
    spec = importlib.util.spec_from_file_location("prepare_experiments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "configs"
    argv = ["prepare_experiments.py", "--seeds", "0", "--output", str(out), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()
    return out


def test_formal_configs_use_the_spec_and_budget_the_extra_turn_rows(tmp_path, monkeypatch):
    out = _prepare(tmp_path, monkeypatch)
    budget = json.loads((out / "parameter_budget.json").read_text())
    assert budget["extra_turn_rows"] == {
        "types": ["DNb05"],
        "eps": 0.1,
        "variants": ["O", "VO", "VO-late"],
    }
    counts = budget["actor_trainable_parameters"]
    # One row weight per readout branch on top of the #55 budget (VO 16,242).
    assert counts["VO"] == 16243 and counts["O"] - counts["V"] == counts["VO"] - counts["RS"] == 1
    for variant, model in budget["resolved_models"].items():
        assert model["gru_odor_gain_compensation"] is (variant == "GRU")
        assert model["readout_extra_turn_types"] == []  # BC adds and calibrates them.
        assert model["odor_shared_adaptation"] is True
        assert model["readout_extra_turn_center_ramp"] is True
        stages = sorted((out / variant / "seed_0").glob("*.yaml"))
        for stage in stages:
            config = yaml.safe_load(stage.read_text())
            assert config["model"]["gru_odor_gain_compensation"] is (variant == "GRU")
            assert config["graph_path"] == "data/subgraphs/olfactory_v2_nokc/graph.npz"
        bc = yaml.safe_load((out / variant / "seed_0/bc.yaml").read_text())
        assert bc["train"]["center_turn_bias"] is False
        rows = bc["train"].get("calibrate_extra_turn")
        if variant in ("V", "RS", "GRU"):
            assert rows is None
        else:
            assert rows == {"types": ["DNb05"], "eps": 0.1}
        rebuilt, _ = _build_bc_actor(bc)
        actual = count_effective_parameters(rebuilt, bc["train"])
        recorded = budget["parameter_counts"][variant]
        assert actual == {key: value for key, value in recorded.items() if key in actual}
        assert actual["declared"] == counts[variant]
        assert actual["declared"] == actual["zero_lr"] + actual["effective"]
        assert abs(recorded["relative_to_effective_vo_percent"]) <= 5
        assert model["budget_target"] == (14546 if variant == "GRU" else None)
    assert budget["reference_effective_parameters"] == 14546
    assert budget["actor_trainable_parameters_basis"].startswith("declared")
    assert "effective BC" in budget["budget_basis"]
    template = yaml.safe_load((ROOT / "configs/train/bc.yaml").read_text())["train"]
    assert budget["bc_learning_rates"] == {
        key: template.get(key)
        for key in (
            "learning_rate",
            "core_bias_learning_rate",
            "readout_learning_rate",
            "readout_bias_learning_rate",
        )
    }


def _build_bc_actor(config):
    contract = SensorContract.from_json(config["sensor_contract_path"])
    policy = build_policy(
        config["variant"],
        config["graph_path"],
        contract,
        config["frontend_path"],
        **config["model"],
    )
    rows = config["train"].get("calibrate_extra_turn")
    if rows:
        types = rows["types"]
        policy.set_readout_extra_turns(types, [0.0] * len(types), [1.0] * len(types), rows["eps"])
    return policy, contract


def test_legacy_gru_resolution_exceeds_effective_vo_budget():
    options = yaml.safe_load((ROOT / "configs/model/variants.yaml").read_text())
    config = yaml.safe_load((ROOT / "configs/train/bc.yaml").read_text())
    config["model"] = options["common"]
    reference, contract = _build_bc_actor(config)
    declared_target = count_trainable_parameters(reference)
    legacy = build_policy(
        "GRU",
        config["graph_path"],
        contract,
        config["frontend_path"],
        **{**options["common"], **options["overrides"]["GRU"], "budget_target": declared_target},
    )
    assert legacy.model_config["hidden_size"] == 7
    assert legacy.model_config["gru_input_size"] is None
    models = {"VO": reference, "GRU": legacy}
    counts = {
        name: count_effective_parameters(model, config["train"])["effective"]
        for name, model in models.items()
    }
    assert counts == {"VO": 14546, "GRU": 15516}
    assert_parameter_budget(models)  # The old declared comparison incorrectly passed.
    with pytest.raises(ValueError, match="effective parameter budgets exceed"):
        assert_parameter_budget(models, effective_counts=counts)


def test_resolved_gru_yaml_and_artifact_round_trip(tmp_path, monkeypatch):
    out = _prepare(tmp_path, monkeypatch, "--device", "cpu")
    budget = json.loads((out / "parameter_budget.json").read_text())
    config = yaml.safe_load((out / "GRU/seed_0/bc.yaml").read_text())
    policy, contract = _build_bc_actor(config)
    assert policy.model_config["hidden_size"] == budget["gru_resolution"]["hidden_size"]
    assert policy.model_config["gru_input_size"] == budget["gru_resolution"]["gru_input_size"]
    assert policy.model_config["gru_input_size"] is not None
    destination = tmp_path / "actor"
    save_actor(destination, policy, config, contract, stage="bc_dagger", seed=0)
    loaded, loaded_contract, card = load_policy_artifact(destination, device="cpu")
    assert loaded_contract.sha256() == contract.sha256()
    assert loaded.model_config == policy.model_config
    assert loaded.gru.hidden_size == policy.gru.hidden_size
    assert loaded.gru.input_size == policy.gru.input_size
    assert count_trainable_parameters(loaded) == budget["parameter_counts"]["GRU"]["effective"]
    assert card["metadata"]["actor_parameter_budget"] == count_effective_parameters(
        loaded, config["train"]
    )
    obs = {
        "vision": torch.full((1, 2, contract.n_ommatidia), 0.5),
        "odor": torch.tensor([[0.1, 0.7]]),
        "valid": torch.ones(1, 3),
    }
    with torch.no_grad():
        original = policy(obs, policy.initial_state(1, "cpu"))
        restored = loaded(obs, loaded.initial_state(1, "cpu"))
    for expected, actual in zip(original, restored, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("stage", ["pretrain_vision", "bc"])
def test_actor_card_counts_use_the_stage_train_section(tmp_path, stage):
    config = yaml.safe_load((ROOT / f"configs/train/{stage}.yaml").read_text())
    options = yaml.safe_load((ROOT / "configs/model/variants.yaml").read_text())
    config["model"] = options["common"]
    policy, contract = _build_bc_actor(config)
    config["train"].update(core_bias_learning_rate=None, readout_learning_rate=0)
    save_actor(tmp_path / "actor", policy, config, contract, stage=stage)
    card = json.loads((tmp_path / "actor/model_card.json").read_text())
    actual = card["metadata"]["actor_parameter_budget"]
    assert actual == count_effective_parameters(policy, config["train"])
    assert actual["zero_lr"] > 0
    assert all(row["name"].startswith("readout.") for row in actual["zero_lr_parameters"])


@pytest.mark.parametrize("algorithm", [None, "ppo", "es"])
def test_formal_chain_ends_at_bc_unless_an_algorithm_is_named(tmp_path, monkeypatch, algorithm):
    out = _prepare(tmp_path, monkeypatch, *(["--algorithm", algorithm] if algorithm else []))
    assert f"--algorithm {algorithm or 'none'} " in (out / "run_all.sh").read_text()
    for variant in ("V", "VO", "GRU"):
        stages = {path.stem for path in (out / variant / "seed_0").glob("*.yaml")}
        assert "bc" in stages and not stages & {"ppo", "es"} - {algorithm}
        bc = yaml.safe_load((out / variant / "seed_0/bc.yaml").read_text())
        assert bc["output"] == f"runs/bc/{variant}/seed_0"
        if algorithm:
            final = yaml.safe_load((out / variant / f"seed_0/{algorithm}.yaml").read_text())
            assert final["init_artifact"] == f"runs/bc/{variant}/seed_0/actor"
