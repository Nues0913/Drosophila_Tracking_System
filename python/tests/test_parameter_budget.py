"""Optimizer-effective actor budgets and artifact matrix validation (#77)."""

import json
from types import SimpleNamespace

import pytest
import torch
from flytrack_core.contracts import SensorContract
from flytrack_train import evaluation, parameters


@pytest.fixture
def policy():
    model = torch.nn.Module()
    model.core = torch.nn.Module()
    model.core.bias = torch.nn.Parameter(torch.zeros(3))
    model.core.raw_alpha = torch.nn.Parameter(torch.zeros(5))
    model.readout = torch.nn.Module()
    model.readout.turn_weight = torch.nn.Parameter(torch.zeros(2))
    model.readout.turn_bias = torch.nn.Parameter(torch.zeros(1))
    model.frontend = torch.nn.Linear(2, 2).requires_grad_(False)
    return model


@pytest.mark.parametrize("core_rate", [None, 0.001, 0])
def test_effective_counts_follow_optimizer_groups(policy, core_rate):
    train = {"core_bias_learning_rate": core_rate, "readout_learning_rate": 0.01}
    result = parameters.count_effective_parameters(policy, train)
    assert result["declared"] == 11
    assert result["zero_lr"] == (3 if core_rate == 0 else 0)
    assert result["effective"] == result["declared"] - result["zero_lr"]
    assert sum(group["numel"] for group in result["groups"]) == result["declared"]
    optimizer = torch.optim.Adam(parameters.parameter_groups(policy, train))
    for actual, expected in zip(result["groups"], optimizer.param_groups, strict=True):
        assert actual["lr"] == expected["lr"]
        assert actual["numel"] == sum(p.numel() for p in expected["params"] if p.requires_grad)


def test_zero_readout_rate_and_bias_override_count_the_correct_parameters(policy):
    result = parameters.count_effective_parameters(
        policy,
        {"core_bias_learning_rate": None, "readout_learning_rate": 0},
    )
    assert result["zero_lr"] == 3
    assert result["zero_lr_parameters"] == [
        {"name": "readout.turn_weight", "numel": 2},
        {"name": "readout.turn_bias", "numel": 1},
    ]
    assert result["effective"] == 8
    result = parameters.count_effective_parameters(
        policy, {"readout_learning_rate": 0, "readout_bias_learning_rate": 0.01}
    )
    assert result["zero_lr"] == 2 and result["effective"] == 9


def test_effective_count_uses_parameter_groups_itself(policy, monkeypatch):
    monkeypatch.setattr(
        parameters, "parameter_groups", lambda *_: [{"lr": 0, "params": list(policy.parameters())}]
    )
    result = parameters.count_effective_parameters(policy, {})
    assert result["effective"] == 0 and result["zero_lr"] == result["declared"] == 11


@pytest.fixture
def artifact_matrix(tmp_path, monkeypatch):
    contract = SensorContract()
    cards = {}
    for variant in ("VO", "GRU"):
        path = tmp_path / variant
        path.mkdir()
        (path / "model_card.json").write_text(json.dumps({"variant": variant}))
        cards[variant] = {
            "trainable_parameters": 100 if variant == "VO" else 150,
            "metadata": {"actor_parameter_budget": {"effective": 100}},
            "files": {"graph.npz": "graph", "frontend.npz": "frontend"},
        }

    def load(path, **kwargs):
        variant = path.name
        return SimpleNamespace(variant=variant), contract, cards[variant]

    monkeypatch.setattr(evaluation, "load_policy_artifact", load)
    config = {
        "sensor_contract": contract.to_dict(),
        "evaluation": {
            "require_equal_budget": False,
            "artifacts": {variant: {0: tmp_path / variant} for variant in cards},
        },
    }
    return config, cards


def test_evaluation_matches_effective_counts_even_when_declared_counts_differ(artifact_matrix):
    config, _ = artifact_matrix
    assert len(evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")) == 2


def test_evaluation_rejects_effective_mismatch_when_declared_counts_match(artifact_matrix):
    config, cards = artifact_matrix
    cards["GRU"]["trainable_parameters"] = cards["VO"]["trainable_parameters"]
    cards["GRU"]["metadata"]["actor_parameter_budget"]["effective"] = 106
    with pytest.raises(ValueError, match="effective parameter counts exceed"):
        evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")


def test_evaluation_rejects_pre_77_card_without_effective_counts(artifact_matrix):
    config, cards = artifact_matrix
    del cards["GRU"]["metadata"]["actor_parameter_budget"]
    with pytest.raises(ValueError, match="before #77: missing effective"):
        evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")
    config["evaluation"]["require_parameter_match"] = False
    assert len(evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")) == 2
