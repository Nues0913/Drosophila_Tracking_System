"""Optimizer-effective actor budgets and artifact matrix validation (#77)."""

import copy
import json
import re
import shutil
from pathlib import Path

import pytest
import torch
import yaml
from flytrack_core.contracts import SensorContract
from flytrack_core.model import build_policy
from flytrack_train import evaluation, parameters
from flytrack_train.config import save_actor


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


@pytest.fixture(scope="module")
def real_actor_artifacts(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path_factory.mktemp("budget_actors")
    options = yaml.safe_load((root / "configs/model/variants.yaml").read_text())
    bc = yaml.safe_load((root / "configs/train/bc.yaml").read_text())
    contract = SensorContract.from_json(root / bc["sensor_contract_path"])
    for name, target in (("VO", None), ("GRU", 14546), ("legacy_GRU", 16243)):
        variant = "GRU" if name == "legacy_GRU" else name
        config = copy.deepcopy(bc)
        config.update(variant=variant, seed=0)
        config["graph_path"] = str(root / bc["graph_path"])
        config["frontend_path"] = str(root / bc["frontend_path"])
        config.pop("init_artifact", None)
        policy = build_policy(
            variant,
            config["graph_path"],
            contract,
            config["frontend_path"],
            **{
                **options["common"],
                **options["overrides"].get(variant, {}),
                "budget_target": target,
            },
        )
        if variant == "VO":
            rows = options["extra_turn_rows"]
            policy.set_readout_extra_turns(rows["types"], [0.0], [1.0], rows["eps"])
        else:
            config["train"].pop("calibrate_extra_turn", None)
        config["model"] = copy.deepcopy(policy.model_config)
        save_actor(
            output / name,
            policy,
            config,
            contract,
            seed=0,
            stage="bc",
            algorithm="bc_dagger",
            environment_steps=0,
        )
    return output, contract


@pytest.fixture
def artifact_matrix(tmp_path, real_actor_artifacts):
    source, contract = real_actor_artifacts
    for name in ("VO", "GRU", "legacy_GRU"):
        shutil.copytree(source / name, tmp_path / name)
    config = {
        "sensor_contract": contract.to_dict(),
        "evaluation": {
            "artifacts": {variant: {0: tmp_path / variant} for variant in ("VO", "GRU")},
        },
    }
    return config, tmp_path


def _card_path(root, variant):
    return root / variant / "model_card.json"


def _edit_card(root, variant, field, value, *, delete=False):
    path = _card_path(root, variant)
    card = json.loads(path.read_text())
    container = card
    for key in field[:-1]:
        container = container[key]
    if delete:
        del container[field[-1]]
    else:
        container[field[-1]] = value
    path.write_text(json.dumps(card))


def test_evaluation_matches_effective_counts_even_when_declared_counts_differ(
    artifact_matrix, monkeypatch
):
    config, root = artifact_matrix
    load = evaluation.load_policy_artifact
    loaded = {}

    def record_load(path, **kwargs):
        assert path.name not in loaded  # Reuse each loaded actor for recomputation.
        result = load(path, **kwargs)
        loaded[path.name] = result[0]
        return result

    monkeypatch.setattr(evaluation, "load_policy_artifact", record_load)
    assert len(evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")) == 2
    assert set(loaded) == {"VO", "GRU"}
    assert loaded["VO"].readout.extra_turn_weight.numel() == 1  # B3 restored before counting.
    assert loaded["GRU"].model_config["hidden_size"] == 16
    assert loaded["GRU"].model_config["gru_input_size"] == 18
    cards = {name: json.loads(_card_path(root, name).read_text()) for name in loaded}
    assert cards["VO"]["trainable_parameters"] == 16243
    assert cards["GRU"]["trainable_parameters"] == 14904
    assert cards["VO"]["metadata"]["actor_parameter_budget"]["effective"] == 14546


def test_evaluation_rejects_legacy_effective_mismatch_when_declared_counts_fit(artifact_matrix):
    config, root = artifact_matrix
    config["evaluation"]["artifacts"]["GRU"][0] = root / "legacy_GRU"
    legacy = json.loads(_card_path(root, "legacy_GRU").read_text())
    assert legacy["model_config"]["hidden_size"] == 7
    assert legacy["metadata"]["actor_parameter_budget"]["effective"] == 15516
    assert abs(legacy["trainable_parameters"] / 16243 - 1) <= 0.05
    with pytest.raises(ValueError, match="effective parameter counts exceed"):
        evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")


def test_evaluation_rejects_pre_77_card_without_effective_counts(artifact_matrix):
    config, root = artifact_matrix
    _edit_card(root, "GRU", ("metadata", "actor_parameter_budget"), None, delete=True)
    with pytest.raises(ValueError, match="metadata.actor_parameter_budget") as exc:
        evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")
    assert str(root / "GRU") in str(exc.value)
    config["evaluation"]["require_parameter_match"] = False
    assert len(evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")) == 2


def test_evaluation_rejects_pm_effective_scalar_counterexample(artifact_matrix):
    config, root = artifact_matrix
    config["evaluation"]["artifacts"]["GRU"][0] = root / "legacy_GRU"
    _edit_card(root, "legacy_GRU", ("metadata", "actor_parameter_budget", "effective"), 14546)
    with pytest.raises(ValueError, match="metadata.actor_parameter_budget.effective") as exc:
        evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")
    assert str(root / "legacy_GRU") in str(exc.value)


@pytest.mark.parametrize(
    "field,value,delete",
    [
        (("declared",), 1, False),
        (("zero_lr",), 0, False),
        (("effective",), "14546", False),
        (("effective",), True, False),
        (("effective",), 14546, True),
        (("groups", 0, "lr"), 0.1, False),
        (("groups", 0, "lr"), float("nan"), False),
        (("groups", 0, "lr"), float("inf"), False),
        (("groups", 0, "lr"), "0.001", False),
        (("groups", 1, "lr"), 1e-15, False),
        (("groups", 0, "numel"), 1, False),
        (("groups", 0, "parameters", 0, "name"), "wrong", False),
        (("groups", 0, "parameters", 0, "numel"), 1, False),
        (("groups", 0, "parameters"), [], False),
        (("groups",), None, False),
        (("groups",), None, True),
        (("zero_lr_parameters", 0, "name"), "wrong", False),
        (("zero_lr_parameters", 0, "numel"), 1, False),
        (("zero_lr_parameters",), [], False),
        (("zero_lr_parameters",), None, True),
        ((), [], False),
        ((), None, False),
    ],
)
def test_evaluation_rejects_inconsistent_budget_details(artifact_matrix, field, value, delete):
    config, root = artifact_matrix
    full_field = ("metadata", "actor_parameter_budget", *field)
    _edit_card(root, "VO", full_field, value, delete=delete)
    name = "metadata.actor_parameter_budget"
    for key in field:
        name += f"[{key}]" if isinstance(key, int) else f".{key}"
    with pytest.raises(ValueError, match=re.escape(name)) as exc:
        evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")
    assert str(root / "VO") in str(exc.value)


@pytest.mark.parametrize("field", [("train",), ()])
@pytest.mark.parametrize("value,delete", [(None, True), (None, False), ([], False)])
def test_evaluation_requires_saved_train_mapping(artifact_matrix, field, value, delete):
    config, root = artifact_matrix
    _edit_card(root, "VO", ("metadata", "config", *field), value, delete=delete)
    with pytest.raises(ValueError, match="metadata.config.train") as exc:
        evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")
    assert str(root / "VO") in str(exc.value)


@pytest.mark.parametrize("rate", [0.001, -0.001, float("nan"), float("inf"), "invalid"])
def test_evaluation_rejects_changed_or_invalid_train_rates(artifact_matrix, rate):
    config, root = artifact_matrix
    _edit_card(root, "VO", ("metadata", "config", "train", "core_bias_learning_rate"), rate)
    with pytest.raises(ValueError, match="metadata.(actor_parameter_budget|config.train)") as exc:
        evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")
    assert str(root / "VO") in str(exc.value)


@pytest.mark.parametrize(
    "value,delete", [(1, False), (16243.0, False), (None, False), (None, True)]
)
def test_evaluation_checks_top_level_declared_count(artifact_matrix, value, delete):
    config, root = artifact_matrix
    _edit_card(root, "VO", ("trainable_parameters",), value, delete=delete)
    with pytest.raises(ValueError, match="trainable_parameters") as exc:
        evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")
    assert str(root / "VO") in str(exc.value)


def test_evaluation_accepts_roundoff_in_group_learning_rate(artifact_matrix):
    config, root = artifact_matrix
    _edit_card(root, "VO", ("metadata", "actor_parameter_budget", "groups", 0, "lr"), 0.001 + 1e-13)
    assert len(evaluation.validate_artifact_matrix(config, ["VO", "GRU"], [0], "cpu")) == 2
