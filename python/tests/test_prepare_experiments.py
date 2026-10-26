"""Formal matrix configs: #55 no-KC graph, #4 P2 encoder and B3 rows (2026-10-24 spec);
the chain ends at BC/DAgger (#35 Decision 6382764327)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

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
        assert model["readout_extra_turn_types"] == []  # BC adds and calibrates them.
        assert model["odor_shared_adaptation"] is True
        assert model["readout_extra_turn_center_ramp"] is True
        stages = sorted((out / variant / "seed_0").glob("*.yaml"))
        for stage in stages:
            config = yaml.safe_load(stage.read_text())
            assert config["graph_path"] == "data/subgraphs/olfactory_v2_nokc/graph.npz"
        bc = yaml.safe_load((out / variant / "seed_0/bc.yaml").read_text())
        assert bc["train"]["center_turn_bias"] is True
        rows = bc["train"].get("calibrate_extra_turn")
        if variant in ("V", "RS", "GRU"):
            assert rows is None
        else:
            assert rows == {"types": ["DNb05"], "eps": 0.1}


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
