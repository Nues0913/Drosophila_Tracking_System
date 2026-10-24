import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from flytrack_core.contracts import file_sha256

ROOT = Path(__file__).resolve().parents[2]


def load_script():
    path = ROOT / "scripts/diagnostics/prepare_odor_task_bc.py"
    spec = importlib.util.spec_from_file_location("prepare_odor_task_bc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def actor(tmp_path, graph):
    directory = tmp_path / "actor"
    directory.mkdir()
    card = {"files": {"graph.npz": file_sha256(graph)}}
    (directory / "model_card.json").write_text(json.dumps(card))
    return directory


def run(monkeypatch, *argv):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(sys, "argv", ["prepare_odor_task_bc.py", *argv])
    load_script().main()


def test_graph_path_follows_the_init_actor_not_bc_yaml(tmp_path, monkeypatch):
    # bc.yaml has used the no-KC graph since 2026-10-24 (#55); an actor on main stays on main.
    main = Path("data/subgraphs/main/graph.npz")
    init = actor(tmp_path, ROOT / main)
    out = tmp_path / "out"
    run(monkeypatch, "--init-artifact", str(init), "--seed", "1", "--output", str(out))
    base = yaml.safe_load((ROOT / "configs/train/bc.yaml").read_text())
    for arm in "ABC":
        config = yaml.safe_load((out / f"{arm}.yaml").read_text())
        assert config["graph_path"] == str(main) != base["graph_path"]
    assert json.loads((out / "plan.json").read_text())["graph_path"] == str(main)


def test_explicit_graph_must_match_the_init_actor(tmp_path, monkeypatch):
    init = actor(tmp_path, ROOT / "data/subgraphs/olfactory_v2_nokc/graph.npz")
    with pytest.raises(SystemExit, match="SHA-256"):
        run(
            monkeypatch,
            *("--init-artifact", str(init), "--seed", "0", "--output", str(tmp_path / "o")),
            *("--graph", "data/subgraphs/main/graph.npz"),
        )
