import importlib.util
import json
from pathlib import Path

import pytest
import yaml


def test_calibration_is_isolated_bounded_and_preserves_formal_budgets(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "graph_calibration", root / "scripts/diagnostics/prepare_graph_calibration.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(root)
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps({"resolved_models": {"VO": {}, "V": {}, "O": {}}}))
    graph = tmp_path / "graph.npz"
    graph.write_bytes(b"test graph identity")
    output = tmp_path / "calibration"
    report = module.prepare(budget, graph, output, 1)
    assert len(report["stages"]) == 10
    assert report["formal_bc_steps_per_chain"] == 60000 + 3 * 30000
    assert report["formal_ppo_steps_per_chain"] == 2000000
    assert report["full_matrix_48_to_72h_verified"] is False
    destinations = set()
    for row in report["stages"]:
        config = yaml.safe_load(Path(row["config"]).read_text())
        assert Path(config["output"]).is_relative_to(output)
        assert config["output"] not in destinations
        destinations.add(config["output"])
        assert "init_artifact" not in config
        assert config["graph_path"] == str(graph)
        assert config["seed"] == 1
        if row["stage"] == "ppo":
            assert config["train"]["total_steps"] == 8192
            assert row["formal_template_train"]["total_steps"] == 2000000
            assert config["train"]["num_envs"] == 8
            assert config["train"]["sequence_batch_size"] == 16
        elif row["stage"] == "bc":
            assert config["train"]["teacher_steps"] == 12000
            assert config["train"]["dagger_rounds"] == 1
            assert row["formal_template_train"]["dagger_rounds"] == 3
        else:
            assert config["train"]["total_steps"] == 12000
            assert config["train"]["epochs"] == 2
    with pytest.raises(FileExistsError):
        module.prepare(budget, graph, output, 1)
