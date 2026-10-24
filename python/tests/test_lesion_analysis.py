"""Integration checks for lesion-analysis world and held-out seed alignment."""

import json
import runpy
import sys
from pathlib import Path

import yaml
from flytrack_core.contracts import SensorContract
from flytrack_core.model import build_policy, save_policy_artifact
from flytrack_train.evaluation import evaluation_environment_seed

ROOT = Path(__file__).resolve().parents[2]


def test_evaluation_environment_seed_uses_heldout_paired_rule():
    assert evaluation_environment_seed(3, 1) == 1700300001
    assert evaluation_environment_seed(3, 1, 2**31 - 1) == 300001


def test_lesion_analysis_uses_evaluation_world_seed_and_success_settings(tmp_path, monkeypatch):
    contract = SensorContract()
    graph_path = ROOT / "data/subgraphs/olfactory_v2_nokc/graph.npz"
    frontend_path = ROOT / "artifacts/frontend/frontend.npz"
    policy = build_policy("O", graph_path, contract, frontend_path)
    artifact = tmp_path / "actor"
    save_policy_artifact(
        artifact,
        policy,
        policy.model_config,
        contract,
        graph_path,
        frontend_path,
        metadata={"seed": 3},
    )

    scenario_path = tmp_path / "occlusion.yaml"
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "name": "occlusion",
                "duration_s": 1,
                "occlusion_start_s": 0,
                "occlusion_duration_s": 1,
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "evaluate.yaml"
    config = {
        "sensor_contract": contract.to_dict(),
        "env": {"episode_seconds": 0.1, "target_motion": "stationary"},
        "scenario_configs": {"occlusion": str(scenario_path)},
        "evaluation": {
            "environment_seed": 1700000000,
            "success_distance_m": 0.000001,
            "success_time_fraction": 0.0,
            "success_bearing_deg": None,
        },
    }
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lesion_analysis.py",
            "--artifact",
            str(artifact),
            "--config",
            str(config_path),
            "--scenario",
            "occlusion",
            "--episodes",
            "2",
            "--output",
            str(output),
        ],
    )

    runpy.run_path(str(ROOT / "scripts/lesion_analysis.py"), run_name="__main__")

    rows = json.loads((output / "episodes.json").read_text(encoding="utf-8"))
    assert len(rows) == 6
    assert [row["environment_seed"] for row in rows] == [
        1700300000,
        1700300001,
        1700300000,
        1700300001,
        1700300000,
        1700300001,
    ]
    assert all(row["seed"] == 3 and row["environment_seed"] >= 1_000_000_000 for row in rows)
    assert all(row["occlusion_events"] == 1 for row in rows)
    assert all(row["success"] == 1.0 for row in rows)
