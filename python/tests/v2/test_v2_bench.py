"""The v1 baseline benchmark's configuration and metadata (#139)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("v2_bench", ROOT / "scripts/v2/bench.py")
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)


def test_ppo_config_keeps_the_formal_batch_and_drops_the_warm_start(tmp_path):
    config = bench.ppo_config(2, tmp_path / "out")
    train = config["train"]
    assert (train["num_envs"], train["rollout_steps"]) == (8, 256)
    assert train["total_steps"] == 2 * 8 * 256
    assert "init_artifact" not in config and config["output"] == str(tmp_path / "out")
    paths = [config[key] for key in ("sensor_contract_path", "graph_path", "frontend_path")]
    assert all(Path(path).is_file() for path in [*paths, *config["scenario_configs"].values()])


def test_environment_identifies_the_host_only_by_its_label():
    info = bench.environment("2060", "cpu")
    assert info["host_label"] == "2060" and len(info["commit"]) == 40
    assert "hostname" not in info and "node" not in info
