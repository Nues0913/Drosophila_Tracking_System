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


def test_ppo_bench_creates_runs_and_reports_a_failing_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    calls = []

    def failing(updates, device, workdir):
        calls.append(workdir)
        raise RuntimeError("v1 PPO exited with 1")

    monkeypatch.setattr(bench, "_time_ppo", failing)
    try:
        bench.bench_ppo("cpu")
    except RuntimeError as error:
        assert "exited with 1" in str(error)
    assert (tmp_path / "runs").is_dir() and calls and calls[0].parent == tmp_path / "runs"


def test_env_bench_restores_the_thread_count(monkeypatch):
    import torch

    before = torch.get_num_threads()
    monkeypatch.setattr(bench, "_bench_env", lambda steps: {"threads": torch.get_num_threads()})
    assert bench.bench_env(1) == {"threads": 1}
    assert torch.get_num_threads() == before
