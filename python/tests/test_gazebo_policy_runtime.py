"""Policy runtime settings for Gazebo cells (#88 G0b): torch threads and device."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def matrix_module():
    spec = importlib.util.spec_from_file_location(
        "gazebo_matrix_runtime_test", ROOT / "scripts/run_gazebo_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_matrix(tmp_path, monkeypatch, *extra):
    matrix = matrix_module()
    monkeypatch.setattr(matrix, "__file__", str(tmp_path / "scripts/run_gazebo_matrix.py"))
    card = tmp_path / "runs/bc/V/seed_0/actor/model_card.json"
    card.parent.mkdir(parents=True)
    card.write_text(json.dumps({"variant": "V", "metadata": {"seed": 0}}))
    digest = matrix.sha256(card)
    commands, environments = [], []

    def fake_run(command, **kwargs):
        commands.append(command)
        if "up" in command:
            env = kwargs["env"]
            environments.append(env)
            cell = tmp_path / env["EVAL_OUTPUT"].removeprefix("/workspace/")
            row = {
                "variant": "V",
                "seed": 0,
                "model_card_sha256": digest,
                "target_motion": "curved",
            }
            (cell / "episodes.jsonl").write_text(json.dumps(row) + "\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["matrix", "--variants", "V", "--scenarios", "normal", "--seeds", "0", "--episodes", "1"]
        + list(extra),
    )
    matrix.main()
    manifest = json.loads((tmp_path / "runs/gazebo_eval/gazebo_matrix_manifest.json").read_text())
    return commands, environments, manifest


def test_matrix_defaults_to_one_cpu_thread_and_records_it(tmp_path, monkeypatch):
    commands, environments, manifest = run_matrix(tmp_path, monkeypatch, "--cpu")
    assert environments[0]["POLICY_TORCH_THREADS"] == "1"
    assert environments[0]["POLICY_DEVICE"] == "cpu"
    assert manifest["policy_torch_threads"] == 1
    assert manifest["policy_device"] == "cpu"
    assert environments[0]["POLICY_VISION_KERNEL"] == "subrays"
    assert manifest["policy_vision_kernel"] == "subrays"
    assert manifest["cpu_rendering"] is True
    assert not any(str(part).endswith("compose.policy-gpu.yaml") for c in commands for part in c)


def test_matrix_forwards_policy_threads_and_cuda_override(tmp_path, monkeypatch):
    commands, environments, manifest = run_matrix(
        tmp_path, monkeypatch, "--policy-threads", "2", "--policy-device", "cuda"
    )
    assert environments[0]["POLICY_TORCH_THREADS"] == "2"
    assert environments[0]["POLICY_DEVICE"] == "cuda"
    assert manifest["policy_torch_threads"] == 2 and manifest["policy_device"] == "cuda"
    assert manifest["cpu_rendering"] is False
    up = next(c for c in commands if "up" in c)
    assert any(str(part).endswith("compose.policy-gpu.yaml") for part in up)


def test_matrix_forwards_and_records_the_vision_kernel(tmp_path, monkeypatch):
    _, environments, manifest = run_matrix(tmp_path, monkeypatch, "--vision-kernel", "gaussian")
    assert environments[0]["POLICY_VISION_KERNEL"] == "gaussian"
    assert manifest["policy_vision_kernel"] == "gaussian"


def test_matrix_rejects_unknown_vision_kernel(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        run_matrix(tmp_path, monkeypatch, "--vision-kernel", "box")


@pytest.mark.parametrize("threads", ["0", "-1"])
def test_matrix_rejects_nonpositive_policy_threads(tmp_path, monkeypatch, threads):
    with pytest.raises(SystemExit):
        run_matrix(tmp_path, monkeypatch, "--policy-threads", threads)


def test_compose_policy_threads_device_and_shared_environment():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
    shared = compose["x-ros"]["environment"]
    policy = compose["services"]["policy"]
    environment = policy["environment"]
    # The policy replaces the anchor's environment, so the shared keys must stay identical.
    assert {key: environment[key] for key in shared} == shared
    assert environment["OMP_NUM_THREADS"] == "${POLICY_TORCH_THREADS:-1}"
    assert environment["MKL_NUM_THREADS"] == "${POLICY_TORCH_THREADS:-1}"
    assert "device:=${POLICY_DEVICE:-cpu}" in policy["command"]
    assert "vision_kernel:=${POLICY_VISION_KERNEL:-subrays}" in policy["command"]
    override = yaml.safe_load((ROOT / "compose.policy-gpu.yaml").read_text())
    assert override == {"services": {"policy": {"gpus": "all"}}}
