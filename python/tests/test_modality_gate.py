"""CPU-only runner gate checks; subprocesses and training artifacts are simulated."""

import importlib.util
import json
import math
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[2] / "scripts/run_experiments.py"
    spec = importlib.util.spec_from_file_location("modality_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sensitivity(mean):
    return {
        "odor_swap_lr": {"mean": mean},
        "odor_to_vision_ratio": 1e-9,
        "no_odor": {"mean": 0.02},
        "no_vision": {"mean": 0.3},
    }


@pytest.mark.parametrize(
    "bc,start,absolute,relative",
    [
        (0.01, 0.001, True, True),
        (1e-3, 1e-4, True, True),
        (math.nextafter(1e-3, 0), 1e-4, False, True),
        (3e-3, 1e-3, True, True),
        (math.nextafter(3e-3, 0), 1e-3, True, False),
        (1e-3, 0.0, True, True),
        (float("nan"), 0.0, False, False),
        (float("inf"), 0.0, False, False),
    ],
)
def test_g4_thresholds(runner, bc, start, absolute, relative):
    metrics = runner.modality_gate("VO", sensitivity(bc), sensitivity(start))
    assert metrics["checks"] == {"absolute": absolute, "relative": relative}
    assert metrics["passed"] is (absolute and relative)
    assert metrics["relative_floor"] == runner.MODALITY_RELATIVE_FACTOR * start
    assert metrics["absolute_floor"] == runner.MODALITY_ABSOLUTE_FLOOR
    assert metrics["bc"]["odor_to_vision_ratio"] == 1e-9  # Never a gate.
    assert metrics["bc"]["no_odor_mean"] == 0.02
    assert metrics["start"]["no_vision_mean"] == 0.3


@pytest.mark.parametrize("mean,passed", [(1e-3, True), (0.0, False)])
def test_g4_without_start(runner, mean, passed):
    metrics = runner.modality_gate("V", sensitivity(mean))
    assert metrics["start"] is None
    assert metrics["checks"] == {"absolute": passed, "relative": None}
    assert metrics["relative_floor"] is None
    assert metrics["passed"] is passed
    assert metrics["enforced"] is False


def test_best_epochs_are_selected_once_per_round(runner, tmp_path):
    path = tmp_path / "training_metrics.json"
    assert runner.bc_best_epochs(tmp_path) is None
    path.write_text(
        json.dumps(
            [
                {"round": 0, "epochs": [{"best_epoch": 1}, {"best_epoch": 2}, {"best_epoch": 2}]},
                {"round": 1, "epochs": [{"best_epoch": 1}]},
                {"round": 2, "epochs": [{"epoch": 1}]},
            ]
        )
    )
    assert runner.bc_best_epochs(tmp_path) == [2, 1, None]
    path.write_text('[{"epochs": [{"epoch": 1}]}]')
    assert runner.bc_best_epochs(tmp_path) is None
    path.write_text("incomplete JSON")
    assert runner.bc_best_epochs(tmp_path) is None


@pytest.fixture
def matrix(runner, tmp_path, monkeypatch):
    """Build arbitrary output layouts and fake only the child processes."""
    configs = tmp_path / "configs"
    rows, calls, trained = {}, [], set()
    health_failures, health_codes, missing_health = {}, {}, set()
    swap_values, sensitivity_errors = {}, set()
    resumed = True

    def prepare(variants, *, seeds=(0,), algorithm="none", resume=True):
        nonlocal resumed
        resumed = resume
        for variant in variants:
            stages = (
                ["pretrain_vision"]
                if variant == "V"
                else ["pretrain_odor"]
                if variant == "O"
                else ["pretrain_vision", "pretrain_odor"]
            )
            for seed in seeds:
                for stage in [*stages, "bc", *([algorithm] if algorithm != "none" else [])]:
                    path = configs / variant / f"seed_{seed}" / f"{stage}.yaml"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Deliberately differs from the standard runs/pretrain layout.
                    output = tmp_path / "artifacts" / f"seed-{seed}" / stage / variant
                    output.mkdir(parents=True, exist_ok=True)
                    train = (
                        {"teacher_steps": 1, "dagger_rounds": 0, "dagger_steps_per_round": 1}
                        if stage == "bc"
                        else {"total_steps": 1}
                    )
                    row = {"variant": variant, "seed": seed, "output": str(output), "train": train}
                    rows[str(path)] = row
                    path.write_text(yaml.safe_dump(row))
                    if stage.startswith("pretrain"):
                        validation = output / "validation"
                        validation.mkdir()
                        (validation / "metrics.json").write_text('{"passed": true}')
        argv = [
            "run_experiments.py",
            "--configs",
            str(configs),
            "--variants",
            *variants,
            "--seeds",
            *map(str, seeds),
            "--algorithm",
            algorithm,
            "--skip-environment-audit",
            "--keep-going",
        ]
        monkeypatch.setattr(sys, "argv", argv)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if Path(command[1]).name == "modality_sensitivity.py":
            output, config_path, *actors = command[2:]
            variant = rows[config_path]["variant"]
            if variant in sensitivity_errors:
                return SimpleNamespace(returncode=1)
            policies = {
                actor: sensitivity(swap_values.get(variant, 0.01) if index == 0 else 1e-4)
                for index, actor in enumerate(actors)
            }
            Path(output).write_text(json.dumps({"policies": policies}))
            return SimpleNamespace(returncode=0)
        row = rows[command[command.index("--config") + 1]]
        variant, output = row["variant"], Path(row["output"])
        if Path(command[1]).name == "dn_health.py":
            stage = command[command.index("--stage") + 1]
            key = variant, stage
            failed = health_failures.get(key, ())
            checks = {
                name: True
                for name in runner.DN_HEALTH_ENGINEERING_CHECKS | runner.DN_HEALTH_QUALITY_CHECKS
            }
            checks.update({name: False for name in failed})
            code = health_codes.get(key, 2 if failed else 0)
            if key not in missing_health:
                path = output / "dn_health" / "metrics.json"
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({"checks": checks, "passed": all(checks.values())}))
            return SimpleNamespace(returncode=code)
        assert command[1:3] == ["-m", "flytrack_train"]
        trained.add(row["output"])
        return SimpleNamespace(returncode=0)

    def plan():
        return json.loads((configs / "execution_plan.json").read_text())["stages"]

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(
        runner, "completed", lambda config, *_: resumed or config["output"] in trained
    )
    return SimpleNamespace(
        prepare=prepare,
        plan=plan,
        calls=calls,
        rows=rows,
        health_failures=health_failures,
        health_codes=health_codes,
        missing_health=missing_health,
        swap_values=swap_values,
        sensitivity_errors=sensitivity_errors,
    )


@pytest.mark.parametrize("variant", ["O", "VO", "VO-late"])
def test_enforced_failure_marks_row_and_keeps_other_chains_running(runner, matrix, variant):
    assert variant in runner.MODALITY_ENFORCED_VARIANTS
    matrix.prepare([variant, "V"])
    matrix.swap_values[variant] = 0.0
    with pytest.raises(SystemExit, match="Modality gate failed"):
        runner.main()
    rows = [row for row in matrix.plan() if row["stage"] == "bc"]
    assert [row["status"] for row in rows] == ["modality_gate_failed", "verified_complete"]
    metrics = json.loads((Path(rows[0]["output"]) / "modality/metrics.json").read_text())
    assert metrics["enforced"] is True and metrics["passed"] is False


@pytest.mark.parametrize("variant", ["V", "GRU", "RS"])
def test_recording_only_variants_do_not_block(runner, matrix, variant):
    matrix.prepare([variant])
    matrix.swap_values[variant] = 0.0
    runner.main()
    row = matrix.plan()[-1]
    assert row["status"] == "verified_complete"
    metrics = json.loads((Path(row["output"]) / "modality/metrics.json").read_text())
    assert metrics["enforced"] is False and metrics["passed"] is False
    assert (metrics["start"] is None) is (variant == "V")


@pytest.mark.parametrize("variant", ["O", "V", "GRU", "RS"])
def test_sensitivity_errors_are_persisted_and_respect_enforcement(runner, matrix, variant):
    matrix.prepare([variant])
    matrix.sensitivity_errors.add(variant)
    if variant in runner.MODALITY_ENFORCED_VARIANTS:
        with pytest.raises(SystemExit, match="Modality gate failed"):
            runner.main()
    else:
        runner.main()
    row = matrix.plan()[-1]
    metrics = json.loads((Path(row["output"]) / "modality/metrics.json").read_text())
    assert metrics["passed"] is False
    assert "Sensitivity exited 1" in metrics["error"]


@pytest.mark.parametrize("resume", [False, True])
def test_bc_gates_run_fresh_and_use_same_chain_start(runner, matrix, monkeypatch, resume):
    matrix.prepare(["VO", "V"], seeds=(0, 7), resume=resume)
    for name in ("STEPS", "SEED", "BUILD"):
        monkeypatch.setenv(name, "invalid-inherited-override")
    monkeypatch.setenv("DEVICE", "cuda")
    runner.main()
    first = matrix.plan()
    first_metrics = [
        json.loads((Path(row["output"]) / "modality/metrics.json").read_text())
        for row in first
        if row["stage"] == "bc"
    ]
    matrix.calls.clear()
    runner.main()
    second = matrix.plan()
    for row, original in zip([r for r in second if r["stage"] == "bc"], first_metrics, strict=True):
        assert json.loads((Path(row["output"]) / "modality/metrics.json").read_text()) == original
    health = [c for c, _ in matrix.calls if Path(c[1]).name == "dn_health.py"]
    modality = [(c, kw) for c, kw in matrix.calls if Path(c[1]).name == "modality_sensitivity.py"]
    assert len(health) == 10 and len(modality) == 4
    for command, kwargs in modality:
        config = matrix.rows[command[3]]
        assert command[4] == str(Path(config["output"]) / "actor")
        assert kwargs["env"]["DEVICE"] == "cpu"
        assert not {"STEPS", "SEED", "BUILD"} & kwargs["env"].keys()
        if config["variant"] == "V":
            assert len(command) == 5
        else:
            start = next(
                row
                for row in second
                if row["stage"] == "pretrain_odor"
                and (row["variant"], row["seed"]) == (config["variant"], config["seed"])
            )
            assert command[5] == str(Path(start["output"]) / "actor")


@pytest.mark.parametrize(
    "quality",
    sorted(
        [
            "omega_correlation",
            "omega_r2",
            "omega_amplitude",
            "eye_swap_response",
        ]
    ),
)
def test_terminal_bc_quality_failure_continues_to_modality(runner, matrix, quality):
    assert quality in runner.DN_HEALTH_QUALITY_CHECKS
    matrix.prepare(["O", "V"], resume=False)
    matrix.health_failures["O", "bc"] = [quality]
    runner.main()
    row = next(r for r in matrix.plan() if r["variant"] == "O" and r["stage"] == "bc")
    assert row["status"] == "complete"
    assert row["dn_health_quality_failures"] == [quality]
    assert (Path(row["output"]) / "modality/metrics.json").is_file()
    assert matrix.plan()[-1]["status"] == "complete"


def test_resumed_terminal_bc_quality_failure_preserves_verified_complete(runner, matrix):
    matrix.prepare(["O"], resume=True)
    matrix.health_failures["O", "bc"] = ["omega_r2"]
    runner.main()
    row = matrix.plan()[-1]
    assert row["status"] == "verified_complete"
    assert row["dn_health_quality_failures"] == ["omega_r2"]
    assert (Path(row["output"]) / "modality/metrics.json").is_file()
    assert not any(command[1:3] == ["-m", "flytrack_train"] for command, _ in matrix.calls)


@pytest.mark.parametrize(
    "failure",
    [
        "finite",
        "not_all_silent",
        "active_motor_count",
        "central_gradient",
        "new_unknown_check",
    ],
)
def test_terminal_bc_engineering_halts_even_with_keep_going(runner, matrix, failure):
    matrix.prepare(["O", "V"])
    matrix.health_failures["O", "bc"] = [failure, "omega_r2"]
    with pytest.raises(SystemExit, match="DN health engineering failure"):
        runner.main()
    rows = matrix.plan()
    row = next(r for r in rows if r["variant"] == "O" and r["stage"] == "bc")
    assert row["status"] == "dn_health_failed"
    assert row["dn_health_engineering_failures"] == [failure]
    assert row["dn_health_quality_failures"] == ["omega_r2"]
    assert all("status" not in r for r in rows if r["variant"] == "V")
    assert not (Path(row["output"]) / "modality/metrics.json").exists()


def test_engineering_halt_lets_running_stage_finish_but_stops_its_next_stage(
    runner,
    matrix,
    monkeypatch,
):
    matrix.prepare(["O", "V"])
    matrix.health_failures["O", "bc"] = ["finite"]
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--jobs", "2"])
    running, halted = threading.Event(), threading.Event()
    check_health, write = runner.check_dn_health, runner.write_json

    def check(python, config_path, destination, stage, environment):
        variant = matrix.rows[config_path]["variant"]
        if variant == "V" and stage == "pretrain":
            running.set()
            assert halted.wait(5), "The other chain must report its engineering failure"
        if variant == "O" and stage == "bc":
            assert running.wait(5), "The other chain must already have a stage in flight"
        return check_health(python, config_path, destination, stage, environment)

    def write_plan(path, report):
        write(path, report)
        if path.name == "execution_plan.json" and any(
            row.get("status") == "dn_health_failed" for row in report["stages"]
        ):
            halted.set()

    monkeypatch.setattr(runner, "check_dn_health", check)
    monkeypatch.setattr(runner, "write_json", write_plan)
    with pytest.raises(SystemExit, match="engineering failure"):
        runner.main()
    rows = [row for row in matrix.plan() if row["variant"] == "V"]
    assert rows[0]["status"] == "verified_complete"
    assert "status" not in rows[1]


def test_terminal_bc_diagnostic_exception_halts_matrix(runner, matrix, monkeypatch):
    matrix.prepare(["O", "V"])
    check_health = runner.check_dn_health

    def check(python, config_path, destination, stage, environment):
        if stage == "bc":
            raise OSError("cannot launch diagnostic")
        return check_health(python, config_path, destination, stage, environment)

    monkeypatch.setattr(runner, "check_dn_health", check)
    with pytest.raises(SystemExit, match="DN health diagnostic error"):
        runner.main()
    rows = matrix.plan()
    assert rows[1]["dn_health_engineering_failures"] == ["diagnostic_error"]
    assert rows[1]["status"] == "dn_health_failed"
    assert all("status" not in row for row in rows if row["variant"] == "V")


@pytest.mark.parametrize("body", ["{", "null", "[]", '{"checks": {}, "passed": true}'])
def test_invalid_dn_metrics_are_engineering(runner, tmp_path, body):
    directory = tmp_path / "dn_health"
    directory.mkdir()
    (directory / "metrics.json").write_text(body)
    assert runner.classify_dn_health(tmp_path, 0) == (["invalid_metrics"], [])


def test_unexpected_dn_exit_is_engineering_even_with_valid_metrics(runner, tmp_path):
    directory = tmp_path / "dn_health"
    directory.mkdir()
    checks = dict.fromkeys(runner.DN_HEALTH_ENGINEERING_CHECKS, True)
    (directory / "metrics.json").write_text(json.dumps({"checks": checks, "passed": True}))
    assert runner.classify_dn_health(tmp_path, 1) == (["diagnostic_exit_code"], [])


@pytest.mark.parametrize("exit_code", [0, 1, 2, 3, -9])
def test_missing_metrics_are_engineering_even_with_success_exit(runner, matrix, exit_code):
    matrix.prepare(["O", "V"])
    matrix.missing_health.add(("O", "bc"))
    matrix.health_codes["O", "bc"] = exit_code
    with pytest.raises(SystemExit, match="engineering failure"):
        runner.main()
    row = next(r for r in matrix.plan() if r["variant"] == "O" and r["stage"] == "bc")
    assert "missing_metrics" in row["dn_health_engineering_failures"]
    assert all("status" not in r for r in matrix.plan() if r["variant"] == "V")


@pytest.mark.parametrize(
    "algorithm,stage",
    [("ppo", "bc"), ("es", "bc"), ("ppo", "ppo"), ("es", "es"), ("none", "pretrain")],
)
def test_nonterminal_bc_and_other_dn_gates_remain_strict(runner, matrix, algorithm, stage):
    matrix.prepare(["O", "V"], algorithm=algorithm)
    matrix.health_failures["O", stage] = ["omega_r2"]
    with pytest.raises(SystemExit, match="DN health gate failed"):
        runner.main()
    rows = matrix.plan()
    row = next(r for r in rows if r.get("status") == "dn_health_failed")
    assert "dn_health_quality_failures" not in row
    assert rows[-1]["status"] == "verified_complete"  # keep-going retains old behavior.


def test_dn_rerun_removes_stale_metrics(runner, tmp_path, monkeypatch):
    path = tmp_path / "dn_health/metrics.json"
    path.parent.mkdir()
    path.write_text('{"passed": true}')
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    assert runner.check_dn_health("python", "config.yaml", tmp_path, "bc", {}) == 1
    assert not path.exists()
    assert runner.classify_dn_health(tmp_path, 1)[0] == ["diagnostic_exit_code", "missing_metrics"]


def test_modality_rerun_does_not_reuse_stale_sensitivity(runner, tmp_path, monkeypatch):
    directory = tmp_path / "modality"
    directory.mkdir()
    (directory / "sensitivity.json").write_text(
        json.dumps(
            {
                "policies": {str(tmp_path / "actor"): sensitivity(0.1)},
            }
        )
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    metrics = runner.check_modality("python", "config.yaml", tmp_path, "VO", None, {})
    assert metrics["passed"] is False and "error" in metrics
    assert not (directory / "sensitivity.json").exists()
