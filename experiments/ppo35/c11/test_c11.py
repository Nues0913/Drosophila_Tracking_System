"""C11 protocol isolation, snapshot races, recovery accounting and artifact bits."""

import copy
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c11_common import (  # noqa: E402
    FORMAL_SEED,
    HERE,
    ROOT,
    S1_FORMAL_SEED,
    S1_SNAP_SEED,
    SMOKE_SEED,
    SNAP_SEED,
    STAGES,
    digest,
    manifest_profile,
    portable,
    read,
    run_profile,
    seeds,
    write,
)
from check_c11 import ppo_complete  # noqa: E402
from driver import (  # noqa: E402
    BudgetExhausted,
    Stopped,
    ledger,
    not_run,
    record,
    recover_pending,
    remaining,
    run_stage,
    stage_env,
)
from prepare_c11 import (  # noqa: E402
    candidate_config,
    config_diff,
    main_hash_check,
    prepare,
    s0_config_diff,
    training_command,
)
from snapshots import Sidecar, bitwise_equal, boundaries, build_actor  # noqa: E402
from summarize_c11 import (  # noqa: E402
    collect,
    curriculum_ranges,
    render,
    stage_for_step,
    training_records,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:Sparse CSR tensor support is in beta state:UserWarning"
)


@pytest.fixture
def base():
    return yaml.safe_load((ROOT / "runs/c10/ppo_c10.yaml").read_text())


def test_config_whitelist(base):
    original = copy.deepcopy(base)
    config = candidate_config(base, Path("runs/c11"))
    assert base == original
    assert set(config_diff(base, config)) == {"output", "train.readout_bias_learning_rate"}
    assert config["train"]["readout_bias_learning_rate"] == 0
    assert config["init_artifact"] == portable(base["init_artifact"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("core_bias_learning_rate", 0),
        ("readout_learning_rate", 0),
        ("total_steps", 100),
        ("critic_warmup_updates", 0),
        ("readout_bias_learning_rate", 0.001),
        ("entropy_coef", 0.02),
    ],
)
def test_config_rejects_other_changes(base, key, value):
    config = candidate_config(base, Path("runs/c11"))
    config["train"][key] = value
    with pytest.raises(ValueError):
        config_diff(base, config)


def test_config_rejects_added_removed_keys(base):
    config = candidate_config(base, Path("runs/c11"))
    del config["model"]["visual_bins"]
    config["train"]["unknown"] = 1
    with pytest.raises(ValueError):
        config_diff(base, config)


@pytest.mark.parametrize(
    "previous,actual,capture,missed",
    [
        (0, 49_999, None, []),
        (0, 50_000, 50_000, []),
        (0, 100_001, 100_000, [50_000]),
        (50_000, 50_000, None, []),
        (50_000, 100_000, 100_000, []),
        (0, 150_000, 150_000, [50_000, 100_000]),
    ],
)
def test_boundaries(previous, actual, capture, missed):
    assert boundaries(previous, actual, 50_000) == {"capture": capture, "missed": missed}


@pytest.mark.parametrize("previous,actual,interval", [(10, 9, 5), (-1, 10, 5), (0, 10, 0)])
def test_invalid_boundaries(previous, actual, interval):
    with pytest.raises(ValueError):
        boundaries(previous, actual, interval)


def publish(out, step, progress=None):
    directory = out / "ppo_c11"
    directory.mkdir(exist_ok=True)
    temporary = directory / "next.pt"
    torch.save({"environment_steps": step, "actor": {"x": torch.tensor([step])}}, temporary)
    temporary.replace(directory / "training_checkpoint.pt")
    write(
        directory / "progress.json", {"environment_steps": step if progress is None else progress}
    )


def test_link_survives_atomic_replace_and_duplicate_poll(tmp_path):
    sidecar = Sidecar(tmp_path, 50_000)
    publish(tmp_path, 50_001)
    sidecar.poll()
    snapshot = tmp_path / "snapshots/step_50001.pt"
    source = tmp_path / "ppo_c11/training_checkpoint.pt"
    assert snapshot.stat().st_ino == source.stat().st_ino
    old = snapshot.read_bytes()
    sidecar.poll()
    assert len(sidecar.state["snapshots"]) == 1
    publish(tmp_path, 100_001)
    assert snapshot.stat().st_ino != source.stat().st_ino
    assert snapshot.read_bytes() == old
    assert torch.load(snapshot, weights_only=True)["environment_steps"] == 50_001


def test_checkpoint_ahead_of_progress_and_resume(tmp_path):
    sidecar = Sidecar(tmp_path, 50_000)
    publish(tmp_path, 100_352, progress=50_176)
    sidecar.poll()
    assert sidecar.state["last_step"] == 100_352
    assert sidecar.state["missed_boundaries"] == [50_000]
    assert sidecar.state["snapshots"][0]["step"] == 100_352
    resumed = Sidecar(tmp_path, 50_000)
    resumed.poll()
    assert resumed.state == sidecar.state


def test_final_snapshot_between_polls(tmp_path):
    sidecar = Sidecar(tmp_path, 50_000)
    publish(tmp_path, 12)
    sidecar.poll()
    assert not sidecar.state["snapshots"]
    sidecar.poll(final=True)
    assert sidecar.state["snapshots"][0]["step"] == 12


def test_sidecar_error_is_diagnostic(tmp_path, monkeypatch):
    sidecar = Sidecar(tmp_path, 4)
    publish(tmp_path, 4)
    original = (tmp_path / "ppo_c11/training_checkpoint.pt").read_bytes()

    def failing_link(*args):
        raise OSError("test failure")

    monkeypatch.setattr("snapshots.os.link", failing_link)
    sidecar.safe_poll()
    assert sidecar.state["errors"] == [{"type": "OSError", "final": False}]
    assert (tmp_path / "ppo_c11/training_checkpoint.pt").read_bytes() == original


def test_rebuild_matches_existing_trained_actor_bits(tmp_path, base):
    # Read C10's completed training artifacts only; do not run training/evaluation seeds.
    config = candidate_config(base, Path("runs/c11"))
    checkpoint = ROOT / "runs/c10/ppo_c10/training_checkpoint.pt"
    actor = tmp_path / "actor"
    assert build_actor(checkpoint, actor, config) == 500_000
    rebuilt = torch.load(actor / "weights.pt", map_location="cpu", weights_only=True)
    trained = torch.load(
        ROOT / "runs/c10/ppo_c10/actor/weights.pt", map_location="cpu", weights_only=True
    )
    assert bitwise_equal(rebuilt, trained)


def test_bitwise_comparison_detects_signed_zero():
    assert not bitwise_equal({"x": torch.tensor([0.0])}, {"x": torch.tensor([-0.0])})
    assert bitwise_equal({"x": torch.tensor([0.0])}, {"x": torch.tensor([0.0])})


@pytest.mark.parametrize(
    "start,count",
    [
        (FORMAL_SEED, 1),
        (FORMAL_SEED + 99, 1),
        (FORMAL_SEED - 1, 2),
        (SNAP_SEED, 1),
        (SNAP_SEED + 19, 1),
        (SNAP_SEED - 1, 2),
    ],
)
def test_smoke_rejects_formal_seed_overlap(start, count):
    # Guard validation only: these seeds are never passed to an environment.
    with pytest.raises(ValueError, match="formal seed"):
        seeds(start, count, True)


def test_smoke_seeds_are_separate():
    assert seeds(SMOKE_SEED, 1, True) == [SMOKE_SEED]
    assert seeds(SMOKE_SEED + 1000, 1, True) == [SMOKE_SEED + 1000]


def reference_fixture(root):
    folder = root / "python/tests"
    folder.mkdir(parents=True)
    source = root / "python/module.py"
    source.write_text("value = 1\n")
    (folder / "test_ppo35_custom.py").write_text("excluded\n")
    reference = root / "main_python.sha256"
    reference.write_text(f"{digest(source)}  python/module.py\n")
    return source, reference


def test_main_hash_inventory_and_exclusion(tmp_path):
    source, reference = reference_fixture(tmp_path)
    result = main_hash_check(tmp_path, reference, False)
    assert result == {"sha256": digest(reference), "warnings": []}
    source.write_text("changed\n")
    with pytest.raises(ValueError):
        main_hash_check(tmp_path, reference, False)
    assert main_hash_check(tmp_path, reference, True)["warnings"]


def test_main_hash_detects_unlisted_files(tmp_path):
    _, reference = reference_fixture(tmp_path)
    (tmp_path / "python/extra.txt").write_text("extra")
    with pytest.raises(ValueError):
        main_hash_check(tmp_path, reference, False)


@pytest.mark.parametrize("entry", ["../escape", "/absolute", "python/../escape", "other/x"])
def test_main_hash_rejects_unsafe_path(tmp_path, entry):
    _, reference = reference_fixture(tmp_path)
    reference.write_text(f"{'0' * 64}  {entry}\n")
    with pytest.raises(ValueError):
        main_hash_check(tmp_path, reference, False)


def test_missing_reference_warns_only_in_smoke(tmp_path):
    reference = tmp_path / "missing"
    assert main_hash_check(tmp_path, reference, True)["sha256"] is None
    with pytest.raises(ValueError):
        main_hash_check(tmp_path, reference, False)


def test_gpu_budget_counts_all_attempts_only_gpu_stages(tmp_path):
    for stage, code, seconds in (
        ("c11_train", 0, 3000),
        ("c11_eval", 1, 300),
        ("c11_eval", 0, 200),
        ("bc_eval", "INTERRUPTED", 400),
        ("snap_eval", 0, 10000),
    ):
        record(tmp_path, stage, code, seconds)
    assert remaining(tmp_path, {"gpu_budget_s": 4500}) == 600


def test_interruption_debits_remaining_budget(tmp_path):
    write(
        tmp_path / "active_c11_eval.json",
        {
            "stage": "c11_eval",
            "started_unix_s": time.time() - 100,
            "remaining_s": 12,
        },
    )
    recover_pending(tmp_path)
    assert remaining(tmp_path, {"gpu_budget_s": 4500}) == 4488
    assert ledger(tmp_path)[0]["exit"] == "INTERRUPTED"
    recover_pending(tmp_path)
    assert len(ledger(tmp_path)) == 1


def test_budget_exhaustion_does_not_launch_stage(tmp_path):
    write(tmp_path / "manifest.json", {"smoke": False, "gpu_budget_s": 4500})
    record(tmp_path, "c11_train", 0, 4497)
    with pytest.raises(BudgetExhausted):
        run_stage(tmp_path, "c11_eval")
    assert not (tmp_path / "active_c11_eval.json").exists()


def test_stop_is_clean_before_launch(tmp_path):
    write(tmp_path / "manifest.json", {"smoke": True})
    (tmp_path / "STOP").touch()
    with pytest.raises(Stopped):
        run_stage(tmp_path, "c11_train")
    assert not (tmp_path / "stages.tsv").exists()


def test_not_run_includes_acceptance_and_pending_stages(tmp_path):
    write(tmp_path / "manifest.json", {"smoke": True, "snap_scheduling": "CPU"})
    write(tmp_path / "c11_train.complete.json", {})
    not_run(tmp_path, "test budget")
    assert read(tmp_path / "acceptance.json")["status"] == "NOT RUN"
    assert read(tmp_path / "acceptance.json")["accepted"] is False
    assert {r["stage"] for r in ledger(tmp_path)} == set(STAGES) - {"c11_train"}
    assert (tmp_path / "summary.md").exists()


def test_completed_receipt_rejects_changed_output(tmp_path):
    write(tmp_path / "manifest.json", {"smoke": True})
    write(tmp_path / "acceptance.json", {"accepted": False})
    write(tmp_path / "acceptance.complete.json", {str(tmp_path / "acceptance.json"): "changed"})
    with pytest.raises(ValueError, match="Completed stage changed"):
        run_stage(tmp_path, "acceptance")


def test_training_and_cpu_stage_environments(monkeypatch):
    monkeypatch.setenv("DETERMINISTIC", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert "DETERMINISTIC" not in stage_env("c11_train", False)
    for stage in ("snap_build", "snap_eval", "snap_dn", "dn_c11"):
        assert stage_env(stage, False)["CUDA_VISIBLE_DEVICES"] == ""
    assert stage_env("c11_eval", True)["CUDA_VISIBLE_DEVICES"] == ""


def test_timeout_is_charged_and_never_validated_as_complete(tmp_path, monkeypatch):
    write(tmp_path / "manifest.json", {"smoke": False, "gpu_budget_s": 4500})
    commands = []

    class TimedOut:
        def __init__(self, command, **kwargs):
            commands.append(command)

        def poll(self):
            return 124

        def wait(self):
            return 124

    monkeypatch.setattr("driver.subprocess.Popen", TimedOut)
    monkeypatch.setattr("driver.stop_group", lambda p: None)
    with pytest.raises(BudgetExhausted, match="timeout"):
        run_stage(tmp_path, "c11_eval")
    assert commands[0][:3] == ["timeout", "--signal=TERM", "--kill-after=2s"]
    assert ledger(tmp_path)[0]["exit"] == "124"
    assert remaining(tmp_path, {"gpu_budget_s": 4500}) < 4500
    assert not (tmp_path / "c11_eval.complete.json").exists()


def test_stop_during_stage_cleans_process_and_preserves_accounting(tmp_path, monkeypatch):
    write(tmp_path / "manifest.json", {"smoke": True})
    stopped = []

    class Running:
        def __init__(self, *args, **kwargs):
            (tmp_path / "STOP").touch()

        def poll(self):
            return None

    monkeypatch.setattr("driver.subprocess.Popen", Running)
    monkeypatch.setattr("driver.stop_group", lambda p: stopped.append(p))
    with pytest.raises(Stopped):
        run_stage(tmp_path, "c11_eval")
    assert isinstance(stopped[0], Running)
    assert ledger(tmp_path)[0]["exit"] == "INTERRUPTED"
    assert not (tmp_path / "active_c11_eval.json").exists()


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["python", "-m", "flytrack_train", "ppo"], True),
        (["python", "run_c10.py", "train", "out"], True),
        (["python", "experiments/ppo35/c11/jobs.py", "out", "c11_train"], True),
        (["python", "experiments/ppo35/c11/jobs.py", "out", "c11_eval"], False),
        (["python", "driver.py", "run", "out"], False),
    ],
)
def test_preflight_recognizes_training_commands(argv, expected):
    assert training_command(argv) == expected


def test_sidecar_startup_cannot_spend_more_than_remaining_budget(tmp_path, monkeypatch):
    write(tmp_path / "manifest.json", {"smoke": False, "gpu_budget_s": 4500})
    record(tmp_path, "c11_train", "INTERRUPTED", 4494)
    clock = [0.0]
    commands = []

    class SlowSidecar:
        def __init__(self, command, **kwargs):
            commands.append(command)

        def poll(self):
            return None

    def advance(seconds):
        clock[0] += seconds

    monkeypatch.setattr("driver.subprocess.Popen", SlowSidecar)
    monkeypatch.setattr("driver.stop_group", lambda p: None)
    monkeypatch.setattr("driver.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("driver.time.sleep", advance)
    with pytest.raises(BudgetExhausted):
        run_stage(tmp_path, "c11_train")
    assert len(commands) == 1 and "sidecar" in commands[0]
    assert 0 < remaining(tmp_path, {"gpu_budget_s": 4500}) < 6


@pytest.mark.parametrize(
    "name,output,seed,formal,snapshot,protocol",
    [
        ("s0", "runs/c11", 0, FORMAL_SEED, SNAP_SEED, "C11-protocol100-v1"),
        ("s1", "runs/c11_s1", 1, S1_FORMAL_SEED, S1_SNAP_SEED, "C11-s1-protocol100-v1"),
    ],
)
def test_run_profiles(monkeypatch, base, name, output, seed, formal, snapshot, protocol):
    monkeypatch.setenv("C11_RUN", name)
    profile = run_profile()
    assert profile.name == name
    assert profile.out() == Path(output)
    assert profile.out(True) == Path(output + "_smoke")
    assert (profile.seed, profile.formal_seed, profile.snap_seed) == (seed, formal, snapshot)
    assert profile.protocol == protocol
    assert manifest_profile({"protocol": protocol}) == profile
    assert candidate_config(base, profile.out())["seed"] == seed


def test_default_profile_and_s0_config_unchanged(monkeypatch, base):
    monkeypatch.delenv("C11_RUN", raising=False)
    assert run_profile().name == "s0"
    expected = yaml.safe_load((ROOT / "runs/c11/ppo_c11.yaml").read_text())
    assert candidate_config(base, Path("runs/c11")) == expected


@pytest.mark.parametrize("name", ["", "s2", "S1", "../s1"])
def test_invalid_profile_fails_in_python_and_shell(monkeypatch, tmp_path, name):
    monkeypatch.setenv("C11_RUN", name)
    with pytest.raises(ValueError, match="C11_RUN"):
        run_profile()
    out = tmp_path / "unused"
    result = subprocess.run(
        ["bash", str(HERE / "run_c11.sh")],
        env={**os.environ, "SMOKE": "1", "OUT": str(out)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "C11_RUN must be s0 or s1" in result.stdout
    assert not out.exists()


@pytest.mark.parametrize("name", ["s0", "s1"])
@pytest.mark.parametrize("smoke", [False, True])
def test_out_guard_precedes_writes(monkeypatch, tmp_path, name, smoke):
    monkeypatch.setenv("C11_RUN", name)
    out = tmp_path / "unused"
    with pytest.raises(ValueError, match="OUT differs"):
        prepare(out, smoke)
    result = subprocess.run(
        ["bash", str(HERE / "run_c11.sh")],
        env={**os.environ, "SMOKE": str(int(smoke)), "OUT": str(out)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "OUT differs" in result.stdout
    assert not out.exists()


def test_s1_config_whitelists(base):
    original = copy.deepcopy(base)
    candidate = candidate_config(base, Path("runs/c11_s1"), "s1")
    assert base == original
    assert set(config_diff(base, candidate, "s1")) == {
        "output",
        "seed",
        "train.readout_bias_learning_rate",
    }
    s0 = yaml.safe_load((ROOT / "runs/c11/ppo_c11.yaml").read_text())
    assert set(s0_config_diff(s0, candidate)) == {"output", "seed"}
    assert candidate["seed"] == 1
    assert candidate["init_artifact"] == portable(base["init_artifact"])


@pytest.mark.parametrize(
    "change",
    ["extra", "missing_output", "missing_seed", "missing_bias", "changed_bias", "removed_key"],
)
def test_s1_c10_whitelist_rejects_extra_or_missing_changes(base, change):
    candidate = candidate_config(base, Path("runs/c11_s1"), "s1")
    if change == "extra":
        candidate["train"]["entropy_coef"] = 0.02
    elif change == "missing_output":
        candidate["output"] = portable(base["output"])
    elif change == "missing_seed":
        candidate["seed"] = 0
    elif change == "missing_bias":
        del candidate["train"]["readout_bias_learning_rate"]
    elif change == "changed_bias":
        candidate["train"]["readout_bias_learning_rate"] = 0.001
    else:
        del candidate["model"]["visual_bins"]
    with pytest.raises(ValueError):
        config_diff(base, candidate, "s1")


@pytest.mark.parametrize("change", ["extra", "missing_output", "missing_seed"])
def test_s1_s0_whitelist_rejects_extra_or_missing_changes(base, change):
    s0 = yaml.safe_load((ROOT / "runs/c11/ppo_c11.yaml").read_text())
    candidate = candidate_config(base, Path("runs/c11_s1"), "s1")
    if change == "extra":
        candidate["train"]["total_steps"] = 100
    elif change == "missing_output":
        candidate["output"] = s0["output"]
    else:
        candidate["seed"] = s0["seed"]
    with pytest.raises(ValueError, match="differences from C11-s0"):
        s0_config_diff(s0, candidate)


@pytest.mark.parametrize("run", ["s0", "s1"])
@pytest.mark.parametrize("first,count", [(S1_FORMAL_SEED, 100), (S1_SNAP_SEED, 20)])
@pytest.mark.parametrize("offset,episodes", [(0, 1), (-1, 2), (None, 1)])
def test_smoke_rejects_s1_blocks(run, first, count, offset, episodes):
    start = first + (count - 1 if offset is None else offset)
    with pytest.raises(ValueError, match="formal seed"):
        seeds(start, episodes, True, run)


@pytest.mark.parametrize("first,count", [(FORMAL_SEED, 100), (SNAP_SEED, 20)])
@pytest.mark.parametrize("offset,episodes", [(0, 1), (-1, 2), (None, 1)])
def test_s1_smoke_rejects_s0_blocks(first, count, offset, episodes):
    start = first + (count - 1 if offset is None else offset)
    with pytest.raises(ValueError, match="formal seed"):
        seeds(start, episodes, True, "s1")


@pytest.mark.parametrize(
    "start,count",
    [
        (FORMAL_SEED, 100),
        (SNAP_SEED, 20),
        (SMOKE_SEED, 1),
        (S1_FORMAL_SEED - 1, 2),
        (S1_FORMAL_SEED + 99, 2),
        (S1_SNAP_SEED - 1, 2),
        (S1_SNAP_SEED + 19, 2),
    ],
)
def test_s1_formal_rejects_other_seed_blocks(start, count):
    with pytest.raises(ValueError, match="own formal seed"):
        seeds(start, count, False, "s1")


@pytest.mark.parametrize("first,count", [(S1_FORMAL_SEED, 100), (S1_SNAP_SEED, 20)])
def test_s1_formal_seed_ranges(first, count):
    assert seeds(first, count, False, "s1") == list(range(first, first + count))
    assert seeds(first + count - 1, 1, False, "s1") == [first + count - 1]
    assert seeds(first + count, 1, True, "s1") == [first + count]


@pytest.mark.parametrize(
    "step,stage",
    [
        (0, 1),
        (75_000, 1),
        (75_001, 2),
        (175_000, 2),
        (175_001, 3),
        (300_000, 3),
        (300_001, 4),
        (375_000, 4),
        (375_001, 5),
        (450_000, 5),
        (450_001, 6),
        (500_000, 6),
    ],
)
def test_curriculum_boundaries_match_training(base, step, stage):
    from flytrack_train.ppo import curriculum_stage

    stages = curriculum_ranges(base["train"])
    assert stage_for_step(step, stages) == stage
    expected = curriculum_stage(base["train"], step / base["train"]["total_steps"])
    assert {key: stages[stage - 1][key] for key in expected} == expected
    assert stages[-1]["end_step"] == 500_000
    assert stages[0]["start_inclusive"] is True
    assert all(not row["start_inclusive"] for row in stages[1:])


def test_snapshot_dn_slope_and_curriculum_without_index(tmp_path, base):
    (tmp_path / "snapshots").mkdir()
    row = {
        "step": 75_000,
        "tracked_fraction": 0.5,
        "omega_median": 0.1,
        "turn_bias": 0.2,
        "core_bias_max_abs": 0.3,
    }
    write(tmp_path / "snap_eval.json", {"rows": [row]})
    write(
        tmp_path / "snapshots/dn_step_75000.json",
        {"dn": {"DNa01": {"slope_median": 0.012}, "DNa02": {"voltage_median": 3.5}}},
    )
    assert training_records(tmp_path, curriculum_ranges(base["train"])) == [
        {**row, "DNa01_slope_median": 0.012, "DNa02_voltage_median": 3.5, "curriculum_stage": 1}
    ]


def test_curriculum_ranges_for_smoke_budget(base):
    stages = curriculum_ranges({**base["train"], "total_steps": 36})
    assert [stage["end_step"] for stage in stages] == [5.4, 12.6, 21.6, 27, 32.4, 36]
    assert [stage_for_step(step, stages) for step in (0, 12, 24, 36)] == [1, 2, 4, 6]


def test_s1_summary_compares_exact_s0_steps_and_endpoints(tmp_path, base):
    write(
        tmp_path / "manifest.json",
        {
            "protocol": run_profile("s1").protocol,
            "smoke": True,
            "configs": {"ppo_c11.yaml": candidate_config(base, Path("runs/c11_s1"), "s1")},
            "snap_scheduling": "CPU",
        },
    )
    s0_rows = read(ROOT / "runs/c11/snap_eval.json")["rows"]
    row = {**s0_rows[1], "tracked_fraction": 0.123}
    write(tmp_path / "snap_eval.json", {"rows": [row]})
    data = collect(tmp_path)
    reference = next(r for r in data["s0_training_records"] if r["step"] == row["step"])
    assert reference["tracked_fraction"] == s0_rows[1]["tracked_fraction"]
    assert data["training_records"][0]["tracked_fraction"] == 0.123
    dn = read(ROOT / "runs/c11/snapshots" / f"dn_step_{row['step']}.json")
    assert reference["DNa01_slope_median"] == dn["dn"]["DNa01"]["slope_median"]
    assert data["s0_acceptance"] == read(ROOT / "runs/c11/acceptance.json")
    assert data["s0_endpoint"]["normal_success"] is not None
    text = render(data)
    assert "| 條件 | C11-s1 | C11 (s0) | C10 |" in text
    assert "C11-s1 DNa01 slope 中位數" in text
    assert "C11 (s0) DNa01 slope 中位數" in text
    assert "1：stationary／gaussian／normal（[0–75,000]）" in text
    assert "6：curved、wander／puff／conflict（(450,000–500,000]）" in text
    assert "| normal 終點 | C11-s1 | C11 (s0) | C10 |" in text
    assert str(ROOT) not in text


@pytest.mark.parametrize("seed", [0, 1])
def test_completion_checks_configured_recovery_seed(tmp_path, base, seed):
    from flytrack_train.recovery import Recovery, atomic_torch_save

    # Rebind a completed CPU fixture without training or running evaluation seeds.
    config = candidate_config(base, Path("runs/c11_s1"), "s1")
    config["seed"] = seed
    source = ROOT / "runs/c10/ppo_c10"
    checkpoint = torch.load(
        source / "training_checkpoint.pt", map_location="cpu", weights_only=True
    )
    identity = Recovery(tmp_path, config, "ppo", seed, "cpu").identity
    atomic_torch_save({**checkpoint, "identity": identity}, tmp_path / "training_checkpoint.pt")
    for name in ("progress.json", "training_metrics.json"):
        shutil.copyfile(source / name, tmp_path / name)
    shutil.copytree(source / "actor", tmp_path / "actor")
    ppo_complete(tmp_path, config, True)
    with pytest.raises(ValueError, match="seed, configuration or source hashes differ"):
        ppo_complete(tmp_path, {**config, "seed": 1 - seed}, True)
