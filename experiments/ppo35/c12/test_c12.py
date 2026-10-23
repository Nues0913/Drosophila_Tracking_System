"""C12 protocol, decision boundaries, recovery and scheduler safety; no GPU work."""

import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml
from c11 import jobs as c11_jobs
from c11.c11_common import PROFILES
from c11.summarize_c11 import curriculum_ranges, stage_for_step

from c12 import c12_jobs, driver, prepare_c12
from c12.c12_common import (
    FORMAL_BLOCKS,
    FORMAL_SEED,
    GPU_STAGES,
    MAIN_COMMIT,
    ROOT,
    RULES,
    SEEDS,
    SMOKE_SEED,
    SNAP_SEED,
    STAGES,
    UNTIL,
    digest,
    early_interpretation,
    overall,
    read,
    seeds,
    write,
)
from c12.prepare_c12 import candidate_config, config_diff, prepare, reference_check
from c12.summarize_c12 import collect, render, summarize


@pytest.fixture
def base():
    return yaml.safe_load((ROOT / "runs/c11/ppo_c11.yaml").read_text())


@pytest.mark.parametrize("seed", [0, 1])
def test_exact_config_whitelist(base, seed):
    original = copy.deepcopy(base)
    candidate = candidate_config(base, Path("runs/c12"), seed)
    assert base == original
    assert set(config_diff(base, candidate, seed)) == {"output", "train.curriculum"} | (
        {"seed"} if seed else set()
    )
    assert candidate["seed"] == seed
    assert all(set(row) == {"c11", "c12"} for row in config_diff(base, candidate, seed).values())
    assert [r["until"] for r in candidate["train"]["curriculum"]] == UNTIL
    for old, new in zip(
        base["train"]["curriculum"][2:], candidate["train"]["curriculum"], strict=True
    ):
        assert {k: v for k, v in old.items() if k != "until"} == {
            k: v for k, v in new.items() if k != "until"
        }


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "change",
    [
        "missing_output",
        "missing_curriculum",
        "extra",
        "removed",
        "total_steps",
        "readout_bias",
        "warmup",
    ],
)
def test_extra_missing_or_changed_settings_fail(base, seed, change):
    candidate = candidate_config(base, Path("runs/c12"), seed)
    if change.startswith("missing_"):
        if change == "missing_output":
            candidate["output"] = base["output"]
        else:
            candidate["train"]["curriculum"] = base["train"]["curriculum"]
    elif change == "extra":
        candidate["train"]["unknown"] = None
    elif change == "removed":
        del candidate["model"]["visual_bins"]
    else:
        key = {
            "total_steps": "total_steps",
            "readout_bias": "readout_bias_learning_rate",
            "warmup": "critic_warmup_updates",
        }[change]
        candidate["train"][key] = 99
    with pytest.raises(ValueError):
        config_diff(base, candidate, seed)


def test_s1_requires_seed_difference(base):
    candidate = candidate_config(base, Path("runs/c12"), 1)
    candidate["seed"] = 0
    with pytest.raises(ValueError):
        config_diff(base, candidate, 1)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("field", ["until", "target_motion", "plume_model", "scenario", "extra"])
def test_curriculum_every_key_and_until_are_fixed(base, index, field):
    candidate = candidate_config(base, Path("runs/c12"), 0)
    candidate["train"]["curriculum"][index][field] = 0.384615 if field == "until" else "changed"
    with pytest.raises(ValueError, match="curriculum"):
        config_diff(base, candidate, 0)


@pytest.mark.parametrize("size", [0, 3, 5])
def test_curriculum_exactly_four_stages(base, size):
    candidate = candidate_config(base, Path("runs/c12"), 0)
    candidate["train"]["curriculum"] = (candidate["train"]["curriculum"] * 2)[:size]
    with pytest.raises(ValueError):
        config_diff(base, candidate, 0)


@pytest.mark.parametrize(
    "passed,label",
    [
        ({"s0": True, "s1": True}, "C12 一致通過"),
        ({"s0": True, "s1": False}, "不一致"),
        ({"s0": False, "s1": True}, "不一致"),
        ({"s0": False, "s1": False}, "都不通過"),
    ],
)
def test_overall_decision(passed, label):
    assert overall(passed) == label


@pytest.mark.parametrize("passed", [{"s0": True}, {"s0": 1, "s1": False}, {}])
def test_overall_requires_two_boolean_decisions(passed):
    with pytest.raises(ValueError):
        overall(passed)


def observations(omega, tracking):
    return [{"omega_median": omega, "tracked_fraction": tracking} for _ in range(4)]


@pytest.mark.parametrize("omega", [0, 0.3, -0.3])
@pytest.mark.parametrize("tracking", [0.3, 1])
def test_early_all_four_including_inclusive_boundary(omega, tracking):
    assert early_interpretation(observations(omega, tracking)) == "前兩段是繞圈的原因"


@pytest.mark.parametrize("omega", [0.6, -0.6, 0.9])
@pytest.mark.parametrize("tracking", [0, 0.1])
def test_early_any_including_inclusive_boundary(omega, tracking):
    rows = observations(0.2, 0.5)
    rows[2] = {"omega_median": omega, "tracked_fraction": tracking}
    assert early_interpretation(rows) == "繞圈和 curriculum 無關"


@pytest.mark.parametrize(
    "omega,tracking",
    [(0.300001, 0.3), (0.3, 0.299999), (0.599999, 0.1), (0.6, 0.100001), (0.4, 0.2)],
)
def test_early_other_cases_are_inconclusive(omega, tracking):
    assert early_interpretation(observations(omega, tracking)) == "不下結論"


@pytest.mark.parametrize("size", [0, 1, 3, 5])
def test_early_missing_or_duplicate_observations_inconclusive(size):
    assert early_interpretation((observations(0.1, 0.9) * 2)[:size]) == "不下結論"


@pytest.mark.parametrize("first,count", FORMAL_BLOCKS)
@pytest.mark.parametrize("offset,episodes", [(0, 1), (-1, 2), (None, 1)])
def test_smoke_rejects_every_formal_block_overlap(first, count, offset, episodes):
    start = first + (count - 1 if offset is None else offset)
    with pytest.raises(ValueError, match="formal seed"):
        seeds(start, episodes, True)


@pytest.mark.parametrize("key", ["SEED_START", "SNAP_SEED_START"])
@pytest.mark.parametrize("first,count", FORMAL_BLOCKS)
def test_smoke_environment_override_fails_before_writes(monkeypatch, key, first, count):
    monkeypatch.setenv(key, str(first))
    with pytest.raises(ValueError, match="formal seed"):
        prepare(Path("runs/c12_smoke"), True)


def test_fixed_smoke_and_formal_ranges():
    assert seeds(SMOKE_SEED, 1, True) == [SMOKE_SEED]
    assert seeds(SMOKE_SEED + 1000, 1, True) == [SMOKE_SEED + 1000]
    assert len(seeds(FORMAL_SEED, 100, False)) == 100
    assert len(seeds(SNAP_SEED, 20, False)) == 20
    with pytest.raises(ValueError):
        seeds(FORMAL_SEED, 99, False)


@pytest.mark.parametrize("smoke", [True, False])
def test_out_guard_precedes_all_writes(tmp_path, smoke):
    out = tmp_path / "unused"
    with pytest.raises(ValueError, match="OUT differs"):
        prepare(out, smoke)
    assert not out.exists()


def reference_fixture(root):
    (root / "python").mkdir()
    source = root / "python/module.py"
    source.write_text("value = 1\n")
    folder = root / "runs/c12"
    folder.mkdir(parents=True)
    (folder / "main_python.sha256").write_text(f"{digest(source)}  python/module.py\n")
    (folder / "main_python.commit").write_text(MAIN_COMMIT + "\n")
    return source, folder


def test_main_hash_and_exact_commit(tmp_path):
    source, folder = reference_fixture(tmp_path)
    assert reference_check(tmp_path, False)["warnings"] == []
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="hashes"):
        reference_check(tmp_path, False)
    assert reference_check(tmp_path, True)["warnings"]


def test_main_inventory_includes_tests_excluded_by_c11(tmp_path):
    reference_fixture(tmp_path)
    folder = tmp_path / "python/tests"
    folder.mkdir()
    (folder / "test_ppo35_extra.py").write_text("extra\n")
    with pytest.raises(ValueError, match="Complete python/ inventory"):
        reference_check(tmp_path, False)
    assert reference_check(tmp_path, True)["warnings"]


@pytest.mark.parametrize("commit", [None, "4ef0acce0c16a00c59bef9d90a9f434de0803713", "wrong"])
def test_main_commit_missing_or_wrong_fails_formal_warns_smoke(tmp_path, commit):
    _, folder = reference_fixture(tmp_path)
    path = folder / "main_python.commit"
    if commit is None:
        path.unlink()
    else:
        path.write_text(commit)
    with pytest.raises(ValueError, match="commit"):
        reference_check(tmp_path, False)
    assert reference_check(tmp_path, True)["warnings"]


def test_gpu_budget_all_attempts_cpu_is_free(tmp_path):
    for stage, code, wall in [
        ("s0_train", 0, 3000),
        ("s1_train", "INTERRUPTED", 1000),
        ("s1_train", 0, 2000),
        ("s0_eval", 1, 100),
        ("s1_eval", 0, 200),
        ("bc_eval", 0, 300),
        ("s0_snap_eval", 0, 9000),
        ("s1_dn", 0, 1000),
    ]:
        driver.record(tmp_path, stage, code, wall)
    assert driver.remaining(tmp_path, {"smoke": False, "gpu_budget_s": 8100}) == 1500
    assert driver.remaining(tmp_path, {"smoke": True, "gpu_budget_s": 8100}) == 8100


def test_recover_interrupted_attempt_only_once(tmp_path):
    write(
        tmp_path / "active_s1_train.json",
        {"stage": "s1_train", "started_unix_s": time.time() - 100, "remaining_s": 12},
    )
    driver.recover_pending(tmp_path)
    driver.recover_pending(tmp_path)
    assert len(driver.ledger(tmp_path)) == 1
    assert driver.remaining(tmp_path, {"smoke": False, "gpu_budget_s": 8100}) == 8088


@pytest.mark.parametrize("stage", GPU_STAGES)
def test_exhausted_budget_never_launches_gpu(tmp_path, monkeypatch, stage):
    write(tmp_path / "manifest.json", {"smoke": False, "gpu_budget_s": 8100})
    driver.record(tmp_path, "s0_train", 0, 8097)
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **k: pytest.fail("GPU launched"))
    with pytest.raises(driver.BudgetExhausted):
        driver.run_stage(tmp_path, stage)
    assert not list(tmp_path.glob("active_*.json"))


def test_budget_not_run_marks_all_remaining_and_both_acceptances(tmp_path):
    write(tmp_path / "manifest.json", {"smoke": False})
    write(tmp_path / "s0_train.complete.json", {})
    driver.not_run(tmp_path, "test exhausted")
    result = read(tmp_path / "acceptance.json")
    assert result["status"] == result["overall"] == "NOT RUN"
    assert result["rules"] == RULES
    assert all(r["status"] == "NOT RUN" for r in result["candidates"].values())
    assert set(result["stages"]) == set(STAGES) - {"s0_train"}
    assert all(r["exit"] == "NOT RUN" for r in driver.ledger(tmp_path))
    assert (tmp_path / "summary.md").exists()


def test_complete_stage_skips_launch_after_hash_and_semantic_validation(tmp_path, monkeypatch):
    write(tmp_path / "manifest.json", {"smoke": True})
    write(tmp_path / "acceptance.json", {"accepted": False})
    write(
        tmp_path / "acceptance.complete.json",
        {str(tmp_path / "acceptance.json"): digest(tmp_path / "acceptance.json")},
    )
    validated = []
    monkeypatch.setattr(driver.subprocess, "run", lambda *a, **k: validated.append(a[0]))
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *a, **k: pytest.fail("Stage reran"))
    driver.run_stage(tmp_path, "acceptance")
    assert validated and "validate" in validated[0]
    assert not driver.ledger(tmp_path)


def test_modified_receipt_artifact_rejected(tmp_path):
    write(tmp_path / "manifest.json", {"smoke": True})
    write(tmp_path / "acceptance.json", {})
    write(tmp_path / "acceptance.complete.json", {str(tmp_path / "acceptance.json"): "changed"})
    with pytest.raises(ValueError, match="Completed stage changed"):
        driver.run_stage(tmp_path, "acceptance")


def test_stop_before_stage_has_no_attempt(tmp_path):
    write(tmp_path / "manifest.json", {"smoke": True})
    (tmp_path / "STOP").touch()
    with pytest.raises(driver.Stopped):
        driver.run_stage(tmp_path, "s0_train")
    assert not driver.ledger(tmp_path)


def test_gpu_timeout_has_no_receipt(tmp_path, monkeypatch):
    write(tmp_path / "manifest.json", {"smoke": False, "gpu_budget_s": 8100})
    commands = []

    class TimedOut:
        def __init__(self, command, **kwargs):
            commands.append(command)

        def poll(self):
            return 124

        def wait(self):
            return 124

    monkeypatch.setattr(driver.subprocess, "Popen", TimedOut)
    monkeypatch.setattr(driver, "stop_group", lambda p: None)
    with pytest.raises(driver.BudgetExhausted, match="timeout"):
        driver.run_stage(tmp_path, "s0_eval")
    assert commands[0][:3] == ["timeout", "--signal=TERM", "--kill-after=2s"]
    assert driver.ledger(tmp_path)[0]["exit"] == "124"
    assert not (tmp_path / "s0_eval.complete.json").exists()


def test_snapshots_require_both_training_receipts(tmp_path):
    write(tmp_path / "s0_train.complete.json", {})
    with pytest.raises(ValueError, match="Both training"):
        driver.snapshot_coordinator(tmp_path)


def test_scheduler_starts_snapshots_after_both_trainings_before_evaluations(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(prepare_c12, "prepare", lambda *a: None)
    monkeypatch.setattr(driver, "run_stage", lambda out, stage: events.append(stage))
    monkeypatch.setattr(driver, "stop_group", lambda p: None)

    class Chain:
        def __init__(self, *a, **kw):
            events.append("CPU snapshots started")
            assert events[:2] == ["s0_train", "s1_train"]

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(driver.subprocess, "Popen", Chain)
    driver.run(tmp_path, True)
    assert events == [
        "s0_train",
        "s1_train",
        "CPU snapshots started",
        "s0_eval",
        "s1_eval",
        "bc_eval",
        "s0_dn",
        "s1_dn",
        "acceptance",
        "summary",
    ]


def test_cpu_seed_resources_have_nice_ten_and_at_most_two_cpus(tmp_path, monkeypatch):
    monkeypatch.setattr(driver.os, "getpriority", lambda *a: 10)
    nice = []
    affinity = []
    stages = []
    monkeypatch.setattr(driver.os, "nice", nice.append)
    monkeypatch.setattr(driver.os, "sched_getaffinity", lambda *a: set(range(16)))
    monkeypatch.setattr(driver.os, "sched_setaffinity", lambda pid, cpus: affinity.append(cpus))
    monkeypatch.setattr(driver, "run_stage", lambda out, stage, **kw: stages.append(stage))
    for seed in SEEDS:
        driver.cpu_chain(tmp_path, seed)
    assert nice == [0, 0]
    assert affinity == [[0, 1], [2, 3]]
    assert stages == [
        f"{s}_{kind}" for s in SEEDS for kind in ("snap_build", "snap_eval", "snap_dn")
    ]


def test_early_rows_match_capture_boundaries_and_never_replace_missed_snapshots(tmp_path):
    for seed in SEEDS:
        folder = tmp_path / seed
        (folder / "snapshots").mkdir(parents=True)
        write(
            folder / "snapshots/index.json",
            {
                "snapshots": [
                    {"step": 51200, "boundary": 50000},
                    {"step": 100352, "boundary": 100000},
                ]
            },
        )
        write(
            folder / "snap_eval.json",
            {
                "rows": [
                    {"step": step, "omega_median": 0.3, "tracked_fraction": 0.3}
                    for step in (0, 51200, 100352)
                ]
            },
        )
    rows = c12_jobs.early_rows(tmp_path)
    assert len(rows) == 4
    assert early_interpretation(rows) == "前兩段是繞圈的原因"
    write(
        tmp_path / "s1/snapshots/index.json", {"snapshots": [{"step": 100352, "boundary": 100000}]}
    )
    assert early_interpretation(c12_jobs.early_rows(tmp_path)) == "不下結論"


@pytest.mark.parametrize(
    "step,stage",
    [
        (0, 1),
        (192500, 1),
        (192501, 2),
        (307500, 2),
        (307501, 3),
        (423000, 3),
        (423001, 4),
        (500000, 4),
    ],
)
def test_exact_new_curriculum_step_ranges(base, step, stage):
    stages = curriculum_ranges(candidate_config(base, Path("runs/c12"), 0)["train"])
    assert stage_for_step(step, stages) == stage
    assert [r["end_step"] for r in stages] == [192500, 307500, 423000, 500000]


def test_summary_reads_exact_prior_steps_and_each_own_curriculum(tmp_path, base):
    config = candidate_config(base, Path("runs/c12"), 0)
    write(tmp_path / "manifest.json", {"smoke": False, "configs": {"ppo_c12_s0.yaml": config}})
    for seed in SEEDS:
        folder = tmp_path / seed
        folder.mkdir()
        rows = [
            {
                "step": 51200,
                "tracked_fraction": 0.123,
                "omega_median": -0.2,
                "turn_bias": 0.05,
                "core_bias_max_abs": 0.1,
            }
        ]
        write(folder / "snap_eval.json", {"rows": rows})
    data = collect(tmp_path)
    assert data["c12"]["s0"]["training_records"][0]["tracked_fraction"] == 0.123
    assert data["c12"]["s0"]["same_step_c11"]["s0"][0]["step"] == 51200
    text = render(data)
    assert "C11 (s0) curriculum 階段" in text and "C11 (s1) curriculum 階段" in text
    assert "192,500" in text and "75,000" in text
    assert str(ROOT) not in text
    summarize(tmp_path)
    summarize(tmp_path, validate_only=True)


def test_shell_rejects_wrong_output_without_writes(tmp_path):
    out = tmp_path / "unused"
    result = subprocess.run(
        ["bash", "experiments/ppo35/c12/run_c12.sh"],
        env={**os.environ, "SMOKE": "1", "OUT": str(out)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and "OUT differs" in result.stdout
    assert not out.exists()


@pytest.mark.parametrize("job", ["evaluate", "snap_eval"])
@pytest.mark.parametrize("profile", ["s0", "s1", "c12"])
def test_shared_jobs_preserve_c11_default_guards_and_allow_c12_override(
    tmp_path, monkeypatch, job, profile
):
    import torch
    from flytrack_core.model import sparse

    # Stop at the guard: no environment, actor, or formal episode is executed.
    class GuardChecked(Exception):
        pass

    calls = []
    if profile == "c12":
        manifest = {
            "protocol": "C12-protocol100-v1",
            "smoke": True,
            "seed_start": SMOKE_SEED,
            "episodes": 1,
            "snap_seed_start": SMOKE_SEED + 1000,
            "snap_episodes": 1,
            "configs": {"eval.yaml": {}},
        }
        original_guard = seeds
        kwargs = {"seed_guard": lambda *args: checked_guard(*args)}
        monkeypatch.setattr(c11_jobs, "seeds", lambda *a: pytest.fail("C11 guard used for C12"))
    else:
        manifest = read(ROOT / PROFILES[profile].out() / "manifest.json")
        original_guard = c11_jobs.seeds
        kwargs = {}
        monkeypatch.setenv("C11_RUN", "s1" if profile == "s0" else "s0")

    def checked_guard(*args):
        original_guard(*args)
        calls.append(args)
        raise GuardChecked

    if profile != "c12":
        monkeypatch.setattr(c11_jobs, "seeds", checked_guard)
    monkeypatch.setattr(c11_jobs, "read", lambda path: manifest)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(sparse, "enable_deterministic", lambda: None)
    monkeypatch.setattr(sys, "argv", ["test"])
    with pytest.raises(GuardChecked):
        if job == "evaluate":
            c11_jobs.evaluate(tmp_path, "c11", **kwargs)
        else:
            c11_jobs.snap_eval(tmp_path, **kwargs)
    start_key, episodes_key = (
        ("seed_start", "episodes") if job == "evaluate" else ("snap_seed_start", "snap_episodes")
    )
    expected = (manifest[start_key], manifest[episodes_key], manifest["smoke"])
    assert calls == [expected if profile == "c12" else (*expected, profile)]


@pytest.mark.parametrize("explicit_items", [False, True])
def test_shared_snapshot_default_actor_selection_is_opt_in(tmp_path, monkeypatch, explicit_items):
    import torch
    from flytrack_core.model import sparse
    from flytrack_train import config

    manifest = {
        "smoke": True,
        "snap_seed_start": SMOKE_SEED + 1000,
        "snap_episodes": 1,
        "configs": {"eval.yaml": {}},
    }
    selected = []
    saved = []
    monkeypatch.setattr(c11_jobs, "read", lambda path: manifest)
    monkeypatch.setattr(c11_jobs, "snapshot_actors", lambda out: selected.append(out) or [])
    monkeypatch.setattr(c11_jobs, "write", lambda path, value: saved.append(value))
    monkeypatch.setattr(torch, "set_num_threads", lambda n: None)
    monkeypatch.setattr(sparse, "enable_deterministic", lambda: None)
    monkeypatch.setattr(config, "contract_from_config", lambda value: None)
    c11_jobs.snap_eval(tmp_path, seed_guard=seeds, **({"items": []} if explicit_items else {}))
    assert selected == ([] if explicit_items else [tmp_path])
    assert saved == [{"rows": []}]
