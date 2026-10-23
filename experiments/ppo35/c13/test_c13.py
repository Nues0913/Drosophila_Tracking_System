"""C13 configuration, CPU measurements, decision boundaries and scheduling."""

import copy
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from c12 import driver as shared

from c13 import anchor_distance, c13_jobs, driver, prepare_c13, summarize_c13
from c13.c13_common import (
    BOUNDARIES,
    FORMAL_BLOCKS,
    FORMAL_SEED,
    GPU_STAGES,
    LAMBDA,
    ROOT,
    RULES,
    SEEDS,
    SMOKE_SEED,
    SNAP_SEED,
    STAGES,
    digest,
    overall,
    read,
    seeds,
    snapshot_interpretation,
    write,
)
from c13.prepare_c13 import candidate_config, config_diff, prepare, reference_check


@pytest.fixture
def bases():
    return {s: yaml.safe_load((ROOT / f"runs/c12/ppo_c12_{s}.yaml").read_text()) for s in SEEDS}


@pytest.mark.parametrize("seed", [0, 1])
def test_only_output_and_new_anchor_key_change(bases, seed):
    base = bases[f"s{seed}"]
    before = copy.deepcopy(base)
    candidate = candidate_config(base, Path("runs/c13"), seed)
    changes = config_diff(base, candidate, seed)
    assert set(changes) == {"output", "train.bc_anchor_coef"}
    assert changes["train.bc_anchor_coef"] == {"c12": None, "c13": LAMBDA}
    assert base == before


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "mutation", ["extra", "removed", "output", "anchor", "lambda", "warmup", "seed", "curriculum"]
)
def test_config_diff_rejects_extra_missing_and_incorrect_lambda(bases, seed, mutation):
    base = bases[f"s{seed}"]
    candidate = candidate_config(base, Path("runs/c13"), seed)
    if mutation == "extra":
        candidate["train"]["unknown"] = None
    elif mutation == "removed":
        del candidate["model"]["visual_bins"]
    elif mutation == "output":
        candidate["output"] = base["output"]
    elif mutation == "anchor":
        del candidate["train"]["bc_anchor_coef"]
    elif mutation == "lambda":
        candidate["train"]["bc_anchor_coef"] = 0.1724
    elif mutation == "warmup":
        candidate["train"]["critic_warmup_updates"] = 11
    elif mutation == "seed":
        candidate["seed"] = 1 - seed
    else:
        candidate["train"]["curriculum"][0]["until"] = 0.38
    with pytest.raises(ValueError, match="config differences"):
        config_diff(base, candidate, seed)


@pytest.mark.parametrize("smoke", [True, False])
@pytest.mark.parametrize(
    "has_helper,source", [(False, "bc_anchor_coef"), (True, "old source"), (False, "")]
)
def test_missing_anchor_support_stops_both_modes(monkeypatch, smoke, has_helper, source):
    module = SimpleNamespace(**({"_bc_anchor_loss": lambda: None} if has_helper else {}))
    monkeypatch.setattr(prepare_c13.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(prepare_c13.inspect, "getsource", lambda module: source)
    monkeypatch.setattr(
        prepare_c13, "reference_check", lambda *a: pytest.fail("Reference check before support")
    )
    with pytest.raises(ValueError, match="main 尚未合併 bc_anchor_coef"):
        prepare(Path("runs/c13_smoke" if smoke else "runs/c13"), smoke)


def test_support_requires_both_helper_and_source(monkeypatch):
    monkeypatch.setattr(
        prepare_c13.importlib,
        "import_module",
        lambda name: SimpleNamespace(_bc_anchor_loss=lambda: None),
    )
    monkeypatch.setattr(prepare_c13.inspect, "getsource", lambda module: "bc_anchor_coef")
    prepare_c13.anchor_support()


def reference_fixture(root, commit="1" * 40):
    (root / "python").mkdir()
    source = root / "python/module.py"
    source.write_text("value = 1\n")
    folder = root / "runs/c13"
    folder.mkdir(parents=True)
    (folder / "main_python.sha256").write_text(f"{digest(source)}  python/module.py\n")
    (folder / "main_python.commit").write_text(commit + "\n")
    return source, folder


@pytest.mark.parametrize("commit", ["1" * 40, "a" * 40])
def test_commit_is_read_from_reference_without_pinned_constant(tmp_path, commit):
    reference_fixture(tmp_path, commit)
    result = reference_check(tmp_path, False)
    assert result["commit"] == commit and not result["warnings"]


@pytest.mark.parametrize(
    "change",
    ["hash", "new_source", "ppo35_test", "missing_commit", "invalid_commit", "missing_hash"],
)
def test_reference_mismatch_stops_formal_warns_smoke(tmp_path, change):
    source, folder = reference_fixture(tmp_path)
    if change == "hash":
        source.write_text("changed\n")
    elif change in ("new_source", "ppo35_test"):
        target = (
            tmp_path / "python/tests/test_ppo35_new.py"
            if change == "ppo35_test"
            else tmp_path / "python/new.py"
        )
        target.parent.mkdir(exist_ok=True)
        target.write_text("new\n")
    elif change == "missing_commit":
        (folder / "main_python.commit").unlink()
    elif change == "invalid_commit":
        (folder / "main_python.commit").write_text("invalid")
    else:
        (folder / "main_python.sha256").unlink()
    with pytest.raises(ValueError):
        reference_check(tmp_path, False)
    assert reference_check(tmp_path, True)["warnings"]


@pytest.mark.parametrize(
    "passed,label",
    [
        ({"s0": True, "s1": True}, "C13 一致通過"),
        ({"s0": True, "s1": False}, "不一致"),
        ({"s0": False, "s1": True}, "不一致"),
        ({"s0": False, "s1": False}, "都不通過"),
    ],
)
def test_overall_decisions(passed, label):
    assert overall(passed) == label


@pytest.mark.parametrize("passed", [{}, {"s0": True}, {"s0": True, "s1": 1}])
def test_overall_requires_boolean_pair(passed):
    with pytest.raises(ValueError):
        overall(passed)


EARLY = RULES["snapshots"]["early_dn"]["label"]
CIRCLE = RULES["snapshots"]["circling"]["label"]


def observations(slope=0.01, omega=0.5, tracking=0.2):
    return [
        {
            "seed": seed,
            "boundary": boundary,
            "step": 51200 if boundary == 50000 else boundary,
            "DNa01_slope_median": slope,
            "omega_median": omega,
            "tracked_fraction": tracking,
        }
        for seed in SEEDS
        for boundary in BOUNDARIES
    ]


@pytest.mark.parametrize(
    "slope,expected", [(0.01, True), (0.010001, True), (0.009999, False), (-0.01, False)]
)
def test_early_dn_inclusive_boundary(slope, expected):
    rows = observations()
    rows[10]["DNa01_slope_median"] = slope
    assert snapshot_interpretation(rows)[EARLY] is expected


@pytest.mark.parametrize(
    "omega,tracking,expected",
    [
        (0.6, 0.1, True),
        (-0.6, 0.1, True),
        (0.7, 0.0, True),
        (0.599999, 0.1, False),
        (0.6, 0.100001, False),
    ],
)
@pytest.mark.parametrize("position", [0, 9, 19])
def test_circling_any_seed_and_boundary_inclusive(omega, tracking, expected, position):
    rows = observations()
    rows[position].update(omega_median=omega, tracked_fraction=tracking)
    assert snapshot_interpretation(rows)[CIRCLE] is expected


def test_snapshot_descriptions_are_independent_and_can_both_be_true():
    assert snapshot_interpretation(observations(0.01, -0.6, 0.1)) == {EARLY: True, CIRCLE: True}


@pytest.mark.parametrize(
    "mode", ["early_missing", "wrong_step", "wrong_boundary", "missing_dn", "duplicate_seed"]
)
def test_early_missing_snapshot_is_inconclusive(mode):
    rows = observations()
    if mode == "early_missing":
        rows.pop(10)
    elif mode == "wrong_step":
        rows[10]["step"] = 53248
    elif mode == "wrong_boundary":
        rows[10]["boundary"] = 100_000
    elif mode == "missing_dn":
        rows[10]["DNa01_slope_median"] = None
    else:
        rows[10]["seed"] = "s0"
    assert snapshot_interpretation(rows)[EARLY] == "不下結論"


@pytest.mark.parametrize("hit", [True, False])
@pytest.mark.parametrize("missing", ["row", "omega", "tracking", "duplicate"])
def test_incomplete_circling_can_establish_only_true(hit, missing):
    rows = observations()
    if missing == "row":
        rows.pop()
    elif missing == "duplicate":
        rows[-1] = rows[1].copy()
    else:
        rows[-1]["omega_median" if missing == "omega" else "tracked_fraction"] = None
    if hit:
        rows[0].update(omega_median=0.6, tracked_fraction=0.1)
    result = snapshot_interpretation(rows)[CIRCLE]
    assert result is True if hit else result == "不下結論"


def test_no_snapshots_is_inconclusive():
    assert snapshot_interpretation([]) == {EARLY: "不下結論", CIRCLE: "不下結論"}


@pytest.mark.parametrize("first,count", FORMAL_BLOCKS)
@pytest.mark.parametrize("offset,episodes", [(0, 1), (-1, 2), (None, 1)])
def test_smoke_rejects_overlap_with_every_formal_block(first, count, offset, episodes):
    start = first + (count - 1 if offset is None else offset)
    with pytest.raises(ValueError, match="formal seed"):
        seeds(start, episodes, True)


@pytest.mark.parametrize("key", ["SEED_START", "SNAP_SEED_START"])
@pytest.mark.parametrize("first,count", FORMAL_BLOCKS)
def test_smoke_formal_environment_override_precedes_preflight(monkeypatch, key, first, count):
    monkeypatch.setenv(key, str(first))
    monkeypatch.setattr(prepare_c13, "anchor_support", lambda: pytest.fail("Preflight reached"))
    with pytest.raises(ValueError, match="formal seed"):
        prepare(Path("runs/c13_smoke"), True)


def test_exact_protocol_ranges():
    assert seeds(SMOKE_SEED, 1, True) == [SMOKE_SEED]
    assert len(seeds(FORMAL_SEED, 100, False)) == 100
    assert len(seeds(SNAP_SEED, 20, False)) == 20
    with pytest.raises(ValueError):
        seeds(FORMAL_SEED, 99, False)


class Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core = torch.nn.Module()
        self.core.bias = torch.nn.Parameter(torch.zeros(2))
        self.readout = torch.nn.Module()
        self.readout.weight = torch.nn.Parameter(torch.zeros(1))
        self.readout.turn_bias = torch.nn.Parameter(torch.zeros(1))
        self.frozen = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.register_buffer("buffer", torch.zeros(1))


def test_actor_distance_excludes_frozen_zero_rate_groups_and_buffers():
    actor, baseline = Actor(), Actor()
    with torch.no_grad():
        actor.core.bias.copy_(torch.tensor([3.0, 4.0]))
        actor.readout.weight.fill_(12)
        actor.readout.turn_bias.fill_(999)
        actor.frozen.fill_(999)
        actor.buffer.fill_(999)
    train = {"readout_bias_learning_rate": 0}
    assert anchor_distance.actor_distance(actor, baseline, train) == 13
    assert (
        anchor_distance.actor_distance(actor, baseline, {**train, "core_bias_learning_rate": 0})
        == 12
    )
    assert (
        anchor_distance.actor_distance(actor, baseline, {**train, "readout_learning_rate": 0}) == 5
    )
    assert anchor_distance.actor_distance(actor, baseline, {"learning_rate": 0}) == 0


def test_distance_subtraction_and_accumulation_are_float64():
    actor, baseline = Actor().double(), Actor().double()
    with torch.no_grad():
        actor.core.bias.copy_(torch.tensor([1e12 + 3, 1e12 + 4], dtype=torch.float64))
        baseline.core.bias.fill_(1e12)
    assert anchor_distance.actor_distance(actor, baseline, {}) == 5
    with torch.no_grad():
        actor.core.bias.fill_(1e20)
    assert anchor_distance.actor_distance(actor, baseline, {}) == pytest.approx(
        2**0.5 * (1e20 - 1e12)
    )


def test_checkpoint_recompute_and_nearest_update(tmp_path, monkeypatch):
    from flytrack_train import config as train_config

    monkeypatch.setattr(train_config, "prepare_policy", lambda *a: (Actor(), None))
    bc = tmp_path / "bc"
    bc.mkdir()
    actor = Actor()
    torch.save(actor.state_dict(), bc / "weights.pt")
    with torch.no_grad():
        actor.core.bias.copy_(torch.tensor([3.0, 4.0]))
    checkpoint = tmp_path / "step.pt"
    torch.save({"actor": actor.state_dict(), "environment_steps": 12}, checkpoint)
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "ppo_c11").mkdir()
    write(
        tmp_path / "snapshots/index.json",
        {"snapshots": [{"step": 12, "boundary": 10, "path": str(checkpoint)}]},
    )
    write(
        tmp_path / "ppo_c11/training_metrics.json",
        [
            {"environment_steps": 10, "bc_anchor_distance": 4},
            {"environment_steps": 14, "bc_anchor_distance": 6},
        ],
    )
    records = anchor_distance.distance_records(tmp_path, {"train": {}}, bc)
    assert [r["actor_distance_l2"] for r in records] == [0, 5]
    assert records[1]["bc_anchor_distance"] == 4
    assert records[1]["training_update_step"] == 10
    assert anchor_distance.nearest_update([], 0)["bc_anchor_distance"] is None


def test_budget_counts_all_gpu_attempts_and_leaves_cpu_free(tmp_path):
    for stage, code, wall in [
        ("s0_train", 0, 4000),
        ("s1_train", "INTERRUPTED", 1000),
        ("s1_train", 0, 2000),
        ("s0_eval", 1, 400),
        ("s1_eval", 0, 300),
        ("bc_eval", 0, 500),
        ("s0_snap_eval", 0, 9000),
    ]:
        driver.record(tmp_path, stage, code, wall)
    assert driver.remaining(tmp_path, {"smoke": False, "gpu_budget_s": 9000}) == 800
    assert driver.remaining(tmp_path, {"smoke": True, "gpu_budget_s": 9000}) == 9000


@pytest.mark.parametrize("stage", GPU_STAGES)
def test_exhausted_budget_launches_nothing(tmp_path, monkeypatch, stage):
    write(tmp_path / "manifest.json", {"smoke": False, "gpu_budget_s": 9000})
    driver.record(tmp_path, "s0_train", 0, 8997)
    monkeypatch.setattr(shared.subprocess, "Popen", lambda *a, **k: pytest.fail("GPU launched"))
    with pytest.raises(driver.BudgetExhausted):
        driver.run_stage(tmp_path, stage)


def test_budget_exhaustion_marks_all_remaining_and_acceptance_not_run(tmp_path, monkeypatch):
    write(tmp_path / "manifest.json", {"smoke": False})
    write(tmp_path / "s0_train.complete.json", {})
    summaries = []
    monkeypatch.setattr(summarize_c13, "summarize", summaries.append)
    driver.not_run(tmp_path, "budget exhausted")
    result = read(tmp_path / "acceptance.json")
    assert result["status"] == result["overall"] == "NOT RUN"
    assert set(result["stages"]) == set(STAGES) - {"s0_train"}
    assert result["rules"] == RULES
    assert all(r["status"] == "NOT RUN" for r in result["candidates"].values())
    assert all(r["exit"] == "NOT RUN" for r in driver.ledger(tmp_path))
    assert summaries == [tmp_path]


def test_completed_stage_skips_launch_with_c13_validation(tmp_path, monkeypatch):
    write(tmp_path / "manifest.json", {"smoke": True})
    write(tmp_path / "acceptance.json", {})
    write(
        tmp_path / "acceptance.complete.json",
        {str(tmp_path / "acceptance.json"): digest(tmp_path / "acceptance.json")},
    )
    validated = []
    monkeypatch.setattr(shared.subprocess, "run", lambda *a, **k: validated.append(a[0]))
    monkeypatch.setattr(shared.subprocess, "Popen", lambda *a, **k: pytest.fail("Stage reran"))
    driver.run_stage(tmp_path, "acceptance")
    assert "c13.driver" in validated[0] and "validate" in validated[0]
    assert not driver.ledger(tmp_path)


def test_stop_precedes_attempt(tmp_path):
    write(tmp_path / "manifest.json", {"smoke": True})
    (tmp_path / "STOP").touch()
    with pytest.raises(driver.Stopped):
        driver.run_stage(tmp_path, "s0_train")
    assert not driver.ledger(tmp_path)


def test_scheduler_starts_snapshot_chain_after_both_training_receipts(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(prepare_c13, "prepare", lambda *a: None)
    monkeypatch.setattr(driver, "run_stage", lambda out, stage: events.append(stage))
    monkeypatch.setattr(shared, "stop_group", lambda process: None)
    monkeypatch.delenv("MONITOR_SH", raising=False)

    class Chain:
        def __init__(self, command, **kwargs):
            assert events == ["s0_train", "s1_train"]
            assert "c13.driver" in command and "snap-chain" in command
            events.append("snapshots")

        def poll(self):
            return 0

        def wait(self):
            return 0

    monkeypatch.setattr(shared.subprocess, "Popen", Chain)
    driver.run(tmp_path, True)
    assert events == [
        "s0_train",
        "s1_train",
        "snapshots",
        "s0_eval",
        "s1_eval",
        "bc_eval",
        "s0_dn",
        "s1_dn",
        "acceptance",
        "summary",
    ]


def test_cpu_chains_nice_ten_and_two_cpus_per_seed(tmp_path, monkeypatch):
    nice, affinity, stages = [], [], []
    monkeypatch.setattr(shared.os, "getpriority", lambda *a: 10)
    monkeypatch.setattr(shared.os, "nice", nice.append)
    monkeypatch.setattr(shared.os, "sched_getaffinity", lambda *a: set(range(16)))
    monkeypatch.setattr(shared.os, "sched_setaffinity", lambda pid, cpus: affinity.append(cpus))
    monkeypatch.setattr(driver, "run_stage", lambda out, stage, **kw: stages.append(stage))
    for seed in SEEDS:
        driver.cpu_chain(tmp_path, seed)
    assert nice == [0, 0] and affinity == [[0, 1], [2, 3]]
    assert len(stages) == 6


def test_snapshot_chain_requires_both_trainings(tmp_path):
    write(tmp_path / "s0_train.complete.json", {})
    with pytest.raises(ValueError, match="Both training"):
        driver.snapshot_coordinator(tmp_path)


def test_shell_rejects_wrong_output_without_writes(tmp_path):
    out = tmp_path / "unused"
    result = subprocess.run(
        ["bash", "experiments/ppo35/c13/run_c13.sh"],
        env={**os.environ, "SMOKE": "1", "OUT": str(out)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and "OUT differs" in result.stdout
    assert not out.exists()


def test_metric_summary_uses_all_updates_and_reports_missing():
    result = summarize_c13.metric_statistics(
        [
            {"bc_anchor_loss": 1, "bc_anchor_distance": 2, "entropy": 3, "actor_grad_norm": 4},
            {"bc_anchor_loss": 5, "bc_anchor_distance": 6, "entropy": 7, "actor_grad_norm": 8},
        ]
    )
    assert result["bc_anchor_loss"] == {"count": 2, "median": 3, "min": 1, "max": 5}
    assert summarize_c13.metric_statistics([])["entropy"]["median"] is None


def test_acceptance_snapshot_descriptions_do_not_change_gates(tmp_path, monkeypatch):
    write(
        tmp_path / "manifest.json",
        {
            "protocol": "C13",
            "smoke": False,
            "episodes": 100,
            "seed_start": FORMAL_SEED,
            "inputs": {"BC_ACTOR": "bc"},
        },
    )
    monkeypatch.setattr(
        c13_jobs, "seed_acceptance", lambda *a: {"accepted": True, "gates_passed": True}
    )
    monkeypatch.setattr(c13_jobs, "snapshot_rows", lambda out: observations(0, 0.6, 0.1))
    result = c13_jobs.acceptance_result(tmp_path)
    assert result["accepted"] and result["overall"] == "C13 一致通過"
    assert result["snapshots"]["interpretations"] == {EARLY: False, CIRCLE: True}


@pytest.fixture
def fake_run_tree(tmp_path, monkeypatch, bases):
    """Only synthetic files; no training, simulation, or delivered-result writes."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(prepare_c13, "ROOT", tmp_path)
    monkeypatch.setattr(summarize_c13, "ROOT", tmp_path)
    monkeypatch.setattr(
        prepare_c13, "relative", lambda p: os.path.relpath(Path(p).resolve(), tmp_path)
    )
    for key in (
        "BC_ACTOR",
        "MEASURE_REPO",
        "MEASURE_PY",
        "SEED_START",
        "SNAP_SEED_START",
        "MONITOR_SH",
    ):
        monkeypatch.delenv(key, raising=False)
    prior = tmp_path / "runs/c12"
    prior.mkdir(parents=True)
    bc = tmp_path / "fixtures/bc"
    bc.mkdir(parents=True)
    (bc / "weights.pt").write_bytes(b"synthetic actor")
    write(bc / "model_card.json", {"files": ["weights.pt"]})
    measure = tmp_path / "fixtures/measure/scripts/diagnostics/modality_sensitivity.py"
    measure.parent.mkdir(parents=True)
    measure.write_text("# synthetic measurement\n")
    main = tmp_path / "fixtures/measure/experiments/odor_chain/measure.py"
    main.parent.mkdir(parents=True)
    main.write_text("# synthetic measurement entry\n")
    configs = {}
    for seed in SEEDS:
        base = copy.deepcopy(bases[seed])
        base["init_artifact"] = "fixtures/bc"
        name = f"ppo_c12_{seed}.yaml"
        configs[name] = base
        (prior / name).write_text(yaml.safe_dump(base, sort_keys=False))
        folder = prior / seed
        (folder / "snapshots").mkdir(parents=True)
        (folder / "ppo_c11").mkdir()
        write(folder / "snap_eval.json", {"rows": []})
        write(folder / "snap_dn.json", {"rows": []})
        write(folder / "snapshots/index.json", {"snapshots": []})
        write(folder / "ppo_c11/training_metrics.json", [])
    evaluation = {"seed": 1_600_600_000, "env": {"episode_seconds": 60}}
    (prior / "eval.yaml").write_text(yaml.safe_dump(evaluation))
    configs["eval.yaml"] = evaluation
    write(prior / "acceptance.json", {"candidates": {}})
    write(prior / "dn_bc.json", {})
    write(
        prior / "manifest.json",
        {
            "inputs": {
                "BC_ACTOR": "fixtures/bc",
                "MEASURE_REPO": "fixtures/measure",
                "MEASURE_PY": str(main.relative_to(tmp_path)),
            },
            "sha256": {
                str(p.relative_to(tmp_path)): digest(p)
                for p in (bc / "weights.pt", bc / "model_card.json")
            },
            "configs": configs,
        },
    )
    monkeypatch.setattr(prepare_c13, "validate_actor", lambda *a: None)
    monkeypatch.setattr(prepare_c13, "anchor_support", lambda: None)
    monkeypatch.setattr(prepare_c13, "training_preflight", lambda: None)
    monkeypatch.setattr(
        prepare_c13, "reference_check", lambda *a: {"commit": "1" * 40, "warnings": []}
    )
    return tmp_path


def test_prepare_smoke_configs_resume_binding_and_portable_outputs(fake_run_tree):
    out = Path("runs/c13_smoke")
    manifest = prepare(out, True)
    assert manifest["gpu_budget_s"] == 9000 and manifest["device"] == "cpu"
    for index, seed in enumerate(SEEDS):
        formal = manifest["formal_c13"][seed]
        assert formal["seed"] == index
        assert formal["train"]["total_steps"] == 500_000
        assert formal["train"]["critic_warmup_updates"] == 10
        assert formal["train"]["bc_anchor_coef"] == LAMBDA
        assert set(manifest["c12_config_diff"][seed]) == {"output", "train.bc_anchor_coef"}
        assert manifest["configs"][f"ppo_c13_{seed}.yaml"]["train"]["total_steps"] == 36
        assert (
            read(out / seed / "manifest.json")["configs"]["ppo_c11.yaml"]
            == manifest["configs"][f"ppo_c13_{seed}.yaml"]
        )
    assert str(fake_run_tree) not in (out / "manifest.json").read_text()
    assert prepare(out, True) == manifest
    (Path("fixtures/bc") / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="BC actor differs"):
        prepare(out, True)


def test_resume_rejects_changed_reference_commit_before_config_writes(fake_run_tree, monkeypatch):
    out = Path("runs/c13_smoke")
    prepare(out, True)
    before = (out / "ppo_c13_s0.yaml").read_bytes()
    monkeypatch.setattr(
        prepare_c13, "reference_check", lambda *a: {"commit": "2" * 40, "warnings": []}
    )
    with pytest.raises(ValueError, match="Run inputs/code/mode changed"):
        prepare(out, True)
    assert (out / "ppo_c13_s0.yaml").read_bytes() == before


def test_summary_exact_steps_numeric_metrics_readonly_and_validation(fake_run_tree, monkeypatch):
    out = Path("runs/c13_smoke")
    prepare(out, True)

    def distances(folder, config, bc_actor):
        # A deliberately different C12 step must stay absent in the joined table.
        step = 50001 if "c12" in folder.parts and folder.name == "s1" else 12
        return [
            {"step": 0, "actor_distance_l2": 0},
            {
                "step": step,
                "actor_distance_l2": 5,
                "bc_anchor_distance": 4,
                "training_update_step": 10,
            },
        ]

    monkeypatch.setattr(summarize_c13, "distance_records", distances)
    write(
        out / "acceptance.json",
        {
            "status": "COMPLETE",
            "overall": "不一致",
            "snapshots": {"interpretations": {EARLY: True, CIRCLE: False}},
            "candidates": {},
        },
    )
    protected = {p: digest(p) for p in Path("runs/c12").rglob("*") if p.is_file()}
    data = summarize_c13.collect(out)
    assert data["c13"]["s0"]["same_step_c12"]["s0"][1]["actor_distance_l2"] == 5
    assert data["c13"]["s0"]["same_step_c12"]["s1"][1] == {"step": 12}
    text = summarize_c13.render(data)
    for label in (
        "C12 (s0)",
        "C12 (s1)",
        "‖θ−θ_BC‖₂",
        "bc_anchor_distance",
        "bc_anchor_loss",
        "最小值",
        "最大值",
        "不一致",
        EARLY,
        CIRCLE,
    ):
        assert label in text
    assert str(fake_run_tree) not in text
    summarize_c13.summarize(out)
    summarize_c13.summarize(out, validate_only=True)
    assert all(digest(p) == value for p, value in protected.items())
    (out / "summary.md").write_text("changed")
    with pytest.raises(ValueError, match="Summary differs"):
        summarize_c13.summarize(out, validate_only=True)


def test_snapshot_descriptions_read_shared_dn_and_eval_sources(tmp_path):
    write(tmp_path / "manifest.json", {"configs": {}})
    report = tmp_path / "dn.json"
    write(report, {"dn": {"DNa01": {"slope_median": 0.01}}})
    for seed in SEEDS:
        folder = tmp_path / seed
        (folder / "snapshots").mkdir(parents=True)
        write(folder / "snapshots/index.json", {"snapshots": [{"boundary": 50000, "step": 51200}]})
        write(folder / "snap_dn.json", {"rows": [{"step": 51200, "report": str(report)}]})
        write(
            folder / "snap_eval.json",
            {
                "rows": [
                    {
                        "step": 51200,
                        "omega_median": 0.6,
                        "tracked_fraction": 0.1,
                        "turn_bias": 0,
                        "core_bias_max_abs": 0,
                    }
                ]
            },
        )
    result = c13_jobs.snapshot_result(tmp_path)
    assert result["interpretations"] == {EARLY: True, CIRCLE: True}
    assert len(result["rows"]) == 2
