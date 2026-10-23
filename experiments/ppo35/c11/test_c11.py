"""C11 protocol isolation, snapshot races, recovery accounting and artifact bits."""

import copy
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from c11_common import (  # noqa: E402
    FORMAL_SEED,
    ROOT,
    SMOKE_SEED,
    SNAP_SEED,
    STAGES,
    digest,
    portable,
    read,
    seeds,
    write,
)
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
    training_command,
)
from snapshots import Sidecar, bitwise_equal, boundaries, build_actor  # noqa: E402

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
