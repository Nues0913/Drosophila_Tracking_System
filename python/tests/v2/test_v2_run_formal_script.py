"""The formal-matrix runner (scripts/v2/run_formal.py, #143 WP9)."""

import argparse
import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("run_formal", ROOT / "scripts/v2/run_formal.py")
run_formal = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_formal)

PROTOCOL = yaml.safe_load((ROOT / "configs/v2/evaluation_protocol.yaml").read_text())
HOSTS = yaml.safe_load((ROOT / "configs/v2/train/hosts.yaml").read_text())
HEAD = "0123456789abcdef" * 2 + "01234567"
ASSIGNMENT = {"4070": [5, 1], "4060": [2, 3], "2060": [4]}
FROZEN = {**PROTOCOL, "status": "frozen", "host_assignment": ASSIGNMENT}


def args(tmp_path, **changes):
    values = {
        "host": "4070", "seeds": [5, 1], "sha": HEAD[:12], "out": tmp_path / "formal",
        "variants": list(PROTOCOL["model_variants"]), "device": "cuda", "smoke": False,
    }  # fmt: skip
    return argparse.Namespace(**{**values, **changes})


def final_dir(path: Path, weights: bytes = b"w") -> Path:
    path.mkdir(parents=True)
    files = {"weights.pt": weights, "model_card.json": b"{}"}
    for name, data in files.items():
        (path / name).write_bytes(data)
    (path / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in files.items())
    )
    return path


def test_preflight_passes_a_frozen_protocol_at_the_decision_sha(tmp_path):
    assert run_formal.preflight(args(tmp_path), FROZEN, HOSTS, HEAD, "", {}) == []


@pytest.mark.parametrize(
    "changes, protocol, dirty, env, reason",
    [
        ({"sha": "fedcba987654"}, FROZEN, "", {}, "is not --sha"),
        ({"sha": HEAD[:8]}, FROZEN, "", {}, "is not --sha"),  # too short to name a commit
        ({}, FROZEN, " M python/x.py", {}, "tracked tree has changes"),
        ({}, FROZEN, "", {"FLYTRACK_CHEM_GAIN": "2"}, "FLYTRACK_CHEM_GAIN"),
        ({"seeds": [0, 1]}, FROZEN, "", {}, "not the protocol's training seeds"),
        ({"seeds": [100]}, FROZEN, "", {}, "not the protocol's training seeds"),
        ({"seeds": [1, 1]}, FROZEN, "", {}, "listed twice"),
        ({}, PROTOCOL | {"status": "draft_until_m4"}, "", {}, "still a draft"),
        ({"host": "3090"}, FROZEN, "", {}, "not in configs/v2/train/hosts.yaml"),
        ({"variants": ["VO2", "GRU3"]}, FROZEN, "", {}, "not in the protocol"),
        ({"seeds": [5, 2]}, FROZEN, "", {}, "not assigned to host 4070"),
        ({}, FROZEN | {"host_assignment": "decided at M4"}, "", {}, "not a host -> seeds"),
    ],
)
def test_preflight_refuses(tmp_path, changes, protocol, dirty, env, reason):
    problems = run_formal.preflight(args(tmp_path, **changes), protocol, HOSTS, HEAD, dirty, env)
    assert any(reason in problem for problem in problems), problems


def test_container_settings_are_not_overrides(tmp_path):
    env = {"FLYTRACK_IMAGE_TAG": "v2", "FLYTRACK_RUN_IMAGE_ID": "sha256:ab", "PATH": "/usr/bin"}
    assert run_formal.preflight(args(tmp_path), FROZEN, HOSTS, HEAD, "", env) == []


def test_smoke_takes_pilot_seeds_and_a_draft_but_never_a_training_seed(tmp_path):
    draft = PROTOCOL | {"status": "draft_until_m4"}
    smoke = args(tmp_path, smoke=True, seeds=[100])
    assert run_formal.preflight(smoke, draft, HOSTS, HEAD, "", {}) == []
    smoke.seeds = [1]
    assert run_formal.preflight(smoke, draft, HOSTS, HEAD, "", {})


def test_host_assignment_keys_may_be_numbers(tmp_path):
    numeric = FROZEN | {"host_assignment": {4070: [5, 1], 4060: [2, 3], 2060: [4]}}
    assert run_formal.preflight(args(tmp_path), numeric, HOSTS, HEAD, "", {}) == []


def test_each_stage_is_evaluated_on_the_held_out_episodes_right_after_it(tmp_path):
    out = ROOT / "runs/v2_formal"
    steps = run_formal.chain(args(tmp_path, out=out), "RW2", 5)
    assert [s["name"] for s in steps] == ["stage1", "heldout_stage1", "stage2", "heldout_stage2"]
    stage1, held1, stage2, held2 = (s["cmd"] for s in steps)
    assert stage1[1:3] == ["-m", "flytrack_v2.train.dagger"]
    assert stage1[stage1.index("--variant") + 1] == "RW2"
    assert stage1[stage1.index("--out") + 1] == "runs/v2_formal/RW2/seed_5/stage1"
    for cmd in (stage1, stage2):
        assert cmd[cmd.index("--seed") + 1] == "5" and cmd[cmd.index("--host") + 1] == "4070"
        assert "--override" not in cmd  # the formal recipe is the configs at the SHA
    assert stage2[stage2.index("--init") + 1] == "runs/v2_formal/RW2/seed_5/stage1/final"
    for stage, cmd in (("stage1", held1), ("stage2", held2)):
        assert cmd[cmd.index("--artifacts") + 1] == f"runs/v2_formal/RW2/seed_5/{stage}/final"
        assert cmd[cmd.index("--seeds") + 1] == "pilot_held_out"
        assert "--no-sensitivity" in cmd and "--episodes" not in cmd


def test_smoke_chain_is_tiny(tmp_path):
    steps = run_formal.chain(args(tmp_path, smoke=True, seeds=[100]), "VO2", 100)
    assert "rounds.env_steps_per_round=72" in steps[0]["cmd"]
    assert "total_env_steps=96" in steps[2]["cmd"]
    assert steps[1]["cmd"][-2:] == ["--episodes", "2"] == steps[3]["cmd"][-2:]


def test_artifact_done_needs_every_checksum_to_match(tmp_path):
    assert not run_formal.artifact_done(tmp_path / "missing")
    final = final_dir(tmp_path / "final")
    assert run_formal.artifact_done(final)
    (final / "weights.pt").write_bytes(b"changed")
    assert not run_formal.artifact_done(final)


def fake_executor(monkeypatch, outcomes, reason="round 4: deterministic speed std is <= 0.02"):
    """Each call pops (exit code, export final/?) for the step; records (step, attempt, resume)
    and appends the runner's log line as execute does."""
    calls = []

    def execute(step, attempt, log, env, resume):
        code, export = outcomes[step["name"]].pop(0)
        calls.append((step["name"], attempt, resume))
        step["out"].mkdir(parents=True, exist_ok=True)
        if step["kind"] == "train":
            (step["out"] / "config.json").write_text("{}")
            if export:
                final_dir(step["out"] / "final")
                if code == 3:
                    (step["out"] / "failure").mkdir()
                    (step["out"] / "failure/reason.json").write_text(json.dumps({"reason": reason}))
        elif export:
            (step["out"] / "manifest.json").write_text("{}")
        record = {"step": step["name"], "run": run_formal.rel(step["run"]), "attempt": attempt,
                  "exit": code, "wall_s": 1.0}  # fmt: skip
        with log.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    monkeypatch.setattr(run_formal, "execute", execute)
    return calls


def all_pass(**changes):
    """Every step succeeds once, except the given steps' outcomes (fresh lists each call)."""
    steps = ("stage1", "heldout_stage1", "stage2", "heldout_stage2")
    return {**{name: [(0, True)] for name in steps}, **changes}


@pytest.mark.parametrize(
    "reason, note",
    [
        ("round 4: deterministic speed std is <= 0.02 m/s. Exported to final/.", "low speed"),
        ("round 4: silent readout DNs. Exported to final/.", "silent readout DNs"),
        (
            "round 4: silent readout DNs, deterministic speed std is <= 0.02 m/s",
            "silent readout DNs, low speed",
        ),
    ],
)  # fmt: skip
def test_stage1_health_warning_is_recorded_with_its_reason(tmp_path, monkeypatch, reason, note):
    steps = run_formal.chain(args(tmp_path), "VO2", 5)
    replay = steps[0]["out"] / "round_3/replay.pt"
    replay.parent.mkdir(parents=True)
    replay.write_bytes(b"r")
    calls = fake_executor(monkeypatch, all_pass(stage1=[(3, True)]), reason)
    log = tmp_path / "log.jsonl"
    assert run_formal.run_chain(steps, log, {}, 0) == f"done (stage1 health: {note})"
    assert [c[0] for c in calls] == ["stage1", "heldout_stage1", "stage2", "heldout_stage2"]
    assert not replay.exists()
    # Kept on the next run, with the same note from the log.
    assert run_formal.run_chain(steps, log, {}, 0, run_formal.history(log)) == (
        f"done (stage1 health: {note})"
    )


def test_a_failed_stage2_still_leaves_stage1_evaluated(tmp_path, monkeypatch):
    steps = run_formal.chain(args(tmp_path), "ER2", 1)
    calls = fake_executor(monkeypatch, all_pass(stage2=[(1, False), (1, False)]))
    assert run_formal.run_chain(steps, tmp_path / "log.jsonl", {}, 0).startswith("failed: stage2")
    assert ("heldout_stage1", 1, False) in calls and (steps[1]["out"] / "manifest.json").exists()


def test_a_crash_resumes_and_attempts_count_across_runs(tmp_path, monkeypatch):
    steps = run_formal.chain(args(tmp_path), "ER2", 1)
    log = tmp_path / "log.jsonl"
    calls = fake_executor(
        monkeypatch, all_pass(stage1=[(1, False), (0, True)], stage2=[(1, False), (1, False)])
    )
    assert run_formal.run_chain(steps, log, {}, 0) == "failed: stage2 (exit 1, 2 attempt(s))"
    assert calls[:2] == [("stage1", 1, False), ("stage1", 2, True)]  # resumed after the crash
    # Running the same command again does not try stage2 a third time ...
    calls.clear()
    assert run_formal.run_chain(steps, log, {}, 0, run_formal.history(log)).startswith("failed")
    assert calls == []
    # ... unless asked to (--retry-failed: no history); the stage resumes from its config.json.
    fake_executor(monkeypatch, {"stage2": [(0, True)], "heldout_stage2": [(0, True)]})
    assert run_formal.run_chain(steps, log, {}, 0, {}) == "done"


def test_monitor_failures_are_not_resumed(tmp_path, monkeypatch):
    steps = run_formal.chain(args(tmp_path), "VO2", 5)
    calls = fake_executor(monkeypatch, all_pass(stage1=[(3, False)]))
    assert run_formal.run_chain(steps, tmp_path / "log.jsonl", {}, 0) == (
        "failed: stage1 (exit 3, 1 attempt(s))"
    )
    assert calls == [("stage1", 1, False)]


def test_a_reboot_mid_stage_resumes_without_counting_an_attempt(tmp_path, monkeypatch):
    steps = run_formal.chain(args(tmp_path), "VO2", 5)
    steps[0]["out"].mkdir(parents=True)
    (steps[0]["out"] / "config.json").write_text("{}")  # started, never logged
    calls = fake_executor(monkeypatch, all_pass())
    assert run_formal.run_chain(steps, tmp_path / "log.jsonl", {}, 0) == "done"
    assert calls[0] == ("stage1", 1, True)


def test_a_verified_final_is_never_run_again(tmp_path, monkeypatch):
    steps = run_formal.chain(args(tmp_path), "VO2", 5)
    calls = fake_executor(monkeypatch, all_pass(stage2=[(1, True)]))
    assert run_formal.run_chain(steps, tmp_path / "log.jsonl", {}, 0) == (
        "done (stage2 exit 1 with final/ verified)"
    )
    assert [c[0] for c in calls].count("stage2") == 1


def test_the_evidence_has_no_weights_and_no_local_paths(tmp_path):
    out = tmp_path / "formal"
    run = out / "VO2/seed_5"
    final_dir(run / "stage1/final")
    (run / "stage1/round_0").mkdir()
    (run / "stage1/round_0/replay.pt").write_bytes(b"r")
    (run / "stage1/tensorboard").mkdir()
    (run / "stage1/tensorboard/events.out.tfevents.1.machine-name").write_bytes(b"t")
    (run / "stage2").mkdir()
    (run / "stage2/checkpoint.pt").write_bytes(b"c")
    (run / "stage2/metrics.jsonl").write_text(json.dumps({"update": 1}) + "\n")
    (run / "stage1.log").write_text(f"warning in {ROOT}/python/x.py, {Path.home()}/.venv\n")
    evidence, weights, sums = run_formal.archive(out, "4070", [run], [])
    with tarfile.open(evidence) as tar:
        names = tar.getnames()
        log = tar.extractfile("VO2/seed_5/stage1.log").read().decode()
    assert "VO2/seed_5/stage2/metrics.jsonl" in names
    assert "VO2/seed_5/stage1/final/model_card.json" in names
    assert not any(n.endswith((".pt",)) or "round_" in n or "tensorboard" in n for n in names)
    assert str(ROOT) not in log and str(Path.home()) not in log and "<repo>/python" in log
    with tarfile.open(weights) as tar:
        assert "VO2/seed_5/stage1/final/weights.pt" in tar.getnames()
    assert sums.read_text().count("  v2_formal_4070_") == 2
