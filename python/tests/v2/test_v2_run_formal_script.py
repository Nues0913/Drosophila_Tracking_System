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
FROZEN = {**PROTOCOL, "status": "frozen"}


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


def test_a_chain_trains_both_stages_then_evaluates_both_on_the_held_out_episodes(tmp_path):
    out = ROOT / "runs/v2_formal"
    steps = run_formal.chain(args(tmp_path, out=out), "RW2", 5)
    assert [s["name"] for s in steps] == ["stage1", "stage2", "heldout_stage1", "heldout_stage2"]
    stage1, stage2, held1, held2 = (s["cmd"] for s in steps)
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
    assert "total_env_steps=96" in steps[1]["cmd"]
    assert steps[2]["cmd"][-2:] == ["--episodes", "2"]


def test_artifact_done_needs_every_checksum_to_match(tmp_path):
    assert not run_formal.artifact_done(tmp_path / "missing")
    final = final_dir(tmp_path / "final")
    assert run_formal.artifact_done(final)
    (final / "weights.pt").write_bytes(b"changed")
    assert not run_formal.artifact_done(final)


def fake_executor(monkeypatch, outcomes):
    """Each call pops (exit code, export final/?) for the step; records the attempts."""
    calls = []

    def execute(step, attempt, log, env):
        code, export = outcomes[step["name"]].pop(0)
        calls.append((step["name"], attempt))
        step["out"].mkdir(parents=True, exist_ok=True)
        if step["kind"] == "train":
            (step["out"] / "config.json").write_text("{}")
            if export:
                final_dir(step["out"] / "final")
        elif export:
            (step["out"] / "manifest.json").write_text("{}")
        return {"exit": code, "wall_s": 1.0}

    monkeypatch.setattr(run_formal, "execute", execute)
    return calls


def test_stage1_health_warning_is_not_a_failure_and_its_replay_is_deleted(tmp_path, monkeypatch):
    steps = run_formal.chain(args(tmp_path), "VO2", 5)
    replay = steps[0]["out"] / "round_3/replay.pt"
    replay.parent.mkdir(parents=True)
    replay.write_bytes(b"r")
    calls = fake_executor(
        monkeypatch,
        {"stage1": [(3, True)], "stage2": [(0, True)],
         "heldout_stage1": [(0, True)], "heldout_stage2": [(0, True)]},
    )  # fmt: skip
    assert run_formal.run_chain(steps, tmp_path / "log.jsonl", {}, 0) == "done"
    assert calls == [("stage1", 1), ("stage2", 1), ("heldout_stage1", 1), ("heldout_stage2", 1)]
    assert not replay.exists()


def test_a_crashed_stage_resumes_once_then_the_chain_stops(tmp_path, monkeypatch):
    steps = run_formal.chain(args(tmp_path), "ER2", 1)
    calls = fake_executor(
        monkeypatch, {"stage1": [(1, False), (0, True)], "stage2": [(1, False)] * 2}
    )
    assert run_formal.run_chain(steps, tmp_path / "log.jsonl", {}, 0) == "stage2"
    assert calls == [("stage1", 1), ("stage1", 2), ("stage2", 1), ("stage2", 2)]
    # The next run keeps the finished stage and retries the failed one.
    calls.clear()
    fake_executor(monkeypatch, {"stage2": [(0, True)], "heldout_stage1": [(0, True)],
                                "heldout_stage2": [(0, True)]})  # fmt: skip
    assert run_formal.run_chain(steps, tmp_path / "log.jsonl", {}, 0) == "done"


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
