"""Shared, read-only validation for the preregistered C10/C9 evaluation protocol."""

import hashlib
import json
import math
import os
from pathlib import Path

SCENARIOS = ("normal", "occlusion", "noise", "conflict")
SEED_START = 1_600_100_000
OPTIONAL = {"reacquisition_time_s", "latency_p50_ms", "latency_p95_ms"}
RUNTIME_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "DETERMINISTIC",
    "CUBLAS_WORKSPACE_CONFIG",
    "EPISODES",
    "SEED_START",
    "LANES",
    "WORKERS",
    "PPO_WORKERS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "SMOKE",
)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value, label, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Undefined/nonfinite {label}")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"Out of range {label}: {value}")
    return value


def finite(value):
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite stage output")


def validate_actor(path):
    path = Path(path)
    card = read(path / "model_card.json")
    if card["variant"] != "V" or not card["files"]:
        raise ValueError(f"Expected a V actor: {path}")
    for name, expected in card["files"].items():
        if digest(path / name) != expected:
            raise ValueError(f"Actor checksum mismatch: {path / name}")
    return card


def verify_c9(root):
    root = Path(root)
    sums = root / "runs/t3c_pack/t3c_c9_7b6a722/trained_actor_SHA256SUMS"
    actor = root / "runs/t3c/ppo_bc/actor"
    checked = {}
    for line in sums.read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        if not name.startswith("ppo_bc/actor/"):
            continue
        path = (root / "runs/t3c" / name).resolve()
        if not path.is_relative_to(actor) or digest(path) != expected:
            raise ValueError(f"C9 checksum mismatch: {name}")
        if name in checked:
            raise ValueError(f"Duplicate C9 checksum: {name}")
        checked[name] = expected
    card = validate_actor(actor)
    required = {f"ppo_bc/actor/{name}" for name in ("model_card.json", *card["files"])}
    if not required.issubset(checked):
        raise ValueError("C9 checksum list does not cover the complete actor")
    return checked


def dn_rows(report, actor=None, steps=None):
    finite(report)
    if actor is not None and report["source"] != str(actor):
        raise ValueError("DN actor source differs")
    if steps is not None and report["steps"] != steps:
        raise ValueError("DN steps differ")
    if report.get("seed") != 1_610_000_000:
        raise ValueError("DN seed differs")
    for name in ("DNa01", "DNa02"):
        row = report["dn"][name]
        number(row["n"], f"{name}.n", 1)
        number(row["voltage_median"], f"{name}.voltage_median")
        number(row["slope_median"], f"{name}.slope_median", 0, 1)
    return report["dn"]


def validate_eval(report, actors, *, episodes, seed_start, smoke, env):
    finite(report)
    if set(report["policies"]) != set(map(str, actors)):
        raise ValueError("Evaluation actor set differs")
    if report["episodes"] != episodes or report["seed_start"] != seed_start:
        raise ValueError("Evaluation episode count/seed block differs")
    if report["env"] != env or not report["deterministic"]:
        raise ValueError("Evaluation environment/determinism differs")
    if report["device"] != ("cpu" if smoke else "cuda"):
        raise ValueError("Evaluation device differs")
    if not smoke and "4060" not in report["device_name"]:
        raise ValueError("Formal evaluation requires the 4060")
    if not report["torch_version"] or report["environment"]["DETERMINISTIC"] != "1":
        raise ValueError("Missing runtime provenance")
    for actor in map(str, actors):
        validate_actor(actor)
        if report["actor_sha256"][actor] != actor_hashes(actor):
            raise ValueError("Evaluated actor changed")
        number(report["actor_wall_s"][actor], "actor wall time", 0)
        scenarios = report["policies"][actor]
        if set(scenarios) != set(SCENARIOS):
            raise ValueError("Scenario set differs")
        for row in scenarios.values():
            rows = row["episodes"]
            if [r["environment_seed"] for r in rows] != list(
                range(seed_start, seed_start + episodes)
            ):
                raise ValueError("Unpaired, missing or duplicate environment seeds")
            for metric in ("success", "collision_rate", "mean_bearing_error_deg"):
                high = 180 if metric == "mean_bearing_error_deg" else 1
                number(row[metric], metric, 0, high)
                values = [number(r[metric], metric, 0, high) for r in rows]
                if not math.isclose(row[metric], sum(values) / episodes, abs_tol=1e-12):
                    raise ValueError(f"Summary differs from episode data: {metric}")
    return report


def actor_hashes(actor):
    card = read(Path(actor) / "model_card.json")
    return {name: digest(Path(actor) / name) for name in ("model_card.json", *card["files"])}


def environment():
    return {key: os.environ.get(key) for key in RUNTIME_ENV}
