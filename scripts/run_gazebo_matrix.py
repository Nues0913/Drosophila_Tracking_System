"""Run paired Gazebo cells in an isolated Compose project after all validation.

No containers are launched merely by importing this file. Existing evaluation
files are never overwritten. First executed for the #88 G1 trial (2026-11-08).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    # The last line can be in the middle of an evaluator write.
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()]


V2_VARIANTS = ("VO2", "V2", "O2", "RW2", "ER2", "GRU2")


def card_seed(card: dict) -> int:
    """Training seed of a v1 card (``metadata.seed``) or a v2 card (``provenance.train_seed``)."""
    if card.get("spec_version") == "v2":
        return int(card.get("provenance", {}).get("train_seed", -1))
    return int(card.get("metadata", {}).get("seed", -1))


def validate_cell_rows(rows, variant, seed, card_hash, target_motion):
    if any(
        row["variant"] != variant
        or int(row["seed"]) != seed
        or row.get("model_card_sha256") != card_hash
        or row.get("target_motion") != target_motion
        for row in rows
    ):
        raise RuntimeError(
            "Recorded checkpoint/model labels or target motion differ from this cell"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("V", "O", "VO", "VO-late", "GRU", "RS") + V2_VARIANTS,
        default=["V", "VO", "VO-late", "GRU", "RS"],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("normal", "occlusion", "noise", "conflict"),
        default=["normal", "occlusion", "noise", "conflict"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--environment-seed", type=int, default=1700000000)
    parser.add_argument("--target-motion", choices=("curved", "wander"), default="curved")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("/dev/null"),
        help="Compose variable file; defaults to /dev/null to avoid reading .env",
    )
    parser.add_argument("--artifact-template", default="runs/bc/{variant}/seed_{seed}/actor")
    parser.add_argument("--output", type=Path, default=Path("runs/gazebo_eval"))
    parser.add_argument("--project-name", default="flytrack-gz-eval")
    parser.add_argument("--domain-id", type=int, default=77)
    parser.add_argument("--cell-timeout", type=float, default=7200)
    parser.add_argument("--cpu", action="store_true")
    # Policy runtime (#88 G0b): torch threads for the policy container and its device.
    parser.add_argument("--policy-threads", type=int, default=1)
    parser.add_argument("--policy-device", choices=("cpu", "cuda"), default="cpu")
    # Camera hex-eye kernel (#111 G0e): "gaussian" reproduces runs before G0e (formal-v1 G2).
    parser.add_argument("--vision-kernel", choices=("subrays", "gaussian"), default="subrays")
    # #140: a v2 card brings its own sensor contract; the world stays v1compat until the
    # wind G1 is registered (#140 7001508986).
    parser.add_argument("--world", choices=("v1compat", "wind"), default="v1compat")
    args = parser.parse_args()
    if (
        args.episodes < 1
        or args.cell_timeout <= 0
        or args.policy_threads < 1
        or min(args.seeds) < 0
        or not 0 <= args.domain_id <= 101
    ):
        parser.error(
            "episodes/timeout/policy-threads must be positive, seeds nonnegative, "
            "domain-id in [0,101]"
        )
    if any(
        len(set(values)) != len(values) for values in (args.variants, args.scenarios, args.seeds)
    ):
        parser.error("matrix axes must not contain duplicates")
    root = Path(__file__).resolve().parent.parent
    output = (root / args.output).resolve()
    if not output.is_relative_to(root / "runs"):
        parser.error("output must be inside the project's mounted runs directory")
    output.mkdir(parents=True, exist_ok=True)
    matrices = []
    # Read all cards before starting a container; the actor loader verifies tensor/file integrity.
    for variant in args.variants:
        for seed in args.seeds:
            artifact = (root / args.artifact_template.format(variant=variant, seed=seed)).resolve()
            if not (
                artifact.is_relative_to(root / "runs")
                or artifact.is_relative_to(root / "artifacts")
            ):
                raise ValueError("Actors must be in mounted runs/ or artifacts/")
            card_path = artifact / "model_card.json"
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if card["variant"] != variant or card_seed(card) != seed:
                raise ValueError(f"Artifact matrix label differs from {artifact}")
            if card.get("runtime_lesion", "none") != "none":
                raise ValueError("Formal Gazebo comparison requires unlesioned actors")
            for scenario in args.scenarios:
                cell = output / variant / f"seed_{seed}" / scenario
                path = cell / "episodes.jsonl"
                if path.exists():
                    raise FileExistsError(
                        f"Preserve existing results in {cell}; choose a new output directory"
                    )
                matrices.append((variant, seed, scenario, artifact, cell, sha256(card_path)))
            if (card.get("spec_version") == "v2") != (variant in V2_VARIANTS):
                raise ValueError(f"{artifact}: card spec_version does not match variant {variant}")
    compose = [
        "docker",
        "compose",
        "--env-file",
        str(args.env_file.resolve()),
        "--project-name",
        args.project_name,
        "-f",
        str(root / "compose.yaml"),
    ]
    if args.cpu:
        compose += ["-f", str(root / "compose.cpu.yaml")]
    if args.policy_device == "cuda":
        compose += ["-f", str(root / "compose.policy-gpu.yaml")]
    compose += ["--profile", "sim"]
    manifest = {
        "project": args.project_name,
        "domain_id": args.domain_id,
        "episodes_per_cell": args.episodes,
        "environment_seed_base": args.environment_seed,
        "target_motion": args.target_motion,
        "policy_torch_threads": args.policy_threads,
        "policy_device": args.policy_device,
        "policy_vision_kernel": args.vision_kernel,
        "cpu_rendering": bool(args.cpu),
        "world": args.world,
        "cells": [],
    }
    try:
        for variant, seed, scenario, artifact, cell, card_hash in matrices:
            cell.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(
                ARTIFACT="/workspace/" + str(artifact.relative_to(root)),
                TRAINING_SEED=str(seed),
                ENVIRONMENT_SEED=str(args.environment_seed + seed * 100000),
                SCENARIO=scenario,
                TARGET_MOTION=args.target_motion,
                MAX_EPISODES=str(args.episodes),
                ROS_DOMAIN_ID=str(args.domain_id),
                EVAL_OUTPUT="/workspace/" + str(cell.relative_to(root)),
                POLICY_TORCH_THREADS=str(args.policy_threads),
                POLICY_DEVICE=args.policy_device,
                POLICY_VISION_KERNEL=args.vision_kernel,
                FLYTRACK_WORLD=args.world,
            )
            if variant in V2_VARIANTS:
                # The deployment contract must hold every field of the model's (#182).
                environment["SENSOR_CONTRACT"] = "/workspace/" + str(
                    (artifact / "sensor_contract.json").relative_to(root)
                )
            else:
                # v1 actors keep the v1 contract; Compose now defaults to the v2 one.
                environment["SENSOR_CONTRACT"] = "/workspace/configs/sensor_contract.json"
            print(f"Gazebo {variant}, model seed {seed}, {scenario}", flush=True)
            cell_started = datetime.now(timezone.utc).isoformat()
            cell_started_monotonic = time.monotonic()
            subprocess.run(
                compose + ["up", "-d", "--wait", "sim", "policy", "eval"],
                cwd=root,
                env=environment,
                check=True,
                timeout=600,
            )
            deadline = time.monotonic() + args.cell_timeout
            while True:
                rows = complete_rows(cell / "episodes.jsonl")
                if len(rows) >= args.episodes:
                    if len(rows) != args.episodes:
                        raise RuntimeError(
                            "More episodes were recorded than the declared cell budget"
                        )
                    validate_cell_rows(rows, variant, seed, card_hash, args.target_motion)
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Only {len(rows)} complete episodes in {cell}")
                # A failed policy must not make the batch silently wait for the whole deadline.
                state = subprocess.run(
                    compose + ["ps", "--all", "--format", "json"],
                    cwd=root,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                statuses = (
                    json.loads(state.stdout)
                    if state.stdout.strip().startswith("[")
                    else [json.loads(line) for line in state.stdout.splitlines() if line.strip()]
                )
                if any(service.get("State") in {"exited", "dead"} for service in statuses):
                    raise RuntimeError(
                        "An evaluation container stopped; inspect this Compose project's logs"
                    )
                time.sleep(1)
            subprocess.run(
                compose + ["down", "--remove-orphans"],
                cwd=root,
                env=environment,
                check=True,
                timeout=120,
            )
            manifest["cells"].append(
                {
                    "variant": variant,
                    "seed": seed,
                    "scenario": scenario,
                    "artifact": str(artifact.relative_to(root)),
                    "model_card_sha256": card_hash,
                    "episodes": len(rows),
                    "output": str(cell.relative_to(root)),
                    "started_wall_utc": cell_started,
                    "finished_wall_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_wall_s": time.monotonic() - cell_started_monotonic,
                }
            )
            (output / "gazebo_matrix_manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n"
            )
    finally:
        # Scoped to this script's project; no volumes or unrelated Compose projects are removed.
        subprocess.run(compose + ["down", "--remove-orphans"], cwd=root, check=False, timeout=120)


if __name__ == "__main__":
    main()
