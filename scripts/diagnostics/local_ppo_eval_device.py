"""Device-selectable runner for the local paired evaluator (confirmation stage).

The episode loop and summaries are imported unchanged from local_ppo_eval.py;
only the actor device differs. The device is part of the recorded
implementation, so local_ppo_compare.py never pairs CPU with CUDA reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import torch
import yaml
from flytrack_train.config import contract_from_config, project_source_fingerprint, write_json
from flytrack_train.parallel import EnvPool
from flytrack_train.worlds import make_env
from local_ppo_eval import evaluate, summarize
from policy_episode_eval import Actor, Fixed

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def implementation(device):
    return {
        "evaluator_sha256": digest(HERE / "local_ppo_eval.py"),
        "device_runner_sha256": digest(Path(__file__)),
        "actor_wrapper_sha256": digest(HERE / "policy_episode_eval.py"),
        "project_source_sha256": project_source_fingerprint()["sha256"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": str(torch.__version__),
    }


def artifact_metadata(source):
    card = json.loads((source / "model_card.json").read_text())
    training_config = card.get("metadata", {}).get("config", {})
    metadata = {k: card.get("metadata", {}).get(k) for k in ("seed", "stage")}
    metadata["init_artifact"] = training_config.get("init_artifact")
    metadata["local_experiment"] = training_config.get("local_experiment", {})
    metadata["model_card_sha256"] = digest(source / "model_card.json")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", nargs="*", type=Path, default=[])
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=1_630_000_000)
    parser.add_argument("--lanes", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cells", nargs="+", default=[
        "curved/normal", "wander/normal", "curved/occlusion", "curved/noise", "curved/conflict",
    ])
    parser.add_argument("--resume", action="store_true",
                        help="Keep cells already written by an identical earlier invocation")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("Output must be inside this workspace")
    if min(args.episodes, args.lanes, args.threads) < 1:
        parser.error("episodes, lanes and threads must be positive")
    if args.base_seed < 1_000_000_000:
        parser.error("Use held-out episode seeds outside the training range")
    if not args.controls and not args.artifacts:
        parser.error("At least one actor or --controls is required")
    cells = [tuple(cell.split("/")) for cell in args.cells]
    if any(len(cell) != 2 for cell in cells) or len(set(cells)) != len(cells):
        parser.error("Cells are distinct motion/scenario pairs")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to this process")
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(args.config.read_text())
    contract = contract_from_config(config)
    seeds = list(range(args.base_seed, args.base_seed + args.episodes))
    report = {
        "config": config, "config_sha256": digest(args.config),
        "environment_seeds": seeds, "runtime_sparse_override": False,
        "device": str(device), "lanes": args.lanes, "policies": {},
        "implementation": implementation(device),
    }
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        for key in ("config_sha256", "environment_seeds", "device", "lanes", "implementation"):
            if previous.get(key) != report[key]:
                raise ValueError(f"Cannot resume: {key} differs from the existing report")
        report["policies"] = previous["policies"]
    policies = [("teacher", None), ("fixed_straight", [0.2, 0.0])] if args.controls else []
    policies += [(str(p), p) for p in args.artifacts]
    started = time.perf_counter()
    for name, source in policies:
        if isinstance(source, Path):
            act, metadata = Actor(source, device), artifact_metadata(source)
        else:
            act, metadata = Fixed(source), {"control": True}
        entry = report["policies"].setdefault(name, {"metadata": metadata, "cells": {}})
        if entry["metadata"] != metadata:
            raise ValueError(f"Cannot resume: artifact identity changed for {name}")
        for motion, scenario in cells:
            key = f"{motion}/{scenario}"
            if key in entry["cells"]:
                continue
            options = {**config.get("env", {}), "target_motion": motion, "scenario": scenario}
            local_config = {**config, "env": options}
            pool = EnvPool(
                [partial(make_env, local_config, contract) for _ in range(args.lanes)],
                workers=args.workers,
            )
            try:
                rows = evaluate(pool, act, contract, options, seeds, args.lanes)
            finally:
                pool.close()
            summary = summarize(rows)
            entry["cells"][key] = {"summary": summary, "episodes": rows}
            report["elapsed_seconds"] = time.perf_counter() - started
            write_json(args.output, report)
            print(name, key, {k: summary[k] for k in (
                "episode_return", "success", "collision_rate", "tracked_fraction",
                "mean_bearing_error_deg",
            )}, f"{time.perf_counter() - started:.0f}s", flush=True)
    report["complete"] = True
    write_json(args.output, report)


if __name__ == "__main__":
    main()
