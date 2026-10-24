"""Paired full-episode evaluation for local BC/PPO experiments.

Retains raw returns and individual episodes; never treats episode seeds as
independent training seeds. Outputs are restricted to the current workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml
from flytrack_core.metrics import EpisodeMetrics
from flytrack_train.config import contract_from_config, project_source_fingerprint, write_json
from flytrack_train.parallel import EnvPool
from flytrack_train.worlds import make_env
from policy_episode_eval import Actor, Fixed

ROOT = Path(__file__).resolve().parents[2]


def evaluate(pool, act, contract, options, seeds, lanes):
    rows = []
    for first in range(0, len(seeds), lanes):
        batch_seeds = seeds[first : first + lanes]
        slots = list(range(len(batch_seeds)))
        obs, infos = zip(*(pool.reset(i, s) for i, s in enumerate(batch_seeds)), strict=True)
        obs, infos = list(obs), list(infos)
        metrics = [EpisodeMetrics.from_env_options(contract.control_dt, options) for _ in slots]
        returns = np.zeros(len(slots), dtype=np.float64)
        omega, teacher = [[] for _ in slots], [[] for _ in slots]
        act.reset(len(slots))
        live = slots.copy()
        while live:
            actions = act(obs, infos, live)
            stepped = pool.step(dict(zip(live, actions, strict=True)))
            next_live = []
            for j, i in enumerate(live):
                observation, reward, terminated, truncated, info = stepped[i]
                applied = info.get("applied_action", actions[j])
                returns[i] += reward
                omega[i].append(float(applied[1]))
                teacher[i].append(float(infos[i]["teacher_action"][1]))
                metrics[i].append(
                    info["target_distance_m"], info["target_bearing_rad"], applied[1],
                    info["collision"], info.get("occluded", False), info.get("time_s"),
                )
                obs[i], infos[i] = observation, info
                if terminated or truncated:
                    terminal = info["episode_metrics"]
                    if not np.isclose(returns[i], terminal["episode_return"], atol=1e-6):
                        raise ValueError("Accumulated reward differs from environment return")
                else:
                    next_live.append(i)
            live = next_live
        for i, seed in enumerate(batch_seeds):
            row = metrics[i].result()
            actual, desired = np.asarray(omega[i]), np.asarray(teacher[i])
            row.update(
                environment_seed=seed,
                episode_return=float(returns[i]),
                episode_seconds=len(actual) * contract.control_dt,
                omega_std=float(actual.std()),
                teacher_omega_std=float(desired.std()),
                omega_corr_with_teacher=float(np.corrcoef(actual, desired)[0, 1])
                if actual.std() > 1e-10 and desired.std() > 1e-10 else None,
            )
            terminal = infos[i]["episode_metrics"]
            if row["success"] != float(terminal["success"]):
                raise ValueError("Evaluator/environment success definitions differ")
            if not np.isclose(row["tracked_fraction"], terminal["tracking_fraction"]):
                raise ValueError("Evaluator/environment tracking fractions differ")
            rows.append(row)
    return rows


def summarize(rows):
    result = {}
    for key in rows[0]:
        if key == "environment_seed":
            continue
        values = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        result[key] = float(np.mean(values)) if values else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", nargs="*", type=Path, default=[])
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--base-seed", type=int, default=1_610_000_000)
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--scenarios", nargs="+", default=["normal"])
    parser.add_argument("--motions", nargs="+", default=["curved"])
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("Output must be inside this workspace")
    if min(args.episodes, args.lanes, args.threads) < 1:
        parser.error("episodes, lanes and threads must be positive")
    if args.base_seed < 1_000_000_000:
        parser.error("Use held-out episode seeds outside the training range")
    if not args.controls and not args.artifacts:
        parser.error("At least one actor or --controls is required")
    torch.set_num_threads(args.threads)
    config = yaml.safe_load(args.config.read_text())
    contract = contract_from_config(config)
    seeds = list(range(args.base_seed, args.base_seed + args.episodes))
    report = {
        "config": config, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment_seeds": seeds, "runtime_sparse_override": args.sparse,
        "device": "cpu", "policies": {},
        "implementation": {
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "actor_wrapper_sha256": hashlib.sha256(
                Path(__file__).with_name("policy_episode_eval.py").read_bytes()
            ).hexdigest(),
            "project_source_sha256": project_source_fingerprint()["sha256"],
        },
    }
    policies = [("teacher", None), ("fixed_straight", [0.2, 0.0])] if args.controls else []
    policies += [(str(p), p) for p in args.artifacts]
    started = time.perf_counter()
    for name, source in policies:
        if isinstance(source, Path):
            act = Actor(source, torch.device("cpu"))
            if args.sparse and hasattr(act.policy, "core"):
                act.policy.core.dense = False
            card = json.loads((source / "model_card.json").read_text())
            metadata = {k: card.get("metadata", {}).get(k) for k in ("seed", "stage")}
            training_config = card.get("metadata", {}).get("config", {})
            metadata["init_artifact"] = training_config.get("init_artifact")
            metadata["local_experiment"] = training_config.get("local_experiment", {})
            metadata["model_card_sha256"] = hashlib.sha256(
                (source / "model_card.json").read_bytes()
            ).hexdigest()
        else:
            act, metadata = Fixed(source), {"control": True}
        report["policies"][name] = {"metadata": metadata, "cells": {}}
        for motion in args.motions:
            for scenario in args.scenarios:
                options = {**config.get("env", {}), "target_motion": motion, "scenario": scenario}
                local_config = {**config, "env": options}
                with_pool = EnvPool(
                    [partial(make_env, local_config, contract) for _ in range(args.lanes)],
                    workers=args.workers,
                )
                try:
                    rows = evaluate(with_pool, act, contract, options, seeds, args.lanes)
                finally:
                    with_pool.close()
                summary = summarize(rows)
                report["policies"][name]["cells"][f"{motion}/{scenario}"] = {
                    "summary": summary, "episodes": rows,
                }
                report["elapsed_seconds"] = time.perf_counter() - started
                write_json(args.output, report)
                print(name, motion, scenario, {k: summary[k] for k in (
                    "episode_return", "success", "collision_rate", "tracked_fraction",
                    "mean_bearing_error_deg",
                )}, flush=True)
    report["complete"] = True
    write_json(args.output, report)


if __name__ == "__main__":
    main()
