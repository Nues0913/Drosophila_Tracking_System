"""Paired odor-lesion evaluation for #4 direction 3 (does the policy need odor?).

Usage: odor_reliance_eval.py OUT.json ARTIFACT [ARTIFACT ...]
Run from the repository root with PYTHONPATH=python. Worlds come from
configs/train/evaluate.yaml (curved target, puff plume, formal success rule).
Each actor runs normal, long occlusion and conflict, each with and without
``set_lesion("odor")``, on the same environment seeds, so every episode pairs.
Long occlusion starts at OCCLUSION_START_S (8) and lasts OCCLUSION_S (12).

Environment: EPISODES (100), SEED_START (1_700_500_000, outside training and the
formal seeds 0-4), LANES (25), WORKERS (8).
The report keeps every episode; ``paired`` holds the #4 Decision statistics:
the sign test on mean |bearing| while occluded and McNemar on success.
"""

import json
import os
import sys
from functools import partial
from math import comb
from pathlib import Path

import numpy as np
import torch
import yaml
from flytrack_core.metrics import EpisodeMetrics
from flytrack_train.config import contract_from_config
from flytrack_train.parallel import EnvPool
from flytrack_train.worlds import make_env
from policy_episode_eval import Actor

LESIONS = ("none", "odor")


def binomial_two_sided(k: int, n: int) -> float:
    """Exact two-sided p for k successes of n fair coin flips (sign test, McNemar)."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def episodes(pool, lanes, act, contract, count, seed_start, env_options):
    rows = []
    for first in range(0, count, len(lanes)):
        slots = lanes[: min(len(lanes), count - first)]
        seeds = [seed_start + first + i for i in range(len(slots))]
        obs, infos = map(
            list, zip(*(pool.reset(s, e) for s, e in zip(slots, seeds, strict=True)), strict=True)
        )
        metrics = [EpisodeMetrics.from_env_options(contract.control_dt, env_options) for _ in slots]
        occluded = [[] for _ in slots]
        live = list(range(len(slots)))
        act.reset(len(slots))
        while live:
            actions = act(obs, infos, live)
            stepped = pool.step({slots[row]: actions[i] for i, row in enumerate(live)})
            still = []
            for i, row in enumerate(live):
                observation, _, terminated, truncated, info = stepped[slots[row]]
                applied = info.get("applied_action", actions[i])
                hidden = bool(info.get("occluded", False))
                metrics[row].append(
                    info["target_distance_m"],
                    info["target_bearing_rad"],
                    applied[1],
                    info["collision"],
                    hidden,
                    info.get("time_s"),
                )
                if hidden:
                    occluded[row].append(abs(float(np.rad2deg(info["target_bearing_rad"]))))
                obs[row], infos[row] = observation, info
                if not (terminated or truncated):
                    still.append(row)
            live = still
        for row, seed in enumerate(seeds):
            result = metrics[row].result()
            result["environment_seed"] = seed
            result["occluded_steps"] = len(occluded[row])
            result["occluded_bearing_error_deg"] = (
                float(np.mean(occluded[row])) if occluded[row] else float("nan")
            )
            rows.append(result)
    return rows


def summarize(rows):
    keys = ("success", "collision_rate", "mean_bearing_error_deg", "occluded_bearing_error_deg")
    return {key: float(np.nanmean([row[key] for row in rows])) for key in keys}


def paired(intact, lesioned):
    """Intact minus odor-lesioned, episode by episode on identical seeds."""
    assert [r["environment_seed"] for r in intact] == [r["environment_seed"] for r in lesioned]
    better = worse = 0
    for a, b in zip(intact, lesioned, strict=True):
        x, y = a["occluded_bearing_error_deg"], b["occluded_bearing_error_deg"]
        if np.isfinite(x) and np.isfinite(y) and x != y:
            better += x < y
            worse += x > y
    only_intact = sum(a["success"] > b["success"] for a, b in zip(intact, lesioned, strict=True))
    only_lesion = sum(a["success"] < b["success"] for a, b in zip(intact, lesioned, strict=True))
    return {
        "occluded_bearing_intact_better": int(better),
        "occluded_bearing_lesion_better": int(worse),
        "occluded_bearing_sign_p": binomial_two_sided(int(better), int(better + worse)),
        "success_only_intact": int(only_intact),
        "success_only_lesion": int(only_lesion),
        "success_mcnemar_p": binomial_two_sided(int(only_intact), int(only_intact + only_lesion)),
    }


def main():
    out, artifacts = Path(sys.argv[1]), sys.argv[2:]
    config = yaml.safe_load(Path("configs/train/evaluate.yaml").read_text(encoding="utf-8"))
    count = int(os.environ.get("EPISODES", 100))
    seed_start = int(os.environ.get("SEED_START", 1_700_500_000))
    lanes = min(count, int(os.environ.get("LANES", 25)))
    occlusion = {
        "occlusion_start_s": float(os.environ.get("OCCLUSION_START_S", 8)),
        "occlusion_duration_s": float(os.environ.get("OCCLUSION_S", 12)),
    }
    worlds = {
        "normal": {"scenario": "normal"},
        "occlusion_long": {"scenario": "occlusion", "scenario_config": occlusion},
        "conflict": {"scenario": "conflict"},
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    contract = contract_from_config(config)
    factories = [
        partial(make_env, config, contract, **w) for w in worlds.values() for _ in range(lanes)
    ]
    pool = EnvPool(factories, workers=int(os.environ.get("WORKERS", 8)))
    report = {
        "schema_version": 1,
        "env": config["env"],
        "worlds": worlds,
        "episodes": count,
        "seed_start": seed_start,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "actors": {},
    }
    try:
        for path in artifacts:
            actor = Actor(path, device)
            if actor.policy.variant == "V":
                raise ValueError("odor lesion needs an odor-sensing variant")
            cells = {}
            for lesion in LESIONS:
                actor.policy.set_lesion(lesion)
                for index, name in enumerate(worlds):
                    slots = list(range(index * lanes, (index + 1) * lanes))
                    rows = episodes(pool, slots, actor, contract, count, seed_start, config["env"])
                    cells[f"{name}/{lesion}"] = {"summary": summarize(rows), "episodes": rows}
                    print(path, name, lesion, cells[f"{name}/{lesion}"]["summary"], flush=True)
            actor.policy.set_lesion("none")
            report["actors"][path] = {
                "cells": cells,
                "paired": {
                    name: paired(
                        cells[f"{name}/none"]["episodes"], cells[f"{name}/odor"]["episodes"]
                    )
                    for name in worlds
                },
            }
            print(path, report["actors"][path]["paired"]["occlusion_long"], flush=True)
    finally:
        pool.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
