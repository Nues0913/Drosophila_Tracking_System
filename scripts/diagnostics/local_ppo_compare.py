"""Compare paired BC/PPO episodes, using PPO seeds as the inference unit."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from flytrack_train.config import write_json

METRICS = (
    "episode_return", "success", "collision_rate", "tracked_fraction",
    "full_horizon_tracked_fraction",
    "mean_bearing_error_deg", "mean_distance_m", "smoothness_rad_s", "episode_seconds",
)


def merge_cells(existing, incoming):
    if existing["metadata"] != incoming["metadata"]:
        raise ValueError("Artifact metadata differs between split evaluation reports")
    for name, cell in incoming["cells"].items():
        if name in existing["cells"] and existing["cells"][name] != cell:
            raise ValueError(f"Conflicting repeated evaluation cell: {name}")
        existing["cells"][name] = copy.deepcopy(cell)


def paired_delta(baseline, candidate, horizon_seconds=60.0):
    left = {row["environment_seed"]: row for row in baseline}
    right = {row["environment_seed"]: row for row in candidate}
    if len(left) != len(baseline) or len(right) != len(candidate):
        raise ValueError("Duplicate evaluation episode seeds")
    if left.keys() != right.keys():
        raise ValueError("BC/PPO episode seeds are not fully paired")
    def value(row, key):
        if key == "full_horizon_tracked_fraction":
            # An early collision cannot improve tracking by deleting its bad tail.
            return row["tracked_fraction"] * row["episode_seconds"] / horizon_seconds
        return row[key]

    return {metric: float(np.mean([
        value(right[s], metric) - value(left[s], metric) for s in left
    ])) for metric in METRICS}


def seed_summary(deltas, draws=10000):
    matrix = np.array([[row[k] for k in METRICS] for row in deltas])
    rng = np.random.default_rng(7301)
    # Never manufacture a seed interval from many episodes of one trained model.
    bootstrap = matrix[rng.integers(len(matrix), size=(draws, len(matrix)))].mean(1)
    result = {}
    for column, key in enumerate(METRICS):
        result[key] = {
            "mean_delta": float(matrix[:, column].mean()),
            "per_seed_delta": matrix[:, column].tolist(),
            "nominal_seed_bootstrap_95": np.quantile(
                bootstrap[:, column], [0.025, 0.975]
            ).tolist() if len(matrix) >= 3 else None,
        }
    return result


def numerical_gates(cells, seed_count, confirmation_verified=False):
    """Predeclared numeric criteria; DN and parameter checks remain separate."""
    primary = cells.get("curved/normal")
    expected = ("curved/normal", "wander/normal", "curved/occlusion",
                "curved/noise", "curved/conflict")
    checks = {
        "five_ppo_seeds": seed_count >= 5,
        "untouched_confirmation_episode_sets": confirmation_verified,
        "all_preservation_cells": all(name in cells for name in expected),
    }
    if primary is None:
        return {"passed": False, "checks": checks, "unverified": ["primary cell"]}
    for name in ("episode_return", "tracked_fraction", "full_horizon_tracked_fraction"):
        metric = primary[name]
        interval = metric["nominal_seed_bootstrap_95"]
        checks[f"{name}_improves_every_seed"] = all(x > 0 for x in metric["per_seed_delta"])
        checks[f"{name}_interval_positive"] = interval is not None and interval[0] > 0
    for name, cell in cells.items():
        for metric, index, threshold, lower in (
            ("success", 0, -0.05, True),
            ("collision_rate", 1, 0.05, False),
            ("mean_bearing_error_deg", 1, 5.0, False),
        ):
            interval = cell[metric]["nominal_seed_bootstrap_95"]
            checks[f"{name}/{metric}_preserved"] = interval is not None and (
                interval[index] >= threshold if lower else interval[index] <= threshold
            )
    return {"passed": all(checks.values()), "checks": checks,
            "unverified": ["DN health", "frozen parameter invariants", "additional BC starts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if not args.output.resolve().is_relative_to(root):
        parser.error("Output must stay in this workspace")
    baseline = None
    policies, config_sha, implementation = {}, None, None
    for path in args.reports:
        report = json.loads(path.read_text())
        if not report.get("complete"):
            raise ValueError(f"Incomplete evaluation: {path}")
        if config_sha is not None and config_sha != report["config_sha256"]:
            raise ValueError("Evaluation configurations differ")
        if implementation is not None and implementation != report.get("implementation"):
            raise ValueError("Evaluation implementations differ")
        if not report.get("implementation"):
            raise ValueError("Evaluation implementation hashes are missing")
        implementation = report["implementation"]
        config_sha = report["config_sha256"]
        horizon_seconds = float(report["config"].get("env", {}).get("episode_seconds", 60.0))
        for name, policy in report["policies"].items():
            if policy["metadata"].get("control"):
                continue
            identity = policy["metadata"]["model_card_sha256"]
            if name == args.baseline:
                if baseline and baseline["metadata"]["model_card_sha256"] != identity:
                    raise ValueError("Baseline artifact changed between evaluations")
                if baseline:
                    merge_cells(baseline, policy)
                else:
                    baseline = copy.deepcopy(policy)
            elif identity in policies:
                previous_name, previous_policy = policies[identity]
                if previous_name != name:
                    raise ValueError("The same artifact has inconsistent names")
                merge_cells(previous_policy, policy)
            else:
                policies[identity] = (name, copy.deepcopy(policy))
    if baseline is None:
        raise ValueError("Baseline not found in supplied evaluations")
    groups = defaultdict(list)
    for name, policy in policies.values():
        metadata = policy["metadata"]
        settings = metadata.get("local_experiment", {})
        if not settings.get("name"):
            raise ValueError(f"Candidate recipe name is missing: {name}")
        if metadata.get("init_artifact") != args.baseline:
            raise ValueError(f"Candidate uses another BC baseline: {name}")
        groups[settings["name"]].append((name, policy))
    result = {"baseline": args.baseline, "config_sha256": config_sha,
              "implementation": implementation, "recipes": {}}
    for recipe, entries in groups.items():
        seeds = [policy["metadata"]["seed"] for _, policy in entries]
        if None in seeds or len(seeds) != len(set(seeds)):
            raise ValueError(f"Missing or duplicate PPO training seeds: {recipe}")
        cells = {}
        for cell, base_cell in baseline["cells"].items():
            differences = [paired_delta(base_cell["episodes"], p["cells"][cell]["episodes"],
                                        horizon_seconds)
                           for _, p in entries]
            cells[cell] = seed_summary(differences)
        confirmation_seeds = list(range(1_630_000_000, 1_630_000_100))
        confirmation_verified = all(
            sorted(row["environment_seed"] for row in base_cell["episodes"])
            == confirmation_seeds
            for base_cell in baseline["cells"].values()
        )
        result["recipes"][recipe] = {
            "ppo_seeds": seeds, "independent_bc_starts": 1, "cells": cells,
            "claim_scope": "PPO-seed reproducibility conditional on this single BC start",
            "sufficient_seed_count_for_confirmation": len(seeds) >= 5,
            "confirmation_episode_sets_verified": confirmation_verified,
            "numerical_gates": numerical_gates(cells, len(seeds), confirmation_verified),
        }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
