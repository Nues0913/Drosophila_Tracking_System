"""Seed-level Student-t gates and the combined decision (study_plan confirmation_extensions).

Reads paired comparisons produced by local_ppo_compare.py; the PPO training seed
is the unit. Original bootstrap gates are copied through unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from flytrack_train.config import write_json
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = "curved/normal"
IMPROVEMENT = ("episode_return", "tracked_fraction", "full_horizon_tracked_fraction")
# metric -> (bound side, threshold): lower bound >= threshold, or upper bound <= threshold.
NONINFERIORITY = {
    "success": ("lower", -0.05),
    "collision_rate": ("upper", 0.05),
    "mean_bearing_error_deg": ("upper", 5.0),
}


def t_bound(values, side, confidence=0.95):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    half = stats.t.ppf(confidence, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - half if side == "lower" else values.mean() + half)


def t_gates(cells):
    checks, bounds = {}, {}
    for metric in IMPROVEMENT:
        bound = t_bound(cells[PRIMARY][metric]["per_seed_delta"], "lower")
        bounds[f"{PRIMARY}/{metric}"] = {"lower_95": bound}
        checks[f"{PRIMARY}/{metric}_t_lower_above_zero"] = bound is not None and bound > 0
    for cell, metrics in cells.items():
        for metric, (side, threshold) in NONINFERIORITY.items():
            bound = t_bound(metrics[metric]["per_seed_delta"], side)
            bounds[f"{cell}/{metric}"] = {f"{side}_95": bound, "threshold": threshold}
            checks[f"{cell}/{metric}_t_noninferior"] = bound is not None and (
                bound >= threshold if side == "lower" else bound <= threshold
            )
    return {"passed": all(checks.values()), "checks": checks, "bounds": bounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--comparison", type=Path, required=True, help="All confirmation seeds")
    parser.add_argument("--declared", type=Path, help="Declared seeds 0-4 only")
    parser.add_argument("--state", type=Path, required=True, help="confirm_<tag>.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-seeds", type=int, default=10)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("Output must stay in this workspace")
    entry = json.loads(args.comparison.read_text())["recipes"][args.recipe]
    state = json.loads(args.state.read_text())
    if state.get("phase") != "complete":
        raise ValueError("Confirmation controller has not completed")
    audits = state["audits"]
    result = {
        "recipe": args.recipe, "ppo_seeds": sorted(entry["ppo_seeds"]),
        "original_gates_all_seeds": entry["numerical_gates"],
        "t_gates_all_seeds": t_gates(entry["cells"]),
        "audits_passed": all(a["passed"] for a in audits.values())
        and len(audits) == len(entry["ppo_seeds"]),
        "confirmation_episode_sets_verified": entry["confirmation_episode_sets_verified"],
        "per_seed_primary": {m: entry["cells"][PRIMARY][m]["per_seed_delta"]
                             for m in (*IMPROVEMENT, "success", "collision_rate",
                                       "mean_bearing_error_deg")},
    }
    checks = {
        "enough_seeds": len(entry["ppo_seeds"]) >= args.minimum_seeds,
        "original_gates_all_seeds": entry["numerical_gates"]["passed"],
        "t_gates_all_seeds": result["t_gates_all_seeds"]["passed"],
        "preservation_audits": result["audits_passed"],
        "untouched_confirmation_episodes": entry["confirmation_episode_sets_verified"],
    }
    if args.declared:
        declared = json.loads(args.declared.read_text())["recipes"][args.recipe]
        if sorted(declared["ppo_seeds"]) != [0, 1, 2, 3, 4]:
            raise ValueError("Declared analysis must use PPO seeds 0-4")
        result["original_gates_declared_seeds"] = declared["numerical_gates"]
        result["t_gates_declared_seeds"] = t_gates(declared["cells"])
        checks["original_gates_declared_seeds"] = declared["numerical_gates"]["passed"]
    result["decision_checks"] = checks
    result["reproducible_improvement_claimed"] = all(checks.values())
    result["scope"] = "One BC start; combine with other BC starts before generalizing."
    write_json(args.output, result)
    print(json.dumps({"claimed": result["reproducible_improvement_claimed"], "checks": checks},
                     indent=2))


if __name__ == "__main__":
    main()
