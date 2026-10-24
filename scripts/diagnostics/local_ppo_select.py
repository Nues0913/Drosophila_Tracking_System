"""Apply a predeclared development selection rule (study_plan.yaml).

Reads only development evidence: paired comparisons for PPO seeds 0-2,
parameter/preservation checks and both DN floors. Never reads confirmation data.
Round 1 used the original pipelines; round 2 uses local_ppo_confirm.py outputs
and also screens every evaluated cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from flytrack_train.config import write_json

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/local_ppo_bc"
sys.path.insert(0, str(ROOT / "scripts"))
from dn_health import health_gate  # noqa: E402

SEEDS = [0, 1, 2]
POSITIVE_EVERY_SEED = ("episode_return", "tracked_fraction", "full_horizon_tracked_fraction")


def assess(recipe, development_round=1):
    name = recipe if development_round == 1 else f"dev{development_round}_{recipe}"
    comparison = json.loads((STUDY / f"development_comparison_{name}.json").read_text())
    entry = comparison["recipes"][recipe]
    if sorted(entry["ppo_seeds"]) != SEEDS or entry["confirmation_episode_sets_verified"]:
        raise ValueError(f"{recipe}: not a complete development-only comparison")
    cell = entry["cells"]["curved/normal"]
    order = np.argsort(entry["ppo_seeds"])
    deltas = {k: [cell[k]["per_seed_delta"][i] for i in order] for k in cell}
    checks = {}
    for seed in SEEDS:
        config = yaml.safe_load(
            (STUDY / "configs/ppo" / recipe / f"seed_{seed}.yaml").read_text()
        )
        output = ROOT / config["output"]
        health = json.loads((output / "dn_health.json").read_text())
        evidence = "parameter_verification.json" if development_round == 1 else (
            "preservation_audit.json")
        checks[f"seed_{seed}_parameters"] = json.loads((output / evidence).read_text())["passed"]
        checks[f"seed_{seed}_ppo_dn"] = health_gate(health, "ppo")["passed"]
        checks[f"seed_{seed}_bc_dn_floor"] = health_gate(health, "bc")["passed"]
    for key in POSITIVE_EVERY_SEED:
        checks[f"{key}_positive_every_seed"] = all(v > 0 for v in deltas[key])
    checks["mean_success_delta_at_least_-0.05"] = float(np.mean(deltas["success"])) >= -0.05
    checks["mean_collision_delta_at_most_0.05"] = float(np.mean(deltas["collision_rate"])) <= 0.05
    if development_round >= 2:
        for cell_name, metrics in entry["cells"].items():
            checks[f"{cell_name}/mean_success_delta_at_least_-0.05"] = (
                metrics["success"]["mean_delta"] >= -0.05)
            checks[f"{cell_name}/mean_collision_delta_at_most_0.05"] = (
                metrics["collision_rate"]["mean_delta"] <= 0.05)
    return {
        "eligible": all(checks.values()), "checks": checks,
        "per_seed_delta": {k: deltas[k] for k in (*POSITIVE_EVERY_SEED, "success",
                                                   "collision_rate", "mean_bearing_error_deg")},
        "mean_delta": {k: float(np.mean(v)) for k, v in deltas.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    plan = yaml.safe_load((STUDY / "study_plan.yaml").read_text())
    if args.round == 1:
        rule, recipes = plan["development_selection"], sorted(plan["candidate_recipes"])
    else:
        rule = plan[f"development_round_{args.round}"]["selection"]
        recipes = sorted(plan[f"development_round_{args.round}"]["recipes"])
    results = {recipe: assess(recipe, args.round) for recipe in recipes}
    eligible = [r for r in recipes if results[r]["eligible"]]
    selected = max(eligible, key=lambda r: (
        results[r]["mean_delta"]["full_horizon_tracked_fraction"],
        results[r]["mean_delta"]["episode_return"],
    )) if eligible else None
    report = {
        "round": args.round, "rule": rule, "recipes": results, "eligible": eligible,
        "selected": selected,
        "scope": "Development episodes and PPO seeds 0-2 only; not confirmation evidence.",
    }
    suffix = "" if args.round == 1 else f"_round_{args.round}"
    write_json(STUDY / f"development_selection{suffix}.json", report)
    print(json.dumps({"eligible": eligible, "selected": selected,
                      "checks": {r: results[r]["checks"] for r in recipes}}, indent=2))


if __name__ == "__main__":
    main()
