"""Evaluate declared sensory/cell lesions and incoming convergence-current mass.

This is a separate post-training analysis, not a replacement for formal model
comparisons. It requires a genuine trained artifact and has not been run.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from flytrack_core.metrics import EpisodeMetrics
from flytrack_core.model import load_policy_artifact
from flytrack_sim import FlyTrackEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/lesion_analysis"))
    parser.add_argument(
        "--scenario", choices=("normal", "occlusion", "noise", "conflict"), default="occlusion"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    policy, contract, card = load_policy_artifact(args.artifact, device=args.device)
    if not hasattr(policy, "set_lesion"):
        raise ValueError("Policy does not expose the declared lesion analysis interface")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for lesion in ("none", "vision", "odor"):
        policy.set_lesion(lesion)
        env = FlyTrackEnv(
            contract=contract,
            scenario=args.scenario,
            episode_seconds=60,
            domain_randomization=False,
        )
        try:
            for episode in range(args.episodes):
                observation, _ = env.reset(seed=args.seed + episode)
                state = policy.initial_state(1, args.device)
                metrics = EpisodeMetrics(contract.control_dt)
                masses = {"visual_exclusive": 0.0, "odor_exclusive": 0.0, "shared": 0.0}
                analyzed_steps = 0
                with torch.no_grad():
                    while True:
                        tensors = {
                            key: torch.as_tensor(value, device=args.device).unsqueeze(0)
                            for key, value in observation.items()
                        }
                        action, state, activity = policy(tensors, state)
                        if card["variant"] != "GRU":
                            contribution = policy.convergence_current_contributions(activity)
                            for name, value in contribution.items():
                                masses[name] += float(value.sum().cpu())
                            analyzed_steps += 1
                        observation, _, terminated, truncated, info = env.step(
                            action[0].cpu().numpy()
                        )
                        metrics.append(
                            info["target_distance_m"],
                            info["target_bearing_rad"],
                            float(info["applied_action"][1]),
                            info["collision"],
                            info["occluded"],
                            info["time_s"],
                        )
                        if terminated or truncated:
                            break
                total = sum(masses.values())
                row = {
                    "variant": card["variant"],
                    "lesion": lesion,
                    "scenario": args.scenario,
                    "episode": episode,
                    "environment_seed": args.seed + episode,
                    **metrics.result(),
                    "analyzed_steps": analyzed_steps,
                }
                for name, value in masses.items():
                    row[f"{name}_fraction"] = value / total if total > 0 else None
                rows.append(
                    {
                        key: None if isinstance(value, float) and not np.isfinite(value) else value
                        for key, value in row.items()
                    }
                )
        finally:
            env.close()
    policy.set_lesion("none")
    (args.output / "episodes.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    with (args.output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "analysis_notes.json").write_text(
        json.dumps(
            {
                "artifact": str(args.artifact),
                "model_card": card,
                "contribution_semantics": (
                    "sum of absolute incoming recurrent edge-current mass, not causal attribution"
                ),
                "lesion_semantics": (
                    "complete corresponding central branch silence"
                    if card["variant"] == "VO-late"
                    else "sensory input suppression only"
                    if card["variant"] == "GRU"
                    else (
                        "input suppression plus modality-exclusive cells; "
                        "shared convergence/motor retained"
                    )
                ),
                "n_convergence": len(policy.convergence_node_ids)
                if card["variant"] != "GRU"
                else None,
                "parameter_budget_unchanged": True,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
