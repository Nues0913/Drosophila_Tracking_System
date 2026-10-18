"""Measure the privileged teacher on calibration seeds separate from formal evaluation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from flytrack_core.metrics import EpisodeMetrics
from flytrack_train.config import contract_from_config, make_env, read_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train/evaluate.yaml"))
    parser.add_argument("--episodes", type=int, default=20)
    # Diagnostic range [1.6e9, 1.7e9): disjoint from training [0, 1e9), decoder
    # validation [1.5e9, 1.6e9) and formal evaluation [1.7e9, ...).
    parser.add_argument("--seed-start", type=int, default=1_650_000_000)
    parser.add_argument("--output", type=Path, default=Path("runs/calibration/teacher.json"))
    args = parser.parse_args()
    if args.episodes < 1 or not 1_600_000_000 <= args.seed_start < 1_700_000_000 - args.episodes:
        parser.error(
            "episodes must be positive and seeds inside the diagnostic range [1.6e9,1.7e9)"
        )
    config = read_config(args.config)
    contract = contract_from_config(config)
    env = make_env(config, contract, scenario="normal")
    rows = []
    started = time.perf_counter()
    try:
        for episode in range(args.episodes):
            seed = args.seed_start + episode
            _, info = env.reset(seed=seed)
            metrics = EpisodeMetrics.from_env_options(contract.control_dt, config.get("env", {}))
            while True:
                _, _, terminated, truncated, info = env.step(info["teacher_action"])
                metrics.append(
                    info["target_distance_m"],
                    info["target_bearing_rad"],
                    float(env.previous_action[1]),
                    bool(info["collision"]),
                    timestamp=env.time_s,
                )
                if terminated or truncated:
                    break
            trajectory = env.trajectory.config
            rows.append(
                {
                    "seed": seed,
                    "target_motion": trajectory.motion,
                    "direction": trajectory.direction,
                    "radius_m": trajectory.radius_m,
                    **metrics.result(),
                }
            )
            report = {
                "purpose": "calibration only; privileged teacher, never a sensory baseline",
                "scenario": "normal",
                "episodes_requested": args.episodes,
                "episodes_completed": len(rows),
                "complete": len(rows) == args.episodes,
                "elapsed_s": time.perf_counter() - started,
                "rows": rows,
                "mean_success": float(np.mean([row["success"] for row in rows])),
                "mean_bearing_error_deg": float(
                    np.mean([row["mean_bearing_error_deg"] for row in rows])
                ),
                "bearing_error_deg_range": [
                    float(np.min([row["mean_bearing_error_deg"] for row in rows])),
                    float(np.max([row["mean_bearing_error_deg"] for row in rows])),
                ],
                "distinct_layouts": len({(row["direction"], row["radius_m"]) for row in rows}),
                "config": config,
            }
            write_json(args.output, report)
            print(
                f"Teacher calibration {len(rows)}/{args.episodes}: "
                f"success={report['mean_success']:.3f}, "
                f"bearing={report['mean_bearing_error_deg']:.3f} deg",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
