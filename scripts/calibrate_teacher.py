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
    parser.add_argument("--seed-start", type=int, default=9_000_000)
    parser.add_argument("--output", type=Path, default=Path("runs/calibration/teacher.json"))
    args = parser.parse_args()
    if args.episodes < 1 or args.seed_start < 0:
        parser.error("episodes must be positive and seed-start nonnegative")
    config = read_config(args.config)
    contract = contract_from_config(config)
    env = make_env(config, contract, scenario="normal")
    rows = []
    started = time.perf_counter()
    try:
        for episode in range(args.episodes):
            seed = args.seed_start + episode
            _, info = env.reset(seed=seed)
            metrics = EpisodeMetrics(contract.control_dt)
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
            rows.append({"seed": seed, **metrics.result()})
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
