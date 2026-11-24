"""2D reference episodes for the v2 G1 scope in Gazebo (#140).

Runs a v2 artifact in the v1 2D world (``FlyTrackEnv``, the world the Gazebo world node mirrors)
on the same per-episode environment seeds the Gazebo orchestrator uses
(``evaluation_environment_seed``: base + training seed * 100,000 + episode index), one actor call
per control step through the ROS node's ``V2PolicyAdapter``. So each Gazebo episode has a 2D
partner with the same scenario draw, actor and deterministic action (the distribution's mode).
The wind input is zero with valid[3] = 0, as in the v1compat world (#140 7001508986).

Usage (repository root):
  PYTHONPATH=python python scripts/v2/gazebo_g1_reference.py --artifact DIR --variant VO2 \
      --seed 100 --scenarios normal occlusion --target-motion curved --episodes 6 --output OUT
OUT/{variant}/seed_{seed}/{scenario}/episodes.jsonl holds one row per episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "flytrack_policy"))

from flytrack_core.contracts import file_sha256  # noqa: E402
from flytrack_core.metrics import EpisodeMetrics  # noqa: E402
from flytrack_policy.artifact import load_deployment_policy  # noqa: E402
from flytrack_train.batches import tensor_observation  # noqa: E402
from flytrack_train.config import make_env, read_config  # noqa: E402
from flytrack_train.evaluation import evaluation_environment_seed  # noqa: E402

CONFIGS = {"curved": "configs/train/evaluate.yaml", "wander": "configs/train/evaluate_wander.yaml"}


def run(args) -> list[dict]:
    config = read_config(ROOT / CONFIGS[args.target_motion])
    options = config.get("evaluation", {})
    if config.get("env", {}).get("target_motion", "curved") != args.target_motion:
        raise ValueError(f"{CONFIGS[args.target_motion]} is not a {args.target_motion} config")
    policy, contract, card, spec = load_deployment_policy(args.artifact, device="cpu")
    if spec != "v2" or card["variant"] != args.variant:
        raise ValueError(f"{args.artifact} is not a v2 {args.variant} artifact")
    if int(card.get("provenance", {}).get("train_seed", -1)) != args.seed:
        raise ValueError(f"{args.artifact} was not trained with seed {args.seed}")
    policy.eval()
    torch.set_num_threads(args.threads)
    card_hash = file_sha256(Path(args.artifact) / "model_card.json")
    bearing = options.get("success_bearing_deg", 45.0)
    rows = []
    for scenario in args.scenarios:
        env = make_env(config, contract, scenario=scenario)
        try:
            for episode in range(args.episodes):
                environment_seed = evaluation_environment_seed(
                    args.seed, episode, args.environment_seed
                )
                observation, _ = env.reset(seed=environment_seed)
                state = policy.initial_state(1, "cpu")
                metrics = EpisodeMetrics(
                    contract.control_dt,
                    float(options.get("success_distance_m", 2)),
                    float(options.get("success_time_fraction", 0.8)),
                    None if bearing is None else float(bearing),
                )
                with torch.no_grad():
                    while True:
                        action, state, _ = policy(tensor_observation([observation], "cpu"), state)
                        physical = action[0].numpy()
                        observation, _, terminated, truncated, info = env.step(physical)
                        metrics.append(
                            info["target_distance_m"],
                            info["target_bearing_rad"],
                            info.get("applied_action", physical)[1],
                            info["collision"],
                            info.get("occluded", False),
                            info.get("time_s"),
                            None,
                        )
                        if terminated or truncated:
                            break
                rows.append(
                    {
                        "variant": args.variant,
                        "seed": args.seed,
                        "scenario": scenario,
                        "target_motion": args.target_motion,
                        "episode": episode,
                        "environment_seed": environment_seed,
                        "model_card_sha256": card_hash,
                        **metrics.result(),
                    }
                )
                print(
                    f"{args.variant} s{args.seed} {scenario} {args.target_motion} ep{episode}: "
                    f"success {rows[-1].get('success')} "
                    f"collision_rate {rows[-1].get('collision_rate')}",
                    flush=True,
                )
        finally:
            env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scenarios", nargs="+", default=["normal", "occlusion"])
    parser.add_argument("--target-motion", choices=tuple(CONFIGS), default="curved")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--environment-seed", type=int, default=1700000000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = run(args)
    for scenario in args.scenarios:
        cell = args.output / args.variant / f"seed_{args.seed}" / scenario
        cell.mkdir(parents=True, exist_ok=True)
        path = cell / "episodes.jsonl"
        if path.exists():
            raise FileExistsError(f"Keep {path}; choose a new output directory")
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows if r["scenario"] == scenario),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
