"""Evaluate formal-v1 actors in the v1compat world and compare with formal-v1 (#139).

The protocol's v1_actor_consistency audit: every formal-v1 actor runs the four scenarios on the
batched v1compat world with v1's own evaluation settings (configs/train/evaluate*.yaml env,
blackout occlusion as in its training, v1 reward), 100 episodes per scenario. Per episode it
also writes the #161 A3 speed-distance slope and the E1 occlusion phases (secondary.jsonl). Its
seed-level success is then compared, seed by seed, with the formal-v1 seed means. The worlds use
hash draws, so episodes are not v1's episodes: agreement is judged on seed means and intervals,
never episode by episode.

Usage: python scripts/v2/audit_v1_actors.py --actors DIR [DIR ...] --formal-seed-means FILE
       --motion curved --output DIR
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from flytrack_core.model import load_policy_artifact
from flytrack_v2.contracts import SensorContractV2
from flytrack_v2.eval.runner import V1Actor, run_episodes, to_records
from flytrack_v2.metrics import validate_episode_record
from flytrack_v2.world.v1compat import V1CompatWorld

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("normal", "occlusion", "noise", "conflict")
EPISODE_SEED_BASE = 1_800_000_000  # outside training (< 1e9) and v1's evaluation (1.7e9)
V1_EVALUATION_SEED_BASE = 1_700_000_000


def episode_seeds(scheme: str, train_seed: int, episodes: int, scenarios: int) -> list[int]:
    """Per row (scenario-major), the episode seed of episode e.

    - per_seed (default): EPISODE_SEED_BASE + 100000 * train_seed + e, as v1 did, so each
      training seed sees its own episodes and seed means carry the episode sampling variance;
    - v1: v1's formal evaluation seeds (flytrack_train.evaluation.evaluation_environment_seed);
      with --draws v1 the episodes are v1's own, so records compare episode by episode.
    The same e gives the same seed in every scenario, as in v1.
    """
    base = V1_EVALUATION_SEED_BASE if scheme == "v1" else EPISODE_SEED_BASE
    one = [(base + 100000 * int(train_seed) + e) % (2**31 - 1) for e in range(episodes)]
    return one * scenarios


def world_for(
    motion: str,
    episodes: int,
    device: str,
    draws: str = "hash",
    scenarios: tuple[str, ...] = SCENARIOS,
) -> tuple[V1CompatWorld, list[dict], int]:
    name = "evaluate.yaml" if motion == "curved" else "evaluate_wander.yaml"
    config = yaml.safe_load((ROOT / "configs/train" / name).read_text(encoding="utf-8"))
    scenario_configs = {
        scenario: yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
        for scenario, path in config["scenario_configs"].items()
    }
    rows = [{"scenario": s, "noise_scale": 0.75} for s in scenarios for _ in range(episodes)]
    world = V1CompatWorld(
        SensorContractV2.from_json(ROOT / "configs/v2/sensor_contract.json"),
        num_envs=len(rows),
        scenarios=[row["scenario"] for row in rows],
        options=config["env"],
        scenario_configs=scenario_configs,
        device=device,
        reward="v1",
        occlusion_view="blackout",
        draws=draws,
    )
    steps = int(np.ceil(config["env"]["episode_seconds"] / world.contract.control_dt))
    return world, rows, steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--actors",
        nargs="+",
        type=Path,
        required=True,
        help="runs/bc directories holding {variant}/seed_{n}/actor",
    )
    parser.add_argument("--formal-seed-means", type=Path, required=True)
    parser.add_argument("--motion", choices=("curved", "wander"), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--draws", choices=("hash", "v1"), default="hash")
    parser.add_argument("--seed-scheme", choices=("per_seed", "v1"), default="per_seed")
    parser.add_argument("--variants", nargs="*", help="only these v1 variants (default: all)")
    parser.add_argument("--scenarios", nargs="*", choices=SCENARIOS, help="default: all four")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    args.output.mkdir(parents=True, exist_ok=True)
    scenarios = tuple(args.scenarios or SCENARIOS)
    world, rows, episode_steps = world_for(
        args.motion, args.episodes, args.device, args.draws, scenarios
    )
    max_steps = episode_steps + 5
    formal = {
        (r["variant"], r["scenario"], int(r["seed"])): r["success"]
        for r in json.loads(args.formal_seed_means.read_text(encoding="utf-8"))
    }
    actors = sorted(p for root in args.actors for p in root.glob("*/seed_*/actor"))
    if args.variants:
        actors = [p for p in actors if p.parent.parent.name in args.variants]
    comparison = []
    invalid = []
    secondary = (args.output / "secondary.jsonl").open("w", encoding="utf-8")
    with (args.output / "episodes.jsonl").open("w", encoding="utf-8") as stream, secondary:
        for path in actors:
            variant, seed = path.parent.parent.name, int(path.parent.name.split("_")[1])
            started = time.perf_counter()
            policy, _, card = load_policy_artifact(path, device=args.device)
            policy.eval()
            seeds = episode_seeds(args.seed_scheme, seed, args.episodes, len(scenarios))
            result = run_episodes(world, V1Actor(policy), seeds, max_steps=max_steps)
            artifact = card["files"]["weights.pt"]
            records = to_records(
                result,
                rows,
                run_id=f"v1audit-{args.motion}-{args.draws}-{args.seed_scheme}",
                variant=f"v1:{variant}",
                train_seed=seed,
                world="v1compat",
                motion=args.motion,
                artifact_sha256=artifact,
                commit=commit,
            )
            for i, record in enumerate(records):
                try:
                    validate_episode_record(record)
                except ValueError as error:  # keep the night's run going; report at the end
                    invalid.append(
                        {"variant": variant, "seed": seed, "row": i, "error": str(error)}
                    )
                stream.write(json.dumps(record) + "\n")
                extra = {key: values[i] for key, values in result.speed_distance.items()}
                extra.update({key: values[i] for key, values in result.occlusion_phases.items()})
                extra.update(
                    variant=variant,
                    seed=seed,
                    scenario=record["scenario"],
                    episode_seed=record["episode_seed"],
                )
                secondary.write(json.dumps(extra) + "\n")
            for scenario in scenarios:
                ours = float(np.mean([r["success"] for r in records if r["scenario"] == scenario]))
                comparison.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "scenario": scenario,
                        "v1compat": ours,
                        "formal": formal.get((variant, scenario, seed)),
                        "seconds": time.perf_counter() - started,
                    }
                )
            print(
                f"{variant} seed {seed}: "
                + ", ".join(
                    f"{c['scenario']} {c['v1compat']:.2f}/{c['formal']}"
                    for c in comparison[-len(scenarios) :]
                ),
                flush=True,
            )
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, indent=1) + "\n", encoding="utf-8"
    )
    (args.output / "invalid_records.json").write_text(json.dumps(invalid, indent=1) + "\n")
    print(f"{len(invalid)} records failed validate_episode_record", flush=True)


if __name__ == "__main__":
    main()
