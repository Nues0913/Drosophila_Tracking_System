"""Write the three #4 direction-3 BC configs (A/B/C) for one host and seed.

All three start from configs/train/bc.yaml and the host's G2 pretrain actor:
A keeps normal-only BC data; B and C draw each episode 50/50 from normal and a
long occlusion (start 8 s; 12 s for B, 30 s for C) with the #33 odor-steering
teacher (k=8) while occluded. Nothing else differs. Refuses to overwrite.
"""

import argparse
import copy
import json
from pathlib import Path

import yaml

ARMS = {"A": None, "B": 12.0, "C": 30.0}
OCCLUSION_START_S = 8.0
ODOR_TEACHER_GAIN = 8.0


def arm_config(base: dict, occlusion_s: float | None) -> dict:
    config = copy.deepcopy(base)
    if occlusion_s is None:
        return config
    env = config["env"]
    env.pop("scenario", None)
    env["scenario_mix"] = [
        {"scenario": "normal", "weight": 0.5},
        {
            "scenario": "occlusion",
            "weight": 0.5,
            "scenario_config": {
                "occlusion_start_s": OCCLUSION_START_S,
                "occlusion_duration_s": occlusion_s,
            },
        },
    ]
    env["occlusion_odor_steering"] = {"gain": ODOR_TEACHER_GAIN}
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--init-artifact", type=Path, required=True, help="G2 odor-pretrain actor")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base", type=Path, default=Path("configs/train/bc.yaml"))
    args = p.parse_args()
    if not (args.init_artifact / "model_card.json").is_file():
        raise SystemExit(f"{args.init_artifact} is not an actor artifact (no model_card.json)")
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; use a new directory")
    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True)
    plan = {"seed": args.seed, "init_artifact": str(args.init_artifact), "arms": {}}
    for arm, occlusion_s in ARMS.items():
        config = arm_config(base, occlusion_s)
        config.update(
            seed=args.seed,
            variant="VO",
            init_artifact=str(args.init_artifact),
            output=str(args.output / arm / "bc"),
        )
        path = args.output / f"{arm}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        plan["arms"][arm] = {
            "config": str(path),
            "occlusion_s": occlusion_s,
            "bc": f"python -m flytrack_train bc --config {path} --workers 4",
            "actor": str(args.output / arm / "bc" / "actor"),
        }
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
