#!/usr/bin/env python3
"""Run the complete generated six-variant, five-seed stage matrix sequentially."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from flytrack_core.contracts import file_sha256
from flytrack_train.config import read_config, write_json
from flytrack_train.recovery import Recovery

VARIANTS = ("V", "O", "VO", "VO-late", "GRU", "RS")


def stage_budget(config, command):
    train = config["train"]
    if command == "bc":
        return int(train["teacher_steps"]) + int(train["dagger_rounds"]) * int(
            train["dagger_steps_per_round"]
        )
    return int(train["total_steps"])


def completed(config, command, device):
    output = Path(config["output"])
    card_path = output / "actor/model_card.json"
    if not card_path.is_file():
        return False
    # Completed stages must still match code, configuration, source data and runtime.
    Recovery(output, {**config, "resume": True}, command, int(config["seed"]), device=device)
    card = json.loads(card_path.read_text())
    if int(card.get("metadata", {}).get("environment_steps", -1)) != stage_budget(config, command):
        raise ValueError(f"Completed stage has the wrong interaction budget: {output}")
    for name, digest in card["files"].items():
        if file_sha256(output / "actor" / name) != digest:
            raise ValueError(f"Completed stage artifact checksum differs: {output}/{name}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=Path("runs/experiment_configs"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--algorithm", choices=("ppo", "es"), default="ppo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds or min(args.seeds) < 0:
        parser.error("Seeds must be distinct nonnegative integers")
    plan = []
    for variant in args.variants:
        stages = (
            ["pretrain_vision"]
            if variant == "V"
            else ["pretrain_odor"]
            if variant == "O"
            else ["pretrain_vision", "pretrain_odor"]
        )
        for seed in args.seeds:
            for stage in [*stages, "bc", args.algorithm]:
                path = args.configs / variant / f"seed_{seed}" / f"{stage}.yaml"
                config = read_config(path)
                if config["variant"] != variant or int(config["seed"]) != seed:
                    raise ValueError(f"Generated configuration has the wrong variant/seed: {path}")
                command = "pretrain" if stage.startswith("pretrain") else stage
                plan.append(
                    {
                        "path": str(path),
                        "stage": stage,
                        "command": command,
                        "variant": variant,
                        "seed": seed,
                        "output": config["output"],
                        "environment_steps": stage_budget(config, command),
                    }
                )
    manifest = {
        "started_unix_s": time.time(),
        "device": str(device),
        "stages": plan,
        "total_planned_environment_steps": sum(row["environment_steps"] for row in plan),
        "selected_variants": args.variants,
        "selected_seeds": args.seeds,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    write_json(args.configs / "execution_plan.json", manifest)
    environment = {**os.environ, "PYTHON_DOTENV_DISABLED": "1", "PYTHONUNBUFFERED": "1"}
    for index, row in enumerate(plan):
        config = read_config(row["path"])
        destination = Path(row["output"])
        if completed(config, row["command"], device):
            status = "verified_complete"
        else:
            destination.mkdir(parents=True, exist_ok=True)
            command = [
                args.python,
                "-m",
                "flytrack_train",
                row["command"],
                "--config",
                row["path"],
                "--device",
                str(device),
            ]
            if (destination / "training_checkpoint.pt").is_file():
                command.append("--resume")
            row["status"] = "running"
            write_json(args.configs / "execution_plan.json", manifest)
            with (destination / "execution.log").open("a", buffering=1) as log:
                print(
                    f"[{index + 1}/{len(plan)}] {row['variant']} seed={row['seed']} "
                    f"{row['stage']}: {command}",
                    flush=True,
                )
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False
                )
            if result.returncode:
                row.update(status="failed", exit_code=result.returncode)
                write_json(args.configs / "execution_plan.json", manifest)
                raise SystemExit(f"Stage failed; inspect {destination / 'execution.log'}")
            if not completed(config, row["command"], device):
                raise RuntimeError(f"Stage exited without a verified actor artifact: {destination}")
            status = "complete"
        if row["command"] == "pretrain":
            validation = destination / "validation"
            metrics = validation / "metrics.json"
            if not metrics.is_file() or not json.loads(metrics.read_text()).get("passed"):
                command = [
                    args.python,
                    "-m",
                    "flytrack_train",
                    "validate-pretrain",
                    "--config",
                    row["path"],
                    "--artifact",
                    str(destination / "actor"),
                    "--checkpoint",
                    str(destination / "pretraining_checkpoint.pt"),
                    "--output",
                    str(validation),
                    "--device",
                    str(device),
                ]
                with (destination / "validation.log").open("a", buffering=1) as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False
                    )
                if result.returncode:
                    row.update(status="heldout_validation_failed", exit_code=result.returncode)
                    write_json(args.configs / "execution_plan.json", manifest)
                    raise SystemExit(f"Held-out decoding gate failed; inspect {validation}")
        row.update(status=status, finished_unix_s=time.time())
        write_json(args.configs / "execution_plan.json", manifest)
    manifest["completed_unix_s"] = time.time()
    write_json(args.configs / "execution_plan.json", manifest)


if __name__ == "__main__":
    main()
