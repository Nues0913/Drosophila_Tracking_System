"""Continue one predeclared recipe after its externally running seed-0 job.

Never starts seed 0: waits for its completed, validated artifact instead. Owns
subprocesses for later seeds and records every stage so interrupted runs resume.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml
from flytrack_core.model import load_policy_artifact
from flytrack_train.config import write_json

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/local_ppo_bc"
BASELINE = "runs/local_ppo_bc/bc_recovery/seed_0/actor"
RECIPES = ("head_low_noise_anchor", "head_low_noise_unanchored")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed(config):
    output = ROOT / config["output"]
    progress = output / "progress.json"
    if not progress.exists() or json.loads(progress.read_text()).get("phase") != "complete":
        return False
    _, _, card = load_policy_artifact(output / "actor", device="cpu")
    saved = card["metadata"]
    if saved.get("seed") != config["seed"]:
        raise ValueError("Artifact training seed differs from configuration")
    if saved.get("environment_steps") != config["train"]["total_steps"]:
        raise ValueError("Completed artifact budget differs from the registered budget")
    expected = json.loads(json.dumps(config))
    expected.setdefault("local_experiment", {})["runner_sha256"] = digest(
        ROOT / "scripts/diagnostics/local_ppo_train.py"
    )
    actual = {k: v for k, v in saved["config"].items() if k != "resume"}
    if actual != expected:
        raise ValueError("Completed actor configuration/runner differs from this recipe")
    checkpoint = torch.load(
        output / "training_checkpoint.pt", map_location="cpu", weights_only=True
    )
    if checkpoint["identity"]["sources"]["init_artifact"] != digest(
        ROOT / BASELINE / "model_card.json"
    ):
        raise ValueError("PPO checkpoint used a different BC parent")
    return True


def verify_parameters(config):
    base, _, _ = load_policy_artifact(ROOT / BASELINE, device="cpu")
    actor_path = ROOT / config["output"] / "actor"
    actor, _, _ = load_policy_artifact(actor_path, device="cpu")
    changes = {}
    reference = base.state_dict()
    for name, value in actor.state_dict().items():
        if value.dtype.is_floating_point:
            changes[name] = float((value - reference[name]).abs().max())
        elif not torch.equal(value, reference[name]):
            raise ValueError(f"Nonfloating actor state changed: {name}")
    if any(v != 0 for name, v in changes.items()
           if not name.startswith("readout.") and name != "log_std"):
        raise ValueError("A frozen non-readout actor parameter changed")
    if not torch.equal(actor.log_std, torch.full_like(actor.log_std, -2.5)):
        raise ValueError("Exploration scale changed")
    if not any(v > 0 for name, v in changes.items() if name.startswith("readout.")):
        raise ValueError("No readout parameter updated")
    report = {"passed": True, "parameter_max_abs_changes": changes,
              "model_card_sha256": digest(actor_path / "model_card.json")}
    write_json(ROOT / config["output"] / "parameter_verification.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.seeds != [0, 1, 2]:
        parser.error("This development pipeline is restricted to declared seeds 0, 1, 2")
    torch.set_num_threads(1)
    accepted = json.loads((STUDY / "baseline_acceptance.json").read_text())
    if not accepted["accepted"] or accepted["model_card_sha256"] != digest(
        ROOT / BASELINE / "model_card.json"
    ):
        raise ValueError("Accepted baseline identity differs")
    if accepted["dn_health_sha256"] != digest(STUDY / accepted["dn_health_file"]):
        raise ValueError("Accepted baseline health evidence changed")
    if accepted["evaluation_sha256"] != digest(STUDY / accepted["evaluation_file"]):
        raise ValueError("Accepted baseline evaluation evidence changed")
    # The descriptor stays open for the complete run; the OS releases it on exit.
    lock = None
    if not args.dry_run:
        lock = (STUDY / f"pipeline_{args.recipe}.lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = STUDY / f"pipeline_{args.recipe}.json"
    reports = [STUDY / "bc_development.json"]
    state = {"recipe": args.recipe, "seed_plan": args.seeds, "completed_seeds": []}

    def status(phase, seed):
        state.update(phase=phase, seed=seed, updated_unix_s=time.time())
        write_json(state_path, state)
        print(phase, "seed", seed, flush=True)

    def run(script, arguments, logfile):
        with logfile.open("a") as handle:
            subprocess.run([sys.executable, str(ROOT / "scripts" / script), *arguments],
                           cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)

    for seed in args.seeds:
        config_path = STUDY / "configs/ppo" / args.recipe / f"seed_{seed}.yaml"
        config = yaml.safe_load(config_path.read_text())
        if args.dry_run:
            print(config_path.relative_to(ROOT), config["output"], flush=True)
            continue
        if seed == 0:
            status("waiting_for_external_seed_0_completion", seed)
            while not completed(config):
                time.sleep(5)
        elif not completed(config):
            status("training", seed)
            command = ["--config", str(config_path), "--workers", "2"]
            if (ROOT / config["output"] / "training_checkpoint.pt").exists():
                command.append("--resume")
            run("diagnostics/local_ppo_train.py", command,
                STUDY / f"ppo_{args.recipe}_seed_{seed}.log")
            if not completed(config):
                raise RuntimeError("Training exited without a complete validated artifact")
        status("checking_parameters", seed)
        verify_parameters(config)
        result = STUDY / f"development_{args.recipe}_seed_{seed}.json"
        actor = str(Path(config["output"]) / "actor")
        if not result.exists() or not json.loads(result.read_text()).get("complete"):
            status("evaluating", seed)
            run("diagnostics/local_ppo_eval.py", [
                "--config", str(STUDY / "configs/study_eval.yaml"),
                "--output", str(result), "--artifacts", actor,
                "--episodes", "24", "--lanes", "24", "--workers", "2",
            ], STUDY / f"development_{args.recipe}_seed_{seed}.log")
        reports.append(result)
        health = ROOT / config["output"] / "dn_health.json"
        if not health.exists():
            status("checking_dn_health", seed)
            run("dn_health.py", [
                "--artifact", actor, "--config", str(STUDY / "configs/study_eval.yaml"),
                "--stage", "ppo", "--sparse", "--output", str(health),
            ], STUDY / f"dn_{args.recipe}_seed_{seed}.log")
        # A failed scientific gate is retained as a result; run remaining seeds.
        state["completed_seeds"].append({
            "seed": seed, "dn_healthy": json.loads(health.read_text())["passed"],
        })
        status("summarizing", seed)
        run("diagnostics/local_ppo_compare.py", [
            "--baseline", BASELINE, "--reports", *[str(path) for path in reports],
            "--output", str(STUDY / f"development_comparison_{args.recipe}.json"),
        ], STUDY / f"development_comparison_{args.recipe}.log")
    if not args.dry_run:
        status("complete", args.seeds[-1])


if __name__ == "__main__":
    main()
