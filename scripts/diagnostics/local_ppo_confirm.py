"""Development or confirmation stage for one recipe from one accepted BC start.

Trains missing declared PPO seeds with the unchanged CPU runner, checks DN
health and binds every actor to its BC parent, evaluates BC and all PPO seeds
on the stage's episode set (confirmation: untouched episodes) with one evaluator
on one device, then runs the paired seed-level comparison. Every stage resumes
from its outputs.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml
from flytrack_core.model import load_policy_artifact
from flytrack_train.config import write_json
from local_ppo_audit import audit

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "runs/local_ppo_bc"
EVAL_CONFIG = STUDY / "configs/study_eval.yaml"
# stage -> (first episode seed, episodes, lanes, declared leading PPO seeds)
STAGES = {
    "confirmation": (1_630_000_000, 100, 50, [0, 1, 2, 3, 4]),
    "development": (1_610_000_000, 24, 24, [0, 1, 2]),
}
PRIMARY = "curved/normal"
PRESERVATION = ("wander/normal", "curved/occlusion", "curved/noise", "curved/conflict")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_acceptance(path, baseline):
    accepted = json.loads(path.read_text())
    if not accepted["accepted"] or Path(accepted["artifact"]) != baseline:
        raise ValueError("BC start is not the accepted artifact")
    if accepted["model_card_sha256"] != digest(ROOT / baseline / "model_card.json"):
        raise ValueError("Accepted BC identity differs")
    for kind in ("dn_health", "evaluation"):
        if accepted[f"{kind}_sha256"] != digest(STUDY / accepted[f"{kind}_file"]):
            raise ValueError(f"Accepted BC {kind} evidence changed")


def runner_known(sha):
    """The current runner, or an archived earlier version with this exact content."""
    if sha == digest(ROOT / "scripts/diagnostics/local_ppo_train.py"):
        return True
    archived = STUDY / "archive" / f"local_ppo_train.{sha}.py"
    return archived.exists() and digest(archived) == sha


def completed(config, baseline):
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
    runner = saved["config"].get("local_experiment", {}).get("runner_sha256")
    if not runner_known(runner):
        raise ValueError("Completed actor was trained by an unarchived runner version")
    expected.setdefault("local_experiment", {})["runner_sha256"] = runner
    if {k: v for k, v in saved["config"].items() if k != "resume"} != expected:
        raise ValueError("Completed actor configuration differs from this recipe")
    if Path(config["init_artifact"]) != baseline:
        raise ValueError("PPO configuration names a different BC parent")
    return True


def report_cells(path):
    if not path.exists():
        return set()
    policies = json.loads(path.read_text())["policies"]
    return set().union(*(set(p["cells"]) for p in policies.values())) if policies else set()


def writer_of(path):
    """PID of a running evaluator already writing this report, if any."""
    found = subprocess.run(["pgrep", "-f", f"local_ppo_eval_device.py .*--output {path}( |$)"],
                           capture_output=True, text=True).stdout.split()
    return int(found[0]) if found else None


class External:
    """Waits for an evaluator started outside this controller (no duplicate writer)."""

    def __init__(self, pid):
        self.pid, self.returncode = pid, None

    def poll(self):
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            self.returncode = 0
        return self.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Names state, log and report files")
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True, help="Directory of seed_N.yaml")
    parser.add_argument("--stage", choices=sorted(STAGES), default="confirmation")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--train-parallel", type=int, default=2)
    parser.add_argument("--train-workers", type=int, default=2)
    parser.add_argument("--eval-parallel", type=int, default=2)
    parser.add_argument("--eval-workers", type=int, default=3)
    parser.add_argument("--lanes", type=int, help="Default from the stage")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bc-report", type=Path,
                        help="Shared BC confirmation report; default <reports>/bc.json")
    args = parser.parse_args()
    first_episode, episodes, lanes, declared = STAGES[args.stage]
    args.seeds = args.seeds or declared
    args.lanes = args.lanes or lanes
    if args.seeds[: len(declared)] != declared or len(set(args.seeds)) != len(args.seeds):
        parser.error(f"Seeds start with the declared {args.stage} seeds {declared}, no repeats")
    check_acceptance(STUDY / args.acceptance, args.baseline)
    lock = (STUDY / f"confirm_{args.tag}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = STUDY / f"confirm_{args.tag}.json"
    state = {"tag": args.tag, "stage": args.stage, "recipe": args.recipe,
             "baseline": str(args.baseline), "seeds": args.seeds, "device": args.device,
             "episodes": [first_episode, episodes], "lanes": args.lanes}
    logs = STUDY / "confirm_logs" / args.tag
    reports = STUDY / args.stage / args.tag
    logs.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    def status(phase, **extra):
        state.update(phase=phase, updated_unix_s=time.time(), **extra)
        write_json(state_path, state)
        print(phase, extra, flush=True)

    def launch(job):
        output = job.get("output")
        pid = writer_of(output) if output else None
        if pid is not None:
            return External(pid)
        handle = job["log"].open("a")
        return subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / job["script"]), *job["arguments"]],
            cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        )

    configs = {}
    for seed in args.seeds:
        path = args.configs / f"seed_{seed}.yaml"
        config = yaml.safe_load(path.read_text())
        if config["seed"] != seed or config["local_experiment"]["name"] != args.recipe:
            raise ValueError(f"Configuration does not declare {args.recipe} seed {seed}")
        configs[seed] = (path, config)
    names = ["bc"] + [f"ppo_seed_{seed}" for seed in configs]
    paths = [args.bc_report or reports / "bc.json"] + [reports / f"{n}.json" for n in names[1:]]
    report_paths = dict(zip(names, paths, strict=True))
    audits, queue = {}, []

    # CPU jobs (training, DN health) and CUDA evaluations are bounded separately,
    # so finished seeds are evaluated while later seeds still train. The primary
    # cell of every policy is evaluated before any preservation cell.
    def evaluate(name, actor, cells=(PRIMARY,), priority=0):
        missing = [c for c in cells if c not in report_cells(report_paths[name])]
        if not missing and cells != (PRIMARY,):
            return
        job = {"name": f"evaluate {name} {'primary' if priority == 0 else 'preservation'}",
               "pool": "gpu", "priority": priority, "output": report_paths[name],
               "script": "diagnostics/local_ppo_eval_device.py", "arguments": [
                   "--config", str(EVAL_CONFIG), "--output", str(report_paths[name]),
                   "--artifacts", str(actor), "--episodes", str(episodes),
                   "--base-seed", str(first_episode), "--lanes", str(args.lanes),
                   "--workers", str(args.eval_workers), "--device", args.device,
                   "--resume", "--cells", *(missing or cells)],
               "log": logs / f"evaluate_{name}.log"}
        if priority == 0:
            job["then"] = lambda: evaluate(name, actor, (PRIMARY, *PRESERVATION), 1)
        if missing:
            queue.append(job)
        else:
            job["then"]()

    def audited(seed):
        config = configs[seed][1]
        output = ROOT / config["output"]
        report = audit(ROOT / args.baseline, output / "actor", output / "dn_health.json",
                       config["train"]["total_steps"])
        write_json(output / "preservation_audit.json", report)
        audits[seed] = {"passed": report["passed"],
                        "failed_checks": [k for k, v in report["checks"].items() if not v]}
        evaluate(f"ppo_seed_{seed}", Path(config["output"]) / "actor")

    def trained(seed):
        config = configs[seed][1]
        if not completed(config, args.baseline):
            raise RuntimeError(f"Seed {seed} has no complete validated artifact")
        actor = Path(config["output"]) / "actor"
        if (ROOT / config["output"] / "dn_health.json").exists():
            audited(seed)
            return
        queue.append({"name": f"dn seed {seed}", "pool": "cpu", "priority": 0,
                      "script": "dn_health.py",
                      "arguments": ["--artifact", str(actor), "--config", str(EVAL_CONFIG),
                                    "--stage", "ppo", "--sparse",
                                    "--output", str(Path(config["output"]) / "dn_health.json")],
                      "log": logs / f"dn_seed_{seed}.log", "then": lambda: audited(seed)})

    evaluate("bc", args.baseline)
    for seed, (path, config) in configs.items():
        if completed(config, args.baseline):
            trained(seed)
            continue
        resume = (ROOT / config["output"] / "training_checkpoint.pt").exists()
        queue.append({"name": f"train seed {seed}", "pool": "cpu", "priority": 1,
                      "script": "diagnostics/local_ppo_train.py",
                      "arguments": ["--config", str(path), "--workers", str(args.train_workers),
                                    "--device", config["local_experiment"].get("device", "cpu")]
                      + (["--resume"] if resume else []),
                      "log": logs / f"train_seed_{seed}.log",
                      "then": lambda seed=seed: trained(seed)})
    limits = {"cpu": args.train_parallel, "gpu": args.eval_parallel}
    running, failed, shown = [], [], None
    while queue or running:
        for job in sorted(queue, key=lambda j: j["priority"]):
            if sum(other["pool"] == job["pool"] for other, _ in running) < limits[job["pool"]]:
                queue.remove(job)
                running.append((job, launch(job)))
        current = ([job["name"] for job, _ in running], [job["name"] for job in queue])
        if current != shown:
            status("running", running=current[0], queued=current[1], audits=audits)
            shown = current
        time.sleep(5)
        for job, process in list(running):
            if process.poll() is None:
                continue
            running.remove((job, process))
            if process.returncode:
                failed.append(job["name"])
            elif "then" in job:
                job["then"]()
    incomplete = [n for n, p in report_paths.items()
                  if report_cells(p) != {PRIMARY, *PRESERVATION}]
    if failed or incomplete:
        status("failed", failed=failed, incomplete=incomplete, audits=audits)
        raise RuntimeError(f"Failed jobs {failed}; incomplete reports {incomplete}")

    status("comparing")
    analyses = {"all_seeds": names}
    if len(args.seeds) > len(declared):
        analyses["declared_seeds"] = ["bc"] + [f"ppo_seed_{seed}" for seed in declared]
    results = {}
    for analysis, members in analyses.items():
        suffix = "" if analysis == "all_seeds" else "_declared"
        comparison = STUDY / f"{args.stage}_comparison_{args.tag}{suffix}.json"
        with (logs / "compare.log").open("a") as handle:
            subprocess.run([
                sys.executable, str(ROOT / "scripts/diagnostics/local_ppo_compare.py"),
                "--baseline", str(args.baseline), "--output", str(comparison),
                "--reports", *[str(report_paths[name]) for name in members],
            ], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
        results[analysis] = json.loads(comparison.read_text())["recipes"][args.recipe][
            "numerical_gates"]
    status("complete", audits=audits, numerical_gates=results)


if __name__ == "__main__":
    main()
