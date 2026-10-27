#!/usr/bin/env python3
"""Run the complete generated six-variant, five-seed stage matrix.

Stages of one (variant, seed) chain always run in order; ``--jobs`` chains run
concurrently (they share the GPU and split the CPU environment workers).

#25 / #4 Decision 6390024036 item 8: when BC ends the chain (--algorithm none),
DN steering-quality failures are recorded; engineering failures halt the matrix
even with --keep-going. Other DN gates retain their strict behavior. After BC's
DN gate, every variant runs CPU modality sensitivity: O, VO and VO-late must
pass G4 (odor swap mean >= 1e-3 rad/s and >= 3 times the pretrain_odor actor).
V, GRU and RS only record modality results; odor/vision ratios never gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from flytrack_core.contracts import file_sha256
from flytrack_train.audit import audit_environment
from flytrack_train.config import contract_from_config, read_config, write_json
from flytrack_train.recovery import Recovery

VARIANTS = ("V", "O", "VO", "VO-late", "GRU", "RS")
MODALITY_ENFORCED_VARIANTS = frozenset(("O", "VO", "VO-late"))
MODALITY_ABSOLUTE_FLOOR = 1e-3
MODALITY_RELATIVE_FACTOR = 3.0
DN_HEALTH_ENGINEERING_CHECKS = frozenset(
    ("finite", "not_all_silent", "active_motor_count", "central_gradient")
)
DN_HEALTH_QUALITY_CHECKS = frozenset(
    ("omega_correlation", "omega_r2", "omega_amplitude", "eye_swap_response")
)


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


def check_dn_health(python, config_path, destination, stage, environment):
    """Run a fresh read-only gate, including when a completed stage is resumed."""
    # A crashed rerun must not leave an older healthy report looking current.
    (destination / "dn_health" / "metrics.json").unlink(missing_ok=True)
    command = [
        python,
        str(Path(__file__).with_name("dn_health.py")),
        "--artifact",
        str(destination / "actor"),
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--episodes",
        "24",
        "--steps",
        "200",
        "--output",
        str(destination / "dn_health" / "metrics.json"),
        "--require-healthy",
    ]
    with (destination / "dn_health.log").open("a", buffering=1) as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False
        )
    return result.returncode


def classify_dn_health(destination, exit_code):
    """Terminal BC only: unknown failed checks and diagnostic errors are engineering."""
    engineering, quality = [], []
    # dn_health.py writes metrics before exiting 2 for an unhealthy actor.
    if exit_code not in (0, 2):
        engineering.append("diagnostic_exit_code")
    try:
        report = json.loads((destination / "dn_health" / "metrics.json").read_text())
        checks = report["checks"]
        if (
            not isinstance(checks, dict)
            or not DN_HEALTH_ENGINEERING_CHECKS <= checks.keys()
            or any(type(value) is not bool for value in checks.values())
            or report["passed"] is not all(checks.values())
            or (exit_code == 0 and not report["passed"])
            or (exit_code == 2 and report["passed"])
        ):
            raise ValueError("Inconsistent DN health metrics")
        for name, passed in checks.items():
            if not passed:
                (quality if name in DN_HEALTH_QUALITY_CHECKS else engineering).append(name)
    except FileNotFoundError:
        engineering.append("missing_metrics")
    except (OSError, ValueError, KeyError, TypeError):
        engineering.append("invalid_metrics")
    return sorted(engineering), sorted(quality)


def modality_gate(variant, bc, start=None, *, best_epoch=None):
    """Evaluate G4 from sensitivity rows; ratios and best epochs are recording only."""

    def summarize(report):
        if report is None:
            return None
        return {
            "odor_swap_lr_mean": report["odor_swap_lr"]["mean"],
            "odor_to_vision_ratio": report["odor_to_vision_ratio"],
            "no_odor_mean": report["no_odor"]["mean"],
            "no_vision_mean": report["no_vision"]["mean"],
        }

    bc, start = summarize(bc), summarize(start)
    mean = bc["odor_swap_lr_mean"] if bc is not None else None
    relative_floor = (
        MODALITY_RELATIVE_FACTOR * start["odor_swap_lr_mean"] if start is not None else None
    )
    finite = mean is not None and math.isfinite(mean)
    absolute = finite and mean >= MODALITY_ABSOLUTE_FLOOR
    relative = (
        finite and math.isfinite(relative_floor) and mean >= relative_floor
        if relative_floor is not None
        else None
    )
    return {
        "variant": variant,
        "enforced": variant in MODALITY_ENFORCED_VARIANTS,
        "bc": bc,
        "start": start,
        "absolute_floor": MODALITY_ABSOLUTE_FLOOR,
        "relative_factor": MODALITY_RELATIVE_FACTOR,
        "relative_floor": relative_floor,
        "checks": {"absolute": absolute, "relative": relative},
        "passed": absolute and relative is not False,
        "best_epoch": best_epoch,
    }


def bc_best_epochs(destination):
    """The last epoch report of each DAgger round records its selected best epoch."""
    path = destination / "training_metrics.json"
    try:
        rounds = json.loads(path.read_text())
        values = [
            row["epochs"][-1].get("best_epoch") if row.get("epochs") else None for row in rounds
        ]
    except (OSError, ValueError, AttributeError, KeyError, TypeError, IndexError):
        # Optional training metadata must never change the modality verdict.
        return None
    return values if any(value is not None for value in values) else None


def check_modality(python, config_path, destination, variant, start_actor, environment):
    """Rerun CPU sensitivity at its default diagnostic seed/steps and persist G4."""
    output = destination / "modality"
    output.mkdir(parents=True, exist_ok=True)
    sensitivity_path = output / "sensitivity.json"
    sensitivity_path.unlink(missing_ok=True)
    (output / "metrics.json").unlink(missing_ok=True)
    actor = destination / "actor"
    metrics = modality_gate(variant, None)
    try:
        metrics["best_epoch"] = bc_best_epochs(destination)
        command = [
            python,
            str(Path(__file__).parent / "diagnostics" / "modality_sensitivity.py"),
            str(sensitivity_path),
            str(config_path),
            str(actor),
            *([str(start_actor)] if start_actor is not None else []),
        ]
        diagnostic_environment = {**environment, "DEVICE": "cpu"}
        for name in ("STEPS", "SEED", "BUILD"):
            diagnostic_environment.pop(name, None)
        with (destination / "modality.log").open("a", buffering=1) as log:
            result = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=diagnostic_environment,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"Sensitivity exited {result.returncode}; inspect modality.log")
        policies = json.loads(sensitivity_path.read_text())["policies"]
        metrics = modality_gate(
            variant,
            policies[str(actor)],
            policies[str(start_actor)] if start_actor is not None else None,
            best_epoch=metrics["best_epoch"],
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
        metrics["error"] = str(error)
    write_json(output / "metrics.json", metrics)
    return metrics


def start_actor_for(plan, row):
    """Find the actual output of this chain's odor pretraining, independent of layout."""
    return next(
        (
            Path(candidate["output"]) / "actor"
            for candidate in plan
            if candidate["variant"] == row["variant"]
            and candidate["seed"] == row["seed"]
            and candidate["stage"] == "pretrain_odor"
        ),
        None,
    )


def audit_worlds(plan, destination):
    """Refuse to train in a world where a blind constant action can "track".

    Every chain shares the stage templates, so auditing the distinct BC and
    final-algorithm worlds once covers the whole matrix.
    """
    reports, seen = [], set()
    for row in plan:
        if row["command"] not in ("bc", "ppo", "es"):
            continue
        config = read_config(row["path"])
        key = json.dumps(config.get("env", {}), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        report = audit_environment(config, contract_from_config(config))
        reports.append({"stage": row["stage"], "config": row["path"], **report})
    write_json(destination, reports)
    failures = [f"{r['stage']}: {'; '.join(r['failures'])}" for r in reports if not r["passed"]]
    if failures:
        raise SystemExit(f"Degenerate training world (see {destination}): " + " | ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, default=Path("runs/experiment_configs"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--algorithm",
        choices=("none", "ppo", "es"),
        default="none",
        help="Stage after BC; none ends every chain at BC/DAgger (#35 Decision)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent (variant, seed) chains")
    parser.add_argument(
        "--workers", default="auto", help="Environment worker processes per job (auto splits CPUs)"
    )
    parser.add_argument("--tf32", action="store_true", help="Pass --tf32 to every stage")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help=(
            "A failed chain stops only itself, except terminal-BC DN engineering failures "
            "always halt the matrix (#4 Decision 6390024036 item 8)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-environment-audit",
        action="store_true",
        help="Do not refuse worlds in which a blind constant action succeeds (diagnostics only)",
    )
    args = parser.parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds or min(args.seeds) < 0:
        parser.error("Seeds must be distinct nonnegative integers")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    workers = (
        max(1, min(8, ((os.cpu_count() or 2) - 1) // args.jobs))
        if args.workers == "auto"
        else int(args.workers)
    )
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
            final = [] if args.algorithm == "none" else [args.algorithm]
            for stage in [*stages, "bc", *final]:
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
    if not args.skip_environment_audit:
        audit_worlds(plan, args.configs / "environment_audit.json")
    lock = threading.Lock()
    halt = threading.Event()
    failures: list[str] = []

    def commit(row=None, **values):
        # Rows are mutated only under the lock: json serialization in another chain's
        # commit would otherwise see a row change size and fail that whole chain.
        with lock:
            if row is not None:
                row.update(values)
            write_json(args.configs / "execution_plan.json", manifest)

    commit()
    environment = {**os.environ, "PYTHON_DOTENV_DISABLED": "1", "PYTHONUNBUFFERED": "1"}
    extra = ["--workers", str(workers), *(["--tf32"] if args.tf32 else [])]

    def run_stage(index, row):
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
                *extra,
            ]
            if (destination / "training_checkpoint.pt").is_file():
                command.append("--resume")
            commit(row, status="running")
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
                commit(row, status="failed", exit_code=result.returncode)
                raise RuntimeError(f"Stage failed; inspect {destination / 'execution.log'}")
            if not completed(config, row["command"], device):
                raise RuntimeError(f"Stage exited without a verified actor artifact: {destination}")
            status = "complete"
        if row["variant"] != "GRU":
            terminal_bc = row["stage"] == "bc" and args.algorithm == "none"
            try:
                exit_code = check_dn_health(
                    args.python, row["path"], destination, row["command"], environment
                )
            except Exception as error:  # noqa: BLE001 - terminal BC must halt on diagnostic errors
                if terminal_bc:
                    halt.set()
                    commit(
                        row,
                        status="dn_health_failed",
                        dn_health_engineering_failures=["diagnostic_error"],
                    )
                raise RuntimeError(f"DN health diagnostic error: {destination}") from error
            if terminal_bc:
                engineering, quality = classify_dn_health(destination, exit_code)
                if engineering:
                    halt.set()
                    commit(
                        row,
                        status="dn_health_failed",
                        exit_code=exit_code,
                        dn_health_engineering_failures=engineering,
                        dn_health_quality_failures=quality,
                    )
                    raise RuntimeError(f"DN health engineering failure; inspect {destination}")
                if quality:
                    commit(row, dn_health_quality_failures=quality)
            elif exit_code:
                commit(row, status="dn_health_failed", exit_code=exit_code)
                raise RuntimeError(
                    f"DN health gate failed; inspect {destination / 'dn_health.log'}"
                )
        if row["stage"] == "bc":
            metrics = check_modality(
                args.python,
                row["path"],
                destination,
                row["variant"],
                start_actor_for(plan, row),
                environment,
            )
            if metrics["enforced"] and not metrics["passed"]:
                commit(row, status="modality_gate_failed")
                raise RuntimeError(f"Modality gate failed; inspect {destination / 'modality'}")
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
                    *extra,
                ]
                with (destination / "validation.log").open("a", buffering=1) as log:
                    result = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=False
                    )
                if result.returncode:
                    commit(row, status="heldout_validation_failed", exit_code=result.returncode)
                    raise RuntimeError(f"Held-out decoding gate failed; inspect {validation}")
        commit(row, status=status, finished_unix_s=time.time())

    chains: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(plan):
        chains.setdefault((row["variant"], row["seed"]), []).append(index)

    def run_chain(indices):
        for index in indices:
            # By default one failure (often systemic, e.g. a gate every variant fails)
            # stops all chains; --keep-going cannot override terminal BC engineering errors.
            if halt.is_set() or (failures and not args.keep_going):
                return
            try:
                run_stage(index, plan[index])
            except Exception as error:  # noqa: BLE001 - reported after other chains stop
                with lock:
                    failures.append(str(error))
                print(error, flush=True)
                return

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        list(executor.map(run_chain, chains.values()))
    if failures:
        raise SystemExit("; ".join(failures))
    with lock:
        manifest["completed_unix_s"] = time.time()
        write_json(args.configs / "execution_plan.json", manifest)


if __name__ == "__main__":
    main()
