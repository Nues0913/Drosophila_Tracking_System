"""Sequential stage runner, persistent GPU accounting, validation and timeout control."""

import csv
import math
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from accept100 import acceptance
from protocol100 import (
    digest,
    dn_rows,
    environment,
    read,
    validate_actor,
    validate_eval,
    verify_c9,
    write,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGES = ("c10_train", "bc_eval", "c10_eval", "dn_c10", "c9_eval", "acceptance")
BUDGET = {"c10_train": "b", "c10_eval": "b", "bc_eval": "a", "c9_eval": "a"}


def cpu_command(*args):
    return ["prlimit", "--as=6442450944", os.environ["PY"], *map(str, args)]


def cpu_env():
    return {**os.environ, "CUDA_VISIBLE_DEVICES": ""}


def outputs(out, stage):
    if stage == "c10_train":
        actor = out / "ppo_c10/actor"
        card = read(actor / "model_card.json")
        return [
            out / "train_runtime.json",
            out / "ppo_c10/progress.json",
            out / "ppo_c10/training_metrics.json",
            out / "ppo_c10/training_checkpoint.pt",
            *[actor / name for name in ("model_card.json", *card["files"])],
        ]
    if stage.endswith("_eval"):
        return [out / f"eval_{stage.removesuffix('_eval')}.json"]
    if stage == "dn_c10":
        return [out / "dn_c10.json"]
    return [out / "acceptance.json", out / "summary.md"]


def validate(out, stage):
    manifest = read(out / "manifest.json")
    smoke = manifest["smoke"]
    if stage == "c10_train":
        from check import ppo_complete

        validate_actor(out / "ppo_c10/actor")
        ppo_complete(out / "ppo_c10", manifest["configs"]["ppo_c10.yaml"], smoke)
        runtime = read(out / "train_runtime.json")
        if (
            runtime["device"] != manifest["device"]
            or runtime["deterministic"] is not False
            or runtime["environment"]["DETERMINISTIC"] is not None
            or runtime["environment"]["CUBLAS_WORKSPACE_CONFIG"] is not None
        ):
            raise ValueError("Training runtime differs")
        if not smoke and "4060" not in runtime["device_name"]:
            raise ValueError("Training did not use the 4060")
    elif stage.endswith("_eval"):
        name = stage.removesuffix("_eval")
        actor = (
            out / "ppo_c10/actor" if name == "c10" else manifest["inputs"][f"{name.upper()}_ACTOR"]
        )
        if name == "c9":
            verify_c9(ROOT)
        validate_eval(
            read(outputs(out, stage)[0]),
            [actor],
            episodes=manifest["episodes"],
            seed_start=manifest["seed_start"],
            smoke=smoke,
            env=manifest["configs"]["eval.yaml"]["env"],
        )
    elif stage == "dn_c10":
        dn_rows(read(out / "dn_c10.json"), out / "ppo_c10/actor", 8 if smoke else 200)
    else:
        acceptance(out, validate_only=True)


def train_entry(out, extra):
    import torch
    from flytrack_train.__main__ import main

    manifest = read(out / "manifest.json")
    if not manifest["smoke"] and (
        not torch.cuda.is_available() or "4060" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("Formal training requires the 4060 CUDA device")
    write(
        out / "train_runtime.json",
        {
            "device": manifest["device"],
            "device_name": "CPU" if manifest["smoke"] else torch.cuda.get_device_name(0),
            "torch_version": str(torch.__version__),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "environment": environment(),
        },
    )
    sys.argv = [
        "flytrack_train",
        "ppo",
        "--config",
        str(out / "ppo_c10.yaml"),
        "--device",
        manifest["device"],
        "--workers",
        os.environ["PPO_WORKERS"],
        *extra,
    ]
    main()


def ledger(out):
    path = out / "stages.tsv"
    if not path.exists():
        return []
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        if row["stage"] not in STAGES or not math.isfinite(float(row["wall_s"])):
            raise ValueError("Invalid stage accounting")
        if float(row["wall_s"]) < 0:
            raise ValueError("Negative stage accounting")
    return rows


def record(out, stage, code, wall):
    path = out / "stages.tsv"
    existed = path.exists()
    with path.open("a") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        if not existed:
            writer.writerow(("stage", "end_utc", "exit", "wall_s"))
        writer.writerow((stage, datetime.now(UTC).isoformat(), code, f"{wall:.9f}"))
        stream.flush()
        os.fsync(stream.fileno())


def remaining(out, manifest, stage):
    bucket = BUDGET[stage]
    used = sum(float(r["wall_s"]) for r in ledger(out) if BUDGET.get(r["stage"]) == bucket)
    return manifest["budgets_s"][bucket] - used


def not_run(out, stage, reason):
    skipped = STAGES[STAGES.index(stage) :]
    for name in skipped:
        record(out, name, "NOT RUN", 0)
    result = {"status": "NOT RUN", "reason": reason, "stages": list(skipped), "accepted": False}
    write(out / "budget_status.json", result)
    write(out / "acceptance.json", result)
    (out / "summary.md").write_text(
        f"# C10 protocol100: NOT RUN\n\n{reason}\n\n"
        + "\n".join(f"- {name}: NOT RUN" for name in skipped)
        + "\n"
    )


def run(out):
    manifest = read(out / "manifest.json")
    smoke = manifest["smoke"]
    pending = out / "active_stage.json"
    # SIGKILL/power loss cannot run finally. Conservatively charge elapsed downtime
    # rather than granting a fresh budget; cap at the budget left when launched.
    if pending.exists():
        row = read(pending)
        wall = max(0, time.time() - row["started_unix_s"])
        if row["remaining_s"] is not None:
            wall = min(wall, row["remaining_s"])
        record(out, row["stage"], "INTERRUPTED", wall)
        pending.unlink()
    if (out / "budget_status.json").exists():
        print("Budget already exhausted; NOT RUN (see budget_status.json)", flush=True)
        return
    monitor = None
    active = None

    def log(message):
        text = f"[{datetime.now(UTC).isoformat()}] {message}"
        print(text, flush=True)
        with (out / "run.log").open("a") as stream:
            stream.write(text + "\n")

    def stop_group(process):
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        # Workers may outlive a parent that exited on TERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if os.environ.get("MONITOR_SH"):
            directory = out / "monitor"
            directory.mkdir(exist_ok=True)
            with (directory / "monitor.log").open("a") as stream:
                monitor = subprocess.Popen(
                    ["bash", os.environ["MONITOR_SH"], str(directory), "2", "0"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
        for stage in STAGES:
            if (out / "STOP").exists():
                log("STOP present; exiting cleanly")
                return
            receipt = out / f"{stage}.complete.json"
            if receipt.exists():
                expected = {str(p): digest(p) for p in outputs(out, stage)}
                if read(receipt) != expected:
                    raise ValueError(f"Completed stage changed: {stage}")
                subprocess.run(
                    cpu_command(HERE / "run_c10.py", "validate", out, stage),
                    env=cpu_env(),
                    check=True,
                )
                log(f"skip {stage} (validated complete)")
                continue
            if stage == "c9_eval":
                verify_c9(ROOT)
                dn_rows(read(manifest["inputs"]["C9_DN"]), manifest["inputs"]["C9_ACTOR"], 200)
                log("C9 actor: all listed ppo_bc checksums verified")
            args = []
            cwd = ROOT
            env = dict(os.environ)
            if stage == "c10_train":
                env.pop("DETERMINISTIC", None)
                env.pop("CUBLAS_WORKSPACE_CONFIG", None)
                args = [str(HERE / "run_c10.py"), "train", str(out)]
                if (out / "ppo_c10/training_checkpoint.pt").exists():
                    args.append("--resume")
            elif stage.endswith("_eval"):
                env.update(DETERMINISTIC="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
                name = stage.removesuffix("_eval")
                actor = (
                    str(out / "ppo_c10/actor")
                    if name == "c10"
                    else manifest["inputs"][f"{name.upper()}_ACTOR"]
                )
                args = [
                    str(HERE / "eval100.py"),
                    str(out / f"eval_{name}.tmp.json"),
                    str(out / "eval.yaml"),
                    actor,
                ]
            elif stage == "dn_c10":
                args = [
                    manifest["inputs"]["MEASURE_PY"],
                    str(out / "dn_c10.tmp.json"),
                    str(out / "ppo_c10/actor"),
                ]
                cwd = Path(manifest["inputs"]["MEASURE_REPO"])
                env = {**cpu_env(), "PYTHONPATH": str(cwd / "python")}
            else:
                args = [str(HERE / "accept100.py"), str(out)]
            gpu = stage in BUDGET and not smoke
            command = [os.environ["PY"], *args] if gpu else cpu_command(*args)
            available = remaining(out, manifest, stage) if gpu else None
            if gpu:
                # Reserve two seconds for timeout's TERM -> KILL grace period.
                if available <= 2:
                    not_run(out, stage, f"Budget ({BUDGET[stage]}) exhausted")
                    return
                command = [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=2s",
                    f"{available - 2:.9f}s",
                    *command,
                ]
            else:
                env["CUDA_VISIBLE_DEVICES"] = ""
            log(f"start {stage}; remaining_s={available}")
            started = time.monotonic()
            code = "INTERRUPTED"
            write(
                pending, {"stage": stage, "started_unix_s": time.time(), "remaining_s": available}
            )
            try:
                with (out / f"{stage}.log").open("a") as stream:
                    active = subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    code = active.wait()
            finally:
                stop_group(active)
                active = None
                wall = time.monotonic() - started
                record(out, stage, code, wall)
                pending.unlink(missing_ok=True)
            if gpu and (code in (124, 137, -9) or wall >= available):
                not_run(out, stage, f"Budget ({BUDGET[stage]}) timeout at {stage}")
                log(f"{stage} and subsequent stages: NOT RUN")
                return
            if code != 0:
                raise RuntimeError(f"{stage} exit={code}; see {stage}.log")
            if stage.endswith("_eval"):
                name = stage.removesuffix("_eval")
                (out / f"eval_{name}.tmp.json").replace(out / f"eval_{name}.json")
            elif stage == "dn_c10":
                (out / "dn_c10.tmp.json").replace(out / "dn_c10.json")
            subprocess.run(
                cpu_command(HERE / "run_c10.py", "validate", out, stage), env=cpu_env(), check=True
            )
            write(receipt, {str(p): digest(p) for p in outputs(out, stage)})
            log(f"{stage} complete; wall_s={wall:.3f}")
        log("ALL DONE")
    finally:
        stop_group(active)
        stop_group(monitor)


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate(Path(sys.argv[2]), sys.argv[3])
    elif sys.argv[1] == "train":
        train_entry(Path(sys.argv[2]), sys.argv[3:])
    else:
        run(Path(sys.argv[1]))
