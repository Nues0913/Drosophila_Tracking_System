"""Two-seed C12 scheduler with one GPU budget and bounded CPU snapshot chains."""

import csv
import fcntl
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from c11 import driver as c11_driver
from c11.driver import (
    BudgetExhausted,
    Stopped,
    cpu_command,
    interrupted,
    log,
    record,
    sanitize_log,
    stop_group,
)
from protocol100 import actor_hashes, dn_rows, finite, validate_eval

from c12.c12_common import (
    GPU_STAGES,
    RULES,
    SEEDS,
    SNAP_KINDS,
    STAGES,
    digest,
    read,
    relative,
    seeds,
    write,
)
from c12.c12_jobs import acceptance_result, snapshot_actors


def ledger(out):
    path = out / "stages.tsv"
    if not path.exists():
        return []
    with path.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if any(
        r["stage"] not in STAGES or not math.isfinite(float(r["wall_s"])) or float(r["wall_s"]) < 0
        for r in rows
    ):
        raise ValueError("Invalid C12 stage accounting")
    return rows


def remaining(out, manifest):
    if manifest["smoke"]:
        return manifest["gpu_budget_s"]
    return manifest["gpu_budget_s"] - sum(
        float(r["wall_s"]) for r in ledger(out) if r["stage"] in GPU_STAGES
    )


def recover_pending(out):
    for path in sorted(out.glob("active_*.json")):
        row = read(path)
        if row["stage"] not in STAGES:
            raise ValueError("Unknown interrupted stage")
        wall = max(0, time.time() - row["started_unix_s"])
        if row["remaining_s"] is not None:
            wall = min(wall, row["remaining_s"])
        record(out, row["stage"], "INTERRUPTED", wall)
        path.unlink()


def outputs(out, stage):
    if stage == "bc_eval":
        return [out / "eval_bc.json"]
    if stage == "bc_snap_eval":
        return [out / "snap_eval_bc.json"]
    if stage == "bc_snap_dn":
        return [out / "dn_bc.json"]
    if stage in ("acceptance", "summary"):
        return (
            [out / "acceptance.json"]
            if stage == "acceptance"
            else [out / "summary.json", out / "summary.md"]
        )
    seed, kind = stage.split("_", 1)
    if seed not in SEEDS:
        raise ValueError("Unknown C12 stage")
    legacy = {"train": "c11_train", "eval": "c11_eval", "dn": "dn_c11"}.get(kind, kind)
    return c11_driver.outputs(out / seed, legacy)


def validate_snap_rows(out, rows, expected):
    m = read(out / "manifest.json")
    if [{k: r[k] for k in ("step", "actor")} for r in rows] != expected:
        raise ValueError("Snapshot evaluation actor set differs")
    expected_seeds = seeds(m["snap_seed_start"], m["snap_episodes"], m["smoke"])
    for row in rows:
        finite(row)
        if row["actor_sha256"] != actor_hashes(row["actor"]) or row["device"] != "cpu":
            raise ValueError("Snapshot actor/runtime differs")
        if row["seed_start"] != m["snap_seed_start"] or row["episodes"] != m["snap_episodes"]:
            raise ValueError("Snapshot evaluation protocol differs")
        if [r["environment_seed"] for r in row["normal"]["episodes"]] != expected_seeds:
            raise ValueError("Snapshot seed pairing differs")


def validate(out, stage):
    m = read(out / "manifest.json")
    if stage == "summary":
        from c12.summarize_c12 import summarize

        summarize(out, validate_only=True)
    elif stage == "acceptance":
        if read(out / "acceptance.json") != acceptance_result(out):
            raise ValueError("Acceptance result differs")
    elif stage in ("s0_eval", "s1_eval", "bc_eval"):
        seeds(m["seed_start"], m["episodes"], m["smoke"])
        actor = (
            m["inputs"]["BC_ACTOR"]
            if stage == "bc_eval"
            else str(out / stage[:2] / "ppo_c11/actor")
        )
        validate_eval(
            read(outputs(out, stage)[0]),
            [actor],
            episodes=m["episodes"],
            seed_start=m["seed_start"],
            smoke=m["smoke"],
            env=m["configs"]["eval.yaml"]["env"],
        )
    elif stage == "bc_snap_eval":
        validate_snap_rows(
            out,
            read(out / "snap_eval_bc.json")["rows"],
            [{"step": 0, "actor": m["inputs"]["BC_ACTOR"]}],
        )
    elif stage == "bc_snap_dn":
        dn_rows(read(out / "dn_bc.json"), m["inputs"]["BC_ACTOR"], 8 if m["smoke"] else 200)
    else:
        seed, kind = stage.split("_", 1)
        folder = out / seed
        if kind in ("train", "dn", "snap_build"):
            c11_driver.validate(folder, {"train": "c11_train", "dn": "dn_c11"}.get(kind, kind))
        elif kind == "snap_eval":
            rows = read(folder / "snap_eval.json")["rows"]
            validate_snap_rows(out, rows, snapshot_actors(out, seed))
            if rows[0] != read(out / "snap_eval_bc.json")["rows"][0]:
                raise ValueError("Shared BC snapshot differs")
        elif kind == "snap_dn":
            rows = read(folder / "snap_dn.json")["rows"]
            if [{k: r[k] for k in ("step", "actor")} for r in rows] != snapshot_actors(out, seed):
                raise ValueError("Snapshot DN actor set differs")
            if rows[0]["report"] != str(out / "dn_bc.json"):
                raise ValueError("Shared BC DN differs")
            for row in rows:
                dn_rows(read(row["report"]), row["actor"], 8 if m["smoke"] else 200)
        else:
            raise ValueError("Unknown C12 stage")


def stage_env(stage, smoke):
    # C11 removes training determinism flags and hides CUDA in CPU stages.
    legacy_stage = (
        "c11_train"
        if stage.endswith("_train")
        else "c11_eval"
        if stage in GPU_STAGES
        else "snap_eval"
    )
    return c11_driver.stage_env(legacy_stage, smoke)


def run_stage(out, stage, in_chain=False):
    if stage not in STAGES:
        raise ValueError("Unknown C12 stage")
    m = read(out / "manifest.json")
    receipt = out / f"{stage}.complete.json"
    env = stage_env(stage, m["smoke"])
    if (out / "STOP").exists():
        raise Stopped
    validation = cpu_command("-m", "c12.driver", "validate", out, stage)
    if receipt.exists():
        if read(receipt) != {str(p): digest(p) for p in outputs(out, stage)}:
            raise ValueError(f"Completed stage changed: {stage}")
        subprocess.run(
            validation,
            env={**env, "CUDA_VISIBLE_DEVICES": ""},
            check=True,
            stdin=subprocess.DEVNULL,
        )
        log(out, f"skip {stage} (validated complete)")
        return
    gpu = stage in GPU_STAGES and not m["smoke"]
    available = remaining(out, m) if gpu else None
    if gpu and available <= 5:
        raise BudgetExhausted(f"GPU budget exhausted before {stage}")
    args = ["-m", "c12.c12_jobs", str(out), stage]
    command = [sys.executable, *args] if gpu else cpu_command(*args)
    pending = out / f"active_{stage}.json"
    process = sidecar = None
    code = "INTERRUPTED"
    started = time.monotonic()
    log(out, f"start {stage}; remaining_s={available}")
    write(pending, {"stage": stage, "started_unix_s": time.time(), "remaining_s": available})
    folder = out / stage[:2] if stage.endswith("_train") else None
    try:
        with (out / f"{stage}.log").open("a") as stream:
            if folder is not None:
                (folder / "sidecar.stop").unlink(missing_ok=True)
                ready = folder / "sidecar.ready"
                ready.unlink(missing_ok=True)
                sidecar = subprocess.Popen(
                    ["nice", "-n", "10", *cpu_command("-m", "c12.driver", "sidecar", folder)],
                    env={**env, "CUDA_VISIBLE_DEVICES": ""},
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                wait_s = (
                    15 if not gpu else max(0, min(15, available - (time.monotonic() - started) - 5))
                )
                deadline = time.monotonic() + wait_s
                while not ready.exists() and sidecar.poll() is None and time.monotonic() < deadline:
                    if (out / "STOP").exists():
                        raise Stopped
                    time.sleep(0.1)
                if not ready.exists():
                    log(out, "WARNING: sidecar startup delayed; training continues")
            if gpu:
                launch_remaining = available - (time.monotonic() - started)
                if launch_remaining <= 5:
                    raise BudgetExhausted(f"GPU budget exhausted before launching {stage}")
                command = [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=2s",
                    f"{launch_remaining - 5:.9f}s",
                    *command,
                ]
            process = subprocess.Popen(
                command,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=not in_chain,
            )
            while process.poll() is None:
                if (out / "STOP").exists():
                    raise Stopped
                time.sleep(0.1)
            code = process.wait()
            if sidecar is not None and code == 0:
                (folder / "sidecar.stop").touch()
                grace = (
                    15
                    if not gpu
                    else max(0.1, min(15, available - (time.monotonic() - started) - 2))
                )
                try:
                    sidecar.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    if gpu:
                        raise BudgetExhausted(
                            "GPU budget exhausted during final snapshot capture"
                        ) from None
                    raise
    finally:
        if in_chain:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        else:
            stop_group(process)
        stop_group(sidecar)
        wall = time.monotonic() - started
        record(out, stage, code, wall)
        pending.unlink(missing_ok=True)
        sanitize_log(out / f"{stage}.log")
    if gpu and (code in (124, 137, -9) or wall >= available):
        raise BudgetExhausted(f"GPU budget timeout at {stage}")
    if code != 0:
        raise RuntimeError(f"{stage} exit={code}; see {stage}.log")
    subprocess.run(
        validation, env={**env, "CUDA_VISIBLE_DEVICES": ""}, check=True, stdin=subprocess.DEVNULL
    )
    write(receipt, {str(p): digest(p) for p in outputs(out, stage)})
    log(out, f"{stage} complete; wall_s={wall:.3f}")


def not_run(out, reason):
    from c12.c12_common import early_interpretation
    from c12.c12_jobs import early_rows, seed_acceptance
    from c12.summarize_c12 import summarize

    pending = [s for s in STAGES if not (out / f"{s}.complete.json").exists()]
    for stage in pending:
        record(out, stage, "NOT RUN", 0)
    candidates = {}
    for seed in SEEDS:
        affected = [s for s in (f"{seed}_eval", f"{seed}_dn", "bc_eval") if s in pending]
        candidates[seed] = (
            {"status": "NOT RUN", "accepted": False, "stages": affected}
            if affected
            else seed_acceptance(out, seed)
        )
    rows = early_rows(out)
    result = {
        "status": "NOT RUN",
        "accepted": False,
        "overall": "NOT RUN",
        "stages": pending,
        "reason": reason,
        "candidates": candidates,
        "rules": RULES,
        "early_snapshots": {"rows": rows, "interpretation": early_interpretation(rows)},
    }
    write(out / "budget_status.json", result)
    write(out / "acceptance.json", result)
    summarize(out)


def cpu_chain(out, seed):
    # This entire process tree inherits nice, affinity, and one-thread BLAS settings.
    os.nice(max(0, 10 - os.getpriority(os.PRIO_PROCESS, 0)))
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        index = SEEDS.index(seed) * 2
        selected = allowed[index : index + 2] or allowed[:2]
        os.sched_setaffinity(0, selected)
    for kind in SNAP_KINDS:
        run_stage(out, f"{seed}_{kind}", in_chain=True)


def snapshot_coordinator(out):
    if any(not (out / f"{s}_train.complete.json").is_file() for s in SEEDS):
        raise ValueError("Both training receipts are required before CPU snapshots")
    children = []
    try:
        os.nice(10)
        for stage in ("bc_snap_eval", "bc_snap_dn"):
            run_stage(out, stage, in_chain=True)
        for seed in SEEDS:
            stream = (out / f"{seed}_snap_chain.log").open("a")
            try:
                # Same process group as coordinator, so STOP kills descendants too.
                child = subprocess.Popen(
                    cpu_command("-m", "c12.driver", "cpu-chain", out, seed),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                )
                children.append(child)
            finally:
                stream.close()
        while any(p.poll() is None for p in children):
            if (out / "STOP").exists():
                raise Stopped
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError("CPU snapshot seed chain failed")
            time.sleep(0.1)
        if any(p.wait() != 0 for p in children):
            raise RuntimeError("CPU snapshot seed chain failed")
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()


def run(out, smoke):
    from c12.prepare_c12 import prepare

    prepare(out, smoke)
    recover_pending(out)
    if (out / "budget_status.json").exists():
        log(out, "Budget already exhausted; NOT RUN")
        return
    monitor = chain = None
    try:
        if (out / "STOP").exists():
            raise Stopped
        if os.environ.get("MONITOR_SH") and not smoke:
            directory = out / "monitor"
            directory.mkdir(exist_ok=True)
            with (directory / "monitor.log").open("a") as stream:
                monitor = subprocess.Popen(
                    ["bash", os.environ["MONITOR_SH"], str(directory), "2", "0"],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
        for stage in ("s0_train", "s1_train"):
            run_stage(out, stage)
        with (out / "snap_chain.log").open("a") as stream:
            chain = subprocess.Popen(
                cpu_command("-m", "c12.driver", "snap-chain", out),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        for stage in ("s0_eval", "s1_eval", "bc_eval", "s0_dn", "s1_dn"):
            run_stage(out, stage)
        while chain.poll() is None:
            if (out / "STOP").exists():
                raise Stopped
            time.sleep(0.1)
        if chain.wait() != 0:
            raise RuntimeError("CPU snapshot chain failed; see snap_chain.log")
        for stage in ("acceptance", "summary"):
            run_stage(out, stage)
        log(out, "ALL DONE")
    except BudgetExhausted as error:
        stop_group(chain)
        chain = None
        recover_pending(out)
        not_run(out, str(error))
        log(out, str(error))
    except Stopped:
        log(out, "STOP present; exiting cleanly")
    finally:
        stop_group(chain)
        stop_group(monitor)
        for path in out.glob("*snap_chain.log"):
            sanitize_log(path)
        if monitor is not None:
            sanitize_log(out / "monitor/monitor.log")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    mode, out = sys.argv[1], Path(relative(sys.argv[2]))
    try:
        if mode == "validate":
            validate(out, sys.argv[3])
        elif mode == "sidecar":
            from c11.snapshots import Sidecar

            Sidecar(out, read(out / "manifest.json")["snapshot_interval"]).run()
        elif mode == "cpu-chain":
            cpu_chain(out, sys.argv[3])
        elif mode == "snap-chain":
            snapshot_coordinator(out)
        elif mode == "run":
            run(out, os.environ.get("SMOKE", "0") == "1")
        else:
            raise ValueError("Unknown driver mode")
    except Stopped:
        pass
