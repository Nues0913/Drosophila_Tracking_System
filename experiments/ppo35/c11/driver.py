"""Resumable C11 stages with one persistent GPU budget and a restricted CPU chain."""

import csv
import fcntl
import math
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from c11_common import (
    GPU_STAGES,
    HERE,
    ROOT,
    SNAP_STAGES,
    STAGES,
    actor_files,
    digest,
    read,
    relative,
    write,
)


def log(out, message):
    line = f"[{datetime.now(UTC).isoformat()}] {message}"
    print(line, flush=True)
    with (out / "run.log").open("a") as stream:
        stream.write(line + "\n")


def ledger(out):
    path = out / "stages.tsv"
    if not path.exists():
        return []
    with path.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        rows = list(csv.DictReader(stream, delimiter="\t"))
    for row in rows:
        if row["stage"] not in STAGES or not math.isfinite(float(row["wall_s"])):
            raise ValueError("Invalid stage accounting")
        if float(row["wall_s"]) < 0:
            raise ValueError("Negative stage accounting")
    return rows


def record(out, stage, code, wall):
    with (out / "stages.tsv").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.seek(0, os.SEEK_END)
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        if stream.tell() == 0:
            writer.writerow(("stage", "end_utc", "exit", "wall_s"))
        writer.writerow((stage, datetime.now(UTC).isoformat(), code, f"{wall:.9f}"))
        stream.flush()
        os.fsync(stream.fileno())


def remaining(out, manifest):
    return manifest["gpu_budget_s"] - sum(
        float(row["wall_s"]) for row in ledger(out) if row["stage"] in GPU_STAGES
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
    if stage == "c11_train":
        files = [out / "train_runtime.json", out / "snapshots/index.json"]
        files += [
            out / "ppo_c11" / n
            for n in (
                "progress.json",
                "training_metrics.json",
                "training_checkpoint.pt",
            )
        ]
        files += actor_files(out / "ppo_c11/actor")
        files += [Path(row["path"]) for row in read(out / "snapshots/index.json")["snapshots"]]
        return files
    if stage in ("c11_eval", "bc_eval"):
        return [out / f"eval_{stage.removesuffix('_eval')}.json"]
    if stage == "dn_c11":
        return [out / "dn_c11.json"]
    if stage == "snap_build":
        report = read(out / "snap_build.json")
        return [out / "snap_build.json", *map(Path, report["files_sha256"])]
    if stage == "snap_eval":
        return [out / "snap_eval.json"]
    if stage == "snap_dn":
        return [
            out / "snap_dn.json",
            *(Path(row["report"]) for row in read(out / "snap_dn.json")["rows"]),
        ]
    if stage == "acceptance":
        return [out / "acceptance.json"]
    return [out / "summary.json", out / "summary.md"]


def validate(out, stage):
    from protocol100 import actor_hashes, dn_rows, finite, validate_actor, validate_eval

    m = read(out / "manifest.json")
    if stage == "c11_train":
        from check import ppo_complete

        ppo_complete(out / "ppo_c11", m["configs"]["ppo_c11.yaml"], m["smoke"])
        import torch
        from snapshots import bitwise_equal

        before = torch.load(
            Path(m["inputs"]["BC_ACTOR"]) / "weights.pt", map_location="cpu", weights_only=True
        )
        after = torch.load(out / "ppo_c11/actor/weights.pt", map_location="cpu", weights_only=True)
        names = ("readout.turn_bias", "readout.forward_bias")
        if not bitwise_equal({k: before[k] for k in names}, {k: after[k] for k in names}):
            raise ValueError("C11 readout biases changed despite zero learning rate")
        runtime = read(out / "train_runtime.json")
        if runtime["device"] != m["device"] or runtime["deterministic"] is not False:
            raise ValueError("Training runtime differs")
        if any(
            runtime["environment"][key] is not None
            for key in (
                "DETERMINISTIC",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        ):
            raise ValueError("Training must use C10 nondeterministic settings")
        if not m["smoke"] and "4060" not in runtime["device_name"]:
            raise ValueError("Formal training requires the 4060")
        rows = read(out / "snapshots/index.json")["snapshots"]
        if (
            not rows
            or max(r["step"] for r in rows) != m["configs"]["ppo_c11.yaml"]["train"]["total_steps"]
        ):
            raise ValueError("Final snapshot missing")
    elif stage in ("c11_eval", "bc_eval"):
        actor = str(out / "ppo_c11/actor") if stage == "c11_eval" else m["inputs"]["BC_ACTOR"]
        validate_eval(
            read(outputs(out, stage)[0]),
            [actor],
            episodes=m["episodes"],
            seed_start=m["seed_start"],
            smoke=m["smoke"],
            env=m["configs"]["eval.yaml"]["env"],
        )
    elif stage == "dn_c11":
        dn_rows(read(out / "dn_c11.json"), str(out / "ppo_c11/actor"), 8 if m["smoke"] else 200)
    elif stage == "snap_build":
        report = read(out / "snap_build.json")
        if report["final_actor_bitwise_equal"] is not True:
            raise ValueError("Final actor comparison failed")
        for row in report["actors"]:
            validate_actor(row["actor"])
        for path, checksum in report["files_sha256"].items():
            if digest(path) != checksum:
                raise ValueError("Rebuilt artifact changed")
    elif stage == "snap_eval":
        from c11_common import seeds
        from jobs import snapshot_actors

        rows = read(out / "snap_eval.json")["rows"]
        if [{k: r[k] for k in ("step", "actor")} for r in rows] != snapshot_actors(out):
            raise ValueError("Snapshot evaluation actor set differs")
        for row in rows:
            finite(row)
            if row["actor_sha256"] != actor_hashes(row["actor"]) or row["device"] != "cpu":
                raise ValueError("Snapshot actor/runtime differs")
            expected = seeds(m["snap_seed_start"], m["snap_episodes"], m["smoke"])
            if row["seed_start"] != m["snap_seed_start"] or row["episodes"] != len(expected):
                raise ValueError("Snapshot evaluation protocol differs")
            if [r["environment_seed"] for r in row["normal"]["episodes"]] != expected:
                raise ValueError("Snapshot seed pairing differs")
    elif stage == "snap_dn":
        from jobs import snapshot_actors

        rows = read(out / "snap_dn.json")["rows"]
        if [{k: r[k] for k in ("step", "actor")} for r in rows] != snapshot_actors(out):
            raise ValueError("Snapshot DN actor set differs")
        for row in rows:
            dn_rows(read(row["report"]), row["actor"], 8 if m["smoke"] else 200)
    elif stage == "acceptance":
        from jobs import acceptance_result

        if read(out / "acceptance.json") != acceptance_result(out):
            raise ValueError("Acceptance result differs")
    else:
        from summarize_c11 import summarize

        summarize(out, validate_only=True)


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
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def cpu_command(*args):
    return ["prlimit", "--as=6442450944", sys.executable, *map(str, args)]


def stage_env(stage, smoke):
    env = dict(os.environ)
    if stage == "c11_train":
        env.pop("DETERMINISTIC", None)
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        env.update(DETERMINISTIC="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if smoke or stage not in GPU_STAGES:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def sanitize_log(path):
    if path.exists():
        value = path.read_text().replace(str(ROOT) + "/", "").replace(str(ROOT.parent) + "/", "../")
        path.write_text(value)


class BudgetExhausted(Exception):
    pass


class Stopped(Exception):
    pass


def run_stage(out, stage, in_chain=False):
    m = read(out / "manifest.json")
    receipt = out / f"{stage}.complete.json"
    env = stage_env(stage, m["smoke"])
    if (out / "STOP").exists():
        raise Stopped
    if receipt.exists():
        if read(receipt) != {str(p): digest(p) for p in outputs(out, stage)}:
            raise ValueError(f"Completed stage changed: {stage}")
        subprocess.run(
            cpu_command(relative(HERE / "driver.py"), "validate", out, stage),
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
    args = [relative(HERE / "jobs.py"), str(out), stage]
    command = [sys.executable, *args] if gpu else cpu_command(*args)
    pending = out / f"active_{stage}.json"
    process = sidecar = None
    code = "INTERRUPTED"
    started = time.monotonic()
    log(out, f"start {stage}; remaining_s={available}")
    write(pending, {"stage": stage, "started_unix_s": time.time(), "remaining_s": available})
    try:
        with (out / f"{stage}.log").open("a") as stream:
            if stage == "c11_train":
                (out / "sidecar.stop").unlink(missing_ok=True)
                ready = out / "sidecar.ready"
                ready.unlink(missing_ok=True)
                sidecar = subprocess.Popen(
                    [
                        "nice",
                        "-n",
                        "10",
                        *cpu_command(relative(HERE / "driver.py"), "sidecar", out),
                    ],
                    env={**env, "CUDA_VISIBLE_DEVICES": ""},
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                ready_wait = (
                    15 if not gpu else max(0, min(15, available - (time.monotonic() - started) - 5))
                )
                deadline = time.monotonic() + ready_wait
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
                (out / "sidecar.stop").touch()
                # Final link is captured even if training finished between polls.
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
        cpu_command(relative(HERE / "driver.py"), "validate", out, stage),
        env={**env, "CUDA_VISIBLE_DEVICES": ""},
        check=True,
        stdin=subprocess.DEVNULL,
    )
    write(receipt, {str(p): digest(p) for p in outputs(out, stage)})
    log(out, f"{stage} complete; wall_s={wall:.3f}")


def not_run(out, reason):
    pending = [stage for stage in STAGES if not (out / f"{stage}.complete.json").exists()]
    for stage in pending:
        record(out, stage, "NOT RUN", 0)
    result = {"status": "NOT RUN", "accepted": False, "stages": pending, "reason": reason}
    write(out / "budget_status.json", result)
    write(out / "acceptance.json", result)
    from summarize_c11 import summarize

    summarize(out)


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def cpu_chain(out):
    os.nice(10)
    if hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, allowed[-min(2, len(allowed)) :])
    for stage in SNAP_STAGES:
        run_stage(out, stage, in_chain=True)


def run(out, smoke):
    from prepare_c11 import prepare

    prepare(out, smoke)
    recover_pending(out)
    if (out / "budget_status.json").exists():
        log(out, "Budget already exhausted; NOT RUN")
        return
    monitor = chain = None
    try:
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
        run_stage(out, "c11_train")
        with (out / "snap_chain.log").open("a") as stream:
            chain = subprocess.Popen(
                cpu_command(relative(HERE / "driver.py"), "cpu-chain", out),
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                stdout=stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        for stage in ("c11_eval", "bc_eval", "dn_c11"):
            run_stage(out, stage)
        while chain.poll() is None:
            if (out / "STOP").exists():
                raise Stopped
            time.sleep(0.1)
        if chain.wait() != 0:
            raise RuntimeError("CPU snapshot chain failed; see snap_chain.log")
        sanitize_log(out / "snap_chain.log")
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
        sanitize_log(out / "snap_chain.log")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    mode, out = sys.argv[1], Path(relative(sys.argv[2]))
    try:
        if mode == "validate":
            validate(out, sys.argv[3])
        elif mode == "sidecar":
            from snapshots import Sidecar

            Sidecar(out, read(out / "manifest.json")["snapshot_interval"]).run()
        elif mode == "cpu-chain":
            cpu_chain(out)
        elif mode == "run":
            run(out, os.environ.get("SMOKE", "0") == "1")
        else:
            raise ValueError("Unknown driver mode")
    except Stopped:
        pass
