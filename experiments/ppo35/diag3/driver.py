"""Resumable sequential execution with persistent GPU accounting and STOP."""

import csv
import math
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from diag.common import ROOT, portable
from diag.driver import stop
from protocol100 import actor_hashes, digest, read, write

HERE = Path(__file__).resolve().parent
STAGES = ("restore_a", "restore_b", "actions_a", "actions_b", "dn_a", "dn_b", "summary")
GPU = {"actions_a", "actions_b"}


def outputs(out, stage):
    if stage.startswith("restore_"):
        actor = out / stage / "actor"
        return [
            out / f"{stage}_check.json",
            actor / "model_card.json",
            *[actor / name for name in read(actor / "model_card.json")["files"]],
        ]
    if stage.startswith("dn_"):
        return [out / f"{stage}.json"]
    if stage.startswith("actions_"):
        return [out / f"{stage}.json", out / f"{stage}.npz"]
    return [out / "summary.json", out / "summary.md"]


def main():
    out = Path(sys.argv[1])
    prereg = ROOT / "runs/c10_diag3/README.md"
    expected = (prereg.parent / "README.md.sha256").read_text().split()[0]
    if digest(prereg) != expected:
        raise ValueError("README SHA mismatch")
    smoke = os.environ["SMOKE"] == "1"
    budget = float(os.environ["GPU_BUDGET"])
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("Invalid GPU_BUDGET")
    bc = os.environ.get("BC_ACTOR", read(ROOT / "runs/c10/manifest.json")["inputs"]["BC_ACTOR"])
    bc = portable(str(Path(bc).resolve()))
    c10, original = "runs/c10/ppo_c10/actor", "runs/c10_diag/restore/actor"
    source_files = [
        ROOT / "runs/c10/eval.yaml",
        ROOT / "runs/c10_diag/summary.json",
        ROOT / "runs/c10/dn_c10.json",
        ROOT / "runs/t3c/dn_bc.json",
        ROOT / "runs/c10_diag/dn_restore.json",
        Path(os.environ["MEASURE_PY"]).resolve(),
    ]
    source_files += [
        ROOT / f"runs/c10_diag/actions_{name}.{suffix}"
        for name in ("bc", "c10", "restore")
        for suffix in ("json", "npz")
    ]
    identity = portable(
        {
            "readme_sha256": expected,
            "smoke": smoke,
            "episodes": int(os.environ["EPISODES"]),
            "lanes": int(os.environ["LANES"]),
            "actors": {
                "bc": {"path": bc, "hashes": actor_hashes(bc)},
                "c10": {"path": c10, "hashes": actor_hashes(c10)},
                "restore": {"path": original, "hashes": actor_hashes(original)},
            },
            "inputs_sha256": {str(p): digest(p) for p in source_files},
        }
    )
    identity_path = out / "inputs.json"
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError("Diagnostic inputs changed; use a fresh output directory")
    write(identity_path, identity)
    ledger = out / "stages.tsv"

    def record(stage, code, wall):
        exists = ledger.exists()
        with ledger.open("a") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            if not exists:
                writer.writerow(("stage", "end_utc", "exit", "wall_s", "gpu"))
            writer.writerow(
                (
                    stage,
                    datetime.now(UTC).isoformat(),
                    code,
                    f"{wall:.9f}",
                    int(stage in GPU and not smoke),
                )
            )
            stream.flush()
            os.fsync(stream.fileno())

    def used():
        if not ledger.exists():
            return 0
        with ledger.open() as stream:
            return sum(
                float(r["wall_s"])
                for r in csv.DictReader(stream, delimiter="\t")
                if r["gpu"] == "1"
            )

    pending = out / "active_stage.json"
    if pending.exists():
        row = read(pending)
        elapsed = max(0, time.time() - row["started"])
        if row["remaining"] is not None:
            elapsed = min(elapsed, row["remaining"])
        record(row["stage"], "INTERRUPTED", elapsed)
        pending.unlink()
    monitor = active = None

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
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
        last = os.environ.get("STOP_AFTER", "summary")
        if last not in STAGES:
            raise ValueError("Invalid STOP_AFTER")
        for stage in STAGES[: STAGES.index(last) + 1]:
            if (out / "STOP").exists():
                print("STOP present", flush=True)
                return
            receipt = out / f"{stage}.complete.json"
            if receipt.exists():
                if read(receipt) != {str(p): digest(p) for p in outputs(out, stage)}:
                    raise ValueError(f"Completed output changed: {stage}")
                print(f"skip {stage}", flush=True)
                continue
            gpu = stage in GPU and not smoke
            remaining = max(0, budget - used()) if gpu else None
            if gpu and remaining <= 2:
                for name in STAGES[STAGES.index(stage) :]:
                    record(name, "NOT RUN", 0)
                write(out / "budget_status.json", {"status": "NOT RUN", "stage": stage})
                return
            env = dict(os.environ)
            cwd = ROOT
            if not gpu:
                env["CUDA_VISIBLE_DEVICES"] = ""
            if stage.startswith("restore_"):
                args = [
                    str(HERE / "make_restore_readout.py"),
                    str(out / stage),
                    c10 if stage == "restore_a" else original,
                    bc,
                ]
            elif stage.startswith("dn_"):
                name = stage.removeprefix("dn_")
                cwd = Path(env["MEASURE_REPO"]).resolve()
                args = [
                    str(Path(env["MEASURE_PY"]).resolve()),
                    str((out / f"{stage}.json").resolve()),
                    str((out / f"restore_{name}/actor").resolve()),
                ]
                env["PYTHONPATH"] = str(cwd / "python")
            elif stage.startswith("actions_"):
                name = stage.removeprefix("actions_")
                args = [
                    str(HERE.parent / "diag/eval_actions.py"),
                    str(out),
                    "runs/c10/eval.yaml",
                    str(out / f"restore_{name}/actor"),
                    name,
                ]
            else:
                args = [str(HERE / "summarize3.py"), str(out)]
            command = [os.environ["PY"], *args]
            if gpu:
                command = [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=2s",
                    f"{remaining - 2:.9f}s",
                    *command,
                ]
            started = time.monotonic()
            code = "INTERRUPTED"
            write(pending, {"stage": stage, "started": time.time(), "remaining": remaining})
            print(f"start {stage}", flush=True)
            try:
                with (out / f"{stage}.log").open("a") as stream:
                    active = subprocess.Popen(
                        command,
                        cwd=cwd,
                        env=env,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    while active.poll() is None:
                        if (out / "STOP").exists():
                            stop(active)
                            return
                        time.sleep(0.2)
                    code = active.returncode
            finally:
                stop(active)
                active = None
                wall = time.monotonic() - started
                record(stage, code, wall)
                pending.unlink(missing_ok=True)
                log = out / f"{stage}.log"
                content = log.read_text()
                content = content.replace(str(ROOT), ".")
                content = content.replace(str(ROOT.parent), "..")
                log.write_text(content)
            if gpu and (code in (124, 137, -9) or wall >= remaining):
                for name in STAGES[STAGES.index(stage) :]:
                    record(name, "NOT RUN", 0)
                write(out / "budget_status.json", {"status": "NOT RUN", "stage": stage})
                return
            if code != 0:
                raise RuntimeError(f"{stage} exit={code}; see {stage}.log")
            for path in outputs(out, stage):
                if path.suffix == ".json":
                    write(path, portable(read(path)))
            write(receipt, {str(p): digest(p) for p in outputs(out, stage)})
            print(f"complete {stage}: {wall:.3f}s", flush=True)
    finally:
        stop(active)
        stop(monitor)


if __name__ == "__main__":
    main()
