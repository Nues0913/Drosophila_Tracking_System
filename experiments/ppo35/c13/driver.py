"""C13 uses C12's scheduler, accounting, STOP, receipts and bounded CPU chains."""

import os
import signal
import sys
from functools import partial
from pathlib import Path

from c12 import driver as shared
from c12.driver import (  # noqa: F401
    BudgetExhausted,
    Stopped,
    ledger,
    outputs,
    record,
    recover_pending,
    remaining,
    stage_env,
)

from c13.c13_common import RULES, SEEDS, STAGES, read, relative, seeds, write
from c13.c13_jobs import acceptance_result as acceptance_result
from c13.c13_jobs import seed_acceptance, snapshot_result

validate = partial(shared.validate, package="c13")
validate_snap_rows = partial(shared.validate_snap_rows, seed_guard=seeds)
run_stage = partial(shared.run_stage, package="c13")
cpu_chain = partial(shared.cpu_chain, package="c13")
snapshot_coordinator = partial(shared.snapshot_coordinator, package="c13")
run = partial(shared.run, package="c13")


def not_run(out, reason):
    from c13.summarize_c13 import summarize

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
    result = {
        "status": "NOT RUN",
        "accepted": False,
        "overall": "NOT RUN",
        "stages": pending,
        "reason": reason,
        "candidates": candidates,
        "rules": RULES,
        "snapshots": snapshot_result(out),
    }
    write(out / "budget_status.json", result)
    write(out / "acceptance.json", result)
    summarize(out)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shared.interrupted)
    signal.signal(signal.SIGTERM, shared.interrupted)
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
    except ValueError as error:
        print(f"C13 stopped: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
