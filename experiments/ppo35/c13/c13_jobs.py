"""C13 decisions and thin adapters for the C12/C11 stage implementations."""

import sys
from functools import partial
from pathlib import Path

from c11.summarize_c11 import curriculum_ranges, optional, training_records
from c12 import c12_jobs

from c13.c13_common import RULES, SEEDS, overall, read, seeds, snapshot_interpretation

seed_acceptance = partial(c12_jobs.seed_acceptance, seed_guard=seeds)
execute = partial(c12_jobs.execute, package="c13")


def snapshot_rows(out):
    manifest = read(out / "manifest.json")
    result = []
    for seed in SEEDS:
        folder = out / seed
        config = manifest.get("configs", {}).get(f"ppo_c13_{seed}.yaml", {})
        stages = curriculum_ranges(config.get("train", {}))
        index = optional(folder / "snapshots/index.json", {"snapshots": []})
        boundaries = {r["step"]: r["boundary"] for r in index["snapshots"]}
        for row in training_records(folder, stages):
            result.append({"seed": seed, "boundary": boundaries.get(row["step"], 0), **row})
    return result


def snapshot_result(out):
    rows = snapshot_rows(out)
    return {"rows": rows, "interpretations": snapshot_interpretation(rows)}


def acceptance_result(out):
    manifest = read(out / "manifest.json")
    candidates = {seed: seed_acceptance(out, seed) for seed in SEEDS}
    return {
        "protocol": manifest["protocol"],
        "smoke": manifest["smoke"],
        "status": "COMPLETE",
        "episodes": manifest["episodes"],
        "seed_start": manifest["seed_start"],
        "bc_actor": manifest["inputs"]["BC_ACTOR"],
        "candidates": candidates,
        "accepted": all(r["accepted"] for r in candidates.values()),
        "overall": overall({s: r["gates_passed"] for s, r in candidates.items()}),
        "snapshots": snapshot_result(out),
        "rules": RULES,
    }


if __name__ == "__main__":
    execute(Path(sys.argv[1]), sys.argv[2])
