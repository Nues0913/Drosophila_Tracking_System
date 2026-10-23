"""C12 stage adapters; reuse C11's trainer, evaluation, snapshots and DN."""

import sys
from pathlib import Path

from accept100 import compare
from c11.jobs import evaluate, measure, snap_eval, train
from c11.snapshots import build_all
from protocol100 import validate_eval

from c12.c12_common import RULES, SEEDS, early_interpretation, overall, read, seeds, write


def snapshot_actors(out, seed):
    m = read(out / "manifest.json")
    return [
        {"step": 0, "actor": m["inputs"]["BC_ACTOR"]},
        *read(out / seed / "snap_build.json")["actors"],
    ]


def early_rows(out):
    result = []
    for seed in SEEDS:
        index_path = out / seed / "snapshots/index.json"
        eval_path = out / seed / "snap_eval.json"
        if not index_path.exists() or not eval_path.exists():
            continue
        index = read(index_path)
        evaluations = {r["step"]: r for r in read(eval_path)["rows"]}
        for boundary in RULES["early_snapshots"]["boundaries"]:
            captures = [r for r in index["snapshots"] if r["boundary"] == boundary]
            if len(captures) == 1 and captures[0]["step"] in evaluations:
                row = evaluations[captures[0]["step"]]
                result.append(
                    {
                        "seed": seed,
                        "boundary": boundary,
                        **{k: row[k] for k in ("step", "omega_median", "tracked_fraction")},
                    }
                )
    return result


def seed_acceptance(out, seed):
    m = read(out / "manifest.json")
    seeds(m["seed_start"], m["episodes"], m["smoke"])
    bc_actor = m["inputs"]["BC_ACTOR"]
    actor = str(out / seed / "ppo_c11/actor")
    baseline = read(out / "eval_bc.json")
    candidate = read(out / seed / "eval_c11.json")
    for report, who in ((baseline, bc_actor), (candidate, actor)):
        validate_eval(
            report,
            [who],
            episodes=m["episodes"],
            seed_start=m["seed_start"],
            smoke=m["smoke"],
            env=m["configs"]["eval.yaml"]["env"],
        )
    return compare(
        baseline["policies"][bc_actor],
        candidate["policies"][actor],
        read(out / seed / "dn_c11.json"),
        episodes=m["episodes"],
        seed_start=m["seed_start"],
        smoke=m["smoke"],
    )


def acceptance_result(out):
    m = read(out / "manifest.json")
    candidates = {seed: seed_acceptance(out, seed) for seed in SEEDS}
    rows = early_rows(out)
    return {
        "protocol": m["protocol"],
        "smoke": m["smoke"],
        "status": "COMPLETE",
        "episodes": m["episodes"],
        "seed_start": m["seed_start"],
        "bc_actor": m["inputs"]["BC_ACTOR"],
        "candidates": candidates,
        "accepted": all(r["accepted"] for r in candidates.values()),
        "overall": overall({s: r["gates_passed"] for s, r in candidates.items()}),
        "early_snapshots": {"rows": rows, "interpretation": early_interpretation(rows)},
        "rules": RULES,
    }


def execute(out, stage):
    if stage in ("acceptance", "summary"):
        if stage == "acceptance":
            write(out / "acceptance.json", acceptance_result(out))
        else:
            from c12.summarize_c12 import summarize

            summarize(out)
        return
    m = read(out / "manifest.json")
    if stage == "bc_eval":
        evaluate(out / "s0", "bc", seed_guard=seeds)
        (out / "s0/eval_bc.json").replace(out / "eval_bc.json")
        return
    if stage == "bc_snap_eval":
        snap_eval(out, items=[{"step": 0, "actor": m["inputs"]["BC_ACTOR"]}], seed_guard=seeds)
        (out / "snap_eval.json").replace(out / "snap_eval_bc.json")
        return
    if stage == "bc_snap_dn":
        measure(out, m["inputs"]["BC_ACTOR"], out / "dn_bc.json")
        return
    seed, kind = stage.split("_", 1)
    if seed not in SEEDS:
        raise ValueError("Unknown C12 stage")
    folder = out / seed
    if kind == "train":
        extra = ["--resume"] if (folder / "ppo_c11/training_checkpoint.pt").exists() else []
        train(folder, extra)
    elif kind == "eval":
        evaluate(folder, "c11", seed_guard=seeds)
    elif kind == "dn":
        measure(folder, folder / "ppo_c11/actor", folder / "dn_c11.json")
    elif kind == "snap_build":
        build_all(folder)
    elif kind == "snap_eval":
        snap_eval(folder, items=read(folder / "snap_build.json")["actors"], seed_guard=seeds)
        rows = read(out / "snap_eval_bc.json")["rows"] + read(folder / "snap_eval.json")["rows"]
        write(folder / "snap_eval.json", {"rows": rows})
    elif kind == "snap_dn":
        rows = [{"step": 0, "actor": m["inputs"]["BC_ACTOR"], "report": str(out / "dn_bc.json")}]
        for item in read(folder / "snap_build.json")["actors"]:
            destination = folder / "snapshots" / f"dn_step_{item['step']}.json"
            measure(folder, item["actor"], destination)
            rows.append({**item, "report": str(destination)})
        write(folder / "snap_dn.json", {"rows": rows})
    else:
        raise ValueError("Unknown C12 stage")


if __name__ == "__main__":
    execute(Path(sys.argv[1]), sys.argv[2])
