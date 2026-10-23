"""Read files only; report numeric C13/C12 comparisons and fixed decisions."""

import statistics
import sys
from pathlib import Path

from c11.summarize_c11 import (
    curriculum_ranges,
    display,
    optional,
    stage_for_step,
    stage_label,
    training_records,
)
from c12.summarize_c12 import CHECKS, METRICS, details

from c13.anchor_distance import distance_records
from c13.c13_common import ROOT, SEEDS, portable, read, write

SNAP_METRICS = {
    **METRICS,
    "actor_distance_l2": "‖θ−θ_BC‖₂（快照重算）",
    "bc_anchor_distance": "bc_anchor_distance（訓練紀錄）",
    "training_update_step": "最近 update 步數",
}
TRAIN_METRICS = ("bc_anchor_loss", "bc_anchor_distance", "actor_grad_norm", "entropy")


def metric_statistics(records):
    result = {}
    for key in TRAIN_METRICS:
        values = [r[key] for r in records if r.get(key) is not None]
        result[key] = {
            "count": len(values),
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return result


def seed_records(folder, config, bc_actor):
    stages = curriculum_ranges(config.get("train", {}))
    rows = {r["step"]: r for r in training_records(folder, stages)}
    distances = distance_records(folder, config, bc_actor) if config else []
    for distance in distances:
        step = distance["step"]
        rows.setdefault(step, {"step": step, "curriculum_stage": stage_for_step(step, stages)})
        rows[step].update(distance)
    return [rows[step] for step in sorted(rows)]


def collect(out):
    manifest = read(out / "manifest.json")
    prior = read(ROOT / "runs/c12/manifest.json")
    data = {
        "smoke": manifest["smoke"],
        "acceptance": optional(out / "acceptance.json", {"status": "NOT RUN", "candidates": {}}),
        "c12_acceptance": read(ROOT / "runs/c12/acceptance.json"),
        "snap_scheduling": manifest.get("snap_scheduling"),
        "c13": {},
        "c12": {},
    }
    for seed in SEEDS:
        config = manifest.get("configs", {}).get(f"ppo_c13_{seed}.yaml", {})
        prior_config = prior["configs"][f"ppo_c12_{seed}.yaml"]
        rows = seed_records(out / seed, config, manifest.get("inputs", {}).get("BC_ACTOR"))
        data["c13"][seed] = {
            "training_records": rows,
            "curriculum_ranges": curriculum_ranges(config.get("train", {})),
            "training_metrics": metric_statistics(
                optional(out / seed / "ppo_c11/training_metrics.json", [])
            ),
        }
        data["c12"][seed] = {
            "training_records": seed_records(
                ROOT / "runs/c12" / seed, prior_config, prior["inputs"]["BC_ACTOR"]
            ),
            "curriculum_ranges": curriculum_ranges(prior_config["train"]),
        }
    for seed in SEEDS:
        data["c13"][seed]["same_step_c12"] = {
            other: [
                next(
                    (r for r in data["c12"][other]["training_records"] if r["step"] == row["step"]),
                    {"step": row["step"]},
                )
                for row in data["c13"][seed]["training_records"]
            ]
            for other in SEEDS
        }
    return portable(data)


def render(data):
    acceptance = data["acceptance"]
    comparisons = [(f"C13 ({s})", acceptance.get("candidates", {}).get(s, {})) for s in SEEDS]
    comparisons += [
        (f"C12 ({s})", data["c12_acceptance"].get("candidates", {}).get(s, {})) for s in SEEDS
    ]
    header = " | ".join(name for name, _ in comparisons)
    separator = " | ".join("---" for _ in comparisons)
    lines = [
        "# C13 protocol100" + (" CPU SMOKE ONLY" if data["smoke"] else ""),
        "",
        f"狀態：{acceptance.get('status', 'NOT RUN')}；"
        f"accepted：{acceptance.get('accepted', False)}；"
        f"整體判定：{acceptance.get('overall', 'NOT RUN')}",
        "",
    ]
    for label, value in acceptance.get("snapshots", {}).get("interpretations", {}).items():
        lines += [f"{label}：{display(value)}", ""]
    lines += [f"| 終點條件 | {header} |", f"| --- | {separator} |"]
    for key, label in CHECKS.items():
        lines.append(
            f"| {label} | "
            + " | ".join(
                r.get("checks", {}).get(key, {}).get("status", "NOT RUN") for _, r in comparisons
            )
            + " |"
        )
    for key in ("gates_passed", "accepted"):
        lines.append(f"| {key} | " + " | ".join(display(r.get(key)) for _, r in comparisons) + " |")
    lines += ["", f"| 條件數值 | {header} |", f"| --- | {separator} |"]
    for key in CHECKS:
        columns = [details(r.get("checks", {}).get(key, {})) for _, r in comparisons]
        for metric in dict.fromkeys(k for column in columns for k in column):
            lines.append(
                f"| {key}.{metric} | " + " | ".join(display(c.get(metric)) for c in columns) + " |"
            )
    for seed in SEEDS:
        lines += ["", f"| C13 ({seed}) curriculum 階段與步數範圍 |", "| --- |"]
        lines += [f"| {stage_label(stage)} |" for stage in data["c13"][seed]["curriculum_ranges"]]
        names = [f"C13 ({seed})", "C12 (s0)", "C12 (s1)"]
        columns = [
            f"{name} {label}"
            for name in names
            for label in ("curriculum 階段", *SNAP_METRICS.values())
        ]
        lines += [
            "",
            f"## C13 ({seed}) 同步數快照",
            "",
            "| 步數 | " + " | ".join(columns) + " |",
            "| ---: | " + " | ".join("---:" for _ in columns) + " |",
        ]
        for index, row in enumerate(data["c13"][seed]["training_records"]):
            values = [row, *[data["c13"][seed]["same_step_c12"][s][index] for s in SEEDS]]
            cells = [display(r.get(k)) for r in values for k in ("curriculum_stage", *SNAP_METRICS)]
            lines.append(f"| {row['step']} | " + " | ".join(cells) + " |")
        lines += [
            "",
            f"| C13 ({seed}) 訓練紀錄 | 數量 | 中位數 | 最小值 | 最大值 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for metric, values in data["c13"][seed]["training_metrics"].items():
            lines.append(
                f"| {metric} | "
                + " | ".join(display(values[k]) for k in ("count", "median", "min", "max"))
                + " |"
            )
    return "\n".join(lines) + "\n"


def summarize(out, validate_only=False):
    data = collect(out)
    text = render(data)
    if validate_only:
        if read(out / "summary.json") != data or (out / "summary.md").read_text() != text:
            raise ValueError("Summary differs from source files")
    else:
        write(out / "summary.json", data)
        temporary = out / "summary.md.writing"
        temporary.write_text(text)
        temporary.replace(out / "summary.md")


if __name__ == "__main__":
    summarize(Path(sys.argv[1]))
