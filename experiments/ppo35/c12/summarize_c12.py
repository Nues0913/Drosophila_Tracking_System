"""Read result files only; emit numeric Chinese tables and fixed rule labels."""

import sys
from pathlib import Path

from c11.summarize_c11 import (
    curriculum_ranges,
    display,
    optional,
    stage_label,
    training_records,
)

from c12.c12_common import ROOT, SEEDS, portable, read, write

METRICS = {
    "tracked_fraction": "追蹤比例",
    "omega_median": "ω 中位數",
    "turn_bias": "turn_bias",
    "core_bias_max_abs": "max abs(core bias)",
    "DNa02_voltage_median": "DNa02 電壓中位數",
    "DNa01_slope_median": "DNa01 slope 中位數",
}
CHECKS = {
    "collision_all_scenarios": "四情境碰撞率",
    "normal_success": "normal 成功率",
    "normal_bearing_sign_test": "normal bearing 精確二項檢定",
    "dn": "DN slope median",
}


def collect(out):
    manifest = read(out / "manifest.json")
    data = {
        "smoke": manifest["smoke"],
        "acceptance": optional(out / "acceptance.json", {"status": "NOT RUN", "candidates": {}}),
        "curriculum_ranges": curriculum_ranges(
            manifest.get("configs", {}).get("ppo_c12_s0.yaml", {}).get("train", {})
        ),
        "snap_scheduling": manifest.get("snap_scheduling"),
        "c12": {},
        "c11": {},
    }
    for seed in SEEDS:
        data["c12"][seed] = {
            "training_records": training_records(out / seed, data["curriculum_ranges"])
        }
        folder = ROOT / ("runs/c11" if seed == "s0" else "runs/c11_s1")
        prior = read(folder / "manifest.json")
        stages = curriculum_ranges(prior["configs"]["ppo_c11.yaml"]["train"])
        data["c11"][seed] = {
            "training_records": training_records(folder, stages),
            "curriculum_ranges": stages,
            "acceptance": read(folder / "acceptance.json"),
        }
    # Exact step matching: no interpolation or nearest-step substitution.
    for seed in SEEDS:
        rows = data["c12"][seed]["training_records"]
        data["c12"][seed]["same_step_c11"] = {
            prior_seed: [
                next(
                    (
                        r
                        for r in data["c11"][prior_seed]["training_records"]
                        if r["step"] == row["step"]
                    ),
                    {"step": row["step"]},
                )
                for row in rows
            ]
            for prior_seed in SEEDS
        }
    return portable(data)


def details(value, prefix=""):
    result = {}
    for name, item in value.items():
        if name in {"test", "passed", "status"}:
            continue
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(item, dict):
            result.update(details(item, path))
        else:
            result[path] = item
    return result


def render(data):
    acceptance = data["acceptance"]
    comparisons = [(f"C12 ({s})", acceptance.get("candidates", {}).get(s, {})) for s in SEEDS]
    comparisons += [(f"C11 ({s})", data["c11"][s]["acceptance"]) for s in SEEDS]
    header = " | ".join(name for name, _ in comparisons)
    separator = " | ".join("---" for _ in comparisons)
    early = acceptance.get("early_snapshots", {})
    lines = [
        "# C12 protocol100" + (" CPU SMOKE ONLY" if data["smoke"] else ""),
        "",
        f"狀態：{acceptance.get('status', 'NOT RUN')}；"
        f"accepted：{acceptance.get('accepted', False)}；"
        f"整體判定：{acceptance.get('overall', 'NOT RUN')}",
        "",
        f"早期快照固定規則：{early.get('interpretation', '不下結論')}",
        "",
        f"| 終點條件 | {header} |",
        f"| --- | {separator} |",
    ]
    for key, label in CHECKS.items():
        values = " | ".join(
            r.get("checks", {}).get(key, {}).get("status", "NOT RUN") for _, r in comparisons
        )
        lines.append(f"| {label} | {values} |")
    for key in ("gates_passed", "accepted"):
        lines.append(f"| {key} | " + " | ".join(display(r.get(key)) for _, r in comparisons) + " |")
    lines += ["", f"| 條件數值 | {header} |", f"| --- | {separator} |"]
    for key in CHECKS:
        columns = [details(r.get("checks", {}).get(key, {})) for _, r in comparisons]
        for metric in dict.fromkeys(k for column in columns for k in column):
            lines.append(
                f"| {key}.{metric} | " + " | ".join(display(c.get(metric)) for c in columns) + " |"
            )
    for model, stages in [
        ("C12", data["curriculum_ranges"]),
        *[(f"C11 ({s})", data["c11"][s]["curriculum_ranges"]) for s in SEEDS],
    ]:
        lines += ["", f"| {model} curriculum 階段與步數範圍 |", "| --- |"]
        lines += [f"| {stage_label(stage)} |" for stage in stages]
    for seed in SEEDS:
        names = [f"C12 ({seed})", "C11 (s0)", "C11 (s1)"]
        columns = [
            f"{name} {label}" for name in names for label in ("curriculum 階段", *METRICS.values())
        ]
        lines += [
            "",
            f"## C12 ({seed}) 同步數快照",
            "",
            "| 步數 | " + " | ".join(columns) + " |",
            "| ---: | " + " | ".join("---:" for _ in columns) + " |",
        ]
        for index, row in enumerate(data["c12"][seed]["training_records"]):
            values = [row, *[data["c12"][seed]["same_step_c11"][s][index] for s in SEEDS]]
            cells = [display(r.get(k)) for r in values for k in ("curriculum_stage", *METRICS)]
            lines.append(f"| {row['step']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| 早期快照 seed | 邊界 | 實際步數 | ω 中位數 | 追蹤比例 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in early.get("rows", []):
        lines.append(
            "| "
            + " | ".join(
                display(row[k])
                for k in ("seed", "boundary", "step", "omega_median", "tracked_fraction")
            )
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
