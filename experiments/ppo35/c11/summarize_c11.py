"""Read existing results; write numeric Chinese C11 profile comparisons."""

import sys
from decimal import Decimal
from pathlib import Path

from c11_common import ROOT, manifest_profile, portable, read, write


def optional(path, default=None):
    return read(path) if path.exists() else default


def curriculum_ranges(train):
    total = train.get("total_steps", 0)
    result, start = [], 0
    for index, config in enumerate(train.get("curriculum", []), 1):
        end = float(Decimal(str(config["until"])) * total)
        end = int(end) if end.is_integer() else end
        result.append(
            {
                **config,
                "stage": index,
                "start_step": start,
                "end_step": end,
                "start_inclusive": index == 1,
            }
        )
        start = end
    return result


def stage_for_step(step, stages):
    if step < 0:
        raise ValueError("Negative curriculum step")
    # Match flytrack_train.ppo.curriculum_stage: progress <= until.
    for stage in stages:
        if step <= stage["end_step"]:
            return stage["stage"]
    if stages:
        raise ValueError("Step exceeds curriculum ranges")
    return None


def stage_label(stage):
    motion = stage["target_motion"]
    if isinstance(motion, list):
        motion = "、".join(motion)
    start = "[" if stage["start_inclusive"] else "("
    bounds = f"{start}{stage['start_step']:,}–{stage['end_step']:,}]"
    return f"{stage['stage']}：{motion}／{stage['plume_model']}／{stage['scenario']}（{bounds}）"


def training_records(out, stages):
    reports = {
        row["step"]: Path(row["report"])
        for row in optional(out / "snap_dn.json", {"rows": []})["rows"]
    }
    result = []
    for row in optional(out / "snap_eval.json", {"rows": []})["rows"]:
        step = row["step"]
        path = out / "snapshots" / f"dn_step_{step}.json"
        dn = optional(path if path.exists() else reports.get(step, path), {}).get("dn", {})
        result.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "step",
                        "tracked_fraction",
                        "omega_median",
                        "turn_bias",
                        "core_bias_max_abs",
                    )
                },
                "DNa02_voltage_median": dn.get("DNa02", {}).get("voltage_median"),
                "DNa01_slope_median": dn.get("DNa01", {}).get("slope_median"),
                "curriculum_stage": stage_for_step(step, stages),
            }
        )
    return result


def endpoint(out):
    evaluation = optional(out / "eval_c11.json", {})
    normal = next(iter(evaluation.get("policies", {}).values()), {}).get("normal", {})
    return {
        "normal_success": normal.get("success"),
        "normal_collision": normal.get("collision_rate"),
        "normal_bearing_error_deg": normal.get("mean_bearing_error_deg"),
    }


def c10_endpoint():
    out = ROOT / "runs/c10"
    evaluation = optional(out / "eval_c10.json", {})
    normal = next(iter(evaluation.get("policies", {}).values()), {}).get("normal", {})
    dn = optional(out / "dn_c10.json", {})
    turn_bias = None
    weights = out / "ppo_c10/actor/weights.pt"
    if weights.exists():
        import torch

        state = torch.load(weights, map_location="cpu", weights_only=True)
        turn_bias = float(state["readout.turn_bias"].abs().max())
    return {
        "step": 500_000,
        "tracked_fraction": normal.get("tracked_fraction"),
        "omega_median": None,
        "turn_bias": turn_bias,
        "core_bias_max_abs": dn.get("core_bias_max_abs"),
        "DNa02_voltage_median": dn.get("dn", {}).get("DNa02", {}).get("voltage_median"),
        "DNa01_slope_median": dn.get("dn", {}).get("DNa01", {}).get("slope_median"),
        "normal_success": normal.get("success"),
        "normal_collision": normal.get("collision_rate"),
        "normal_bearing_error_deg": normal.get("mean_bearing_error_deg"),
    }


def collect(out):
    m = read(out / "manifest.json")
    profile = manifest_profile(m)
    train = m.get("configs", {}).get("ppo_c11.yaml", {}).get("train", {})
    stages = curriculum_ranges(train)
    acceptance = optional(out / "acceptance.json", {"status": "NOT RUN", "accepted": False})
    c10_acceptance = optional(ROOT / "runs/c10/acceptance_c10_only.json")
    if c10_acceptance is None:
        c10_acceptance = optional(ROOT / "runs/c10/acceptance.json", {"status": "NOT RUN"})
    c10_acceptance = {k: v for k, v in c10_acceptance.items() if k != "note"}
    data = portable(
        {
            "run_profile": profile.name,
            "smoke": m["smoke"],
            "acceptance": acceptance,
            "c10_acceptance": c10_acceptance,
            "training_records": training_records(out, stages),
            "curriculum_ranges": stages,
            "c10_endpoint": c10_endpoint(),
            "c11_endpoint": endpoint(out),
            "snapshot_index": optional(out / "snapshots/index.json", {}),
            "final_actor_bitwise_equal": optional(out / "snap_build.json", {}).get(
                "final_actor_bitwise_equal"
            ),
            "snap_scheduling": m["snap_scheduling"],
        }
    )
    if profile.name == "s1":
        s0 = ROOT / "runs/c11"
        s0_train = read(s0 / "manifest.json")["configs"]["ppo_c11.yaml"]["train"]
        data.update(
            s0_training_records=training_records(s0, curriculum_ranges(s0_train)),
            s0_acceptance=optional(s0 / "acceptance.json", {"status": "NOT RUN"}),
            s0_endpoint=endpoint(s0),
        )
    return portable(data)


def display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def render(data):
    acceptance = data["acceptance"]
    s1 = data["run_profile"] == "s1"
    model = "C11-s1" if s1 else "C11"
    comparisons = [(model, acceptance)]
    if s1:
        comparisons.append(("C11 (s0)", data["s0_acceptance"]))
    comparisons.append(("C10", data["c10_acceptance"]))
    headers = " | ".join(name for name, _ in comparisons)
    separator = " | ".join("---" for _ in comparisons)
    lines = [
        f"# {model} protocol100",
        "",
        f"smoke: {data['smoke']}；accepted: {acceptance.get('accepted', False)}；"
        f"gates_passed: {acceptance.get('gates_passed', 'NOT RUN')}",
        "",
        f"| 條件 | {headers} |",
        f"| --- | {separator} |",
    ]
    names = {
        "collision_all_scenarios": "四情境碰撞率",
        "normal_success": "normal 成功率",
        "normal_bearing_sign_test": "normal bearing 精確二項檢定",
        "dn": "DN slope median",
    }
    for key, label in names.items():
        values = " | ".join(
            result.get("checks", {}).get(key, {}).get("status", "NOT RUN")
            for _, result in comparisons
        )
        lines.append(f"| {label} | {values} |")
    values = " | ".join(str(result.get("accepted", False)) for _, result in comparisons)
    lines.append(f"| 最終 accepted | {values} |")
    lines += ["", f"| 條件數值 | {headers} |", f"| --- | {separator} |"]

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

    for key in names:
        columns = [details(result.get("checks", {}).get(key, {})) for _, result in comparisons]
        for metric in dict.fromkeys(metric for column in columns for metric in column):
            values = " | ".join(display(column.get(metric)) for column in columns)
            lines.append(f"| {key}.{metric} | {values} |")

    stages = {stage["stage"]: stage for stage in data["curriculum_ranges"]}
    lines += ["", "| curriculum 階段與步數範圍 |", "| --- |"]
    lines += [f"| {stage_label(stage)} |" for stage in stages.values()]
    keys = (
        "tracked_fraction",
        "omega_median",
        "turn_bias",
        "core_bias_max_abs",
        "DNa02_voltage_median",
        "DNa01_slope_median",
    )
    labels = (
        "追蹤比例",
        "ω 中位數",
        "max abs(turn_bias)",
        "max abs(core bias)",
        "DNa02 電壓中位數",
        "DNa01 slope 中位數",
    )
    c10 = data["c10_endpoint"]
    s0_rows = {row["step"]: row for row in data.get("s0_training_records", [])}
    columns = [f"{model} {label}" for label in labels]
    if s1:
        columns += [f"C11 (s0) {label}" for label in labels]
    lines += [
        "",
        "| 模型 | 步數 | curriculum 階段 | " + " | ".join(columns) + " |",
        "| --- | ---: | --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for row in data["training_records"]:
        stage = stages.get(row["curriculum_stage"])
        values = [display(row[key]) for key in keys]
        if s1:
            values += [display(s0_rows.get(row["step"], {}).get(key)) for key in keys]
        lines.append(
            "| "
            + ("BC" if row["step"] == 0 else model)
            + f" | {row['step']} | {stage_label(stage) if stage else '—'} | "
            + " | ".join(values)
            + " |"
        )
    values = [display(c10[key]) for key in keys] + (["—"] * len(keys) if s1 else [])
    lines.append(f"| C10 | {c10['step']} | — | " + " | ".join(values) + " |")
    endpoints = [data["c11_endpoint"]]
    if s1:
        endpoints.append(data["s0_endpoint"])
    endpoints.append(c10)
    lines += ["", f"| normal 終點 | {headers} |", f"| --- | {separator} |"]
    for key in ("normal_success", "normal_collision", "normal_bearing_error_deg"):
        values = " | ".join(display(endpoint[key]) for endpoint in endpoints)
        lines.append(f"| {key} | {values} |")
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
