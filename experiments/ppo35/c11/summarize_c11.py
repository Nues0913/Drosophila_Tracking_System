"""Read existing files only; write numeric Chinese C11/C10 summaries."""

import sys
from pathlib import Path

from c11_common import ROOT, portable, read, write


def optional(path, default=None):
    return read(path) if path.exists() else default


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
        "normal_success": normal.get("success"),
        "normal_collision": normal.get("collision_rate"),
        "normal_bearing_error_deg": normal.get("mean_bearing_error_deg"),
    }


def collect(out):
    m = read(out / "manifest.json")
    rows = optional(out / "snap_eval.json", {"rows": []})["rows"]
    dn = {
        row["step"]: read(row["report"])["dn"]["DNa02"]["voltage_median"]
        for row in optional(out / "snap_dn.json", {"rows": []})["rows"]
    }
    table = [
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
            "DNa02_voltage_median": dn.get(row["step"]),
        }
        for row in rows
    ]
    acceptance = optional(out / "acceptance.json", {"status": "NOT RUN", "accepted": False})
    c10_acceptance = optional(ROOT / "runs/c10/acceptance_c10_only.json")
    if c10_acceptance is None:
        c10_acceptance = optional(ROOT / "runs/c10/acceptance.json", {"status": "NOT RUN"})
    c10_acceptance = {k: v for k, v in c10_acceptance.items() if k != "note"}
    evaluation = optional(out / "eval_c11.json", {})
    normal = next(iter(evaluation.get("policies", {}).values()), {}).get("normal", {})
    return portable(
        {
            "smoke": m["smoke"],
            "acceptance": acceptance,
            "c10_acceptance": c10_acceptance,
            "training_records": table,
            "c10_endpoint": c10_endpoint(),
            "c11_endpoint": {
                "normal_success": normal.get("success"),
                "normal_collision": normal.get("collision_rate"),
                "normal_bearing_error_deg": normal.get("mean_bearing_error_deg"),
            },
            "snapshot_index": optional(out / "snapshots/index.json", {}),
            "final_actor_bitwise_equal": optional(out / "snap_build.json", {}).get(
                "final_actor_bitwise_equal"
            ),
            "snap_scheduling": m["snap_scheduling"],
        }
    )


def display(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def render(data):
    acceptance = data["acceptance"]
    lines = [
        "# C11 protocol100",
        "",
        f"smoke: {data['smoke']}；accepted: {acceptance.get('accepted', False)}；"
        f"gates_passed: {acceptance.get('gates_passed', 'NOT RUN')}",
        "",
        "| 條件 | C11 | C10 |",
        "| --- | --- | --- |",
    ]
    names = {
        "collision_all_scenarios": "四情境碰撞率",
        "normal_success": "normal 成功率",
        "normal_bearing_sign_test": "normal bearing 精確二項檢定",
        "dn": "DN slope median",
    }
    for key, label in names.items():
        left = acceptance.get("checks", {}).get(key, {})
        right = data["c10_acceptance"].get("checks", {}).get(key, {})
        lines.append(
            f"| {label} | {left.get('status', 'NOT RUN')} | {right.get('status', 'NOT RUN')} |"
        )
    lines.append(
        f"| 最終 accepted | {acceptance.get('accepted', False)} | "
        f"{data['c10_acceptance'].get('accepted', False)} |"
    )
    lines += ["", "| 條件數值 | C11 | C10 |", "| --- | --- | --- |"]

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
        left = details(acceptance.get("checks", {}).get(key, {}))
        right = details(data["c10_acceptance"].get("checks", {}).get(key, {}))
        for metric in dict.fromkeys([*left, *right]):
            lines.append(
                f"| {key}.{metric} | {display(left.get(metric))} | {display(right.get(metric))} |"
            )
    lines += [
        "",
        "| 模型 | 步數 | 追蹤比例 | ω 中位數 | "
        "max abs(turn_bias) | max abs(core bias) | DNa02 電壓中位數 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    keys = (
        "step",
        "tracked_fraction",
        "omega_median",
        "turn_bias",
        "core_bias_max_abs",
        "DNa02_voltage_median",
    )
    c10 = data["c10_endpoint"]
    for row in data["training_records"]:
        lines.append(
            "| "
            + ("BC" if row["step"] == 0 else "C11")
            + " | "
            + " | ".join(display(row[k]) for k in keys)
            + " |"
        )
    lines.append("| C10 | " + " | ".join(display(c10[k]) for k in keys) + " |")
    lines += ["", "| normal 終點 | C11 | C10 |", "| --- | ---: | ---: |"]
    for key in ("normal_success", "normal_collision", "normal_bearing_error_deg"):
        lines.append(f"| {key} | {display(data['c11_endpoint'][key])} | {display(c10[key])} |")
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
