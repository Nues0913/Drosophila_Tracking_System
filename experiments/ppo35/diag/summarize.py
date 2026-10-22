"""Read only JSON/npz inputs; emit numeric summaries without interpretation."""

import sys
from pathlib import Path

import numpy as np
from common import ROOT
from protocol100 import SCENARIOS, read, write

CLASSES = ("停止", "原地轉", "遠離目標", "其他")
METRICS = ("success", "collision_rate", "mean_bearing_error_deg", "tracked_fraction")


def classify(v, omega, initial, final, v_max, omega_max):
    median_v = np.median(v)
    if median_v < 0.1 * v_max:
        return CLASSES[0]
    if median_v < 0.5 * v_max and np.median(np.abs(omega)) >= 0.5 * omega_max:
        return CLASSES[1]
    if final - initial > 0.5:
        return CLASSES[2]
    return CLASSES[3]


def distribution(actions, v_max, omega_max):
    def stats(values):
        return {
            "mean": float(np.mean(values)),
            "quantiles": {str(q): float(np.percentile(values, q)) for q in (5, 25, 50, 75, 95)},
        }

    v, omega = actions.T
    return {
        "v": stats(v),
        "omega": stats(omega),
        "abs_omega": stats(np.abs(omega)),
        "omega_positive_fraction": float(np.mean(omega > 0)),
        "omega_negative_fraction": float(np.mean(omega < 0)),
        "omega_near_zero_fraction": float(np.mean(np.abs(omega) < 0.05 * omega_max)),
        "v_max": v_max,
        "omega_max": omega_max,
    }


def scenario_report(path):
    policies = read(path)["policies"]
    if len(policies) != 1:
        raise ValueError("Expected one actor per evaluation")
    return next(iter(policies.values()))


def differences(actual, expected):
    """Compare every episode field, retaining all missing/extra seeds and values."""
    a = {row["environment_seed"]: row for row in actual}
    b = {row["environment_seed"]: row for row in expected}
    return [
        {
            "environment_seed": seed,
            "metric": key,
            "actual": a.get(seed, {}).get(key),
            "expected": b.get(seed, {}).get(key),
        }
        for seed in sorted(a.keys() | b.keys())
        for key in sorted(a.get(seed, {}).keys() | b.get(seed, {}).keys())
        if a.get(seed, {}).get(key) != b.get(seed, {}).get(key)
    ]


def markdown(summary):
    lines = [
        "# C10 診斷數字",
        "",
        "## 四情境",
        "",
        "| Actor | 情境 | 成功率 | 碰撞率 | 方位誤差（度） | 追蹤比例 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, scenarios in summary["scenarios"].items():
        for scenario, row in scenarios.items():
            lines.append(
                f"| {name} | {scenario} | " + " | ".join(str(row[key]) for key in METRICS) + " |"
            )
    lines += ["", "## DN", "", "| Actor | DN | 電壓中位數 | slope 中位數 |", "|---|---|---:|---:|"]
    for name, kinds in summary["dn"].items():
        for kind, row in kinds.items():
            lines.append(f"| {name} | {kind} | {row['voltage_median']} | {row['slope_median']} |")
    lines += [
        "",
        "## 動作分佈",
        "",
        "| Actor | 動作 | 平均 | P5 | P25 | P50 | P75 | P95 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["actions"].items():
        for key in ("v", "omega", "abs_omega"):
            stats = row[key]
            values = [stats["mean"], *stats["quantiles"].values()]
            lines.append(f"| {name} | {key} | " + " | ".join(map(str, values)) + " |")
    lines += [
        "",
        "| Actor | v_max | omega_max | ω > 0 比例 | ω < 0 比例 | 小轉速比例 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["actions"].items():
        keys = (
            "v_max",
            "omega_max",
            "omega_positive_fraction",
            "omega_negative_fraction",
            "omega_near_zero_fraction",
        )
        lines.append(f"| {name} | " + " | ".join(str(row[k]) for k in keys) + " |")
    lines += ["", "## 回合分類", "", "| Actor | 分類 | 回合數 | 比例 |", "|---|---|---:|---:|"]
    for name, row in summary["actions"].items():
        for label, values in row["classes"].items():
            lines.append(f"| {name} | {label} | {values['count']} | {values['fraction']} |")
    lines += ["", "## 逐回合一致性", "", "| Actor | 狀態 | 差異數 |", "|---|---|---:|"]
    for name, row in summary["consistency"].items():
        lines.append(f"| {name} | {row['status']} | {len(row['differences'])} |")
        if row["differences"]:
            lines += ["", "| seed | 指標 | 實際 | 參考 |", "|---:|---|---:|---:|"]
            for diff in row["differences"]:
                lines.append(
                    f"| {diff['environment_seed']} | {diff['metric']} | "
                    f"{diff['actual']} | {diff['expected']} |"
                )
            lines.append("")
    thresholds = summary["tracking_thresholds"]
    lines += [
        "",
        "## normal 追蹤比例門檻",
        "",
        "| 下門檻 | 上門檻 | 還原值 | 區間 |",
        "|---:|---:|---:|---|",
        "| " + " | ".join(str(thresholds[k]) for k in ("low", "high", "value", "interval")) + " |",
        "",
    ]
    return "\n".join(lines)


def main():
    out = Path(sys.argv[1])
    baseline = {
        name: scenario_report(ROOT / f"runs/c10/eval_{name}.json") for name in ("bc", "c10")
    }
    baseline["restore"] = scenario_report(out / "eval_restore.json")
    summary = {"scenarios": {}, "dn": {}, "actions": {}, "consistency": {}}
    contract = read(out / "restore/actor/sensor_contract.json")
    for name, scenarios in baseline.items():
        summary["scenarios"][name] = {
            scenario: {metric: scenarios[scenario][metric] for metric in METRICS}
            for scenario in SCENARIOS
        }
        dn_path = {
            "bc": ROOT / "runs/t3c/dn_bc.json",
            "c10": ROOT / "runs/c10/dn_c10.json",
            "restore": out / "dn_restore.json",
        }[name]
        summary["dn"][name] = {
            kind: {
                key: read(dn_path)["dn"][kind][key] for key in ("voltage_median", "slope_median")
            }
            for kind in ("DNa01", "DNa02")
        }
        report = read(out / f"actions_{name}.json")
        with np.load(out / f"actions_{name}.npz", allow_pickle=False) as data:
            actions = distribution(data["applied_action"], contract["v_max"], contract["omega_max"])
            counts = dict.fromkeys(CLASSES, 0)
            labels = []
            for seed, initial in zip(
                data["episode_seeds"], data["initial_distance_m"], strict=True
            ):
                mask = data["environment_seed"] == seed
                v, omega = data["applied_action"][mask].T
                label = classify(
                    v,
                    omega,
                    initial,
                    data["target_distance_m"][mask][-1],
                    contract["v_max"],
                    contract["omega_max"],
                )
                counts[label] += 1
                labels.append({"environment_seed": int(seed), "class": label})
            actions["classes"] = {
                key: {"count": n, "fraction": n / sum(counts.values())} for key, n in counts.items()
            }
            actions["episodes"] = labels
            summary["actions"][name] = actions
        skipped = report["smoke"] and name != "restore"
        diff = (
            []
            if skipped
            else differences(report["normal"]["episodes"], scenarios["normal"]["episodes"])
        )
        summary["consistency"][name] = {
            "status": "SKIPPED_SMOKE" if skipped else ("DIFFERENT" if diff else "EXACT"),
            "differences": diff,
        }
    tracked = baseline["restore"]["normal"]["tracked_fraction"]
    summary["tracking_thresholds"] = {
        "low": 0.15,
        "high": 0.40,
        "value": tracked,
        "interval": "[0.40,1]"
        if tracked >= 0.40
        else ("[0,0.15]" if tracked <= 0.15 else "(0.15,0.40)"),
    }
    write(out / "summary.json", summary)
    (out / "summary.md").write_text(markdown(summary))


if __name__ == "__main__":
    main()
