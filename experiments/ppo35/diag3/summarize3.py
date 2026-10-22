"""Read JSON/npz only; report preregistered numbers and paired sign tests."""

import math
import os
import sys
from pathlib import Path

import numpy as np
from diag.common import ROOT
from diag.summarize import METRICS, distribution
from protocol100 import read, write


def partition(a, b):
    if a >= 0.40:
        return 1
    if b >= 0.40:
        return 2
    if a <= 0.15 and b <= 0.15:
        return 3
    return 4


def sign_test(better, worse):
    n = better + worse
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / 2**n)


def paired(actual, reference, metric, smoke=False):
    def indexed(rows):
        result = {r["environment_seed"]: r[metric] for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate episode seeds")
        return result

    a, b = indexed(actual), indexed(reference)
    if not smoke and a.keys() != b.keys():
        raise ValueError("Episode seeds differ")
    seeds = sorted(a.keys() & b.keys())
    if not seeds:
        raise ValueError("No paired episodes")
    better = sum(a[s] > b[s] for s in seeds)
    worse = sum(a[s] < b[s] for s in seeds)
    return {
        "episodes": len(seeds),
        "better": better,
        "worse": worse,
        "ties": len(seeds) - better - worse,
        "p_value": sign_test(better, worse),
    }


def main():
    out = Path(sys.argv[1])
    smoke = os.environ.get("SMOKE") == "1"
    old = ROOT / "runs/c10_diag"
    previous = read(old / "summary.json")
    summary = {"actors": {}, "paired": {}, "consistency": {}}
    reports = {}
    for name in ("bc", "c10", "restore", "a", "b"):
        folder = out if name in ("a", "b") else old
        report = read(folder / f"actions_{name}.json")
        reports[name] = report["normal"]["episodes"]
        with np.load(folder / f"actions_{name}.npz", allow_pickle=False) as data:
            actions = distribution(data["applied_action"], report["v_max"], report["omega_max"])
        dn_path = {
            "bc": ROOT / "runs/t3c/dn_bc.json",
            "c10": ROOT / "runs/c10/dn_c10.json",
            "restore": old / "dn_restore.json",
            "a": out / "dn_a.json",
            "b": out / "dn_b.json",
        }[name]
        dn = read(dn_path)["dn"]
        row = {
            **{k: report["normal"][k] for k in METRICS},
            "omega_median": actions["omega"]["quantiles"]["50"],
            "omega_negative_fraction": actions["omega_negative_fraction"],
            **{k: dn[k]["voltage_median"] for k in ("DNa01", "DNa02")},
        }
        summary["actors"][name] = row
        if name not in ("a", "b"):
            expected = {
                **previous["scenarios"][name]["normal"],
                "omega_median": previous["actions"][name]["omega"]["quantiles"]["50"],
                "omega_negative_fraction": previous["actions"][name]["omega_negative_fraction"],
                **{k: previous["dn"][name][k]["voltage_median"] for k in ("DNa01", "DNa02")},
            }
            summary["consistency"][name] = {
                "skipped": int(smoke),
                "differences": []
                if smoke
                else [
                    {"metric": k, "actual": row[k], "expected": expected[k]}
                    for k in row
                    if row[k] != expected[k]
                ],
            }
    for name in ("a", "b"):
        summary["paired"][name] = {
            ref: {
                k: paired(reports[name], reports[ref], k, smoke)
                for k in ("success", "tracked_fraction")
            }
            for ref in ("bc", "c10", "restore")
        }
    a, b = (summary["actors"][n]["tracked_fraction"] for n in ("a", "b"))
    summary["partition"] = {"rule": partition(a, b), "a": a, "b": b}
    write(out / "summary.json", summary)
    lines = [
        "# C10 診斷 (iii) 數字",
        "",
        "| Actor | 成功率 | 碰撞率 | 方位誤差 | 追蹤比例 | ω 中位數 | ω < 0 比例 | DNa01 | DNa02 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["actors"].items():
        lines.append(f"| {name} | " + " | ".join(map(str, row.values())) + " |")
    lines += [
        "",
        "| Actor | 配對 | 指標 | 回合數 | 較好 | 較差 | 相同 | p |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, refs in summary["paired"].items():
        for ref, metrics in refs.items():
            for metric, row in metrics.items():
                lines.append(
                    f"| {name} | {ref} | {metric} | " + " | ".join(map(str, row.values())) + " |"
                )
    lines += [
        "",
        "| 條號 | iii-a | iii-b |",
        "|---:|---:|---:|",
        f"| {partition(a, b)} | {a} | {b} |",
        "",
        "| Actor | 一致性略過 | 差異數 |",
        "|---|---:|---:|",
    ]
    for name, row in summary["consistency"].items():
        lines.append(f"| {name} | {row['skipped']} | {len(row['differences'])} |")
        for diff in row["differences"]:
            lines.append(f"\n{diff['metric']}: {diff['actual']}, {diff['expected']}\n")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
