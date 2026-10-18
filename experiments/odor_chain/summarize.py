"""Collect every group's stage measurements into summary.json and summary.md.

Usage: summarize.py BASE   (OUT or OUT/smoke)
Criteria (plan, part 2): BC odor/vision ratio >= 0.05, normal success no more
than 10 points below G0, and the BC DN-health gate passes.
"""

import json
import sys
from pathlib import Path

import torch

STAGES = ("init", "vision", "odor", "bc")


def load(path):
    return json.loads(Path(path).read_text())


def fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    return f"{value:.{digits}g}" if isinstance(value, float) else str(value)


def identical_checkpoints(left, right):
    a = torch.load(left, map_location="cpu", weights_only=True)
    b = torch.load(right, map_location="cpu", weights_only=True)
    return all(
        a[part].keys() == b[part].keys()
        and all(torch.equal(value, b[part][key]) for key, value in a[part].items())
        for part in ("actor", "decoder")
    )


def main():
    base = Path(sys.argv[1])
    manifest = load(base / "configs/manifest.json")
    summary = {
        "base": str(base),
        "seed": manifest["seed"],
        "smoke": manifest["smoke"],
        "groups": {},
    }
    for group in manifest["groups"]:
        row = {"stages": {}}
        for stage in STAGES:
            measured = load(base / "measure" / f"{group}_{stage}.json")
            entry = {
                "odor_input_gain": measured["odor_input_gain"],
                "ratio": measured["odor_to_vision_ratio"],
                "no_odor": measured["no_odor"],
                "odor_swap_lr": measured["odor_swap_lr"],
                "no_vision": measured["no_vision"],
                "DNa01_voltage": measured["dn"]["DNa01"]["voltage_median"],
                "DNa02_voltage": measured["dn"]["DNa02"]["voltage_median"],
                "DNa01_slope": measured["dn"]["DNa01"]["slope_median"],
                "DNa02_slope": measured["dn"]["DNa02"]["slope_median"],
            }
            if stage in ("vision", "odor"):
                validation = load(base / group / stage / "validation" / "metrics.json")
                entry["decoder_bearing_mae_deg"] = validation["bearing_mae_deg"]
                entry["decoder_presence_accuracy"] = validation["presence_accuracy"]
                entry["decoder_passed"] = validation["passed"]
                entry["final_loss"] = load(base / group / stage / "training_metrics.json")[-1][
                    "loss"
                ]
            row["stages"][stage] = entry
        health = load(base / group / "bc" / "dn_health" / "metrics.json")
        episodes = {
            lesion: load(base / group / "bc" / "episodes" / f"{lesion}.json")["scenarios"]
            for lesion in ("none", "odor")
        }
        row["bc"] = {
            "dn_health_passed": health["passed"],
            "dn_health_checks": health["checks"],
            "normal_success": episodes["none"]["normal"]["success"],
            "normal_bearing_deg": episodes["none"]["normal"]["mean_bearing_error_deg"],
            "occlusion_success": episodes["none"]["occlusion"]["success"],
            "occluded_bearing_deg": episodes["none"]["occlusion"]["occlusion_bearing_error_deg"],
            "occluded_bearing_deg_odor_lesion": episodes["odor"]["occlusion"][
                "occlusion_bearing_error_deg"
            ],
            "normal_success_odor_lesion": episodes["odor"]["normal"]["success"],
        }
        row["bc"]["odor_lesion_occluded_bearing_delta_deg"] = (
            row["bc"]["occluded_bearing_deg_odor_lesion"] - row["bc"]["occluded_bearing_deg"]
        )
        summary["groups"][group] = row
    reference = (
        summary["groups"]["G0"]["bc"]["normal_success"] if "G0" in summary["groups"] else None
    )
    for row in summary["groups"].values():
        bc = row["bc"]
        row["criteria"] = {
            "bc_ratio_ge_0.05": row["stages"]["bc"]["ratio"] >= 0.05,
            "normal_success_within_10pt_of_G0": (
                bc["normal_success"] >= reference - 0.10 if reference is not None else None
            ),
            "bc_gate_passed": bc["dn_health_passed"],
        }
        row["criteria"]["all"] = all(row["criteria"].values()) if reference is not None else None
    # Vision pretraining sees zero odor input, so G1 and G2 should match (bitwise
    # on deterministic CPU; GPU kernels may differ slightly).
    summary["G1_G2_vision_identical"] = (
        identical_checkpoints(
            base / "G1/vision/pretraining_checkpoint.pt",
            base / "G2/vision/pretraining_checkpoint.pt",
        )
        if {"G1", "G2"} <= summary["groups"].keys()
        else None
    )
    (base / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    lines = [
        f"seed={manifest['seed']}; smoke={manifest['smoke']}",
        "",
        "| 組 | 階段 | 增益 | 比例 | 左右交換 | 去氣味 | 去視覺 | DNa01 V | DNa02 V "
        "| decoder MAE° |",
    ]
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for group, row in summary["groups"].items():
        for stage, entry in row["stages"].items():
            lines.append(
                f"| {group} | {stage} | {fmt(entry['odor_input_gain'])} | {fmt(entry['ratio'])} "
                f"| {fmt(entry['odor_swap_lr'])} | {fmt(entry['no_odor'])} "
                f"| {fmt(entry['no_vision'])} | {fmt(entry['DNa01_voltage'])} "
                f"| {fmt(entry['DNa02_voltage'])} "
                f"| {fmt(entry.get('decoder_bearing_mae_deg'))} |"
            )
    lines += [
        "",
        "| 組 | normal 成功率 | occlusion 成功率 | 遮蔽中方位誤差° | odor lesion 後° | Δ° "
        "| BC gate | 比例≥0.05 | 成功率 ≥ G0−10pt | 全部通過 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for group, row in summary["groups"].items():
        bc, criteria = row["bc"], row["criteria"]
        lines.append(
            f"| {group} | {fmt(bc['normal_success'])} | {fmt(bc['occlusion_success'])} "
            f"| {fmt(bc['occluded_bearing_deg'])} | {fmt(bc['occluded_bearing_deg_odor_lesion'])} "
            f"| {fmt(bc['odor_lesion_occluded_bearing_delta_deg'])} "
            f"| {fmt(bc['dn_health_passed'])} | {fmt(criteria['bc_ratio_ge_0.05'])} "
            f"| {fmt(criteria['normal_success_within_10pt_of_G0'])} | {fmt(criteria['all'])} |"
        )
    lines += ["", f"G1/G2 vision pretrain 完全相同：{fmt(summary['G1_G2_vision_identical'])}"]
    (base / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
