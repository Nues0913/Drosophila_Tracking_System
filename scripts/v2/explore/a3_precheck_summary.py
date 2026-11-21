"""Summarize held-out metrics, plastic gains and closed-loop traces for #138 A3."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from statistics import mean

import torch
from flytrack_v2.model import load_policy

PAIR_KINDS = ("odor", "wind", "intensity")
PAIR_FIELDS = ("slope", "student_rms", "teacher_rms")
CONTACTS = ("target", "wall")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
_CONTACT = re.compile(
    rf"\s*during occlusion, (target|wall): (\d+); at the hit, student speed ({_NUMBER}), "
    rf"teacher label speed ({_NUMBER}) m/s,"
)
_MOTION = re.compile(r"^== .+ (curved|wander): collisions (\d+)/(\d+), phase (\{[^}]+\})")
_TRACK = re.compile(rf"\s*occluded, (.+?):\s*({_NUMBER})% of steps, turn sign agrees ({_NUMBER}),")


def _mean(values):
    return mean(values) if values else None


def held_out_metrics(path: Path, first: int, last: int) -> dict:
    rows = []
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if "update" in row and first <= row["update"] <= last:
                        rows.append(row)
    out = {"nll": _mean([r["held_out_nll"] for r in rows if r.get("held_out_nll") is not None])}
    for kind in PAIR_KINDS:
        keys = [f"held_out_pair_{kind}_present_{field}" for field in PAIR_FIELDS]
        records = [tuple(row.get(key) for key in keys) for row in rows]
        out[kind] = {
            field: _mean([record[i] for record in records if record[i] is not None])
            for i, field in enumerate(PAIR_FIELDS)
        }
        out[kind]["error_reduction"] = _mean(
            [
                2 * slope - (student / teacher) ** 2
                for slope, student, teacher in records
                if slope is not None and student is not None and teacher
            ]
        )
    return out


@torch.no_grad()
def plastic_gains(path: Path) -> dict | None:
    if not (path / "model_card.json").is_file():
        return None
    policy = load_policy(path, device="cpu")
    if not policy.config.plastic_labels:
        return None
    core = policy.core
    gains = core.gains()[core.plastic_index]
    ratios = gains / policy.config.gain_init
    tolerance = {"rtol": 8 * torch.finfo(gains.dtype).eps, "atol": 0.0}
    return {
        "min": float(ratios.min()),
        "median": float(torch.quantile(ratios, 0.5)),
        "max": float(ratios.max()),
        "at_cap": int(torch.isclose(gains, core.plastic_hi, **tolerance).sum()),
        "at_floor": int(torch.isclose(gains, core.plastic_lo, **tolerance).sum()),
    }


def collision_trace(path: Path) -> dict | None:
    if not path.is_file():
        return None
    motions = {}
    current = None
    pending = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := _CONTACT.match(line):
            kind, count, student, teacher = match.groups()
            pending[kind] = {
                "episodes": int(count),
                "student_speed_m_s": float(student),
                "teacher_speed_m_s": float(teacher),
            }
        elif match := _MOTION.match(line):
            motion, collisions, episodes, phase = match.groups()
            phases = ast.literal_eval(phase)
            during = phases.get("during")
            complete = during == sum(record["episodes"] for record in pending.values())
            by_contact = {
                kind: pending.get(
                    kind,
                    {
                        "episodes": 0 if complete else None,
                        "student_speed_m_s": None,
                        "teacher_speed_m_s": None,
                    },
                )
                for kind in CONTACTS
            }
            current = motions[motion] = {
                "episodes": int(episodes),
                "collisions": int(collisions),
                "phases": phases,
                "by_contact": {
                    kind: row["episodes"] if during == int(collisions) else None
                    for kind, row in by_contact.items()
                },
                "during_occlusion": by_contact,
                "occluded_tracking": None,
            }
            pending = {}
        elif current is not None and (match := _TRACK.match(line)):
            rule, share, agreement = match.groups()
            if current["occluded_tracking"] is None:
                current["occluded_tracking"] = {}
            current["occluded_tracking"][rule.strip()] = {
                "step_share": float(share) / 100,
                "turn_sign_agreement": float(agreement),
            }
    return {
        "path": str(path),
        "motions": motions or None,
    }


def _fmt(value) -> str:
    return "null" if value is None else f"{value:.5g}"


def _collision_cell(trace: dict | None) -> str:
    if trace is None:
        return "null"
    parts = []
    for motion, data in (trace["motions"] or {}).items():
        targets = _fmt(data["by_contact"]["target"])
        walls = _fmt(data["by_contact"]["wall"])
        during = data["during_occlusion"]
        details = [
            f"{kind} {_fmt(during[kind]['episodes'])} at "
            f"{_fmt(during[kind]['student_speed_m_s'])}/{_fmt(during[kind]['teacher_speed_m_s'])}"
            for kind in CONTACTS
        ]
        tracking = [
            f"{rule} {value['turn_sign_agreement']:.0%}"
            for rule, value in (data["occluded_tracking"] or {}).items()
        ]
        parts.append(
            f"{motion} {data['collisions']}/{data['episodes']}, target/wall {targets}/{walls}, "
            f"during {', '.join(details)}, agreement {', '.join(tracking) or 'null'}"
        )
    return f"{'<br>'.join(parts) or 'null'} ([trace](<{trace['path']}>))"


def _write_markdown(path: Path, rows: list[dict], first: int, last: int) -> None:
    lines = [
        "# A3 precheck",
        "",
        f"Held-out updates {first}–{last}. Pair figures give slope, student RMS, teacher RMS "
        "and error reduction. Collision speeds are student/teacher medians in m/s.",
        "",
        "| Row | Odor turn | Wind turn | Intensity speed | NLL | Plastic min/median/max and "
        "cap/floor | Closed loop |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        held = row["held_out"]
        pairs = [
            " / ".join(_fmt(held[kind][key]) for key in (*PAIR_FIELDS, "error_reduction"))
            for kind in PAIR_KINDS
        ]
        gains = row["plastic_gains"]
        gain_cell = ""
        if gains is not None:
            ratios = " / ".join(_fmt(gains[key]) for key in ("min", "median", "max"))
            gain_cell = f"{ratios}, {gains['at_cap']}/{gains['at_floor']}"
        collision = _collision_cell(row["collision_trace"])
        cells = [row["row"], *pairs, _fmt(held["nll"]), gain_cell, collision]
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outroot", type=Path)
    parser.add_argument(
        "--updates", type=int, nargs=2, default=(81, 100), metavar=("FIRST", "LAST")
    )
    args = parser.parse_args()
    first, last = args.updates
    if not 1 <= first <= last:
        parser.error("Expected an increasing range of positive update numbers")
    outroot = args.outroot.resolve()
    rows = []
    folders = [path for path in outroot.glob("scale_*") if path.is_dir()]
    folders.sort(key=lambda path: (float(path.name.removeprefix("scale_")), path.name))
    if (outroot / "a2").is_dir():
        folders.insert(0, outroot / "a2")
    for folder in folders:
        rows.append(
            {
                "row": folder.name,
                "held_out": held_out_metrics(folder / "metrics.jsonl", first, last),
                "plastic_gains": None if folder.name == "a2" else plastic_gains(folder / "final"),
                "collision_trace": collision_trace(folder / "collision_trace.txt"),
            }
        )
    summary = {"held_out_updates": [first, last], "rows": rows}
    (outroot / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _write_markdown(outroot / "summary.md", rows, first, last)
    print(f"Wrote summary.json and summary.md for {len(rows)} rows (updates {first}–{last})")


if __name__ == "__main__":
    main()
