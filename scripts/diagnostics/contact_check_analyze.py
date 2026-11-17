"""Compare Gazebo contacts with the geometric collision flag (#116 G0f).

Usage: python scripts/diagnostics/contact_check_analyze.py EXPORT.json OUT.json
EXPORT.json comes from contact_check_export.py. Per episode (from its first active event to its
"completed" event), with all times on the simulation clock:
  flag_s           the "completed" event's stamp when it reports collision (scenario_manager's
                   tick that set the flag), else null
  geometric_s      first agent pose sample in the episode at which the same geometric rule holds
                   (agent pose with the latest target pose at or before it); it isolates the
                   tick/sampling delay from the rule itself
  first_contact_s  first non-floor contact after the episode start, and what it touched
  contacts_before_flag   contact messages with start < stamp < flag_s (the acceptance count),
                   with the objects touched; for episodes without a flag, contacts inside the
                   active window
  agent pose at the first contact: heading, incidence to the touched wall (degrees from its normal)
  and pitch (degrees, positive nose down; the Gazebo agent tips forward when it brakes), and the
  largest |pitch| in the episode's active window
Deltas are first_contact_s - flag_s and first_contact_s - geometric_s (negative = the contact
came first). Contacts up to POST_S after the episode end are kept to report contacts after a flag.
"""

from __future__ import annotations

import bisect
import json
import math
import sys
from pathlib import Path

POST_S = 1.0


def touched(name: str) -> str:
    """Model name of a scoped collision name (e.g. north_wall::wall::wall -> north_wall)."""
    return name.split("::")[0]


def episodes(events: list[dict]) -> list[dict]:
    first, end = {}, {}
    for event in events:
        episode = event["episode_id"]
        if event["active"]:
            first.setdefault(episode, event)
        elif episode in first and event["parameters"].get("phase") == "completed":
            end.setdefault(episode, event)
    rows = []
    for episode, start_event in sorted(first.items()):
        if episode not in end:
            continue
        p = start_event["parameters"]
        finish = end[episode]
        rows.append(
            {
                "episode_id": episode,
                "seed": start_event["seed"],
                "start_s": float(p["episode_start_s"]),
                "end_s": finish["stamp"],
                "collision": bool(finish["parameters"]["collision"]),
                "parameters": p,
            }
        )
    return rows


def geometric(agent, target, p) -> bool:
    half = float(p["arena_size_m"]) / 2
    cx, cy = float(p["arena_center_x"]), float(p["arena_center_y"])
    radius = float(p["agent_radius_m"])
    x, y = agent[1], agent[2]
    if not (cx - half + radius <= x <= cx + half - radius) or not (
        cy - half + radius <= y <= cy + half - radius
    ):
        return True
    if target is not None and math.hypot(x - target[1], y - target[2]) <= radius + float(
        p["target_radius_m"]
    ):
        return True
    for obstacle in p.get("obstacles", []):
        if "radius" in obstacle:
            if math.hypot(x - obstacle["x"], y - obstacle["y"]) <= radius + obstacle["radius"]:
                return True
        else:
            nx = min(max(x, obstacle["xmin"]), obstacle["xmax"])
            ny = min(max(y, obstacle["ymin"]), obstacle["ymax"])
            if math.hypot(x - nx, y - ny) <= radius:
                return True
    return False


def first_geometric(agent, target, row):
    target_t = [sample[0] for sample in target]
    for sample in agent:
        if sample[0] < row["start_s"]:
            continue
        if sample[0] > row["end_s"]:
            break
        index = bisect.bisect_right(target_t, sample[0]) - 1
        latest = target[index] if index >= 0 and target[index][0] >= row["start_s"] else None
        if geometric(sample, latest, row["parameters"]):
            return sample[0]
    return None


WALL_NORMALS = {
    "north_wall": (0, 1),
    "east_wall": (1, 0),
    "south_wall": (0, -1),
    "west_wall": (-1, 0),
}


def pose_at(agent, t):
    index = bisect.bisect_right([sample[0] for sample in agent], t) - 1
    return agent[index] if index >= 0 else None


def contact_pose(agent, first) -> dict:
    pose = pose_at(agent, first["stamp"])
    if pose is None:
        return {}
    out = {"heading_deg": math.degrees(pose[3])}
    if len(pose) > 5:
        out["z_m"], out["pitch_deg"] = pose[4], math.degrees(pose[5])
    walls = [touched(x["other"]) for x in first["contacts"] if touched(x["other"]) in WALL_NORMALS]
    if walls:
        nx, ny = WALL_NORMALS[walls[0]]
        cosine = math.cos(pose[3]) * nx + math.sin(pose[3]) * ny
        out["incidence_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    return out


def analyse(export: dict) -> dict:
    rows = []
    contacts = export["contacts"]
    for row in episodes(export["events"]):
        start, end = row["start_s"], row["end_s"]
        flag = end if row["collision"] else None
        limit = flag if flag is not None else end
        window = [c for c in contacts if start < c["stamp"] <= end + POST_S]
        before = [c for c in window if c["stamp"] < limit]
        first = window[0] if window else None
        geometric_s = first_geometric(export["agent"], export["target"], row)
        objects = sorted({touched(x["other"]) for c in before for x in c["contacts"]})
        result = {
            "episode_id": row["episode_id"],
            "seed": row["seed"],
            "start_s": start,
            "end_s": end,
            "flag_s": flag,
            "geometric_s": geometric_s,
            "first_contact_s": first["stamp"] if first else None,
            "first_contact_objects": sorted({touched(x["other"]) for x in first["contacts"]})
            if first
            else [],
            "first_contact_links": sorted({first["link"]}) if first else [],
            "contacts_before_flag": len(before),
            "contacts_before_flag_objects": objects,
        }
        if first:
            result["first_contact_pose"] = contact_pose(export["agent"], first)
        active = [s for s in export["agent"] if start <= s[0] <= end and len(s) > 5]
        if active:
            result["max_abs_pitch_deg"] = max(abs(math.degrees(s[5])) for s in active)
        if first and flag is not None:
            result["contact_minus_flag_s"] = first["stamp"] - flag
        if first and geometric_s is not None:
            result["contact_minus_geometric_s"] = first["stamp"] - geometric_s
        rows.append(result)
    flagged = [r for r in rows if r["flag_s"] is not None]
    summary = {
        "episodes": len(rows),
        "geometric_flags": len(flagged),
        "flags_with_contact": sum(r["first_contact_s"] is not None for r in flagged),
        "episodes_with_contact_before_flag": sum(r["contacts_before_flag"] > 0 for r in rows),
        "contact_messages_before_flag": sum(r["contacts_before_flag"] for r in rows),
        "objects_before_flag": sorted({o for r in rows for o in r["contacts_before_flag_objects"]}),
        "unflagged_with_contact": sum(
            r["flag_s"] is None and r["contacts_before_flag"] > 0 for r in rows
        ),
    }
    for key in ("contact_minus_flag_s", "contact_minus_geometric_s"):
        values = sorted(r[key] for r in rows if key in r)
        if values:
            summary[key] = {
                "n": len(values),
                "min": values[0],
                "median": values[len(values) // 2]
                if len(values) % 2
                else (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2,
                "max": values[-1],
            }
    return {"summary": summary, "episodes": rows}


def main() -> None:
    export, output = Path(sys.argv[1]), Path(sys.argv[2])
    result = analyse(json.loads(export.read_text(encoding="utf-8")))
    output.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))


if __name__ == "__main__":
    main()
