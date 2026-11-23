"""Pair the v2 G1 Gazebo episodes with their 2D references, episode by episode (#140).

Usage: python scripts/v2/gazebo_g1_compare.py GAZEBO_DIR REFERENCE_DIR OUT.json
Both directories hold {variant}/seed_{seed}/{scenario}/episodes.jsonl for each target motion
under GAZEBO_DIR/<run>/ and REFERENCE_DIR/<run>/ (run = one orchestrator call, e.g. curved,
wander). Episodes are paired by (run, variant, seed, scenario, environment_seed).

Reported per pair: Gazebo valid_episode, success in both, collision (collision_rate > 0) in
both, and whether the outcome agrees (same success and same collision). The formal-v1 G1 scope
reported the same episode-level agreement (formal_v1_report.md: 47/48 on each host).
"""

import json
import sys
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cells(root: Path):
    for path in sorted(root.glob("*/*/seed_*/*/episodes.jsonl")):
        run, variant, seed, scenario = path.relative_to(root).parts[:4]
        yield (run, variant, int(seed.removeprefix("seed_")), scenario), rows(path)


def main() -> None:
    gazebo, reference, output = (Path(a) for a in sys.argv[1:4])
    two_d = {}
    for key, episodes in cells(reference):
        for e in episodes:
            two_d[key + (int(e["environment_seed"]),)] = e
    pairs, missing = [], []
    for key, episodes in cells(gazebo):
        for g in episodes:
            k = key + (int(g["environment_seed"]),)
            r = two_d.get(k)
            if r is None:
                missing.append(list(k))
                continue
            gs, rs = bool(g.get("success")), bool(r.get("success"))
            gc, rc = (g.get("collision_rate") or 0) > 0, (r.get("collision_rate") or 0) > 0
            pairs.append(
                {
                    "run": k[0],
                    "variant": k[1],
                    "seed": k[2],
                    "scenario": k[3],
                    "environment_seed": k[4],
                    "gazebo_valid": bool(g.get("valid_episode")),
                    "gazebo_success": gs,
                    "reference_success": rs,
                    "gazebo_collision": gc,
                    "reference_collision": rc,
                    "agree": gs == rs and gc == rc,
                    "gazebo_mean_distance_m": g.get("mean_distance_m"),
                    "reference_mean_distance_m": r.get("mean_distance_m"),
                    "gazebo_tracked_fraction": g.get("tracked_fraction"),
                    "reference_tracked_fraction": r.get("tracked_fraction"),
                }
            )
    summary = {
        "rule": __doc__,
        "pairs": len(pairs),
        "missing_reference": missing,
        "gazebo_valid": sum(p["gazebo_valid"] for p in pairs),
        "agree": sum(p["agree"] for p in pairs),
        "success_gazebo": sum(p["gazebo_success"] for p in pairs),
        "success_reference": sum(p["reference_success"] for p in pairs),
        "by_variant": {},
        "episodes": pairs,
    }
    for p in pairs:
        v = summary["by_variant"].setdefault(
            p["variant"], {"pairs": 0, "agree": 0, "valid": 0, "gz_success": 0, "2d_success": 0}
        )
        v["pairs"] += 1
        v["agree"] += p["agree"]
        v["valid"] += p["gazebo_valid"]
        v["gz_success"] += p["gazebo_success"]
        v["2d_success"] += p["reference_success"]
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"pairs {summary['pairs']}, valid {summary['gazebo_valid']}, agree {summary['agree']}, "
        f"success gazebo {summary['success_gazebo']} / 2D {summary['success_reference']}, "
        f"missing reference {len(missing)}"
    )
    for variant, v in summary["by_variant"].items():
        print(f"  {variant}: {v}")
    for p in pairs:
        if not p["agree"] or not p["gazebo_valid"]:
            print(
                f"  differs: {p['run']} {p['variant']} {p['scenario']} {p['environment_seed']} "
                f"gz success {p['gazebo_success']} coll {p['gazebo_collision']} valid "
                f"{p['gazebo_valid']} | 2D success {p['reference_success']} coll "
                f"{p['reference_collision']}"
            )


if __name__ == "__main__":
    main()
