"""Pair the v2 G1 Gazebo episodes with their 2D references, episode by episode (#140).

Usage: python scripts/v2/gazebo_g1_compare.py GAZEBO_DIR REFERENCE_DIR OUT.json
Both directories hold {variant}/seed_{seed}/{scenario}/episodes.jsonl for each target motion
under GAZEBO_DIR/<run>/ and REFERENCE_DIR/<run>/ (run = one orchestrator call, e.g. curved,
wander). Episodes are paired by (run, variant, seed, scenario, environment_seed).

Before any agreement is counted, both sides must hold exactly one record per key and the same
set of keys, each record's variant, seed, scenario and target_motion must match its path, and
the two records of a pair must name the same non-empty model_card_sha256. Any violation is
listed and the script exits 1 without writing a summary.

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


class PairingError(RuntimeError):
    """The two sides cannot be paired one to one with the same model."""


def index(root: Path, side: str, problems: list[str]) -> dict[tuple, dict]:
    table: dict[tuple, dict] = {}
    for key, episodes in cells(root):
        run, variant, seed, scenario = key
        for e in episodes:
            k = key + (int(e["environment_seed"]),)
            expected = {
                "variant": variant,
                "seed": seed,
                "scenario": scenario,
                "target_motion": run,
            }
            for field, value in expected.items():
                if field in e and e[field] != value:
                    problems.append(f"{side} {list(k)}: {field} {e[field]!r} != path {value!r}")
            if k in table:
                problems.append(f"{side} {list(k)}: duplicate episode key")
                continue
            table[k] = e
    return table


def pair(gazebo: Path, reference: Path) -> list[dict]:
    """Pair the two trees one to one; raise PairingError listing every problem."""
    problems: list[str] = []
    gz = index(gazebo, "gazebo", problems)
    two_d = index(reference, "2D", problems)
    for k in sorted(gz.keys() - two_d.keys()):
        problems.append(f"gazebo {list(k)}: no 2D reference")
    for k in sorted(two_d.keys() - gz.keys()):
        problems.append(f"2D {list(k)}: no Gazebo episode")
    if not gz:
        problems.append("gazebo: no episodes")
    pairs = []
    for k in sorted(gz.keys() & two_d.keys()):
        g, r = gz[k], two_d[k]
        gm, rm = g.get("model_card_sha256"), r.get("model_card_sha256")
        if not gm or not rm or gm != rm:
            problems.append(f"{list(k)}: model_card_sha256 gazebo {gm!r} != 2D {rm!r}")
        gs, rs = bool(g.get("success")), bool(r.get("success"))
        gc, rc = (g.get("collision_rate") or 0) > 0, (r.get("collision_rate") or 0) > 0
        pairs.append(
            {
                "run": k[0],
                "variant": k[1],
                "seed": k[2],
                "scenario": k[3],
                "environment_seed": k[4],
                "model_card_sha256": gm,
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
    if problems:
        raise PairingError("\n".join(problems))
    return pairs


def main() -> None:
    gazebo, reference, output = (Path(a) for a in sys.argv[1:4])
    try:
        pairs = pair(gazebo, reference)
    except PairingError as error:
        print(f"pairing failed:\n{error}", file=sys.stderr)
        raise SystemExit(1) from None
    missing: list = []
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
