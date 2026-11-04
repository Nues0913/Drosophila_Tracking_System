"""Read-only A1 lesion statistics, deterministic failure selection and trace tables.

Each bootstrap row gets a NEW np.random.default_rng(1301), 10,000 draws, via
flytrack_train.evaluation.bootstrap_mean. This deliberately differs from formal
evaluation's shared RNG stream: rows are independent of iteration/input order.
Episodes are averaged within each seed before seeds are bootstrapped or paired.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from ._a1_common import input_files, load_rows, provenance, write_report
else:
    from _a1_common import input_files, load_rows, provenance, write_report

VARIANTS = ("VO", "VO-late", "RS", "GRU")
SCENARIOS = ("normal", "occlusion", "conflict")
LESIONS = ("none", "vision", "odor")
SEEDS = tuple(range(5))
EPISODES = tuple(range(20))
CURRENT_METRICS = tuple(
    f"{name}_fraction" for name in ("visual_exclusive", "odor_exclusive", "shared")
)
METRICS = (
    "success",
    "collision_rate",
    "mean_distance_m",
    "mean_bearing_error_deg",
    "tracked_fraction",
    "within_distance_fraction",
)
ALIASES = {
    "collision_rate": "collision",
    "mean_bearing_error_deg": "bearing_error_deg",
    "tracked_fraction": "tracking_fraction",
}
BOOTSTRAP_RULE = (
    "Within-seed episode means; seed-level bootstrap_mean with a fresh "
    "np.random.default_rng(1301) per output row, 10000 draws, percentile 95% CI. "
    "Unlike formal summaries, no RNG stream is shared between rows. "
    "Null values are undefined, excluded from means and counted explicitly."
)
SELECTION_RULE = (
    "PM classification: collision_rate > 0 => collision; otherwise successful rows "
    "are excluded; within_distance_fraction < 0.8 => distance; remaining failures "
    "=> bearing. Select occlusion/collision and normal/distance, VO-late then VO, "
    "one smallest episode per seed 0..4. Fill missing seed slots from the next "
    "seed in cyclic ascending order (wrapping 4 to 0), choosing the smallest "
    "unused episode; keep five distinct episodes per variant and category."
)


def metric(row: dict, name: str):
    if name in row:
        return row[name]
    alias = ALIASES.get(name)
    if alias and alias in row:
        return row[alias]
    raise ValueError(f"Missing metric {name} in episode {episode_key(row)}")


def episode_key(row: dict) -> tuple:
    return row["variant"], row["scenario"], int(row["seed"]), int(row["episode"])


def indexed(rows: list[dict]) -> dict:
    result = {}
    for row in rows:
        key = episode_key(row)
        if key in result:
            raise ValueError(f"Duplicate episode: {key}")
        result[key] = row
    return result


def finite_mean(values) -> float | None:
    values = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(values)) if values else None


def bootstrap(values) -> dict:
    from flytrack_train.evaluation import bootstrap_mean

    return bootstrap_mean(
        [value for value in values if value is not None], np.random.default_rng(1301), 10000
    )


def failure_category(row: dict) -> str:
    if float(metric(row, "collision_rate")) > 0:
        return "collision"
    if bool(metric(row, "success")):
        return "success"
    if float(metric(row, "within_distance_fraction")) < 0.8:
        return "distance"
    return "bearing"


def select_failure_cases(rows: list[dict]) -> list[dict]:
    indexed(rows)
    cases = []
    for scenario, category in (("occlusion", "collision"), ("normal", "distance")):
        for variant in ("VO-late", "VO"):
            candidates = {
                seed: sorted(
                    (
                        row
                        for row in rows
                        if row["variant"] == variant
                        and row["scenario"] == scenario
                        and int(row["seed"]) == seed
                        and failure_category(row) == category
                    ),
                    key=lambda row: int(row["episode"]),
                )
                for seed in SEEDS
            }
            chosen = {}
            used = set()
            for seed in SEEDS:
                if candidates[seed]:
                    chosen[seed] = candidates[seed][0]
                    used.add(episode_key(candidates[seed][0]))
            for missing_seed in SEEDS:
                if missing_seed in chosen:
                    continue
                for offset in range(1, len(SEEDS)):
                    donor = (missing_seed + offset) % len(SEEDS)
                    available = [row for row in candidates[donor] if episode_key(row) not in used]
                    if available:
                        chosen[missing_seed] = available[0]
                        used.add(episode_key(available[0]))
                        break
                else:
                    raise ValueError(f"Cannot fill five {variant}/{scenario}/{category} cases")
            for requested_seed in SEEDS:
                row = chosen[requested_seed]
                replacement = int(row["seed"]) != requested_seed
                cases.append(
                    {
                        "variant": variant,
                        "scenario": scenario,
                        "category": category,
                        "seed": int(row["seed"]),
                        "episode": int(row["episode"]),
                        "requested_seed": requested_seed,
                        "replacement": replacement,
                        "selection_note": (
                            f"seed {requested_seed} unavailable; filled from seed {row['seed']}"
                        )
                        if replacement
                        else "smallest qualifying episode for this seed",
                        "f3_metrics": dict(row),
                    }
                )
    return cases


def consistency(rows: list[dict], f3: list[dict], posthoc: list[dict]) -> list[dict]:
    formal_index, posthoc_index = indexed(f3), indexed(posthoc)
    groups = defaultdict(list)
    for row in rows:
        if row["lesion"] != "none":
            continue
        key = episode_key(row)
        reference = posthoc_index if row["variant"] == "GRU" else formal_index
        if key not in reference:
            raise ValueError(f"No reference episode for lesion=none: {key}")
        other = reference[key]
        distance_a = metric(row, "mean_distance_m")
        distance_b = metric(other, "mean_distance_m")
        if (
            distance_a is None
            or distance_b is None
            or not np.all(np.isfinite([distance_a, distance_b]))
        ):
            raise ValueError(f"Nonfinite consistency distance: {key}")
        groups[(row["variant"], row["scenario"])].append(
            (
                float(metric(row, "success")) == float(metric(other, "success")),
                (float(metric(row, "collision_rate")) > 0)
                == (float(metric(other, "collision_rate")) > 0),
                abs(float(distance_a) - float(distance_b)),
            )
        )
    return [
        {
            "variant": variant,
            "scenario": scenario,
            "reference": "posthoc" if variant == "GRU" else "f3",
            "episodes": len(values),
            "success_agreement": float(np.mean([v[0] for v in values])),
            "collision_agreement": float(np.mean([v[1] for v in values])),
            "max_abs_mean_distance_m_difference": max(v[2] for v in values),
        }
        for (variant, scenario), values in sorted(groups.items())
    ]


def lesion_statistics(rows: list[dict], allow_partial: bool = False) -> dict:
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["variant"], row["scenario"], row["lesion"], int(row["seed"]))
        episode = int(row["episode"])
        if (
            key[0] not in VARIANTS
            or key[1] not in SCENARIOS
            or key[2] not in LESIONS
            or key[3] not in SEEDS
            or episode not in EPISODES
        ):
            raise ValueError(f"Unexpected lesion matrix entry: {key}, episode {episode}")
        if (*key, episode) in seen:
            raise ValueError(f"Duplicate lesion episode: {key}, {episode}")
        seen.add((*key, episode))
        groups[key].append(row)
    missing = []
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            for lesion in LESIONS:
                for seed in SEEDS:
                    missing_episodes = [
                        ep for ep in EPISODES if (variant, scenario, lesion, seed, ep) not in seen
                    ]
                    if missing_episodes:
                        missing.append(
                            {
                                "variant": variant,
                                "scenario": scenario,
                                "lesion": lesion,
                                "seed": seed,
                                "episodes_present": 20 - len(missing_episodes),
                                "missing_episodes": missing_episodes,
                            }
                        )
    if missing and not allow_partial:
        raise ValueError(
            f"Incomplete lesion matrix: {len(missing)} missing/incomplete cells; "
            "use --allow-partial"
        )
    seed_rows = []
    for (variant, scenario, lesion, seed), episodes in sorted(groups.items()):
        episodes = sorted(episodes, key=lambda row: int(row["episode"]))
        row_metrics = METRICS + (() if variant == "GRU" else CURRENT_METRICS)
        averages, counts = {}, {}
        for name in row_metrics:
            values = [metric(row, name) for row in episodes]
            averages[name] = finite_mean(values)
            counts[name] = sum(bool(value is not None and np.isfinite(value)) for value in values)
        seed_rows.append(
            {
                "variant": variant,
                "scenario": scenario,
                "lesion": lesion,
                "seed": seed,
                "episodes": len(episodes),
                **averages,
                "defined_episode_counts": counts,
            }
        )
    summaries, comparisons, paired_seed_rows = [], [], []
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            row_metrics = METRICS + (() if variant == "GRU" else CURRENT_METRICS)
            by_lesion = {
                lesion: {
                    row["seed"]: row
                    for row in seed_rows
                    if row["variant"] == variant
                    and row["scenario"] == scenario
                    and row["lesion"] == lesion
                }
                for lesion in LESIONS
            }
            for lesion in LESIONS:
                for name in row_metrics:
                    summaries.append(
                        {
                            "variant": variant,
                            "scenario": scenario,
                            "lesion": lesion,
                            "metric": name,
                            **bootstrap([row[name] for row in by_lesion[lesion].values()]),
                        }
                    )
            for lesion in ("vision", "odor"):
                paired = sorted(by_lesion["none"].keys() & by_lesion[lesion].keys())
                differences = defaultdict(list)
                for seed in paired:
                    pair = {
                        "variant": variant,
                        "scenario": scenario,
                        "comparison": f"none - {lesion}",
                        "seed": seed,
                    }
                    for name in row_metrics:
                        a, b = by_lesion["none"][seed][name], by_lesion[lesion][seed][name]
                        pair[name] = a - b if a is not None and b is not None else None
                        differences[name].append(pair[name])
                    paired_seed_rows.append(pair)
                for name in row_metrics:
                    comparisons.append(
                        {
                            "variant": variant,
                            "scenario": scenario,
                            "comparison": f"none - {lesion}",
                            "metric": name,
                            **bootstrap(differences[name]),
                        }
                    )
    return {
        "complete": not missing,
        "expected_matrix": {
            "variants": list(VARIANTS),
            "scenarios": list(SCENARIOS),
            "lesions": list(LESIONS),
            "seeds": list(SEEDS),
            "episodes_per_cell": 20,
        },
        "bootstrap_rule": BOOTSTRAP_RULE,
        "missing_cells": missing,
        "seed_means": seed_rows,
        "lesion_summary": summaries,
        "paired_seed_differences": paired_seed_rows,
        "paired_comparisons": comparisons,
    }


def trace_statistics(rows: list[dict], cases: list[dict], allow_partial: bool = False) -> dict:
    # Full-current and selected-failure replays may overlap. Deduplicate only
    # identical scientific records (the trace filenames may differ).
    trace_index = {}
    for row in rows:
        if row["lesion"] != "none":
            continue
        key = episode_key(row)
        if key in trace_index:
            before = {k: v for k, v in trace_index[key].items() if k != "trace"}
            after = {k: v for k, v in row.items() if k != "trace"}
            if before != after:
                raise ValueError(f"Conflicting duplicate trace: {key}")
        trace_index[key] = row
    groups = defaultdict(list)
    missing = []
    for variant in ("VO", "VO-late", "RS"):
        for seed in SEEDS:
            available = []
            absent = []
            for episode in EPISODES:
                row = trace_index.get((variant, "occlusion", seed, episode))
                if row is None:
                    absent.append(episode)
                else:
                    if "current_segments" not in row:
                        raise ValueError(f"Missing segmented current: {episode_key(row)}")
                    available.append(row)
            if available:
                groups[(variant, seed)] = available
            if absent:
                missing.append({"variant": variant, "seed": seed, "missing_episodes": absent})
    missing_cases = [case for case in cases if episode_key(case) not in trace_index]
    if not allow_partial and (missing or missing_cases):
        raise ValueError(
            f"Incomplete traces: {len(missing)} current cells, "
            f"{len(missing_cases)} failure cases missing"
        )
    seed_rows = []
    for (variant, seed), episodes in sorted(groups.items()):
        for segment in ("occluded", "non_occluded"):
            for name in ("odor_exclusive", "visual_exclusive"):
                values = [row["current_segments"][segment]["fraction"][name] for row in episodes]
                seed_rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "segment": segment,
                        "metric": f"{name}_fraction",
                        "episodes": len(episodes),
                        "defined_episodes": sum(
                            bool(v is not None and np.isfinite(v)) for v in values
                        ),
                        "mean": finite_mean(values),
                    }
                )
    summaries, comparisons, pairs = [], [], []
    for variant in ("VO", "VO-late", "RS"):
        for name in ("odor_exclusive_fraction", "visual_exclusive_fraction"):
            segment_rows = {
                segment: {
                    row["seed"]: row
                    for row in seed_rows
                    if row["variant"] == variant
                    and row["metric"] == name
                    and row["segment"] == segment
                }
                for segment in ("occluded", "non_occluded")
            }
            for segment, values in segment_rows.items():
                summaries.append(
                    {
                        "variant": variant,
                        "segment": segment,
                        "metric": name,
                        **bootstrap([row["mean"] for row in values.values()]),
                    }
                )
            differences = []
            for seed in sorted(
                segment_rows["occluded"].keys() & segment_rows["non_occluded"].keys()
            ):
                a = segment_rows["occluded"][seed]["mean"]
                b = segment_rows["non_occluded"][seed]["mean"]
                difference = a - b if a is not None and b is not None else None
                differences.append(difference)
                pairs.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "metric": name,
                        "comparison": "occluded - non_occluded",
                        "difference": difference,
                    }
                )
            comparisons.append(
                {
                    "variant": variant,
                    "metric": name,
                    "comparison": "occluded - non_occluded",
                    **bootstrap(differences),
                }
            )
    failure_rows = []
    for case in cases:
        row = trace_index.get(episode_key(case))
        if row is None:
            continue
        descriptors = row["descriptors"]
        failure_rows.append(
            {
                "variant": row["variant"],
                "scenario": row["scenario"],
                "seed": int(row["seed"]),
                "episode": int(row["episode"]),
                "category": case["category"],
                "replay_category": failure_category(row),
                "collision_types": row["collision_types"],
                "collision_time_s": row["collision_time_s"],
                "occlusion": {
                    key: value
                    for key, value in (descriptors["occlusion"] or {}).items()
                    if key != "windows"
                },
                "whole_episode": descriptors["whole_episode"],
            }
        )
    ranges = []
    for category in ("collision", "distance"):
        selected = [row for row in failure_rows if row["category"] == category]
        quantities = defaultdict(list)
        for row in selected:
            quantities["collision_time_s"].append(row["collision_time_s"])
            for scope in ("occlusion", "whole_episode"):
                for name, value in row[scope].items():
                    quantities[f"{scope}.{name}"].append(value)
        for name, values in sorted(quantities.items()):
            defined = [float(v) for v in values if v is not None and np.isfinite(v)]
            ranges.append(
                {
                    "category": category,
                    "metric": name,
                    "episodes": len(selected),
                    "defined_episodes": len(defined),
                    "median": float(np.median(defined)) if defined else None,
                    "min": min(defined) if defined else None,
                    "max": max(defined) if defined else None,
                }
            )
    return {
        "complete": not missing and not missing_cases,
        "bootstrap_rule": BOOTSTRAP_RULE,
        "current_rule": "300 unique occlusion/none episodes: VO, VO-late, RS; seeds 0..4; "
        "episodes 0..19. Average per-episode segment fractions within seed; "
        "zero-mass or unobserved segments are undefined, never zero-filled.",
        "missing_current_cells": missing,
        "missing_failure_cases": missing_cases,
        "current_seed_means": seed_rows,
        "current_summary": summaries,
        "current_paired_seed_differences": pairs,
        "current_paired_comparisons": comparisons,
        "failure_cases": failure_rows,
        "failure_ranges": ranges,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select-failures", help="Select the 20 predeclared F3 failures")
    select.add_argument("--f3", type=Path, required=True)
    select.add_argument("--output", "-o", type=Path, required=True)
    lesion = commands.add_parser("lesion", help="Summarize the 4 x 3 x 3 x 5 x 20 A1-1 matrix")
    lesion.add_argument("--input", type=Path, required=True)
    lesion.add_argument("--f3", type=Path, required=True)
    lesion.add_argument("--posthoc", type=Path, required=True)
    lesion.add_argument("--allow-partial", action="store_true")
    lesion.add_argument("--output", "-o", type=Path, required=True)
    traces = commands.add_parser(
        "traces", help="Summarize the 300 current traces and selected failures"
    )
    traces.add_argument("--input", type=Path, required=True)
    traces.add_argument(
        "--failures", type=Path, required=True, help="select-failures/failure_cases.json"
    )
    traces.add_argument("--allow-partial", action="store_true")
    traces.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select-failures":
        paths = [args.f3]
        cases = select_failure_cases(load_rows(paths))
        payload = {
            "selection_rule": SELECTION_RULE,
            "cases": cases,
            "provenance": provenance(paths),
        }
        write_report(args.output, "failure_cases", payload, {"cases": cases}, command=args.command)
        print(f"Selected {len(cases)} cases", flush=True)
        return
    paths = input_files(args.input)
    rows = load_rows(paths)
    if args.command == "lesion":
        paths.extend([args.f3, args.posthoc])
        agreements = consistency(rows, load_rows([args.f3]), load_rows([args.posthoc]))
        payload = lesion_statistics(rows, args.allow_partial)
        payload["consistency"] = agreements
        tables = {
            name: payload[name]
            for name in (
                "seed_means",
                "lesion_summary",
                "paired_seed_differences",
                "paired_comparisons",
                "consistency",
                "missing_cells",
            )
        }
    else:
        selection = json.loads(args.failures.read_text(encoding="utf-8"))
        paths.append(args.failures)
        # Hash the replay manifests as well: they identify the underlying actor,
        # configuration, and compressed step files without loading those files.
        paths.extend(
            sorted(
                {
                    path.parent / "analysis_notes.json"
                    for path in paths
                    if (path.parent / "analysis_notes.json").is_file()
                }
            )
        )
        payload = trace_statistics(rows, selection["cases"], args.allow_partial)
        tables = {
            name: payload[name]
            for name in (
                "current_seed_means",
                "current_summary",
                "current_paired_seed_differences",
                "current_paired_comparisons",
                "failure_cases",
                "failure_ranges",
                "missing_current_cells",
                "missing_failure_cases",
            )
        }
    payload["provenance"] = provenance(paths)
    write_report(args.output, args.command, payload, tables, command=args.command)
    print(f"{args.command}: {len(rows)} input episodes; complete={payload['complete']}", flush=True)


if __name__ == "__main__":
    main()
