"""Summarize Gazebo runtime coverage and the G3 latency from recorded G2 outputs (#88).

Usage:
  python scripts/summarize_gazebo_runtime.py --host 4070=DIR [--host 4060=DIR ...] \
      --reference 4070 --output runs/evaluation/gazebo_runtime.json

Each DIR is one host's runs/gazebo_eval tree (unpack every host separately), as the G2 runner wrote:
  {curved,wander,posthoc_curved,posthoc_wander}/seed_{k}/gazebo_matrix_manifest.json
  {...}/seed_{k}/{variant}/seed_{k}/{scenario}/{episodes.jsonl,control_traces.jsonl}
Only the manifest's cells are read; any other file under DIR is ignored.

Definitions (G2 Decision on #88, fixed before the G2 data):
- inference rate = sum(inference_wall_samples) / sum(measured_duration_s), per simulated second;
- timed fraction = control_traces rows with causal_clock_valid / control_traces rows;
- sensor->command latency per command = command_published_monotonic_ns minus the oldest
  source_monotonic_ns among its valid samples; percentiles interpolate linearly (numpy's
  default);
- G3 latency of a variant = p95 over every timed command of the reference host's curved normal
  cells, pooled over the training seeds run there; it passes at <= 50 ms. The G3 success half
  (curved normal success >= 0.8, training-seed mean) comes from the registered evaluation.
Every other group (wander, other scenarios, other hosts, single seeds) is reported, never used
for G3. Behaviour outcomes are not summarized here.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

G3_P95_MAX_MS = 50.0
RUNS = {
    "curved": ("formal", "curved"),
    "wander": ("formal", "wander"),
    "posthoc_curved": ("posthoc", "curved"),
    "posthoc_wander": ("posthoc", "wander"),
}


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q / 100
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def command_latency_ms(trace: dict) -> float | None:
    if not trace.get("causal_clock_valid"):
        return None
    starts = [s["source_monotonic_ns"] for s in trace["samples"] if s.get("causal_clock_valid")]
    if not starts:
        return None
    return (int(trace["command_published_monotonic_ns"]) - min(starts)) / 1e6


def host_argument(value: str) -> tuple[str, Path]:
    name, separator, directory = value.partition("=")
    if not separator or not name or not directory:
        raise argparse.ArgumentTypeError("Expected --host NAME=DIR")
    return name, Path(directory)


def collect(hosts: list[tuple[str, Path]]) -> list[dict]:
    """One record per recorded cell, with its raw command latencies."""
    cells, seen = [], set()
    for host, root in hosts:
        for run, (kind, motion) in RUNS.items():
            for manifest_path in sorted((root / run).glob("seed_*/gazebo_matrix_manifest.json")):
                block = manifest_path.parent
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("target_motion") != motion:
                    raise ValueError(f"{manifest_path}: target_motion is not {motion}")
                for cell in manifest["cells"]:
                    variant, seed, scenario = cell["variant"], int(cell["seed"]), cell["scenario"]
                    key = (kind, motion, variant, seed, scenario)
                    if key in seen:
                        raise ValueError(f"Cell recorded twice across hosts or blocks: {key}")
                    seen.add(key)
                    directory = block / variant / f"seed_{seed}" / scenario
                    episodes = rows(directory / "episodes.jsonl")
                    traces = rows(directory / "control_traces.jsonl")
                    samples = sum(int(e.get("inference_wall_samples") or 0) for e in episodes)
                    duration = sum(float(e.get("measured_duration_s") or 0) for e in episodes)
                    latencies = [v for v in map(command_latency_ms, traces) if v is not None]
                    cells.append(
                        {
                            "host": host,
                            "kind": kind,
                            "motion": motion,
                            "variant": variant,
                            "seed": seed,
                            "scenario": scenario,
                            "episodes": len(episodes),
                            "invalid_episodes": sum(not e.get("valid_episode") for e in episodes),
                            "inference_samples": samples,
                            "measured_duration_s": duration,
                            "trace_rows": len(traces),
                            "timed_commands": len(latencies),
                            "evaluator_filled_samples": sum(
                                int(e.get("causal_wall_evaluator_filled_samples") or 0)
                                for e in episodes
                            ),
                            "latencies_ms": latencies,
                        }
                    )
    return cells


def pooled(cells: list[dict]) -> dict:
    latencies = [value for cell in cells for value in cell["latencies_ms"]]
    samples = sum(c["inference_samples"] for c in cells)
    duration = sum(c["measured_duration_s"] for c in cells)
    traces = sum(c["trace_rows"] for c in cells)
    return {
        "cells": len(cells),
        "episodes": sum(c["episodes"] for c in cells),
        "invalid_episodes": sum(c["invalid_episodes"] for c in cells),
        "seeds": sorted({c["seed"] for c in cells}),
        "inference_rate_hz": samples / duration if duration else None,
        "min_cell_inference_rate_hz": min(
            (
                c["inference_samples"] / c["measured_duration_s"]
                for c in cells
                if c["measured_duration_s"]
            ),
            default=None,
        ),
        "timed_fraction": sum(c["timed_commands"] for c in cells) / traces if traces else None,
        "trace_rows_per_inference_sample": traces / samples if samples else None,
        "evaluator_filled_samples": sum(c["evaluator_filled_samples"] for c in cells),
        "timed_commands": len(latencies),
        "sensor_to_command_p50_ms": percentile(latencies, 50),
        "sensor_to_command_p95_ms": percentile(latencies, 95),
    }


def summarize(hosts: list[tuple[str, Path]], reference: str) -> dict:
    if reference not in {name for name, _ in hosts}:
        raise ValueError("--reference must name one of the --host entries")
    if len({name for name, _ in hosts}) != len(hosts):
        raise ValueError("Host names must be unique")
    cells = collect(hosts)
    groups = defaultdict(list)
    for cell in cells:
        groups[
            (cell["host"], cell["kind"], cell["motion"], cell["variant"], cell["scenario"])
        ].append(cell)
    g3 = {}
    for kind, variant in sorted({(c["kind"], c["variant"]) for c in cells}):
        chosen = [
            c
            for c in cells
            if c["host"] == reference
            and c["kind"] == kind
            and c["variant"] == variant
            and c["motion"] == "curved"
            and c["scenario"] == "normal"
        ]
        summary = pooled(chosen)
        p95 = summary["sensor_to_command_p95_ms"]
        g3[f"{kind}/{variant}"] = {
            **summary,
            "per_seed_p95_ms": {
                str(c["seed"]): percentile(c["latencies_ms"], 95)
                for c in sorted(chosen, key=lambda c: c["seed"])
            },
            "latency_passed": None if p95 is None else p95 <= G3_P95_MAX_MS,
        }
    return {
        "rule": __doc__,
        "reference_host": reference,
        "g3_p95_max_ms": G3_P95_MAX_MS,
        "g3_latency": g3,
        "groups": [
            {
                "host": host,
                "kind": kind,
                "motion": motion,
                "variant": variant,
                "scenario": scenario,
                **pooled(members),
            }
            for (host, kind, motion, variant, scenario), members in sorted(groups.items())
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", type=host_argument, action="append", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.host, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, row in result["g3_latency"].items():
        p95 = row["sensor_to_command_p95_ms"]
        print(
            f"G3 latency {name:16s} seeds {row['seeds']} commands {row['timed_commands']} "
            f"p95 {'n/a' if p95 is None else f'{p95:.1f}'} ms passed {row['latency_passed']}"
        )


if __name__ == "__main__":
    main()
