"""Decompose each inference command along its oldest contributing sample path."""

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

STAGES = (
    "source_to_ingress",
    "ingress_to_forward",
    "forward_to_received",
    "received_to_control",
    "control_to_published",
)


def ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def load_bag(bag):
    traces, events = [], []
    checks = []
    message_counts = Counter()
    for db in sorted(bag.glob("*.db3")):
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        checks.append(conn.execute("PRAGMA quick_check").fetchone()[0])
        topics = {
            row[0]: (row[1], row[2]) for row in conn.execute("SELECT id,name,type FROM topics")
        }
        for topic_id, count in conn.execute(
            "SELECT topic_id,COUNT(*) FROM messages GROUP BY topic_id"
        ):
            message_counts[topics[topic_id][0]] += count
        selected = {
            topic_id: get_message(kind)
            for topic_id, (name, kind) in topics.items()
            if name in {"/fly/control_trace", "/world/scenario"}
        }
        placeholders = ",".join("?" for _ in selected)
        query = (
            "SELECT topic_id,timestamp,data FROM messages "
            f"WHERE topic_id IN ({placeholders}) ORDER BY timestamp,id"
        )
        for topic_id, _, data in conn.execute(query, list(selected)):
            name = topics[topic_id][0]
            message = deserialize_message(data, selected[topic_id])
            if name == "/fly/control_trace":
                if message.inference_performed:
                    traces.append(message)
            else:
                events.append(message)
        conn.close()
    return traces, events, checks, message_counts


def analyze_trace(trace, episode):
    candidates = []
    for index, sample in enumerate(trace.samples):
        if not trace.valid[index]:
            continue
        times = [
            int(sample.source_monotonic_ns),
            int(sample.ingress_monotonic_ns),
            int(sample.forward_monotonic_ns),
            int(trace.policy_received_monotonic_ns[index]),
            int(trace.control_started_monotonic_ns),
            int(trace.command_published_monotonic_ns),
        ]
        candidates.append((times[0], sample.sensor, sample.clock_domain, times))
    if not candidates:
        return None
    source, sensor, domain, times = min(candidates, key=lambda x: (x[0], x[1]))
    valid = (
        bool(trace.clock_domain)
        and domain == trace.clock_domain
        and all(value > 0 for value in times)
        and all(first <= second for first, second in zip(times, times[1:], strict=False))
    )
    if not valid:
        return None
    stages = [(end - start) / 1e6 for start, end in zip(times, times[1:], strict=False)]
    total = (times[-1] - times[0]) / 1e6
    if not math.isclose(sum(stages), total, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Stage durations do not sum to the per-command total")
    stamp = ns(trace.header.stamp)
    return {
        "episode": int(episode) if episode is not None else None,
        "command_sequence": int(trace.command_sequence),
        "command_sim_stamp_ns": stamp,
        "oldest_sensor": sensor,
        "oldest_source_monotonic_ns": source,
        "stage_ms": dict(zip(STAGES, stages, strict=True)),
        "total_ms": total,
        "active": False,
    }


def distribution(rows):
    if not rows:
        return {"n": 0}
    totals = np.asarray([row["total_ms"] for row in rows])
    over = totals >= 50
    n_worst = max(1, math.ceil(len(rows) * 0.05))
    worst = sorted(rows, key=lambda row: (-row["total_ms"], row["command_sequence"]))[:n_worst]
    stage_values = {stage: np.asarray([row["stage_ms"][stage] for row in rows]) for stage in STAGES}
    worst_stage_totals = {stage: sum(row["stage_ms"][stage] for row in worst) for stage in STAGES}
    worst_total = sum(worst_stage_totals.values())
    return {
        "n": len(rows),
        "total_ms": {
            f"p{p}": float(np.percentile(totals, p)) for p in (0, 25, 50, 75, 90, 95, 99, 100)
        },
        "ge_50ms": {"n": int(over.sum()), "fraction": float(over.mean())},
        "stages_ms": {
            stage: {
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "mean": float(values.mean()),
            }
            for stage, values in stage_values.items()
        },
        "worst_5_percent": {
            "n": n_worst,
            "total_ms_min": min(row["total_ms"] for row in worst),
            "total_ms_max": max(row["total_ms"] for row in worst),
            "stages_mean_ms": {stage: worst_stage_totals[stage] / n_worst for stage in STAGES},
            "stage_time_share": {
                stage: worst_stage_totals[stage] / worst_total for stage in STAGES
            },
            "dominant_stage_counts": dict(
                Counter(max(STAGES, key=lambda stage: row["stage_ms"][stage]) for row in worst)
            ),
            "oldest_sensor_counts": dict(Counter(row["oldest_sensor"] for row in worst)),
            "top_10_commands": [
                {
                    "episode": row["episode"],
                    "sequence": row["command_sequence"],
                    "total_ms": row["total_ms"],
                    "oldest_sensor": row["oldest_sensor"],
                    "stages_ms": row["stage_ms"],
                }
                for row in worst[:10]
            ],
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    traces, events, checks, counts = load_bag(args.bag)
    intervals, episode_spans = {}, {}
    for event in events:
        episode = int(event.episode_id)
        phase = json.loads(event.parameters_json).get("phase")
        if phase == "resetting":
            span = episode_spans.setdefault(episode, {"start": ns(event.header.stamp), "end": None})
            span["start"] = min(span["start"], ns(event.header.stamp))
        if event.active and phase == "tracking":
            intervals[episode] = {"start": ns(event.header.stamp), "end": None}
        elif not event.active and phase == "completed" and episode in intervals:
            intervals[episode]["end"] = ns(event.header.stamp)
            episode_spans.setdefault(episode, {"start": 0, "end": None})["end"] = ns(
                event.header.stamp
            )

    rows, invalid = [], 0
    for trace in traces:
        command_stamp = ns(trace.header.stamp)
        episode = next(
            (
                number
                for number, span in episode_spans.items()
                if span["start"] <= command_stamp <= span["end"]
            ),
            None,
        )
        row = analyze_trace(trace, episode)
        if row is None:
            invalid += 1
            continue
        row["episode_assigned"] = episode is not None
        interval = intervals.get(row["episode"])
        if interval and interval["end"] is not None:
            row["active"] = interval["start"] <= row["command_sim_stamp_ns"] <= interval["end"]
        rows.append(row)

    groups = {
        "full_bag": rows,
        "active": [row for row in rows if row["active"]],
    }
    for episode in sorted(intervals):
        groups[f"episode_{episode}"] = [
            row for row in rows if row["episode"] == episode and row["active"]
        ]
    result = {
        "bag": args.bag.name,
        "trace_count_inference": len(traces),
        "invalid_causal_traces": invalid,
        "episode_unassigned_traces": sum(row["episode"] is None for row in rows),
        "sqlite_quick_check": checks,
        "topic_message_counts": dict(counts),
        "active_intervals_ns": intervals,
        "groups": {name: distribution(group) for name, group in groups.items()},
        "cold_start_first_active_command": {
            str(episode): next(
                (
                    {
                        "total_ms": row["total_ms"],
                        "oldest_sensor": row["oldest_sensor"],
                        "stages_ms": row["stage_ms"],
                    }
                    for row in sorted(
                        (r for r in rows if r["episode"] == episode and r["active"]),
                        key=lambda r: r["command_sim_stamp_ns"],
                    )
                ),
                None,
            )
            for episode in sorted(intervals)
        },
        "all_rows": rows,
        "method": (
            "For every inference_performed command, select the valid contributing sample "
            "with the earliest same-host source CLOCK_MONOTONIC timestamp. Split that single "
            "sample-to-command path into the five adjacent clock intervals; verify their sum "
            "equals the complete endpoint difference. Active windows come from scenario "
            "tracking/completed events. Worst 5% is the slowest ceil(5% of rows), ranked by total."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: value["n"] for name, value in result["groups"].items()}))
    print("invalid", invalid)


if __name__ == "__main__":
    main()
