#!/usr/bin/env python3
"""Recompute Issue #49 replay timing from its immutable raw JSONL archive."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

SENSORS = ("left", "right", "odor")
SPAN_NAMES = ("image_decode", "hex_eye_sample", "buffer_update")


def rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def linear_quantile(values: list[int | float], probability: float) -> float | None:
    """NumPy-style linear percentile, returned in milliseconds for ns input."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    result = ordered[lower] + (index - lower) * (ordered[upper] - ordered[lower])
    return result / 1_000_000


def stats(values: list[int | float]) -> dict[str, int | float | None]:
    return {
        "n": len(values),
        "p50_ms": linear_quantile(values, 0.50),
        "p95_ms": linear_quantile(values, 0.95),
        "p99_ms": linear_quantile(values, 0.99),
    }


def measurement_bounds(monitor: list[dict[str, Any]]) -> tuple[int, int]:
    starts = [
        r["host_monotonic_ns"]
        for r in monitor
        if r.get("event") == "phase_start" and r.get("phase") == "measurement"
    ]
    ends = [
        r["host_monotonic_ns"]
        for r in monitor
        if r.get("event") == "phase_end" and r.get("phase") == "measurement"
    ]
    if len(starts) != 1 or len(ends) != 1 or ends[0] <= starts[0]:
        raise ValueError("expected exactly one valid measurement interval")
    return starts[0], ends[0]


def matched_sensor_callbacks(
    monitor: list[dict[str, Any]], profiler: list[dict[str, Any]], start: int, end: int
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[tuple[int, int], list[int]]]]:
    published: dict[str, dict[tuple[int, int], list[int]]] = {
        sensor: defaultdict(list) for sensor in SENSORS
    }
    for row in monitor:
        if row.get("event") == "sensor_publish" and row.get("phase") == "measurement":
            published[row["sensor"]][tuple(row["header_stamp"])].append(row["publish_monotonic_ns"])

    callbacks: dict[str, dict[tuple[str, tuple[int, int], int], dict[str, Any]]] = {
        sensor: {} for sensor in SENSORS
    }
    for row in profiler:
        if row.get("event") != "command_profile" or not start <= row["callback_entry_ns"] <= end:
            continue
        for callback in row.get("sensor_callbacks_since_previous_command", []):
            if not callback.get("accepted") or not start <= callback["callback_entry_ns"] <= end:
                continue
            sensor = callback["sensor"]
            key = (sensor, tuple(callback["header_stamp"]), callback["callback_entry_ns"])
            callbacks[sensor][key] = callback

    result: dict[str, list[dict[str, Any]]] = {sensor: [] for sensor in SENSORS}
    for sensor in SENSORS:
        for (_, header, entry), callback in callbacks[sensor].items():
            candidates = [stamp for stamp in published[sensor].get(header, []) if stamp <= entry]
            matched = max(candidates) if candidates else None
            result[sensor].append({"callback": callback, "publish_ns": matched})
    return result, published


def analyze_case(raw: Path, device: str) -> dict[str, Any]:
    monitor_on = rows_from_jsonl(raw / device / f"{device}-profile_on" / "monitor.jsonl")
    profiler = rows_from_jsonl(raw / device / f"{device}-profile_on" / "profiler.jsonl")
    monitor_off = rows_from_jsonl(raw / device / f"{device}-profile_off" / "monitor.jsonl")
    start, end = measurement_bounds(monitor_on)
    if measurement_bounds(monitor_off)[1] <= measurement_bounds(monitor_off)[0]:
        raise ValueError("invalid profile-off measurement interval")

    profiles = [
        r
        for r in profiler
        if r.get("event") == "command_profile" and start <= r["callback_entry_ns"] <= end
    ]
    traces_on = {
        r["command_sequence"]: r
        for r in monitor_on
        if r.get("event") == "control_trace" and r.get("phase") == "measurement"
    }
    traces_off = [
        r
        for r in monitor_off
        if r.get("event") == "control_trace" and r.get("phase") == "measurement"
    ]

    def valid_trace(trace: dict[str, Any]) -> bool:
        return all(trace.get("valid", [])) and len(trace.get("valid", [])) == 3

    on_cohort = [
        row
        for row in profiles
        if all(row.get("valid", []))
        and row.get("control_trace_status") == "emitted"
        and row.get("command_sequence") in traces_on
        and valid_trace(traces_on[row["command_sequence"]])
    ]
    off_cohort = [row for row in traces_off if valid_trace(row)]
    elapsed_on = [row["publish_done_ns"] - row["callback_entry_ns"] for row in on_cohort]
    elapsed_off = [
        row["command_published_monotonic_ns"] - row["control_started_monotonic_ns"]
        for row in off_cohort
    ]

    all_callbacks = [row for row in profiles]
    inference = [row for row in profiles if row.get("inference_performed")]
    callback_all = [row["callback_exit_ns"] - row["callback_entry_ns"] for row in all_callbacks]
    callback_inference = [row["callback_exit_ns"] - row["callback_entry_ns"] for row in inference]
    model = [
        row["spans_ns"]["model_forward_and_completion"][1]
        - row["spans_ns"]["model_forward_and_completion"][0]
        for row in inference
    ]

    sensor_data, published = matched_sensor_callbacks(monitor_on, profiler, start, end)
    sensor_result: dict[str, Any] = {}
    for sensor, matched in sensor_data.items():
        waits: list[int] = []
        spans: dict[str, list[int]] = {name: [] for name in SPAN_NAMES}
        missing = 0
        for item in matched:
            callback = item["callback"]
            publish_ns = item["publish_ns"]
            if publish_ns is None:
                missing += 1
                continue
            waits.append(callback["callback_entry_ns"] - publish_ns)
            for name in SPAN_NAMES:
                pair = callback.get("spans_ns", {}).get(name)
                if pair:
                    spans[name].append(pair[1] - pair[0])
        pub_count = sum(len(items) for items in published[sensor].values())
        sensor_result[sensor] = {
            "published": pub_count,
            "accepted_callbacks": len(matched),
            "matched_callbacks": len(waits),
            "unmatched_callbacks": missing,
            "publish_to_callback_entry": stats(waits),
            "spans": {name: stats(values) for name, values in spans.items()},
        }

    latest_callback_gap: list[int] = []
    nonempty = 0
    for row in profiles:
        callbacks = [
            cb
            for cb in row.get("sensor_callbacks_since_previous_command", [])
            if cb.get("accepted")
        ]
        if callbacks:
            nonempty += 1
            latest_exit = max(cb["callback_exit_ns"] for cb in callbacks)
            latest_callback_gap.append(row["callback_entry_ns"] - latest_exit)

    return {
        "device": device,
        "measurement_commands": len(profiles),
        "inference_commands": len(inference),
        "trace_received_measurement": len(traces_on),
        "on_valid_trace_cohort_n": len(on_cohort),
        "off_valid_trace_cohort_n": len(off_cohort),
        "off_start_to_publish": stats(elapsed_off),
        "on_start_to_publish": stats(elapsed_on),
        "callback_all_commands": stats(callback_all),
        "callback_inference_only": stats(callback_inference),
        "model_inference_only": stats(model),
        "commands_with_sensor_callback_since_previous": nonempty,
        "latest_sensor_callback_exit_to_control_entry": stats(latest_callback_gap),
        "sensors": sensor_result,
    }


def locate_raw(path: Path) -> Path:
    if (path / "cpu").is_dir():
        return path
    if (path / "raw" / "cpu").is_dir():
        return path / "raw"
    raise ValueError("input must point to archive raw/ or extracted package/raw/")


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Issue #49 offline replay analysis",
        "",
        "Quantiles use linear interpolation at `(n - 1) * p`; all durations are milliseconds. "
        "Sensor callback rows are deduplicated by sensor, header stamp, and callback-entry "
        "timestamp. Publication matching uses the latest same-sensor/header publication "
        "timestamp not later than callback entry.",
        "",
    ]
    for device in ("cpu", "cuda"):
        case = result[device]
        lines += [
            f"## {device.upper()}",
            "",
            f"Measurement command callbacks: {case['measurement_commands']}; inference "
            f"commands: {case['inference_commands']}; received ControlTrace records: "
            f"{case['trace_received_measurement']}. Valid-trace cohort: profile off "
            f"n={case['off_valid_trace_cohort_n']}, profile on "
            f"n={case['on_valid_trace_cohort_n']}.",
            "",
            "| Cohort / span | n | p50 | p95 | p99 |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, key in [
            ("Profile off control-start → command-published", "off_start_to_publish"),
            ("Profile on callback-entry → publish-done", "on_start_to_publish"),
            ("Profile on all command callbacks", "callback_all_commands"),
            ("Profile on inference-only callbacks", "callback_inference_only"),
            ("Model forward and completion, inference-only", "model_inference_only"),
            (
                "Latest sensor callback exit → control callback entry",
                "latest_sensor_callback_exit_to_control_entry",
            ),
        ]:
            s = case[key]
            lines.append(
                f"| {label} | {s['n']} | {s['p50_ms']:.3f} | "
                f"{s['p95_ms']:.3f} | {s['p99_ms']:.3f} |"
            )
        lines += [
            "",
            "The latest-sensor gap includes only commands with at least one accepted "
            f"sensor callback since the previous command "
            f"(n={case['commands_with_sensor_callback_since_previous']}); it is not "
            "publish-to-callback queue wait.",
            "",
            "| Sensor | Published | Accepted rows | Matched | Unmatched | "
            "Publish → callback entry p50 / p95 / p99 | Decode p95 | "
            "Hex-eye sample p95 | Buffer update p95 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for sensor in SENSORS:
            data = case["sensors"][sensor]
            wait = data["publish_to_callback_entry"]
            spans = data["spans"]

            def fmt(key: str, spans=spans) -> str:
                val = spans[key]["p95_ms"]
                return "—" if val is None else f"{val:.3f}"

            lines.append(
                f"| {sensor} | {data['published']} | {data['accepted_callbacks']} | "
                f"{data['matched_callbacks']} | {data['unmatched_callbacks']} | "
                f"{wait['p50_ms']:.3f} / {wait['p95_ms']:.3f} / "
                f"{wait['p99_ms']:.3f} | {fmt('image_decode')} | "
                f"{fmt('hex_eye_sample')} | {fmt('buffer_update')} |"
            )
        lines += [""]
    lines += [
        "## Limits",
        "",
        "This is an offline recomputation of the archived single replay. It reports one paired "
        "observation per device; it cannot establish causal or generally expected profiler "
        "overhead. The sensor publication-to-callback interval includes DDS and executor "
        "scheduling and does not isolate either. It must not be subtracted from the earlier "
        "bag acquisition-to-command G3 metric. Missing ControlTrace records remain missing; "
        "the profiler sidecar's emitted status is not subscriber-delivery confirmation.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="extracted raw/ or package/ directory")
    parser.add_argument("--output", type=Path, help="write rendered Markdown to this path")
    parser.add_argument("--json", type=Path, help="write machine-readable results to this path")
    args = parser.parse_args()
    raw = locate_raw(args.input)
    result = {device: analyze_case(raw, device) for device in ("cpu", "cuda")}
    rendered = render(result)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
