"""Validate per-command same-host causal clocks and summarize full sensor latency."""

from __future__ import annotations

import numpy as np


class CausalLatency:
    def __init__(self):
        self.traces = 0
        self.invalid = 0
        self.rows = []
        self.durations = {
            key: []
            for key in (
                "sensor_to_command",
                "source_to_ros_ingress",
                "corruption",
                "sensory_dds",
                "input_wait",
                "control",
            )
        }

    def append(self, trace):
        if not trace.inference_performed:
            return None
        self.traces += 1
        selected = []
        row = {
            "command_sequence": trace.command_sequence,
            "command_sim_stamp_ns": trace.header.stamp.sec * 1000000000
            + trace.header.stamp.nanosec,
            "clock_domain": trace.clock_domain,
            "control_started_monotonic_ns": trace.control_started_monotonic_ns,
            "command_published_monotonic_ns": trace.command_published_monotonic_ns,
            "samples": [],
        }
        valid = bool(trace.clock_domain)
        for index, sample in enumerate(trace.samples):
            if not trace.valid[index]:
                continue
            times = [
                sample.source_monotonic_ns,
                sample.ingress_monotonic_ns,
                sample.forward_monotonic_ns,
                trace.policy_received_monotonic_ns[index],
                trace.control_started_monotonic_ns,
                trace.command_published_monotonic_ns,
            ]
            sample_valid = (
                sample.clock_domain == trace.clock_domain
                and all(value > 0 for value in times)
                and all(a <= b for a, b in zip(times, times[1:], strict=False))
            )
            valid &= sample_valid
            selected.append(times)
            row["samples"].append(
                {
                    "sensor": sample.sensor,
                    "source_stamp_ns": sample.header.stamp.sec * 1000000000
                    + sample.header.stamp.nanosec,
                    "clock_domain": sample.clock_domain,
                    "source_monotonic_ns": times[0],
                    "ros_ingress_monotonic_ns": times[1],
                    "forward_monotonic_ns": times[2],
                    "policy_received_monotonic_ns": times[3],
                    "causal_clock_valid": sample_valid,
                }
            )
        valid &= bool(selected)
        row["causal_clock_valid"] = bool(valid)
        self.rows.append(row)
        if not valid:
            self.invalid += 1
            return row
        # The oldest actual contributing sample defines the per-command end-to-end delay.
        self.durations["sensor_to_command"].append(max(t[-1] - t[0] for t in selected) / 1e6)
        self.durations["source_to_ros_ingress"].append(max(t[1] - t[0] for t in selected) / 1e6)
        self.durations["corruption"].append(max(t[2] - t[1] for t in selected) / 1e6)
        self.durations["sensory_dds"].append(max(t[3] - t[2] for t in selected) / 1e6)
        self.durations["input_wait"].append(max(t[4] - t[3] for t in selected) / 1e6)
        self.durations["control"].append((selected[0][5] - selected[0][4]) / 1e6)
        return row

    def result(self):
        result = {
            "causal_wall_trace_samples": self.traces,
            "causal_wall_invalid_traces": self.invalid,
            "causal_wall_trace_coverage": (
                (self.traces - self.invalid) / self.traces if self.traces else 0.0
            ),
            "causal_wall_basis": (
                "same-host CLOCK_MONOTONIC: complete sensor message before "
                "source publication to completed cmd_vel publication; "
                "oldest valid contributing sample per command"
            ),
        }
        for stage, values in self.durations.items():
            for percentile in (50, 95):
                result[f"{stage}_wall_p{percentile}_ms"] = (
                    float(np.percentile(values, percentile)) if values else float("nan")
                )
        return result
