"""Opt-in JSONL timing sidecar for short policy replay diagnostics."""

from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

SENSORS = ("left", "right", "odor")


def header_key(sensor: str, header) -> tuple[str, int, int]:
    return sensor, int(header.stamp.sec), int(header.stamp.nanosec)


class PolicyProfiler:
    """Write correlated callback and control timings only when explicitly enabled."""

    def __init__(self, enabled: bool = False, output_path: str = "", metadata: dict | None = None):
        self.enabled = bool(enabled)
        self.run_id = ""
        self.command_sequence = 0
        self._stream = None
        self._records_since_flush = 0
        self._sensor_rows = {}
        self._sensor_callbacks = []
        if not self.enabled:
            return
        if not output_path:
            raise ValueError("profiler_output_path is required when profiler_enabled=true")
        self.run_id = uuid.uuid4().hex
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("x", encoding="utf-8")
        self._write(
            {
                "event": "run_start",
                "run_id": self.run_id,
                "monotonic_ns": time.monotonic_ns(),
                "clock": "CLOCK_MONOTONIC",
                "metadata": metadata or {},
            }
        )

    def _write(self, record: dict) -> None:
        if self._stream is not None:
            self._stream.write(json.dumps(record, sort_keys=True) + "\n")
            self._records_since_flush += 1
            if self._records_since_flush >= 100:
                self._stream.flush()
                self._records_since_flush = 0

    def begin_sensor_callback(self, sensor: str, header, entry_ns: int):
        if not self.enabled:
            return None
        return {
            "sensor": sensor,
            "header_stamp": [int(header.stamp.sec), int(header.stamp.nanosec)],
            "callback_entry_ns": int(entry_ns),
            "spans_ns": {},
        }

    @staticmethod
    def span(token, name: str, start_ns: int, end_ns: int) -> None:
        if token is not None:
            token["spans_ns"][name] = [int(start_ns), int(end_ns)]

    def end_sensor_callback(self, token, exit_ns: int, accepted: bool) -> None:
        if token is None:
            return
        token["callback_exit_ns"] = int(exit_ns)
        token["accepted"] = bool(accepted)
        record = {"event": "sensor_callback", "run_id": self.run_id, **token}
        key = (token["sensor"], *token["header_stamp"])
        self._sensor_rows[key] = record
        while len(self._sensor_rows) > 2048:
            self._sensor_rows.pop(next(iter(self._sensor_rows)))
        self._sensor_callbacks.append(record)

    def sensor_references(self, runtime_stamps, latest, valid) -> list[dict]:
        references = []
        if not self.enabled:
            return references
        for index, sensor in enumerate(SENSORS):
            header_key_value = None
            received_ns = 0
            current = latest[index]
            stamp = float(runtime_stamps[index])
            if current is not None and math.isfinite(stamp):
                header, received_ns = current
                header_seconds = header.stamp.sec + header.stamp.nanosec * 1e-9
                if abs(header_seconds - stamp) < 1e-9:
                    header_key_value = header_key(sensor, header)
            reference = {"sensor": sensor, "valid": bool(valid[index])}
            if header_key_value is not None:
                row = self._sensor_rows.get(header_key_value)
                callback_profiled = (
                    row is not None and row["accepted"] and row["callback_entry_ns"] == received_ns
                )
                reference.update(
                    {
                        "header_stamp": list(header_key_value[1:]),
                        "policy_received_ns": int(received_ns),
                        "callback_profiled": callback_profiled,
                    }
                )
                if callback_profiled:
                    reference["callback_entry_ns"] = row["callback_entry_ns"]
                    reference["callback_exit_ns"] = row["callback_exit_ns"]
                    reference["callback_spans_ns"] = row["spans_ns"]
            references.append(reference)
        return references

    def record_command(
        self,
        *,
        command_sequence: int,
        trace_sequence: int | None,
        trace_status: str,
        sim_stamp,
        callback_entry_ns: int,
        callback_exit_ns: int,
        publish_start_ns: int,
        publish_done_ns: int,
        spans_ns: dict,
        samples: list[dict],
        inference_performed: bool,
        valid,
    ) -> None:
        if not self.enabled:
            return
        self._write(
            {
                "event": "command_profile",
                "run_id": self.run_id,
                "command_sequence": int(command_sequence),
                "control_trace_sequence": (None if trace_sequence is None else int(trace_sequence)),
                "control_trace_status": trace_status,
                "sim_stamp": [int(sim_stamp.sec), int(sim_stamp.nanosec)],
                "inference_performed": bool(inference_performed),
                "valid": [bool(value) for value in valid],
                "callback_entry_ns": int(callback_entry_ns),
                "callback_exit_ns": int(callback_exit_ns),
                "publish_start_ns": int(publish_start_ns),
                "publish_done_ns": int(publish_done_ns),
                "spans_ns": spans_ns,
                "samples": samples,
                "sensor_callbacks_since_previous_command": self._sensor_callbacks,
            }
        )
        self._sensor_callbacks = []

    def next_command_sequence(self) -> int:
        sequence = self.command_sequence
        self.command_sequence += 1
        return sequence

    def record_reset(self, sim_stamp, monotonic_ns: int) -> None:
        if not self.enabled:
            return
        self._write(
            {
                "event": "reset",
                "run_id": self.run_id,
                "sim_stamp": [int(sim_stamp.sec), int(sim_stamp.nanosec)],
                "monotonic_ns": int(monotonic_ns),
                "sensor_callbacks_since_previous_command": self._sensor_callbacks,
            }
        )
        self._sensor_callbacks = []
        self._sensor_rows.clear()

    def close(self) -> None:
        if self._stream is None:
            return
        if self._sensor_callbacks:
            self._write(
                {
                    "event": "sensor_callbacks_unassigned",
                    "run_id": self.run_id,
                    "sensor_callbacks": self._sensor_callbacks,
                }
            )
        self._write(
            {"event": "run_end", "run_id": self.run_id, "monotonic_ns": time.monotonic_ns()}
        )
        self._stream.flush()
        self._stream.close()
        self._stream = None
