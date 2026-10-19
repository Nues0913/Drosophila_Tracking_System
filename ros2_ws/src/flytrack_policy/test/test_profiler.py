import json
from types import SimpleNamespace

import pytest
from flytrack_policy.profiling import PolicyProfiler


def header(sec=1, nanosec=250_000_000):
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))


def test_disabled_profiler_has_no_file_or_records(tmp_path):
    output_path = tmp_path / "disabled.jsonl"
    profiler = PolicyProfiler(enabled=False, output_path=str(output_path))

    token = profiler.begin_sensor_callback("left", header(), 100)
    profiler.end_sensor_callback(token, 200, True)
    profiler.record_reset(header(), 300)
    profiler.close()

    assert not output_path.exists()


def test_enabled_profiler_correlates_sensor_identity_and_command(tmp_path):
    output_path = tmp_path / "profile.jsonl"
    profiler = PolicyProfiler(enabled=True, output_path=str(output_path))
    sensor_header = header()
    token = profiler.begin_sensor_callback("left", sensor_header, 100)
    profiler.span(token, "image_decode", 110, 130)
    profiler.end_sensor_callback(token, 150, accepted=True)
    samples = profiler.sensor_references(
        [1.25, float("nan"), float("nan")],
        [(sensor_header, 100), None, None],
        [True, False, False],
    )

    assert samples[0]["callback_profiled"] is True
    assert samples[0]["header_stamp"] == [1, 250_000_000]
    assert samples[0]["callback_spans_ns"]["image_decode"] == [110, 130]
    profiler.record_command(
        command_sequence=0,
        trace_sequence=0,
        trace_status="publisher_called",
        sim_stamp=header(sec=2, nanosec=0).stamp,
        callback_entry_ns=90,
        callback_exit_ns=250,
        publish_start_ns=220,
        publish_done_ns=230,
        spans_ns={"action_limit": [180, 200]},
        samples=samples,
        inference_performed=True,
        valid=[True, False, False],
    )
    profiler.record_reset(header(sec=3, nanosec=0).stamp, 300)
    profiler.close()

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows[0]["event"] == "run_start"
    assert rows[0]["schema_version"] == 2
    command_row, reset_row, end_row = rows[1:]
    assert command_row["run_id"] == reset_row["run_id"]
    assert command_row["event"] == "command_profile"
    assert command_row["command_sequence"] == command_row["control_trace_sequence"] == 0
    assert command_row["control_trace_status"] == "publisher_called"
    assert command_row["samples"][0]["header_stamp"] == [1, 250_000_000]
    assert command_row["sensor_callbacks_since_previous_command"][0]["sensor"] == "left"
    assert reset_row["event"] == "reset"
    assert reset_row["discarded_pending_count"] == 0
    assert reset_row["discarded_pending_trace_sequences"] == []
    assert end_row["event"] == "run_end"


def test_enabled_profiler_refuses_to_append_to_existing_sidecar(tmp_path):
    output_path = tmp_path / "profile.jsonl"
    output_path.write_text("existing\n")

    with pytest.raises(FileExistsError):
        PolicyProfiler(enabled=True, output_path=str(output_path))
