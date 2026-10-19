import json
from types import SimpleNamespace

import numpy as np
import pytest


class FakeRuntime:
    def __init__(self):
        self.stamps = np.full(3, np.nan)
        self.last_inference_latency_ms = None
        self.last_profile_spans_ns = {}
        self._ticks = iter(
            [
                ValueError("synthetic tick failure"),
                (np.array([0.1, 0.0]), None, np.array([False, False, False])),
            ]
        )

    def tick(self, now):
        result = next(self._ticks)
        if isinstance(result, Exception):
            raise result
        return result

    def reset(self, minimum_stamp=None):
        pass


class FakeTracer:
    def __init__(self):
        self.sequence = 0
        self.latest = [None, None, None]
        self.emitted = []

    def command(self, *args, **kwargs):
        sequence = self.sequence
        self.emitted.append(sequence)
        self.sequence += 1
        return sequence, "publisher_called"


def test_control_callback_does_not_claim_trace_for_failed_tick(tmp_path):
    pytest.importorskip("torch")
    from flytrack_policy.node import ConnectomePolicyNode
    from flytrack_policy.profiling import PolicyProfiler

    output_path = tmp_path / "callback-profile.jsonl"
    profiler = PolicyProfiler(enabled=True, output_path=str(output_path))
    tracer = FakeTracer()
    clock_time = SimpleNamespace(
        nanoseconds=1_000_000_000,
        to_msg=lambda: SimpleNamespace(
            sec=1,
            nanosec=0,
        ),
    )
    logger = SimpleNamespace(error=lambda *args, **kwargs: None)
    fake_node = SimpleNamespace(
        profiler=profiler,
        tracer=tracer,
        runtime=FakeRuntime(),
        get_clock=lambda: SimpleNamespace(now=lambda: clock_time),
        command_publisher=SimpleNamespace(publish=lambda message: None),
        publish_stop=lambda: None,
        latency_publisher=SimpleNamespace(publish=lambda message: None),
        get_logger=lambda: logger,
        last_activity_stamp=-float("inf"),
    )

    ConnectomePolicyNode.control_callback(fake_node)
    ConnectomePolicyNode.control_callback(fake_node)
    profiler.close()

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    commands = [row for row in records if row["event"] == "command_profile"]
    assert [(row["command_sequence"], row["control_trace_sequence"]) for row in commands] == [
        (0, None),
        (1, 0),
    ]
    assert [row["control_trace_status"] for row in commands] == [
        "not_submitted",
        "publisher_called",
    ]
    assert tracer.emitted == [0]
