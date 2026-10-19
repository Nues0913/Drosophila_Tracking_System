import json

from builtin_interfaces.msg import Time
from flytrack_interfaces.msg import ControlTrace, SensorTiming
from flytrack_policy.profiling import PolicyProfiler
from flytrack_policy.timing import ControlTracer
from std_msgs.msg import Header


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class FakeNode:
    def __init__(self):
        self.publisher = FakePublisher()

    def create_publisher(self, message_type, topic, qos):
        assert message_type is ControlTrace
        assert topic == "/fly/control_trace"
        return self.publisher

    def create_subscription(self, message_type, topic, callback, qos):
        assert message_type is SensorTiming
        assert topic == "/fly/sensor_timing"
        self.timing_callback = callback
        return object()


def sensor_timing(sensor: str, sec: int = 1, nanosec: int = 250_000_000):
    message = SensorTiming()
    message.sensor = sensor
    message.header = Header(stamp=Time(sec=sec, nanosec=nanosec))
    message.clock_domain = "test-clock"
    message.source_monotonic_ns = 123456
    return message


def make_tracer(on_state=None):
    node = FakeNode()
    tracer = ControlTracer(node, on_trace_state=on_state)
    header = Header(stamp=Time(sec=1, nanosec=250_000_000))
    tracer.latest = [(header, 100), (header, 110), (header, 120)]
    return tracer, node


def submit(tracer):
    return tracer.command(
        Time(sec=2, nanosec=0),
        started_ns=200,
        published_ns=210,
        valid=[True, True, True],
        runtime_stamps=[1.25, 1.25, 1.25],
        inference=True,
    )


def test_tracer_reports_immediate_publish_after_submission():
    states = []
    tracer, node = make_tracer(lambda *event: states.append(event))
    for sensor in ("left", "right", "odor"):
        tracer.on_timing(sensor_timing(sensor))

    assert submit(tracer) == (0, "publisher_called")
    assert len(node.publisher.messages) == 1
    assert node.publisher.messages[0].command_sequence == 0
    assert states[0][1:3] == ("pending", "submitted_to_tracer")
    assert states[-1][1:3] == ("publisher_called", "required_timings_available")


def test_pending_trace_is_published_when_required_timing_arrives():
    states = []
    tracer, node = make_tracer(lambda *event: states.append(event))

    assert submit(tracer) == (0, "pending")
    assert tracer.pending_trace_sequences() == [0]
    for sensor in ("left", "right"):
        tracer.on_timing(sensor_timing(sensor))
    assert node.publisher.messages == []
    tracer.on_timing(sensor_timing("odor"))

    assert [message.command_sequence for message in node.publisher.messages] == [0]
    assert states[-1][1:3] == ("publisher_called", "required_timings_available")


def test_pending_trace_timeout_publishes_with_timeout_reason(monkeypatch):
    now = [100]
    monkeypatch.setattr("flytrack_policy.timing.time.monotonic_ns", lambda: now[0])
    states = []
    tracer, node = make_tracer(lambda *event: states.append(event))

    assert submit(tracer) == (0, "pending")
    now[0] = 1_000_000_101
    tracer.drain()

    assert [message.command_sequence for message in node.publisher.messages] == [0]
    assert states[-1][1:3] == ("publisher_called", "deadline_expired")


def test_reset_reports_discarded_sequences_without_publishing_or_rewinding():
    tracer, node = make_tracer()
    assert submit(tracer) == (0, "pending")

    assert tracer.reset() == [0]
    assert tracer.pending_trace_sequences() == []
    assert node.publisher.messages == []
    assert submit(tracer) == (1, "pending")


def test_profiler_sidecar_links_pending_state_to_reset_discard(tmp_path):
    output = tmp_path / "trace-state.jsonl"
    profiler = PolicyProfiler(enabled=True, output_path=str(output))
    tracer, _ = make_tracer(on_state=profiler.record_trace_state)
    sequence, status = submit(tracer)
    profiler.record_command(
        command_sequence=0,
        trace_sequence=sequence,
        trace_status=status,
        sim_stamp=Time(sec=2, nanosec=0),
        callback_entry_ns=100,
        callback_exit_ns=200,
        publish_start_ns=180,
        publish_done_ns=190,
        spans_ns={},
        samples=[],
        inference_performed=True,
        valid=[True, True, True],
    )
    profiler.record_reset(
        Time(sec=3, nanosec=0),
        300,
        discarded_trace_sequences=tracer.pending_trace_sequences(),
    )
    tracer.reset()
    profiler.close()

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    command = next(row for row in rows if row["event"] == "command_profile")
    states = [row for row in rows if row["event"] == "control_trace_state"]
    reset = next(row for row in rows if row["event"] == "reset")
    assert command["control_trace_sequence"] == 0
    assert command["control_trace_status"] == "pending"
    assert [(row["trace_sequence"], row["state"]) for row in states] == [(0, "pending")]
    assert reset["discarded_pending_count"] == 1
    assert reset["discarded_pending_trace_sequences"] == [0]
    assert reset["discard_reason"] == "reset_cleared_pending"


def test_profiler_off_tracer_preserves_trace_and_reset_behavior():
    tracer, node = make_tracer()
    assert submit(tracer) == (0, "pending")
    assert tracer.reset() == [0]
    assert node.publisher.messages == []

    header = Header(stamp=Time(sec=1, nanosec=250_000_000))
    tracer.latest = [(header, 100), (header, 110), (header, 120)]
    for sensor in ("left", "right", "odor"):
        tracer.on_timing(sensor_timing(sensor))

    assert submit(tracer) == (1, "publisher_called")
    assert [message.command_sequence for message in node.publisher.messages] == [1]
