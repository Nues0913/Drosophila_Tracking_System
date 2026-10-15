"""Only complete, ordered same-host sample traces can establish wall latency."""

from types import SimpleNamespace

from flytrack_eval.latency import CausalLatency


def trace(source=1000000, domain="same-host", received=3000000):
    stamp = SimpleNamespace(sec=1, nanosec=0)
    sample = SimpleNamespace(
        sensor="left",
        header=SimpleNamespace(stamp=stamp),
        clock_domain=domain,
        source_monotonic_ns=source,
        ingress_monotonic_ns=2000000,
        forward_monotonic_ns=2500000,
    )
    return SimpleNamespace(
        inference_performed=True,
        command_sequence=1,
        header=SimpleNamespace(stamp=stamp),
        clock_domain="same-host",
        control_started_monotonic_ns=4000000,
        command_published_monotonic_ns=6000000,
        valid=[True, False, False],
        samples=[sample, sample, sample],
        policy_received_monotonic_ns=[received, 0, 0],
    )


def test_full_latency_includes_bridge_and_scheduling():
    accumulator = CausalLatency()
    row = accumulator.append(trace())
    assert row["causal_clock_valid"]
    result = accumulator.result()
    assert result["sensor_to_command_wall_p95_ms"] == 5.0
    assert result["control_wall_p95_ms"] == 2.0
    assert result["source_to_ros_ingress_wall_p95_ms"] == 1.0
    assert result["causal_wall_trace_coverage"] == 1.0


def test_missing_source_and_cross_host_clocks_do_not_become_measurements():
    accumulator = CausalLatency()
    accumulator.append(trace(source=0))
    accumulator.append(trace(domain="other-host"))
    accumulator.append(trace(received=2000000))
    result = accumulator.result()
    assert result["causal_wall_invalid_traces"] == 3
    assert result["causal_wall_trace_coverage"] == 0.0


def test_newest_arriving_sensor_does_not_hide_oldest_contributing_sample():
    message = trace()
    message.valid = [True, True, False]
    newer = trace(source=1800000).samples[0]
    newer.sensor = "right"
    message.samples[1] = newer
    message.policy_received_monotonic_ns[1] = 3000000
    accumulator = CausalLatency()
    accumulator.append(message)
    assert accumulator.result()["sensor_to_command_wall_p95_ms"] == 5.0
