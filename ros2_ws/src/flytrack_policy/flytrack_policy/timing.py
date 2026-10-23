"""Causal command traces assembled independently of the actor data path."""

from __future__ import annotations

import math
import os
import time
from collections import OrderedDict
from copy import deepcopy

from flytrack_interfaces.msg import ControlTrace, SensorTiming
from rclpy.qos import QoSProfile, ReliabilityPolicy

TIMING_QOS = QoSProfile(depth=256, reliability=ReliabilityPolicy.RELIABLE)
SENSORS = ("left", "right", "odor")


def key(sensor, stamp):
    return sensor, int(stamp.sec), int(stamp.nanosec)


class ControlTracer:
    def __init__(self, node, on_trace_state=None):
        self.domain = os.environ.get("FLYTRACK_CLOCK_DOMAIN", "")
        self.on_trace_state = on_trace_state
        self.publisher = node.create_publisher(ControlTrace, "/fly/control_trace", TIMING_QOS)
        self.subscription = node.create_subscription(
            SensorTiming, "/fly/sensor_timing", self.on_timing, TIMING_QOS
        )
        self.sequence = 0
        self.pending = []
        self.reset()

    def reset(self):
        discarded = self.pending_trace_sequences()
        self.timings = OrderedDict()
        self.latest = [None, None, None]
        self.pending = []
        return discarded

    def pending_trace_sequences(self):
        return [trace.command_sequence for _, trace, _ in self.pending]

    def report_state(self, trace, state, reason):
        if self.on_trace_state is not None:
            self.on_trace_state(trace.command_sequence, state, reason, time.monotonic_ns())

    def on_timing(self, message):
        sample = key(message.sensor, message.header.stamp)
        previous = self.timings.get(sample)
        if previous is None or message.source_monotonic_ns:
            self.timings[sample] = deepcopy(message)
        while len(self.timings) > 2048:
            self.timings.popitem(last=False)
        self.drain()

    def received(self, sensor_index, header, monotonic_ns):
        self.latest[sensor_index] = deepcopy(header), monotonic_ns

    def command(self, stamp, started_ns, published_ns, valid, runtime_stamps, inference):
        trace = ControlTrace()
        trace.header.stamp, trace.header.frame_id = stamp, "base_link"
        trace.command_sequence = self.sequence
        self.sequence += 1
        trace.clock_domain = self.domain
        trace.control_started_monotonic_ns = started_ns
        trace.command_published_monotonic_ns = published_ns
        trace.inference_performed = bool(inference)
        trace.valid = [bool(value) for value in valid]
        identities = []
        for index, sensor in enumerate(SENSORS):
            sample = SensorTiming()
            sample.sensor = sensor
            current = self.latest[index]
            identity = None
            if current is not None and math.isfinite(runtime_stamps[index]):
                header, received_ns = current
                seconds = header.stamp.sec + header.stamp.nanosec * 1e-9
                if abs(seconds - runtime_stamps[index]) < 1e-9:
                    sample.header = deepcopy(header)
                    identity = key(sensor, header.stamp)
                    trace.policy_received_monotonic_ns[index] = received_ns
            trace.samples[index] = sample
            identities.append(identity)
        self.pending.append((time.monotonic_ns() + 1000000000, trace, identities))
        self.report_state(trace, "pending", "submitted_to_tracer")
        self.drain()
        status = (
            "pending"
            if trace.command_sequence in self.pending_trace_sequences()
            else "publisher_called"
        )
        return trace.command_sequence, status

    def drain(self):
        remaining = []
        now = time.monotonic_ns()
        for deadline, trace, identities in self.pending:
            complete = True
            for index, identity in enumerate(identities):
                if not trace.valid[index]:
                    continue
                timing = self.timings.get(identity)
                if timing is None or not timing.source_monotonic_ns:
                    complete = False
                if timing is not None:
                    trace.samples[index] = deepcopy(timing)
            if complete or now >= deadline:
                self.publisher.publish(trace)
                reason = "required_timings_available" if complete else "deadline_expired"
                self.report_state(trace, "publisher_called", reason)
            else:
                remaining.append((deadline, trace, identities))
        self.pending = remaining
