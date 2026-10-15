"""Join source-publication timing with preserved sensor sample identities."""

from __future__ import annotations

import os
from collections import OrderedDict
from copy import deepcopy

from flytrack_interfaces.msg import SensorTiming
from rclpy.qos import QoSProfile, ReliabilityPolicy

TIMING_QOS = QoSProfile(depth=256, reliability=ReliabilityPolicy.RELIABLE)


def sample_key(sensor, stamp):
    return sensor, int(stamp.sec), int(stamp.nanosec)


class SensorTimingRelay:
    def __init__(self, node):
        self.node = node
        self.domain = os.environ.get("FLYTRACK_CLOCK_DOMAIN", "")
        self.publisher = node.create_publisher(SensorTiming, "/fly/sensor_timing", TIMING_QOS)
        self.sources = OrderedDict()
        self.forwarded = OrderedDict()

    def reset(self):
        self.sources.clear()
        self.forwarded.clear()

    @staticmethod
    def _trim(values):
        while len(values) > 2048:
            values.popitem(last=False)

    def camera_source(self, sensor, message):
        try:
            sec, nanosec, wall, domain = message.data.split(",")
            key = sensor, int(sec), int(nanosec)
            wall = int(wall)
            if wall <= 0 or not 0 <= key[2] < 1000000000:
                raise ValueError("invalid source timing")
        except (ValueError, TypeError):
            self.node.get_logger().warning("Rejected malformed camera source timing")
            return
        self.sources[key] = wall, domain
        self._trim(self.sources)
        if key in self.forwarded:
            # Metadata may arrive after the image on the independent ROS bridge.
            timing = deepcopy(self.forwarded[key])
            timing.source_monotonic_ns, timing.clock_domain = wall, domain
            self.publisher.publish(timing)

    def record(self, header, sensor, ingress_ns, forward_ns, source_ns=0, domain=None):
        key = sample_key(sensor, header.stamp)
        source_ns, domain = self.sources.get(key, (source_ns, domain or self.domain))
        timing = SensorTiming()
        timing.header = deepcopy(header)
        timing.sensor, timing.clock_domain = sensor, domain
        timing.source_monotonic_ns = source_ns
        timing.ingress_monotonic_ns, timing.forward_monotonic_ns = ingress_ns, forward_ns
        self.forwarded[key] = timing
        self._trim(self.forwarded)
        self.publisher.publish(timing)
