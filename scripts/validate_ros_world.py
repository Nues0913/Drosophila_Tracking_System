"""Exercise Gazebo sensors, drive bridge and two resets without a learned actor.

This diagnostic owns the policy reset service while it runs. It refuses to run
beside an actual policy. Its output proves world/bridge behavior only; it is not
a learned-policy evaluation or a scientific tracking result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import rclpy
from flytrack_interfaces.msg import OdorReading, ScenarioEvent, SensorTiming
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class WorldProbe(Node):
    def __init__(self):
        super().__init__("flytrack_world_validation")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        self.set_parameters([Parameter("use_sim_time", value=True)])
        self.counts = dict(clock=0, left=0, right=0, odor=0, agent=0, target=0)
        self.stamps = {key: [] for key in self.counts}
        self.events = []
        self.errors = []
        self.completed = set()
        self.active_episode = None
        self.episode_start = None
        self.poses = {}
        self.reset_calls = 0
        self.clock = None
        self.timings = {}
        self.clock_domain = os.environ.get("FLYTRACK_CLOCK_DOMAIN", "")
        self.command = self.create_publisher(Twist, "/cmd_vel", 1)
        self.reset = self.create_service(Trigger, "/connectome_policy_node/reset", self.on_reset)
        sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        event = QoSProfile(
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Clock,
            "/clock",
            self.on_clock,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(
            Image, "/fly/camera/left/image_raw", lambda msg: self.on_image("left", msg), sensor
        )
        self.create_subscription(
            Image, "/fly/camera/right/image_raw", lambda msg: self.on_image("right", msg), sensor
        )
        self.create_subscription(OdorReading, "/fly/odor", self.on_odor, sensor)
        self.create_subscription(PoseStamped, "/world/agent/pose", self.on_agent, 10)
        self.create_subscription(PoseStamped, "/world/target/pose", self.on_target, 10)
        self.create_subscription(ScenarioEvent, "/world/scenario", self.on_event, event)
        self.create_subscription(SensorTiming, "/fly/sensor_timing", self.on_timing, 256)
        self.create_timer(0.05, self.drive)

    def record(self, key, stamp):
        seconds = stamp_seconds(stamp)
        self.counts[key] += 1
        self.stamps[key].append(seconds)

    def on_clock(self, msg):
        self.clock = stamp_seconds(msg.clock)
        self.record("clock", msg.clock)

    def on_image(self, key, msg):
        self.record(key, msg.header.stamp)
        if (msg.width, msg.height, msg.encoding.lower()) != (160, 120, "mono8"):
            self.errors.append(f"{key}: unexpected camera dimensions/encoding")
        if len(msg.data) != msg.step * msg.height or msg.step < msg.width:
            self.errors.append(f"{key}: invalid image stride/payload")

    def on_odor(self, msg):
        self.record("odor", msg.header.stamp)
        if not all(math.isfinite(value) and value >= 0 for value in (msg.left, msg.right)):
            self.errors.append("nonfinite or negative odor")

    def on_agent(self, msg):
        self.record("agent", msg.header.stamp)
        if self.active_episode is not None:
            self.poses.setdefault(self.active_episode, []).append(
                [
                    stamp_seconds(msg.header.stamp),
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                ]
            )

    def on_target(self, msg):
        self.record("target", msg.header.stamp)

    def on_timing(self, msg):
        identity = msg.sensor, msg.header.stamp.sec, msg.header.stamp.nanosec
        previous = self.timings.get(identity)
        if previous is None or msg.source_monotonic_ns:
            self.timings[identity] = msg

    def on_reset(self, request, response):
        self.reset_calls += 1
        self.command.publish(Twist())
        response.success, response.message = (
            True,
            "World validation reset acknowledged; no learned actor is present",
        )
        return response

    def on_event(self, msg):
        values = json.loads(msg.parameters_json)
        self.events.append(
            {
                "episode": msg.episode_id,
                "active": msg.active,
                "phase": values["phase"],
                "stamp": stamp_seconds(msg.header.stamp),
            }
        )
        if msg.active:
            self.active_episode = msg.episode_id
            self.episode_start = values["episode_start_s"]
        else:
            self.active_episode = None
            self.episode_start = None
            self.command.publish(Twist())
            if values["phase"] == "completed":
                self.completed.add(msg.episode_id)

    def drive(self):
        message = Twist()
        if self.episode_start is not None and self.clock is not None:
            elapsed = self.clock - self.episode_start
            if 0.5 <= elapsed <= 1.5:
                message.linear.x = 0.1
        self.command.publish(message)

    def report(self):
        rates = {}
        for key, stamps in self.stamps.items():
            differences = [b - a for a, b in zip(stamps, stamps[1:], strict=False) if b > a]
            rates[key] = (len(differences) / sum(differences)) if differences else None
            if any(b < a for a, b in zip(stamps, stamps[1:], strict=False)):
                self.errors.append(f"{key}: timestamp went backwards")
        for key, count in self.counts.items():
            if count < 2:
                self.errors.append(f"{key}: fewer than two samples")
        if len(self.completed) < 2 or self.reset_calls < 4:
            self.errors.append("two episodes and their before/after reset calls did not complete")
        displacements = {}
        for episode, values in self.poses.items():
            if len(values) >= 2:
                first, last = values[0], values[-1]
                displacement = math.hypot(last[1] - first[1], last[2] - first[2])
                displacements[str(episode)] = displacement
                if displacement < 0.02:
                    self.errors.append(
                        f"episode {episode}: drive command produced less than 2cm displacement"
                    )
        timed = {sensor: 0 for sensor in ("left", "right", "odor")}
        forwarded = dict(timed)
        for sample in self.timings.values():
            forwarded[sample.sensor] += 1
            if sample.source_monotonic_ns:
                ordered = (
                    0
                    < sample.source_monotonic_ns
                    <= sample.ingress_monotonic_ns
                    <= sample.forward_monotonic_ns
                )
                if not ordered or sample.clock_domain != self.clock_domain:
                    self.errors.append("source timing has an invalid domain or causal order")
                else:
                    timed[sample.sensor] += 1
        for sensor in timed:
            if timed[sensor] < 10 or timed[sensor] / max(1, forwarded[sensor]) < 0.95:
                self.errors.append(f"{sensor}: source timing coverage below 95% or 10 samples")
        return {
            "scope": "world/bridge validation with diagnostic reset service; no learned actor",
            "passed": not self.errors,
            "errors": sorted(set(self.errors)),
            "counts": self.counts,
            "observed_stamp_rates_hz": rates,
            "completed_episodes": sorted(self.completed),
            "reset_calls": self.reset_calls,
            "agent_displacement_m": displacements,
            "source_timing_forwarded_samples": forwarded,
            "source_timing_complete_samples": timed,
            "events": self.events,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    rclpy.init()
    node = WorldProbe()
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline and len(node.completed) < 2:
            rclpy.spin_once(node, timeout_sec=0.1)
            policy_nodes = {name for name, _ in node.get_node_names_and_namespaces()}
            if "connectome_policy_node" in policy_nodes:
                raise RuntimeError(
                    "Stop the actual policy before using the diagnostic reset service"
                )
        report = node.report()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        node.command.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
