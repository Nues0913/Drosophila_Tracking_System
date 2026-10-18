"""Own rosbag's process and flush its index on normal ROS shutdown."""

from __future__ import annotations

import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

DEFAULT_TOPICS = [
    "/clock",
    "/fly/camera/left/image_raw",
    "/fly/camera/right/image_raw",
    "/fly/odor",
    "/fly/neural_activity",
    "/fly/inference_latency_ms",
    "/fly/sensor_timing",
    "/fly/control_trace",
    "/cmd_vel",
    "/world/agent/pose",
    "/world/target/pose",
    "/world/scenario",
    "/eval/metrics",
]


class BagRecorderNode(Node):
    def __init__(self):
        super().__init__("flytrack_bag_recorder")
        self.declare_parameter("output_dir", "runs/rosbags")
        self.declare_parameter("topics", DEFAULT_TOPICS)
        self.declare_parameter("storage_id", "sqlite3")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        output = Path(str(self.get_parameter("output_dir").value))
        output.mkdir(parents=True, exist_ok=True)
        bag = output / datetime.now(timezone.utc).strftime("flytrack_%Y%m%dT%H%M%S_%fZ")
        arguments = [
            "ros2",
            "bag",
            "record",
            "--output",
            str(bag),
            "--storage",
            str(self.get_parameter("storage_id").value),
        ]
        if bool(self.get_parameter("use_sim_time").value):
            arguments.append("--use-sim-time")
        arguments.extend(list(self.get_parameter("topics").value))
        self.process = subprocess.Popen(arguments)
        self.reported_exit = False
        self.create_timer(1.0, self.check_process)
        self.get_logger().info(f"Recording to {bag}")

    def check_process(self):
        code = self.process.poll()
        if code is not None and not self.reported_exit:
            self.reported_exit = True
            self.get_logger().error(f"rosbag process exited with status {code}")

    def destroy_node(self):
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BagRecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # normal launch shutdown; the finally block still cleans up
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
