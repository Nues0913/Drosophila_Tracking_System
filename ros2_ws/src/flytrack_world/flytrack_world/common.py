"""Small ROS adapter helpers for units, QoS and shared event contracts."""

import json
import math
from dataclasses import fields
from pathlib import Path

import rclpy
from flytrack_core.contracts import SensorContract
from flytrack_core.world import ScenarioSpec
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

SENSOR_QOS = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
CAMERA_QOS = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
POSE_QOS = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
EVENT_QOS = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL
)
SCENARIO_FIELDS = {field.name for field in fields(ScenarioSpec)}
PROTOCOL_FIELDS = {
    "phase",
    "episode_start_s",
    "occluded",
    "collision",
    "target_motion",
    "plume_model",
    "plume_config",
    "trajectory_config",
    "layout",
    "arena_size_m",
    "arena_center_x",
    "arena_center_y",
    "agent_radius_m",
    "target_radius_m",
    "obstacles",
    "initial_agent_pose",
    "initial_target_pose",
    "training_seed",
    "environment_seed",
}


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def now_seconds(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


def pose_xy_yaw(message):
    pose = message.pose
    q = pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return [pose.position.x, pose.position.y, yaw]


def param(node: Node, name: str, default):
    return node.declare_parameter(name, default).value


class WorldNode(Node):
    def __init__(self, name: str):
        super().__init__(name)
        path = param(self, "sensor_contract_json", "")
        self.contract = (
            SensorContract.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
            if path
            else SensorContract()
        )
        self.scenario_name = param(self, "scenario", "normal")
        self.base_seed = int(param(self, "seed", 0))
        self.episode_seconds = float(param(self, "episode_seconds", 30.0))
        self.world_frame = param(self, "world_frame", "world")
        self.active = False
        self.episode_id = None
        self.started_at_s = 0.0
        scenario_path = param(self, "scenario_config_path", "")
        values = {}
        if scenario_path:
            raw = Path(scenario_path).read_text(encoding="utf-8")
            if Path(scenario_path).suffix.lower() == ".json":
                values = json.loads(raw)
            else:
                import yaml  # ROS adapter dependency; core remains numpy/torch-only.

                values = yaml.safe_load(raw)
            if not isinstance(values, dict):
                raise ValueError("scenario config must be a mapping")
        self.scenario_spec = ScenarioSpec.from_dict(
            {**values, "duration_s": self.episode_seconds}, name=self.scenario_name
        )

    def consume_event(self, message) -> tuple[bool, dict]:
        """A new active episode resets state; phase events preserve neural/world history."""
        values = json.loads(message.parameters_json or "{}")
        if not isinstance(values, dict):
            raise ValueError("scenario event parameters must be an object")
        unknown = set(values) - SCENARIO_FIELDS - PROTOCOL_FIELDS
        if unknown:
            raise ValueError(f"unknown scenario event parameters: {sorted(unknown)}")
        self.scenario_spec = ScenarioSpec.from_dict(
            {key: value for key, value in values.items() if key in SCENARIO_FIELDS},
            name=message.scenario,
        )
        self.active = bool(message.active)
        new_episode = self.active and message.episode_id != self.episode_id
        if new_episode:
            self.episode_id = message.episode_id
            self.started_at_s = float(
                values.get("episode_start_s", stamp_seconds(message.header.stamp))
            )
        return new_episode, values


def spin_main(node_type, args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = node_type()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # normal launch shutdown; the finally block still cleans up
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
