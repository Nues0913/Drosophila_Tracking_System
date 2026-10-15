"""Truth belongs to this observer, never to the policy subscription graph."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import rclpy
from flytrack_core.contracts import VARIANTS
from flytrack_core.metrics import EpisodeMetrics
from flytrack_interfaces.msg import ControlTrace, OdorReading, ScenarioEvent, TrackingMetrics
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

from .latency import CausalLatency


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def yaw(pose):
    q = pose.orientation
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class EvaluatorNode(Node):
    def __init__(self):
        super().__init__("flytrack_evaluator")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", True)
        if not self.get_parameter("use_sim_time").value:
            raise ValueError("Gazebo evaluation requires use_sim_time=true")
        self.declare_parameter("variant", "VO")
        self.declare_parameter("training_seed", -1)
        self.declare_parameter("artifact", "")
        self.variant = str(self.get_parameter("variant").value)
        self.training_seed = int(self.get_parameter("training_seed").value)
        self.model_card_sha256 = ""
        artifact = str(self.get_parameter("artifact").value)
        if artifact:
            card_bytes = (Path(artifact) / "model_card.json").read_bytes()
            card = json.loads(card_bytes)
            if card.get("schema_version") != 1 or card.get("variant") not in VARIANTS:
                raise ValueError("Evaluator artifact has an unsupported model card")
            seed = card.get("metadata", {}).get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(
                    "Evaluator artifact model card must identify its nonnegative training seed"
                )
            self.variant, self.training_seed = card["variant"], seed
            self.model_card_sha256 = hashlib.sha256(card_bytes).hexdigest()
            self.get_logger().info(
                f"Artifact model card sets variant={self.variant}, "
                f"training_seed={self.training_seed}; artifact overrides labeling parameters"
            )
        self.declare_parameter("output_dir", "runs/gazebo_eval")
        self.declare_parameter("control_dt", 0.05)
        self.declare_parameter("arena_size_m", 10.0)
        self.declare_parameter("agent_radius_m", 0.18)
        self.declare_parameter("collision_distance_m", 0.33)
        self.declare_parameter("pose_timeout_s", 0.2)
        self.declare_parameter("success_distance_m", 2.0)
        self.declare_parameter("success_time_fraction", 0.8)
        self.declare_parameter("minimum_sample_coverage", 0.95)
        self.output = Path(str(self.get_parameter("output_dir").value))
        self.output.mkdir(parents=True, exist_ok=True)
        self.agent = self.target = None
        self.omega = 0.0
        self.sensor_stamps = {}
        self.pending_latency = None
        self.inference_latencies = []
        self.causal_latency = CausalLatency()
        self.event = None
        self.parameters = {}
        self.accumulator = None
        self.started_s = None
        self.last_sample_s = None
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        sensor = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
        scenario = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(PoseStamped, "/world/agent/pose", self.on_agent, reliable)
        self.create_subscription(PoseStamped, "/world/target/pose", self.on_target, reliable)
        self.create_subscription(ScenarioEvent, "/world/scenario", self.on_scenario, scenario)
        self.create_subscription(
            Twist,
            "/cmd_vel",
            self.on_command,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        for name in ("left", "right"):
            self.create_subscription(
                Image,
                f"/fly/camera/{name}/image_raw",
                lambda message, key=name: self.on_sensor(key, message),
                sensor,
            )
        self.create_subscription(
            OdorReading, "/fly/odor", lambda message: self.on_sensor("odor", message), sensor
        )
        self.create_subscription(
            Float64, "/fly/inference_latency_ms", self.on_inference_latency, sensor
        )
        self.create_subscription(
            ControlTrace,
            "/fly/control_trace",
            self.on_control_trace,
            QoSProfile(depth=256, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.publisher = self.create_publisher(TrackingMetrics, "/eval/metrics", reliable)
        self.create_timer(float(self.get_parameter("control_dt").value), self.sample)

    def on_agent(self, message):
        self.agent = message

    def on_target(self, message):
        self.target = message

    def on_sensor(self, key, message):
        self.sensor_stamps[key] = stamp_seconds(message.header.stamp)

    def on_command(self, message):
        self.omega = float(message.angular.z)
        now = self.get_clock().now().nanoseconds * 1e-9
        fresh = [stamp for stamp in self.sensor_stamps.values() if 0 <= now - stamp <= 0.2]
        if fresh:
            # Twist has no header. This is latest-sensor age at command receipt,
            # without an exact causal association.
            self.pending_latency = max(0.0, now - min(fresh)) * 1000

    def on_inference_latency(self, message):
        value = float(message.data)
        if self.accumulator is not None and math.isfinite(value) and value >= 0:
            self.inference_latencies.append(value)

    def on_control_trace(self, message):
        if self.accumulator is None or stamp_seconds(message.header.stamp) < self.started_s:
            return
        row = self.causal_latency.append(message)
        if row is not None:
            row["episode_id"] = self.event.episode_id
            with (self.output / "control_traces.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    def on_scenario(self, message):
        if message.active:
            if self.event and message.episode_id == self.event.episode_id:
                return
            if self.accumulator is not None:
                self.finish("superseded_by_new_episode")
            try:
                parameters = json.loads(message.parameters_json or "{}")
                if not isinstance(parameters, dict):
                    raise ValueError("parameters_json must be a mapping")
            except (json.JSONDecodeError, ValueError) as error:
                self.get_logger().error(str(error))
                return
            started_s = float(
                parameters.get("episode_start_s", stamp_seconds(message.header.stamp))
            )
            if (
                not math.isfinite(started_s)
                or started_s > stamp_seconds(message.header.stamp) + 1e-6
            ):
                self.get_logger().error(
                    "Scenario episode start is invalid or later than its event stamp"
                )
                return
            self.event, self.parameters = message, parameters
            self.started_s = started_s
            self.last_sample_s = None
            self.sensor_stamps.clear()
            self.pending_latency = None
            self.inference_latencies = []
            self.causal_latency = CausalLatency()
            self.omega = 0.0
            self.accumulator = EpisodeMetrics(
                float(self.get_parameter("control_dt").value),
                float(self.get_parameter("success_distance_m").value),
                float(self.get_parameter("success_time_fraction").value),
            )
        elif self.event is not None and message.episode_id == self.event.episode_id:
            try:
                terminal = json.loads(message.parameters_json or "{}")
                self.accumulator.collision |= bool(terminal.get("collision", False))
            except (json.JSONDecodeError, AttributeError):
                pass
            self.finish("scenario_complete")

    def _collision(self, distance):
        position = self.agent.pose.position
        half = (
            float(self.parameters.get("arena_size_m", self.get_parameter("arena_size_m").value)) / 2
        )
        radius = float(
            self.parameters.get("agent_radius_m", self.get_parameter("agent_radius_m").value)
        )
        cx = float(self.parameters.get("arena_center_x", 0))
        cy = float(self.parameters.get("arena_center_y", 0))
        if max(abs(position.x - cx), abs(position.y - cy)) + radius >= half:
            return True
        target_radius = float(
            self.parameters.get(
                "target_radius_m", float(self.get_parameter("collision_distance_m").value) - radius
            )
        )
        if distance <= radius + target_radius:
            return True
        for obstacle in self.parameters.get("obstacles", []):
            if "radius" in obstacle:
                if math.hypot(
                    position.x - float(obstacle["x"]), position.y - float(obstacle["y"])
                ) <= radius + float(obstacle["radius"]):
                    return True
                continue
            x = min(max(position.x, float(obstacle["xmin"])), float(obstacle["xmax"]))
            y = min(max(position.y, float(obstacle["ymin"])), float(obstacle["ymax"]))
            if math.hypot(position.x - x, position.y - y) <= radius:
                return True
        return False

    def sample(self):
        if self.accumulator is None or self.agent is None or self.target is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        timeout = float(self.get_parameter("pose_timeout_s").value)
        if self.last_sample_s is not None and now <= self.last_sample_s:
            return
        stamps = (stamp_seconds(self.agent.header.stamp), stamp_seconds(self.target.header.stamp))
        if any(stamp < self.started_s or not 0 <= now - stamp <= timeout for stamp in stamps):
            return
        dx = self.target.pose.position.x - self.agent.pose.position.x
        dy = self.target.pose.position.y - self.agent.pose.position.y
        bearing = math.atan2(
            math.sin(math.atan2(dy, dx) - yaw(self.agent.pose)),
            math.cos(math.atan2(dy, dx) - yaw(self.agent.pose)),
        )
        elapsed = now - self.started_s
        start = float(self.parameters.get("occlusion_start_s", 5.0))
        duration = float(self.parameters.get("occlusion_duration_s", 3.0))
        occluded = self.event.scenario == "occlusion" and start <= elapsed < start + duration
        distance = math.hypot(dx, dy)
        if not all(math.isfinite(value) for value in (distance, bearing, self.omega, elapsed)):
            return
        self.accumulator.append(
            distance,
            bearing,
            self.omega,
            self._collision(distance),
            occluded,
            elapsed,
            self.pending_latency,
        )
        self.pending_latency = None
        self.last_sample_s = now

    def finish(self, reason):
        if self.accumulator is None:
            return
        elapsed = max(0.0, self.get_clock().now().nanoseconds * 1e-9 - self.started_s)
        dt = float(self.get_parameter("control_dt").value)
        duration = float(self.parameters.get("duration_s", elapsed))
        terminal_collision = reason == "scenario_complete" and self.accumulator.collision
        measured_duration = min(duration, elapsed) if terminal_collision else duration
        expected_samples = max(1, int(math.ceil(measured_duration / dt)))
        coverage = min(1.0, len(self.accumulator.distances) / expected_samples)
        complete = reason == "scenario_complete" and (
            elapsed + 2 * dt >= duration or terminal_collision
        )
        valid = complete and coverage >= float(self.get_parameter("minimum_sample_coverage").value)
        if self.accumulator.distances:
            values = self.accumulator.result()
        else:
            values = {
                "success": 0.0,
                "mean_distance_m": float("nan"),
                "mean_bearing_error_deg": float("nan"),
                "reacquisition_time_s": float("nan"),
                "smoothness_rad_s": float("nan"),
                "collision_rate": float(self.accumulator.collision),
                "latency_p50_ms": float("nan"),
                "latency_p95_ms": float("nan"),
                "tracked_fraction": 0.0,
                "samples": 0,
                "occlusion_events": 0,
                "completed_recaptures": 0,
                "censored_recaptures": 0,
            }
        values["success"] = float(bool(values["success"]) and valid)
        values.update(
            {
                "sample_coverage": coverage,
                "expected_samples": expected_samples,
                "valid_episode": valid,
                "duration_s": duration,
                "measured_duration_s": measured_duration,
                "elapsed_s": elapsed,
                "latency_basis": (
                    "simulation-clock age of latest sensor at command receipt; diagnostic only"
                ),
            }
        )
        # Inference wall time excludes sensor generation, DDS and control-timer delay.
        if self.inference_latencies:
            values["inference_wall_p50_ms"] = float(np.percentile(self.inference_latencies, 50))
            values["inference_wall_p95_ms"] = float(np.percentile(self.inference_latencies, 95))
        else:
            values["inference_wall_p50_ms"] = values["inference_wall_p95_ms"] = float("nan")
        values["inference_wall_samples"] = len(self.inference_latencies)
        values.update(self.causal_latency.result())
        if not valid:
            self.get_logger().warning(
                f"Episode {self.event.episode_id}: incomplete or insufficient "
                f"truth coverage ({coverage:.1%})"
            )
        message = TrackingMetrics()
        message.header.stamp = self.get_clock().now().to_msg()
        message.episode_id = self.event.episode_id
        message.variant = self.variant
        message.scenario = self.event.scenario
        configured_seed = self.training_seed
        message.seed = (
            configured_seed
            if configured_seed >= 0
            else int(self.parameters.get("training_seed", self.event.seed))
        )
        message.success = bool(values["success"])
        for key in (
            "mean_distance_m",
            "mean_bearing_error_deg",
            "reacquisition_time_s",
            "smoothness_rad_s",
            "collision_rate",
            "latency_p50_ms",
            "latency_p95_ms",
        ):
            setattr(message, key, float(values[key]))
        self.publisher.publish(message)
        row = {
            "episode_id": message.episode_id,
            "variant": message.variant,
            "scenario": message.scenario,
            "seed": message.seed,
            "environment_seed": self.event.seed,
            "model_card_sha256": self.model_card_sha256,
            "reason": reason,
            **values,
        }
        clean = {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in row.items()
        }
        with (self.output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(clean, ensure_ascii=False, allow_nan=False) + "\n")
        path = self.output / "episodes.csv"
        exists = path.exists() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(clean))
            if not exists:
                writer.writeheader()
            writer.writerow(clean)
        self.get_logger().info(
            f"Episode {message.episode_id}: success={message.success}, "
            f"distance={message.mean_distance_m:.3f}m"
        )
        self.accumulator = self.event = None

    def destroy_node(self):
        if self.accumulator is not None:
            self.finish("evaluator_shutdown")
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EvaluatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
