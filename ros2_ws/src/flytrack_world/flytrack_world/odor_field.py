"""Truth-aware world node that publishes only bilateral local concentration."""

import os
import time

import numpy as np
from flytrack_core.sensing import odor_sensor_positions
from flytrack_core.world import make_plume
from flytrack_interfaces.msg import OdorReading, ScenarioEvent
from geometry_msgs.msg import PoseStamped

from .common import (
    EVENT_QOS,
    POSE_QOS,
    SENSOR_QOS,
    WorldNode,
    now_seconds,
    param,
    pose_xy_yaw,
    spin_main,
    stamp_seconds,
)


class OdorFieldNode(WorldNode):
    def __init__(self):
        super().__init__("odor_field")
        self.plume_model = param(self, "plume_model", "puff")
        self.plume_config = {
            "wind_x": float(param(self, "wind_x", 0.15)),
            "wind_y": float(param(self, "wind_y", 0.0)),
        }
        self.sensor_forward_m = float(param(self, "sensor_forward_m", 0.15))
        self.agent_pose = self.target_pose = None
        self.agent_stamp = self.target_stamp = -np.inf
        self.plume = make_plume(self.plume_model, self.plume_config, self.base_seed)
        self.publisher = self.create_publisher(OdorReading, "/sim/odor", SENSOR_QOS)
        self.create_subscription(PoseStamped, "/world/agent/pose", self._agent, POSE_QOS)
        self.create_subscription(PoseStamped, "/world/target/pose", self._target, POSE_QOS)
        self.create_subscription(ScenarioEvent, "/world/scenario", self._scenario, EVENT_QOS)
        self.create_timer(1 / self.contract.odor_hz, self._tick)

    def _agent(self, message):
        self.agent_pose, self.agent_stamp = (
            np.asarray(pose_xy_yaw(message)),
            stamp_seconds(message.header.stamp),
        )

    def _target(self, message):
        self.target_pose, self.target_stamp = (
            np.asarray(pose_xy_yaw(message)),
            stamp_seconds(message.header.stamp),
        )

    def _scenario(self, message):
        new, values = self.consume_event(message)
        if new:
            self.plume = make_plume(
                values.get("plume_model", self.plume_model),
                {**self.plume_config, **values.get("plume_config", {})},
                int(message.seed),
            )
            # Reject pre-reset truth even if the bridge last published it recently.
            self.agent_pose = self.target_pose = None
            self.agent_stamp = self.target_stamp = -np.inf

    def _tick(self):
        if not self.active or self.agent_pose is None or self.target_pose is None:
            return
        now = now_seconds(self)
        relative = now - self.started_at_s
        if relative < 0:
            self.active = False
            return
        if min(self.agent_stamp, self.target_stamp) <= self.started_at_s or any(
            now - stamp > self.contract.watchdog_s or stamp > now + 1e-6
            for stamp in (self.agent_stamp, self.target_stamp)
        ):
            return
        sources = [self.target_pose[:2]]
        if self.scenario_spec.name == "conflict":
            sources.append([self.scenario_spec.decoy_x, self.scenario_spec.decoy_y])
        self.plume.advance(relative, np.asarray(sources))
        reading = self.plume.concentration(
            odor_sensor_positions(
                self.agent_pose, self.contract.odor_baseline_m, self.sensor_forward_m
            )
        )
        message = OdorReading()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.world_frame
        message.left, message.right, message.valid = float(reading[0]), float(reading[1]), True
        message.clock_domain = os.environ.get("FLYTRACK_CLOCK_DOMAIN", "")
        message.source_monotonic_ns = time.monotonic_ns()
        self.publisher.publish(message)


def main(args=None):
    spin_main(OdorFieldNode, args)
