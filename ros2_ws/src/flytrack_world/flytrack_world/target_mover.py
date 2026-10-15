"""Animate an analytic target through the bridged Gazebo SetEntityPose service."""

import math

from flytrack_core.world import TargetTrajectory
from flytrack_interfaces.msg import ScenarioEvent
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose

from .common import EVENT_QOS, WorldNode, now_seconds, param, spin_main


def set_pose_request(entity_name: str, pose, z: float):
    request = SetEntityPose.Request()
    request.entity.name, request.entity.type = entity_name, Entity.MODEL
    request.pose.position.x, request.pose.position.y, request.pose.position.z = (
        float(pose[0]),
        float(pose[1]),
        float(z),
    )
    request.pose.orientation.z, request.pose.orientation.w = (
        math.sin(float(pose[2]) / 2),
        math.cos(float(pose[2]) / 2),
    )
    return request


class TargetMoverNode(WorldNode):
    def __init__(self):
        super().__init__("target_mover")
        world = param(self, "world_name", "tracking")
        self.entity_name = param(self, "entity_name", "target")
        self.target_z_m = float(param(self, "target_z_m", 0.25))
        self.motion = param(self, "target_motion", "curved")
        self.rate_hz = float(param(self, "rate_hz", 50.0))
        self.service_timeout_s = float(param(self, "service_timeout_s", 1.0))
        self.client = self.create_client(SetEntityPose, f"/world/{world}/set_pose")
        self.trajectory = TargetTrajectory({"motion": self.motion}, self.base_seed)
        self.pending = None
        self.create_subscription(ScenarioEvent, "/world/scenario", self._scenario, EVENT_QOS)
        self.create_timer(1 / self.rate_hz, self._tick)

    def _scenario(self, message):
        new, values = self.consume_event(message)
        if not self.active and self.pending is not None:
            self.pending.cancel()
            self.pending = None
        if new:
            self.trajectory = TargetTrajectory(
                {
                    "motion": values.get("target_motion", self.motion),
                    **values.get("trajectory_config", {}),
                },
                int(message.seed),
            )

    def _complete(self, future):
        if future is not self.pending:
            return
        self.pending = None
        try:
            if not future.result().success:
                self.get_logger().error("Gazebo rejected target SetEntityPose")
        except Exception as error:
            self.get_logger().error(f"target SetEntityPose failed: {error}")

    def _tick(self):
        if not self.active:
            return
        now = now_seconds(self)
        if self.pending is not None:
            if now - self._pending_started_s > self.service_timeout_s:
                expired = self.pending
                self.pending = None
                expired.cancel()
                self.get_logger().error(
                    "target SetEntityPose timed out; retrying current trajectory pose"
                )
            return
        if not self.client.service_is_ready():
            return
        relative = now - self.started_at_s
        if relative < 0:
            self.active = False
            return
        pose, _ = self.trajectory.state(relative)
        self._pending_started_s = now
        self.pending = self.client.call_async(
            set_pose_request(self.entity_name, pose, self.target_z_m)
        )
        self.pending.add_done_callback(self._complete)


def main(args=None):
    spin_main(TargetMoverNode, args)
