"""Episode reset coordinator with asynchronous Gazebo/policy service gates.

Inactive events close the sensor forwarding gate. Policy reset first clears
cached commands; repeated zero commands then settle the robot before teleport.
A second reset after pose changes establishes the fresh-sensor barrier.
"""

import json
import math

import numpy as np
from flytrack_core.world import ScenarioSpec, TargetTrajectory
from flytrack_interfaces.msg import ScenarioEvent
from geometry_msgs.msg import PoseStamped, Twist
from ros_gz_interfaces.srv import SetEntityPose
from std_srvs.srv import Trigger

from .common import EVENT_QOS, POSE_QOS, WorldNode, now_seconds, param, pose_xy_yaw, spin_main
from .target_mover import set_pose_request


class ScenarioManagerNode(WorldNode):
    def __init__(self):
        super().__init__("scenario_manager")
        self.world_name = param(self, "world_name", "tracking")
        self.training_seed = int(param(self, "training_seed", self.base_seed))
        self.agent_name = param(self, "agent_name", "agent")
        self.target_name = param(self, "target_name", "target")
        self.agent_z_m = float(param(self, "agent_z_m", 0.12))
        self.target_z_m = float(param(self, "target_z_m", 0.25))
        self.target_motion = param(self, "target_motion", "curved")
        self.target_speed_m_s = float(param(self, "target_speed_m_s", 0.2))
        self.plume_model = param(self, "plume_model", "puff")
        self.arena_size_m = float(param(self, "arena_size_m", 10.0))
        self.arena_center_x = float(param(self, "arena_center_x", 0.0))
        self.arena_center_y = float(param(self, "arena_center_y", 0.0))
        self.agent_radius_m = float(param(self, "agent_radius_m", 0.18))
        self.target_radius_m = float(param(self, "target_radius_m", 0.15))
        self.initial_agent_pose = json.loads(param(self, "initial_agent_pose_json", "null"))
        self.obstacles = json.loads(param(self, "obstacles_json", "[]"))
        self.trajectory_config = json.loads(param(self, "trajectory_config_json", "{}"))
        self.plume_config = json.loads(param(self, "plume_config_json", "{}"))
        spec = {
            **self.scenario_spec.to_dict(),
            "duration_s": self.episode_seconds,
            "occlusion_start_s": float(
                param(self, "occlusion_start_s", self.scenario_spec.occlusion_start_s)
            ),
            "occlusion_duration_s": float(
                param(self, "occlusion_duration_s", self.scenario_spec.occlusion_duration_s)
            ),
        }
        spec.update(json.loads(param(self, "scenario_parameters_json", "{}")))
        self.scenario_spec = ScenarioSpec.from_dict(spec, name=self.scenario_name)
        self.max_episodes = int(
            param(self, "max_episodes", 0)
        )  # 0 runs until the evaluator stops it.
        self.inter_episode_s = float(param(self, "inter_episode_s", 0.3))
        minimum_settle = (
            max(
                self.contract.v_max / self.contract.linear_accel_max,
                self.contract.omega_max / self.contract.angular_accel_max,
            )
            + 0.1
        )
        self.settle_s = max(minimum_settle, float(param(self, "settle_s", minimum_settle)))
        self.service_timeout_s = float(param(self, "service_timeout_s", 5.0))
        self.reset_endpoint = param(self, "policy_reset_service", "/connectome_policy_node/reset")
        self.pose_client = self.create_client(SetEntityPose, f"/world/{self.world_name}/set_pose")
        self.reset_client = self.create_client(Trigger, self.reset_endpoint)
        self.publisher = self.create_publisher(ScenarioEvent, "/world/scenario", EVENT_QOS)
        self.stop_publisher = self.create_publisher(Twist, "/cmd_vel", 1)
        self.create_subscription(PoseStamped, "/world/agent/pose", self._agent_pose, POSE_QOS)
        self.create_subscription(PoseStamped, "/world/target/pose", self._target_pose, POSE_QOS)
        self._agent = self._target = None
        self._agent_stamp = self._target_stamp = -np.inf
        self._episode_index = 0
        self._episode_seed = self.base_seed
        self._pending = None
        self._phase = "waiting_services"
        self._last_clock_s = now_seconds(self)
        self._next_reset_s = self._last_clock_s + self.inter_episode_s
        self._last_occluded = False
        self._collision = False
        self._event_parameters = {}
        self._prepare_poses()
        self._publish_event(False, "resetting")
        self.create_timer(self.contract.control_dt, self._tick)

    def _agent_pose(self, message):
        self._agent, self._agent_stamp = (
            np.asarray(pose_xy_yaw(message)),
            message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
        )

    def _target_pose(self, message):
        self._target, self._target_stamp = (
            np.asarray(pose_xy_yaw(message)),
            message.header.stamp.sec + message.header.stamp.nanosec * 1e-9,
        )

    def _prepare_poses(self):
        self._episode_seed = self.base_seed + self._episode_index
        half = self.arena_size_m / 2 - 0.5
        config = {
            "motion": self.target_motion,
            "speed_m_s": self.target_speed_m_s,
            "center_x": self.arena_center_x,
            "center_y": self.arena_center_y,
            "initial_x": self.arena_center_x,
            "initial_y": self.arena_center_y,
            "xmin": self.arena_center_x - half,
            "xmax": self.arena_center_x + half,
            "ymin": self.arena_center_y - half,
            "ymax": self.arena_center_y + half,
            **self.trajectory_config,
        }
        self._trajectory_config = config
        target, _ = TargetTrajectory(config, self._episode_seed).state(0)
        if self.initial_agent_pose is None:
            yaw = target[2]
            agent = np.asarray(
                [target[0] - 1.5 * math.cos(yaw), target[1] - 1.5 * math.sin(yaw), yaw]
            )
        else:
            agent = np.asarray(self.initial_agent_pose, dtype=np.float64)
        if agent.shape != (3,) or not np.all(np.isfinite(agent)):
            raise ValueError("initial_agent_pose_json must be finite [x,y,yaw] or null")
        self._initial_agent, self._initial_target = agent, target

    def _publish_event(self, active: bool, phase: str):
        now = now_seconds(self)
        values = {
            **self.scenario_spec.to_dict(),
            "phase": phase,
            "episode_start_s": self.started_at_s,
            "occluded": active and self.scenario_spec.occluded_at(now - self.started_at_s),
            "collision": self._collision,
            "target_motion": self.target_motion,
            "training_seed": self.training_seed,
            "environment_seed": self._episode_seed,
            "plume_model": self.plume_model,
            "plume_config": self.plume_config,
            "trajectory_config": self._trajectory_config,
            "arena_size_m": self.arena_size_m,
            "arena_center_x": self.arena_center_x,
            "arena_center_y": self.arena_center_y,
            "agent_radius_m": self.agent_radius_m,
            "target_radius_m": self.target_radius_m,
            "obstacles": self.obstacles,
            "initial_agent_pose": self._initial_agent.tolist(),
            "initial_target_pose": self._initial_target.tolist(),
        }
        message = ScenarioEvent()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.world_frame
        message.episode_id, message.seed = self._episode_index, self._episode_seed
        message.scenario, message.active = self.scenario_name, active
        message.parameters_json = json.dumps(values, sort_keys=True, allow_nan=False)
        self.publisher.publish(message)

    def _submit(self, client, request, phase: str, next_operation):
        self._phase = phase
        self._pending_started_s = now_seconds(self)
        self._pending = client.call_async(request)
        expected = self._pending

        def finished(future):
            if future is not self._pending:
                return
            self._pending = None
            try:
                response = future.result()
                if response is None or not response.success:
                    detail = getattr(response, "message", "request was rejected")
                    raise RuntimeError(detail)
                next_operation()
            except Exception as error:
                self.get_logger().error(f"{phase}: {error}; keeping the episode inactive")
                self._phase = "waiting_services"
                self._next_reset_s = now_seconds(self) + self.inter_episode_s

        expected.add_done_callback(finished)

    def _reset_target(self):
        self._submit(
            self.pose_client,
            set_pose_request(self.target_name, self._initial_target, self.target_z_m),
            "resetting_target",
            self._reset_policy,
        )

    def _reset_policy(self):
        self._submit(self.reset_client, Trigger.Request(), "resetting_policy", self._start_episode)

    def _begin_settle(self):
        self._phase = "settling"
        self._settle_until_s = now_seconds(self) + self.settle_s

    def _teleport_agent(self):
        self._submit(
            self.pose_client,
            set_pose_request(self.agent_name, self._initial_agent, self.agent_z_m),
            "resetting_agent",
            self._reset_target,
        )

    def _start_episode(self):
        self._phase = "active"
        self.active = True
        self.started_at_s = now_seconds(self)
        self._agent = self._target = None
        self._agent_stamp = self._target_stamp = -np.inf
        self._last_occluded = self.scenario_spec.occluded_at(0)
        self._collision = False
        self._publish_event(True, "occlusion" if self._last_occluded else "tracking")

    def _has_collision(self) -> bool:
        if self._agent is None:
            return False
        lower = np.asarray([self.arena_center_x, self.arena_center_y]) - self.arena_size_m / 2
        upper = lower + self.arena_size_m
        if np.any(self._agent[:2] < lower + self.agent_radius_m) or np.any(
            self._agent[:2] > upper - self.agent_radius_m
        ):
            return True
        if (
            self._target is not None
            and np.linalg.norm(self._agent[:2] - self._target[:2])
            <= self.agent_radius_m + self.target_radius_m
        ):
            return True
        for obstacle in self.obstacles:
            if "radius" in obstacle:
                if (
                    np.linalg.norm(self._agent[:2] - [obstacle["x"], obstacle["y"]])
                    <= self.agent_radius_m + obstacle["radius"]
                ):
                    return True
            else:
                closest = np.clip(
                    self._agent[:2],
                    [obstacle["xmin"], obstacle["ymin"]],
                    [obstacle["xmax"], obstacle["ymax"]],
                )
                if np.linalg.norm(self._agent[:2] - closest) <= self.agent_radius_m:
                    return True
        return False

    def _end_episode(self, now: float):
        self.active = False
        self._publish_event(False, "completed")
        self.stop_publisher.publish(Twist())
        self._episode_index += 1
        self._phase = (
            "done"
            if self.max_episodes and self._episode_index >= self.max_episodes
            else "waiting_services"
        )
        self._next_reset_s = now + self.inter_episode_s
        if self._phase != "done":
            self._prepare_poses()

    def _tick(self):
        now = now_seconds(self)
        if now < self._last_clock_s - 1e-6:
            self.active = False
            if self._pending is not None:
                self._pending.cancel()
                self._pending = None
            self._episode_index += 1
            self._prepare_poses()
            self._phase = "waiting_services"
            self._next_reset_s = now + self.inter_episode_s
            self._publish_event(False, "clock_reset")
        self._last_clock_s = now
        if self._phase != "active":
            self.stop_publisher.publish(Twist())
            if self._pending is not None:
                if now - self._pending_started_s > self.service_timeout_s:
                    self._pending.cancel()
                    self._pending = None
                    self._phase = "waiting_services"
                    self._next_reset_s = now + self.inter_episode_s
                    self.get_logger().error("episode reset service timed out; sensors remain gated")
                return
            if (
                self._phase == "settling"
                and now >= self._settle_until_s
                and self.pose_client.service_is_ready()
            ):
                self._teleport_agent()
                return
            if (
                self._phase == "waiting_services"
                and now >= self._next_reset_s
                and self.pose_client.service_is_ready()
                and self.reset_client.service_is_ready()
            ):
                self._publish_event(False, "resetting")
                self._submit(
                    self.reset_client, Trigger.Request(), "clearing_policy", self._begin_settle
                )
            return
        relative = now - self.started_at_s
        occluded = self.scenario_spec.occluded_at(relative)
        if occluded != self._last_occluded:
            self._last_occluded = occluded
            self._publish_event(True, "occlusion" if occluded else "tracking")
        if (
            self._agent_stamp >= self.started_at_s
            and now - self._agent_stamp <= self.contract.watchdog_s
            and self._target_stamp >= self.started_at_s
            and now - self._target_stamp <= self.contract.watchdog_s
        ):
            self._collision |= self._has_collision()
        if relative >= self.episode_seconds or self._collision:
            self._end_episode(now)


def main(args=None):
    spin_main(ScenarioManagerNode, args)
