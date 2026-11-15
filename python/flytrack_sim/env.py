"""Planar closed-loop sensory tracking with asynchronous camera/odor clocks.

Actor observations contain only sensory measurements. Privileged state, teacher
labels and ground-truth metrics live exclusively in ``info`` for offline use.
"""

import math
from functools import partial

import gymnasium as gym
import numpy as np
from flytrack_core.contracts import ActionLimits, SensorContract
from flytrack_core.sensing import HexEye, SensorCorruptor, odor_sensor_positions
from flytrack_core.world import (
    LayoutConfig,
    ScenarioSpec,
    TargetTrajectory,
    TrajectoryConfig,
    make_plume,
    motion_choices,
    sample_episode_layout,
    wrap_angle,
)

from .teacher import teacher_action

PRIVILEGED_STATE_FIELDS = (
    "agent_x",
    "agent_y",
    "agent_sin_yaw",
    "agent_cos_yaw",
    "target_x",
    "target_y",
    "target_vx",
    "target_vy",
    "previous_v",
    "previous_omega",
    "wind_x",
    "wind_y",
)


class FlyTrackEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, contract: SensorContract | None = None, scenario: str = "normal", **kwargs):
        super().__init__()
        if isinstance(contract, dict):
            contract = SensorContract.from_dict(contract)
        self.contract = contract or SensorContract()
        options = dict(kwargs.pop("config", {}) or {})
        options.update(kwargs)
        self.render_mode = options.pop("render_mode", None)
        if self.render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or rgb_array")
        self.arena_size_m = float(options.pop("arena_size_m", 10.0))
        self.arena_center = np.asarray(options.pop("arena_center_xy", [0.0, 0.0]), dtype=np.float64)
        if self.arena_center.shape != (2,) or not np.all(np.isfinite(self.arena_center)):
            raise ValueError("arena_center_xy must be finite [x,y]")
        half_size = self.arena_size_m / 2
        self.walls = np.asarray(
            [*(self.arena_center - half_size), *(self.arena_center + half_size)]
        )
        self.agent_radius_m = float(options.pop("agent_radius_m", 0.18))
        self.target_radius_m = float(options.pop("target_radius_m", 0.15))
        self.following_distance_m = float(options.pop("following_distance_m", 1.0))
        self.distance_tolerance_m = float(options.pop("distance_tolerance_m", 0.2))
        distance_cap = options.pop("distance_penalty_cap_m", 3.0)
        self.distance_penalty_cap_m = None if distance_cap is None else float(distance_cap)
        if self.distance_penalty_cap_m is not None and (
            not np.isfinite(self.distance_penalty_cap_m) or self.distance_penalty_cap_m < 0
        ):
            raise ValueError("distance_penalty_cap_m must be finite and nonnegative or null")
        self.success_distance_m = float(options.pop("success_distance_m", 2.0))
        # A tracked step must also face the target: distance alone let a blind
        # circling policy count as "tracking" with ~94 deg bearing error.
        bearing_limit = options.pop("success_bearing_deg", 45.0)
        self.success_bearing_deg = None if bearing_limit is None else float(bearing_limit)
        self.success_fraction = float(options.pop("success_fraction", 0.8))
        self.lost_distance_m = float(options.pop("lost_distance_m", 3.0))
        self.episode_seconds = float(options.pop("episode_seconds", 30.0))
        self.max_steps = int(np.ceil(self.episode_seconds / self.contract.control_dt))
        self.target_motion = options.pop("target_motion", "curved")
        self.motion_choices = motion_choices(self.target_motion)
        self.target_speed_m_s = float(options.pop("target_speed_m_s", 0.20))
        self.plume_model = options.pop("plume_model", "puff")
        self.plume_config = dict(options.pop("plume_config", {}) or {})
        self.trajectory_config = dict(options.pop("trajectory_config", {}) or {})
        self.scenario_config = dict(options.pop("scenario_config", {}) or {})
        odor_steering = options.pop("occlusion_odor_steering", None)
        self.occlusion_odor_steering = None
        if odor_steering is not None:
            if not isinstance(odor_steering, dict) or set(odor_steering) - {"gain"}:
                raise ValueError("occlusion_odor_steering must be null or a mapping with gain")
            gain = odor_steering.get("gain", 8.0)
            try:
                valid_gain = (
                    not isinstance(gain, (str, bool))
                    and np.ndim(gain) == 0
                    and np.isfinite(gain)
                    and gain >= 0
                )
            except TypeError:
                valid_gain = False
            if not valid_gain:
                raise ValueError("occlusion odor gain must be finite and nonnegative")
            self.occlusion_odor_steering = {"gain": float(gain)}
        # #35 M1 (exploration): while occluded the teacher label repeats the last unoccluded label
        # (re-clipped to the acceleration limit), so it uses nothing the student could not have:
        # no privileged pose and no distance rule. Off by default.
        hold = options.pop("occlusion_hold_teacher", False)
        if not isinstance(hold, bool):
            raise ValueError("occlusion_hold_teacher must be a bool")
        if hold and self.occlusion_odor_steering is not None:
            raise ValueError("occlusion_hold_teacher and occlusion_odor_steering are exclusive")
        self.occlusion_hold_teacher = hold
        self._held_teacher = None
        self.initial_agent_pose = options.pop("initial_agent_pose", None)
        # Per-episode layout diversity is on by default; layout_preset: legacy (or the
        # individual LayoutConfig keys) reproduces the old fixed layout.
        layout_options = {
            name: options.pop(name)
            for name in list(options)
            if name in LayoutConfig.__dataclass_fields__
        }
        if options.get("layout_preset", "varied") not in ("varied", "legacy"):
            raise ValueError("layout_preset must be varied or legacy")
        if options.pop("layout_preset", "varied") == "legacy":
            layout_options["preset"] = "legacy"
        self.layout = LayoutConfig.from_dict(layout_options)
        # Audits of blind fixed-action policies may skip rendering and odor sampling.
        self.sensing = bool(options.pop("sensing", True))
        self.obstacles = [dict(obstacle) for obstacle in options.pop("obstacles", [])]
        self.subrays = int(options.pop("subrays", 7))
        self.camera_height_m = float(options.pop("camera_height_m", 0.27))
        self.camera_forward_m = float(options.pop("camera_forward_m", 0.14))
        self.camera_baseline_m = float(options.pop("camera_baseline_m", 0.12))
        self.target_height_m = float(options.pop("target_height_m", 0.5))
        self.domain_randomization = bool(options.pop("domain_randomization", True))
        self.default_seed = int(options.pop("seed", 0))
        self.reward_weights = {
            "progress": 2.0,
            "distance": 0.3,
            "smoothness": 0.05,
            "collision": 10.0,
            "lost": 0.5,
            "survival": "auto",
        }
        supplied_weights = options.pop("reward_weights", {}) or {}
        if set(supplied_weights) - set(self.reward_weights):
            raise ValueError("unknown reward weight name")
        self.reward_weights.update(supplied_weights)
        for name, value in self.reward_weights.items():
            if name == "survival" and isinstance(value, str) and value == "auto":
                continue
            try:
                valid = not isinstance(value, str) and np.isfinite(value) and value >= 0
            except TypeError:
                valid = False
            if not valid:
                raise ValueError("reward weights must be finite and nonnegative (survival: auto)")
        if options:
            raise ValueError(f"unknown environment options: {sorted(options)}")
        if (
            min(
                self.arena_size_m,
                self.agent_radius_m,
                self.target_radius_m,
                self.following_distance_m,
                self.episode_seconds,
            )
            <= 0
        ):
            raise ValueError("physical sizes and episode duration must be positive")
        if not 0 <= self.success_fraction <= 1:
            raise ValueError("success_fraction must be in [0,1]")
        if self.success_bearing_deg is not None and not 0 < self.success_bearing_deg <= 180:
            raise ValueError("success_bearing_deg must be in (0,180] or null")
        self.scenario_spec = ScenarioSpec.from_dict(
            {**self.scenario_config, "duration_s": self.episode_seconds}, name=scenario
        )
        self.retina = HexEye(self.contract)
        self.limits = ActionLimits.from_contract(self.contract)
        self.reward_penalty_bounds = self._step_penalty_bounds()
        self.max_step_penalty = math.fsum(self.reward_penalty_bounds.values())
        self.survival_bonus = (
            self.max_step_penalty
            if self.reward_weights["survival"] == "auto"
            else float(self.reward_weights["survival"])
        )
        self.action_space = gym.spaces.Box(
            np.asarray([0.0, -self.contract.omega_max], np.float32),
            np.asarray([self.contract.v_max, self.contract.omega_max], np.float32),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "vision": gym.spaces.Box(0.0, 1.0, (2, self.contract.n_ommatidia), np.float32),
                "odor": gym.spaces.Box(0.0, np.inf, (2,), np.float32),
                "valid": gym.spaces.Box(0.0, 1.0, (3,), np.float32),
            }
        )
        self._episode_id = -1
        self._seeded = False
        self._running = False

    def _step_penalty_bounds(self) -> dict[str, float]:
        """Bound all non-collision penalties without sampling an episode layout."""
        speed = float(self.trajectory_config.get("speed_m_s", self.target_speed_m_s))
        if not np.isfinite(speed) or speed < 0:
            raise ValueError("target speed must be finite and nonnegative")
        # Layout sampling never changes speed; explicit trajectory speed wins.
        # Reflections and wander's clipped/interpolated path cannot exceed it.
        target_speed = speed if any(m != "stationary" for m in self.motion_choices) else 0.0
        cap = self.distance_penalty_cap_m
        if cap is None:
            # All target motions stay inside their trajectory bounds. Use the
            # farthest arena/target corners when the distance penalty is uncapped.
            lower = np.asarray(
                [
                    self.trajectory_config.get(k, v + 0.5)
                    for k, v in zip(("xmin", "ymin"), self.walls[:2], strict=True)
                ]
            )
            upper = np.asarray(
                [
                    self.trajectory_config.get(k, v - 0.5)
                    for k, v in zip(("xmax", "ymax"), self.walls[2:], strict=True)
                ]
            )
            separation = np.maximum(abs(upper - self.walls[:2]), abs(self.walls[2:] - lower))
            max_distance = float(np.linalg.norm(separation))
            cap = max(
                0.0,
                max_distance - self.following_distance_m - self.distance_tolerance_m,
                self.following_distance_m - self.distance_tolerance_m,
            )
        dt = self.contract.control_dt
        limits = self.limits
        delta_v = min(limits.v_max, limits.linear_accel_max * dt)
        delta_omega = min(2 * limits.omega_max, limits.angular_accel_max * dt)
        max_delta_squared = max(
            delta_v**2 + delta_omega**2,
            # The stale-sensor watchdog bypasses acceleration limits to stop.
            limits.v_max**2 + limits.omega_max**2,
        )
        # Applied actions, their subtraction and dot product use float32;
        # rounding a speed limit can also slightly increase displacement.
        rounding_margin = 1 + 8 * float(np.finfo(np.float32).eps)
        max_delta_squared *= rounding_margin
        w = self.reward_weights
        bounds = {
            "distance": w["distance"] * cap,
            "lost": w["lost"],
            "smoothness": w["smoothness"] * max_delta_squared,
            "progress": w["progress"] * (limits.v_max + target_speed) * dt * rounding_margin,
        }
        if any(not np.isfinite(value) or value < 0 for value in bounds.values()):
            raise ValueError("reward penalty bounds must be finite and nonnegative")
        if not np.isfinite(math.fsum(bounds.values())):
            raise ValueError("total reward penalty bound must be finite")
        return bounds

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is None and not self._seeded:
            seed = self.default_seed
        super().reset(seed=seed)
        self._seeded = True
        self._episode_id += 1
        if seed is not None:
            self._next_episode_seed = int(seed)
        self._episode_seed = self._next_episode_seed
        self._next_episode_seed += 1
        initial = (options or {}).get("agent_pose", self.initial_agent_pose)
        config, initial = sample_episode_layout(
            self._episode_seed,
            self.motion_choices,
            {
                "speed_m_s": self.target_speed_m_s,
                "center_x": self.arena_center[0],
                "center_y": self.arena_center[1],
                "initial_x": self.arena_center[0],
                "initial_y": self.arena_center[1],
                "xmin": self.walls[0] + 0.5,
                "ymin": self.walls[1] + 0.5,
                "xmax": self.walls[2] - 0.5,
                "ymax": self.walls[3] - 0.5,
            },
            self.trajectory_config,
            self.layout,
            self.walls,
            self.agent_radius_m,
            agent_pose=initial,
        )
        self.episode_trajectory_config = config
        self.trajectory = TargetTrajectory(TrajectoryConfig.from_dict(config), self._episode_seed)
        self.target_pose, self.target_velocity = self.trajectory.state(0.0)
        self.agent_pose = np.asarray(initial, dtype=np.float64).copy()
        if self.agent_pose.shape != (3,) or not np.all(np.isfinite(self.agent_pose)):
            raise ValueError("initial agent pose must be finite [x,y,yaw]")
        self.previous_action = np.zeros(2, dtype=np.float32)
        self._held_teacher = None
        self.time_s = 0.0
        self.step_count = 0
        self.episode_return = 0.0
        self.collision = False
        self._running = True
        self.plume = make_plume(self.plume_model, self.plume_config, self._episode_seed)
        self.corruptor = SensorCorruptor(self._episode_seed)
        self._vision = np.zeros((2, self.contract.n_ommatidia), np.float32)
        self._odor = np.zeros(2, np.float32)
        self._sensor_stamps = np.full(3, -np.inf)
        self._next_vision_s = self._next_odor_s = 0.0
        self._previous_distance = np.linalg.norm(self.target_pose[:2] - self.agent_pose[:2])
        self._distance_sum = self._bearing_sum = self._smoothness_sum = 0.0
        self._tracked_steps = self._near_steps = 0
        self._previous_occluded = self.scenario_spec.occluded_at(0)
        self._reacquisition_start_s = None
        self._reacquisition_times: list[float] = []
        self._wall_intensity = (
            float(self.np_random.uniform(0.25, 0.65)) if self.domain_randomization else 0.45
        )
        self._target_intensity = (
            float(self.np_random.uniform(0.75, 1.0)) if self.domain_randomization else 0.9
        )
        self._target_visual_radius = self.target_radius_m * (
            float(self.np_random.uniform(0.8, 1.2)) if self.domain_randomization else 1
        )
        self.plume.advance(0, self._source_positions)
        self._acquire_vision()
        self._acquire_odor()
        observation = self._observation()
        return observation, self._info(observation)

    def _source_positions(self, time_s: float) -> np.ndarray:
        target_pose, _ = self.trajectory.state(time_s)
        positions = [target_pose[:2]]
        if self.scenario_spec.name == "conflict":
            positions.append([self.scenario_spec.decoy_x, self.scenario_spec.decoy_y])
        return np.asarray(positions)

    def _acquire_vision(self):
        if not self.sensing:
            self._sensor_stamps[:2] = self.time_s
            self._next_vision_s += 1 / self.contract.vision_hz
            return
        disks = [
            *self.obstacles,
            {
                "x": self.target_pose[0],
                "y": self.target_pose[1],
                "radius": self._target_visual_radius,
                "height": self.target_height_m,
                "intensity": self._target_intensity,
            },
        ]
        corruption = self.scenario_spec.corruption_at(self.time_s)
        for eye in (0, 1):
            raw = self.retina.raycast(
                self.agent_pose,
                self.walls,
                disks,
                eye,
                subrays=self.subrays,
                wall_intensity=self._wall_intensity,
                camera_height_m=self.camera_height_m,
                camera_forward_m=self.camera_forward_m,
                camera_baseline_m=self.camera_baseline_m,
            )
            image, valid = self.corruptor.image(raw, corruption, eye)
            if valid:
                self._vision[eye] = image
                self._sensor_stamps[eye] = self.time_s
        self._next_vision_s += 1 / self.contract.vision_hz

    def _acquire_odor(self):
        if not self.sensing:
            self._sensor_stamps[2] = self.time_s
            self._next_odor_s += 1 / self.contract.odor_hz
            return
        positions = odor_sensor_positions(self.agent_pose, self.contract.odor_baseline_m)
        odor, valid = self.corruptor.odor(
            self.plume.concentration(positions), self.scenario_spec.corruption_at(self.time_s)
        )
        if valid:
            self._odor = odor
            self._sensor_stamps[2] = self.time_s
        self._next_odor_s += 1 / self.contract.odor_hz

    def _observation(self) -> dict[str, np.ndarray]:
        valid = (self.time_s - self._sensor_stamps <= self.contract.watchdog_s).astype(np.float32)
        return {
            "vision": (self._vision * valid[:2, None]).astype(np.float32),
            "odor": (self._odor * valid[2]).astype(np.float32),
            "valid": valid,
        }

    def _move_agent(self, action: np.ndarray, dt: float):
        v, omega = action
        yaw = self.agent_pose[2]
        if abs(omega) < 1e-8:
            self.agent_pose[:2] += dt * v * np.asarray([np.cos(yaw), np.sin(yaw)])
        else:
            end_yaw = yaw + omega * dt
            self.agent_pose[0] += v / omega * (np.sin(end_yaw) - np.sin(yaw))
            self.agent_pose[1] += v / omega * (np.cos(yaw) - np.cos(end_yaw))
        self.agent_pose[2] = float(wrap_angle(yaw + omega * dt))

    def _has_collision(self) -> bool:
        xy = self.agent_pose[:2]
        if np.any(xy < self.walls[:2] + self.agent_radius_m) or np.any(
            xy > self.walls[2:] - self.agent_radius_m
        ):
            return True
        if np.linalg.norm(xy - self.target_pose[:2]) <= self.agent_radius_m + self.target_radius_m:
            return True
        return any(
            np.linalg.norm(xy - [obstacle["x"], obstacle["y"]])
            <= self.agent_radius_m + obstacle["radius"]
            for obstacle in self.obstacles
        )

    def step(self, action):
        if not self._running:
            raise RuntimeError("reset must be called before step and after an episode ends")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError("an environment action must be exactly [v,omega]")
        applied = self.limits.clip(action, self.previous_action, self.contract.control_dt)
        if not self._observation()["valid"].any():
            applied = np.zeros(2, np.float32)  # same emergency stop as deployment watchdog
        end_s = min(self.episode_seconds, (self.step_count + 1) * self.contract.control_dt)
        while self.time_s < end_s - 1e-10:
            next_s = min(
                end_s, self.time_s + self.contract.neural_dt, self._next_vision_s, self._next_odor_s
            )
            self._move_agent(applied, next_s - self.time_s)
            self.time_s = next_s
            self.target_pose, self.target_velocity = self.trajectory.state(self.time_s)
            if self.sensing:
                self.plume.advance(self.time_s, self._source_positions)
            self.collision |= self._has_collision()
            if self._next_vision_s <= self.time_s + 1e-9:
                self._acquire_vision()
            if self._next_odor_s <= self.time_s + 1e-9:
                self._acquire_odor()
            if self.collision:
                break
        displacement = self.target_pose[:2] - self.agent_pose[:2]
        distance = float(np.linalg.norm(displacement))
        bearing = float(
            wrap_angle(np.arctan2(displacement[1], displacement[0]) - self.agent_pose[2])
        )
        action_change = applied - self.previous_action
        w = self.reward_weights
        distance_penalty = max(
            0, abs(distance - self.following_distance_m) - self.distance_tolerance_m
        )
        if self.distance_penalty_cap_m is not None:
            distance_penalty = min(distance_penalty, self.distance_penalty_cap_m)
        reward = (
            w["progress"] * (self._previous_distance - distance)
            - w["distance"] * distance_penalty
            - w["smoothness"] * float(action_change @ action_change)
            - w["collision"] * self.collision
            - w["lost"] * (distance > self.lost_distance_m)
        )
        if not self.collision and self.survival_bonus:
            reward += self.survival_bonus
        self._distance_sum += distance
        self._bearing_sum += abs(bearing)
        self._smoothness_sum += abs(float(action_change[1]))
        near = distance <= self.success_distance_m
        self._near_steps += near
        self._tracked_steps += near and (
            self.success_bearing_deg is None
            or abs(math.degrees(bearing)) <= self.success_bearing_deg
        )
        self._previous_distance = distance
        self.previous_action = applied
        self.step_count += 1
        self.episode_return += reward
        occluded = self.scenario_spec.occluded_at(self.time_s)
        if self._previous_occluded and not occluded:
            self._reacquisition_start_s = (
                self.scenario_spec.occlusion_start_s + self.scenario_spec.occlusion_duration_s
            )
        if self._reacquisition_start_s is not None and abs(bearing) < np.deg2rad(15):
            self._reacquisition_times.append(max(0.0, self.time_s - self._reacquisition_start_s))
            self._reacquisition_start_s = None
        self._previous_occluded = occluded
        terminated = bool(self.collision)
        truncated = bool(
            self.time_s >= self.episode_seconds - 1e-9 or self.step_count >= self.max_steps
        )
        self._running = not (terminated or truncated)
        observation = self._observation()
        info = self._info(observation)
        if not self._running:
            info["episode_metrics"] = self.episode_metrics()
            info["episode"] = {"r": self.episode_return, "l": self.step_count, "t": self.time_s}
        return observation, float(reward), terminated, truncated, info

    def _target_visible(self, bearing: float, distance: float) -> bool:
        if self.scenario_spec.occluded_at(self.time_s):
            return False
        half_fov = math.radians(self.contract.horizontal_fov_deg / 2)
        center = math.radians(self.contract.eye_yaw_deg)
        if not (
            abs(float(wrap_angle(bearing - center))) <= half_fov
            or abs(float(wrap_angle(bearing + center))) <= half_fov
        ):
            return False
        direction = (self.target_pose[:2] - self.agent_pose[:2]) / max(distance, 1e-8)
        for obstacle in self.obstacles:
            relative = np.asarray([obstacle["x"], obstacle["y"]]) - self.agent_pose[:2]
            along = float(relative @ direction)
            if (
                0 < along < distance
                and np.linalg.norm(relative - along * direction) < obstacle["radius"]
            ):
                return False
        return True

    def _info(self, observation: dict[str, np.ndarray] | None = None) -> dict:
        displacement = self.target_pose[:2] - self.agent_pose[:2]
        distance = float(np.linalg.norm(displacement))
        bearing = float(
            wrap_angle(np.arctan2(displacement[1], displacement[0]) - self.agent_pose[2])
        )
        privileged = np.asarray(
            [
                *self.agent_pose[:2],
                np.sin(self.agent_pose[2]),
                np.cos(self.agent_pose[2]),
                *self.target_pose[:2],
                *self.target_velocity,
                *self.previous_action,
                *self.plume.wind,
            ],
            dtype=np.float32,
        )
        teacher_options = {}
        if self.occlusion_odor_steering is not None:
            teacher_options = {
                "occluded": self.scenario_spec.occluded_at(self.time_s),
                "observation": self._observation() if observation is None else observation,
                "occlusion_odor_gain": self.occlusion_odor_steering["gain"],
            }
        label = teacher_action(
            self.agent_pose,
            self.target_pose,
            self.previous_action,
            self.contract,
            self.following_distance_m,
            target_velocity=self.target_velocity,
            **teacher_options,
        )
        if self.occlusion_hold_teacher:
            if not self.scenario_spec.occluded_at(self.time_s):
                self._held_teacher = np.asarray(label, dtype=np.float32).copy()
            elif self._held_teacher is not None:
                label = ActionLimits.from_contract(self.contract).clip(
                    self._held_teacher, self.previous_action, self.contract.control_dt
                )
        return {
            "privileged_state": privileged,
            "teacher_action": label,
            "target_bearing_rad": bearing,
            "target_distance_m": distance,
            "target_visible": self._target_visible(bearing, distance),
            "occluded": self.scenario_spec.occluded_at(self.time_s),
            "collision": self.collision,
            "lost": distance > self.lost_distance_m,
            "reacquisition_time_s": self._reacquisition_times[-1]
            if self._reacquisition_times
            else float("nan"),
            "scenario": self.scenario_spec.name,
            "target_motion": self.trajectory.config.motion,
            "scenario_active": self._running,
            "episode_id": self._episode_id,
            "episode_seed": self._episode_seed,
            "time_s": self.time_s,
            "sensor_timestamps_s": self._sensor_stamps.copy(),
            "agent_pose": self.agent_pose.copy(),
            "target_pose": self.target_pose.copy(),
            "applied_action": self.previous_action.copy(),
        }

    def episode_metrics(self) -> dict:
        count = max(1, self.step_count)
        fraction = self._tracked_steps / count
        return {
            "success": bool(fraction >= self.success_fraction and not self.collision),
            "tracking_fraction": fraction,
            "within_distance_fraction": self._near_steps / count,
            "mean_distance_m": self._distance_sum / count,
            "bearing_error_deg": math.degrees(self._bearing_sum / count),
            "smoothness_rad_s": self._smoothness_sum / count,
            "collision": bool(self.collision),
            "collision_rate": float(self.collision),
            "reacquisition_time_s": float(np.mean(self._reacquisition_times))
            if self._reacquisition_times
            else float("nan"),
            "reacquisition_events": len(self._reacquisition_times),
            "reacquisition_censored": bool(self._reacquisition_start_s is not None),
            "episode_return": self.episode_return,
            "episode_length": self.step_count,
        }

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        image = np.full((512, 512, 3), 235, np.uint8)
        yy, xx = np.mgrid[:512, :512]
        for pose, radius, color in [
            (self.target_pose, self.target_radius_m, [225, 65, 50]),
            (self.agent_pose, self.agent_radius_m, [45, 90, 210]),
        ]:
            px = (pose[0] - self.walls[0]) / self.arena_size_m * 511
            py = 511 - (pose[1] - self.walls[1]) / self.arena_size_m * 511
            mask = (xx - px) ** 2 + (yy - py) ** 2 <= (radius / self.arena_size_m * 511) ** 2
            image[mask] = color
        for obstacle in self.obstacles:
            px = (obstacle["x"] - self.walls[0]) / self.arena_size_m * 511
            py = 511 - (obstacle["y"] - self.walls[1]) / self.arena_size_m * 511
            image[
                (xx - px) ** 2 + (yy - py) ** 2
                <= (obstacle["radius"] / self.arena_size_m * 511) ** 2
            ] = 90
        return image

    def close(self):
        self._running = False


def make_vector_env(
    num_envs: int,
    contract: SensorContract | None = None,
    scenario: str = "normal",
    seed: int = 0,
    **kwargs,
):
    """Gymnasium >=1.1 SAME_STEP auto-reset; final_obs/final_info retain endings.

    ``vector.reset(seed=seed)`` explicitly reproduces all environment streams.
    Initial factories also get seed+i when reset is called without a seed.
    SyncVectorEnv is CPU orchestration; it does not claim CUDA world batching.
    """
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    factories = [
        partial(FlyTrackEnv, contract, scenario, seed=seed + index, **kwargs)
        for index in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(factories, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
