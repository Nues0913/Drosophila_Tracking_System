"""Sensor buffers, episode state, watchdog, and acceleration limits without ROS."""

from __future__ import annotations

import math
from collections import deque
from time import monotonic_ns, perf_counter

import numpy as np
import torch


def state_is_finite(state) -> bool:
    """True when every tensor in a (nested tuple) policy state is finite; v2 states are tuples."""
    if isinstance(state, torch.Tensor):
        return bool(torch.isfinite(state).all())
    if isinstance(state, (tuple, list)):
        return all(state_is_finite(item) for item in state)
    return True


WIND = 3  # sensor index of the antennae: after the two eyes (0, 1) and odor (2)


class PolicyRuntime:
    """Buffers the sensors, arms once every one is fresh, and steps the policy.

    ``wind=True`` (a v2 policy in the wind world, #140) adds the antenna reading as a fourth
    sensor: it must also be fresh to arm, and the observation carries ``wind`` and a 4-channel
    ``valid``. Without it there are three sensors, as in v1 and the v2 v1compat world, and a v2
    policy sees wind 0 with valid 0 (V2PolicyAdapter).
    """

    def __init__(
        self,
        policy,
        contract,
        device: str = "cpu",
        profiler_enabled: bool = False,
        wind: bool = False,
    ):
        self.policy, self.contract, self.device = policy, contract, torch.device(device)
        self.profiler_enabled = bool(profiler_enabled)
        self.wind_enabled = bool(wind)
        self.sensor_count = 4 if self.wind_enabled else 3
        self.history = deque(maxlen=contract.history_steps)
        self.reset()

    def reset(self, minimum_stamp: float | None = None) -> None:
        self.state = self.policy.initial_state(1, self.device)
        self.vision = np.zeros((2, self.contract.n_ommatidia), dtype=np.float32)
        self.odor = np.zeros(2, dtype=np.float32)
        self.wind = np.zeros(2, dtype=np.float32)
        self.stamps = np.full(self.sensor_count, np.nan, dtype=np.float64)
        self.sensor_available = np.zeros(self.sensor_count, dtype=bool)
        self.armed = False
        self.minimum_stamp = minimum_stamp
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_tick = None
        self.last_inference_latency_ms = None
        self.last_profile_spans_ns = {}
        self.history.clear()

    def update_vision(self, eye: int, values: np.ndarray, stamp: float) -> bool:
        values = np.asarray(values, dtype=np.float32)
        if (
            eye not in (0, 1)
            or not math.isfinite(stamp)
            or (self.minimum_stamp is not None and stamp <= self.minimum_stamp)
        ):
            return False
        if np.isfinite(self.stamps[eye]) and stamp < self.stamps[eye]:
            return False
        if values.shape != (self.contract.n_ommatidia,) or not np.isfinite(values).all():
            self.mark_missing(eye, stamp)
            return False
        self.vision[eye] = values
        self.stamps[eye] = stamp
        self.sensor_available[eye] = True
        return True

    def mark_missing(self, sensor: int, stamp: float) -> None:
        if (
            sensor not in range(self.sensor_count)
            or not math.isfinite(stamp)
            or (self.minimum_stamp is not None and stamp <= self.minimum_stamp)
        ):
            return
        if np.isfinite(self.stamps[sensor]) and stamp < self.stamps[sensor]:
            return
        self.stamps[sensor] = stamp
        self.sensor_available[sensor] = False

    def update_odor(self, values: np.ndarray, stamp: float, valid: bool = True) -> bool:
        values = np.asarray(values, dtype=np.float32)
        if self.minimum_stamp is not None and stamp <= self.minimum_stamp:
            return False
        if (
            values.shape != (2,)
            or not math.isfinite(stamp)
            or not np.isfinite(values).all()
            or not valid
        ):
            # Dropout must immediately mark missing, even if a previous packet was fresh.
            self.mark_missing(2, stamp)
            return False
        if np.isfinite(self.stamps[2]) and stamp < self.stamps[2]:
            return False
        self.odor = np.maximum(values, 0)
        self.stamps[2] = stamp
        self.sensor_available[2] = True
        return True

    def update_wind(self, values: np.ndarray, stamp: float, valid: bool = True) -> bool:
        """The antenna reading; signed, so unlike odor it is not clipped at 0."""
        if not self.wind_enabled:
            raise ValueError("this runtime has no wind sensor (wind=False)")
        values = np.asarray(values, dtype=np.float32)
        if self.minimum_stamp is not None and stamp <= self.minimum_stamp:
            return False
        if (
            values.shape != (2,)
            or not math.isfinite(stamp)
            or not np.isfinite(values).all()
            or not valid
        ):
            self.mark_missing(WIND, stamp)
            return False
        if np.isfinite(self.stamps[WIND]) and stamp < self.stamps[WIND]:
            return False
        self.wind = values
        self.stamps[WIND] = stamp
        self.sensor_available[WIND] = True
        return True

    def tick(self, now: float) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        self.last_inference_latency_ms = None
        self.last_profile_spans_ns = {}
        if not math.isfinite(now):
            raise ValueError("Runtime clock must be finite")
        if self.previous_tick is not None and now < self.previous_tick:
            self.reset(minimum_stamp=now)
        age = now - self.stamps
        # A delayed packet from the previous clock epoch must not block fresh packets.
        future = np.isfinite(age) & (age < -1e-6)
        self.stamps[future] = np.nan
        self.sensor_available[future] = False
        valid = (
            self.sensor_available
            & np.isfinite(age)
            & (age >= -1e-6)
            & (age <= self.contract.watchdog_s)
        )
        self.previous_tick = now
        if not self.armed and valid.all():
            self.armed = True
        if not valid.any() or not self.armed:
            self.previous_action[:] = 0
            return self.previous_action.copy(), None, valid
        input_started_ns = monotonic_ns() if self.profiler_enabled else 0
        observation = {
            "vision": self.vision.copy(),
            "odor": self.odor.copy(),
            "valid": valid.astype(np.float32),
        }
        if self.wind_enabled:
            # The wind valid flag gates it as in the 2D world (wind * valid[:, 3]).
            observation["wind"] = self.wind * np.float32(valid[WIND])
        self.history.append(observation)
        inference_started = perf_counter()
        tensors = {
            key: torch.as_tensor(value, device=self.device).unsqueeze(0)
            for key, value in observation.items()
        }
        if self.profiler_enabled:
            self.last_profile_spans_ns["tick_input_conversion"] = [
                input_started_ns,
                monotonic_ns(),
            ]
        if self.profiler_enabled and self.device.type == "cuda":
            sync_started_ns = monotonic_ns()
            torch.cuda.synchronize(self.device)
            self.last_profile_spans_ns["device_sync_before_forward"] = [
                sync_started_ns,
                monotonic_ns(),
            ]
        model_started_ns = monotonic_ns() if self.profiler_enabled else 0
        with torch.inference_mode():
            action, next_state, activity = self.policy(tensors, self.state)
        if self.profiler_enabled and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self.profiler_enabled:
            self.last_profile_spans_ns["model_forward_and_completion"] = [
                model_started_ns,
                monotonic_ns(),
            ]
        finite_started_ns = monotonic_ns() if self.profiler_enabled else 0
        if (
            not torch.isfinite(action).all()
            or not state_is_finite(next_state)
            or not torch.isfinite(activity).all()
        ):
            self.reset(minimum_stamp=now)
            return self.previous_action.copy(), None, valid
        if self.profiler_enabled:
            self.last_profile_spans_ns["finite_check"] = [finite_started_ns, monotonic_ns()]
        self.state = next_state
        transfer_started_ns = monotonic_ns() if self.profiler_enabled else 0
        desired = action[0].detach().cpu().numpy()
        activity_array = activity[0].detach().cpu().numpy()
        if self.profiler_enabled:
            self.last_profile_spans_ns["device_to_host"] = [transfer_started_ns, monotonic_ns()]
        self.last_inference_latency_ms = (perf_counter() - inference_started) * 1000
        limit_started_ns = monotonic_ns() if self.profiler_enabled else 0
        desired = np.clip(
            desired, [0, -self.contract.omega_max], [self.contract.v_max, self.contract.omega_max]
        )
        # The model advances exactly one fixed control interval, regardless of real-time factor.
        limits = (
            np.array([self.contract.linear_accel_max, self.contract.angular_accel_max])
            * self.contract.control_dt
        )
        self.previous_action = (
            self.previous_action + np.clip(desired - self.previous_action, -limits, limits)
        ).astype(np.float32)
        if self.profiler_enabled:
            self.last_profile_spans_ns["action_limit"] = [limit_started_ns, monotonic_ns()]
        return self.previous_action.copy(), activity_array, valid
