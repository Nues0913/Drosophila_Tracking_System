"""#35 S3: teacher_accel_clip=False labels the desired command; execution is still acceleration-limited."""

import numpy as np
import pytest
from flytrack_core.contracts import ActionLimits, SensorContract
from flytrack_sim import FlyTrackEnv
from flytrack_sim.teacher import teacher_action

OPTS = dict(scenario="normal", episode_seconds=1.0, sensing=False)


def test_default_labels_are_unchanged():
    contract = SensorContract(eye_radius=1)
    plain = FlyTrackEnv(contract, **OPTS)
    explicit = FlyTrackEnv(contract, **OPTS, teacher_accel_clip=True)
    (_, a), (_, b) = plain.reset(seed=3), explicit.reset(seed=3)
    for _ in range(10):
        assert a["teacher_action"].tobytes() == b["teacher_action"].tobytes()
        command = np.array([0.1, 0.0], np.float32)
        _, _, _, _, a = plain.step(command)
        _, _, _, _, b = explicit.step(command)


def test_desired_label_skips_only_the_acceleration_clip():
    contract = SensorContract()
    limits = ActionLimits.from_contract(contract)
    agent, target, previous = np.array([0.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0]), np.zeros(2, np.float32)
    clipped = teacher_action(agent, target, previous, contract)
    desired = teacher_action(agent, target, previous, contract, accel_clip=False)
    # 0.7 * (3 - 1) = 1.4 m/s is bounded to v_max; the formal label can only move 0.05 m/s from 0.
    assert desired[0] == pytest.approx(limits.v_max)
    assert clipped[0] == pytest.approx(limits.linear_accel_max * contract.control_dt)


def test_env_still_limits_the_executed_command():
    contract = SensorContract(eye_radius=1)
    env = FlyTrackEnv(contract, **OPTS, teacher_accel_clip=False)
    _, info = env.reset(seed=5)
    previous = env.previous_action.copy()
    env.step(np.asarray(info["teacher_action"], np.float32))
    delta = np.abs(env.previous_action - previous)
    assert delta[0] <= contract.linear_accel_max * contract.control_dt + 1e-6
    assert delta[1] <= contract.angular_accel_max * contract.control_dt + 1e-6


def test_option_must_be_bool():
    with pytest.raises(ValueError, match="bool"):
        FlyTrackEnv(teacher_accel_clip="no")
