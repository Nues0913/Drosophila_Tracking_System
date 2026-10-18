"""Survival reward bounds, collision incentives and legacy replay."""

import math

import numpy as np
import pytest
from flytrack_core.contracts import SensorContract
from flytrack_sim import FlyTrackEnv


def make_env(**options):
    return FlyTrackEnv(SensorContract(eye_radius=1), sensing=False, **options)


@pytest.mark.parametrize("gamma", [0.99, 1.0])
@pytest.mark.parametrize("motion", ["stationary", "curved", "linear", "wander"])
def test_collision_is_worse_than_worst_survival_from_random_states(gamma, motion):
    env = make_env(target_motion=motion, episode_seconds=60)
    rng = np.random.default_rng(19)
    # Different distances, speeds and remaining horizons. Force a real wall
    # crossing at the first integration substep; future survival uses only bounds.
    for seed in range(8):
        env.reset(seed=seed)
        env.agent_pose = np.array(
            [env.walls[2] - env.agent_radius_m - 1e-6, rng.uniform(-3, 3), rng.uniform(-0.4, 0.4)]
        )
        env.previous_action = np.array([rng.uniform(0.1, 0.5), rng.uniform(-1.5, 1.5)], np.float32)
        env._previous_distance = np.linalg.norm(env.target_pose[:2] - env.agent_pose[:2])
        remaining_steps = int(rng.integers(1, env.max_steps + 1))
        _, reward, terminated, _, _ = env.step([env.limits.v_max, 0])
        assert terminated
        # Triangle inequality bounds positive progress on the collision step,
        # even if distance, lost and smoothness contribute no penalty at all.
        collision_upper = env.reward_penalty_bounds["progress"] - env.reward_weights["collision"]
        assert reward <= collision_upper
        worst_surviving_step = env.survival_bonus - env.max_step_penalty
        survival_lower = math.fsum(worst_surviving_step * gamma**t for t in range(remaining_steps))
        assert reward < survival_lower
        assert collision_upper < survival_lower
    env.close()


@pytest.mark.parametrize("motion,plume", [("stationary", "gaussian"), ("curved", "puff")])
def test_random_surviving_steps_are_nonnegative_and_episode_returns_accumulate(motion, plume):
    env = make_env(target_motion=motion, plume_model=plume, episode_seconds=60)
    rng = np.random.default_rng(731)
    checked = 0
    for seed in (19, 20):
        env.reset(seed=seed)
        total = 0.0
        while True:
            _, reward, terminated, truncated, info = env.step(rng.uniform([0, -1.5], [0.5, 1.5]))
            total += reward
            if not terminated:
                assert reward >= 0
                checked += 1
            if terminated or truncated:
                assert info["episode"]["r"] == total
                assert info["episode_metrics"]["episode_return"] == total
                assert info["episode"]["l"] == env.step_count
                assert info["episode_metrics"]["collision"] == terminated
                break
    assert checked > 100
    env.close()


def test_bound_includes_watchdog_stop_that_bypasses_acceleration_limits():
    env = make_env(
        target_motion="stationary",
        reward_weights={"progress": 0, "distance": 0, "lost": 0, "smoothness": 1},
    )
    env.reset(seed=0)
    env.previous_action = np.array([env.limits.v_max, env.limits.omega_max], np.float32)
    env._sensor_stamps[:] = -np.inf
    _, reward, terminated, _, info = env.step(env.previous_action)
    assert not terminated
    np.testing.assert_array_equal(info["applied_action"], [0, 0])
    assert reward >= 0
    assert env.max_step_penalty >= env.limits.v_max**2 + env.limits.omega_max**2
    env.close()


def test_progress_bound_covers_speed_limit_rounding_in_float32():
    env = FlyTrackEnv(
        SensorContract(eye_radius=1, v_max=0.8),
        sensing=False,
        target_motion="stationary",
        initial_agent_pose=[2, 0, 0],
        layout_preset="legacy",
        reward_weights={"distance": 0, "lost": 0, "smoothness": 0},
    )
    env.reset(seed=0)
    env.previous_action = np.array([0.8, 0], np.float32)
    _, reward, terminated, _, _ = env.step([0.8, 0])
    assert not terminated
    assert reward >= 0
    env.close()


@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("motion", ["stationary", "curved"])
def test_legacy_reward_is_bitwise_identical_to_original_expression(collision, motion):
    env = make_env(
        target_motion=motion,
        episode_seconds=2,
        distance_penalty_cap_m=None,
        reward_weights={"survival": 0},
        initial_agent_pose=[4.8, 0, 0] if collision else [-4, -4, 0],
    )
    env.reset(seed=21)
    rng = np.random.default_rng(21)
    total = 0.0
    while True:
        previous_distance = env._previous_distance
        previous_action = env.previous_action.copy()
        action = [env.limits.v_max, 0] if collision else rng.uniform([0, -1.5], [0.5, 1.5])
        _, reward, terminated, truncated, info = env.step(action)
        distance = info["target_distance_m"]
        change = info["applied_action"] - previous_action
        w = env.reward_weights
        original = (
            w["progress"] * (previous_distance - distance)
            - w["distance"]
            * max(0, abs(distance - env.following_distance_m) - env.distance_tolerance_m)
            - w["smoothness"] * float(change @ change)
            - w["collision"] * env.collision
            - w["lost"] * (distance > env.lost_distance_m)
        )
        assert reward.hex() == float(original).hex()
        total += original
        if terminated or truncated:
            assert terminated == collision
            assert float(info["episode"]["r"]).hex() == float(total).hex()
            break
    env.close()


@pytest.mark.parametrize("cap,expected", [(None, None), (0, 0), (3, -3)])
def test_distance_cap_can_be_disabled_or_set_to_zero(cap, expected):
    env = make_env(
        target_motion="stationary",
        layout_preset="legacy",
        initial_agent_pose=[-4, -4, 0],
        distance_penalty_cap_m=cap,
        reward_weights={"progress": 0, "distance": 1, "smoothness": 0, "lost": 0, "survival": 0},
    )
    env.reset(seed=0)
    _, reward, terminated, _, info = env.step([0, 0])
    assert not terminated
    if expected is None:
        expected = -(info["target_distance_m"] - 1 - 0.2)
    assert reward == expected
    env.close()


@pytest.mark.parametrize("bonus", [0, 0.25, 4])
def test_explicit_survival_bonus_is_added_only_without_collision(bonus):
    weights = {"progress": 0, "distance": 0, "smoothness": 0, "lost": 0, "survival": bonus}
    for collision in (False, True):
        env = make_env(
            target_motion="stationary",
            initial_agent_pose=[4.819999 if collision else 4, 0, 0],
            reward_weights=weights,
        )
        env.reset(seed=0)
        _, reward, terminated, _, _ = env.step([env.limits.v_max, 0])
        assert terminated == collision
        assert reward == (-env.reward_weights["collision"] if collision else bonus)
        env.close()


@pytest.mark.parametrize(
    "motion", ["stationary", "linear", "curved", "wander", ["stationary", "wander"]]
)
def test_bounds_use_effective_trajectory_speed_weights_and_action_limits(motion):
    contract = SensorContract(
        eye_radius=1, v_max=0.8, omega_max=2, linear_accel_max=40, angular_accel_max=100
    )
    env = FlyTrackEnv(
        contract,
        target_motion=motion,
        target_speed_m_s=0.1,
        trajectory_config={"speed_m_s": 0.7, "motion": "stationary"},
        distance_penalty_cap_m=1.25,
        reward_weights={"distance": 0.6, "lost": 0.9, "smoothness": 0.1, "progress": 3},
    )
    bounds = env.reward_penalty_bounds
    speed = 0 if motion == "stationary" else 0.7
    assert bounds["progress"] == pytest.approx(3 * (0.8 + speed) * contract.control_dt)
    assert bounds["distance"] == 0.75
    assert bounds["lost"] == 0.9
    # Large accelerations allow a full -omega_max -> +omega_max change.
    assert bounds["smoothness"] == pytest.approx(0.1 * (0.8**2 + 4**2), rel=2e-6)
    assert env.max_step_penalty == math.fsum(bounds.values())
    assert env.survival_bonus == env.max_step_penalty
    env.close()


def test_uncapped_auto_bound_uses_arena_and_explicit_trajectory_extent():
    small = make_env(distance_penalty_cap_m=None)
    large = make_env(
        arena_size_m=20,
        distance_penalty_cap_m=None,
        trajectory_config={"xmin": -15, "xmax": 15, "ymin": -20, "ymax": 20},
    )
    expected_distance_bound = 0.3 * (np.hypot(25, 30) - 1 - 0.2)
    assert large.reward_penalty_bounds["distance"] == pytest.approx(expected_distance_bound)
    assert large.survival_bonus > small.survival_bonus
    small.close()
    large.close()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("option", ["distance_penalty_cap_m", "survival"])
def test_new_reward_options_reject_negative_and_nonfinite_values(option, value):
    options = {option: value} if option != "survival" else {"reward_weights": {option: value}}
    with pytest.raises(ValueError, match="finite and nonnegative"):
        make_env(**options)


@pytest.mark.parametrize("value", [None, "automatic", "1.5", {}, complex(1, 2)])
def test_survival_rejects_invalid_non_numeric_values(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        make_env(reward_weights={"survival": value})


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_bound_rejects_invalid_effective_target_speed(value):
    with pytest.raises(ValueError, match="target speed"):
        make_env(trajectory_config={"speed_m_s": value})
