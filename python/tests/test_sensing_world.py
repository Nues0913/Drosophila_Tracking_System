"""Behavior checks authored for later validation; this implementation turn did not run them."""

import numpy as np
import pytest
from flytrack_core.contracts import SensorContract
from flytrack_core.sensing import (
    CorruptionConfig,
    HexEye,
    ORNAdapter,
    SensorCorruptor,
    odor_sensor_positions,
)
from flytrack_core.world import GaussianPlume, PuffPlume, ScenarioSpec, TargetTrajectory


def test_hex_projector_preserves_constant_brightness_for_rgb_and_gray():
    contract = SensorContract(eye_radius=2, camera_width=32, camera_height=24)
    eye = HexEye(contract)
    assert eye.directions_rad.shape == (2, 19, 2)
    gray = np.full((24, 32), 128, np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    np.testing.assert_allclose(eye.sample_image(gray), 128 / 255, atol=1e-6)
    np.testing.assert_allclose(eye.sample_image(rgb), eye.sample_image(gray), atol=1e-6)
    assert eye.sha256() == HexEye(contract).sha256()
    assert eye.sha256() != HexEye(SensorContract(eye_radius=2, eye_yaw_deg=40)).sha256()


def test_bilateral_sensor_geometry_tracks_agent_heading():
    sensors = odor_sensor_positions(np.asarray([1.0, 2.0, np.pi / 2]), 0.12, 0.15)
    np.testing.assert_allclose(sensors, [[0.94, 2.15], [1.06, 2.15]], atol=1e-8)


def test_orn_old_threshold_then_adaptation_and_missing_freezes_state():
    orn = ORNAdapter(num_types=2, K=[0.3, 0.6])
    response = orn.update(np.asarray([0.3, 0.3]), 0.1)
    np.testing.assert_allclose(response[:, 0], 0.5)
    np.testing.assert_allclose(response[:, 1], 0.2)
    np.testing.assert_allclose(orn.K[:, 1], 0.585)
    previous = orn.K.copy()
    missing = orn.update(np.asarray([100.0, np.nan]), 0.1, valid=False)
    assert not missing.any()
    np.testing.assert_array_equal(orn.K, previous)
    orn.reset((3,))
    assert orn.update(np.full((3, 2), 0.3), 0.05).shape == (3, 2, 2)


def test_dropout_differs_from_valid_blackout_and_is_seed_reproducible():
    a, b = SensorCorruptor(4), SensorCorruptor(4)
    image = np.ones(10, np.float32)
    config = CorruptionConfig(image_noise_std=0.1, brightness_std=0.2)
    np.testing.assert_array_equal(a.image(image, config)[0], b.image(image, config)[0])
    black, black_valid = a.image(image, CorruptionConfig(blackout=True))
    dropped, dropped_valid = a.image(image, CorruptionConfig(image_dropout=1))
    assert black_valid and not dropped_valid
    assert not black.any() and not dropped.any()


def test_puff_sampling_never_changes_world_rng_and_emission_tracks_moving_source():
    first, second = PuffPlume(seed=11), PuffPlume(seed=11)

    def trajectory(t):
        return np.asarray([[t * 0.2, 0.0]])

    points = np.asarray([[0.0, 0.0], [0.5, 0.0]])
    for time in (0.1, 0.2, 0.5, 1.0):
        first.advance(time, trajectory)
        first.concentration(points)
        first.concentration(points)
        second.advance(time, trajectory)
    np.testing.assert_array_equal(first.centers, second.centers)
    np.testing.assert_array_equal(first.concentration(points), second.concentration(points))
    assert len(first.centers) > 0
    assert np.ptp(first.centers[:, 0]) > 0
    with pytest.raises(ValueError, match="backwards"):
        first.advance(0.5, trajectory)


def test_gaussian_curriculum_has_positive_near_source_and_bounded_trajectory():
    plume = GaussianPlume()
    plume.advance(0, [[0.0, 0.0]])
    concentration = plume.concentration(np.asarray([[0.0, 0.0], [0.0, 4.0]]))
    assert concentration[0] > concentration[1] >= 0
    trajectory = TargetTrajectory({"motion": "linear", "speed_m_s": 0.5}, seed=8)
    for time in (0, 10, 30, 100):
        pose, velocity = trajectory.state(time)
        assert np.all(pose[:2] >= -4.5) and np.all(pose[:2] <= 4.5)
        assert np.linalg.norm(velocity) <= 0.5 + 1e-9


def test_scenario_window_and_unknown_config_are_explicit():
    spec = ScenarioSpec(name="occlusion", occlusion_start_s=1.0, occlusion_duration_s=2.0)
    assert not spec.occluded_at(0.99) and spec.occluded_at(1.0)
    assert spec.occluded_at(2.99) and not spec.occluded_at(3.0)
    with pytest.raises(ValueError, match="unknown scenario"):
        ScenarioSpec.from_dict({"odor_dropot": 0.5})


def test_env_sensor_only_action_limits_and_terminal_protocol():
    pytest.importorskip("gymnasium")
    from flytrack_sim import FlyTrackEnv

    env = FlyTrackEnv(
        SensorContract(eye_radius=2),
        target_motion="stationary",
        plume_model="gaussian",
        episode_seconds=0.1,
        domain_randomization=False,
    )
    observation, info = env.reset(seed=21)
    assert set(observation) == {"vision", "odor", "valid"}
    assert observation["vision"].shape == (2, 19)
    assert info["privileged_state"].shape == (12,)
    observation, _, terminated, truncated, info = env.step([100, 100])
    assert not terminated and not truncated
    np.testing.assert_allclose(info["applied_action"], [0.05, 0.15], atol=1e-7)
    _, _, _, truncated, info = env.step([0, 0])
    assert truncated and "episode_metrics" in info
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0, 0])
    env.close()


def test_vector_same_step_preserves_terminal_observation_and_info():
    pytest.importorskip("gymnasium")
    from flytrack_sim import make_vector_env

    env = make_vector_env(
        2,
        SensorContract(eye_radius=2),
        seed=2,
        target_motion="stationary",
        plume_model="gaussian",
        episode_seconds=0.05,
    )
    env.reset(seed=2)
    _, _, _, truncated, infos = env.step(np.zeros((2, 2), np.float32))
    assert truncated.all()
    assert "final_obs" in infos and "final_info" in infos
    assert infos["_final_obs"].all() and infos["_final_info"].all()
    env.close()
