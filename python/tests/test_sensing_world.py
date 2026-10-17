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


def test_layout_is_varied_by_default_seeded_and_legacy_reproducible():
    pytest.importorskip("gymnasium")
    from flytrack_sim import FlyTrackEnv

    clockwise = TargetTrajectory({"direction": -1}, seed=3)
    counter = TargetTrajectory({"direction": 1}, seed=3)
    pose, _ = clockwise.state(0.0)
    np.testing.assert_allclose(pose[:2], counter.state(0.0)[0][:2])
    turn = lambda trajectory: np.angle(  # noqa: E731
        np.exp(1j * (trajectory.state(1.0)[0][2] - trajectory.state(0.0)[0][2]))
    )
    assert turn(clockwise) < 0 < turn(counter)
    with pytest.raises(ValueError, match="direction"):
        TargetTrajectory({"direction": 0.5})

    def start(**options):
        env = FlyTrackEnv(SensorContract(eye_radius=2), episode_seconds=0.1, **options)
        poses = []
        for seed in range(16):
            env.reset(seed=seed)
            config = env.trajectory.config
            poses.append((config.direction, config.radius_m, *env.target_pose, *env.agent_pose))
        env.close()
        return np.array(poses)

    default = start()
    # Both circling directions, several radii and offset starts: no constant turn fits.
    assert set(default[:, 0]) == {-1.0, 1.0}
    assert np.ptp(default[:, 1]) > 1.0 and (1.5 <= default[:, 1]).all()
    assert (default[:, 1] <= 3.5).all()
    np.testing.assert_array_equal(default, start())
    legacy = start(layout_preset="legacy")
    assert (legacy[:, 0] == 1).all() and (legacy[:, 1] == 2.0).all()
    # Legacy: agent exactly 1.5 m behind the target, aligned with its heading.
    target, agent = legacy[:, 2:5], legacy[:, 5:8]
    behind = target[:, :2] - 1.5 * np.stack((np.cos(target[:, 2]), np.sin(target[:, 2])), axis=1)
    np.testing.assert_allclose(agent[:, :2], behind, atol=1e-12)
    np.testing.assert_allclose(agent[:, 2], target[:, 2], atol=1e-12)
    # The 2026-10-17 scenario-test world (direction + offsets, fixed radius) is unchanged.
    scenario_test = start(
        randomize_target_direction=True,
        start_lateral_offset_m=0.5,
        start_heading_offset_deg=30,
        randomize_target_layout=False,
    )
    assert set(scenario_test[:, 0]) == {-1.0, 1.0} and (scenario_test[:, 1] == 2.0).all()
    with pytest.raises(ValueError, match="layout"):
        FlyTrackEnv(SensorContract(eye_radius=2), start_heading_offset_deg=270)
    with pytest.raises(ValueError, match="layout_preset"):
        FlyTrackEnv(SensorContract(eye_radius=2), layout_preset="fixed")


def test_wander_target_is_smooth_bounded_and_seed_reproducible():
    from flytrack_core.world.trajectories import WANDER_DT_S

    config = {"motion": "wander", "initial_x": 3.5, "initial_y": -3.0, "heading_rad": 0.3}
    trajectory = TargetTrajectory(config, seed=11)
    times = np.arange(0, 600, 0.05)
    poses = np.array([trajectory.state(time)[0] for time in times])
    assert (np.abs(poses[:, :2]) <= 4.5 + 1e-9).all()
    # Constant speed and a bounded, continuous turn rate (no instantaneous reversal).
    step = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    np.testing.assert_allclose(step, 0.2 * 0.05, rtol=0.05)
    turn = np.abs(np.angle(np.exp(1j * np.diff(poses[:, 2])))) / 0.05
    assert turn.max() < 0.4 + 1.0 * np.pi + 1e-6
    assert np.ptp(poses[:, 0]) > 4 and np.ptp(poses[:, 1]) > 4
    # Pure function of (config, seed, time): query order and a fresh object agree.
    replay = TargetTrajectory(config, seed=11)
    np.testing.assert_allclose(replay.state(300.0)[0], trajectory.state(300.0)[0])
    np.testing.assert_allclose(replay.state(1.234)[0], trajectory.state(1.234)[0])
    other = TargetTrajectory(config, seed=12)
    assert np.linalg.norm(other.state(60.0)[0][:2] - trajectory.state(60.0)[0][:2]) > 0.1
    assert trajectory.state(WANDER_DT_S / 2)[0][0] == pytest.approx(
        3.5 + 0.2 * np.cos(0.3) * WANDER_DT_S / 2, abs=1e-3
    )


def test_explicit_agent_pose_skips_the_unused_sampled_start():
    pytest.importorskip("gymnasium")
    from flytrack_sim import FlyTrackEnv

    # A stationary target facing -x at x=2.4 in a 6 m arena puts the sampled
    # "behind the target" start outside the walls; an explicit start must still work.
    options = dict(
        arena_size_m=6,
        target_motion="stationary",
        trajectory_config={"initial_x": 2.4, "initial_y": 0.0, "heading_rad": np.pi},
        layout_preset="legacy",
        episode_seconds=1,
    )
    with pytest.raises(ValueError, match="outside the arena"):
        FlyTrackEnv(SensorContract(eye_radius=2), **options).reset(seed=0)
    env = FlyTrackEnv(SensorContract(eye_radius=2), initial_agent_pose=[0, 0, 0], **options)
    env.reset(seed=0)
    np.testing.assert_array_equal(env.agent_pose, [0, 0, 0])
    env.reset(seed=1, options={"agent_pose": [0.5, -1.0, 0.25]})
    np.testing.assert_array_equal(env.agent_pose, [0.5, -1.0, 0.25])
    np.testing.assert_allclose(env.target_pose[:2], [2.4, 0.0])
    assert abs(np.cos(env.target_pose[2]) + 1) < 1e-12  # heading pi (wrapped to +-pi)
    # Layout draws do not depend on the explicit start (direction stays seeded).
    sampled = FlyTrackEnv(SensorContract(eye_radius=2), episode_seconds=0.1)
    pinned = FlyTrackEnv(SensorContract(eye_radius=2), episode_seconds=0.1)
    for seed in range(8):
        sampled.reset(seed=seed)
        pinned.reset(seed=seed, options={"agent_pose": [0, 0, 0]})
        assert sampled.trajectory.config == pinned.trajectory.config


def test_layout_respects_explicit_trajectory_options_and_motion_mixtures():
    from flytrack_core.world import LayoutConfig, sample_episode_layout

    walls = np.asarray([-5.0, -5.0, 5.0, 5.0])
    defaults = {"xmin": -4.5, "ymin": -4.5, "xmax": 4.5, "ymax": 4.5}
    motions, agents = set(), []
    for seed in range(200):
        config, agent = sample_episode_layout(
            seed, ["curved", "wander"], defaults, {"radius_m": 2.0}, LayoutConfig(), walls, 0.18
        )
        motions.add(config["motion"])
        agents.append(agent)
        if config["motion"] == "curved":
            assert config["radius_m"] == 2.0  # explicit options are never resampled
    assert motions == {"curved", "wander"}
    assert (np.abs(np.asarray(agents)[:, :2]) < 5 - 0.18).all()
    with pytest.raises(ValueError, match="target_motion"):
        sample_episode_layout(0, [], defaults, {}, LayoutConfig(), walls, 0.18)


def test_success_requires_facing_the_target():
    from flytrack_core.metrics import EpisodeMetrics

    sideways = EpisodeMetrics()
    legacy = EpisodeMetrics(bearing_threshold_deg=None)
    for metrics in (sideways, legacy):
        for _ in range(100):
            metrics.append(1.2, np.deg2rad(94), 0.1, False)
    # The constant left turn of the old world: within 2 m but side-on to the target.
    assert sideways.result()["success"] == 0 and sideways.result()["within_distance_fraction"] == 1
    assert legacy.result()["success"] == 1
