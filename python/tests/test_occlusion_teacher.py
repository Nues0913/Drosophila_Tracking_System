"""Occluded steering labels must use exactly the student's sensory input."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from flytrack_core.contracts import SensorContract
from flytrack_sim import FlyTrackEnv
from flytrack_sim.teacher import teacher_action


def observation(odor, valid=1):
    return {"odor": np.asarray(odor, np.float32), "valid": np.asarray([1, 1, valid], np.float32)}


def action(odor, **overrides):
    kwargs = dict(
        agent_pose=np.array([0, 0, 0]),
        target_pose=np.array([0, 2, 0]),
        previous_action=np.zeros(2, np.float32),
        contract=SensorContract(),
        occluded=True,
        observation=observation(odor),
        occlusion_odor_gain=8.0,
    )
    kwargs.update(overrides)
    return teacher_action(**kwargs)


@pytest.mark.parametrize("odor,sign", [([2, 1], 1), ([1, 2], -1), ([1, 1], 0), ([0, 0], 0)])
def test_odor_turn_sign_and_same_forward_rule(odor, sign):
    turn = action(odor)
    assert np.sign(turn[1]) == sign
    assert turn[0] == action(odor, occlusion_odor_gain=None)[0]


def test_normalized_ratio_and_action_limits():
    contract = SensorContract(angular_accel_max=100)
    # 8 * (0.101 - 0.099) / 0.2 = 0.08, below the velocity clamp.
    result = action([0.101, 0.099], contract=contract)
    assert result[1] == pytest.approx(0.08, abs=1e-6)
    assert action([1, 0], contract=contract)[1] == contract.omega_max
    # Keep the existing acceleration limit, even during an odor sign reversal.
    previous = np.array([0, -0.6], np.float32)
    limited = action([1, 0], previous_action=previous)
    assert limited[1] == pytest.approx(-0.45)


@pytest.mark.parametrize("scale", [1e-30, 1, 1e30])
def test_ratio_does_not_depend_on_absolute_concentration(scale):
    np.testing.assert_array_equal(action(np.array([2, 1]) * scale), action([2, 1]))


@pytest.mark.parametrize(
    "odor,valid", [([0, 0], 1), ([1, 2], 0), ([np.nan, 2], 1), ([1, np.inf], 1)]
)
def test_missing_or_invalid_odor_steers_neutrally(odor, valid):
    assert action(odor, observation=observation(odor, valid))[1] == 0


def test_steering_does_not_follow_hidden_target_when_odor_is_fixed():
    first = action([1, 2], target_pose=np.array([0, 2, 0]))
    second = action([1, 2], target_pose=np.array([0, -2, 0]))
    assert first[1] == second[1] < 0


def test_disabled_and_nonoccluded_paths_are_bitwise_identical():
    baseline = action([1, 2], occlusion_odor_gain=None)
    inactive = action([1, 2], occluded=False)
    np.testing.assert_array_equal(baseline, inactive)
    assert baseline.tobytes() == inactive.tobytes()


def test_active_odor_mode_needs_student_observation():
    with pytest.raises(ValueError, match="student's observation"):
        action([1, 2], observation=None)


def test_env_passes_same_observation_object_and_never_resamples_plume(monkeypatch):
    import flytrack_sim.env as env_module

    env = FlyTrackEnv(
        SensorContract(eye_radius=1),
        scenario="occlusion",
        episode_seconds=0.2,
        scenario_config={"occlusion_start_s": 0, "occlusion_duration_s": 0.2},
        occlusion_odor_steering={"gain": 8},
    )
    passed = []

    def capture(*args, **kwargs):
        passed.append(kwargs["observation"])
        return teacher_action(*args, **kwargs)

    monkeypatch.setattr(env_module, "teacher_action", capture)
    obs, _ = env.reset(seed=21)
    assert passed[-1] is obs
    obs, _, _, _, info = env.step([0, 0])
    assert passed[-1] is obs
    # Override the received, corrupted sample, independently of the plume truth.
    env._odor[:] = [0, 1]
    env._sensor_stamps[2] = env.time_s
    obs = env._observation()
    monkeypatch.setattr(env.plume, "concentration", lambda *args: pytest.fail("truth resampled"))
    info = env._info(obs)
    assert passed[-1] is obs
    expected = teacher_action(
        env.agent_pose,
        env.target_pose,
        env.previous_action,
        env.contract,
        env.following_distance_m,
        target_velocity=env.target_velocity,
        occluded=True,
        observation=obs,
        occlusion_odor_gain=8,
    )
    np.testing.assert_array_equal(info["teacher_action"], expected)
    env.close()


def test_environment_only_changes_teacher_inside_occlusion_window():
    opts = dict(
        scenario="occlusion",
        episode_seconds=0.3,
        scenario_config={"occlusion_start_s": 0.1, "occlusion_duration_s": 0.1},
        initial_agent_pose=[0, 0, 0],
        target_motion="stationary",
        plume_model="gaussian",
        domain_randomization=False,
    )
    disabled = FlyTrackEnv(SensorContract(eye_radius=1), **opts)
    enabled = FlyTrackEnv(SensorContract(eye_radius=1), **opts, occlusion_odor_steering={"gain": 8})
    off, on = disabled.reset(seed=21), enabled.reset(seed=21)
    assert disabled.occlusion_odor_steering is None
    for _ in range(7):
        for key in off[0]:
            np.testing.assert_array_equal(off[0][key], on[0][key])
        if not enabled.scenario_spec.occluded_at(enabled.time_s):
            assert off[1]["teacher_action"].tobytes() == on[1]["teacher_action"].tobytes()
        if not enabled._running:
            break
        a = disabled.step([0, 0])
        b = enabled.step([0, 0])
        assert a[1:4] == b[1:4]  # reward and terminal flags are unchanged.
        off, on = (a[0], a[4]), (b[0], b[4])
    disabled.close()
    enabled.close()


@pytest.mark.parametrize("option", [True, False, "auto", 8, {"gian": 8}])
def test_teacher_option_rejects_wrong_shape_or_unknown_keys(option):
    with pytest.raises(ValueError, match="mapping with gain"):
        FlyTrackEnv(occlusion_odor_steering=option)


@pytest.mark.parametrize("gain", [-1, np.nan, np.inf, None, True, "8", {}, [], [8]])
def test_teacher_option_rejects_invalid_gain(gain):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        FlyTrackEnv(occlusion_odor_steering={"gain": gain})


def test_calibration_cli_selects_occlusion_and_reports_collision(tmp_path):
    root = Path(__file__).resolve().parents[2]
    contract_path = tmp_path / "sensor.json"
    contract_path.write_text(json.dumps(SensorContract(eye_radius=1).to_dict()))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "sensor_contract_path": str(contract_path),
                "env": {
                    "episode_seconds": 0.1,
                    "target_motion": "stationary",
                    "plume_model": "gaussian",
                    "scenario_config": {"occlusion_start_s": 0, "occlusion_duration_s": 0.05},
                    "occlusion_odor_steering": {"gain": 8},
                },
            }
        )
    )
    output = tmp_path / "calibration.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/calibrate_teacher.py"),
            "--config",
            str(config),
            "--scenario",
            "occlusion",
            "--episodes",
            "1",
            "--output",
            str(output),
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "python")},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["scenario"] == "occlusion" and report["complete"]
    assert report["mean_collision_rate"] == report["rows"][0]["collision_rate"]
