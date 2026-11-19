"""The Gazebo wind world (#140) against the 2D wind world's model; no ROS needed."""

import json
import math
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flytrack_v2")

from flytrack_core.contracts import SensorContract  # noqa: E402
from flytrack_core.world import PlumeConfig, PuffPlume  # noqa: E402
from flytrack_v2.contracts import SensorContractV2  # noqa: E402
from flytrack_v2.world import wind as wind2d  # noqa: E402
from flytrack_world.contract import has_wind, load_required_contract  # noqa: E402
from flytrack_world.wind_field import (  # noqa: E402
    WindPlume,
    antenna,
    plume_config,
    wind_dropped,
    wind_rng,
)


def test_antenna_matches_the_2d_formula():
    rng = np.random.default_rng(0)
    winds = rng.normal(0, 0.5, (64, 2))
    headings = rng.uniform(-math.pi, math.pi, 64)
    expected = wind2d.antenna(torch.as_tensor(winds), torch.as_tensor(headings), 45.0, 0.5).numpy()
    ours = np.stack([antenna(w, h, 45.0, 0.5) for w, h in zip(winds, headings, strict=True)])
    np.testing.assert_allclose(ours, expected, atol=1e-12)


def test_antenna_signs_follow_the_contract():
    # Wind from straight ahead (blowing toward -x at heading 0): both antennae equal.
    ahead = antenna(np.array([-0.5, 0.0]), 0.0, 45.0, 0.5)
    np.testing.assert_allclose(ahead, [math.cos(math.radians(45))] * 2, atol=1e-12)
    # Wind from the left (coming from +y): left > right, and upwind_angle recovers +90 deg.
    left = antenna(np.array([0.0, -0.5]), 0.0, 45.0, 0.5)
    assert left[0] > left[1]
    theta = wind2d.upwind_angle(torch.as_tensor(left[None]), 45.0)
    assert float(theta) == pytest.approx(math.pi / 2)


def test_plume_settings_are_the_2d_wind_worlds():
    w = wind2d.WindConfig()
    one, two = plume_config(w, 1), plume_config(w, 2)
    assert (one.wind_x, one.wind_y) == (0.0, 0.0)
    assert one.emission_hz == w.emission_hz and one.lifetime_s == w.lifetime_s
    assert one.decay_tau_s == 1e30
    # v1compat.py: min(sources * (lifetime * emission + 3), capacity)
    assert one.max_puffs == 203 and two.max_puffs == w.capacity


def test_without_meander_it_is_v1s_puff_plume_in_a_fixed_wind():
    w = replace(wind2d.WindConfig(), direction_std_deg=0.0)
    ours = WindPlume(w, 1, seed=7)
    fixed = PuffPlume(
        PlumeConfig(
            **{**plume_config(w, 1).__dict__, "wind_x": ours.wind[0], "wind_y": ours.wind[1]}
        ),
        seed=7,
    )
    for time_s in np.arange(0.02, 12.0, 0.02):
        source = [[math.cos(time_s), math.sin(time_s)]]
        ours.advance(time_s, source)
        fixed.advance(time_s, source)
    np.testing.assert_array_equal(ours.centers, fixed.centers)
    np.testing.assert_array_equal(ours.ages, fixed.ages)
    points = np.array([[0.2, 0.1], [-0.3, 0.4]])
    np.testing.assert_array_equal(ours.concentration(points), fixed.concentration(points))


def test_wind_meanders_with_the_configured_speed_and_spread():
    w = wind2d.WindConfig()
    offsets = []
    for seed in range(40):
        plume = WindPlume(w, 1, seed=seed)
        for time_s in np.arange(0.5, 60.0, 0.5):
            plume.advance(time_s, [[0.0, 0.0]])
            assert np.hypot(*plume.wind) == pytest.approx(w.speed_m_s)
            offsets.append(plume.eta)
    # Stationary OU from the first tick: std = direction_std_deg (loose: 40 x 60 s, tau 10 s).
    assert math.degrees(np.std(offsets)) == pytest.approx(w.direction_std_deg, rel=0.2)


def test_seeds_reproduce_and_differ():
    w = wind2d.WindConfig()
    a, b, c = WindPlume(w, 1, 3), WindPlume(w, 1, 3), WindPlume(w, 1, 4)
    for plume in (a, b, c):
        plume.advance(5.0, [[0.0, 0.0]])
    np.testing.assert_array_equal(a.centers, b.centers)
    assert a.direction == b.direction != c.direction
    np.testing.assert_array_equal(a.antenna_reading(0.3, 45, 0.5), b.antenna_reading(0.3, 45, 0.5))


def test_antenna_noise_is_the_configured_std():
    w = wind2d.WindConfig()
    plume = WindPlume(w, 1, seed=1)
    clean = antenna(plume.wind, 0.0, 45, 0.5)
    readings = np.stack([plume.antenna_reading(0.0, 45, 0.5) for _ in range(4000)])
    assert np.std(readings - clean) == pytest.approx(w.antenna_noise_std, rel=0.05)


def test_v2_contract_is_loaded_as_v2_and_needs_every_wind_field(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(SensorContractV2().to_dict()), encoding="utf-8")
    loaded = load_required_contract(str(path))
    assert isinstance(loaded, SensorContractV2) and has_wind(loaded)
    assert loaded.sha256() == SensorContractV2().sha256()

    v1 = load_required_contract_from(tmp_path, SensorContract().to_dict())
    assert type(v1) is SensorContract and not has_wind(v1)

    partial = {**SensorContract().to_dict(), "wind_hz": 50.0}
    with pytest.raises(ValueError, match="missing wind fields"):
        load_required_contract_from(tmp_path, partial)


def load_required_contract_from(directory, values):
    path = directory / "other.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return load_required_contract(str(path))


def test_wind_dropout_uses_its_own_stream_and_probability():
    assert not wind_dropped(wind_rng(0), 0.0)
    rng = wind_rng(5)
    rate = np.mean([wind_dropped(rng, 0.25) for _ in range(8000)])
    assert rate == pytest.approx(0.25, abs=0.02)
    # Independent of SensorCorruptor's eye/odor streams (children 0-2 of the same seed).
    children = np.random.SeedSequence(5).spawn(3)
    assert all(np.random.default_rng(child).random() != wind_rng(5).random() for child in children)
