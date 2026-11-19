"""Paired fixed-trajectory comparisons for the #161 D3 reward Decision."""

import importlib.util
from pathlib import Path

import pytest
import torch
from flytrack_v2.world.v1compat import track_v2_reward

ROOT = Path(__file__).resolve().parents[3]
ROWS = 4
_spec = importlib.util.spec_from_file_location(
    "track_v2_trajectories", ROOT / "scripts/v2/track_v2_trajectories.py"
)
trajectories = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trajectories)


@pytest.fixture(scope="module", params=trajectories.MOTIONS)
def measurements(request):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = trajectories.measure_trajectories(
            request.param,
            rows=ROWS,
            episode_seconds=60,
            trajectories=("A", "B2", "C10", "C10stop"),
        )
    finally:
        torch.set_num_threads(threads)
    return report["trajectories"]


def _assert_order(cells, better, worse, metrics):
    seeds = range(trajectories.SEED_START, trajectories.SEED_START + ROWS)
    higher = cells[better]["episodes"]
    lower = cells[worse]["episodes"]
    violations = []
    for seed, left, right in zip(seeds, higher, lower, strict=True):
        assert left["seed"] == right["seed"] == seed
        for metric in metrics:
            if not left[metric] > right[metric]:
                violations.append((seed, metric, left[metric], right[metric]))
    assert not violations, f"expected {better} > {worse}; {violations}"


@pytest.mark.slow
def test_stopping_beats_wall_collision_on_every_row(measurements):
    for stopped, crashed in zip(
        measurements["C10stop"]["episodes"], measurements["C10"]["episodes"], strict=True
    ):
        assert crashed["collision"]
        assert not stopped["collision"]
        assert stopped["steps"] == 1200
        assert stopped["final_wall_distance_m"] == pytest.approx(0.5, abs=1e-4)
        assert stopped["final_speed_m_s"] == 0.0
    _assert_order(measurements, "C10stop", "C10", trajectories.RETURN_FIELDS)


@pytest.mark.slow
def test_tracking_beats_constant_lag_on_every_row(measurements):
    _assert_order(measurements, "A", "B2", ("return",))


def test_non_collision_reward_is_bounded_by_smoothness_penalty():
    distances = torch.tensor([0.2, 1.0, 2.0, 4.0, 8.0])
    tracked = torch.tensor([True, True, False, False, False])
    collision = torch.zeros_like(tracked)
    change = torch.tensor([[0.0, 0.5], [0.2, 0.0], [0.0, 0.0], [0.0, 0.0], [0.3, 0.4]])
    lower_bound = -0.05 * change.to(torch.float64).norm(dim=-1)

    reward = track_v2_reward(tracked, distances, collision, change, following=1.0)

    assert (reward >= lower_bound).all()
    torch.testing.assert_close(reward[-2:], lower_bound[-2:])


def test_collision_keeps_the_original_terms():
    tracked = torch.tensor([True, False])
    distances = torch.tensor([0.5, 5.0], dtype=torch.float64)
    collision = torch.ones_like(tracked)
    change = torch.tensor([[0.3, 0.4], [0.0, 0.0]], dtype=torch.float64)

    reward = track_v2_reward(tracked, distances, collision, change, following=1.0)

    expected = torch.tensor([1.0 - 0.15 - 10.0 - 0.025, -0.9 - 10.0], dtype=torch.float64)
    torch.testing.assert_close(reward, expected)


def test_tracked_at_following_distance_earns_1_9_minus_smoothness():
    tracked = torch.tensor([True, True])
    distances = torch.ones(2, dtype=torch.float64)
    collision = torch.zeros_like(tracked)
    change = torch.tensor([[0.0, 0.0], [0.3, 0.4]], dtype=torch.float64)

    reward = track_v2_reward(tracked, distances, collision, change, following=1.0)

    torch.testing.assert_close(reward, torch.tensor([1.9, 1.9 - 0.025], dtype=torch.float64))
