"""The batched evaluation runner on a scripted world (#139)."""

import math

import pytest
import torch
from flytrack_v2.eval.runner import (
    EPISODE_KEYS,
    ObservationAblation,
    SpeedClamp,
    rotate_wind,
    run_episodes,
    swap_odor,
    to_records,
)
from flytrack_v2.interfaces import StepResult
from flytrack_v2.metrics import validate_episode_record


class _World:
    """Row i ends its first episode after seed % 4 + 3 steps; distance grows 0.1 m per step."""

    def __init__(self, batch, occluded_steps=(), bearing=None):
        self.num_envs, self.device, self.contract = batch, torch.device("cpu"), None
        self.occluded_steps, self.bearing = set(occluded_steps), bearing

    def reset(self, mask=None, episode_seeds=None):
        self.seeds = episode_seeds.clone()
        self.t = torch.zeros(self.num_envs)
        self.length = (self.seeds % 4 + 3).to(torch.float32)
        return self._obs()

    def _obs(self):
        return {"vision": torch.zeros(self.num_envs, 2, 721), "valid": torch.ones(self.num_envs, 4)}

    def step(self, action):
        self.t += 1
        ended = self.t >= self.length
        b = self.num_envs
        info = {
            "distance_m": 0.1 * self.t,
            "tracked": self.t % 2 == 0,
            "collision": torch.zeros(b, dtype=torch.bool),
            "bearing_rad": torch.zeros(b) if self.bearing is None else self.bearing(self.t),
            "occluded": torch.tensor([int(t) in self.occluded_steps for t in self.t]),
        }
        for key in EPISODE_KEYS:
            info[key] = torch.zeros(b)
        info["episode_steps"] = self.t.clone()
        info["episode_seed"] = self.seeds.to(torch.float32)
        info["episode_success"] = torch.ones(b)
        info["episode_tracking_fraction"] = torch.full((b,), 0.9)
        info["episode_within_distance_fraction"] = torch.full((b,), 0.95)
        info["episode_occlusion_tracking_fraction"] = torch.full((b,), math.nan)
        info["episode_reacquisition_time_s"] = torch.full((b,), math.nan)
        self.t = torch.where(ended, torch.zeros_like(self.t), self.t)  # automatic reset
        flags = torch.zeros(b, dtype=torch.bool)
        return StepResult(self._obs(), torch.zeros(b), flags, ended, info, self._obs(), {})


class _Actor:
    """Speed rises with the distance the world reports through the step count."""

    def reset(self, batch, device):
        self.t = torch.zeros(batch)
        self.resets = 0

    def act(self, obs, reset_mask):
        if reset_mask is not None:
            self.resets += int(reset_mask.sum())
            self.t = torch.where(reset_mask, torch.zeros_like(self.t), self.t)
        self.t += 1
        return torch.stack((0.05 * self.t, torch.zeros_like(self.t)), dim=-1)


def test_each_row_keeps_only_its_first_episode():
    world = _World(4)
    result = run_episodes(world, _Actor(), [0, 1, 2, 5], max_steps=20, sync_every=1)
    assert result.values["episode_steps"] == [3.0, 4.0, 5.0, 4.0]
    assert result.values["episode_seed"] == [0.0, 1.0, 2.0, 5.0]
    # step t commands 0.05 t after seeing 0.1 (t - 1) m: slope 0.5 m/s per m, from the
    # second step (the first has no distance seen before it) of the first episode only
    assert result.speed_distance["slope_m_s_per_m"] == pytest.approx([0.5] * 4)
    assert result.speed_distance["n"] == [2.0, 3.0, 4.0, 3.0]


def test_rows_that_never_finish_raise():
    with pytest.raises(RuntimeError, match="did not finish"):
        run_episodes(_World(2), _Actor(), [0, 1], max_steps=2)


def test_speed_clamp_holds_speed_and_flattens_the_slope():
    result = run_episodes(_World(2), SpeedClamp(_Actor(), 0.2), [0, 1], max_steps=20)
    assert result.speed_distance["mean_speed_m_s"] == pytest.approx([0.2, 0.2])
    assert result.speed_distance["slope_m_s_per_m"] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_records_from_a_run_pass_the_episode_schema():
    result = run_episodes(_World(2), _Actor(), [3, 4], max_steps=20)
    rows = [{"scenario": "normal"}, {"scenario": "noise", "noise_scale": 0.75}]
    records = to_records(
        result,
        rows,
        run_id="t",
        variant="v1:VO",
        train_seed=0,
        world="v1compat",
        motion="curved",
        artifact_sha256="0" * 64,
        commit="abc",
    )
    for record in records:
        validate_episode_record(record)
    assert records[1]["noise_scale"] == 0.75 and records[0]["noise_scale"] is None


def test_occlusion_phases_follow_the_reacquisition_rule():
    # one row, 6 steps: occluded at steps 2-3, bearing 30 deg until step 5, then 0
    world = _World(1, occluded_steps=(2, 3), bearing=lambda t: torch.where(t < 5, 0.52, 0.0))
    result = run_episodes(world, _Actor(), [3], max_steps=10, sync_every=1)
    phases = result.occlusion_phases
    assert [phases[f"{n}_steps"][0] for n in ("before", "during", "gap", "after")] == [1, 2, 1, 2]
    # tracked on even steps: before (1) no; during (2, 3) half; gap (4) yes; after (5, 6) half
    assert phases["before_tracking_fraction"][0] == 0.0
    assert phases["during_tracking_fraction"][0] == 0.5
    assert phases["gap_tracking_fraction"][0] == 1.0
    assert phases["after_tracking_fraction"][0] == 0.5


def test_an_occlusion_that_never_ended_has_null_reacquisition_fields():
    result = run_episodes(_World(1), _Actor(), [1], max_steps=10)
    (record,) = to_records(
        result,
        [{"scenario": "occlusion"}],
        run_id="t",
        variant="VO2",
        train_seed=0,
        world="v1compat",
        motion="curved",
        artifact_sha256="0" * 64,
        commit="abc",
    )
    assert record["reacquisition_censored"] is None and record["reacquisition_time_s"] is None
    validate_episode_record(record)
    with pytest.raises(ValueError, match="needs its time"):
        validate_episode_record({**record, "reacquisition_censored": False})


class _Echo:
    def reset(self, batch, device):
        pass

    def act(self, obs, reset_mask):
        self.seen = obs
        return torch.zeros(len(obs["odor"]), 2)


def _wind(theta_deg, alpha_deg=45.0, a=1.3):
    theta, alpha = math.radians(theta_deg), math.radians(alpha_deg)
    return torch.tensor([[a * math.cos(theta - alpha), a * math.cos(theta + alpha)]])


def test_odor_swap_keeps_the_total_and_leaves_the_rest():
    echo = _Echo()
    obs = {"odor": torch.tensor([[0.2, 0.7]]), "wind": _wind(20), "valid": torch.ones(1, 4)}
    ObservationAblation(echo, swap_odor).act(obs, None)
    assert echo.seen["odor"].tolist() == [[pytest.approx(0.7), pytest.approx(0.2)]]
    assert echo.seen["wind"] is obs["wind"] and echo.seen["valid"] is obs["valid"]


@pytest.mark.parametrize("theta,delta", [(0, 30), (20, -50), (-70, 180), (135, 90)])
def test_rotate_wind_reports_the_rotated_direction(theta, delta):
    echo = _Echo()
    obs = {"odor": torch.zeros(1, 2), "wind": _wind(theta)}
    ObservationAblation(echo, rotate_wind(delta, 45.0)).act(obs, None)
    torch.testing.assert_close(echo.seen["wind"], _wind(theta + delta))
