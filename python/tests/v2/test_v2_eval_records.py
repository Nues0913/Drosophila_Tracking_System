"""Record-set checks before aggregation (#161 C5)."""

import pytest
from flytrack_v2.eval.records import check_records, expected_grid


def _record(variant="VO2", seed=0, episode=1, artifact="a" * 64, **override):
    record = {
        "run_id": "eval",
        "variant": variant,
        "train_seed": seed,
        "world": "wind",
        "scenario": "normal",
        "motion": "curved",
        "episode_seed": episode,
        "success": True,
        "tracking_fraction": 0.9,
        "within_distance_fraction": 0.95,
        "collision": False,
        "steps": 1200,
        "mean_distance_m": 1.2,
        "bearing_error_deg": 10.0,
        "mean_speed_m_s": 0.2,
        "speed_std_m_s": 0.05,
        "occlusion_tracking_fraction": None,
        "reacquisition_time_s": None,
        "reacquisition_censored": None,
        "noise_scale": None,
        "artifact_sha256": artifact,
        "commit": "abc",
    }
    return {**record, **override}


def _paired():
    return [
        _record(v, s, e, artifact=v[0] * 64) for v in ("VO2", "V2") for s in (0, 1) for e in (1, 2)
    ]


def test_a_complete_paired_set_passes():
    assert len(check_records(_paired(), expected_episode_seeds=[1, 2])) == 8


def test_a_record_that_contradicts_itself_is_refused():
    records = _paired()
    records[0] = {**records[0], "collision": True}
    with pytest.raises(ValueError, match="record 0: success disagrees"):
        check_records(records)


def test_an_episode_counted_twice_is_refused():
    with pytest.raises(ValueError, match="repeats record"):
        check_records([*_paired(), _record("VO2", 0, 1, artifact="V" * 64)])


def test_one_actor_cannot_mix_artifacts():
    records = _paired()
    records[1] = {**records[1], "artifact_sha256": "b" * 64}
    with pytest.raises(ValueError, match="mixes 2 artifacts"):
        check_records(records)


def test_missing_and_unpaired_episodes_are_refused():
    records = [r for r in _paired() if not (r["variant"] == "V2" and r["episode_seed"] == 2)]
    with pytest.raises(ValueError, match=r"V2 in .*missing \[2\]"):
        check_records(records)
    with pytest.raises(ValueError, match=r"missing \[3\]"):
        check_records(_paired(), expected_episode_seeds=[1, 2, 3])


_PROTOCOL = {
    "worlds": ["v1compat", "wind"],
    "motions": {"primary": "curved", "secondary": "wander"},
    "episodes_per_cell": 2,
    "scenarios": {
        "normal": {},
        "noise": {"scale": 0.75, "sensitivity_scales": [0.5, 1.0]},
    },
    "episode_seeds": {"formal_evaluation": {"base": 1000, "per_train_seed": 100}},
}


def _from_grid(grid):
    return [
        _record(v, s, e, artifact=v[0] * 64, world=w, scenario=scenario, motion=m, noise_scale=n)
        for (v, s, w, scenario, m, n), seeds in grid.items()
        for e in sorted(seeds)
    ]


def test_expected_grid_uses_each_training_seed_episodes_and_the_noise_scales():
    grid = expected_grid(_PROTOCOL, "formal_evaluation", variants=["VO2"], train_seeds=[0, 3])
    # 2 seeds x 2 worlds x 2 motions x (normal + noise at 3 scales)
    assert len(grid) == 2 * 2 * 2 * 4
    assert grid[("VO2", 3, "wind", "noise", "wander", 1.0)] == {1300, 1301}
    assert grid[("VO2", 0, "v1compat", "normal", "curved", None)] == {1000, 1001}
    main = expected_grid(
        _PROTOCOL, "formal_evaluation", variants=["VO2"], train_seeds=[0], sensitivity=False
    )
    assert {key[3:] for key in main} == {("normal", m, None) for m in ("curved", "wander")} | {
        ("noise", m, 0.75) for m in ("curved", "wander")
    }


def test_the_grid_refuses_a_missing_block_and_another_seeds_episodes():
    grid = expected_grid(_PROTOCOL, "formal_evaluation", variants=["VO2", "V2"], train_seeds=[0, 1])
    records = _from_grid(grid)
    assert len(check_records(records, grid=grid)) == len(records)
    # every wander episode of VO2 gone: invisible to the per-cell check, caught by the grid
    no_wander = [r for r in records if not (r["variant"] == "VO2" and r["motion"] == "wander")]
    check_records([r for r in no_wander if r["motion"] == "curved"])
    with pytest.raises(ValueError, match="expected cells have no records"):
        check_records(no_wander, grid=grid)
    # seed 1 evaluated on seed 0's episode numbers
    shifted = [
        {**r, "episode_seed": r["episode_seed"] - 100} if r["train_seed"] == 1 else r
        for r in records
    ]
    with pytest.raises(ValueError, match=r"'train_seed': 1.*episode seeds differ"):
        check_records(shifted, grid=grid)
