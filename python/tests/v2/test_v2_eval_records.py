"""Record-set checks before aggregation (#161 C5)."""

import pytest
from flytrack_v2.eval.records import check_records


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
