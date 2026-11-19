"""The evaluation summary JSON (#139, docs/v2/metrics.md)."""

import json

import pytest
from flytrack_v2.eval.summary import (
    ComparisonSpec,
    build_summary,
    records_sha256,
    selection_seed_means,
    validate_summary,
)


def _record(variant, seed, episode, success, motion="curved", **override):
    tracking = 0.9 if success else 0.5
    record = {
        "run_id": "eval",
        "variant": variant,
        "train_seed": seed,
        "world": "wind",
        "scenario": "occlusion",
        "motion": motion,
        "episode_seed": episode,
        "success": success,
        "tracking_fraction": tracking,
        "within_distance_fraction": 0.95,
        "collision": False,
        "steps": 1200,
        "mean_distance_m": 1.2,
        "bearing_error_deg": 10.0,
        "mean_speed_m_s": 0.2,
        "speed_std_m_s": 0.05,
        "occlusion_tracking_fraction": tracking,
        "reacquisition_time_s": 1.5,
        "reacquisition_censored": False,
        "noise_scale": None,
        "artifact_sha256": f"{variant}-{seed}".ljust(64, "0"),
        "commit": "abc",
    }
    return {**record, **override}


def _records():
    """VO2 succeeds on 6 of 10 episodes per seed and motion plus one per seed; V2 on 3."""
    records = []
    for seed in range(5):
        for motion in ("curved", "wander"):
            for episode in range(10):
                records.append(_record("VO2", seed, episode, episode < 6 + (seed % 2), motion))
                records.append(_record("V2", seed, episode, episode < 3, motion))
    return records


G_B = ComparisonSpec(
    "G-B",
    "primary",
    "superiority",
    {"variant": "VO2", "world": "wind", "scenario": "occlusion"},
    {"variant": "V2", "world": "wind", "scenario": "occlusion"},
    minimum_difference=0.10,
)


def test_selection_pools_both_motions_per_seed():
    means = selection_seed_means(_records(), G_B.candidate, "success")
    assert means == {0: 0.6, 1: 0.7, 2: 0.6, 3: 0.7, 4: 0.6}
    wander = selection_seed_means(_records(), {**G_B.candidate, "motion": "wander"}, "success")
    assert wander == means
    with pytest.raises(ValueError, match="no records match"):
        selection_seed_means(_records(), {"variant": "GRU2"}, "success")


def test_selection_never_pools_variants():
    with pytest.raises(ValueError, match=r"several variants \['V2', 'VO2'\]"):
        selection_seed_means(_records(), {"world": "wind", "scenario": "occlusion"}, "success")
    only_vo2 = [r for r in _records() if r["variant"] == "VO2"]
    assert selection_seed_means(only_vo2, {"world": "wind"}, "success") == (
        selection_seed_means(only_vo2, {"variant": "VO2"}, "success")
    )


def test_summary_reports_the_gate_and_the_secondary_family():
    per_motion = [
        ComparisonSpec(
            f"G-B:{motion}",
            "G-B by motion",
            "superiority",
            {**G_B.candidate, "motion": motion},
            {**G_B.reference, "motion": motion},
            secondary=True,
        )
        for motion in ("curved", "wander")
    ]
    summary = build_summary(
        _records(), [G_B, *per_motion], protocol_seed=7, protocol_sha256="p" * 64, draws=200
    )
    gate, curved, wander = summary["comparisons"]
    assert gate["mean"] == pytest.approx(0.34) and gate["gate_passed"] is True
    assert gate["p_holm"] == gate["p_value"]  # a family of one
    assert curved["secondary"] and curved["gate_passed"] is None
    assert curved["p_holm"] == pytest.approx(
        min(1.0, 2 * min(curved["p_value"], wander["p_value"]))
    )
    cell = next(
        c
        for c in summary["cells"]
        if c["variant"] == "V2" and c["motion"] == "wander" and c["metric"] == "success"
    )
    assert cell["mean"] == pytest.approx(0.3) and cell["t95"] == [pytest.approx(0.3)] * 2
    assert summary["records_sha256"] == records_sha256(reversed(_records()))
    json.dumps(summary)  # plain JSON
    validate_summary(summary)


def test_summary_is_reproducible_from_the_protocol_seed():
    first = build_summary(_records(), [G_B], protocol_seed=3, protocol_sha256="p" * 64, draws=200)
    again = build_summary(_records(), [G_B], protocol_seed=3, protocol_sha256="p" * 64, draws=200)
    assert first == again


def test_summary_refuses_unpairable_records_and_mixed_families():
    records = _records()
    records.pop()  # V2 loses one wander episode of seed 4
    with pytest.raises(ValueError, match="episode seeds differ"):
        build_summary(records, [G_B], protocol_seed=0, protocol_sha256="p" * 64)
    mixed = ComparisonSpec(**{**G_B.__dict__, "name": "x", "secondary": True})
    with pytest.raises(ValueError, match="mixes gates"):
        build_summary(_records(), [G_B, mixed], protocol_seed=0, protocol_sha256="p" * 64)


def test_validate_summary_catches_inconsistent_rows():
    summary = build_summary(_records(), [G_B], protocol_seed=0, protocol_sha256="p" * 64, draws=200)
    bad = json.loads(json.dumps(summary))
    bad["comparisons"][0]["p_holm"] = bad["comparisons"][0]["p_value"] / 2
    with pytest.raises(ValueError, match="Holm cannot lower p"):
        validate_summary(bad)
    bad = json.loads(json.dumps(summary))
    bad["cells"][0]["n_seeds"] = 4
    with pytest.raises(ValueError, match="n_seeds"):
        validate_summary(bad)
