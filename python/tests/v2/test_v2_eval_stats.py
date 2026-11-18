"""Seed-level statistics and registered comparisons with known answers (#139)."""

import numpy as np
import pytest
from flytrack_v2.eval.stats import (
    Comparison,
    Interval,
    bootstrap_interval,
    gate_passes,
    holm,
    paired_comparison,
    restricted_reacquisition,
    seed_means,
)
from scipy import stats as scipy_stats


def _rng():
    return np.random.default_rng(0)


def test_episodes_average_within_a_seed_before_anything_else():
    records = [
        {
            "variant": "VO2",
            "world": "wind",
            "scenario": "normal",
            "motion": "curved",
            "noise_scale": None,
            "train_seed": seed,
            "success": success,
        }
        for seed, success in [(0, True), (0, False), (0, True), (1, False), (1, False)]
    ]
    cell = ("VO2", "wind", "normal", "curved", None)
    assert seed_means(records) == {cell: {0: pytest.approx(2 / 3), 1: 0.0}}


def test_missing_values_are_skipped_not_counted_as_zero():
    records = [
        {
            "variant": "V2",
            "world": "wind",
            "scenario": "occlusion",
            "motion": "curved",
            "noise_scale": None,
            "train_seed": 0,
            "reacquisition_time_s": value,
        }
        for value in (1.0, None, 3.0)
    ]
    (seeds,) = seed_means(records, "reacquisition_time_s").values()
    assert seeds == {0: 2.0}


def test_bootstrap_of_identical_seeds_has_a_zero_width_interval():
    interval = bootstrap_interval([0.8] * 5, _rng(), draws=200)
    assert (interval.mean, interval.low, interval.high, interval.n_seeds) == (0.8, 0.8, 0.8, 5)


def test_bootstrap_interval_brackets_the_mean():
    interval = bootstrap_interval([0.1, 0.4, 0.5, 0.7, 0.9], _rng())
    assert interval.low < interval.mean < interval.high
    with pytest.raises(ValueError):
        bootstrap_interval([0.1, float("nan")], _rng())


def test_clear_superiority_passes_and_a_tie_does_not():
    reference = {s: 0.2 for s in range(5)}
    better = paired_comparison(
        "G-B", {s: 0.5 + 0.01 * s for s in range(5)}, reference, "superiority"
    )
    assert better.difference.low > 0 and better.p_value < 0.01
    tie = paired_comparison("tie", dict(reference), reference, "superiority")
    assert tie.p_value == 1.0


def test_paired_t_matches_scipy_and_the_review_example():
    """#152 review: differences 0.001-0.005 give one-sided paired t p = 0.0066."""
    candidate = {s: 0.5 + 0.001 * (s + 1) for s in range(5)}
    reference = {s: 0.5 for s in range(5)}
    result = paired_comparison("small", candidate, reference, "superiority")
    expected = scipy_stats.ttest_1samp(
        [0.001, 0.002, 0.003, 0.004, 0.005], 0.0, alternative="greater"
    ).pvalue
    assert result.p_value == pytest.approx(expected) and result.p_value == pytest.approx(
        0.0066, abs=1e-4
    )
    assert result.difference.low < result.bootstrap.low  # t is wider than the 5-seed bootstrap


def test_noninferiority_uses_the_negative_margin():
    reference = {s: 0.80 for s in range(5)}
    slightly_worse = {s: 0.78 + 0.002 * s for s in range(5)}  # about -0.016
    result = paired_comparison("G-C", slightly_worse, reference, "noninferiority", 0.05)
    assert result.threshold == -0.05 and result.p_value < 0.01
    clearly_worse = {s: 0.70 + 0.002 * s for s in range(5)}  # about -0.096
    result = paired_comparison("G-C", clearly_worse, reference, "noninferiority", 0.05)
    assert result.p_value > 0.99


def test_pairing_requires_the_same_seeds_and_names_the_missing_ones():
    with pytest.raises(ValueError, match=r"unmatched \[1\]"):
        paired_comparison("x", {0: 1.0, 1: 1.0}, {0: 1.0}, "superiority")


def test_each_comparison_draws_from_its_own_generator():
    a = {s: 0.5 + 0.03 * s for s in range(5)}
    b = {s: 0.4 + 0.01 * s for s in range(5)}
    first = paired_comparison("G-B", a, b, "superiority", protocol_seed=7)
    paired_comparison("other", b, a, "superiority", protocol_seed=7)
    again = paired_comparison("G-B", a, b, "superiority", protocol_seed=7)
    assert first.bootstrap == again.bootstrap


def test_holm_step_down_matches_the_textbook_example():
    interval = Interval(0.0, 0.0, 0.0, 5)
    family = [
        Comparison(n, "superiority", 0.0, interval, interval, p)
        for n, p in (("a", 0.01), ("b", 0.04), ("c", 0.03), ("d", 0.005))
    ]
    adjusted = {c.name: (c.p_holm, c.passed) for c in holm(family)}
    # sorted p: 0.005, 0.01, 0.03, 0.04 -> 0.02, 0.03, 0.06, 0.06 (monotone)
    assert adjusted == {
        "d": (pytest.approx(0.02), True),
        "a": (pytest.approx(0.03), True),
        "c": (pytest.approx(0.06), False),
        "b": (pytest.approx(0.06), False),
    }


def test_gate_needs_holm_and_the_minimum_difference():
    reference = {s: 0.2 for s in range(5)}
    small = paired_comparison(
        "G-B", {s: 0.25 + 0.002 * s for s in range(5)}, reference, "superiority"
    )
    large = paired_comparison(
        "G-D", {s: 0.45 + 0.002 * s for s in range(5)}, reference, "superiority"
    )
    with pytest.raises(ValueError, match="holm"):
        gate_passes(small, 0.10)
    small, large = holm([small, large])
    assert not gate_passes(small, 0.10)  # significant but only about +0.054
    assert gate_passes(large, 0.10)


def test_restricted_reacquisition_counts_censored_episodes_at_the_bound():
    base = {"reacquisition_time_s": None}
    records = [
        {**base, "reacquisition_censored": None},  # no occlusion
        {**base, "reacquisition_censored": True},
        {"reacquisition_censored": False, "reacquisition_time_s": 3.0},
        {"reacquisition_censored": False, "reacquisition_time_s": 40.0},
    ]
    values = [
        r["restricted_reacquisition_time_s"] for r in restricted_reacquisition(records[:3], 22.0)
    ]
    assert values == [None, 22.0, 3.0]
    with pytest.raises(ValueError, match="exceeds the bound"):
        restricted_reacquisition(records, 22.0)  # 40 s cannot be observed before the end
    with pytest.raises(ValueError):
        restricted_reacquisition(records, 0.0)


def test_equal_differences_use_the_exact_sign_flip_value():
    """#155 review: five seeds all +0.01 gave p = 0; the exact one-sided value is 1/32."""
    reference = {s: 0.5 for s in range(5)}
    result = paired_comparison("flat", {s: 0.51 for s in range(5)}, reference, "superiority")
    assert result.degenerate and result.p_value == pytest.approx(1 / 32)
    (adjusted,) = holm([result])
    assert adjusted.degenerate and adjusted.passed  # 1/32 < 0.05 for a family of one
    tie = paired_comparison("tie", dict(reference), reference, "superiority")
    assert tie.degenerate and tie.p_value == 1.0


def test_superiority_refuses_a_margin():
    with pytest.raises(ValueError, match="gate_passes"):
        paired_comparison("G-B", {0: 0.6, 1: 0.7}, {0: 0.4, 1: 0.5}, "superiority", 0.10)
