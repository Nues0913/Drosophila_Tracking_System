"""Seed-level statistics and registered comparisons with known answers (#139)."""

import numpy as np
import pytest
from flytrack_v2.eval.stats import (
    Comparison,
    Interval,
    bootstrap_interval,
    holm,
    paired_comparison,
    seed_means,
)


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
        "G-B", {s: 0.5 + 0.01 * s for s in range(5)}, reference, "superiority", 0.0, _rng()
    )
    assert better.difference.low > 0 and better.p_value < 0.01
    tie = paired_comparison("tie", dict(reference), reference, "superiority", 0.0, _rng())
    assert tie.p_value == 1.0


def test_noninferiority_uses_the_negative_margin():
    reference = {s: 0.80 for s in range(5)}
    slightly_worse = {s: 0.78 for s in range(5)}  # -0.02, inside the 0.05 margin
    result = paired_comparison("G-C", slightly_worse, reference, "noninferiority", 0.05, _rng())
    assert result.threshold == -0.05 and result.p_value < 0.01
    clearly_worse = {s: 0.70 for s in range(5)}  # -0.10
    result = paired_comparison("G-C", clearly_worse, reference, "noninferiority", 0.05, _rng())
    assert result.p_value == 1.0


def test_pairing_requires_the_same_seeds():
    with pytest.raises(ValueError, match="same training seeds"):
        paired_comparison("x", {0: 1.0, 1: 1.0}, {0: 1.0}, "superiority", 0.0, _rng())


def test_holm_step_down_matches_the_textbook_example():
    interval = Interval(0.0, 0.0, 0.0, 5)
    family = [
        Comparison(n, "superiority", 0.0, interval, p)
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
