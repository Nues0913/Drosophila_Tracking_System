"""Seed-level statistics and registered comparisons for v2 (#139; protocol schema 4).

The unit is the training seed: episodes are averaged within a seed first, then the seed means
are bootstrapped (protocol `statistics`). Comparisons pair variants by training seed, because
seed s of every variant shares the same episode seeds.

Decisions use a one-sided paired t test (n - 1 degrees of freedom) and Holm's step-down over
the registered family (#152 Decision): with five seeds a bootstrap test is anti-conservative
(about 12% false positives at a nominal 5%). Bootstrap intervals are still reported. Each
comparison draws from its own generator, derived from (protocol seed, comparison name), so a
rerun gives the same intervals whatever the call order. Nothing here asks for bitwise reruns.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

CELL_KEYS = ("variant", "world", "scenario", "motion", "noise_scale")


def seed_means(
    records: Iterable[Mapping], metric: str = "success", keys: Sequence[str] = CELL_KEYS
) -> dict[tuple, dict[int, float]]:
    """{cell: {train_seed: mean of metric over that seed's episodes}}; booleans count as 0/1."""
    sums: dict[tuple, dict[int, list[float]]] = {}
    for record in records:
        value = record[metric]
        if value is None:
            continue
        cell = tuple(record[key] for key in keys)
        sums.setdefault(cell, {}).setdefault(int(record["train_seed"]), []).append(float(value))
    return {
        cell: {seed: float(np.mean(values)) for seed, values in sorted(seeds.items())}
        for cell, seeds in sums.items()
    }


def _bootstrap(values: np.ndarray, rng: np.random.Generator, draws: int) -> np.ndarray:
    if draws <= 0:
        raise ValueError("draws must be positive")
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap needs a nonempty vector of finite seed values")
    return rng.choice(values, size=(draws, len(values)), replace=True).mean(1)


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n_seeds: int


def bootstrap_interval(
    values: Sequence[float],
    rng: np.random.Generator,
    draws: int = 10000,
    confidence: float = 0.95,
) -> Interval:
    values = np.asarray(values, dtype=np.float64)
    samples = _bootstrap(values, rng, draws)
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return Interval(float(values.mean()), float(low), float(high), len(values))


@dataclass(frozen=True)
class Comparison:
    name: str
    kind: str  # superiority or noninferiority
    threshold: float  # the difference H1 must exceed: 0 (superiority) or -margin
    difference: Interval  # mean and two-sided t interval, n - 1 degrees of freedom
    bootstrap: Interval  # reported alongside; never used for a decision
    p_value: float  # one-sided paired t test, H0: difference <= threshold
    p_holm: float | None = None
    passed: bool | None = None  # Holm-adjusted p < alpha, set by holm()
    # All seed differences equal (standard error 0): p is the exact sign-flip value instead of
    # a t value, and reports must say so (#155 review).
    degenerate: bool = False


def comparison_rng(protocol_seed: int, name: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{int(protocol_seed)}:{name}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def paired_comparison(
    name: str,
    candidate: Mapping[int, float],
    reference: Mapping[int, float],
    kind: str,
    margin: float = 0.0,
    protocol_seed: int = 0,
    draws: int = 10000,
    confidence: float = 0.95,
) -> Comparison:
    """candidate - reference, paired by training seed.

    superiority: H1 difference > 0 (G-B, G-D also need the mean to reach their minimum, see
    gate_passes); noninferiority: H1 difference > -margin (G-C uses 0.05).
    """
    if kind not in ("superiority", "noninferiority"):
        raise ValueError("kind must be superiority or noninferiority")
    if kind == "superiority" and margin != 0:
        raise ValueError(
            "superiority tests against 0; a gate's minimum difference goes to gate_passes"
        )
    if set(candidate) != set(reference) or not candidate:
        missing = sorted(set(candidate) ^ set(reference))
        raise ValueError(f"{name}: both sides need the same training seeds; unmatched {missing}")
    seeds = sorted(candidate)
    if len(seeds) < 2:
        raise ValueError(f"{name}: a paired t test needs at least two seeds")
    differences = np.asarray([candidate[s] - reference[s] for s in seeds], dtype=np.float64)
    threshold = 0.0 if kind == "superiority" else -float(margin)
    n, mean = len(differences), float(differences.mean())
    se = float(differences.std(ddof=1)) / math.sqrt(n)
    degenerate = se == 0
    if degenerate:
        # Exact one-sided sign-flip test: n equal differences above the threshold reach
        # p = 1 / 2^n (1/32 for five seeds), not 0.
        p_value, low, high = (0.5**n if mean > threshold else 1.0), mean, mean
    else:
        p_value = float(scipy_stats.t.sf((mean - threshold) / se, n - 1))
        half = float(scipy_stats.t.ppf(1 - (1 - confidence) / 2, n - 1)) * se
        low, high = mean - half, mean + half
    samples = _bootstrap(differences, comparison_rng(protocol_seed, name), draws)
    tail = (1 - confidence) / 2
    boot_low, boot_high = np.quantile(samples, [tail, 1 - tail])
    return Comparison(
        name,
        kind,
        threshold,
        Interval(mean, low, high, n),
        Interval(mean, float(boot_low), float(boot_high), n),
        p_value,
        degenerate=degenerate,
    )


def holm(comparisons: Sequence[Comparison], alpha: float = 0.05) -> list[Comparison]:
    """Holm step-down over one registered family; returns comparisons in input order."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    order = sorted(range(len(comparisons)), key=lambda index: comparisons[index].p_value)
    count = len(comparisons)
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * comparisons[index].p_value))
        adjusted[index] = running
    return [
        Comparison(
            c.name,
            c.kind,
            c.threshold,
            c.difference,
            c.bootstrap,
            c.p_value,
            p_holm=adjusted[i],
            passed=adjusted[i] < alpha,
            degenerate=c.degenerate,
        )
        for i, c in enumerate(comparisons)
    ]


def gate_passes(comparison: Comparison, minimum_difference: float | None = None) -> bool:
    """A registered gate: Holm-adjusted p < alpha, and the mean reaches its minimum if any
    (G-B and G-D: 0.10). The interval is reported, not tested (#152 Decision)."""
    if comparison.passed is None:
        raise ValueError(f"{comparison.name}: run holm() over the registered family first")
    reaches = minimum_difference is None or comparison.difference.mean >= minimum_difference
    return bool(comparison.passed and reaches)


def restricted_reacquisition(records: Iterable[Mapping], bound_s: float) -> list[dict]:
    """Copies of the records with restricted_reacquisition_time_s: a censored episode counts
    as the observable bound (occlusion end to episode end), so variants that often fail to
    reacquire are not reported as fast. Report it with the censored fraction
    (seed_means(records, "reacquisition_censored")), never the uncensored mean alone."""
    if not math.isfinite(bound_s) or bound_s <= 0:
        raise ValueError("bound_s must be positive and finite")
    out = []
    for record in records:
        censored = record["reacquisition_censored"]
        if censored is None:
            value = None
        elif censored:
            value = float(bound_s)
        else:
            value = float(record["reacquisition_time_s"])
            if value > bound_s:
                # Beyond the observable bound means a wrong bound or episode length (#155 review).
                raise ValueError(f"reacquisition time {value} s exceeds the bound {bound_s} s")
        out.append({**record, "restricted_reacquisition_time_s": value})
    return out
