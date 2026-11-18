"""Seed-level statistics and registered comparisons for v2 (#139; protocol schema 4).

The unit is the training seed: episodes are averaged within a seed first, then the seed means
are bootstrapped (protocol `statistics`). Comparisons pair variants by training seed, because
seed s of every variant shares the same episode seeds. One-sided bootstrap p-values feed a
Holm step-down over the registered family. Nothing here asks for bitwise reruns: a rerun
reproduces when it lands inside these intervals.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

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
    threshold: float  # difference H1 must exceed: margin (superiority) or -margin
    difference: Interval
    p_value: float  # one-sided bootstrap, H0: difference <= threshold
    p_holm: float | None = None
    passed: bool | None = None


def paired_comparison(
    name: str,
    candidate: Mapping[int, float],
    reference: Mapping[int, float],
    kind: str,
    margin: float,
    rng: np.random.Generator,
    draws: int = 10000,
    confidence: float = 0.95,
) -> Comparison:
    """candidate - reference, paired by training seed.

    superiority: H1 difference > margin (G-B uses margin 0 and asks the mean for >= 0.10
    separately); noninferiority: H1 difference > -margin (G-C uses 0.05).
    """
    if kind not in ("superiority", "noninferiority"):
        raise ValueError("kind must be superiority or noninferiority")
    if set(candidate) != set(reference) or not candidate:
        raise ValueError(f"{name}: both sides need the same training seeds")
    seeds = sorted(candidate)
    differences = np.asarray([candidate[s] - reference[s] for s in seeds], dtype=np.float64)
    samples = _bootstrap(differences, rng, draws)
    threshold = float(margin) if kind == "superiority" else -float(margin)
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    interval = Interval(float(differences.mean()), float(low), float(high), len(seeds))
    # Add-one estimate: a p-value of exactly 0 would claim more than the draws can show.
    p_value = (int(np.sum(samples <= threshold)) + 1) / (draws + 1)
    return Comparison(name, kind, threshold, interval, p_value)


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
            c.p_value,
            p_holm=adjusted[i],
            passed=adjusted[i] < alpha,
        )
        for i, c in enumerate(comparisons)
    ]
