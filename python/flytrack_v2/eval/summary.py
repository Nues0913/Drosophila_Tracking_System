"""The summary JSON of one evaluation (docs/v2/metrics.md, #139).

build_summary checks the record set (check_records), then reports

- cells: every metric's seed means per (variant, world, scenario, motion, noise_scale), with a
  t interval over seeds and a bootstrap interval (reported only, never tested);
- comparisons: the registered comparisons, each a paired t test over training seeds between
  two record selections, Holm-adjusted within its family, and its gate (Holm p < alpha and the
  minimum difference, if any).

A selection is a dict of record fields to values; a seed's value is the mean of the metric over
all of that seed's matching episodes, so {"world": "wind", "scenario": "occlusion"} pools both
motions (G-B as registered) and adding "motion" gives the per-motion secondary analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats

from flytrack_v2.eval.records import check_records
from flytrack_v2.eval.stats import (
    CELL_KEYS,
    Comparison,
    bootstrap_interval,
    comparison_rng,
    gate_passes,
    holm,
    paired_comparison,
    seed_means,
)

SUMMARY_SPEC_VERSION = 1
SUMMARY_METRICS = (
    "success",
    "tracking_fraction",
    "within_distance_fraction",
    "collision",
    "mean_speed_m_s",
    "speed_std_m_s",
    "occlusion_tracking_fraction",
    "reacquisition_censored",
)


@dataclass(frozen=True)
class ComparisonSpec:
    """One registered comparison: candidate - reference on `metric`, paired by train_seed."""

    name: str
    family: str  # the Holm family it is corrected within
    kind: str  # superiority or noninferiority
    candidate: Mapping[str, object]
    reference: Mapping[str, object]
    metric: str = "success"
    margin: float = 0.0  # noninferiority only
    minimum_difference: float | None = None  # G-B, G-D: 0.10
    secondary: bool = False  # reported, never a gate
    extra: dict = field(default_factory=dict)  # free-form notes copied to the output


def selection_seed_means(
    records: Sequence[Mapping], selection: Mapping[str, object], metric: str
) -> dict[int, float]:
    """{train_seed: mean of metric over the seed's records matching every selection field}.

    A selection may pool motions or worlds on purpose (G-B pools both motions), but never
    variants: records of two variants under one selection raise (#172 review).
    """
    values: dict[int, list[float]] = defaultdict(list)
    variants = set()
    for record in records:
        if all(record[key] == value for key, value in selection.items()):
            variants.add(record["variant"])
            if record[metric] is not None:
                values[int(record["train_seed"])].append(float(record[metric]))
    if len(variants) > 1:
        raise ValueError(
            f"{dict(selection)} matches the records of several variants {sorted(variants)}; "
            "name the variant in the selection"
        )
    if not values:
        raise ValueError(f"no records match {dict(selection)} with a value for {metric}")
    return {seed: float(np.mean(v)) for seed, v in sorted(values.items())}


_EPISODE = ("train_seed", "world", "scenario", "motion", "noise_scale", "episode_seed")


def check_pairing(records: Sequence[Mapping], spec: ComparisonSpec) -> None:
    """Refuse a comparison whose two sides do not cover the same episodes (#192 R1).

    A paired test over seed means only pairs if both sides average the same episodes: with a
    motion missing on one side, the seed means compare an easy motion with a hard one.
    """

    def covered(selection: Mapping) -> set[tuple]:
        return {
            tuple(r[key] for key in _EPISODE)
            for r in records
            if all(r[key] == value for key, value in selection.items())
        }

    candidate, reference = covered(spec.candidate), covered(spec.reference)
    if candidate != reference:
        only_c, only_r = (
            sorted(candidate - reference, key=str),
            sorted(reference - candidate, key=str),
        )
        example = dict(zip(_EPISODE, (only_c or only_r)[0], strict=True))
        raise ValueError(
            f"{spec.name}: the two sides do not cover the same episodes ({len(only_c)} only in "
            f"the candidate, {len(only_r)} only in the reference, e.g. {example})"
        )


def t_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float] | None:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return None
    mean = float(values.mean())
    half = (
        float(scipy_stats.t.ppf(1 - (1 - confidence) / 2, len(values) - 1))
        * float(values.std(ddof=1))
        / math.sqrt(len(values))
    )
    return mean - half, mean + half


def records_sha256(records: Iterable[Mapping]) -> str:
    """Order-free hash of a record set; the summary names the records it came from."""
    lines = sorted(json.dumps(dict(r), sort_keys=True, separators=(",", ":")) for r in records)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _cells(records, protocol_seed: int, draws: int) -> list[dict]:
    rows = []
    for metric in SUMMARY_METRICS:
        for cell, by_seed in sorted(seed_means(records, metric).items(), key=str):
            values = list(by_seed.values())
            name = f"cell:{metric}:{json.dumps(cell)}"
            boot = bootstrap_interval(values, comparison_rng(protocol_seed, name), draws)
            t = t_interval(values)
            rows.append(
                {
                    **dict(zip(CELL_KEYS, cell, strict=True)),
                    "metric": metric,
                    "seeds": {str(seed): value for seed, value in by_seed.items()},
                    "n_seeds": len(values),
                    "mean": boot.mean,
                    "t95": None if t is None else list(t),
                    "bootstrap95": [boot.low, boot.high],
                }
            )
    return rows


def _comparison_row(spec: ComparisonSpec, c: Comparison, alpha: float) -> dict:
    return {
        "name": spec.name,
        "family": spec.family,
        "kind": spec.kind,
        "metric": spec.metric,
        "candidate": dict(spec.candidate),
        "reference": dict(spec.reference),
        "threshold": c.threshold,
        "minimum_difference": spec.minimum_difference,
        "n_seeds": c.difference.n_seeds,
        "mean": c.difference.mean,
        "t95": [c.difference.low, c.difference.high],
        "bootstrap95": [c.bootstrap.low, c.bootstrap.high],
        "p_value": c.p_value,
        "p_holm": c.p_holm,
        "alpha": alpha,
        # all seed differences equal: p is the exact sign-flip value and t95 has zero width
        "degenerate": c.degenerate,
        "secondary": spec.secondary,
        "gate_passed": None if spec.secondary else gate_passes(c, spec.minimum_difference),
        **({"extra": spec.extra} if spec.extra else {}),
    }


def build_summary(
    records: Iterable[Mapping],
    specs: Sequence[ComparisonSpec],
    *,
    protocol_seed: int,
    protocol_sha256: str,
    expected_episode_seeds: Sequence[int] | None = None,
    expected: Mapping[tuple, Iterable[int]] | None = None,
    alpha: float = 0.05,
    draws: int = 10000,
) -> dict:
    """The summary JSON. The formal summary passes ``expected`` (records.expected_grid)."""
    records = check_records(records, expected_episode_seeds, grid=expected)
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("comparison names must be unique")
    by_family: dict[str, list[tuple[ComparisonSpec, Comparison]]] = defaultdict(list)
    for spec in specs:
        check_pairing(records, spec)
        candidate = selection_seed_means(records, spec.candidate, spec.metric)
        reference = selection_seed_means(records, spec.reference, spec.metric)
        comparison = paired_comparison(
            spec.name,
            candidate,
            reference,
            spec.kind,
            margin=spec.margin,
            protocol_seed=protocol_seed,
            draws=draws,
        )
        by_family[spec.family].append((spec, comparison))
    adjusted: dict[str, dict] = {}
    for family, pairs in by_family.items():
        if len({spec.secondary for spec, _ in pairs}) > 1:
            raise ValueError(f"family {family} mixes gates and secondary analyses")
        for (spec, _), c in zip(pairs, holm([c for _, c in pairs], alpha), strict=True):
            adjusted[spec.name] = _comparison_row(spec, c, alpha)
    summary = {
        "spec_version": SUMMARY_SPEC_VERSION,
        "protocol_sha256": protocol_sha256,
        "protocol_seed": int(protocol_seed),
        "records_sha256": records_sha256(records),
        "n_records": len(records),
        "commits": sorted({r["commit"] for r in records}),
        "comparisons": [adjusted[name] for name in names],
        "cells": _cells(records, protocol_seed, draws),
    }
    validate_summary(summary)
    return summary


def _check_interval(where: str, mean: float, interval) -> None:
    if interval is None:
        return
    low, high = interval
    if not (math.isfinite(low) and math.isfinite(high) and low - 1e-12 <= mean <= high + 1e-12):
        raise ValueError(f"{where}: interval {interval} does not bracket the mean {mean}")


def validate_summary(summary: Mapping) -> None:
    """Shape and internal consistency of a summary JSON (docs/v2/metrics.md)."""
    keys = {
        "spec_version",
        "protocol_sha256",
        "protocol_seed",
        "records_sha256",
        "n_records",
        "commits",
        "comparisons",
        "cells",
    }
    if set(summary) != keys:
        raise ValueError(f"summary keys must be {sorted(keys)}")
    if summary["spec_version"] != SUMMARY_SPEC_VERSION:
        raise ValueError("unknown summary spec_version")
    for row in summary["comparisons"]:
        where = f"comparison {row['name']}"
        for p in ("p_value", "p_holm"):
            if not 0 <= row[p] <= 1:
                raise ValueError(f"{where}: {p} must be in [0, 1]")
        if row["p_holm"] < row["p_value"] - 1e-15:
            raise ValueError(f"{where}: Holm cannot lower p")
        _check_interval(where, row["mean"], row["t95"])
        _check_interval(where, row["mean"], row["bootstrap95"])
        if row["secondary"] != (row["gate_passed"] is None):
            raise ValueError(f"{where}: exactly the gates carry gate_passed")
    for row in summary["cells"]:
        where = f"cell {row['variant']} {row['scenario']} {row['metric']}"
        if row["n_seeds"] != len(row["seeds"]):
            raise ValueError(f"{where}: n_seeds disagrees with seeds")
        _check_interval(where, row["mean"], row["t95"])
        _check_interval(where, row["mean"], row["bootstrap95"])
