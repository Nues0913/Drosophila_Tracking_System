"""Checks a set of episode records must pass before any aggregation (#161 C5).

validate_episode_record checks one record; these check the set: no episode counted twice, one
artifact per trained actor, every expected episode present, and the same episode seeds for
every variant compared inside a cell, so that paired comparisons really pair.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from flytrack_v2.metrics import validate_episode_record

# A cell holds the episodes that variants are compared on.
CELL = ("train_seed", "world", "scenario", "motion", "noise_scale")


def check_records(
    records: Iterable[Mapping], expected_episode_seeds: Sequence[int] | None = None
) -> list[Mapping]:
    """Validate every record and the set; raise ValueError listing every problem found."""
    records = list(records)
    problems: list[str] = []
    for index, record in enumerate(records):
        try:
            validate_episode_record(record)
        except ValueError as error:
            problems.append(f"record {index}: {error}")
    if problems:
        raise ValueError("; ".join(problems[:20]))
    seen: dict[tuple, int] = {}
    artifacts: dict[tuple, set[str]] = defaultdict(set)
    seeds: dict[tuple, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for index, record in enumerate(records):
        cell = tuple(record[key] for key in CELL)
        key = (record["variant"], *cell, record["episode_seed"])
        if key in seen:
            problems.append(f"record {index} repeats record {seen[key]}: {key}")
        seen[key] = index
        artifacts[(record["variant"], record["train_seed"])].add(record["artifact_sha256"])
        seeds[cell][record["variant"]].add(record["episode_seed"])
    for actor, hashes in sorted(artifacts.items(), key=str):
        if len(hashes) > 1:
            problems.append(f"{actor} mixes {len(hashes)} artifacts")
    expected = None if expected_episode_seeds is None else set(expected_episode_seeds)
    for cell, by_variant in sorted(seeds.items(), key=str):
        reference = expected if expected is not None else next(iter(by_variant.values()))
        for variant, episode_seeds in sorted(by_variant.items()):
            if episode_seeds != reference:
                missing = sorted(reference - episode_seeds)[:5]
                extra = sorted(episode_seeds - reference)[:5]
                problems.append(
                    f"{variant} in {cell}: episode seeds differ (missing {missing}, extra {extra})"
                )
    if problems:
        raise ValueError("; ".join(problems[:20]))
    return records
