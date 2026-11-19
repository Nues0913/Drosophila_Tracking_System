"""Checks a set of episode records must pass before any aggregation (#161 C5).

validate_episode_record checks one record; these check the set: no episode counted twice, one
artifact per trained actor, every expected episode present, and the same episode seeds for
every variant compared inside a cell, so that paired comparisons really pair.

Inside a cell the variants are compared against each other, so a variant that is missing from
whole cells (say every wander episode) is invisible there. The formal summary therefore also
passes expected_grid(protocol, ...): every variant x training seed x world x scenario x motion x
noise scale the protocol asks for, with that training seed's own episode numbers, as
check_records(..., grid=...) (#192 R1).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from flytrack_v2.metrics import V1_REFERENCE_PREFIX, validate_episode_record

# A cell holds the episodes that variants are compared on.
CELL = ("train_seed", "world", "scenario", "motion", "noise_scale")
# One trained actor's episodes in one cell.
GRID_KEY = ("variant", *CELL)


def expected_grid(
    protocol: Mapping,
    use: str,
    *,
    variants: Sequence[str],
    train_seeds: Sequence[int],
    worlds: Sequence[str] | None = None,
    motions: Sequence[str] | None = None,
    sensitivity: bool = True,
    episodes: int | None = None,
) -> dict[tuple, frozenset[int]]:
    """{GRID_KEY values: episode seeds} the protocol expects from an evaluation (#192 R1).

    Episode seeds are base + per_train_seed * train_seed + e, e < episodes_per_cell, the same in
    every world, scenario and motion; the noise scenario appears at the protocol's scale and,
    with ``sensitivity``, at each sensitivity scale; other scenarios have noise_scale None.
    """
    entry = protocol["episode_seeds"][use]
    worlds = list(worlds or protocol["worlds"])
    motions = list(motions or (protocol["motions"]["primary"], protocol["motions"]["secondary"]))
    episodes = episodes or protocol["episodes_per_cell"]
    noise = protocol["scenarios"]["noise"]
    scales = [noise["scale"], *(noise["sensitivity_scales"] if sensitivity else [])]
    grid = {}
    for seed in train_seeds:
        first = entry["base"] + entry["per_train_seed"] * seed
        seeds = frozenset(range(first, first + episodes))
        for variant in variants:
            for world in worlds:
                for motion in motions:
                    for scenario in protocol["scenarios"]:
                        for scale in scales if scenario == "noise" else [None]:
                            grid[(variant, seed, world, scenario, motion, scale)] = seeds
    return grid


def check_records(
    records: Iterable[Mapping],
    expected_episode_seeds: Sequence[int] | None = None,
    *,
    grid: Mapping[tuple, Iterable[int]] | None = None,
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
    # Every v2 variant must cover the same cells (#192 R1), with or without a grid: a variant
    # missing a whole motion is invisible inside the cells. Formal-v1 reference actors
    # ("v1:<variant>", #192 R2) run in v1compat's main run only; the grid and
    # summary.check_pairing cover them.
    cells_by_variant: dict[str, set[tuple]] = defaultdict(set)
    for cell, by_variant in seeds.items():
        for variant in by_variant:
            if not variant.startswith(V1_REFERENCE_PREFIX):
                cells_by_variant[variant].add(cell)
    if len({frozenset(cells) for cells in cells_by_variant.values()}) > 1:
        union = set().union(*cells_by_variant.values())
        for variant, cells in sorted(cells_by_variant.items()):
            if cells != union:
                missing = sorted(union - cells, key=str)
                problems.append(
                    f"{variant} misses {len(missing)} cells the other variants have, e.g. "
                    f"{dict(zip(CELL, missing[0], strict=True))}"
                )
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
    if grid is not None:
        actual: dict[tuple, set[int]] = defaultdict(set)
        for record in records:
            actual[tuple(record[key] for key in GRID_KEY)].add(record["episode_seed"])
        missing_cells = sorted(set(grid) - set(actual), key=str)
        extra_cells = sorted(set(actual) - set(grid), key=str)
        if missing_cells:
            problems.append(
                f"{len(missing_cells)} expected cells have no records, e.g. "
                f"{dict(zip(GRID_KEY, missing_cells[0], strict=True))}"
            )
        if extra_cells:
            problems.append(
                f"{len(extra_cells)} cells are not in the protocol, e.g. "
                f"{dict(zip(GRID_KEY, extra_cells[0], strict=True))}"
            )
        for key in sorted(set(grid) & set(actual), key=str):
            want = set(grid[key])
            if actual[key] != want:
                missing = sorted(want - actual[key])[:5]
                extra = sorted(actual[key] - want)[:5]
                problems.append(
                    f"{dict(zip(GRID_KEY, key, strict=True))}: episode seeds differ "
                    f"(missing {missing}, extra {extra})"
                )
    if problems:
        raise ValueError("; ".join(problems[:20]))
    return records
