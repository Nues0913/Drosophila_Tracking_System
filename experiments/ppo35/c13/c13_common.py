"""Preregistered C13 constants, seed isolation and independent snapshot rules."""

from c12.c12_common import FORMAL_BLOCKS as C12_BLOCKS
from c12.c12_common import (  # noqa: F401
    GPU_STAGES,
    ROOT,
    SEEDS,
    SMOKE_SEED,
    SNAP_KINDS,
    STAGES,
    actor_files,
    digest,
    portable,
    read,
    relative,
    write,
)
from protocol100 import number

LAMBDA = 0.1723
FORMAL_SEED = 1_600_800_000
SNAP_SEED = 1_600_900_000
FORMAL_BLOCKS = (*C12_BLOCKS, (FORMAL_SEED, 100), (SNAP_SEED, 20))
BOUNDARIES = tuple(range(50_000, 500_001, 50_000))
RULES = {
    "overall": {
        "both_pass": "C13 一致通過",
        "one_pass": "不一致",
        "neither_pass": "都不通過",
    },
    "snapshots": {
        "early_dn": {
            "label": "錨定保住了早期的 DN 結構",
            "step": 51_200,
            "both_seeds_slope_min": 0.01,
        },
        "circling": {
            "label": "錨定不足以防止繞圈",
            "boundaries": list(BOUNDARIES),
            "any_abs_omega_min": 0.6,
            "any_tracking_max": 0.1,
        },
        "missing": "不下結論",
        "independent": True,
        "affects_acceptance": False,
    },
}


def seeds(start, episodes, smoke):
    if type(start) is not int or type(episodes) is not int or start < 0 or episodes < 1:
        raise ValueError("Invalid evaluation seed range")
    if smoke:
        if any(
            start < first + count and start + episodes > first for first, count in FORMAL_BLOCKS
        ):
            raise ValueError("Smoke must not use any formal seed block")
    elif (start, episodes) not in ((FORMAL_SEED, 100), (SNAP_SEED, 20)):
        raise ValueError("C13 must use its exact formal seed blocks")
    return list(range(start, start + episodes))


def overall(passed):
    if set(passed) != set(SEEDS) or any(type(value) is not bool for value in passed.values()):
        raise ValueError("Both seed decisions are required")
    return RULES["overall"][("neither_pass", "one_pass", "both_pass")[sum(passed.values())]]


def snapshot_interpretation(rows):
    early, circling = [], []
    for row in rows:
        if row["seed"] not in SEEDS:
            raise ValueError("Unknown snapshot seed")
        if row.get("boundary") == 50_000 and row["step"] == RULES["snapshots"]["early_dn"]["step"]:
            if row.get("DNa01_slope_median") is not None:
                number(row["DNa01_slope_median"], "DNa01_slope_median")
                early.append(row)
        if row.get("boundary") in BOUNDARIES:
            if row.get("omega_median") is not None and row.get("tracked_fraction") is not None:
                number(row["omega_median"], "omega_median")
                number(row["tracked_fraction"], "tracked_fraction", 0, 1)
                circling.append(row)
    early_complete = len(early) == 2 and {r["seed"] for r in early} == set(SEEDS)
    expected = {(s, b) for s in SEEDS for b in BOUNDARIES}
    complete = (
        len(circling) == len(expected)
        and {(r["seed"], r["boundary"]) for r in circling} == expected
    )
    hit = any(abs(r["omega_median"]) >= 0.6 and r["tracked_fraction"] <= 0.1 for r in circling)
    return {
        RULES["snapshots"]["early_dn"]["label"]: (
            all(r["DNa01_slope_median"] >= 0.01 for r in early) if early_complete else "不下結論"
        ),
        RULES["snapshots"]["circling"]["label"]: True if hit else False if complete else "不下結論",
    }
