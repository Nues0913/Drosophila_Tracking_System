"""C12 constants, seed isolation and preregistered decision rules; CPU only."""

from c11.c11_common import FORMAL_BLOCKS as C11_BLOCKS
from c11.c11_common import (  # noqa: F401
    ROOT,
    SMOKE_SEED,
    actor_files,
    digest,
    portable,
    read,
    relative,
    write,
)
from protocol100 import number

FORMAL_SEED = 1_600_600_000
SNAP_SEED = 1_600_700_000
FORMAL_BLOCKS = (*C11_BLOCKS, (FORMAL_SEED, 100), (SNAP_SEED, 20))
MAIN_COMMIT = "84830e7d01efeb224aedeb3a9db0f08b981a8c8e"
UNTIL = [0.385, 0.615, 0.846, 1.0]
SEEDS = ("s0", "s1")
GPU_STAGES = ("s0_train", "s1_train", "s0_eval", "s1_eval", "bc_eval")
SNAP_KINDS = ("snap_build", "snap_eval", "snap_dn")
STAGES = (
    *GPU_STAGES,
    "s0_dn",
    "s1_dn",
    "bc_snap_eval",
    "bc_snap_dn",
    *(f"{seed}_{kind}" for seed in SEEDS for kind in SNAP_KINDS),
    "acceptance",
    "summary",
)
RULES = {
    "overall": {
        "both_pass": "C12 一致通過",
        "one_pass": "不一致",
        "neither_pass": "都不通過",
    },
    "early_snapshots": {
        "boundaries": [50_000, 100_000],
        "required_rows": 4,
        "all_abs_omega_max": 0.3,
        "all_tracking_min": 0.3,
        "all_label": "前兩段是繞圈的原因",
        "any_abs_omega_min": 0.6,
        "any_tracking_max": 0.1,
        "any_label": "繞圈和 curriculum 無關",
        "otherwise": "不下結論",
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
        raise ValueError("C12 must use its exact formal seed blocks")
    return list(range(start, start + episodes))


def overall(passed):
    if set(passed) != set(SEEDS) or any(type(value) is not bool for value in passed.values()):
        raise ValueError("Both seed decisions are required")
    count = sum(passed.values())
    return RULES["overall"][("neither_pass", "one_pass", "both_pass")[count]]


def early_interpretation(rows):
    rule = RULES["early_snapshots"]
    # Missing captures never establish a conclusion from fewer than four observations.
    if len(rows) != rule["required_rows"]:
        return rule["otherwise"]
    for row in rows:
        number(row["omega_median"], "omega_median")
        number(row["tracked_fraction"], "tracked_fraction", 0, 1)
    if all(abs(r["omega_median"]) <= 0.3 and r["tracked_fraction"] >= 0.3 for r in rows):
        return rule["all_label"]
    if any(abs(r["omega_median"]) >= 0.6 and r["tracked_fraction"] <= 0.1 for r in rows):
        return rule["any_label"]
    return rule["otherwise"]
