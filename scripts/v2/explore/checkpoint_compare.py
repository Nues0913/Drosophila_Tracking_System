"""#138 exploratory track: score a candidate against the criteria registered in 7044168731.

python checkpoint_compare.py BASE_EVAL NEW_EVAL [V2_EVAL]  (evaluate_v2.py Stage 2 output dirs)

BASE_EVAL is the current recipe's VO2 on the same training seed, NEW_EVAL the candidate's VO2,
V2_EVAL the V2 to compare the candidate with (criterion 2, G-B's measure). Per criterion:
1. wind occlusion success (both motions pooled), new - base >= 0.10, paired episode counts;
2. new VO2 - V2 on the same measure >= 0.10;
3. E1 (occlusion_baseline - ablation) of during_tracking_fraction >= 0.05 for odor_swap or
   wind_rotate_90 (base shown for reference);
4. G-A (curved normal, each world) >= 0.95; normal wander (both worlds) drop <= 0.05; all
   1,600 episodes drop <= 0.02;
5. NaN check on the recorded episode metrics (throughput and both stages are checked separately).
"""

import json
import math
import sys

import numpy as np


def episodes(root):
    rows = [json.loads(line) for line in open(f"{root}/episodes.jsonl")]
    return {(r["world"], r["scenario"], r["motion"], r["episode_seed"]): r for r in rows}


def secondary(root, run):
    rows = [json.loads(line) for line in open(f"{root}/secondary.jsonl")]
    return {(r["world"], r["motion"], r["episode_seed"]): r for r in rows if r["run"] == run}


def success(rows, keep):
    picked = [r for k, r in rows.items() if keep(k)]
    return float(np.mean([r["success"] for r in picked])), len(picked)


def paired(a, b, keep):
    keys = [k for k in a if keep(k)]
    return sum(a[k]["success"] and not b[k]["success"] for k in keys), sum(
        b[k]["success"] and not a[k]["success"] for k in keys
    )


def e1(root):
    base = secondary(root, "occlusion_baseline")
    out = {}
    for label in ("odor_swap_occlusion", "wind_rotate_90_occlusion"):
        ablated = secondary(root, label)
        diffs = [
            base[k]["during_tracking_fraction"] - ablated[k]["during_tracking_fraction"]
            for k in base
            if base[k]["during_tracking_fraction"] is not None
            and ablated[k]["during_tracking_fraction"] is not None
        ]
        out[label] = {"difference": float(np.mean(diffs)) if diffs else None, "n": len(diffs)}
    return out


base, new = episodes(sys.argv[1]), episodes(sys.argv[2])
assert base.keys() == new.keys(), "the two evaluations must cover the same episodes"
wind_occ = lambda k: k[0] == "wind" and k[1] == "occlusion"  # noqa: E731
wander_normal = lambda k: k[1] == "normal" and k[2] == "wander"  # noqa: E731
report = {}

b1, n1 = success(base, wind_occ)
c1, _ = success(new, wind_occ)
only_new, only_base = paired(new, base, wind_occ)
report["1_occlusion"] = {"base": b1, "new": c1, "difference": c1 - b1, "n": n1,
                         "only_new": only_new, "only_base": only_base, "passed": c1 - b1 >= 0.10}

if len(sys.argv) > 3:
    v2 = episodes(sys.argv[3])
    v, _ = success(v2, wind_occ)
    report["2_fusion"] = {"new_vo2": c1, "v2": v, "difference": c1 - v, "passed": c1 - v >= 0.10}

e_new, e_base = e1(sys.argv[2]), e1(sys.argv[1])
report["3_e1"] = {"new": e_new, "base": e_base,
                  "passed": any((r["difference"] or -1) >= 0.05 for r in e_new.values())}

g_a = {w: success(new, lambda k, w=w: k[0] == w and k[1] == "normal" and k[2] == "curved")[0]
       for w in ("v1compat", "wind")}
bw, _ = success(base, wander_normal)
nw, _ = success(new, wander_normal)
ball, nall = success(base, lambda k: True)
call, _ = success(new, lambda k: True)
report["4_no_regression"] = {
    "g_a": g_a,
    "normal_wander": {"base": bw, "new": nw, "drop": bw - nw},
    "all": {"base": ball, "new": call, "drop": ball - call, "n": nall},
    "passed": min(g_a.values()) >= 0.95 and bw - nw <= 0.05 and ball - call <= 0.02,
}

nan = sum(
    1
    for r in new.values()
    for v in r.values()
    if isinstance(v, float) and not math.isfinite(v)
)
report["5_nan_metrics"] = nan

# per cell, for the write-up
cells = {}
for k in sorted({k[:3] for k in base}):
    keep = lambda key, k=k: key[:3] == k  # noqa: E731
    cells["/".join(k)] = {
        "base": success(base, keep)[0],
        "new": success(new, keep)[0],
        "collision_base": float(np.mean([r["collision"] for key, r in base.items() if keep(key)])),
        "collision_new": float(np.mean([r["collision"] for key, r in new.items() if keep(key)])),
    }
report["cells"] = cells
print(json.dumps(report, indent=1))
