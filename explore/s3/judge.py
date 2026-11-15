"""#35 S3 exploration judgment: the S2 rules (#35 6901989849), registered for S3 on #35 before any S3 data.
Standard library only.

Usage: python3 -I judge.py EVAL_DIR TRACE_ROOT TAG OUT.json
EVAL_DIR holds baseline_VO-late/ and candidate_TAG/ evaluations (episodes.json) on block 1,900,000,000
(curved, 4 scenarios, seeds 0 and 1, 100 episodes per cell). TRACE_ROOT/TAG/VO-late_seed_{k}/ holds
episode_trace.py outputs of the candidate, normal episodes 0-19 of the same block. Per seed:
  1. normal success >= baseline + 0.10            (above the R0 retrain spread 0.10)
  2. occlusion success >= baseline - 0.05
  3. speed is modulated: median within-episode std of applied v after 3 s >= 0.02 m/s, and the pooled
     correlation of applied v with the teacher's desired speed
     v* = min(0.5, max(0, 0.7 (distance - 1)) * max(0, cos bearing)) is >= 0.3
  4. normal mean bearing error <= baseline + 5 degrees
  5. normal collision rate <= baseline + 0.05     (the D3 failure: faster everywhere, more collisions)
The candidate TAG passes only if both seeds meet all five.
"""

import gzip
import json
import math
import statistics
import sys
from pathlib import Path

eval_dir, trace_root, tag, output = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])


def cell(rows, seed, scenario):
    chosen = [r for r in rows if r["variant"] == "VO-late" and int(r["seed"]) == seed and r["scenario"] == scenario]
    if len(chosen) != 100:
        sys.exit(f"seed {seed} {scenario}: {len(chosen)} episodes, expected 100")
    return chosen


def mean(rows, key):
    return statistics.fmean(float(r[key]) for r in rows)


def speed(seed):
    stds, v_all, d_all = [], [], []
    paths = sorted((trace_root / tag / f"VO-late_seed_{seed}").glob("trace/*.jsonl.gz"))
    for path in paths:
        rows = [json.loads(line) for line in gzip.open(path, "rt")]
        late = [r for r in rows if r["time_s"] >= 3.0]
        v = [r["applied_action"][0] for r in late]
        if len(v) > 1:
            stds.append(statistics.pstdev(v))
        v_all += v
        d_all += [min(0.5, max(0.0, 0.7 * (r["target_distance_m"] - 1.0)) * max(0.0, math.cos(r["target_bearing_rad"])))
                  for r in late]
    if len(stds) != 20:
        sys.exit(f"seed {seed}: {len(stds)} traced normal episodes, expected 20")
    corr = statistics.correlation(v_all, d_all) if statistics.pstdev(v_all) > 0 else 0.0
    return statistics.median(stds), corr


candidate = json.loads((eval_dir / f"candidate_{tag}" / "episodes.json").read_text())
baseline = json.loads((eval_dir / "baseline_VO-late" / "episodes.json").read_text())
result = {"rule": __doc__, "tag": tag, "cells": [], "passed": None}
for seed in (0, 1):
    cn, bn = cell(candidate, seed, "normal"), cell(baseline, seed, "normal")
    co, bo = cell(candidate, seed, "occlusion"), cell(baseline, seed, "occlusion")
    v_std, v_corr = speed(seed)
    row = {
        "seed": seed,
        "normal_success": [mean(bn, "success"), mean(cn, "success")],
        "occlusion_success": [mean(bo, "success"), mean(co, "success")],
        "normal_bearing_deg": [mean(bn, "mean_bearing_error_deg"), mean(cn, "mean_bearing_error_deg")],
        "normal_collision": [mean(bn, "collision_rate"), mean(cn, "collision_rate")],
        "speed_std_median": v_std,
        "speed_corr_desired": v_corr,
    }
    row["criteria"] = {
        "1_normal_plus_0.10": row["normal_success"][1] >= row["normal_success"][0] + 0.10,
        "2_occlusion_minus_0.05": row["occlusion_success"][1] >= row["occlusion_success"][0] - 0.05,
        "3_speed_modulated": v_std >= 0.02 and v_corr >= 0.3,
        "4_bearing_plus_5": row["normal_bearing_deg"][1] <= row["normal_bearing_deg"][0] + 5,
        "5_collision_plus_0.05": row["normal_collision"][1] <= row["normal_collision"][0] + 0.05,
    }
    row["all"] = all(row["criteria"].values())
    result["cells"].append(row)
    print(f"{tag} seed {seed}: normal {row['normal_success'][0]:.2f}->{row['normal_success'][1]:.2f} "
          f"occlusion {row['occlusion_success'][0]:.2f}->{row['occlusion_success'][1]:.2f} "
          f"bearing {row['normal_bearing_deg'][0]:.1f}->{row['normal_bearing_deg'][1]:.1f} "
          f"collision {row['normal_collision'][0]:.2f}->{row['normal_collision'][1]:.2f} "
          f"v std {v_std:.4f} corr(v, v*) {v_corr:+.2f}  {row['criteria']}")
result["passed"] = all(r["all"] for r in result["cells"])
output.write_text(json.dumps(result, indent=2) + "\n")
print(f"S3 {tag} PASSED" if result["passed"] else f"S3 {tag} NOT PASSED")
