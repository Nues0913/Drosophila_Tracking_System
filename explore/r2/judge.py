"""#35 R2 judgment (rules registered on the R2 issue before any R2 data). Standard library only.

Usage: python3 -I judge.py EVAL_DIR OUT.json
EVAL_DIR holds pair/episodes.json (formal VO-late and the R2 side-preserving RS, seeds 0-4) and
formal_rs/episodes.json (the formal RS, seeds 0-4), all on block 1,910,000,000, curved, 4 scenarios,
100 episodes per seed. Per scenario: success per training seed (mean of its episodes), then
  d = mean over seeds of (VO-late - RS_side), 95% CI from 10,000 seed-paired bootstrap draws (seed 2026).
Primary reading (normal): "the connectome beats side-preserving random wiring" if the CI lower bound > 0;
"not established" otherwise. The other scenarios, RS_side - formal RS and every success rate are reported.
"""

import json
import random
import statistics
import sys
from pathlib import Path

eval_dir, output = Path(sys.argv[1]), Path(sys.argv[2])
SCENARIOS = ("normal", "occlusion", "noise", "conflict")


def per_seed(rows, variant, scenario):
    seeds = {}
    for r in rows:
        if r["variant"] == variant and r["scenario"] == scenario:
            seeds.setdefault(int(r["seed"]), []).append(float(r["success"]))
    if sorted(seeds) != [0, 1, 2, 3, 4] or any(len(v) != 100 for v in seeds.values()):
        sys.exit(f"{variant} {scenario}: expected seeds 0-4 with 100 episodes each")
    return {s: statistics.fmean(v) for s, v in seeds.items()}


def paired(a, b):
    diffs = [a[s] - b[s] for s in sorted(a)]
    rng = random.Random(2026)
    draws = sorted(statistics.fmean(rng.choice(diffs) for _ in diffs) for _ in range(10000))
    return statistics.fmean(diffs), draws[249], draws[9749], diffs


pair = json.loads((eval_dir / "pair" / "episodes.json").read_text())
formal_rs = json.loads((eval_dir / "formal_rs" / "episodes.json").read_text())
result = {"rule": __doc__, "scenarios": {}}
for scenario in SCENARIOS:
    vol = per_seed(pair, "VO-late", scenario)
    rs_side = per_seed(pair, "RS", scenario)
    rs_formal = per_seed(formal_rs, "RS", scenario)
    d, lo, hi, diffs = paired(vol, rs_side)
    d2, lo2, hi2, _ = paired(rs_side, rs_formal)
    result["scenarios"][scenario] = {
        "success": {"VO-late": statistics.fmean(vol.values()), "RS_side": statistics.fmean(rs_side.values()),
                    "RS_formal": statistics.fmean(rs_formal.values())},
        "VO-late_minus_RS_side": {"mean": d, "ci95": [lo, hi], "per_seed": diffs},
        "RS_side_minus_RS_formal": {"mean": d2, "ci95": [lo2, hi2]},
    }
    print(f"{scenario:9s} VO-late {statistics.fmean(vol.values()):.3f}  RS_side {statistics.fmean(rs_side.values()):.3f}  "
          f"RS_formal {statistics.fmean(rs_formal.values()):.3f} | VO-late - RS_side {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
          f"RS_side - RS_formal {d2:+.3f} [{lo2:+.3f}, {hi2:+.3f}]")
normal = result["scenarios"]["normal"]["VO-late_minus_RS_side"]
result["primary"] = "established" if normal["ci95"][0] > 0 else "not established"
output.write_text(json.dumps(result, indent=2) + "\n")
print("R2 primary (normal, VO-late > RS_side):", result["primary"])
