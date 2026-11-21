# ruff: noqa: E501, B023  (#138 exploratory analysis output lines)
"""#138 A2: read the paired-teaching records of a Stage 1 run (described only, not a criterion).

python a2_monitor.py RUN/stage1/metrics.jsonl

Per held-out record (and the training records averaged between them): for odor and wind, on
all paired steps, on the steps with the odor present and no wall turn, and on those 9-16 steps
after the fork: the slope of the student's turn difference on the teacher's, both RMS, the step
count, and the error reduction against always predicting no difference (#138 review),
1 - sum((ds - dt)^2) / sum(dt^2) = 2 slope - (RMS_student / RMS_teacher)^2: 0 is as good as
ignoring the transform, 1 follows the teacher, below 0 responds but further off.
"""

import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
rows = [r for r in rows if "update" in r]
SUBSETS = (("all", ""), ("present", "_present"), ("present 9-16", "_present_d9_16"))


def summary(get):
    out = []
    for kind in ("odor", "wind"):
        for label, suffix in SUBSETS:
            key = f"pair_{kind}{suffix}"
            slope, rs, rt, n = (
                get(f"{key}_{x}") for x in ("slope", "student_rms", "teacher_rms", "n")
            )
            if None in (slope, rs, rt) or not rt:
                continue
            gain = 2 * slope - (rs / rt) ** 2
            out.append(
                f"{kind:4s} {label:12s} slope {slope:+.4f} rms s/t {rs:.4f}/{rt:.3f} n {n or 0:6.0f} reduction {gain:+.4f}"
            )
    return out


last = 0
for r in rows:
    if "held_out_pair_odor_slope" not in r:
        continue
    span = [x for x in rows if last < x["update"] <= r["update"]]
    print(f"== update {r['update']} (round {r.get('round')}), held-out:")
    for line in summary(lambda k: r.get(f"held_out_{k}")):
        print("   " + line)

    def train_mean(key):
        values = [x[key] for x in span if key in x]
        return sum(values) / len(values) if values else None

    print(
        f"   training mean over updates {last + 1}-{r['update']}: pair_loss {train_mean('pair_loss')}, nll {train_mean('nll')}"
    )
    for line in summary(train_mean):
        print("   train " + line)
    last = r["update"]
