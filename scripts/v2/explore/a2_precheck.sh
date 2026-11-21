#!/bin/bash
# #138 A2/A3 pre-check (7058912295): CPU, A's graph, fresh initialization, every row in the
# occlusion scenario, one 400-step round (8 rows per world and motion), 100 updates on that
# fixed replay (replay_ratio 8), held-out every 5 updates on a quarter of the rows.
#
#   scripts/v2/explore/a2_precheck.sh OUT PAIR_WEIGHT [extra --override KEY=VALUE ...]
#
# Prints the paired records (a2_monitor.py) and the means over updates 81-100. The A2 numbers
# in 7058912295 are pair weight 1, 10 and 100 at 36bc649 on the 2060 (held-out, weight 10:
# odor student RMS 0.0006, wind 0.038, NLL 0.833).
set -eu
OUT=$1; W=$2; shift 2
PY=${PYTHON:-.venv/bin/python}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONPATH=python
"$PY" -m flytrack_v2.train.dagger --variant VO2 --seed 100 --host 2060 --device cpu --out "$OUT" \
  --override model.graph_path=data/subgraphs/v2_cx_lat/graph.npz pair_weight="$W" \
  "rounds.beta=[1.0]" envs_per_instance=8 rounds.env_steps_per_round=12800 replay_ratio=8 \
  batch_sequences=16 held_out.every_updates=5 held_out.sequences=16 held_out.fraction=0.25 \
  scenarios.normal=0.0 scenarios.occlusion=1.0 scenarios.noise=0.0 scenarios.conflict=0.0 \
  desync_first_episode=false "$@" > "$OUT.log" 2>&1 || echo "dagger exit $? (3 = exported, low speed)"
"$PY" scripts/v2/explore/a2_monitor.py "$OUT/metrics.jsonl" | tail -16
"$PY" - "$OUT/metrics.jsonl" <<'EOF'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
rows = [r for r in rows if "update" in r and 81 <= r["update"] <= 100]
for key in ("pair_odor_present_student_rms", "pair_odor_present_slope",
            "pair_wind_present_student_rms", "pair_wind_present_slope", "nll", "pair_loss"):
    for prefix in ("", "held_out_"):
        values = [r[prefix + key] for r in rows if prefix + key in r]
        if values:
            print(f"{prefix + key:40s} mean over updates 81-100: {sum(values) / len(values):.4f}")
EOF
