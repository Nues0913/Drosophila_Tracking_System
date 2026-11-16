#!/usr/bin/env bash
# #35 S5 exploration (PM): evaluate the S5 actors (explore/s5/fit_readout.py; runs/s5/VO-late/seed_{0,1}/actor)
# on block 1,980,000,000, trace normal episodes 0-19 on CPU (one torch thread per seed) and judge with the S2
# rules (explore/s4/judge.py). The baseline evaluation of the block is the S6 one: pass its directory.
# Usage: explore/s5/run_s5_eval.sh BASELINE_EVAL_DIR      (e.g. ../drosophila-s6/runs/s6/eval/baseline_VO-late)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
baseline=$1
WT=$PWD; log=runs/s5/run_s5.log; mkdir -p runs/s5/eval
note() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$log"; }
note "start head $(git rev-parse HEAD)"
sha256sum runs/s5/VO-late/seed_{0,1}/actor/model_card.json "$baseline/episodes.json" | tee -a "$log"
[ -d runs/s5/eval/baseline_VO-late ] || cp -r "$baseline" runs/s5/eval/baseline_VO-late
D=(docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" -v "$WT:/workspace"
   -w /workspace -e PYTHONPATH=/workspace/python -e HOME=/tmp flytrack-ml:0.1)
T=("${D[@]:0:3}" --cpus 1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 "${D[@]:3}")
if [ ! -f runs/s5/eval/candidate_s5/episodes.json ]; then
  note "evaluate s5"
  "${D[@]}" python -m flytrack_train evaluate --config explore/s5/eval_s5.yaml --device cuda > runs/s5/eval_s5.log 2>&1
  note "evaluate s5 done"
fi
trace() {
  local dest=runs/s5/trace/s5/VO-late_seed_$1
  [ -f "$dest/episodes.json" ] && return
  rm -rf "${dest:?}"
  note "trace seed $1"
  "${T[@]}" python scripts/diagnostics/episode_trace.py --artifact runs/s5/VO-late/seed_$1/actor \
    --config explore/s5/eval_s5.yaml --scenario normal --lesion none --device cpu \
    --episodes 0-19 --output "$dest" > runs/s5/trace_$1.log 2>&1
}
trace 0 & p0=$!; trace 1 & p1=$!
wait $p0; wait $p1
python3 -I explore/s4/judge.py runs/s5/eval runs/s5/trace s5 runs/s5/judgment_s5.json | tee -a "$log"
note "done"
