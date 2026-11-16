#!/usr/bin/env bash
# #35 S4 exploration (PM): BC with forward rows over the 134 bilateral DN types that train the core,
# recalibrated every round, + the S2 gradient balance + S3's desired-command labels. VO-late seeds 0 and 1
# from the formal pretrain actors, formal BC budget. Resumable; finished steps are skipped.
# Steps: both BC chains in parallel (GPU) -> evaluate baseline (formal actors) and s4 on block
# 1,930,000,000 -> trace s4 normal episodes 0-19 on CPU, one torch thread per seed -> explore/s4/judge.py.
# Needs runs/pretrain/VO-late/seed_{0,1}/odor/actor and runs/bc/VO-late/seed_{0,1}/actor (formal-v1).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
test -z "$(git status --short --untracked-files=no)" || { echo "commit the worktree first" >&2; exit 2; }
WT=$PWD; log=runs/s4/run_s4.log; mkdir -p runs/s4
note() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$log"; }
note "start head $(git rev-parse HEAD); local time $(date +%H:%M)"
sha256sum runs/bc/VO-late/seed_{0,1}/actor/model_card.json runs/pretrain/VO-late/seed_{0,1}/*/actor/weights.pt | tee -a "$log"
D=(docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" -v "$WT:/workspace"
   -w /workspace -e PYTHONPATH=/workspace/python -e HOME=/tmp flytrack-ml:0.1)
T=("${D[@]:0:3}" --cpus 1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 "${D[@]:3}")
bc() {
  local seed=$1 out=runs/s4/bc/VO-late/seed_$1 resume=()
  if [ -f "$out/actor/model_card.json" ]; then note "skip BC seed $seed (done)"; return; fi
  [ -d "$out" ] && resume=(--resume)
  note "BC seed $seed ${resume[*]}"
  "${D[@]}" python -m flytrack_train bc --config explore/s4/configs/bc_s4_VO-late_seed_$seed.yaml --device cuda \
    "${resume[@]}" > runs/s4/bc_seed_$seed.log 2>&1
  note "BC seed $seed done"
}
bc 0 & p0=$!; bc 1 & p1=$!
wait $p0; wait $p1
for name in baseline s4; do
  dest=runs/s4/eval/candidate_s4; [ $name = baseline ] && dest=runs/s4/eval/baseline_VO-late
  if [ -f $dest/episodes.json ]; then note "skip eval $name (done)"; continue; fi
  note "evaluate $name"
  "${D[@]}" python -m flytrack_train evaluate --config explore/s4/configs/eval_$name.yaml --device cuda > runs/s4/eval_$name.log 2>&1
  note "evaluate $name done"
done
trace() {
  local dest=runs/s4/trace/s4/VO-late_seed_$1
  [ -f "$dest/episodes.json" ] && return
  rm -rf "${dest:?}"
  note "trace seed $1"
  "${T[@]}" python scripts/diagnostics/episode_trace.py --artifact runs/s4/bc/VO-late/seed_$1/actor \
    --config explore/s4/configs/eval_s4.yaml --scenario normal --lesion none --device cpu \
    --episodes 0-19 --output "$dest" > runs/s4/trace_$1.log 2>&1
}
trace 0 & p0=$!; trace 1 & p1=$!
wait $p0; wait $p1
python3 -I explore/s4/judge.py runs/s4/eval runs/s4/trace s4 runs/s4/judgment_s4.json | tee -a "$log"
note "done"
