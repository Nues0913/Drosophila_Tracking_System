#!/usr/bin/env bash
# #35 S6 exploration (PM): does the formal DNp09 speed readout start to work once DNp09 gets its real LC9 input
# back? BC (formal recipe + S2 gradient balance 0.3 + S3 desired-command labels, no DN-population rows) on the
# derived graph, VO-late seeds 0 and 1, from the formal pretrain actors transplanted onto that graph.
# Resumable; finished steps are skipped.
# Steps: derive the graph if missing (needs RAW_DIR, default ../drosophila/data/raw) -> transplant the pretrain
# actors (CPU) -> both BC chains in parallel (GPU) -> evaluate baseline (formal actors) and s6 on block
# 1,980,000,000 -> trace s6 normal episodes 0-19 on CPU, one torch thread per seed -> explore/s4/judge.py
# (the S2 rules, unchanged).
# Needs runs/pretrain/VO-late/seed_{0,1}/odor/actor and runs/bc/VO-late/seed_{0,1}/actor (formal-v1).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
test -z "$(git status --short --untracked-files=no)" || { echo "commit the worktree first" >&2; exit 2; }
WT=$PWD; log=runs/s6/run_s6.log; mkdir -p runs/s6
RAW_DIR=${RAW_DIR:-$(dirname "$WT")/drosophila/data/raw}
note() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$log"; }
note "start head $(git rev-parse HEAD); local time $(date +%H:%M)"
D=(docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" -v "$WT:/workspace"
   -w /workspace -e PYTHONPATH=/workspace/python -e HOME=/tmp flytrack-ml:0.1)
T=("${D[@]:0:3}" --cpus 1 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 "${D[@]:3}")
if [ ! -f runs/s6/graph/graph.npz ]; then
  note "derive the graph from $RAW_DIR"
  docker run --rm --network none --cpus 1 --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$RAW_DIR:/raw:ro" -v "$WT:/workspace" \
    -w /workspace flytrack-ml:0.1 python -I explore/s6/derive_graph.py /raw data/subgraphs/olfactory_v2_nokc runs/s6/graph \
    > runs/s6/derive_graph.log 2>&1
fi
sha256sum runs/s6/graph/graph.npz runs/bc/VO-late/seed_{0,1}/actor/model_card.json runs/pretrain/VO-late/seed_{0,1}/*/actor/weights.pt | tee -a "$log"
for seed in 0 1; do
  if [ ! -f runs/s6/init/VO-late/seed_$seed/actor/model_card.json ]; then
    note "transplant seed $seed"
    "${T[@]}" python explore/s6/transplant.py runs/pretrain/VO-late/seed_$seed/odor/actor runs/s6/graph/graph.npz \
      runs/s6/init/VO-late/seed_$seed/actor 2>&1 | tail -1 | tee -a "$log"
  fi
done
bc() {
  local seed=$1 out=runs/s6/bc/VO-late/seed_$1 resume=()
  if [ -f "$out/actor/model_card.json" ]; then note "skip BC seed $seed (done)"; return; fi
  [ -d "$out" ] && resume=(--resume)
  note "BC seed $seed ${resume[*]}"
  "${D[@]}" python -m flytrack_train bc --config explore/s6/configs/bc_s6_VO-late_seed_$seed.yaml --device cuda \
    "${resume[@]}" > runs/s6/bc_seed_$seed.log 2>&1
  note "BC seed $seed done"
}
bc 0 & p0=$!; bc 1 & p1=$!
wait $p0; wait $p1
for name in baseline s6; do
  dest=runs/s6/eval/candidate_s6; [ $name = baseline ] && dest=runs/s6/eval/baseline_VO-late
  if [ -f $dest/episodes.json ]; then note "skip eval $name (done)"; continue; fi
  note "evaluate $name"
  "${D[@]}" python -m flytrack_train evaluate --config explore/s6/configs/eval_$name.yaml --device cuda > runs/s6/eval_$name.log 2>&1
  note "evaluate $name done"
done
trace() {
  local dest=runs/s6/trace/s6/VO-late_seed_$1
  [ -f "$dest/episodes.json" ] && return
  rm -rf "${dest:?}"
  note "trace seed $1"
  "${T[@]}" python scripts/diagnostics/episode_trace.py --artifact runs/s6/bc/VO-late/seed_$1/actor \
    --config explore/s6/configs/eval_s6.yaml --scenario normal --lesion none --device cpu \
    --episodes 0-19 --output "$dest" > runs/s6/trace_$1.log 2>&1
}
trace 0 & p0=$!; trace 1 & p1=$!
wait $p0; wait $p1
python3 -I explore/s4/judge.py runs/s6/eval runs/s6/trace s6 runs/s6/judgment_s6.json | tee -a "$log"
note "done"
