#!/usr/bin/env bash
# One fixed candidate, two arms. SMOKE=1 is an explicitly separate CPU protocol.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT=$PWD
: "${BC_ACTOR:?Set BC_ACTOR to the BC actor directory}"
: "${PRETRAIN_ACTOR:?Set PRETRAIN_ACTOR to the pretrain actor directory}"
: "${MEASURE_PY:?Set MEASURE_PY to odor_chain/measure.py}"
: "${MEASURE_REPO:?Set MEASURE_REPO to the measure repository root}"
export BC_ACTOR PRETRAIN_ACTOR MEASURE_PY MEASURE_REPO
BC_ACTOR=$(realpath "$BC_ACTOR")
PRETRAIN_ACTOR=$(realpath "$PRETRAIN_ACTOR")
MEASURE_PY=$(realpath "$MEASURE_PY")
MEASURE_REPO=$(realpath "$MEASURE_REPO")
export OUT=${OUT:-runs/t3c} SMOKE=${SMOKE:-0}
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
export PYTHONPATH="$ROOT/python" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=${PY:-$ROOT/.venv/bin/python}
[[ "$PY" = /* ]] || PY=$ROOT/$PY
CHECK=$ROOT/experiments/ppo35/check.py
cpu_py(){ CUDA_VISIBLE_DEVICES='' prlimit --as=6442450944 "$PY" "$@"; }
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run.log"; }
p1= p2= monitor_pid=
cleanup(){
  local pid
  for pid in "$p1" "$p2" "$monitor_pid"; do
    if [[ -n "$pid" ]]; then
      kill -- -"$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
stop(){ if [[ -e "$OUT/STOP" ]]; then log "STOP present; exiting cleanly"; exit 0; fi; }
stop
cpu_py experiments/ppo35/prepare.py | tee -a "$OUT/run.log"
if [[ -n "${MONITOR_SH:-}" ]]; then
  test -f "$MONITOR_SH"
  mkdir -p "$OUT/monitor"
  setsid bash "$MONITOR_SH" "$OUT/monitor" 2 0 > "$OUT/monitor/monitor.log" 2>&1 &
  monitor_pid=$!
fi
DEVICE=cuda
PPO_WORKERS=${PPO_WORKERS:-auto}
TRAIN_PY=("$PY")
if [[ "$SMOKE" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES=''
  DEVICE=cpu
  PPO_WORKERS=0
  TRAIN_PY=(prlimit --as=6442450944 "$PY")
fi
launch_ppo(){
  local arm=$1 var=$2 output=$OUT/ppo_$1
  local extra=()
  if [[ -f "$output/actor/model_card.json" && -f "$output/progress.json" ]] &&
      cpu_py -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["phase"] != "complete")' "$output/progress.json"; then
    cpu_py "$CHECK" ppo "$OUT" "$arm"
    log "skip PPO $arm (validated complete)"
    return
  fi
  [[ ! -f "$output/training_checkpoint.pt" ]] || extra+=(--resume)
  log "start PPO $arm ${extra[*]}"
  setsid "${TRAIN_PY[@]}" -m flytrack_train ppo --config "$OUT/ppo_$arm.yaml" \
    --device "$DEVICE" --workers "$PPO_WORKERS" "${extra[@]}" > "$OUT/ppo_$arm.log" 2>&1 &
  printf -v "$var" '%s' "$!"
}
stop
launch_ppo bc p1
launch_ppo direct p2
if [[ -n "$p1" ]]; then
  wait "$p1"; p1=
  cpu_py "$CHECK" ppo "$OUT" bc
fi
log "PPO bc complete"
stop
if [[ -n "$p2" ]]; then
  wait "$p2"; p2=
  cpu_py "$CHECK" ppo "$OUT" direct
fi
log "PPO direct complete"
stop
if [[ -f "$OUT/episode_eval.json" ]]; then
  cpu_py "$CHECK" eval "$OUT"
  log "skip evaluation (validated complete)"
else
  log "start episode evaluation"
  EPISODES=30 LANES=30 WORKERS=12
  if [[ "$SMOKE" == 1 ]]; then EPISODES=1; LANES=1; WORKERS=0; fi
  export EPISODES LANES WORKERS FIXED='{}'
  "${TRAIN_PY[@]}" scripts/diagnostics/policy_episode_eval.py "$OUT/episode_eval.tmp.json" \
    "$OUT/eval.yaml" "$BC_ACTOR" "$OUT/ppo_bc/actor" "$OUT/ppo_direct/actor" \
    > "$OUT/episode_eval.log" 2>&1
  cpu_py "$CHECK" normalize-eval "$OUT" "$OUT/episode_eval.tmp.json"
  mv "$OUT/episode_eval.tmp.json" "$OUT/episode_eval.json"
fi
log "evaluation complete"
stop
for name in pretrain_vision bc ppo_bc ppo_direct; do
  stop
  if [[ -f "$OUT/dn_$name.json" ]]; then
    cpu_py "$CHECK" dn "$OUT" "$name"
    log "skip DN $name (validated complete)"
    continue
  fi
  actor=$OUT/$name/actor
  [[ "$name" != bc ]] || actor=$BC_ACTOR
  [[ "$name" != pretrain_vision ]] || actor=$PRETRAIN_ACTOR
  log "start DN $name (CPU)"
  (cd "$MEASURE_REPO" && CUDA_VISIBLE_DEVICES='' PYTHONPATH="$MEASURE_REPO/python" \
    prlimit --as=6442450944 "$PY" "$MEASURE_PY" "$OUT/dn_$name.tmp.json" "$actor" \
    > "$OUT/dn_$name.log" 2>&1)
  cpu_py "$CHECK" dn "$OUT" "$name" "$OUT/dn_$name.tmp.json"
  mv "$OUT/dn_$name.tmp.json" "$OUT/dn_$name.json"
  log "DN $name complete"
done
stop
if [[ -f "$OUT/acceptance.json" && -f "$OUT/summary.md" ]]; then
  cpu_py "$CHECK" accept-check "$OUT" > /dev/null
  log "skip acceptance (validated complete)"
else
  cpu_py "$CHECK" accept "$OUT" | tee -a "$OUT/run.log"
fi
test -s "$OUT/acceptance.json"
test -s "$OUT/summary.md"
log "ALL DONE"
