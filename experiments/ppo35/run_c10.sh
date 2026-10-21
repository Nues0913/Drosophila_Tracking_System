#!/usr/bin/env bash
# CPU smoke and formal CUDA runs must use distinct OUT directories.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT=$PWD
: "${BC_ACTOR:?Set BC_ACTOR to the BC actor directory}"
: "${MEASURE_REPO:?Set MEASURE_REPO to the measure repository root}"
export MEASURE_PY=${MEASURE_PY:-$MEASURE_REPO/experiments/odor_chain/measure.py}
export BC_ACTOR MEASURE_REPO
export OUT=${OUT:-runs/c10} SMOKE=${SMOKE:-0}
[[ "$SMOKE" == 0 || "$SMOKE" == 1 ]] || { echo 'SMOKE must be 0 or 1'; exit 1; }
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
export OUT
exec 9>"$OUT/driver.lock"
flock -n 9 || { echo 'This OUT already has a running driver'; exit 1; }
export PYTHONPATH="$ROOT/python" PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export EPISODES=100 SEED_START=1600100000
export LANES=${LANES:-100} WORKERS=${WORKERS:-12} PPO_WORKERS=${PPO_WORKERS:-auto}
if [[ "$SMOKE" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES='' EPISODES=1 LANES=1 WORKERS=0 PPO_WORKERS=0
fi
PY=${PY:-$ROOT/.venv/bin/python}
[[ "$PY" = /* ]] || PY=$ROOT/$PY
export PY
[[ ! -f "$OUT/STOP" ]] || { echo 'STOP present; exiting cleanly'; exit 0; }
CUDA_VISIBLE_DEVICES='' prlimit --as=6442450944 "$PY" experiments/ppo35/prepare_c10.py
exec "$PY" experiments/ppo35/run_c10.py "$OUT"
