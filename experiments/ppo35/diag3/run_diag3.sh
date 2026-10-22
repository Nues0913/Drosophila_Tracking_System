#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PY=${PY:-$PWD/.venv/bin/python}
export PYTHONPATH="$PWD/python:$PWD/experiments/ppo35:$PWD/experiments/ppo35/diag"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SMOKE=${SMOKE:-0} GPU_BUDGET=${GPU_BUDGET:-1200}
export MEASURE_REPO=${MEASURE_REPO:-../flytrack-odor}
export MEASURE_PY=${MEASURE_PY:-$MEASURE_REPO/experiments/odor_chain/measure.py}
export EPISODES=100 SEED_START=1600100000 LANES=${LANES:-100} WORKERS=${WORKERS:-12}
export DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$SMOKE" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES='' EPISODES=1 LANES=1 WORKERS=0
elif [[ "$SMOKE" != 0 ]]; then
  echo 'SMOKE must be 0 or 1'; exit 1
fi
OUT=runs/c10_diag3
[[ "$SMOKE" != 1 ]] || OUT=runs/c10_diag3_smoke
mkdir -p "$OUT"
exec 9>"$OUT/driver.lock"
flock -n 9 || { echo 'Driver already running'; exit 1; }
exec "$PY" experiments/ppo35/diag3/driver.py "$OUT" < /dev/null
