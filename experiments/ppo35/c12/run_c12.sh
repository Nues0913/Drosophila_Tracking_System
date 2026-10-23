#!/usr/bin/env bash
set -euo pipefail
exec < /dev/null
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export SMOKE=${SMOKE:-0}
[[ "$SMOKE" == 0 || "$SMOKE" == 1 ]] || { echo 'SMOKE must be 0 or 1'; exit 1; }
EXPECTED=runs/c12
if [[ "$SMOKE" == 1 ]]; then EXPECTED=runs/c12_smoke; fi
export OUT=${OUT:-$EXPECTED}
[[ "$OUT" == "$EXPECTED" ]] || { echo 'OUT differs from C12 mode'; exit 1; }
mkdir -p "$OUT"
exec 9>"$OUT/driver.lock"
flock -n 9 || { echo 'This OUT already has a running driver'; exit 1; }
[[ ! -f "$OUT/STOP" ]] || { echo 'STOP present; exiting cleanly'; exit 0; }
export PYTHONPATH=python:experiments/ppo35:experiments/ppo35/c11
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore:Sparse CSR tensor support is in beta state:UserWarning"
export LANES=${LANES:-100} WORKERS=${WORKERS:-12} PPO_WORKERS=${PPO_WORKERS:-auto}
if [[ "$SMOKE" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES='' LANES=1 WORKERS=0 PPO_WORKERS=0
fi
PY=${PY:-.venv/bin/python}
exec "$PY" -m c12.driver run "$OUT"
