#!/usr/bin/env bash
# Local equivalent of the required CI checks (plan 4.1, #131): lint and the fast test tier.
# Usage: scripts/v2/check.sh [extra pytest arguments]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-.venv/bin/python}
RUFF="$(dirname "$PY")/ruff"
"$RUFF" check python scripts
"$RUFF" format --check python scripts
parallel=()
if "$PY" -c 'import xdist' 2>/dev/null; then
  parallel=(-n auto)
fi
"$PY" -m pytest -m "not slow and not gpu and not ros and not legacy and not bench" -q "${parallel[@]}" "$@"
