#!/usr/bin/env bash
# CPU smoke with SIGKILL/recovery and a STOP request. Use a fresh OUT.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=experiments/odor_chain
export SEED=${SEED:-1}
export CHAIN_GROUPS=${CHAIN_GROUPS-"G0 G1"}
export OUT=${OUT:-runs/odor_gain/chain_seed${SEED}_resume_smoke}
export SMOKE=1 CUDA_VISIBLE_DEVICES=''
BASE=$OUT/smoke
read -r first_group _ <<< "$CHAIN_GROUPS"
[ ! -e "$OUT" ] || { echo "Use a fresh OUT for smoke_test.sh" >&2; exit 1; }
[[ "$OUT" == runs/* && "$OUT" != *..* ]] || { echo "OUT must be under runs/" >&2; exit 1; }
mkdir -p "$BASE"
LOG=$BASE/smoke_test.log
driver=
cleanup() {
    if [ -n "$driver" ] && kill -0 "$driver" 2>/dev/null; then
        kill -9 -- "-$driver" 2>/dev/null || true
    fi
    rm -f "$OUT/STOP"
}
trap cleanup EXIT
note() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
wait_for() {
    local deadline=$(($(date +%s) + $1)) pid=$2
    shift 2
    until "$@"; do
        kill -0 "$pid" 2>/dev/null || { note "driver exited before: $*"; return 1; }
        [ "$(date +%s)" -lt "$deadline" ] || { note "timeout waiting for: $*"; return 1; }
        sleep 0.1
    done
}
vision_dataset() {
    [ -s "$BASE/$first_group/vision/dataset.json" ] &&
        [ -s "$BASE/$first_group/vision/training_checkpoint.pt" ]
}
vision_checked() { [ -s "$BASE/logs/${first_group}_vision.check.json" ]; }

note "run 1: kill during $first_group vision fit"
setsid bash "$ROOT/run.sh" > "$BASE/run1.log" 2>&1 &
driver=$!
wait_for 1800 "$driver" vision_dataset
[ ! -s "$BASE/$first_group/vision/actor/model_card.json" ] || {
    note "stage finished before the kill; resume untested"; exit 1;
}
note "SIGKILL process group $driver"
kill -9 -- "-$driver"
wait "$driver" 2>/dev/null || true
driver=

note "run 2: resume, then STOP after vision is checked"
setsid bash "$ROOT/run.sh" > "$BASE/run2.log" 2>&1 &
driver=$!
wait_for 3600 "$driver" vision_checked
touch "$OUT/STOP"
status=0
wait "$driver" || status=$?
driver=
note "run 2 exit $status"
[ "$status" = 0 ]
grep -q "start $first_group vision --resume" "$BASE/run2.log"
grep -q 'STOP file' "$BASE/run2.log"
test -s "$BASE/stopped.txt"
test -e "$OUT/STOP"
[ ! -e "$BASE/$first_group/bc" ]
note "resume and STOP verified"

rm "$OUT/STOP"
note "run 3: continue to the end"
bash "$ROOT/run.sh" > "$BASE/run3.log" 2>&1
test -s "$BASE/summary.json"
note "smoke complete"
