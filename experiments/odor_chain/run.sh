#!/usr/bin/env bash
# VO chain per group: pretrain vision -> pretrain odor -> BC, with
# measurements after every stage. Groups run one after another.
#
#   SEED=1 CHAIN_GROUPS="G0 G1 G2" bash experiments/odor_chain/run.sh
#   SMOKE=1 SEED=1 CHAIN_GROUPS="G0 G1" bash experiments/odor_chain/run.sh
#
# Stop between stages: touch "$OUT/STOP" (also stops "$OUT/smoke"). Remove
# it and rerun the identical command to continue.
# Resume: rerun the same command. Finished stages/measurements are skipped; a
# training stage with training_checkpoint.pt but no actor/model_card.json gets
# --resume (source and config hashes must match); one killed before its first
# checkpoint is cleared and restarted. Any failure, missing output or nonfinite
# value stops the driver at once with a nonzero exit.
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=python
export PYTHONDONTWRITEBYTECODE=1
ROOT=experiments/odor_chain
export SEED=${SEED:-0}
export CHAIN_GROUPS=${CHAIN_GROUPS-"G0 G1 G2 G3"}
export OUT=${OUT:-runs/odor_gain/chain_seed${SEED}}
export SMOKE=${SMOKE:-0}
export WORKERS=${WORKERS:-auto}
read -r -a groups <<< "$CHAIN_GROUPS"

if [ "$SMOKE" = 1 ]; then
    export CUDA_VISIBLE_DEVICES=''
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
    export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
    BASE=$OUT/smoke DEVICE=cpu MEM=6G
    EVAL_EPISODES=2 DN_EPISODES=1 DN_STEPS=16
else
    BASE=$OUT DEVICE=cuda MEM=10G
    EVAL_EPISODES=30 DN_EPISODES=24 DN_STEPS=200
fi

py() {
    if [ "$SMOKE" = 1 ]; then
        prlimit --as=6442450944 .venv/bin/python "$@"
    else
        systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 \
            .venv/bin/python "$@"
    fi
}

# Validate paths, seed, groups and the immutable run manifest before any cleanup.
py "$ROOT/prepare.py"
LOGS=$BASE/logs
mkdir -p "$LOGS" "$BASE/measure"
trap 'status=$?; printf "%s\n" "$status" > "$BASE/exit_code.txt"' EXIT
trap 'log "FAILED (status $?) at run.sh line $LINENO; see $LOGS"' ERR
rm -f "$BASE/exit_code.txt"
date -Is > "$BASE/started_at.txt"

log() { printf '%s %s\n' "$(date -Is)" "$*"; }
STOP_FILE=$OUT/STOP
stop_point() {  # description of the step about to start
    if [ -e "$STOP_FILE" ]; then
        log "STOP file $STOP_FILE present; stopping before $*"
        printf '%s stopped before %s\n' "$(date -Is)" "$*" >> "$BASE/stopped.txt"
        exit 0
    fi
}
complete() { [ -s "$1/actor/model_card.json" ] && grep -q '"phase": "complete"' "$1/progress.json"; }

stop_point "initial measurement"

train() {  # group stage
    local config=$BASE/configs/$1_$2.yaml out=$BASE/$1/$2 command=pretrain resume=()
    [ "$2" = bc ] && command=bc
    if complete "$out"; then
        log "skip $1 $2"
    else
        stop_point "training $1 $2"
        if [ -s "$out/training_checkpoint.pt" ]; then
            resume=(--resume)
        else
            rm -rf "${out:?}"  # nothing committed yet: start clean
        fi
        log "start $1 $2 ${resume[*]}"
        py -m flytrack_train "$command" --config "$config" --device "$DEVICE" \
            --workers "$WORKERS" "${resume[@]}" >> "$LOGS/$1_$2.log" 2>&1
        log "done $1 $2"
    fi
    py "$ROOT/check.py" "$out" "$config" > "$LOGS/$1_$2.check.json"
}

measure() {  # group stage
    local out=$BASE/measure/$1_$2.json
    [ -s "$out" ] && return 0
    stop_point "measuring $1 $2"
    if [ "$2" = init ]; then
        py "$ROOT/measure.py" "$out.tmp" --init "$BASE/configs/$1_vision.yaml"
    else
        py "$ROOT/measure.py" "$out.tmp" "$BASE/$1/$2/actor"
    fi >> "$LOGS/$1_measure.log" 2>&1
    mv "$out.tmp" "$out"
}

validate() {  # group stage; a decoder below threshold is a result, not an error
    local out=$BASE/$1/$2 status=0
    [ -s "$out/validation/metrics.json" ] && return 0
    stop_point "validate-pretrain $1 $2"
    rm -rf "$out/validation"
    py -m flytrack_train validate-pretrain --config "$BASE/configs/$1_$2.yaml" \
        --artifact "$out/actor" --checkpoint "$out/pretraining_checkpoint.pt" \
        --output "$out/validation" --device "$DEVICE" --workers "$WORKERS" \
        >> "$LOGS/$1_$2.validate.log" 2>&1 || status=$?
    if [ ! -s "$out/validation/metrics.json" ]; then
        log "validate-pretrain $1 $2 failed with status $status"
        exit 1
    fi
    py -c 'import json, math, sys
m = json.load(open(sys.argv[1]))
assert m["bearing_mae_deg"] is not None and math.isfinite(m["bearing_mae_deg"]), m
print(sys.argv[1], "passed", m["passed"], "mae", round(m["bearing_mae_deg"], 2))' \
        "$out/validation/metrics.json"
}

bc_diagnostics() {  # group
    local out=$BASE/$1/bc
    if [ ! -s "$out/dn_health/metrics.json" ]; then
        stop_point "dn_health $1"
        py scripts/dn_health.py --artifact "$out/actor" --config "$BASE/configs/$1_bc.yaml" \
            --stage bc --episodes "$DN_EPISODES" --steps "$DN_STEPS" \
            --output "$out/dn_health/metrics.json" >> "$LOGS/$1_dn_health.log" 2>&1
    fi
    py -c 'import json, sys
m = json.load(open(sys.argv[1]))
print(sys.argv[1], "passed", m["passed"], m["checks"])' "$out/dn_health/metrics.json"
    for lesion in none odor; do
        [ -s "$out/episodes/$lesion.json" ] && continue
        stop_point "episode eval $1 lesion=$lesion"
        mkdir -p "$out/episodes"
        EPISODES=$EVAL_EPISODES WORKERS=$WORKERS py "$ROOT/episode_eval.py" \
            "$out/episodes/$lesion.json.tmp" "$BASE/configs/$1_bc.yaml" "$out/actor" "$lesion" \
            >> "$LOGS/$1_episodes.log" 2>&1
        mv "$out/episodes/$lesion.json.tmp" "$out/episodes/$lesion.json"
    done
}

for group in "${groups[@]}"; do
    measure "$group" init
    for stage in vision odor; do
        train "$group" "$stage"
        measure "$group" "$stage"
        validate "$group" "$stage"
    done
    train "$group" bc
    measure "$group" bc
    bc_diagnostics "$group"
    log "group $group complete"
done

stop_point "summary"
py "$ROOT/summarize.py" "$BASE" > "$LOGS/summarize.log" 2>&1
test -s "$BASE/summary.json"
date -Is > "$BASE/completed_at.txt"
log "all done"
