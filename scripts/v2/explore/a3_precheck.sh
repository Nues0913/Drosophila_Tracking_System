#!/bin/bash
# #138 A3: extend a2_precheck.sh to full episodes and compare plastic LR scales.
# Usage: a3_precheck.sh OUTROOT [scales...] [--override KEY=VALUE ...]
# Environment: PYTHON, DEVICE, PLASTIC_LABELS, COLLISION_N, OMP_NUM_THREADS, SPEED_TYPES.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 OUTROOT [scales...] [--override KEY=VALUE ...]" >&2
    exit 2
fi
outroot=$(realpath -m -- "$1")
shift
scales=()
while [[ $# -gt 0 && $1 != --override ]]; do
    if [[ ! $1 =~ ^[+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$ ]]; then
        echo "Expected a numeric plastic LR scale, got $1" >&2
        exit 2
    fi
    scales+=("$1")
    shift
done
if [[ ${#scales[@]} -eq 0 ]]; then
    scales=(1 10 30 100)
fi
if [[ $# -gt 0 ]]; then
    shift
fi
extra=("$@")
for override in "${extra[@]}"; do
    case "${override%%=*}" in
        model.plastic_labels|model.plastic_lr_scale)
            echo "Pass labels via PLASTIC_LABELS and scales as positional arguments" >&2
            exit 2
            ;;
    esac
done

labels=${PLASTIC_LABELS:-}
if [[ -n $labels ]]; then
    labels=$(realpath -- "$labels")
fi
cd "$(dirname "$0")/../../.."
labels=${labels:-scripts/v2/explore/a3_plastic_labels.json}
py=${PYTHON:-.venv/bin/python}
device=${DEVICE:-cpu}
speed_types=${SPEED_TYPES:-[DNp09,DNb05]}
collision_n=${COLLISION_N:-20}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} PYTHONPATH=python
mkdir -p -- "$outroot"

run_row() {
    local row=$1
    shift
    local out="$outroot/$row"
    local status=0
    local resume=()
    if [[ -d $out/final ]]; then
        echo "$row: using final/"
    else
        if [[ -f $out/config.json ]]; then
            resume=(--resume)
        fi
        echo "$row: training on $device"
        # Same DAgger command as a2_precheck.sh, with a selectable device and resume.
        # 32 rollout lanes x 1200 ticks, 8 replays / (48 sequences x 64 ticks) = 100 updates.
        "$py" -m flytrack_v2.train.dagger --variant VO2 --seed 100 --host 2060 \
            --device "$device" --out "$out" "${resume[@]}" \
            --override model.graph_path=data/subgraphs/v2_cx_lat/graph.npz \
            "rounds.beta=[1.0]" envs_per_instance=8 rounds.env_steps_per_round=38400 \
            replay_ratio=8 batch_sequences=48 held_out.every_updates=5 held_out.sequences=16 \
            held_out.fraction=0.25 scenarios.normal=0.0 scenarios.occlusion=1.0 \
            scenarios.noise=0.0 scenarios.conflict=0.0 desync_first_episode=false \
            episode_seconds=60 "${extra[@]}" pair_weight=10 "model.speed_types=$speed_types" "$@" \
            < /dev/null >> "$outroot/$row.log" 2>&1 || status=$?
        if [[ $status -eq 3 && -f $out/final/SHA256SUMS ]]; then
            echo "$row: DAgger exported with a failed health check (exit 3)"
        elif [[ $status -ne 0 || ! -d $out/final ]]; then
            echo "$row: training did not export a policy (exit $status). See $outroot/$row.log" >&2
            return "$((status == 0 ? 1 : status))"
        fi
    fi
    if [[ -f $out/collision_trace.txt ]]; then
        echo "$row: using collision_trace.txt"
    else
        "$py" scripts/v2/explore/collision_trace.py "$collision_n" "$row=$out/final" \
            < /dev/null > "$out/collision_trace.tmp" 2>> "$outroot/$row.log"
        mv -- "$out/collision_trace.tmp" "$out/collision_trace.txt"
        echo "$row: wrote collision_trace.txt"
    fi
}

rows=(a2 "${scales[@]/#/scale_}")
label_override=
for row in "${rows[@]}"; do
    if [[ -e $outroot/STOP ]]; then
        echo "STOP before $row"
        break
    fi
    if [[ $row == a2 || -d $outroot/$row/final ]]; then
        run_row "$row"
    else
        if [[ -z $label_override ]]; then
            label_override=$("$py" scripts/v2/explore/a3_labels.py --labels "$labels" \
                --emit-override < /dev/null)
        fi
        run_row "$row" "$label_override" "model.plastic_lr_scale=!!float ${row#scale_}"
    fi
done
"$py" scripts/v2/explore/a3_precheck_summary.py "$outroot" < /dev/null
