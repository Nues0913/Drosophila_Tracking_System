#!/usr/bin/env bash
# #35 S3 exploration kit (PM design, run on the 4060): BC with the teacher's desired command as the label
# (env.teacher_accel_clip: false), VO-late seeds 0 and 1, from the formal-v1 pretrain actors with the formal
# BC budget; evaluate it and the formal-v1 baseline on the fresh block 1,900,000,000; trace normal
# episodes 0-19; judge with the rules registered on #35; pack the results.
# Usage, from the root of a clean checkout of branch explore/35-s3-desired-labels:  bash explore/s3/run_s3.sh
# S3_STAGE_ONLY=1 stops after downloading the actors and checking their cards (a quick pre-flight).
# Needs: docker with --gpus, image flytrack-ml:0.1, gh logged in. Resumable: finished steps are skipped.
# About 1-2 h per BC chain and about 1 h for the evaluations on a 4060.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=runs/s3
mkdir -p $R
log=$R/run_s3.log
note() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$log"; }
test -z "$(git status --short --untracked-files=no)" || { echo "commit or stash tracked changes first" >&2; exit 2; }
HEAD=$(git rev-parse HEAD)
note "start head $HEAD; $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
D=(docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" -v "$PWD:/workspace"
   -w /workspace -e PYTHONPATH=/workspace/python -e HOME=/tmp flytrack-ml:0.1)

# 1. Actors from release formal-v1 (pretrain = BC init; formal BC = baseline), each tarball checked
#    against its published SHA-256 and extracted into its own empty directory.
declare -A DIGEST=(
  [formal_v1_pretrain_seed0.tar.gz]=017bde9b7e101444736c1e1f36cdc6f9ff74e78af71745f68c0e1c1716ece83d
  [formal_v1_pretrain_seed1.tar.gz]=c6f8c6d6ad3899cdde1e693b62de6f0847c15e11ec0410dd50ca50f35c5c3d07
  [formal_v1_4060_F1.tar.gz]=2f0e2acd6adcf9add500e2b54f116661e1cffffa795daa2e47078d315cea44ac
  [formal_v1_4070_F2.tar.gz]=be05b1c780e9531f601136b6b50dc156d8ca2a612e7721c5770f0597f80fcffa
)
if ! sha256sum -c --quiet explore/s3/expected_cards.sha256 2>/dev/null; then
  mkdir -p $R/dl
  for name in "${!DIGEST[@]}"; do
    [ -f "$R/dl/$name" ] || gh release download formal-v1 -R Nues0913/Drosophila_Tracking_System -D $R/dl -p "$name"
    echo "${DIGEST[$name]}  $R/dl/$name" | sha256sum -c --quiet
    x=$R/dl/x_${name%.tar.gz}; rm -rf "${x:?}"; mkdir -p "$x"
    tar -xzf "$R/dl/$name" -C "$x" --no-same-owner --no-same-permissions
  done
  for seed in 0 1; do
    pre=$(find $R/dl/x_formal_v1_pretrain_seed$seed -type d -path "*runs/pretrain/VO-late/seed_$seed" | head -1)
    mkdir -p runs/pretrain/VO-late && rm -rf "runs/pretrain/VO-late/seed_${seed:?}" && cp -r "$pre" runs/pretrain/VO-late/seed_$seed
    pkg=$R/dl/x_formal_v1_4060_F1; [ $seed = 1 ] && pkg=$R/dl/x_formal_v1_4070_F2
    bc=$(find $pkg -type d -path "*runs/bc/VO-late/seed_$seed/actor" | head -1)
    mkdir -p runs/bc/VO-late/seed_$seed && rm -rf "runs/bc/VO-late/seed_${seed:?}/actor" && cp -r "$bc" runs/bc/VO-late/seed_$seed/actor
  done
fi
sha256sum -c explore/s3/expected_cards.sha256 | tee -a "$log"
if [ "${S3_STAGE_ONLY:-0}" = 1 ]; then note "staging checked (S3_STAGE_ONLY=1); stop"; exit 0; fi

# 2. BC, 3. evaluations, 4. traces, 5. judgment.
for seed in 0 1; do
  out=$R/bc/VO-late/seed_$seed
  if [ -f "$out/actor/model_card.json" ]; then note "skip BC seed $seed (done)"; continue; fi
  resume=(); [ -d "$out" ] && resume=(--resume)
  note "BC seed $seed ${resume[*]}"
  "${D[@]}" python -m flytrack_train bc --config explore/s3/configs/bc_s3_VO-late_seed_$seed.yaml --device cuda \
    "${resume[@]}" > $R/bc_seed_$seed.log 2>&1
  note "BC seed $seed done"
done
for name in baseline s3; do
  dest=$R/eval/candidate_s3; [ $name = baseline ] && dest=$R/eval/baseline_VO-late
  if [ -f $dest/episodes.json ]; then note "skip eval $name (done)"; continue; fi
  note "evaluate $name"
  "${D[@]}" python -m flytrack_train evaluate --config explore/s3/configs/eval_$name.yaml --device cuda > $R/eval_$name.log 2>&1
  note "evaluate $name done"
done
for seed in 0 1; do
  dest=$R/trace/s3/VO-late_seed_$seed
  [ -f "$dest/episodes.json" ] && continue
  note "trace seed $seed"
  "${D[@]}" python scripts/diagnostics/episode_trace.py --artifact $R/bc/VO-late/seed_$seed/actor \
    --config explore/s3/configs/eval_s3.yaml --scenario normal --lesion none --device cpu \
    --episodes 0-19 --output "$dest" > $R/trace_$seed.log 2>&1
done
python3 -I explore/s3/judge.py $R/eval $R/trace s3 $R/judgment_s3.json | tee -a "$log"

# 6. Pack for release research-35 (actors, evaluations, traces, logs, judgment; no datasets).
P=$R/pack/s3_desired_labels_4060; rm -rf "${P:?}"; mkdir -p $P
cp -r $R/eval $R/trace $R/judgment_s3.json $R/run_s3.log $R/bc_seed_*.log $R/eval_*.log explore/s3 $P/
for seed in 0 1; do mkdir -p $P/actors/seed_$seed && cp -r $R/bc/VO-late/seed_$seed/actor $P/actors/seed_$seed/; done
echo "head $HEAD" > $P/ENVIRONMENT.txt; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $P/ENVIRONMENT.txt
docker image inspect flytrack-ml:0.1 --format '{{.Id}}' >> $P/ENVIRONMENT.txt
(cd $P && find . -type f ! -name SHA256SUMS -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS)
tar -C $R/pack -czf $R/pack/s3_desired_labels_4060.tar.gz s3_desired_labels_4060
(cd $R/pack && sha256sum s3_desired_labels_4060.tar.gz > s3_desired_labels_4060.tar.gz.sha256 && cat s3_desired_labels_4060.tar.gz.sha256)
note "done; upload $R/pack/s3_desired_labels_4060.tar.gz and .sha256 to release research-35"
