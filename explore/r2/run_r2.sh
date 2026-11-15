#!/usr/bin/env bash
# #35 R2 exploration kit (PM design): train a side-preserving random-wiring control (RS chain with
# model.random_side_preserving: true and one random graph per training seed), seeds 0-4, with the formal
# budget (pretrain vision -> pretrain odor -> BC); evaluate it next to the formal VO-late and the formal RS
# on the fresh block 1,910,000,000; judge with the rules registered on the R2 issue; pack the results.
# Usage, from the root of a clean checkout of branch explore/35-r2-side-rs:  bash explore/r2/run_r2.sh
# R2_STAGE_ONLY=1 stops after downloading the formal actors and checking their cards (a quick pre-flight).
# Needs docker with --gpus, image flytrack-ml:0.1, gh logged in. Resumable: finished steps are skipped.
# Roughly 15 training stages (about 1.5-2 h per seed on a 4060/4070) plus about 1.5 h of evaluation.
set -euo pipefail
cd "$(dirname "$0")/../.."
R=runs/r2
mkdir -p $R
log=$R/run_r2.log
note() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$log"; }
test -z "$(git status --short --untracked-files=no)" || { echo "commit or stash tracked changes first" >&2; exit 2; }
HEAD=$(git rev-parse HEAD)
note "start head $HEAD; $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
D=(docker run --rm --gpus all --network none --user "$(id -u):$(id -g)" -v "$PWD:/workspace"
   -w /workspace -e PYTHONPATH=/workspace/python -e HOME=/tmp flytrack-ml:0.1)

# 1. Formal actors from release formal-v1: VO-late (pair) and RS (descriptive), seeds 0-4.
declare -A DIGEST=(
  [formal_v1_4060_F1.tar.gz]=2f0e2acd6adcf9add500e2b54f116661e1cffffa795daa2e47078d315cea44ac
  [formal_v1_4070_F2.tar.gz]=be05b1c780e9531f601136b6b50dc156d8ca2a612e7721c5770f0597f80fcffa
)
if ! sha256sum -c --quiet explore/r2/expected_cards.sha256 >/dev/null 2>&1; then
  mkdir -p $R/dl
  for name in "${!DIGEST[@]}"; do
    [ -f "$R/dl/$name" ] || gh release download formal-v1 -R Nues0913/Drosophila_Tracking_System -D $R/dl -p "$name"
    echo "${DIGEST[$name]}  $R/dl/$name" | sha256sum -c --quiet
    x=$R/dl/x_${name%.tar.gz}; rm -rf "${x:?}"; mkdir -p "$x"
    tar -xzf "$R/dl/$name" -C "$x" --no-same-owner --no-same-permissions
  done
  for seed in 0 1 2 3 4; do
    pkg=$R/dl/x_formal_v1_4060_F1; case $seed in 1|3) pkg=$R/dl/x_formal_v1_4070_F2;; esac
    for pair in "VO-late stage_pair" "RS stage_formal"; do
      set -- $pair
      src=$(find $pkg -type d -path "*runs/bc/$1/seed_$seed/actor" | head -1)
      dest=$R/$2/$1/seed_$seed
      mkdir -p "$dest" && rm -rf "${dest:?}/actor" && cp -r "$src" "$dest/actor"
    done
  done
fi
sha256sum -c explore/r2/expected_cards.sha256 | tee -a "$log"
if [ "${R2_STAGE_ONLY:-0}" = 1 ]; then note "staging checked (R2_STAGE_ONLY=1); stop"; exit 0; fi

# 2. Train: per seed, pretrain vision -> pretrain odor -> BC (same CLI as the formal chain).
for seed in 0 1 2 3 4; do
  for stage in pretrain_vision pretrain_odor bc; do
    out=$(sed -n 's/^output: //p' explore/r2/configs/${stage}_RS_seed_$seed.yaml)
    if [ -f "$out/actor/model_card.json" ]; then note "skip $stage seed $seed (done)"; continue; fi
    command=pretrain; [ $stage = bc ] && command=bc
    resume=(); [ -d "$out" ] && resume=(--resume)
    note "$stage seed $seed ${resume[*]}"
    "${D[@]}" python -m flytrack_train $command --config explore/r2/configs/${stage}_RS_seed_$seed.yaml \
      --device cuda "${resume[@]}" > $R/${stage}_seed_$seed.log 2>&1
    note "$stage seed $seed done"
  done
  dest=$R/stage_pair/RS/seed_$seed
  mkdir -p "$dest" && rm -rf "${dest:?}/actor" && cp -r $R/bc/RS/seed_$seed/actor "$dest/actor"
done

# 3. Evaluate, 4. judge.
for name in pair formal_rs; do
  if [ -f $R/eval/$name/episodes.json ]; then note "skip eval $name (done)"; continue; fi
  note "evaluate $name"
  "${D[@]}" python -m flytrack_train evaluate --config explore/r2/configs/eval_$name.yaml --device cuda > $R/eval_$name.log 2>&1
  note "evaluate $name done"
done
python3 -I explore/r2/judge.py $R/eval $R/judgment_r2.json | tee -a "$log"

# 5. Pack for release research-35 (actors, evaluations, logs, judgment; no datasets).
P=$R/pack/r2_side_preserving_rs; rm -rf "${P:?}"; mkdir -p $P/actors
cp -r $R/eval $R/judgment_r2.json $R/run_r2.log $R/*_seed_*.log $R/eval_*.log explore/r2 $P/
for seed in 0 1 2 3 4; do mkdir -p $P/actors/seed_$seed && cp -r $R/bc/RS/seed_$seed/actor $P/actors/seed_$seed/; done
echo "head $HEAD" > $P/ENVIRONMENT.txt; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $P/ENVIRONMENT.txt
docker image inspect flytrack-ml:0.1 --format '{{.Id}}' >> $P/ENVIRONMENT.txt
(cd $P && find . -type f ! -name SHA256SUMS -printf '%P\n' | LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS)
tar -C $R/pack -czf $R/pack/r2_side_preserving_rs.tar.gz r2_side_preserving_rs
(cd $R/pack && sha256sum r2_side_preserving_rs.tar.gz > r2_side_preserving_rs.tar.gz.sha256 && cat r2_side_preserving_rs.tar.gz.sha256)
note "done; upload $R/pack/r2_side_preserving_rs.tar.gz and .sha256 to release research-35"
