#!/usr/bin/env bash
# Usage: bash experiments/odor_chain/pack.sh OUT [user]
set -euo pipefail
cd "$(dirname "$0")/../.."
base=${1:?Usage: pack.sh OUT [user]}
pack_user=${2:-${USER:-$(id -un)}}
[[ "$pack_user" =~ ^[[:alnum:]_.-]+$ ]] || { echo "Invalid user name" >&2; exit 1; }
test -s "$base/configs/manifest.json"
seed=$(awk '$1 == "\"seed\":" { gsub(",", "", $2); print $2; exit }' "$base/configs/manifest.json")
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "No seed in manifest" >&2; exit 1; }
destination=$(dirname "$base")
archive=explore4_seed${seed}_${pack_user}.tar.gz
tar -czf "$destination/$archive" -C "$base" \
    --exclude='datasets' --exclude='dataset' --exclude='*.pt' --exclude='actor' \
    --exclude='*.tar.gz' --exclude='*.sha256' .
(cd "$destination" && sha256sum "$archive" > "$archive.sha256")
printf '%s\n' "$destination/$archive" "$destination/$archive.sha256"
