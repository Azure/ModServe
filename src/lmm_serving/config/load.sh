#!/usr/bin/env bash
_src="${BASH_SOURCE[0]:-$0}"
_cfg_dir="$(cd "$(dirname "$_src")" && pwd)"          # src/lmm_serving/config
export PKG_DIR="$(cd "$_cfg_dir/.." && pwd)"          # src/lmm_serving
export REPO_ROOT="$(cd "$_cfg_dir/../../.." && pwd)"  # repo root

while IFS=$'\t' read -r _k _v; do
    [ -z "$_k" ] && continue
    eval "[ -n \"\${$_k+x}\" ]" && continue                 # already set → keep (override wins)
    export "$_k=$(REPO_ROOT="$REPO_ROOT" envsubst <<<"$_v")"  # expand ${REPO_ROOT}/${HOME}
done < <(yq -r 'to_entries | .[] | .key + "\t" + (.value | tostring)' "$_cfg_dir/config.yaml")

unset _k _v _src _cfg_dir
