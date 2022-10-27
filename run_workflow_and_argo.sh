#!/usr/bin/env bash

set -e

if [[ -n "${DEBUG_SHOW_WORKFLOW}" ]]; then
  set -x
fi

functions_dir=$(dirname $0)
. $functions_dir/functions.sh

envs_file="$functions_dir/envs.sh"
if [[ -f "$envs_file" && -r "$envs_file" ]]; then
    . "$envs_file"
fi

tmpdir="${TMPDIR:-/tmp}"

config_path="$tmpdir/config.yml"
generated_config_path="$tmpdir/generated-config.yml"

store_config "$config_path"

if [[ -n "${DEBUG_SHOW_WORKFLOW}" ]]; then
  echo "===CONFIG==="
  cat "$config_path"
fi

argo_binary=$(get_argo_binary)

gordo workflow generate --machine-config "$config_path" --output-file "$generated_config_path"

if [[ -n "${DEBUG_SHOW_WORKFLOW}" ]]; then
  echo "===GENERATED CONFIG==="
  cat "$generated_config_path"
fi

argo_submit "$generated_config_path" "$argo_binary"
