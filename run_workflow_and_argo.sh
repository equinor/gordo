#!/usr/bin/env bash

set -e

function_dir=$(dirname $0)
. $function_dir/functions.sh

if [[ -n "${DEBUG_SHOW_WORKFLOW}" ]]; then
  set -x
fi

tmpdir="${TMPDIR:-/tmp}"

config_path="$tmpdir/config.yml"
generated_config_path="$tmpdir/generated-config.yml"

store_config "$config_path"

if [[ -n "${DEBUG_SHOW_WORKFLOW}" ]]; then
  echo "===CONFIG==="
  cat "$config_path"
fi

gordo workflow generate --machine-config "$config_path" --output-file "$generated_config_path"

if [[ -n "${DEBUG_SHOW_WORKFLOW}" ]]; then
  echo "===GENERATED CONFIG==="
  cat "$generated_config_path"
fi

argo_submit "$generated_config_path"
