#!/usr/bin/env bash
set -e
if [[ -n "${DEBUG_SHOW_WORKFLOW}" ]]; then
  set -x
fi
if [[ -z "${MACHINE_CONFIG}" && -z "${GORDO_NAME}" ]]; then
    cp /code/config_crd.yml /tmp/config.yml
elif [[ -z "${MACHINE_CONFIG}" ]]; then
    # $GORDO_NAME is set
    kubectl get gordos ${GORDO_NAME} -o json > /tmp/config.yml
else
    echo "$MACHINE_CONFIG" > /tmp/config.yml
fi

if [[ -n "${DEBUG_SHOW_WORKFLOW}" ]]; then
  echo "===CONFIG==="
  cat /tmp/config.yml
fi

gordo workflow generate --machine-config /tmp/config.yml --output-file /tmp/generated-config.yml

if [[ -n "${DEBUG_SHOW_WORKFLOW}" ]]; then
  echo "===GENERATED CONFIG==="
  cat /tmp/generated-config.yml
fi

argo lint /tmp/generated-config.yml
if [ "$ARGO_SUBMIT" = true ] ; then
    argo submit /tmp/generated-config.yml
fi
