#!/usr/bin/env bash
set -e
if [[ -z "${MACHINE_CONFIG}" && -z "${GORDO_NAME}" ]]; then
    cp /code/config.yml /tmp/config.yml
elif [[ -z "${MACHINE_CONFIG}" ]]; then
    # $GORDO_NAME is set
    kubectl get gordos ${GORDO_NAME} -o json  | jq ".spec.config" > /tmp/config.yml
else
    echo "$MACHINE_CONFIG" > /tmp/config.yml
fi

workflow_generator --machine-config /tmp/config.yml --output-file /tmp/generated-config.yml
argo lint /tmp/generated-config.yml
if [ "$ARGO_SUBMIT" = true ] ; then
    argo submit /tmp/generated-config.yml
fi
