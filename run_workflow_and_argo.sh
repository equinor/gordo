#!/usr/bin/env bash
set -e
if [[ -z "${MACHINE_CONFIG}" ]]; then
    cp /code/config.yml /tmp/config.yml
else
    echo "$MACHINE_CONFIG" > /tmp/config.yml
fi

gordo-components workflow-generator --machine-config /tmp/config.yml --output-file /tmp/generated-config.yml
argo lint /tmp/generated-config.yml
if [ "$ARGO_SUBMIT" = true ] ; then
    argo submit /tmp/generated-config.yml
fi
