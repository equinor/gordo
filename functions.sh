#!/usr/bin/env bash

function store_config {
    config_path="$1"
    if [[ -z "${MACHINE_CONFIG}" && -z "${GORDO_NAME}" ]]; then
        cp /code/config_crd.yml "$config_path"
    elif [[ -z "${MACHINE_CONFIG}" ]]; then
        # $GORDO_NAME is set
        kubectl get gordos ${GORDO_NAME} -o json > "$config_path"
    else
        echo "$MACHINE_CONFIG" > "$config_path"
    fi
}

function argo_submit { 
    generated_config_path="$1"
    argo lint "$generated_config_path"
    if [ "$ARGO_SUBMIT" = true ] ; then
        if [[ -n "$ARGO_SERVICE_ACCOUNT" ]]; then
            argo submit --serviceaccount "$ARGO_SERVICE_ACCOUNT" "$generated_config_path"
        else
            argo submit "$generated_config_path"
        fi
    fi
}
