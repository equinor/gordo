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

function get_argo_binary {
    argo_binary="argo"
    if [ -n "$ARGO_VERSION_NUMBER" ]; then
        if [ -z "$ARGO_VERSIONS" ]; then
            echo "ARGO_VERSION env var is empty" 2>&1
            exit 1
        fi
        for number in `echo $ARGO_VERSIONS | jq -rM .[].number` 
        do
            if [ "$number" = "null" ]; then
                number=""
            fi
            if [ "$number" = "$ARGO_VERSION_NUMBER"]; then
                found="true"
                break
            fi
        done
        if [ -z "$found" ]; then
            echo "Unable to find number ($ARGO_VERSION_NUMBER) in '$ARGO_VERSIONS'" 2>&1
            exit 1
        fi
        argo_binary="argo$number"
    fi
    echo $argo_binary
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
