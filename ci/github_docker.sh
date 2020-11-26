#!/bin/bash

set -e

DOCKER_DEV_IMAGE=${DOCKER_DEV_REGISTRY}/gordo
DOCKER_PROD_IMAGE=${DOCKER_PROD_REGISTRY}/gordo

if [[ $GITHUB_REF == refs/tags/* ]]; then
    VERSION=${GITHUB_REF#refs/tags/}
else
    VERSION=${GITHUB_SHA::8}
fi

IMAGE_TYPE="dev"
STABLE=""
if [ "$GITHUB_EVENT_NAME" == "release" ]; then
    IMAGE_TYPE="prod"
    prerelease=`cat "$GITHUB_EVENT_PATH" | jq -rM .release.prerelease`
    if [ "$prerelease" == "false" ]; then
        STABLE="true"
    fi
fi

function output_tags {
    var_name=$1
    image_name=$2
    tags=$DOCKER_DEV_IMAGE/$image_name:$VERSION,$DOCKER_DEV_IMAGE/$image_name:latest
    if [ "$IMAGE_TYPE" == "prod" ]; then
        tags=$tags,$DOCKER_PROD_IMAGE/$image_name:$VERSION,$DOCKER_PROD_IMAGE/$image_name:latest
    fi
    if [ "$STABLE" == "true" ]; then
        tags=$tags,$DOCKER_DEV_IMAGE/$image_name:stable
        if [ "$IMAGE_TYPE" == "prod" ]; then
            tags=$tags,$DOCKER_PROD_IMAGE/$image_name:stable
        fi
    fi
    echo ::set-output name=$var_name::$tags
}

echo ::set-output name=version::${VERSION}
echo ::set-output name=stable::${STABLE}
echo ::set-output name=image_type::${IMAGE_TYPE}
echo ::set-output name=created::$(date -u +'%Y-%m-%dT%H:%M:%SZ')
echo ::set-output name=base_image::gordo_base:$VERSION
output_tags "tags_gordo_client" "gordo-client"
output_tags "tags_gordo_deploy" "gordo-deploy"
output_tags "tags_gordo_model_builder" "gordo-model-builder"
output_tags "tags_gordo_model_server" "gordo-model-server"
