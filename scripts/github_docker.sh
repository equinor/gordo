#!/bin/bash

set -e

function show_help {
    echo "Usage: $0 [OPTION]..."
    echo
    echo "Provides the variables such as docker images tags for GitHub workflow."
    echo
    echo "-r    Provide pr-<number> label."
    echo
    exit $1
}

while getopts "hp" opt; do
  case "$opt" in
    h)
        show_help 0
        ;;
    r)
        with_pr="true"
        ;;
  esac
done

GORDO_REPOSITORY=${GORDO_REPOSITORY:-gordo}
GORDO_IMAGE_NAME=${GORDO_IMAGE_NAME:-gordo-base}

DOCKER_DEV_IMAGE=${DOCKER_DEV_REGISTRY}/${GORDO_REPOSITORY}
DOCKER_PROD_IMAGE=${DOCKER_PROD_REGISTRY}/${GORDO_REPOSITORY}

IMAGE_TYPE="dev"
if [[ $GITHUB_REF == refs/tags/* ]]; then
    VERSION=${GITHUB_REF#refs/tags/}
elif [[ $GITHUB_REF == refs/pull/* ]]; then
    number=`cat "$GITHUB_EVENT_PATH" | jq -rM .number`
    if [ -n "$number" ]; then
        if [ -n "$with_pr" ]; then
            VERSION=pr-$number
        fi
        IMAGE_TYPE="pr"
    fi
fi

if [ -z "$VERSION" ]; then
    VERSION=${GITHUB_SHA::8}
fi

RELEASE=""
if [ "$GITHUB_EVENT_NAME" == "release" ]; then
    RELEASE="prerelease"
    IMAGE_TYPE="prod"
    prerelease=`cat "$GITHUB_EVENT_PATH" | jq -rM .release.prerelease`
    if [ "$prerelease" == "false" ]; then
        RELEASE="release"
    fi
fi

function version_tags {
    image=$1
    version=(${2//./ })
    output=$image:$2
    if [ ${#version[@]} -ge 3 ]; then
        output=$output,$image:${version[0]}
        output=$output,$image:${version[0]}.${version[1]}
    fi
    echo $output
}

function set_output_tags {
    image_name=$1
    if [ "$IMAGE_TYPE" == "pr" ]; then
        tags=$DOCKER_DEV_IMAGE/$image_name:$VERSION
    else
        tags=$(version_tags "$DOCKER_DEV_IMAGE/$image_name" "$VERSION")
        if [ -n "$RELEASE" ]; then
            tags=$tags,$DOCKER_DEV_IMAGE/$image_name:latest
        fi
        if [ "$IMAGE_TYPE" == "prod" ]; then
            tags=$tags,$DOCKER_PROD_IMAGE/$image_name:latest,$(version_tags "$DOCKER_PROD_IMAGE/$image_name" "$VERSION")
        fi
        if [ "$RELEASE" == "release" ]; then
            tags=$tags,$DOCKER_DEV_IMAGE/$image_name:stable
            if [ "$IMAGE_TYPE" == "prod" ]; then
                tags=$tags,$DOCKER_PROD_IMAGE/$image_name:stable
            fi
        fi
    fi
    echo $tags
}

function get_push_to_cr {
    if [ -n "$tags_gordo_base" ]; then
        push_to_cr="true"
    fi
    if [ -z "$with_pr" ] && [ "$IMAGE_TYPE" == "pr" ]; then
        push_to_cr=""
    fi
    echo $push_to_cr
}

BASE_IMAGE=$DOCKER_DEV_IMAGE/base
if [ "$IMAGE_TYPE" == "prod" ]; then
    BASE_IMAGE=$DOCKER_PROD_IMAGE/base
fi

echo ::set-output name=version::${VERSION}
echo ::set-output name=release_type::${RELEASE}
echo ::set-output name=image_type::${IMAGE_TYPE}
echo ::set-output name=created::$(date -u +'%Y-%m-%dT%H:%M:%SZ')
echo ::set-output name=base_image::$BASE_IMAGE:$VERSION
echo ::set-output name=tags_gordo_base::$(set_output_tags "$GORDO_IMAGE_NAME")
echo ::set-output name=push_to_cr::$(get_push_to_cr)
