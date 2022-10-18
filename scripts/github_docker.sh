#!/bin/bash

set -e

function show_help {
    echo "Usage: $0 [OPTION]..."
    echo
    echo "Provides the variables such as docker images tags for GitHub workflow."
    echo
    echo "-p    Provide pr-<number> label."
    echo
    exit $1
}

while getopts "hp" opt; do
  case "$opt" in
    h)
        show_help 0
        ;;
    p)
        with_pr="true"
        ;;
  esac
done

DOCKER_DEV_IMAGE=${DOCKER_DEV_REGISTRY}/gordo
DOCKER_PROD_IMAGE=${DOCKER_PROD_REGISTRY}/gordo

IMAGE_TYPE="dev"
if [[ $GITHUB_REF == refs/tags/* ]]; then
    VERSION=${GITHUB_REF#refs/tags/}
elif [[ $GITHUB_REF == refs/pull/* ]]; then
    if [ -n "$with_pr" ]; then
        number=`cat "$GITHUB_EVENT_PATH" | jq -rM .number`
        if [ -n "$number" ]; then
            VERSION=pr-$number
            IMAGE_TYPE="pr"
        fi
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

BASE_IMAGE=$DOCKER_DEV_IMAGE/base
if [ "$IMAGE_TYPE" == "prod" ]; then
    BASE_IMAGE=$DOCKER_PROD_IMAGE/base
fi

echo ::set-output name=version::${VERSION}
echo ::set-output name=release_type::${RELEASE}
echo ::set-output name=image_type::${IMAGE_TYPE}
echo ::set-output name=created::$(date -u +'%Y-%m-%dT%H:%M:%SZ')
echo ::set-output name=base_image::$BASE_IMAGE:$VERSION

tags_gordo_base=$(set_output_tags "gordo-base")
echo ::set-output name=tags_gordo_base::$tags_gordo_base

if [ -n "$tags_gordo_base" ]; then
    non_empty_tags="true"
fi
echo ::set-output name=non_empty_tags::$non_empty_tags
