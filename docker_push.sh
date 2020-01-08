#!/bin/bash
# Script to push an docker image to a docker registry. The docker image can
# either exist and be provided in the env variable DOCKER_IMAGE, or it will be
# built if DOCKER_IMAGE is empty and DOCKER_FILE is provided
#
# If $DOCKER_USERNAME is set then it will attempt to log in to the
# $DOCKER_REGISTRY using $DOCKER_USERNAME and DOCKER_PASSWORD, otherwise it
# assumes that you are already logged in.
#
# Pushes a docker image with name DOCKER_NAME, and tag tag with the 8 first
# chars of the git commit, and if the current commit is tagged it also pushes
# with those tags. So e.g. if git HEAD is commit "3f6f963" and with tag "0.0.2"
# then two tags are pushed, both "gordo-infrastructure/gordo-deploy:3f6f963" and
# "gordo-infrastructure/gordo-deploy:0.0.2" if DOCKER_NAME is
# "gordo-infrastructure/gordo-deploy".
#
# Expects the following environment variables to be set:
# DOCKER_NAME: Required. Docker name to push to.
# DOCKER_FILE: Semi-Required. Dockerfile to build. Either DOCKER_IMAGE or
#              DOCKER_FILE must be set.
# DOCKER_IMAGE: Semi-Required. The local docker image to push. Either
#               DOCKER_IMAGE or DOCKER_FILE must be set.
# DOCKER_USERNAME: If set then it uses it an the password to log in to the
#                  registry
# DOCKER_PASSWORD: If set then it uses it an the username to log in to the
#                  registry
# DOCKER_REGISTRY: Docker registry to push to. Defaults to
#                  auroradevacr.azurecr.io
# GORDO_PROD_MODE: If false then pushed tags will include a -dev suffix.
#                  Defaults to false
# DOCKER_REPO: The docker repository of concern

if [[ -z "${DOCKER_REGISTRY}" ]]; then
    echo "DOCKER_REGISTRY must be set, exiting"
    exit 1
fi

if [[ -z "${DOCKER_NAME}" ]]; then
    echo "DOCKER_NAME must be set, exiting"
    exit 1
fi

if [[ -z "${DOCKER_USERNAME}" ]]; then
    echo "DOCKER_USERNAME not set: we assume that you are already logged in to the docker registry."
else
    # Logging in to the docker registry, exiting script if it fails
    echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin $DOCKER_REGISTRY || exit 1
fi

export tmp_tag=$(date +%Y-%m-%d)

if [[ -z "${DOCKER_IMAGE}" ]]; then
    if [[ -z "${DOCKER_FILE}" ]]; then
        echo "DOCKER_IMAGE or DOCKER_FILE must be provided, exiting"
        exit 1
    fi
    docker build -t $tmp_tag  -f $DOCKER_FILE .
    export DOCKER_IMAGE=$tmp_tag
fi

# Ensure we're getting the latest version, including any dirty state of the repo
# replacing any '+' development identifier with an underscore for docker compatibility
export version=$(docker run --rm $DOCKER_IMAGE gordo --version | tr + _)


if [[ -z "${GORDO_PROD_MODE}" ]]; then
    echo "Skipping pushing of 'latest' image"
else
    # if we're in prod mode, we'll push the latest image.
    docker tag $DOCKER_IMAGE $DOCKER_REGISTRY/$DOCKER_REPO/$DOCKER_NAME:latest
    docker push $DOCKER_REGISTRY/$DOCKER_REPO/$DOCKER_NAME:latest
fi

docker tag $DOCKER_IMAGE $DOCKER_REGISTRY/$DOCKER_REPO/$DOCKER_NAME:$version
docker push $DOCKER_REGISTRY/$DOCKER_REPO/$DOCKER_NAME:$version

git tag --points-at HEAD | while read -r tag ; do
    docker tag $DOCKER_IMAGE $DOCKER_REGISTRY/$DOCKER_REPO/$DOCKER_NAME:$tag
    docker push $DOCKER_REGISTRY/$DOCKER_REPO/$DOCKER_NAME:$tag
done
