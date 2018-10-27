export DOCKER_REGISTRY := auroradevacr.azurecr.io

MODEL_BUILDER_IMG_NAME := gordo-components/gordo-model-builder
MODEL_SERVER_IMG_NAME  := gordo-components/gordo-model-server
MODEL_SERVER_BASE_IMG  := gordo-components/gordo-serverbase

# Create the image capable to building/training a model
model-builder:
	docker build . -f Dockerfile-ModelBuilder -t $(MODEL_BUILDER_IMG_NAME)

# Create the image which serves built models
model-server:
	docker build . -f Dockerfile-ModelServer -t $(MODEL_SERVER_BASE_IMG)
	cd ./gordo_components/runtime && s2i build . -e HTTPS_PROXY=$(HTTPS_PROXY) \
	 $(MODEL_SERVER_BASE_IMG) $(MODEL_SERVER_IMG_NAME)

push-server: model-server
	export DOCKER_NAME=$(MODEL_SERVER_IMG_NAME);\
	export DOCKER_IMAGE=$(MODEL_SERVER_IMG_NAME);\
	./docker_push.sh

push-builder: model-builder
	export DOCKER_NAME=$(MODEL_BUILDER_IMG_NAME);\
	export DOCKER_IMAGE=$(MODEL_BUILDER_IMG_NAME);\
	./docker_push.sh

# Publish images to the currently logged in docker repo
push-dev-images: push-builder push-server

push-prod-images: export GORDO_PROD_MODE:="true"
push-prod-images:	push-builder push-server

images: model-builder model-server

test:
	python setup.py test

all: test images push-dev-images

.PHONY: model-builder model-server push-server push-builder push-dev-images push-prod-images images test all
