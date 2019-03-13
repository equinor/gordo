export DOCKER_REGISTRY := auroradevacr.azurecr.io

MODEL_BUILDER_IMG_NAME := gordo-components/gordo-model-builder
MODEL_SERVER_IMG_NAME  := gordo-components/gordo-model-server
WATCHMAN_IMG_NAME := gordo-components/gordo-watchman

# Create the image capable to building/training a model
model-builder:
	docker build . -f Dockerfile-ModelBuilder -t $(MODEL_BUILDER_IMG_NAME)

# Create the image which serves built models
model-server:
	docker build . -f Dockerfile-ModelServer -t $(MODEL_SERVER_IMG_NAME)

# Create the image which reports status of expected model endpoints for the project
watchman:
	docker build . -f Dockerfile-Watchman -t $(WATCHMAN_IMG_NAME)

push-server: model-server
	export DOCKER_NAME=$(MODEL_SERVER_IMG_NAME);\
	export DOCKER_IMAGE=$(MODEL_SERVER_IMG_NAME);\
	./docker_push.sh

push-builder: model-builder
	export DOCKER_NAME=$(MODEL_BUILDER_IMG_NAME);\
	export DOCKER_IMAGE=$(MODEL_BUILDER_IMG_NAME);\
	./docker_push.sh

push-watchman: watchman
	export DOCKER_NAME=$(WATCHMAN_IMG_NAME);\
	export DOCKER_IMAGE=$(WATCHMAN_IMG_NAME);\
	./docker_push.sh

# Publish images to the currently logged in docker repo
push-dev-images: push-builder push-server push-watchman

push-prod-images: export GORDO_PROD_MODE:="true"
push-prod-images: push-builder push-server push-watchman

# Make the python source distribution
sdist:
# Ensure the dist directory is empty/non-existant before sdist
	rm -rf ./dist/
	python setup.py sdist

images: model-builder model-server watchman

test:
	python setup.py test

coverage:
	coverage run --source=gordo_components setup.py testall
	coverage report -m

docs:
	cd ./docs && $(MAKE) html

all: test images push-dev-images

.PHONY: model-builder model-server push-server push-builder push-dev-images push-prod-images images test all docs
