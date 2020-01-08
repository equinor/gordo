
MODEL_BUILDER_IMG_NAME := gordo-model-builder
MODEL_SERVER_IMG_NAME  := gordo-model-server
CLIENT_IMG_NAME := gordo-client
WORKFLOW_GENERATOR_IMG_NAME := gordo-deploy

# Create the image capable of rendering argo workflow generator
workflow-generator:
	docker build . -f Dockerfile-GordoDeploy -t $(WORKFLOW_GENERATOR_IMG_NAME)

# Publish image to the currently logged in docker repo
push-workflow-generator: workflow-generator
	export DOCKER_NAME=$(WORKFLOW_GENERATOR_IMG_NAME);\
	export DOCKER_IMAGE=$(WORKFLOW_GENERATOR_IMG_NAME);\
	./docker_push.sh

# Create the image capable to building/training a model
model-builder:
	docker build . -f Dockerfile-ModelBuilder -t $(MODEL_BUILDER_IMG_NAME)

# Create the image which serves built models
model-server:
	docker build . -f Dockerfile-ModelServer -t $(MODEL_SERVER_IMG_NAME)

client:
	docker build . -f Dockerfile-Client -t $(CLIENT_IMG_NAME)

push-server: model-server
	export DOCKER_NAME=$(MODEL_SERVER_IMG_NAME);\
	export DOCKER_IMAGE=$(MODEL_SERVER_IMG_NAME);\
	bash ./docker_push.sh

push-builder: model-builder
	export DOCKER_NAME=$(MODEL_BUILDER_IMG_NAME);\
	export DOCKER_IMAGE=$(MODEL_BUILDER_IMG_NAME);\
	bash ./docker_push.sh

push-client: client
	export DOCKER_NAME=$(CLIENT_IMG_NAME);\
	export DOCKER_IMAGE=$(CLIENT_IMG_NAME);\
	bash ./docker_push.sh

# Publish development images
push-dev-images:

	# Push everything to auroradevacr.azurecr.io/gordo-components
	export DOCKER_REGISTRY=auroradevacr.azurecr.io;\
	export DOCKER_REPO=gordo;\
	$(MAKE) push-builder push-server push-client push-workflow-generator

	# Also push workflow-generator to auroradevacr.azurecr.io/gordo-infrastructure
	# as gordo-controller still expects it to be located there.
	export DOCKER_REGISTRY=auroradevacr.azurecr.io;\
	export DOCKER_REPO=gordo-infrastructure;\
	$(MAKE) push-workflow-generator

push-prod-images: export GORDO_PROD_MODE:="true"
push-prod-images: push-builder push-server push-client push-workflow-generator

# Make the python source distribution
sdist:
# Ensure the dist directory is empty/non-existant before sdist
	rm -rf ./dist/
	python setup.py sdist

images: model-builder model-server client

test:
	python setup.py test

testall:
	python setup.py testall

docs:
	cd ./docs && $(MAKE) html

all: test images push-dev-images

.PHONY: model-builder model-server client watchman push-server push-builder push-client push-dev-images push-prod-images images test all docs workflow-generator push-workflow-generator
