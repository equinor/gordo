BASE_IMG_NAME := gordo/base
MODEL_BUILDER_IMG_NAME := gordo-model-builder
MODEL_SERVER_IMG_NAME  := gordo-model-server
CLIENT_IMG_NAME := gordo-client
WORKFLOW_GENERATOR_IMG_NAME := gordo-deploy

base:
	docker build . -f Dockerfile -t $(BASE_IMG_NAME)

# Create the image capable of rendering argo workflow generator
workflow-generator: base
	docker build -f Dockerfile-GordoDeploy --build-arg BASE_IMAGE=$(BASE_IMG_NAME) -t $(WORKFLOW_GENERATOR_IMG_NAME) .

# Publish image to the currently logged in docker repo
push-workflow-generator: workflow-generator
	export DOCKER_NAME=$(WORKFLOW_GENERATOR_IMG_NAME);\
	export DOCKER_IMAGE=$(WORKFLOW_GENERATOR_IMG_NAME);\
	./docker_push.sh

# Create the image capable to building/training a model
model-builder: base
	docker build -f Dockerfile-ModelBuilder --build-arg BASE_IMAGE=$(BASE_IMG_NAME) -t $(MODEL_BUILDER_IMG_NAME) .

# Create the image which serves built models
model-server: base
	docker build -f Dockerfile-ModelServer --build-arg BASE_IMAGE=$(BASE_IMG_NAME) -t $(MODEL_SERVER_IMG_NAME) .

client: base
	docker build -f Dockerfile-Client --build-arg BASE_IMAGE=$(BASE_IMG_NAME) -t $(CLIENT_IMG_NAME) .

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

############### Scan docker images ####################

scan:
	@images="${MODEL_BUILDER_IMG_NAME} ${MODEL_SERVER_IMG_NAME} ${CLIENT_IMG_NAME}"; \
	uname_S=$(shell uname -s 2>/dev/null || echo not); \
	trivy=$(shell which trivy); \
	if [ -z "$$trivy" ]; then \
		if [ "$$uname_S" = "Darwin" ]; then \
			machine="macOS"; \
		elif [ "$$uname_S"  = "Linux" ]; then \
			machine="Linux"; \
		else \
			echo "Unable to determine platform '$$uname_S'"; exit 1; \
		fi; \
		TRIVY_VERSION=$(shell curl --silent "https://api.github.com/repos/aquasecurity/trivy/releases/latest" | grep '"tag_name":' | sed -E 's/.*"v([^"]+)".*/\1/'); \
		echo "Downloading trivy.."; \
		[ -n "$$TRIVY_VERSION" ] && [ -n "$$machine" ] && curl -Ls "https://github.com/aquasecurity/trivy/releases/download/v$${TRIVY_VERSION}/trivy_$${TRIVY_VERSION}_$${machine}-64bit.tar.gz" | tar zx --wildcards '*trivy' || { echo "Download or extract failed for '$${machine}' version '$${TRIVY_VERSION}'."; exit 1; }; \
		trivy="./trivy"; \
	else \
		TRIVY_VERSION=$(shell trivy -v 2>/dev/null | head -1 | cut -d ' ' -f 2); \
	fi; \
	echo "Trivy version is $${TRIVY_VERSION} and platform is $${uname_S}"; \
	for i in $$images; do \
		echo "Scanning image - '$$i'"; \
		$$trivy --clear-cache && $$trivy --exit-code 1 -severity HIGH,CRITICAL --light --no-progress --ignore-unfixed "$$i"; \
	done; \

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

.PHONY: model-builder model-server client watchman push-server push-builder push-client push-dev-images push-prod-images images test all docs workflow-generator push-workflow-generator base
