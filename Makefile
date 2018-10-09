
MODEL_BUILDER_IMG_NAME := milesg/gordo-flow-builder:latest
MODEL_SERVER_IMG_NAME  := milesg/gordo-flow-server:latest


# Create the image capable to building/training a model
model-builder:
	docker build . -f Dockerfile-ModelBuilder -t $(MODEL_BUILDER_IMG_NAME)

# Create the image which serves built models
model-server:
	cd ./gordo_flow/runtime && s2i build . -e HTTPS_PROXY=http://www-proxy.statoil.no:80/ \
	 seldonio/seldon-core-s2i-python3 $(MODEL_SERVER_IMG_NAME)

# Publish images to the currently logged in docker repo
push-images:
	docker push $(MODEL_BUILDER_IMG_NAME)
	docker push $(MODEL_SERVER_IMG_NAME)

test:
	python setup.py test

all: test model-builder model-server push-images
