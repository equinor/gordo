
MODEL_BUILDER_IMG_NAME := milesg/gordo-components-builder:latest
MODEL_SERVER_IMG_NAME  := milesg/gordo-components-server:latest
MODEL_SERVER_BASE_IMG  := milesg/gordo-components-serverbase:latest

# Create the image capable to building/training a model
model-builder:
	docker build . -f Dockerfile-ModelBuilder -t $(MODEL_BUILDER_IMG_NAME)

# Create the image which serves built models
model-server:
	docker build . -f Dockerfile-ModelServer -t $(MODEL_SERVER_BASE_IMG)
	cd ./gordo_components/runtime && s2i build . -e HTTPS_PROXY=$(HTTPS_PROXY) \
	 $(MODEL_SERVER_BASE_IMG) $(MODEL_SERVER_IMG_NAME)

# Publish images to the currently logged in docker repo
push-images:
	docker push $(MODEL_BUILDER_IMG_NAME)
	docker push $(MODEL_SERVER_IMG_NAME)

test:
	python setup.py test

all: test model-builder model-server push-images
