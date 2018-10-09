
MODEL_BUILDER_IMG_NAME := azure-repo/gordo-flow-builder:latest
MODEL_SERVER_IMG_NAME  := azure-repo/gordo-flow-server:latest


model-builder:
	docker build . -f Dockerfile-ModelBuilder -t $(MODEL_BUILDER_IMG_NAME)

model-server:
	cd ./gordo_flow/runtime && s2i build . -e HTTPS_PROXY=http://www-proxy.statoil.no:80/ \
	 seldonio/seldon-core-s2i-python3 $(MODEL_SERVER_IMG_NAME)

test:
	python setup.py test
