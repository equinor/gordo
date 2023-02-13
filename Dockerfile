# Gordo base image
FROM python:3.9-bullseye as builder

# Copy source code
COPY . /code
# Copy .git to deduce version number
COPY .git /code/

WORKDIR /code
RUN rm -rf /code/dist \
    && python setup.py sdist \
    && mv /code/dist/$(ls /code/dist | head -1) /code/dist/gordo-packed.tar.gz

# Extract a few big dependencies which docker will cache even when other dependencies change
RUN cat /code/requirements/full_requirements.txt | grep tensorflow== > /code/prereq.txt \
    && cat /code/requirements/full_requirements.txt | grep pyarrow== >> /code/prereq.txt \
    && cat /code/requirements/full_requirements.txt | grep scipy== >> /code/prereq.txt \
    && cat /code/requirements/full_requirements.txt | grep catboost== >> /code/prereq.txt

FROM python:3.9-slim-bullseye

# Nonroot user for running CMD
RUN groupadd -g 999 gordo && \
    useradd -r -u 999 -g gordo gordo

ENV HOME "/home/gordo"
ENV PATH "${HOME}/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y \
    curl \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install requirements separately for improved docker caching
COPY --from=builder /code/prereq.txt .
RUN pip install --no-deps -r prereq.txt --no-cache-dir

COPY requirements/full_requirements.txt .
RUN pip install -r full_requirements.txt --no-cache-dir

# Install gordo, packaged from earlier 'python setup.py sdist'
COPY --from=builder /code/dist/gordo-packed.tar.gz .
RUN pip install gordo-packed.tar.gz[full]

# Install GordoDeploy dependencies
ARG HTTPS_PROXY
ARG KUBECTL_VERSION="v1.22.4"

#donwload & install kubectl
RUN curl -sSL -o /usr/local/bin/kubectl https://storage.googleapis.com/kubernetes-release/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl &&\
  chmod +x /usr/local/bin/kubectl

# Copy scripts
COPY ./functions.sh ${HOME}/functions.sh
COPY ./run_workflow_and_argo.sh ${HOME}/run_workflow_and_argo.sh

# Baking in example configs for running tests, as docker.client.containers.run
# bind doesn't seem to work correctly for non-root users
# volumes={repo_dir: {"bind": "/home/gordo", "mode": "ro"}},
COPY ./examples ${HOME}/examples
COPY ./resources ${HOME}/resources

# Install ModelBuilder dependencies
ADD build.sh ${HOME}/build.sh

# build.sh (build the model) as executable default command
RUN cp ${HOME}/build.sh /usr/bin/build \
    && chmod a+x /usr/bin/build

# Run things from gordo's home to have write access when needed (e.g. Catboost tmp files)
WORKDIR ${HOME}

#download & install argo
ENV ARGO_VERSIONS="[{\"number\":3,\"version\":\"3.4.3\"},{\"number\":2,\"version\":\"2.12.13\"}]"
COPY scripts/download_argo.py ./download_argo.py
RUN python3 ./download_argo.py -o /usr/local/bin

# Cleanup
RUN dpkg -r curl && rm ./download_argo.py

# Make gordo own all in its home
RUN chown -R gordo:gordo ${HOME}

# Switch user
USER 999 
