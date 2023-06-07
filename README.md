

<h1 align="center">Gordo</h1>
<div align="center">
 <!-- Uncomment line below once we decided on 'logo.png' -->
 <!--<img align="center" src="logo.png" width="250" height="250">-->
 <br />
 <strong>
   Building thousands of models with timeseries data to monitor systems.
 </strong>
</div>

<br />

<div align="center">
  <a href="https://github.com/equinor/gordo/actions?query=branch=master">
    <img src="https://github.com/equinor/gordo/workflows/CI/badge.svg?branch=master" alt="Build Status"/>
  </a>
</div>

---
## About

Gordo fulfills the role of inhaling config files and supplying components to the pipeline of:

1. Fetching data
2. Training model
3. Serving model

## Components

* [gordo-controller](https://github.com/equinor/gordo-controller/) - Kubernetes controller for the Gordo CRDs.
* [gordo-core](https://github.com/equinor/gordo-core/) - Gordo core library.
* [gordo-client](https://github.com/equinor/gordo-client/) - Gordo server's client. It can make predictions from deployed models.

---

[Documentation is available on Read the Docs](https://gordo1.readthedocs.io/)

---
## Install

[gordo-helm](https://github.com/equinor/gordo-helm) - you can use [gordo](https://github.com/equinor/gordo-helm/tree/main/charts/gordo) helm chart from this repository to deploy gordo infrastructure to your Kubernetes cluster. 

### Python package 

`pip install --upgrade gordo`  

With additional extras:
`pip install gordo[postgres,mlflow]`  

Bleeding edge:  
`pip install git+https://github.com/equinor/gordo.git`


## Developer manual

This section will explain how to start development of Gordo.

### Setup

Create and activate a virtual environment first. As a default option, it can be [venv](https://docs.python.org/3/library/venv.html) module.

Install pip-tools
```
pip install --upgrade pip
pip install --upgrade pip-tools
```

Install requirements
```
pip install -r requirements/full_requirements.txt
pip install -r requirements/test_requirements.txt
```

Install package:
```
python3 setup.py install
```

#### How to update packages

Note: you have to install `pip-tools` version higher then `6` for requirements to have same multi-line output format.

To update some package in `full_requirements.txt`:
- Change its version in `requirements.in` file;
- Compile and upgrade requirements:
```shell
pip-compile --upgrade --output-file=full_requirements.txt mlflow_requirements.in postgres_requirements.in requirements.in  
```

### Examples

See our [example](./examples) notebooks for how to develop with `gordo` locally.

### How to run tests locally

List of commands to run tests can be found [here](/setup.cfg).
Running of tests takes some time, so it's faster to run tests in parallel:
```
pytest -n auto -m 'not dockertest' --ignore benchmarks
```
Run docker-related tests:
```
pytest -m 'dockertest'
```

> **_NOTE:_**  To run tests it's required for your system to has (note: commands might differ from your OS):
> - Running docker daemon.
> - Available 5432 port for `postgres` container.

> **_NOTE:_** this example is for Pycharm IDE to use `breakpoints` in the code of the tests.  
> On the configuration setup for test running add to `Additional arguments:` in `pytest` 
> section following string: `--ignore benchmarks --cov-report= --no-cov ` 

### Build the documentation

This command will run the local documentation server:

```console
> cd docs/
> make watch
```
