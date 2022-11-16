

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

## Table of content
* [About](#About)
* [Examples](#Examples)
* [Install](#Install)
* [Uninstall](#Uninstall)
* [Developer manual](#Developer-manual)
    * [How to prepare working environment](#How-to-prepare-working-environment)
        * [How to update packages](#How-to-update-packages)
    * [How to run tests locally](#How-to-run-tests-locally)
        * [Tests system requirements](#Tests-system-requirements)
        * [Run tests](#Run-tests)

## About

Gordo fulfills the role of inhaling config files and supplying components to the pipeline of:

1. Fetching data
2. Training model
3. Serving model

---

## Examples

See our [example](./examples) notebooks for how to develop with `gordo` locally.

---

## Install 
`pip install --upgrade gordo`  

With additional extras:
`pip install gordo[postgres,mlflow]`  

Bleeding edge:  
`pip install git+https://github.com/equinor/gordo.git`

## Uninstall
`pip uninstall gordo`

## Developer manual
This section will explain how to start development of Gordo.

### How to prepare working environment
- Install pip-tools
```
pip install --upgrade pip
pip install --upgrade pip-tools
```

- Install requirements
```
pip install -r requirements/full_requirements.txt
pip install -r requirements/test_requirements.txt
```

#### How to update packages
Note: you have to install `pip-tools` version higher then `6` for requirements to have same multi-line output format.

To update some package in `full_requirements.txt`:
- Change its version in `requirements.in` file;
- Compile and upgrade requirements:
```shell
pip-compile --upgrade --output-file=full_requirements.txt mlflow_requirements.in postgres_requirements.in requirements.in  
```

### How to run tests locally

#### Tests system requirements
To run tests it's required for your system to has (note: commands might differ from your OS):
- Running docker process;
- Available 5432 port for postgres container 
(`postgresql` container is used, so better to stop your local instance for tests running). 

#### Run tests
List of commands to run tests can be found [here](/setup.cfg).
Running of tests takes some time, so it's faster to run tests in parallel:
```
python3 setup.py test
```

> **_NOTE:_** this example is for Pycharm IDE to use `breakpoints` in the code of the tests.  
> On the configuration setup for test running add to `Additional arguments:` in `pytest` 
> section following string: `--ignore benchmarks --cov-report= --no-cov ` 
> or TEMPORARY remove `--cov-report=xml` and `--cov=gordo` from `pytest.ini` file.
