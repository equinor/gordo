

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
  <a href="https://codecov.io/gh/equinor/gordo">
    <img src="https://codecov.io/gh/equinor/gordo/branch/master/graph/badge.svg" alt="Codecov"/>
  </a>
  <a href="https://gordo.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/gordo/badge/?version=latest" alt="Documentation"/>
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
    * [How to run tests locally](#How-to-run-tests-locally)
        * [Tests system requirements](#Tests-system-requirements)
        * [Run tests](#Run-tests)
        * [How to run tests in debug mode](#How-to-run-tests-in-debug-mode)
    * [Code quality checks](#Code-quality-checks)
        * [Run code quality checks locally](#Run-code-quality-checks-locally)

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

Bleeding edge:  
`pip install git+https://github.com/equinor/gordo.git`

## Uninstall
`pip uninstall gordo`

## Developer manual
This section will explain how to start development of Gordo.

### How to prepare working environment
- install requirements
```shell script
# create and activate virtualenv. Note: you should use python3.7 (project's tensorflow version is not compatible with python3.8)
# then:
make install_app_requirements
```

### How to run tests locally

#### Tests system requirements
To run tests it's required for your system to has (note: commands might differ from your OS):
- running docker process;
- available 5432 port for postgres container.

#### Run tests
List of commands to run tests can be found [here](/setup.cfg).  
Running of tests takes some time, so it's faster to run tests in parallel:
```shell script
# example
pytest tests/gordo/client/test_client.py --ignore benchmarks --cov-report= --no-cov -n auto -m 'not dockertest' 
# or if you have multiple python versions and they're not resolved properly:
python3.7 -m pytest ... 
```

#### How to run tests in debug mode
Note: this example is for Pycharm IDE to use `breakpoints` in the code of the tests.  
On the configuration setup for test running add to `Additional arguments:` in `pytest` section following string: `--ignore benchmarks --cov-report= --no-cov `

### Code quality checks
This repo uses [black](https://black.readthedocs.io/en/stable/) and [flakehell](https://github.com/life4/flakehell/blob/master/README.md) (with plugins) for code formatting and analys.  
Code quality checks on the CI automatically. Checks run only on *CHANGED* code (not on all files in the repo).  

#### Run code quality checks locally
Before pushing code to the "remote" - check if your changes are good:
```shell
# this will compare changes in your local branch with origin/master.
# note that origin/master should be up to date and merged into current branch. 
make code-quality-locally
```
In plans to add pre-commit hooks not ot do it manually.
