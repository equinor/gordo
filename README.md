

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
      * [How to update packages](#How-to-update-packages)
    * [How to run tests locally](#How-to-run-tests-locally)
        * [Tests system requirements](#Tests-system-requirements)
        * [Run tests](#Run-tests)
        * [How to run tests in debug mode](#How-to-run-tests-in-debug-mode)

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
pip install --upgrade pip
pip install --upgrade pip-tools
# Some of the packages are in private pypi (Azure artifacts), so you have to specify its url.
# After running next command you will be prompted with <PAT name> and <PAT password> for such pypi-url.
# You might get PAT (personal assess token) by [this instruction](https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops&tabs=preview-page#create-a-pat) 
# in Azure DevOps. This PAT should only have "Packaging -> Read" scope.
pip install --extra-index-url <https://private-pypi-repo-url/> -r requirements/full_requirements.txt
pip install -r requirements/test_requirements.txt
```

#### How to update packages
Note: you have to install `pip-tools` version higher then `6` for requirements to have same multi-line output format.    

To update some package in `full_requirements.txt`:
- change its version in `requirements.in` file;
- (todo once) get credentials to access private pypi 
(for more details see [How to prepare working environment](#How-to-prepare-working-environment) section);
- compile requirements:
```shell
# this command might be changed with time, so its better to take it from top of the `full_requirements.txt` file.
pip-compile --extra-index-url <https://private-pypi-repo-url/> --no-emit-index-url --output-file=full_requirements.txt mlflow_requirements.in postgres_requirements.in requirements.in  
```

### How to run tests locally

#### Tests system requirements
To run tests it's required for your system to has (note: commands might differ from your OS):
- running docker process;
- available 5432 port for postgres container 
(`postgresql` container is used, so better to stop your local instance for tests running). 

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
On the configuration setup for test running add to `Additional arguments:` in `pytest` 
section following string: `--ignore benchmarks --cov-report= --no-cov ` 
or TEMPORARY remove `--cov-report=xml` and `--cov=gordo` from `pytest.ini` file.
