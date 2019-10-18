

<h1 align="center">Gordo Components</h1>
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
  <a href="https://circleci.com/gh/equinor/gordo-components">
    <img src="https://circleci.com/gh/equinor/gordo-components/tree/master.svg?style=svg" alt="Build Status"/>
  </a>
  <a href="https://codecov.io/gh/equinor/gordo-components">
    <img src="https://codecov.io/gh/equinor/gordo-components/branch/master/graph/badge.svg" alt="Codecov"/>
  </a>
  <a href="https://gordo-components.readthedocs.io/en/latest/?badge=latest">
    <img src="https://readthedocs.org/projects/gordo-components/badge/?version=latest" alt="Documentation"/>
  </a> 
  <a href="https://dependabot.com">
    <img src="https://api.dependabot.com/badges/status?host=github&repo=equinor/gordo-components" alt="Dependabot"/>
  </a>
</div>

---

## About:
Gordo-Components is part of the common ML ops provided by `gordo`

It fulfills the role of inhaling config files and supplying components to the pipeline of:

1. Fetching data
2. Training model
3. Serving model

It is designed to be used _by_ `gordo` and not (at present) as a standalone tool.

---

## Examples

See our [example](./examples) notebooks for how to develop with `gordo` locally.

---

## Install: 
`pip install --upgrade gordo-components`  

Bleeding edge:  
`pip install git+https://github.com/equinor/gordo-components.git`

## Uninstall:
`pip uninstall gordo-components`
