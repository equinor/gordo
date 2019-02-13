# Gordo Components

[![Build Status](https://travis-ci.com/equinor/gordo-components.svg?token=9cHSKigsoXktTGTEJsVA&branch=master)](https://travis-ci.com/equinor/gordo-components)

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
`python setup.py install`  
or...  
`pip install git+https://github.com/Statoil/gordo-components.git`

## Uninstall:
`pip uninstall gordo-components`