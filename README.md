# gordo-flow

---

## About:
Gordo-Flow is part of the common ML ops provided by `gordo`  (_TODO: Link to repo once created_)

It fulfills the role of inhaling config files for model creation and data fetching by creating two seperate containers to perform the following:

1. Model configuration & Building/training -> deposit serialized model
2. Model serving

It is designed to be used _by_ `gordo` and not (at present) as a standalone tool.

## Install: 
`python setup.py install`  
or...  
`pip install git+https://github.com/Statoil/gordo-flow.git`

## Uninstall:
`pip uninstall gordo-flow`