
Datasets
========
.. automodule:: gordo.machine.dataset.datasets
    :members:
    :undoc-members:
    :show-inheritance:

Base Dataset
------------
.. automodule:: gordo.machine.dataset.base
    :members:
    :undoc-members:
    :show-inheritance:

Row Filtering tools
-------------------

Tooling used for filtering data based on filter strings and applying any
buffering around 'cut' times.

.. automodule:: gordo.machine.dataset.filter_rows
    :members:
    :undoc-members:
    :show-inheritance:

Sensor Tag
----------

Tools for dealing with sensor tag representations.

.. automodule:: gordo.machine.dataset.sensor_tag
    :members:
    :undoc-members:
    :show-inheritance:

DataProvider
------------

Each dataset requires a data provider, which is
responsible to provide the data to the logic of
the dataset.

.. toctree::
    :maxdepth: 4
    :caption: DataProvider

    ./data_provider.rst
