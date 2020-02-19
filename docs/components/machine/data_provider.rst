Data Providers & Readers
========================

Data providers and readers are responsible to doing the actual reading/collection
of the raw data for a given dataset. Datasets like :class:`gordo.machine.dataset.datasets.TimeSeriesDataset`
for example can use multiple data providers in the case where a single provider
cannot gather all the required data.

Data Providers
--------------
.. automodule:: gordo.machine.dataset.data_provider.providers
    :members:
    :undoc-members:
    :show-inheritance:

IROC Reader
-----------
.. automodule:: gordo.machine.dataset.data_provider.iroc_reader
    :members:
    :undoc-members:
    :show-inheritance:

NCS Reader
----------
.. automodule:: gordo.machine.dataset.data_provider.ncs_reader
    :members:
    :undoc-members:
    :show-inheritance:

Azure Utils
-----------
.. automodule:: gordo.machine.dataset.data_provider.azure_utils
    :members:
    :undoc-members:
    :show-inheritance:

Base Data Provider
------------------
.. automodule:: gordo.machine.dataset.data_provider.base
    :members:
    :undoc-members:
    :show-inheritance:
