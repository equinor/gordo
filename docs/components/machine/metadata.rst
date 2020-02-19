Metadata
--------

Each Machine is entitled to have Metadata, which can be set at the Machine.metadata
level inside the config, but will result in a standardized output of metadata
under ``user_defined`` and ``build_metadata``. Where ``user_defined`` can go
arbitrarily deep, depending on the amount of metadata the user wishes to enter.

``build_metadata`` is more predictable. During the course of building a Machine
the system will insert certain metadata given about the build time, and model
metrics (depending on configuration).

.. automodule:: gordo.machine.metadata.metadata
    :members:
    :undoc-members:
    :show-inheritance:
