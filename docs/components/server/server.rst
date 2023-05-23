ML Server
---------
The ML Server is responsible for giving different "views" into the model
being served.


Server
======
.. automodule:: gordo.server.server
    :members:
    :undoc-members:
    :show-inheritance:

Blueprints
==========
A collection of implemented views into the Model being served.

.. automodule:: gordo.server.blueprints
    :members:
    :undoc-members:
    :show-inheritance:

.. toctree::
    :maxdepth: 4
    :caption: Views:

    base.rst
    anomaly.rst

Utils
=====
Shared utility functions and decorators which are used by the Views

.. automodule:: gordo.server.utils
    :members:
    :undoc-members:
    :show-inheritance:


Model IO
========
The general model input/output operations applied by the views

.. automodule:: gordo.server.model_io
    :members:
    :undoc-members:
    :show-inheritance:
