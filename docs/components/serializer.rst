Serializer
----------

The serializer is the core component used in the conversion of a Gordo config
file into Python objects which interact in order to construct a full ML model
capable of being served on Kubernetes.

Things like the ``dataset`` and ``model`` keys within the YAML config represents
objects which will be (de)serialized by the serializer to complete this goal.


.. automodule:: gordo.serializer.serializer
    :members:
    :undoc-members:
    :show-inheritance:


From Definition
===============

The ability to take a 'raw' representation of an object in ``dict`` form
and load it into a Python object.

.. automodule:: gordo.serializer.from_definition
    :members:
    :undoc-members:
    :show-inheritance:


Into Definitiion
================

The ability to take a Python object, such as a scikit-learn
pipeline and convert it into a primitive ``dict``, which can then be inserted
into a YAML config file.

.. automodule:: gordo.serializer.into_definition
    :members:
    :undoc-members:
    :show-inheritance:
