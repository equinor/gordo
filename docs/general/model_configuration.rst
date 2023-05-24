Model configuration
-------------------

Example of the Gordo model configuration:

.. literalinclude:: ../../examples/model-configuration.yaml

A Gordo model is typically wrapped by the class :class:`gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector`.
This class holds generic methods for model cross-validation, training and fitting.

In turn, this method is wrapped around the class `\ ``ModelBuilder`` <https://github.com/equinor/gordo/blob/master/gordo/builder/build_model.py>`_\ , which is the top-level 

We will focus on the output that is created using the ``.cross_validate()``\ -method, which is used when using the model for predictions.

`\ ``Machine`` <https://github.com/equinor/gordo/blob/master/gordo/machine/machine.py>`_ class holds basically all information that is contained in one `Gordo-config <../model_configuration>`_.
The `\ ``ModelBuilder`` <https://github.com/equinor/gordo/blob/master/gordo/builder/build_model.py>`_ class takes a ``Machine`` and does the heavy lifting when it comes to data fetching, cross-validation and model training.

The ``dataset`` as well as ``model`` elements are extracted.
A `\ ``DiffBasedAnomalyDetector`` <https://github.com/equinor/gordo/blob/master/gordo/machine/model/anomaly/diff.py>`_ is created from the ``model`` dictionary, using the `\ ``from_definition`` <https://github.com/equinor/gordo/blob/master/gordo/serializer/from_definition.py>`_ of the ``serializer`` module in Gordo, and replaces the ``model``.
