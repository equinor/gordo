Model configuration
-------------------

Example of the Gordo model configuration:

.. literalinclude:: ../../examples/model-configuration.yaml

A Gordo model is typically wrapped by the class :class:`gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector`.
This class holds generic methods for model cross-validation, training and fitting.

In turn, this method is wrapped around the class :class:`gordo.builder.build_model.ModelBuilder`, which is the top-level

We will focus on the output that is created using :func:`sklearn.model_selection.cross_validate` method,
which is used when using the model for predictions.

:class:`gordo.machine.machine.Machine` class holds basically all information that is contained in the one Gordo config.
The :class:`gordo.builder.build_model.ModelBuilder` class takes a :class:`gordo.machine.machine.Machine` and does the heavy lifting
when it comes to data fetching, cross-validation and model training.