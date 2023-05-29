Configuration
-------------

Example of the Gordo model configuration:

.. literalinclude:: ../../examples/model-configuration.yaml

We can deserialize this configuration into a model object with using :mod:`gordo.serializer.serializer` module.

..
    Commented temporory due to issue wierd exception:
        File ~/gordo/docs/../gordo/machine/model/models.py:326, in KerasBaseEstimator.get_params(self, **params)
            324 with open('get_params', 'a') as f:
            325     f.write("before get_params call\n")
        --> 326 params = super(BaseEstimator, self).get_params(**params)
            327 with open('get_params', 'a') as f:
            328     f.write("-----\n")
        AttributeError: 'super' object has no attribute 'get_params'
    .. ipython::

        In [1]: import yaml

        In [2]: with open('../examples/model-configuration.yaml', 'r') as f:
           ...:     config=yaml.safe_load(f)
           ...:

        In [3]: from gordo.serializer import from_definition

        In [4]: model=from_definition(config['model'])

        In [5]: repr(model)

A Gordo model is typically wrapped by the class :class:`gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector`.
This class holds generic methods for model cross-validation, training and fitting.

In turn, this method is wrapped around the class :class:`gordo.builder.build_model.ModelBuilder`, which is the top-level

We will focus on the output that is created using :func:`sklearn.model_selection.cross_validate` method,
which is used when using the model for predictions.

:class:`gordo.machine.machine.Machine` class holds basically all information that is contained in the one Gordo config.
The :class:`gordo.builder.build_model.ModelBuilder` class takes a :class:`gordo.machine.machine.Machine` and does the heavy lifting
when it comes to data fetching, cross-validation and model training.