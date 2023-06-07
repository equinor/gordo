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

Evaluation specification
^^^^^^^^^^^^^^^^^^^^^^^^

Alongside the ML-model itself, all aspects of the cross-validation evaluation is parameterized in the config:

.. code-block:: yaml

   - evaluation:
        cv: 
          sklearn.model_selection.TimeSeriesSplit:
            n_splits: 3
        cv_mode: full_build
        scoring_scaler: sklearn.preprocessing.MinMaxScaler
        metrics:
        - explained_variance_score
        - r2_score
        - mean_squared_error
        - mean_absolute_error

Alternatively, the ``cv_mode`` can be set to ``cross_val_only`` which will not fit the final model.

Cross-validation methods
^^^^^^^^^^^^^^^^^^^^^^^^

Setting ``cv`` to :class:`sklearn.model_selection.TimeSeriesSplit` , the dataset is split as depicted below.
Independent of the number of splits, the test set always is of the same size.

An alternative is to use `k-fold <https://scikit-learn.org/stable/modules/cross_validation.html>`_ cross-validation.
Here, one can decide to shuffle the data before it is split into folds.
In contradiction to the time-series-split above, which augments the considered data in each fold with time-consecutive observations, this method is uncoupled from the time dimension.
This must be considered when comparing results from different folds.

The following parameters can then be set as such:

.. code-block:: yaml

   - evaluation:
        cv: 
          sklearn.model_selection.KFold:
            n_splits: 3
            shuffle: True
            random_state: 0

Borrowed from `scikit-learn <https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py>`_ , which performs the actual split/train for us.
