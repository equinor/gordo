Model output and evaluation
---------------------------

A Gordo model is typically of the class :class:`gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector`.
This is the class that is initiated to both train/fit a model and later is used to gain predictions.

We will focus on the output that is created using the ``AnomalyDetectorBase.anomaly()`` - method, which is used when using the model for predictions.

Predictions are created based on input matrix :math:`X`, and a known output :math:`Y`, which in most cases of the default auto-encoder is simply :math:`Y=X`.
Based on these values, the following is returned for $n$ observations and $m$ input variables:

.. list-table::
   :header-rows: 1

   * - Name/index
     - Description
     - Notation
     - Dimension
   * - ``model-input``
     - Unscaled input data
     - :math:`X`
     - n x m
   * - ``model-outut``
     - Unscaled model predictions
     - :math:`\hat{Y}`
     - n x m
   * - ``tag-anomaly-scaled``
     - Absolute scaled residual
     - :math:`R^{s}=|Y^{s}-\hat{Y^{s}}|`
     - n x m
   * - ``total-anomaly-scaled``
     - Mean squared error
     - :math:`\frac{1}{n}\sum_{j=1}^{n} {r_j^s}^{2} \quad \forall i=(1,...,n)`
     - n x 1
   * - ``tag-anomaly-unscaled``
     - Absolute residuals
     - :math:`R = |Y-\hat{Y}|`
     - n x m 
   * - ``total-anomaly-unscaled``
     - Mean squared error
     - :math:`\frac{1}{n}\sum_{j=1}^n r_j^{2} \quad \forall i=(1,...,n)`
     - n x m
   * - ``anomaly-confidence``
     - Anomaly confidence each tag
     - ``tag-anomaly-scaled`` / ``feature-thresholds-per-fold`` (last fold considered)
     - n x m
   * - ``total-anomaly-confidence``
     - Anomaly confidence complete model
     - ``total-anomaly-scaled`` / ``aggregate-thresholds-per-fold`` (last-fold considered)
     - n x 1
   * - ``smooth-tag-anomaly-scaled``
     - 
     - 
     - (n-window) x m
   * - ``smooth-total-anomaly-scaled``
     - 
     - 
     - (n-window) x 1
   * - ``smooth-tag-anomaly-unscaled``
     - 
     - 
     - (n-window) x m
   * - ``smooth-total-anomaly-unscaled``
     - 
     - 
     - (n-window) x 1


Smoothed values
^^^^^^^^^^^^^^^

All output variables with the ``smooth``-prefix are generated using the ``smoothing_method`` and ``window`` stated in the model configuration.
If *both* are set to ``None``\ , the output is not produced.
The ``window`` decides over how many points the smoothing should be considered.

You can select one of the following methods:

* ``smm`` simple moving median
* ``sma`` simple moving average or
* ``ewma`` exponential weighted moving average

.. code-block:: yaml

     model:
       gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
         smoothing_method: smm
         window: 144

For given

* start timestamp :math:`t_1`,
* stop timestamp :math:`t_n` and
* resolution :math:`r = \delta(t_i) =  t_i - t_{i-1}` (e.g. 10min),

the vector length of the dataset is :math:`(t_{stop} - t_{start}) / r = n`.
E.g. 24h / 10min = 144 for 24h of data.

This means, that, using the default smoothing window of $w=144$, one must fetch 24h extra data *prior* to the considered time period, resulting in time series length :math:`|T| = n + w`, with :math:`T = \{ t_{-w}, ... , t_{1}, ..., t_{n} \}`.

Anomaly confidence
^^^^^^^^^^^^^^^^^^

During model training, cross validation is applied to the class instance through the ``.cross_validate()``\ -method.
In this process, confidence thresholds are created for individual tags as well as for the complete model.
The cross-validation scheme is explained in the `model evaluation and training metadata <>` section, and the calculation of the anomaly confidence.

Based on these thresholds, the following metrics are reported:


* ``anomaly-confidence`` = ``tag-anomaly-scaled`` / ``feature-thresholds-per-fold(last fold)``
* ``total-anomaly-confidence`` = ``total-anomaly-scaled`` / ``aggregate-thresholds-per-fold(last-fold)``

Scaling of data during cross-validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before the cross-validation is executed, the ``scoring_scaler`` is extracted.
This is used to transform the data before the wanted metrics are calculated.

An internal method, :func:`gordo.builder.build_model.ModelBuilder.build_metrics_dict`, is called prior to the cross-validation, which specified list ``metrics`` and the ``scoring_scaler``.
This method builds a callable dictionary by using the nested function :func:`gordo.machine.model.utils.metric_wrapper`.
This generated dictionary now contains information about the ``scoring_scaler`` which will be used later.
