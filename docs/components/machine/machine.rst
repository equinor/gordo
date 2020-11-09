Machine
-------

A Machine is the central unity of
a model, dataset, metadata and everything
needed to create and build a ML model to
be served by a deployment.

An example of a ``Machine`` in the context of a YAML config, could be
the following:

.. code-block:: yaml

    - name: ct-23-0001
      dataset:
        tags:
          - TAG 1
          - TAG 2
          - TAG 3
        train_start_date: 2016-11-07T09:11:30+01:00
        train_end_date: 2018-09-15T03:01:00+01:00
      metadata:
        arbitrary-key: arbitrary-value
      model:
        gordo.machine.model.anomaly.diff.DiffBasedAnomalyDetector:
          base_estimator:
            sklearn.pipeline.Pipeline:
              steps:
                - sklearn.preprocessing.MinMaxScaler
                - gordo.machine.model.models.KerasAutoEncoder:
                    kind: feedforward_hourglass

And to construct this into a python object:

.. code-block:: python

    >>> from gordo.machine import Machine
    >>> # `config` is the result of the parsed and loaded yaml element above
    >>> machine = Machine.from_config(config, project_name='test-proj')
    >>> machine.name
    ct-23-0001

.. automodule:: gordo.machine.machine
    :members:
    :undoc-members:
    :show-inheritance:

.. toctree::
    :maxdepth: 4
    :caption: Validators

    ./validators.rst

.. toctree::
    :maxdepth: 4
    :caption: Model

    ./model/model.rst

.. toctree::
    :maxdepth: 4
    :caption: Metadata

    ./metadata.rst