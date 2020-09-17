Quick start
-----------

The concept of ``Gordo`` is to (as of now) process, only, *timeseries*
datasets which are comprised of sensors/tag identifies. The workflow
launches the collection of these tags, building of a defined model and
subsequent deployment of a ML Server which acts as a REST interface
in front of the model.

A typical config file might look like this:

.. code-block:: yaml

    apiVersion: equinor.com/v1
    kind: Gordo
    metadata:
      name: test-project
    spec:
      deploy-version: 0.39.0
      config:

        machines:

          # This machine specifies all keys, and will train a model on one month
          # worth of data, as shown in its train_start/end_date dataset keys.
          - name: some-name-here
            dataset:
              train_start_date: 2018-01-01T00:00:00Z
              train_end_date: 2018-02-01T00:00:00Z
              resolution: 2T  # Resample timeseries at 2min intervals (pandas freq strings)
              tags:
                - tag-1
                - tag-2
            model:
              sklearn.pipeline.Pipeline:
                steps:
                  - sklearn.preprocessing.MinMaxScaler
                  - gordo.model.models.KerasAutoEncoder:
                      kind: feedforward_hourglass
            metadata:
              key1: some-value

          # This machine does NOT specify all keys, it is missing 'model' but will
          # have the 'model' under 'globals' inserted as its default.
          # And will train a model on one month as well.
          - name: some-name-here
            dataset:
              train_start_date: 2018-01-01T00:00:00Z
              train_end_date: 2018-02-01T00:00:00Z
              resolution: 2T  # Resample timeseries at 2min intervals (pandas freq strings)
              tags:
                - tag-1
                - tag-2
            metadata:
              key1: some-different-value-if-you-want
              nested-keys-allowed:
                - correct: true

        globals:
          model:
            sklearn.pipeline.Pipeline:
              steps:
                - sklearn.preprocessing.MinMaxScaler
                - gordo.model.models.KerasAutoEncoder:
                    kind: feedforward_model

          metadata:
            what-does-this-do: "This metadata will get mapped to every machine's metadata!"


One can experiment locally with Gordo through the Jupyter Notebooks provided in
the `examples <https://github.com/equinor/gordo/tree/master/examples>`_
directory of the repository.