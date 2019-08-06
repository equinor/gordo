Endpoints
---------

==================
Project index page
==================

Going to the base path of the project, ie. ``/gordo/v0/my-project/`` will return the
project level index, with returns a collection of the metadata surrounding the models currently deployed and their status.
Each ``endpoint`` key has an associated ``endpoint-metadata`` key which is the direct transferal of metadata returned from
the ML servers at their :ref:`ml-server-metadata-route` route.

.. code-block:: python

    {'endpoints': [{'endpoint': '/gordo/v0/sample-project/sample-model1/',
                'endpoint-metadata': {'env': {'MODEL_LOCATION': '/gordo/models/sample1/1560329965701/model1'},
                                      'gordo-server-version': '0.22.1.dev12+gfb2e8cb.d20190612',
                                      'metadata': {'dataset': {'filter': '',
                                                               'resolution': '10T',
                                                               'tag_list': [{'asset': 'ASSET-A',
                                                                             'name': 'TAG-1'},
                                                                            {'asset': 'ASSET-B',
                                                                             'name': 'TAG-2'},
                                                                            {'asset': 'ASSET-C',
                                                                             'name': 'TAG-3'}],
                                                               'train_end_date': '2019-03-01 '
                                                                                 '00:00:00+00:00',
                                                               'train_start_date': '2019-01-01 '
                                                                                   '00:00:00+00:00'},
                                                   'model': {'cross-validation': {'cv-duration-sec': 15.931376934051514,
                                                                                  'scores': {'explained-variance': {'max': 9.62193288008469e-16,
                                                                                                                    'mean': 2.9605947323337506e-16,
                                                                                                                    'min': -2.220446049250313e-16,
                                                                                                                    'raw-scores': [-2.220446049250313e-16,
                                                                                                                                   9.62193288008469e-16,
                                                                                                                                   1.4802973661668753e-16],
                                                                                                                    'std': 4.946644983939441e-16}}},
                                                             'data-query-duration-sec': 44.260337114334106,
                                                             'history': {'acc': [0.9912910438978463,
                                                                                 0.9988231140402495,
                                                                                 0.9988231140402495,
                                                                                 0.9988231140402495,
                                                                                 0.9988231140472643,
                                                                                 0.9988231140402495,
                                                                                 0.9988231140402495,
                                                                                 0.9988231140402495,
                                                                                 0.9988231140402495,
                                                                                 0.9988231140402495,
                                                                                 0.9988231140402495,
                                                                                 0.9988231140402495,
                                                                                 0.9988231140402495],
                                                                         'loss': [0.2624536094275365,
                                                                                  0.1044679939132186,
                                                                                  0.04376118538868986,
                                                                                  0.021631221938998864,
                                                                                  0.013675790901569965,
                                                                                  0.010892559444415396,
                                                                                  0.009980763574608167,
                                                                                  0.009708345731069488,
                                                                                  0.009639194397312994,
                                                                                  0.009623750014195754,
                                                                                  0.009622064676664956,
                                                                                  0.009621601143707853,
                                                                                  0.009620696473377858]}}}}}]}

----

==============================
Machine Learning Server Routes
==============================

When a model is deployed from a config file, it results in a ML
server capable of the following paths:

Under normal Equinor deployments, paths listed below should be prefixed with ``/gordo/v0/<project-name>/<model-name>``.
Otherwise, the paths listed below are the raw exposed endpoints from the server's perspective.

----

/
=

This is the Swagger UI for the given model. Allows for manual testing of endpoints via a GUI interface.

----

.. _prediction-endpoint:

/prediction/
============

The ``/prediction`` endpoint will return the basic values a model
is capable of returning. Namely, this will be:

- ``model-output``:
    - The raw model output, after calling ``.predict`` on the model or pipeline
      or ``.transform`` if the pipeline/model does not have a ``.predict`` method.
- ``original-input``:
    - Represents the data supplied to the Pipeline, the raw untransformed values.

Sample response:

.. code-block:: python

    {'data': [{'model-output': [3.2764337898700404],
               'original-input': [1.0, 2.0]}
               ],
     'tags': [{'asset': 'ASSET-1', 'name': 'TAG-1'},
              {'asset': 'ASSET-2', 'name': 'TAG-2'}],
     'time-seconds': '0.0165'}


The endpoint accepts both POST and GET requests.

``GET`` requests take ``start`` and  ``end`` timestamps with timezone information in the URL query:

.. code-block:: python

    >>> import requests
    >>> requests.get("https://my-server.io/prediction?start=2019-01-01T00:00:00+01:00&end=2019-01-01T05:00:00+01:00")  # doctest: +SKIP
    >>>
    >>> # or...
    >>> params = {"start": "2019-01-01T00:00:00+01:00", "end": "2019-01-01T05:00:00+01:00"}
    >>> requests.get("https://my-server.io/prediction", params=params)  # doctest: +SKIP

**NOTE:** The requested time interval must be less than 24hrs in time span.

``POST`` requests take raw data:

.. code-block:: python

    >>> import requests
    >>>
    >>> # Single sample:
    >>> requests.post("https://my-server.io/prediction", json={"X": [1, 2, 3, 4]})  # doctest: +SKIP
    >>>
    >>> # Multiple samples:
    >>> requests.post("https://my-server.io/prediction", json={"X": [[1, 2, 3, 4], [5, 6, 7, 8]]})  # doctest: +SKIP

**NOTE:** The client must provide the correct number of input features, ie. if the model was trained on 4 features,
the client should provide 4 feature sample(s).

----

/anomaly/prediction/
====================

The ``/anomaly/prediction`` endpoint will return the data supplied by the ``/prediction`` endpoint
but reserved for models which output the same shape as their input, expected to be AutoEncoder type
models.

By this restriction, additional _features_ are calculated and returned:

- ``tag-anomaly``:
    - Anomaly per feature/tag calculated from the expected tag input (y) and the model's output for those tags (yhat)
- ``total-anomaly``:
    - This is the total anomaly for the given point as calculated by the model.

Sample response:

.. code-block:: python

    {'data': [{'end': [None],
               'tag-anomaly': [
                    2.746687859183499,
                    2.4497416272485886,
                    2.508896707372706
               ],
               'total-anomaly': [60.4668517758241],
               'model-output': [
                    37.94235610961914,
                    10.5305765271186829,
                    12.7146536707878113
               ],
               'original-input': [1, 2, 3],
               'start': [None]}],
     'tags': [{'asset': 'ASSET-A', 'name': 'TAG-1'},
              {'asset': 'ASSET-B', 'name': 'TAG-2'},
              {'asset': 'ASSET-C', 'name': 'TAG-3'}],
     'time-seconds': '0.0866'}


This endpoint accepts both ``GET`` and ``POST`` requests.
Model requests are exactly the same as :ref:`prediction-endpoint` but it is expected the model
being served is of an AutoEncoder variety.

----

/download-model/
================

Returns the current model being served. Loadable via ``gordo_components.serializer.loads(downloaded_bytes)``

----

.. _ml-server-metadata-route:

/metadata/
==========

Various metadata surrounding the current model and environment.
