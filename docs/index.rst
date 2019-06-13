.. Gordo Components Documentation

Welcome to Gordo Components' documentation!
===========================================

.. _overview:

Overview
^^^^^^^^

Gordo Components is a collection of tools to create a distributed
ML service represented by a specific pipeline. Generally, any
``sklearn.pipeline.Pipeline`` object can be defined within a config file
and deployed as a REST API on Kubernetes.

.. toctree::
    :maxdepth: 4
    :caption: Project Resources:

    ./general/quickstart.rst
    ./general/architecture.rst
    ./general/endpoints.rst

.. toctree::
    :maxdepth: 4
    :caption: Components:

    ./components/model/model.rst
    ./components/builder.rst
    ./components/data_provider.rst
    ./components/dataset.rst
    ./components/serializer.rst
    ./components/watchman.rst
    ./components/server.rst
    ./components/cli.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
