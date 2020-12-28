.. Gordo Documentation

Welcome to Gordo' documentation!
===========================================

.. _overview:

Overview
^^^^^^^^

Gordo is a collection of tools to create a distributed
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

    ./components/machine/machine.rst
    ./components/builder.rst
    ./components/serializer.rst
    ./components/server/server.rst
    ./components/cli.rst
    ./components/workflow.rst
    ./components/util.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
