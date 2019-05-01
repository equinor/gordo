Architecture
------------

``Gordo`` is based on parsing a config file written in ``yaml`` that is converted into an ``Argo`` workflow. This is
deployed with ``ArgoCD`` onto a ``Kubernetes`` cluster. The main interface after building the models is a set of
``REST`` APIs

.. _c4: https://c4model.com/

For illustrating the architecture, we use the C4_ approach.


.. image:: ../_static/Gordo_C4.svg
