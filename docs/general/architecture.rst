Architecture
------------

Gordo is based on parsing a config file written in Yaml
that is converted into an `Argo <https://argoproj.github.io/argo-workflows/>`_ workflow. This is
deployed with `ArgoCD <https://argo-cd.readthedocs.io/en/stable/>`_ onto a Kubernetes cluster.
The main interface after building the models is a set of ``REST`` APIs

.. image:: ../_static/architecture_diagram.png
