Architecture
------------

Gordo is based on parsing a config file written in Yaml
that is converted into an `Argo <https://argoproj.github.io/argo-workflows/>`_ workflow. This is
deployed with `ArgoCD <https://argo-cd.readthedocs.io/en/stable/>`_ onto a Kubernetes cluster.
The main interface after building the models is a set of ``REST`` APIs

.. image:: ../_static/architecture_diagram.png

`Gordo <https://github.com/equinor/gordo-helm/blob/main/charts/gordo/templates/crds/gordos.equinor.com.yaml>`_ is a `CustomResourceDefinition <https://kubernetes.io/docs/tasks/extend-kubernetes/custom-resources/custom-resource-definitions/>`_
represents the project and can contains multiple Machine Learning models.

.. note::
    In order to reproduce all examples below you can install `this Helm chart <https://github.com/equinor/gordo-helm>`_ on your local `Minikube cluster <https://github.com/equinor/gordo-helm/tree/main/charts/gordo#development-manual>`_.

Simplest possible project with 2 models we could find in ``examples/test-project.yaml``:

.. literalinclude:: ../../examples/test-project.yaml

To deploy this project to the cluster:

.. code-block:: console

    > kubectl apply -f examples/test-project.yaml
    gordo.equinor.com/test-project created

Check status of deployed project:

.. code-block:: console

    > kubectl get gordo
    NAME           MODEL-COUNT   MODELS-BUILT   SUBMITTED   DEPLOY VERSION
    test-project   2             0              1           latest