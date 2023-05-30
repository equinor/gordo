Anomaly Models
--------------

Models which implment a :func:`gordo.machine.model.anomaly.base.AnomalyDetectorBase.anomaly` and can be served under the
model server :ref:`post-prediction` endpoint.


AnomalyDetectorBase
^^^^^^^^^^^^^^^^^^^

The base class for all other anomaly detector models

.. automodule:: gordo.machine.model.anomaly.base
    :members:
    :undoc-members:
    :show-inheritance:


DiffBasedAnomalyDetector
^^^^^^^^^^^^^^^^^^^^^^^^

Calculates the absolute value prediction differences between y and yhat as well
as the absolute difference error between both matrices via :func:`numpy.linalg.norm`

.. automodule:: gordo.machine.model.anomaly.diff
    :members:
    :undoc-members:
    :show-inheritance: