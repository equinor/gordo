Anomaly Models
--------------

Models which implment a ``.anomaly(X, y)`` and can be served under the
model server ``/anomaly/prediction`` endpoint.


DiffBasedAnomalyDetector
========================

Calculates the absolute value prediction differences between y and yhat as well
as the absolute difference error between both matrices via ``numpy.linalg.norm(..., axis=1)``

.. automodule:: gordo_components.model.anomaly.diff
    :members:
    :undoc-members:
    :show-inheritance: