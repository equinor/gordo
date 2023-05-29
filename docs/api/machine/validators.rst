Descriptors
-----------

Collection of descriptors to verify types and conditions of the Machine
attributes when loading.

And example of which is if the machine name is set to a value which isn't
a valid URL string, thus causing early failure before k8s itself discovers that the
name isn't valid. (See: :class:`gordo.machine.validators.ValidUrlString`)

.. automodule:: gordo.machine.validators
    :members:
    :undoc-members:
    :show-inheritance:
