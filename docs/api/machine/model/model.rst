Models
------

Models are a collection of `Scikit-Learn <https://scikit-learn.org/stable/>`_
like models, built specifically to fulfill a need. One example of which is
the ``KerasAutoEncoder``.

Other scikit-learn compliant models can be used within the config files without
any additional configuration.


Base Model
==========

The base model is designed to be inherited from any other models which need
to be implemented within Gordo due to special model requirements. ie. PyTorch,
Keras, etc.

.. automodule:: gordo.machine.model.base
    :members:
    :undoc-members:
    :show-inheritance:

Custom Gordo models
===================

This group of models are already implemented and ready to be used within
config files, by simply specifying their full path. For example:
``gordo.machine.model.models.KerasAutoEncoder``


.. automodule:: gordo.machine.model.models
    :members:
    :undoc-members:
    :show-inheritance:


.. toctree::
    :maxdepth: 4
    :caption: Model Extensions:

    model-factories.rst
    transformer-funcs.rst
    transformers.rst
    anomaly.rst


Utils
=====
Shared utility functions used by models and other components interacting
with the model's.

.. automodule:: gordo.machine.model.utils
    :members:
    :undoc-members:
    :show-inheritance: