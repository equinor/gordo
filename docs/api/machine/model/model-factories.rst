Model factories
---------------

Model factories are stand alone functions which take an arbitrary number of
primitive parameters (int, float, list, dict, etc) and return a model which
can then be used in the ``kind`` parameter of some Scikit-Learn like wrapper model.

An example of this is ``KerasAutoEncoder`` which accepts a ``kind`` argument
(as all custom gordo models do) and can be given `feedforward_model`. Meaning
that function will be used to create the underlying Keras model for
``KerasAutoEncoder``


feedforward factories
=====================

.. automodule:: gordo.machine.model.factories.feedforward_autoencoder
    :members:
    :undoc-members:
    :show-inheritance:

lstm factories
==============

.. automodule:: gordo.machine.model.factories.lstm_autoencoder
    :members:
    :undoc-members:
    :show-inheritance:
