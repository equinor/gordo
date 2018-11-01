# -*- coding: utf-8 -*-

import inspect
from typing import Union

from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from gordo_components.model.base import GordoBaseModel
from gordo_components.model.factories import KerasFeedForwardFactory, KerasLstmFactory


FACTORIES = {
    'feedforward': KerasFeedForwardFactory,
    'lstm': KerasLstmFactory
}


class KerasModel(KerasRegressor, GordoBaseModel):
    """ 
    The Keras Model implementation as Scikit-Learn / Gordo model API
    """

    def __init__(self,
                 n_features: int,
                 kind: Union[str, callable] = 'feedforward_symetric',
                 **kwargs):
        """
        Initialized the model

        n_features: int - Number of features in the expected dataset for model,
                          this will be passed to the factory build functions or
                          the 'kind' parameter if a callable
        kind: Union[callable, str] - The structure of model to build, valid 
            values ['symetric'] if defined as a string. If a callable is passed,
            the first arg should be 'n_features' and then accept optional kwargs
            as defined in 'kwargs' which will be passed to this callable. This
            callable should yield a keras model.
        kwargs: dict -  Any additional args which are passed to the factory 
                        building method and/or any additional args to be passed 
                        to Keras' fit() method
        """
        kwargs.update({'n_features': n_features})
        self.kwargs = kwargs

        # Determine the build_fn, which shall return a Keras model
        if callable(kind):
            self.build_fn = kind
        else:
            factory = FACTORIES.get(kind.split('_')[0])
            self.build_fn = getattr(factory, f'build_{kind}_model')

    @property
    def sk_params(self):
        return self.kwargs

    def __call__(self):
        return self.build_fn(**self.sk_params)
