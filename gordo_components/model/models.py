# -*- coding: utf-8 -*-

import inspect
import logging
from typing import Union

from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from gordo_components.model.base import GordoBaseModel
from gordo_components.model.factories import *

from gordo_components.model.register import register_model_builder


logger = logging.getLogger(__name__)


class KerasModel(KerasRegressor, GordoBaseModel):

    def __init__(self, kind: Union[str, callable], **kwargs):
        """
        Initialized a Scikit-Learn API compatitble Keras model with a pre-registered function or a builder function
        directly.

        Example use:
        ```
        from gordo_components.model import models


        def this_function_returns_a_special_keras_model(n_features, extra_param1, extra_param2):
            ...

        scikit_based_model = KerasModel(kind=this_function_returns_a_special_keras_model,
                                        extra_param1='special_parameter',
                                        extra_param2='another_parameter')

        scikit_based_model.fit(X, y)
        scikit_based_model.predict(X)
        ```

        kind: Union[callable, str] - The structure of the model to build. As designated by any registered builder
                                     functions, registered with gordo_compontents.model.register.register_model_builder
                                     Alternatively, one may pass a builder function directly to this argument. Such a
                                     function should accept `n_features` as it's first argument, and pass any additional
                                     parameters to `**kwargs`

        kwargs: dict -  Any additional args which are passed to the factory 
                        building function and/or any additional args to be passed
                        to Keras' fit() method
        """
        self.build_fn = None
        self.kwargs = kwargs

        class_name = self.__class__.__name__
        
        if callable(kind):
            register_model_builder(type=class_name)(kind)
            self.kind = self.kind.__name__
        else:
            if kind not in register_model_builder.factories[class_name]:
                raise ValueError(f'kind: {kind} is not an available model for type: {class_name}!')
            self.kind = kind

    @property
    def sk_params(self):
        return self.kwargs

    def __call__(self):
        build_fn = register_model_builder.factories[self.__class__.__name__][self.kind]
        return build_fn(**self.sk_params)

    def fit(self, X, y, sample_weight=None, **kwargs):
        self.kwargs.update({'n_features': X.shape[1]})
        return super().fit(X, y, sample_weight=None, **kwargs)
