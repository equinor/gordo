# -*- coding: utf-8 -*-

"""
Functions to be used within sklearn's FunctionTransformer
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html

Each function SHALL take an X, and optionally a y.

Functions CAN take additional arguments which should be given during the initialization of the FunctionTransformer

Example:

>>> from sklearn.preprocessing import FunctionTransformer
>>> import numpy as np
>>> def my_function(X, another_arg):
...     # Some fancy X manipulation...
...     return X
>>> transformer = FunctionTransformer(func=my_function, kw_args={'another_arg': 'this thing'})
>>> out = transformer.fit_transform(np.random.random(100).reshape(10, 10))
"""


def multiply_by(X, factor):
    """
    Multiplies X by a given factor
    """
    return X * factor
