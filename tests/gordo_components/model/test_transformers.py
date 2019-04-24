# -*- coding: utf-8 -*-

import unittest
import numpy as np

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


class GordoFunctionTransformerFuncsTestCase(unittest.TestCase):
    """
    Test all functions within gordo_components meants for use in a Scikit-Learn
    FunctionTransformer work as expected
    """

    def _validate_transformer(self, transformer):
        """
        Inserts a transformer into the middle of a pipeline and runs it
        """
        pipe = Pipeline([("pca1", PCA()), ("custom", transformer), ("pca2", PCA())])
        X = np.random.random(size=100).reshape(10, 10)
        pipe.fit_transform(X)

    def test_multiply_by_function_transformer(self):
        from gordo_components.model.transformer_funcs.general import multiply_by

        # Provide a require argument
        tf = FunctionTransformer(func=multiply_by, kw_args={"factor": 2})
        self._validate_transformer(tf)

        # Ignore the required argument
        tf = FunctionTransformer(func=multiply_by)
        with self.assertRaises(TypeError):
            self._validate_transformer(tf)
