# -*- coding: utf-8 -*-

import unittest

import pytest
import yaml
import pandas as pd
import numpy as np

from gordo.machine.model.transformers.imputer import InfImputer
from gordo import serializer

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


class GordoFunctionTransformerFuncsTestCase(unittest.TestCase):
    """
    Test all functions within gordo meants for use in a Scikit-Learn
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
        from gordo.machine.model.transformer_funcs.general import multiply_by

        # Provide a require argument
        tf = FunctionTransformer(func=multiply_by, kw_args={"factor": 2})
        self._validate_transformer(tf)

        # Ignore the required argument
        tf = FunctionTransformer(func=multiply_by)
        with self.assertRaises(TypeError):
            self._validate_transformer(tf)


@pytest.mark.parametrize("strategy", ["extremes", "minmax"])
def test_infimputer_basic(strategy):
    """
    Functionality of the InfImputer
    """
    base_x = np.random.random((100, 10)).astype(np.float32)

    flat_view = base_x.ravel()

    pos_inf_idxs = np.random.randint(0, len(flat_view), size=100)
    neg_inf_idxs = np.random.randint(0, len(flat_view), size=100)

    flat_view[pos_inf_idxs] = np.inf
    flat_view[neg_inf_idxs] = -np.inf

    # Our base x should now be littered with pos/neg inf values
    assert np.isposinf(base_x).sum() > 0, "Expected some positive infinity values here"
    assert np.isneginf(base_x).sum() > 0, "Expected some negative infinity values here"

    imputer = InfImputer(strategy=strategy, delta=2.0)

    # Test imputer on numpy array
    X = base_x.copy()
    X = imputer.fit_transform(X)
    assert np.isposinf(X).sum() == 0, "Expected no positive infinity values here"
    assert np.isneginf(X).sum() == 0, "Expected no negative infinity values here"

    if strategy == "extremes":
        # All pos infs in base_x should be filled in X now as the max dtype value
        assert np.all(X[np.where(np.isposinf(base_x))] == np.finfo(X.dtype).max)

        # All neg infs in base_x should be filled in X now as the min dtype value
        assert np.all(X[np.where(np.isneginf(base_x))] == np.finfo(X.dtype).min)

    # min max is a bit more difficult to assert however.
    elif strategy == "minmax":

        # Identify the features in the base array which have pos infs
        features_with_pos_infs = np.where(np.isposinf(base_x))[1]

        # Get the maxes of those features in the imputed X
        filled_maxes = X[:, features_with_pos_infs].max(axis=0)

        # Get the previous maxes
        previous_maxes = np.ma.masked_invalid(base_x[:, features_with_pos_infs]).max(
            axis=0
        )

        # Compare that each new max is the previous max + 2.
        # These were previously 'inf' values, replaced by previous max + 2
        for filled_max, previous_max in zip(filled_maxes, previous_maxes):
            assert np.isclose(filled_max, previous_max + 2.0)

        ### Repeat the process for negative infs ###

        # Identify the features in the base array which have pos infs
        features_with_neg_infs = np.where(np.isneginf(base_x))[1]

        # Get the maxes of those features in the imputed X
        filled_mins = X[:, features_with_neg_infs].min(axis=0)

        # Get the previous maxes
        previous_mins = np.ma.masked_invalid(base_x[:, features_with_neg_infs]).min(
            axis=0
        )

        # Now compare, that each new min is the min - 2. wq
        for filled_min, previous_min in zip(filled_mins, previous_mins):
            assert np.isclose(filled_min, previous_min - 2.0)

    # Test imputer on pandas dataframe
    X = pd.DataFrame(base_x.copy())
    X = imputer.fit_transform(X)
    assert np.isposinf(X).sum() == 0, "Expected no positive infinity values here"
    assert np.isneginf(X).sum() == 0, "Expected no negative infinity values here"


def test_infimputer_fill_values():
    """
    InfImputer when fill values are provided
    """
    base_x = np.random.random((100, 10)).astype(np.float32)

    flat_view = base_x.ravel()

    pos_inf_idxs = [1, 2, 3, 4, 5]
    neg_inf_idxs = [6, 7, 8, 9, 10]

    flat_view[pos_inf_idxs] = np.inf
    flat_view[neg_inf_idxs] = -np.inf

    # Our base x should now be littered with pos/neg inf values
    assert np.isposinf(base_x).sum() > 0, "Expected some positive infinity values here"
    assert np.isneginf(base_x).sum() > 0, "Expected some negative infinity values here"

    imputer = InfImputer(inf_fill_value=9999.0, neg_inf_fill_value=-9999.0)
    X = imputer.fit_transform(base_x)
    np.equal(
        X.ravel()[[pos_inf_idxs]], np.array([9999.0, 9999.0, 9999.0, 9999.0, 9999.0]),
    )
    np.equal(
        X.ravel()[[neg_inf_idxs]],
        np.array([-9999.0, -9999.0, -9999.0, -9999.0, -9999.0]),
    )


@pytest.mark.parametrize(
    "config_str",
    [
        """
    sklearn.pipeline.Pipeline:
      steps:
        - gordo.machine.model.transformers.imputer.InfImputer
    """,
        """
    sklearn.pipeline.Pipeline:
      steps:
        - gordo.machine.model.transformers.imputer.InfImputer:
            inf_fill_value: 10
    """,
        """gordo.machine.model.transformers.imputer.InfImputer""",
    ],
)
def test_imputer_from_definition(config_str: str):
    """
    Ensure it plays well with the gordo serializer
    """
    config = yaml.safe_load(config_str)
    model = serializer.from_definition(config)

    if isinstance(model, Pipeline):
        assert isinstance(model.steps[-1][1], InfImputer)
    else:
        assert isinstance(model, InfImputer)

    serializer.from_definition(serializer.into_definition(model))
