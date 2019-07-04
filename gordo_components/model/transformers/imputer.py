# -*- coding: utf-8 -*-

from typing import Union

import pandas as pd
import numpy as np
from numpy.ma import masked_invalid

from sklearn.base import TransformerMixin


class InfImputer(TransformerMixin):
    def __init__(
        self,
        inf_fill_value=None,
        neg_inf_fill_value=None,
        strategy="minmax",
        delta: float = 2.0,
    ):
        """
        Fill inf/-inf values of a 2d array/dataframe with imputed or provided values
        By default it will find the min and max of each feature/column and fill -infs/infs
        with those values +/- ``delta``

        Parameters
        ----------
        inf_fill_value: numeric
            Value to fill 'inf' values
        neg_inf_fill_value: numeric
            Value to fill '-inf' values
        strategy: str
            How to fill values, irrelevant if fill value is provided.
            choices: 'extremes', 'minmax'
            -'extremes' will use the min and max values for the current datatype.
            such that 'inf' in a float32 dataset will have float32's largest value inserted.
            - 'minmax' will look at the min and max values in the feature where the -inf / inf
            appears and fill with the max/min found in that feature.
        delta: float
            Only applicable if ``strategy='minmax'``
            Will add/subtract the max/min value, by feature, by this delta. If the max value
            in a feature was 10 and ``delta=2`` any inf value will be filled with 12.
            Likewise, if the min feature was -10 any -inf will be filled with -12.
        """
        self.inf_fill_value = inf_fill_value
        self.neg_inf_fill_value = neg_inf_fill_value
        self.strategy = strategy
        self.delta = delta

    def get_params(self, deep=True):
        return {
            "inf_fill_value": self.inf_fill_value,
            "neg_inf_fill_value": self.neg_inf_fill_value,
            "strategy": self.strategy,
            "delta": self.delta,
        }

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):

        # Store the min/max for features in training.
        if self.strategy == "minmax":

            data = pd.DataFrame(X)  # ensure a dataframe

            # Calculate max/min allowable values
            max_allowable_value = np.finfo(data.values.dtype).max
            min_allowable_value = np.finfo(data.values.dtype).min

            # Get the max/min values in each feature, ignoring infs
            _posinf_fill_values = data.apply(lambda col: masked_invalid(col).max())
            _neginf_fill_values = data.apply(lambda col: masked_invalid(col).min())

            # Calculate a 1d arrays of fill values for each feature
            self._posinf_fill_values = _posinf_fill_values.apply(
                lambda val: val + self.delta
                if max_allowable_value - self.delta > val
                else max_allowable_value
            )
            self._neginf_fill_values = _neginf_fill_values.apply(
                lambda val: val - self.delta
                if min_allowable_value + self.delta < val
                else min_allowable_value
            )

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray], y=None):

        # Ensure we're dealing with numpy array if it's a dataframe or similar
        X = X.values if hasattr(X, "values") else X

        # Apply specific fill values if provided.
        if self.inf_fill_value is not None:
            X[np.isposinf(X)] = self.inf_fill_value
        if self.neg_inf_fill_value is not None:
            X[np.isneginf(X)] = self.neg_inf_fill_value

        # May still be left over infs, if only one fill value was supplied for example
        if self.strategy is not None:
            return getattr(self, f"_fill_{self.strategy}")(X)
        return X

    def _fill_extremes(self, X: np.ndarray):
        """
        Fill negative and postive infs with their dtype's min/max values
        """
        X[np.isposinf(X)] = np.finfo(X.dtype).max
        X[np.isneginf(X)] = np.finfo(X.dtype).min
        return X

    def _fill_minmax(self, X: np.ndarray):
        """
        Fill inf/-inf values in features of the array based on their min & max values.
        Compounded by the ``power`` value so long as the result doesn't exceed the
        current array's dtype's max/min. Otherwise it will use those.
        """

        # For each feature fill inf/-inf with pre-calculate fill values
        for feature_idx, (posinf_fill, neginf_fill) in enumerate(
            zip(self._posinf_fill_values, self._neginf_fill_values)
        ):
            X[:, feature_idx][np.isposinf(X[:, feature_idx])] = posinf_fill
            X[:, feature_idx][np.isneginf(X[:, feature_idx])] = neginf_fill
        return X
