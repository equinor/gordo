import logging
import sklearn_pandas

from copy import copy
from sklearn.base import BaseEstimator
from typing import List, Union

logger = logging.getLogger(__name__)


class DataFrameMapper(sklearn_pandas.DataFrameMapper):
    _default_kwargs = {"df_out": True}

    def __init__(
        self,
        columns: List[Union[str, List[str]]],
        transformers: List[BaseEstimator] = None,
        **kwargs
    ):
        self.columns = columns
        self.transformers = transformers
        features = self._build_features(columns, transformers)
        base_kwargs = copy(self._default_kwargs)
        base_kwargs.update(kwargs)
        super().__init__(features=features, **base_kwargs)

    @staticmethod
    def _build_features(
        columns: List[Union[str, List[str]]], transformers: List[BaseEstimator]
    ):
        features = []
        for column in columns:
            features.append((column, transformers))
        return features

    def __getstate__(self):
        state = super().__getstate__()
        state["columns"] = self.columns
        state["transformers"] = self.transformers
        del state["features"]
        return state

    def __setstate__(self, state):
        features = self._build_features(state.get("columns"), state.get("transformers"))
        state["features"] = features
        super().__setstate__(state)


__all__ = ["DataFrameMapper"]
