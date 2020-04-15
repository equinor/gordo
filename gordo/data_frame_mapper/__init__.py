import logging
import sklearn_pandas

from pydoc import locate
from copy import copy, deepcopy
from typing import List, Union, Optional

logger = logging.getLogger(__name__)


class DataFrameMapper(sklearn_pandas.DataFrameMapper):
    _default_kwargs = {"df_out": True}

    def __init__(
            self,
            columns: List[Union[str, List[str]]],
            classes: Optional[List[dict]] = None,
            **kwargs
    ):
        self.columns = columns
        self.classes = classes
        features = self._build_features(columns, classes)
        base_kwargs = copy(self._default_kwargs)
        base_kwargs.update(kwargs)
        super().__init__(features=features, **base_kwargs)

    @staticmethod
    def _build_features(
            columns: List[Union[str, List[str]]], classes: Optional[List[dict]] = None,
    ):
        if classes is not None:
            classes = deepcopy(classes)
            for i, v in enumerate(classes):
                if isinstance(v, dict):
                    if "class" not in v:
                        raise ValueError('"class" attribute is empty')
                    if isinstance(v["class"], str):
                        cls = locate(v["class"])
                        classes[i]["class"] = cls
        logger.debug("_build_features for columns=%s, classes=%s", columns, classes)
        return sklearn_pandas.gen_features(columns=columns, classes=classes)

    def __getstate__(self):
        state = super().__getstate__()
        state["columns"] = self.columns
        state["classes"] = self.classes
        del state["features"]
        return state

    def __setstate__(self, state):
        features = self._build_features(state.get("columns"), state.get("classes"))
        state["features"] = features
        super().__setstate__(state)


__all__ = ['DataFrameMapper']
