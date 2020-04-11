import logging

from pydoc import locate
from sklearn_pandas import DataFrameMapper, gen_features
from copy import copy
from typing import List, Union, Optional

logger = logging.getLogger(__name__)


class DataFrameMapper(DataFrameMapper):

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
            classes = copy(classes)
            for i, v in enumerate(classes):
                if "class" not in v:
                    raise ValueError('"class" attribute is empty')
                if isinstance(v["class"], str):
                    cls = locate(v["class"])
                    classes[i]["class"] = cls
        logger.debug("_build_features for columns=%s, classes=%s)", columns, classes)
        return gen_features(columns=columns, classes=classes)

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
