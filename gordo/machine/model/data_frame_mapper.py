from pydoc import locate
from sklearn_pandas import DataFrameMapper, gen_features
from copy import copy
from typing import List, Union, Optional


class DataFrameMapper(DataFrameMapper):

    _default_kwargs = {
        "df_out": True
    }

    def __init__(self, columns: List[Union[str, List[str]]], classes: Optional[List[dict]] = None, **kwargs):
        if classes is not None:
            classes = copy(classes)
            self._prepare_classes(classes)
        features = gen_features(columns=columns, classes=classes)
        base_kwargs = copy(self._default_kwargs)
        base_kwargs.update(kwargs)
        super().__init__(features=features, **kwargs)

    @staticmethod
    def _prepare_classes(classes: List[dict]):
        for i, v in enumerate(classes):
            if "class" not in v:
                raise ValueError("\"class\" attribute is empty")
            if isinstance(v["class"], str):
                cls = locate(v["class"])
                classes[i]["class"] = cls
