import importlib
import numpy as np
import pandas as pd
import xarray as xr

from copy import copy
from typing import Any, Union, Tuple
from abc import ABCMeta, abstractmethod
from ..metadata import BuildMetadata


class BaseEvaluator(metaclass=ABCMeta):
    @abstractmethod
    def fit_model(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame, xr.DataArray],
        y: Union[np.ndarray, pd.DataFrame, xr.DataArray],
    ) -> Tuple[Any, BuildMetadata]:
        ...


def create_evaluator(**kwargs) -> BaseEvaluator:
    evaluator_type = "gordo.machine.evaluation.cv.CrossValidation"
    if "type" in kwargs:
        kwargs = copy(kwargs)
        evaluator_type = kwargs.pop("type")
    module_name, class_name = evaluator_type.rsplit(".", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise ValueError(
            "Unable to find evaluator class '%s' in module '%s'"
            % (module_name, class_name)
        )
    return getattr(module, class_name)(**kwargs)
