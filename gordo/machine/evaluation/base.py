import numpy as np
import pandas as pd
import xarray as xr

from typing import Any, Union, Tuple
from abc import ABCMeta, abstractmethod
from ..metadata import BuildMetadata


class BaseEvaluation(metaclass=ABCMeta):

    @abstractmethod
    def fit_model(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame, xr.DataArray],
        y: Union[np.ndarray, pd.DataFrame, xr.DataArray],
    ) -> Tuple[Any, BuildMetadata]:
        ...
