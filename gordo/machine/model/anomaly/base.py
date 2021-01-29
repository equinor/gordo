import abc
import pandas as pd
import xarray as xr
from typing import Optional, Union
from datetime import timedelta
from sklearn.base import BaseEstimator

from gordo.machine.model.base import GordoBase


class AnomalyDetectorBase(BaseEstimator, GordoBase, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def anomaly(
        self,
        X: Union[pd.DataFrame, xr.DataArray],
        y: Union[pd.DataFrame, xr.DataArray],
        frequency: Optional[timedelta] = None,
    ) -> Union[pd.DataFrame, xr.Dataset]:
        """
        Take X, y and optionally frequency; returning a dataframe containing
        anomaly score(s)
        """
        ...
