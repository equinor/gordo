import abc
import pandas as pd
from typing import Optional
from datetime import timedelta
from sklearn.base import BaseEstimator

from gordo.machine.model.base import GordoBase


class AnomalyDetectorBase(BaseEstimator, GordoBase, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def anomaly(
        self, X: pd.DataFrame, y: pd.DataFrame, frequency: Optional[timedelta] = None
    ) -> pd.DataFrame:
        """
        Take X, y and optionally frequency; returning a dataframe containing
        anomaly score(s)
        """
        ...
