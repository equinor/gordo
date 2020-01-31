# -*- coding: utf-8 -*-

import abc
from typing import Optional, Union

import numpy as np
import pandas as pd

from gordo.base import GordoBase


class GordoBaseModel(GordoBase, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        """Initialize the model"""
        ...

    @abc.abstractmethod
    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame],
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Score the model; must implement the correct default scorer based on model type
        """
        ...

    @abc.abstractmethod
    def get_metadata(self):
        """Get model specific metadata, if any"""
