# -*- coding: utf-8 -*-

import abc
from typing import Optional, Union

import numpy as np
import pandas as pd


class GordoBase(abc.ABC):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        """Initialize the model"""
        ...

    @abc.abstractmethod
    def get_params(self, deep=False):
        """Return a dict containing all parameters used to initialized object"""
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
