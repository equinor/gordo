# -*- coding: utf-8 -*-

import abc
import pandas as pd
from typing import Iterable


class GordoBaseDataProvider(abc.ABC):

    @abc.abstractmethod
    def make_dataframe_generator(self) -> Iterable[pd.DataFrame]:
        """
        Load the required data as an iterable of dataframes
        """
        ...
