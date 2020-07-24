import logging
import pandas as pd

from typing import Iterable, Dict, Tuple, Union, Type, List
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from collections import defaultdict

logger = logging.getLogger(__name__)


class Preprocessor(metaclass=ABCMeta):
    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def prepare_series(self, series: Iterable[pd.Series]) -> Iterable[pd.Series]:
        ...

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        ...


_types: Dict[str, Type[Preprocessor]] = {}


def preprocessor(preprocessor_type):
    def wrapper(cls):
        if preprocessor_type in _types:
            raise ValueError(
                "Preprocessor with name '%s' has already been added" % preprocessor_type
            )
        _types[preprocessor_type] = cls
        return cls

    return wrapper


def create_preprocessor(preprocessor_type, *args, **kwargs):
    if preprocessor_type not in _types:
        raise ValueError("Can't find a preprocessor with name '%s'" % preprocessor_type)
    return _types[preprocessor_type](*args, **kwargs)


def normalize_preprocessor(value):
    if isinstance(value, dict):
        if "type" not in value:
            raise ValueError("A preprocessor type is empty")
        value = deepcopy(value)
        preprocessor_type = value.pop("type")
        return create_preprocessor(preprocessor_type, **value)
    return value


def gap2str(gap_start: pd.Timestamp, gap_end: pd.Timestamp):
    return "from %s to %s" % (gap_start.isoformat(), gap_end.isoformat())


@preprocessor("fill_gaps")
class FillGapsPreprocessor(Preprocessor):
    def __init__(
        self,
        gap_size: Union[str, pd.Timedelta],
        replace_value: float,
        replace_lower_values: bool = False,
    ):
        if isinstance(gap_size, str):
            gap_size = pd.Timedelta(gap_size)
        self.gap_size = gap_size
        self.replace_value = replace_value
        self.replace_lower_values = replace_lower_values
        self._gaps: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(
            list
        )

    def reset(self):
        self._gaps = defaultdict(list)

    def prepare_series(self, series: Iterable[pd.Series]) -> Iterable[pd.Series]:
        result = []
        for value in series:
            result.append(value)
            name = value.name
            idx = value.index.to_series()
            df = pd.concat([idx, idx.diff().rename("Diff")], axis=1)
            filtered_df = df[df["Diff"] > self.gap_size]
            gaps = (
                (row["Time"], row["Time"] + row["Diff"])
                for _, row in filtered_df.iterrows()
            )

            self._gaps[name].extend(gaps)
        for name, gaps in self._gaps.items():  # type: ignore
            logger.info(
                "Found %d gap%s in '%s' time-series",
                len(gaps),  # type: ignore
                "s" if len(gaps) > 1 else "",  # type: ignore
                name,
            )
            gaps_str = ", ".join(
                gap2str(gap_start, gap_end) for gap_start, gap_end in gaps
            )
            logger.debug("Gaps for '%s': %s", gaps_str)
        else:
            logger.info("Have not found any gaps in all time-series")
        return result

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        replace_value = self.replace_value
        logger.info(
            "Preparing %d tags data DataFrame with %d gaps",
            len(self._gaps),
            sum(len(gaps) for gaps in self._gaps.values()),
        )
        for name, gaps in self._gaps.items():
            if self.replace_lower_values:
                condition = df[name] <= replace_value
            else:
                condition = df[name] == replace_value
            values_count = df.loc[condition, name].count()
            if values_count:
                logger.warning(
                    "Found %d values %s to replace_value='%s' in '%s'",
                    values_count,
                    "lower or equal" if self.replace_lower_values else "equal",
                    replace_value,
                    name,
                )
            if self.replace_lower_values:
                df.loc[df[name] < replace_value, name] = replace_value
            for gap_start, gap_end in gaps:
                df.iloc[
                    (df.index > gap_start) & (df.index < gap_end),
                    df.columns.get_loc(name),
                ] = replace_value
        return df
