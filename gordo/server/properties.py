import copy

import pandas as pd

from typing import Optional, Any
from flask import g

from gordo_core.sensor_tag import SensorTag

from gordo.utils import normalize_sensor_tags


def find_path_in_dict(path: list[str], data: dict) -> Any:
    """
    Find a path in `dict` recursively

    Examples
    --------
    >>> find_path_in_dict(["parent", "child"], {"parent": {"child": 42}})
    42

    Parameters
    ----------
    path: List[str]
    data: dict

    Returns
    -------

    """
    reversed_path = copy.copy(path)
    reversed_path.reverse()
    curr_data = data
    while len(reversed_path):
        key = reversed_path.pop()
        if key not in curr_data:
            exception_path = ".".join(path[: len(path) - len(reversed_path)])
            raise KeyError("'%s' is absent" % exception_path)
        curr_data = curr_data[key]
    return curr_data


def get_frequency():
    """
    The frequency the model was trained with in the dataset
    """
    return pd.tseries.frequencies.to_offset(g.metadata["dataset"]["resolution"])


def load_build_dataset_metadata():
    try:
        build_dataset_metadata = find_path_in_dict(
            ["metadata", "build_metadata", "dataset"], g.metadata
        )
    except KeyError as e:
        raise ValueError("Unable to load build dataset metadata: %s" % str(e))
    return build_dataset_metadata


def get_normalize_additional_fields(dataset: dict[str, Any]):
    additional_fields: dict[str, Optional[str]] = {}
    if "default_tag" in dataset and dataset["default_tag"]:
        additional_fields = dataset["default_tag"]
    # Keeping back-compatibility for a while
    elif dataset.get("asset", None):
        additional_fields["asset"] = dataset["asset"]
    return additional_fields


def get_tags() -> list[SensorTag]:
    """
    The input tags for this model

    Returns
    -------
    list[SensorTag]
    """
    dataset = g.metadata["dataset"]
    tag_list = dataset["tag_list"]
    build_dataset_metadata = load_build_dataset_metadata()
    additional_fields = get_normalize_additional_fields(dataset)
    return normalize_sensor_tags(build_dataset_metadata, tag_list, **additional_fields)


def get_target_tags() -> list[SensorTag]:
    """
    The target tags for this model

    Returns
    -------
    list[SensorTag]
    """
    # TODO refactor this part to have the same tag preparation logic as in TimeSeriesDataset
    orig_target_tag_list = []
    if "target_tag_list" in g.metadata["dataset"]:
        orig_target_tag_list = g.metadata["dataset"]["target_tag_list"]
    if orig_target_tag_list:
        build_dataset_metadata = load_build_dataset_metadata()
        additional_fields = get_normalize_additional_fields(g.metadata["dataset"])
        return normalize_sensor_tags(
            build_dataset_metadata, orig_target_tag_list, **additional_fields
        )
    else:
        return get_tags()
