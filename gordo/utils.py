from collections import OrderedDict
from typing import Optional, List, Dict, cast, Union

import inject
from gordo_dataset.assets_config import AssetsConfig
from gordo_dataset.dataset_metadata import sensor_tags_from_build_metadata
from gordo_dataset.sensor_tag import (
    SensorTag,
    Tag,
    normalize_sensor_tag,
    extract_tag_name,
)

TagsList = List[Union[Dict, List, str, SensorTag]]


@inject.autoparams("assets_config")
def normalize_sensor_tags(
    build_dataset_metadata: dict,
    tag_list: TagsList,
    assets_config: AssetsConfig,
    asset: Optional[str] = None,
) -> List[SensorTag]:
    """
    Load tag information from the metadata

    Parameters
    ----------
    build_dataset_metadata: dict
        build_metadata.dataset part of the metadata
    tag_list: TagsList
        Tag list that needs to be loaded
    assets_config: AssetsConfig
    asset: Optional[str]
        Asset name. Useful if the current `dataset` has specified one

    Returns
    -------
    List[SensorTag]

    """
    tags: Dict[str, Tag] = OrderedDict()
    for tag in tag_list:
        tag_name: str
        if type(tag) is str:
            tags[cast(str, tag)] = tag
        else:
            normalized_tag = normalize_sensor_tag(assets_config, tag, asset)
            tag_name = extract_tag_name(normalized_tag)
            tags[tag_name] = normalized_tag
    normalized_sensor_tags = sensor_tags_from_build_metadata(
        build_dataset_metadata,
        set(tags.keys()),
        with_legacy_tag_normalization=True,
        assets_config=assets_config,
        asset=asset,
    )
    normalized_tag_list: List[SensorTag] = []
    for tag_name in tags.keys():
        normalized_tag_list.append(normalized_sensor_tags[tag_name])
    return normalized_tag_list
