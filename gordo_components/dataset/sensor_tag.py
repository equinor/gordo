import re
import logging
from collections import namedtuple
from typing import Union, List, Dict


logger = logging.getLogger(__name__)

SensorTag = namedtuple("SensorTag", ["name", "asset"])

TagPatternToAsset = namedtuple("TagToAsset", ["tag_regexp", "asset_name"])

TAG_TO_ASSET = [
    TagPatternToAsset(re.compile(r"^asgb.", re.IGNORECASE), "1191-asgb"),
    TagPatternToAsset(re.compile(r"^gra.", re.IGNORECASE), "1755-gra"),
    TagPatternToAsset(re.compile(r"^1125.", re.IGNORECASE), "1125-kvb"),
    TagPatternToAsset(re.compile(r"^trb.", re.IGNORECASE), "1775-trob"),
    TagPatternToAsset(re.compile(r"^trc.", re.IGNORECASE), "1776-troc"),
    TagPatternToAsset(re.compile(r"^tra.", re.IGNORECASE), "1130-troa"),
    TagPatternToAsset(re.compile(r"^1218.", re.IGNORECASE), "1218-gkr"),
    TagPatternToAsset(re.compile(r"^per.", re.IGNORECASE), "1295-pera"),
    TagPatternToAsset(re.compile(r"^gfa.", re.IGNORECASE), "1110-gfa"),
    TagPatternToAsset(re.compile(r"^ninenine.+::.+", re.IGNORECASE), "ninenine"),
    TagPatternToAsset(re.compile(r"^uon_ef.+::.+", re.IGNORECASE), "uon_ef"),
    TagPatternToAsset(re.compile(r"^1138.", re.IGNORECASE), "1138-val"),
    TagPatternToAsset(re.compile(r"^nor.", re.IGNORECASE), "1180-nor"),
    TagPatternToAsset(re.compile(r"^asga.", re.IGNORECASE), "1190-asga"),
    TagPatternToAsset(re.compile(r"^1900.", re.IGNORECASE), "1900-jsv"),
    TagPatternToAsset(re.compile(r"^1901.", re.IGNORECASE), "1901-jsv"),
    TagPatternToAsset(re.compile(r"^1902.", re.IGNORECASE), "1902-jsv"),
    TagPatternToAsset(re.compile(r"^1903.", re.IGNORECASE), "1903-jsv"),
    TagPatternToAsset(re.compile(r"^vis.", re.IGNORECASE), "1230-vis"),
]


def _asset_from_tag_name(tag_name: str):
    """
    Resolves a tag to the asset it belongs to, if possible.
    Returns None if it does not match any of the tag-regexps we know.
    """
    logger.debug(f"Looking for pattern for tag {tag_name}")

    for pattern in TAG_TO_ASSET:
        if pattern.tag_regexp.match(tag_name):
            logger.info(
                f"Found pattern {pattern.tag_regexp} in tag {tag_name}, "
                f"returning {pattern.asset_name}"
            )
            return pattern.asset_name
    raise ValueError(f"Unable to find asset for tag with name {tag_name}")


def _normalize_sensor_tag(sensor: Union[Dict, List, str, SensorTag], asset: str = None):
    if isinstance(sensor, Dict):
        return SensorTag(sensor["name"], sensor["asset"])

    elif isinstance(sensor, str):
        if asset is None:
            return SensorTag(sensor, _asset_from_tag_name(sensor))
        else:
            return SensorTag(sensor, asset)

    elif isinstance(sensor, List):
        return SensorTag(sensor[0], sensor[1])

    elif isinstance(sensor, SensorTag):
        return sensor

    raise ValueError(
        f"Sensor {sensor} with type {type(sensor)}cannot be converted to a valid "
        f"SensorTag"
    )


def normalize_sensor_tags(
    sensors: List[Union[Dict, str, SensorTag]], asset: str = None
) -> List[SensorTag]:
    """
    Converts a list of sensors in different formats, into a list of SensorTag elements.
    Note, if you input a list of SensorTag elements, these will just be returned.

    Parameters
    ----------
    sensors : List[Union[Mapping, str, SensorTag]]
            List of sensors
    asset : str
            Optional asset code to put on sensors that don't have it

    Returns
    -------
    List[SensorTag]
            List of SensorTags

    """
    logging.info(
        f"Normalizing list of sensors in some format into SensorTags: {sensors}"
    )
    return [
        _normalize_sensor_tag(sensor_tag_element, asset)
        for sensor_tag_element in sensors
    ]


def to_list_of_strings(sensor_tag_list: List[SensorTag]):
    return [sensor_tag.name for sensor_tag in sensor_tag_list]
