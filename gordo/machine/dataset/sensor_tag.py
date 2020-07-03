import re
import logging
from collections import namedtuple
from typing import Union, List, Dict


logger = logging.getLogger(__name__)

SensorTag = namedtuple("SensorTag", ["name", "asset"])

TagPatternToAsset = namedtuple("TagToAsset", ["tag_regexp", "asset_name"])

TAG_TO_ASSET = [
    TagPatternToAsset(re.compile(r"^ninenine.+::.+", re.IGNORECASE), "ninenine"),
    TagPatternToAsset(re.compile(r"^uon_ef.+::.+", re.IGNORECASE), "uon_ef"),
    TagPatternToAsset(re.compile(r"^gfa.", re.IGNORECASE), "1110-gfa"),
    TagPatternToAsset(re.compile(r"^gfb.", re.IGNORECASE), "1111-gfb"),
    TagPatternToAsset(re.compile(r"^gfc.", re.IGNORECASE), "1112-gfc"),
    TagPatternToAsset(re.compile(r"^1125.", re.IGNORECASE), "1125-kvb"),
    TagPatternToAsset(re.compile(r"^tra.", re.IGNORECASE), "1130-troa"),
    TagPatternToAsset(re.compile(r"^asgb.", re.IGNORECASE), "1191-asgb"),
    TagPatternToAsset(re.compile(r"^kri.", re.IGNORECASE), "1175-kri"),
    TagPatternToAsset(re.compile(r"^1138.", re.IGNORECASE), "1138-val"),
    TagPatternToAsset(re.compile(r"^hd.", re.IGNORECASE), "1170-hd"),
    TagPatternToAsset(re.compile(r"^nor.", re.IGNORECASE), "1180-nor"),
    TagPatternToAsset(re.compile(r"^asga.", re.IGNORECASE), "1190-asga"),
    TagPatternToAsset(re.compile(r"^1218.", re.IGNORECASE), "1218-gkr"),
    TagPatternToAsset(re.compile(r"^1219.", re.IGNORECASE), "1219-aha"),
    TagPatternToAsset(re.compile(r"^vis.", re.IGNORECASE), "1230-vis"),
    TagPatternToAsset(re.compile(r"^per-pa.", re.IGNORECASE), "1294-pera"),
    TagPatternToAsset(re.compile(r"^per-pb.", re.IGNORECASE), "1298-perb"),
    TagPatternToAsset(re.compile(r"^per.", re.IGNORECASE), "1299-perf"),
    TagPatternToAsset(re.compile(r"^gra.", re.IGNORECASE), "1755-gra"),
    TagPatternToAsset(re.compile(r"^hea.", re.IGNORECASE), "1760-hea"),
    TagPatternToAsset(re.compile(r"^osc.", re.IGNORECASE), "1765-OSC"),
    TagPatternToAsset(re.compile(r"^oss.", re.IGNORECASE), "1766-OSS"),
    TagPatternToAsset(re.compile(r"^ose.", re.IGNORECASE), "1767-OSE"),
    TagPatternToAsset(re.compile(r"^trb.", re.IGNORECASE), "1775-trob"),
    TagPatternToAsset(re.compile(r"^trc.", re.IGNORECASE), "1776-troc"),
    TagPatternToAsset(re.compile(r"^1900.", re.IGNORECASE), "1900-jsv"),
    TagPatternToAsset(re.compile(r"^1901.", re.IGNORECASE), "1901-jsv"),
    TagPatternToAsset(re.compile(r"^1902.", re.IGNORECASE), "1902-jsv"),
    TagPatternToAsset(re.compile(r"^1903.", re.IGNORECASE), "1903-jsv"),
    TagPatternToAsset(re.compile(r"^1904.", re.IGNORECASE), "1904-jsv"),
]


def _asset_from_tag_name(tag_name: str, default_asset: str = None):
    """
    Resolves a tag to the asset it belongs to, if possible.

    If `default_asset` is provided then it will be used as the fallback in case it
    is impossible to resolve tag_name to an asset.

    Parameters
    ----------
    tag_name : str
        Tag to deduce the asset from.
    default_asset : str
        Asset to report if we are not able to resolve tag_name to an assets.

    Returns
    -------
    str
        The asset for provided tag.

    Raises
    ------
    SensorTagNormalizationError
        If we are not able to resolve the tag to an asset and default_asset is not set.
    """
    logger.debug(f"Looking for pattern for tag {tag_name}")

    for pattern in TAG_TO_ASSET:
        if pattern.tag_regexp.match(tag_name):
            logger.debug(
                f"Found pattern {pattern.tag_regexp} in tag {tag_name}, "
                f"returning {pattern.asset_name}"
            )
            return pattern.asset_name
    if default_asset:
        return default_asset
    else:
        raise SensorTagNormalizationError(
            f"Unable to find asset for tag with name {tag_name}"
        )


def _normalize_sensor_tag(
    sensor: Union[Dict, List, str, SensorTag],
    asset: str = None,
    default_asset: str = None,
):
    if isinstance(sensor, Dict):
        return SensorTag(sensor["name"], sensor["asset"])

    elif isinstance(sensor, str):
        if asset is None:
            return SensorTag(
                sensor, _asset_from_tag_name(sensor, default_asset=default_asset)
            )
        else:
            return SensorTag(sensor, asset)

    elif isinstance(sensor, List):
        return SensorTag(sensor[0], sensor[1])

    elif isinstance(sensor, SensorTag):
        return sensor

    raise SensorTagNormalizationError(
        f"Sensor {sensor} with type {type(sensor)} cannot be converted to a valid "
        f"SensorTag"
    )


def normalize_sensor_tags(
    sensors: List[Union[Dict, str, SensorTag]],
    asset: str = None,
    default_asset: str = None,
) -> List[SensorTag]:
    """
    Converts a list of sensors in different formats, into a list of SensorTag elements.

    Notes
    -----
    If you input a list of SensorTag elements, these will just be returned.
    If you input a list of lists or a list of dicts they are expected to contain the
    tag-name and the asset, and no deduction will happen.

    Parameters
    ----------
    sensors : List[Union[Mapping, str, SensorTag]]
            List of sensors
    asset : str
            Optional asset code to put on sensors that don't have it. If this is
            provided we will not attempt to deduce the asset from the tagame.
    default_asset : str
            Optional asset code to put on sensors that dont have it and which we are not
            able to resolve to an asset.

    Returns
    -------
    List[SensorTag]
            List of SensorTags

    """
    logger.debug(
        f"Normalizing list of sensors in some format into SensorTags: {sensors}"
    )
    return [
        _normalize_sensor_tag(sensor_tag_element, asset, default_asset)
        for sensor_tag_element in sensors
    ]


def to_list_of_strings(sensor_tag_list: List[SensorTag]):
    return [sensor_tag.name for sensor_tag in sensor_tag_list]


class SensorTagNormalizationError(ValueError):
    """Error indicating that something went wrong normalizing a sensor tag"""

    pass
