import re
from collections import namedtuple, Mapping
import logging

logger = logging.getLogger(__name__)

SensorTag = namedtuple("SensorTag", ["name", "asset"])

TAG_TO_ASSET = [
    (re.compile(r"^asgb."), "1191-asgb"),
    (re.compile(r"^gra."), "1755-gra"),
    (re.compile(r"^1125."), "1125-kvb"),
    (re.compile(r"^trb."), "1775-trob"),
    (re.compile(r"^trc."), "1776-troc"),
    (re.compile(r"^tra."), "1130-troa"),
]


def _asset_from_tag_name(tag_name):
    """
    Resolves a tag to the asset it belongs to, if possible.
    Returns None if it does not match any of the tag-regexps we know.
    """
    tag_name = tag_name.lower()
    logger.debug(f"Looking for pattern for tag {tag_name}")

    for pattern in TAG_TO_ASSET:
        if pattern[0].match(tag_name):
            logger.info(
                f"Found pattern {pattern[0]} in tag {tag_name}, returning {pattern[1]}"
            )
            return pattern[1]
    raise ValueError(f"Unable to find asset for tag with name {tag_name}")


def normalize_sensor_tag(sensor):
    if isinstance(sensor, Mapping):
        return SensorTag(sensor["name"], sensor["asset"])

    elif isinstance(sensor, str):
        return SensorTag(sensor, _asset_from_tag_name(sensor))

    elif isinstance(sensor, SensorTag):
        return sensor

    raise ValueError(
        f"Sensor {sensor} with type {type(sensor)}cannot be converted to a valid "
        f"SensorTag"
    )
