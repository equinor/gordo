import pytest

from gordo_components.dataset.sensor_tag import normalize_sensor_tags
from gordo_components.dataset.sensor_tag import SensorTag


TAG_NAME1 = "MyBeautifulTag1"
TAG_NAME2 = "MyBeautifulTag2"
asset_nonsense = "ImaginaryAsset"


def test_normalize_sensor_tags_from_dict():
    tag_list_as_list_of_dict = [
        {"name": TAG_NAME1, "asset": asset_nonsense},
        {"name": TAG_NAME2, "asset": asset_nonsense},
    ]
    tag_list_as_list_of_sensor_tag = normalize_sensor_tags(tag_list_as_list_of_dict)
    assert tag_list_as_list_of_sensor_tag == [
        SensorTag(TAG_NAME1, asset_nonsense),
        SensorTag(TAG_NAME2, asset_nonsense),
    ]


def test_normalize_sensor_tags_from_string():
    with pytest.raises(ValueError):
        tag_list_as_list_of_strings_nonsense = [TAG_NAME1, TAG_NAME2]
        normalize_sensor_tags(tag_list_as_list_of_strings_nonsense)

    tag_list_as_list_of_known_tag_strings = ["TRC-123", "GRA-214", "ASGB-212"]
    tag_list_as_list_of_sensor_tag = normalize_sensor_tags(
        tag_list_as_list_of_known_tag_strings
    )

    assert tag_list_as_list_of_sensor_tag == [
        SensorTag("TRC-123", "1776-troc"),
        SensorTag("GRA-214", "1755-gra"),
        SensorTag("ASGB-212", "1191-asgb"),
    ]


def test_normalize_sensor_tag_from_sensor_tag():
    sensor_tags = [
        SensorTag("TRC-123", "1776-troc"),
        SensorTag("GRA-214", "1755-gra"),
        SensorTag("ASGB-212", "1191-asgb"),
    ]

    normalized_sensor_tags = normalize_sensor_tags(sensor_tags)

    assert sensor_tags == normalized_sensor_tags
