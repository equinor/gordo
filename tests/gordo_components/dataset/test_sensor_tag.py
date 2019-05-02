import pytest

from gordo_components.dataset.sensor_tag import normalize_sensor_tags
from gordo_components.dataset.sensor_tag import SensorTag


TAG_NAME1 = "MyBeautifulTag1"
TAG_NAME2 = "MyBeautifulTag2"
asset_nonsense = "ImaginaryAsset"


@pytest.mark.parametrize(
    "good_input_tags,asset,expected_output_tags",
    [
        (
            [
                {"name": TAG_NAME1, "asset": asset_nonsense},
                {"name": TAG_NAME2, "asset": asset_nonsense},
            ],
            "ThisAssetCodeWillBeIgnored",
            [
                SensorTag(TAG_NAME1, asset_nonsense),
                SensorTag(TAG_NAME2, asset_nonsense),
            ],
        ),
        (
            ["TRC-123", "GRA-214", "ASGB-212"],
            "ThisWillBeTheAsset",
            [
                SensorTag("TRC-123", "ThisWillBeTheAsset"),
                SensorTag("GRA-214", "ThisWillBeTheAsset"),
                SensorTag("ASGB-212", "ThisWillBeTheAsset"),
            ],
        ),
        (
            ["TRC-123", "GRA-214", "ASGB-212"],
            None,  # Will deduce asset
            [
                SensorTag("TRC-123", "1776-troc"),
                SensorTag("GRA-214", "1755-gra"),
                SensorTag("ASGB-212", "1191-asgb"),
            ],
        ),
        (
            [
                SensorTag("TRC-123", "1776-troc"),
                SensorTag("GRA-214", "1755-gra"),
                SensorTag("ASGB-212", "1191-asgb"),
            ],
            "CouldWriteAssetHereButWeDontCare",
            [
                SensorTag("TRC-123", "1776-troc"),
                SensorTag("GRA-214", "1755-gra"),
                SensorTag("ASGB-212", "1191-asgb"),
            ],
        ),
        (
            [[TAG_NAME1, asset_nonsense], [TAG_NAME2, asset_nonsense]],
            None,
            [
                SensorTag(TAG_NAME1, asset_nonsense),
                SensorTag(TAG_NAME2, asset_nonsense),
            ],
        ),
    ],
)
def test_normalize_sensor_tags_ok(good_input_tags, asset, expected_output_tags):
    tag_list_as_list_of_sensor_tag = normalize_sensor_tags(good_input_tags, asset)
    assert tag_list_as_list_of_sensor_tag == expected_output_tags


def test_normalize_sensor_tags_not_ok():
    with pytest.raises(ValueError):
        tag_list_as_list_of_strings_nonsense = [TAG_NAME1, TAG_NAME2]
        normalize_sensor_tags(tag_list_as_list_of_strings_nonsense)
