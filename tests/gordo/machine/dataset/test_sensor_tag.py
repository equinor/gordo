import pytest

from gordo.machine.dataset.sensor_tag import (
    normalize_sensor_tags,
    SensorTagNormalizationError,
)
from gordo.machine.dataset.sensor_tag import SensorTag


NON_RESOLVABLE_TAG_NAME1 = "MyBeautifulTag1"
NON_RESOLVABLE_TAG_NAME2 = "MyBeautifulTag2"
asset_nonsense = "ImaginaryAsset"


@pytest.mark.parametrize(
    "good_input_tags,asset,default_asset,expected_output_tags",
    [
        (
            [
                {"name": NON_RESOLVABLE_TAG_NAME1, "asset": asset_nonsense},
                {"name": NON_RESOLVABLE_TAG_NAME2, "asset": asset_nonsense},
            ],
            "ThisAssetCodeWillBeIgnoredSinceAssetMustBeProvidedInDict",
            "AsWillThisSinceAssetMustBeProvidedInDict",
            [
                SensorTag(NON_RESOLVABLE_TAG_NAME1, asset_nonsense),
                SensorTag(NON_RESOLVABLE_TAG_NAME2, asset_nonsense),
            ],
        ),
        (
            ["TRC-123", "GRA-214", "ASGB-212"],
            "ThisWillBeTheAsset",
            "AndThisWillBeIgnoredSinceAllTagsAreResolvable",
            [
                SensorTag("TRC-123", "ThisWillBeTheAsset"),
                SensorTag("GRA-214", "ThisWillBeTheAsset"),
                SensorTag("ASGB-212", "ThisWillBeTheAsset"),
            ],
        ),
        (
            ["TRC-123", "GRA-214", "ASGB-212"],
            None,  # Will deduce asset
            None,
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
            "WillBeIgnoredSinceInputIsSensorTags",
            [
                SensorTag("TRC-123", "1776-troc"),
                SensorTag("GRA-214", "1755-gra"),
                SensorTag("ASGB-212", "1191-asgb"),
            ],
        ),
        (
            [
                [NON_RESOLVABLE_TAG_NAME1, asset_nonsense],
                [NON_RESOLVABLE_TAG_NAME2, asset_nonsense],
            ],
            "WillBeIgnoredSinceInputIsList",
            "WillBeIgnoredSinceInputIsList",
            [
                SensorTag(NON_RESOLVABLE_TAG_NAME1, asset_nonsense),
                SensorTag(NON_RESOLVABLE_TAG_NAME2, asset_nonsense),
            ],
        ),
        (
            ["TRC-123", NON_RESOLVABLE_TAG_NAME2, "ASGB-212"],
            None,
            "fallbackkasset",
            [
                SensorTag("TRC-123", "1776-troc"),
                SensorTag(NON_RESOLVABLE_TAG_NAME2, "fallbackkasset"),
                SensorTag("ASGB-212", "1191-asgb"),
            ],
        ),
    ],
)
def test_normalize_sensor_tags_ok(
    good_input_tags, asset, default_asset, expected_output_tags
):
    tag_list_as_list_of_sensor_tag = normalize_sensor_tags(
        good_input_tags, asset, default_asset=default_asset
    )
    assert tag_list_as_list_of_sensor_tag == expected_output_tags


def test_normalize_sensor_tags_not_ok():
    with pytest.raises(SensorTagNormalizationError):
        tag_list_as_list_of_strings_nonsense = [
            NON_RESOLVABLE_TAG_NAME1,
            NON_RESOLVABLE_TAG_NAME2,
        ]
        normalize_sensor_tags(tag_list_as_list_of_strings_nonsense)
