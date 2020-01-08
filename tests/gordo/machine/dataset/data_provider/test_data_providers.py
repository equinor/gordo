import re
import unittest
from datetime import datetime
from typing import Iterable, List, Pattern, Any

import pandas as pd
import pytest

from gordo.machine.dataset.data_provider.base import GordoBaseDataProvider
from gordo.machine.dataset.data_provider import providers
from gordo.machine.dataset.data_provider.providers import (
    load_series_from_multiple_providers,
)
from gordo.machine.dataset.sensor_tag import SensorTag


class MockProducerRegExp(GordoBaseDataProvider):
    def can_handle_tag(self, tag: SensorTag):
        return self.regexp.match(tag.name)

    def load_series(
        self,
        train_start_date: datetime,
        train_end_date: datetime,
        tag_list: List[SensorTag],
        dry_run=False,
    ) -> Iterable[pd.Series]:
        for tag in tag_list:
            if self.regexp.match(tag.name):
                yield pd.Series(name=str(self.regexp.pattern))
            else:
                raise ValueError(f"Unable to find base path from tag {tag.name}")

    def __init__(self, regexp: Pattern[Any], **kwargs):
        """
        Mock producer which can handle tags which follow the regexp, and yields empty
        dataframes with one column being the regexp pattern.

        Parameters
        ----------
        regexp
            Regular expression for tags the mock producer can accept
        """
        self.regexp = regexp  # type: Pattern[Any]


class LoadMultipleDataFramesTest(unittest.TestCase):
    @classmethod
    def setUp(self):
        # Producer only accepting tags which starts with "ab"
        self.ab_producer = MockProducerRegExp(re.compile("ab.*"))
        # Producer only accepting tags which contain a "b"
        self.containing_b_producer = MockProducerRegExp(re.compile(".*b.*"))

    def test_load_multiple_raises_with_no_matches(self):
        """If no provider matches a tag then load_series_from_multiple_providers
        raises a ValueError when the generator is realized"""
        with self.assertRaises(ValueError):
            list(
                load_series_from_multiple_providers(
                    [self.ab_producer, self.containing_b_producer],
                    None,
                    None,
                    [
                        SensorTag("ab", None),
                        SensorTag("tag_not_matching_any_of_the_regexps", None),
                    ],
                )
            )

    def test_load_multiple_matches_loads_from_first(self):
        """When a tag can be read from multiple providers it is the first provider in
        the list of providers which gets the job"""
        series_collection = list(
            load_series_from_multiple_providers(
                [self.ab_producer, self.containing_b_producer],
                None,
                None,
                [SensorTag("abba", None)],
            )
        )
        self.assertEqual(series_collection[0].name, "ab.*")

    def test_load_from_multiple_providers(self):
        """ Two tags, each belonging to different data producers, and both gets loaded
        """
        series_collection = list(
            load_series_from_multiple_providers(
                [self.ab_producer, self.containing_b_producer],
                None,
                None,
                [SensorTag("abba", None), SensorTag("cba", None)],
            )
        )
        self.assertEqual(series_collection[0].name, "ab.*")
        self.assertEqual(series_collection[1].name, ".*b.*")


@pytest.mark.parametrize(
    "provider,expected_params",
    (
        (
            providers.RandomDataProvider(200, max_size=205),
            {"min_size": 200, "max_size": 205},
        ),
        (
            providers.InfluxDataProvider("measurement", value_name="Value"),
            {"measurement": "measurement", "value_name": "Value"},
        ),
    ),
)
def test_data_provider_serializations(
    provider: GordoBaseDataProvider, expected_params: dict
):
    """
    Test a given provider can be serialized to dict and back
    """

    encoded = provider.to_dict()

    # Verify the expected parameter kwargs match
    for k, v in expected_params.items():
        assert encoded[k] == v

    # Should have inserted the name of the class as 'type'
    assert provider.__class__.__name__ == encoded["type"]

    # Should be able to recreate the object from encoded directly
    cloned = provider.__class__.from_dict(encoded)
    assert type(cloned) == type(provider)
