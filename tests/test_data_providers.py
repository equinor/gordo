import re
import unittest
from datetime import datetime
from typing import Iterable, List, Pattern, Any

import pandas as pd

from gordo_components.data_provider.base import GordoBaseDataProvider
from gordo_components.data_provider.providers import (
    load_dataframes_from_multiple_providers,
)


class MockProducerRegExp(GordoBaseDataProvider):
    def can_handle_tag(self, tag):
        return self.regexp.match(tag)

    def load_dataframes(
        self, from_ts: datetime, to_ts: datetime, tag_list: List[str]
    ) -> Iterable[pd.DataFrame]:
        for tag in tag_list:
            if self.regexp.match(tag):
                yield pd.DataFrame(columns=[str(self.regexp.pattern)])
            else:
                raise ValueError(f"Unable to find base path from tag {tag}")

    def __init__(self, regexp: Pattern[Any], **kwargs):
        """
        Mock producer which can handle tags which follow the regexp, and yields empty
        dataframes with one column being the regexp pattern.

        Parameters
        ----------
        regexp
            Regular expression for tags the mock producer can accept
        """
        super().__init__(**kwargs)
        self.regexp = regexp  # type: Pattern[Any]


class LoadMultipleDataFramesTest(unittest.TestCase):
    @classmethod
    def setUp(self):
        # Producer only accepting tags which starts with "ab"
        self.ab_producer = MockProducerRegExp(re.compile("ab.*"))
        # Producer only accepting tags which contain a "b"
        self.containing_b_producer = MockProducerRegExp(re.compile(".*b.*"))

    def test_load_multiple_raises_with_no_matches(self):
        """If no provider matches a tag then load_dataframes_from_multiple_providers
        raises a ValueError when the generator is realized"""
        with self.assertRaises(ValueError):
            list(
                load_dataframes_from_multiple_providers(
                    [self.ab_producer, self.containing_b_producer],
                    None,
                    None,
                    ["ab", "tag_not_matching_any_of_the_regexps"],
                )
            )

    def test_load_multiple_matches_loads_from_first(self):
        """When a tag can be read from multiple providers it is the first provider in
        the list of providers which gets the job"""
        dfs = list(
            load_dataframes_from_multiple_providers(
                [self.ab_producer, self.containing_b_producer], None, None, ["abba"]
            )
        )
        self.assertEqual(dfs[0].columns[0], "ab.*")

    def test_load_from_multiple_providers(self):
        """ Two tags, each belonging to different data producers, and both gets loaded
        """
        dfs = list(
            load_dataframes_from_multiple_providers(
                [self.ab_producer, self.containing_b_producer],
                None,
                None,
                ["abba", "cba"],
            )
        )
        self.assertEqual(dfs[0].columns[0], "ab.*")
        self.assertEqual(dfs[1].columns[0], ".*b.*")
