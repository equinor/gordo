import unittest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from gordo_components.dataset.filter_rows import pandas_filter_rows


class TestFilterRows(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.df = pd.DataFrame(list(np.ndindex((10, 2))), columns=["Tag  1", "Tag 2"])

    def test_filter_rows_basic(self):
        df = self.df
        self.assertEqual(len(pandas_filter_rows(df, "`Tag  1` <= `Tag 2`")), 3)
        self.assertEqual(len(pandas_filter_rows(df, "`Tag  1` == `Tag 2`")), 2)
        self.assertEqual(
            len(pandas_filter_rows(df, "(`Tag  1` <= `Tag 2`) | `Tag 2` < 2")), 20
        )
        self.assertEqual(
            len(pandas_filter_rows(df, "(`Tag  1` <= `Tag 2`) | `Tag 2` < 0.9")), 9
        )

        assert_frame_equal(
            pandas_filter_rows(df, "(`Tag  1` <= `Tag 2`)"),
            pandas_filter_rows(df, "~(`Tag  1` > `Tag 2`)"),
        )

    def test_filter_rows_catches_illegal(self):
        with self.assertRaises(ValueError):
            pandas_filter_rows(self.df, "sys.exit(0)")
        with self.assertRaises(ValueError):
            pandas_filter_rows(self.df, "lambda x:x")
        with self.assertRaises(ValueError):
            pandas_filter_rows(self.df, "__import__('os').system('clear')"), ValueError
