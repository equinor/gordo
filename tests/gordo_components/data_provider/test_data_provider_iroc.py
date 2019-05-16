import unittest
from io import StringIO
from unittest import mock

from dateutil.parser import isoparse  # type: ignore

from gordo_components.data_provider.iroc_reader import IrocReader, read_iroc_file
from gordo_components.dataset.sensor_tag import SensorTag

IROC_HAPPY_TAG_LIST = [
    SensorTag("NINENINE.OPCIS::NNFCDPC01.AI1410J0", None),
    SensorTag("NINENINE.OPCIS::NNFCDPC01.AI1840C1J0", None),
    SensorTag("NINENINE.OPCIS::NNFCDPC01.AI1840E1J0", None),
]

HAPPY_FROM_TS = isoparse("2018-05-02T01:56:00+00:00")
HAPPY_TO_TS = isoparse("2018-05-03T01:56:00+00:00")

# Functioning CSV for IROC. Has 6 lines, 5 different timestamps
# (2018-05-02T06:44:29.7830000Z occurs twice) and 3 different tags.
IROC_HAPPY_PATH_CSV = u"""tag,value,timestamp,status
NINENINE.OPCIS::NNFCDPC01.AI1410J0,5,2018-05-02T06:00:11.3860000Z,Analog Normal
NINENINE.OPCIS::NNFCDPC01.AI1410J0,76.86899,2018-05-02T06:44:29.7830000Z,Analog Normal
NINENINE.OPCIS::NNFCDPC01.AI1840C1J0,-23.147645,2018-05-02T06:43:53.8490000Z,Analog Normal
NINENINE.OPCIS::NNFCDPC01.AI1840C1J0,-10.518037,2018-05-02T06:44:29.9130000Z,Analog Normal
NINENINE.OPCIS::NNFCDPC01.AI1840E1J0,48.92137,2018-05-02T06:43:59.7240000Z,Analog Normal
NINENINE.OPCIS::NNFCDPC01.AI1840E1J0,-0.497645,2018-05-02T06:44:29.7830000Z,Analog Normal
                """


class IrocDataSourceTestCase(unittest.TestCase):
    def test_read_iroc_file_basic(self):
        """A basic happy-path testing of reading an iroc CSV with some values"""
        f = StringIO(IROC_HAPPY_PATH_CSV)
        res_df = read_iroc_file(
            file_obj=f,
            from_ts=HAPPY_FROM_TS,
            to_ts=HAPPY_TO_TS,
            tag_list=IROC_HAPPY_TAG_LIST,
        )
        for tag in IROC_HAPPY_TAG_LIST:
            self.assertIn(tag.name, res_df.columns)
        # We have one row per distinct timestamp in the input-csv
        self.assertEqual(5, len(res_df))

    def test_read_iroc_file_with_errors(self):
        """Reading a csv with some of the 'values' beeing strings instead of numbers
        leads to those lines being ignored completely"""
        f = StringIO(
            u"""tag,value,timestamp,status
NINENINE.OPCIS::NNFCDPC01.AI1410J0,5,2018-05-02T06:00:11.3860000Z,Analog Normal
NINENINE.OPCIS::NNFCDPC01.AI1410J0,76.86899,2018-05-02T06:44:29.7830000Z,Analog Normal
NINENINE.OPCIS::NNFCDPC01.AI1840C1J0,lameness,2018-05-02T06:43:53.8490000Z,Analog Normal
NINENINE.OPCIS::NNFCDPC01.AI1840C1J0,-10.518037,2018-05-02T06:44:29.9130000Z,Analog Normal
NINENINE.OPCIS::NNFCDPC01.AI1840E1J0,wtf,2018-05-02T06:43:59.7240000Z,Analog Normal
NINENINE.OPCIS::NNFCDPC01.AI1840E1J0,-0.497645,2018-05-02T06:44:29.7830000Z,Analog Normal
                """
        )

        res_df = read_iroc_file(
            file_obj=f,
            from_ts=isoparse("2018-05-02T01:56:00+00:00"),
            to_ts=isoparse("2018-05-03T01:56:00+00:00"),
            tag_list=IROC_HAPPY_TAG_LIST,
        )
        for tag in IROC_HAPPY_TAG_LIST:
            self.assertIn(tag.name, res_df.columns)
        # Two of the lines have invalid values (non-numeric), so those are ignored
        # and there are only 3 distinct timestamps left.
        self.assertEqual(3, len(res_df))

    @mock.patch.object(
        IrocReader,
        "_fetch_all_iroc_files_from_paths",
        side_effect=lambda all_base_paths, from_ts, to_ts, tag_list: [
            read_iroc_file(
                file_obj=StringIO(IROC_HAPPY_PATH_CSV),
                from_ts=from_ts,
                to_ts=to_ts,
                tag_list=tag_list,
            )
        ],
    )
    def test_load_series_no_data(self, _mocked_method):
        """load_series will raise ValueError if it does not find any tags"""
        iroc_reader = IrocReader(client=None, threads=1)
        with self.assertRaises(ValueError):
            list(
                iroc_reader.load_series(
                    from_ts=isoparse("2018-05-02T01:56:00+00:00"),
                    to_ts=isoparse("2018-05-03T01:56:00+00:00"),
                    tag_list=[SensorTag("jalla", None)],  # Not a tag in the input
                )
            )

    def test_load_series_no_tag_list(self):
        """load_series will return an empty generator when called with no tags"""
        iroc_reader = IrocReader(client=None, threads=1)
        res = list(
            iroc_reader.load_series(
                from_ts=isoparse("2018-05-02T01:56:00+00:00"),
                to_ts=isoparse("2018-05-03T01:56:00+00:00"),
                tag_list=[],
            )
        )
        self.assertEqual([], res)

    def test_load_series_checks_date(self):
        """load_series will raise ValueError if to_ts<from_ts"""
        iroc_reader = IrocReader(client=None, threads=1)
        with self.assertRaises(ValueError):
            list(
                iroc_reader.load_series(
                    from_ts=isoparse("2018-05-03T01:56:00+00:00"),
                    to_ts=isoparse("2018-05-02T01:56:00+00:00"),
                    tag_list=[SensorTag("jalla", None)],  # Not a tag in the input
                )
            )

    @mock.patch.object(
        IrocReader,
        "_fetch_all_iroc_files_from_paths",
        side_effect=lambda all_base_paths, from_ts, to_ts, tag_list: [
            read_iroc_file(
                file_obj=StringIO(IROC_HAPPY_PATH_CSV),
                from_ts=from_ts,
                to_ts=to_ts,
                tag_list=tag_list,
            )
        ],
    )
    def test_load_series_missing_columns_data(self, _mocked_method):
        """load_series will raise ValueError if there is a single tag it can not
        find"""
        iroc_reader = IrocReader(client=None, threads=1)
        with self.assertRaises(ValueError):
            list(
                iroc_reader.load_series(
                    from_ts=isoparse("2018-05-02T01:56:00+00:00"),
                    to_ts=isoparse("2018-05-03T01:56:00+00:00"),
                    tag_list=IROC_HAPPY_TAG_LIST + [SensorTag("jalla", None)],
                    # "jalla" is not a tag
                )
            )

    @mock.patch.object(
        IrocReader,
        "_fetch_all_iroc_files_from_paths",
        side_effect=lambda all_base_paths, from_ts, to_ts, tag_list: [
            read_iroc_file(
                file_obj=StringIO(IROC_HAPPY_PATH_CSV),
                from_ts=from_ts,
                to_ts=to_ts,
                tag_list=tag_list,
            )
        ],
    )
    def test_load_series_happy_path(self, _mocked_method):
        """Happy-path testing of load_dataframe"""
        iroc_reader = IrocReader(client=None, threads=1)
        res = list(
            iroc_reader.load_series(
                from_ts=isoparse("2018-05-02T01:56:00+00:00"),
                to_ts=isoparse("2018-05-03T01:56:00+00:00"),
                tag_list=IROC_HAPPY_TAG_LIST,
            )
        )
        # We get one dataframe per tag, so 3
        self.assertEqual(3, len(res))
