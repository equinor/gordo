import unittest
from io import StringIO
from unittest import mock

from dateutil.parser import isoparse  # type: ignore

from gordo.machine.dataset.data_provider.iroc_reader import IrocReader, read_iroc_file
from gordo.machine.dataset.sensor_tag import normalize_sensor_tags
from gordo.machine.dataset.sensor_tag import SensorTag

IROC_HAPPY_TAG_LIST = [
    SensorTag("NINENINE.OPCIS::NNFCDPC01.AI1410J0", "NINENINE"),
    SensorTag("NINENINE.OPCIS::NNFCDPC01.AI1840C1J0", "NINENINE"),
    SensorTag("NINENINE.OPCIS::NNFCDPC01.AI1840E1J0", "NINENINE"),
]

HAPPY_FROM_TS = isoparse("2018-05-02T01:56:00+00:00")
HAPPY_TO_TS = isoparse("2018-05-03T01:56:00+00:00")

IROC_MANY_ASSETS_TAG_LIST = [
    "NINENINE.OPCIS::NNFCDPC01.AI1410J0",
    "NINENINE.OPCIS::NNFCDPC01.AI1840C1J0",
    "NINENINE.OPCIS::NNFCDPC01.AI1840E1J0",
    "UON_EF.OPCIS::LO006-B1H.PRCASXIN",
    "UON_EF.OPCIS::LO006-B1H.PRTUBXIN",
    "UON_EF.OPCIS::LO006-B1H_M1.PRSTAXIN",
    "UON_EF.OPCIS::LO006-B1H_M1.RTGASDIN",
]

IROC_NO_ASSET_TAG_LIST = [
    SensorTag("NOT.OPCIS::NNFCDPC01.AI1410J0", "NOT"),
    SensorTag("AN.OPCIS::NNFCDPC01.AI1840C1J0", "AN"),
    SensorTag("IROC.OPCIS::NNFCDPC01.AI1840E1J0", "IROC"),
    SensorTag("ASSET.OPCIS::LO006-B1H.PRCASXIN", "ASSET"),
]

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

IROC_MANY_ASSETS_SENSOR_TAG_LIST = [
    SensorTag("NINENINE.OPCIS::NNFCDPC01.AI1410J0", "ninenine"),
    SensorTag("NINENINE.OPCIS::NNFCDPC01.AI1840C1J0", "ninenine"),
    SensorTag("NINENINE.OPCIS::NNFCDPC01.AI1840E1J0", "ninenine"),
    SensorTag("UON_EF.OPCIS::LO006-B1H.PRCASXIN", "uon_ef"),
    SensorTag("UON_EF.OPCIS::LO006-B1H.PRTUBXIN", "uon_ef"),
    SensorTag("UON_EF.OPCIS::LO006-B1H_M1.PRSTAXIN", "uon_ef"),
    SensorTag("UON_EF.OPCIS::LO006-B1H_M1.RTGASDIN", "uon_ef"),
]


def test_normalize_iroc_tags():
    normalized_tags = normalize_sensor_tags(IROC_MANY_ASSETS_TAG_LIST)
    assert normalized_tags == IROC_MANY_ASSETS_SENSOR_TAG_LIST


class IrocDataSourceTestCase(unittest.TestCase):
    def test_read_iroc_file_basic(self):
        """A basic happy-path testing of reading an iroc CSV with some values"""
        f = StringIO(IROC_HAPPY_PATH_CSV)
        res_df = read_iroc_file(
            file_obj=f,
            train_start_date=HAPPY_FROM_TS,
            train_end_date=HAPPY_TO_TS,
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
            train_start_date=isoparse("2018-05-02T01:56:00+00:00"),
            train_end_date=isoparse("2018-05-03T01:56:00+00:00"),
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
        side_effect=lambda all_base_paths, train_start_date, train_end_date, tag_list: [
            read_iroc_file(
                file_obj=StringIO(IROC_HAPPY_PATH_CSV),
                train_start_date=train_start_date,
                train_end_date=train_end_date,
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
                    train_start_date=isoparse("2018-05-02T01:56:00+00:00"),
                    train_end_date=isoparse("2018-05-03T01:56:00+00:00"),
                    tag_list=[SensorTag("jalla", None)],  # Not a tag in the input
                )
            )

    def test_load_series_no_tag_list(self):
        """load_series will return an empty generator when called with no tags"""
        iroc_reader = IrocReader(client=None, threads=1)
        res = list(
            iroc_reader.load_series(
                train_start_date=isoparse("2018-05-02T01:56:00+00:00"),
                train_end_date=isoparse("2018-05-03T01:56:00+00:00"),
                tag_list=[],
            )
        )
        self.assertEqual([], res)

    def test_can_handle_tag_ok(self):
        iroc_reader = IrocReader(client=None, threads=1)
        assert iroc_reader.can_handle_tag(SensorTag("UON_EF.xxx", "UON_EF"))

    def test_can_handle_tag_unknown_asset(self):
        iroc_reader = IrocReader(client=None, threads=1)
        assert not iroc_reader.can_handle_tag(SensorTag("UON_EF.xxx", "UNKNOWǸ_ASSET"))

    def test_can_handle_tag_no_asset(self):
        iroc_reader = IrocReader(client=None, threads=1)
        assert not iroc_reader.can_handle_tag(SensorTag("UON_EF.xxx", None))

    def test_load_series_many_assets(self):
        """load_series will return an empty generator when called with tags
        related to several assets"""
        iroc_reader = IrocReader(client=None, threads=1)
        with self.assertRaises(ValueError):
            list(
                iroc_reader.load_series(
                    train_start_date=isoparse("2018-05-02T01:56:00+00:00"),
                    train_end_date=isoparse("2018-05-03T01:56:00+00:00"),
                    tag_list=IROC_MANY_ASSETS_SENSOR_TAG_LIST,  # Not a tag in the input
                )
            )

    def test_load_series_no_asset_found(self):
        """load_series will return an empty generator when called with tags
        that cannot be related to any asset"""
        iroc_reader = IrocReader(client=None, threads=1)
        with self.assertRaises(ValueError):
            list(
                iroc_reader.load_series(
                    train_start_date=isoparse("2018-05-02T01:56:00+00:00"),
                    train_end_date=isoparse("2018-05-03T01:56:00+00:00"),
                    tag_list=IROC_NO_ASSET_TAG_LIST,  # Not a tag in the input
                )
            )

    def test_load_series_checks_date(self):
        """load_series will raise ValueError if train_end_date<train_start_date"""
        iroc_reader = IrocReader(client=None, threads=1)
        with self.assertRaises(ValueError):
            list(
                iroc_reader.load_series(
                    train_start_date=isoparse("2018-05-03T01:56:00+00:00"),
                    train_end_date=isoparse("2018-05-02T01:56:00+00:00"),
                    tag_list=[SensorTag("jalla", None)],  # Not a tag in the input
                )
            )

    @mock.patch.object(
        IrocReader,
        "_fetch_all_iroc_files_from_paths",
        side_effect=lambda all_base_paths, train_start_date, train_end_date, tag_list: [
            read_iroc_file(
                file_obj=StringIO(IROC_HAPPY_PATH_CSV),
                train_start_date=train_start_date,
                train_end_date=train_end_date,
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
                    train_start_date=isoparse("2018-05-02T01:56:00+00:00"),
                    train_end_date=isoparse("2018-05-03T01:56:00+00:00"),
                    tag_list=IROC_HAPPY_TAG_LIST + [SensorTag("jalla", None)],
                    # "jalla" is not a tag
                )
            )

    @mock.patch.object(
        IrocReader,
        "_fetch_all_iroc_files_from_paths",
        side_effect=lambda all_base_paths, train_start_date, train_end_date, tag_list: [
            read_iroc_file(
                file_obj=StringIO(IROC_HAPPY_PATH_CSV),
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                tag_list=tag_list,
            )
        ],
    )
    def test_load_series_happy_path(self, _mocked_method):
        """Happy-path testing of load_dataframe"""
        iroc_reader = IrocReader(client=None, threads=1)
        res = list(
            iroc_reader.load_series(
                train_start_date=isoparse("2018-05-02T01:56:00+00:00"),
                train_end_date=isoparse("2018-05-03T01:56:00+00:00"),
                tag_list=IROC_HAPPY_TAG_LIST,
            )
        )
        # We get one dataframe per tag, so 3
        self.assertEqual(3, len(res))

    @mock.patch.object(
        IrocReader,
        "_fetch_all_iroc_files_from_paths",
        side_effect=lambda all_base_paths, train_start_date, train_end_date, tag_list: [
            read_iroc_file(
                file_obj=StringIO(IROC_HAPPY_PATH_CSV),
                train_start_date=train_start_date,
                train_end_date=train_end_date,
                tag_list=tag_list,
            )
        ],
    )
    def test_load_series_happy_path_different_timezones(self, _mocked_method):
        """Happy-path testing of load_dataframe"""
        iroc_reader = IrocReader(client=None, threads=1)
        res = list(
            iroc_reader.load_series(
                train_start_date=isoparse("2018-05-02T01:56:00+02:00"),
                train_end_date=isoparse("2018-05-03T01:56:00+00:00"),
                tag_list=IROC_HAPPY_TAG_LIST,
            )
        )
        # We get one dataframe per tag, so 3
        self.assertEqual(3, len(res))

    def test_load_series_dry_run_raises(self):
        iroc_reader = IrocReader(client=None)

        with self.assertRaises(NotImplementedError):
            list(
                iroc_reader.load_series(
                    train_start_date=isoparse("2018-05-02T01:56:00+02:00"),
                    train_end_date=isoparse("2018-05-03T01:56:00+00:00"),
                    tag_list=IROC_HAPPY_TAG_LIST,
                    dry_run=True,
                )
            )
