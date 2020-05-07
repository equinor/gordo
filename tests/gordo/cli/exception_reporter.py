import json
import pytest

from io import StringIO

from gordo.cli.exceptions_reporter import (
    ReportLevel,
    ExceptionsReporter,
    DEFAULT_EXIT_CODE,
)


def test_report_level():
    level = ReportLevel.get_by_name("MESSAGE")
    assert level == ReportLevel.MESSAGE
    level = ReportLevel.get_by_name("DIFFERENT")
    assert level is None
    level = ReportLevel.get_by_name("DIFFERENT", ReportLevel.MESSAGE)
    assert level == ReportLevel.MESSAGE
    levels = ReportLevel.get_names()
    assert len(levels) == 3


class _Test1Exception(Exception):
    pass


class _Test2Exception(Exception):
    pass


@pytest.fixture
def reporter1():
    return ExceptionsReporter(((_Test1Exception, 110),))


def get_result(sio):
    value = sio.getvalue()
    return json.loads(value)


def test_reporter1(reporter1):
    assert reporter1.exception_exit_code(_Test1Exception()) == 110
    assert reporter1.exception_exit_code(_Test2Exception()) == DEFAULT_EXIT_CODE


def test_with_message_report_level(reporter1):
    report_file = StringIO()
    reporter1.report(ReportLevel.MESSAGE, _Test1Exception("Test message"), report_file)
    assert get_result(report_file) == {
        "type": "_Test1Exception",
        "message": "Test message",
    }


def test_with_type_report_level(reporter1):
    report_file = StringIO()
    reporter1.report(ReportLevel.TYPE, _Test1Exception("Test message"), report_file)
    assert get_result(report_file) == {
        "type": "_Test1Exception",
    }


def test_with_exit_code_report_level(reporter1):
    report_file = StringIO()
    reporter1.report(
        ReportLevel.EXIT_CODE, _Test1Exception("Test message"), report_file
    )
    assert get_result(report_file) == {}


def test_with_unicode_chars(reporter1):
    report_file = StringIO()
    reporter1.report(ReportLevel.MESSAGE, _Test1Exception("你好 world!"), report_file)
    assert get_result(report_file) == {
        "type": "_Test1Exception",
        "message": " world!",
    }
