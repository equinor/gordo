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


class _Test3Exception(_Test1Exception):
    pass


@pytest.fixture
def reporter1():
    return ExceptionsReporter(((_Test1Exception, 110),))


def get_result(sio):
    value = sio.getvalue()
    return json.loads(value)


def test_sort_exceptions():
    exceptions = ((Exception, 10), (IOError, 20), (FileNotFoundError, 30))
    sorted_exceptions = ExceptionsReporter.sort_exceptions(exceptions)
    assert sorted_exceptions == [
        (FileNotFoundError, 30),
        (OSError, 20),
        (Exception, 10),
    ]


def test_reporter1(reporter1):
    assert reporter1.exception_exit_code(_Test1Exception()) == 110
    assert reporter1.exception_exit_code(_Test2Exception()) == DEFAULT_EXIT_CODE
    assert reporter1.exception_exit_code(_Test3Exception()) == 110


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


def test_with_max_message_len(reporter1):
    report_file = StringIO()
    reporter1.report(
        ReportLevel.MESSAGE,
        _Test1Exception("Hello world!"),
        report_file,
        max_message_len=8,
    )
    assert get_result(report_file) == {
        "type": "_Test1Exception",
        "message": "Hello...",
    }
    report_file = StringIO()
    reporter1.report(
        ReportLevel.MESSAGE,
        _Test1Exception("Hello world!"),
        report_file,
        max_message_len=20,
    )
    assert get_result(report_file) == {
        "type": "_Test1Exception",
        "message": "Hello world!",
    }
    report_file = StringIO()
    reporter1.report(
        ReportLevel.MESSAGE, _Test1Exception("Hello"), report_file, max_message_len=4
    )
    assert get_result(report_file) == {
        "type": "_Test1Exception",
        "message": "",
    }
