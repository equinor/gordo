import sys
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
    assert len(levels) == 4


class _Test1Exception(Exception):
    pass


class _Test2Exception(Exception):
    pass


class _Test3Exception(_Test1Exception):
    pass


traceback_example = [
    "Traceback (most recent call last):\n",
    '  File "bar.py", line 13, in <module>\n    foo()\n',
    '  File "bar.py", line 10, in foo\n    bar()\n',
    '  File "bar.py", line 7, in bar\n'
    '    raise Exception("Something bad happening")\n',
    "Exception: Something bad happening\n",
]


@pytest.fixture
def reporter1():
    return ExceptionsReporter(((_Test1Exception, 110),))


def test_sort_exceptions():
    exceptions = ((Exception, 10), (IOError, 20), (FileNotFoundError, 30))
    sorted_exceptions = ExceptionsReporter.sort_exceptions(exceptions)
    assert sorted_exceptions == [
        (FileNotFoundError, 30),
        (OSError, 20),
        (Exception, 10),
    ]


def test_trim_formatted_traceback():
    result = ExceptionsReporter.trim_formatted_traceback(traceback_example, 118)
    assert result == [
        "...\n",
        '  File "bar.py", line 7, in bar\n'
        '    raise Exception("Something bad happening")\n',
        "Exception: Something bad happening\n",
    ]
    result = ExceptionsReporter.trim_formatted_traceback(traceback_example, 117)
    assert result == ["...\n", "Exception: Something bad happening\n"]


def test_reporter1(reporter1):
    assert reporter1.exception_exit_code(_Test1Exception) == 110
    assert reporter1.exception_exit_code(_Test2Exception) == DEFAULT_EXIT_CODE
    assert reporter1.exception_exit_code(_Test3Exception) == 110
    assert reporter1.exception_exit_code(None) == 0


def test_reporting_out_of_exception_scope(reporter1):
    sio = StringIO()
    reporter1.report(ReportLevel.MESSAGE, None, None, None, sio)
    assert json.loads(sio.getvalue()) == {}


def report(e, reporter, report_level, report_file, **kwargs):
    try:
        raise e
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        reporter.report(
            report_level, exc_type, exc_value, exc_traceback, report_file, **kwargs
        )


def report_to_string(e, reporter, report_level, **kwargs):
    sio = StringIO()
    report(e, reporter, report_level, sio, **kwargs)
    value = sio.getvalue()
    return json.loads(value)


def test_with_message_report_level(reporter1):
    result = report_to_string(
        _Test1Exception("Test message"), reporter1, ReportLevel.MESSAGE
    )
    assert result == {
        "type": "_Test1Exception",
        "message": "Test message",
    }


def test_with_traceback_report_level(reporter1):
    result = report_to_string(
        _Test1Exception("Test message"), reporter1, ReportLevel.TRACEBACK
    )
    assert result["type"] == "_Test1Exception"
    assert "traceback" in result
    assert "Test message" in result["traceback"]


def test_with_type_report_level(reporter1):
    result = report_to_string(
        _Test1Exception("Test message"), reporter1, ReportLevel.TYPE
    )
    assert result == {
        "type": "_Test1Exception",
    }


def test_with_exit_code_report_level(reporter1):
    result = report_to_string(
        _Test1Exception("Test message"), reporter1, ReportLevel.EXIT_CODE
    )
    assert result == {}


def test_with_unicode_chars(reporter1):
    result = report_to_string(
        _Test1Exception("\t你好 world!\n"), reporter1, ReportLevel.MESSAGE
    )
    assert result == {
        "type": "_Test1Exception",
        "message": "\t?? world!\n",
    }


def test_with_max_message_len(reporter1):
    result = report_to_string(
        _Test1Exception("Hello world!"),
        reporter1,
        ReportLevel.MESSAGE,
        max_message_len=8,
    )
    assert result == {
        "type": "_Test1Exception",
        "message": "Hello...",
    }
    result = report_to_string(
        _Test1Exception("Hello world!"),
        reporter1,
        ReportLevel.MESSAGE,
        max_message_len=20,
    )
    assert result == {
        "type": "_Test1Exception",
        "message": "Hello world!",
    }
    result = report_to_string(
        _Test1Exception("Hello"), reporter1, ReportLevel.MESSAGE, max_message_len=4
    )
    assert result == {
        "type": "_Test1Exception",
        "message": "",
    }
