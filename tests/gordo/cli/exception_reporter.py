import json

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
    assert level == ReportLevel.EXIT_CODE
    level = ReportLevel.get_by_name("DIFFERENT", ReportLevel.MESSAGE)
    assert level == ReportLevel.MESSAGE
    levels = ReportLevel.get_names()
    assert len(levels) == 3


class _Test1Exception(Exception):
    pass


class _Test2Exception(Exception):
    pass


def test_exceptions_reporter():
    reporter = ExceptionsReporter(((_Test1Exception, 110),))
    assert reporter.exception_exit_code(_Test1Exception()) == 110
    assert reporter.exception_exit_code(_Test2Exception()) == DEFAULT_EXIT_CODE

    def get_result(sio):
        value = sio.getvalue()
        return json.loads(value)

    report_file = StringIO()
    reporter.report(ReportLevel.MESSAGE, _Test1Exception("Test message"), report_file)
    assert get_result(report_file) == {
        "type": "_Test1Exception",
        "message": "Test message",
    }
    report_file = StringIO()
    reporter.report(ReportLevel.TYPE, _Test1Exception("Test message"), report_file)
    assert get_result(report_file) == {
        "type": "_Test1Exception",
    }
    report_file = StringIO()
    reporter.report(ReportLevel.EXIT_CODE, _Test1Exception("Test message"), report_file)
    assert get_result(report_file) == {}
