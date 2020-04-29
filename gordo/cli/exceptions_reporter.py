import json
import traceback

from typing import Tuple, Iterable, Type, IO
from collections import OrderedDict
from enum import Enum


class ReportLevel(Enum):
    EXIT_CODE = 0
    TYPE = 1
    MESSAGE = 2

    @classmethod
    def get_by_name(cls, name: str, default: "ReportLevel" = None) -> "ReportLevel":
        for level in cls:
            if name == level.name:
                return level
        return default

    @classmethod
    def get_names(cls):
        return [level.name for level in cls]


DEFAULT_EXIT_CODE = 1


class ExceptionsReporter:
    def __init__(
        self,
        exceptions: Iterable[Tuple[Type[Exception], int]],
        default_exit_code: int = DEFAULT_EXIT_CODE,
    ):
        self.exceptions = OrderedDict(exceptions)
        self.default_exit_code = default_exit_code

    def exception_exit_code(self, e: Exception):
        return self.exceptions.get(e.__class__, self.default_exit_code)

    def report(self, level: ReportLevel, e: Exception, report_file: IO[str]):
        report = {}
        if e.__class__ in self.exceptions:
            if level in (ReportLevel.MESSAGE, ReportLevel.TYPE):
                report["type"] = e.__class__.__name__
            if level == ReportLevel.MESSAGE:
                report["message"] = str(e)
        json.dump(report, report_file)

    def safe_report(self, level: ReportLevel, e: Exception, report_file_path: str):
        try:
            with open(report_file_path, "w") as report_file:
                self.report(level, e, report_file)
        except Exception:
            traceback.print_exc()
