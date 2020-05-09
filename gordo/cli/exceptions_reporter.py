import json
import traceback

from typing import Tuple, Iterable, Type, IO, Optional, List
from types import TracebackType
from collections import Counter
from enum import Enum

from gordo.util import replace_all_non_ascii_chars


class ReportLevel(Enum):
    EXIT_CODE = 0
    TYPE = 1
    MESSAGE = 2

    @classmethod
    def get_by_name(
        cls, name: str, default: Optional["ReportLevel"] = None
    ) -> Optional["ReportLevel"]:
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
        self.exceptions_items = self.sort_exceptions(exceptions)
        self.default_exit_code = default_exit_code

    @staticmethod
    def sort_exceptions(
        exceptions: Iterable[Tuple[Type[Exception], int]]
    ) -> List[Tuple[Type[Exception], int]]:
        inheritance_levels = Counter()
        for exc, exit_code in exceptions:
            for e, _ in exceptions:
                if e is not exc and issubclass(exc, e):
                    inheritance_levels[e] += 1
        sorted_exceptions = list(exceptions)

        def key(v):
            exc, exit_code = v
            return (inheritance_levels[exc], exit_code)

        return sorted(sorted_exceptions, key=key)

    def found_exception_item(self, exc_type: Type[Exception]):
        for item in self.exceptions_items:
            if issubclass(exc_type, item[0]):
                return item
        return None

    def exception_exit_code(self, exc_type: Type[Exception]):
        item = self.found_exception_item(exc_type)
        return item[1] if item is not None else self.default_exit_code

    def report(
        self,
        level: ReportLevel,
        exc_type: Type[Exception],
        exc_value: Exception,
        exc_traceback: TracebackType,
        report_file: IO[str],
        max_message_len: Optional[int] = None,
    ):
        report = {}

        def add_report(k: str, v: str):
            report[k] = replace_all_non_ascii_chars(v, "?")

        if self.found_exception_item(exc_type) is not None:
            if level in (ReportLevel.MESSAGE, ReportLevel.TYPE):
                add_report("type", exc_type.__name__)
            if level == ReportLevel.MESSAGE:
                add_report("message", str(exc_value))
                if max_message_len is not None:
                    message = report["message"]
                    if len(message) > max_message_len:
                        message = message[: max_message_len - 3]
                        if len(message) <= 3:
                            report["message"] = ""
                        else:
                            report["message"] = message + "..."
        json.dump(report, report_file)

    def safe_report(
        self,
        level: ReportLevel,
        exc_type: Type[Exception],
        exc_value: Exception,
        exc_traceback: TracebackType,
        report_file_path: str,
        max_message_len: Optional[int] = None,
    ):
        try:
            with open(report_file_path, "w") as report_file:
                self.report(
                    level,
                    exc_type,
                    exc_value,
                    exc_traceback,
                    report_file,
                    max_message_len,
                )
        except Exception:
            traceback.print_exc()
