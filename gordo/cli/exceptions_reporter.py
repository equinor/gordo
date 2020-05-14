import json
import traceback

from typing import Tuple, Iterable, Type, IO, Optional, List, Dict
from types import TracebackType
from collections import Counter
from enum import Enum

from gordo.util import replace_all_non_ascii_chars


class ReportLevel(Enum):
    EXIT_CODE = 0
    TYPE = 1
    MESSAGE = 2
    TRACEBACK = 3

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
    """
        Helper which can save the exception information in JSON format into the file.
        This class might be used for storing exception information after termination of the Kubernetes pod.
        `Information <https://kubernetes.io/docs/tasks/debug-application-cluster/determine-reason-pod-failure/#customizing-the-termination-message>`_
    """

    def __init__(
        self,
        exceptions: Iterable[Tuple[Type[Exception], int]],
        default_exit_code: int = DEFAULT_EXIT_CODE,
        traceback_limit: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        exceptions
            Exceptions list with preferred exit codes for each of them
        default_exit_code
            Default exit code. It might be used as `sys.exit()` code
        traceback_limit
            Limit for `traceback.format_exception()`
        """
        self.exceptions_items = self.sort_exceptions(exceptions)
        self.default_exit_code = default_exit_code
        self.traceback_limit = traceback_limit

    @staticmethod
    def sort_exceptions(
        exceptions: Iterable[Tuple[Type[Exception], int]]
    ) -> List[Tuple[Type[Exception], int]]:
        inheritance_levels: Dict[Type[BaseException], int] = Counter()
        for exc, exit_code in exceptions:
            for e, _ in exceptions:
                if e is not exc and issubclass(exc, e):
                    inheritance_levels[e] += 1
        sorted_exceptions = list(exceptions)

        def key(v):
            exc, exit_code = v
            return (inheritance_levels[exc], exit_code)

        return sorted(sorted_exceptions, key=key)

    @staticmethod
    def trim_message(message: str, max_length: int) -> str:
        if len(message) > max_length:
            message = message[: max_length - 3]
            return "" if len(message) <= 3 else message + "..."
        return message

    @staticmethod
    def trim_formatted_traceback(
        formatted_traceback: List[str], max_length: int
    ) -> List[str]:
        if sum(len(line) for line in formatted_traceback) <= max_length:
            return formatted_traceback
        length = 4
        result = []
        for line in reversed(formatted_traceback):
            length += len(line)
            if length > max_length:
                result.append("...\n")
                break
            else:
                result.append(line)
        return list(reversed(result))

    def found_exception_item(self, exc_type: Type[BaseException]):
        for item in self.exceptions_items:
            if issubclass(exc_type, item[0]):
                return item
        return None

    def exception_exit_code(self, exc_type: Optional[Type[BaseException]]) -> int:
        """
        Possible `sys.exit()` code for given exception type

        Parameters
        ----------
        exc_type
            The exception type

        Returns
        -------
        int
        """
        if exc_type is None:
            return 0
        item = self.found_exception_item(exc_type)
        return item[1] if item is not None else self.default_exit_code

    def report(
        self,
        level: ReportLevel,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
        report_file: IO[str],
        max_message_len: Optional[int] = None,
    ):
        """
        Report exception to the file.
        `exc_type`, `exc_value`, `exc_traceback` might be values returned by `sys.exc_info()`

        Parameters
        ----------
        level: ReportLevel
            Level of the report verbosity
        exc_type
            The exception type
        exc_value
            The exception
        exc_traceback
            The exception traceback
        report_file
            File like object for reporting
        max_message_len
            The maximum length of `message` or `traceback`.
            Actual for the environments with limitation of storage capacity.
            For example, 2024 bytes is the maximum size of the k8s pod termination message content
        """
        report = {}

        if exc_type is not None and exc_value is not None and exc_traceback is not None:
            if self.found_exception_item(exc_type) is not None:
                if level in (
                    ReportLevel.MESSAGE,
                    ReportLevel.TYPE,
                    ReportLevel.TRACEBACK,
                ):
                    report["type"] = replace_all_non_ascii_chars(exc_type.__name__, "?")
                if level == ReportLevel.MESSAGE:
                    report["message"] = replace_all_non_ascii_chars(str(exc_value), "?")
                    if max_message_len is not None:
                        report["message"] = self.trim_message(
                            report["message"], max_message_len
                        )
                elif level == ReportLevel.TRACEBACK:
                    formatted_traceback = traceback.format_exception(
                        exc_type, exc_value, exc_traceback, limit=self.traceback_limit
                    )
                    formatted_traceback = [
                        replace_all_non_ascii_chars(v, "?") for v in formatted_traceback
                    ]
                    if max_message_len is not None:
                        formatted_traceback = self.trim_formatted_traceback(
                            formatted_traceback, max_message_len
                        )
                    report["traceback"] = "".join(formatted_traceback)
        json.dump(report, report_file)

    def safe_report(
        self,
        level: ReportLevel,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
        report_file_path: str,
        max_message_len: Optional[int] = None,
    ):
        """
        Basically this is a wrapper for `ExceptionsReporter.report()` function
        with additional internal exceptions handling

        Parameters
        ----------
        level
        exc_type
        exc_value
        exc_traceback
        report_file_path
        max_message_len

        Returns
        -------

        """
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
