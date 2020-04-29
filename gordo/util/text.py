import re

non_ascii_re = re.compile(r"[^\u0020-\u007f]+")


def cuts_all_non_ascii_chars(s: str) -> str:
    return non_ascii_re.sub("", s)
