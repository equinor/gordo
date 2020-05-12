import re

non_ascii_re = re.compile(r"[^\s\u0020-\u007f]")


def replace_all_non_ascii_chars(s: str, replace_with: str = "") -> str:
    return non_ascii_re.sub(replace_with, s)
