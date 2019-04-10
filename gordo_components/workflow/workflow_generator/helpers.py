import re


def _get_sanitized_string(original_string):
    return re.sub(r"[^0-9a-zA-Z-]+", "-", original_string)
