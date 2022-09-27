import subprocess
import re

import dictdiffer

from typing import Optional
from packaging import version

_version_re = re.compile(r"^argo:\s+v?(.+)$")


class ArgoVersionDeterminingError(Exception):
    pass


def patch_dict(original_dict: dict, patch_dictionary: dict) -> dict:
    """Patches a dict with another. Patching means that any path defines in the
    patch is either added (if it does not exist), or replaces the existing value (if
    it exists). Nothing is removed from the original dict, only added/replaced.

    Parameters
    ----------
    original_dict : dict
        Base dictionary which will get paths added/changed
    patch_dictionary: dict
        Dictionary which will be overlaid on top of original_dict

    Examples
    --------
    >>> patch_dict({"highKey":{"lowkey1":1, "lowkey2":2}}, {"highKey":{"lowkey1":10}})
    {'highKey': {'lowkey1': 10, 'lowkey2': 2}}
    >>> patch_dict({"highKey":{"lowkey1":1, "lowkey2":2}}, {"highKey":{"lowkey3":3}})
    {'highKey': {'lowkey1': 1, 'lowkey2': 2, 'lowkey3': 3}}
    >>> patch_dict({"highKey":{"lowkey1":1, "lowkey2":2}}, {"highKey2":4})
    {'highKey': {'lowkey1': 1, 'lowkey2': 2}, 'highKey2': 4}

    Returns
    -------
    dict
        A new dictionary which is the result of overlaying `patch_dictionary` on top of
        `original_dict`

    """
    diff = dictdiffer.diff(original_dict, patch_dictionary)
    adds_and_mods = [(f, d, s) for (f, d, s) in diff if f != "remove"]
    return dictdiffer.patch(adds_and_mods, original_dict)


def determine_argo_version() -> version.Version:
    command = ["argo", "version", "--show"]
    message_suffix = ". Command: '%s'" % " ".join(command)
    try:
        result = subprocess.run(command, timeout=30, capture_output=True, check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        raise ArgoVersionDeterminingError(
            "Getting argo version exception" + message_suffix
        )
    if result is not None:
        stdout: Optional[str] = None
        try:
            stdout = result.stdout.decode("utf-8")
        except UnicodeDecodeError:
            raise ArgoVersionDeterminingError(
                ("Unable to encode output %s" % str(stdout)) + message_suffix
            )
        if stdout is not None:
            m = _version_re.match(stdout.rstrip())
            if not m:
                raise ArgoVersionDeterminingError(
                    ("Unable to parse version from %s" % str(stdout)) + message_suffix
                )
            argo_version = version.parse(m[1])
            if not isinstance(argo_version, version.Version):
                raise ArgoVersionDeterminingError(
                    ("Parsed version has wrong format '%s'" % argo_version)
                    + message_suffix
                )
            return argo_version
