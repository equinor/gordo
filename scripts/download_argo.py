#!/usr/bin/env python

import sys
import argparse
import re
import os
import json
import stat

from subprocess import Popen, PIPE
from packaging import version
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, IO, cast

if sys.version_info.major != 3 or sys.version_info.minor < 7:
    raise RuntimeError("Unsupported python version: %s" % sys.version)

DOWNLOAD_URL = (
    "https://github.com/argoproj/argo-workflows/releases/download/v{version}/{arch}.gz"
)
ARCH = "argo-linux-amd64"
PROCESS_TIMEOUT = 60 * 5

BINARY_NAME = "argo"

_arch_re = re.compile(r"^[\w\-]+$")
_version_re = re.compile(r"^v?((\d+)\.(\d+)\.(\d+))")

_executable_mask = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH


@dataclass
class ArgoVersion:
    version: str
    number: Optional[int]
    is_default: bool

    _required_fields = ["version", "number"]

    @classmethod
    def load_list(cls, items: List[Dict[str, Any]]):
        is_default = True
        versions: List[ArgoVersion] = []
        numbers = set()
        for item in items:
            for field in cls._required_fields:
                if field not in item:
                    raise ValueError("'%s' is empty in %s" % (field, item))
            version, number = item["version"], int(item["number"])
            m = _version_re.match(version)
            if not m:
                raise ValueError("'%s' version is malformed" % version)
            if number in numbers:
                raise ValueError("Duplicates for version number %d", number)
            numbers.add(number)
            versions.append(cls(m[1], number, is_default))
            is_default = False
        return versions

    @property
    def binary_name(self):
        return BINARY_NAME + str(self.number)


def get_download_url(version: str, arch: str):
    return DOWNLOAD_URL.format(version=version, arch=arch)


def make_file_executable(file_path: str):
    s = os.stat(file_path)
    os.chmod(file_path, s.st_mode | _executable_mask)


def download_gz_binary(url: str, output_file: str, timeout: int = None):
    with open(output_file, "wb") as f:
        p1 = Popen(["curl", "-sL", url], stdout=PIPE)
        p2 = Popen(["gzip", "-d"], stdin=p1.stdout, stdout=f)
        cast(IO[bytes], p1.stdout).close()
        p2.communicate(timeout=timeout)
        if p2.returncode != 0:
            raise RuntimeError("Failed to download %s" % url)


def symlink(src: str, dst: str):
    if os.path.exists(dst):
        print("Removing %s" % dst)
        os.unlink(dst)
    print("Creating symlink %s -> %s" % (src, dst))
    os.symlink(src, dst)


def download_argo_versions(
    argo_versions: List[ArgoVersion], output_directory: str, arch: str, timeout: int
):
    for argo_version in argo_versions:
        url = get_download_url(argo_version.version, arch)
        output_file = os.path.join(output_directory, argo_version.binary_name)
        print("Downloading argo %s to %s" % (argo_version.version, output_file))
        download_gz_binary(url, output_file, timeout=timeout)
        print("Making %s executable" % output_file)
        make_file_executable(output_file)
        if argo_version.is_default:
            dst = os.path.join(output_directory, BINARY_NAME)
            symlink(argo_version.binary_name, dst)


def usage(
    parser: argparse.ArgumentParser, message: str = None, returncode: int = 1, f=None
):
    if f is None:
        f = sys.stdout
    if message:
        print(message + "\n", file=f)
    parser.print_help(file=f)
    sys.exit(returncode)


def main():
    parser = argparse.ArgumentParser(description="Download argo CLIs binaries")

    parser.add_argument(
        "-v",
        "--argo-versions",
        default=os.environ.get("ARGO_VERSIONS"),
        help='Argo versions list in JSON format. Takes ARGO_VERSIONS environment variable as default value. Example: [{"version": "3.4.2"}]',
    )
    parser.add_argument(
        "-o", "--output-directory", required=True, help="Output directory"
    )
    parser.add_argument(
        "--arch",
        default=ARCH,
        help="Binary architecture. Default: '%s'" % ARCH,
    )
    parser.add_argument(
        "-t",
        "--process-timeout",
        default=PROCESS_TIMEOUT,
        help="Subprocesses timeout in seconds. Default: %d" % PROCESS_TIMEOUT,
    )
    args = parser.parse_args()

    if not args.argo_versions:
        usage(parser, message="--argo-versions is empty")

    if not _arch_re.match(args.arch):
        raise ValueError("'%s' malformed arch" % args.arch)

    argo_versions = ArgoVersion.load_list(json.loads(args.argo_versions))

    download_argo_versions(
        argo_versions,
        output_directory=args.output_directory,
        arch=args.arch,
        timeout=args.process_timeout,
    )


if __name__ == "__main__":
    main()
