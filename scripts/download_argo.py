#!/usr/bin/env python

import sys
import argparse
import re
import os
import json

from subprocess import Popen, PIPE
from packaging import version
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

if sys.version_info.major != 3 or sys.version_info.minor < 7:
    raise RuntimeError("Unsupported python version: %s" % sys.version)

DOWNLOAD_URL = (
    "https://github.com/argoproj/argo-workflows/releases/download/v{version}/{arch}.gz"
)
ARCH = "argo-linux-amd64"
PROCESS_TIMEOUT = 60 * 5

BINARY_NAME = "argo"

_arch_re = re.compile(r"^[\w\-]+$")
_version_re = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


@dataclass
class ArgoVersion:
    version: str
    number: Optional[int]

    @classmethod
    def load(cls, item: Dict[str, Any]):
        if "version" not in item:
            raise ValueError("'version' is empty in %s" % item)
        version = item["version"]
        if not _version_re.match(version):
            raise ValueError("'%s' version is malformed" % version)
        number = int(item["number"]) if item.get("number") is not None else None
        return cls(version, number)

    @property
    def binary_name(self):
        return BINARY_NAME + self.str_number

    @property
    def str_number(self):
        return str(self.number) if self.number is not None else ""


def load_argo_versions(items: List[Dict[str, Any]]):
    numbers = set()
    argo_versions: List[ArgoVersion] = []
    for item in items:
        argo_version = ArgoVersion.load(item)
        if argo_version.number in numbers:
            raise ValueError(
                "Duplicates for version number '%s'", argo_version.str_number
            )
        numbers.add(argo_version.number)
        argo_versions.append(argo_version)
    return argo_versions


def get_download_url(version: str, arch: str):
    return DOWNLOAD_URL.format(version=version, arch=arch)


def download_gz_binary(url: str, output_file: str, timeout: int = None):
    with open(output_file, "wb") as f:
        p1 = Popen(["curl", "-sL", url], stdout=PIPE)
        p2 = Popen(["gzip", "-d"], stdin=p1.stdout, stdout=f)
        p1.stdout.close()
        p2.communicate(timeout=timeout)
        if p2.returncode != 0:
            raise RuntimeError("Failed to download %s" % url)


def download_argo_versions(
    argo_versions: List[ArgoVersion], output_directory: str, arch: str, timeout: int
):
    for argo_version in argo_versions:
        url = get_download_url(argo_version.version, arch)
        output_file = os.path.join(output_directory, argo_version.binary_name)
        print("Downloading argo %s to '%s'" % (argo_version.version, output_file))
        download_gz_binary(url, output_file, timeout=timeout)


def usage(
    parser: argparse.ArgumentParser, message: str = None, returncode: int = 1, file=None
):
    if file is None:
        file = sys.stdout
    if message:
        print(message + "\n", file=file)
    parser.print_help(file=file)
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

    argo_versions = load_argo_versions(json.loads(args.argo_versions))

    download_argo_versions(
        argo_versions,
        output_directory=args.output_directory,
        arch=args.arch,
        timeout=args.process_timeout,
    )


if __name__ == "__main__":
    main()
