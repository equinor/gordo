#!/usr/bin/env python

import sys
import argparse
import re
import os
import json

from subprocess import Popen, PIPE
from packaging import version
from dataclasses import dataclass
from typing import List, Dict, Any

if sys.version_info.major != 3 or sys.version_info.minor < 7:
    raise RuntimeError("Unsupported python version: %s" % sys.version)

DOWNLOAD_URL = (
    "https://github.com/argoproj/argo-workflows/releases/download/v{version}/{arch}.gz"
)
ARCH = "argo-linux-amd64"
PROCESS_TIMEOUT = 60

_arch_re = re.compile(r"^[\w\-]+$")
_version_re = re.compile(r"^(\d+)\.(\d+)\.(\d+)")


@dataclass
class ArgoVersion:
    major: int
    version: str

    @classmethod
    def load(cls, item: Dict[str, Any]):
        if "version" not in item:
            raise ValueError("version is empty in %s" % item)
        version = item["version"]
        m = _version_re.match(version)
        if not m:
            raise ValueError("'%s' version is malformed" % version])
        return cls(int(item.get("major", m[1])), version)

    def binary_name(self):
        return "argo%d" % self.major


def load_argo_versions(items: List[Dict[str, Any]]):
    majors = set()
    argo_versions: List[ArgoVersion] = []
    for item in items:
        argo_version = ArgoVersion.load(item)
        if argo_version.major in majors:
            raise ValueError("Duplicates for major version %d", argo_version.major)
        majors.add(argo_version.major)
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


def download_argo_versions(argo_versions: List[ArgoVersion], output_directory: str, arch: str)
    for argo_version in argo_versions:
        url = get_download_url(


def main():
    parser = argparse.ArgumentParser(description="Download argo CLIs binaries")

    parser.add_argument(
            "-v", "--argo-versions", required=True, default=os.environ.get("ARGO_VERSIONS"), help='Argo versions to download. In JSON format. ARGO_VERSIONS env variable by default. Example: [{"version": "3.4.2"}]'
    )
    parser.add_argument("-o", "--output-directory", required=True, help="Output directory")
    parser.add_argument(
        "--arch",
        default=ARCH,
        help="Binary architecture. Default: '%s'" % ARCH,
    )
    parser.add_argument(
        "-t",
        "--process-timeout",
        default=PROCESS_TIMEOUT,
        help="Subprocesses timeout in seconds. Default: %d" % PROCESS_TIMEOUT
    )
    args = parser.parse_args()

    if not _arch_re.match(args.arch):
        raise ValueError("'%s' malformed arch" % args.arch)

    argo_versions = load_argo_versions(json.load(arg.argo_versions))

    download_gz_binary(get_download_url(args.argo_version, args.arch), output_file=args.output_file)


if __name__ == "__main__":
    main()
