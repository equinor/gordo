#!/usr/bin/env python

import sys
import argparse

from subprocess import Popen, PIPE
from packaging import version

if sys.version_info.major != 3 or sys.version_info.minor < 7:
    raise RuntimeError("Unsupported python version: %s" % sys.version)

ARGO_DOWNLOAD_URL = (
    "https://github.com/argoproj/argo-workflows/releases/download/{version}/{arch}.gz"
)

DEFAULT_ARCH = "argo-linux-amd64"


def get_argo_download_url(version: str, arch: str):
    return ARGO_DOWNLOAD_URL.format(version=version, arch=arch)


def download_argo_binary(
    minor_version: int, argo_version: str, argo_arch: str, output_file: str
):
    with open(output_file, "wb") as f:
        p1 = Popen(
            ["curl", "-sLO", get_argo_download_url(argo_version, argo_arch)],
            stdout=PIPE,
        )
        p2 = Popen(["gzip", "-d"], stdin=p1.stdout, stdout=f)
        p1.stdout.close()
        p2.communicate()


def main():
    parser = argparse.ArgumentParser(description="Download argo CLIs binaries")

    parser.add_argument(
        "-m", "--minor-version", required=True, type=int, help="Minor version"
    )
    parser.add_argument(
        "-v", "--argo-version", required=True, help="Argo version to download"
    )
    parser.add_argument("-o", "--output-file", required=True, help="Output file")
    parser.add_argument(
        "--arch",
        default=DEFAULT_ARCH,
        help="Binary architecture. Default: '%s'" % DEFAULT_ARCH,
    )
    args = parser.parse_args()

    download_argo_binary(
        minor_version=args.minor_version,
        argo_version=args.argo_version,
        argo_arch=args.arch,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
