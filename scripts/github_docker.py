#!/usr/bin/env python

import sys
import os
import re
import json
import argparse

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple, Dict

if sys.version_info.major != 3 or sys.version_info.minor < 7:
    raise RuntimeError("Unsupported python version: %s" % sys.version)


re_tags = re.compile(r"^refs\/tags\/v?([^\/]*).*?$")
re_pull = re.compile(r"^refs\/pull\/(\d+).*?$")


class ImageType(Enum):
    dev = "dev"
    pr = "pr"
    prod = "prod"
    sha = "sha"


class Release(Enum):
    release = "release"
    prerelease = "prerelease"


def get_github_event(environ: Dict[str, str]):
    with open(environ["GITHUB_EVENT_PATH"], "r") as f:  # type: ignore
        return json.load(f)


@dataclass
class Settings:
    image_names: List[str]
    docker_image: str
    docker_prod_image: str
    base_image: str

    @classmethod
    def from_environ(cls, repository: str, image_names: List[str], environ=None):
        if environ is None:
            environ = os.environ
        if "DOCKER_REGISTRY" not in environ:
            raise RuntimeError("DOCKER_REGISTRY environment variable is empty")
        docker_image = environ["DOCKER_REGISTRY"] + "/" + repository
        docker_prod_image = ""
        if "DOCKER_PROD_REGISTRY" in environ:
            docker_prod_image = environ["DOCKER_PROD_REGISTRY"] + "/" + repository
        base_image = docker_image + "/base"
        return cls(
            image_names=image_names,
            docker_image=docker_image,
            docker_prod_image=docker_prod_image,
            base_image=base_image,
        )

    def get_docker_images(self, labels: List[str], for_prod: bool = False):
        docker_image = self.docker_prod_image if for_prod else self.docker_image
        if not docker_image:
            return
        for image_name in self.image_names:
            for label in labels:
                yield docker_image + "/" + image_name + ":" + label


@dataclass
class Context:
    image_type: ImageType
    version: str
    release: Optional[Release] = None


def prepare_context(environ=None):
    if environ is None:
        environ = os.environ
    image_type, version, release = ImageType.dev, "", None
    github_ref = environ["GITHUB_REF"]
    m = re_tags.match(github_ref)
    if m:
        version = m[1]
    if not version:
        m = re_pull.match(github_ref)
        if m:
            image_type, version = ImageType.pr, "pr-" + m[1]
    if environ["GITHUB_EVENT_NAME"] == "release":
        release = Release.prerelease
        prerelease = get_github_event(environ)["release"]["prerelease"]
        if not prerelease:
            release = Release.release
            image_type = ImageType.prod
    if not version:
        image_type, version = ImageType.sha, environ["GITHUB_SHA"][:8]
    return Context(image_type, version, release)


def version_labels(version: str) -> List[str]:
    labels = [version]
    splited = version.split(".")
    if len(splited) == 3:
        labels.append(splited[0])
        labels.append(splited[0] + "." + splited[1])
    return labels


def get_output_tags(settings: Settings, context: Context) -> List[str]:
    image_type = context.image_type
    tags = []
    if image_type is ImageType.pr:
        tags.extend(settings.get_docker_images([context.version]))
    else:
        tags.extend(settings.get_docker_images([context.version]))
        if context.release:
            tags.extend(settings.get_docker_images(["latest"]))
            if context.release is Release.release:
                dev_labels = version_labels(context.version)
                tags.extend(settings.get_docker_images(dev_labels))
        if image_type is ImageType.prod:
            tags.extend(settings.get_docker_images(["latest"], for_prod=True))
            prod_labels = version_labels(context.version)
            tags.extend(settings.get_docker_images(prod_labels, for_prod=True))
        if context.release is Release.release:
            tags.extend(settings.get_docker_images(["stable"]))
            if image_type is ImageType.prod:
                tags.extend(settings.get_docker_images(["stable"], for_prod=True))
    return tags


def bool_variable(value: bool):
    return "true" if value else ""


def get_variables(
    repository: str,
    image_names: List[str],
    with_pr: bool = False,
    with_sha: bool = False,
):
    variables: List[Tuple[str, str]] = []
    settings = Settings.from_environ(repository, image_names)
    push_image, login_cr, login_prod_cr = True, True, bool(settings.docker_prod_image)

    context = prepare_context()
    image_type = context.image_type
    if (image_type is ImageType.pr and not with_pr) or (
        image_type is ImageType.sha and not with_sha
    ):
        push_image, login_cr, login_prod_cr = False, False, False
    if image_type is not ImageType.prod:
        login_prod_cr = False
    tags = []
    if push_image:
        tags = get_output_tags(settings, context)

    variables = []
    variables.append(("version", context.version))
    variables.append(
        ("release_type", context.release.value if context.release is not None else "")
    )
    variables.append(("image_type", context.image_type.value))
    dt = datetime.utcnow()
    variables.append(("created", dt.strftime("%Y-%m-%dT%H:%M:%SZ")))
    variables.append(("base_image", settings.base_image + ":" + context.version))
    variables.append(("tags_gordo_base", ",".join(tags)))
    variables.append(("push_image", bool_variable(push_image)))
    variables.append(("login_cr", bool_variable(login_cr)))
    variables.append(("login_prod_cr", bool_variable(login_prod_cr)))
    return variables


def render_set_output(variables: List[Tuple[str, str]], environ=None):
    for name, value in variables:
        print("::set-output name=%s::%s" % (name, value))


def render_github_output(variables: List[Tuple[str, str]], environ=None):
    if environ is None:
        environ = os.environ
    with open(environ["GITHUB_OUTPUT"], "a") as f:
        for name, value in variables:
            f.write("%s=%s\n" % (name, value))


_renders = {
    "set-output": render_set_output,
    "github-output": render_github_output,
}
_default_render = "github-output"


def main():
    parser = argparse.ArgumentParser(
        description="Provides the variables such as docker images tags for GitHub workflow"
    )

    parser.add_argument(
        "-r",
        "--repository",
        required=True,
        help="Docker repository. Full url: <registry>/<repository>/<image_name>",
    )
    parser.add_argument(
        "-i", "--image-name", required=True, action="append", help="Docker image name"
    )
    parser.add_argument(
        "-p", "--with-pr", action="store_true", help="Run for PRs events"
    )
    parser.add_argument(
        "-s", "--with-sha", action="store_true", help="Run for non-releases"
    )
    parser.add_argument(
        "-e",
        "--render",
        choices=list(_renders.keys()),
        default=_default_render,
        help="Variables render type. Default: '%s'" % _default_render,
    )

    args = parser.parse_args()
    render = _renders[args.render]

    variables = get_variables(
        repository=args.repository,
        image_names=args.image_name,
        with_pr=args.with_pr,
        with_sha=args.with_sha,
    )
    render(variables)


if __name__ == "__main__":
    main()
