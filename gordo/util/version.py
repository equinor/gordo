import re
from typing import Optional, Union

from dataclasses import dataclass
from enum import Enum
from abc import ABCMeta, abstractmethod


class Version(metaclass=ABCMeta):
    @abstractmethod
    def get_version(self):
        ...


class Special(Enum):
    LATEST = "latest"
    STABLE = "stable"

    @classmethod
    def find(cls, version: str) -> Optional["Special"]:
        for special in cls:
            if special.value == version:
                return special
        return None


@dataclass(frozen=True)
class GordoSpecial(Version):
    special: Special

    def get_version(self):
        return self.special.value


release_re = re.compile(r"^(\d{1,5})(\.(\d+)(\.(\d+)(.*?)?)?)?$")


@dataclass(frozen=True)
class GordoRelease(Version):
    major: int
    minor: Optional[int] = None
    patch: Optional[int] = None
    suffix: Optional[str] = None

    def without_patch(self) -> bool:
        return self.suffix is None and self.patch is None

    def only_major_minor(self) -> bool:
        return (
            self.major is not None and self.minor is not None and self.without_patch()
        )

    def only_major(self) -> bool:
        return self.major is not None and self.minor is None and self.without_patch()

    def get_version(self):
        version = str(self.major)
        if self.minor is not None:
            version += "." + str(self.minor)
        if self.patch is not None:
            version += "." + str(self.patch)
        if self.suffix is not None:
            version += self.suffix
        return version


pr_prefix = "pr-"


@dataclass(frozen=True)
class GordoPR(Version):
    number: int

    def get_version(self):
        return "%s%d" % (pr_prefix, self.number)


sha_re = re.compile(r"^[0-9a-z]{8,40}$")


@dataclass(frozen=True)
class GordoSHA(Version):
    sha: str

    def get_version(self):
        return self.sha


def parse_version(
    gordo_version: str,
) -> Union[GordoRelease, GordoSpecial, GordoPR, GordoSHA]:
    """
    Parsing gordo version. Also supported gordo docker images tags

    Parameters
    ----------
    gordo_version: str

    Example
    -------
    >>> parse_version('2.3.5')
    GordoRelease(major=2, minor=3, patch=5, suffix=None)
    >>> parse_version('latest')
    GordoSpecial(special=<Special.LATEST: 'latest'>)

    Returns
    -------
    Union[GordoRelease, GordoSpecial, GordoPR, GordoSHA]

    """
    special_version = Special.find(gordo_version)
    if special_version is not None:
        return GordoSpecial(special_version)
    if gordo_version.find(pr_prefix) == 0:
        try:
            number = int(gordo_version[len(pr_prefix) :])
        except ValueError:
            raise ValueError("Malformed gordo PR version '%s'" % gordo_version)
        return GordoPR(number)
    m = release_re.match(gordo_version)
    if m:
        (major, _, minor, _, patch, suffix) = m.groups()
        return GordoRelease(
            int(major),
            int(minor) if minor else None,
            int(patch) if patch else None,
            suffix if suffix else None,
        )
    m = sha_re.match(gordo_version)
    if m:
        return GordoSHA(gordo_version)
    raise ValueError("Malformed gordo version '%s'" % gordo_version)
