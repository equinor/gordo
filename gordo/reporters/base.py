import abc

from gordo.machine import Machine
from gordo.base import GordoBase


class BaseReporter(GordoBase, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def report(self, machine: Machine):
        """Report/log the machine"""
