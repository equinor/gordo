# -*- coding: utf-8 -*-

import abc


class GordoBase(abc.ABC):

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """Initialize the model"""

    @abc.abstractmethod
    def get_params(self, deep=False):
        """Return a dict containing all parameters used to initialized object"""

    @abc.abstractclassmethod
    def load_from_dir(cls, directory: str):
        """Load this model from a directory"""

    @abc.abstractmethod
    def save_to_dir(self, directory: str):
        """Save this model to a directory"""
