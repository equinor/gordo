# -*- coding: utf-8 -*-

import re
import datetime
import pandas as pd


class BaseDescriptor:
    """
    Base descriptor class

    New object should override __set__(self, instance, value) method to check
    if 'value' meets required needs.
    """

    def __get__(self, instance, owner):
        return instance.__dict__[self.name]

    def __set_name__(self, owner, name):
        self.name = name

    def __set__(self, instance, value):
        raise NotImplementedError("Setting value not implemented for this Validator!")


class ValidDataset(BaseDescriptor):
    """
    Descriptor for attributes requiring type gordo_infrastructure.config_elements.Dataset
    """

    def __set__(self, instance, value):

        # Avoid circular dependency imports
        from gordo_components.workflow.config_elements.dataset import Dataset

        if not isinstance(value, Dataset):
            raise TypeError(
                f"Expected value to be an instance of Dataset config element, found {value}"
            )
        instance.__dict__[self.name] = value


class ValidDatasetKwargs(BaseDescriptor):
    """
    Descriptor for attributes requiring type gordo_infrastructure.config_elements.Dataset
    """

    def _verify_resolution(self, resolution: str):
        """
        Verifies that a resolution string is supported in pandas
        """
        try:
            pd.tseries.frequencies.to_offset(resolution)
        except ValueError:
            raise ValueError(
                'Values for "resolution" must match pandas frequency terms: '
                "http://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html"
            )

    def __set__(self, instance, value):
        if not isinstance(value, dict):
            raise TypeError(f"Expected kwargs to be an instance of dict, found {value}")

        # Check that if 'resolution' is defined, it's one of supported pandas resampling frequencies
        if "resolution" in value:
            self._verify_resolution(value["resolution"])
        instance.__dict__[self.name] = value


class ValidModel(BaseDescriptor):
    """
    Descriptor for attributes requiring type Union[dict, str]

    TODO: Once we have access to gordo_components we can validate the provided model itself is valid
    """

    def __set__(self, instance, value):
        if not isinstance(value, dict) and not isinstance(value, str):
            raise ValueError(
                f"Must provide a valid model config, either model path or mapping config"
            )
        instance.__dict__[self.name] = value


class ValidMetadata(BaseDescriptor):
    """
    Descriptor for attributes requiring type Optional[dict]
    """

    def __set__(self, instance, value):
        if value is not None and not isinstance(value, dict):
            raise ValueError(f"Can either be None or an instance of dict")
        instance.__dict__[self.name] = value


class ValidDatetime(BaseDescriptor):
    """
    Descriptor for attributes requiring valid datetime.datetime attribute
    """

    def __set__(self, instance, value):
        if not isinstance(value, datetime.datetime):
            raise ValueError(f"'{value}' is not a valid datetime.datetime object!")
        instance.__dict__[self.name] = value


class ValidTagList(BaseDescriptor):
    """
    Descriptor for attributes requiring a non-empty list of strings
    """

    def __set__(self, instance, value):
        if (
            len(value) == 0
            or not isinstance(value, list)
            or not isinstance(value[0], str)
        ):
            raise ValueError(f"Requires setting a non-empty list of strings")
        instance.__dict__[self.name] = value


class ValidUrlString(BaseDescriptor):
    """
    Descriptor for use in objects which require valid URL values.
    Where 'valid URL values' is Gordo's version: alphanumeric with dashes.

    Use:

    class MySpecialClass:

        url_attribute = ValidUrlString()

        ...

    myspecialclass = MySpecialClass()

    myspecialclass.url_attribute = 'this-is-ok'
    myspecialclass.url_attribute = 'this will r@ise a ValueError'
    """

    def __set__(self, instance, value):
        if not self.valid_url_string(value):
            raise ValueError(
                f"'{value}' is not a valid Gordo url value. Only lower-case alphanumeric with dashes allowed.'"
            )
        if len(value) > 63:
            raise ValueError(
                "Should be less than 63 chars, as required by Kubernetes/AKS DNS requirements."
            )
        instance.__dict__[self.name] = value

    @staticmethod
    def valid_url_string(string: str) -> bool:
        """
        What we (Gordo) deem to be a suitable URL is the same as kubernetes
        lowercase alphanumeric with dashes but not ending or starting with a dash

        Parameters
        ----------
            string: str - String to check

        Returns
        -------
            bool
        """
        match = re.match(
            r"^([a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*)$",
            string,
        )
        return bool(match)
