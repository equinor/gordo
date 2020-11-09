# -*- coding: utf-8 -*-
import collections
import copy
import re
import datetime

import pandas as pd
import dateutil.parser
import logging

from gordo.serializer import from_definition
from gordo_dataset.sensor_tag import SensorTag


logger = logging.getLogger(__name__)


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
    Descriptor for attributes requiring type :class:`gordo.workflow.config_elements.Dataset`
    """

    def __set__(self, instance, value):

        # Avoid circular dependency imports
        from gordo_dataset.base import GordoBaseDataset

        if not isinstance(value, GordoBaseDataset):
            raise TypeError(
                f"Expected value to be an instance of GordoBaseDataset, found {value}"
            )
        instance.__dict__[self.name] = value


class ValidDatasetKwargs(BaseDescriptor):
    """
    Descriptor for attributes requiring type :class:`gordo.workflow.config_elements.Dataset`
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
    """

    def __set__(self, instance, value):
        if getattr(instance, "_strict", True):
            try:
                from_definition(value)
            except Exception as e:
                raise ValueError(f"Pipeline from definition failed: {e}")
        instance.__dict__[self.name] = value


class ValidMetadata(BaseDescriptor):
    """
    Descriptor for attributes requiring type Optional[dict]
    """

    def __set__(self, instance, value):
        from gordo.machine.metadata import Metadata

        if value is not None and not any(
            isinstance(value, Obj) for Obj in (dict, Metadata)
        ):
            raise ValueError(f"Can either be None or an instance of dict or Metadata")
        instance.__dict__[self.name] = value


class ValidDataProvider(BaseDescriptor):
    """
    Descriptor for DataProvider
    """

    def __set__(self, instance, value):

        # Avoid circular dependency imports
        from gordo_dataset.data_provider.base import GordoBaseDataProvider

        if not isinstance(value, GordoBaseDataProvider):
            raise TypeError(
                f"Expected value to be an instance of GordoBaseDataProvider, "
                f"found {value} "
            )
        instance.__dict__[self.name] = value


class ValidMachineRuntime(BaseDescriptor):
    """
    Descriptor for runtime dict in a machine object. Must be a valid runtime, but also
    must contain server.resources.limits/requests.memory/cpu to be valid.
    """

    def __set__(self, instance, value):
        if not isinstance(value, dict):
            raise ValueError(f"Runtime must be an instance of dict")
        value = self._verify_reporters(value)
        value = fix_runtime(value)
        instance.__dict__[self.name] = value

    @staticmethod
    def _verify_reporters(value: dict):
        """
        Verify the expected existence and structure of runtime.reporters
        """
        runtime = copy.deepcopy(value)
        if "reporters" not in runtime:
            runtime["reporters"] = list()
        else:
            assert isinstance(runtime["reporters"], list), "reporters should be a list"
        assert all(
            isinstance(rptr, dict) or isinstance(rptr, str)
            for rptr in runtime["reporters"]
        ), "All elements in 'reporters' should be a dict or str instances."
        return runtime


def fix_runtime(runtime_dict):
    """A valid runtime description must satisfy that any resource
    description must have that limit >= requests. This function will bump any limits
    that is too low."""
    runtime_dict = copy.deepcopy(runtime_dict)
    # We must also limit/request errors

    for key, val in runtime_dict.items():
        if isinstance(val, collections.Mapping):
            resource = val.get("resources")
            if resource:
                runtime_dict[key]["resources"] = fix_resource_limits(resource)
    return runtime_dict


def fix_resource_limits(resources: dict) -> dict:
    """
    Resource limitations must be higher or equal to resource requests, if they are
    both specified. This bumps any limits to the corresponding request if they are both
    set.

    Parameters
    ----------
    resources: dict
        Dictionary with possible requests/limits

    Examples
    --------
    >>> fix_resource_limits({"requests": {"cpu": 10}, "limits":{"cpu":9}})
    {'requests': {'cpu': 10}, 'limits': {'cpu': 10}}
    >>> fix_resource_limits({"requests": {"cpu": 10}})
    {'requests': {'cpu': 10}}


    Returns
    -------
    dict:
        A copy of `resource_dict` with the any limits bumped to the corresponding request if
        they are both set.
    """
    resources = copy.deepcopy(resources)
    requests = resources.get("requests", dict())
    limits = resources.get("limits", dict())
    request_memory = requests.get("memory")
    limits_memory = limits.get("memory")
    requests_cpu = requests.get("cpu")
    limits_cpu = limits.get("cpu")

    for r in [request_memory, limits_memory, requests_cpu, limits_cpu]:
        if r is not None and not isinstance(r, int):
            raise ValueError(
                f"Resource descriptions must be integers, and '{r}' is not."
            )
    if (
        limits_memory is not None
        and request_memory is not None
        and request_memory > limits_memory
    ):
        logger.warning(
            f"Memory limit {limits_memory} can not be smaller than memory "
            f"request {request_memory}, increasing memory limit to be equal"
            f" to request. "
        )
        limits["memory"] = request_memory
    if (
        limits_cpu is not None
        and requests_cpu is not None
        and requests_cpu > limits_cpu
    ):
        logger.warning(
            f"CPU limit {limits.get('cpu')} can not be smaller than cpu request"
            f" {requests.get('cpu')}, increasing cpu limit to be equal to request."
        )
        limits["cpu"] = requests_cpu
    return resources


class ValidDatetime(BaseDescriptor):
    """
    Descriptor for attributes requiring valid datetime.datetime attribute
    """

    def __set__(self, instance, value):
        datetime_value = None
        if isinstance(value, datetime.datetime):
            datetime_value = value
        elif isinstance(value, str):
            datetime_value = dateutil.parser.isoparse(value)
        else:
            raise ValueError(
                f"'{value}' is not a valid datetime.datetime object or string!"
            )

        if datetime_value.tzinfo is None:
            raise ValueError(f"Provide timezone to timestamp '{value}'")

        instance.__dict__[self.name] = datetime_value


class ValidTagList(BaseDescriptor):
    """
    Descriptor for attributes requiring a non-empty list of strings
    """

    def __set__(self, instance, value):
        if (
            len(value) == 0
            or not isinstance(value, list)
            or not any(isinstance(value[0], inst) for inst in (str, dict, SensorTag))
        ):
            raise ValueError(f"Requires setting a non-empty list of strings")
        instance.__dict__[self.name] = value


class ValidUrlString(BaseDescriptor):
    """
    Descriptor for use in objects which require valid URL values.
    Where 'valid URL values' is Gordo's version: alphanumeric with dashes.

    Use:

    .. code-block:: python

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
                f"'{value}' should be less than 63 chars, as required by Kubernetes/AKS DNS requirements."
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
        return bool(
            re.match(
                r"^([a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*)$",
                string,
            )
        )
