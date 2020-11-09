# -*- coding: utf-8 -*-
import json
import logging
from datetime import datetime
from typing import Dict, Any, Union, Optional

import numpy as np
import yaml

from gordo_dataset.base import GordoBaseDataset
from gordo.machine.validators import (
    ValidUrlString,
    ValidMetadata,
    ValidModel,
    ValidDataset,
    ValidMachineRuntime,
)
from gordo.machine.metadata import Metadata
from gordo.workflow.workflow_generator.helpers import patch_dict


logger = logging.getLogger(__name__)


class Machine:
    """
    Represents a single machine in a config file
    """

    name = ValidUrlString()
    project_name = ValidUrlString()
    host = ValidUrlString()
    model = ValidModel()
    dataset = ValidDataset()
    metadata = ValidMetadata()
    runtime = ValidMachineRuntime()
    _strict = True

    def __init__(
        self,
        name: str,
        model: dict,
        dataset: Union[GordoBaseDataset, dict],
        project_name: str,
        evaluation: Optional[dict] = None,
        metadata: Optional[Union[dict, Metadata]] = None,
        runtime=None,
    ):

        if runtime is None:
            runtime = dict()
        if evaluation is None:
            evaluation = dict(cv_mode="full_build")
        if metadata is None:
            metadata = dict()
        self.name = name
        self.model = model
        self.dataset = (
            dataset
            if isinstance(dataset, GordoBaseDataset)
            else GordoBaseDataset.from_dict(dataset)
        )
        self.runtime = runtime
        self.evaluation = evaluation
        self.metadata = (
            metadata
            if isinstance(metadata, Metadata)
            else Metadata.from_dict(metadata)  # type: ignore
        )
        self.project_name = project_name

        self.host = f"gordoserver-{self.project_name}-{self.name}"

    @classmethod
    def from_config(  # type: ignore
        cls, config: Dict[str, Any], project_name: str, config_globals=None
    ):
        """
        Construct an instance from a block of YAML config file which represents
        a single Machine; loaded as a ``dict``.

        Parameters
        ----------
        config: dict
            The loaded block of config which represents a 'Machine' in YAML
        project_name: str
            Name of the project this Machine belongs to.
        config_globals:
            The block of config within the YAML file within `globals`

        Returns
        -------
        :class:`~Machine`
        """
        if config_globals is None:
            config_globals = dict()

        name = config["name"]
        model = config.get("model") or config_globals.get("model")

        local_runtime = config.get("runtime", dict())
        runtime = patch_dict(config_globals.get("runtime", dict()), local_runtime)

        dataset_config = patch_dict(
            config.get("dataset", dict()), config_globals.get("dataset", dict())
        )
        dataset = GordoBaseDataset.from_dict(dataset_config)
        evaluation = patch_dict(
            config_globals.get("evaluation", dict()), config.get("evaluation", dict())
        )

        metadata = Metadata(
            user_defined={
                "global-metadata": config_globals.get("metadata", dict()),
                "machine-metadata": config.get("metadata", dict()),
            }
        )
        return cls(
            name,
            model,
            dataset,
            metadata=metadata,
            runtime=runtime,
            project_name=project_name,
            evaluation=evaluation,
        )

    def __str__(self):
        return yaml.dump(self.to_dict())

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> "Machine":
        """
        Get an instance from a dict taken from :func:`~Machine.to_dict`
        """
        # No special treatment required, just here for consistency.
        return cls(**d)

    def to_dict(self):
        """
        Convert to a ``dict`` representation along with all attributes which
        can also be converted to a ``dict``. Can reload with :func:`~Machine.from_dict`
        """
        return {
            "name": self.name,
            "dataset": self.dataset.to_dict(),
            "model": self.model,
            "metadata": self.metadata.to_dict(),
            "runtime": self.runtime,
            "project_name": self.project_name,
            "evaluation": self.evaluation,
        }

    def report(self):
        """ 
        Run any reporters in the machine's runtime for the current state.

        Reporters implement the :class:`gordo.reporters.base.BaseReporter` and
        can be specified in a config file of the machine for example:

        .. code-block:: yaml

            runtime:
              reporters:
                - gordo.reporters.postgres.PostgresReporter:
                    host: my-special-host

        """
        # Avoid circular dependency with reporters which import Machine
        from gordo.reporters.base import BaseReporter

        for reporter in map(BaseReporter.from_dict, self.runtime.get("reporters", [])):
            logger.debug(f"Using reporter: {reporter}")
            reporter.report(self)


class MachineEncoder(json.JSONEncoder):
    """
    A JSONEncoder for machine objects, handling datetime.datetime objects as strings
    and handles any numpy numeric instances; both of which common in the ``dict``
    representation of a :class:`~gordo.machine.Machine`

    Example
    -------
    >>> from pytz import UTC
    >>> s = json.dumps({"now":datetime.now(tz=UTC)}, cls=MachineEncoder, indent=4)
    >>> s = '{"now": "2019-11-22 08:34:41.636356+"}'
    """

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S.%f+%z")
        # Typecast builtin and numpy ints and floats to builtin types
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        elif np.issubdtype(type(obj), np.integer):
            return int(obj)
        else:
            return json.JSONEncoder.default(self, obj)
