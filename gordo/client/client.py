# -*- coding: utf-8 -*-

import sys  # noqa
import copy
import requests
import logging
import itertools
import typing
from functools import wraps
from time import sleep
from threading import Lock
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


import pandas as pd
from sklearn.base import BaseEstimator
from werkzeug.exceptions import BadRequest

from gordo import serializer
from gordo.client.io import _handle_response
from gordo.client.io import HttpUnprocessableEntity
from gordo.client.utils import PredictionResult
from gordo.machine.dataset.data_provider.base import GordoBaseDataProvider
from gordo.server import utils as server_utils
from gordo.machine import Machine
from gordo.machine.metadata import Metadata


logger = logging.getLogger(__name__)


class Client:
    """
    Basic client shipped with Gordo

    Enables some basic communication with a deployed Gordo project
    """

    _mutex = Lock()
    machines: List[Machine] = []

    def __init__(
        self,
        project: str,
        target: typing.Optional[str] = None,
        host: str = "localhost",
        port: int = 443,
        scheme: str = "https",
        metadata: typing.Optional[dict] = None,
        data_provider: typing.Optional[GordoBaseDataProvider] = None,
        prediction_forwarder: typing.Optional[
            Callable[[pd.DataFrame, Machine, dict, pd.DataFrame], None]
        ] = None,
        batch_size: int = 100000,
        parallelism: int = 10,
        forward_resampled_sensors: bool = False,
        n_retries: int = 5,
        use_parquet: bool = False,
        session: Optional[requests.Session] = None,
        revision: Optional[str] = None,
    ):
        """

        Parameters
        ----------
        project: str
            Name of the project.
        target: Optional[str]
            Target name if desired to only make predictions against one target.
            Leave as None to run predictions against all targets in controller.
        host: str
            Host of where to find controller and other services.
        port: int
            Port to communicate on.
        scheme: str
            The request scheme to use, ie 'https'.
        metadata: Optional[dict]
            Arbitrary mapping of key-value pairs to save to influx with
            prediction runs in 'tags' property
        data_provider: Optional[GordoBaseDataProvider]
            The data provider to use for the dataset. If not set, the client
            will fall back to using the GET /prediction machine
        prediction_forwarder: Optional[Callable[[pd.DataFrame, Machine, dict, pd.DataFrame], None]]
            callable which will take a dataframe of predictions,
            ``Machine``, the metadata, and the dataframe of resampled sensor
            values and forward them somewhere.
        batch_size: int
            How many samples to send to the server, only applicable when data
            provider is supplied.
        parallelism: int
            The maximum number of tasks to run at a given time when
            running predictions
        forward_resampled_sensors: bool
            If true then forward resampled sensor values to the prediction_forwarder
        n_retries: int
            Number of times the client should attempt to retry a failed prediction request. Each time the client
            retires the time it sleeps before retrying is exponentially calculated.
        use_parquet: bool
            Pass the data to the server using the parquet protocol. Default is True
            and recommended as it's more efficient for larger batch sizes. If False JSON
            is used for sending the data back and forth.
        session: Optional[requests.Session]
            The http session object to use for making requests.
        revision: Optional[str]
            Specific project revision to use when connecting to the server.
            Defaults to asking the server for the latest revision its capable of serving.
        """

        self.base_url = f"{scheme}://{host}:{port}"
        self.server_endpoint = f"{self.base_url}/gordo/v0/{project}"
        self.metadata = metadata if metadata is not None else dict()
        self.prediction_forwarder = prediction_forwarder
        self.data_provider = data_provider
        self.use_parquet = use_parquet
        self.project_name = project

        # Default, failing back to /prediction on http code 422
        self.prediction_path = "/anomaly/prediction"
        self.batch_size = batch_size
        self.parallelism = parallelism
        self.forward_resampled_sensors = forward_resampled_sensors
        self.n_retries = n_retries
        self.query = f"?format={'parquet' if use_parquet else 'json'}"
        self.target = target
        self.session = session or requests.Session()
        self.revision = revision

        self.machines = self.get_machines()

    @property
    def session(self):
        return self._session

    @session.setter
    def session(self, session: requests.Session):
        session.get = self._expired_revision_check(session.get)  # type: ignore
        session.post = self._expired_revision_check(session.post)  # type: ignore
        self._session = session

    def _expired_revision_check(self, f):
        """
        Wrap a request function to check for 410 status codes, and update
        the current revision if so.
        """

        @wraps(f)
        def _wrapper(*args, **kwargs):
            resp = f(*args, **kwargs)
            if resp.status_code == 410:
                # 410 Gone - Unable to serve this revision, update and try again.
                self.update_revision()
                return f(*args, **kwargs)
            else:
                return resp

        return _wrapper

    def update_revision(self):
        self.revision = None  # Triggers fetching the latest revision

    @property
    def revision(self):
        """
        Revision being used against the server
        """
        return self._revision

    @revision.setter
    def revision(self, revision: Optional[str]):
        """
        Verifies the server can satisfy this revision.
        If the supplied value is ``None`` it will get the latest revision
        from the server.
        """
        req = requests.Request(
            "GET", f"{self.base_url}/gordo/v0/{self.project_name}/revisions"
        )
        resp = self.session.send(req.prepare())
        if not resp.ok:
            raise IOError(f"Failed to get revisions: {resp.content.decode()}")

        if revision:
            supported_revisions = resp.json()["available-revisions"]
            if revision not in supported_revisions:
                raise LookupError(
                    f"Revision '{revision}' cannot be served "
                    f"from revisions: {supported_revisions}"
                )
            self._revision = revision
        else:
            self._revision = resp.json()["latest"]

        # Update session headers, to send the revision we want.
        self.session.headers.update({"revision": self._revision})
        self.get_metadata(force_refresh=True)

    def get_machines(self) -> List[Machine]:
        # Thread safe single access and updating of machines.
        with self._mutex:
            machines = self._machines_from_server()
            return self._filter_machines(machines=machines, target=self.target)

    @staticmethod
    def _filter_machines(
        machines: typing.List[Machine], target: typing.Optional[str] = None
    ) -> typing.List[Machine]:
        """
        Based on the current configuration, filter out machines which the client
        should not care about.

        Parameters
        ----------
        machines: List[Machine]
            List of Machine objs
        target: Optional[str] (None)
            Name of the target/machine/machine we should filter down to

        Returns
        -------
        List[Machine]
            The filtered ``Machine``s
        """

        original_machines = copy.copy(machines)

        # Filter down to single machine if requested
        if target:
            machines = [ep for ep in machines if ep.name == target]

            # And check for single result and that it's healthy
            if len(machines) != 1:
                raise ValueError(
                    f"Found {'multiple' if len(machines) else 'no'} machines matching "
                    f"target name '{target}' in {original_machines}"
                )

        # finally, raise an error if all this left us without any machines
        if not machines:
            raise ValueError(
                f"Found no machines out of supplied machines: {original_machines} after filtering"
            )
        return machines

    def _machines_from_server(self) -> typing.List[Machine]:
        """
        Get a list of machines by querying controller
        """
        resp = self.session.get(f"{self.server_endpoint}/models")
        if not resp.ok:
            raise IOError(f"Failed to get machines: {repr(resp.content)}")
        else:
            model_names = resp.json()["models"]
            with ThreadPoolExecutor(max_workers=self.parallelism) as executor:
                machines = executor.map(self._machine_from_server, model_names)
                return list(machines)

    def _machine_from_server(self, name: str) -> Machine:
        """
        Create a :class:`gordo.workflow.config_elements.machine.Machine`
        from this model's endpoint on the server

        Parameters
        ----------
        name: str
            Name of this machine

        Returns
        -------
        Machine
        """
        resp = self.session.get(
            f"{self.base_url}/gordo/v0/{self.project_name}/{name}/metadata"
        )
        if resp.ok:
            metadata = resp.json()["metadata"]
            return Machine(**metadata)
        else:
            raise IOError(
                f"Unable to create machine '{name}': {resp.content.decode(errors='ignore')}"
            )

    def download_model(self) -> typing.Dict[str, BaseEstimator]:
        """
        Download the actual model(s) from the ML server /download-model

        Returns
        -------
        Dict[str, BaseEstimator]
            Mapping of target name to the model
        """
        models = dict()
        for machine in self.machines:
            resp = self.session.get(
                f"{self.base_url}/gordo/v0/{self.project_name}/{machine.name}/download-model"
            )
            if resp.ok:
                models[machine.name] = serializer.loads(resp.content)
            else:
                raise IOError(f"Failed to download model: '{repr(resp.content)}'")
        return models

    def get_metadata(self, force_refresh: bool = False) -> typing.Dict[str, Metadata]:
        """
        Get the metadata for each target

        Parameters
        ----------
        force_refresh : bool
            Even if the previous request was cached, make a new request for metadata.

        Returns
        -------
        Dict[str, Metadata]
            Mapping of target names to their metadata
        """
        if hasattr(self, "metadata_") and not force_refresh:
            return self.metadata_.copy()

        if force_refresh:
            self.machines = self.get_machines()  # Forced refresh if
        self.metadata_: Dict[str, Metadata] = {
            ep.name: ep.metadata for ep in self.machines
        }
        return self.metadata_.copy()

    def predict(
        self,
        start: datetime,
        end: datetime,
        refresh_machines: bool = False,
        machine_names: Optional[List[str]] = None,
    ) -> typing.Iterable[typing.Tuple[str, pd.DataFrame, typing.List[str]]]:
        """
        Start the prediction process.

        Parameters
        ----------
        start: datetime
        end: datetime
        refresh_machines : bool
            Before running predictions, refresh the current machines. Default
            ``False`` and will use the machines obtained during initialization.
        machine_names: Optional[List[str]]
            Optionally only target certain machines, referring to them by name.

        Returns
        -------
        List[Tuple[str, pandas.core.DataFrame, List[str]]
            A list of tuples, where:
              0th element is the target name
              1st element is the dataframe of the predictions; complete with a DateTime index.
              2nd element is a list of error messages (if any) for running the predictions
        """
        if refresh_machines:
            self.machines = self.get_machines()

        # Select machines if the machine_names were provided.
        if machine_names:
            machines = [ep for ep in self.machines if ep.name in machine_names]
        else:
            machines = self.machines

        # For every machine, start making predictions for the time range
        with ThreadPoolExecutor(max_workers=self.parallelism) as executor:
            jobs = executor.map(
                lambda ep: self.predict_single_machine(ep, start, end), machines
            )
            return [(j.name, j.predictions, j.error_messages) for j in jobs]

    def predict_single_machine(
        self, machine: Machine, start: datetime, end: datetime
    ) -> PredictionResult:
        """
        Get predictions based on the /prediction POST machine of Gordo ML Servers

        Parameters
        ----------
        machine: Machine
            Named tuple which has 'machine' specifying the full url to the base ml server
        start: datetime
        end: datetime

        Returns
        -------
        dict
            Prediction response from /prediction GET
        """

        # Fetch all of the raw data
        X, y = self._raw_data(machine, start, end)

        # Forward sensor data
        if self.prediction_forwarder is not None and self.forward_resampled_sensors:
            self.prediction_forwarder(resampled_sensor_data=X)  # type: ignore

        max_indx = len(X.index) - 1  # Maximum allowable index values

        # Start making batch predictions
        with ThreadPoolExecutor(max_workers=self.parallelism) as executor:
            jobs = executor.map(
                lambda i: self._send_prediction_request(
                    X,
                    y,
                    chunk=slice(i, i + self.batch_size),
                    machine=machine,
                    start=X.index[i],
                    end=X.index[
                        i + self.batch_size
                        if i + self.batch_size <= max_indx
                        else max_indx
                    ],
                ),
                range(0, X.shape[0], self.batch_size),
            )

            # Accumulate the batched predictions
            prediction_dfs = list()
            error_messages: List[str] = list()
            for prediction_result in jobs:
                if prediction_result.predictions is not None:
                    prediction_dfs.append(prediction_result.predictions)
                error_messages.extend(prediction_result.error_messages)

            predictions = (
                pd.concat(prediction_dfs).sort_index()
                if prediction_dfs
                else pd.DataFrame()
            )
        return PredictionResult(
            name=machine.name, predictions=predictions, error_messages=error_messages
        )

    def _send_prediction_request(
        self,
        X: pd.DataFrame,
        y: typing.Optional[pd.DataFrame],
        chunk: slice,
        machine: Machine,
        start: datetime,
        end: datetime,
    ):
        """
        Post a slice of data to the machine

        Parameters
        ----------
        X: pandas.core.DataFrame
            The data for the model, in pandas representation
        chunk: slice
            The slice to take from DataFrame.iloc for the batch size
        machine: Machine
        start: datetime
        end: datetime

        Notes
        -----
        PredictionResult.predictions may be None if the prediction process fails

        Returns
        -------
        PredictionResult
        """

        kwargs: Dict[str, Any] = dict(
            url=f"{self.base_url}/gordo/v0/{self.project_name}/{machine.name}{self.prediction_path}{self.query}"
        )

        # We're going to serialize the data as either JSON or Arrow
        if self.use_parquet:
            kwargs["files"] = {
                "X": server_utils.dataframe_into_parquet_bytes(X.iloc[chunk]),
                "y": server_utils.dataframe_into_parquet_bytes(y.iloc[chunk])
                if y is not None
                else None,
            }
        else:
            kwargs["json"] = {
                "X": server_utils.dataframe_to_dict(X.iloc[chunk]),
                "y": server_utils.dataframe_to_dict(y.iloc[chunk])
                if y is not None
                else None,
            }

        # Start attempting to get predictions for this batch
        for current_attempt in itertools.count(start=1):
            try:
                try:
                    resp = _handle_response(self.session.post(**kwargs))
                except HttpUnprocessableEntity:
                    self.prediction_path = "/prediction"
                    kwargs[
                        "url"
                    ] = f"{self.base_url}/gordo/v0/{self.project_name}/{machine.name}{self.prediction_path}{self.query}"
                    resp = _handle_response(self.session.post(**kwargs))
            # If it was an IO or TimeoutError, we can retry
            except (
                IOError,
                TimeoutError,
                BadRequest,
                requests.ConnectionError,
                requests.HTTPError,
            ) as exc:
                if current_attempt <= self.n_retries:
                    time_to_sleep = min(2 ** (current_attempt + 2), 300)
                    logger.warning(
                        f"Failed to get response on attempt {current_attempt} out of {self.n_retries} attempts."
                    )
                    sleep(time_to_sleep)
                    continue
                else:
                    msg = (
                        f"Failed to get predictions for dates {start} -> {end} "
                        f"for target: '{machine.name}' Error: {exc}"
                    )
                    logger.error(msg)

                    return PredictionResult(
                        name=machine.name, predictions=None, error_messages=[msg]
                    )

            # No point in retrying a BadRequest
            except BadRequest as exc:
                msg = (
                    f"Failed with BadRequest error for dates {start} -> {end} "
                    f"for target: '{machine.name}' Error: {exc}"
                )
                logger.error(msg)
                return PredictionResult(
                    name=machine.name, predictions=None, error_messages=[msg]
                )

            # Process response and return if no exception
            else:

                predictions = self.dataframe_from_response(resp)

                # Forward predictions to any other consumer if registered.
                if self.prediction_forwarder is not None:
                    self.prediction_forwarder(  # type: ignore
                        predictions=predictions, machine=machine, metadata=self.metadata
                    )
                return PredictionResult(
                    name=machine.name, predictions=predictions, error_messages=[]
                )

    def _raw_data(
        self, machine: Machine, start: datetime, end: datetime
    ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch the required raw data in this time range which would
        satisfy this machine's /prediction POST

        Parameters
        ----------
        machine: Machine
            Named tuple representing the machine info from controller
        start: datetime
        end: datetime

        Returns
        -------
        Tuple[pandas.core.DataFrame, pandas.core.DataFrame]
            The dataframes representing X and y.
        """

        # We want to adjust for any model offset. If the model outputs less than it got in, it requires
        # extra data than what we're being asked to get predictions for.
        # just to give us some buffer zone.
        resolution = machine.dataset.resolution
        n_intervals = machine.metadata.build_metadata.model.model_offset + 5
        start = self._adjust_for_offset(
            dt=start, resolution=resolution, n_intervals=n_intervals
        )

        # Re-create the machine's dataset but updating to use the client's
        # data provider and changing the dates of data we want.
        config = machine.dataset.to_dict()
        config.update(
            dict(
                data_provider=self.data_provider,
                train_start_date=start,
                train_end_date=end,
            )
        )
        dataset = machine.dataset.from_dict(config)
        return dataset.get_data()

    @staticmethod
    def _adjust_for_offset(dt: datetime, resolution: str, n_intervals: int = 100):
        """
        Adjust the given date by multiplying ``n_intervals`` by ``resolution``. Such that
        a date of 12:00:00 with ``n_intervals=2`` and ``resolution='10m'`` (10 minutes)
        would result in 11:40

        Parameters
        ----------
        dt: datetime
            Initial datetime to adjust.
        resolution: str
            A string code capable of being parsed by :meth::`pandas.Timedelta`.
        n_intervals: int
            Number of resolution steps to take earlier than the given date.

        Returns
        -------
        datetime
            The new offset datetime object.

        Examples
        --------
        >>> import dateutil
        >>> date = dateutil.parser.isoparse("2019-01-01T12:00:00+00:00")
        >>> offset_date = Client._adjust_for_offset(dt=date, resolution='15m', n_intervals=5)
        >>> str(offset_date)
        '2019-01-01 10:45:00+00:00'
        """
        return dt - (pd.Timedelta(resolution) * n_intervals)

    @staticmethod
    def dataframe_from_response(response: typing.Union[dict, bytes]) -> pd.DataFrame:
        """
        The response from the server, parsed as either JSON / dict or raw bytes,
        of which would be expected to be loadable from :func:`server.utils.dataframe_from_parquet_bytes`

        Parameters
        ----------
        response: Union[dict, bytes]
            The parsed response from the ML server.

        Returns
        -------
        pandas.DataFrame
        """
        if isinstance(response, dict):
            predictions = server_utils.dataframe_from_dict(response["data"])
        else:
            predictions = server_utils.dataframe_from_parquet_bytes(response)
        return predictions


def make_date_ranges(
    start: datetime, end: datetime, max_interval_days: int, freq: str = "H"
):
    """
    Split start and end datetimes into a list of datetime intervals.
    If the interval between start and end is less than ``max_interval_days`` then
    the resulting list will contain the original start & end. ie. [(start, end)]

    Otherwise it will split the intervals by ``freq``, parse-able by pandas.

    Parameters
    ----------
    start: datetime
    end: datetime
    max_interval_days: int
        Maximum days between start and end before splitting into intervals
    freq: str
        String frequency parse-able by Pandas

    Returns
    -------
    List[Tuple[datetime, datetime]]
    """
    if (end - start).days >= max_interval_days:
        # Split into 1hr data ranges
        date_range = pd.date_range(start, end, freq=freq)
        return [
            (date_range[i], date_range[i + 1]) for i in range(0, len(date_range) - 1)
        ]
    else:
        return [(start, end)]
