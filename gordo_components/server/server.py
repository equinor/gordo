# -*- coding: utf-8 -*-

import io
import os
import logging
import timeit
import dateutil.parser  # type: ignore
import typing
import traceback
from functools import wraps
from datetime import datetime

import numpy as np
import pandas as pd

from flask import Flask, request, send_file, g, url_for, current_app
from flask_restplus import Resource, fields, Api as BaseApi
from sklearn.base import BaseEstimator

from gordo_components import __version__, serializer
from gordo_components.dataset.datasets import TimeSeriesDataset
import gordo_components.dataset.sensor_tag as sensor_tag
from gordo_components.data_provider.base import GordoBaseDataProvider

logger = logging.getLogger(__name__)


class Config:
    """Server config"""

    MODEL_LOCATION_ENV_VAR = "MODEL_LOCATION"


class Api(BaseApi):
    @property
    def specs_url(self):
        return url_for(self.endpoint("specs"), _external=False)


api = Api(
    title="Gordo API Docs",
    version=__version__,
    description="Documentation for the Gordo ML Server",
    default_label="Gordo Endpoints",
)

# POST type declarations
API_MODEL_INPUT_POST = api.model(
    "Prediction - Multiple Samples", {"X": fields.List(fields.List(fields.Float))}
)
API_MODEL_OUTPUT_POST = api.model(
    "Prediction - Output from POST", {"output": fields.List(fields.List(fields.Float))}
)


# GET type declarations
API_MODEL_INPUT_GET = api.model(
    "Prediction - Time range prediction",
    {"start": fields.DateTime, "end": fields.DateTime},
)
_tags = {
    fields.String: fields.Float
}  # tags of single prediction record {'tag-name': tag-value}
_single_prediction_record = {
    "start": fields.DateTime,
    "end": fields.DateTime,
    "tags": fields.Nested(_tags),
    "total_abnormality": fields.Float,
}
API_MODEL_OUTPUT_GET = api.model(
    "Prediction - Output from GET",
    {"output": fields.List(fields.Nested(_single_prediction_record))},
)


class PredictionApiView(Resource):
    """
    Serve model predictions via GET and POST methods

    If ``GET``
        Will take a ``start`` and ``end`` ISO format datetime string
        and give back predictions in the following example::

            {'output': [
                {'end': '2016-01-01 00:20:00+00:00',
                 'start': '2016-01-01 00:10:00+00:00',
                 'tags': {'tag-0': 0.6031807382481357,
                          'tag-1': 0.6850933943812884,
                          'tag-2': 0.37281826556659486,
                          'tag-3': 0.6688453143800668,
                          'tag-4': 0.3860472585679212,
                          'tag-5': 0.6704775418728366,
                          'tag-6': 0.36539023141775234,
                          'tag-7': 0.929348645859519,
                          'tag-8': 0.555020406599076,
                          'tag-9': 0.3908480506569232},
                 'total_anomaly': 1.8644325873977061}
             ],
             'status-code': 200,
             'time-seconds': '2.5015'}
    if ``POST``
        Will take the raw data in ``X`` and expected to be of the correct shape.
        This is much lower level, and reflects that raw input for the model.

        Returned example format::

            {'output': [
                [0.9620815728153441,
                 0.48481952794697486,
                 0.10449727234407678,
                 0.5820399931840917,],
                [0.7328459309239063,
                 0.7933340297343985,
                 0.4012604048104457,
                 0.9799145817366233,],
             'status-code': 200,
             'time-seconds': '0.0568'}

        where ``output`` represents the actual output of the model.
    """

    @staticmethod
    def _parse_iso_datetime(datetime_str: str) -> datetime:
        parsed_date = dateutil.parser.isoparse(datetime_str)  # type: ignore
        if parsed_date.tzinfo is None:
            raise ValueError(
                f"Provide timezone to timestamp {datetime_str}."
                f" Example: for UTC timezone use {datetime_str + 'Z'} or {datetime_str + '+00:00'} "
            )
        return parsed_date

    def get_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get the raw output from the current model given X.
        Will try to `predict` and then `transform`, raising an error
        if both fail.

        Parameters
        ----------
        X: np.ndarray
            2d array of sample(s)

        Returns
        -------
        np.ndarray
            The raw output of the model in numpy array form.
        """
        try:
            return current_app.model.predict(X)  # type: ignore

        # Model may only be a transformer
        except AttributeError:
            try:
                return current_app.model.transform(X)  # type: ignore
            except Exception as exc:
                tb = traceback.format_exc()
                logger.error(
                    f"Failed to predict or transform; error: {exc} - \nTraceback: {tb}"
                )
                raise

    @api.response(200, "Success", API_MODEL_OUTPUT_POST)
    @api.expect(API_MODEL_INPUT_POST, validate=False)
    @api.doc(
        params={
            "X": "Nested list of samples to predict, or single list considered as one sample"
        }
    )
    def post(self):
        """
        Get predictions
        """

        context = dict()  # type: typing.Dict[str, typing.Any]
        context["status-code"] = 200
        start_time = timeit.default_timer()

        X = request.json.get("X")

        if X is None:
            return {"error": 'Cannot predict without "X"'}, 400

        X = np.asanyarray(X)

        if X.dtype == np.dtype("O"):
            return (
                {
                    "error": "Either provided non numerical elements or records with different shapes."
                    "  ie. [[0, 1, 2], [0, 1]]"
                },
                400,
            )

        # Reshape X to sample 1 record if a single record was given
        X = X.reshape(1, -1) if len(X.shape) == 1 else X

        try:
            context["output"] = self.get_predictions(X).tolist()
        except ValueError as err:
            logger.error(f"Failed to predict or transform; error: {err}")
            context["error"] = f"ValueError: {str(err)}"
            context["status-code"] = 400
        # Model may only be a transformer, probably an AttributeError, but catch all to avoid logging other
        # exceptions twice if it happens.
        except Exception as exc:
            logger.error(f"Failed to predict or transform; error: {exc}")
            context["error"] = "Something unexpected happened; check your input data"
            context["status-code"] = 400

        context["time-seconds"] = f"{timeit.default_timer() - start_time:.4f}"
        return context, context["status-code"]

    @api.response(200, "Success", API_MODEL_OUTPUT_POST)
    @api.doc(
        params={
            "start": "An ISO formatted datetime with timezone info string indicating prediction range start",
            "end": "An ISO formatted datetime with timezone info string indicating prediction range end",
        }
    )
    def get(self):

        context = dict()  # type: typing.Dict[str, typing.Any]
        context["status-code"] = 200
        start_time = timeit.default_timer()

        params = request.get_json() or request.args

        if not all(k in params for k in ("start", "end")):
            return (
                {
                    "error": "must provide iso8601 formatted dates with "
                    "timezone-information for parameters 'start' and 'end'"
                },
                400,
            )

        try:
            start = self._parse_iso_datetime(params["start"])
            end = self._parse_iso_datetime(params["end"])
        except ValueError:
            logger.error(
                f"Failed to parse start and/or end date to ISO: start: "
                f"{params['start']} - end: {params['end']}"
            )
            return (
                {
                    "error": "Could not parse start/end date(s) into ISO datetime. "
                    "must provide iso8601 formatted dates for both."
                },
                400,
            )

        # Make request time span of one day
        if (end - start).days:
            return {"error": "Need to request a time span less than 24 hours."}, 400

        freq = pd.tseries.frequencies.to_offset(
            current_app.metadata["dataset"]["resolution"]
        )

        dataset = TimeSeriesDataset(
            data_provider=g.data_provider,
            from_ts=start - freq.delta,
            to_ts=end,
            resolution=current_app.metadata["dataset"]["resolution"],
            tag_list=sensor_tag.normalize_sensor_tags(
                current_app.metadata["dataset"]["tag_list"]
            ),
        )
        X, _y = dataset.get_data()

        # Want resampled buckets equal or greater than start, but less than end
        # b/c if end == 00:00:00 and req = 10 mins, a resampled bucket starting
        # at 00:00:00 would imply it has data until 00:10:00; which is passed
        # the requested end datetime
        X = X[(X.index > start - freq.delta) & (X.index + freq.delta < end)]

        try:
            xhat = self.get_predictions(X).tolist()

        # Model may only be a transformer, probably an AttributeError, but catch all to avoid logging other
        # exceptions twice if it happens.
        except Exception as exc:
            logger.critical(f"Failed to predict or transform; error: {exc}")
            return (
                {"error": "Something unexpected happened; check your input data"},
                400,
            )

        # In GET requests we need to pair the resulting predictions with their
        # specific timestamp and additionally match the predictions to the corresponding tags.
        data = []

        # This tags list is just for display/informative purposes, skipping the asset
        tags = [tag["name"] for tag in current_app.metadata["dataset"]["tag_list"]]

        for prediction, time_stamp in zip(xhat, X.index[-len(xhat) :]):

            # Auto encoders return double their input.
            # First half is input to model, second half is output of model
            tag_inputs = np.array(prediction[: len(tags)])
            tag_outputs = np.array(prediction[len(tags) :])
            tag_errors = np.abs(tag_inputs - tag_outputs)
            data.append(
                {
                    "start": f"{time_stamp}",
                    "end": f"{time_stamp + freq}",
                    "tags": {tag: error for tag, error in zip(tags, tag_errors)},
                    "total_anomaly": np.linalg.norm(tag_inputs - tag_outputs),
                }
            )
        context["output"] = data
        context["time-seconds"] = f"{timeit.default_timer() - start_time:.4f}"
        return context, context["status-code"]


class MetaDataView(Resource):
    """
    Serve model / server metadata
    """

    def get(self):
        """
        Get metadata about this endpoint, also serves as /healthcheck endpoint
        """
        model_location_env_var = current_app.config["MODEL_LOCATION_ENV_VAR"]
        return {
            "gordo-server-version": __version__,
            "metadata": current_app.metadata,
            "env": {model_location_env_var: os.environ.get(model_location_env_var)},
        }


class DownloadModel(Resource):
    """
    Download the trained model

    suitable for reloading via ``gordo_components.serializer.loads()``
    """

    @api.doc(
        description="Download model, loadable via gordo_components.serializer.loads"
    )
    def get(self):
        """
        Responds with a serialized copy of the current model being served.

        Returns
        -------
        bytes
            Results from ``gordo_components.serializer.dumps()``
        """
        serialized_model = serializer.dumps(current_app.model)
        buff = io.BytesIO(serialized_model)
        return send_file(buff, attachment_filename="model.tar.gz")


def adapt_proxy_deployment(wsgi_app: typing.Callable) -> typing.Callable:
    """
    Decorator specific to fixing behind-proxy-issues when on Kubernetes and
    using Envoy proxy.

    Parameters
    ----------
    wsgi_app: typing.Callable
        The underlying WSGI application of a flask app, for example

    Notes
    -----
    Special note about deploying behind Ambassador, or prefixed proxy paths in general:

    When deployed on kubernetes/ambassador there is a prefix in-front of the
    server. ie::

        /gordo/v0/some-project-name/some-target

    The server itself only knows about routes to the right of such a prefix:
    such as ``/metadata`` or ``/predictions`` when in reality, the full path is::

        /gordo/v0/some-project-name/some-target/metadata

    This is solved by getting the current application's assigned prefix,
    where ``HTTP_X_ENVOY_ORIGINAL_PATH`` is the *full* path, including the prefix.
    and ``PATH_INFO`` is the actual relative path the server knows about.

    This function wraps the WSGI app itself to map the current full path
    to the assigned route function.

    ie. ``/metadata`` -> metadata route function, by default, but updates
    ``/gordo/v0/some-project-name/some-target/metadata`` -> metadata route function

    Returns
    -------
    Callable

    Example
    -------
    >>> app = Flask(__name__)
    >>> app.wsgi_app = adapt_proxy_deployment(app.wsgi_app)
    """

    @wraps(wsgi_app)
    def wrapper(environ, start_response):

        # Script name can be "/gordo/v0/some-project-name/some-target/metadata"
        script_name = environ.get("HTTP_X_ENVOY_ORIGINAL_PATH", "")
        if script_name:

            # PATH_INFO could be either "/" or some local route such as "/metadata"
            path_info = environ.get("PATH_INFO", "")
            if path_info.rstrip("/"):

                # PATH_INFO must be something like "/metadata" or other local path
                # To figure out the prefix/script_name we remove it from the
                # full HTTP_X_ENVOY_ORIGINAL_PATH, so that something such as
                # /gordo/v0/some-project-name/some-target/metadata, becomes
                # /gordo/v0/some-project-name/some-target/
                script_name = script_name.replace(path_info, "")
            environ["SCRIPT_NAME"] = script_name

            # Now we can just ensure the PATH_INFO reflects the locally known path
            # such as /metadata and not /gordo/v0/some-project-name/some-target/metadata
            if path_info.startswith(script_name):
                environ["PATH_INFO"] = path_info[len(script_name) :]

        scheme = environ.get("HTTP_X_FORWARDED_PROTO", "")
        if scheme:
            environ["wsgi.url_scheme"] = scheme
        return wsgi_app(environ, start_response)

    return wrapper


def load_model_and_metadata(
    model_dir_env_var: str
) -> typing.Tuple[BaseEstimator, dict]:
    """
    Loads a model and metadata from the path found in ``model_dir_env_var``
    environment variable

    Parameters
    ----------
    model_dir_env_var: str
        The name of the environment variable which stores the location of the model

    Returns
    -------
    BaseEstimator, dict
        Tuple where the 0th element is the model, and the 1st element is the metadata
        associated with the model
    """
    logger.debug("Determining model location...")
    model_location = os.getenv(model_dir_env_var)
    if model_location is None:
        raise ValueError(f'Environment variable "{model_dir_env_var}" not set!')
    if not os.path.isdir(model_location):
        raise NotADirectoryError(
            f'The supplied directory: "{model_location}" does not exist!'
        )

    model = serializer.load(model_location)
    metadata = serializer.load_metadata(model_location)
    return model, metadata


def build_app(data_provider: typing.Optional[GordoBaseDataProvider] = None):
    """
    Build app and any associated routes
    """
    app = Flask(__name__)
    app.config.from_object(Config())
    api.init_app(app)
    api.add_resource(PredictionApiView, "/prediction")
    api.add_resource(MetaDataView, "/metadata", "/healthcheck")
    api.add_resource(DownloadModel, "/download-model")

    app.wsgi_app = adapt_proxy_deployment(app.wsgi_app)  # type: ignore
    app.url_map.strict_slashes = False  # /path and /path/ are ok.

    @app.before_request
    def _reg_data_provider():
        g.data_provider = data_provider

    with app.app_context():
        app.model, app.metadata = load_model_and_metadata(
            app.config["MODEL_LOCATION_ENV_VAR"]
        )

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 5555,
    debug: bool = False,
    data_provider: typing.Optional[GordoBaseDataProvider] = None,
):
    app = build_app(data_provider=data_provider)
    app.run(host, port, debug=debug)


if __name__ == "__main__":
    run_server()
