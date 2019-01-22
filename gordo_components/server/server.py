# -*- coding: utf-8 -*-

import io
import os
import logging
import timeit
import dateutil.parser  # type: ignore
import typing

from functools import wraps
from typing import Callable
from datetime import datetime

import numpy as np
import pandas as pd

from flask import Flask, request, send_file, g
from flask_restplus import Resource, fields, Api

from gordo_components import __version__, serializer
from gordo_components.dataset.datasets import TimeSeriesDataset
from gordo_components.data_provider.base import GordoBaseDataProvider


logger = logging.getLogger(__name__)


MODEL_LOCATION_ENV_VAR = "MODEL_LOCATION"
MODEL = None
MODEL_METADATA = None

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


class Base(Resource):

    model = None
    metadata = dict()  # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set the global variables for MODEL and MODEL_METADATA if they haven't been set already.
        if MODEL is None:
            load_model_and_metadata()

        self.model = MODEL
        self.metadata = MODEL_METADATA


class PredictionApiView(Base):
    """
    Serve model predictions
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
        X: np.ndarray - 2d array of sample(s)

        Returns
        -------
        np.ndarray
        """
        try:
            return self.model.predict(X)  # type: ignore

        # Model may only be a transformer
        except AttributeError:
            try:
                return self.model.transform(X)  # type: ignore
            except Exception as exc:
                logger.error(f"Failed to predict or transform; error: {exc}")
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
            logger.critical(f"Failed to predict or transform; error: {err}")
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

        parameters = request.get_json()
        if not any(k in parameters for k in ("start", "end")):
            return (
                {
                    "error": "must provide iso8601 formatted dates with "
                    "timezone-information for parameters 'start' and 'end'"
                },
                400,
            )

        try:
            start = self._parse_iso_datetime(parameters["start"])
            end = self._parse_iso_datetime(parameters["end"])
        except ValueError:
            logger.error(
                f"Failed to parse start and/or end date to ISO: start: "
                f"{parameters['start']} - end: {parameters['end']}"
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

        freq = pd.tseries.frequencies.to_offset(self.metadata["dataset"]["resolution"])

        dataset = TimeSeriesDataset(
            data_provider=g.data_provider,
            from_ts=start - freq.delta,
            to_ts=end,
            resolution=self.metadata["dataset"]["resolution"],
            tag_list=self.metadata["dataset"]["tag_list"],
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
        tags = self.metadata["dataset"]["tag_list"]
        for prediction, time_stamp in zip(xhat, X.index):

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


class MetaDataView(Base):
    """
    Serve model / server metadata
    """

    def get(self):
        """
        Get metadata about this endpoint, also serves as /healthcheck endpoint
        """
        return {
            "gordo-server-version": __version__,
            "metadata": self.metadata,
            "env": {MODEL_LOCATION_ENV_VAR: os.environ.get(MODEL_LOCATION_ENV_VAR)},
        }


class DownloadModel(Base):
    """
    Download the trained model

    suitable for reloading via gordo_components.serializer.loads()
    """

    @api.doc(
        description="Download model, loadable via gordo_components.serializer.loads"
    )
    def get(self):
        """
        Download model - loadable via gordo_components.serializer.loads
        """
        serialized_model = serializer.dumps(self.model)
        buff = io.BytesIO(serialized_model)
        return send_file(buff, attachment_filename="model.tar.gz")


def adapt_proxy_deployment(wsgi_app: typing.Callable):
    """
    Special note about deploying behind Ambassador, or prefixed proxy paths in general:

        When deployed on kubernetes/ambassador there is a prefix in-front of the
        server. ie. /gordo/v0/some-project-name/some-target.

        The server itself only knows about routes to the right of such a prefix:
        such as '/metadata' or '/predictions when in reality, the full path is
        /gordo/v0/some-project-name/some-target/metadata

        This is solved by getting the current application's assigned prefix,
        where HTTP_X_ENVOY_ORIGINAL_PATH is the *full* path, including the prefix.
        and PATH_INFO is the actual relative path the server knows about.

        This function wraps the WSGI app itself to map the current full path
        to the assigned route function.

        ie. /metadata -> metadata route function, by default, but updates
        /gordo/v0/some-project-name/some-target/metadata -> metadata route function
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


def load_model_and_metadata():
    """
    Loads a model from having the 'MODEL_LOCATION' environment variable
    and sets the global variables 'MODEL' to the loaded model, and 'MODEL_METADATA'
    to any existing metadata for that model.
    """
    logger.debug("Determining model location...")
    model_location = os.getenv(MODEL_LOCATION_ENV_VAR)
    if model_location is None:
        raise ValueError(f'Environment variable "{MODEL_LOCATION_ENV_VAR}" not set!')
    if not os.path.isdir(model_location):
        raise NotADirectoryError(
            f'The supplied directory: "{model_location}" does not exist!'
        )

    global MODEL, MODEL_METADATA
    MODEL = serializer.load(model_location)
    MODEL_METADATA = serializer.load_metadata(model_location)


def build_app(data_provider: typing.Optional[GordoBaseDataProvider] = None):
    """
    Build app and any associated routes
    """
    app = Flask(__name__)
    api.init_app(app)
    api.add_resource(PredictionApiView, "/prediction")
    api.add_resource(MetaDataView, "/metadata", "/healthcheck")
    api.add_resource(DownloadModel, "/download-model")

    app.wsgi_app = adapt_proxy_deployment(app.wsgi_app)  # type: ignore
    app.url_map.strict_slashes = False  # /path and /path/ are ok.

    @app.before_request
    def _reg_data_provider():
        g.data_provider = data_provider

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
