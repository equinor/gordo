# -*- coding: utf-8 -*-

import os
import io
import logging
import timeit
import typing
import dateutil.parser  # noqa
from datetime import datetime

import numpy as np
import pandas as pd

from flask import Blueprint, current_app, request, g, send_file, make_response, jsonify
from flask_restplus import Resource, fields

from gordo_components import __version__, serializer
from gordo_components.dataset.datasets import TimeSeriesDataset
from gordo_components.server.mixins import ModelMixin
from gordo_components.server.rest_api import Api
from gordo_components.dataset.sensor_tag import SensorTag, normalize_sensor_tags


logger = logging.getLogger(__name__)

base_blueprint = Blueprint("base_model_view", __name__)

api = Api(
    app=base_blueprint,
    title="Gordo Base Model View API Docs",
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


class BaseModelView(Resource, ModelMixin):
    """
    The base model view.

    Both POST and GET should return the same data response, but can vary
    in how it collects the data.

    POST expects data to be provided, GET uses TimeSeriesDataset
    """

    methods = ["GET", "POST"]

    @property
    def frequency(self):
        """
        The frequency the model was trained with in the dataset
        """
        return pd.tseries.frequencies.to_offset(
            current_app.metadata["dataset"]["resolution"]
        )

    @property
    def tags(self) -> typing.List[SensorTag]:
        return normalize_sensor_tags(current_app.metadata["dataset"]["tag_list"])

    @staticmethod
    def _parse_iso_datetime(datetime_str: str) -> datetime:
        parsed_date = dateutil.parser.isoparse(datetime_str)  # type: ignore
        if parsed_date.tzinfo is None:
            raise ValueError(
                f"Provide timezone to timestamp {datetime_str}."
                f" Example: for UTC timezone use {datetime_str + 'Z'} or {datetime_str + '+00:00'} "
            )
        return parsed_date

    @api.response(200, "Success", API_MODEL_OUTPUT_POST)
    @api.doc(
        params={
            "start": "An ISO formatted datetime with timezone info string indicating prediction range start",
            "end": "An ISO formatted datetime with timezone info string indicating prediction range end",
        }
    )
    def get(self):
        """
        Process a GET request by fetching data ourselves
        """
        context = dict()  # type: typing.Dict[str, typing.Any]
        context["status-code"] = 200
        start_time = timeit.default_timer()

        params = request.get_json() or request.args

        if not all(k in params for k in ("start", "end")):
            message = dict(
                message="must provide iso8601 formatted dates with timezone-information for parameters 'start' and 'end'"
            )
            return make_response((jsonify(message), 400))

        try:
            start = self._parse_iso_datetime(params["start"])
            end = self._parse_iso_datetime(params["end"])
        except ValueError:
            logger.error(
                f"Failed to parse start and/or end date to ISO: start: "
                f"{params['start']} - end: {params['end']}"
            )
            message = dict(
                message="Could not parse start/end date(s) into ISO datetime. must provide iso8601 formatted dates for both."
            )
            return make_response((jsonify(message), 400))

        # Make request time span of one day
        if (end - start).days:
            message = dict(message="Need to request a time span less than 24 hours.")
            return make_response((jsonify(message), 400))

        dataset = TimeSeriesDataset(
            data_provider=g.data_provider,
            from_ts=start - self.frequency.delta,
            to_ts=end,
            resolution=current_app.metadata["dataset"]["resolution"],
            tag_list=self.tags,
        )
        X, _y = dataset.get_data()

        # Want resampled buckets equal or greater than start, but less than end
        # b/c if end == 00:00:00 and req = 10 mins, a resampled bucket starting
        # at 00:00:00 would imply it has data until 00:10:00; which is passed
        # the requested end datetime
        X = X[
            (X.index > start - self.frequency.delta)
            & (X.index + self.frequency.delta < end)
        ]
        return self._process_request(context=context, X=X, start_time=start_time)

    @api.response(200, "Success", API_MODEL_OUTPUT_POST)
    @api.expect(API_MODEL_INPUT_POST, validate=False)
    @api.doc(
        params={
            "X": "Nested list of samples to predict, or single list considered as one sample"
        }
    )
    def post(self):
        """
        Process a POST request by using provided user data
        """

        context = dict()  # type: typing.Dict[str, typing.Any]
        context["status-code"] = 200
        start_time = timeit.default_timer()

        X = request.json.get("X")

        if X is None:
            message = dict(message='Cannot predict without "X"')
            return make_response((jsonify(message), 400))

        X = np.asanyarray(X)

        if X.dtype == np.dtype("O"):
            message = dict(
                error="Either provided non numerical elements or records with different shapes. ie. [[0, 1, 2], [0, 1]]"
            )
            return make_response((jsonify(message), 400))

        # Reshape X to sample 1 record if a single record was given
        X = X.reshape(1, -1) if len(X.shape) == 1 else X
        return self._process_request(context=context, X=X, start_time=start_time)

    def _process_request(
        self,
        context: dict,
        X: typing.Union[pd.DataFrame, np.ndarray],
        start_time: float,
    ):
        """
        Construct a response which fetches model outputs (transformed) as well
        as original input, transformed model input and, if applicable, inverse
        transformed model output.

        Parameters
        ----------
        context: dict
            Current context mapping of the request. Must be dict where
            items are capable of JSON serialization
        X: Union[pandas.DataFrame, numpy.ndarray]
            Data to use for gathering outputs in the response
        start_time: float
            Start time of the original request

        Returns
        -------
        flask.Response
        """
        self.X = X

        try:
            output = self.get_model_output(model=current_app.model, X=X)
            transformed_model_input = self.get_transformed_input(
                model=current_app.model, X=X
            )
            if len(output.shape) == len(X.shape) and output.shape[1] == X.shape[1]:
                inverse_transformed_model_output = self.get_inverse_transformed_input(
                    model=current_app.model, X=output
                )
                self._output_matches_input_shape = True
            else:
                inverse_transformed_model_output = None
                self._output_matches_input_shape = False

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
        else:
            data = self.make_base_dataframe(
                original_input=X.values if isinstance(X, pd.DataFrame) else X,
                model_output=output,
                transformed_model_input=transformed_model_input,
                inverse_transformed_model_output=inverse_transformed_model_output,
                index=X.index if isinstance(X, pd.DataFrame) else None,
            )
            context["data"] = data.to_dict(orient="records")

        context["tag-names"] = self.tags
        context["time-seconds"] = f"{timeit.default_timer() - start_time:.4f}"
        return make_response((jsonify(context), context["status-code"]))

    def make_base_dataframe(
        self,
        original_input: np.ndarray,
        model_output: np.ndarray,
        transformed_model_input: np.ndarray,
        inverse_transformed_model_output: typing.Optional[np.ndarray] = None,
        index: typing.Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Construct a uniform dataframe of the data going in and out of the model
        Set here as a set and clear way of house the base response generates
        its dataframe.

        Parameters
        ----------
        original_input: np.ndarray
        model_output: np.ndarray
        transformed_model_input: np.ndarray
        inverse_transformed_model_output: Optional[np.ndarray]
        index: Optional[np.ndarray]

        Returns
        -------
        pd.DataFrame
        """

        def column_suffixes(array: np.ndarray) -> typing.Iterator[str]:
            if array.shape[1] == len(self.tags):
                return (tag.name for tag in self.tags)
            else:
                return map(str, range(model_output.shape[1]))

        # The initial data passed to the pipeline/model
        data = pd.DataFrame(
            original_input, columns=(f"original-input-{tag.name}" for tag in self.tags)
        )

        # Join in the model output column(s)
        data = data.join(
            pd.DataFrame(
                model_output,
                columns=(f"model-output-{i}" for i in column_suffixes(model_output)),
            ),
            how="outer",
        )

        # Join in the transformed model input column(s)
        data = data.join(
            pd.DataFrame(
                transformed_model_input,
                columns=(
                    f"transformed-model-input-{i}"
                    for i in column_suffixes(transformed_model_input)
                ),
            ),
            how="outer",
        )

        # If it exists, join in the inverse transformed model ouput
        if inverse_transformed_model_output is not None:
            data = data.join(
                pd.DataFrame(
                    inverse_transformed_model_output,
                    columns=(
                        f"inverse-transformed-model-output-{tag.name}"
                        for tag in self.tags
                    ),
                ),
                how="outer",
            )

        if index is not None:
            # Attempt to convert to ISO formatted str if it can, otherwise just the value
            # This is because objects in the dataframe need to be JSON serializable
            data.index = index
        return data


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


api.add_resource(BaseModelView, "/prediction")
api.add_resource(MetaDataView, "/metadata", "/healthcheck")
api.add_resource(DownloadModel, "/download-model")
