# -*- coding: utf-8 -*-

import os
import io
import logging
import traceback
import timeit
import typing

import pandas as pd

from flask import Blueprint, current_app, g, send_file, make_response, jsonify
from flask_restplus import Resource, fields

from gordo_components import __version__, serializer
from gordo_components.server.rest_api import Api
from gordo_components.server import utils as server_utils
from gordo_components.model import utils as model_utils
from gordo_components.dataset.sensor_tag import SensorTag, normalize_sensor_tags
from gordo_components.server import model_io


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


class BaseModelView(Resource):
    """
    The base model view.

    Both POST and GET should return the same data response, but can vary
    in how it collects the data.

    POST expects data to be provided, GET uses TimeSeriesDataset

    A typical response might look like this::

         {
        'data': [
            {
           'end': ['2016-01-01T00:10:00+00:00'],
           'model-output': [0.0005317790200933814,
                            -0.0001525811239844188,
                            0.0008310950361192226,
                            0.0015755111817270517],
           'original-input': [0.9135588550070414,
                              0.3472517774179448,
                              0.8994921857179736,
                              0.11982773108991263],
           'start': ['2016-01-01T00:00:00+00:00'],
            },
            ...
        ],

     'tags': [{'asset': None, 'name': 'tag-0'},
              {'asset': None, 'name': 'tag-1'},
              {'asset': None, 'name': 'tag-2'},
              {'asset': None, 'name': 'tag-3'}],
     'time-seconds': '0.1937'}
    """

    methods = ["GET", "POST"]

    y: pd.DataFrame = None
    X: pd.DataFrame = None

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

    @property
    def target_tags(self) -> typing.List[SensorTag]:
        if "target_tag_list" in current_app.metadata["dataset"]:
            return normalize_sensor_tags(
                current_app.metadata["dataset"]["target_tag_list"]
            )
        else:
            return []

    @api.response(200, "Success", API_MODEL_OUTPUT_POST)
    @api.doc(
        params={
            "start": "An ISO formatted datetime with timezone info string indicating prediction range start",
            "end": "An ISO formatted datetime with timezone info string indicating prediction range end",
        }
    )
    @server_utils.extract_X_y
    def get(self):
        """
        Process a GET request by fetching data ourselves
        """
        return self._process_request()

    @api.response(200, "Success", API_MODEL_OUTPUT_POST)
    @api.expect(API_MODEL_INPUT_POST, validate=False)
    @api.doc(params={"X": "Nested or single list of sample(s) to predict"})
    @server_utils.extract_X_y
    def post(self):
        """
        Process a POST request by using provided user data
        """
        return self._process_request()

    def _process_request(self):
        """
        Construct a response which fetches model outputs (transformed) as well
        as original input, transformed model input and, if applicable, inverse
        transformed model output.

        Parameters
        ----------
        context: dict
            Current context mapping of the request. Must be dict where
            items are capable of JSON serialization

        Returns
        -------
        flask.Response
        """
        context = dict()  # type: typing.Dict[str, typing.Any]
        context["status-code"] = 200
        context["tags"] = self.tags
        context["target-tags"] = self.target_tags

        data = None
        context: typing.Dict[typing.Any, typing.Any] = dict()
        X = g.X
        process_request_start_time_s = timeit.default_timer()

        try:
            output = model_io.get_model_output(model=current_app.model, X=X)
        except ValueError as err:
            tb = traceback.format_exc()
            logger.error(
                f"Failed to predict or transform; error: {err} - \nTraceback: {tb}"
            )
            context["error"] = f"ValueError: {str(err)}"
            return make_response((jsonify(context), 400))

        # Model may only be a transformer, probably an AttributeError, but catch all to avoid logging other
        # exceptions twice if it happens.
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(
                f"Failed to predict or transform; error: {exc} - \nTraceback: {tb}"
            )
            context["error"] = "Something unexpected happened; check your input data"
            return make_response((jsonify(context), 400))

        else:
            get_model_output_time_s = timeit.default_timer()
            logger.debug(
                f"Calculating model output took "
                f"{get_model_output_time_s-process_request_start_time_s} s"
            )
            data = model_utils.make_base_dataframe(
                tags=self.tags,
                model_input=X.values if isinstance(X, pd.DataFrame) else X,
                model_output=output,
                target_tag_list=self.target_tags,
                index=X.index,
            )
            context["data"] = server_utils.dataframe_to_dict(data)
            return make_response((jsonify(context), context.pop("status-code", 200)))


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
