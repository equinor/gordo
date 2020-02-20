# -*- coding: utf-8 -*-

import os
import io
import logging
import traceback
import timeit
import typing

import pandas as pd
from flask import Blueprint, current_app, g, send_file, make_response, jsonify, request
from flask_restplus import Resource, fields

from gordo import __version__, serializer
from gordo.server.rest_api import Api
from gordo.server import utils as server_utils
from gordo.machine.model import utils as model_utils
from gordo.machine.dataset.sensor_tag import SensorTag, normalize_sensor_tags
from gordo.server import model_io


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
_tags = {
    fields.String: fields.Float
}  # tags of single prediction record {'tag-name': tag-value}
_single_prediction_record = {
    "start": fields.DateTime,
    "end": fields.DateTime,
    "tags": fields.Nested(_tags),
    "total_abnormality": fields.Float,
}


class BaseModelView(Resource):
    """
    The base model view.
    """

    methods = ["POST"]

    y: pd.DataFrame = None
    X: pd.DataFrame = None

    @property
    def frequency(self):
        """
        The frequency the model was trained with in the dataset
        """
        return pd.tseries.frequencies.to_offset(g.metadata["dataset"]["resolution"])

    @property
    def tags(self) -> typing.List[SensorTag]:
        """
        The input tags for this model

        Returns
        -------
        typing.List[SensorTag]
        """
        return normalize_sensor_tags(
            g.metadata["dataset"]["tag_list"],
            asset=g.metadata["dataset"].get("asset"),
            default_asset=g.metadata["dataset"].get("default_asset"),
        )

    @property
    def target_tags(self) -> typing.List[SensorTag]:
        """
        The target tags for this model

        Returns
        -------
        typing.List[SensorTag]
        """
        if "target_tag_list" in g.metadata["dataset"]:
            return normalize_sensor_tags(
                g.metadata["dataset"]["target_tag_list"],
                asset=g.metadata["dataset"].get("asset"),
                default_asset=g.metadata["dataset"].get("default_asset"),
            )
        else:
            return []

    @api.response(200, "Success", API_MODEL_OUTPUT_POST)
    @api.expect(API_MODEL_INPUT_POST, validate=False)
    @api.doc(params={"X": "Nested or single list of sample(s) to predict"})
    @server_utils.model_required
    @server_utils.extract_X_y
    def post(self):
        """
        Process a POST request by using provided user data

        A typical response might look like this

        .. code-block:: python

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

                'tags': [
                    {'asset': None, 'name': 'tag-0'},
                    {'asset': None, 'name': 'tag-1'},
                    {'asset': None, 'name': 'tag-2'},
                    {'asset': None, 'name': 'tag-3'}
                ],
                'time-seconds': '0.1937'
            }
        """
        context: typing.Dict[typing.Any, typing.Any] = dict()
        X = g.X
        process_request_start_time_s = timeit.default_timer()

        try:
            output = model_io.get_model_output(model=g.model, X=X)
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
            if request.args.get("format") == "parquet":
                return send_file(
                    io.BytesIO(server_utils.dataframe_into_parquet_bytes(data)),
                    mimetype="application/octet-stream",
                )
            else:
                context["data"] = server_utils.dataframe_to_dict(data)
                return make_response(
                    (jsonify(context), context.pop("status-code", 200))
                )


class MetaDataView(Resource):
    """
    Serve model / server metadata
    """

    @server_utils.metadata_required
    def get(self):
        """
        Get metadata about this endpoint, also serves as /healthcheck endpoint
        """
        model_collection_env_var = current_app.config["MODEL_COLLECTION_DIR_ENV_VAR"]
        return {
            "gordo-server-version": __version__,
            "metadata": g.metadata,
            "env": {model_collection_env_var: os.environ.get(model_collection_env_var)},
        }


class DownloadModel(Resource):
    """
    Download the trained model

    suitable for reloading via :func:`gordo.serializer.serializer.loads`
    """

    @api.doc(description="Download model, loadable via gordo.serializer.loads")
    @server_utils.model_required
    def get(self):
        """
        Responds with a serialized copy of the current model being served.

        Returns
        -------
        bytes
            Results from ``gordo.serializer.dumps()``
        """
        serialized_model = serializer.dumps(g.model)
        buff = io.BytesIO(serialized_model)
        return send_file(buff, attachment_filename="model.tar.gz")


class ModelListView(Resource):
    """
    List the current models capable of being served by the server
    """

    @api.doc(description="List the name of the models capable of being served.")
    def get(self, gordo_project: str):
        try:
            available_models = os.listdir(g.collection_dir)
        except FileNotFoundError:
            available_models = []
        finally:
            return jsonify({"models": available_models})


class RevisionListView(Resource):
    """
    List the available revisions the model can serve.
    """

    @api.doc(description="Available revisions of the project that can be served.")
    def get(self, gordo_project: str):
        try:
            available_revisions = os.listdir(os.path.join(g.collection_dir, ".."))
        except FileNotFoundError:
            logger.error(
                f"Attempted to list directories above {g.collection_dir} but failed with: {traceback.format_exc()}"
            )
            available_revisions = [g.current_revision]
        return jsonify(
            {"latest": g.current_revision, "available-revisions": available_revisions}
        )


class ExpectedModels(Resource):
    @api.doc(description="Models that the server expects to be able to serve.")
    def get(self, gordo_project: str):
        return jsonify({"expected-models": current_app.config["EXPECTED_MODELS"]})


api.add_resource(ModelListView, "/gordo/v0/<gordo_project>/models")
api.add_resource(ExpectedModels, "/gordo/v0/<gordo_project>/expected-models")
api.add_resource(BaseModelView, "/gordo/v0/<gordo_project>/<gordo_name>/prediction")
api.add_resource(
    MetaDataView,
    "/gordo/v0/<gordo_project>/<gordo_name>/metadata",
    "/gordo/v0/<gordo_project>/<gordo_name>/healthcheck",
)
api.add_resource(DownloadModel, "/gordo/v0/<gordo_project>/<gordo_name>/download-model")
api.add_resource(RevisionListView, "/gordo/v0/<gordo_project>/revisions")
