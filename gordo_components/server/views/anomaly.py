# -*- coding: utf-8 -*-

import logging
import timeit
from datetime import datetime

import numpy as np
import pandas as pd

from flask import Blueprint, make_response, Response, jsonify
from flask_restplus import fields

from gordo_components import __version__
from gordo_components.server.mixins import ModelMixin
from gordo_components.server.rest_api import Api
from gordo_components.server.views.base import BaseModelView


logger = logging.getLogger(__name__)

anomaly_blueprint = Blueprint("ioc_anomaly_blueprint", __name__, url_prefix="/anomaly")

api = Api(
    app=anomaly_blueprint,
    title="Gordo API IOC Anomaly Docs",
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


class AnomalyView(BaseModelView, ModelMixin):
    """
    Serve model predictions via GET and POST methods

    Will take a ``start`` and ``end`` ISO format datetime string if a GET request
    or will take the raw input given in a POST request
    and give back predictions in the following example::

        {'data': [
            {
              'error-transformed-tag-0': 0.0,
              'error-transformed-tag-1': 0.0,
              'error-transformed-tag-2': 0.0,
              'error-transformed-tag-3': 0.0,
              'error-untransformed-tag-0': 0.420411857997552,
              'error-untransformed-tag-1': 0.914222253049058,
              'error-untransformed-tag-2': 0.5068556967346384,
              'error-untransformed-tag-3': 0.7270567536334099,
              'inverse-transformed-model-output-tag-0': 0.0007441184134222567,
              'inverse-transformed-model-output-tag-1': 0.0007441249908879399,
              'inverse-transformed-model-output-tag-2': 0.0007441261550411582,
              'inverse-transformed-model-output-tag-3': 0.0007441251655109227,
              'model-output-tag-0': 0.0007441184134222567,
              'model-output-tag-1': 0.0007441249908879399,
              'model-output-tag-2': 0.0007441261550411582,
              'model-output-tag-3': 0.0007441251655109227,
              'original-input-tag-0': 0.42115597641097424,
              'original-input-tag-1': 0.914966378039946,
              'original-input-tag-2': 0.5075998228896795,
              'original-input-tag-3': 0.7278008787989209,
              'total-transformed-error': nan,
              'total-untransformed-error': nan,
              'transformed-model-input-tag-0': 0.0007441184134222567,
              'transformed-model-input-tag-1': 0.0007441249908879399,
              'transformed-model-input-tag-2': 0.0007441261550411582,
              'transformed-model-input-tag-3': 0.0007441251655109227
            },
            ...
         ],
         'status-code': 200,
         'time-seconds': '2.5015'}
    """

    @api.response(200, "Success", API_MODEL_OUTPUT_POST)
    @api.expect(API_MODEL_INPUT_POST, validate=False)
    @api.doc(
        params={
            "X": "Nested list of samples to predict, or single list considered as one sample"
        }
    )
    def post(self):
        start_time = timeit.default_timer()
        base_response = super().post()
        return self._process_base_response(base_response, start_time)

    @api.response(200, "Success", API_MODEL_OUTPUT_POST)
    @api.doc(
        params={
            "start": "An ISO formatted datetime with timezone info string indicating prediction range start",
            "end": "An ISO formatted datetime with timezone info string indicating prediction range end",
        }
    )
    def get(self):
        start_time = timeit.default_timer()
        base_response: Response = super().get()
        return self._process_base_response(base_response, start_time)

    def _process_base_response(self, response: Response, start_time: float = None):
        """
        Process a base response from POST or GET endpoints, where it is expected in
        the anomaly endpoint that the keys "output", "transformed-model-input" and "inverse-transformed-output"
        are expected to be present in ``.json`` of the Response.

        Parameters
        ----------
        response: Response
            The response from the ``BaseModelView`` which represents the raw model output
        start_time: Optional[float]
            Start time to use when timing the processing time of the request, will construct a new
            one if not provided.

        Returns
        -------
        flask.Response
            The formatted anomaly representation response object.
        """
        if start_time is None:
            start_time = timeit.default_timer()

        # If something went wrong with the basic model view, no point going further
        if not 200 <= response.status_code <= 299:
            return response

        # If client is accessing the anomaly endpoint where the model doesn't give
        # back the transformed model input, we can't do anomaly formatting
        if (
            hasattr(self, "_output_matches_input_shape")
            and not self._output_matches_input_shape
        ):
            message = dict(
                message="Cannot perform anomaly detection where model output shape is different than its input"
            )
            return make_response((jsonify(message), 400))

        X: pd.DataFrame = pd.DataFrame.from_records(response.json["data"])

        # In GET requests we need to pair the resulting predictions with their
        # specific timestamp and additionally match the predictions to the corresponding tags.
        anomaly_df = self.make_anomaly_df(X)

        context = response.json.copy()
        context["data"] = anomaly_df.to_dict(orient="records")
        context["time-seconds"] = f"{timeit.default_timer() - start_time:.4f}"
        return make_response(jsonify(context), context.get("status-code", 200))

    def make_anomaly_df(self, X: pd.DataFrame):
        """
        Create an anomaly dataframe from the base provided dataframe.
        It is expected that the 'inverse-transformed-model-output' columns are
        included for the features.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame created by the base model view.

        Returns
        -------
        pd.DataFrame
            A superset of the original base dataframe with added anomaly specific
            features
        """

        # Set the start and end times, setting them as ISO strings after calculations,
        # to ensure they are JSON encoded successfully and with tz info
        X["start"] = self.X.index if isinstance(self.X, pd.DataFrame) else None
        X["end"] = X["start"].map(
            lambda start: (start + self.frequency).isoformat()
            if isinstance(start, datetime)
            else None
        )
        X["start"] = X["start"].map(
            lambda start: start.isoformat() if hasattr(start, "isoformat") else None
        )

        # Calculate the total anomaly between all tags for the transformed data
        X["total-transformed-error"] = np.linalg.norm(
            X[[f"transformed-model-input-{tag.name}" for tag in self.tags]].values
            - X[[f"model-output-{tag.name}" for tag in self.tags]].values
        )

        # Calculate the total anomaly between all tags for the original/untransformed
        X["total-untransformed-error"] = np.linalg.norm(
            X[[f"original-input-{tag.name}" for tag in self.tags]].values
            - X[
                [f"inverse-transformed-model-output-{tag.name}" for tag in self.tags]
            ].values
        )

        # Calculate anomaly values by tag; both transformed and untransformed data
        for tag in self.tags:
            X[f"error-transformed-{tag.name}"] = (
                X[f"model-output-{tag.name}"] - X[f"transformed-model-input-{tag.name}"]
            ).abs()

            X[f"error-untransformed-{tag.name}"] = (
                X[f"inverse-transformed-model-output-{tag.name}"]
                - X[f"original-input-{tag.name}"]
            ).abs()

        return X


api.add_resource(AnomalyView, "/prediction")
