# -*- coding: utf-8 -*-

import logging
import timeit
from datetime import datetime

import numpy as np
import pandas as pd

from flask import Blueprint, make_response, Response, jsonify
from flask_restplus import fields

from gordo_components import __version__
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


class AnomalyView(BaseModelView):
    """
    Serve model predictions via GET and POST methods

    Will take a ``start`` and ``end`` ISO format datetime string if a GET request
    or will take the raw input given in a POST request
    and give back predictions in the following example::

        {
        'data': [
            {
           'end': ['2016-01-01T00:10:00+00:00'],
           'error-transformed': [0.913027075986948,
                                 0.3474043585419292,
                                 0.8986610906818544,
                                 0.11825221990818557],
           'error-untransformed': [0.913027075986948,
                                   0.3474043585419292,
                                   0.8986610906818544,
                                   0.11825221990818557],
           'inverse-transformed-model-output': [0.0005317790200933814,
                                                -0.0001525811239844188,
                                                0.0008310950361192226,
                                                0.0015755111817270517],
           'model-output': [0.0005317790200933814,
                            -0.0001525811239844188,
                            0.0008310950361192226,
                            0.0015755111817270517],
           'original-input': [0.9135588550070414,
                              0.3472517774179448,
                              0.8994921857179736,
                              0.11982773108991263],
           'start': ['2016-01-01T00:00:00+00:00'],
           'total-transformed-error': [1.3326228173185086],
           'total-untransformed-error': [1.3326228173185086],
           'transformed-model-input': [0.9135588550070414,
                                       0.3472517774179448,
                                       0.8994921857179736,
                                       0.11982773108991263]
            },
            ...
        ],

     'tags': [{'asset': None, 'name': 'tag-0'},
              {'asset': None, 'name': 'tag-1'},
              {'asset': None, 'name': 'tag-2'},
              {'asset': None, 'name': 'tag-3'}],
     'time-seconds': '0.1937'}
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
        if not self._output_matches_input_shape:
            message = {
                "message": "Cannot perform anomaly detection: model output shape != input shape"
            }
            return make_response((jsonify(message), 400))

        # Now create an anomaly dataframe from the base response dataframe
        anomaly_df = self.make_anomaly_df(self._data)

        context = response.json.copy()
        context["data"] = self.multi_lvl_column_dataframe_to_dict(anomaly_df)
        context["time-seconds"] = f"{timeit.default_timer() - start_time:.4f}"
        return make_response(jsonify(context), context.pop("status-code", 200))

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
        X["start"] = (
            self.X.index
            if isinstance(self.X, pd.DataFrame)
            and isinstance(self.X.index, pd.DatetimeIndex)
            else None
        )
        X["end"] = X["start"].map(
            lambda start: (start + self.frequency).isoformat()
            if isinstance(start, datetime)
            else None
        )
        X["start"] = X["start"].map(
            lambda start: start.isoformat() if hasattr(start, "isoformat") else None
        )

        # Calculate the total anomaly between all tags for the original/untransformed
        X["total-transformed-error"] = np.linalg.norm(
            X["transformed-model-input"] - X["model-output"], axis=1
        )
        X["total-untransformed-error"] = np.linalg.norm(
            X["original-input"] - X["inverse-transformed-model-output"], axis=1
        )

        # Calculate anomaly values by tag for transformed values
        error_transformed = (X["model-output"] - X["transformed-model-input"]).abs()
        error_transformed.columns = pd.MultiIndex.from_tuples(
            ("error-transformed", i) for i in error_transformed.columns
        )
        X = X.join(error_transformed)

        # Calculate anomaly values by tag for untransformed values
        error_untransformed = (
            X["inverse-transformed-model-output"] - X["original-input"]
        ).abs()
        error_untransformed.columns = pd.MultiIndex.from_tuples(
            ("error-untransformed", i) for i in error_untransformed.columns
        )
        X = X.join(error_untransformed)

        return X


api.add_resource(AnomalyView, "/prediction")
