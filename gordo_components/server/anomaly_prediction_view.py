# -*- coding: utf-8 -*-

import logging
import timeit
import typing

import numpy as np
import pandas as pd

from flask import g, request
from flask.blueprints import Blueprint
from flask_restplus import fields

from gordo_components import __version__
from gordo_components.dataset.datasets import TimeSeriesDataset
from gordo_components.server.base import BasePredictionView, Api


logger = logging.getLogger(__name__)


anomaly_blueprint = Blueprint(
    "ioc_anomaly_prediction_view", __name__, url_prefix="/ioc-anomaly"
)

api = Api(
    app=anomaly_blueprint,
    title="Gordo Anomaly API Docs",
    version=__version__,
    description="Documentation for the Gordo ML Anomaly endpoint(s)",
    default_label="Gordo Anomaly Endpoints",
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


class IocAnomalyPredictionView(BasePredictionView):
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

    @api.response(200, "Success", API_MODEL_OUTPUT_POST)
    @api.expect(API_MODEL_INPUT_POST, validate=False)
    @api.doc(
        params={
            "X": "Nested list of samples to predict, or single list considered as one sample"
        }
    )
    def post(self):
        """
        Get predictions, without any post processing steps.
        """
        return super().post()

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


anomaly_blueprint.add_url_rule(
    "/prediction", view_func=IocAnomalyPredictionView.as_view("ioc_anomaly_prediction")
)
