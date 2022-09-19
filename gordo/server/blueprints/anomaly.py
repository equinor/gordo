# -*- coding: utf-8 -*-

import io
import logging
import timeit
import typing

from flask import Blueprint, make_response, jsonify, g, request, send_file

from gordo.server import utils, properties

logger = logging.getLogger(__name__)

anomaly_blueprint = Blueprint("ioc_anomaly_blueprint", __name__)


DELETED_FROM_RESPONSE_COLUMNS = (
    "smooth-tag-anomaly-scaled",
    "smooth-total-anomaly-scaled",
    "smooth-tag-anomaly-unscaled",
    "smooth-total-anomaly-unscaled",
)


def _create_anomaly_response(start_time: float = None):
    """
    Use the current ``X`` and ``y`` to create an anomaly specific response
    using the trained ML model's ``.anomaly()`` method.

    Parameters
    ----------
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

    # To use this endpoint, we need a 'y' to calculate the errors.
    if g.y is None:
        message = {"message": "Cannot perform anomaly without 'y' to compare against."}
        return make_response((jsonify(message), 400))

    # Now create an anomaly dataframe from the base response dataframe
    try:
        anomaly_df = g.model.anomaly(g.X, g.y, frequency=properties.get_frequency())
    except AttributeError:
        msg = {
            "message": f"Model is not an AnomalyDetector, it is of type: {type(g.model)}"
        }
        return make_response(jsonify(msg), 422)  # 422 Unprocessable Entity

    if request.args.get("all_columns") is None:
        columns_for_delete = []
        for column in anomaly_df:
            if column[0] in DELETED_FROM_RESPONSE_COLUMNS:
                columns_for_delete.append(column)
        anomaly_df = anomaly_df.drop(columns=columns_for_delete)

    if request.args.get("format") == "parquet":
        return send_file(
            io.BytesIO(utils.dataframe_into_parquet_bytes(anomaly_df)),
            mimetype="application/octet-stream",
        )
    else:
        context: typing.Dict[typing.Any, typing.Any] = dict()
        context["data"] = utils.dataframe_to_dict(anomaly_df)
        context["time-seconds"] = f"{timeit.default_timer() - start_time:.4f}"
        return make_response(jsonify(context), context.pop("status-code", 200))


@anomaly_blueprint.route(
    "/gordo/v0/<gordo_project>/<gordo_name>/anomaly/prediction", methods=["POST"]
)
@utils.model_required
@utils.extract_X_y
def post_anomaly_prediction():
    """
    Serve model predictions via POST method.

    Gives back predictions looking something like this
    (depending on anomaly model being served)::

        {
        'data': [
            {
           'end': ['2016-01-01T00:10:00+00:00'],
           'tag-anomaly-scaled': [0.913027075986948,
                                  0.3474043585419292,
                                  0.8986610906818544,
                                  0.11825221990818557],
           'tag-anomaly-unscaled': [10.2335327305725986948,
                                  4.234343958392+3293,
                                  10.379394390232232,
                                  3.32093438982743929],
           'model-output': [0.0005317790200933814,
                            -0.0001525811239844188,
                            0.0008310950361192226,
                            0.0015755111817270517],
           'original-input': [0.9135588550070414,
                              0.3472517774179448,
                              0.8994921857179736,
                              0.11982773108991263],
           'start': ['2016-01-01T00:00:00+00:00'],
           'total-anomaly-unscaled': [1.3326228173185086],
           'total-anomaly-scaled': [0.3020328328002392],
            },
            ...
        ],

     'tags': [{'asset': None, 'name': 'tag-0'},
              {'asset': None, 'name': 'tag-1'},
              {'asset': None, 'name': 'tag-2'},
              {'asset': None, 'name': 'tag-3'}],
     'time-seconds': '0.1937'}
    """
    start_time = timeit.default_timer()
    return _create_anomaly_response(start_time)
