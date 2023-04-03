# -*- coding: utf-8 -*-

import os
import io
import logging
import traceback
import timeit

import pandas as pd
from flask import Blueprint, current_app, g, send_file, make_response, jsonify, request

from gordo import __version__, serializer
from gordo.server import utils as server_utils
from gordo.machine.model import utils as model_utils
from gordo.server.utils import (
    validate_gordo_name,
    validate_revision,
    delete_revision,
)
from gordo.server.properties import get_tags, get_target_tags
from gordo.server import model_io
from typing import Any


logger = logging.getLogger(__name__)

base_blueprint = Blueprint("base_model_view", __name__)


@base_blueprint.route(
    "/gordo/v0/<gordo_project>/<gordo_name>/prediction", methods=["POST"]
)
@server_utils.model_required
@server_utils.extract_X_y
def post_prediction():
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
    context: dict[Any, Any] = dict()
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
            tags=get_tags(),
            model_input=X.values if isinstance(X, pd.DataFrame) else X,
            model_output=output,
            target_tag_list=get_target_tags(),
            index=X.index,
        )
        if request.args.get("format") == "parquet":
            return send_file(
                io.BytesIO(server_utils.dataframe_into_parquet_bytes(data)),
                mimetype="application/octet-stream",
            )
        else:
            context["data"] = server_utils.dataframe_to_dict(data)
            return make_response((jsonify(context), context.pop("status-code", 200)))


@base_blueprint.route(
    "/gordo/v0/<gordo_project>/<gordo_name>/revision/<revision>", methods=["DELETE"]
)
def delete_model_revision(gordo_name: str, revision: str, **kwargs):
    """
    Delete provided model revision from the disk.
    """
    validate_gordo_name(gordo_name)
    if not validate_revision(revision):
        return make_response(
            (jsonify({"error": "Revision should only contains numbers."}), 422)
        )
    if revision == g.current_revision:
        return make_response(
            (jsonify({"error": "Unable to delete current revision."}), 409)
        )
    revision_dir = os.path.join(g.collection_dir, "..", revision)
    delete_revision(revision_dir, gordo_name)
    return make_response(jsonify({"ok": True}), 200)


@base_blueprint.route(
    "/gordo/v0/<gordo_project>/<gordo_name>/metadata", methods=["GET"]
)
@base_blueprint.route(
    "/gordo/v0/<gordo_project>/<gordo_name>/healthcheck", methods=["GET"]
)
@server_utils.metadata_required
def get_metadata():
    """
    Serve model / server metadata

    Get metadata about this endpoint, also serves as /healthcheck endpoint
    """
    model_collection_env_var = current_app.config["MODEL_COLLECTION_DIR_ENV_VAR"]
    metadata = {}
    if g.info:
        metadata = g.info
    metadata.update(
        {
            "gordo-server-version": __version__,
            "metadata": g.metadata,
            "env": {model_collection_env_var: os.environ.get(model_collection_env_var)},
        }
    )
    return metadata


@base_blueprint.route(
    "/gordo/v0/<gordo_project>/<gordo_name>/download-model", methods=["GET"]
)
@server_utils.model_required
def get_download_model():
    """
    Download the trained model

    Responds with a serialized copy of the current model being served.

    Returns
    -------
    bytes
        Results from ``gordo.serializer.dumps()``
    """
    serialized_model = serializer.dumps(g.model)
    buff = io.BytesIO(serialized_model)
    return send_file(buff, download_name="model.pickle")


@base_blueprint.route("/gordo/v0/<gordo_project>/models", methods=["GET"])
def get_model_list(gordo_project: str):
    """
    List the current models capable of being served by the server
    """
    available_models = []
    try:
        available_models = os.listdir(g.collection_dir)
    except FileNotFoundError:
        available_models = []
    finally:
        return jsonify({"models": available_models})


@base_blueprint.route("/gordo/v0/<gordo_project>/revisions", methods=["GET"])
def get_revision_list(gordo_project: str):
    """
    List the available revisions the model can serve.
    """
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


@base_blueprint.route("/gordo/v0/<gordo_project>/expected-models", methods=["GET"])
def get(gordo_project: str):
    return jsonify({"expected-models": current_app.config["EXPECTED_MODELS"]})
