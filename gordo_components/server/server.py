# -*- coding: utf-8 -*-

import os
import logging
import timeit

import numpy as np

from flask import Flask, request
from flask_restplus import Resource, fields, Api
from gordo_components import __version__, serializer


logger = logging.getLogger(__name__)

api = Api()


MODEL_LOCATION_ENV_VAR = "MODEL_LOCATION"
MODEL = None
MODEL_METADATA = None


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


API_MODEL_INPUT = api.model(
    "Prediction - Multiple Samples", {"X": fields.List(fields.List(fields.Float))}
)
API_MODEL_OUTPUT = api.model(
    "Prediction - Output", {"output": fields.List(fields.List(fields.Float))}
)


class PredictionApiView(Resource):
    """
    Serve model predictions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set the global variables for MODEL and MODEL_METADATA if they haven't been set already.
        if MODEL is None:
            load_model_and_metadata()

        self.model = MODEL
        self.metadata = MODEL_METADATA

    @api.response(200, "Success", API_MODEL_OUTPUT)
    @api.expect(API_MODEL_INPUT, validate=False)
    @api.doc(
        params={
            "X": "Nested list of samples to predict, or single list considered as one sample"
        }
    )
    def post(self):
        """
        Get predictions
        """
        data = request.json

        X = data.get("X")

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

        context = dict()
        context["status-code"] = 200  # Default
        start = timeit.default_timer()

        try:
            context["output"] = self.model.predict(X).tolist()

        # Model may only be a transformer, probably an AttributeError, but catch all to avoid logging other
        # exceptions twice if it happens.
        except Exception as exc:
            try:
                context["output"] = self.model.transform(X).tolist()
            except Exception as exc:
                logger.critical(f"Failed to predict or transform; error: {exc}")
                context[
                    "error"
                ] = "Something unexpected happened; please check your input data"
                context["status-code"] = 400

        context["time-seconds"] = f"{timeit.default_timer() - start:.4f}"
        return context, context["status-code"]


class MetaDataView(Resource):
    """
    Serve model / server metadata
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set the global variables for MODEL and MODEL_METADATA if they haven't been set already.
        if MODEL_METADATA is None:
            load_model_and_metadata()
        self.metadata = MODEL_METADATA

    def get(self):
        """
        Get metadata about this endpoint, also serves as /healthcheck endpoint
        """
        return {
            "version": __version__,
            "model-metadata": self.metadata,
            MODEL_LOCATION_ENV_VAR: os.environ.get(MODEL_LOCATION_ENV_VAR),
        }


def build_app():
    """
    Build app and any associated routes
    """
    app = Flask(__name__)
    api.init_app(app)
    api.add_resource(PredictionApiView, "/predictions")
    api.add_resource(MetaDataView, "/metadata", "/healthcheck")
    return app


def run_server(host: str = "0.0.0.0", port: int = 5555, debug: bool = False):
    app = build_app()
    app.run(host, port, debug=debug)


if __name__ == "__main__":
    run_server()
