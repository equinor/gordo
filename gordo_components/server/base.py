# -*- coding: utf-8 -*-

import dateutil
import logging
import timeit

from datetime import datetime

import numpy as np

from flask import url_for, request
from flask_restplus import Resource, Api as BaseApi


logger = logging.getLogger(__name__)


MODEL_LOCATION_ENV_VAR = "MODEL_LOCATION"
MODEL = None
MODEL_METADATA = None


class Api(BaseApi):
    """
    Update the default specs_url to return relative url
    """

    @property
    def specs_url(self):
        return url_for(self.endpoint("specs"), _external=False)


class BasePredictionView(Resource):
    """
    Base Resource which sets module level globals for ``MODEL`` and ``MODEL_METADATA``
    upon first invokation.
    """

    model = None
    metadata = dict()  # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Set the global variables for MODEL and MODEL_METADATA if they haven't been set already.
        if MODEL is None:
            from gordo_components.server.server import load_model_and_metadata

            load_model_and_metadata()

        self.model = MODEL
        self.metadata = MODEL_METADATA

    def post(self):
        """
        Simply take input data and run against the model, responding with the
        output without any post processing
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
        X: np.ndarray
            2d array of sample(s)

        Returns
        -------
        np.ndarray
            The raw output of the model in numpy array form.
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
