# -*- coding: utf-8 -*-

import os
import io
import logging
import timeit
import typing
import traceback
import dateutil.parser  # noqa
from datetime import datetime

import numpy as np
import pandas as pd

from flask import Blueprint, current_app, request, g, send_file, make_response, jsonify
from flask_restplus import Resource, fields

from gordo_components import __version__, serializer
from gordo_components.dataset.datasets import TimeSeriesDataset
from gordo_components.server import model_io
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


class BaseModelView(Resource):
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

    @property
    def target_tags(self) -> typing.List[SensorTag]:
        if "target_tag_list" in current_app.metadata["dataset"]:
            return normalize_sensor_tags(
                current_app.metadata["dataset"]["target_tag_list"]
            )
        else:
            return []

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
        logger.debug("Fetching data from data provider")
        before_data_fetch = timeit.default_timer()
        dataset = TimeSeriesDataset(
            data_provider=g.data_provider,
            from_ts=start - self.frequency.delta,
            to_ts=end,
            resolution=current_app.metadata["dataset"]["resolution"],
            tag_list=self.tags,
        )
        X, _y = dataset.get_data()
        logger.debug(
            f"Fetching data from data provider took "
            f"{timeit.default_timer()-before_data_fetch} seconds"
        )
        # Want resampled buckets equal or greater than start, but less than end
        # b/c if end == 00:00:00 and req = 10 mins, a resampled bucket starting
        # at 00:00:00 would imply it has data until 00:10:00; which is passed
        # the requested end datetime
        X = X[
            (X.index > start - self.frequency.delta)
            & (X.index + self.frequency.delta < end)
        ]
        return self._process_request(context=context, X=X)

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

        if X.shape[1] != len(self.tags):
            message = dict(
                message=f"Expected n features to be {len(self.tags)} but got {X.shape[1]}"
            )
            return make_response((jsonify(message), 400))

        return self._process_request(context=context, X=X)

    def _process_request(
        self, context: dict, X: typing.Union[pd.DataFrame, np.ndarray]
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

        Returns
        -------
        flask.Response
        """
        self.X = X
        data = None
        process_request_start_time_s = timeit.default_timer()

        try:
            output = model_io.get_model_output(model=current_app.model, X=X)
            get_model_output_time_s = timeit.default_timer()
            logger.debug(
                f"Calculating model output took "
                f"{get_model_output_time_s-process_request_start_time_s} s"
            )
            transformed_model_input = model_io.get_transformed_input(
                model=current_app.model, X=X
            )
            get_transformed_input_time_s = timeit.default_timer()
            logger.debug(
                f"Calculating model transformed input took "
                f"{get_transformed_input_time_s- get_model_output_time_s} s "
            )
            if len(output.shape) == len(X.shape) and output.shape[1] == X.shape[1]:
                inverse_transformed_model_output = model_io.get_inverse_transformed_input(
                    model=current_app.model, X=output
                )

                logger.debug(
                    f"Calculating model inverse transformed output took "
                    f"{timeit.default_timer() - get_transformed_input_time_s} s"
                )
                self._output_matches_input_shape = True
            else:
                inverse_transformed_model_output = None
                self._output_matches_input_shape = False

        except ValueError as err:
            tb = traceback.format_exc()
            logger.error(
                f"Failed to predict or transform; error: {err} - \nTraceback: {tb}"
            )
            context["error"] = f"ValueError: {str(err)}"
            context["status-code"] = 400

        # Model may only be a transformer, probably an AttributeError, but catch all to avoid logging other
        # exceptions twice if it happens.
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error(
                f"Failed to predict or transform; error: {exc} - \nTraceback: {tb}"
            )
            context["error"] = "Something unexpected happened; check your input data"
            context["status-code"] = 400
        else:
            data = self.make_base_dataframe(
                tags=self.tags,
                original_input=X.values if isinstance(X, pd.DataFrame) else X,
                model_output=output,
                transformed_model_input=transformed_model_input,
                inverse_transformed_model_output=inverse_transformed_model_output,
                index=X.index if isinstance(X, pd.DataFrame) else None,
            )
            self._data = data  # Assign the base response DF for any children to use

        context["tags"] = self.tags
        context["target-tags"] = self.target_tags

        if data is not None:
            context["data"] = self.multi_lvl_column_dataframe_to_dict(data)
        return make_response((jsonify(context), context.pop("status-code", 200)))

    @staticmethod
    def multi_lvl_column_dataframe_to_dict(df: pd.DataFrame) -> typing.List[dict]:
        """
        Convert a dataframe which has a pandas.MultiIndex as columns into a dict
        where each key is the top level column name, and the value is the array
        of columns under the top level name.
        """

        # Note: It is possible to do this more simply with nested dict comprehension,
        # but ends up making it ~5x slower.

        # This gets the 'top level' names of the multi level column names
        # it will contain names like 'model-output' and then sub column(s)
        # of the actual model output.
        names = df.columns.get_level_values(0).unique()

        # Now a series where each row has an index with the name of the feature
        # which corresponds to 'names' above.
        records = (
            # Stack the dataframe so second level column names become second level indexs
            df.stack()
            # For each column now, unstack the previous second level names (which are now the indexes of the series)
            # back into a dataframe with those names, and convert to list; if it's a Series we'll need to reshape it
            .apply(
                lambda col: col.reindex(df[col.name].columns, level=1)
                .unstack()
                .dropna(axis=1)
                .values.tolist()
                if isinstance(df[col.name], pd.DataFrame)
                else col.unstack()
                .rename(columns={"": col.name})[col.name]
                .values.reshape(-1, 1)
                .tolist()
            )
        )

        results: typing.List[dict] = []

        for i, name in enumerate(names):

            # For each top level name, we'll select its column, unstack so that
            # previous second level names moved into the index will then be column
            # names again, and convert that to a list, matched to the name.
            values = map(lambda row: {name: row}, records[name])

            # If we have results, we'll update the record/row data with this
            # current name. ie {'col1': [1, 2}} -> {'col1': [1, 2], 'col2': [3, 4]}
            # if the current column name is 'col2' and values [3, 4] for the first row,
            # and so on.
            if i == 0:
                results = list(values)
            else:
                [rec.update(d) for rec, d in zip(results, values)]

        return results

    @staticmethod
    def make_base_dataframe(
        tags: typing.Union[typing.List[SensorTag], typing.List[str]],
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
        tags: List[Union[str, SensorTag]]
        original_input: np.ndarray
        model_output: np.ndarray
        transformed_model_input: np.ndarray
        inverse_transformed_model_output: Optional[np.ndarray]
        index: Optional[np.ndarray]

        Returns
        -------
        pd.DataFrame
        """
        start_time_s = timeit.default_timer()
        names_n_values = (
            ("original-input", original_input),
            ("model-output", model_output),
            ("transformed-model-input", transformed_model_input),
            ("inverse-transformed-model-output", inverse_transformed_model_output),
        )

        index = (
            index[-len(model_output) :]
            if index is not None
            else range(len(model_output))
        )

        # Loop over the names and values, less any which are None
        data: pd.DataFrame
        name: str
        values: np.ndarray
        for i, (name, values) in enumerate(
            filter(lambda nv: nv[1] is not None, names_n_values)
        ):

            # Create the second level of column names, either as the tag names
            # or simple range of numbers
            if values.shape[1] == len(tags):
                # map(...) to satisfy mypy to match second possible outcome
                second_lvl_names = map(
                    str,
                    (tag.name if isinstance(tag, SensorTag) else tag for tag in tags),
                )
            else:
                second_lvl_names = map(str, range(values.shape[1]))

            # Columns will be multi level with the title of the output on top
            # and specific names below, ie. ('model-output', 'tag-0') as a column
            columns = pd.MultiIndex.from_tuples(
                (name, sub_name) for sub_name in second_lvl_names
            )

            # Pass valudes, offsetting any differences in length compared to index, as set by model-output size
            other: pd.DataFrame = pd.DataFrame(
                values[-len(model_output) :], columns=columns, index=index
            )

            if not i:
                data = other
            else:
                data = data.join(other)
        logger.debug(
            f"make_base_dataframe took {timeit.default_timer() - start_time_s} s"
        )
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
