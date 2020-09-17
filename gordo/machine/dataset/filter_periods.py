from pprint import pformat
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from gordo.machine.dataset.datasets import pandas_filter_rows
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class WrongFilterMethodType(TypeError):
    pass


class FilterPeriods:
    """Model class with methods for data pre-processing.
    Performs a series of algorithms that drops noisy data.

    Either a rolling median or an isolation forest algorithm is executed.
    Both provide drop periods in a dict-type element on the class object
    `object.drop_periods["iforest"]` and `object.drop_periods["median"]`,
    and data is filtered accordingly.

    Parameters:
    data: pandas.DataFrame
        Data frame containing already filtered data (global max/min + dropped known periods).
        Time consecutively is not required.
    granularity: str
        The bucket size for grouping all incoming time data (e.g. "10T").
        Available strings come from https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
    **kwargs:
        See below.

    Keyword arguments:
    filter_method: str
        Which method should be used for data cleaning, either "median" (default), "iforest" or "all" which returns
        results for both methods.
    iforest_smooth: bool
        If exponential weighted smoothing should be applied to data before isolation forest
        algorithm is run.
    """

    def __init__(
        self,
        granularity: Optional[str] = "10T",
        filter_method: Optional[str] = "median",
        window=144,
        n_iqr=5,
        iforest_smooth=False,
        contamination=0.03,
    ):
        self.granularity = granularity
        self.filter_method = filter_method
        if self.filter_method not in ["median", "iforest", "all"]:
            raise WrongFilterMethodType
        self._window = window
        self._n_iqr = n_iqr
        self._iforest_smooth = iforest_smooth
        self._contamination = contamination

    def filter_data(self, data):
        """Method for filtering data.
        Returns the filtered dataset, a dict containing the different periods that have been dropped arranged
        by filtering method and the actual predictions from the filter model.
        """
        predictions = {}
        if self.filter_method in ["median", "all"]:
            predictions["median"] = self._rolling_median(data)

        if self.filter_method in ["iforest", "all"]:
            self._train(data)
            predictions["iforest"] = self._predict(data)

        drop_periods = self._drop_periods(predictions)
        data = self._filter_data(data, drop_periods)
        return data, drop_periods, predictions

    def _init_model(self, data):
        """Return a new instance of the models."""
        self.isolationforest = IsolationForest(
            n_estimators=300,  # The number of base estimators in the ensemble.
            max_samples=min(
                1000, data.shape[0]
            ),  # Samples to draw from X to train each base estimator
            contamination=self._contamination,  # default = "auto"
            max_features=1.0,  # Features to draw from X to train each base estimator.
            bootstrap=False,
            n_jobs=-1,  # ``-1`` means using all processors
            random_state=42,
            verbose=0,
        )
        self.minmaxscaler = MinMaxScaler()
        logger.info("Initialized isolationforest: \n%s.", self.isolationforest)
        logger.info("Initialized minmaxscaler: \n%s.", self.minmaxscaler)

    def _train(self, data):
        """Train the model.
        Smooth if necessary.
        """
        data = data.copy()
        if self._iforest_smooth:
            data = data.ewm(halflife=6).mean()

        logger.info("Fitting model")
        self._init_model(data)
        self.model = self.isolationforest.fit(data)

        logger.info(
            "Created new isolationforest model:\n%s\nFitted on data of shape '%s'.",
            self.model,
            data.shape,
        )

    def _predict(self, data):
        """Make predictions.
        """
        logger.info("Calculating predictions for isolation forest")
        assert isinstance(data, pd.DataFrame)

        score = -self.model.decision_function(data)
        self.iforest_scores = self._describe(score)
        score = self.minmaxscaler.fit_transform(score.reshape(-1, 1)).squeeze()
        self.iforest_scores_transformed = self._describe(score)

        pred = self.model.predict(data)
        pred = pd.DataFrame({"pred": pred, "score": score, "timestamp": data.index})
        logger.info("Anomaly ratio: %s", list(pred).count(-1) / pred.shape[0])
        return pred

    def _rolling_median(self, data):
        """Function for filtering using a rolling median approach."""
        logger.info("Calculating predictions for rolling median")
        roll = data.rolling(self._window, center=True)
        r_md = roll.median()
        r_iqr = roll.quantile(0.75) - roll.quantile(0.25)
        high = r_md + self._n_iqr * r_iqr
        low = r_md - self._n_iqr * r_iqr
        mask = ((data < low) | (data > high)).any(1).astype("int") * -1
        pred = pd.DataFrame({"pred": mask})
        pred.index.name = "timestamp"
        pred = pred.reset_index()
        logger.info("Anomaly ratio: %s", list(pred).count(-1) / pred.shape[0])
        return pred

    def _drop_periods(self, predictions):
        """Create drop period list.

        Only keep anomaly flagged observations (-1), and create a time-diff between these.
        Time consecutive anomalies will have a time-diff equal to the specified
        granularity of the data, e.g. 10 minutes.
        Minimum two consecutive anomalies must be flagged for a drop period to be initiated.
        """
        logger.info("Creating list of drop period dicts")
        drop_periods = {}

        if self.filter_method == "all":
            pred_types = ["iforest", "median"]

        else:
            pred_types = [self.filter_method]

        for pred_type in pred_types:
            t = predictions[pred_type].query("pred == -1")[["timestamp"]]
            t["delta_t"] = (
                t["timestamp"]
                .diff()
                .fillna(pd.Timedelta(seconds=0))
                .astype("timedelta64[m]")
                .astype("int")
            )
            t = t.reset_index(drop=True)

            # get granularity in minutes
            granularity = pd.Timedelta(self.granularity).total_seconds() / 60

            start = []
            end = []
            for i in range(len(t)):
                # start conditions:
                # [i == 0 (start) OR
                # time delta from previous > granularity (gap)]
                if (i == 0) or (t["delta_t"][i] > granularity):
                    start.append(str(t["timestamp"][i]))
                # stop conditions:
                # [i + 1 == len(t) (end of data) OR
                # time delta to next > granularity (gap)]
                if (i + 1 == len(t)) or (t["delta_t"][i + 1] > granularity):
                    end.append(str(t["timestamp"][i]))
            drop_periods[pred_type] = pd.DataFrame(
                {"drop_start": start, "drop_end": end}
            ).to_dict("records")

        return drop_periods

    def _filter_data(self, data, drop_periods):
        """Drops periods defined previously from dataset."""
        row_filter = []
        for drop_method, p in drop_periods.items():
            for line in p:
                row_filter.append(
                    f"~('{line['drop_start']}' <= index <= '{line['drop_end']}')"
                )

        if row_filter:
            n_prior = len(data)
            data = pandas_filter_rows(df=data, filter_str=row_filter, buffer_size=0)
            logger.info(f"Dropped {n_prior - len(data)} rows")
            return data
        else:
            logger.info("No rows dropped")
            return data

    @staticmethod
    def _describe(score):
        """Format array-like for logging summary statistics."""
        return pformat(pd.Series(score).describe().round(3).to_dict())
