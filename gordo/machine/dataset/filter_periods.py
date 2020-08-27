from pprint import pformat
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from gordo.machine.dataset.datasets import pandas_filter_rows
import logging

logger = logging.getLogger(__name__)


class WrongFilterMethodType(TypeError):
    pass

class filter_periods:
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

    def __init__(self, data, granularity, **kwargs):
        self.data = data.copy()
        self.granularity = granularity
        self.filter_method = kwargs.get("filter_method", "median")

        if self.filter_method not in ["median", "iforest", "all"]:
            raise WrongFilterMethodType

        self.predictions = {}
        if self.filter_method in ["median", "all"]:
            self._window = kwargs.get("window", 144)
            self._n_iqr = kwargs.get("n_iqr", 5)
            self._rolling_median()

        if self.filter_method in ["iforest", "all"]:
            self._iforest_smooth = kwargs.get("iforest_smooth", False)
            self._contamination = kwargs.get("contamination", 0.03)
            self._train()
            self._predict()

        self._drop_periods()
        self._filter_data()

    def _init_model(self):
        """Return a new instance of the models.
        """
        self.isolationforest = IsolationForest(
            n_estimators=300,  # The number of base estimators in the ensemble.
            max_samples=min(
                1000, self.data.shape[0]
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

    def _train(self):
        """Train the model.
        Smooth if necessary.
        """
        data = self.data.copy()
        if self._iforest_smooth:
            data = data.ewm(halflife=6).mean()

        logger.info("Fitting model")
        self._init_model()
        self.model = self.isolationforest.fit(data)

        logger.info(
            "Created new isolationforest model:\n%s\nFitted on data of shape '%s'.",
            self.model,
            self.data.shape,
        )

    def _predict(self):
        """Make predictions.
        """
        logger.info("Calculating predictions for isolation forest")
        assert isinstance(self.data, pd.DataFrame)

        score = -self.model.decision_function(self.data)
        self.iforest_scores = self._describe(score)
        score = self.minmaxscaler.fit_transform(score.reshape(-1, 1)).squeeze()
        self.iforest_scores_transformed = self._describe(score)

        pred = self.model.predict(self.data)
        self.predictions["iforest"] = pd.DataFrame(
            {"pred": pred, "score": score, "timestamp": self.data.index}
        )
        logger.info("Anomaly ratio: %s", list(pred).count(-1) / pred.shape[0])

    def _rolling_median(self):
        logger.info("Calculating predictions for rolling median")
        roll = self.data.rolling(self._window, center=True)
        r_md = roll.median()
        r_iqr = roll.quantile(0.75) - roll.quantile(0.25)
        high = r_md + self._n_iqr * r_iqr
        low = r_md - self._n_iqr * r_iqr
        mask = ((self.data < low) | (self.data > high)).any(1).astype("int") * -1
        pred = pd.DataFrame({"pred": mask})
        pred.index.name = "timestamp"
        pred = pred.reset_index()
        self.predictions["median"] = pred
        logger.info("Anomaly ratio: %s", list(pred).count(-1) / pred.shape[0])

    def _drop_periods(self):
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
            t = self.predictions[pred_type].query("pred == -1")[["timestamp"]]
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

        self.drop_periods = drop_periods

    def _filter_data(self):
        """Drops periods defined previously from dataset """

        row_filter = []
        for drop_method, p in self.drop_periods.items():
            for line in p:
                row_filter.append(
                    f"~('{line['drop_start']}' <= index <= '{line['drop_end']}')"
                )

        if row_filter:
            self.data = pandas_filter_rows(
                df=self.data, filter_str=row_filter, buffer_size=0
            )

    @staticmethod
    def _describe(score):
        """Format array-like for logging summary statistics.
        """
        return pformat(pd.Series(score).describe().round(3).to_dict())
