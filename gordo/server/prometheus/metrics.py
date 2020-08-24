import timeit

from copy import copy
from flask import Flask, g, request, Request, Response
from typing import Optional, Tuple, List, Dict, Iterable
from prometheus_client.multiprocess import MultiProcessCollector
from prometheus_client.registry import CollectorRegistry
from prometheus_client import Counter, Histogram, Gauge
from http import HTTPStatus


def create_registry():
    registry = CollectorRegistry()
    MultiProcessCollector(registry)
    return registry


def to_status_code(response_status):
    if isinstance(response_status, HTTPStatus):
        return response_status.value
    else:
        return response_status


def url_rule_to_str(url_rule):
    return url_rule.rule


def current_time():
    return timeit.default_timer()


class GordoServerPrometheusMetrics:
    """
    Container for encapsulating all Prometheus related logic
    The simplest way to use with preexisting Flask application:

    >>> from flask import Flask
    >>> from prometheus_client.registry import CollectorRegistry
    >>> app = Flask("test")
    >>> @app.route('/hello')
    ... def hello():
    ...     return 'Hello, World'
    >>> prometheus_metrics = GordoServerPrometheusMetrics(registry=CollectorRegistry())
    >>> prometheus_metrics.prepare_app(app)
    """

    prefix = "gordo_server"
    main_labels = ("method", "path", "status_code")

    @staticmethod
    def main_label_values(req: Request, resp: Response):
        return (
            req.method,
            url_rule_to_str(req.url_rule),
            to_status_code(resp.status_code),
        )

    def __init__(
        self,
        args_labels: Optional[Iterable[Tuple[str, str]]] = None,
        info: Optional[Dict[str, str]] = None,
        ignore_paths: Optional[Iterable[str]] = None,
        registry: Optional[CollectorRegistry] = None,
    ):
        self.args_labels = args_labels if args_labels is not None else []
        if ignore_paths is not None:
            ignore_paths = set(ignore_paths)
        self.ignore_paths = ignore_paths if ignore_paths is not None else {}
        self.info = info
        self.label_names: List[str] = []
        self.label_values: List[str] = []
        self.args_names: List[str] = []

        if registry is None:
            registry = create_registry()
        self.registry = registry
        self.init_labels()
        self.request_duration_seconds = Histogram(
            "%s_request_duration_seconds" % self.prefix,
            "HTTP request duration, in seconds",
            self.label_names,
            registry=registry,
        )
        self.request_count = Counter(
            "%s_requests_total" % self.prefix,
            "Total HTTP requests",
            self.label_names,
            registry=registry,
        )

    def init_labels(self):
        label_names, label_values = [], []
        if self.info is not None:
            for name, value in self.info.items():
                label_names.append(name)
                label_values.append(value)
            gauge_info = Gauge(
                self.prefix + "_info",
                "Gordo information",
                label_names,
                registry=self.registry,
            )
            gauge_info = gauge_info.labels(*label_values)
            gauge_info.set(1)
        args_names = []
        for arg_name, label_name in self.args_labels:
            args_names.append(arg_name)
            label_names.append(label_name)
        self.args_names = args_names
        label_names.extend(self.main_labels)
        self.label_names = label_names
        self.label_values = label_values

    def request_label_values(self, req: Request, resp: Response):
        label_values = copy(self.label_values)
        view_args = req.view_args
        for arg_name in self.args_names:
            value = view_args.get(arg_name, "") if view_args is not None else ""
            label_values.append(value)
        label_values.extend(self.main_label_values(req, resp))
        return label_values

    def prepare_app(self, app: Flask):
        @app.before_request
        def _start_prometheus():
            g.prometheus_metrics = self
            g.prometheus_start_time = current_time()

        @app.after_request
        def _end_prometheus(response: Response) -> Response:
            url_rule = url_rule_to_str(request.url_rule)
            if self.ignore_paths is not None and url_rule in self.ignore_paths:
                return response
            label_values = self.request_label_values(request, response)
            self.request_duration_seconds.labels(*label_values).observe(
                current_time() - g.prometheus_start_time
            )
            self.request_count.labels(*label_values).inc(1)
            del g.prometheus_metrics
            return response
