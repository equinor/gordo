import timeit

from copy import copy
from flask import Flask, g, request, Request, Response
from typing import Optional, Tuple, List, Dict, Iterable
from prometheus_client import multiprocess
from prometheus_client import generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
from prometheus_client import Counter, Histogram, Gauge
from http import HTTPStatus


def create_registry():
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return registry


def to_status_code(response_status):
    if isinstance(response_status, HTTPStatus):
        return response_status.value
    else:
        return response_status


class GordoServerPrometheusMetrics:
    PREFIX = 'gordo_server'
    MAIN_LABELS = ("method", "path", "status_code")

    @staticmethod
    def main_label_values(req: Request, resp: Response):
        return req.method, req.url_rule, to_status_code(resp.status_code)

    def __init__(self,
                 args_labels: Iterable[Tuple[str, str]],
                 info: Dict[str, str],
                 ignore_paths: Optional[Iterable[str]] = None,
                 registry: Optional[CollectorRegistry] = None):
        self.args_labels = args_labels
        if ignore_paths is not None:
            ignore_paths = set(ignore_paths)
        self.ignore_paths = ignore_paths

        self.info = info
        self.label_names, self.label_values = [], []
        self.args_names = []
        self.init_labels()

        if registry is None:
            registry = create_registry()
        self.registry = registry
        self.request_duration_seconds = Histogram(
            self.PREFIX+"_request_duration_seconds",
            "HTTP request duration, in seconds",
            self.label_names,
            registry=registry
        )
        self.request_count = Counter(
            self.PREFIX+"_requests_total",
            "Total HTTP requests",
            self.label_names,
            registry=registry
        )

    def init_labels(self):
        label_names, label_values = [], []
        if self.info is not None:
            for name, value in self.info.items():
                label_names.append(name)
                label_values.append(value)
            gauge_info = Gauge(
                self.PREFIX+'_info', "Gordo information", label_names, registry=self.registry
            )
            gauge_info = gauge_info.labels(*label_values)
            gauge_info.set(1)
        args_names = []
        for arg_name, label_name in self.args_labels:
            args_names.append(arg_name)
            label_names.append(label_name)
        self.args_names = args_names
        label_names.extend(self.MAIN_LABELS)
        self.label_names = label_names
        self.label_values = label_values

    def request_label_values(self, req: Request, resp: Response):
        label_values = copy(self.label_values)
        args = req.args
        for arg_name in self.args_names:
            value = args.get(arg_name, None)
            if value is not None:
                label_values.append(value)
        label_values.extend(self.main_label_values(req, resp))
        return label_values

    def prepare_app(self, app: Flask):
        @app.before_request
        def _start_prometheus():
            g.prometheus_start_time = timeit.default_timer()

        @app.after_request
        def _end_prometheus(response: Response):
            if request.url_rule in self.ignore_paths:
                return
            label_values = self.request_label_values(request, response)
            self.request_duration_seconds.labels(*label_values).observe(g.prometheus_start_time)
            self.request_count.labels(*label_values).inc(1)
