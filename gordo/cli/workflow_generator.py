import logging
import time
import pkg_resources
import os

from typing import Dict, Any, List, Tuple, Optional, cast

import click
import json
from jinja2 import Environment, BaseLoader

from gordo import __version__
from gordo.workflow.config_elements.normalized_config import NormalizedConfig
from gordo.workflow.workflow_generator import workflow_generator as wg
from gordo.workflow.config_elements.schemas import (
    SecurityContext,
    PodSecurityContext,
    EnvVar,
)
from gordo.cli.exceptions_reporter import ReportLevel
from gordo.util.version import parse_version
from gordo.workflow.workflow_generator.helpers import (
    parse_argo_version,
    determine_argo_version,
    ArgoVersionError,
)
from gordo_core.back_compatibles import DEFAULT_BACK_COMPATIBLES
from .custom_types import JSONParam, REParam


logger = logging.getLogger(__name__)


PREFIX = "WORKFLOW_GENERATOR"
DEFAULT_BUILDER_EXCEPTIONS_REPORT_LEVEL = ReportLevel.TRACEBACK

ML_SERVER_HPA_TYPES = ["none", "k8s_cpu", "keda"]
DEFAULT_ML_SERVER_HPA_TYPE = "k8s_cpu"

DEFAULT_KEDA_PROMETHEUS_METRIC_NAME = "gordo_server_request_duration_seconds_count"
DEFAULT_KEDA_PROMETHEUS_QUERY = 'sum(rate(gordo_server_request_duration_seconds_count{project=~"{{project_name}}",path=~".*prediction"}[30s]))'
DEFAULT_KEDA_PROMETHEUS_THRESHOLD = "1.0"


def get_builder_exceptions_report_level(config: NormalizedConfig) -> ReportLevel:
    orig_report_level = None
    try:
        orig_report_level = config.globals["runtime"]["builder"][
            "exceptions_report_level"
        ]
    except KeyError:
        pass
    if orig_report_level is not None:
        report_level = ReportLevel.get_by_name(orig_report_level)
        if report_level is None:
            raise ValueError(
                "Invalid 'runtime.builder.exceptions_report_level' value '%s'"
                % orig_report_level
            )
    else:
        report_level = DEFAULT_BUILDER_EXCEPTIONS_REPORT_LEVEL
    return report_level


DEFAULT_CUSTOM_MODEL_BUILDER_ENVS = "[]"


def validate_generate_context(context):
    if context["ml_server_hpa_type"] == "keda":
        if not context["with_keda"]:
            raise click.ClickException(
                '"--ml-server-hpa-type=keda" is only be supported with "--with-keda" flag'
            )
        if not context["prometheus_server_address"]:
            raise click.ClickException(
                '--prometheus-server-address should be specified for "--ml-server-hpa-type=keda"'
            )


KEDA_PROMETHEUS_QUERY_ARGS = ["project_name"]


def prepare_keda_prometheus_query(context):
    keda_prometheus_query = context["keda_prometheus_query"]
    if keda_prometheus_query:
        template = Environment(loader=BaseLoader).from_string(keda_prometheus_query)
        kwargs = {k: context[k] for k in KEDA_PROMETHEUS_QUERY_ARGS}
        return template.render(**kwargs)


def prepare_resources_labels(value: str) -> List[Tuple[str, Any]]:
    resources_labels: List[Tuple[str, Any]] = []
    if value:
        try:
            json_value = json.loads(value)
        except json.JSONDecodeError as e:
            raise click.ClickException(
                '"--resources-labels=%s" contains invalid JSON value: %s'
                % (value, str(e))
            )
        if isinstance(json_value, dict):
            resources_labels = cast(List[Tuple[str, Any]], list(json_value.items()))
        else:
            type_name = type(json_value).__name__
            raise click.ClickException(
                '"--resources-labels=%s" contains value with type "%s" instead "dict"'
                % (value, type_name)
            )
    return resources_labels


def prepare_argo_version(argo_binary: Optional[str] = None) -> str:
    if argo_binary is not None:
        raw_argo_version = determine_argo_version(argo_binary)
    else:
        raw_argo_version = determine_argo_version()
    argo_version = parse_argo_version(raw_argo_version)
    if argo_version is None:
        raise ArgoVersionError(
            "Unable to parse %s version: '%s'" % (argo_binary, argo_version)
        )
    return cast(str, argo_version)


@click.group("workflow")
@click.pass_context
def workflow_cli(gordo_ctx):
    pass


@click.command("generate")
@click.option(
    "--machine-config",
    type=str,
    help="Machine configuration file",
    envvar=f"{PREFIX}_MACHINE_CONFIG",
    required=True,
)
@click.option("--workflow-template", type=str, help="Template to expand")
@click.option(
    "--owner-references",
    type=wg._valid_owner_ref,
    default=None,
    allow_from_autoenv=True,
    help="Kubernetes owner references to inject into all created resources. "
    "Should be a nonempty yaml/json list of owner-references, each owner-reference "
    "a dict containing at least the keys 'uid', 'name', 'kind', and 'apiVersion'",
    envvar=f"{PREFIX}_OWNER_REFERENCES",
)
@click.option(
    "--gordo-version",
    type=str,
    default=wg._docker_friendly_version(__version__),
    help="Version of gordo to use, if different than this one",
    envvar=f"{PREFIX}_GORDO_VERSION",
)
@click.option(
    "--project-name",
    type=str,
    help="Name of the project which own the workflow.",
    allow_from_autoenv=True,
    envvar=f"{PREFIX}_PROJECT_NAME",
    required=True,
)
@click.option(
    "--project-revision",
    type=str,
    default=int(time.time() * 1000),  # unix time milliseconds
    help="Revision of the project which own the workflow.",
    envvar=f"{PREFIX}_PROJECT_REVISION",
)
@click.option(
    "--output-file",
    type=str,
    required=False,
    help="Optional file to render to",
    envvar=f"{PREFIX}_OUTPUT_FILE",
)
@click.option(
    "--namespace",
    type=str,
    default="kubeflow",
    help="Which namespace to deploy services into",
    envvar=f"{PREFIX}_NAMESPACE",
)
@click.option(
    "--split-workflows",
    type=int,
    default=30,
    help="Split workflows containg more than this number of models into several "
    "workflows, where each workflow contains at most this nr of models. The "
    "workflows are outputted sequentially with '---' in between, which allows "
    "kubectl to apply them all at once.",
    envvar=f"{PREFIX}_SPLIT_WORKFLOWS",
)
@click.option(
    "--n-servers",
    type=int,
    default=None,
    help="Max number of ML Servers to use, defaults to N machines * 10",
    envvar=f"{PREFIX}_N_SERVERS",
)
@click.option(
    "--docker-repository",
    type=str,
    default="equinor",
    help="The docker repo to use for pulling component images from",
    envvar=f"{PREFIX}_DOCKER_REPOSITORY",
)
@click.option(
    "--docker-registry",
    type=str,
    default="auroradevacr.azurecr.io",  # TODO: Change to docker.io after migrating
    help="The docker registry to use for pulling component images from",
    envvar=f"{PREFIX}_DOCKER_REGISTRY",
)
@click.option(
    "--retry-backoff-duration",
    type=str,
    default="15s",
    help="retryStrategy.backoff.duration for workflow steps",
    envvar=f"{PREFIX}_RETRY_BACKOFF_DURATION",
)
@click.option(
    "--retry-backoff-factor",
    type=int,
    default=2,
    help="retryStrategy.backoff.factor for workflow steps",
    envvar=f"{PREFIX}_RETRY_BACKOFF_FACTOR",
)
@click.option(
    "--gordo-server-workers",
    type=int,
    help="The number of worker processes for handling Gordo server requests.",
    envvar=f"{PREFIX}_GORDO_SERVER_WORKERS",
)
@click.option(
    "--gordo-server-threads",
    type=int,
    help="The number of worker threads for handling requests.",
    envvar=f"{PREFIX}_GORDO_SERVER_THREADS",
)
@click.option(
    "--gordo-server-probe-timeout",
    type=int,
    help="timeoutSeconds value for livenessProbe and readinessProbe of Gordo server Deployment",
    envvar=f"{PREFIX}_GORDO_SERVER_PROBE_TIMEOUT",
)
@click.option(
    "--without-prometheus",
    is_flag=True,
    help="Do not deploy Prometheus for Gordo servers monitoring",
    envvar=f"{PREFIX}_WITHOUT_PROMETHEUS",
)
@click.option(
    "--prometheus-metrics-server-workers",
    help="Number of workers for Prometheus metrics servers",
    envvar=f"{PREFIX}_PROMETHEUS_METRICS_SERVER_WORKERS",
    default=1,
)
@click.option(
    "--image-pull-policy",
    help="Default imagePullPolicy for all gordo's images",
    envvar=f"{PREFIX}_IMAGE_PULL_POLICY",
)
@click.option(
    "--with-keda",
    is_flag=True,
    help="Enable support for the KEDA autoscaler",
    envvar=f"{PREFIX}_WITH_KEDA",
)
@click.option(
    "--ml-server-hpa-type",
    help="HPA type for the ML server",
    envvar=f"{PREFIX}_ML_SERVER_HPA_TYPE",
    type=click.Choice(ML_SERVER_HPA_TYPES),
    default=DEFAULT_ML_SERVER_HPA_TYPE,
)
@click.option(
    "--custom-model-builder-envs",
    help="List of custom environment variables in ",
    envvar=f"{PREFIX}_CUSTOM_MODEL_BUILDER_ENVS",
    default=DEFAULT_CUSTOM_MODEL_BUILDER_ENVS,
    type=JSONParam(List[EnvVar]),
)
@click.option(
    "--prometheus-server-address",
    help='Prometheus url. Required for "--ml-server-hpa-type=keda"',
    envvar=f"{PREFIX}_PROMETHEUS_SERVER_ADDRESS",
)
@click.option(
    "--keda-prometheus-metric-name",
    help="metricName value for the KEDA prometheus scaler",
    envvar=f"{PREFIX}_KEDA_PROMETHEUS_METRIC_NAME",
    default=DEFAULT_KEDA_PROMETHEUS_METRIC_NAME,
)
@click.option(
    "--keda-prometheus-query",
    help="query value for the KEDA prometheus scaler",
    envvar=f"{PREFIX}_KEDA_PROMETHEUS_QUERY",
    default=DEFAULT_KEDA_PROMETHEUS_QUERY,
)
@click.option(
    "--keda-prometheus-threshold",
    help="threshold value for the KEDA prometheus scaler",
    envvar=f"{PREFIX}_KEDA_PROMETHEUS_THRESHOLD",
    default=DEFAULT_KEDA_PROMETHEUS_THRESHOLD,
)
@click.option(
    "--resources-labels",
    help="Additional labels for resources. Have to be empty string or a dictionary in JSON format",
    envvar=f"{PREFIX}_RESOURCE_LABELS",
    default="",
)
@click.option(
    "--server-termination-grace-period",
    help="terminationGracePeriodSeconds for the gordo server",
    envvar=f"{PREFIX}_SERVER_TERMINATION_GRACE_PERIOD",
    type=int,
    default=60,
)
@click.option(
    "--server-target-cpu-utilization-percentage",
    help="targetCPUUtilizationPercentage for gordo-server's HPA",
    envvar=f"{PREFIX}_SERVER_TARGET_CPU_UTILIZATION_PERCENTAGE",
    type=int,
    default=50,
)
@click.option(
    "--gordo-server-readiness-initial-delay",
    help="initialDelaySeconds for gordo-server's readinessProbe",
    envvar=f"{PREFIX}_GORDO_SERVER_READINESS_INITIAL_DELAY",
    type=int,
    default=5,
)
@click.option(
    "--gordo-server-liveness-initial-delay",
    help="initialDelaySeconds for gordo-server's livenessProbe",
    envvar=f"{PREFIX}_GORDO_SERVER_LIVENESS_INITIAL_DELAY",
    type=int,
    default=600,
)
@click.option(
    "--security-context",
    help="Containers securityContext in JSON format",
    envvar=f"{PREFIX}_SECURITY_CONTEXT",
    type=JSONParam(SecurityContext),
)
@click.option(
    "--pod-security-context",
    help="Global Workflow securityContext in JSON format",
    envvar=f"{PREFIX}_POD_SECURITY_CONTEXT",
    type=JSONParam(PodSecurityContext),
)
@click.option(
    "--default-data-provider",
    help="Default data_provider.type for dataset",
    envvar=f"{PREFIX}_DEFAULT_DATA_PROVIDER",
)
@click.option(
    "--model-builder-class",
    help="ModelBuilder class",
    envvar="MODEL_BUILDER_CLASS",
)
@click.option(
    "--argo-binary",
    default="argo",
    type=REParam(r"^argo\d*$"),
    help="Argo binary path",
)
@click.pass_context
def workflow_generator_cli(gordo_ctx, **ctx):
    """
    Machine Configuration to Argo Workflow
    """

    context: Dict[Any, Any] = ctx.copy()
    yaml_content = wg.get_dict_from_yaml(context["machine_config"])

    model_builder_env = None
    if context["custom_model_builder_envs"]:
        custom_model_builder_envs = cast(
            List[EnvVar], context["custom_model_builder_envs"]
        )
        model_builder_env = [
            env_var.dict(exclude_none=True) for env_var in custom_model_builder_envs
        ]
    # Create normalized config
    config = NormalizedConfig(
        yaml_content,
        project_name=context["project_name"],
        model_builder_env=model_builder_env,
        back_compatibles=DEFAULT_BACK_COMPATIBLES,
        default_data_provider=context["default_data_provider"],
        json_path="spec.config",
    )

    try:
        log_level = config.globals["runtime"]["log_level"]
    except KeyError:
        log_level = os.getenv("GORDO_LOG_LEVEL", gordo_ctx.obj["log_level"])

    logging.getLogger("gordo").setLevel(log_level.upper())
    context["log_level"] = log_level.upper()

    validate_generate_context(context)

    context["argo_version"] = prepare_argo_version(context.get("argo_binary"))

    context["resources_labels"] = prepare_resources_labels(context["resources_labels"])

    if context["pod_security_context"]:
        pod_security_context = cast(PodSecurityContext, context["pod_security_context"])
        context["pod_security_context"] = pod_security_context.dict(exclude_none=True)

    if context["security_context"]:
        security_context = cast(SecurityContext, context["security_context"])
        context["security_context"] = security_context.dict(exclude_none=True)

    version = parse_version(context["gordo_version"])
    if "image_pull_policy" not in context or not context["image_pull_policy"]:
        context["image_pull_policy"] = wg.default_image_pull_policy(version)
    logger.info(
        "Generate config with gordo_version=%s and imagePullPolicy=%s",
        context["gordo_version"],
        context["image_pull_policy"],
    )

    context["max_server_replicas"] = (
        context.pop("n_servers") or len(config.machines) * 10
    )

    context["volumes"] = None
    if "volumes" in config.globals["runtime"]:
        context["volumes"] = config.globals["runtime"]["volumes"]

    builder_runtime = config.globals["runtime"]["builder"]
    # We know these exist since we set them in the default globals
    builder_resources = builder_runtime["resources"]
    context["model_builder_resources_requests_memory"] = builder_resources["requests"][
        "memory"
    ]
    context["model_builder_resources_requests_cpu"] = builder_resources["requests"][
        "cpu"
    ]
    context["model_builder_resources_limits_memory"] = builder_resources["limits"][
        "memory"
    ]
    context["model_builder_resources_limits_cpu"] = builder_resources["limits"]["cpu"]
    context["model_builder_image"] = config.globals["runtime"]["builder"]["image"]

    context["builder_runtime"] = builder_runtime

    builder_runtime_env = []
    if "env" in builder_runtime:
        builder_runtime_env = builder_runtime["env"]

    if builder_runtime_env:
        if context["model_builder_class"]:
            builder_runtime_env.append(
                {"name": "MODEL_BUILDER_CLASS", "value": context["model_builder_class"]}
            )

    context["builder_runtime_env"] = builder_runtime_env

    context["server_resources"] = config.globals["runtime"]["server"]["resources"]
    context["server_image"] = config.globals["runtime"]["server"]["image"]

    context["prometheus_metrics_server_resources"] = config.globals["runtime"][
        "prometheus_metrics_server"
    ]["resources"]

    context["prometheus_metrics_server_image"] = config.globals["runtime"][
        "prometheus_metrics_server"
    ]["image"]

    context["deployer_image"] = config.globals["runtime"]["deployer"]["image"]

    # These are also set in the default globals, and guaranteed to exist
    client_resources = config.globals["runtime"]["client"]["resources"]
    context["client_resources_requests_memory"] = client_resources["requests"]["memory"]
    context["client_resources_requests_cpu"] = client_resources["requests"]["cpu"]
    context["client_resources_limits_memory"] = client_resources["limits"]["memory"]
    context["client_resources_limits_cpu"] = client_resources["limits"]["cpu"]

    context["client_image"] = config.globals["runtime"]["client"]["image"]

    context["client_max_instances"] = config.globals["runtime"]["client"][
        "max_instances"
    ]

    influx_resources = config.globals["runtime"]["influx"]["resources"]
    context["influx_resources_requests_memory"] = influx_resources["requests"]["memory"]
    context["influx_resources_requests_cpu"] = influx_resources["requests"]["cpu"]
    context["influx_resources_limits_memory"] = influx_resources["limits"]["memory"]
    context["influx_resources_limits_cpu"] = influx_resources["limits"]["cpu"]

    nr_of_models_with_clients = len(
        [
            machine
            for machine in config.machines
            if machine.runtime.get("influx", {}).get("enable", True)
        ]
    )
    context["client_total_instances"] = nr_of_models_with_clients

    # Should we start up influx/grafana at all, i.e. is there at least one request
    # for it?"
    enable_influx = nr_of_models_with_clients > 0
    context["enable_influx"] = enable_influx

    context["postgres_host"] = f"gordo-postgres-{config.project_name}"

    context["keda_prometheus_query"] = prepare_keda_prometheus_query(context)

    # If enabling influx, we setup a postgres reporter to send metadata
    # to allowing querying about the machine from grafana
    if enable_influx:
        pg_reporter = {
            "gordo.reporters.postgres.PostgresReporter": {
                "host": context["postgres_host"]
            }
        }
        for machine in config.machines:
            machine.runtime["reporters"].append(pg_reporter)

    # Determine if MlFlowReporter should be enabled per machine
    for machine in config.machines:
        try:
            enabled = machine.runtime["builder"]["remote_logging"]["enable"]
        except KeyError:
            continue
        else:
            if enabled:
                machine.runtime["reporters"].append(
                    "gordo.reporters.mlflow.MlFlowReporter"
                )

    context["machines"] = config.machines

    # Context requiring pre-processing
    context["target_names"] = [machine.name for machine in config.machines]

    # Json dump owner_references, if not None, otherwise pop it out of the context
    if context["owner_references"]:
        context["owner_references"] = json.dumps(context["owner_references"])
    else:
        context.pop("owner_references")

    builder_exceptions_report_level = get_builder_exceptions_report_level(config)
    context["builder_exceptions_report_level"] = builder_exceptions_report_level.name
    if builder_exceptions_report_level != ReportLevel.EXIT_CODE:
        context["builder_exceptions_report_file"] = "/tmp/exception.json"

    if context["workflow_template"]:
        template = wg.load_workflow_template(context["workflow_template"])
    else:
        workflow_template = pkg_resources.resource_filename(
            "gordo.workflow.workflow_generator.resources", "argo-workflow.yml.template"
        )
        template = wg.load_workflow_template(workflow_template)

    # Clear output file
    if context["output_file"]:
        open(context["output_file"], "w").close()  # type: ignore
    project_workflow = 0
    for i in range(0, len(config.machines), context["split_workflows"]):  # type: ignore
        logger.info(
            f"Generating workflow for machines {i} to {i + context['split_workflows']}"
        )
        context["machines"] = config.machines[i : i + context["split_workflows"]]
        context["project_workflow"] = str(project_workflow)

        if context["output_file"]:
            s = template.stream(**context)
            with open(context["output_file"], "a") as f:  # type: ignore
                if i != 0:
                    f.write("\n---\n")
                s.dump(f)
        else:
            output = template.render(**context)
            if i != 0:
                print("\n---\n")
            print(output)
        project_workflow += 1


workflow_cli.add_command(workflow_generator_cli)

if __name__ == "__main__":
    workflow_cli()
