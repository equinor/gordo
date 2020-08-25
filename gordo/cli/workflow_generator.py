import logging
import time
import pkg_resources
import json
import os

from typing import Dict, Any

import click

from gordo import __version__
from gordo.workflow.config_elements.normalized_config import NormalizedConfig
from gordo.workflow.workflow_generator import workflow_generator as wg
from gordo.cli.exceptions_reporter import ReportLevel


logger = logging.getLogger(__name__)


PREFIX = "WORKFLOW_GENERATOR"
DEFAULT_BUILDER_EXCEPTIONS_REPORT_LEVEL = ReportLevel.TRACEBACK


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
    default="gordo",
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
@click.pass_context
def workflow_generator_cli(gordo_ctx, **ctx):
    """
    Machine Configuration to Argo Workflow
    """

    context: Dict[Any, Any] = ctx.copy()
    yaml_content = wg.get_dict_from_yaml(context["machine_config"])

    try:
        log_level = yaml_content["globals"]["runtime"]["log_level"]
    except KeyError:
        log_level = os.getenv("GORDO_LOG_LEVEL", gordo_ctx.obj["log_level"])

    logging.getLogger("gordo").setLevel(log_level.upper())
    context["log_level"] = log_level.upper()

    # Create normalized config
    config = NormalizedConfig(yaml_content, project_name=context["project_name"])

    context["max_server_replicas"] = (
        context.pop("n_servers") or len(config.machines) * 10
    )

    # We know these exist since we set them in the default globals
    builder_resources = config.globals["runtime"]["builder"]["resources"]
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

    context["server_resources"] = config.globals["runtime"]["server"]["resources"]
    context["prometheus_metrics_server_resources"] = config.globals["runtime"][
        "prometheus_metrics_server"
    ]["resources"]

    # These are also set in the default globals, and guaranteed to exist
    client_resources = config.globals["runtime"]["client"]["resources"]
    context["client_resources_requests_memory"] = client_resources["requests"]["memory"]
    context["client_resources_requests_cpu"] = client_resources["requests"]["cpu"]
    context["client_resources_limits_memory"] = client_resources["limits"]["memory"]
    context["client_resources_limits_cpu"] = client_resources["limits"]["cpu"]

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


@click.command("unique-tags")
@click.option(
    "--machine-config", type=str, required=True, help="Machine configuration file"
)
@click.option(
    "--output-file-tag-list",
    type=str,
    required=False,
    help="Optional file to dump list of unique tags",
)
def unique_tag_list_cli(machine_config: str, output_file_tag_list: str):

    yaml_content = wg.get_dict_from_yaml(machine_config)

    machines = NormalizedConfig(yaml_content, project_name="test-proj-name").machines

    tag_list = set(tag for machine in machines for tag in machine.dataset.tag_list)

    if output_file_tag_list:
        with open(output_file_tag_list, "w") as output_file:
            for tag in tag_list:
                output_file.write(f"{tag.name}\n")
    else:
        for tag in tag_list:
            print(tag.name)


workflow_cli.add_command(workflow_generator_cli)
workflow_cli.add_command(unique_tag_list_cli)

if __name__ == "__main__":
    workflow_cli()
