import logging
import time
import pkg_resources
import json
import sys

from typing import Dict, Any

import click

from gordo_components import __version__
from gordo_components.workflow.watchman_to_sql.watchman_to_sql import watchman_to_sql
from gordo_components.workflow.config_elements.normalized_config import NormalizedConfig
from gordo_components.workflow.workflow_generator import workflow_generator as wg


logger = logging.getLogger(__name__)


PREFIX = "WORKFLOW_GENERATOR"


@click.group("workflow")
def workflow_cli():
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
    "--project-version",
    type=str,
    default=int(time.time() * 1000),  # unix time milliseconds
    help="Version of the project which own the workflow.",
    envvar=f"{PREFIX}_PROJECT_VERSION",
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
    "--ambassador-namespace",
    type=str,
    default="ambassador",
    help="Namespace we should expect to find the Ambassador service in.",
    envvar=f"{PREFIX}_AMBASSADOR_NAMESPACE",
)
@click.option(
    "--split-workflows",
    type=int,
    default=50,
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
    default="gordo-components",
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
def workflow_generator_cli(**ctx: dict):
    """
    Machine Configuration to Argo Workflow
    """

    context: Dict[Any, Any] = ctx.copy()

    yaml_content = wg.get_dict_from_yaml(context["machine_config"])

    # Create normalized config
    config = NormalizedConfig(yaml_content, project_name=context["project_name"])

    context["machines"] = config.machines
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
    context["enable_influx"] = nr_of_models_with_clients > 0

    # Context requiring pre-processing
    context["target_names"] = [machine.name for machine in config.machines]

    # Json dump owner_references, if not None, otherwise pop it out of the context
    if context["owner_references"]:
        context["owner_references"] = json.dumps(context["owner_references"])
    else:
        context.pop("owner_references")

    if context["workflow_template"]:
        template = wg.load_workflow_template(context["workflow_template"])
    else:
        workflow_template = pkg_resources.resource_filename(
            "gordo_components.workflow.workflow_generator.resources",
            "argo-workflow.yml.template",
        )
        template = wg.load_workflow_template(workflow_template)

    # Clear output file
    if context["output_file"]:
        open(context["output_file"], "w").close()  # type: ignore
    for i in range(0, len(config.machines), context["split_workflows"]):  # type: ignore
        logger.info(
            f"Generating workflow for machines {i} to {i + context['split_workflows']}"
        )
        context["machines"] = config.machines[i : i + context["split_workflows"]]

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

    tag_list = set(tag for machine in machines for tag in machine.dataset.tags)
    if output_file_tag_list:
        with open(output_file_tag_list, "w") as output_file:
            for item in tag_list:
                output_file.write("%s\n" % item)
    else:
        for tag in tag_list:
            print(tag)


@click.command("watchman-to-sql")
@click.option("--watchman-address", help="Address of watchman", required=True, type=str)
@click.option("--sql-host", help="Host of the sql server", required=True, type=str)
@click.option(
    "--sql-port", help="Port of the sql server", required=False, type=int, default=5432
)
@click.option(
    "--sql-database",
    help="Username of the sql server",
    required=False,
    type=str,
    default="postgres",
)
@click.option(
    "--sql-username",
    help="Username of the sql server",
    required=False,
    type=str,
    default="postgres",
)
@click.option(
    "--sql-password",
    help="Port of the sql server",
    required=False,
    type=str,
    default=None,
)
def watchman_to_sql_cli(
    watchman_address, sql_host, sql_port, sql_database, sql_username, sql_password
):
    """
    Program to fetch metadata from watchman and push the metadata to a postgres sql
    database. Pushes to the table `machine`.
    """
    if watchman_to_sql(
        watchman_address, sql_host, sql_port, sql_database, sql_username, sql_password
    ):
        sys.exit(0)
    else:
        sys.exit(1)


workflow_cli.add_command(workflow_generator_cli)
workflow_cli.add_command(unique_tag_list_cli)
workflow_cli.add_command(watchman_to_sql_cli)

if __name__ == "__main__":
    workflow_cli()
