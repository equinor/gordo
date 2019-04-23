# -*- coding: utf-8 -*-

import os
import yaml
import dateutil.parser
import logging
import jinja2
import time
import typing
import pkg_resources

import click

from gordo_components import __version__
from gordo_components.workflow.config_elements.normalized_config import NormalizedConfig

logger = logging.getLogger(__name__)


class Kwargs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __getattr__(self, item):
        return self.kwargs.get(item)


def _docker_friendly_version(version):
    """
    Some untagged versions may have a '+' in which case is not a valid
    docker tag
    """
    return version.replace("+", "_")


@click.command("workflow-generator")
@click.option(
    "--machine-config",
    help="Machine configuration file",
    envvar="WORKFLOW_GENERATOR_MACHINE_CONFIG",
)
@click.option(
    "--workflow-template",
    help="Template to expand",
    envvar="WORKFLOW_GENERATOR_WORKFLOW_TEMPLATE",
)
@click.option(
    "--project-name",
    help="Name of the project which own the workflow.",
    envvar="WORKFLOW_GENERATOR_PROJECT_NAME",
)
@click.option(
    "--project-version",
    type=str,
    default=time.time(),
    help="Version of the project which own the workflow.",
    envvar="WORKFLOW_GENERATOR_PROJECT_VERSION",
)
@click.option(
    "--output-file",
    help="Optional file to render to",
    envvar="WORKFLOW_GENERATOR_OUTPUT_FILE",
)
@click.option(
    "--namespace",
    type=str,
    default="kubeflow",
    help="Which namespace to deploy services into",
    envvar="WORKFLOW_GENERATOR_NAMESPACE",
)
def workflow_generator_cli(**kwargs):
    return workflow_generator(Kwargs(**kwargs))


def _timestamp_constructor(_loader, node):
    parsed_date = dateutil.parser.isoparse(node.value)
    if parsed_date.tzinfo is None:
        raise ValueError(
            "Provide timezone to timestamp {}."
            " Example: for UTC timezone use {} or {} ".format(
                node.value, node.value + "Z", node.value + "+00:00"
            )
        )
    return parsed_date


def get_dict_from_yaml(config_file: typing.Union[str, typing.IO[str]]) -> dict:
    """
    Read a config file or file like object of YAML into a dict
    """
    yaml.FullLoader.add_constructor(  # type: ignore
        tag="tag:yaml.org,2002:timestamp", constructor=_timestamp_constructor
    )
    if hasattr(config_file, "read"):
        return yaml.load(config_file, Loader=yaml.FullLoader)  # type: ignore
    try:
        path_to_config_file = os.path.abspath(str(config_file))
        with open(path_to_config_file, "r") as yamlfile:
            return yaml.load(yamlfile, Loader=yaml.FullLoader)  # type: ignore
    except FileNotFoundError:
        raise FileNotFoundError(
            "Unable to find config file <{}>".format(path_to_config_file)
        )


def load_workflow_template(workflow_template: str) -> jinja2.Template:
    """
    Loads the Jinja2 Template from a specified path

    Parameters
    ----------
    workflow_template: str
        Path to a workflow template

    Returns
    -------
    jinja2.Template
        Loaded but non-rendered jinja2 template for the workflow
    """
    path_to_workflow_template = os.path.abspath(workflow_template)
    template_dir = os.path.dirname(path_to_workflow_template)

    templateEnv = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir), undefined=jinja2.StrictUndefined
    )
    return templateEnv.get_template(os.path.basename(workflow_template))


@click.command("machine-config-unique-tags")
@click.option(
    "--machine-config",
    help="Machine configuration file",
    envvar="UNIQUE_TAGS_MACHINE_CONFIG",
)
@click.option(
    "--output-file-tag-list",
    help="Optional file to dump list of unique tags",
    envvar="UNIQUE_TAGS_OUTPUT_FILE_TAG_LIST",
)
def machine_config_unique_tags_cli(**kwargs):
    return machine_config_unique_tags(Kwargs(**kwargs))


def machine_config_unique_tags(kwargs: Kwargs):
    """
    Creates a parser to extract list of unique tags from gordo config file
    """
    yaml_content = get_dict_from_yaml(kwargs.machine_config)
    machines = NormalizedConfig.from_config(yaml_content).machines
    tag_list = set(tag for machine in machines for tag in machine.dataset.tags)
    if kwargs.output_file_tag_list:
        with open(kwargs.output_file_tag_list, "w") as output_file:
            for item in tag_list:
                output_file.write("%s\n" % item)
    else:
        for tag in tag_list:
            print(tag)


def workflow_generator(kwargs: Kwargs):

    context = dict()  # type: typing.Dict[str, typing.Any]

    yaml_content = get_dict_from_yaml(kwargs.machine_config)

    context["components_version"] = _docker_friendly_version(__version__)
    context["project_name"] = kwargs.project_name
    context["project_version"] = kwargs.project_version
    context["namespace"] = kwargs.namespace

    # Create normalized config
    config = NormalizedConfig.from_config(yaml_content)

    context["machines"] = config.machines

    # Context requiring pre-processing
    context["sanitized_tags"] = {
        tag: tag_sanitized
        for machine in config.machines
        for tag, tag_sanitized in zip(
            machine.dataset.tags, machine.dataset.sanitized_tags
        )
    }
    context["target_names"] = [machine.name for machine in config.machines]

    if kwargs.workflow_template:
        workflow_template = kwargs.workflow_template
        template = load_workflow_template(workflow_template)
    else:
        workflow_template = pkg_resources.resource_filename(
            "gordo_components.workflow.workflow_generator.resources",
            "argo-workflow.yml.template",
        )
        template = load_workflow_template(workflow_template)

    if kwargs.output_file:
        s = template.stream(**context)
        s.dump(kwargs.output_file)
    else:
        output = template.render(**context)
        print(output)
