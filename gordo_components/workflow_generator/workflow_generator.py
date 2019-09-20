import argparse
import json
import os
import yaml
import dateutil.parser
import logging
import configargparse
import jinja2
import time
import pkg_resources
import io

from typing import Union

from gordo_components.config_elements.normalized_config import NormalizedConfig
from gordo_components import __version__

logger = logging.getLogger(__name__)


def _docker_friendly_version(version):
    """
    Some untagged versions may have a '+' in which case is not a valid
    docker tag
    """
    return version.replace("+", "_")


def _valid_owner_ref(owner_reference_str: str):
    """Validates that the parameter can be loaded (as yaml) into a non-empty list of
    valid owner-references.

    A valid owner reference is a dict containing at least the keys 'uid', 'name',
    'kind', and 'apiVersion'. Raises `argparse.ArgumentTypeError` if not, or returns
    the list of owner references if they seem valid.

    Parameters
    ----------
    owner_reference_str: str
        String representation of the list of owner-references, should be parsable as
        yaml/json

    Returns
    -------
    list[dict]
        The list of owner-references

    """
    owner_ref = yaml.safe_load(owner_reference_str)
    if not type(owner_ref) == list or len(owner_ref) < 1:
        raise argparse.ArgumentTypeError(
            "Owner-references must be a list with at least one element"
        )
    for oref in owner_ref:
        if (
            "uid" not in oref
            or "name" not in oref
            or "kind" not in oref
            or "apiVersion" not in oref
        ):
            raise argparse.ArgumentTypeError(
                "All elements in owner-references must contain a uid, name, kind, "
                "and apiVersion key "
            )
    return owner_ref


def create_argparse():
    parser = configargparse.ArgumentParser(
        description="Machine Configuration to Argo Workflow",
        add_env_var_help=True,
        auto_env_var_prefix="WORKFLOW_GENERATOR_",
    )
    default_components_version = __version__

    parser.add_argument(
        "--machine-config", type=str, required=True, help="Machine configuration file"
    )
    parser.add_argument(
        "--workflow-template", type=str, required=False, help="Template to expand"
    )
    parser.add_argument(
        "--owner-references",
        type=_valid_owner_ref,
        required=False,
        default=None,
        help="Kubernetes owner references to inject into all created resources. "
        "Should be a nonempty yaml/json list of owner-references, each owner-reference "
        "a dict containing at least the keys 'uid', 'name', 'kind', and 'apiVersion'",
    )
    parser.add_argument(
        "--model-builder-version",
        type=str,
        required=False,
        default=default_components_version,
        help="Version of model-builder",
    )
    parser.add_argument(
        "--model-server-version",
        type=str,
        required=False,
        default=default_components_version,
        help="Version of server-version",
    )
    parser.add_argument(
        "--watchman-version",
        type=str,
        required=False,
        default=default_components_version,
        help="Version of watchman",
    )
    parser.add_argument(
        "--client-version",
        type=str,
        required=False,
        default=default_components_version,
        help="Version of prediction client",
    )
    parser.add_argument(
        "--cleanup-version",
        type=str,
        required=False,
        default=_docker_friendly_version(__version__),
        help="Version of cleanup image (gordo-deploy)",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="Name of the project which own the workflow.",
    )
    parser.add_argument(
        "--project-version",
        type=str,
        required=False,
        default=int(time.time() * 1000),  # unix time milliseconds
        help="Version of the project which own the workflow.",
    )
    parser.add_argument(
        "--output-file", type=str, required=False, help="Optional file to render to"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        required=False,
        default="kubeflow",
        help="Which namespace to deploy services into",
    )
    parser.add_argument(
        "--ambassador-namespace",
        type=str,
        required=False,
        default="ambassador",
        help="Namespace we should expect to find the Ambassador service in.",
    )
    parser.add_argument(
        "--split-workflows",
        type=int,
        required=False,
        default=50,
        help="Split workflows containg more than this number of models into several "
        "workflows, where each workflow contains at most this nr of models. The "
        "workflows are outputted sequentially with '---' in between, which allows "
        "kubectl to apply them all at once.",
    )

    return parser


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


def get_dict_from_yaml(config_file: Union[str, io.StringIO]):
    """
    Read a config file or file like object of YAML into a dict
    """
    # We must override the default constructor for timestamp to ensure the result
    # has tzinfo. Yaml loader never adds tzinfo, but converts to UTC.
    yaml.FullLoader.add_constructor(
        tag="tag:yaml.org,2002:timestamp", constructor=_timestamp_constructor
    )
    if hasattr(config_file, "read"):
        return yaml.load(config_file, Loader=yaml.FullLoader)
    try:
        path_to_config_file = os.path.abspath(config_file)  # type: ignore
        with open(path_to_config_file, "r") as yamlfile:  # type: ignore
            yaml_content = yaml.load(yamlfile, Loader=yaml.FullLoader)
            return yaml_content
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


def create_argparse_for_taglist():
    parser = configargparse.ArgumentParser(
        description="Print unique tags from machine configuration file to stdout"
        "or output file if parameter output-file-tag-list is provided",
        add_env_var_help=True,
        auto_env_var_prefix="UNIQUE_TAGS_",
    )

    parser.add_argument(
        "--machine-config", type=str, required=True, help="Machine configuration file"
    )

    parser.add_argument(
        "--output-file-tag-list",
        type=str,
        required=False,
        help="Optional file to dump list of unique tags",
    )
    return parser


def main_tag_list(args=None):
    """
    Creates a parser to extract list of unique tags from gordo config file
    """

    parser = create_argparse_for_taglist()

    if args is None:
        args, data = parser.parse_known_args()
    else:
        args = parser.parse_args(args)

    yaml_content = get_dict_from_yaml(args.machine_config)

    machines = NormalizedConfig(yaml_content, project_name="test-proj-name").machines

    tag_list = set(tag for machine in machines for tag in machine.dataset.tags)
    if args.output_file_tag_list:
        with open(args.output_file_tag_list, "w") as output_file:
            for item in tag_list:
                output_file.write("%s\n" % item)
    else:
        for tag in tag_list:
            print(tag)


def main(args=None):
    parser = create_argparse()

    if args is None:
        args, data = parser.parse_known_args()
    else:
        args = parser.parse_args(args)

    context = dict()

    yaml_content = get_dict_from_yaml(args.machine_config)

    # Context directly from args.
    context["model_builder_version"] = args.model_builder_version
    context["model_server_version"] = args.model_server_version
    context["watchman_version"] = args.watchman_version
    context["client_version"] = args.client_version
    context["cleanup_version"] = args.cleanup_version
    context["project_name"] = args.project_name
    context["project_version"] = args.project_version
    context["namespace"] = args.namespace
    context["ambassador_namespace"] = args.ambassador_namespace
    # Create normalized config

    config = NormalizedConfig(yaml_content, project_name=args.project_name)

    context["machines"] = config.machines

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

    if args.owner_references:
        context["owner_references"] = json.dumps(args.owner_references)

    if args.workflow_template:
        workflow_template = args.workflow_template
        template = load_workflow_template(workflow_template)
    else:
        workflow_template = pkg_resources.resource_filename(
            "gordo_components.workflow_generator.resources",
            "argo-workflow.yml.template",
        )
        template = load_workflow_template(workflow_template)

    # Clear output file
    if args.output_file:
        open(args.output_file, "w").close()
    for i in range(0, len(config.machines), args.split_workflows):
        logger.info(
            f"Generating workflow for machines {i} to {i + args.split_workflows}"
        )
        context["machines"] = config.machines[i : i + args.split_workflows]

        if args.output_file:
            s = template.stream(**context)
            with open(args.output_file, "a") as f:
                if i != 0:
                    f.write("\n---\n")
                s.dump(f)
        else:
            output = template.render(**context)
            if i != 0:
                print("\n---\n")
            print(output)


if __name__ == "__main__":
    main()
