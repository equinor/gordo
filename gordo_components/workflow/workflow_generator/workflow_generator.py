import os
import yaml
import dateutil.parser
import logging
import jinja2
import io

from typing import Union

logger = logging.getLogger(__name__)


def _docker_friendly_version(version):
    """
    Some untagged versions may have a '+' in which case is not a valid
    docker tag
    """
    return version.replace("+", "_")


def _valid_owner_ref(owner_reference_str: str):
    """
    Validates that the parameter can be loaded (as yaml) into a non-empty list of
    valid owner-references.

    A valid owner reference is a dict containing at least the keys 'uid', 'name',
    'kind', and 'apiVersion'. Raises `TypeError` if not, or returns
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
        raise TypeError("Owner-references must be a list with at least one element")
    for oref in owner_ref:
        if (
            "uid" not in oref
            or "name" not in oref
            or "kind" not in oref
            or "apiVersion" not in oref
        ):
            raise TypeError(
                "All elements in owner-references must contain a uid, name, kind, "
                "and apiVersion key "
            )
    return owner_ref


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


def get_dict_from_yaml(config_file: Union[str, io.StringIO]) -> dict:
    """
    Read a config file or file like object of YAML into a dict
    """
    # We must override the default constructor for timestamp to ensure the result
    # has tzinfo. Yaml loader never adds tzinfo, but converts to UTC.
    yaml.FullLoader.add_constructor(
        tag="tag:yaml.org,2002:timestamp", constructor=_timestamp_constructor
    )
    if hasattr(config_file, "read"):
        yaml_content = yaml.load(config_file, Loader=yaml.FullLoader)
    else:
        try:
            path_to_config_file = os.path.abspath(config_file)  # type: ignore
            with open(path_to_config_file, "r") as yamlfile:  # type: ignore
                yaml_content = yaml.load(yamlfile, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Unable to find config file <{path_to_config_file}>"
            )
    # Handle multiple versions of workflow config structure
    if "spec" in yaml_content:
        yaml_content = yaml_content["spec"]["config"]

    return yaml_content


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
