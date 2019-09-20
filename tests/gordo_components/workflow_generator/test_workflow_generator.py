# -*- coding: utf-8 -*-

import logging
import os
import sys
import unittest

from contextlib import contextmanager
from io import StringIO
from unittest.mock import patch

import docker
import pytest
import yaml

from gordo_components.workflow_generator import workflow_generator as wg
from gordo_components.config_elements.normalized_config import NormalizedConfig


logger = logging.getLogger(__name__)


@contextmanager
def capture_stdout():
    new_out = StringIO()
    old_out = sys.stdout
    try:
        sys.stdout = new_out
        yield sys.stdout
    finally:
        sys.stdout = old_out


def _generate_test_workflow_str(
    path_to_config_files, config_filename, project_name="test-proj-name"
):
    """ Reads a test-config file with workflow_generator, and returns the string
    content of the generated workflow"""
    config_file = os.path.join(path_to_config_files, config_filename)
    args = ["--machine-config", config_file, "--project-name", project_name]
    with patch("sys.argv", args), capture_stdout() as output:
        wg.main(sys.argv)
        getvalue = output.getvalue()
    return getvalue


def _get_env_for_machine_build_serve_task(machine, expanded_template):
    templates = expanded_template["spec"]["templates"]
    do_all = [task for task in templates if task["name"] == "do-all"][0]
    model_builder_machine = [
        task
        for task in do_all["dag"]["tasks"]
        if task["name"] == f"model-builder-{machine}"
    ][0]
    model_builder_machine_env = {
        e["name"]: e["value"] for e in model_builder_machine["arguments"]["parameters"]
    }
    return model_builder_machine_env


class WorkflowGeneratorTestCase(unittest.TestCase):

    path_to_config_files = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"
    )

    def _generate_test_workflow_yaml(
        self, config_filename, project_name="test-proj-name"
    ):
        """ Reads a test-config file with workflow_generator, and returns the parsed
        yaml of the generated workflow """
        getvalue = _generate_test_workflow_str(
            self.path_to_config_files, config_filename, project_name=project_name
        )
        expanded_template = yaml.load(getvalue, Loader=yaml.FullLoader)
        return expanded_template

    @pytest.mark.slowtest
    def test_argo_lint(self):
        """
        Test the our example config file, assumed to be valid, produces a valid workflow via `argo lint`
        """
        client = docker.from_env()

        # Build the argo dockerfile
        repo_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..")
        file = os.path.join(repo_dir, "Dockerfile-argo")

        logger.info("Building Argo docker image...")
        img, _ = client.images.build(
            path=repo_dir,
            dockerfile=file,
            tag="temp-argo",
            use_config_proxy=True,
            buildargs={"HTTPS_PROXY": ""},
        )

        logger.info(
            "Running workflow generator and argo lint on examples/config.yaml..."
        )
        result = client.containers.run(
            img.id,
            command="bash -c 'workflow_generator "
            "--project-name some-project "
            "--machine-config /code/examples/config.yaml "
            "--output-file /tmp/out.yaml "
            "&& argo lint /tmp/out.yaml'",
            auto_remove=True,
            stderr=True,
            stdout=True,
            detach=False,
            volumes={repo_dir: {"bind": "/code", "mode": "ro"}},
        )
        self.assertEqual(
            result.decode().strip().split("\n")[-1], "Workflow manifests validated"
        )
        client.images.remove(img.id, force=True)

    def test_basic_generation(self):

        """ model must be included in the config file """
        """ start/end dates ...always included? or default to specific dates if not included?"""

        project_name = "some-fancy-project-name"
        model_config = (
            "{'sklearn.pipeline.Pipeline': {'steps': ['sklearn.preprocessing.data.MinMaxScaler',"
            " {'gordo_components.model.models.KerasAutoEncoder': {'kind': 'feedforward_hourglass'}}]}}"
        )

        config_filename = "config-test-with-models.yml"
        expanded_template = _generate_test_workflow_str(
            self.path_to_config_files, config_filename, project_name=project_name
        )

        self.assertTrue(
            project_name in expanded_template,
            msg=f"Expected to find project name: {project_name} "
            f"in output: {expanded_template}",
        )
        self.assertTrue(
            model_config in expanded_template,
            msg=f"Expected to find model config: {model_config} "
            f"in output: {expanded_template}",
        )

        yaml_content = wg.get_dict_from_yaml(
            os.path.join(self.path_to_config_files, config_filename)
        )
        machines = NormalizedConfig(yaml_content, project_name=project_name).machines
        self.assertTrue(len(machines) == 2)

    def test_runtime_overrides_server(self):
        expanded_template = self._generate_test_workflow_yaml(
            "config-test-runtime-resource.yaml"
        )

        model_builder_ct_23_0002_env = _get_env_for_machine_build_serve_task(
            "ct-23-0002", expanded_template
        )

        model_builder_ct_23_0003_env = _get_env_for_machine_build_serve_task(
            "ct-23-0003", expanded_template
        )

        # ct_23_0002 uses the global overriden requests, but default limits
        self.assertEqual(
            model_builder_ct_23_0002_env["server-resources-requests-memory"], 111
        )

        # This value must be changed if we change the default values
        self.assertEqual(
            model_builder_ct_23_0002_env["server-resources-limits-cpu"], 1000
        )

        # ct_23_0003 uses locally overriden request memory
        self.assertEqual(
            model_builder_ct_23_0003_env["server-resources-requests-memory"], 201
        )

    def test_overrides_builder_datasource(self):

        expanded_template = self._generate_test_workflow_yaml(
            "config-test-datasource.yml"
        )

        model_builder_machine_1_env = _get_env_for_machine_build_serve_task(
            "machine-1", expanded_template
        )
        model_builder_machine_2_env = _get_env_for_machine_build_serve_task(
            "machine-2", expanded_template
        )
        model_builder_machine_3_env = _get_env_for_machine_build_serve_task(
            "machine-3", expanded_template
        )

        # ct_23_0002 uses the global overriden requests, but default limits
        self.assertEqual(
            "{'type': 'DataLakeProvider', 'threads': 20}",
            model_builder_machine_1_env["data-provider"],
        )

        # This value must be changed if we change the default values
        self.assertEqual(
            "{'type': 'custom', 'threads': 20}",
            model_builder_machine_2_env["data-provider"],
        )

        # ct_23_0003 uses locally overriden request memory
        self.assertEqual(
            "{'type': 'DataLakeProvider', 'threads': 10}",
            model_builder_machine_3_env["data-provider"],
        )

    def test_runtime_overrides_builder(self):
        expanded_template = self._generate_test_workflow_yaml(
            "config-test-runtime-resource.yaml"
        )
        templates = expanded_template["spec"]["templates"]
        model_builder_task = [
            task for task in templates if task["name"] == "model-builder"
        ][0]
        model_builder_resource = model_builder_task["container"]["resources"]

        # We use yaml overriden memory (both request and limits).
        self.assertEqual(model_builder_resource["requests"]["memory"], "121M")

        # This was specified to 120 in the config file, but is bumped to match the
        # request
        self.assertEqual(model_builder_resource["limits"]["memory"], "121M")
        # requests.cpu is all default
        self.assertEqual(model_builder_resource["requests"]["cpu"], "500m")

    def test_runtime_overrides_client_para(self):
        """It is possible to override the parallelization of the client
        through the globals-section of the config file"""
        expanded_template = self._generate_test_workflow_yaml(
            "config-test-runtime-resource.yaml"
        )
        templates = expanded_template["spec"]["templates"]
        client_task = [
            task for task in templates if task["name"] == "gordo-client-waiter"
        ][0]

        client_env = {e["name"]: e["value"] for e in client_task["script"]["env"]}

        self.assertEqual(client_env["GORDO_MAX_CLIENTS"], "10")

    def test_runtime_overrides_client(self):
        expanded_template = self._generate_test_workflow_yaml(
            "config-test-runtime-resource.yaml"
        )
        templates = expanded_template["spec"]["templates"]
        model_client_task = [
            task for task in templates if task["name"] == "gordo-client"
        ][0]
        model_client_resource = model_client_task["script"]["resources"]

        # We use yaml overriden memory (both request and limits).
        self.assertEqual(model_client_resource["requests"]["memory"], "221M")

        # This was specified to 120 in the config file, but is bumped to match the
        # request
        self.assertEqual(model_client_resource["limits"]["memory"], "221M")
        # requests.cpu is all default
        self.assertEqual(model_client_resource["requests"]["cpu"], "100m")

    def test_runtime_overrides_influx(self):
        expanded_template = self._generate_test_workflow_yaml(
            "config-test-runtime-resource.yaml"
        )
        templates = expanded_template["spec"]["templates"]
        influx_task = [
            task for task in templates if task["name"] == "gordo-influx-statefulset"
        ][0]
        influx_statefulset_definition = yaml.load(
            influx_task["resource"]["manifest"], Loader=yaml.FullLoader
        )
        influx_resource = influx_statefulset_definition["spec"]["template"]["spec"][
            "containers"
        ][0]["resources"]
        # We use yaml overriden memory (both request and limits).
        self.assertEqual(influx_resource["requests"]["memory"], "321M")

        # This was specified to 120 in the config file, but is bumped to match the
        # request
        self.assertEqual(influx_resource["limits"]["memory"], "321M")
        # requests.cpu is default
        self.assertEqual(influx_resource["requests"]["cpu"], "520m")
        self.assertEqual(influx_resource["limits"]["cpu"], "10040m")

    def test_disable_influx(self):
        """It works to disable influx globally"""
        expanded_template = self._generate_test_workflow_yaml(
            "config-test-disable-influx.yml"
        )
        templates = expanded_template["spec"]["templates"]
        do_all = [task for task in templates if task["name"] == "do-all"][0]
        influx_tasks = [
            task["name"] for task in do_all["dag"]["tasks"] if "influx" in task["name"]
        ]
        client_tasks = [
            task["name"] for task in do_all["dag"]["tasks"] if "client" in task["name"]
        ]

        # The cleanup should be the only influx-related task
        self.assertEqual(influx_tasks, ["influx-cleanup"])
        self.assertEqual(client_tasks, [])

    def test_selective_influx(self):
        """It works to enable/disable influx per machine"""
        expanded_template = self._generate_test_workflow_yaml(
            "config-test-selective-influx.yml"
        )
        templates = expanded_template["spec"]["templates"]
        do_all = [task for task in templates if task["name"] == "do-all"][0]
        influx_tasks = [
            task["name"] for task in do_all["dag"]["tasks"] if "influx" in task["name"]
        ]
        client_tasks = [
            task["name"] for task in do_all["dag"]["tasks"] if "client" in task["name"]
        ]

        # Now we should have both influx and influx-cleanup
        self.assertEqual(influx_tasks, ["influx-cleanup", "gordo-influx"])

        # And we have a single client task for the one client we want running
        self.assertEqual(client_tasks, ["gordo-client-ct-23-0002"])

    def test_main_tag_list(self):
        config_file = os.path.join(
            self.path_to_config_files, "config-test-tag-list.yml"
        )
        args = ["--machine-config", config_file]

        with patch("sys.argv", args), capture_stdout() as output:
            wg.main_tag_list(sys.argv)
            output_tags = output.getvalue()

        output_tags = set(output_tags.split(sep="\n")[:-1])
        expected_output_tags = {"Tag 1", "Tag 2", "Tag 3", "Tag 4", "Tag 5"}

        self.assertTrue(
            output_tags == expected_output_tags,
            msg=f"Expected to find: {expected_output_tags}, outputted "
            f"{output_tags}",
        )

    def test_valid_dateformats(self):
        output_workflow = _generate_test_workflow_str(
            self.path_to_config_files, "config-test-allowed-timestamps.yml"
        )

        self.assertEqual(
            output_workflow.count("2016-11-07"), 6
        )  # Three from the dataset and three from the start for tag fetching
        self.assertEqual(output_workflow.count("2017-11-07"), 3)

    def test_model_names_embedded(self):
        """Tests that the generated workflow contains the names of the machines
        it builds a workflow for in the metadata/annotation as a yaml-parsable structure
        """
        output_workflow = self._generate_test_workflow_yaml(
            "config-test-allowed-timestamps.yml"
        )
        parsed_machines = yaml.load(
            output_workflow["metadata"]["annotations"]["gordo-models"]
        )
        self.assertEqual(parsed_machines, ["machine-1", "machine-2", "machine-3"])

    def test_missing_timezone(self):
        with self.assertRaises(ValueError):
            self._generate_test_workflow_yaml("config-test-missing-timezone.yml")

        with self.assertRaises(ValueError):
            self._generate_test_workflow_yaml("config-test-missing-timezone-quoted.yml")

    def test_validates_resource_format(self):
        """We validate that resources are integers"""
        with self.assertRaises(ValueError):
            _generate_test_workflow_str(
                self.path_to_config_files, "config-test-failing-resource-format.yml"
            )
