# -*- coding: utf-8 -*-

import logging
import os
import unittest

import docker
import pytest

from click.testing import CliRunner

from gordo_components.cli import cli
from gordo_components.workflow.workflow_generator import workflow_generator as wg
from gordo_components.workflow.config_elements.normalized_config import NormalizedConfig


logger = logging.getLogger(__name__)


class WorkflowGeneratorTestCase(unittest.TestCase):

    path_to_config_files = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data"
    )

    @pytest.mark.dockertest
    def test_argo_lint(self):
        """
        Test the our example config file, assumed to be valid, produces a valid workflow via `argo lint`
        """
        client = docker.from_env()

        # Build the argo dockerfile
        repo_dir = os.path.join(os.path.dirname(__file__), "..", "..")
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
            command="bash -c 'gordo-components workflow-generator "
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
        self.assertTrue(
            result.decode().strip().endswith("Workflow manifests validated")
        )
        client.images.remove(img.id, force=True)

    def test_basic_generation(self):

        """ model must be included in the config file """
        """ start/end dates ...always included? or default to specific dates if not included?"""

        project_name = "SOME-FANCY-PROJECT-NAME"
        model_config = (
            "{'sklearn.pipeline.Pipeline': {'steps': ['sklearn.preprocessing.data.MinMaxScaler',"
            " {'gordo_components.model.models.KerasAutoEncoder': {'kind': 'feedforward_hourglass'}}]}}"
        )
        config_file = os.path.join(
            self.path_to_config_files, "config-test-with-models.yml"
        )

        args = [
            "workflow-generator",
            "--machine-config",
            config_file,
            "--project-name",
            project_name,
        ]

        runner = CliRunner()
        result = runner.invoke(cli.gordo, args=args)
        self.assertEqual(result.exit_code, 0)

        expanded_template = result.output

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

        yaml_content = wg.get_dict_from_yaml(config_file)
        machines = NormalizedConfig.from_config(yaml_content).machines
        self.assertTrue(len(machines) == 2)

    def test_main_tag_list(self):
        config_file = os.path.join(
            self.path_to_config_files, "config-test-tag-list.yml"
        )
        args = ["machine-config-unique-tags", "--machine-config", config_file]

        runner = CliRunner()
        result = runner.invoke(cli.gordo, args=args)
        self.assertEqual(result.exit_code, 0)

        output_tags = set(result.output.split(sep="\n")[:-1])
        expected_output_tags = {"Tag 1", "Tag 2", "Tag 3", "Tag 4", "Tag 5"}

        self.assertTrue(
            output_tags == expected_output_tags,
            msg=f"Expected to find: {expected_output_tags}, outputted "
            f"{output_tags}",
        )
