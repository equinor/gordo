import pytest

from gordo.workflow.config_elements.normalized_config import NormalizedConfig


def test_splited_docker_images():
    config = {"machines": [], "globals": {"runtime": {}}}
    normalized_config = NormalizedConfig(config, "test", "1.0.0")
    config_globals = normalized_config.globals
    config_runtime = config_globals["runtime"]
    assert config_runtime["deployer"]["image"] == "gordo-deploy"
    assert config_runtime["server"]["image"] == "gordo-model-server"
    assert config_runtime["prometheus_metrics_server"]["image"] == "gordo-model-server"
    assert config_runtime["builder"]["image"] == "gordo-model-builder"
    assert config_runtime["client"]["image"] == "gordo-client"


def test_unified_docker_images():
    config = {"machines": [], "globals": {"runtime": {}}}
    normalized_config = NormalizedConfig(config, "test", "1.3.0")
    config_globals = normalized_config.globals
    config_runtime = config_globals["runtime"]
    assert config_runtime["deployer"]["image"] == "gordo-base"
    assert config_runtime["server"]["image"] == "gordo-base"
    assert config_runtime["prometheus_metrics_server"]["image"] == "gordo-base"
    assert config_runtime["builder"]["image"] == "gordo-base"
    assert config_runtime["client"]["image"] == "gordo-base"


def test_custom_docker_images():
    config = {
        "machines": [],
        "globals": {
            "runtime": {
                "deployer": {"image": "my-deployer"},
                "server": {"image": "my-server"},
                "builder": {"image": "my-builder"},
            }
        },
    }
    normalized_config = NormalizedConfig(config, "test", "1.1.0")
    config_globals = normalized_config.globals
    config_runtime = config_globals["runtime"]
    assert config_runtime["deployer"]["image"] == "my-deployer"
    assert config_runtime["server"]["image"] == "my-server"
    assert config_runtime["prometheus_metrics_server"]["image"] == "gordo-model-server"
    assert config_runtime["builder"]["image"] == "my-builder"
    assert config_runtime["client"]["image"] == "gordo-client"
