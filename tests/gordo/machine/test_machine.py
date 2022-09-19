import pytest
import peewee
import json
import yaml

from gordo.machine import Machine
from gordo.machine.constants import MACHINE_YAML_FIELDS
from gordo.reporters.postgres import PostgresReporter, Machine as PostgresMachine
from gordo_core.sensor_tag import SensorTag


def test_builder_with_reporter(postgresdb, metadata):
    """
    Verify a model can take a reporter and .report() will run any given reporters
    """
    reporter = PostgresReporter(host="localhost")
    metadata["runtime"]["reporters"].append(reporter.to_dict())

    machine = Machine.from_dict(metadata)

    with pytest.raises(peewee.DoesNotExist):
        PostgresMachine.get(PostgresMachine.name == machine.name)
    machine.report()
    PostgresMachine.get(PostgresMachine.name == machine.name)


def create_machine():
    return Machine.from_config(
        dict(
            name="special-model-name",
            model={"sklearn.decomposition.PCA": {"svd_solver": "auto"}},
            dataset={
                "type": "RandomDataset",
                "train_start_date": "2017-12-25 06:00:00Z",
                "train_end_date": "2017-12-30 06:00:00Z",
                "tag_list": [SensorTag("Tag 1"), SensorTag("Tag 2")],
                "target_tag_list": [SensorTag("Tag 3"), SensorTag("Tag 4")],
            },
            project_name="test-proj",
        )
    )


def test_to_json():
    machine = create_machine()
    json_result = machine.to_json()
    result = json.loads(json_result)
    for field in MACHINE_YAML_FIELDS:
        if field in result:
            json.loads(result[field])


def test_to_yaml():
    machine = create_machine()
    yaml_result = machine.to_yaml()
    result = yaml.load(yaml_result)
    for field in MACHINE_YAML_FIELDS:
        if field in result:
            yaml.load(result[field])
