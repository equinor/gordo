import pytest
import peewee

from gordo.machine import Machine
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


def test_to_json_and_to_yaml():
    machine = Machine.from_config(dict(
         name="special-model-name",
         model={"sklearn.decomposition.PCA": {"svd_solver": "auto"}},
         dataset={
            "type": "RandomDataset",
            "train_start_date": "2017-12-25 06:00:00Z",
            "train_end_date": "2017-12-30 06:00:00Z",
            "tag_list": [SensorTag("Tag 1"), SensorTag("Tag 2")],
            "target_tag_list": [SensorTag("Tag 3"), SensorTag("Tag 4")]
        },
        project_name='test-proj'
    ))
    expected_json = """
    {"name": "special-model-name", "dataset": "{\\"target_tag_list\\": [{\\"name\\": \\"Tag 3\\"}, {\\"name\\": \\"Tag 4\\"}], \\"additional_tags\\": null, \\"default_tag\\": null, \\"data_provider\\": {\\"min_size\\": 100, \\"max_size\\": 300, \\"type\\": \\"gordo_core.data_providers.providers.RandomDataProvider\\"}, \\"resolution\\": \\"10T\\", \\"row_filter\\": \\"\\", \\"known_filter_periods\\": null, \\"aggregation_methods\\": \\"mean\\", \\"row_filter_buffer_size\\": 0, \\"asset\\": null, \\"n_samples_threshold\\": 0, \\"low_threshold\\": -10000, \\"high_threshold\\": 500000, \\"interpolation_method\\": \\"linear_interpolation\\", \\"interpolation_limit\\": \\"48H\\", \\"filter_periods\\": null, \\"train_start_date\\": \\"2017-12-25 06:00:00Z\\", \\"train_end_date\\": \\"2017-12-30 06:00:00Z\\", \\"tag_list\\": [{\\"name\\": \\"Tag 1\\"}, {\\"name\\": \\"Tag 2\\"}], \\"type\\": \\"gordo_core.time_series.RandomDataset\\"}", "model": "{\\"sklearn.decomposition.PCA\\": {\\"svd_solver\\": \\"auto\\"}}", "metadata": "{\\"user_defined\\": {\\"global-metadata\\": {}, \\"machine-metadata\\": {}}, \\"build_metadata\\": {\\"model\\": {\\"model_offset\\": 0, \\"model_creation_date\\": null, \\"model_builder_version\\": \\"1.11.1.dev26+gdb77f9d.d20220727\\", \\"cross_validation\\": {\\"scores\\": {}, \\"cv_duration_sec\\": null, \\"splits\\": {}}, \\"model_training_duration_sec\\": null, \\"model_meta\\": {}}, \\"dataset\\": {\\"query_duration_sec\\": null, \\"dataset_meta\\": {}}}}", "runtime": "{\\"reporters\\": []}", "project_name": "test-proj", "evaluation": "{\\"cv_mode\\": \\"full_build\\"}"}
    """
    assert machine.to_json() == expected_json.strip()
    print(machine.to_yaml())
