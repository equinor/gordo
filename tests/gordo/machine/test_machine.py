import pytest
import peewee

from gordo.machine import Machine
from gordo.reporters.postgres import PostgresReporter, Machine as PostgresMachine


def test_builder_with_reporter(postgresdb, metadata):
    """
    Verify a model can take a reporter and .report() will run any given reporters
    """
    reporter = PostgresReporter(host="localhost")
    metadata["runtime"]["reporters"].append(reporter.to_dict())

    machine = Machine(**metadata)

    with pytest.raises(peewee.DoesNotExist):
        PostgresMachine.get(PostgresMachine.name == machine.name)
    machine.report()
    PostgresMachine.get(PostgresMachine.name == machine.name)
