import pytest
import peewee
from gordo.reporters.postgres import PostgresReporter, Machine as PostgresMachine
from gordo.machine import Machine


def test_postgres_reporter(postgresdb, metadata):
    """
    Check logging  of a machine into postgres
    """
    reporter1 = PostgresReporter(host="localhost")
    machine1 = Machine(**metadata)

    # Before inserting, the machine does not exist.
    with pytest.raises(peewee.DoesNotExist):
        PostgresMachine.get(PostgresMachine.name == machine1.name)

    reporter1.report(machine1)

    record = PostgresMachine.get(PostgresMachine.name == machine1.name)
    assert record.name == machine1.name

    # Create another logger to ensure nothing happened to the DB
    reporter2 = PostgresReporter(host="localhost")
    machine2 = Machine(**metadata)
    machine2.name = "another-machine"

    reporter2.report(machine2)

    # The first machine is still there
    record = PostgresMachine.get(PostgresMachine.name == machine1.name)
    assert record.name == machine1.name

    # And the second
    record = PostgresMachine.get(PostgresMachine.name == machine2.name)
    assert record.name == machine2.name
