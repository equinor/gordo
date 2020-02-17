import pytest
import peewee
from gordo.reporters.postgres import PostgresReporter, Machine as PostgresMachine
from gordo.machine import Machine
from gordo.reporters.exceptions import ReporterException


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


def test_postgres_exceptions(metadata, postgresdb):

    # No database to connect to
    with pytest.raises(ReporterException):
        PostgresReporter(host="does-not-exist")

    # Bad machine to report
    reporter = PostgresReporter(host="localhost")
    with pytest.raises(ReporterException):
        reporter.report("This it not a a machine instance.")  # type: ignore


def test_overwrite_report(postgresdb, metadata):
    """
    Ensure saving same machine is ok.
    """
    reporter1 = PostgresReporter(host="localhost")
    reporter2 = PostgresReporter(host="localhost")

    machine1 = Machine(**metadata)
    machine2 = Machine(**metadata)

    reporter1.report(machine1)

    # Reporting twice should be ok.
    reporter2.report(machine2)

    results = PostgresMachine.select().where(PostgresMachine.name == machine1.name)
    assert len([result for result in results]) == 1
