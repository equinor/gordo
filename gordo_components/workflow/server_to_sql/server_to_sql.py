import logging
from urllib.parse import urlparse, ParseResult
from typing import List

from gordo_components.workflow.config_elements.machine import Machine as MachineConfig
from gordo_components.client import Client


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from playhouse.postgres_ext import (
    Model,
    PostgresqlExtDatabase,
    CharField,
    BinaryJSONField,
    DateTimeTZField,
)

db = PostgresqlExtDatabase(None)


class Machine(Model):
    name = CharField(index=True, unique=True)
    metadata = BinaryJSONField()
    train_start_date = DateTimeTZField()
    train_end_date = DateTimeTZField()

    class Meta:
        database = db

    def __repr__(self):
        return f"Machine {self.__data__} "


def machine_config_to_machine_data(machine: MachineConfig):
    return dict(
        name=machine.name,
        metadata=machine.metadata,
        train_start_date=machine.dataset.train_start_date,
        train_end_date=machine.dataset.train_end_date,
    )


def server_to_sql(
    project_name: str,
    server_address: str,
    sql_host: str,
    sql_port: int = 5432,
    sql_database: str = "postgres",
    sql_username: str = "postgres",
    sql_password: str = None,
):
    """
    Connects to server, downloads all metadata, and proceeds to push them to postgres.

    Parameters
    ----------
    project_name: str
        Name of the project to run against.
    server_address:
        Address of the server to connect to. ie. https://myhost.com
    sql_host: str
        Host where postgres resides, e.g. "localhost"
    sql_port: int
        Port for postgres
    sql_database: str
        Database name the metadata storage
    sql_username: str
        Username to use for authenticating towards postgres
    sql_password: str
        Password to use for authenticating towards postgres

    Returns
    -------
    bool
        Returns True if we successfully got all metadata for all machines, False
        otherwise

    """
    sql_parameters = {"host": sql_host, "port": sql_port, "user": sql_username}
    if sql_password:
        sql_parameters.update({"password": sql_password})
    db.init(sql_database, **sql_parameters)
    Machine.create_table(safe=True)
    machines = get_machines_from_server(project_name, server_address)
    got_all_machines = None not in machines
    logger.debug(f"Fetched a total of {len(machines)} machines")
    machines = [machine for machine in machines if machine is not None]
    logger.debug(f"Fetched a total of {len(machines)} healthy machines")
    with db.atomic():
        for machine in machines:
            logger.debug(f"Inserting machine {machine['name']} in sql")  # type: ignore
            Machine.insert(machine).on_conflict_ignore().execute()
    return got_all_machines


def get_machines_from_server(
    project_name: str, server_address: str
) -> List[MachineConfig]:
    """
    TODO: Parameters
    """
    address: ParseResult = urlparse(server_address)

    client = Client(
        project=project_name,
        host=address.netloc,
        port=address.port or 443 if address.scheme == "https" else 80,
        scheme=address.scheme,
    )
    return client.machines
