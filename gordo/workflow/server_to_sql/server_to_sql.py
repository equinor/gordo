import logging
from urllib.parse import urlparse, ParseResult
from typing import List

from gordo.machine import Machine as MachineConfig
from gordo.client import Client


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from playhouse.postgres_ext import (
    Model,
    PostgresqlExtDatabase,
    CharField,
    BinaryJSONField,
)

db = PostgresqlExtDatabase(None)


class Machine(Model):
    name = CharField(index=True, unique=True)
    dataset = BinaryJSONField()
    model = BinaryJSONField()
    metadata = BinaryJSONField()

    class Meta:
        database = db

    def __repr__(self):
        return f"Machine {self.__data__} "


def machine_config_to_machine_data(machine: MachineConfig):
    return dict(
        name=machine.name,
        dataset=machine.dataset.to_dict(),
        model=machine.model,
        metadata=machine.metadata.to_dict(),
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
        Address of the server to connect to. e.g. "https://myhost.com".
    sql_host: str
        Host where postgres resides, e.g. "localhost".
    sql_port: int
        Port for postgres.
    sql_database: str
        Database name the metadata storage.
    sql_username: str
        Username to use for authenticating towards postgres.
    sql_password: str
        Password to use for authenticating towards postgres.

    Returns
    -------
    bool
        Returns True if we successfully got all metadata for all machines.

    """
    sql_parameters = {"host": sql_host, "port": sql_port, "user": sql_username}
    if sql_password:
        sql_parameters.update({"password": sql_password})
    db.init(sql_database, **sql_parameters)
    Machine.create_table(safe=True)
    machines = get_machines_from_server(project_name, server_address)
    logger.debug(f"Fetched a total of {len(machines)} machines")
    with db.atomic():
        for machine in machines:
            logger.debug(f"Inserting machine {machine.name} in sql")  # type: ignore
            Machine.insert(
                machine_config_to_machine_data(machine)
            ).on_conflict_ignore().execute()
    return True


def get_machines_from_server(
    project_name: str, server_address: str
) -> List[MachineConfig]:
    """
    Get a list of machines from the server

    Parameters
    ----------
    project_name: str
        Project name to query machines for.
    server_address: str
        The address of the server e.g. "https://myhost.com".

    Returns
    -------
    List[gordo.workflow.config_elements.machine.Machine]
    """
    address: ParseResult = urlparse(server_address)

    client = Client(
        project=project_name,
        host=address.netloc,
        port=address.port or 443 if address.scheme == "https" else 80,
        scheme=address.scheme,
    )
    return client.machines
