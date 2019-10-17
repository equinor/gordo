from typing import List, Dict, Optional

import requests
import logging


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

    @classmethod
    def from_endpoint(cls, ep):
        """
        Returns a Machine object constructed from a valid endpoint.

        Parameters
        ----------
        ep: Dict
            Endpoint from watchman

        Returns
        -------
        Machine

        """
        return cls(**endpoint_to_machine_data(ep))

    def __repr__(self):
        return f"Machine {self.__data__} "


def endpoint_to_machine_data(ep):
    name = ep["endpoint-metadata"]["metadata"]["name"]
    train_start_date = ep["endpoint-metadata"]["metadata"]["dataset"][
        "train_start_date"
    ]
    train_end_date = ep["endpoint-metadata"]["metadata"]["dataset"]["train_end_date"]
    return dict(
        name=name,
        metadata=ep,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
    )


def watchman_to_sql(
    watchman_address: str,
    sql_host: str,
    sql_port: int = 5432,
    sql_database: str = "postgres",
    sql_username: str = "postgres",
    sql_password: str = None,
):
    """
    Connects to watchman, downloads all metadata, and proceeds to push them to postgres.

    Parameters
    ----------
    watchman_address: str
        Full URL (including http/https) to watchman
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
    machines = get_machines_from_watchman(watchman_address)
    got_all_machines = None not in machines
    logger.debug(f"Fetched a total of {len(machines)} machines")
    machines = [machine for machine in machines if machine is not None]
    logger.debug(f"Fetched a total of {len(machines)} healthy machines")
    with db.atomic():
        for machine in machines:
            logger.debug(f"Inserting machine {machine['name']} in sql")  # type: ignore
            Machine.insert(machine).on_conflict_ignore().execute()
    return got_all_machines


def get_machines_from_watchman(watchman_address: str) -> List[Optional[Machine]]:
    """
    Returns a list of `Machine` objects from watchman, with None indicating a failed
    machine.

    Parameters
    ----------
    watchman_address: str
        Full address of watchman, including "http" / "https"

    Returns
    -------
    List[Optional[Machine]]
        List of Machine objects, with possible None values indicating a failed machine.

    """
    ret: List[Optional[Machine]] = []
    response = requests.get(watchman_address, timeout=5)
    if response.ok:
        watchman_response = response.json()
        ret = _extract_machines_from_watchman_response(watchman_response)
    return ret


def _extract_machines_from_watchman_response(
    watchman_response: Dict
) -> List[Optional[Machine]]:
    """
    Extracts the list of machines from the watchman json response, with None indicating
    a failed machine.

    Parameters
    ----------
    watchman_response: Dict
        Response from watchman

    Examples
    --------
    >>> example_response = {
    ...     "endpoints": [
    ...         {
    ...             "endpoint-metadata": {
    ...                    "metadata": {
    ...                       "user-defined": {"machine-name": "test_machine1"},
    ...                        "dataset": {
    ...                            "train_start_date": "2019-05-08T14:10:38Z",
    ...                            "train_end_date": "2019-06-08T14:10:38Z",
    ...                        },
    ...                        "name": "machine-1"
    ...                    }},
    ...             "healthy": False
    ...          },
    ...          {
    ...             "endpoint-metadata": {
    ...                    "metadata": {
    ...                       "user-defined": {"machine-name": "test_machine2"},
    ...                        "dataset": {
    ...                            "train_start_date": "2019-05-08T14:10:38Z",
    ...                            "train_end_date": "2019-06-08T14:10:38Z",
    ...                        },
    ...                        "name": "machine-2"
    ...                    }},
    ...             "healthy": True
    ...          }
    ...     ]
    ... }
    >>> machines = _extract_machines_from_watchman_response(example_response)
    >>> len(machines)
    2
    >>> len([m for m in machines if m is not None] )
    1

    Returns
    -------
    List[Optional[Machine]]
        List of Machine objects, with possible None values indicating a failed machine.
    """
    machines = []
    if "endpoints" in watchman_response:
        for ep in watchman_response["endpoints"]:
            if ep["healthy"]:
                machines.append(endpoint_to_machine_data(ep))
            else:
                logger.info(f"Found non-healthy endpoint {ep}, ignoring")
                machines.append(None)
    return machines
