import json
import logging
from playhouse.postgres_ext import (
    Model,
    PostgresqlExtDatabase,
    CharField,
    BinaryJSONField,
)
from playhouse.shortcuts import dict_to_model

from .base import BaseReporter
from gordo.util.utils import capture_args
from gordo.machine import Machine as GordoMachine
from gordo.builder.mlflow_utils import MachineEncoder


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


db = PostgresqlExtDatabase(None)


class PostgresReporter(BaseReporter):
    """
    Reporter storing the :class:`gordo.machine.Machine` into a Postgres database.
    """

    db = db

    @capture_args
    def __init__(
        self,
        host: str,
        port: int = 5432,
        user: str = "postgres",
        password: str = "postgres",
        database: str = "postgres",
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        sql_parameters = {"host": self.host, "port": self.port, "user": self.user}
        if self.password:
            sql_parameters.update({"password": self.password})
        self.db.init(self.database, **sql_parameters)
        Machine.create_table(safe=True)

    def report(self, machine: GordoMachine):
        """
        Log a machine to Postgres where top level keys, 'name', 'dataset', 'model',
        and 'metadata' mappings to BinaryJSON fields.

        Parameters
        ----------
        machine: gordo.machine.Machine

        Returns
        -------
        None
        """
        with self.db.atomic():
            logger.info(f"Inserting machine {machine.name} in sql")  # type: ignore

            # Ensure it's serializable using MachineEncoder
            record = json.loads(json.dumps(machine.to_dict(), cls=MachineEncoder))
            dict_to_model(Machine, record, ignore_unknown=True).save()


class Machine(Model):
    name = CharField(index=True, unique=True)
    dataset = BinaryJSONField()
    model = BinaryJSONField()
    metadata = BinaryJSONField()

    class Meta:
        database = db

    def __repr__(self):
        return f"Machine {self.__data__} "
