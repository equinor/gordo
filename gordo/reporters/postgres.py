import json
import logging

import peewee
from playhouse.postgres_ext import (
    Model,
    PostgresqlExtDatabase,
    CharField,
    BinaryJSONField,
)
from playhouse.shortcuts import dict_to_model, model_to_dict

from .base import BaseReporter
from gordo.util.utils import capture_args
from gordo.machine import Machine as GordoMachine
from gordo.machine.machine import MachineEncoder
from .exceptions import ReporterException


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


db = PostgresqlExtDatabase(None)


class PostgresReporterException(ReporterException):
    pass


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

        try:
            self.db.init(self.database, **sql_parameters)
            Machine.create_table(safe=True)
        except Exception as exc:
            raise PostgresReporterException(exc)

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
        try:
            with self.db.atomic():
                logger.info(f"Inserting machine {machine.name} in sql")  # type: ignore

                # Ensure it's serializable using MachineEncoder
                record = json.loads(json.dumps(machine.to_dict(), cls=MachineEncoder))
                model = dict_to_model(Machine, record, ignore_unknown=True)
                try:
                    Machine.get(Machine.name == machine.name)
                except peewee.DoesNotExist:
                    model.save()
                else:
                    query = Machine.update(**model_to_dict(model)).where(
                        Machine.name == machine.name
                    )
                    query.execute()

        except Exception as exc:
            raise PostgresReporterException(exc)


class Machine(Model):
    name = CharField(index=True, unique=True)
    dataset = BinaryJSONField()
    model = BinaryJSONField()
    metadata = BinaryJSONField()

    class Meta:
        primary_key = False
        database = db
        table_name = "machine"

    def __repr__(self):
        return f"Machine {self.__data__} "
