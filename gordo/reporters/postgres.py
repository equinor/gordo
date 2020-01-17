import logging
from playhouse.postgres_ext import (
    Model,
    PostgresqlExtDatabase,
    CharField,
    BinaryJSONField,
)

from .base import BaseReporter
from gordo.util.utils import capture_args
from gordo.machine import Machine as GordoMachine


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
            logger.debug(f"Inserting machine {machine.name} in sql")  # type: ignore
            Machine.insert(
                dict(
                    name=machine.name,
                    dataset=machine.dataset.to_dict(),
                    model=machine.model,
                    metadata=machine.metadata.to_dict(),
                )
            ).on_conflict_ignore().execute()


class Machine(Model):
    name = CharField(index=True, unique=True)
    dataset = BinaryJSONField()
    model = BinaryJSONField()
    metadata = BinaryJSONField()

    class Meta:
        database = db

    def __repr__(self):
        return f"Machine {self.__data__} "
