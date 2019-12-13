import tests.utils as tu
from gordo_components.workflow.server_to_sql import server_to_sql as sts


def test_get_machines_from_server(ml_server):
    machines = sts.get_machines_from_server(
        tu.GORDO_PROJECT, f"https://localhost/gordo/v0/{tu.GORDO_PROJECT}/"
    )
    assert isinstance(machines, list)
    assert len(machines) == len(tu.GORDO_TARGETS)
