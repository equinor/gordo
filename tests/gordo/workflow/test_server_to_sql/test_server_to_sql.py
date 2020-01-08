from gordo.workflow.server_to_sql import server_to_sql as sts


def test_get_machines_from_server(gordo_project, gordo_targets, ml_server):
    machines = sts.get_machines_from_server(
        gordo_project, f"https://localhost/gordo/v0/{gordo_project}/"
    )
    assert isinstance(machines, list)
    assert len(machines) == len(gordo_targets)
