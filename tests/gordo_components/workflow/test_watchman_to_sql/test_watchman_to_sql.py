from gordo_components.workflow.watchman_to_sql.watchman_to_sql import (
    _extract_machines_from_watchman_response,
)


def test_extract_machine_from_watchman_response():
    example_response = {
        "endpoints": [
            {
                "healthy": True,
                "endpoint-metadata": {
                    "metadata": {
                        "name": "test_machine",
                        "dataset": {
                            "train_start_date": "2019-05-08T14:10:38Z",
                            "train_end_date": "2019-06-08T14:10:38Z",
                        },
                    }
                },
            },
            {"healthy": False},
        ]
    }

    all_machines = _extract_machines_from_watchman_response(example_response)

    assert len(all_machines) == 2
    working_machines = [machine for machine in all_machines if machine is not None]
    assert len(working_machines) == 1
