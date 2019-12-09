from gordo_components.workflow.controller_to_sql.controller_to_sql import (
    _extract_machines_from_controller_response,
)


def test_extract_machine_from_controller_response():
    example_response = [
        {
            "spec": {
                "config": {
                    "name": "test_machine",
                    "dataset": {
                        "train_start_date": "2019-05-08T14:10:38Z",
                        "train_end_date": "2019-06-08T14:10:38Z",
                    },
                }
            }
        }
    ]

    all_machines = _extract_machines_from_controller_response(example_response)

    assert len(all_machines) == 1
