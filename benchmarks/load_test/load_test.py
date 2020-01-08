import os
import requests
import subprocess
import yaml
from collections import namedtuple

import numpy as np
import jinja2
import click
from locust import HttpLocust

from gordo.client.utils import EndpointMetadata

TEMPLATE_FILE = "task_set.py.jinja2"
Task = namedtuple("Task", "name path json")


def fetch_metadata(project_name, host, port, ambassador, watchman_port):
    if ambassador:
        data = requests.get(
            f"{host}:{port}/gordo/v0/{project_name}/", proxies={"http": None}
        )
    else:
        data = requests.get(f"{host}:{watchman_port}", proxies={"http": None})

    json_data = yaml.safe_load(data.content)

    endpoint_metadata = [
        EndpointMetadata(endpoint) for endpoint in json_data["endpoints"]
    ]
    tags = {
        metadata.name: {
            "tag_list": len(metadata.tag_list),
            "target_tag_list": len(metadata.target_tag_list),
        }
        for metadata in endpoint_metadata
    }

    return tags


def generate_random_data(len_x, len_y, samples=100):
    if len_y > 0:
        return {
            "X": np.random.random((samples, len_x)).tolist(),
            "y": np.random.random((samples, len_y)).tolist(),
        }
    return {"X": np.random.random((samples, len_x)).tolist()}


def make_tasks(tags, endpoint):
    tasks = list()
    for name, dict_values in tags.items():
        data = generate_random_data(
            dict_values["tag_list"], dict_values["target_tag_list"]
        )
        path = f"/gordo/v0/{endpoint}/{name}/prediction"
        tasks.append(Task(name=f"fn_{name.replace('-', '_')}", path=path, json=data))

    return tasks


@click.command()
@click.option("--project-name", type=str, help="Project name", required=True)
@click.option(
    "--host",
    type=str,
    default="http://127.0.0.1",
    help="Host is usually 127.0.0.1 (localhost)",
)
@click.option("--port", type=int, default=8888, help="Port which hosts the project")
@click.option(
    "--ambassador", is_flag=True, help="Use if routing should go through ambassador"
)
@click.option(
    "--watchman-port",
    type=int,
    default=8889,
    help="Use if you wanna port-forward watchman to a different port",
)
def main(project_name, host, port, ambassador, watchman_port):
    template_loader = jinja2.FileSystemLoader(searchpath=os.path.dirname(__file__))
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template(TEMPLATE_FILE)

    tags = fetch_metadata(project_name, host, port, ambassador, watchman_port)
    tasks = make_tasks(tags, project_name)
    path = os.path.join(os.path.dirname(__file__), "task_set.py")
    template.stream(tasks=tasks).dump(path)

    subprocess.run(["locust", "-f", f"{__file__}", "--host", f"{host}:{port}"])


if __name__ == "__main__":
    main()


class MyLocust(HttpLocust):
    from benchmarks.load_test.task_set import Tasks

    task_set = Tasks
    min_wait = 1000
    max_wait = 1000
