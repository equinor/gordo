# -*- coding: utf-8 -*-

import click
import sys
import subprocess

from .helpers import _azure_cli_exists


@click.group("cluster")
def cluster():
    """
    Set of helpful cluster commands relating for Gordo
    """
    pass


@click.command("login")
@click.option("--name", type=str, prompt="Name of cluster to login to")
@click.option("--admin", prompt="Login as admin?", default=True, is_flag=True)
def cluster_login_cli(name: str, admin: bool):
    """
    Sign into a cluster.

    \b
    Example: "gordo-components cluster login"

    \b
    Parameters
    ----------
    name: str
        Name of cluster to login into, expected the name and resource group name
        are the same
    admin: bool
        Whether login should be as administrator or not.
    \b
    Returns
    -------
    None
    """
    if not _azure_cli_exists():
        sys.exit(1)

    cmd = ["az", "aks", "get-credentials", "--resource-group", name, "--name", name]
    if admin:
        cmd.append("--admin")

    try:
        subprocess.check_output(cmd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as err:
        click.secho(f" \u23F0 Failed to login: {err.stderr}", fg="red")
    else:
        click.secho(f"\U0001F680 Successful login for: '{name}'")


cluster.add_command(cluster_login_cli)
