# -*- coding: utf-8 -*-

import click
import sys
import subprocess
import json

from .helpers import _azure_cli_exists


@click.group("argocd")
def argocd():
    """
    Subcommand for ArgoCD releated activites
    """
    pass


@click.command("login")
@click.option(
    "--host",
    type=str,
    prompt="ArgoCD host, ie. 'grpc.auroracicd06.omnia-aurora-equinor.net'",
)
@click.option(
    "--keyvault-name", type=str, prompt="Keyvault name which holds the password"
)
@click.option(
    "--secret-name", type=str, prompt="Secret name for password held in keyvault"
)
@click.option("--username", type=str, prompt="Username", default="admin")
def argocd_login_cli(host: str, secret_name: str, keyvault_name: str, username: str):
    """
    \b
    Log into an ArgoCD resource

    \f
    Example
    -------
    gordo-components argocd login

    \b
    Parameters
    ----------
    host: str
        Host to login to
    secret_name: str
        Name of the secret to extract from the keyvault
    keyvault_name: str
        Name of the keyvault to look for the secret
    username: str
        Username to login with.

    Returns
    -------
    None
    """
    if not _azure_cli_exists():
        sys.exit(1)

    # Get the password
    cmd = f"az keyvault secret show --name {secret_name} --vault-name {keyvault_name}".split()

    try:
        result = subprocess.check_output(cmd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as err:
        click.secho(f"\u23F0 Failed to get password: {err.stderr}", fg="red")
        sys.exit(1)

    password = json.loads(result.decode("utf-8"))["value"]

    cmd = f"argocd login {host} --password {password} --username {username} --insecure".split()
    try:
        subprocess.check_output(cmd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as err:
        click.secho(f" \u23F0 Failed to login -> {err.stderr} ", fg="red")
        sys.exit(1)
    else:
        click.secho(f"\U0001F389 Logged into ArgoCD on {host}!")


@click.command("clusters")
def argocd_list_clusters_cli():
    """
    List available clusters to the current ArgoCD context
    """
    try:
        result = subprocess.check_output(
            "argocd cluster list".split(), stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as err:
        click.secho(f"Failed to list clusters: {err.stderr}")
    else:
        click.secho(f'List of available clusters: \n{"-" * 40}')
        click.secho(result.decode(), fg="green")


@click.group("add")
def argocd_add_cli_group():
    """
    Subcommand for adding a repo or app to ArgoCD
    """
    pass


@click.command("repo")
@click.option(
    "--git-url",
    type=str,
    prompt='SSH git url to repo, ie "git@github.com:equinor/my-project.git',
)
@click.option(
    "--ssh-private-key",
    type=click.Path(exists=True, dir_okay=False),
    prompt="Path to the private key associated with this repo",
)
def argocd_add_repo_cli(git_url: str, ssh_private_key: str):
    """
    Add a repo to ArgoCD

    \b
    Parameters
    ----------
    git_url: str
        The ssh styled git url. ie. git@github.com:equinor/my-project.git
    ssh_private_key: str
        The path to the private ssh key associated with this repo
    """
    if not _azure_cli_exists():
        sys.exit(1)

    cmd = f"argocd repo add {git_url} --ssh-private-key-path {ssh_private_key}".split()
    try:
        subprocess.check_output(cmd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as err:
        click.secho(f"\u23F0  Failed to add repo: {err.stderr}")
        sys.exit(1)
    else:
        click.secho(f"\U0001F680 Added repo!")


@click.command("app")
@click.option("--name", prompt="Name to give this application on ArgoCD", type=str)
@click.option(
    "--git-url",
    prompt='SSH git url to repo, ie "git@github.com:equinor/my-project.git" for app',
    type=str,
)
@click.option(
    "--auto-sync", is_flag=True, prompt="Do you want to enable auto sync?", default=True
)
@click.option(
    "--namespace", prompt="The namespace to launch this app into", default="kubeflow"
)
@click.option(
    "--destination-server",
    prompt="""The desintation cluster to launch applications into 
    see "gordo-components argocd clusters" for a list of available clusters/servers. 
    This should include the "https:// and port" prefix""",
    type=str,
)
@click.option(
    "--path",
    prompt="The path of the app within the repo, if not top level",
    default="kubernetes-deployment",
)
@click.option(
    "--env", default="default", prompt="The app environment to use when launching."
)
@click.option(
    "--force",
    default=False,
    is_flag=True,
    prompt="Force an application update, helpful if changing an app's"
    " parameter such as its destination cluster",
)
def argocd_add_app_cli(
    name: str,
    git_url: str,
    auto_sync: bool,
    namespace: str,
    destination_server: str,
    path: str,
    env: str,
    force: bool,
):
    """
    Add an application to ArgoCD

    \b
    Example
    -------
    gordo-components argocd add app

    \b
    Parameters
    ----------
    name: str
        Name of the application you want to give as reference in ArgoCD
    git_url:
        The url got your git repo for this project
    auto_sync: bool
        Auto sync will automatically sync the application after changes are
        detected in the git repo
    namespace: str
        Namespace in Kubernetes to deploy application to
    destination_server: str
        The base url of the desintation server. ie https://my-cluster:443
    path: str
        If the application cannot be auto detected, for example if it's within
        a nested directory of your repo, you can specify the repo path to it.
    env: str
        If your application has different environments, you can specify it here,
        default is 'default'

    """
    cmd = (
        f"argocd app create {name} --repo {git_url} --dest-namespace {namespace} "
        f"--dest-server {destination_server} --env {env}".split()
    )

    if path:
        cmd += f"--path {path}".split()
    if auto_sync:
        cmd += f"--sync-policy automated".split()
    if force:
        cmd += ["--upsert"]

    try:
        subprocess.check_output(cmd, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as err:
        click.secho(f"\u23F0 Failed to add application: {err.stderr}", fg="red")
        sys.exit(1)
    else:
        click.secho(f"\U0001F389 Added application '{name}'!")


argocd.add_command(argocd_login_cli)
argocd.add_command(argocd_list_clusters_cli)

# Subcommand of 'argocd'
argocd.add_command(argocd_add_cli_group)

# Subcommands of 'argocd add'
argocd_add_cli_group.add_command(argocd_add_repo_cli)
argocd_add_cli_group.add_command(argocd_add_app_cli)
