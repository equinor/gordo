# -*- coding: utf-8 -*-

import subprocess
import click


def _azure_cli_exists() -> bool:
    try:
        subprocess.check_output(["az", "--version"], stderr=subprocess.PIPE)
    except (FileNotFoundError, subprocess.CalledProcessError):
        click.secho(
            f"\u23F0  Failed to find azure CLI install in current environment. "
            f"Running 'pip install azure-cli' may fix this.",
            fg="red",
        )
        return False
    else:
        click.secho("\U0001F382 Found install of Azure CLI", fg="green")
        return True
