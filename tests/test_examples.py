# -*- coding: utf-8 -*-
import glob
import importlib
import logging
import nbformat
import os
import sys

from nbconvert.exporters import PythonExporter

logger = logging.getLogger(__name__)


def test_notebooks(tmpdir):
    """
    Ensures all notebooks will run without error
    """
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    notebooks = glob.glob(os.path.join(repo_root, "examples", "*.ipynb"))

    logger.info(f"Found {len(notebooks)} notebooks to test.")

    for notebook in notebooks:

        logger.info(f"Testing notebook: {os.path.basename(notebook)}")

        with open(notebook) as f:
            nb = nbformat.read(f, as_version=4)
            exporter = PythonExporter()
            source, _meta = exporter.from_notebook_node(nb)

            with open(os.path.join(tmpdir, "tmpmodule.py"), "w") as f:
                f.writelines(source)
            with open(os.path.join(tmpdir, "__init__.py"), "w") as f:
                f.write("from .tmpmodule import *")

            # Import this module to 'run' the code
            module_dir = os.path.join(tmpdir, "..")
            sys.path.insert(0, module_dir)

            importlib.import_module(os.path.basename(tmpdir), ".")

            sys.path.remove(module_dir)
