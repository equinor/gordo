# -*- coding: utf-8 -*-

import dateutil.parser
import glob
import importlib
import logging
import mock
import nbformat
import os
import sys

import pandas as pd
import numpy as np

from nbconvert.exporters import PythonExporter

from gordo.machine.dataset.data_provider.providers import DataLakeProvider
from gordo.machine.dataset.datasets import TimeSeriesDataset
from gordo.machine.dataset.sensor_tag import SensorTag

logger = logging.getLogger(__name__)

CONFIG = dict(
    train_start_date=dateutil.parser.isoparse("2014-07-01T00:10:00+00:00"),
    train_end_date=dateutil.parser.isoparse("2015-01-01T00:00:00+00:00"),
    tag_list=[SensorTag(name=f"Tag {i}", asset=None) for i in range(10)],
)


def _fake_data():
    data = pd.DataFrame(
        {
            f"sensor_{i}": np.random.random(size=100)
            for i in range(len(CONFIG["tag_list"]))
        }
    )
    return data, data


@mock.patch.object(TimeSeriesDataset, "get_data", return_value=_fake_data())
def test_faked_DataLakeBackedDataset(MockDataset):

    provider = DataLakeProvider(storename="dataplatformdlsprod", interactive=True)
    dataset = TimeSeriesDataset(data_provider=provider, **CONFIG)

    # Should be able to call get_data without being asked to authenticate in tests
    X, y = dataset.get_data()


@mock.patch.object(TimeSeriesDataset, "get_data", return_value=_fake_data())
def test_notebooks(MockDataset, tmpdir):
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
