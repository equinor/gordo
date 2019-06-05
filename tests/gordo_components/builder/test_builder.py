# -*- coding: utf-8 -*-

import unittest
import pytest
import os
import dateutil.parser
import yaml

from typing import List, Optional, Dict
from tempfile import TemporaryDirectory
from gordo_components.builder.build_model import (
    _save_model_for_workflow,
    provide_saved_model,
)
from gordo_components.builder import build_model
from gordo_components.dataset.sensor_tag import SensorTag


def get_random_data():
    data = {
        "type": "RandomDataset",
        "from_ts": dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
        "to_ts": dateutil.parser.isoparse("2017-12-30 06:00:00Z"),
        "tag_list": [SensorTag("Tag 1", None), SensorTag("Tag 2", None)],
    }
    return data


class ModelBuilderTestCase(unittest.TestCase):
    """
    Test functionality of the builder processes
    """

    def test_output_dir(self):
        """
        Test building of model will create subdirectories for model saving if needed.
        """
        from gordo_components.builder import build_model

        with TemporaryDirectory() as tmpdir:

            model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
            data_config = get_random_data()
            output_dir = os.path.join(tmpdir, "some", "sub", "directories")

            model, metadata = build_model(
                name="model-name",
                model_config=model_config,
                data_config=data_config,
                metadata={},
            )

            self.metadata_check(metadata, False)

            _save_model_for_workflow(
                model=model, metadata=metadata, output_dir=output_dir
            )

            # Assert the model was saved at the location
            # using gordo_components.serializer should create some subdir(s)
            # which start with 'n_step'
            dirs = [d for d in os.listdir(output_dir) if d.startswith("n_step")]
            self.assertGreaterEqual(
                len(dirs),
                1,
                msg="Expected saving of model to create at "
                f"least one subdir, but got {len(dirs)}",
            )

    def test_model_builder_model_withouth_pipeline(self):

        # MinMax is only a transformer and does not have a score either.
        raw_model_config = """
        sklearn.preprocessing.data.MinMaxScaler:
            feature_range: [-1, 1]
        """

        model_config = yaml.load(raw_model_config, Loader=yaml.FullLoader)
        data_config = get_random_data()

        model, metadata = build_model(
            name="model-name",
            model_config=model_config,
            data_config=data_config,
            metadata={},
        )

        self.metadata_check(metadata, False)

    def test_model_builder_save_history(self):
        """Checks that the metadata contains the keras model build history"""
        raw_model_config = """
        gordo_components.model.models.KerasAutoEncoder:
            kind: feedforward_hourglass
        """

        model_config = yaml.load(raw_model_config, Loader=yaml.FullLoader)
        data_config = get_random_data()

        model, metadata = build_model(
            name="model-name",
            model_config=model_config,
            data_config=data_config,
            metadata={},
        )

        self.metadata_check(metadata, True)

    def test_model_builder_pipeline(self):
        raw_model_config = """
        sklearn.pipeline.Pipeline:
            steps:
              - sklearn.preprocessing.data.MinMaxScaler
              - sklearn.decomposition.pca.PCA:
                  svd_solver: auto
        """

        model_config = yaml.load(raw_model_config, Loader=yaml.FullLoader)
        data_config = get_random_data()

        model, metadata = build_model(
            name="model-name",
            model_config=model_config,
            data_config=data_config,
            metadata={},
        )

        self.metadata_check(metadata, False)

    def test_model_builder_pipeline_in_pipeline(self):
        from gordo_components.builder import build_model
        import yaml

        raw_model_config = """
            sklearn.pipeline.Pipeline:
                steps:
                  - sklearn.pipeline.Pipeline:
                      steps:
                        - sklearn.preprocessing.data.MinMaxScaler
                  - sklearn.pipeline.Pipeline:
                      steps:
                        - sklearn.decomposition.pca.PCA:
                            svd_solver: auto
            """

        model_config = yaml.load(raw_model_config, Loader=yaml.FullLoader)
        data_config = get_random_data()

        model, metadata = build_model(
            name="model-name",
            model_config=model_config,
            data_config=data_config,
            metadata={},
        )

        self.metadata_check(metadata, False)

    def metadata_check(self, metadata, check_history):
        self.assertTrue("name" in metadata)
        self.assertTrue("model" in metadata)
        self.assertTrue("cross-validation" in metadata["model"])
        self.assertTrue("scores" in metadata["model"]["cross-validation"])

        # Scores is allowed to be an empty dict. in a case where the pipeline/transformer
        # doesn't implement a .score()
        if metadata["model"]["cross-validation"]["scores"] != dict():
            self.assertTrue(
                "explained-variance" in metadata["model"]["cross-validation"]["scores"]
            )
        if check_history:
            self.assertTrue("history" in metadata["model"])
            self.assertTrue("params" in metadata["model"]["history"])
            self.assertTrue("loss" in metadata["model"]["history"])
            self.assertTrue("acc" in metadata["model"]["history"])

    def test_provide_saved_model_simple_happy_path(self):
        """
        Test provide_saved_model with no caching
        """

        with TemporaryDirectory() as tmpdir:

            model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
            data_config = get_random_data()
            output_dir = os.path.join(tmpdir, "model")

            model_location = provide_saved_model(
                name="model-name",
                model_config=model_config,
                data_config=data_config,
                metadata={},
                output_dir=output_dir,
            )

            # Assert the model was saved at the location
            # using gordo_components.serializer should create some subdir(s)
            # which start with 'n_step'
            dirs = [d for d in os.listdir(model_location) if d.startswith("n_step")]
            self.assertGreaterEqual(
                len(dirs),
                1,
                msg="Expected saving of model to create at "
                f"least one subdir, but got {len(dirs)}",
            )


@pytest.mark.parametrize(
    "should_be_equal,metadata,tag_list,replace_cache",
    [
        (True, None, None, False),
        (False, {"metadata": "something"}, None, False),
        (False, None, [SensorTag("extra_tag", None)], False),
        (False, None, None, True),  # replace_cache gives a new model location
    ],
)
def test_provide_saved_model_caching(
    should_be_equal: bool,
    metadata: Optional[Dict],
    tag_list: Optional[List[SensorTag]],
    replace_cache,
):
    """
    Test provide_saved_model with caching and possible cache busting if metadata,
    tag_list, or replace_cache is set.

    Builds two models and checks if their locations are the same, which will be if and
    only if there is caching.

    Parameters
    ----------
    should_be_equal : bool
        Do we expect the two generated models to be at the same location or not? I.e. do
        we expect caching.
    metadata
        Optional metadata which will be used as metadata for the second model.
    tag_list
        Optional list of strings which be used as the taglist in the dataset for the
        second model.
    replace_cache: bool
        Should we force a model cache replacement?

    """

    if tag_list is None:
        tag_list = []
    if metadata is None:
        metadata = dict()
    with TemporaryDirectory() as tmpdir:

        model_config = {"sklearn.decomposition.pca.PCA": {"svd_solver": "auto"}}
        data_config = get_random_data()
        output_dir = os.path.join(tmpdir, "model")
        registry_dir = os.path.join(tmpdir, "registry")

        model_location = provide_saved_model(
            name="model-name",
            model_config=model_config,
            data_config=data_config,
            output_dir=output_dir,
            metadata={},
            model_register_dir=registry_dir,
        )

        if tag_list:
            data_config["tag_list"] = tag_list
        new_output_dir = os.path.join(tmpdir, "model2")
        model_location2 = provide_saved_model(
            name="model-name",
            model_config=model_config,
            data_config=data_config,
            output_dir=new_output_dir,
            metadata=metadata,
            model_register_dir=registry_dir,
            replace_cache=replace_cache,
        )
        if should_be_equal:
            assert model_location == model_location2
        else:
            assert model_location != model_location2
