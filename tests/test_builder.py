# -*- coding: utf-8 -*-

import unittest
import os
import dateutil
from tempfile import TemporaryDirectory
from gordo_components.builder.build_model import _save_model_for_workflow


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

            model_config = {
                "gordo_components.model.models.KerasAutoEncoder": {
                    "kind": "feedforward_hourglass"
                }
            }
            data_config = {
                "type": "RandomDataset",
                "from_ts": dateutil.parser.isoparse("2017-12-25 06:00:00Z"),
                "to_ts": dateutil.parser.isoparse("2017-12-30 06:00:00Z"),
                "tag_list": ["Tag 1", "Tag 2"],
            }
            output_dir = os.path.join(tmpdir, "some", "sub", "directories")

            model, metadata = build_model(
                model_config=model_config, data_config=data_config, metadata={}
            )

            # cross val metadata checks
            self.assertTrue("model" in metadata)
            self.assertTrue("cross-validation" in metadata["model"])

            print(metadata)
            self.assertTrue("history" in metadata["model"])
            self.assertTrue("params" in metadata["model"]["history"])
            self.assertTrue("loss" in metadata["model"]["history"])
            self.assertTrue("acc" in metadata["model"]["history"])
            self.assertTrue("scores" in metadata["model"]["cross-validation"])
            self.assertTrue(
                "explained-variance" in metadata["model"]["cross-validation"]["scores"]
            )

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
