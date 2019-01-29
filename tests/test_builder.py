# -*- coding: utf-8 -*-

import unittest
import os
from tempfile import TemporaryDirectory


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
                    "kind": "feedforward_symetric"
                }
            }
            data_config = {"type": "RandomDataset"}
            output_dir = os.path.join(tmpdir, "some", "sub", "directories")

            build_model(
                output_dir=output_dir,
                model_config=model_config,
                data_config=data_config,
                metadata={},
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
