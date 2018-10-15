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

            model_config = {'type': 'keras'}
            data_config = {'type': 'random'}
            output_dir = os.path.join(tmpdir, 'some', 'sub', 'directories')

            build_model(output_dir=output_dir, 
                        model_config=model_config, 
                        data_config=data_config)

            # Assert the model was saved at the location specified
            saved_model = os.path.join(output_dir, 'model.pkl')
            self.assertTrue(os.path.exists(saved_model))
