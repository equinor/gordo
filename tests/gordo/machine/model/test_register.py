# -*- coding: utf-8 -*-

import unittest
from gordo.machine.model.register import register_model_builder


class RegisterTestCase(unittest.TestCase):
    def test_fail_no_required_params(self):
        """
        @register required a function which has certain parameters.

        n_features
        """
        # Fail without required param(s)
        with self.assertRaises(ValueError):

            @register_model_builder(type="KerasAutoEncoder")
            def build_fn():
                pass

        # Pass with required param(s)
        @register_model_builder(type="KerasAutoEncoder")  # pragma: no flakes
        def build_fn(n_features):
            pass

        # Call to ensure that register didn't 'eat' the function
        build_fn(1)

    def test_hold_multiple_funcs(self):
        """
        Ensure the register holds references to multiple funcs
        """

        @register_model_builder(type="KerasAutoEncoder")
        def func1(n_features):
            pass

        @register_model_builder(type="KerasAutoEncoder")
        def func2(n_features):
            pass

        self.assertTrue(
            all(
                func_name in register_model_builder.factories["KerasAutoEncoder"]
                for func_name in ["func1", "func2"]
            )
        )
