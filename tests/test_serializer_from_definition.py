# -*- coding: utf-8 -*-

import logging
import unittest
import yaml
import copy

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

from gordo_components.model.models import KerasAutoEncoder
from gordo_components.serializer import pipeline_from_definition
import gordo_components.model.transformer_funcs.general


logger = logging.getLogger(__name__)


class ConfigToScikitLearnPipeTestCase(unittest.TestCase):
    def setUp(self):

        self.various_raw_yamls = [
            # This has full parameter names define
            """
            sklearn.pipeline.Pipeline:
                steps:
                    - sklearn.decomposition.pca.PCA:
                        n_components: 2
                        copy: true
                        whiten: false
                        svd_solver:  auto
                        tol: 0.0
                        iterated_power: auto
                        random_state:
                    - sklearn.preprocessing._function_transformer.FunctionTransformer:
                        func: gordo_components.model.transformer_funcs.general.multiply_by
                        kw_args:
                            factor: 1
                    - sklearn.pipeline.FeatureUnion:
                        transformer_list:
                        - sklearn.decomposition.pca.PCA:
                            n_components: 3
                            copy: true
                            whiten: false
                            svd_solver: auto
                            tol: 0.0
                            iterated_power: auto
                            random_state:
                        - sklearn.pipeline.Pipeline:
                            steps:
                            - sklearn.preprocessing.data.MinMaxScaler:
                                feature_range:
                                - 0
                                - 1
                                copy: true
                            - sklearn.decomposition.truncated_svd.TruncatedSVD:
                                n_components: 2
                                algorithm: randomized
                                n_iter: 5
                                random_state:
                                tol: 0.0
                            memory:
                        n_jobs: 1
                        transformer_weights:
                    - gordo_components.model.models.KerasAutoEncoder:
                        kind: feedforward_model
            """,
            # This has only some named parameters included
            """
            sklearn.pipeline.Pipeline:
                steps:
                    - sklearn.decomposition.pca.PCA:
                        n_components: 2
                    - sklearn.preprocessing._function_transformer.FunctionTransformer:
                        func: gordo_components.model.transformer_funcs.general.multiply_by
                        kw_args:
                            factor: 1
                    - sklearn.pipeline.FeatureUnion:
                        - sklearn.decomposition.pca.PCA:
                            n_components: 3
                        - sklearn.pipeline.Pipeline:
                            - sklearn.preprocessing.data.MinMaxScaler:
                                feature_range: [0, 1]
                            - sklearn.decomposition.truncated_svd.TruncatedSVD:
                                n_components: 2
                    - gordo_components.model.models.KerasAutoEncoder:
                        kind: feedforward_model
            """,
            # Define pipeline memory with something other than None w/o metadata
            """
            sklearn.pipeline.Pipeline:
                steps:
                - sklearn.decomposition.pca.PCA:
                    n_components: 2
                    copy: true
                    whiten: false
                    svd_solver: auto
                    tol: 0.0
                    iterated_power: auto
                    random_state:
                - sklearn.preprocessing._function_transformer.FunctionTransformer:
                        func: gordo_components.model.transformer_funcs.general.multiply_by
                        kw_args:
                            factor: 1
                - sklearn.pipeline.FeatureUnion:
                    transformer_list:
                    - sklearn.decomposition.pca.PCA:
                        n_components: 3
                        copy: true
                        whiten: false
                        svd_solver: auto
                        tol: 0.0
                        iterated_power: auto
                        random_state:
                    - sklearn.pipeline.Pipeline:
                        steps:
                        - sklearn.preprocessing.data.MinMaxScaler:
                            feature_range:
                            - 0
                            - 1
                            copy: true
                        - sklearn.decomposition.truncated_svd.TruncatedSVD:
                            n_components: 2
                            algorithm: randomized
                            n_iter: 5
                            random_state:
                            tol: 0.0
                        memory: /tmp
                    n_jobs: 1
                    transformer_weights:
                - gordo_components.model.models.KerasAutoEncoder:
                    kind: feedforward_model
            """,
        ]

    def test_pipeline_from_definition(self):

        for raw_yaml in self.various_raw_yamls:
            config = yaml.load(raw_yaml)
            logger.debug("{}".format(config))

            config_clone = copy.deepcopy(config)  # To ensure no mutation occurs
            pipe = pipeline_from_definition(config)

            # Test that the original config matches the one passed; no mutation
            self.assertEqual(config, config_clone)

            # Special tests that defining non-default argument holds for a
            # 'key:  ' is evaled to 'key=None'
            if "memory: /tmp" in raw_yaml:
                self.assertEqual(pipe.steps[2][1].transformer_list[1][1].memory, "/tmp")
            self._verify_pipe(pipe)

    def _verify_pipe(self, pipe):
        # We should now have a Pipeline.steps == 3
        self.assertTrue(len(pipe.steps), 3)

        # Note: All steps in a Pipeline or Transformer are tuples of (str, obj)
        # That is the reason for the semi-ugly indexing which follows.

        # STEP 1 TEST: Test expected PCA step
        step1 = pipe.steps[0][1]
        self.assertIsInstance(step1, PCA)
        self.assertEqual(step1.n_components, 2)

        # STEP 2 TEST: Test expected FunctionTransformer step
        step2 = pipe.steps[1][1]
        self.assertIsInstance(step2, FunctionTransformer)
        self.assertEqual(
            step2.func, gordo_components.model.transformer_funcs.general.multiply_by
        )

        # STEP 3 TEST: Test expected FeatureUnion Step
        step3 = pipe.steps[2][1]
        self.assertIsInstance(step3, FeatureUnion)

        # First transformer of feature_transformers should be PCA(n_components=3)
        self.assertIsInstance(step3.transformer_list[0][1], PCA)
        self.assertEqual(step3.transformer_list[0][1].n_components, 3)

        # Third transformer in feature_transformers should be Pipeline
        sub_pipeline = step3.transformer_list[1][1]
        self.assertIsInstance(sub_pipeline, Pipeline)

        # First step in the sub pipeline is MinMaxScalar
        self.assertIsInstance(sub_pipeline.steps[0][1], MinMaxScaler)

        # Next step in the sub pipeline is TruncatedSVD w/ n_components=2
        self.assertIsInstance(sub_pipeline.steps[1][1], TruncatedSVD)
        self.assertEqual(sub_pipeline.steps[1][1].n_components, 2)

        # STEP 4 TEST:  Finally, the last step should be a KerasModel
        step4 = pipe.steps[3][1]
        self.assertIsInstance(step4, KerasAutoEncoder)
        self.assertTrue(step4.kind, "feedforward_model")
