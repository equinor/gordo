# -*- coding: utf-8 -*-

import logging
import unittest
import yaml
import copy
import pydoc

import pytest
import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.multioutput import MultiOutputRegressor

from gordo import serializer
from gordo.serializer import from_definition
import gordo.machine.model.transformer_funcs.general
from gordo.machine.model.register import register_model_builder

from tests.gordo.serializer.definition_test_model import DefinitionTestModel

logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "definition",
    [
        """ 
    sklearn.multioutput.MultiOutputRegressor:
      estimator: sklearn.ensemble.RandomForestRegressor
    """,
        """ 
    sklearn.multioutput.MultiOutputRegressor:
      estimator: 
        sklearn.ensemble.RandomForestRegressor:
          n_estimators: 20
    """,
        """ 
    sklearn.multioutput.MultiOutputRegressor:
      estimator: 
        sklearn.pipeline.Pipeline:
            steps:
                - sklearn.ensemble.RandomForestRegressor:
                    n_estimators: 20
    """,
        """
        sklearn.multioutput.MultiOutputRegressor:
            estimator: 
                sklearn.pipeline.Pipeline:
                    steps:
                        - sklearn.cluster.FeatureAgglomeration:
                            n_clusters: 2
                            pooling_func: numpy.mean
                        - sklearn.linear_model.LinearRegression
        """,
    ],
)
def test_load_from_definition(definition):
    """
    Ensure serializer can load models which take other models as parameters.
    """
    X, y = np.random.random((10, 10)), np.random.random((10, 2))
    definition = yaml.load(definition, Loader=yaml.SafeLoader)
    model = serializer.from_definition(definition)
    assert isinstance(model, MultiOutputRegressor)
    model.fit(X, y)
    model.predict(X)


def test_from_definition_test_model():
    config = """
    tests.gordo.serializer.definition_test_model.DefinitionTestModel:
        depth: "300"
    """
    definition = yaml.load(config)
    model = serializer.from_definition(definition)
    assert type(model) == DefinitionTestModel
    assert model.depth == 300


class ConfigToScikitLearnPipeTestCase(unittest.TestCase):
    def setup_gen(self):
        self.factories = register_model_builder.factories
        for model in self.factories.keys():
            for model_kind in self.factories[model].keys():
                templates = [
                    # This has full parameter names define
                    (
                        f"""
                    sklearn.pipeline.Pipeline:
                        steps:
                            - sklearn.decomposition.PCA:
                                n_components: 2
                                copy: true
                                whiten: false
                                svd_solver:  auto
                                tol: 0.0
                                iterated_power: auto
                                random_state:
                            - sklearn.preprocessing._function_transformer.FunctionTransformer:
                                func: gordo.machine.model.transformer_funcs.general.multiply_by
                                kw_args:
                                    factor: 1
                            - sklearn.pipeline.FeatureUnion:
                                transformer_list:
                                - sklearn.decomposition.PCA:
                                    n_components: 3
                                    copy: true
                                    whiten: false
                                    svd_solver: auto
                                    tol: 0.0
                                    iterated_power: auto
                                    random_state:
                                - sklearn.pipeline.Pipeline:
                                    steps:
                                    - sklearn.preprocessing.MinMaxScaler:
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
                            - gordo.machine.model.models.{model}: 
                                kind: {model_kind}
                    """,
                        pydoc.locate(f"gordo.machine.model.models.{model}"),
                        model_kind,
                    ),
                    (
                        f"""
                    sklearn.pipeline.Pipeline:
                        steps:
                            - sklearn.decomposition.PCA:
                                n_components: 2
                            - sklearn.preprocessing._function_transformer.FunctionTransformer:
                                func: gordo.machine.model.transformer_funcs.general.multiply_by
                                kw_args:
                                    factor: 1
                            - sklearn.pipeline.FeatureUnion:
                                - sklearn.decomposition.PCA:
                                    n_components: 3
                                - sklearn.pipeline.Pipeline:
                                    - sklearn.preprocessing.MinMaxScaler:
                                        feature_range: [0, 1]
                                    - sklearn.decomposition.truncated_svd.TruncatedSVD:
                                        n_components: 2
                            - gordo.machine.model.models.{model}:
                                kind: {model_kind}
                    """,
                        pydoc.locate(f"gordo.machine.model.models.{model}"),
                        model_kind,
                    ),
                    # Define pipeline memory with something other than None w/o metadata
                    (
                        f"""
                    sklearn.pipeline.Pipeline:
                        steps:
                        - sklearn.decomposition.PCA:
                            n_components: 2
                            copy: true
                            whiten: false
                            svd_solver: auto
                            tol: 0.0
                            iterated_power: auto
                            random_state:
                        - sklearn.preprocessing._function_transformer.FunctionTransformer:
                                func: gordo.machine.model.transformer_funcs.general.multiply_by
                                kw_args:
                                    factor: 1
                        - sklearn.pipeline.FeatureUnion:
                            transformer_list:
                            - sklearn.decomposition.PCA:
                                n_components: 3
                                copy: true
                                whiten: false
                                svd_solver: auto
                                tol: 0.0
                                iterated_power: auto
                                random_state:
                            - sklearn.pipeline.Pipeline:
                                steps:
                                - sklearn.preprocessing.MinMaxScaler:
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
                        - gordo.machine.model.models.{model}:
                            kind: {model_kind}
                    """,
                        pydoc.locate(f"gordo.machine.model.models.{model}"),
                        model_kind,
                    ),
                ]
                for template in templates:
                    yield template

    def test_pydoc_locate_class(self):
        self.factories = register_model_builder.factories
        for model in self.factories.keys():
            self.assertTrue(pydoc.locate(f"gordo.machine.model.models.{model}"))

    def test_from_definition(self):

        for raw_yaml, model, model_kind in self.setup_gen():
            self.assertTrue(model)
            logger.info(raw_yaml)
            config = yaml.load(raw_yaml)
            logger.debug("{}".format(config))

            config_clone = copy.deepcopy(config)  # To ensure no mutation occurs
            pipe = from_definition(config)

            # Test that the original config matches the one passed; no mutation
            self.assertEqual(config, config_clone)

            # Special tests that defining non-default argument holds for a
            # 'key:  ' is evaled to 'key=None'
            if "memory: /tmp" in raw_yaml:
                self.assertEqual(pipe.steps[2][1].transformer_list[1][1].memory, "/tmp")
            self._verify_pipe(pipe, model, model_kind)

    def _verify_pipe(self, pipe, model, model_kind):
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
            step2.func, gordo.machine.model.transformer_funcs.general.multiply_by
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
        self.assertIsInstance(step4, model)
        self.assertTrue(step4.kind, model_kind)
