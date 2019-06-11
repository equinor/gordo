# -*- coding: utf-8 -*-

import unittest
import logging
import ruamel.yaml
import io
import pydoc

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

from gordo_components.model.models import KerasAutoEncoder
from gordo_components.serializer import (
    pipeline_into_definition,
    pipeline_from_definition,
)
from gordo_components.model.register import register_model_builder

logger = logging.getLogger(__name__)


class PipelineToConfigTestCase(unittest.TestCase):
    def setUp(self):
        self.variations_of_same_pipeline = [
            # Normal
            Pipeline(
                [
                    ("pca1", PCA(n_components=2)),
                    (
                        "fu",
                        FeatureUnion(
                            [
                                ("pca2", PCA(n_components=3)),
                                (
                                    "pipe",
                                    Pipeline(
                                        [
                                            ("minmax", MinMaxScaler()),
                                            ("truncsvd", TruncatedSVD(n_components=2)),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                    ("ae", KerasAutoEncoder(kind="feedforward_hourglass")),
                ]
            ),
            # MinMax initialized (wrongly) with a list
            Pipeline(
                [
                    ("pca1", PCA(n_components=2)),
                    (
                        "fu",
                        FeatureUnion(
                            [
                                ("pca2", PCA(n_components=3)),
                                (
                                    "pipe",
                                    Pipeline(
                                        [
                                            ("minmax", MinMaxScaler([0, 1])),
                                            ("truncsvd", TruncatedSVD(n_components=2)),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                    ("ae", KerasAutoEncoder(kind="feedforward_hourglass")),
                ]
            ),
            # MinMax initialized with tuple
            Pipeline(
                [
                    ("pca1", PCA(n_components=2)),
                    (
                        "fu",
                        FeatureUnion(
                            [
                                ("pca2", PCA(n_components=3)),
                                (
                                    "pipe",
                                    Pipeline(
                                        [
                                            ("minmax", MinMaxScaler((0, 1))),
                                            ("truncsvd", TruncatedSVD(n_components=2)),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                    ("ae", KerasAutoEncoder(kind="feedforward_hourglass")),
                ]
            ),
            # First pipeline without explicit steps param, other with.
            Pipeline(
                [
                    ("pca1", PCA(n_components=2)),
                    (
                        "fu",
                        FeatureUnion(
                            [
                                ("pca2", PCA(n_components=3)),
                                (
                                    "pipe",
                                    Pipeline(
                                        steps=[
                                            ("minmax", MinMaxScaler((0, 1))),
                                            ("truncsvd", TruncatedSVD(n_components=2)),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                    ("ae", KerasAutoEncoder(kind="feedforward_hourglass")),
                ]
            ),
        ]

    def test_pipeline_into_definition(self):

        expected_definition = (
            """
            sklearn.pipeline.Pipeline:
              memory:
                steps:
                    - sklearn.decomposition.pca.PCA:
                        copy: true
                        iterated_power: auto
                        n_components: 2
                        random_state:
                        svd_solver: auto
                        tol: 0.0
                        whiten: false
                    - sklearn.pipeline.FeatureUnion:
                        n_jobs:
                        transformer_list:
                        - sklearn.decomposition.pca.PCA:
                            copy: true
                            iterated_power: auto
                            n_components: 3
                            random_state:
                            svd_solver: auto
                            tol: 0.0
                            whiten: false
                        - sklearn.pipeline.Pipeline:
                            memory:
                            steps:
                            - sklearn.preprocessing.data.MinMaxScaler:
                                copy: true
                                feature_range:
                                  - 0
                                  - 1
                            - sklearn.decomposition.truncated_svd.TruncatedSVD:
                                algorithm: randomized
                                n_components: 2
                                n_iter: 5
                                random_state:
                                tol: 0.0
                            verbose: false
                        transformer_weights:
                        verbose: false
                    - gordo_components.model.models.KerasAutoEncoder:
                        kind: feedforward_hourglass
                verbose: false
            """.rstrip()
            .strip()
            .replace(" ", "")
        )

        for pipe in self.variations_of_same_pipeline:

            definition = pipeline_into_definition(pipe)

            # Using ruamel over PyYaml, better output option support
            stream = io.StringIO()
            ruamel.yaml.dump(definition, stream, Dumper=ruamel.yaml.RoundTripDumper)
            stream.seek(0)

            current_output = stream.read().rstrip().strip().replace(" ", "")
            self.assertEqual(
                current_output,
                expected_definition,
                msg=f"Failed output:\n{current_output}\nExpected:----------------\n{expected_definition}",
            )

    def test_into_from(self):
        """
        Pass Pipeline into definition, and then from that definition
        """
        from gordo_components.model.transformer_funcs.general import multiply_by

        self.factories = register_model_builder.factories
        for model in self.factories.keys():

            for model_kind in self.factories[model].keys():
                pipe = Pipeline(
                    [
                        ("step_0", PCA(n_components=2)),
                        (
                            "step_1",
                            FeatureUnion(
                                [
                                    ("step_0", PCA(n_components=3)),
                                    (
                                        "step_1",
                                        Pipeline(
                                            steps=[
                                                ("step_0", MinMaxScaler((0, 1))),
                                                (
                                                    "step_1",
                                                    TruncatedSVD(n_components=2),
                                                ),
                                            ]
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        (
                            "step_2",
                            FunctionTransformer(
                                func=multiply_by, kw_args={"factor": 1}
                            ),
                        ),
                        (
                            "step_3",
                            pydoc.locate(f"gordo_components.model.models.{model}")(
                                kind=model_kind
                            ),
                        ),
                    ]
                )

                pipeline_from_definition(pipeline_into_definition(pipe))

    def test_from_into(self):
        """
        Create pipeline from definition, and create from that definition
        """
        self.factories = register_model_builder.factories
        for model in self.factories.keys():
            for model_kind in self.factories[model].keys():
                definition = f"""
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
                                inverse_func: gordo_components.model.transformer_funcs.general.multiply_by
                                inv_kw_args:
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
                                    verbose: false
                                n_jobs: 1
                                transformer_weights:
                                verbose: false
                            - gordo_components.model.models.{model}:
                                kind: {model_kind}
                        memory:
                        verbose: false
                    """
                definition = ruamel.yaml.load(definition, Loader=ruamel.yaml.Loader)
                pipe = pipeline_from_definition(definition)
                pipeline_into_definition(pipe)


def test_captures_kwarg_to_init():
    """
    Our models allow kwargs which are put into the underlying keras model or to construct
    the underlying model.
    We want to ensure into defintion captures kwargs which are part of the model
    parameters but not part of the __init__ signature
    """
    ae = KerasAutoEncoder(kind="feedforward_hourglass", some_fancy_param="Howdy!")
    definition = pipeline_into_definition(ae)
    parameters = definition[
        f"{KerasAutoEncoder.__module__}.{KerasAutoEncoder.__name__}"
    ]
    assert "some_fancy_param" in parameters
    assert parameters["some_fancy_param"] == "Howdy!"

    # And make sure we can init again
    KerasAutoEncoder(**parameters)
