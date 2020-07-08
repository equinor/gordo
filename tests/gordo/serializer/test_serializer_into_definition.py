# -*- coding: utf-8 -*-

import logging
import yaml
import json
import pydoc

import pytest

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

from gordo.machine.model.models import KerasAutoEncoder
from gordo.serializer import into_definition, from_definition
from gordo.machine.model.register import register_model_builder

from tests.gordo.serializer.definition_test_model import DefinitionTestModel

logger = logging.getLogger(__name__)


@pytest.fixture
def variations_of_same_pipeline():
    return [
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


def test_into_definition(variations_of_same_pipeline):

    expected_definition = """
        sklearn.pipeline.Pipeline:
            memory: null
            steps:
                - sklearn.decomposition._pca.PCA:
                    copy: true
                    iterated_power: auto
                    n_components: 2
                    random_state: null
                    svd_solver: auto
                    tol: 0.0
                    whiten: false
                - sklearn.pipeline.FeatureUnion:
                    n_jobs: null
                    transformer_list:
                    - sklearn.decomposition._pca.PCA:
                        copy: true
                        iterated_power: auto
                        n_components: 3
                        random_state: null
                        svd_solver: auto
                        tol: 0.0
                        whiten: false
                    - sklearn.pipeline.Pipeline:
                        memory: null
                        steps:
                        - sklearn.preprocessing._data.MinMaxScaler:
                            copy: true
                            feature_range:
                              - 0
                              - 1
                        - sklearn.decomposition._truncated_svd.TruncatedSVD:
                            algorithm: randomized
                            n_components: 2
                            n_iter: 5
                            random_state: null
                            tol: 0.0
                        verbose: false
                    transformer_weights: null
                    verbose: false
                - gordo.machine.model.models.KerasAutoEncoder:
                    kind: feedforward_hourglass
            verbose: false
        """

    expected_definition = yaml.safe_load(expected_definition)

    for pipe in variations_of_same_pipeline:

        definition = into_definition(from_definition(into_definition(pipe)))

        assert json.dumps(definition) == json.dumps(
            expected_definition
        ), f"Failed output:\n{definition}\nExpected:----------------\n{expected_definition}"


def test_into_from():
    """
    Pass Pipeline into definition, and then from that definition
    """
    from gordo.machine.model.transformer_funcs.general import multiply_by

    factories = register_model_builder.factories
    for model in factories.keys():

        for model_kind in factories[model].keys():
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
                                            ("step_1", TruncatedSVD(n_components=2)),
                                        ]
                                    ),
                                ),
                            ]
                        ),
                    ),
                    (
                        "step_2",
                        FunctionTransformer(func=multiply_by, kw_args={"factor": 1}),
                    ),
                    (
                        "step_3",
                        pydoc.locate(f"gordo.machine.model.models.{model}")(
                            kind=model_kind
                        ),
                    ),
                ]
            )

            from_definition(into_definition(pipe))


def test_from_into():
    """
    Create pipeline from definition, and create from that definition
    """
    factories = register_model_builder.factories
    for model in factories.keys():
        for model_kind in factories[model].keys():
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
                            func: gordo.machine.model.transformer_funcs.general.multiply_by
                            kw_args:
                                factor: 1
                            inverse_func: gordo.machine.model.transformer_funcs.general.multiply_by
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
                        - gordo.machine.model.models.{model}:
                            kind: {model_kind}
                    memory:
                    verbose: false
                """
            definition = yaml.safe_load(definition)
            pipe = from_definition(definition)
            into_definition(pipe)


def test_captures_kwarg_to_init():
    """
    Our models allow kwargs which are put into the underlying keras model or to construct
    the underlying model.
    We want to ensure into defintion captures kwargs which are part of the model
    parameters but not part of the __init__ signature
    """
    ae = KerasAutoEncoder(kind="feedforward_hourglass", some_fancy_param="Howdy!")
    definition = into_definition(ae)
    parameters = definition[
        f"{KerasAutoEncoder.__module__}.{KerasAutoEncoder.__name__}"
    ]
    assert "some_fancy_param" in parameters
    assert parameters["some_fancy_param"] == "Howdy!"

    # And make sure we can init again
    KerasAutoEncoder(**parameters)


def test_from_definition_test_model():
    model = DefinitionTestModel(300)
    definition = into_definition(model)
    expected_definition = {
        "tests.gordo.serializer.definition_test_model.DefinitionTestModel": {
            "depth": 300
        }
    }
    assert expected_definition == definition
