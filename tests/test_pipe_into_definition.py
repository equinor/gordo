# -*- coding: utf-8 -*-

import unittest
import logging
import ruamel.yaml
import io

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import MinMaxScaler

from gordo_components.model.models import KerasModel
from gordo_components.pipeline_translator import pipeline_into_definition, pipeline_from_definition


logger = logging.getLogger(__name__)


class PipelineToConfigTestCase(unittest.TestCase):

    def setUp(self):
        self.variations_of_same_pipeline = [
            # Normal
            Pipeline([
                ('pca1', PCA(n_components=2)),
                ('fu', FeatureUnion([
                    ('pca2', PCA(n_components=3)),
                    ('pipe', Pipeline([
                        ('minmax', MinMaxScaler()),
                        ('truncsvd', TruncatedSVD(n_components=2))
                    ]))
                ])),
                ('ae', KerasModel(kind='feedforward_symetric'))
            ]),

            # MinMax initialized (wrongly) with a list
            Pipeline([
                ('pca1', PCA(n_components=2)),
                ('fu', FeatureUnion([
                    ('pca2', PCA(n_components=3)),
                    ('pipe', Pipeline([
                        ('minmax', MinMaxScaler([0, 1])),
                        ('truncsvd', TruncatedSVD(n_components=2))
                    ]))
                ])),
                ('ae', KerasModel(kind='feedforward_symetric'))
            ]),

            # MinMax initialized with tuple
            Pipeline([
                ('pca1', PCA(n_components=2)),
                ('fu', FeatureUnion([
                    ('pca2', PCA(n_components=3)),
                    ('pipe', Pipeline([
                        ('minmax', MinMaxScaler((0, 1))),
                        ('truncsvd', TruncatedSVD(n_components=2))
                    ]))
                ])),
                ('ae', KerasModel(kind='feedforward_symetric'))
            ]),

            # First pipeline without explicit steps param, other with.
            Pipeline([
                ('pca1', PCA(n_components=2)),
                ('fu', FeatureUnion([
                    ('pca2', PCA(n_components=3)),
                    ('pipe', Pipeline(steps=[
                        ('minmax', MinMaxScaler((0, 1))),
                        ('truncsvd', TruncatedSVD(n_components=2))
                    ]))
                ])),
                ('ae', KerasModel(kind='feedforward_symetric'))
            ])
        ]

    def test_pipeline_into_definition(self):

        expected_definition = \
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
                    - gordo_components.model.models.KerasModel:
                        kind: feedforward_symetric
                memory:
            """.rstrip().strip().replace(' ', '')

        for pipe in self.variations_of_same_pipeline:

            definition = pipeline_into_definition(pipe)

            # Using ruamel over PyYaml, better output option support
            stream = io.StringIO()
            ruamel.yaml.dump(definition, stream, Dumper=ruamel.yaml.RoundTripDumper)
            stream.seek(0)

            current_output = stream.read().rstrip().strip().replace(' ', '')
            self.assertEqual(current_output, expected_definition)

    def test_into_from(self):
        """
        Pass Pipeline into definition, and then from that definition
        """
        pipe = Pipeline([
            ('step_0', PCA(n_components=2)),
            ('step_1', FeatureUnion([
                ('step_0', PCA(n_components=3)),
                ('step_1', Pipeline(steps=[
                    ('step_0', MinMaxScaler((0, 1))),
                    ('step_1', TruncatedSVD(n_components=2))
                ]))
            ])),
            ('step_2', KerasModel(kind='feedforward_symetric'))
        ])

        pipeline_from_definition(pipeline_into_definition(pipe))

    def test_from_into(self):
        """
        Create pipeline from definition, and create from that definition
        """
        definition = \
            '''
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
                    - gordo_components.model.models.KerasModel:
                        kind: feedforward_symetric
                memory:
            '''
        definition = ruamel.yaml.load(definition, Loader=ruamel.yaml.Loader)
        pipeline_into_definition(pipeline_from_definition(definition))
