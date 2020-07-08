import pytest
import yaml
import tensorflow as tf
import numpy as np
from sklearn.pipeline import Pipeline

from gordo import serializer
from gordo.machine.model.models import KerasRawModelRegressor


@pytest.mark.parametrize(
    "spec_str",
    (
        """
        compile:
            loss: mse
            optimizer: 
              tensorflow.keras.optimizers.SGD:
                learning_rate: 0.001
        spec:
            tensorflow.keras.models.Sequential:
                layers:
                    - tensorflow.keras.layers.Dense: 
                        units: 10
                    - tensorflow.keras.layers.Dense:
                        units: 32
                        kernel_regularizer:
                            tensorflow.keras.regularizers.L1L2:
                                l1: 0.2
                    - tensorflow.keras.layers.Dense:
                        units: 1
    """,
        """
        compile:
            loss: mse
            optimizer: adam
        spec:
            tensorflow.keras.models.Sequential:
                layers:
                    - tensorflow.keras.layers.Input:
                        shape: [9]
                    - tensorflow.keras.layers.Reshape:
                        target_shape: [3, 3]
                    - tensorflow.keras.layers.LSTM:
                        units: 12
                    - tensorflow.keras.layers.Flatten
                    - tensorflow.keras.layers.Dense:
                        units: 1
    """,
    ),
)
def test_raw_keras_basic(spec_str: str):
    """
    Can load a keras.Sequential model from a config/yaml definition(s)
    """
    spec = yaml.safe_load(spec_str)
    pipe = KerasRawModelRegressor(spec)
    model = pipe()
    assert isinstance(model, tf.keras.models.Sequential)


def test_raw_keras_part_of_pipeline():
    """
    It should play well, when tucked into a sklearn.pipeline.Pipeline
    """
    X, y = np.random.random((100, 4)), np.random.random((100, 1))

    config_str = """
    sklearn.pipeline.Pipeline:
        steps:
            - sklearn.decomposition.pca.PCA:
                n_components: 4
            - gordo.machine.model.models.KerasRawModelRegressor:
                kind:
                    compile:
                        loss: mse
                        optimizer: adam
                    spec:
                        tensorflow.keras.models.Sequential:
                            layers:
                                - tensorflow.keras.layers.Dense:
                                    units: 4
                                - tensorflow.keras.layers.Dense:
                                    units: 1
    """
    config = yaml.safe_load(config_str)
    pipe = serializer.from_definition(config)
    assert isinstance(pipe, Pipeline)

    pipe.fit(X, y)
    out = pipe.predict(X)
    assert len(out) == len(y)
