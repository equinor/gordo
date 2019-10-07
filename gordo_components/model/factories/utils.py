# -*- coding: utf-8 -*-

import math
from typing import Tuple


def hourglass_calc_dims(
    compression_factor: float, encoding_layers: int, n_features: int
) -> Tuple[int, ...]:
    """
    Calculate the layer dimensions given the number of layers, compression, and features

    Parameters
    ----------
    compression_factor: float
        How small the smallest layer is as a ratio of n_features
        (smallest layer is rounded up to nearest integer). Must satisfy
        0 <= compression_factor <= 1.
    encoding_layers: int
        Number of layers from the input layer (exclusive) to the
        narrowest layer (inclusive). Must be > 0. The total nr of layers
        including input and output layer will be 2*encoding_layers + 1.
    n_features_out: Optional[int]
        Number of features the model will output, default to ``n_features``.

    Returns
    -------
    dims: Tuple[int,...]
         Number of neurons per layers for the encoder, reversed for the decoder.
         Must have len > 0
    """
    if not (1 >= compression_factor >= 0):
        raise ValueError("compression_factor must be 0 <= compression_factor <= 1")
    if encoding_layers < 1:
        raise ValueError("encoding_layers must be >= 1")
    smallest_layer = max(min(math.ceil(compression_factor * n_features), n_features), 1)
    diff = n_features - smallest_layer
    average_slope = diff / encoding_layers
    dims = tuple(
        round(n_features - (i * average_slope)) for i in range(1, encoding_layers + 1)
    )
    return dims


def check_dim_func_len(prefix: str, dim: Tuple[int, ...], func: Tuple[str, ...]):
    """
    Check that the number of layer dimensions and layer functions are equal

    Parameters
    ----------
    prefix: str
        Parameter name prefix for error message generation (Options: "encoding" or "decoding").
    dim: tuple of int
        Tuple of numbers with the number of neurons in the encoding or decoding part.
    func: Tuple[str,...]
        Tuple of numbers with the number of neurons in the decoding part.
    """
    if len(dim) != len(func):
        raise ValueError(
            f"The length (i.e. the number of network layers) of {prefix}_dim "
            f"({len(dim)}) and {prefix}_func ({len(func)}) must be equal. If only "
            f"{prefix}_dim or {prefix}_func was passed, ensure that its length matches "
            f"that of the {prefix} parameter not passed."
        )
