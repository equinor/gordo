# -*- coding: utf-8 -*-

import math


def hourglass_calc_dims(compression_factor, encoding_layers, n_features):
    if not (1 >= compression_factor >= 0):
        raise ValueError("compression_factor must be 0 <= compression_factor <= 1")
    if encoding_layers < 1:
        raise ValueError("encoding_layers must be >= 1")
    smallest_layer = max(min(math.ceil(compression_factor * n_features), n_features), 1)
    diff = n_features - smallest_layer
    average_slope = diff / encoding_layers
    dims = [
        round(n_features - (i * average_slope)) for i in range(1, encoding_layers + 1)
    ]
    return dims
