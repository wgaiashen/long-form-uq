"""SAPLMA feature: mean-pool the output-token hidden states at one layer.

The pooling over tokens happens once, in generate(), which caches the pooled vector
at ALL layers (Tier 2). So here we only SELECT the layer we want for the probe, which
makes trying different layers a free sweep on cached data. A middle layer is usually
more informative than the last one, so expect to compare a few and keep the best.
"""
import numpy as np


def select_layer(pooled_all_layers: np.ndarray, layer: int) -> np.ndarray:
    """pooled_all_layers: (n_examples, n_layers, hidden) -> (n_examples, hidden).

    `layer` indexes the cached layers (e.g. a middle index). The returned matrix is
    exactly what the probe trains on.
    """
    return pooled_all_layers[:, layer, :]
