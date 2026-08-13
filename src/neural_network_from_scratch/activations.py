"""Activation functions used by NN-01."""

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Apply the sigmoid activation element-wise."""
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative_from_activation(
    activation: np.ndarray,
) -> np.ndarray:
    """Return the sigmoid derivative from an existing activation."""
    return activation * (1.0 - activation)
