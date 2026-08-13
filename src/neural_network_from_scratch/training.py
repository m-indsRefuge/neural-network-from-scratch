"""Deterministic training loop for NN-01."""

import numpy as np

from neural_network_from_scratch.losses import binary_cross_entropy
from neural_network_from_scratch.network import (
    Parameters,
    backward,
    forward,
    initialize_parameters,
    update_parameters,
)


def train(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> tuple[Parameters, list[float]]:
    """Train NN-01 using full-batch gradient descent."""
    parameters = initialize_parameters(seed)
    losses: list[float] = []

    for _ in range(epochs):
        predictions, cache = forward(inputs, parameters)
        loss = binary_cross_entropy(targets, predictions)
        losses.append(loss)

        gradients = backward(targets, parameters, cache)
        parameters = update_parameters(
            parameters,
            gradients,
            learning_rate,
        )

    return parameters, losses
