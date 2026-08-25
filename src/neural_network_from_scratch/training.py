"""Deterministic training loop for NN-01."""

from collections.abc import Callable

import numpy as np

from neural_network_from_scratch.losses import binary_cross_entropy
from neural_network_from_scratch.network import (
    Parameters,
    backward,
    forward,
    initialize_parameters,
    update_parameters,
)
from neural_network_from_scratch.observability import TrainingSnapshot


def train(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
    observer: Callable[[TrainingSnapshot], None] | None = None,
) -> tuple[Parameters, list[float]]:
    """Train NN-01 using full-batch gradient descent."""
    parameters = initialize_parameters(seed)
    losses: list[float] = []

    for epoch_index in range(epochs):
        predictions, cache = forward(inputs, parameters)
        loss = binary_cross_entropy(targets, predictions)
        losses.append(loss)

        gradients = backward(targets, parameters, cache)

        if observer is not None:
            observer(
                TrainingSnapshot(
                    epoch=epoch_index + 1,
                    loss=loss,
                    parameters=parameters,
                    gradients=gradients,
                    cache=cache,
                )
            )

        parameters = update_parameters(
            parameters,
            gradients,
            learning_rate,
        )

    return parameters, losses