"""Core NN-01 parameter structures and network operations."""

from dataclasses import dataclass

import numpy as np

from neural_network_from_scratch.activations import sigmoid


@dataclass(frozen=True)
class Parameters:
    """Trainable parameter arrays for the fixed NN-01 architecture."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray


@dataclass(frozen=True)
class ForwardCache:
    """Values retained from a forward pass for later backpropagation."""

    inputs: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    predictions: np.ndarray


def initialize_parameters(seed: int) -> Parameters:
    """Initialize the 337 trainable parameters deterministically."""
    rng = np.random.default_rng(seed)

    return Parameters(
        w1=rng.normal(0.0, np.sqrt(1.0 / 2.0), size=(2, 16)),
        b1=np.zeros((1, 16)),
        w2=rng.normal(0.0, np.sqrt(1.0 / 16.0), size=(16, 16)),
        b2=np.zeros((1, 16)),
        w3=rng.normal(0.0, np.sqrt(1.0 / 16.0), size=(16, 1)),
        b3=np.zeros((1, 1)),
    )


def parameter_count(parameters: Parameters) -> int:
    """Return the total number of trainable scalar parameters."""
    return sum(
        array.size
        for array in (
            parameters.w1,
            parameters.b1,
            parameters.w2,
            parameters.b2,
            parameters.w3,
            parameters.b3,
        )
    )


def forward(
    inputs: np.ndarray,
    parameters: Parameters,
) -> tuple[np.ndarray, ForwardCache]:
    """Run one NN-01 forward pass."""
    z1 = inputs @ parameters.w1 + parameters.b1
    a1 = sigmoid(z1)

    z2 = a1 @ parameters.w2 + parameters.b2
    a2 = sigmoid(z2)

    z3 = a2 @ parameters.w3 + parameters.b3
    predictions = sigmoid(z3)

    cache = ForwardCache(
        inputs=inputs,
        a1=a1,
        a2=a2,
        predictions=predictions,
    )

    return predictions, cache
