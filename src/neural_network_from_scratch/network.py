"""Core NN-01 parameter structures and network operations."""

from dataclasses import dataclass

import numpy as np

from neural_network_from_scratch.activations import (
    sigmoid,
    sigmoid_derivative_from_activation,
)


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



@dataclass(frozen=True)
class Gradients:
    """Analytical gradients for every NN-01 trainable parameter."""

    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    w3: np.ndarray
    b3: np.ndarray


def backward(
    targets: np.ndarray,
    parameters: Parameters,
    cache: ForwardCache,
) -> Gradients:
    """Propagate prediction error backward through NN-01."""
    sample_count = targets.shape[0]

    dz3 = (cache.predictions - targets) / sample_count
    dw3 = cache.a2.T @ dz3
    db3 = np.sum(dz3, axis=0, keepdims=True)

    da2 = dz3 @ parameters.w3.T
    dz2 = da2 * sigmoid_derivative_from_activation(cache.a2)
    dw2 = cache.a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ parameters.w2.T
    dz1 = da1 * sigmoid_derivative_from_activation(cache.a1)
    dw1 = cache.inputs.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    return Gradients(
        w1=dw1,
        b1=db1,
        w2=dw2,
        b2=db2,
        w3=dw3,
        b3=db3,
    )


def update_parameters(
    parameters: Parameters,
    gradients: Gradients,
    learning_rate: float,
) -> Parameters:
    """Return new parameters after one gradient-descent step."""
    return Parameters(
        w1=parameters.w1 - learning_rate * gradients.w1,
        b1=parameters.b1 - learning_rate * gradients.b1,
        w2=parameters.w2 - learning_rate * gradients.w2,
        b2=parameters.b2 - learning_rate * gradients.b2,
        w3=parameters.w3 - learning_rate * gradients.w3,
        b3=parameters.b3 - learning_rate * gradients.b3,
    )
