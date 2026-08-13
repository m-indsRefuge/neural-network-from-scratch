"""Independent numerical gradient checking for NN-01."""

from dataclasses import dataclass

import numpy as np

from neural_network_from_scratch.losses import binary_cross_entropy
from neural_network_from_scratch.network import (
    Gradients,
    Parameters,
    backward,
    forward,
)

_PARAMETER_NAMES = ("w1", "b1", "w2", "b2", "w3", "b3")


@dataclass(frozen=True)
class GradientCheckResult:
    """Error measurements comparing analytical and numerical gradients."""

    relative_error: float
    max_absolute_error: float


def _copy_parameters(parameters: Parameters) -> Parameters:
    return Parameters(
        w1=parameters.w1.copy(),
        b1=parameters.b1.copy(),
        w2=parameters.w2.copy(),
        b2=parameters.b2.copy(),
        w3=parameters.w3.copy(),
        b3=parameters.b3.copy(),
    )


def _flatten_gradients(gradients: Gradients) -> np.ndarray:
    return np.concatenate(
        [getattr(gradients, name).ravel() for name in _PARAMETER_NAMES]
    )


def _loss_for_parameters(
    inputs: np.ndarray,
    targets: np.ndarray,
    parameters: Parameters,
) -> float:
    predictions, _ = forward(inputs, parameters)
    return binary_cross_entropy(targets, predictions)


def gradient_check(
    inputs: np.ndarray,
    targets: np.ndarray,
    parameters: Parameters,
    epsilon: float = 1e-5,
) -> GradientCheckResult:
    """Compare handwritten gradients with central finite differences."""
    _, cache = forward(inputs, parameters)
    analytical = backward(targets, parameters, cache)
    analytical_flat = _flatten_gradients(analytical)

    numerical_parts: list[np.ndarray] = []

    for name in _PARAMETER_NAMES:
        parameter_array = getattr(parameters, name)
        numerical = np.zeros_like(parameter_array, dtype=float)

        for index in np.ndindex(parameter_array.shape):
            plus = _copy_parameters(parameters)
            minus = _copy_parameters(parameters)

            getattr(plus, name)[index] += epsilon
            getattr(minus, name)[index] -= epsilon

            loss_plus = _loss_for_parameters(inputs, targets, plus)
            loss_minus = _loss_for_parameters(inputs, targets, minus)

            numerical[index] = (loss_plus - loss_minus) / (2.0 * epsilon)

        numerical_parts.append(numerical.ravel())

    numerical_flat = np.concatenate(numerical_parts)
    difference = analytical_flat - numerical_flat

    max_absolute_error = float(np.max(np.abs(difference)))

    denominator = (
        np.linalg.norm(analytical_flat)
        + np.linalg.norm(numerical_flat)
        + 1e-12
    )
    relative_error = float(np.linalg.norm(difference) / denominator)

    return GradientCheckResult(
        relative_error=relative_error,
        max_absolute_error=max_absolute_error,
    )
