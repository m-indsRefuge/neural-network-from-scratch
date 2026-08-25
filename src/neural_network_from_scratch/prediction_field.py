"""Probability-field probing for two-dimensional NN-01 inputs."""

from dataclasses import dataclass

import numpy as np

from neural_network_from_scratch.network import Parameters, forward


@dataclass(frozen=True)
class PredictionField:
    """Predicted probabilities sampled across a rectangular 2D input grid."""

    x_values: np.ndarray
    y_values: np.ndarray
    probabilities: np.ndarray


def probe_prediction_field(
    parameters: Parameters,
    *,
    resolution: int,
) -> PredictionField:
    """Evaluate NN-01 across an evenly spaced unit-square input grid."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")

    x_values = np.linspace(0.0, 1.0, resolution)
    y_values = np.linspace(0.0, 1.0, resolution)

    grid_x, grid_y = np.meshgrid(
        x_values,
        y_values,
        indexing="xy",
    )

    inputs = np.column_stack(
        (
            grid_x.ravel(),
            grid_y.ravel(),
        )
    )

    predictions, _ = forward(inputs, parameters)

    probabilities = predictions.reshape(
        resolution,
        resolution,
    )

    return PredictionField(
        x_values=x_values,
        y_values=y_values,
        probabilities=probabilities,
    )