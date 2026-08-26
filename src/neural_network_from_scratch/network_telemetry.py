"""Current-state internal telemetry for the fixed live NN-01 network."""

from dataclasses import dataclass

import numpy as np

from neural_network_from_scratch.experiments import Experiment
from neural_network_from_scratch.network import (
    Gradients,
    Parameters,
    backward,
    forward,
)

TOPOLOGY = (2, 16, 16, 1)

_PARAMETER_SHAPES = {
    "w1": (2, 16),
    "b1": (1, 16),
    "w2": (16, 16),
    "b2": (1, 16),
    "w3": (16, 1),
    "b3": (1, 1),
}


class NetworkTelemetryError(ValueError):
    """Raised when current NN-01 state cannot form valid telemetry."""


@dataclass(frozen=True)
class LayerActivationSummary:
    """Python-owned batch statistics for one network layer."""

    mean: np.ndarray
    min: np.ndarray
    max: np.ndarray
    spread: np.ndarray

    def to_dict(self) -> dict[str, object]:
        """Return the summary as JSON-compatible numeric lists."""
        return {
            "mean": self.mean.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "spread": self.spread.tolist(),
        }


@dataclass(frozen=True)
class NetworkActivations:
    """Exact current batch activations for every visible NN-01 layer."""

    input: np.ndarray
    hidden_1: np.ndarray
    hidden_2: np.ndarray
    output: np.ndarray

    def to_dict(self) -> dict[str, object]:
        """Return activations as JSON-compatible numeric matrices."""
        return {
            "input": self.input.tolist(),
            "hidden_1": self.hidden_1.tolist(),
            "hidden_2": self.hidden_2.tolist(),
            "output": self.output.tolist(),
        }


@dataclass(frozen=True)
class NetworkActivationSummary:
    """Per-layer aggregate summaries for the current training batch."""

    input: LayerActivationSummary
    hidden_1: LayerActivationSummary
    hidden_2: LayerActivationSummary
    output: LayerActivationSummary

    def to_dict(self) -> dict[str, object]:
        """Return each layer's Python-owned aggregate statistics."""
        return {
            "input": self.input.to_dict(),
            "hidden_1": self.hidden_1.to_dict(),
            "hidden_2": self.hidden_2.to_dict(),
            "output": self.output.to_dict(),
        }


@dataclass(frozen=True)
class NetworkTelemetry:
    """Complete non-mutating current-state telemetry for NN-01."""

    topology: tuple[int, int, int, int]
    activations: NetworkActivations
    activation_summary: NetworkActivationSummary
    parameters: Parameters
    gradients: Gradients

    def to_dict(self) -> dict[str, object]:
        """Return all real numeric values in the browser contract shape."""
        return {
            "topology": list(self.topology),
            "activations": self.activations.to_dict(),
            "activation_summary": self.activation_summary.to_dict(),
            "parameters": _arrays_to_dict(self.parameters),
            "gradients": _arrays_to_dict(self.gradients),
        }


def _arrays_to_dict(
    arrays: Parameters | Gradients,
) -> dict[str, object]:
    return {
        name: getattr(arrays, name).tolist()
        for name in _PARAMETER_SHAPES
    }


def _validate_shape(
    name: str,
    values: np.ndarray,
    expected_shape: tuple[int, ...],
) -> None:
    if values.shape != expected_shape:
        raise NetworkTelemetryError(
            f"{name} must have shape {expected_shape}, got {values.shape}"
        )


def _validate_inputs_and_targets(
    experiment: Experiment,
) -> int:
    inputs = experiment.inputs

    if inputs.ndim != 2 or inputs.shape[1:] != (2,):
        raise NetworkTelemetryError(
            "inputs must have shape (samples, 2)"
        )

    sample_count = inputs.shape[0]

    if experiment.targets.shape != (sample_count, 1):
        raise NetworkTelemetryError(
            "targets must have shape (samples, 1)"
        )

    return sample_count


def _validate_parameter_shapes(
    arrays: Parameters | Gradients,
) -> None:
    for name, expected_shape in _PARAMETER_SHAPES.items():
        _validate_shape(
            name,
            getattr(arrays, name),
            expected_shape,
        )


def _summary(
    values: np.ndarray,
) -> LayerActivationSummary:
    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)

    return LayerActivationSummary(
        mean=np.mean(values, axis=0),
        min=minimum,
        max=maximum,
        spread=maximum - minimum,
    )


def probe_network_telemetry(
    experiment: Experiment,
    parameters: Parameters,
) -> NetworkTelemetry:
    """Observe the current NN-01 model without updating its parameters."""
    sample_count = _validate_inputs_and_targets(experiment)
    _validate_parameter_shapes(parameters)

    _, cache = forward(
        experiment.inputs,
        parameters,
    )

    activations = NetworkActivations(
        input=cache.inputs,
        hidden_1=cache.a1,
        hidden_2=cache.a2,
        output=cache.predictions,
    )

    for name, values, width in (
        ("inputs", activations.input, 2),
        ("a1", activations.hidden_1, 16),
        ("a2", activations.hidden_2, 16),
        ("predictions", activations.output, 1),
    ):
        _validate_shape(
            name,
            values,
            (sample_count, width),
        )

    gradients = backward(
        experiment.targets,
        parameters,
        cache,
    )
    _validate_parameter_shapes(gradients)

    return NetworkTelemetry(
        topology=TOPOLOGY,
        activations=activations,
        activation_summary=NetworkActivationSummary(
            input=_summary(activations.input),
            hidden_1=_summary(activations.hidden_1),
            hidden_2=_summary(activations.hidden_2),
            output=_summary(activations.output),
        ),
        parameters=parameters,
        gradients=gradients,
    )
