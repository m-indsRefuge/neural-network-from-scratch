from dataclasses import fields, replace

import numpy as np
import pytest

from neural_network_from_scratch.experiments import linear_split_experiment
from neural_network_from_scratch.network import (
    Parameters,
    backward,
    forward,
    initialize_parameters,
)
from neural_network_from_scratch.network_telemetry import (
    NetworkTelemetryError,
    probe_network_telemetry,
)


def test_current_probe_exposes_the_fixed_complete_network() -> None:
    experiment = linear_split_experiment()
    parameters = initialize_parameters(experiment.seed)

    telemetry = probe_network_telemetry(experiment, parameters)

    assert telemetry.topology == (2, 16, 16, 1)

    assert telemetry.activations.input.shape == (8, 2)
    assert telemetry.activations.hidden_1.shape == (8, 16)
    assert telemetry.activations.hidden_2.shape == (8, 16)
    assert telemetry.activations.output.shape == (8, 1)

    for parameter in fields(Parameters):
        assert getattr(telemetry.parameters, parameter.name).shape == getattr(
            parameters,
            parameter.name,
        ).shape
        assert getattr(telemetry.gradients, parameter.name).shape == getattr(
            parameters,
            parameter.name,
        ).shape


def test_current_probe_matches_handwritten_forward_and_backward_results() -> None:
    experiment = linear_split_experiment()
    parameters = initialize_parameters(experiment.seed)

    telemetry = probe_network_telemetry(experiment, parameters)
    predictions, cache = forward(experiment.inputs, parameters)
    gradients = backward(experiment.targets, parameters, cache)

    np.testing.assert_allclose(telemetry.activations.input, experiment.inputs)
    np.testing.assert_allclose(telemetry.activations.hidden_1, cache.a1)
    np.testing.assert_allclose(telemetry.activations.hidden_2, cache.a2)
    np.testing.assert_allclose(telemetry.activations.output, predictions)

    for parameter in fields(Parameters):
        np.testing.assert_allclose(
            getattr(telemetry.parameters, parameter.name),
            getattr(parameters, parameter.name),
        )
        np.testing.assert_allclose(
            getattr(telemetry.gradients, parameter.name),
            getattr(gradients, parameter.name),
        )


def test_current_probe_aggregate_statistics_match_numpy_for_every_layer() -> None:
    experiment = linear_split_experiment()
    telemetry = probe_network_telemetry(
        experiment,
        initialize_parameters(experiment.seed),
    )

    layers = {
        "input": telemetry.activations.input,
        "hidden_1": telemetry.activations.hidden_1,
        "hidden_2": telemetry.activations.hidden_2,
        "output": telemetry.activations.output,
    }

    for layer_name, values in layers.items():
        summary = getattr(telemetry.activation_summary, layer_name)
        expected_min = np.min(values, axis=0)
        expected_max = np.max(values, axis=0)

        np.testing.assert_allclose(summary.mean, np.mean(values, axis=0))
        np.testing.assert_allclose(summary.min, expected_min)
        np.testing.assert_allclose(summary.max, expected_max)
        np.testing.assert_allclose(summary.spread, expected_max - expected_min)

        assert summary.mean.shape == (values.shape[1],)


def test_current_probe_serializes_all_real_signed_matrices_deterministically() -> None:
    experiment = linear_split_experiment()
    parameters = initialize_parameters(experiment.seed)

    first = probe_network_telemetry(experiment, parameters).to_dict()
    second = probe_network_telemetry(experiment, parameters).to_dict()

    assert first == second
    assert first["topology"] == [2, 16, 16, 1]

    _, cache = forward(experiment.inputs, parameters)
    gradients = backward(experiment.targets, parameters, cache)

    for parameter in fields(Parameters):
        name = parameter.name

        assert first["parameters"][name] == getattr(parameters, name).tolist()
        assert first["gradients"][name] == getattr(gradients, name).tolist()

    flattened_weights = np.concatenate(
        (
            parameters.w1.ravel(),
            parameters.w2.ravel(),
            parameters.w3.ravel(),
        )
    )

    assert np.any(flattened_weights < 0.0)
    assert np.any(flattened_weights > 0.0)


def test_current_probe_rejects_malformed_parameter_shape() -> None:
    experiment = linear_split_experiment()
    parameters = initialize_parameters(experiment.seed)
    malformed = replace(parameters, w1=np.zeros((3, 16)))

    with pytest.raises(
        NetworkTelemetryError,
        match=r"w1 must have shape \(2, 16\)",
    ):
        probe_network_telemetry(experiment, malformed)


def test_current_probe_rejects_malformed_input_shape() -> None:
    experiment = linear_split_experiment()
    malformed = replace(experiment, inputs=np.zeros((8, 3)))

    with pytest.raises(
        NetworkTelemetryError,
        match=r"inputs must have shape \(samples, 2\)",
    ):
        probe_network_telemetry(
            malformed,
            initialize_parameters(malformed.seed),
        )
