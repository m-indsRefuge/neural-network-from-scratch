import numpy as np
import pytest

from neural_network_from_scratch.experiment_runner import run_experiment
from neural_network_from_scratch.experiments import linear_split_experiment
from neural_network_from_scratch.network import forward
from neural_network_from_scratch.prediction_field import probe_prediction_field


def test_prediction_field_has_expected_grid_shape_and_coordinates() -> None:
    experiment = linear_split_experiment()
    parameters, _ = run_experiment(experiment)

    field = probe_prediction_field(
        parameters,
        resolution=5,
    )

    np.testing.assert_array_equal(
        field.x_values,
        np.linspace(0.0, 1.0, 5),
    )
    np.testing.assert_array_equal(
        field.y_values,
        np.linspace(0.0, 1.0, 5),
    )

    assert field.probabilities.shape == (5, 5)


def test_prediction_field_matches_direct_network_predictions() -> None:
    experiment = linear_split_experiment()
    parameters, _ = run_experiment(experiment)

    field = probe_prediction_field(
        parameters,
        resolution=3,
    )

    for row, y_value in enumerate(field.y_values):
        for column, x_value in enumerate(field.x_values):
            point = np.array([[x_value, y_value]])

            prediction, _ = forward(point, parameters)

            np.testing.assert_allclose(
                field.probabilities[row, column],
                prediction[0, 0],
                rtol=1e-12,
                atol=1e-15,
            )


def test_prediction_field_contains_both_class_regions() -> None:
    experiment = linear_split_experiment()
    parameters, _ = run_experiment(experiment)

    field = probe_prediction_field(
        parameters,
        resolution=21,
    )

    assert np.any(field.probabilities < 0.5)
    assert np.any(field.probabilities >= 0.5)


def test_prediction_field_rejects_resolution_below_two() -> None:
    experiment = linear_split_experiment()
    parameters, _ = run_experiment(experiment)

    with pytest.raises(ValueError, match="resolution must be at least 2"):
        probe_prediction_field(
            parameters,
            resolution=1,
        )