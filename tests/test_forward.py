import numpy as np

from neural_network_from_scratch.network import (
    initialize_parameters,
    parameter_count,
)


def test_parameter_shapes_match_nn01_architecture() -> None:
    parameters = initialize_parameters(seed=7)

    assert parameters.w1.shape == (2, 16)
    assert parameters.b1.shape == (1, 16)
    assert parameters.w2.shape == (16, 16)
    assert parameters.b2.shape == (1, 16)
    assert parameters.w3.shape == (16, 1)
    assert parameters.b3.shape == (1, 1)


def test_parameter_count_is_exactly_337() -> None:
    parameters = initialize_parameters(seed=7)

    assert parameter_count(parameters) == 337


def test_initialization_is_reproducible_with_fixed_seed() -> None:
    first = initialize_parameters(seed=7)
    second = initialize_parameters(seed=7)

    np.testing.assert_array_equal(first.w1, second.w1)
    np.testing.assert_array_equal(first.b1, second.b1)
    np.testing.assert_array_equal(first.w2, second.w2)
    np.testing.assert_array_equal(first.b2, second.b2)
    np.testing.assert_array_equal(first.w3, second.w3)
    np.testing.assert_array_equal(first.b3, second.b3)


def test_biases_start_at_zero() -> None:
    parameters = initialize_parameters(seed=7)

    np.testing.assert_array_equal(parameters.b1, np.zeros((1, 16)))
    np.testing.assert_array_equal(parameters.b2, np.zeros((1, 16)))
    np.testing.assert_array_equal(parameters.b3, np.zeros((1, 1)))


from neural_network_from_scratch.network import forward


def test_forward_shapes_match_nn01_architecture() -> None:
    parameters = initialize_parameters(seed=7)
    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    predictions, cache = forward(inputs, parameters)

    assert cache.inputs.shape == (4, 2)
    assert cache.a1.shape == (4, 16)
    assert cache.a2.shape == (4, 16)
    assert predictions.shape == (4, 1)


def test_forward_predictions_are_probabilities() -> None:
    parameters = initialize_parameters(seed=7)
    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    predictions, _ = forward(inputs, parameters)

    assert np.all(predictions > 0.0)
    assert np.all(predictions < 1.0)
