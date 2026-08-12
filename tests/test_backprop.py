import numpy as np

from neural_network_from_scratch.network import (
    backward,
    forward,
    initialize_parameters,
    update_parameters,
)


def test_backward_gradient_shapes_match_parameters() -> None:
    parameters = initialize_parameters(seed=7)

    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    targets = np.array(
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ]
    )

    _, cache = forward(inputs, parameters)
    gradients = backward(targets, parameters, cache)

    assert gradients.w1.shape == parameters.w1.shape
    assert gradients.b1.shape == parameters.b1.shape
    assert gradients.w2.shape == parameters.w2.shape
    assert gradients.b2.shape == parameters.b2.shape
    assert gradients.w3.shape == parameters.w3.shape
    assert gradients.b3.shape == parameters.b3.shape



def test_update_parameters_applies_gradient_descent_without_mutation() -> None:
    parameters = initialize_parameters(seed=7)

    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    targets = np.array(
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ]
    )

    original = {
        name: getattr(parameters, name).copy()
        for name in ("w1", "b1", "w2", "b2", "w3", "b3")
    }

    _, cache = forward(inputs, parameters)
    gradients = backward(targets, parameters, cache)

    learning_rate = 0.5
    updated = update_parameters(parameters, gradients, learning_rate)

    for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
        expected = original[name] - learning_rate * getattr(gradients, name)

        np.testing.assert_allclose(getattr(updated, name), expected)
        np.testing.assert_array_equal(getattr(parameters, name), original[name])
