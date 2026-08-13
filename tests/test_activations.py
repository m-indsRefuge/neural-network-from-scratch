import numpy as np

from neural_network_from_scratch.activations import (
    sigmoid,
    sigmoid_derivative_from_activation,
)


def test_sigmoid_zero_is_half() -> None:
    result = sigmoid(np.array([0.0]))
    np.testing.assert_allclose(result, np.array([0.5]))


def test_sigmoid_preserves_shape() -> None:
    values = np.zeros((4, 16))
    assert sigmoid(values).shape == (4, 16)


def test_sigmoid_derivative_from_activation() -> None:
    activation = np.array([0.25, 0.5, 0.75])
    expected = activation * (1.0 - activation)
    np.testing.assert_allclose(
        sigmoid_derivative_from_activation(activation),
        expected,
    )
