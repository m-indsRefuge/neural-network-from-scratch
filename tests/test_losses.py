import numpy as np
import pytest

from neural_network_from_scratch.losses import binary_cross_entropy


def test_binary_cross_entropy_known_value() -> None:
    targets = np.array([[1.0], [0.0]])
    predictions = np.array([[0.8], [0.2]])

    expected = -np.log(0.8)

    assert binary_cross_entropy(targets, predictions) == pytest.approx(expected)


def test_binary_cross_entropy_perfect_predictions_are_near_zero() -> None:
    targets = np.array([[1.0], [0.0]])
    predictions = np.array([[1.0], [0.0]])

    assert binary_cross_entropy(targets, predictions) < 1e-10
