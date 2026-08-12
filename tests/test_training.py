import numpy as np

from neural_network_from_scratch.network import forward
from neural_network_from_scratch.training import train


def _xor_data() -> tuple[np.ndarray, np.ndarray]:
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

    return inputs, targets


def test_training_reduces_xor_loss_substantially() -> None:
    inputs, targets = _xor_data()

    _, losses = train(
        inputs,
        targets,
        seed=7,
        epochs=10_000,
        learning_rate=1.0,
    )

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.01


def test_training_classifies_all_xor_inputs_correctly() -> None:
    inputs, targets = _xor_data()

    parameters, _ = train(
        inputs,
        targets,
        seed=7,
        epochs=10_000,
        learning_rate=1.0,
    )

    predictions, _ = forward(inputs, parameters)
    classifications = (predictions >= 0.5).astype(float)

    np.testing.assert_array_equal(classifications, targets)


def test_training_is_deterministic_for_fixed_seed() -> None:
    inputs, targets = _xor_data()

    parameters_a, losses_a = train(
        inputs,
        targets,
        seed=7,
        epochs=10_000,
        learning_rate=1.0,
    )

    parameters_b, losses_b = train(
        inputs,
        targets,
        seed=7,
        epochs=10_000,
        learning_rate=1.0,
    )

    np.testing.assert_array_equal(losses_a, losses_b)

    for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
        np.testing.assert_array_equal(
            getattr(parameters_a, name),
            getattr(parameters_b, name),
        )
