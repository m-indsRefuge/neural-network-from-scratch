import numpy as np

from neural_network_from_scratch.network import backward, forward
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


def test_training_observer_receives_consistent_snapshot_per_epoch() -> None:
    inputs, targets = _xor_data()
    snapshots = []

    _, losses = train(
        inputs,
        targets,
        seed=7,
        epochs=3,
        learning_rate=1.0,
        observer=snapshots.append,
    )

    assert len(snapshots) == 3
    assert [snapshot.epoch for snapshot in snapshots] == [1, 2, 3]
    assert [snapshot.loss for snapshot in snapshots] == losses

    for snapshot in snapshots:
        predictions, expected_cache = forward(inputs, snapshot.parameters)

        np.testing.assert_array_equal(
            snapshot.cache.inputs,
            expected_cache.inputs,
        )
        np.testing.assert_array_equal(
            snapshot.cache.a1,
            expected_cache.a1,
        )
        np.testing.assert_array_equal(
            snapshot.cache.a2,
            expected_cache.a2,
        )
        np.testing.assert_array_equal(
            snapshot.cache.predictions,
            predictions,
        )

        expected_gradients = backward(
            targets,
            snapshot.parameters,
            snapshot.cache,
        )

        for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
            np.testing.assert_array_equal(
                getattr(snapshot.gradients, name),
                getattr(expected_gradients, name),
            )


def test_observer_does_not_change_training_result() -> None:
    inputs, targets = _xor_data()

    parameters_without_observer, losses_without_observer = train(
        inputs,
        targets,
        seed=7,
        epochs=5,
        learning_rate=1.0,
    )

    snapshots = []

    parameters_with_observer, losses_with_observer = train(
        inputs,
        targets,
        seed=7,
        epochs=5,
        learning_rate=1.0,
        observer=snapshots.append,
    )

    np.testing.assert_array_equal(
        losses_with_observer,
        losses_without_observer,
    )

    for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
        np.testing.assert_array_equal(
            getattr(parameters_with_observer, name),
            getattr(parameters_without_observer, name),
        )