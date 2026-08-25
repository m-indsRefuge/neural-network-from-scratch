import numpy as np

from neural_network_from_scratch.experiments import Experiment, xor_experiment


def test_xor_experiment_has_canonical_configuration() -> None:
    experiment = xor_experiment()

    assert isinstance(experiment, Experiment)
    assert experiment.name == "xor"
    assert experiment.seed == 7
    assert experiment.epochs == 10_000
    assert experiment.learning_rate == 1.0

    np.testing.assert_array_equal(
        experiment.inputs,
        np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ]
        ),
    )

    np.testing.assert_array_equal(
        experiment.targets,
        np.array(
            [
                [0.0],
                [1.0],
                [1.0],
                [0.0],
            ]
        ),
    )


def test_xor_experiment_returns_independent_dataset_arrays() -> None:
    experiment_a = xor_experiment()
    experiment_b = xor_experiment()

    assert experiment_a.inputs is not experiment_b.inputs
    assert experiment_a.targets is not experiment_b.targets

    experiment_a.inputs[0, 0] = 99.0
    experiment_a.targets[0, 0] = 99.0

    assert experiment_b.inputs[0, 0] == 0.0
    assert experiment_b.targets[0, 0] == 0.0