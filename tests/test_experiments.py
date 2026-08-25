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


def test_linear_split_experiment_has_canonical_configuration() -> None:
    from neural_network_from_scratch.experiments import linear_split_experiment

    experiment = linear_split_experiment()

    assert isinstance(experiment, Experiment)
    assert experiment.name == "linear_split"
    assert experiment.seed == 7
    assert experiment.epochs == 500
    assert experiment.learning_rate == 1.0

    np.testing.assert_array_equal(
        experiment.inputs,
        np.array(
            [
                [0.10, 0.20],
                [0.20, 0.40],
                [0.40, 0.20],
                [0.30, 0.50],
                [0.60, 0.50],
                [0.50, 0.70],
                [0.70, 0.60],
                [0.85, 0.80],
            ]
        ),
    )

    np.testing.assert_array_equal(
        experiment.targets,
        np.array(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [1.0],
                [1.0],
                [1.0],
                [1.0],
            ]
        ),
    )


def test_linear_split_experiment_is_learnable_by_nn01() -> None:
    from neural_network_from_scratch.experiment_runner import run_experiment
    from neural_network_from_scratch.experiments import linear_split_experiment
    from neural_network_from_scratch.network import forward

    experiment = linear_split_experiment()

    parameters, losses = run_experiment(experiment)

    predictions, _ = forward(experiment.inputs, parameters)
    classifications = (predictions >= 0.5).astype(float)

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.01
    np.testing.assert_array_equal(classifications, experiment.targets)