from dataclasses import replace

import numpy as np

from neural_network_from_scratch.experiment_runner import run_experiment
from neural_network_from_scratch.experiments import xor_experiment
from neural_network_from_scratch.training import train


def test_run_experiment_matches_direct_training() -> None:
    experiment = replace(xor_experiment(), epochs=5)

    experiment_parameters, experiment_losses = run_experiment(experiment)

    direct_parameters, direct_losses = train(
        experiment.inputs,
        experiment.targets,
        seed=experiment.seed,
        epochs=experiment.epochs,
        learning_rate=experiment.learning_rate,
    )

    np.testing.assert_array_equal(experiment_losses, direct_losses)

    for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
        np.testing.assert_array_equal(
            getattr(experiment_parameters, name),
            getattr(direct_parameters, name),
        )


def test_run_experiment_forwards_training_observer() -> None:
    experiment = replace(xor_experiment(), epochs=3)
    snapshots = []

    _, losses = run_experiment(
        experiment,
        observer=snapshots.append,
    )

    assert len(snapshots) == 3
    assert [snapshot.epoch for snapshot in snapshots] == [1, 2, 3]
    assert [snapshot.loss for snapshot in snapshots] == losses