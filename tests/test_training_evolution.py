from dataclasses import replace

import numpy as np
import pytest

from neural_network_from_scratch.experiment_runner import run_experiment
from neural_network_from_scratch.experiments import linear_split_experiment
from neural_network_from_scratch.training_evolution import capture_training_evolution


def test_capture_training_evolution_keeps_only_requested_epochs() -> None:
    experiment = replace(linear_split_experiment(), epochs=10)

    evolution = capture_training_evolution(
        experiment,
        epochs=(1, 3, 7, 10),
    )

    assert [snapshot.epoch for snapshot in evolution.snapshots] == [
        1,
        3,
        7,
        10,
    ]


def test_capture_training_evolution_preserves_training_result() -> None:
    experiment = replace(linear_split_experiment(), epochs=10)

    evolution = capture_training_evolution(
        experiment,
        epochs=(1, 5, 10),
    )

    direct_parameters, direct_losses = run_experiment(experiment)

    np.testing.assert_array_equal(
        evolution.losses,
        direct_losses,
    )

    for name in ("w1", "b1", "w2", "b2", "w3", "b3"):
        np.testing.assert_array_equal(
            getattr(evolution.parameters, name),
            getattr(direct_parameters, name),
        )


def test_capture_training_evolution_snapshot_losses_match_training_history() -> None:
    experiment = replace(linear_split_experiment(), epochs=10)

    evolution = capture_training_evolution(
        experiment,
        epochs=(1, 4, 10),
    )

    for snapshot in evolution.snapshots:
        assert snapshot.loss == evolution.losses[snapshot.epoch - 1]


def test_capture_training_evolution_rejects_invalid_epochs() -> None:
    experiment = replace(linear_split_experiment(), epochs=10)

    with pytest.raises(
        ValueError,
        match="capture epochs must be between 1 and experiment epochs",
    ):
        capture_training_evolution(
            experiment,
            epochs=(0, 5),
        )

    with pytest.raises(
        ValueError,
        match="capture epochs must be between 1 and experiment epochs",
    ):
        capture_training_evolution(
            experiment,
            epochs=(5, 11),
        )


def test_capture_training_evolution_rejects_duplicate_epochs() -> None:
    experiment = replace(linear_split_experiment(), epochs=10)

    with pytest.raises(
        ValueError,
        match="capture epochs must be unique",
    ):
        capture_training_evolution(
            experiment,
            epochs=(1, 5, 5, 10),
        )