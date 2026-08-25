from dataclasses import fields, replace

import numpy as np
import pytest

from neural_network_from_scratch.experiment_runner import run_experiment
from neural_network_from_scratch.experiments import linear_split_experiment
from neural_network_from_scratch.live_training import LiveTrainingSession
from neural_network_from_scratch.network import Parameters


def _assert_parameters_equal(
    actual: Parameters,
    expected: Parameters,
) -> None:
    for field in fields(Parameters):
        np.testing.assert_array_equal(
            getattr(actual, field.name),
            getattr(expected, field.name),
        )


def test_live_training_session_starts_before_first_epoch() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=3,
    )

    session = LiveTrainingSession(experiment)

    assert session.epoch == 0
    assert session.losses == []
    assert session.is_complete is False


def test_step_advances_exactly_one_epoch() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=3,
    )

    session = LiveTrainingSession(experiment)

    snapshot = session.step()

    assert snapshot.epoch == 1
    assert session.epoch == 1
    assert session.losses == [snapshot.loss]
    assert session.is_complete is False


def test_step_snapshot_matches_existing_training_observer_semantics() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=1,
    )

    observed = []

    run_experiment(
        experiment,
        observer=observed.append,
    )

    session = LiveTrainingSession(experiment)
    snapshot = session.step()

    expected = observed[0]

    assert snapshot.epoch == expected.epoch
    assert snapshot.loss == expected.loss

    _assert_parameters_equal(
        snapshot.parameters,
        expected.parameters,
    )

    for field in fields(expected.gradients):
        np.testing.assert_array_equal(
            getattr(snapshot.gradients, field.name),
            getattr(expected.gradients, field.name),
        )

    np.testing.assert_array_equal(
        snapshot.cache.inputs,
        expected.cache.inputs,
    )
    np.testing.assert_array_equal(
        snapshot.cache.a1,
        expected.cache.a1,
    )
    np.testing.assert_array_equal(
        snapshot.cache.a2,
        expected.cache.a2,
    )
    np.testing.assert_array_equal(
        snapshot.cache.predictions,
        expected.cache.predictions,
    )


def test_repeated_steps_exactly_match_normal_training() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=4,
    )

    expected_parameters, expected_losses = run_experiment(experiment)

    session = LiveTrainingSession(experiment)

    for _ in range(experiment.epochs):
        session.step()

    assert session.epoch == experiment.epochs
    assert session.losses == expected_losses
    assert session.is_complete is True

    _assert_parameters_equal(
        session.parameters,
        expected_parameters,
    )


def test_step_rejects_training_beyond_configured_epochs() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=2,
    )

    session = LiveTrainingSession(experiment)

    session.step()
    session.step()

    with pytest.raises(
        RuntimeError,
        match="training session is complete",
    ):
        session.step()

def test_training_can_pause_between_steps_and_resume_exactly() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=4,
    )

    expected_parameters, expected_losses = run_experiment(experiment)

    session = LiveTrainingSession(experiment)

    session.step()

    paused_epoch = session.epoch
    paused_losses = session.losses.copy()
    paused_parameters = session.parameters

    assert session.epoch == paused_epoch
    assert session.losses == paused_losses
    _assert_parameters_equal(
        session.parameters,
        paused_parameters,
    )

    while not session.is_complete:
        session.step()

    assert session.epoch == experiment.epochs
    assert session.losses == expected_losses

    _assert_parameters_equal(
        session.parameters,
        expected_parameters,
    )
