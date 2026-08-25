import base64
from dataclasses import fields, replace

import numpy as np

from neural_network_from_scratch.decision_boundary import extract_decision_boundary
from neural_network_from_scratch.experiments import linear_split_experiment
from neural_network_from_scratch.live_training import LiveTrainingSession
from neural_network_from_scratch.live_view import (
    build_live_view_state,
    encode_probability_field,
)
from neural_network_from_scratch.losses import binary_cross_entropy
from neural_network_from_scratch.network import Parameters, backward, forward
from neural_network_from_scratch.prediction_field import probe_prediction_field


def test_initial_live_view_describes_epoch_zero_model() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=2,
    )
    session = LiveTrainingSession(experiment)

    state = build_live_view_state(
        session,
        experiment,
        resolution=5,
    )

    predictions, _ = forward(
        experiment.inputs,
        session.parameters,
    )

    expected_loss = binary_cross_entropy(
        experiment.targets,
        predictions,
    )

    assert state.epoch == 0
    assert state.loss == expected_loss
    assert state.is_complete is False
    assert state.resolution == 5


def test_live_view_after_step_describes_updated_parameters() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=2,
    )
    session = LiveTrainingSession(experiment)

    snapshot = session.step()

    state = build_live_view_state(
        session,
        experiment,
        resolution=5,
    )

    predictions, _ = forward(
        experiment.inputs,
        session.parameters,
    )

    expected_loss = binary_cross_entropy(
        experiment.targets,
        predictions,
    )

    assert snapshot.epoch == 1
    assert state.epoch == 1
    assert state.loss == expected_loss

    # TrainingSnapshot describes the coherent pre-update state.
    # LiveViewState describes the model after that update completed.
    assert state.loss != snapshot.loss


def test_live_view_network_telemetry_describes_current_post_update_parameters() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=2,
    )
    session = LiveTrainingSession(experiment)

    snapshot = session.step()

    state = build_live_view_state(
        session,
        experiment,
        resolution=5,
    )
    predictions, cache = forward(
        experiment.inputs,
        session.parameters,
    )
    gradients = backward(
        experiment.targets,
        session.parameters,
        cache,
    )

    assert state.network is not None
    assert state.network_error is None
    assert state.network.topology == (2, 16, 16, 1)
    assert state.loss == binary_cross_entropy(
        experiment.targets,
        predictions,
    )
    assert state.loss == state.loss_history[-1]
    assert state.loss != snapshot.loss

    np.testing.assert_allclose(
        state.network.activations.output,
        predictions,
    )

    for parameter in fields(Parameters):
        np.testing.assert_allclose(
            getattr(state.network.parameters, parameter.name),
            getattr(session.parameters, parameter.name),
        )
        np.testing.assert_allclose(
            getattr(state.network.gradients, parameter.name),
            getattr(gradients, parameter.name),
        )


def test_live_view_network_telemetry_reads_do_not_mutate_training() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=2,
    )
    session = LiveTrainingSession(experiment)

    before_epoch = session.epoch
    before_losses = session.losses.copy()
    before_history = session.loss_history
    before_parameters = {
        parameter.name: getattr(session.parameters, parameter.name).copy()
        for parameter in fields(Parameters)
    }

    first = build_live_view_state(
        session,
        experiment,
        resolution=5,
    )
    second = build_live_view_state(
        session,
        experiment,
        resolution=5,
    )

    assert first.to_dict() == second.to_dict()
    assert session.epoch == before_epoch
    assert session.losses == before_losses
    assert session.loss_history == before_history

    for parameter in fields(Parameters):
        np.testing.assert_array_equal(
            getattr(session.parameters, parameter.name),
            before_parameters[parameter.name],
        )


def test_probability_field_encoding_matches_uint8_quantization() -> None:
    experiment = linear_split_experiment()
    session = LiveTrainingSession(experiment)

    field = probe_prediction_field(
        session.parameters,
        resolution=5,
    )

    encoded = encode_probability_field(
        field.probabilities,
    )

    decoded = np.frombuffer(
        base64.b64decode(encoded),
        dtype=np.uint8,
    )

    expected = np.rint(
        np.clip(
            field.probabilities,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(np.uint8).ravel()

    np.testing.assert_array_equal(
        decoded,
        expected,
    )

    assert decoded.size == 25


def test_probability_field_encoding_is_deterministic() -> None:
    experiment = linear_split_experiment()
    session = LiveTrainingSession(experiment)

    field = probe_prediction_field(
        session.parameters,
        resolution=5,
    )

    first = encode_probability_field(
        field.probabilities,
    )
    second = encode_probability_field(
        field.probabilities,
    )

    assert first == second


def test_live_view_reports_completion_and_serializes() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=1,
    )
    session = LiveTrainingSession(experiment)

    session.step()

    state = build_live_view_state(
        session,
        experiment,
        resolution=5,
    )

    payload = state.to_dict()

    assert state.epoch == 1
    assert state.is_complete is True
    assert state.network is not None
    assert state.network_error is None

    assert payload == {
        "epoch": state.epoch,
        "loss": state.loss,
        "loss_history": state.loss_history,
        "is_complete": state.is_complete,
        "resolution": state.resolution,
        "field": state.field,
        "decision_boundary": state.decision_boundary,
        "network": state.network.to_dict(),
        "network_error": None,
    }


def test_live_view_contains_server_owned_loss_history() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=3,
    )
    session = LiveTrainingSession(experiment)

    session.step()
    session.step()

    state = build_live_view_state(
        session,
        experiment,
        resolution=5,
    )

    assert state.loss_history == session.loss_history
    assert state.loss == session.loss_history[-1]


def test_live_view_boundary_uses_full_precision_prediction_field() -> None:
    experiment = linear_split_experiment()
    session = LiveTrainingSession(experiment)

    field = probe_prediction_field(
        session.parameters,
        resolution=5,
    )
    expected = extract_decision_boundary(field)

    state = build_live_view_state(
        session,
        experiment,
        resolution=5,
    )

    assert state.decision_boundary == expected
