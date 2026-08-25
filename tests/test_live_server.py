import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields, replace
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from neural_network_from_scratch.experiment_runner import run_experiment
from neural_network_from_scratch.experiments import linear_split_experiment
from neural_network_from_scratch.live_server import (
    LiveObservatoryServer,
    create_live_server,
)
from neural_network_from_scratch.network import Parameters, backward, forward


@contextmanager
def _running_server(
    *,
    epochs: int = 4,
    resolution: int = 5,
) -> Iterator[tuple[LiveObservatoryServer, str]]:
    experiment = replace(
        linear_split_experiment(),
        epochs=epochs,
    )

    server = create_live_server(
        experiment=experiment,
        port=0,
        resolution=resolution,
    )

    thread = Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    host, port = server.server_address[:2]

    try:
        yield server, f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _read_json(url: str) -> dict:
    with urlopen(url) as response:
        return json.loads(response.read())


def _post_json(url: str) -> dict:
    request = Request(
        url,
        method="POST",
    )

    with urlopen(request) as response:
        return json.loads(response.read())


def _assert_parameters_equal(
    actual: Parameters,
    expected: Parameters,
) -> None:
    for field in fields(Parameters):
        np.testing.assert_array_equal(
            getattr(actual, field.name),
            getattr(expected, field.name),
        )


def _copy_parameters(
    parameters: Parameters,
) -> Parameters:
    return Parameters(
        **{
            parameter.name: getattr(
                parameters,
                parameter.name,
            ).copy()
            for parameter in fields(Parameters)
        }
    )


def test_server_binds_only_to_loopback() -> None:
    server = create_live_server(
        port=0,
        resolution=5,
    )

    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()


def test_get_root_serves_live_observatory() -> None:
    with _running_server() as (_, base_url):
        with urlopen(f"{base_url}/") as response:
            html = response.read().decode("utf-8")

        assert response.status == 200
        assert 'id="belief-canvas"' in html
        assert 'id="step-button"' in html
        assert 'id="train-button"' in html
        assert 'id="pause-button"' in html


def test_get_state_does_not_advance_training() -> None:
    with _running_server() as (server, base_url):
        before = server.session.epoch

        state = _read_json(
            f"{base_url}/api/state"
        )

        assert state["epoch"] == 0
        assert state["resolution"] == 5
        assert server.session.epoch == before


def test_get_state_returns_deterministic_current_network_without_mutation() -> None:
    with _running_server(
        epochs=3,
        resolution=5,
    ) as (server, base_url):
        before_epoch = server.session.epoch
        before_history = server.session.loss_history
        before_parameters = _copy_parameters(
            server.session.parameters,
        )

        first = _read_json(f"{base_url}/api/state")
        second = _read_json(f"{base_url}/api/state")

        assert first["network_error"] is None
        assert first["network"]["topology"] == [2, 16, 16, 1]
        assert first["network"] == second["network"]
        assert server.session.epoch == before_epoch
        assert server.session.loss_history == before_history
        _assert_parameters_equal(
            server.session.parameters,
            before_parameters,
        )


def test_post_step_network_telemetry_uses_current_parameters() -> None:
    with _running_server(
        epochs=3,
        resolution=5,
    ) as (server, base_url):
        state = _post_json(f"{base_url}/api/step")

        predictions, cache = forward(
            server.experiment.inputs,
            server.session.parameters,
        )
        gradients = backward(
            server.experiment.targets,
            server.session.parameters,
            cache,
        )

        assert state["epoch"] == server.session.epoch == 1
        assert state["network_error"] is None
        np.testing.assert_allclose(
            state["network"]["activations"]["output"],
            predictions,
        )

        for parameter in fields(Parameters):
            np.testing.assert_allclose(
                state["network"]["parameters"][parameter.name],
                getattr(server.session.parameters, parameter.name),
            )
            np.testing.assert_allclose(
                state["network"]["gradients"][parameter.name],
                getattr(gradients, parameter.name),
            )


def test_post_step_advances_exactly_once() -> None:
    with _running_server() as (server, base_url):
        state = _post_json(
            f"{base_url}/api/step"
        )

        assert state["epoch"] == 1
        assert server.session.epoch == 1


def test_step_requests_advance_sequentially() -> None:
    with _running_server() as (_, base_url):
        first = _post_json(
            f"{base_url}/api/step"
        )
        second = _post_json(
            f"{base_url}/api/step"
        )

        assert first["epoch"] == 1
        assert second["epoch"] == 2


def test_final_step_reports_completion() -> None:
    with _running_server(
        epochs=1,
    ) as (server, base_url):
        state = _post_json(
            f"{base_url}/api/step"
        )

        assert state["epoch"] == 1
        assert state["is_complete"] is True
        assert server.session.is_complete is True


def test_step_after_completion_returns_409_without_mutation() -> None:
    with _running_server(
        epochs=1,
    ) as (server, base_url):
        completed = _post_json(
            f"{base_url}/api/step"
        )

        assert completed["is_complete"] is True
        assert server.session.epoch == 1

        request = Request(
            f"{base_url}/api/step",
            method="POST",
        )

        try:
            urlopen(request)
        except HTTPError as error:
            payload = json.loads(error.read())

            assert error.code == 409
            assert payload == {
                "error": "training session is complete",
            }
        else:
            raise AssertionError(
                "Expected HTTP 409"
            )

        assert server.session.epoch == 1


def test_unknown_route_returns_404() -> None:
    with _running_server() as (_, base_url):
        try:
            urlopen(
                f"{base_url}/missing"
            )
        except HTTPError as error:
            assert error.code == 404
        else:
            raise AssertionError(
                "Expected HTTP 404"
            )


def test_wrong_method_on_known_route_returns_405() -> None:
    with _running_server() as (_, base_url):
        request = Request(
            f"{base_url}/api/state",
            method="POST",
        )

        try:
            urlopen(request)
        except HTTPError as error:
            assert error.code == 405
        else:
            raise AssertionError(
                "Expected HTTP 405"
            )


def test_http_stepping_remains_equivalent_to_normal_training() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=4,
    )

    expected_parameters, expected_losses = run_experiment(
        experiment
    )

    server = create_live_server(
        experiment=experiment,
        port=0,
        resolution=5,
    )

    thread = Thread(
        target=server.serve_forever,
        daemon=True,
    )
    thread.start()

    host, port = server.server_address[:2]
    base_url = f"http://{host}:{port}"

    try:
        states = []

        for _ in range(experiment.epochs):
            states.append(
                _post_json(
                    f"{base_url}/api/step"
                )
            )

        assert server.session.epoch == experiment.epochs
        assert server.session.losses == expected_losses
        assert states[-1]["is_complete"] is True

        _assert_parameters_equal(
            server.session.parameters,
            expected_parameters,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_get_state_returns_full_history_without_mutation() -> None:
    with _running_server(
        epochs=3,
        resolution=5,
    ) as (server, base_url):
        _post_json(f"{base_url}/api/step")
        _post_json(f"{base_url}/api/step")

        before_epoch = server.session.epoch
        before_history = server.session.loss_history

        state = _read_json(f"{base_url}/api/state")

        assert state["epoch"] == 2
        assert state["loss_history"] == list(before_history)
        assert state["loss"] == before_history[-1]
        assert server.session.epoch == before_epoch
        assert server.session.loss_history == before_history


def test_post_step_grows_live_history_exactly_once() -> None:
    with _running_server(
        epochs=3,
        resolution=5,
    ) as (_, base_url):
        initial = _read_json(f"{base_url}/api/state")
        stepped = _post_json(f"{base_url}/api/step")

        assert len(initial["loss_history"]) == 1
        assert len(stepped["loss_history"]) == 2
        assert stepped["epoch"] == 1
        assert stepped["loss"] == stepped["loss_history"][-1]


def test_http_state_contains_server_boundary_geometry() -> None:
    with _running_server(
        resolution=5,
    ) as (_, base_url):
        state = _read_json(f"{base_url}/api/state")

        assert "decision_boundary" in state
        assert isinstance(state["decision_boundary"], list)


def test_completion_409_does_not_extend_live_history() -> None:
    with _running_server(
        epochs=1,
        resolution=5,
    ) as (server, base_url):
        completed = _post_json(f"{base_url}/api/step")
        completed_history = tuple(completed["loss_history"])

        request = Request(
            f"{base_url}/api/step",
            method="POST",
        )

        try:
            urlopen(request)
        except HTTPError as error:
            assert error.code == 409
        else:
            raise AssertionError("Expected HTTP 409")

        assert server.session.epoch == 1
        assert server.session.loss_history == completed_history
