import json
import re

from neural_network_from_scratch.experiments import linear_split_experiment
from neural_network_from_scratch.live_observatory_html import (
    render_live_observatory_html,
)


def _viewer_data(html: str) -> dict:
    match = re.search(
        r'<script id="experiment-data" '
        r'type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )

    assert match is not None

    return json.loads(match.group(1))


def test_live_viewer_uses_one_canvas_without_recorded_frames() -> None:
    html = render_live_observatory_html(
        linear_split_experiment(),
        resolution=5,
    )

    assert html.count('id="belief-canvas"') == 1
    assert 'data-evolution-frame="true"' not in html
    assert 'id="evolution-data"' not in html


def test_live_viewer_embeds_only_immutable_experiment_data() -> None:
    experiment = linear_split_experiment()

    html = render_live_observatory_html(
        experiment,
        resolution=5,
    )

    data = _viewer_data(html)

    assert data["resolution"] == 5
    assert data["training_inputs"] == experiment.inputs.tolist()
    assert data["training_targets"] == experiment.targets.ravel().tolist()

    assert "frames" not in data
    assert "probabilities" not in data


def test_live_viewer_contains_control_and_speed_contract() -> None:
    html = render_live_observatory_html(
        linear_split_experiment(),
        resolution=5,
    )

    assert 'id="step-button"' in html
    assert 'id="train-button"' in html
    assert 'id="pause-button"' in html
    assert 'id="speed-select"' in html

    assert 'value="1000"' in html
    assert 'value="200"' in html
    assert 'value="100"' in html
    assert 'value="0"' in html

    assert 'id="epoch-value"' in html
    assert 'id="loss-value"' in html
    assert 'id="status-message"' in html


def test_live_viewer_references_live_http_endpoints() -> None:
    html = render_live_observatory_html(
        linear_split_experiment(),
        resolution=5,
    )

    assert 'fetch("/api/state")' in html
    assert 'fetch("/api/step"' in html
    assert 'method: "POST"' in html


def test_live_viewer_output_is_deterministic() -> None:
    experiment = linear_split_experiment()

    first = render_live_observatory_html(
        experiment,
        resolution=5,
    )

    second = render_live_observatory_html(
        experiment,
        resolution=5,
    )

    assert first == second


def test_live_viewer_has_learning_dynamics_surfaces() -> None:
    html = render_live_observatory_html(
        linear_split_experiment(),
        resolution=5,
    )

    assert 'id="belief-canvas"' in html
    assert 'id="loss-canvas"' in html
    assert "function drawDecisionBoundary" in html
    assert "function drawLossHistory" in html


def test_browser_consumes_server_learning_dynamics() -> None:
    html = render_live_observatory_html(
        linear_split_experiment(),
        resolution=5,
    )

    assert "state.decision_boundary" in html
    assert "state.loss_history" in html

    assert "marchingSquares" not in html
    assert "computeDecisionBoundary" not in html


def test_loss_history_draws_visible_current_point() -> None:
    html = render_live_observatory_html(
        linear_split_experiment(),
        resolution=5,
    )

    assert "function drawLossHistory" in html
    assert "lossContext.arc(" in html
    assert "lossContext.fill()" in html
