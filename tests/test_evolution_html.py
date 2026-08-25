import json
import re
from dataclasses import replace

from neural_network_from_scratch.evolution_html import (
    render_training_evolution_html,
)
from neural_network_from_scratch.experiments import linear_split_experiment
from neural_network_from_scratch.training_evolution import (
    capture_training_evolution,
)


def _small_evolution():
    experiment = replace(
        linear_split_experiment(),
        epochs=5,
    )

    evolution = capture_training_evolution(
        experiment,
        epochs=(1, 3, 5),
    )

    return experiment, evolution


def _embedded_data(html: str) -> dict:
    match = re.search(
        r'<script id="evolution-data" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )

    assert match is not None

    return json.loads(match.group(1))


def test_evolution_html_uses_one_canvas_and_no_svg_field_cells() -> None:
    experiment, evolution = _small_evolution()

    html = render_training_evolution_html(
        experiment,
        evolution,
        resolution=5,
    )

    assert html.count('id="belief-canvas"') == 1
    assert "<svg" not in html
    assert 'data-field-cell="true"' not in html
    assert 'data-training-point="true"' not in html


def test_evolution_html_embeds_real_frame_probabilities() -> None:
    experiment, evolution = _small_evolution()

    html = render_training_evolution_html(
        experiment,
        evolution,
        resolution=5,
    )

    data = _embedded_data(html)

    assert data["resolution"] == 5
    assert [frame["epoch"] for frame in data["frames"]] == [1, 3, 5]
    assert len(data["frames"]) == 3

    for frame in data["frames"]:
        assert len(frame["probabilities"]) == 25

    assert len(data["training_inputs"]) == 8
    assert len(data["training_targets"]) == 8


def test_evolution_html_contains_interactive_controls() -> None:
    experiment, evolution = _small_evolution()

    html = render_training_evolution_html(
        experiment,
        evolution,
        resolution=5,
    )

    assert 'id="epoch-slider"' in html
    assert 'id="play-toggle"' in html
    assert 'id="epoch-value"' in html
    assert 'id="loss-value"' in html


def test_evolution_html_is_deterministic() -> None:
    experiment, evolution = _small_evolution()

    html_a = render_training_evolution_html(
        experiment,
        evolution,
        resolution=5,
    )
    html_b = render_training_evolution_html(
        experiment,
        evolution,
        resolution=5,
    )

    assert html_a == html_b


def test_real_observatory_viewer_stays_under_three_megabytes() -> None:
    experiment = linear_split_experiment()

    evolution = capture_training_evolution(
        experiment,
        epochs=(1, 5, 10, 25, 50, 100, 250, 500),
    )

    html = render_training_evolution_html(
        experiment,
        evolution,
        resolution=101,
    )

    assert len(html.encode("utf-8")) < 3_000_000