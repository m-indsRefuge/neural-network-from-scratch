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


def test_evolution_html_contains_one_frame_per_snapshot() -> None:
    experiment, evolution = _small_evolution()

    html = render_training_evolution_html(
        experiment,
        evolution,
        resolution=5,
    )

    assert html.count(
        '<div class="evolution-frame" data-evolution-frame="true"'
    ) == 3
    assert 'data-epoch="1"' in html
    assert 'data-epoch="3"' in html
    assert 'data-epoch="5"' in html


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


def test_evolution_html_embeds_real_prediction_fields() -> None:
    experiment, evolution = _small_evolution()

    html = render_training_evolution_html(
        experiment,
        evolution,
        resolution=5,
    )

    assert html.count('data-field-cell="true"') == 3 * 25
    assert html.count('data-training-point="true"') == 3 * 8


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