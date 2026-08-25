import xml.etree.ElementTree as ET

import numpy as np

from neural_network_from_scratch.field_svg import render_prediction_field_svg
from neural_network_from_scratch.prediction_field import PredictionField

SVG_NS = {"svg": "http://www.w3.org/2000/svg"}


def _small_field() -> PredictionField:
    return PredictionField(
        x_values=np.array([0.0, 1.0]),
        y_values=np.array([0.0, 1.0]),
        probabilities=np.array(
            [
                [0.0, 0.25],
                [0.75, 1.0],
            ]
        ),
    )


def test_renderer_emits_valid_svg_with_one_cell_per_probability() -> None:
    svg = render_prediction_field_svg(
        _small_field(),
        training_inputs=np.empty((0, 2)),
        training_targets=np.empty((0, 1)),
    )

    root = ET.fromstring(svg)

    assert root.tag == "{http://www.w3.org/2000/svg}svg"

    cells = root.findall(".//svg:rect[@data-field-cell='true']", SVG_NS)

    assert len(cells) == 4
    assert [float(cell.attrib["data-probability"]) for cell in cells] == [
        0.0,
        0.25,
        0.75,
        1.0,
    ]


def test_renderer_overlays_training_points_with_class_identity() -> None:
    training_inputs = np.array(
        [
            [0.25, 0.25],
            [0.75, 0.75],
        ]
    )
    training_targets = np.array(
        [
            [0.0],
            [1.0],
        ]
    )

    svg = render_prediction_field_svg(
        _small_field(),
        training_inputs=training_inputs,
        training_targets=training_targets,
    )

    root = ET.fromstring(svg)

    points = root.findall(
        ".//svg:circle[@data-training-point='true']",
        SVG_NS,
    )

    assert len(points) == 2
    assert [point.attrib["data-class"] for point in points] == ["0", "1"]


def test_renderer_is_deterministic() -> None:
    field = _small_field()
    training_inputs = np.array([[0.5, 0.5]])
    training_targets = np.array([[1.0]])

    svg_a = render_prediction_field_svg(
        field,
        training_inputs=training_inputs,
        training_targets=training_targets,
    )
    svg_b = render_prediction_field_svg(
        field,
        training_inputs=training_inputs,
        training_targets=training_targets,
    )

    assert svg_a == svg_b