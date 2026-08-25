"""Deterministic SVG rendering for Neural Observatory prediction fields."""

import xml.etree.ElementTree as ET

import numpy as np

from neural_network_from_scratch.prediction_field import PredictionField

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
CANVAS_SIZE = 640.0


def _probability_color(probability: float) -> str:
    """Map a class probability to a deterministic RGB field color."""
    probability = float(np.clip(probability, 0.0, 1.0))

    class_zero = np.array([31.0, 20.0, 66.0])
    class_one = np.array([35.0, 220.0, 235.0])

    rgb = np.rint(
        class_zero + probability * (class_one - class_zero)
    ).astype(int)

    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def render_prediction_field_svg(
    field: PredictionField,
    *,
    training_inputs: np.ndarray,
    training_targets: np.ndarray,
) -> str:
    """Render a probability field and its training samples as deterministic SVG."""
    row_count, column_count = field.probabilities.shape

    cell_width = CANVAS_SIZE / column_count
    cell_height = CANVAS_SIZE / row_count

    root = ET.Element(
        "svg",
        {
            "xmlns": SVG_NAMESPACE,
            "viewBox": f"0 0 {CANVAS_SIZE:g} {CANVAS_SIZE:g}",
            "role": "img",
            "aria-label": "Neural network prediction field",
        },
    )

    field_group = ET.SubElement(
        root,
        "g",
        {"data-layer": "prediction-field"},
    )

    for row in range(row_count):
        for column in range(column_count):
            probability = float(field.probabilities[row, column])

            x = column * cell_width

            # SVG y increases downward, while our field y-values increase upward.
            y = CANVAS_SIZE - ((row + 1) * cell_height)

            ET.SubElement(
                field_group,
                "rect",
                {
                    "x": f"{x:.6f}",
                    "y": f"{y:.6f}",
                    "width": f"{cell_width:.6f}",
                    "height": f"{cell_height:.6f}",
                    "fill": _probability_color(probability),
                    "data-field-cell": "true",
                    "data-probability": f"{probability:.17g}",
                },
            )

    point_group = ET.SubElement(
        root,
        "g",
        {"data-layer": "training-points"},
    )

    for point, target in zip(
        training_inputs,
        training_targets,
        strict=True,
    ):
        x_value = float(point[0])
        y_value = float(point[1])
        class_value = int(float(target[0]) >= 0.5)

        cx = x_value * CANVAS_SIZE
        cy = (1.0 - y_value) * CANVAS_SIZE

        if class_value == 0:
            fill = "rgb(255,79,216)"
        else:
            fill = "rgb(82,230,255)"

        ET.SubElement(
            point_group,
            "circle",
            {
                "cx": f"{cx:.6f}",
                "cy": f"{cy:.6f}",
                "r": "9",
                "fill": fill,
                "stroke": "rgb(255,255,255)",
                "stroke-width": "2",
                "data-training-point": "true",
                "data-class": str(class_value),
            },
        )

    return ET.tostring(
        root,
        encoding="unicode",
        short_empty_elements=True,
    )