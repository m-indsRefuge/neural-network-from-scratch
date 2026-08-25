import numpy as np

from neural_network_from_scratch.decision_boundary import extract_decision_boundary
from neural_network_from_scratch.prediction_field import PredictionField


def test_field_without_threshold_crossing_has_no_boundary() -> None:
    field = PredictionField(
        x_values=np.array([0.0, 1.0]),
        y_values=np.array([0.0, 1.0]),
        probabilities=np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ]
        ),
    )

    assert extract_decision_boundary(field) == ()

def test_vertical_threshold_crossing_produces_interpolated_segment() -> None:
    field = PredictionField(
        x_values=np.array([0.0, 1.0]),
        y_values=np.array([0.0, 1.0]),
        probabilities=np.array(
            [
                [0.25, 0.75],
                [0.25, 0.75],
            ]
        ),
    )

    assert extract_decision_boundary(field) == (
        ((0.5, 0.0), (0.5, 1.0)),
    )



def test_decision_boundary_is_deterministic() -> None:
    field = PredictionField(
        x_values=np.array([0.0, 0.5, 1.0]),
        y_values=np.array([0.0, 0.5, 1.0]),
        probabilities=np.array(
            [
                [0.1, 0.4, 0.8],
                [0.2, 0.6, 0.9],
                [0.3, 0.7, 0.95],
            ]
        ),
    )

    first = extract_decision_boundary(field)
    second = extract_decision_boundary(field)

    assert first == second


def test_saddle_cell_produces_two_deterministic_segments() -> None:
    field = PredictionField(
        x_values=np.array([0.0, 1.0]),
        y_values=np.array([0.0, 1.0]),
        probabilities=np.array(
            [
                [0.75, 0.25],
                [0.25, 0.75],
            ]
        ),
    )

    boundary = extract_decision_boundary(field)

    assert len(boundary) == 2
