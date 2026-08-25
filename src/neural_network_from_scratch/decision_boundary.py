from neural_network_from_scratch.prediction_field import PredictionField

Point = tuple[float, float]
Segment = tuple[Point, Point]


def _interpolate(
    start: Point,
    end: Point,
    start_value: float,
    end_value: float,
    threshold: float,
) -> Point:
    fraction = (threshold - start_value) / (end_value - start_value)

    return (
        start[0] + fraction * (end[0] - start[0]),
        start[1] + fraction * (end[1] - start[1]),
    )


def extract_decision_boundary(
    field: PredictionField,
    threshold: float = 0.5,
) -> tuple[Segment, ...]:
    """Extract threshold contour segments from a prediction field."""
    segments: list[Segment] = []

    for row in range(len(field.y_values) - 1):
        for column in range(len(field.x_values) - 1):
            x0 = float(field.x_values[column])
            x1 = float(field.x_values[column + 1])
            y0 = float(field.y_values[row])
            y1 = float(field.y_values[row + 1])

            bottom_left = float(field.probabilities[row, column])
            bottom_right = float(field.probabilities[row, column + 1])
            top_right = float(field.probabilities[row + 1, column + 1])
            top_left = float(field.probabilities[row + 1, column])

            edges = (
                ((x0, y0), (x1, y0), bottom_left, bottom_right),
                ((x1, y0), (x1, y1), bottom_right, top_right),
                ((x1, y1), (x0, y1), top_right, top_left),
                ((x0, y1), (x0, y0), top_left, bottom_left),
            )

            crossings: list[Point] = []

            for start, end, start_value, end_value in edges:
                start_high = start_value >= threshold
                end_high = end_value >= threshold

                if start_high == end_high:
                    continue

                crossings.append(
                    _interpolate(
                        start,
                        end,
                        start_value,
                        end_value,
                        threshold,
                    )
                )

            if len(crossings) == 2:
                segments.append((crossings[0], crossings[1]))
                continue

            if len(crossings) == 4:
                case = (
                    (1 if bottom_left >= threshold else 0)
                    | (2 if bottom_right >= threshold else 0)
                    | (4 if top_right >= threshold else 0)
                    | (8 if top_left >= threshold else 0)
                )
                center_high = (
                    (
                        bottom_left
                        + bottom_right
                        + top_right
                        + top_left
                    )
                    / 4.0
                    >= threshold
                )

                pair_adjacent = (
                    (case == 5 and center_high)
                    or (case == 10 and not center_high)
                )

                if pair_adjacent:
                    segments.extend(
                        (
                            (crossings[0], crossings[1]),
                            (crossings[2], crossings[3]),
                        )
                    )
                else:
                    segments.extend(
                        (
                            (crossings[0], crossings[3]),
                            (crossings[1], crossings[2]),
                        )
                    )

    return tuple(segments)
