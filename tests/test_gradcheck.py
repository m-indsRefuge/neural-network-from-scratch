import numpy as np

from neural_network_from_scratch.gradcheck import gradient_check
from neural_network_from_scratch.network import initialize_parameters


def test_gradient_check_agrees_with_handwritten_backpropagation() -> None:
    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    targets = np.array(
        [
            [0.0],
            [1.0],
            [1.0],
            [0.0],
        ]
    )

    parameters = initialize_parameters(seed=7)

    result = gradient_check(
        inputs,
        targets,
        parameters,
        epsilon=1e-5,
    )

    assert result.relative_error < 1e-6
    assert result.max_absolute_error < 1e-6
