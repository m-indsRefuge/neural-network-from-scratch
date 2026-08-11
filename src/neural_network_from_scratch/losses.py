"""Loss functions used by NN-01."""

import numpy as np


def binary_cross_entropy(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    """Return mean binary cross-entropy loss."""
    epsilon = 1e-12
    safe_predictions = np.clip(predictions, epsilon, 1.0 - epsilon)

    return float(
        -np.mean(
            targets * np.log(safe_predictions)
            + (1.0 - targets) * np.log(1.0 - safe_predictions)
        )
    )
