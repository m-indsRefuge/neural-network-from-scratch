"""Deterministic experiment definitions for the Neural Observatory."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Experiment:
    """Configuration and dataset for one deterministic training experiment."""

    name: str
    inputs: np.ndarray
    targets: np.ndarray
    seed: int
    epochs: int
    learning_rate: float


def xor_experiment() -> Experiment:
    """Return the canonical deterministic NN-01 XOR experiment."""
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

    return Experiment(
        name="xor",
        inputs=inputs,
        targets=targets,
        seed=7,
        epochs=10_000,
        learning_rate=1.0,
    )