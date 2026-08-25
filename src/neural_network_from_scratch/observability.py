"""Typed observability contracts for neural-network training."""

from dataclasses import dataclass

from neural_network_from_scratch.network import (
    ForwardCache,
    Gradients,
    Parameters,
)


@dataclass(frozen=True)
class TrainingSnapshot:
    """Immutable view of one NN-01 training step."""

    epoch: int
    loss: float
    parameters: Parameters
    gradients: Gradients
    cache: ForwardCache