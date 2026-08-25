"""Post-update visual state for the live Neural Observatory."""

import base64
from dataclasses import asdict, dataclass

import numpy as np

from neural_network_from_scratch.experiments import Experiment
from neural_network_from_scratch.live_training import LiveTrainingSession
from neural_network_from_scratch.losses import binary_cross_entropy
from neural_network_from_scratch.network import forward
from neural_network_from_scratch.prediction_field import (
    probe_prediction_field,
)


@dataclass(frozen=True)
class LiveViewState:
    """Browser-facing view of the current authoritative NN state."""

    epoch: int
    loss: float
    is_complete: bool
    resolution: int
    field: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def encode_probability_field(
    probabilities: np.ndarray,
) -> str:
    """Encode probabilities as deterministic uint8 base64 telemetry."""
    quantized = np.rint(
        np.clip(
            probabilities,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(np.uint8)

    return base64.b64encode(
        quantized.tobytes(order="C")
    ).decode("ascii")


def build_live_view_state(
    session: LiveTrainingSession,
    experiment: Experiment,
    *,
    resolution: int,
) -> LiveViewState:
    """Build the current post-update browser view without training."""
    predictions, _ = forward(
        experiment.inputs,
        session.parameters,
    )

    loss = binary_cross_entropy(
        experiment.targets,
        predictions,
    )

    field = probe_prediction_field(
        session.parameters,
        resolution=resolution,
    )

    return LiveViewState(
        epoch=session.epoch,
        loss=loss,
        is_complete=session.is_complete,
        resolution=resolution,
        field=encode_probability_field(
            field.probabilities,
        ),
    )