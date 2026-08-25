"""Post-update visual state for the live Neural Observatory."""

import base64
from dataclasses import dataclass

import numpy as np

from neural_network_from_scratch.decision_boundary import (
    Segment,
    extract_decision_boundary,
)
from neural_network_from_scratch.experiments import Experiment
from neural_network_from_scratch.live_training import LiveTrainingSession
from neural_network_from_scratch.network_telemetry import (
    NetworkTelemetry,
    NetworkTelemetryError,
    probe_network_telemetry,
)
from neural_network_from_scratch.prediction_field import (
    probe_prediction_field,
)


@dataclass(frozen=True)
class LiveViewState:
    """Browser-facing view of the current authoritative NN state."""

    epoch: int
    loss: float
    loss_history: tuple[float, ...]
    is_complete: bool
    resolution: int
    field: str
    decision_boundary: tuple[Segment, ...]
    network: NetworkTelemetry | None
    network_error: str | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "epoch": self.epoch,
            "loss": self.loss,
            "loss_history": self.loss_history,
            "is_complete": self.is_complete,
            "resolution": self.resolution,
            "field": self.field,
            "decision_boundary": self.decision_boundary,
            "network": (
                None
                if self.network is None
                else self.network.to_dict()
            ),
            "network_error": self.network_error,
        }


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
    parameters = session.parameters
    field = probe_prediction_field(
        parameters,
        resolution=resolution,
    )
    loss_history = session.loss_history

    try:
        network = probe_network_telemetry(
            experiment,
            parameters,
        )
        network_error = None
    except NetworkTelemetryError as error:
        network = None
        network_error = str(error)

    return LiveViewState(
        epoch=session.epoch,
        loss=loss_history[-1],
        loss_history=loss_history,
        is_complete=session.is_complete,
        resolution=resolution,
        field=encode_probability_field(
            field.probabilities,
        ),
        decision_boundary=extract_decision_boundary(field),
        network=network,
        network_error=network_error,
    )
