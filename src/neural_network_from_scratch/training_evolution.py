"""Selected training-state capture for Neural Observatory evolution views."""

from dataclasses import dataclass

from neural_network_from_scratch.experiment_runner import run_experiment
from neural_network_from_scratch.experiments import Experiment
from neural_network_from_scratch.network import Parameters
from neural_network_from_scratch.observability import TrainingSnapshot


@dataclass(frozen=True)
class TrainingEvolution:
    """Final training result plus selected chronological training snapshots."""

    parameters: Parameters
    losses: list[float]
    snapshots: tuple[TrainingSnapshot, ...]


def capture_training_evolution(
    experiment: Experiment,
    *,
    epochs: tuple[int, ...],
) -> TrainingEvolution:
    """Run an experiment while retaining only selected training epochs."""
    if len(set(epochs)) != len(epochs):
        raise ValueError("capture epochs must be unique")

    if any(epoch < 1 or epoch > experiment.epochs for epoch in epochs):
        raise ValueError(
            "capture epochs must be between 1 and experiment epochs"
        )

    requested_epochs = set(epochs)
    snapshots: list[TrainingSnapshot] = []

    def observer(snapshot: TrainingSnapshot) -> None:
        if snapshot.epoch in requested_epochs:
            snapshots.append(snapshot)

    parameters, losses = run_experiment(
        experiment,
        observer=observer,
    )

    return TrainingEvolution(
        parameters=parameters,
        losses=losses,
        snapshots=tuple(snapshots),
    )