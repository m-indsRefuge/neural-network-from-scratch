"""Execution boundary for deterministic Neural Observatory experiments."""

from collections.abc import Callable

from neural_network_from_scratch.experiments import Experiment
from neural_network_from_scratch.network import Parameters
from neural_network_from_scratch.observability import TrainingSnapshot
from neural_network_from_scratch.training import train


def run_experiment(
    experiment: Experiment,
    *,
    observer: Callable[[TrainingSnapshot], None] | None = None,
) -> tuple[Parameters, list[float]]:
    """Run one experiment using the existing NN-01 training engine."""
    return train(
        experiment.inputs,
        experiment.targets,
        seed=experiment.seed,
        epochs=experiment.epochs,
        learning_rate=experiment.learning_rate,
        observer=observer,
    )