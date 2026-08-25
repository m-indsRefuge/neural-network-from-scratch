"""Stepwise live-training runtime for Neural Observatory."""

from neural_network_from_scratch.experiments import Experiment
from neural_network_from_scratch.losses import binary_cross_entropy
from neural_network_from_scratch.network import (
    Parameters,
    backward,
    forward,
    initialize_parameters,
    update_parameters,
)
from neural_network_from_scratch.observability import TrainingSnapshot


class LiveTrainingSession:
    """Stateful NN-01 training session advanced one epoch at a time."""

    def __init__(self, experiment: Experiment) -> None:
        self._experiment = experiment
        self._parameters = initialize_parameters(experiment.seed)
        self._losses: list[float] = []
        self._loss_history: list[float] = [
            self._loss_for_parameters(self._parameters)
        ]
        self._epoch = 0

    def _loss_for_parameters(
        self,
        parameters: Parameters,
    ) -> float:
        predictions, _ = forward(
            self._experiment.inputs,
            parameters,
        )

        return binary_cross_entropy(
            self._experiment.targets,
            predictions,
        )

    @property
    def loss_history(self) -> tuple[float, ...]:
        """Return post-update live-view losses from epoch zero onward."""
        return tuple(self._loss_history)

    @property
    def epoch(self) -> int:
        """Return the number of completed training epochs."""
        return self._epoch

    @property
    def losses(self) -> list[float]:
        """Return losses observed during completed epochs."""
        return self._losses

    @property
    def parameters(self) -> Parameters:
        """Return the current trainable parameters."""
        return self._parameters

    @property
    def is_complete(self) -> bool:
        """Return whether all configured epochs have been completed."""
        return self._epoch >= self._experiment.epochs

    def step(self) -> TrainingSnapshot:
        """Perform exactly one NN-01 training epoch."""
        if self.is_complete:
            raise RuntimeError("training session is complete")

        predictions, cache = forward(
            self._experiment.inputs,
            self._parameters,
        )

        loss = binary_cross_entropy(
            self._experiment.targets,
            predictions,
        )

        gradients = backward(
            self._experiment.targets,
            self._parameters,
            cache,
        )

        next_epoch = self._epoch + 1

        snapshot = TrainingSnapshot(
            epoch=next_epoch,
            loss=loss,
            parameters=self._parameters,
            gradients=gradients,
            cache=cache,
        )

        self._losses.append(loss)

        self._parameters = update_parameters(
            self._parameters,
            gradients,
            self._experiment.learning_rate,
        )

        self._epoch = next_epoch
        self._loss_history.append(
            self._loss_for_parameters(self._parameters)
        )

        return snapshot
