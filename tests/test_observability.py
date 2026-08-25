import numpy as np

from neural_network_from_scratch.network import (
    ForwardCache,
    Gradients,
    Parameters,
)
from neural_network_from_scratch.observability import TrainingSnapshot


def test_training_snapshot_preserves_training_state() -> None:
    parameters = Parameters(
        w1=np.zeros((2, 16)),
        b1=np.zeros((1, 16)),
        w2=np.zeros((16, 16)),
        b2=np.zeros((1, 16)),
        w3=np.zeros((16, 1)),
        b3=np.zeros((1, 1)),
    )

    cache = ForwardCache(
        inputs=np.zeros((4, 2)),
        a1=np.zeros((4, 16)),
        a2=np.zeros((4, 16)),
        predictions=np.zeros((4, 1)),
    )

    gradients = Gradients(
        w1=np.zeros((2, 16)),
        b1=np.zeros((1, 16)),
        w2=np.zeros((16, 16)),
        b2=np.zeros((1, 16)),
        w3=np.zeros((16, 1)),
        b3=np.zeros((1, 1)),
    )

    snapshot = TrainingSnapshot(
        epoch=1,
        loss=0.5,
        parameters=parameters,
        gradients=gradients,
        cache=cache,
    )

    assert snapshot.epoch == 1
    assert snapshot.loss == 0.5
    assert snapshot.parameters is parameters
    assert snapshot.gradients is gradients
    assert snapshot.cache is cache