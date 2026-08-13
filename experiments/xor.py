"""Run the canonical NN-01 XOR learning experiment."""

from time import perf_counter

import numpy as np

from neural_network_from_scratch.network import forward, parameter_count
from neural_network_from_scratch.training import train


def main() -> None:
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

    start = perf_counter()

    parameters, losses = train(
        inputs,
        targets,
        seed=7,
        epochs=10_000,
        learning_rate=1.0,
    )

    duration = perf_counter() - start
    predictions, _ = forward(inputs, parameters)

    print("NN-01 XOR EXPERIMENT")
    print("====================")
    print("Architecture: 2 -> 16 -> 16 -> 1")
    print(f"Trainable parameters: {parameter_count(parameters)}")
    print("Seed: 7")
    print("Epochs: 10000")
    print("Learning rate: 1.0")
    print(f"Initial loss: {losses[0]:.10f}")
    print(f"Final loss:   {losses[-1]:.10f}")
    print(f"Duration:     {duration:.4f} seconds")
    print()
    print("Predictions:")

    for sample, prediction, target in zip(
        inputs,
        predictions,
        targets,
        strict=True,
    ):
        predicted_class = int(prediction[0] >= 0.5)

        print(
            f"{sample.astype(int).tolist()} -> "
            f"{prediction[0]:.10f} -> "
            f"class {predicted_class} "
            f"(expected {int(target[0])})"
        )


if __name__ == "__main__":
    main()
