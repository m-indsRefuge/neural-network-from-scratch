# Neural Network From Scratch

A deliberately small neural network built from first principles using Python
and NumPy.

The goal of this project is educational: expose and verify the mechanics that
machine-learning frameworks normally hide behind automatic differentiation,
training APIs, and optimizer abstractions.

## NN-01

NN-01 is a fully connected neural network trained to learn XOR.

Architecture:

    2 inputs
        |
        v
    16 hidden neurons
        |
        v
    16 hidden neurons
        |
        v
    1 output

Trainable parameters: **337**

Parameter shapes:

- W1: `(2, 16)`
- b1: `(1, 16)`
- W2: `(16, 16)`
- b2: `(1, 16)`
- W3: `(16, 1)`
- b3: `(1, 1)`

## Implemented from first principles

NN-01 implements:

- deterministic parameter initialization
- matrix-based forward propagation
- sigmoid activation
- binary cross-entropy loss
- handwritten analytical backpropagation
- full-batch gradient descent
- numerical finite-difference gradient checking
- deterministic training
- automated tests

No machine-learning framework performs the learning mathematics.

## Constraints

- Python 3.12+
- NumPy
- CPU only
- no PyTorch
- no TensorFlow
- no JAX
- no automatic differentiation
- no pretrained models
- no GPU dependency

## Learning problem

NN-01 learns XOR:

    0 0 -> 0
    0 1 -> 1
    1 0 -> 1
    1 1 -> 0

Canonical training configuration:

- seed: `7`
- epochs: `10,000`
- learning rate: `1.0`
- optimization: full-batch gradient descent

## Numerical gradient verification

The handwritten analytical gradients are independently checked using central
finite differences:

    dJ/dθ ≈ [J(θ + ε) - J(θ - ε)] / (2ε)

All 337 trainable scalar parameters are checked.

Acceptance thresholds:

- relative error `< 1e-6`
- maximum absolute error `< 1e-6`

## Observed XOR result

A canonical run produced:

    Initial loss: 0.6996885651
    Final loss:   0.0000952025

Predictions:

    [0, 0] -> 0.0000573491 -> class 0
    [0, 1] -> 0.9999102815 -> class 1
    [1, 0] -> 0.9999024411 -> class 1
    [1, 1] -> 0.0001361193 -> class 0

All four XOR inputs were classified correctly.

Training duration is reported by the experiment as observational telemetry.
It varies with machine state and system load and is not an NN-01 acceptance
criterion.

## Run the experiment

From the repository root:

    uv run python experiments/xor.py

## Run the tests

    uv run pytest -q

## Project purpose

NN-01 is intentionally small.

Its purpose is not to compete with modern machine-learning frameworks, but to
build a transparent mental model of:

    parameters
        |
        v
    forward propagation
        |
        v
    loss
        |
        v
    backpropagation
        |
        v
    gradients
        |
        v
    parameter updates
        |
        v
    learning

Future experiments can build on this foundation while preserving NN-01 as the
small, auditable reference implementation.
