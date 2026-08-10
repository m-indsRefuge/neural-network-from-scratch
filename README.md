# Neural Network From Scratch

A deliberately small neural network built from first principles using Python
and NumPy.

The purpose of this project is not to create a production machine-learning
framework. It exists to understand, implement, test, and verify the mechanics
behind neural-network learning.

## NN-01

Initial architecture:

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

## NN-01 will implement

- parameter initialization
- matrix-based forward propagation
- sigmoid activation
- binary cross-entropy loss
- handwritten backpropagation
- gradient descent
- numerical gradient checking
- deterministic XOR training

## Constraints

- Python 3.12+
- NumPy
- CPU only
- no PyTorch
- no TensorFlow
- no JAX
- no automatic differentiation
- no pretrained model
- no GPU dependency

## Initial problem

NN-01 will learn XOR:

    0 0 -> 0
    0 1 -> 1
    1 0 -> 1
    1 1 -> 0

## Current status

Repository and environment scaffold.

Neural-network implementation has not started.
