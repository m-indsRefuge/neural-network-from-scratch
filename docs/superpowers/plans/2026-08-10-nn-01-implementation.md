# NN-01 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, and train the NN-01 `2 → 16 → 16 → 1` neural network entirely from first principles using NumPy.

**Architecture:** Samples are row-wise. The network uses three fully connected affine transforms with sigmoid nonlinearities. Backpropagation is derived and implemented manually, then independently verified with central finite differences before XOR training is accepted.

**Tech Stack:** Python 3.12+, NumPy, pytest, Ruff, uv, CPU execution.

## Global Constraints

- Architecture: `2 → 16 → 16 → 1`.
- Exactly 337 trainable parameters.
- Python 3.12+.
- NumPy only for neural-network mathematics.
- CPU execution only.
- No PyTorch, TensorFlow, JAX, automatic differentiation, pretrained models, or GPU-specific code.
- No general-purpose layer framework or optimizer abstraction.
- Fixed random seeds must reproduce results.
- Analytical gradients must be independently verified numerically.
- XOR is the NN-01 training problem.

---

## File Map

```text
src/neural_network_from_scratch/
    __init__.py
    activations.py
    losses.py
    network.py
    gradcheck.py
    training.py

tests/
    test_package.py
    test_activations.py
    test_losses.py
    test_forward.py
    test_backprop.py
    test_gradcheck.py
    test_training.py

experiments/
    xor.py
```

Responsibilities:

- `activations.py`: sigmoid and sigmoid derivative from activation.
- `losses.py`: binary cross-entropy.
- `network.py`: parameters, forward pass, backpropagation, parameter updates.
- `gradcheck.py`: finite-difference verification independent of analytical backpropagation.
- `training.py`: deterministic training loop.
- `experiments/xor.py`: human-readable NN-01 demonstration.

---

### Task 1: Repository Hygiene and Test Baseline

**Files:**
- Create: `.gitattributes`
- Modify: `pyproject.toml`
- Modify: `src/neural_network_from_scratch/__init__.py`
- Create: `tests/test_package.py`

**Produces:** clean metadata, LF-normalized tracked text, no generated CLI stub, and a passing pytest baseline.

- [ ] **Step 1: Write the package test**

```python
import neural_network_from_scratch


def test_package_imports() -> None:
    assert neural_network_from_scratch.__doc__ is not None
```

- [ ] **Step 2: Run the test before cleanup**

Run:

```text
uv run pytest tests/test_package.py -v
```

- [ ] **Step 3: Clean the generated scaffold**

Set this project description in `pyproject.toml`:

```toml
description = "A neural network implemented from first principles with Python and NumPy."
```

Remove `[project.scripts]` entirely. Add:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 100
```

Replace `src/neural_network_from_scratch/__init__.py` with:

```python
"""Neural-network fundamentals implemented from first principles."""
```

Create `.gitattributes`:

```text
* text=auto eol=lf
```

- [ ] **Step 4: Validate**

Run:

```text
uv sync --dev
uv run pytest -q
uv run ruff check .
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Commit**

```text
chore: clean NN-01 project scaffold
```

---

### Task 2: Sigmoid Activation and Binary Cross-Entropy

**Files:**
- Create: `src/neural_network_from_scratch/activations.py`
- Create: `src/neural_network_from_scratch/losses.py`
- Create: `tests/test_activations.py`
- Create: `tests/test_losses.py`

**Interfaces:**

```python
sigmoid(values: np.ndarray) -> np.ndarray
sigmoid_derivative_from_activation(activation: np.ndarray) -> np.ndarray
binary_cross_entropy(targets: np.ndarray, predictions: np.ndarray) -> float
```

- [ ] **Step 1: Write failing sigmoid tests**

```python
def test_sigmoid_zero_is_half() -> None:
    result = sigmoid(np.array([0.0]))
    np.testing.assert_allclose(result, np.array([0.5]))


def test_sigmoid_preserves_shape() -> None:
    values = np.zeros((4, 16))
    assert sigmoid(values).shape == (4, 16)


def test_sigmoid_derivative_from_activation() -> None:
    activation = np.array([0.25, 0.5, 0.75])
    expected = activation * (1.0 - activation)
    np.testing.assert_allclose(
        sigmoid_derivative_from_activation(activation),
        expected,
    )
```

- [ ] **Step 2: Verify failure**

```text
uv run pytest tests/test_activations.py -v
```

- [ ] **Step 3: Implement sigmoid**

```python
def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative_from_activation(activation: np.ndarray) -> np.ndarray:
    return activation * (1.0 - activation)
```

- [ ] **Step 4: Write the BCE known-value test**

```python
def test_binary_cross_entropy_known_value() -> None:
    targets = np.array([[1.0], [0.0]])
    predictions = np.array([[0.8], [0.2]])
    expected = -np.log(0.8)
    assert binary_cross_entropy(targets, predictions) == pytest.approx(expected)
```

- [ ] **Step 5: Implement BCE**

```python
def binary_cross_entropy(targets: np.ndarray, predictions: np.ndarray) -> float:
    epsilon = 1e-12
    safe = np.clip(predictions, epsilon, 1.0 - epsilon)
    return float(
        -np.mean(
            targets * np.log(safe)
            + (1.0 - targets) * np.log(1.0 - safe)
        )
    )
```

- [ ] **Step 6: Validate and commit**

```text
uv run pytest tests/test_activations.py tests/test_losses.py -v
uv run ruff check .
git commit -m "feat: add activation and loss functions"
```

---

### Task 3: Parameter Initialization and Forward Propagation

**Files:**
- Create: `src/neural_network_from_scratch/network.py`
- Create: `tests/test_forward.py`

**Interfaces:**

```python
Parameters
ForwardCache
initialize_parameters(seed: int) -> Parameters
parameter_count(parameters: Parameters) -> int
forward(inputs: np.ndarray, parameters: Parameters) -> tuple[np.ndarray, ForwardCache]
```

Required parameter shapes:

```text
W1 (2, 16)    b1 (1, 16)
W2 (16, 16)   b2 (1, 16)
W3 (16, 1)    b3 (1, 1)
```

- [ ] **Step 1: Test exact parameter shapes and count**

```python
assert parameters.w1.shape == (2, 16)
assert parameters.b1.shape == (1, 16)
assert parameters.w2.shape == (16, 16)
assert parameters.b2.shape == (1, 16)
assert parameters.w3.shape == (16, 1)
assert parameters.b3.shape == (1, 1)
assert parameter_count(parameters) == 337
```

- [ ] **Step 2: Verify failure**

```text
uv run pytest tests/test_forward.py -v
```

- [ ] **Step 3: Implement deterministic initialization**

Use:

```python
rng = np.random.default_rng(seed)
w1 = rng.normal(0.0, np.sqrt(1.0 / 2.0), size=(2, 16))
w2 = rng.normal(0.0, np.sqrt(1.0 / 16.0), size=(16, 16))
w3 = rng.normal(0.0, np.sqrt(1.0 / 16.0), size=(16, 1))
```

All biases start at zero.

- [ ] **Step 4: Test forward shapes**

For the four XOR samples:

```python
assert predictions.shape == (4, 1)
assert cache.a1.shape == (4, 16)
assert cache.a2.shape == (4, 16)
assert np.all((predictions > 0.0) & (predictions < 1.0))
```

- [ ] **Step 5: Implement forward propagation**

```python
z1 = inputs @ parameters.w1 + parameters.b1
a1 = sigmoid(z1)
z2 = a1 @ parameters.w2 + parameters.b2
a2 = sigmoid(z2)
z3 = a2 @ parameters.w3 + parameters.b3
predictions = sigmoid(z3)
```

- [ ] **Step 6: Validate and commit**

```text
uv run pytest tests/test_forward.py -v
uv run pytest -q
uv run ruff check .
git commit -m "feat: add NN-01 forward propagation"
```

---

### Task 4: Handwritten Backpropagation and Parameter Updates

**Files:**
- Modify: `src/neural_network_from_scratch/network.py`
- Create: `tests/test_backprop.py`

**Interfaces:**

```python
Gradients
backward(targets: np.ndarray, parameters: Parameters, cache: ForwardCache) -> Gradients
update_parameters(parameters: Parameters, gradients: Gradients, learning_rate: float) -> Parameters
```

- [ ] **Step 1: Test gradient shapes**

Every gradient must match its parameter shape.

- [ ] **Step 2: Implement output-layer gradient**

```python
sample_count = targets.shape[0]
dz3 = (cache.predictions - targets) / sample_count
dw3 = cache.a2.T @ dz3
db3 = np.sum(dz3, axis=0, keepdims=True)
```

- [ ] **Step 3: Implement hidden-layer gradients**

```python
da2 = dz3 @ parameters.w3.T
dz2 = da2 * sigmoid_derivative_from_activation(cache.a2)
dw2 = cache.a1.T @ dz2
db2 = np.sum(dz2, axis=0, keepdims=True)

da1 = dz2 @ parameters.w2.T
dz1 = da1 * sigmoid_derivative_from_activation(cache.a1)
dw1 = cache.inputs.T @ dz1
db1 = np.sum(dz1, axis=0, keepdims=True)
```

- [ ] **Step 4: Test immutable gradient-descent updates**

Verify each updated array equals `parameter - learning_rate * gradient` and the original parameters are unchanged.

- [ ] **Step 5: Validate and commit**

```text
uv run pytest tests/test_backprop.py -v
uv run pytest -q
uv run ruff check .
git commit -m "feat: add handwritten backpropagation"
```

---

### Task 5: Independent Numerical Gradient Checking

**Files:**
- Create: `src/neural_network_from_scratch/gradcheck.py`
- Create: `tests/test_gradcheck.py`

**Interface:**

```python
GradientCheckResult
gradient_check(
    inputs: np.ndarray,
    targets: np.ndarray,
    parameters: Parameters,
    epsilon: float = 1e-5,
) -> GradientCheckResult
```

The numerical checker must not use `backward()` for its numerical reference derivatives.

- [ ] **Step 1: Write a fixed-seed agreement test**

```python
assert result.relative_error < 1e-6
assert result.max_absolute_error < 1e-6
```

- [ ] **Step 2: Implement central finite differences for all 337 scalar parameters**

```text
J_plus  = loss(parameter + epsilon)
J_minus = loss(parameter - epsilon)
numerical_gradient = (J_plus - J_minus) / (2 * epsilon)
```

Restore the original scalar before proceeding.

- [ ] **Step 3: Flatten analytical and numerical gradients in one stable order**

```text
w1, b1, w2, b2, w3, b3
```

Calculate:

```python
difference = analytical - numerical
max_absolute_error = float(np.max(np.abs(difference)))
relative_error = float(
    np.linalg.norm(difference)
    / (np.linalg.norm(analytical) + np.linalg.norm(numerical) + 1e-12)
)
```

- [ ] **Step 4: Validate and commit**

```text
uv run pytest tests/test_gradcheck.py -v
uv run pytest -q
uv run ruff check .
git commit -m "test: verify backpropagation numerically"
```

This gate must pass before training is accepted.

---

### Task 6: Deterministic XOR Training and NN-01 Acceptance

**Files:**
- Create: `src/neural_network_from_scratch/training.py`
- Create: `tests/test_training.py`
- Create: `experiments/xor.py`
- Modify: `README.md`

**Interface:**

```python
train(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> tuple[Parameters, list[float]]
```

Canonical NN-01 configuration:

```text
seed = 7
epochs = 10000
learning_rate = 1.0
```

- [ ] **Step 1: Write XOR training assertions**

Use the four XOR rows and assert:

```python
assert losses[-1] < losses[0]
assert losses[-1] < 0.01
np.testing.assert_array_equal(classes, targets.astype(int))
```

- [ ] **Step 2: Implement the minimal training loop**

Each epoch performs only:

```text
forward → loss → backward → gradient-descent update
```

- [ ] **Step 3: Test reproducibility**

Two full runs with the same seed and configuration must produce identical loss histories and parameter arrays.

- [ ] **Step 4: Build `experiments/xor.py`**

Print architecture, parameter count, initial loss, final loss, training duration, and each XOR input with raw prediction, thresholded class, and expected class.

- [ ] **Step 5: Run the acceptance gate**

```text
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q src
uv run python experiments/xor.py
git diff --check
```

Required outcome:

- forward tensor shapes correct;
- loss verified against known values;
- analytical gradients match numerical gradients;
- parameters change during training;
- XOR final loss below `0.01`;
- all four XOR rows classify correctly;
- fixed-seed runs reproduce exactly;
- full pytest suite passes;
- CPU experiment completes comfortably;
- no ML framework performs the neural-network mathematics.

- [ ] **Step 6: Update README**

Document architecture, 337-parameter count, forward equations, loss, backpropagation, numerical gradient verification, XOR results, and the exact reproduction command.

- [ ] **Step 7: Commit**

```text
feat: complete NN-01 verified XOR network
```

---

## Plan Self-Review

### Spec Coverage

- Parameter initialization → Task 3
- Forward propagation → Task 3
- Sigmoid → Task 2
- Binary cross-entropy → Task 2
- Analytical backpropagation → Task 4
- Gradient descent → Tasks 4 and 6
- Numerical gradient checking → Task 5
- Deterministic training → Task 6
- Tests → Tasks 1–6
- Exact 337-parameter architecture → Task 3
- XOR → Task 6
- CPU-only / NumPy-only constraints → Global Constraints
- NN-02 and NN-03 remain out of scope

### Scope Check

No API, GUI, autograd engine, optimizer hierarchy, GPU backend, agent, persistence layer, or unrelated infrastructure is introduced.

### Type and Naming Consistency

Stable names throughout: `Parameters`, `ForwardCache`, `Gradients`, `initialize_parameters`, `parameter_count`, `forward`, `backward`, `update_parameters`, `gradient_check`, and `train`.

No placeholders remain.
