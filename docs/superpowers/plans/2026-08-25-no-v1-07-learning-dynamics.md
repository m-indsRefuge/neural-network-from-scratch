# NO-V1-07 Learning Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, server-authoritative live learning dynamics to the Neural Observatory: a full epoch-0-to-current BCE loss curve and a Python-computed p = 0.5 decision boundary.

**Architecture:** Preserve the existing NN-01 training contract and `LiveTrainingSession.losses`, which contains the pre-update losses used by ordinary deterministic training. Add a separate post-update `loss_history` owned by `LiveTrainingSession`; compute decision-boundary segments in Python from the full-precision `PredictionField`; expose both through `LiveViewState`; and render them in the existing self-contained browser viewer without making JavaScript authoritative.

**Tech Stack:** Python 3.12+, NumPy, standard-library HTTP server, pytest, Ruff, browser Canvas/JavaScript.

**Spec:** `docs/superpowers/specs/2026-08-25-no-v1-07-learning-dynamics-design.md`

## Global Constraints

- No PyTorch, TensorFlow, JAX, plotting dependency, contour dependency, WebSocket dependency, or SSE dependency.
- Python remains authoritative for training progression, loss history, prediction-field generation, and decision-boundary computation.
- `TrainingSnapshot` semantics remain pre-update and unchanged.
- `LiveTrainingSession.losses` remains the existing pre-update training-loss sequence and must continue to exactly match ordinary deterministic training.
- New `LiveTrainingSession.loss_history` represents post-update live-view state and contains epoch 0 through the current epoch inclusive.
- `len(session.loss_history) == session.epoch + 1`.
- Current `LiveViewState.loss == LiveViewState.loss_history[-1]`.
- The decision boundary is p = 0.5 and is derived from full-precision probabilities before field quantization.
- Browser code renders server-provided history and contour geometry; it must not reconstruct either authoritative signal.
- The existing localhost-only, single-threaded HTTP lifecycle and HTTP 409 completion boundary remain unchanged.
- Prediction-field visual transport remains uint8/base64.
- No neuron, activation, weight, or gradient visualization is introduced in this milestone.

---

### Task 1: Add Server-Owned Post-Update Loss History

**Files:**
- Modify: `src/neural_network_from_scratch/live_training.py`
- Modify: `tests/test_live_training.py`

**Interfaces:**
- Preserves: `LiveTrainingSession.losses -> list[float]` with existing pre-update semantics.
- Produces: `LiveTrainingSession.loss_history -> tuple[float, ...]`.
- Invariant: `len(loss_history) == epoch + 1`.
- Invariant: `loss_history[-1]` is the BCE loss of `session.parameters`.

- [ ] **Step 1: Write failing epoch-zero history test**

Add to `tests/test_live_training.py`:

```python
from neural_network_from_scratch.losses import binary_cross_entropy
from neural_network_from_scratch.network import forward


def test_live_loss_history_starts_with_epoch_zero_model() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=3,
    )
    session = LiveTrainingSession(experiment)

    predictions, _ = forward(
        experiment.inputs,
        session.parameters,
    )
    expected = binary_cross_entropy(
        experiment.targets,
        predictions,
    )

    assert session.epoch == 0
    assert session.losses == []
    assert session.loss_history == (expected,)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
uv run pytest tests/test_live_training.py::test_live_loss_history_starts_with_epoch_zero_model -q
```

Expected: FAIL because `LiveTrainingSession` has no `loss_history`.

- [ ] **Step 3: Add failing post-step coherence tests**

Add:

```python
def test_live_loss_history_appends_post_update_loss() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=3,
    )
    session = LiveTrainingSession(experiment)

    initial_loss = session.loss_history[-1]
    snapshot = session.step()

    predictions, _ = forward(
        experiment.inputs,
        session.parameters,
    )
    current_loss = binary_cross_entropy(
        experiment.targets,
        predictions,
    )

    assert session.epoch == 1
    assert len(session.loss_history) == 2
    assert session.loss_history == (
        initial_loss,
        current_loss,
    )

    assert session.losses == [snapshot.loss]
    assert session.loss_history[-1] != snapshot.loss


def test_live_loss_history_length_tracks_epoch_plus_one() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=4,
    )
    session = LiveTrainingSession(experiment)

    assert len(session.loss_history) == 1

    while not session.is_complete:
        session.step()
        assert len(session.loss_history) == session.epoch + 1
```

- [ ] **Step 4: Implement minimal post-update history**

In `live_training.py`, add:

```python
class LiveTrainingSession:
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
```

After updating parameters and setting the epoch, append the current-model loss:

```python
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
```

Do not alter the existing append to `self._losses`.

- [ ] **Step 5: Run focused and regression tests**

```powershell
uv run pytest tests/test_live_training.py -q
uv run ruff check src/neural_network_from_scratch/live_training.py tests/test_live_training.py
git diff --check
```

Expected: green.

- [ ] **Step 6: Commit Task 1**

```powershell
git add src/neural_network_from_scratch/live_training.py tests/test_live_training.py
git diff --cached --check
git commit -m "feat: track live post-update loss history"
```

---

### Task 2: Extract the p = 0.5 Decision Boundary in Python

**Files:**
- Create: `src/neural_network_from_scratch/decision_boundary.py`
- Create: `tests/test_decision_boundary.py`

**Interfaces:**
- Consumes: `PredictionField`.
- Produces: `Point = tuple[float, float]`.
- Produces: `Segment = tuple[Point, Point]`.
- Produces: `extract_decision_boundary(field: PredictionField, *, threshold: float = 0.5) -> tuple[Segment, ...]`.
- Geometry order is deterministic: cells are traversed row-major and segments use deterministic edge pairing.

- [ ] **Step 1: Write failing no-crossing and simple-crossing tests**

Create `tests/test_decision_boundary.py`:

```python
import numpy as np

from neural_network_from_scratch.decision_boundary import (
    extract_decision_boundary,
)
from neural_network_from_scratch.prediction_field import PredictionField


def _field(probabilities: list[list[float]]) -> PredictionField:
    values = np.asarray(probabilities, dtype=float)
    resolution = values.shape[0]

    return PredictionField(
        x_values=np.linspace(0.0, 1.0, resolution),
        y_values=np.linspace(0.0, 1.0, resolution),
        probabilities=values,
    )


def test_contour_is_empty_when_field_never_crosses_threshold() -> None:
    field = _field([
        [0.1, 0.2],
        [0.3, 0.4],
    ])

    assert extract_decision_boundary(field) == ()


def test_contour_interpolates_simple_vertical_crossing() -> None:
    field = _field([
        [0.0, 1.0],
        [0.0, 1.0],
    ])

    assert extract_decision_boundary(field) == (
        (
            (0.5, 0.0),
            (0.5, 1.0),
        ),
    )
```

- [ ] **Step 2: Verify RED**

```powershell
uv run pytest tests/test_decision_boundary.py -q
```

Expected: import/module failure.

- [ ] **Step 3: Implement deterministic marching-squares extraction**

Create `src/neural_network_from_scratch/decision_boundary.py`:

```python
"""Deterministic contour extraction for Neural Observatory fields."""

from typing import TypeAlias

from neural_network_from_scratch.prediction_field import PredictionField

Point: TypeAlias = tuple[float, float]
Segment: TypeAlias = tuple[Point, Point]

_EDGE_PAIRS = {
    1: ((0, 3),),
    2: ((0, 1),),
    3: ((1, 3),),
    4: ((1, 2),),
    6: ((0, 2),),
    7: ((2, 3),),
    8: ((2, 3),),
    9: ((0, 2),),
    11: ((1, 2),),
    12: ((1, 3),),
    13: ((0, 1),),
    14: ((0, 3),),
}


def _interpolate(
    start: Point,
    end: Point,
    start_value: float,
    end_value: float,
    threshold: float,
) -> Point:
    fraction = (threshold - start_value) / (end_value - start_value)

    return (
        start[0] + fraction * (end[0] - start[0]),
        start[1] + fraction * (end[1] - start[1]),
    )


def _saddle_pairs(
    case: int,
    center_value: float,
    threshold: float,
) -> tuple[tuple[int, int], ...]:
    center_is_high = center_value >= threshold

    if case == 5:
        if center_is_high:
            return ((0, 1), (2, 3))
        return ((0, 3), (1, 2))

    if center_is_high:
        return ((0, 3), (1, 2))
    return ((0, 1), (2, 3))


def extract_decision_boundary(
    field: PredictionField,
    *,
    threshold: float = 0.5,
) -> tuple[Segment, ...]:
    """Return deterministic marching-squares contour segments."""
    segments: list[Segment] = []

    for row in range(field.y_values.size - 1):
        for column in range(field.x_values.size - 1):
            x0 = float(field.x_values[column])
            x1 = float(field.x_values[column + 1])
            y0 = float(field.y_values[row])
            y1 = float(field.y_values[row + 1])

            bl = float(field.probabilities[row, column])
            br = float(field.probabilities[row, column + 1])
            tr = float(field.probabilities[row + 1, column + 1])
            tl = float(field.probabilities[row + 1, column])

            case = (
                (1 if bl >= threshold else 0)
                | (2 if br >= threshold else 0)
                | (4 if tr >= threshold else 0)
                | (8 if tl >= threshold else 0)
            )

            if case in (0, 15):
                continue

            corners = (
                ((x0, y0), bl),
                ((x1, y0), br),
                ((x1, y1), tr),
                ((x0, y1), tl),
            )

            edges = (
                (corners[0], corners[1]),
                (corners[1], corners[2]),
                (corners[3], corners[2]),
                (corners[0], corners[3]),
            )

            points: dict[int, Point] = {}

            for edge_index, (start, end) in enumerate(edges):
                start_point, start_value = start
                end_point, end_value = end

                if (
                    (start_value >= threshold)
                    == (end_value >= threshold)
                ):
                    continue

                points[edge_index] = _interpolate(
                    start_point,
                    end_point,
                    start_value,
                    end_value,
                    threshold,
                )

            if case in (5, 10):
                center_value = (bl + br + tr + tl) / 4.0
                pairs = _saddle_pairs(
                    case,
                    center_value,
                    threshold,
                )
            else:
                pairs = _EDGE_PAIRS[case]

            for first_edge, second_edge in pairs:
                segments.append(
                    (
                        points[first_edge],
                        points[second_edge],
                    )
                )

    return tuple(segments)
```

- [ ] **Step 4: Add determinism and saddle tests**

```python
def test_contour_extraction_is_deterministic() -> None:
    field = _field([
        [0.1, 0.8, 0.9],
        [0.2, 0.6, 0.3],
        [0.9, 0.4, 0.1],
    ])

    first = extract_decision_boundary(field)
    second = extract_decision_boundary(field)

    assert first == second


def test_saddle_case_is_resolved_deterministically() -> None:
    field = _field([
        [0.9, 0.1],
        [0.1, 0.9],
    ])

    segments = extract_decision_boundary(field)

    assert len(segments) == 2
    assert segments == extract_decision_boundary(field)
```

- [ ] **Step 5: Run quality gates**

```powershell
uv run pytest tests/test_decision_boundary.py -q
uv run ruff check src/neural_network_from_scratch/decision_boundary.py tests/test_decision_boundary.py
git diff --check
```

- [ ] **Step 6: Commit Task 2**

```powershell
git add src/neural_network_from_scratch/decision_boundary.py tests/test_decision_boundary.py
git diff --cached --check
git commit -m "feat: extract NN decision boundary"
```

---

### Task 3: Extend the Live View Telemetry Contract

**Files:**
- Modify: `src/neural_network_from_scratch/live_view.py`
- Modify: `tests/test_live_view.py`

**Interfaces:**
- Consumes: `LiveTrainingSession.loss_history`.
- Consumes: `extract_decision_boundary(PredictionField)`.
- Produces: `LiveViewState.loss_history: tuple[float, ...]`.
- Produces: `LiveViewState.decision_boundary: tuple[Segment, ...]`.
- Invariant: `LiveViewState.loss == LiveViewState.loss_history[-1]`.

- [ ] **Step 1: Write failing live-state contract tests**

Add to `tests/test_live_view.py`:

```python
from neural_network_from_scratch.decision_boundary import (
    extract_decision_boundary,
)


def test_live_view_contains_server_owned_loss_history() -> None:
    experiment = replace(
        linear_split_experiment(),
        epochs=2,
    )
    session = LiveTrainingSession(experiment)

    session.step()

    state = build_live_view_state(
        session,
        experiment,
        resolution=5,
    )

    assert state.loss_history == session.loss_history
    assert len(state.loss_history) == state.epoch + 1
    assert state.loss == state.loss_history[-1]


def test_live_view_boundary_uses_full_precision_prediction_field() -> None:
    experiment = linear_split_experiment()
    session = LiveTrainingSession(experiment)

    field = probe_prediction_field(
        session.parameters,
        resolution=5,
    )

    state = build_live_view_state(
        session,
        experiment,
        resolution=5,
    )

    assert state.decision_boundary == extract_decision_boundary(field)
```

Update the existing serialization expectation to include:

```python
"loss_history": state.loss_history,
"decision_boundary": state.decision_boundary,
```

- [ ] **Step 2: Verify RED**

```powershell
uv run pytest tests/test_live_view.py -q
```

Expected: failures for missing `loss_history` and `decision_boundary` fields.

- [ ] **Step 3: Extend `LiveViewState` and build from one full-precision field**

Add to `live_view.py`:

```python
from neural_network_from_scratch.decision_boundary import (
    Segment,
    extract_decision_boundary,
)
```

Extend the dataclass:

```python
@dataclass(frozen=True)
class LiveViewState:
    epoch: int
    loss: float
    loss_history: tuple[float, ...]
    is_complete: bool
    resolution: int
    field: str
    decision_boundary: tuple[Segment, ...]
```

Inside `build_live_view_state`, probe the full-precision field exactly once:

```python
field = probe_prediction_field(
    session.parameters,
    resolution=resolution,
)

loss_history = session.loss_history

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
)
```

Remove the duplicate current-loss `forward()` and `binary_cross_entropy()` calculation from `build_live_view_state`.

- [ ] **Step 4: Run focused and full tests**

```powershell
uv run pytest tests/test_live_view.py tests/test_live_training.py tests/test_decision_boundary.py -q
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q src
git diff --check
```

Expected: green.

- [ ] **Step 5: Commit Task 3**

```powershell
git add src/neural_network_from_scratch/live_view.py tests/test_live_view.py
git diff --cached --check
git commit -m "feat: expose live learning dynamics telemetry"
```

---

### Task 4: Render the Loss Curve and Decision Boundary

**Files:**
- Modify: `src/neural_network_from_scratch/live_observatory_html.py`
- Modify: `tests/test_live_observatory_html.py`

**Interfaces:**
- Consumes JSON `state.loss_history`.
- Consumes JSON `state.decision_boundary`.
- Produces a second browser canvas: `id="loss-canvas"`.
- Keeps the existing field canvas: `id="belief-canvas"`.

- [ ] **Step 1: Write failing browser contract tests**

Add to `tests/test_live_observatory_html.py`:

```python
def test_live_viewer_has_learning_dynamics_surfaces() -> None:
    html = render_live_observatory_html(
        linear_split_experiment(),
        resolution=5,
    )

    assert 'id="belief-canvas"' in html
    assert 'id="loss-canvas"' in html
    assert "function drawDecisionBoundary" in html
    assert "function drawLossHistory" in html


def test_browser_consumes_server_learning_dynamics() -> None:
    html = render_live_observatory_html(
        linear_split_experiment(),
        resolution=5,
    )

    assert "state.decision_boundary" in html
    assert "state.loss_history" in html

    assert "marchingSquares" not in html
    assert "computeDecisionBoundary" not in html
```

- [ ] **Step 2: Verify RED**

```powershell
uv run pytest tests/test_live_observatory_html.py -q
```

Expected: failures for missing loss canvas and rendering functions.

- [ ] **Step 3: Add the two-panel layout**

Update the viewer markup so the visual region contains:

```html
<div class="dynamics-grid">
    <div class="stage">
        <canvas
            id="belief-canvas"
            width="640"
            height="640"
            aria-label="Live neural network prediction field"
        ></canvas>
    </div>

    <div class="loss-stage">
        <canvas
            id="loss-canvas"
            width="480"
            height="640"
            aria-label="Live binary cross-entropy loss history"
        ></canvas>
    </div>
</div>
```

Use CSS equivalent to:

```css
.dynamics-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.85fr);
    gap: 1px;
    background: #2d3446;
}

.loss-stage {
    min-width: 0;
    background: #0b0e16;
}

#belief-canvas,
#loss-canvas {
    display: block;
    width: 100%;
    height: 100%;
}

@media (max-width: 900px) {
    .dynamics-grid {
        grid-template-columns: 1fr;
    }
}
```

- [ ] **Step 4: Render server-provided contour geometry**

Add after `drawField()`:

```javascript
function drawDecisionBoundary(segments) {
    context.save();
    context.beginPath();

    segments.forEach((segment) => {
        const start = segment[0];
        const end = segment[1];

        context.moveTo(
            start[0] * size,
            (1 - start[1]) * size
        );

        context.lineTo(
            end[0] * size,
            (1 - end[1]) * size
        );
    });

    context.lineWidth = 3;
    context.strokeStyle = "rgb(255,255,255)";
    context.stroke();
    context.restore();
}
```

Call it in `renderState` between the field and training points:

```javascript
drawField(values);
drawDecisionBoundary(state.decision_boundary);
drawTrainingPoints();
```

Do not derive crossings from `state.field` in JavaScript.

- [ ] **Step 5: Render the full linear loss history**

Initialize:

```javascript
const lossCanvas = document.getElementById("loss-canvas");
const lossContext = lossCanvas.getContext("2d");
```

Add:

```javascript
function drawLossHistory(history) {
    const width = lossCanvas.width;
    const height = lossCanvas.height;

    const left = 54;
    const right = 20;
    const top = 28;
    const bottom = 44;

    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;

    lossContext.clearRect(0, 0, width, height);

    lossContext.strokeStyle = "rgb(72,81,106)";
    lossContext.lineWidth = 1;
    lossContext.beginPath();
    lossContext.moveTo(left, top);
    lossContext.lineTo(left, height - bottom);
    lossContext.lineTo(width - right, height - bottom);
    lossContext.stroke();

    if (history.length === 0) {
        return;
    }

    const maxLoss = Math.max(...history, 1e-12);
    const maxEpoch = Math.max(history.length - 1, 1);

    lossContext.beginPath();

    history.forEach((loss, epoch) => {
        const x = left + (epoch / maxEpoch) * plotWidth;
        const y = (
            height
            - bottom
            - (loss / maxLoss) * plotHeight
        );

        if (epoch === 0) {
            lossContext.moveTo(x, y);
        } else {
            lossContext.lineTo(x, y);
        }
    });

    lossContext.strokeStyle = "rgb(82,230,255)";
    lossContext.lineWidth = 2;
    lossContext.stroke();

    lossContext.fillStyle = "rgb(155,166,190)";
    lossContext.font = "12px ui-monospace, monospace";
    lossContext.fillText("BCE loss", left, 18);
    lossContext.fillText(
        `epoch ${history.length - 1}`,
        width - right - 82,
        height - 14
    );
}
```

In `renderState`, call:

```javascript
drawLossHistory(state.loss_history);
```

The scale is linear. Do not introduce a browser-owned history array.

- [ ] **Step 6: Run viewer gates**

```powershell
uv run pytest tests/test_live_observatory_html.py -q
uv run ruff check src/neural_network_from_scratch/live_observatory_html.py tests/test_live_observatory_html.py
git diff --check
```

Expected: green.

- [ ] **Step 7: Commit Task 4**

```powershell
git add src/neural_network_from_scratch/live_observatory_html.py tests/test_live_observatory_html.py
git diff --cached --check
git commit -m "feat: render live learning dynamics"
```

---

### Task 5: Verify HTTP Integration and Final Acceptance

**Files:**
- Modify: `tests/test_live_server.py`
- No production behavior change is expected unless integration tests expose a defect.

**Interfaces:**
- `GET /api/state` returns current `loss_history` and `decision_boundary` without mutation.
- `POST /api/step` grows `loss_history` by exactly one item.
- HTTP 409 after completion leaves epoch and loss history unchanged.

- [ ] **Step 1: Add HTTP loss-history tests**

Add to `tests/test_live_server.py`:

```python
def test_get_state_returns_full_history_without_mutation() -> None:
    with _running_server(
        epochs=3,
        resolution=5,
    ) as (server, base_url):
        _post_json(f"{base_url}/api/step")
        _post_json(f"{base_url}/api/step")

        before_epoch = server.session.epoch
        before_history = server.session.loss_history

        state = _read_json(f"{base_url}/api/state")

        assert state["epoch"] == 2
        assert state["loss_history"] == list(before_history)
        assert state["loss"] == before_history[-1]
        assert server.session.epoch == before_epoch
        assert server.session.loss_history == before_history


def test_post_step_grows_live_history_exactly_once() -> None:
    with _running_server(
        epochs=3,
        resolution=5,
    ) as (_, base_url):
        initial = _read_json(f"{base_url}/api/state")
        stepped = _post_json(f"{base_url}/api/step")

        assert len(initial["loss_history"]) == 1
        assert len(stepped["loss_history"]) == 2
        assert stepped["epoch"] == 1
        assert stepped["loss"] == stepped["loss_history"][-1]
```

- [ ] **Step 2: Add boundary and completion immutability tests**

```python
def test_http_state_contains_server_boundary_geometry() -> None:
    with _running_server(
        resolution=5,
    ) as (_, base_url):
        state = _read_json(f"{base_url}/api/state")

        assert "decision_boundary" in state
        assert isinstance(state["decision_boundary"], list)


def test_completion_409_does_not_extend_live_history() -> None:
    with _running_server(
        epochs=1,
        resolution=5,
    ) as (server, base_url):
        completed = _post_json(f"{base_url}/api/step")
        completed_history = tuple(completed["loss_history"])

        request = Request(
            f"{base_url}/api/step",
            method="POST",
        )

        try:
            urlopen(request)
        except HTTPError as error:
            assert error.code == 409
        else:
            raise AssertionError("Expected HTTP 409")

        assert server.session.epoch == 1
        assert server.session.loss_history == completed_history
```

- [ ] **Step 3: Run server and deterministic-equivalence gates**

```powershell
uv run pytest tests/test_live_server.py -q
uv run pytest tests/test_live_training.py::test_repeated_steps_exactly_match_normal_training -q
```

Expected: green.

- [ ] **Step 4: Run full automated repository acceptance**

```powershell
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q src
git diff --check
git status --short
```

Expected: all tests pass and only expected Task 5 changes remain before commit.

- [ ] **Step 5: Commit HTTP integration tests**

```powershell
git add tests/test_live_server.py
git diff --cached --check
git commit -m "test: verify live learning dynamics integration"
```

- [ ] **Step 6: Launch the live Observatory for manual acceptance**

```powershell
uv run python -m neural_network_from_scratch.live_server
```

Verify manually:

1. Epoch 0 shows one loss-history point and the current field.
2. Step advances exactly one epoch and extends the graph by one point.
3. Train grows the complete curve continuously.
4. Pause stops scheduling further epochs.
5. Step while paused advances exactly one epoch.
6. The p = 0.5 boundary moves as the model learns.
7. The boundary remains visibly overlaid on the prediction field.
8. Refresh while paused preserves epoch, complete loss history, field, and boundary.
9. 1x, 5x, 10x, and Max still work.
10. Epoch 500 leaves the final curve and boundary visible and disables further training.

- [ ] **Step 7: Verify the post-completion HTTP boundary**

With the server still running at epoch 500:

```powershell
$baseUrl = "http://127.0.0.1:<CURRENT_PORT>"

$state = Invoke-WebRequest -Uri "$baseUrl/api/state" -Method Get -SkipHttpErrorCheck
Write-Host "STATE_STATUS=$($state.StatusCode)"

$response = Invoke-WebRequest -Uri "$baseUrl/api/step" -Method Post -SkipHttpErrorCheck
Write-Host "STEP_STATUS=$($response.StatusCode)"
Write-Host "STEP_BODY=$($response.Content)"
```

Expected:

```text
STATE_STATUS=200
STEP_STATUS=409
STEP_BODY={"error":"training session is complete"}
```

- [ ] **Step 8: Run final post-commit verification**

```powershell
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q src
git diff --check
git status --short
git log -7 --oneline
```

Expected: fully green with a clean worktree.

---

## Plan Self-Review

### Spec coverage

- Full epoch-0-to-current loss history: Task 1.
- Python/server authority: Tasks 1 and 3.
- Existing training-loss semantics preserved: Task 1.
- Linear loss chart: Task 4.
- Full-precision p = 0.5 contour: Tasks 2 and 3.
- Browser-only rendering: Task 4.
- Refresh reconstruction: Tasks 1, 3, and 5.
- HTTP behavior and 409 boundary: Task 5.
- Deterministic training equivalence: Tasks 1, 3, and 5.
- No neuron/weight/gradient scope expansion: Global Constraints.
- Full quality gates: Tasks 3, 4, and 5.

### Type consistency

- `loss_history` is `tuple[float, ...]` in Python and a JSON array in JavaScript.
- `decision_boundary` is `tuple[Segment, ...]` in Python and an array of two-point segments in JavaScript.
- `Segment` is `tuple[Point, Point]`.
- `Point` is `tuple[float, float]`.
- `LiveViewState.loss` is the final post-update `loss_history` entry.
- Existing `LiveTrainingSession.losses` remains unchanged and pre-update.

### Scope

This plan intentionally stops at learning dynamics. Internal neuron, activation, weight, and gradient visualization remains a later milestone.
