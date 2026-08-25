# NO-V1-06B Live Observatory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Implement task-by-task using RED → GREEN → regression gate → commit.

**Goal:** Connect the stepwise NN-01 runtime to a browser Observatory that observes real training over localhost HTTP.

**Architecture:** Python owns the experiment, LiveTrainingSession, loss, inference, prediction fields, and all mutation. A single-threaded standard-library HTTP server exposes current state and one-step mutation. The browser renders one reusable Canvas and controls Step, Train, Pause, and observation speed.

**Tech Stack:** Python 3.12, NumPy, http.server, urllib, HTML/CSS, vanilla JavaScript, Canvas 2D.

**Spec:** docs/superpowers/specs/2026-08-25-no-v1-06b-live-observatory-design.md

## Global Constraints

- Bind only to 127.0.0.1.
- No new runtime dependencies.
- Use HTTPServer, not ThreadingHTTPServer.
- Python remains authoritative for training.
- Preserve LiveTrainingSession.step() semantics.
- Browser state is post-update state.
- Quantization affects visual transport only.
- Use one Canvas and bounded browser memory.
- Step performs exactly one epoch.
- Train sends sequential requests only.
- Pause schedules no future request.
- Speeds: 1x, 5x, 10x, Max.
- No Tauri, React, WebSockets, SSE, persistence, or background trainer.
- Linear Split only for this milestone.
- TDD for every production change.

---

## Task 1 — Post-Update Live View State

**Create**
- src/neural_network_from_scratch/live_view.py
- tests/test_live_view.py

**Produce**

    @dataclass(frozen=True)
    class LiveViewState:
        epoch: int
        loss: float
        is_complete: bool
        resolution: int
        field: str

    def encode_probability_field(
        probabilities: np.ndarray,
    ) -> str

    def build_live_view_state(
        session: LiveTrainingSession,
        experiment: Experiment,
        *,
        resolution: int,
    ) -> LiveViewState

### RED

Write tests proving:

1. Epoch-zero view uses initialized session parameters.
2. Post-step view uses updated session parameters.
3. Post-step live loss differs from the pre-update TrainingSnapshot loss where expected.
4. 5x5 field encodes exactly 25 uint8 values.
5. Encoding equals:

    np.rint(
        np.clip(probabilities, 0.0, 1.0) * 255.0
    ).astype(np.uint8)

6. Encoding is deterministic.
7. Final configured epoch reports is_complete=True.

Run:

    uv run pytest tests/test_live_view.py -q

Expected RED: live_view module does not exist.

### GREEN

Implement encode_probability_field with:

    base64.b64encode(
        quantized.tobytes(order="C")
    ).decode("ascii")

build_live_view_state must:

1. read session.parameters;
2. forward the training inputs;
3. compute current BCE loss;
4. probe the prediction field;
5. encode that field;
6. return LiveViewState.

It must never call session.step().

Run:

    uv run pytest tests/test_live_view.py -q
    uv run pytest -q
    uv run ruff check .
    uv run python -m compileall -q src
    git diff --check

Commit:

    git add src/neural_network_from_scratch/live_view.py tests/test_live_view.py
    git diff --cached --check
    git commit -m "feat: add live Observatory view state"

---

## Task 2 — Live Browser Viewer

**Create**
- src/neural_network_from_scratch/live_observatory_html.py
- tests/test_live_observatory_html.py

**Produce**

    def render_live_observatory_html(
        experiment: Experiment,
        *,
        resolution: int,
    ) -> str

### RED

Tests must prove:

1. Exactly one id="belief-canvas".
2. No recorded evolution frames.
3. No embedded prediction-frame history.
4. Immutable metadata contains resolution, training_inputs, training_targets.
5. Controls exist:
   - step-button
   - train-button
   - pause-button
   - speed-select
6. Speed values exist:
   - 1000
   - 200
   - 100
   - 0
7. Epoch, loss, and status readouts exist.
8. JavaScript references GET /api/state.
9. JavaScript references POST /api/step.
10. Output is deterministic.

Run:

    uv run pytest tests/test_live_observatory_html.py -q

Expected RED: live_observatory_html module does not exist.

### GREEN

The viewer must:

- draw the existing purple-to-cyan field;
- overlay pink/cyan training points;
- base64-decode field data into Uint8Array;
- repaint the same Canvas;
- fetch /api/state on startup;
- issue one POST /api/step for Step;
- keep automatic Train requests sequential;
- stop scheduling on Pause;
- allow an in-flight request to complete;
- stop Train on request errors;
- display a visible error message;
- disable Step and Train at completion.

Train loop:

    request step
    receive response
    render
    wait selected delay
    request next step

Delays:

    1x   1000 ms
    5x    200 ms
    10x   100 ms
    Max     0 ms

Run:

    uv run pytest tests/test_live_observatory_html.py -q
    uv run pytest -q
    uv run ruff check .
    uv run python -m compileall -q src
    git diff --check

Commit:

    git add src/neural_network_from_scratch/live_observatory_html.py tests/test_live_observatory_html.py
    git diff --cached --check
    git commit -m "feat: add live Observatory browser controls"

---

## Task 3 — Localhost HTTP Bridge

**Create**
- src/neural_network_from_scratch/live_server.py
- tests/test_live_server.py

**Produce**

    class LiveObservatoryServer(HTTPServer)

    def create_live_server(
        *,
        experiment: Experiment | None = None,
        port: int = 0,
        resolution: int = 101,
    ) -> LiveObservatoryServer

    def main() -> None

create_live_server must always bind:

    ("127.0.0.1", port)

No host parameter is permitted.

### RED

Use a test-only daemon Thread to run serve_forever while urllib acts as the HTTP client.

Tests must prove:

1. Server binds to 127.0.0.1.
2. GET / returns live viewer HTML.
3. GET /api/state returns epoch zero without mutation.
4. POST /api/step advances exactly once.
5. Two POST requests advance epochs 1 then 2.
6. Completion response reports is_complete=True.
7. POST after completion returns HTTP 409.
8. Post-completion request does not mutate epoch.
9. Unknown route returns 404.
10. Unsupported method on a known route returns 405.
11. HTTP-driven stepping produces final parameters exactly equal to run_experiment.

Run:

    uv run pytest tests/test_live_server.py -q

Expected RED: live_server module does not exist.

### GREEN

Use:

    from http.server import HTTPServer, BaseHTTPRequestHandler

The server owns:

    experiment
    session = LiveTrainingSession(experiment)
    resolution
    viewer_html

Routes:

    GET  /
    GET  /api/state
    POST /api/step

GET /api/state calls build_live_view_state only.

POST /api/step:

    if session.is_complete:
        return 409

    session.step()

    state = build_live_view_state(
        session,
        experiment,
        resolution=resolution,
    )

    return 200 with state

Use compact deterministic JSON with exact Content-Length.

Known path plus wrong method returns 405.
Unknown path returns 404.
Unexpected application errors return JSON 500 without replacing the session.

Override log_message() to suppress default test-server logging.

main() must print:

    NEURAL_OBSERVATORY_URL=http://127.0.0.1:<port>/

then call serve_forever(), allowing Ctrl+C to terminate normally.

Run:

    uv run pytest tests/test_live_server.py -q
    uv run pytest -q
    uv run ruff check .
    uv run python -m compileall -q src
    git diff --check

Commit:

    git add src/neural_network_from_scratch/live_server.py tests/test_live_server.py
    git diff --cached --check
    git commit -m "feat: stream live NN training over localhost"

---

## Manual Acceptance

Launch:

    uv run python -m neural_network_from_scratch.live_server

Then open the printed 127.0.0.1 URL.

Verify:

1. Epoch-zero field appears immediately.
2. Step changes epoch 0 → 1.
3. A second Step changes 1 → 2.
4. Train at 1x visibly advances approximately once per second.
5. Pause freezes further scheduling.
6. Step while paused advances exactly one epoch.
7. 5x operates around 200 ms delay.
8. 10x operates around 100 ms delay.
9. Max adds no intentional delay but remains sequential.
10. Refresh while paused preserves server-owned epoch.
11. Training reaches epoch 500.
12. Completion disables Step and Train.
13. Final field remains visible.
14. Further POST /api/step returns 409.
15. Final automated gates remain green.

Final gate:

    uv run pytest -q
    uv run ruff check .
    uv run python -m compileall -q src
    git diff --check
    git status --short

## Definition of Done

NO-V1-06B is complete when every displayed trained state is produced on demand by a real NN-01 gradient-descent step in the running Python process, and both automated and manual acceptance pass.