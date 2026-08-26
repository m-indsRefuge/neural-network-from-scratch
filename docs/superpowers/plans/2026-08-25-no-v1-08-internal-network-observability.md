# NO-V1-08 Internal Network Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the live Neural Observatory with coherent post-update internal
network telemetry and a polished light Canvas 2D scientific instrument.

**Architecture:** A new Python current-state probe forwards and backpropagates
only against the already-current parameters, then packages real activations,
aggregate statistics, parameters, and gradients into `LiveViewState`. The
browser consumes that payload only to project a 35-node/304-edge Canvas 2D
network, local selection/overlay state, and animation between received states;
it remains unable to calculate model truth.

**Tech Stack:** Python 3.12, NumPy, stdlib `http.server`, HTML/CSS, Canvas 2D,
and browser-native JavaScript. No additional runtime or development dependency.

**Spec:** `docs/superpowers/specs/2026-08-25-no-v1-08-internal-network-observability-design.md`

## Global Constraints

- Preserve `TrainingSnapshot` as a PRE-UPDATE observation and preserve
  `LiveTrainingSession.losses` semantics exactly.
- `LiveViewState` represents one POST-UPDATE epoch: its parameters,
  activations, gradients, field, contour, loss, and final loss-history value
  must refer to the same current model.
- The observation probe must never update parameters, advance epoch, append a
  history value, or otherwise mutate training.
- Python owns forward propagation, backward propagation, gradients, parameter
  values, prediction fields, and decision-boundary extraction.
- JavaScript only projects returned values, handles local UI state, and
  interpolates visual properties between received states.
- Keep the fixed `[2, 16, 16, 1]` architecture visible as 35 nodes and all 304
  actual weight edges. Do not add arbitrary-topology support.
- Keep dense prediction-field uint8/base64 transport and current linear BCE
  loss semantics.
- Do not add ML frameworks, WebGL, Three.js, historical internal replay,
  epoch scrubbing, or any package dependency.
- Use TDD: write and prove each focused failing test before its production
  behavior, then run the matching GREEN check and `uv run ruff check .` plus
  `git diff --check` before each task commit.
- Keep commits focused and unpushed; do not modify historical `.worktrees`.

## Planned File Structure

- Create `src/neural_network_from_scratch/network_telemetry.py`: typed
  current-state probe, fixed-shape validation, aggregate reductions, and JSON
  serialization of real internal values.
- Modify `src/neural_network_from_scratch/live_view.py`: attach valid
  `network` telemetry or explicit `network_error` to post-update live state.
- Modify `src/neural_network_from_scratch/live_observatory_html.py`: replace
  the dark two-panel viewer with the light three-instrument Canvas 2D surface,
  renderer, local interactions, and retained controls.
- Modify `tests/test_live_view.py` and `tests/test_live_server.py`: prove
  temporal coherence, non-mutation, HTTP serialization, and existing behavior.
- Create `tests/test_network_telemetry.py`: prove probe shapes, exact values,
  aggregate reductions, signed matrix preservation, and deterministic reads.
- Modify `tests/test_live_observatory_html.py`: prove durable browser
  structural and authority contracts without snapshotting CSS/HTML wholesale.

---

### Task 1: Add the Typed Current-State Network Telemetry Probe

**Files:**

- Create: `src/neural_network_from_scratch/network_telemetry.py`
- Create: `tests/test_network_telemetry.py`

**Interfaces:**

- Consumes: `Experiment`, current `Parameters`, handwritten `forward`, and
  handwritten `backward`.
- Produces: `probe_network_telemetry(experiment, parameters) -> NetworkTelemetry`.
- Produces: `NetworkTelemetry.to_dict() -> dict[str, object]` with numeric
  lists for topology, activations, summaries, parameters, and gradients.
- Produces: `NetworkTelemetryError(ValueError)` for invalid fixed-network
  telemetry shapes.

- [ ] **Step 1: Write failing probe-contract tests**

  Add tests with independent array checks:

  ```python
  def test_current_probe_exposes_the_fixed_complete_network() -> None:
      experiment = linear_split_experiment()
      parameters = initialize_parameters(experiment.seed)

      telemetry = probe_network_telemetry(experiment, parameters)

      assert telemetry.topology == (2, 16, 16, 1)
      assert telemetry.activations.input.shape == (8, 2)
      assert telemetry.activations.hidden_1.shape == (8, 16)
      assert telemetry.activations.hidden_2.shape == (8, 16)
      assert telemetry.activations.output.shape == (8, 1)
      assert telemetry.parameters.w2.shape == (16, 16)
      assert telemetry.gradients.w3.shape == (16, 1)


  def test_current_probe_aggregate_statistics_match_numpy() -> None:
      experiment = linear_split_experiment()
      telemetry = probe_network_telemetry(
          experiment, initialize_parameters(experiment.seed)
      )

      expected_mean = np.mean(telemetry.activations.hidden_1, axis=0)
      expected_min = np.min(telemetry.activations.hidden_1, axis=0)
      expected_max = np.max(telemetry.activations.hidden_1, axis=0)

      np.testing.assert_allclose(telemetry.activation_summary.hidden_1.mean, expected_mean)
      np.testing.assert_allclose(telemetry.activation_summary.hidden_1.min, expected_min)
      np.testing.assert_allclose(telemetry.activation_summary.hidden_1.max, expected_max)
      np.testing.assert_allclose(
          telemetry.activation_summary.hidden_1.spread, expected_max - expected_min
      )
  ```

  Also test all six parameter and gradient matrices serialize with their real
  signed values, repeated probes serialize identically, and a malformed
  `w1` shape raises `NetworkTelemetryError` rather than a fallback payload.

- [ ] **Step 2: Prove RED**

  Run:

  ```powershell
  uv run pytest tests/test_network_telemetry.py -q
  ```

  Expected: collection fails because `network_telemetry` and its probe do not
  exist yet; do not continue until that is the failure reason.

- [ ] **Step 3: Implement the smallest typed probe**

  Create frozen dataclasses with these public names and responsibilities:

  ```python
  @dataclass(frozen=True)
  class LayerActivationSummary:
      mean: np.ndarray
      min: np.ndarray
      max: np.ndarray
      spread: np.ndarray

  @dataclass(frozen=True)
  class NetworkActivations:
      input: np.ndarray
      hidden_1: np.ndarray
      hidden_2: np.ndarray
      output: np.ndarray

  @dataclass(frozen=True)
  class NetworkActivationSummary:
      input: LayerActivationSummary
      hidden_1: LayerActivationSummary
      hidden_2: LayerActivationSummary
      output: LayerActivationSummary

  @dataclass(frozen=True)
  class NetworkTelemetry:
      topology: tuple[int, int, int, int]
      activations: NetworkActivations
      activation_summary: NetworkActivationSummary
      parameters: Parameters
      gradients: Gradients
  ```

  Implement `probe_network_telemetry()` to validate all required parameter and
  input shapes, run `forward()` then `backward()` against the current supplied
  parameters, validate cache/gradient shapes, and reduce each layer on
  `axis=0` with `spread = max - min`. Implement explicit conversion helpers so
  `to_dict()` emits `input`, `hidden_1`, `hidden_2`, and `output` JSON keys
  plus all six parameter and six gradient matrices. Do not call
  `update_parameters()`.

- [ ] **Step 4: Prove GREEN and inspect the slice**

  Run:

  ```powershell
  uv run pytest tests/test_network_telemetry.py -q
  uv run ruff check .
  git diff --check
  ```

  Expected: focused tests pass, style is clean, and the diff contains only the
  new telemetry module and its tests.

- [ ] **Step 5: Commit Task 1**

  ```powershell
  git add src/neural_network_from_scratch/network_telemetry.py tests/test_network_telemetry.py
  git diff --cached --check
  git commit -m "feat: add current network telemetry probe"
  ```

---

### Task 2: Attach Coherent Post-Update Telemetry to Live State

**Files:**

- Modify: `src/neural_network_from_scratch/live_view.py`
- Modify: `tests/test_live_view.py`
- Modify: `tests/test_live_server.py`

**Interfaces:**

- Consumes: `probe_network_telemetry(experiment, session.parameters)`.
- Produces: `LiveViewState.network: NetworkTelemetry | None`.
- Produces: `LiveViewState.network_error: str | None`.
- Preserves: existing fields, `encode_probability_field`, post-update field
  construction, full-precision contour extraction, and HTTP endpoint shapes.

- [ ] **Step 1: Write failing temporal-coherence tests**

  Add to `tests/test_live_view.py`:

  ```python
  def test_live_view_after_step_uses_one_post_update_epoch_everywhere() -> None:
      experiment = replace(linear_split_experiment(), epochs=2)
      session = LiveTrainingSession(experiment)
      pre_update_snapshot = session.step()

      state = build_live_view_state(session, experiment, resolution=5)
      predictions, cache = forward(experiment.inputs, session.parameters)
      gradients = backward(experiment.targets, session.parameters, cache)

      assert state.epoch == 1
      assert state.loss == binary_cross_entropy(experiment.targets, predictions)
      assert state.loss == state.loss_history[-1]
      np.testing.assert_allclose(state.network.activations.output, predictions)
      np.testing.assert_allclose(state.network.gradients.w2, gradients.w2)
      assert state.network.parameters.w1.tolist() == session.parameters.w1.tolist()
      assert state.loss != pre_update_snapshot.loss
  ```

  Add a second test that calls `build_live_view_state()` twice at epoch 0 and
  asserts equal dictionaries while epoch, all parameter arrays, `losses`, and
  `loss_history` remain unchanged. Extend `test_live_server.py` with real
  loopback coverage:

  ```python
  def test_http_state_includes_deterministic_current_network_without_mutation() -> None:
      with _running_server(epochs=3, resolution=5) as (server, base_url):
          before_epoch = server.session.epoch
          before_parameters = server.session.parameters

          first = _read_json(f"{base_url}/api/state")
          second = _read_json(f"{base_url}/api/state")

          assert first["network"]["topology"] == [2, 16, 16, 1]
          assert first["network"] == second["network"]
          assert server.session.epoch == before_epoch
          _assert_parameters_equal(server.session.parameters, before_parameters)
  ```

  Add a post-step server assertion that uses direct current-parameter
  `forward`/`backward` values for the response's output activations and
  gradients. This belongs in Task 2 so it is written RED before live-state
  serialization exists.

- [ ] **Step 2: Prove RED**

  Run:

  ```powershell
  uv run pytest tests/test_live_view.py tests/test_live_server.py -q
  ```

  Expected: failures report missing `LiveViewState.network` and serialized
  network data; existing NO-V1-07 tests must not be edited to make RED pass.

- [ ] **Step 3: Integrate telemetry without changing training**

  Extend the frozen live-state dataclass and replace the generic `asdict()`
  serialization with an explicit `to_dict()` that preserves the exact existing
  JSON fields, appends `network` with `NetworkTelemetry.to_dict()`, and appends
  `network_error`. Build the existing field and decision boundary from current
  parameters, then invoke the probe on those same current parameters.

  Catch only `NetworkTelemetryError` at the telemetry boundary. Preserve valid
  field/loss state and return `network=None` plus a stable descriptive
  `network_error`; do not mask unrelated HTTP/server failures or mutate the
  session. The server endpoints continue to use `build_live_view_state()` with
  no new inspection endpoint.

- [ ] **Step 4: Prove GREEN and regression compatibility**

  Run:

  ```powershell
  uv run pytest tests/test_live_view.py tests/test_live_server.py tests/test_live_training.py -q
  uv run ruff check .
  git diff --check
  ```

  Expected: the new post-update assertions pass and old pre-update snapshot
  assertions stay green.

- [ ] **Step 5: Commit Task 2**

  ```powershell
  git add src/neural_network_from_scratch/live_view.py tests/test_live_view.py tests/test_live_server.py
  git diff --cached --check
  git commit -m "feat: expose coherent live network telemetry"
  ```

---

### Task 3: Establish the Light Scientific Observatory Shell

**Files:**

- Modify: `src/neural_network_from_scratch/live_observatory_html.py`
- Modify: `tests/test_live_observatory_html.py`

**Interfaces:**

- Consumes: immutable experiment `resolution`, `training_inputs`,
  `training_targets`, and `learning_rate`; live `state` remains fetched.
- Produces: Canvas ids `network-canvas`, `belief-canvas`, and `loss-canvas`.
- Produces: IDs `aggregate-button`, `weight-overlay-toggle`,
  `gradient-overlay-toggle`, `network-inspection`, and `network-error`.
- Preserves: `step-button`, `train-button`, `pause-button`, `speed-select`,
  `epoch-value`, `loss-value`, `status-message`, and the existing endpoint
  calls/speed values.

- [ ] **Step 1: Write failing structural UI tests**

  Add tests that assert the three Canvas instruments, the Aggregate control,
  local overlay controls, inspection/error containers, the retained live
  controls, and semantic light-theme token names exist:

  ```python
  def test_live_viewer_has_scientific_network_instruments() -> None:
      html = render_live_observatory_html(linear_split_experiment(), resolution=5)

      assert 'id="network-canvas"' in html
      assert 'id="belief-canvas"' in html
      assert 'id="loss-canvas"' in html
      assert 'id="aggregate-button"' in html
      assert 'id="weight-overlay-toggle"' in html
      assert 'id="gradient-overlay-toggle"' in html
      assert 'id="network-inspection"' in html
      assert 'id="network-error"' in html
      assert "--surface-page" in html
      assert "--accent-positive" in html
      assert "@media" in html
  ```

  Update immutable-data expectations to include `learning_rate` and preserve
  the existing input/target/resolution assertions.

- [ ] **Step 2: Prove RED**

  Run:

  ```powershell
  uv run pytest tests/test_live_observatory_html.py -q
  ```

  Expected: tests fail for the absent network Canvas and light-theme/control
  contracts, not for a malformed test fixture.

- [ ] **Step 3: Rebuild the semantic shell and token system**

  Replace the dark page styles with a single `:root` token set including
  `--surface-page`, `--surface-panel`, `--surface-inset`, `--text-primary`,
  `--text-secondary`, `--text-muted`, `--accent-positive`,
  `--accent-negative`, `--accent-neutral`, `--accent-selection`,
  `--border-subtle`, `--shadow-panel`, and `--shadow-neuron`.

  Build semantic network, decision-surface, and learning-dynamics panels with
  the network first in DOM order. Place existing runtime readouts in the
  header, retain the existing controls and speed values, add local aggregate
  and overlay controls, and add visually hidden or inert initial inspection
  and error containers. Include `learning_rate` in the immutable JSON script.
  Use CSS grid to make the network primary on desktop and stack the network,
  decision surface, then loss surface on narrow screens.

- [ ] **Step 4: Prove GREEN**

  Run:

  ```powershell
  uv run pytest tests/test_live_observatory_html.py -q
  uv run ruff check .
  git diff --check
  ```

  Expected: the DOM contract passes without relying on a full HTML snapshot.

- [ ] **Step 5: Commit Task 3**

  ```powershell
  git add src/neural_network_from_scratch/live_observatory_html.py tests/test_live_observatory_html.py
  git diff --cached --check
  git commit -m "feat: add scientific Observatory shell"
  ```

---

### Task 4: Render the Complete Aggregate Network from Real Telemetry

**Files:**

- Modify: `src/neural_network_from_scratch/live_observatory_html.py`
- Modify: `tests/test_live_observatory_html.py`

**Interfaces:**

- Consumes: `state.network.topology`, `activations`, `activation_summary`,
  and `parameters` supplied by Task 2.
- Produces: browser functions `validateNetworkTelemetry`, `buildNetworkLayout`,
  `drawNetwork`, `drawNetworkEdges`, and `drawNetworkNodes`.
- Produces: fixed display constants for weight magnitude rather than state-wide
  normalization.

- [ ] **Step 1: Write failing renderer-authority tests**

  Add focused structural checks:

  ```python
  def test_network_renderer_consumes_all_real_network_layers() -> None:
      html = render_live_observatory_html(linear_split_experiment(), resolution=5)

      assert "function validateNetworkTelemetry" in html
      assert "function buildNetworkLayout" in html
      assert "function drawNetworkEdges" in html
      assert "function drawNetworkNodes" in html
      assert "state.network.parameters.w1" in html
      assert "state.network.parameters.w2" in html
      assert "state.network.parameters.w3" in html
      assert "state.network.activation_summary" in html
      assert "WEIGHT_DISPLAY_SCALE" in html
  ```

  Add a separate test that expects the renderer to reject a non-
  `[2, 16, 16, 1]` telemetry layout through the explicit error path rather
  than selecting a decorative subset.

- [ ] **Step 2: Prove RED**

  Run:

  ```powershell
  uv run pytest tests/test_live_observatory_html.py -q
  ```

  Expected: missing renderer functions and real-matrix consumers cause RED.

- [ ] **Step 3: Implement aggregate Canvas projection**

  Add a fixed layer-layout function for exactly 2/16/16/1 nodes. Iterate every
  entry of `w1`, `w2`, and `w3` to draw all 304 actual edges; retain a subdued
  baseline layer so dense edges remain legible. Use a fixed nonlinear
  `weightPresentation(abs(weight))` scale and real signed weight hues.

  Use Python-provided aggregate `mean` for node center intensity and
  Python-provided `spread` for the aggregate outer ring. Draw all 35 radial
  nodes with restrained depth offsets and highlights. `validateNetworkTelemetry`
  checks only required JSON structure and fixed dimensions, then reports a
  readable network-instrument error if it cannot safely project the payload.

- [ ] **Step 4: Prove GREEN and inspect source authority**

  Run:

  ```powershell
  uv run pytest tests/test_live_observatory_html.py -q
  uv run ruff check .
  git diff --check
  ```

  Expected: tests prove the renderer reads all real matrices and aggregate
  inputs; no test asserts incidental pixel coordinates or full style text.

- [ ] **Step 5: Commit Task 4**

  ```powershell
  git add src/neural_network_from_scratch/live_observatory_html.py tests/test_live_observatory_html.py
  git diff --cached --check
  git commit -m "feat: render aggregate network telemetry"
  ```

---

### Task 5: Add Gradient, Bias, Inspection, and Visual-Interpolation Channels

**Files:**

- Modify: `src/neural_network_from_scratch/live_observatory_html.py`
- Modify: `tests/test_live_observatory_html.py`

**Interfaces:**

- Consumes: `state.network.gradients`, non-input parameter biases, and immutable
  `data.learning_rate`.
- Produces: `drawGradientOverlay`, `renderNetworkInspection`, hover hit tests,
  `queueServerState`, and `requestAnimationFrame` presentation transitions.
- Preserves: only server responses establish model states and step requests
  remain sequential.

- [ ] **Step 1: Write failing distinct-channel and no-math tests**

  Add tests asserting real gradient/bias/learning-rate consumers and local
  animation mechanisms:

  ```python
  def test_network_renderer_has_distinct_real_gradient_and_inspection_channels() -> None:
      html = render_live_observatory_html(linear_split_experiment(), resolution=5)

      assert "function drawGradientOverlay" in html
      assert "state.network.gradients.w1" in html
      assert "state.network.gradients.w2" in html
      assert "state.network.gradients.w3" in html
      assert "data.learning_rate" in html
      assert "function renderNetworkInspection" in html
      assert "requestAnimationFrame" in html
  ```

  Add a policy test that rejects browser-owning names or algorithms such as
  `function forward`, `function backward`, `function sigmoid`,
  `function computeGradients`, `marchingSquares`, and
  `computeDecisionBoundary`.

- [ ] **Step 2: Prove RED**

  Run:

  ```powershell
  uv run pytest tests/test_live_observatory_html.py -q
  ```

  Expected: missing gradient/inspection functions cause failures; do not change
  the forbidden-name assertions to fit implementation shortcuts.

- [ ] **Step 3: Implement truthful secondary channels**

  Draw a thin gradient halo per actual edge with a separate fixed gradient
  scale. Add compact non-input bias indicators supplied by `b1`, `b2`, and
  `b3`; inspection reports node activation/statistics, bias, and bias gradient
  or edge source/destination, signed weight, signed gradient, and
  `-data.learning_rate * gradient`.

  Implement Canvas hover hit tests that emphasize the related real edges and
  de-emphasize unrelated edges. Maintain a FIFO received-state queue and use
  `requestAnimationFrame` solely to interpolate paint properties between its
  adjacent real payloads. Never calculate an activation, gradient, parameter,
  field, or contour in JavaScript.

- [ ] **Step 4: Prove GREEN**

  Run:

  ```powershell
  uv run pytest tests/test_live_observatory_html.py -q
  uv run ruff check .
  git diff --check
  ```

  Expected: the browser contract demonstrates distinct source-backed channels
  and keeps the banned mathematical implementations absent.

- [ ] **Step 5: Commit Task 5**

  ```powershell
  git add src/neural_network_from_scratch/live_observatory_html.py tests/test_live_observatory_html.py
  git diff --cached --check
  git commit -m "feat: add network inspection overlays"
  ```

---

### Task 6: Add Exact Sample Mode and Retheme the Decision Surface

**Files:**

- Modify: `src/neural_network_from_scratch/live_observatory_html.py`
- Modify: `tests/test_live_observatory_html.py`

**Interfaces:**

- Consumes: immutable `training_inputs`, `training_targets`, and existing
  Python-provided batch activation rows.
- Produces: local `selectedSampleIndex`, `pickTrainingSample`, and Aggregate
  reset behavior.
- Preserves: field base64 decoding, server-provided decision boundary,
  coordinate orientation, and no selection-related HTTP request.

- [ ] **Step 1: Write failing sample-mode behavior tests**

  Add lightweight UI-contract tests:

  ```python
  def test_sample_selection_is_local_and_uses_python_batch_rows() -> None:
      html = render_live_observatory_html(linear_split_experiment(), resolution=5)

      assert "let selectedSampleIndex = null" in html
      assert "function pickTrainingSample" in html
      assert "state.network.activations.input[selectedSampleIndex]" in html
      assert "state.network.activations.hidden_1[selectedSampleIndex]" in html
      assert "state.network.activations.hidden_2[selectedSampleIndex]" in html
      assert "state.network.activations.output[selectedSampleIndex]" in html
      assert "function setAggregateMode" in html
  ```

  Assert that sample selection functions do not contain `fetch("/api/step"`,
  `loadState()`, or writes to server-owned state. Assert selected-sample ring
  rendering is present in the decision surface.

- [ ] **Step 2: Prove RED**

  Run:

  ```powershell
  uv run pytest tests/test_live_observatory_html.py -q
  ```

  Expected: failures identify absent selection state and exact row projection.

- [ ] **Step 3: Implement local sample selection and surface retheme**

  Add decision-surface hit testing against the immutable point positions. A hit
  sets the local index; a hit on the selected index or Aggregate button clears
  it. `renderState()` preserves a valid selected index as later states arrive.
  `drawNetwork()` chooses either Python summaries or exactly one existing
  Python activation row; it labels the selected immutable target without a new
  inference request.

  Retheme `drawField`, `drawDecisionBoundary`, and `drawTrainingPoints` with
  the shared light scientific palette. Preserve field data, p = 0.5 contour
  geometry, coordinate mapping, and a highly visible selection ring.

- [ ] **Step 4: Prove GREEN**

  Run:

  ```powershell
  uv run pytest tests/test_live_observatory_html.py -q
  uv run ruff check .
  git diff --check
  ```

  Expected: local-only sample selection and server-owned field/contour
  contracts are both covered.

- [ ] **Step 5: Commit Task 6**

  ```powershell
  git add src/neural_network_from_scratch/live_observatory_html.py tests/test_live_observatory_html.py
  git diff --cached --check
  git commit -m "feat: add Observatory sample mode"
  ```

---

### Task 7: Retheme Learning Dynamics, Controls, and Network Error States

**Files:**

- Modify: `src/neural_network_from_scratch/live_observatory_html.py`
- Modify: `tests/test_live_observatory_html.py`

**Interfaces:**

- Consumes: existing `state.loss_history`, `state.is_complete`,
  `state.network_error`, and current control state.
- Produces: light linear BCE chart, local overlay toggles, and explicit network
  error display.
- Preserves: visible epoch-0 marker, 1×/5×/10×/Max controls, Step/Train/Pause
  state machine, completion behavior, and browser refresh semantics.

- [ ] **Step 1: Write failing regression and resilience tests**

  Add tests that require these durable contracts:

  ```python
  def test_light_dynamics_and_controls_consume_authoritative_live_state() -> None:
      html = render_live_observatory_html(linear_split_experiment(), resolution=5)

      assert "function drawLossHistory" in html
      assert "state.loss_history" in html
      assert "lossContext.arc(" in html
      assert "state.network_error" in html
      assert "function renderNetworkError" in html
      assert 'id="weight-overlay-toggle"' in html
      assert 'id="gradient-overlay-toggle"' in html
      assert 'value="1000"' in html
      assert 'value="200"' in html
      assert 'value="100"' in html
      assert 'value="0"' in html
  ```

  Keep tests for `fetch("/api/state")`, `fetch("/api/step"`, POST, and all
  retained controls. Do not assert every color or whitespace token.

- [ ] **Step 2: Prove RED**

  Run:

  ```powershell
  uv run pytest tests/test_live_observatory_html.py -q
  ```

  Expected: missing error display or local overlay control behavior fails.

- [ ] **Step 3: Finish synchronized secondary instruments**

  Redraw the BCE chart with light panel background, subtle grid/axes, clear
  linear scale, retained full history, current epoch marker, and current-loss
  marker at epoch 0. Wire local checkbox/button overlay state into
  `drawNetwork()` only. Render a visible role-alert error when `network_error`
  is non-null or the browser's structural projection validation fails, while
  field/loss/control rendering remains available.

  Verify the Train loop still sends only sequential step requests, Pause only
  stops future scheduling, and completion disables mutation without altering
  server state.

- [ ] **Step 4: Prove GREEN**

  Run:

  ```powershell
  uv run pytest tests/test_live_observatory_html.py -q
  uv run ruff check .
  git diff --check
  ```

  Expected: secondary instrument and control contracts remain explicit and
  NO-V1-07 behavior continues to pass.

- [ ] **Step 5: Commit Task 7**

  ```powershell
  git add src/neural_network_from_scratch/live_observatory_html.py tests/test_live_observatory_html.py
  git diff --cached --check
  git commit -m "feat: complete scientific Observatory controls"
  ```

---

### Task 8: Run Complete-State Integration and Live Acceptance

**Files:**

- No planned source or test edit. This is the fresh integrated validation and
  manual/live-acceptance task after Tasks 1–7 have completed.

**Interfaces:**

- Consumes: actual `LiveObservatoryServer` over loopback HTTP.
- Proves: the TDD contracts introduced in Task 2 and the browser contracts
  introduced in Tasks 3–7 survive a fresh end-to-end run.

- [ ] **Step 1: Run the fresh integrated automated test set**

  Run the task-specific test set that was written RED in Tasks 1–7:

  ```powershell
  uv run pytest tests/test_network_telemetry.py tests/test_live_view.py tests/test_live_server.py tests/test_live_observatory_html.py -q
  ```

  Expected: all task-owned tests pass against the integrated worktree. If a
  failure reveals a defect, stop this validation task, return to the earliest
  responsible implementation task, write a focused RED regression test there,
  and make the minimal repair before rerunning this task.

- [ ] **Step 2: Run the live browser/manual acceptance**

  Start the actual server:

  ```powershell
  uv run python -m neural_network_from_scratch.live_server
  ```

  If browser or Obscura MCP tooling is available in this execution environment,
  first prove availability and use it. Otherwise provide the operator exact
  manual steps covering epoch 0, one Step, aggregate inspection, sample
  selection, selected-sample Train follow, node/edge hover, overlay toggles,
  Pause/Step, all speed modes, refresh, completion at epoch 500, and the exact
  post-completion 409 body.

- [ ] **Step 3: Verify the post-completion boundary without mutation**

  Once the live session reaches epoch 500, issue one read and one rejected
  mutation request from PowerShell:

  ```powershell
  $baseUrl = "http://127.0.0.1:<CURRENT_PORT>"
  $state = Invoke-WebRequest -Uri "$baseUrl/api/state" -Method Get -SkipHttpErrorCheck
  $response = Invoke-WebRequest -Uri "$baseUrl/api/step" -Method Post -SkipHttpErrorCheck
  Write-Host "STATE_STATUS=$($state.StatusCode)"
  Write-Host "STEP_STATUS=$($response.StatusCode)"
  Write-Host "STEP_BODY=$($response.Content)"
  ```

  Expected: `STATE_STATUS=200`, `STEP_STATUS=409`, and
  `STEP_BODY={"error":"training session is complete"}`. Compare current
  state before/after rejection if live automation is available; the existing
  loopback test covers parameter/history immutability.

- [ ] **Step 4: Run final clean-branch gate and review**

  Run fresh:

  ```powershell
  uv run pytest -q
  uv run ruff check .
  uv run python -m compileall -q src
  git diff --check
  git diff --cached --check
  git status --short
  git log --oneline --decorate -12
  git diff main...HEAD --check
  git diff --stat main...HEAD
  ```

  Expected: all intended work is committed, the feature worktree is clean, the
  branch remains `build/no-v1-08-internal-network-observability`, and no
  history, push, PR, merge, or cleanup action occurs.

---

## Plan Self-Review

### Spec coverage

- PRE-UPDATE snapshot and existing losses semantics: Global Constraints and
  Tasks 2 and 8.
- Post-update coherence/current-state non-mutation: Tasks 1, 2, and 8.
- Full typed network telemetry and Python aggregate meaning: Tasks 1 and 2.
- Exact 35-node/304-edge Canvas 2D presentation: Task 4.
- Weight sign, fixed magnitude mapping, gradients, biases, hover, and
  requestAnimationFrame-only visual interpolation: Task 5.
- Aggregate/sample modes, exact batch rows, selection persistence, and decision
  field semantics: Task 6.
- Entire light scientific theme, loss chart, controls, responsive behavior, and
  explicit instrument errors: Tasks 3 and 7.
- No browser mathematical authority: Tasks 4, 5, and 7 tests.
- Existing HTTP/completion behavior and full regression/manual acceptance:
  Task 8.
- No historical internal replay or expanded topology/framework scope: Global
  Constraints.

### Placeholder scan

The plan has no unresolved placeholders. Every implementation task names its
exact files, input/output interfaces, RED command, GREEN command, and focused
commit boundary. Task 8 intentionally makes no production edit; a discovered
integration defect returns to the earliest responsible implementation task for
its own focused RED/GREEN repair, preserving the scope boundary.

### Type consistency

- `NetworkTelemetry` owns raw NumPy arrays in Python and `to_dict()` creates
  JSON numeric lists; `LiveViewState.network` is typed or null with
  `network_error`.
- Browser keys are `input`, `hidden_1`, `hidden_2`, and `output`, matching the
  telemetry serializer and sample-mode indexing.
- Parameter and gradient keys are the existing `w1`, `b1`, `w2`, `b2`, `w3`,
  and `b3` names.
- The fixed topology is a Python tuple and an equivalent JSON array.
- `LiveViewState.loss` continues to equal the final post-update
  `loss_history` value.

### Scope

The plan deliberately stops after current-state observability. It creates no
historical parameter store, no epoch scrubbing, no topology editor, no new ML
framework, no WebGL surface, and no unrelated repository cleanup.
