# NO-V1-08 — Internal Network Observability + Scientific Visual System

Date: 2026-08-25

Status: Approved implementation design

## Purpose

NO-V1-08 turns the live Neural Observatory into a truthful instrument for the
fixed handwritten NN-01 classifier. It adds current internal-network telemetry
and a permanent light scientific visual system so the observer can inspect the
same post-update epoch through its full network, prediction field, decision
boundary, BCE history, and controls.

The network remains the fixed `2 → 16 → 16 → 1` NN-01 architecture. The
instrument must visibly represent all 35 nodes and all 304 weight connections,
with the model's 337 trainable parameters still authoritative in Python.

## Scope

NO-V1-08 includes:

- a typed, non-mutating current-state network probe;
- real batch activations, aggregate activation statistics, parameters, and
  analytical gradients in the browser-facing live state;
- Python-side shape validation and an explicit instrument-error payload;
- a Canvas 2D network renderer with restrained 2.5D presentation;
- real aggregate and exact selected-sample views;
- full weight, gradient, bias, hover, and inspection encodings;
- a light scientific laboratory theme across the entire live Observatory;
- synchronized decision-surface, BCE-history, header, and control migration;
- responsive layout and preserved NO-V1-07 live-control behavior.

NO-V1-08 does not include:

- any training-algorithm, optimizer, initialization, or dependency change;
- PyTorch, TensorFlow, JAX, automatic differentiation, WebGL, or Three.js;
- historical parameter, activation, or gradient replay;
- epoch scrubbing, topology editing, arbitrary architecture support, or
  neuron-specific historical charts;
- browser-side forward propagation, backpropagation, gradient calculation,
  parameter derivation, or decision-boundary extraction.

## Authority and Temporal Semantics

Python remains the only mathematical authority. JavaScript receives numeric
telemetry from Python and owns only layout, hit testing, visual projection,
selection, overlay visibility, and animation between adjacent received states.

`TrainingSnapshot` preserves its current PRE-UPDATE meaning exactly. During
`LiveTrainingSession.step()`, it records the parameters, forward cache,
gradients, and loss before `update_parameters()` runs. Existing consumers and
the existing `LiveTrainingSession.losses` sequence keep that behavior.

The browser never displays that pre-update snapshot as live state. The existing
`LiveTrainingSession.loss_history` remains post-update telemetry: it contains
the epoch-0 current-model loss, then one loss for every completed update. The
existing `LiveViewState.loss` remains the final entry of that history.

For a browser response at completed epoch `N`, the following must all describe
the current post-update parameters at epoch `N`:

- `LiveViewState.epoch`;
- `LiveViewState.loss` and the final `loss_history` item;
- the prediction field and its p = 0.5 contour;
- the network parameters and biases;
- the training-batch activations;
- the analytical gradients calculated at those current parameters.

The resulting invariant is:

```text
epoch N
  = current parameters and biases
  = current forward cache on the training batch
  = current analytical gradients on that cache
  = current prediction field
  = current decision boundary
  = current post-update BCE
```

## Current-State Network Probe

A dedicated typed probe, separate from `TrainingSnapshot`, will observe the
current model without changing `LiveTrainingSession` state. It will:

1. read the current `session.parameters`;
2. call the handwritten `forward(experiment.inputs, parameters)` once;
3. call the handwritten `backward(experiment.targets, parameters, cache)` once;
4. derive aggregate statistics from that exact cache; and
5. serialize the resulting signed numeric values for `LiveViewState`.

The probe never calls `update_parameters()`, `LiveTrainingSession.step()`, or
any other mutation. Repeated state requests with unchanged session state must
therefore produce equal network payloads and leave epoch, parameters,
`losses`, and `loss_history` untouched.

The probe validates the fixed NN-01 contract before constructing telemetry:

```text
inputs       (samples, 2)
a1           (samples, 16)
a2           (samples, 16)
predictions  (samples, 1)

w1           (2, 16)      b1 (1, 16)
w2           (16, 16)     b2 (1, 16)
w3           (16, 1)      b3 (1, 1)
```

Corresponding gradient arrays must have the same shapes. Inputs, targets, and
all activation arrays must agree on the sample count. A malformed probe result
does not trigger a fallback calculation or training mutation. Instead the
live state remains available for the existing field/loss instruments and
contains an explicit `network_error` diagnostic with no invented network
values. The browser presents that diagnostic in the network panel.

## Network Telemetry Contract

`LiveViewState` will retain its existing fields and gain `network` and
`network_error` fields. On a valid probe, `network_error` is null and `network`
is a strongly typed dataclass hierarchy serialized to plain JSON lists and
numbers. Its externally visible structure is:

```text
network
├── topology: [2, 16, 16, 1]
├── activations
│   ├── input       [sample][2]
│   ├── hidden_1    [sample][16]
│   ├── hidden_2    [sample][16]
│   └── output      [sample][1]
├── activation_summary
│   ├── input       { mean, min, max, spread }
│   ├── hidden_1    { mean, min, max, spread }
│   ├── hidden_2    { mean, min, max, spread }
│   └── output      { mean, min, max, spread }
├── parameters
│   ├── w1, b1, w2, b2, w3, b3
└── gradients
    ├── w1, b1, w2, b2, w3, b3
```

Every statistic is a per-neuron list. `mean`, `min`, and `max` use NumPy's
batch-axis reductions. `spread` is explicitly `max - min`, so JavaScript does
not decide its mathematical meaning. All matrix values retain their numeric
sign and ordinary JSON float precision; the batch is intentionally not
compressed.

The exact training inputs and targets already embedded as immutable experiment
data remain the source for point positions and labels. They are not a second
inference result. In sample mode, JavaScript selects the corresponding existing
row from Python-provided `network.activations`; it does not issue another
request or calculate activations.

The dense prediction field remains uint8/base64 transport. Python continues to
derive the p = 0.5 contour from its full-precision field before serialization.

## Live HTTP Contract

`GET /api/state` calls the same non-mutating live-view construction path as
before, now including the current-state network probe. `POST /api/step` first
performs exactly one normal `LiveTrainingSession.step()`, then builds the
post-update live state. No new endpoint is needed for inspection or selection.

The completed-session contract is unchanged: after epoch 500,
`POST /api/step` returns HTTP 409 with exactly
`{"error":"training session is complete"}` and leaves epoch, loss history,
and parameters unchanged. Refreshing the page reconstructs all instruments
from server-owned current state and retained loss history.

## Network Instrument Rendering

The browser uses one Canvas 2D network instrument. It reads the server's
`topology`, requires the fixed `[2, 16, 16, 1]` contract, and draws every
layered connection from `w1`, `w2`, and `w3`: 32 input-to-hidden-1, 256
hidden-1-to-hidden-2, and 16 hidden-2-to-output, for 304 total. It draws all
35 node bodies. The browser must surface an instrument error instead of
guessing when the received contract is malformed.

Rendering order is back to front:

1. structural depth guides;
2. baseline connections;
3. real weighted connections;
4. real gradient overlays;
5. neuron bodies;
6. activation rings and highlights;
7. labels and inspection values;
8. hover and selection emphasis.

The 2.5D appearance is visual only: slightly offset depth planes, restrained
shadows, radial node shading, highlights, opacity, anti-aliasing, and optional
gentle connection curvature. It conveys hierarchy and legibility, never a
mathematical spatial dimension.

### Node encoding

Aggregate mode is the default. Each node's center intensity maps to the
Python-provided mean activation. Its outer ring maps only to the corresponding
Python-provided `spread`. Exact values remain in the inspection readout.

Sample mode maps each node to the exact selected activation row. Its aggregate
spread ring is removed or uses a clearly different selected-sample indicator;
it never pretends to be aggregate variability. Input nodes show selected
`x1/x2`, the output node shows the selected prediction, and the immutable
target label identifies its expected class.

Each non-input node also receives a compact associated bias indicator. Hover
or focus inspection exposes its bias and bias gradient without adding 33
decorative bias edges.

### Connection and gradient encoding

Each edge reads one actual weight and gradient from the appropriate server
matrix. A fixed warm/coral and cool hue encode weight sign with a neutral
midpoint. Absolute magnitude maps through a fixed deterministic nonlinear
function, for example `1 - exp(-abs(value) / fixed_scale)`, to line thickness
and opacity. The scale is a constant declared in browser presentation code,
not a per-frame or per-state maximum, so one unchanged weight does not appear
to change because another weight changed.

Gradients use a distinct thin secondary halo/overlay whose intensity maps from
`abs(dL/dw)` through its own fixed scale. Their sign remains inspectable as a
number. The renderer may apply a short arrival pulse, but it must not depict a
derivative as particles travelling backward.

Hovering a node emphasizes that node and its incoming/outgoing real edges
while de-emphasizing unrelated edges. Hovering an edge exposes source and
destination labels, exact weight, exact gradient, and the implied
`-learning_rate * gradient` using the configured learning rate embedded as
immutable experiment metadata.

## Interaction and Motion

Clicking a training point in the decision-surface Canvas calculates only which
already-rendered immutable point was hit. It sets a local selected sample index
and reprojects the current server payload. It does not fetch, train, pause,
step, mutate the server, or alter parameters. A visible Aggregate control
clears selection; clicking the selected point may also clear it. The selected
index persists across subsequent server states, so live training follows that
same sample as weights change.

Weight and gradient overlay controls are local UI toggles, initially visible,
and do not affect Python state. A compact mode readout reports Aggregate or
Sample `#N`.

Every received server state is kept in arrival order for presentation. Canvas
rendering uses `requestAnimationFrame` to interpolate visual properties only
between adjacent real states. It may drop browser animation frames when busy,
but it must neither manufacture a mathematical state nor discard or reorder a
received model state.

## Scientific Visual System and Layout

The whole Observatory moves from the existing dark interface to a light,
precise laboratory instrument. A `:root` token layer supplies coherent page,
panel, inset, text, positive/cool, negative/coral, neutral, selection, border,
and shadow values. Canvas colors use the same token-derived palette constants.

The desktop composition gives the network primary area and places the decision
surface and linear BCE chart as synchronized secondary instruments. The header
retains title, epoch, loss, and runtime state. The bottom control region keeps
Step, Train, Pause, and the existing 1×, 5×, 10×, and Max speed values while
adding compact Aggregate/Sample and overlay controls. On narrow screens the
network appears first, then decision surface, then learning dynamics; controls
remain usable.

The decision surface keeps real field data, real training point positions, the
server-provided contour, coordinate orientation, and click hit testing. It
adds a clear selected-sample ring and light theme styling only. The BCE chart
keeps its full server-provided history and linear scale, adds subtle grids and
axes, and keeps a visible current epoch/loss marker even at epoch 0.

## Error Handling and Resilience

Rendering failures are isolated from training. Python validates authoritative
telemetry before serialization. The browser validates only structural
assumptions needed to project that telemetry; it never repairs or invents
values. When the network telemetry is absent or malformed, the network panel
shows a visible instrument error while the decision surface, loss history, and
server remain usable where their own state is valid.

HTTP mutation, status, and completion behavior retains NO-V1-07 semantics.
Local selection, overlays, hover state, and animation state never write to the
session.

## Test Strategy

The implementation is TDD-driven in small slices. Tests will prove:

- valid epoch-0 telemetry, exact topology, all required shapes, deterministic
  repeated reads, and non-mutation of epoch/parameters;
- post-step temporal coherence among current parameters, activations,
  gradients, field, contour, loss, and loss history;
- Python aggregate means, minima, maxima, and `max - min` spreads against
  independent NumPy reductions;
- exact selected-sample activation rows, inputs, predictions, and targets;
- complete signed parameter and gradient matrix serialization with no omitted
  values or fabricated values;
- HTML structural contracts for all three canvases, view/overlay controls,
  renderer, hit testing, scientific tokens, and responsive layout;
- absence of browser-owned forward, sigmoid evaluation, backpropagation,
  gradient-calculation, and contour-extraction implementations;
- existing live HTTP behavior, refresh reconstruction, control semantics, and
  post-completion 409 immutability.

Focused tests run after each TDD slice alongside `uv run ruff check .` and
`git diff --check`. Full pytest, Ruff, compileall, both diff checks, Git status,
and final branch-diff review remain required before completion.

## Acceptance Criteria

NO-V1-08 is ready for operator review when the feature branch contains only
the scoped committed work, the automated gates are freshly green, and live
manual acceptance has established the following with real server telemetry:

1. epoch 0 renders the light scientific network, field, BCE marker, default
   aggregate state, all network layers, weight edges, and gradient overlays;
2. Step advances exactly once and all instruments change coherently;
3. aggregate inspection values match server-provided values;
4. selecting a training point reveals that exact sample's path without
   advancing epoch;
5. a selected sample remains selected while Train changes its real path;
6. raw neuron, positive/negative edge, bias, and gradient values are
   inspectable;
7. overlay controls are local-only and pause/step/speed controls retain their
   NO-V1-07 behavior;
8. refresh reconstructs server-owned current state;
9. epoch 500 yields a renderable complete state with 501 losses; and
10. the post-completion HTTP 409 body and immutability contract remain exact.

The milestone stops at an unpushed, clean feature branch after these checks.
