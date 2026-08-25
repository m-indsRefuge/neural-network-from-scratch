# NO-V1-07 — Learning Dynamics

Date: 2026-08-25
Status: Approved design

## Purpose

NO-V1-07 extends the live Neural Observatory from showing only the current
prediction field to showing the network's learning dynamics over time.

The milestone keeps the live prediction field as the primary visualization
and adds:

- a full-history live binary cross-entropy loss curve;
- a live p = 0.5 decision boundary over the prediction field.

The result should let a user watch both:
- how the model's overall error changes across training; and
- how its classification boundary forms and stabilizes.

## Scope

NO-V1-07 includes:

- full loss history from epoch 0 through the current epoch;
- server-authoritative loss history;
- a linear-scale loss graph;
- a p = 0.5 decision boundary;
- server-side contour computation from the full-precision prediction field;
- live payload extension to carry loss history and contour geometry;
- browser rendering of the new signals.

NO-V1-07 does not include:

- neuron diagrams;
- activation heatmaps;
- weight visualizations;
- gradient visualizations;
- log-scale loss plots;
- experiment switching;
- persistence;
- Tauri packaging changes;
- WebSockets or SSE;
- new runtime dependencies.

Those remain later work.

## Architectural Principle

Python remains authoritative.

The browser must not become a second source of model state.

The server owns:
- training progression;
- loss history;
- current parameters;
- prediction-field generation;
- contour computation.

The browser only:
- requests current state or one step of training;
- renders the returned field, curve, and contour;
- exposes controls.

## Runtime Relationship To NO-V1-06B

NO-V1-07 builds on the existing live Observatory runtime.

The following behaviors remain unchanged:

- localhost-only HTTP server;
- single-threaded request handling;
- Step performs exactly one training epoch;
- Train issues sequential requests only;
- Pause stops scheduling future requests;
- completion at the final epoch disables further mutation;
- POST /api/step returns HTTP 409 after completion.

NO-V1-07 only expands the live telemetry and browser rendering.

## Loss History

Loss history is authoritative server state.

It must cover the complete session trajectory from epoch 0 to the current
epoch, inclusive.

This means:

- epoch 0 has a recorded initial loss;
- after each successful step, exactly one new loss entry is appended;
- the current scalar loss equals the final entry in the history.

Because history is server-owned, a browser refresh must reconstruct the full
loss curve exactly as of the current epoch.

The browser must not maintain an independent authoritative loss history.

## Decision Boundary

The decision boundary is the p = 0.5 contour of the model's full-precision
prediction field.

It must be computed in Python from the same authoritative current parameters
that define the prediction field.

The browser must not infer the contour from the quantized display field.

That restriction is important because the displayed field is transported as
uint8/base64 telemetry, while the contour must reflect the full-precision
probabilities.

## Contour Extraction

NO-V1-07 introduces a dedicated server-side contour extractor.

The recommended algorithm is marching squares over the existing prediction
grid.

Input:
- x coordinates;
- y coordinates;
- full-precision probability matrix;
- threshold = 0.5.

Output:
- deterministic contour geometry suitable for browser rendering.

For this milestone, the contour geometry may be represented as line segments
or connected polylines.

Contour simplification is out of scope unless payload size proves
problematic.

The extraction must be deterministic.

## Live View Contract

The live payload expands conceptually from:

- epoch
- loss
- is_complete
- resolution
- field

to:

- epoch
- loss
- loss_history
- is_complete
- resolution
- field
- decision_boundary

The live state remains post-update state.

When the browser displays epoch N:
- the field must describe the model after N completed updates;
- the scalar loss must describe that same current model;
- the final entry in loss_history must match that loss;
- the decision boundary must correspond to that same current model.

## Browser Rendering

The live browser keeps the prediction field central.

The updated Observatory layout should include:

- the prediction field with training-point overlays;
- a visible decision-boundary overlay on that field;
- a dedicated loss chart;
- existing epoch/loss readouts;
- Step / Train / Pause / speed controls.

The loss graph uses:
- the full history returned by Python;
- a linear y-axis;
- epoch on the x-axis.

The boundary overlay must be visibly distinct from the background field and
training points.

The browser renders the signals returned by Python. It does not compute
authoritative history or contour geometry.

## Refresh Semantics

Refreshing the browser must preserve:

- the current epoch;
- the current field;
- the current decision boundary;
- the full loss history through the current epoch.

That means a refresh at epoch 237 must reconstruct the 0→237 loss curve and
the current boundary exactly from server-owned state.

## Deterministic Integrity

NO-V1-07 must not alter NN-01 training mathematics.

The live server, extended telemetry, loss history, and contour extraction
must remain observational only.

The required invariant remains:

ordinary deterministic training
==
the same number of live training steps

for:
- final parameters;
- training progression;
- completion epoch;
- per-epoch loss sequence, where compared.

## Testing Strategy

Implementation remains test-driven.

Tests should cover:

### Loss history
- initial history contains epoch-0 loss;
- one step appends exactly one new loss;
- current scalar loss equals the final history entry;
- history length corresponds to epoch + 1;
- history is deterministic.

### Contour extraction
- no-boundary cases behave correctly;
- simple crossing cases produce expected geometry;
- extraction is deterministic;
- contour threshold is exactly 0.5;
- contour is derived from full-precision probabilities.

### Live state contract
- live state includes the new fields;
- loss history and contour are post-update coherent;
- field, scalar loss, and contour all refer to the same current parameters.

### Browser contract
- HTML includes a loss chart surface;
- HTML includes boundary-rendering logic;
- HTML consumes server-provided loss history;
- HTML consumes server-provided contour geometry;
- browser does not embed prerecorded training history.

### HTTP behavior
- /api/state returns loss history and contour;
- /api/step returns updated loss history and contour;
- history grows only on successful training steps;
- completion behavior remains unchanged.

### Regression gates
- all existing tests remain green;
- Ruff passes;
- compileall passes;
- git diff --check passes.

## Acceptance Criteria

NO-V1-07 is complete when:

1. The live viewer shows a full loss curve from epoch 0 through current.
2. The current scalar loss equals the final loss-history value.
3. The browser refresh reconstructs the current full curve from Python state.
4. The live viewer shows a p = 0.5 decision boundary over the field.
5. The boundary is computed in Python from full-precision field values.
6. The browser does not compute authoritative loss history.
7. The browser does not compute the authoritative contour.
8. Step / Train / Pause remain correct.
9. Completion at the final epoch still rejects further POST /api/step
   requests with HTTP 409.
10. Deterministic training equivalence remains intact.
11. The full repository test and quality gates pass.

## Milestone Boundary

NO-V1-07 proves observable learning dynamics.

It does not yet expose the internal mechanics of the network such as
neurons, activations, weights, or gradients.

Those belong to the next observability milestone.