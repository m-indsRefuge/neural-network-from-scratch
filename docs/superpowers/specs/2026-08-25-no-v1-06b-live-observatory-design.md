# NO-V1-06B — Live Observatory HTTP Bridge

Date: 2026-08-25
Status: Approved design
Parent checkpoint: dea9d3c — feat: add stepwise live training session

## Purpose

NO-V1-06B changes the Neural Observatory from replaying previously captured
training states to observing NN-01 while training actually occurs.

Python remains authoritative for all neural-network state, inference, loss,
gradients, parameter updates, and training progression. The browser is only
a control and visualization surface.

## Scope

The first live Observatory supports the deterministic linear_split experiment.

The UI provides:

- live prediction field
- training-point overlay
- current epoch
- current loss
- Step
- Train
- Pause
- 1x, 5x, 10x, and Max observation speeds
- completion handling

Out of scope:

- Tauri
- React
- WebSockets
- Server-Sent Events
- threads
- background training
- persistence
- reset
- experiment selection
- loss charts
- neuron or gradient visualization
- decision contours
- new runtime dependencies

## Architecture

Data flow:

    Browser
       |
       | POST /api/step
       v
    Local HTTP server
       |
       v
    LiveTrainingSession.step()
       |
       +-- forward
       +-- loss
       +-- backward
       +-- parameter update
       |
       v
    post-update live view state
       |
       v
    JSON response
       |
       v
    Canvas repaint

Every displayed trained epoch originates from a real
LiveTrainingSession.step() call.

## Local Server Boundary

The server binds only to 127.0.0.1.

It uses Python standard-library HTTP facilities and introduces no new runtime
dependency.

One server process owns one LiveTrainingSession.

The server remains single-threaded for this milestone so two parameter
updates cannot happen concurrently.

## HTTP API

GET /

Serves the self-contained live Observatory HTML. Does not mutate training.

GET /api/state

Returns the current model view without advancing training.

At epoch zero this represents the deterministic initialized network.

POST /api/step

Performs exactly one real gradient-descent epoch and returns the resulting
post-update live view.

If training is complete, return HTTP 409 with an error stating that the
training session is complete.

Unknown routes return 404.
Unsupported methods return 405.
Unexpected internal failures return 500 where appropriate.

An error must never silently reset the training session.

## Epoch Semantics

TrainingSnapshot keeps its existing pre-update semantics.

LiveTrainingSession.step() also remains unchanged.

The live browser state is computed after step() returns, using the session's
updated parameters.

Therefore when the UI displays epoch N, its loss and prediction field
describe the model after N completed parameter updates.

## Live View

The dynamic state contains:

- epoch
- loss
- is_complete
- resolution
- encoded prediction field

Immutable experiment data such as training inputs and targets should be sent
once rather than repeated with every epoch.

Parameters, gradients, and activations stay inside Python in this milestone.

## Prediction Field Transport

NN-01 continues computing ordinary floating-point probabilities.

Only the visual transport is quantized.

Each probability is mapped onto an unsigned 8-bit value from 0 through 255.

For a 101 by 101 field this produces 10,201 bytes before transport encoding.

The byte array is base64 encoded in JSON.

The browser decodes it into a Uint8Array and applies the existing Observatory
color mapping.

This quantization must not alter training, loss, gradients, activations,
parameters, or deterministic results.

## Canvas

The live UI uses one reusable Canvas.

Each new epoch replaces the previous displayed field.

The browser must not accumulate one SVG, DOM tree, Canvas, or prediction
matrix per epoch.

Training points are drawn over the field using the existing class colors.

## Controls

Step:

One activation sends one POST /api/step request and therefore performs one
real training epoch.

Train:

Training is browser-driven and sequential:

    request step
    receive result
    render result
    wait for selected observation delay
    request next step

The browser must never queue multiple concurrent training requests.

Pause:

Pause prevents another request from being scheduled.

If one request is already in flight, that epoch may complete and render.
No further request follows it.

Python therefore needs no explicit pause state.

## Observation Speeds

Approximate delays after rendering:

    1x   = 1000 ms
    5x   = 200 ms
    10x  = 100 ms
    Max  = 0 ms

Speed changes observation cadence only. It must never affect learning rate,
dataset, update count, training algorithm, or numerical output.

Max remains sequential.

## Initial State

Server startup creates the linear_split experiment and one
LiveTrainingSession.

The initial epoch is zero.

GET /api/state computes the genuine initial loss and prediction field from
the deterministically initialized parameters.

The Observatory therefore opens showing the real pre-training model rather
than an empty field.

## Browser Lifecycle

Refreshing the browser does not reset training.

The browser reconnects through GET /api/state and displays the server-owned
session state.

Closing the browser stops training because no further step requests arrive.

Stopping the Python server discards the in-memory session. Persistence is
out of scope.

## Completion

At the final configured epoch:

- is_complete becomes true
- the final post-update field remains visible
- Train stops automatically
- Step and Train become disabled
- further POST /api/step calls return HTTP 409
- no additional mutation occurs

## Failure Handling

A failed HTTP request stops automatic Train scheduling.

The browser displays a concise error.

A malformed or failed request must never be interpreted as a successful
training epoch.

## Deterministic Equivalence

Live scheduling must not change NN-01 mathematics.

The required invariant is:

    ordinary deterministic training
    ==
    the same number of LiveTrainingSession steps

for final parameter arrays, training progression, and completion epoch.

The HTTP and browser layers do not compute these authoritative values.

## Testing

TDD remains mandatory.

Tests will cover:

- epoch-zero live state
- post-step live state
- prediction field dimensions
- deterministic field encoding
- unsigned 8-bit transport range
- GET / serving the viewer
- GET /api/state without mutation
- POST /api/step advancing exactly once
- sequential stepping
- completion behavior
- HTTP 409 after completion
- 404 unknown routes
- 405 unsupported methods
- full existing regression suite
- Ruff
- compileall
- git diff --check

No dependency file should change.

## Acceptance Criteria

NO-V1-06B is complete when:

1. The server binds only to 127.0.0.1.
2. The viewer opens at genuine epoch zero.
3. Step performs one real epoch.
4. Train performs sequential real epochs.
5. Pause prevents future steps after any in-flight step completes.
6. 1x, 5x, 10x, and Max work.
7. Epoch and loss come from Python-owned state.
8. Canvas fields come from live Python-generated predictions.
9. Browser refresh preserves the current session.
10. Final epoch completion stops training correctly.
11. Post-completion stepping returns HTTP 409 without mutation.
12. Live training stays deterministically equivalent to ordinary training.
13. No threads, WebSockets, Tauri, React, server framework, or new runtime
    dependency is introduced.
14. All repository gates pass.

## Milestone Boundary

NO-V1-06B proves live observation only.

Loss history, decision contours, neuron activations, weights, gradients,
additional experiments, and desktop packaging remain later milestones.