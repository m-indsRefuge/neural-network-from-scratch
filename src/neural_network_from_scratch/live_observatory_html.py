"""Self-contained browser UI for the live Neural Observatory."""

import json
from html import escape

from neural_network_from_scratch.experiments import Experiment


def render_live_observatory_html(
    experiment: Experiment,
    *,
    resolution: int,
) -> str:
    """Render the browser control surface for live NN training."""
    data = {
        "resolution": resolution,
        "training_inputs": experiment.inputs.tolist(),
        "training_targets": experiment.targets.ravel().tolist(),
        "learning_rate": experiment.learning_rate,
    }

    serialized_data = json.dumps(
        data,
        separators=(",", ":"),
        ensure_ascii=True,
    )

    escaped_name = escape(experiment.name)
    display_name = escaped_name.replace("_", " ").title()

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neural Observatory — {escaped_name}</title>
<style>
:root {{
    color-scheme: dark;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
    background: #090b12;
    color: #f4f7ff;
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    min-height: 100vh;
    background:
        radial-gradient(circle at top, #17172d 0%, #090b12 55%);
}}

.observatory {{
    width: min(1100px, calc(100% - 32px));
    margin: 0 auto;
    padding: 32px 0 48px;
}}

.header {{
    display: flex;
    justify-content: space-between;
    gap: 24px;
    align-items: end;
    margin-bottom: 20px;
}}

.eyebrow {{
    margin: 0 0 6px;
    font-size: 12px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #9ba6be;
}}

h1 {{
    margin: 0;
    font-size: clamp(26px, 4vw, 42px);
    font-weight: 650;
}}

.live-indicator {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
    color: #9ba6be;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}}

.live-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgb(82,230,255);
}}

.readout {{
    display: flex;
    gap: 28px;
    text-align: right;
}}

.metric-label {{
    display: block;
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8f99ad;
}}

.metric-value {{
    display: block;
    margin-top: 4px;
    font-size: 18px;
    font-variant-numeric: tabular-nums;
}}

.viewer {{
    overflow: hidden;
    border: 1px solid #2d3446;
    border-radius: 18px;
    background: #11141d;
}}

.dynamics-grid {{
    display: grid;
    grid-template-columns:
        minmax(0, 1.45fr)
        minmax(320px, 0.85fr);
    gap: 1px;
    background: #2d3446;
}}

.stage {{
    width: 100%;
    aspect-ratio: 1 / 1;
    max-height: 720px;
    background: #080a10;
}}

.loss-stage {{
    min-width: 0;
    background: #0b0e16;
}}

#belief-canvas,
#loss-canvas {{
    display: block;
    width: 100%;
    height: 100%;
}}

.controls {{
    display: grid;
    grid-template-columns: auto auto auto minmax(150px, 1fr);
    gap: 12px;
    align-items: center;
    padding: 18px;
    border-top: 1px solid #2d3446;
}}

button,
select {{
    height: 44px;
    border: 1px solid #48516a;
    border-radius: 10px;
    background: #171c28;
    color: #f4f7ff;
    font: inherit;
}}

button {{
    min-width: 92px;
    padding: 0 18px;
    cursor: pointer;
}}

button:hover:not(:disabled) {{
    background: #202738;
}}

button:disabled {{
    cursor: default;
    opacity: 0.45;
}}

select {{
    width: 100%;
    padding: 0 12px;
}}

button:focus-visible,
select:focus-visible {{
    outline: 2px solid #ffffff;
    outline-offset: 3px;
}}

.status-bar {{
    display: flex;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 18px;
    border-top: 1px solid #242a39;
    color: #9ba6be;
    font-size: 13px;
}}

.status-label {{
    color: #788397;
}}

.legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    margin-top: 14px;
    color: #aab4c8;
    font-size: 13px;
}}

.legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
}}

.dot {{
    width: 10px;
    height: 10px;
    border-radius: 50%;
}}

.class-zero {{
    background: rgb(255,79,216);
}}

.class-one {{
    background: rgb(82,230,255);
}}

.note {{
    margin-top: 12px;
    color: #788397;
    font-size: 12px;
}}

@media (max-width: 900px) {{
    .dynamics-grid {{
        grid-template-columns: 1fr;
    }}
}}

@media (max-width: 700px) {{
    .header {{
        display: block;
    }}

    .readout {{
        margin-top: 14px;
        text-align: left;
    }}

    .controls {{
        grid-template-columns: 1fr 1fr;
    }}
}}
</style>
<style>
:root {{
    color-scheme: light;
    --surface-page: #f4f6f7;
    --surface-panel: #ffffff;
    --surface-inset: #eef2f4;
    --text-primary: #182129;
    --text-secondary: #52616d;
    --text-muted: #7a8790;
    --accent-positive: #187f9c;
    --accent-negative: #c35d52;
    --accent-neutral: #aab5bc;
    --accent-selection: #e29a2e;
    --border-subtle: #d9e0e3;
    --shadow-panel: 0 18px 40px rgba(41, 58, 66, 0.10);
    --shadow-neuron: 0 8px 18px rgba(32, 52, 61, 0.18);
    background: var(--surface-page);
    color: var(--text-primary);
}}

body {{
    min-width: 320px;
    background:
        linear-gradient(rgba(99, 123, 133, 0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99, 123, 133, 0.045) 1px, transparent 1px),
        radial-gradient(circle at 12% 0%, #ffffff 0%, var(--surface-page) 52%);
    background-size: 32px 32px, 32px 32px, auto;
}}

.observatory {{
    width: min(1480px, calc(100% - 36px));
    padding: 28px 0 40px;
}}

.header {{
    align-items: center;
    margin-bottom: 18px;
}}

.eyebrow {{
    color: var(--accent-positive);
    font-weight: 700;
}}

h1 {{
    color: var(--text-primary);
    font-weight: 650;
    letter-spacing: -0.035em;
}}

.live-indicator {{
    color: var(--text-secondary);
}}

.live-dot {{
    background: var(--accent-positive);
    box-shadow: 0 0 0 4px rgba(24, 127, 156, 0.12);
}}

.readout {{
    gap: 22px;
    padding: 12px 16px;
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.82);
    box-shadow: 0 5px 14px rgba(41, 58, 66, 0.05);
}}

.metric-label {{
    color: var(--text-muted);
}}

.metric-value {{
    color: var(--text-primary);
    font-weight: 650;
}}

.viewer {{
    overflow: visible;
    border: 0;
    border-radius: 0;
    background: transparent;
}}

.instrument-grid {{
    display: grid;
    grid-template-columns: minmax(0, 1.55fr) minmax(330px, 0.85fr);
    grid-template-rows: minmax(300px, 1fr) minmax(220px, 0.72fr);
    gap: 16px;
}}

.instrument-panel {{
    min-width: 0;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
    border-radius: 18px;
    background: var(--surface-panel);
    box-shadow: var(--shadow-panel);
}}

.network-stage {{
    grid-row: 1 / span 2;
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto;
}}

.decision-stage,
.loss-stage {{
    display: grid;
    grid-template-rows: auto minmax(0, 1fr) auto;
}}

.instrument-heading {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    padding: 15px 18px 12px;
    border-bottom: 1px solid var(--border-subtle);
}}

.instrument-title {{
    margin: 0;
    color: var(--text-primary);
    font-size: 13px;
    font-weight: 750;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}}

.instrument-subtitle {{
    color: var(--text-muted);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
}}

.network-canvas-wrap {{
    min-height: 0;
    padding: 12px 14px 4px;
    background:
        radial-gradient(circle at 48% 6%, #ffffff 0%, var(--surface-inset) 100%);
}}

.decision-canvas-wrap {{
    min-height: 0;
    padding: 10px;
    background: var(--surface-inset);
}}

.loss-canvas-wrap {{
    min-height: 0;
    padding: 6px 12px 4px;
    background: var(--surface-inset);
}}

#network-canvas,
#belief-canvas,
#loss-canvas {{
    display: block;
    width: 100%;
    height: 100%;
}}

#network-canvas {{
    min-height: 420px;
}}

#belief-canvas {{
    aspect-ratio: 1 / 1;
}}

.network-footer,
.panel-footer {{
    min-height: 44px;
    padding: 10px 16px;
    border-top: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    font-size: 12px;
}}

.network-footer {{
    display: flex;
    justify-content: space-between;
    gap: 14px;
}}

.network-inspection {{
    min-width: 0;
    overflow: hidden;
    color: var(--text-secondary);
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

.network-error {{
    color: #a5423b;
    font-weight: 650;
}}

.controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: center;
    margin-top: 16px;
    padding: 14px 16px;
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    background: var(--surface-panel);
    box-shadow: var(--shadow-panel);
}}

button,
select {{
    border-color: var(--border-subtle);
    background: #ffffff;
    color: var(--text-primary);
    box-shadow: 0 1px 2px rgba(41, 58, 66, 0.05);
}}

button:hover:not(:disabled) {{
    border-color: var(--accent-positive);
    background: #f6fbfc;
}}

button:focus-visible,
select:focus-visible,
input:focus-visible {{
    outline: 2px solid var(--accent-selection);
    outline-offset: 3px;
}}

#train-button {{
    border-color: var(--accent-positive);
    background: var(--accent-positive);
    color: #ffffff;
}}

#train-button:hover:not(:disabled) {{
    background: #116a84;
}}

.control-divider {{
    width: 1px;
    align-self: stretch;
    background: var(--border-subtle);
}}

.overlay-control {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 650;
}}

.overlay-control input {{
    accent-color: var(--accent-positive);
}}

.status-bar {{
    margin-top: 12px;
    padding: 10px 14px;
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.74);
    color: var(--text-secondary);
}}

.status-label {{
    color: var(--text-muted);
}}

.legend {{
    margin: 0;
    padding: 10px 16px;
    border-top: 1px solid var(--border-subtle);
    color: var(--text-secondary);
}}

.class-zero {{
    background: var(--accent-negative);
}}

.class-one {{
    background: var(--accent-positive);
}}

.note {{
    margin: 16px 2px 0;
    color: var(--text-muted);
}}

@media (max-width: 900px) {{
    .instrument-grid {{
        grid-template-columns: 1fr;
        grid-template-rows: auto;
    }}

    .network-stage {{
        grid-row: auto;
    }}

    #network-canvas {{
        min-height: 360px;
    }}
}}

@media (max-width: 700px) {{
    .observatory {{
        width: min(100% - 20px, 1480px);
        padding-top: 16px;
    }}

    .readout {{
        display: inline-flex;
        margin-top: 14px;
    }}

    .instrument-grid {{
        gap: 12px;
    }}

    .network-footer {{
        display: block;
    }}

    .network-error {{
        display: block;
        margin-top: 6px;
    }}

    .control-divider {{
        display: none;
    }}
}}
</style>
</head>
<body>
<main class="observatory">
    <header class="header">
        <div>
            <p class="eyebrow">Neural Observatory</p>
            <h1>{display_name}</h1>
            <div class="live-indicator">
                <span class="live-dot"></span>
                Live Training
            </div>
        </div>

        <div class="readout" aria-live="polite">
            <div>
                <span class="metric-label">Epoch</span>
                <span class="metric-value" id="epoch-value">—</span>
            </div>

            <div>
                <span class="metric-label">Loss</span>
                <span class="metric-value" id="loss-value">—</span>
            </div>
        </div>
    </header>

    <section class="viewer" aria-label="Neural Observatory instruments">
        <div class="instrument-grid">
            <section
                class="instrument-panel network-stage"
                aria-labelledby="network-title"
            >
                <div class="instrument-heading">
                    <h2 class="instrument-title" id="network-title">
                        Network Observatory
                    </h2>
                    <span class="instrument-subtitle">2 → 16 → 16 → 1</span>
                </div>

                <div class="network-canvas-wrap">
                    <canvas
                        id="network-canvas"
                        width="900"
                        height="620"
                        aria-label="Live neural network internals"
                    ></canvas>
                </div>

                <div class="network-footer">
                    <span
                        class="network-inspection"
                        id="network-inspection"
                        aria-live="polite"
                    >
                        Hover a neuron or connection to inspect live values.
                    </span>
                    <span
                        class="network-error"
                        id="network-error"
                        role="alert"
                        hidden
                    ></span>
                </div>
            </section>

            <section
                class="instrument-panel decision-stage"
                aria-labelledby="decision-title"
            >
                <div class="instrument-heading">
                    <h2 class="instrument-title" id="decision-title">
                        Decision Surface
                    </h2>
                    <span class="instrument-subtitle">p = 0.5 boundary</span>
                </div>

                <div class="decision-canvas-wrap">
                    <canvas
                        id="belief-canvas"
                        width="520"
                        height="520"
                        aria-label="Live neural network prediction field"
                    ></canvas>
                </div>

                <div class="legend">
                    <span class="legend-item">
                        <span class="dot class-zero"></span>
                        Class 0 samples
                    </span>
                    <span class="legend-item">
                        <span class="dot class-one"></span>
                        Class 1 samples
                    </span>
                </div>
            </section>

            <section
                class="instrument-panel loss-stage"
                aria-labelledby="loss-title"
            >
                <div class="instrument-heading">
                    <h2 class="instrument-title" id="loss-title">
                        Learning Dynamics
                    </h2>
                    <span class="instrument-subtitle">Linear BCE</span>
                </div>

                <div class="loss-canvas-wrap">
                    <canvas
                        id="loss-canvas"
                        width="520"
                        height="270"
                        aria-label="Live binary cross-entropy loss history"
                    ></canvas>
                </div>

                <div class="panel-footer">
                    Full post-update loss history from epoch 0 onward.
                </div>
            </section>
        </div>

        <div class="controls" aria-label="Training and network controls">
            <button type="button" id="step-button" disabled>
                Step
            </button>

            <button type="button" id="train-button" disabled>
                Train
            </button>

            <button type="button" id="pause-button" disabled>
                Pause
            </button>

            <select id="speed-select" aria-label="Observation speed">
                <option value="1000">1×</option>
                <option value="200">5×</option>
                <option value="100" selected>10×</option>
                <option value="0">Max</option>
            </select>

            <span class="control-divider" aria-hidden="true"></span>

            <button
                type="button"
                id="aggregate-button"
                aria-pressed="true"
            >
                Aggregate
            </button>

            <label class="overlay-control">
                <input id="weight-overlay-toggle" type="checkbox" checked>
                Weights
            </label>

            <label class="overlay-control">
                <input id="gradient-overlay-toggle" type="checkbox" checked>
                Gradients
            </label>
        </div>

        <div class="status-bar">
            <span class="status-label">Runtime</span>
            <span id="status-message" aria-live="polite">
                Connecting…
            </span>
        </div>
    </section>

    <p class="note">
        Every displayed value is observed from the running handwritten Python NN-01.
    </p>
</main>

<script id="experiment-data" type="application/json">{serialized_data}</script>

<script>
(() => {{
    const data = JSON.parse(
        document.getElementById("experiment-data").textContent
    );

    const canvas = document.getElementById("belief-canvas");
    const context = canvas.getContext("2d");

    const lossCanvas = document.getElementById("loss-canvas");
    const lossContext = lossCanvas.getContext("2d");

    const stepButton = document.getElementById("step-button");
    const trainButton = document.getElementById("train-button");
    const pauseButton = document.getElementById("pause-button");
    const speedSelect = document.getElementById("speed-select");

    const epochValue = document.getElementById("epoch-value");
    const lossValue = document.getElementById("loss-value");
    const statusMessage = document.getElementById("status-message");

    const size = canvas.width;
    const resolution = data.resolution;
    const cellSize = size / resolution;

    let currentState = null;
    let running = false;
    let requestInFlight = false;

    function delay(milliseconds) {{
        return new Promise((resolve) => {{
            window.setTimeout(resolve, milliseconds);
        }});
    }}

    function decodeField(encoded) {{
        const binary = atob(encoded);
        const values = new Uint8Array(binary.length);

        for (let index = 0; index < binary.length; index += 1) {{
            values[index] = binary.charCodeAt(index);
        }}

        return values;
    }}

    function probabilityColor(value) {{
        const probability = value / 255;

        const zero = [31, 20, 66];
        const one = [35, 220, 235];

        const red = Math.round(
            zero[0] + probability * (one[0] - zero[0])
        );
        const green = Math.round(
            zero[1] + probability * (one[1] - zero[1])
        );
        const blue = Math.round(
            zero[2] + probability * (one[2] - zero[2])
        );

        return `rgb(${{red}},${{green}},${{blue}})`;
    }}

    function drawField(values) {{
        context.clearRect(0, 0, size, size);

        for (let row = 0; row < resolution; row += 1) {{
            for (let column = 0; column < resolution; column += 1) {{
                const index = row * resolution + column;

                const x = column * cellSize;
                const y = size - ((row + 1) * cellSize);

                context.fillStyle = probabilityColor(values[index]);

                context.fillRect(
                    x,
                    y,
                    cellSize + 0.5,
                    cellSize + 0.5
                );
            }}
        }}
    }}

    function drawDecisionBoundary(segments) {{
        context.save();
        context.beginPath();

        segments.forEach((segment) => {{
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
        }});

        context.lineWidth = 3;
        context.strokeStyle = "rgb(255,255,255)";
        context.stroke();
        context.restore();
    }}

    function drawLossHistory(history) {{
        const width = lossCanvas.width;
        const height = lossCanvas.height;

        const left = 54;
        const right = 20;
        const top = 28;
        const bottom = 44;

        const plotWidth = width - left - right;
        const plotHeight = height - top - bottom;

        lossContext.clearRect(
            0,
            0,
            width,
            height
        );

        lossContext.strokeStyle = "rgb(72,81,106)";
        lossContext.lineWidth = 1;
        lossContext.beginPath();
        lossContext.moveTo(left, top);
        lossContext.lineTo(left, height - bottom);
        lossContext.lineTo(
            width - right,
            height - bottom
        );
        lossContext.stroke();

        if (history.length === 0) {{
            return;
        }}

        const maxLoss = Math.max(
            ...history,
            1e-12
        );
        const maxEpoch = Math.max(
            history.length - 1,
            1
        );

        lossContext.beginPath();

        history.forEach((loss, epoch) => {{
            const x =
                left
                + (epoch / maxEpoch) * plotWidth;

            const y =
                height
                - bottom
                - (loss / maxLoss) * plotHeight;

            if (epoch === 0) {{
                lossContext.moveTo(x, y);
            }}
            else {{
                lossContext.lineTo(x, y);
            }}
        }});

        lossContext.strokeStyle = "rgb(82,230,255)";
        lossContext.lineWidth = 2;
        lossContext.stroke();

        const currentEpoch = history.length - 1;
        const currentLoss = history[currentEpoch];

        const currentX =
            left
            + (currentEpoch / maxEpoch) * plotWidth;

        const currentY =
            height
            - bottom
            - (currentLoss / maxLoss) * plotHeight;

        lossContext.beginPath();
        lossContext.arc(
            currentX,
            currentY,
            4.5,
            0,
            Math.PI * 2
        );
        lossContext.fillStyle = "rgb(82,230,255)";
        lossContext.fill();

        lossContext.fillStyle = "rgb(155,166,190)";
        lossContext.font = "12px ui-monospace, monospace";

        lossContext.fillText(
            "BCE loss",
            left,
            18
        );

        lossContext.fillText(
            "epoch " + String(history.length - 1),
            width - right - 82,
            height - 14
        );
    }}

    function drawTrainingPoints() {{
        data.training_inputs.forEach((point, index) => {{
            const target = data.training_targets[index];
            const classValue = target >= 0.5 ? 1 : 0;

            const x = point[0] * size;
            const y = (1 - point[1]) * size;

            context.beginPath();
            context.arc(
                x,
                y,
                9,
                0,
                Math.PI * 2
            );

            context.fillStyle = classValue === 0
                ? "rgb(255,79,216)"
                : "rgb(82,230,255)";

            context.fill();

            context.lineWidth = 2;
            context.strokeStyle = "rgb(255,255,255)";
            context.stroke();
        }});
    }}

    function updateControls() {{
        const unavailable = currentState === null;
        const complete = currentState?.is_complete ?? false;

        stepButton.disabled =
            unavailable ||
            running ||
            requestInFlight ||
            complete;

        trainButton.disabled =
            unavailable ||
            running ||
            requestInFlight ||
            complete;

        pauseButton.disabled = !running;
    }}

    function renderState(state) {{
        const values = decodeField(state.field);

        drawField(values);
        drawDecisionBoundary(state.decision_boundary);
        drawTrainingPoints();
        drawLossHistory(state.loss_history);

        epochValue.textContent = String(state.epoch);
        lossValue.textContent = Number(state.loss).toFixed(6);

        currentState = state;

        if (state.is_complete) {{
            running = false;
            statusMessage.textContent = "Training complete";
        }}

        updateControls();
    }}

    async function loadState() {{
        const response = await fetch("/api/state");

        if (!response.ok) {{
            throw new Error(
                `State request failed: ${{response.status}}`
            );
        }}

        const state = await response.json();

        renderState(state);

        statusMessage.textContent = state.is_complete
            ? "Training complete"
            : "Ready";
    }}

    async function stepOnce() {{
        if (
            requestInFlight ||
            currentState === null ||
            currentState.is_complete
        ) {{
            return;
        }}

        requestInFlight = true;
        updateControls();

        try {{
            const response = await fetch("/api/step", {{
                method: "POST",
            }});

            const payload = await response.json();

            if (!response.ok) {{
                throw new Error(
                    payload.error ??
                    `Step request failed: ${{response.status}}`
                );
            }}

            renderState(payload);
        }}
        finally {{
            requestInFlight = false;
            updateControls();
        }}
    }}

    async function trainLoop() {{
        if (
            running ||
            requestInFlight ||
            currentState === null ||
            currentState.is_complete
        ) {{
            return;
        }}

        running = true;
        statusMessage.textContent = "Training";
        updateControls();

        try {{
            while (
                running &&
                !currentState.is_complete
            ) {{
                await stepOnce();

                if (
                    !running ||
                    currentState.is_complete
                ) {{
                    break;
                }}

                const delayMs = Number(speedSelect.value);

                if (delayMs > 0) {{
                    await delay(delayMs);
                }}
            }}
        }}
        catch (error) {{
            running = false;
            statusMessage.textContent =
                error instanceof Error
                    ? error.message
                    : "Training request failed";
        }}
        finally {{
            if (
                !currentState?.is_complete &&
                statusMessage.textContent === "Training"
            ) {{
                statusMessage.textContent = "Paused";
            }}

            updateControls();
        }}
    }}

    function pauseTraining() {{
        running = false;

        if (!currentState?.is_complete) {{
            statusMessage.textContent = "Paused";
        }}

        updateControls();
    }}

    stepButton.addEventListener("click", async () => {{
        statusMessage.textContent = "Stepping";

        try {{
            await stepOnce();

            if (!currentState?.is_complete) {{
                statusMessage.textContent = "Paused";
            }}
        }}
        catch (error) {{
            running = false;

            statusMessage.textContent =
                error instanceof Error
                    ? error.message
                    : "Step request failed";

            updateControls();
        }}
    }});

    trainButton.addEventListener("click", () => {{
        void trainLoop();
    }});

    pauseButton.addEventListener("click", () => {{
        pauseTraining();
    }});

    updateControls();

    loadState().catch((error) => {{
        running = false;

        statusMessage.textContent =
            error instanceof Error
                ? error.message
                : "Unable to connect";

        updateControls();
    }});
}})();
</script>
</body>
</html>
"""
