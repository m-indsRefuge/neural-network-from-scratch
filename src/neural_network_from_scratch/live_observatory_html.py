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

.mode-readout {{
    color: var(--text-muted);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
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

#belief-canvas {{
    cursor: crosshair;
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

            <span class="mode-readout" id="view-mode-value">
                Aggregate
            </span>

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

    const networkCanvas = document.getElementById("network-canvas");
    const networkContext = networkCanvas.getContext("2d");
    const networkInspection = document.getElementById("network-inspection");
    const networkError = document.getElementById("network-error");

    const NETWORK_TOPOLOGY = [2, 16, 16, 1];
    const NETWORK_LAYER_KEYS = [
        "input",
        "hidden_1",
        "hidden_2",
        "output",
    ];
    const NETWORK_LAYER_LABELS = [
        "INPUT · 2",
        "HIDDEN 1 · 16",
        "HIDDEN 2 · 16",
        "OUTPUT · 1",
    ];
    const WEIGHT_DISPLAY_SCALE = 0.75;
    const GRADIENT_DISPLAY_SCALE = 0.025;
    const NETWORK_TRANSITION_DURATION = 220;

    const weightOverlayToggle = document.getElementById(
        "weight-overlay-toggle",
    );
    const gradientOverlayToggle = document.getElementById(
        "gradient-overlay-toggle",
    );
    const aggregateButton = document.getElementById("aggregate-button");
    const viewModeValue = document.getElementById("view-mode-value");

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
    let networkPresentationState = null;
    let networkStateQueue = [];
    let networkAnimationFrame = null;
    let currentNetworkLayout = null;
    let currentNetworkConnections = [];
    let hoveredNetworkItem = null;
    let selectedSampleIndex = null;

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

    function validateNetworkMatrix(name, matrix, rows, columns) {{
        const hasExpectedRows = Array.isArray(matrix)
            && matrix.length === rows;
        const hasExpectedColumns = hasExpectedRows
            && matrix.every((row) => {{
                return Array.isArray(row)
                    && row.length === columns
                    && row.every(Number.isFinite);
            }});

        if (!hasExpectedColumns) {{
            throw new Error(
                "Network telemetry matrix "
                + name
                + " has an invalid shape"
            );
        }}
    }}

    function validateNetworkTelemetry(network) {{
        if (network === null || typeof network !== "object") {{
            throw new Error("Network telemetry is unavailable");
        }}

        const topologyMatches = Array.isArray(network.topology)
            && network.topology.length === NETWORK_TOPOLOGY.length
            && network.topology.every((width, index) => {{
                return width === NETWORK_TOPOLOGY[index];
            }});

        if (!topologyMatches) {{
            throw new Error("Network topology must be [2, 16, 16, 1]");
        }}

        NETWORK_LAYER_KEYS.forEach((layerKey, layerIndex) => {{
            const expectedWidth = NETWORK_TOPOLOGY[layerIndex];
            const activations = network.activations?.[layerKey];
            const summary = network.activation_summary?.[layerKey];

            const validActivations = Array.isArray(activations)
                && activations.length === data.training_inputs.length
                && activations.every((row) => {{
                    return Array.isArray(row)
                        && row.length === expectedWidth
                        && row.every(Number.isFinite);
                }});

            const validSummary = summary !== undefined
                && ["mean", "min", "max", "spread"].every((key) => {{
                    const values = summary[key];
                    return Array.isArray(values)
                        && values.length === expectedWidth
                        && values.every(Number.isFinite);
                }});

            if (!validActivations || !validSummary) {{
                throw new Error(
                    "Network telemetry layer "
                    + layerKey
                    + " has an invalid shape"
                );
            }}
        }});

        [
            ["w1", 2, 16],
            ["b1", 1, 16],
            ["w2", 16, 16],
            ["b2", 1, 16],
            ["w3", 16, 1],
            ["b3", 1, 1],
        ].forEach(([name, rows, columns]) => {{
            validateNetworkMatrix(
                name,
                network.parameters?.[name],
                rows,
                columns,
            );
            validateNetworkMatrix(
                "gradient " + name,
                network.gradients?.[name],
                rows,
                columns,
            );
        }});

        return network;
    }}

    function buildNetworkLayout() {{
        const width = networkCanvas.width;
        const height = networkCanvas.height;
        const layerX = [
            width * 0.10,
            width * 0.37,
            width * 0.65,
            width * 0.91,
        ];
        const top = 76;
        const bottom = height - 44;

        return NETWORK_TOPOLOGY.map((nodeCount, layerIndex) => {{
            const interval = nodeCount === 1
                ? 0
                : (bottom - top) / (nodeCount - 1);

            return Array.from(
                {{ length: nodeCount }},
                (_, nodeIndex) => {{
                    return {{
                        x: layerX[layerIndex],
                        y: nodeCount === 1
                            ? (top + bottom) / 2
                            : top + nodeIndex * interval,
                        layerIndex,
                        nodeIndex,
                    }};
                }},
            );
        }});
    }}

    function traceConnection(source, target) {{
        const bend = (target.x - source.x) * 0.38;

        networkContext.beginPath();
        networkContext.moveTo(source.x, source.y);
        networkContext.bezierCurveTo(
            source.x + bend,
            source.y,
            target.x - bend,
            target.y,
            target.x,
            target.y,
        );
    }}

    function weightPresentation(weight) {{
        const intensity = 1 - Math.exp(
            -Math.abs(weight) / WEIGHT_DISPLAY_SCALE
        );

        return {{
            intensity,
            color: weight < 0
                ? "rgb(195, 93, 82)"
                : "rgb(24, 127, 156)",
        }};
    }}

    function drawStructuralDepthGuides(layout) {{
        const height = networkCanvas.height;

        networkContext.save();
        networkContext.setLineDash([3, 7]);
        networkContext.lineWidth = 1;
        networkContext.strokeStyle = "rgba(96, 117, 126, 0.24)";

        layout.forEach((layer, layerIndex) => {{
            const x = layer[0].x;

            networkContext.beginPath();
            networkContext.moveTo(x + 5, 42);
            networkContext.lineTo(x + 5, height - 26);
            networkContext.stroke();

            networkContext.fillStyle = "rgb(82, 97, 106)";
            networkContext.font = "600 11px ui-sans-serif, system-ui, sans-serif";
            networkContext.textAlign = "center";
            networkContext.fillText(
                NETWORK_LAYER_LABELS[layerIndex],
                x,
                26,
            );
        }});

        networkContext.restore();
    }}

    function interpolateNumber(start, end, progress) {{
        return start + (end - start) * progress;
    }}

    function buildNetworkConnections(network, layout) {{
        const connections = [];
        const edgeMatrices = [
            {{
                matrixName: "w1",
                weights: network.parameters.w1,
                gradients: network.gradients.w1,
                sourceLayer: 0,
                targetLayer: 1,
            }},
            {{
                matrixName: "w2",
                weights: network.parameters.w2,
                gradients: network.gradients.w2,
                sourceLayer: 1,
                targetLayer: 2,
            }},
            {{
                matrixName: "w3",
                weights: network.parameters.w3,
                gradients: network.gradients.w3,
                sourceLayer: 2,
                targetLayer: 3,
            }},
        ];

        edgeMatrices.forEach((edgeMatrix) => {{
            edgeMatrix.weights.forEach((row, sourceIndex) => {{
                row.forEach((weight, targetIndex) => {{
                    connections.push({{
                        key: edgeMatrix.matrixName
                            + ":"
                            + sourceIndex
                            + "-"
                            + targetIndex,
                        matrixName: edgeMatrix.matrixName,
                        sourceLayer: edgeMatrix.sourceLayer,
                        targetLayer: edgeMatrix.targetLayer,
                        sourceIndex,
                        targetIndex,
                        source: layout[edgeMatrix.sourceLayer][sourceIndex],
                        target: layout[edgeMatrix.targetLayer][targetIndex],
                        weight,
                        gradient: edgeMatrix.gradients[sourceIndex][targetIndex],
                    }});
                }});
            }});
        }});

        return connections;
    }}

    function connectionIsEmphasized(connection) {{
        if (hoveredNetworkItem === null) {{
            return true;
        }}

        if (hoveredNetworkItem.kind === "connection") {{
            return hoveredNetworkItem.key === connection.key;
        }}

        return (
            connection.sourceLayer === hoveredNetworkItem.layerIndex
            && connection.sourceIndex === hoveredNetworkItem.nodeIndex
        ) || (
            connection.targetLayer === hoveredNetworkItem.layerIndex
            && connection.targetIndex === hoveredNetworkItem.nodeIndex
        );
    }}

    function presentationConnectionValue(
        connection,
        previousNetwork,
        valueName,
        progress,
    ) {{
        const currentValue = connection[valueName];

        if (previousNetwork === null) {{
            return currentValue;
        }}

        const previousValue = previousNetwork[
            valueName === "weight" ? "parameters" : "gradients"
        ][connection.matrixName][connection.sourceIndex][connection.targetIndex];

        return interpolateNumber(
            previousValue,
            currentValue,
            progress,
        );
    }}

    function drawNetworkEdges(connections, previousNetwork, progress) {{
        connections.forEach((connection) => {{
            const weight = presentationConnectionValue(
                connection,
                previousNetwork,
                "weight",
                progress,
            );
            const presentation = weightPresentation(weight);
            const emphasized = connectionIsEmphasized(connection);
            const emphasis = emphasized ? 1 : 0.16;

            networkContext.save();
            traceConnection(connection.source, connection.target);
            networkContext.lineWidth = 1;
            networkContext.strokeStyle = "rgba(137, 151, 158, 0.20)";
            networkContext.globalAlpha = emphasis;
            networkContext.stroke();

            if (weightOverlayToggle.checked) {{
                traceConnection(connection.source, connection.target);
                networkContext.lineWidth = 0.45 + presentation.intensity * 2.9;
                networkContext.strokeStyle = presentation.color;
                networkContext.globalAlpha = (
                    0.08 + presentation.intensity * 0.72
                ) * emphasis;
                networkContext.stroke();
            }}

            networkContext.restore();
        }});
    }}

    function gradientPresentation(gradient) {{
        const intensity = 1 - Math.exp(
            -Math.abs(gradient) / GRADIENT_DISPLAY_SCALE
        );

        return {{
            intensity,
            color: gradient < 0
                ? "rgb(145, 84, 168)"
                : "rgb(226, 154, 46)",
        }};
    }}

    function drawGradientOverlay(connections, previousNetwork, progress) {{
        if (!gradientOverlayToggle.checked) {{
            return;
        }}

        connections.forEach((connection) => {{
            const gradient = presentationConnectionValue(
                connection,
                previousNetwork,
                "gradient",
                progress,
            );
            const presentation = gradientPresentation(gradient);
            const emphasized = connectionIsEmphasized(connection);
            const emphasis = emphasized ? 1 : 0.12;
            const pulse = 0.58 + (1 - progress) * 0.42;

            networkContext.save();
            traceConnection(connection.source, connection.target);
            networkContext.lineWidth = 0.35 + presentation.intensity * 1.55;
            networkContext.strokeStyle = presentation.color;
            networkContext.globalAlpha = (
                0.05 + presentation.intensity * 0.48
            ) * pulse * emphasis;
            networkContext.stroke();
            networkContext.restore();
        }});
    }}

    function selectedSampleActivations(state) {{
        if (
            selectedSampleIndex === null
            || selectedSampleIndex >= data.training_inputs.length
        ) {{
            return null;
        }}

        const input = state.network.activations.input[selectedSampleIndex];
        const hidden1 = state.network.activations.hidden_1[selectedSampleIndex];
        const hidden2 = state.network.activations.hidden_2[selectedSampleIndex];
        const output = state.network.activations.output[selectedSampleIndex];

        if (
            input === undefined
            || hidden1 === undefined
            || hidden2 === undefined
            || output === undefined
        ) {{
            return null;
        }}

        return {{
            input,
            hidden_1: hidden1,
            hidden_2: hidden2,
            output,
        }};
    }}

    function aggregateNodePresentation(
        network,
        previousNetwork,
        progress,
        layerIndex,
        nodeIndex,
        sampleActivations,
        previousSampleActivations,
    ) {{
        const layerKey = NETWORK_LAYER_KEYS[layerIndex];
        const summary = network.activation_summary[layerKey];

        if (sampleActivations !== null) {{
            const currentActivation = sampleActivations[layerKey][nodeIndex];
            const previousActivation = previousSampleActivations === null
                ? currentActivation
                : previousSampleActivations[layerKey][nodeIndex];

            return {{
                activation: interpolateNumber(
                    previousActivation,
                    currentActivation,
                    progress,
                ),
                spread: 0,
                isSample: true,
            }};
        }}

        if (previousNetwork === null) {{
            return {{
                activation: summary.mean[nodeIndex],
                spread: summary.spread[nodeIndex],
                isSample: false,
            }};
        }}

        const previousSummary = previousNetwork.activation_summary[layerKey];

        return {{
            activation: interpolateNumber(
                previousSummary.mean[nodeIndex],
                summary.mean[nodeIndex],
                progress,
            ),
            spread: interpolateNumber(
                previousSummary.spread[nodeIndex],
                summary.spread[nodeIndex],
                progress,
            ),
            isSample: false,
        }};
    }}

    function nodeBias(network, layerIndex, nodeIndex) {{
        const biasMatrices = [
            null,
            {{
                values: network.parameters.b1[0],
                gradients: network.gradients.b1[0],
            }},
            {{
                values: network.parameters.b2[0],
                gradients: network.gradients.b2[0],
            }},
            {{
                values: network.parameters.b3[0],
                gradients: network.gradients.b3[0],
            }},
        ];
        const bias = biasMatrices[layerIndex];

        return bias === null
            ? null
            : {{
                value: bias.values[nodeIndex],
                gradient: bias.gradients[nodeIndex],
            }};
    }}

    function nodeIsEmphasized(node) {{
        if (hoveredNetworkItem === null) {{
            return true;
        }}

        if (hoveredNetworkItem.kind === "node") {{
            return (
                hoveredNetworkItem.layerIndex === node.layerIndex
                && hoveredNetworkItem.nodeIndex === node.nodeIndex
            );
        }}

        return (
            hoveredNetworkItem.sourceLayer === node.layerIndex
            && hoveredNetworkItem.sourceIndex === node.nodeIndex
        ) || (
            hoveredNetworkItem.targetLayer === node.layerIndex
            && hoveredNetworkItem.targetIndex === node.nodeIndex
        );
    }}

    function drawNetworkNodes(
        network,
        layout,
        previousNetwork,
        progress,
        sampleActivations,
        previousSampleActivations,
    ) {{
        layout.forEach((layer) => {{
            layer.forEach((node) => {{
                const values = aggregateNodePresentation(
                    network,
                    previousNetwork,
                    progress,
                    node.layerIndex,
                    node.nodeIndex,
                    sampleActivations,
                    previousSampleActivations,
                );
                const activation = Math.max(0, Math.min(1, values.activation));
                const spread = Math.max(0, Math.min(1, values.spread));
                const radius = node.layerIndex === 3 ? 16 : 11;
                const emphasized = nodeIsEmphasized(node);
                const emphasis = emphasized ? 1 : 0.30;
                const bias = nodeBias(
                    network,
                    node.layerIndex,
                    node.nodeIndex,
                );

                networkContext.save();
                networkContext.globalAlpha = emphasis;
                networkContext.beginPath();
                networkContext.arc(
                    node.x + 4,
                    node.y + 5,
                    radius,
                    0,
                    Math.PI * 2,
                );
                networkContext.fillStyle = "rgba(44, 61, 69, 0.18)";
                networkContext.fill();

                networkContext.beginPath();
                networkContext.arc(
                    node.x,
                    node.y,
                    values.isSample
                        ? radius + 4
                        : radius + 3 + spread * 4,
                    0,
                    Math.PI * 2,
                );
                networkContext.lineWidth = values.isSample
                    ? 2.2
                    : 1 + spread * 2.5;
                networkContext.strokeStyle = values.isSample
                    ? "rgba(226, 154, 46, 0.78)"
                    : "rgba(24, 127, 156, 0.20)";
                networkContext.stroke();

                const gradient = networkContext.createRadialGradient(
                    node.x - radius * 0.35,
                    node.y - radius * 0.4,
                    1,
                    node.x,
                    node.y,
                    radius,
                );
                gradient.addColorStop(
                    0,
                    "rgba(255, 255, 255, 0.96)",
                );
                gradient.addColorStop(
                    0.35,
                    "rgba(101, 190, 211, " + (0.22 + activation * 0.45) + ")",
                );
                gradient.addColorStop(
                    1,
                    "rgba(24, 83, 102, " + (0.20 + activation * 0.65) + ")",
                );

                networkContext.beginPath();
                networkContext.arc(
                    node.x,
                    node.y,
                    radius,
                    0,
                    Math.PI * 2,
                );
                networkContext.fillStyle = gradient;
                networkContext.fill();
                networkContext.lineWidth = emphasized ? 2.2 : 1.2;
                networkContext.strokeStyle = "rgba(255, 255, 255, 0.85)";
                networkContext.stroke();

                if (bias !== null) {{
                    const biasPresentation = weightPresentation(bias.value);
                    const biasGradient = gradientPresentation(bias.gradient);

                    networkContext.fillStyle = biasPresentation.color;
                    networkContext.globalAlpha = 0.28 + biasPresentation.intensity * 0.60;
                    networkContext.fillRect(
                        node.x + radius + 5,
                        node.y - 6,
                        3,
                        12,
                    );
                    networkContext.fillStyle = biasGradient.color;
                    networkContext.globalAlpha = 0.20 + biasGradient.intensity * 0.64;
                    networkContext.fillRect(
                        node.x + radius + 9,
                        node.y - 4,
                        2,
                        8,
                    );
                }}

                networkContext.restore();
            }});
        }});
    }}

    function drawNetwork(
        network,
        previousNetwork = null,
        progress = 1,
        sampleActivations = null,
        previousSampleActivations = null,
    ) {{
        const layout = buildNetworkLayout();
        const connections = buildNetworkConnections(network, layout);

        currentNetworkLayout = layout;
        currentNetworkConnections = connections;

        networkContext.clearRect(
            0,
            0,
            networkCanvas.width,
            networkCanvas.height,
        );
        drawStructuralDepthGuides(layout);
        drawNetworkEdges(connections, previousNetwork, progress);
        drawGradientOverlay(connections, previousNetwork, progress);
        drawNetworkNodes(
            network,
            layout,
            previousNetwork,
            progress,
            sampleActivations,
            previousSampleActivations,
        );
    }}

    function drawNetworkForState(state, previousState = null, progress = 1) {{
        drawNetwork(
            state.network,
            previousState === null
                ? null
                : previousState.network,
            progress,
            selectedSampleActivations(state),
            previousState === null
                ? null
                : selectedSampleActivations(previousState),
        );
        renderNetworkInspection(state);
    }}

    function formatRawValue(value) {{
        return String(value);
    }}

    function networkNodeLabel(layerIndex, nodeIndex) {{
        const names = ["Input", "Hidden 1", "Hidden 2", "Output"];
        return names[layerIndex] + "[" + nodeIndex + "]";
    }}

    function renderNetworkInspection(state) {{
        const network = state.network;
        const sampleActivations = selectedSampleActivations(state);

        if (hoveredNetworkItem === null) {{
            networkInspection.textContent = sampleActivations === null
                ? "Aggregate batch activity · hover a neuron or connection"
                : "Sample #"
                    + (selectedSampleIndex + 1)
                    + " exact activity · hover a neuron or connection";
            return;
        }}

        if (hoveredNetworkItem.kind === "node") {{
            const layerKey = NETWORK_LAYER_KEYS[hoveredNetworkItem.layerIndex];
            const summary = network.activation_summary[layerKey];
            const index = hoveredNetworkItem.nodeIndex;
            const bias = nodeBias(
                network,
                hoveredNetworkItem.layerIndex,
                index,
            );
            const biasReadout = bias === null
                ? ""
                : " · bias "
                    + formatRawValue(bias.value)
                    + " · bias gradient "
                    + formatRawValue(bias.gradient);

            if (sampleActivations !== null) {{
                networkInspection.textContent = "Sample #"
                    + (selectedSampleIndex + 1)
                    + " · "
                    + networkNodeLabel(
                        hoveredNetworkItem.layerIndex,
                        index,
                    )
                    + " · activation "
                    + formatRawValue(sampleActivations[layerKey][index])
                    + " · target "
                    + formatRawValue(
                        data.training_targets[selectedSampleIndex],
                    )
                    + biasReadout;
                return;
            }}

            networkInspection.textContent = networkNodeLabel(
                hoveredNetworkItem.layerIndex,
                index,
            )
                + " · mean "
                + formatRawValue(summary.mean[index])
                + " · min "
                + formatRawValue(summary.min[index])
                + " · max "
                + formatRawValue(summary.max[index])
                + " · spread "
                + formatRawValue(summary.spread[index])
                + biasReadout;
            return;
        }}

        const connection = hoveredNetworkItem;
        const update = -data.learning_rate * connection.gradient;

        networkInspection.textContent = networkNodeLabel(
            connection.sourceLayer,
            connection.sourceIndex,
        )
            + " → "
            + networkNodeLabel(
                connection.targetLayer,
                connection.targetIndex,
            )
            + " · weight "
            + formatRawValue(connection.weight)
            + " · gradient "
            + formatRawValue(connection.gradient)
            + " · update "
            + formatRawValue(update);
    }}

    function pointOnConnection(connection, progress) {{
        const inverse = 1 - progress;
        const bend = (connection.target.x - connection.source.x) * 0.38;
        const controlA = {{
            x: connection.source.x + bend,
            y: connection.source.y,
        }};
        const controlB = {{
            x: connection.target.x - bend,
            y: connection.target.y,
        }};

        return {{
            x: inverse ** 3 * connection.source.x
                + 3 * inverse ** 2 * progress * controlA.x
                + 3 * inverse * progress ** 2 * controlB.x
                + progress ** 3 * connection.target.x,
            y: inverse ** 3 * connection.source.y
                + 3 * inverse ** 2 * progress * controlA.y
                + 3 * inverse * progress ** 2 * controlB.y
                + progress ** 3 * connection.target.y,
        }};
    }}

    function distanceToSegment(point, start, end) {{
        const deltaX = end.x - start.x;
        const deltaY = end.y - start.y;
        const lengthSquared = deltaX ** 2 + deltaY ** 2;
        const projection = lengthSquared === 0
            ? 0
            : Math.max(
                0,
                Math.min(
                    1,
                    ((point.x - start.x) * deltaX + (point.y - start.y) * deltaY)
                        / lengthSquared,
                ),
            );
        const nearestX = start.x + projection * deltaX;
        const nearestY = start.y + projection * deltaY;

        return Math.hypot(point.x - nearestX, point.y - nearestY);
    }}

    function connectionDistance(point, connection) {{
        let shortest = Number.POSITIVE_INFINITY;
        let previous = pointOnConnection(connection, 0);

        for (let step = 1; step <= 14; step += 1) {{
            const next = pointOnConnection(connection, step / 14);
            shortest = Math.min(
                shortest,
                distanceToSegment(point, previous, next),
            );
            previous = next;
        }}

        return shortest;
    }}

    function findNetworkHover(point) {{
        if (currentNetworkLayout === null) {{
            return null;
        }}

        for (const layer of currentNetworkLayout) {{
            for (const node of layer) {{
                const radius = node.layerIndex === 3 ? 21 : 17;

                if (Math.hypot(point.x - node.x, point.y - node.y) <= radius) {{
                    return {{
                        kind: "node",
                        layerIndex: node.layerIndex,
                        nodeIndex: node.nodeIndex,
                    }};
                }}
            }}
        }}

        let closest = null;
        let closestDistance = 6;

        currentNetworkConnections.forEach((connection) => {{
            const distance = connectionDistance(point, connection);

            if (distance < closestDistance) {{
                closest = connection;
                closestDistance = distance;
            }}
        }});

        return closest === null
            ? null
            : {{
                kind: "connection",
                ...closest,
            }};
    }}

    function sameNetworkHover(left, right) {{
        if (left === null || right === null) {{
            return left === right;
        }}

        if (left.kind !== right.kind) {{
            return false;
        }}

        return left.kind === "node"
            ? left.layerIndex === right.layerIndex
                && left.nodeIndex === right.nodeIndex
            : left.key === right.key;
    }}

    function redrawNetworkPresentation() {{
        if (
            networkPresentationState === null
            || networkAnimationFrame !== null
        ) {{
            return;
        }}

        drawNetworkForState(networkPresentationState);
    }}

    function queueServerState(state) {{
        if (networkPresentationState === null) {{
            networkPresentationState = state;
            drawNetworkForState(state);
            return;
        }}

        networkStateQueue.push(state);

        if (networkAnimationFrame === null) {{
            startNextNetworkTransition();
        }}
    }}

    function startNextNetworkTransition() {{
        const nextState = networkStateQueue.shift();

        if (nextState === undefined) {{
            networkAnimationFrame = null;
            return;
        }}

        const previousState = networkPresentationState;
        const startedAt = performance.now();

        const animate = (timestamp) => {{
            const progress = Math.min(
                1,
                (timestamp - startedAt) / NETWORK_TRANSITION_DURATION,
            );

            drawNetworkForState(
                nextState,
                previousState,
                progress,
            );

            if (progress < 1) {{
                networkAnimationFrame = window.requestAnimationFrame(animate);
                return;
            }}

            networkPresentationState = nextState;
            networkAnimationFrame = null;

            if (networkStateQueue.length > 0) {{
                startNextNetworkTransition();
            }}
        }};

        networkAnimationFrame = window.requestAnimationFrame(animate);
    }}

    function redrawDecisionSurface() {{
        if (currentState === null) {{
            return;
        }}

        drawField(decodeField(currentState.field));
        drawDecisionBoundary(currentState.decision_boundary);
        drawTrainingPoints();
    }}

    function setAggregateMode() {{
        selectedSampleIndex = null;
        aggregateButton.setAttribute("aria-pressed", "true");
        viewModeValue.textContent = "Aggregate";
        redrawDecisionSurface();
        redrawNetworkPresentation();
    }}

    function setSampleMode(index) {{
        selectedSampleIndex = index;
        aggregateButton.setAttribute("aria-pressed", "false");
        viewModeValue.textContent = "Sample #" + String(index + 1);
        redrawDecisionSurface();
        redrawNetworkPresentation();
    }}

    function pickTrainingSample(point) {{
        let selected = null;
        let closestDistance = 16;

        data.training_inputs.forEach((sample, index) => {{
            const samplePoint = {{
                x: sample[0] * size,
                y: (1 - sample[1]) * size,
            }};
            const distance = Math.hypot(
                point.x - samplePoint.x,
                point.y - samplePoint.y,
            );

            if (distance < closestDistance) {{
                selected = index;
                closestDistance = distance;
            }}
        }});

        return selected;
    }}

    function renderNetwork(state) {{
        if (
            typeof state.network_error === "string"
            && state.network_error.length > 0
        ) {{
            networkError.textContent = state.network_error;
            networkError.hidden = false;
            return;
        }}

        try {{
            validateNetworkTelemetry(state.network);
            networkError.hidden = true;
            queueServerState(state);
        }}
        catch (error) {{
            networkContext.clearRect(
                0,
                0,
                networkCanvas.width,
                networkCanvas.height,
            );
            networkError.textContent = error instanceof Error
                ? error.message
                : "Unable to render network telemetry";
            networkError.hidden = false;
        }}
    }}

    aggregateButton.addEventListener("click", () => {{
        setAggregateMode();
    }});

    canvas.addEventListener("click", (event) => {{
        const bounds = canvas.getBoundingClientRect();
        const selected = pickTrainingSample({{
            x: (event.clientX - bounds.left) * size / bounds.width,
            y: (event.clientY - bounds.top) * size / bounds.height,
        }});

        if (selected === null) {{
            return;
        }}

        if (selectedSampleIndex === selected) {{
            setAggregateMode();
            return;
        }}

        setSampleMode(selected);
    }});

    networkCanvas.addEventListener("mousemove", (event) => {{
        const bounds = networkCanvas.getBoundingClientRect();
        const point = {{
            x: (event.clientX - bounds.left) * networkCanvas.width / bounds.width,
            y: (event.clientY - bounds.top) * networkCanvas.height / bounds.height,
        }};
        const nextHover = findNetworkHover(point);

        if (!sameNetworkHover(hoveredNetworkItem, nextHover)) {{
            hoveredNetworkItem = nextHover;
            redrawNetworkPresentation();
        }}
    }});

    networkCanvas.addEventListener("mouseleave", () => {{
        if (hoveredNetworkItem !== null) {{
            hoveredNetworkItem = null;
            redrawNetworkPresentation();
        }}
    }});

    function probabilityColor(value) {{
        const probability = value / 255;

        const zero = [244, 236, 231];
        const one = [183, 222, 231];

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

        context.lineWidth = 5;
        context.strokeStyle = "rgba(255, 255, 255, 0.86)";
        context.stroke();

        context.lineWidth = 2.25;
        context.strokeStyle = "rgb(35, 50, 57)";
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

            if (selectedSampleIndex === index) {{
                context.beginPath();
                context.arc(
                    x,
                    y,
                    15,
                    0,
                    Math.PI * 2
                );
                context.lineWidth = 3;
                context.strokeStyle = "rgb(226, 154, 46)";
                context.stroke();
            }}

            context.beginPath();
            context.arc(
                x,
                y,
                8,
                0,
                Math.PI * 2
            );

            context.fillStyle = classValue === 0
                ? "rgb(195, 93, 82)"
                : "rgb(24, 127, 156)";

            context.fill();

            context.lineWidth = 2;
            context.strokeStyle = "rgba(255, 255, 255, 0.92)";
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
        renderNetwork(state);

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
