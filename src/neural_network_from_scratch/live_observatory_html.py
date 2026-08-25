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

.stage {{
    width: 100%;
    aspect-ratio: 1 / 1;
    max-height: 720px;
    background: #080a10;
}}

#belief-canvas {{
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

    <section class="viewer">
        <div class="stage">
            <canvas
                id="belief-canvas"
                width="640"
                height="640"
                aria-label="Live neural network prediction field"
            ></canvas>
        </div>

        <div class="controls">
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
        </div>

        <div class="status-bar">
            <span class="status-label">Runtime</span>
            <span id="status-message" aria-live="polite">
                Connecting…
            </span>
        </div>
    </section>

    <div class="legend">
        <span class="legend-item">
            <span class="dot class-zero"></span>
            Class 0 training samples
        </span>

        <span class="legend-item">
            <span class="dot class-one"></span>
            Class 1 training samples
        </span>
    </div>

    <p class="note">
        Each displayed trained state is produced by the running Python NN-01.
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
        drawTrainingPoints();

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