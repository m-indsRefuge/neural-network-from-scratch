"""Compact interactive HTML rendering for NN training evolution."""

import json
from html import escape

from neural_network_from_scratch.experiments import Experiment
from neural_network_from_scratch.prediction_field import probe_prediction_field
from neural_network_from_scratch.training_evolution import TrainingEvolution


def _viewer_data(
    experiment: Experiment,
    evolution: TrainingEvolution,
    *,
    resolution: int,
) -> dict:
    """Build compact serializable data for the browser-only renderer."""
    frames = []

    for snapshot in evolution.snapshots:
        field = probe_prediction_field(
            snapshot.parameters,
            resolution=resolution,
        )

        frames.append(
            {
                "epoch": snapshot.epoch,
                "loss": snapshot.loss,
                "probabilities": field.probabilities.ravel().tolist(),
            }
        )

    return {
        "resolution": resolution,
        "frames": frames,
        "training_inputs": experiment.inputs.tolist(),
        "training_targets": experiment.targets.ravel().tolist(),
    }


def render_training_evolution_html(
    experiment: Experiment,
    evolution: TrainingEvolution,
    *,
    resolution: int,
) -> str:
    """Render selected training snapshots as a compact canvas-based viewer."""
    if not evolution.snapshots:
        raise ValueError("training evolution must contain at least one snapshot")

    data = _viewer_data(
        experiment,
        evolution,
        resolution=resolution,
    )

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

.readout {{
    display: flex;
    gap: 24px;
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
    grid-template-columns: auto 1fr;
    gap: 16px;
    align-items: center;
    padding: 18px;
    border-top: 1px solid #2d3446;
}}

button {{
    min-width: 92px;
    height: 44px;
    padding: 0 18px;
    border: 1px solid #48516a;
    border-radius: 10px;
    background: #171c28;
    color: #f4f7ff;
    font: inherit;
    cursor: pointer;
}}

button:hover {{
    background: #202738;
}}

button:focus-visible,
input:focus-visible {{
    outline: 2px solid #ffffff;
    outline-offset: 3px;
}}

input[type="range"] {{
    width: 100%;
    accent-color: #25dceb;
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

@media (max-width: 640px) {{
    .header {{
        display: block;
    }}

    .readout {{
        margin-top: 14px;
        text-align: left;
    }}

    .controls {{
        grid-template-columns: 1fr;
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
        </div>

        <div class="readout" aria-live="polite">
            <div>
                <span class="metric-label">Epoch</span>
                <span class="metric-value" id="epoch-value"></span>
            </div>
            <div>
                <span class="metric-label">Loss</span>
                <span class="metric-value" id="loss-value"></span>
            </div>
        </div>
    </header>

    <section class="viewer">
        <div class="stage">
            <canvas
                id="belief-canvas"
                width="640"
                height="640"
                aria-label="Neural network prediction field"
            ></canvas>
        </div>

        <div class="controls">
            <button type="button" id="play-toggle">Play</button>

            <input
                id="epoch-slider"
                type="range"
                min="0"
                max="{len(evolution.snapshots) - 1}"
                step="1"
                value="0"
                aria-label="Training epoch frame"
            >
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
        Every field cell is a real forward prediction from the handwritten NN-01.
    </p>
</main>

<script id="evolution-data" type="application/json">{serialized_data}</script>
<script>
(() => {{
    const data = JSON.parse(
        document.getElementById("evolution-data").textContent
    );

    const canvas = document.getElementById("belief-canvas");
    const context = canvas.getContext("2d");
    const slider = document.getElementById("epoch-slider");
    const playButton = document.getElementById("play-toggle");
    const epochValue = document.getElementById("epoch-value");
    const lossValue = document.getElementById("loss-value");

    const size = canvas.width;
    const resolution = data.resolution;
    const cellSize = size / resolution;

    let timer = null;

    function probabilityColor(probability) {{
        const p = Math.max(0, Math.min(1, probability));

        const zero = [31, 20, 66];
        const one = [35, 220, 235];

        const red = Math.round(zero[0] + p * (one[0] - zero[0]));
        const green = Math.round(zero[1] + p * (one[1] - zero[1]));
        const blue = Math.round(zero[2] + p * (one[2] - zero[2]));

        return `rgb(${{red}},${{green}},${{blue}})`;
    }}

    function drawField(frame) {{
        context.clearRect(0, 0, size, size);

        for (let row = 0; row < resolution; row += 1) {{
            for (let column = 0; column < resolution; column += 1) {{
                const index = row * resolution + column;
                const probability = frame.probabilities[index];

                const x = column * cellSize;
                const y = size - ((row + 1) * cellSize);

                context.fillStyle = probabilityColor(probability);
                context.fillRect(x, y, cellSize + 0.5, cellSize + 0.5);
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
            context.arc(x, y, 9, 0, Math.PI * 2);

            context.fillStyle = classValue === 0
                ? "rgb(255,79,216)"
                : "rgb(82,230,255)";

            context.fill();

            context.lineWidth = 2;
            context.strokeStyle = "rgb(255,255,255)";
            context.stroke();
        }});
    }}

    function showFrame(index) {{
        const safeIndex = Math.max(
            0,
            Math.min(data.frames.length - 1, Number(index))
        );

        const frame = data.frames[safeIndex];

        drawField(frame);
        drawTrainingPoints();

        slider.value = String(safeIndex);
        epochValue.textContent = frame.epoch;
        lossValue.textContent = Number(frame.loss).toFixed(6);
    }}

    function stopPlayback() {{
        if (timer !== null) {{
            window.clearInterval(timer);
            timer = null;
        }}

        playButton.textContent = "Play";
    }}

    function startPlayback() {{
        stopPlayback();
        playButton.textContent = "Pause";

        timer = window.setInterval(() => {{
            const current = Number(slider.value);

            if (current >= data.frames.length - 1) {{
                stopPlayback();
                return;
            }}

            showFrame(current + 1);
        }}, 900);
    }}

    slider.addEventListener("input", () => {{
        stopPlayback();
        showFrame(slider.value);
    }});

    playButton.addEventListener("click", () => {{
        if (timer !== null) {{
            stopPlayback();
            return;
        }}

        if (Number(slider.value) >= data.frames.length - 1) {{
            showFrame(0);
        }}

        startPlayback();
    }});

    showFrame(0);
}})();
</script>
</body>
</html>
"""