// ═══════════════════════════════════════════════════════════
//  Demo 3 — Positional Encoding Visualizer for Hydrology
//  CE 639: AI for Civil Engineering · Lecture 13
// ═══════════════════════════════════════════════════════════

const SVG_NS = "http://www.w3.org/2000/svg";
const COLORS = { purple: "#63045a", blue: "#2d3e78", orange: "#d46e00", green: "#2d7d5a", muted: "#6d6172", line: "#ddd2e1" };

// Named time-scale meanings for even dimensions
const DIM_LABELS = [
    "Daily (PET cycle)", "~6-day storm", "Weekly soil lag", "10-day recharge",
    "Bi-weekly", "Monthly flow", "Monsoon onset", "Seasonal",
    "Inter-seasonal", "Quarterly", "Annual (half)", "Annual",
    "Multi-year", "Inter-annual", "Decadal", "—",
];

// ── State ──
const state = { seqLen: 90, dModel: 64, highlightDim: 0, encType: "sin" };

// ── PE computation ──
function computePE(L, d, encType) {
    // Returns L×d matrix
    if (encType === "learned") {
        // Random but deterministic (seeded): just use a fixed random-ish init
        const pe = [];
        for (let pos = 0; pos < L; pos++) {
            const row = [];
            for (let i = 0; i < d; i++) {
                // Pseudo-random but deterministic
                const v = Math.sin(pos * 12.9898 + i * 78.233) * 0.5;
                row.push(v);
            }
            pe.push(row);
        }
        return pe;
    }
    // Sinusoidal
    const pe = [];
    for (let pos = 0; pos < L; pos++) {
        const row = [];
        for (let i = 0; i < d; i++) {
            const div = Math.pow(10000, (2 * Math.floor(i / 2)) / d);
            if (i % 2 === 0) {
                row.push(Math.sin(pos / div));
            } else {
                row.push(Math.cos(pos / div));
            }
        }
        pe.push(row);
    }
    return pe;
}

function period(i, d) {
    // Period of dimension i (sinusoidal)
    const div = Math.pow(10000, (2 * Math.floor(i / 2)) / d);
    return 2 * Math.PI * div;
}

function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom < 1e-10 ? 0 : dot / denom;
}

// ── SVG helpers ──
function svgEl(tag, attrs = {}) {
    const el = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    return el;
}

function lerpColor(c1, c2, t) {
    t = Math.max(0, Math.min(1, t));
    const a = [parseInt(c1.slice(1, 3), 16), parseInt(c1.slice(3, 5), 16), parseInt(c1.slice(5, 7), 16)];
    const b = [parseInt(c2.slice(1, 3), 16), parseInt(c2.slice(3, 5), 16), parseInt(c2.slice(5, 7), 16)];
    return "#" + a.map((v, i) => Math.round(v + (b[i] - a[i]) * t).toString(16).padStart(2, "0")).join("");
}

function peColor(v) {
    // v in [-1,1] → color
    if (v < 0) return lerpColor("#2d3e78", "#f3eff7", (v + 1));
    return lerpColor("#f3eff7", "#63045a", v);
}

function simColor(v) {
    // v in [-1,1] → color (similarity)
    if (v < 0) return lerpColor("#c84f3d", "#f9f3f7", (v + 1));
    return lerpColor("#f9f3f7", "#2d7d5a", v);
}

// ── Draw PE heatmap (positions × dims, display first 32 dims) ──
function drawPEHeatmap(pe) {
    const svg = document.getElementById("peHeatmap");
    svg.innerHTML = "";
    const L = pe.length;
    const d = Math.min(pe[0].length, 32);
    const W = 760, H = 220;
    const PL = 36, PT = 20, PR = 8, PB = 18;
    const cw = (W - PL - PR) / L;
    const ch = (H - PT - PB) / d;

    for (let pos = 0; pos < L; pos++) {
        for (let i = 0; i < d; i++) {
            const v = pe[pos][i];
            const isHighlight = Math.floor(state.highlightDim / 2) * 2 === i || Math.floor(state.highlightDim / 2) * 2 + 1 === i;
            let col = peColor(v);
            const rect = svgEl("rect", {
                x: PL + pos * cw, y: PT + i * ch,
                width: Math.max(1, cw - 0.2), height: Math.max(1, ch - 0.2),
                fill: col,
                opacity: isHighlight ? 1 : 0.7,
                "stroke-width": isHighlight ? 0 : 0,
                rx: 0,
            });
            svg.appendChild(rect);
            if (isHighlight) {
                rect.setAttribute("filter", "brightness(1.1)");
            }
        }
    }

    // Highlight row border
    const hDimGroup = Math.floor(state.highlightDim / 2) * 2;
    const hy = PT + hDimGroup * ch;
    const hh = ch * 2;
    const border = svgEl("rect", {
        x: PL, y: hy, width: L * cw, height: hh,
        fill: "none", stroke: "#D46E00", "stroke-width": 2, rx: 2,
    });
    svg.appendChild(border);

    // Axis labels
    const posStep = Math.max(1, Math.floor(L / 8));
    for (let pos = 0; pos < L; pos += posStep) {
        const x = PL + pos * cw + cw / 2;
        svg.appendChild(svgEl("text", { x, y: H - 3, "text-anchor": "middle", "font-size": 9, fill: COLORS.muted }))
            .textContent = `D${pos + 1}`;
    }
    for (let i = 0; i < d; i += 4) {
        svg.appendChild(svgEl("text", { x: PL - 3, y: PT + i * ch + ch / 2 + 3, "text-anchor": "end", "font-size": 9, fill: COLORS.muted }))
            .textContent = `i=${i}`;
    }
    svg.appendChild(svgEl("text", { x: PL + L * cw / 2, y: PT - 6, "text-anchor": "middle", "font-size": 9, fill: COLORS.muted }))
        .textContent = `PE(pos, dim) — ${L} positions × ${d} dims shown`;
}

// ── Draw similarity matrix ──
function drawSimMatrix(pe) {
    const svg = document.getElementById("simMatrix");
    svg.innerHTML = "";
    const L = pe.length;
    const W = 760, H = 240;
    const maxN = Math.min(L, 60); // sample positions
    const step = Math.max(1, Math.floor(L / maxN));
    const positions = [];
    for (let p = 0; p < L; p += step) positions.push(p);
    const N = positions.length;
    const PL = 30, PT = 18, PR = 8, PB = 16;
    const cs = Math.min((W - PL - PR) / N, (H - PT - PB) / N);
    const OX = PL, OY = PT;

    for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
            const sim = cosineSim(pe[positions[i]], pe[positions[j]]);
            const rect = svgEl("rect", {
                x: OX + j * cs, y: OY + i * cs, width: Math.max(1, cs - 0.5), height: Math.max(1, cs - 0.5),
                fill: simColor(sim), rx: cs > 5 ? 1 : 0,
            });
            svg.appendChild(rect);
        }
    }

    // Axis labels
    const labelStep = Math.max(1, Math.floor(N / 6));
    for (let i = 0; i < N; i += labelStep) {
        svg.appendChild(svgEl("text", { x: OX + i * cs + cs / 2, y: PT - 3, "text-anchor": "middle", "font-size": 9, fill: COLORS.muted }))
            .textContent = `D${positions[i] + 1}`;
        svg.appendChild(svgEl("text", { x: OX - 3, y: OY + i * cs + cs / 2 + 3, "text-anchor": "end", "font-size": 9, fill: COLORS.muted }))
            .textContent = `D${positions[i] + 1}`;
    }
    svg.appendChild(svgEl("text", { x: OX + N * cs / 2, y: H - 1, "text-anchor": "middle", "font-size": 9, fill: COLORS.muted }))
        .textContent = "Dark green = similar positions, dark red = dissimilar";
}

// ── Draw waveform for highlighted dimension ──
function drawWaveform(pe) {
    const svg = document.getElementById("waveform");
    svg.innerHTML = "";
    const L = pe.length;
    const dimIdx = Math.min(state.highlightDim * 2, pe[0].length - 1);
    const vals = pe.map(row => row[dimIdx]);
    const W = 760, H = 110;
    const PL = 36, PT = 14, PR = 8, PB = 24;
    const cw = (W - PL - PR) / L;
    const mid = PT + (H - PT - PB) / 2;
    const amp = (H - PT - PB) / 2 - 4;

    // Zero line
    svg.appendChild(svgEl("line", { x1: PL, y1: mid, x2: PL + (L - 1) * cw, y2: mid, stroke: COLORS.line, "stroke-width": 1, "stroke-dasharray": "4 3" }));

    // Area fill
    const points = vals.map((v, i) => `${PL + i * cw},${mid - v * amp}`).join(" ");
    const areaTop = vals.map((v, i) => `${PL + i * cw},${mid - v * amp}`).join(" L");
    const areaPath = `M ${PL},${mid} L ${areaTop} L ${PL + (L - 1) * cw},${mid} Z`;
    svg.appendChild(svgEl("path", { d: areaPath, fill: "rgba(99,4,90,0.08)" }));

    // Line
    const linePath = `M ` + vals.map((v, i) => `${PL + i * cw},${mid - v * amp}`).join(" L ");
    svg.appendChild(svgEl("path", { d: linePath, fill: "none", stroke: COLORS.purple, "stroke-width": 2, "stroke-linejoin": "round" }));

    // Period label
    const per = period(dimIdx, state.dModel);
    const perStr = per < state.seqLen * 2 ? `${per.toFixed(1)} days` : ">> sequence";
    const midPos = Math.min(Math.floor(per / 2), L - 1);
    if (midPos > 0) {
        svg.appendChild(svgEl("line", { x1: PL, y1: PT + 4, x2: PL + midPos * cw + cw / 2, y2: PT + 4, stroke: COLORS.orange, "stroke-width": 1.5 }));
        svg.appendChild(svgEl("text", { x: (PL + PL + midPos * cw + cw / 2) / 2, y: PT + 3, "text-anchor": "middle", "font-size": 9, fill: COLORS.orange }))
            .textContent = `T/2 ≈ ${(per / 2).toFixed(1)}d`;
    }

    // Axis
    const step2 = Math.max(1, Math.floor(L / 8));
    for (let p = 0; p < L; p += step2) {
        svg.appendChild(svgEl("text", { x: PL + p * cw, y: H - 4, "text-anchor": "middle", "font-size": 9, fill: COLORS.muted }))
            .textContent = `D${p + 1}`;
    }
    ["−1", "0", "1"].forEach((t, i) => {
        svg.appendChild(svgEl("text", { x: PL - 3, y: PT + (i === 0 ? 4 : i === 1 ? amp : amp * 2 - 4), "text-anchor": "end", "font-size": 9, fill: COLORS.muted }))
            .textContent = t;
    });
    svg.appendChild(svgEl("text", { x: PL + L * cw / 2, y: H - 12, "text-anchor": "middle", "font-size": 9, fill: COLORS.muted }))
        .textContent = `Dim ${dimIdx} (${dimIdx % 2 === 0 ? 'sin' : 'cos'}) — period ≈ ${perStr}`;
}

// ── Period table ──
function renderPeriodTable() {
    const el = document.getElementById("periodTable");
    el.innerHTML = "";
    const d = state.dModel;
    const maxPeriod = period(0, d);
    for (let i = 0; i < Math.min(8, d / 2); i++) {
        const dim = i * 2;
        const per = period(dim, d);
        const t = Math.min(1, per / maxPeriod);
        const labIdx = Math.min(i, DIM_LABELS.length - 1);
        const row = document.createElement("div");
        row.className = "period-row" + (state.highlightDim === i ? " selected" : "");
        row.innerHTML = `
      <div class="period-dim">i=${dim}</div>
      <div class="period-bar-wrap"><div class="period-bar" style="width:${Math.max(4, t * 100)}%"></div></div>
      <div class="period-val">${per >= 1000 ? ">1000d" : per.toFixed(1) + "d"}</div>
      <div class="period-lbl">${DIM_LABELS[labIdx]}</div>
    `;
        el.appendChild(row);
    }
}

// ── Math panel ──
function renderMath(pe) {
    const { dModel, highlightDim, encType } = state;
    const dimIdx = Math.min(highlightDim * 2, dModel - 1);
    const per = period(dimIdx, dModel);
    const div = Math.pow(10000, (2 * Math.floor(dimIdx / 2)) / dModel);

    document.getElementById("mathEven").innerHTML =
        `PE(pos, 2i) = sin(pos / 10000<sup>2i/d</sup>)<br>` +
        `<strong>d = ${dModel}</strong>`;

    document.getElementById("mathOdd").innerHTML =
        `PE(pos, 2i+1) = cos(pos / 10000<sup>2i/d</sup>)<br>` +
        `Same 10000<sup>2i/d</sup> as adjacent even dim → paired (sin, cos)`;

    if (encType === "sin") {
        document.getElementById("mathDim").innerHTML =
            `Dim ${dimIdx}: divisor = 10000<sup>${(2 * Math.floor(dimIdx / 2)) + "/" + (dModel)}</sup> = ${div.toFixed(1)}<br>` +
            `Period = 2π × ${div.toFixed(1)} = <strong>${per.toFixed(1)} days</strong><br>` +
            `Hydro scale: ${DIM_LABELS[Math.min(Math.floor(dimIdx / 2), DIM_LABELS.length - 1)]}`;
    } else {
        document.getElementById("mathDim").innerHTML =
            `<em>Learned encoding</em>: randomly initialized Wₚₑ ∈ ℝ<sup>L×d</sup><br>` +
            `No inherent periodicity — must learn temporal structure from data<br>` +
            `Cannot extrapolate beyond training length L`;
    }
}

// ── Metrics ──
function updateMetrics(pe) {
    const { seqLen, dModel, highlightDim, encType } = state;
    const dimIdx = Math.min(highlightDim * 2, dModel - 1);
    const per = encType === "sin" ? period(dimIdx, dModel) : null;
    const sim180 = seqLen >= 180 ? cosineSim(pe[0], pe[180]) : cosineSim(pe[0], pe[Math.min(seqLen - 1, Math.floor(seqLen / 2))]);

    document.getElementById("metricL").textContent = seqLen;
    document.getElementById("metricD").textContent = dModel;
    document.getElementById("metricPeriod").textContent = per != null ? (per.toFixed(1) + "d") : "N/A";
    document.getElementById("metricSim180").textContent = sim180.toFixed(3);
    document.getElementById("metricRange").textContent = "2.000";
}

// ── Interpretation ──
function updateInterpretation(pe) {
    const { seqLen, dModel, encType } = state;
    const per0 = period(0, dModel);
    const perMid = period(Math.floor(dModel / 2), dModel);
    const perHigh = period(dModel - 2, dModel);
    const sim180 = cosineSim(pe[0], pe[Math.min(seqLen - 1, 180)]);

    let items = [];
    if (encType === "learned") {
        items.push(`<strong>Learned</strong> encodings are randomly initialized — no inherent temporal structure. The model must learn periodicity purely from data.`);
        items.push(`Unlike sinusoidal PE, learned encodings <strong>cannot extrapolate</strong> beyond the training sequence length.`);
    } else {
        items.push(`<strong>Sinusoidal</strong> PE creates a unique fingerprint for each position using d=${dModel} wavelengths ranging from ${per0.toFixed(1)} to ${perHigh.toFixed(0)} days.`);
        items.push(`Low-dim period ≈ ${per0.toFixed(1)} days — marks individual days (diurnal PET cycles). High-dim period >> ${seqLen} days — marks inter-annual trends.`);
        items.push(`Cosine similarity between Day 1 and Day ~180 = ${sim180.toFixed(3)}. ${Math.abs(sim180) > 0.7 ? "High similarity — PE captures half-year recurrence naturally." : "Moderate similarity — half-year mark has a distinct signature."}`);
        items.push(`Dimensions with period ≈ 28 days capture the monthly soil-moisture recharge cycle, critical for multi-step streamflow forecasting.`);
    }
    document.getElementById("interpretation").innerHTML = "<ul>" + items.map(t => `<li>${t}</li>`).join("") + "</ul>";
}

// ── Notes ──
function updateNotes() {
    const { seqLen, dModel, highlightDim, encType } = state;
    const dimIdx = Math.min(highlightDim * 2, dModel - 1);
    const per = encType === "sin" ? period(dimIdx, dModel).toFixed(1) : "N/A";
    document.getElementById("heatmapNote").textContent =
        `${seqLen} positions × ${Math.min(dModel, 32)} dims shown. Orange border = highlighted dimension pair. ${encType === "sin" ? "Blue=−1, white=0, purple=+1." : "Learned (random init)."}`;
    document.getElementById("waveNote").textContent =
        `Dim ${dimIdx} (${dimIdx % 2 === 0 ? "sin" : "cos"}) — period ≈ ${per} days. ${encType === "sin" ? DIM_LABELS[Math.min(Math.floor(dimIdx / 2), DIM_LABELS.length - 1)] : "no structure"}.`;
}

// ── Control labels ──
function updateControlLabels() {
    const { seqLen, dModel, highlightDim, encType } = state;
    document.getElementById("seqLenValue").textContent = seqLen;
    document.getElementById("dModelValue").textContent = dModel;
    document.getElementById("dimValue").textContent = `Dim ${highlightDim * 2}`;
    document.getElementById("encTypeValue").textContent = encType === "sin" ? "Sinusoidal" : "Learned";
}

// ── Render ──
let cachedPE = null;

function render() {
    updateControlLabels();
    const pe = computePE(state.seqLen, state.dModel, state.encType);
    cachedPE = pe;
    drawPEHeatmap(pe);
    drawSimMatrix(pe);
    drawWaveform(pe);
    renderPeriodTable();
    renderMath(pe);
    updateMetrics(pe);
    updateInterpretation(pe);
    updateNotes();
}

// ── Events ──
function attachEvents() {
    document.getElementById("seqLenSlider").addEventListener("input", e => { state.seqLen = +e.target.value; render(); });
    document.getElementById("dModelSlider").addEventListener("input", e => { state.dModel = +e.target.value; render(); });
    document.getElementById("dimSlider").addEventListener("input", e => { state.highlightDim = +e.target.value; render(); });

    document.getElementById("encSin").addEventListener("click", () => {
        state.encType = "sin";
        document.getElementById("encSin").classList.add("active");
        document.getElementById("encLearned").classList.remove("active");
        render();
    });
    document.getElementById("encLearned").addEventListener("click", () => {
        state.encType = "learned";
        document.getElementById("encLearned").classList.add("active");
        document.getElementById("encSin").classList.remove("active");
        render();
    });
    document.getElementById("resetBtn").addEventListener("click", () => {
        state.seqLen = 90; state.dModel = 64; state.highlightDim = 0; state.encType = "sin";
        document.getElementById("seqLenSlider").value = 90;
        document.getElementById("dModelSlider").value = 64;
        document.getElementById("dimSlider").value = 0;
        document.getElementById("encSin").classList.add("active");
        document.getElementById("encLearned").classList.remove("active");
        render();
    });
}

attachEvents();
render();
