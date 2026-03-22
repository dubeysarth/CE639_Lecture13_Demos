// ═══════════════════════════════════════════════════════════
//  Demo 2 — Multi-Head Attention & Head Specialization
//  CE 639: AI for Civil Engineering · Lecture 13
// ═══════════════════════════════════════════════════════════

const SVG_NS = "http://www.w3.org/2000/svg";
const COLORS = { purple: "#63045a", blue: "#2d3e78", orange: "#d46e00", green: "#2d7d5a", rose: "#c84f3d", muted: "#6d6172", line: "#ddd2e1" };

// Head specialization labels and their "bias" toward certain time ranges
const HEAD_SPECS = [
    { label: "Flood Onset", bias: (i, j, L) => gaussBias(i, j, Math.floor(L * 0.28), 3) },
    { label: "Baseflow Recession", bias: (i, j, L) => decayBias(i, j) },
    { label: "Seasonal Trend", bias: (i, j, L) => seasonBias(i, j, L) },
    { label: "Rainfall Lag", bias: (i, j, L) => lagBias(i, j, 2, L) },
    { label: "Peak Correlation", bias: (i, j, L) => gaussBias(i, j, Math.floor(L * 0.28), 6) },
    { label: "Antecedent SWF", bias: (i, j, L) => antecedentBias(i, j, 5) },
    { label: "Storm Window", bias: (i, j, L) => stormBias(i, j, L) },
    { label: "Inter-annual", bias: (i, j, L) => longRangeBias(i, j, L) },
];

function gaussBias(i, j, center, sigma) {
    return Math.exp(-((j - center) ** 2) / (2 * sigma ** 2)) * Math.exp(-((i - center) ** 2) / (2 * sigma ** 2));
}
function decayBias(i, j) {
    const diff = i - j;
    return diff >= 0 ? Math.exp(-diff * 0.4) : 0;
}
function seasonBias(i, j, L) {
    return 0.5 + 0.5 * Math.cos(2 * Math.PI * (i - j) / L);
}
function lagBias(i, j, lag, L) {
    return Math.exp(-((i - j - lag) ** 2) / 4);
}
function antecedentBias(i, j, days) {
    const d = i - j;
    return (d > 0 && d <= days) ? (1 - d / days) : 0;
}
function stormBias(i, j, L) {
    const stormCenter = Math.floor(L * 0.28);
    return Math.max(0, 1 - Math.abs(j - stormCenter) / 4) * 0.8 + decayBias(i, j) * 0.2;
}
function longRangeBias(i, j, L) {
    return Math.exp(-Math.abs(i - j) * 0.05);
}

// Generate synthetic streamflow series
function genStreamflow(L) {
    const sf = [];
    for (let i = 0; i < L; i++) {
        let x = 18 + Math.sin(i * 0.38) * 6 + (Math.sin(i * 0.15) * 3);
        if (i >= Math.floor(L * 0.25) && i <= Math.floor(L * 0.35)) {
            const t = i - Math.floor(L * 0.25);
            x += [45, 78, 55, 30, 18, 10][t] || 0;
        }
        sf.push(Math.max(5, x));
    }
    return sf;
}

// Build an attention matrix for a head using its bias function
function buildHeadAttn(headIdx, L) {
    const spec = HEAD_SPECS[headIdx % HEAD_SPECS.length];
    const raw = [];
    for (let i = 0; i < L; i++) {
        const row = [];
        for (let j = 0; j < L; j++) {
            const b = spec.bias(i, j, L);
            const noise = 0.08 * (Math.sin(i * 3.7 + j * 2.1 + headIdx * 1.3) * 0.5 + 0.5);
            row.push(b + noise);
        }
        // softmax over row
        const mx = Math.max(...row);
        const e = row.map(v => Math.exp((v - mx) * 6));
        const s = e.reduce((a, b) => a + b, 0);
        raw.push(e.map(v => v / s));
    }
    return raw;
}

function entropy(probs) {
    return -probs.reduce((s, p) => s + (p > 1e-12 ? p * Math.log2(p) : 0), 0);
}

function avgHeadEntropy(attn) {
    const perRow = attn.map(row => entropy(row));
    return perRow.reduce((a, b) => a + b, 0) / perRow.length;
}

// ── State ──
const state = { nHeads: 4, dModel: 64, seqLen: 28, viewMode: "all", selectedHead: null };

// ═══════════════ Color helpers ═══════════════

function lerpColor(c1, c2, t) {
    const a = [parseInt(c1.slice(1, 3), 16), parseInt(c1.slice(3, 5), 16), parseInt(c1.slice(5, 7), 16)];
    const b = [parseInt(c2.slice(1, 3), 16), parseInt(c2.slice(3, 5), 16), parseInt(c2.slice(5, 7), 16)];
    const r = a.map((v, i) => Math.round(v + (b[i] - a[i]) * t));
    return `#${r.map(v => v.toString(16).padStart(2, "0")).join("")}`;
}

function attnColorFn(v) {
    if (v < 0.5) return lerpColor("#F9F4FB", "#9563A0", v * 2);
    return lerpColor("#9563A0", "#63045a", (v - 0.5) * 2);
}

// ═══════════════ SVG helpers ═══════════════

function svgEl(tag, attrs = {}) {
    const el = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
    return el;
}

function drawAttnMatrix(svg, attn, W, H, cellBorder = 0.5) {
    svg.innerHTML = "";
    const n = attn.length;
    const PT = 18, PL = 18, PR = 4, PB = 4;
    const cs = Math.min((W - PL - PR) / n, (H - PT - PB) / n);
    const OX = PL, OY = PT;

    // Max val for scaling
    const maxVal = Math.max(...attn.flat());

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const t = attn[i][j] / maxVal;
            const rect = svgEl("rect", {
                x: OX + j * cs, y: OY + i * cs, width: cs - cellBorder, height: cs - cellBorder,
                fill: attnColorFn(t), rx: cs > 8 ? 2 : 0,
            });
            svg.appendChild(rect);
        }
    }
    // Axis labels (sparse)
    const step = Math.max(1, Math.floor(n / 4));
    for (let i = 0; i < n; i += step) {
        const lbl = svgEl("text", { x: OX + i * cs + cs / 2, y: PT - 3, "text-anchor": "middle", "font-size": 9, fill: COLORS.muted });
        lbl.textContent = i + 1;
        svg.appendChild(lbl);
        const lbl2 = svgEl("text", { x: PL - 3, y: OY + i * cs + cs / 2 + 3, "text-anchor": "end", "font-size": 9, fill: COLORS.muted });
        lbl2.textContent = i + 1;
        svg.appendChild(lbl2);
    }
}

// ═══════════════ Head grid rendering ═══════════════

function renderHeadsGrid() {
    const container = document.getElementById("headsGrid");
    container.innerHTML = "";
    const { nHeads, seqLen, dModel, viewMode } = state;
    const dk = Math.floor(dModel / nHeads);
    const sf = genStreamflow(seqLen);

    if (viewMode === "concat") {
        // Show the average attention (simulates concatenated view)
        const combined = Array.from({ length: seqLen }, (_, i) =>
            Array.from({ length: seqLen }, (_, j) => {
                let s = 0; for (let h = 0; h < nHeads; h++) s += HEAD_SPECS[h % HEAD_SPECS.length].bias(i, j, seqLen);
                return s / nHeads;
            })
        );
        // softmax rows
        const softComb = combined.map(row => {
            const mx = Math.max(...row);
            const e = row.map(v => Math.exp((v - mx) * 5));
            const s = e.reduce((a, b) => a + b, 0);
            return e.map(v => v / s);
        });
        const card = document.createElement("div");
        card.className = "head-card";
        card.style.cursor = "default";
        card.innerHTML = `<div class="head-label">
      <span class="head-badge">Concatenated Output</span>
      <span class="head-specialization">Average pattern across all ${nHeads} head(s)</span>
    </div>`;
        const svg = svgEl("svg", { viewBox: `0 0 500 360`, class: "head-svg" });
        drawAttnMatrix(svg, softComb, 500, 360, 0.5);
        card.appendChild(svg);
        container.appendChild(card);
        return;
    }

    // Grid layout
    const cols = nHeads <= 2 ? 1 : nHeads <= 4 ? 2 : nHeads <= 6 ? 3 : 4;
    container.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

    for (let h = 0; h < nHeads; h++) {
        const attn = buildHeadAttn(h, seqLen);
        const spec = HEAD_SPECS[h % HEAD_SPECS.length];
        const ent = avgHeadEntropy(attn);
        const sharpness = ent < 1.5 ? "Sharp" : ent < 2.5 ? "Moderate" : "Diffuse";

        const card = document.createElement("div");
        card.className = "head-card" + (state.selectedHead === h ? " selected" : "");
        card.dataset.head = h;
        card.innerHTML = `<div class="head-label">
      <span class="head-badge">Head ${h + 1}</span>
      <span class="head-specialization">${spec.label} · ${sharpness}</span>
    </div>`;

        const cellBorder = seqLen > 30 ? 0 : 0.5;
        const svgH = nHeads <= 2 ? 260 : nHeads <= 4 ? 200 : 140;
        const svg = svgEl("svg", { viewBox: `0 0 300 ${svgH}`, class: "head-svg" });
        drawAttnMatrix(svg, attn, 300, svgH, cellBorder);
        card.appendChild(svg);
        card.addEventListener("click", () => { state.selectedHead = h; render(); });
        container.appendChild(card);
    }
}

// ═══════════════ Expanded head detail ═══════════════

function renderExpandedHead() {
    const hint = document.getElementById("selectedHeadContent");
    const expandedSvg = document.getElementById("expandedAttn");
    const expandedNote = document.getElementById("expandedNote");

    if (state.selectedHead === null) {
        hint.style.display = "flex"; expandedSvg.style.display = "none"; expandedNote.style.display = "none";
        return;
    }
    hint.style.display = "none"; expandedSvg.style.display = "block"; expandedNote.style.display = "block";

    const h = state.selectedHead;
    const attn = buildHeadAttn(h, state.seqLen);
    const spec = HEAD_SPECS[h % HEAD_SPECS.length];
    drawAttnMatrix(expandedSvg, attn, 400, 360, 0.5);

    const ent = avgHeadEntropy(attn);
    expandedNote.textContent = `Head ${h + 1} specializes in: "${spec.label}". Average entropy = ${ent.toFixed(2)} bits. Each row shows how one time-step distributes attention over all others.`;
}

// ═══════════════ Math panel ═══════════════

function renderMath() {
    const { nHeads, dModel, seqLen } = state;
    const dk = Math.floor(dModel / nHeads);
    const dv = dk;
    const ff = dModel * 4;

    document.getElementById("mathPerHead").innerHTML =
        `headᵢ: Wᵢ_Q, Wᵢ_K ∈ ℝ<sup>d×${dk}</sup>, Wᵢ_V ∈ ℝ<sup>d×${dv}</sup><br>` +
        `<strong>d_model=${dModel}, h=${nHeads}, d_k=${dk}</strong>`;

    document.getElementById("mathHeadAttn").innerHTML =
        `headᵢ = softmax( QᵢKᵢᵀ / √${dk} ) · Vᵢ<br>` +
        `Attention matrix: [${seqLen}×${seqLen}]  →  output: [${seqLen}×${dk}]`;

    document.getElementById("mathMultiHead").innerHTML =
        `MultiHead(Q,K,V) = Concat(head₁,…,head${nHeads}) · W_O<br>` +
        `Concat shape: [${seqLen}×${dModel}]  →  W_O ∈ ℝ<sup>${dModel}×${dModel}</sup><br>` +
        `<strong>Output: [${seqLen}×${dModel}]</strong>`;
}

// ═══════════════ Metrics ═══════════════

function updateMetrics() {
    const { nHeads, dModel, seqLen } = state;
    const dk = Math.floor(dModel / nHeads);
    // Params: 4 * h * d * dk (Q,K,V for each head + W_O)
    const params = 4 * dModel * dModel;
    const allEntropies = Array.from({ length: nHeads }, (_, h) => avgHeadEntropy(buildHeadAttn(h, seqLen)));
    const avgEnt = allEntropies.reduce((a, b) => a + b, 0) / nHeads;

    document.getElementById("metricH").textContent = nHeads;
    document.getElementById("metricDk").textContent = dk;
    document.getElementById("metricParams").textContent = params.toLocaleString();
    document.getElementById("metricDiversity").textContent = avgEnt.toFixed(2) + " bits";
    document.getElementById("metricComplexity").textContent = `O(L²·d)`;
}

// ═══════════════ Interpretation ═══════════════

function updateInterpretation() {
    const { nHeads, dModel, seqLen } = state;
    const dk = Math.floor(dModel / nHeads);
    const specs = Array.from({ length: nHeads }, (_, h) => HEAD_SPECS[h % HEAD_SPECS.length].label);

    let items = [];
    if (nHeads === 1) {
        items.push(`With <strong>1 head</strong>, the single attention map must simultaneously capture flood onsets, recessions, and seasonal patterns. No specialization is possible.`);
    } else {
        items.push(`With <strong>${nHeads} heads</strong>, each head specializes: ${specs.map((s, i) => `Head ${i + 1} → ${s}`).join(", ")}.`);
    }
    items.push(`Each head operates in a d_k = <strong>${dk}</strong>-dimensional subspace. Smaller d_k forces each head to use a more compact, specialized representation.`);
    items.push(`The ${seqLen}-day streamflow series contains a flash-flood event around day ${Math.floor(seqLen * 0.28) + 1}. Different heads should concentrate vs. spread attention around this event.`);
    if (nHeads >= 4) {
        items.push(`Head diversity across ${nHeads} heads allows the model to learn rainfall-to-peak lag, baseflow recession coefficient, and inter-event soil moisture simultaneously.`);
    }

    const el = document.getElementById("interpretation");
    el.innerHTML = "<ul>" + items.map(t => `<li>${t}</li>`).join("") + "</ul>";
}

// ═══════════════ Control labels ═══════════════

function updateControlLabels() {
    const { nHeads, dModel, seqLen, viewMode } = state;
    document.getElementById("nHeadsValue").textContent = nHeads;
    document.getElementById("dModelValue").textContent = dModel;
    document.getElementById("seqLenValue").textContent = seqLen;
    document.getElementById("viewModeValue").textContent = viewMode === "all" ? "All Heads" : "Concatenated";
}

// ═══════════════ Render ═══════════════

function render() {
    updateControlLabels();
    renderHeadsGrid();
    renderExpandedHead();
    renderMath();
    updateMetrics();
    updateInterpretation();
}

// ═══════════════ Events ═══════════════

function attachEvents() {
    document.getElementById("nHeadsSlider").addEventListener("input", e => {
        state.nHeads = +e.target.value;
        if (state.selectedHead !== null && state.selectedHead >= state.nHeads) state.selectedHead = null;
        render();
    });
    document.getElementById("dModelSlider").addEventListener("input", e => {
        // Ensure dModel divisible by nHeads
        state.dModel = Math.round(+e.target.value / state.nHeads) * state.nHeads;
        render();
    });
    document.getElementById("seqLenSlider").addEventListener("input", e => {
        state.seqLen = +e.target.value; state.selectedHead = null; render();
    });

    document.getElementById("viewAll").addEventListener("click", () => {
        state.viewMode = "all";
        document.getElementById("viewAll").classList.add("active");
        document.getElementById("viewConcat").classList.remove("active");
        render();
    });
    document.getElementById("viewConcat").addEventListener("click", () => {
        state.viewMode = "concat";
        document.getElementById("viewConcat").classList.add("active");
        document.getElementById("viewAll").classList.remove("active");
        render();
    });

    document.getElementById("resetBtn").addEventListener("click", () => {
        state.nHeads = 4; state.dModel = 64; state.seqLen = 28;
        state.viewMode = "all"; state.selectedHead = null;
        document.getElementById("nHeadsSlider").value = 4;
        document.getElementById("dModelSlider").value = 64;
        document.getElementById("seqLenSlider").value = 28;
        document.getElementById("viewAll").classList.add("active");
        document.getElementById("viewConcat").classList.remove("active");
        render();
    });
}

attachEvents();
render();
