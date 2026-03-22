// ═══════════════════════════════════════════════════════════
//  Demo 1 — Scaled Dot-Product Attention Calculator
//  CE 639: AI for Civil Engineering · Lecture 13
// ═══════════════════════════════════════════════════════════

const SVG_NS = "http://www.w3.org/2000/svg";
const COLORS = {
  purple: "#63045a", blue: "#2d3e78", orange: "#d46e00",
  green: "#2d7d5a", rose: "#c84f3d", muted: "#6d6172", line: "#ddd2e1",
};

// ── Default Q, K, V matrices (4 days × 3 features: P, PET, SWF) ──
const DEFAULT_Q = [
  [0.80, 0.60, 0.30],
  [1.20, 0.55, 0.50],
  [0.40, 0.70, 0.85],
  [0.95, 0.65, 0.90],
];
const DEFAULT_K = [
  [0.75, 0.58, 0.28],
  [1.15, 0.52, 0.48],
  [0.38, 0.72, 0.88],
  [0.90, 0.62, 0.87],
];
const DEFAULT_V = [
  [14.5, 12.0,  8.2],
  [28.3, 22.0, 18.6],
  [47.1, 35.5, 30.2],
  [38.8, 29.3, 24.5],
];

const DAY_LABELS = ["Day 1", "Day 2", "Day 3", "Day 4"];
const FEAT_LABELS = ["P", "PET", "SWF"];

// ── State ──
const state = {
  Q: DEFAULT_Q.map(r => [...r]),
  K: DEFAULT_K.map(r => [...r]),
  V: DEFAULT_V.map(r => [...r]),
  dk: 3,
  scaled: true,
  mode: "dot",   // "dot" | "bahdanau"
  queryRow: 3,
};

// Bahdanau weight vector (fixed, pretend trained)
const BAH_V = [0.4, -0.3, 0.6, 0.2, -0.5, 0.35];

// ═══════════════ Math helpers ═══════════════

function softmax(arr) {
  const mx = Math.max(...arr);
  const e = arr.map(x => Math.exp(x - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(x => x / s);
}

function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

function norm(v) {
  return Math.sqrt(v.reduce((s, x) => s + x * x, 0));
}

function entropy(probs) {
  return -probs.reduce((s, p) => s + (p > 1e-12 ? p * Math.log2(p) : 0), 0);
}

function tanh(x) { return Math.tanh(x); }

// Compute raw scores for a query row
function computeScores(q, K, mode) {
  if (mode === "dot") {
    return K.map(k => dot(q, k));
  } else {
    // Bahdanau: score(q, k) = vᵀ · tanh(W₁q + W₂k) using BAH_V on concat
    return K.map(k => {
      const concat = [...q, ...k];
      const t = concat.map((x, i) => tanh(x * BAH_V[i % BAH_V.length]));
      return t.reduce((s, x) => s + x, 0);
    });
  }
}

function computeAttn(state) {
  const { Q, K, V, dk, scaled, mode, queryRow } = state;
  const q = Q[queryRow];
  const rawScores = computeScores(q, K, mode);
  const scale = scaled ? 1 / Math.sqrt(dk) : 1;
  const scaledScores = rawScores.map(s => s * scale);
  const weights = softmax(scaledScores);
  // Attended output
  const attended = Array(V[0].length).fill(0);
  weights.forEach((w, i) => V[i].forEach((v, j) => { attended[j] += w * v; }));
  return { rawScores, scaledScores, weights, attended };
}

// ═══════════════ DOM helpers ═══════════════

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  return el;
}

function svgText(x, y, txt, attrs = {}) {
  const el = svgEl("text", { x, y, ...attrs });
  el.textContent = txt;
  return el;
}

function fmt(v, d = 3) { return Number(v).toFixed(d); }

// ═══════════════ Matrix grid rendering ═══════════════

function buildMatrixGrid(containerId, mat, matName) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  container.style.gridTemplateColumns = "1fr";

  // Column header
  const header = document.createElement("div");
  header.className = "mat-row";
  const spacer = document.createElement("div");
  spacer.className = "mat-row-label";
  header.appendChild(spacer);
  FEAT_LABELS.forEach(lbl => {
    const h = document.createElement("div");
    h.className = "mat-cell-wrap";
    h.style.cssText = "flex:1;text-align:center;font-size:0.78rem;font-weight:700;color:var(--muted);padding-bottom:2px";
    h.textContent = lbl;
    header.appendChild(h);
  });
  container.appendChild(header);

  mat.forEach((row, ri) => {
    const rowEl = document.createElement("div");
    rowEl.className = "mat-row";
    const lbl = document.createElement("div");
    lbl.className = "mat-row-label";
    lbl.textContent = DAY_LABELS[ri];
    rowEl.appendChild(lbl);
    row.forEach((val, ci) => {
      const wrap = document.createElement("div");
      wrap.className = "mat-cell-wrap";
      const cell = document.createElement("input");
      cell.type = "number";
      cell.step = "0.05";
      cell.className = "mat-cell" + (ri === state.queryRow && matName === "Q" ? " query-row" : "");
      cell.value = fmt(val, 2);
      cell.addEventListener("change", e => {
        let v = parseFloat(e.target.value);
        if (isNaN(v)) { v = 0; cell.value = "0.00"; }
        state[matName][ri][ci] = v;
        render();
      });
      wrap.appendChild(cell);
      rowEl.appendChild(wrap);
    });
    container.appendChild(rowEl);
  });
}

function refreshQueryHighlight() {
  document.querySelectorAll("#matQ .mat-cell").forEach((cell, idx) => {
    const ri = Math.floor(idx / 3);
    cell.classList.toggle("query-row", ri === state.queryRow);
  });
}

// ═══════════════ Heatmap SVG helper ═══════════════

function makeHeatmap(values, labels, W, H, title, colorFn) {
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "pipe-svg" });
  const PL = 52, PT = 20, PR = 8, PB = 18;
  const cw = (W - PL - PR) / values.length;
  const ch = H - PT - PB;

  values.forEach((v, i) => {
    const x = PL + i * cw;
    const col = colorFn(v, values);
    const rect = svgEl("rect", { x, y: PT, width: cw - 1, height: ch, fill: col, rx: 4 });
    svg.appendChild(rect);
    // Value label
    const brightness = luminance(col);
    const textFill = brightness > 0.4 ? "#2f2436" : "#ffffff";
    svg.appendChild(svgText(x + cw / 2 - 0.5, PT + ch / 2 + 4.5, fmt(v, 3), {
      "text-anchor": "middle", "font-size": 11, fill: textFill, "font-weight": "600",
    }));
    // Day label
    svg.appendChild(svgText(x + cw / 2, H - 2, labels[i], {
      "text-anchor": "middle", "font-size": 10, fill: COLORS.muted,
    }));
  });
  // Y label
  const yLbl = svgText(14, PT + ch / 2, title, {
    "text-anchor": "middle", "font-size": 10, fill: COLORS.muted,
    transform: `rotate(-90, 14, ${PT + ch / 2})`,
  });
  svg.appendChild(yLbl);
  return svg;
}

function luminance(hex) {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

function lerpColor(c1, c2, t) {
  const a = [parseInt(c1.slice(1,3),16), parseInt(c1.slice(3,5),16), parseInt(c1.slice(5,7),16)];
  const b = [parseInt(c2.slice(1,3),16), parseInt(c2.slice(3,5),16), parseInt(c2.slice(5,7),16)];
  const r = a.map((v, i) => Math.round(v + (b[i] - a[i]) * t));
  return `#${r.map(v => v.toString(16).padStart(2,"0")).join("")}`;
}

function scoreColorFn(v, arr) {
  const mn = Math.min(...arr), mx = Math.max(...arr);
  const range = mx - mn === 0 ? 1 : mx - mn;
  const t = (v - mn) / range;
  if (t < 0.5) return lerpColor("#E6F0F8", "#378ADD", t * 2);
  return lerpColor("#378ADD", "#1a3a6e", (t - 0.5) * 2);
}

function attnColorFn(v) {
  return lerpColor("#FDF5E8", "#D85A30", Math.min(1, v * 2.5));
}

// ═══════════════ Pipeline rendering ═══════════════

function renderPipeline(result) {
  const { rawScores, scaledScores, weights, attended } = result;
  const container = document.getElementById("pipelineSteps");
  container.innerHTML = "";

  const dk = state.dk;
  const scale = 1 / Math.sqrt(dk);
  const modeLbl = state.mode === "dot" ? "Q · Kᵀ" : "vᵀ tanh(W₁Q + W₂K)";

  // ── Step 1: Raw scores ──
  addPipeStep(container, "1", "Raw Scores", `score(q, kᵢ) = ${modeLbl}`,
    makeHeatmap(rawScores, DAY_LABELS, 400, 72, "score", scoreColorFn),
    `The query for ${DAY_LABELS[state.queryRow]} is dot-producted with each key row. Higher = more similar feature vectors.`
  );

  // ── Step 2: Scaled scores ──
  if (state.scaled) {
    addPipeStep(container, "2", "Scaled Scores", `score / √d_k = score × ${fmt(scale, 3)}`,
      makeHeatmap(scaledScores, DAY_LABELS, 400, 72, "scaled", scoreColorFn),
      `Dividing by √${dk} = ${fmt(Math.sqrt(dk), 3)} prevents softmax saturation when d_k is large.`
    );
  }

  // ── Step 3: Softmax → weights ──
  const wSvg = makeHeatmap(weights, DAY_LABELS, 400, 72, "weight", v => attnColorFn(v));
  addPipeStep(container, state.scaled ? "3" : "2", "Attention Weights", "αᵢ = softmax(scoreᵢ)",
    wSvg,
    `Sum of all weights = 1.0. Highest weight: ${DAY_LABELS[weights.indexOf(Math.max(...weights))]} (${fmt(Math.max(...weights), 3)}). Entropy = ${fmt(entropy(weights), 2)} bits.`
  );

  // ── Step 4: Attended output ──
  addPipeStep(container, state.scaled ? "4" : "3", "Output = Σ αᵢ · Vᵢ",
    "context = Σᵢ αᵢ · vᵢ",
    makeHeatmap(attended, FEAT_LABELS, 400, 72, "output", (v, arr) => {
      const mn = Math.min(...arr), mx = Math.max(...arr);
      const t = mx === mn ? 0.5 : (v - mn) / (mx - mn);
      return lerpColor("#EAF7F0", "#1D9E75", t);
    }),
    `The output is a weighted blend of Value rows. It represents the attended hydrological state: P=${fmt(attended[0],1)} mm, PET=${fmt(attended[1],1)}, SWF=${fmt(attended[2],3)}.`
  );
}

function addPipeStep(container, num, title, formula, vizEl, note) {
  const step = document.createElement("div");
  step.className = "pipe-step";
  step.innerHTML = `
    <div class="pipe-step-label">
      <span class="pipe-badge">Step ${num}</span>
      <span class="pipe-title">${title}</span>
    </div>
    <div class="pipe-formula">${formula}</div>
  `;
  step.appendChild(vizEl);
  const noteEl = document.createElement("p");
  noteEl.className = "pipe-note";
  noteEl.textContent = note;
  step.appendChild(noteEl);
  container.appendChild(step);
}

// ═══════════════ Math panel ═══════════════

function renderMath(result) {
  const { rawScores, scaledScores, weights, attended } = result;
  const dk = state.dk;
  const scale = 1 / Math.sqrt(dk);
  const q = state.Q[state.queryRow];
  const qStr = `[${q.map(v => fmt(v,2)).join(", ")}]`;
  const sStr = rawScores.map(v => fmt(v,3)).join(", ");
  const ssStr = scaledScores.map(v => fmt(v,3)).join(", ");
  const wStr = weights.map(v => fmt(v,3)).join(", ");
  const aStr = attended.map(v => fmt(v,2)).join(", ");

  const modeFormula = state.mode === "dot"
    ? `score(qᵢ, kⱼ) = qᵢ · kⱼ = ${qStr} · K`
    : `score(qᵢ, kⱼ) = vᵀ · tanh(W₁qᵢ + W₂kⱼ)`;

  document.getElementById("mathScore").innerHTML =
    `${modeFormula}<br><strong>scores = [${sStr}]</strong>`;

  const scaleBlock = document.getElementById("mathScaleBlock");
  scaleBlock.style.display = state.scaled ? "" : "none";
  document.getElementById("mathScale").innerHTML =
    `scaled = scores × (1/√${dk}) = scores × ${fmt(scale,3)}<br><strong>[${ssStr}]</strong>`;

  document.getElementById("mathSoftmax").innerHTML =
    `αᵢ = exp(scoreᵢ) / Σ exp(scoreⱼ)<br><strong>α = [${wStr}]</strong>  (sum = 1.000)`;

  document.getElementById("mathOutput").innerHTML =
    `context = Σᵢ αᵢ × Vᵢ<br><strong>[${aStr}]</strong>`;
}

// ═══════════════ Metric cards ═══════════════

function updateMetrics(result) {
  const { weights, attended } = result;
  const dk = state.dk;
  document.getElementById("metricDk").textContent = dk;
  document.getElementById("metricScale").textContent = fmt(1 / Math.sqrt(dk), 3);
  document.getElementById("metricMaxAttn").textContent = fmt(Math.max(...weights), 3);
  document.getElementById("metricEntropy").textContent = fmt(entropy(weights), 2) + " bits";
  document.getElementById("metricNorm").textContent = fmt(norm(attended), 2);
}

// ═══════════════ Control labels ═══════════════

function updateControlLabels() {
  document.getElementById("dkValue").textContent = state.dk;
  document.getElementById("tempValue").textContent = state.scaled ? "ON" : "OFF";
  document.getElementById("modeValue").textContent = state.mode === "dot" ? "Dot-Product" : "Bahdanau";
  document.getElementById("qRowValue").textContent = DAY_LABELS[state.queryRow];
}

// ═══════════════ Interpretation ═══════════════

function updateInterpretation(result) {
  const { weights, attended } = result;
  const maxI = weights.indexOf(Math.max(...weights));
  const minI = weights.indexOf(Math.min(...weights));
  const qRow = state.queryRow;
  const H = entropy(weights);
  const scaleStr = state.scaled ? "scaled (÷√d_k)" : "unscaled";
  const modeStr = state.mode === "dot" ? "Scaled Dot-Product" : "Bahdanau Additive";

  let items = [];
  items.push(`Using <strong>${modeStr}</strong> attention, ${scaleStr}.`);
  items.push(`<strong>${DAY_LABELS[qRow]}</strong> (the query) allocates ${fmt(weights[maxI]*100,1)}% of its attention to <strong>${DAY_LABELS[maxI]}</strong>, likely because their feature vectors are most similar.`);
  if (H < 1.0) {
    items.push(`Entropy is low (${fmt(H,2)} bits) — attention is <strong>sharp and focused</strong>. The model strongly prefers one context day.`);
  } else if (H < 1.8) {
    items.push(`Entropy is medium (${fmt(H,2)} bits) — attention is <strong>distributed</strong> across a few relevant days.`);
  } else {
    items.push(`Entropy is high (${fmt(H,2)} bits) — attention is <strong>nearly uniform</strong>. The model finds all days equally relevant (or is saturating without scaling).`);
  }
  items.push(`The attended value vector represents: P = ${fmt(attended[0],1)} mm/day, PET = ${fmt(attended[1],1)}, SWF = ${fmt(attended[2],3)}.`);

  const el = document.getElementById("interpretation");
  el.innerHTML = "<ul>" + items.map(t => `<li>${t}</li>`).join("") + "</ul>";
}

// ═══════════════ Main render ═══════════════

function render() {
  const result = computeAttn(state);
  buildMatrixGrid("matQ", state.Q, "Q");
  buildMatrixGrid("matK", state.K, "K");
  buildMatrixGrid("matV", state.V, "V");
  refreshQueryHighlight();
  renderPipeline(result);
  renderMath(result);
  updateMetrics(result);
  updateControlLabels();
  updateInterpretation(result);
}

// ═══════════════ Event listeners ═══════════════

function attachEvents() {
  document.getElementById("dkSlider").addEventListener("input", e => {
    state.dk = +e.target.value; render();
  });

  document.getElementById("qRowSlider").addEventListener("input", e => {
    state.queryRow = +e.target.value; render();
  });

  function makeScaleToggle(id, val) {
    document.getElementById(id).addEventListener("click", () => {
      state.scaled = val;
      document.getElementById("scaleOn").classList.toggle("active", val);
      document.getElementById("scaleOff").classList.toggle("active", !val);
      render();
    });
  }
  makeScaleToggle("scaleOn", true);
  makeScaleToggle("scaleOff", false);

  function makeModeToggle(id, val) {
    document.getElementById(id).addEventListener("click", () => {
      state.mode = val;
      document.getElementById("modeDot").classList.toggle("active", val === "dot");
      document.getElementById("modeBah").classList.toggle("active", val === "bahdanau");
      render();
    });
  }
  makeModeToggle("modeDot", "dot");
  makeModeToggle("modeBah", "bahdanau");

  document.getElementById("resetBtn").addEventListener("click", () => {
    state.Q = DEFAULT_Q.map(r => [...r]);
    state.K = DEFAULT_K.map(r => [...r]);
    state.V = DEFAULT_V.map(r => [...r]);
    state.dk = 3; state.scaled = true; state.mode = "dot"; state.queryRow = 3;
    document.getElementById("dkSlider").value = 3;
    document.getElementById("qRowSlider").value = 3;
    document.getElementById("scaleOn").classList.add("active");
    document.getElementById("scaleOff").classList.remove("active");
    document.getElementById("modeDot").classList.add("active");
    document.getElementById("modeBah").classList.remove("active");
    render();
  });

  document.getElementById("randomBtn").addEventListener("click", () => {
    const r = () => parseFloat((Math.random() * 1.8 - 0.2).toFixed(2));
    state.Q = Array.from({length:4}, () => [r(), r(), Math.max(0, Math.min(1, r()))]);
    state.K = Array.from({length:4}, () => [r(), r(), Math.max(0, Math.min(1, r()))]);
    state.V = Array.from({length:4}, () => [r()*30+5, r()*20+8, r()*20+5]);
    render();
  });
}

// ── Boot ──
attachEvents();
render();
