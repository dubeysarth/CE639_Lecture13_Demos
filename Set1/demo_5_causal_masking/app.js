// ═══════════════════════════════════════════════════════════
//  Demo 5 — Causal Masking & Autoregressive Decoding Debugger
//  CE 639: AI for Civil Engineering · Lecture 13
// ═══════════════════════════════════════════════════════════

const SVG_NS = "http://www.w3.org/2000/svg";
const COLORS = { purple: "#63045a", blue: "#2d3e78", orange: "#d46e00", green: "#2d7d5a", rose: "#c84f3d", muted: "#6d6172", line: "#ddd2e1" };

// ── Synthetic streamflow: history + "true" future ──
function makeStreamflow(L, T) {
    const hist = [];
    for (let i = 0; i < L; i++) {
        let x = 20 + 8 * Math.sin(i * 0.45) + 3 * Math.sin(i * 0.9);
        if (i >= Math.floor(L * 0.55) && i <= Math.floor(L * 0.75)) x += [10, 22, 38, 28, 18, 10, 5][i - Math.floor(L * 0.55)] || 0;
        hist.push(Math.max(5, x));
    }
    // "True" future (unknown to model, just for display)
    const last = hist[L - 1];
    const future = [];
    for (let i = 0; i < T; i++) {
        const rec = last * Math.exp(-0.15 * i) + 15 * (1 - Math.exp(-0.15 * i));
        future.push(rec + 2 * Math.sin(i * 0.8) + (Math.random() - 0.5) * 2);
    }
    return { hist, future };
}

// Simulate attention weights for step `step` (decoder position 0..step)
function computeStepAttn(step, T) {
    // For each (i,j) with j <= i and j <= step: simulate attention weights
    const mat = [];
    for (let i = 0; i <= step; i++) {
        const row = [];
        for (let j = 0; j <= i; j++) {
            const d = i - j;
            const w = Math.exp(-d * 0.35) * (0.7 + 0.3 * Math.sin(j * 0.7 + i * 0.3));
            row.push(w);
        }
        // softmax
        const mx = Math.max(...row);
        const e = row.map(v => Math.exp((v - mx) * 4));
        const s = e.reduce((a, b) => a + b, 0);
        mat.push(e.map(v => v / s));
    }
    return mat;
}

// Cross-attention weights: decoder step → history positions
function computeCrossAttn(step, L) {
    const recencyPeak = Math.max(0, L - 3 - step * 0.5);
    const eventPeak = Math.floor(L * 0.65);
    const scores = Array.from({ length: L }, (_, j) => {
        const rw = Math.exp(-Math.abs(j - recencyPeak) * 0.15);
        const ew = Math.exp(-Math.abs(j - eventPeak) * 0.25) * (step < 3 ? 0.8 : 0.3);
        return rw * 0.6 + ew + (Math.sin(j * 0.3 + step) * 0.05);
    });
    const mx = Math.max(...scores);
    const e = scores.map(v => Math.exp((v - mx) * 5));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
}

// ── State ──
const state = { T: 6, L: 12, currentStep: 0, totalSteps: 0, playing: false, speed: 1200 };
let steps = [];
let playTimer = null;

// ── Build all steps ──
function buildSteps() {
    const { T, L } = state;
    const { hist, future } = makeStreamflow(L, T);
    const generatedSf = [];
    const allSteps = [];

    // Step 0: initialization
    allSteps.push({
        type: "init", step: 0, generated: [],
        msg: "The decoder starts with only the last observed streamflow value Q(t) as its seed. All T future positions are masked (−∞).",
        formula: "M_ij = { 0 if j ≤ i, −∞ if j > i }",
        formulaNote: "At step 0, the only unmasked position is (0,0) — the decoder can only attend to its own seed.",
        badge: "Seed", mode: "Initialize",
        hist, future,
    });

    for (let step = 0; step < T; step++) {
        const crossW = computeCrossAttn(step, L);
        const topJ = crossW.indexOf(Math.max(...crossW));
        const selfAttn = computeStepAttn(step, T);
        const pred = future[step] + (Math.random() - 0.5) * 1.5;
        generatedSf.push(pred);

        allSteps.push({
            type: "decode", step: step + 1, generated: [...generatedSf],
            selfAttn, crossW, topHistJ: topJ,
            msg: `Generating Q(t+${step + 1}): The decoder query at position ${step} attends to its ${step + 1} available past position(s) via self-attention, then reads historical streamflow via cross-attention. Max cross-attention to Day ${topJ + 1} of history.`,
            formula: `A[${step},:] = softmax( (q·Kᵀ / √d_k) + M[${step},:] )`,
            formulaNote: `Only columns j ≤ ${step} are unmasked. Columns j > ${step} receive −∞ before softmax → weight 0.`,
            badge: `Step ${step + 1}/${T}`, mode: `Generating Q(t+${step + 1})`,
            hist, future,
        });
    }

    allSteps.push({
        type: "done", step: T + 1, generated: [...generatedSf],
        msg: `Decoding complete! All ${T} streamflow values Q(t+1)…Q(t+${T}) have been generated. The full lower-triangular attention matrix is now visible — each row shows how that forecast step attended to prior predictions.`,
        formula: `Q(t+1)…Q(t+${T}) generated autoregressively`,
        formulaNote: "Causal masking guaranteed no future information was used at any step.",
        badge: "Complete", mode: "Done",
        hist, future,
    });

    return allSteps;
}

// ═══ SVG helpers ═══
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

function fmt(v, d = 2) { return Number(v).toFixed(d); }

// ═══ Render mask SVG ═══
function renderMask(step) {
    const svg = document.getElementById("maskSvg");
    svg.innerHTML = "";
    const T = state.T;
    const W = 540, H = 340;
    const PL = 44, PT = 30, PR = 10, PB = 30;
    const cs = Math.min((W - PL - PR) / T, (H - PT - PB) / T);
    const OX = PL, OY = PT;

    const curStep = step.type === "done" ? T - 1 : step.step - 1;
    const selfAttn = step.selfAttn || null;

    for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
            let col, label = "";
            if (j > i) {
                // Masked
                col = "#ede7f0"; label = "−∞";
            } else if (j <= curStep && i <= curStep && selfAttn && i < selfAttn.length && j < selfAttn[i].length) {
                // Has attention weight
                const w = selfAttn[i][j];
                col = lerpColor("#E6F0F8", "#2d3e78", w * 2);
                label = fmt(w, 2);
            } else if (j <= i && i <= curStep) {
                col = "rgba(45,62,120,0.15)";
                label = "0";
            } else {
                col = "rgba(220,215,228,0.5)";
            }

            // Highlight current row
            const isCurrentRow = i === curStep && step.type !== "done" && step.type !== "init";
            if (isCurrentRow && j <= i) col = lerpColor("#FAEEDA", "#d46e00", j <= curStep && selfAttn ? Math.min(1, (selfAttn[i]?.[j] || 0) * 2) : 0.3);
            if (isCurrentRow && j === i) col = "#d46e00";

            const rect = svgEl("rect", { x: OX + j * cs, y: OY + i * cs, width: cs - 1, height: cs - 1, fill: col, rx: 3 });
            svg.appendChild(rect);

            if (cs > 24 && label) {
                const brightness = j > i ? 0.6 : 0.3;
                const tc = j > i ? COLORS.muted : (lerpColor("#2d3e78", "#ffffff", brightness).startsWith("#2d") ? "#ffffff" : "#2f2436");
                const el = svgEl("text", { x: OX + j * cs + cs / 2, y: OY + i * cs + cs / 2 + 4, "text-anchor": "middle", "font-size": Math.min(11, cs * 0.4), fill: j > i ? COLORS.muted : "rgba(255,255,255,0.92)", "font-weight": "600" });
                el.textContent = cs > 34 ? (j > i ? "×" : label) : (j > i ? "×" : "");
                svg.appendChild(el);
            }
        }
        // Row label
        const yl = svgEl("text", { x: PL - 5, y: OY + i * cs + cs / 2 + 4, "text-anchor": "end", "font-size": 10, fill: COLORS.muted });
        yl.textContent = `Q(t+${i + 1})`;
        svg.appendChild(yl);
        // Col label
        const xl = svgEl("text", { x: OX + i * cs + cs / 2, y: PT - 5, "text-anchor": "middle", "font-size": 10, fill: COLORS.muted });
        xl.textContent = `t+${i + 1}`;
        svg.appendChild(xl);
    }

    // Border around current step
    if (step.type === "decode" && curStep >= 0 && curStep < T) {
        const bx = OX, by = OY + curStep * cs;
        svg.appendChild(svgEl("rect", { x: bx, y: by, width: T * cs, height: cs, fill: "none", stroke: COLORS.orange, "stroke-width": 2, rx: 4 }));
    }

    // Legend
    ["Attended", "Blocked", "Active row"].forEach((lbl, i) => {
        const col = ["rgba(45,62,120,.7)", "#ede7f0", COLORS.orange][i];
        const rx = OX + (T * cs) + 14, ry = OY + i * 22;
        svg.appendChild(svgEl("rect", { x: rx, y: ry, width: 12, height: 12, fill: col, rx: 3 }));
        svg.appendChild(svgEl("text", { x: rx + 15, y: ry + 10, "font-size": 10, fill: COLORS.muted })).textContent = lbl;
    });
}

// ═══ Render forecast chart ═══
function renderForecast(step) {
    const svg = document.getElementById("forecastSvg");
    svg.innerHTML = "";
    const { hist, future, generated } = step;
    const L2 = hist.length, T2 = future.length;
    const allVals = [...hist, ...future];
    const mn = Math.min(...allVals) * 0.9, mx = Math.max(...allVals) * 1.1;
    const W = 540, H = 200, PL = 44, PT = 12, PR = 10, PB = 28;
    const tw = W - PL - PR, th = H - PT - PB;
    const N = L2 + T2;
    const xp = i => PL + i / (N - 1) * tw, yp = v => PT + th - (v - mn) / (mx - mn) * th;

    // Grid
    [mn + (mx - mn) * .25, mn + (mx - mn) * .5, mn + (mx - mn) * .75].forEach(v => {
        svg.appendChild(svgEl("line", { x1: PL, y1: yp(v), x2: PL + tw, y2: yp(v), stroke: COLORS.line, "stroke-width": .5 }));
        svg.appendChild(svgEl("text", { x: PL - 4, y: yp(v) + 3, "text-anchor": "end", "font-size": 9, fill: COLORS.muted })).textContent = v.toFixed(0);
    });

    // History fill
    const hfPath = "M" + [yp(mn), (hist[0])].join("") + "M" + hist.map((_, i) => `${xp(i)},${yp(hist[i])}`).join("L");
    svg.appendChild(svgEl("path", { d: `M${xp(0)},${yp(mn)} L` + hist.map((_, i) => `${xp(i)},${yp(hist[i])}`).join("L") + `L${xp(L2 - 1)},${yp(mn)}Z`, fill: "rgba(55,138,221,.12)" }));
    svg.appendChild(svgEl("path", { d: "M" + hist.map((_, i) => `${xp(i)},${yp(hist[i])}`).join("L"), fill: "none", stroke: "#378ADD", "stroke-width": 2, "stroke-linejoin": "round" }));

    // Separator
    svg.appendChild(svgEl("line", { x1: xp(L2 - 1), y1: PT, x2: xp(L2 - 1), y2: PT + th, stroke: COLORS.muted, "stroke-width": 1, "stroke-dasharray": "4 4" }));
    svg.appendChild(svgEl("text", { x: xp(L2 - 1) + 3, y: PT + 12, "font-size": 9, fill: COLORS.muted })).textContent = "t (now)";

    // True future (dashed, ghost)
    if (future && future.length > 0) {
        const futurePath = "M" + future.map((_, i) => `${xp(L2 + i)},${yp(future[i])}`).join("L");
        svg.appendChild(svgEl("path", { d: futurePath, fill: "none", stroke: "rgba(109,97,114,.3)", "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
    }

    // Generated so far
    if (generated && generated.length > 0) {
        const genPath = "M" + [`${xp(L2 - 1)},${yp(hist[L2 - 1])}`, ...generated.map((_, i) => `${xp(L2 + i)},${yp(generated[i])}`)].join("L");
        svg.appendChild(svgEl("path", { d: `M${xp(L2 - 1)},${yp(hist[L2 - 1])} L` + generated.map((_, i) => `${xp(L2 + i)},${yp(generated[i])}`).join("L") + `L${xp(L2 + generated.length - 1)},${yp(mn)}Z`, fill: "rgba(212,110,0,.10)" }));
        svg.appendChild(svgEl("path", { d: genPath, fill: "none", stroke: COLORS.orange, "stroke-width": 2.2, "stroke-linejoin": "round" }));
        // Dots
        generated.forEach((v, i) => {
            const isLast = i === generated.length - 1;
            svg.appendChild(svgEl("circle", { cx: xp(L2 + i), cy: yp(v), r: isLast ? 5 : 3.5, fill: isLast ? COLORS.orange : "rgba(212,110,0,.7)" }));
        });
    }

    // Axis labels
    svg.appendChild(svgEl("text", { x: PL + tw / 2, y: H - 4, "text-anchor": "middle", "font-size": 10, fill: COLORS.muted })).textContent = `← History (${L2} days) ·  Forecast (${T2} days) →`;
    svg.appendChild(svgEl("text", { x: PL - 3, y: PT + th / 2, "text-anchor": "middle", "font-size": 9, fill: COLORS.muted, transform: `rotate(-90,${PL - 14},${PT + th / 2})` })).textContent = "Q (m³/s)";
}

// ═══ Render text panels ═══
function renderText(step) {
    const s = step;
    // Formula card
    document.getElementById("stepBadge").textContent = s.badge;
    document.getElementById("formulaMode").textContent = s.mode;
    document.getElementById("formulaEquation").textContent = s.formula;
    document.getElementById("formulaNote").textContent = s.formulaNote;

    // Step summary
    document.getElementById("stepSummary").textContent = s.msg;

    // Exact math
    const T = state.T;
    const curStep = s.type === "decode" ? s.step - 1 : (s.type === "done" ? T - 1 : 0);
    const selfAttn = s.selfAttn;
    let exactHtml = "—";
    if (selfAttn && selfAttn.length > 0) {
        const row = selfAttn[selfAttn.length - 1];
        const wStr = row.map((w, j) => `α[${curStep},${j}]=${fmt(w, 3)}`).join(", ");
        exactHtml = `Row ${curStep} attention weights (after softmax):<br><strong>${wStr}</strong><br>Sum = ${row.reduce((a, b) => a + b, 0).toFixed(3)}`;
    }
    document.getElementById("exactMath").innerHTML = exactHtml;

    // Cross-attention
    if (s.crossW && s.topHistJ !== undefined) {
        const top3 = s.crossW.map((w, j) => ({ w, j })).sort((a, b) => b.w - a.w).slice(0, 3);
        document.getElementById("crossAttn").innerHTML =
            `Cross-attn peak at Day ${s.topHistJ + 1} of history (weight ${fmt(s.crossW[s.topHistJ], 3)}).<br>Top 3: ` +
            top3.map(({ w, j }) => `Day ${j + 1}→${fmt(w, 3)}`).join(", ") +
            `<br><em>The decoder is reading from the most relevant historical period.</em>`;
    } else {
        document.getElementById("crossAttn").textContent = "—";
    }

    // Causal mask formula
    document.getElementById("maskFormula").innerHTML =
        `M<sub>ij</sub> = { 0 if j ≤ i, &minus;∞ if j > i }<br><br>` +
        `A = softmax( (QK<sup>T</sup> / √d_k) + M )<br>` +
        `At step ${curStep}: j ∈ {0,…,${curStep}} active, j ∈ {${curStep + 1},…,${T - 1}} blocked`;

    // Interpretation
    const genN = s.generated ? s.generated.length : 0;
    let items = [];
    if (s.type === "init") {
        items.push(`Decoder is seeded with Q(t) = ${fmt(s.hist[s.hist.length - 1], 1)} m³/s. All T=${T} future positions are masked.`);
        items.push(`The full attention matrix is ${T}×${T}. Currently only position (0,0) is unmasked — the decoder cannot attend to anything yet.`);
    } else if (s.type === "decode") {
        items.push(`Step ${curStep + 1}/${T}: generating Q(t+${curStep + 1}). The query at row ${curStep} can attend to ${curStep + 1} positions (columns 0–${curStep}).`);
        items.push(`As the step progresses, the "context" grows — later predictions benefit from more prior forecasts to attend to.`);
        items.push(`Cross-attention to history ensures the model remembers the storm event ${state.L - (s.topHistJ || 0)} days ago.`);
    } else {
        items.push(`All ${T} streamflow values Q(t+1)…Q(t+${T}) generated. The full lower-triangular attention matrix is now complete.`);
        items.push(`Causal masking prevented the model from "cheating" at any step — physical causality was maintained throughout.`);
    }
    document.getElementById("interpretation").innerHTML = "<ul>" + items.map(t => `<li>${t}</li>`).join("") + "</ul>";
}

// ═══ Metrics ═══
function updateMetrics(step) {
    const T = state.T, L = state.L;
    const curStep = step.type === "decode" ? step.step - 1 : (step.type === "done" ? T - 1 : 0);
    const unmasked = step.type === "init" ? 1 : (curStep + 1) * (curStep + 2) / 2;
    const masked = T * T - unmasked;
    const genN = step.generated ? step.generated.length : 0;
    document.getElementById("metricStep").textContent = state.currentStep;
    document.getElementById("metricUnmasked").textContent = step.type === "init" ? "1" : Math.round(unmasked);
    document.getElementById("metricMasked").textContent = step.type === "done" ? 0 : Math.round(masked);
    document.getElementById("metricForecast").textContent = `${genN} / ${T}`;
    document.getElementById("metricHistory").textContent = step.crossW ? `${L} keys` : "—";
    document.getElementById("maskCaption").textContent =
        `T=${T} forecast steps. Row i = query position. Col j = key position. Lower triangle = attended, upper = masked.`;
}

// ═══ Master render ═══
function render() {
    const step = steps[state.currentStep] || steps[0];
    renderMask(step);
    renderForecast(step);
    renderText(step);
    updateMetrics(step);
    const slider = document.getElementById("stepSlider");
    slider.max = steps.length - 1;
    slider.value = state.currentStep;
}

// ═══ Transport controls ═══
function goToStep(i) {
    state.currentStep = Math.max(0, Math.min(steps.length - 1, i));
    render();
}

function startPlay() {
    if (state.playing) return;
    state.playing = true;
    document.getElementById("playBtn").textContent = "Pause";
    document.getElementById("playBtn").style.background = COLORS.orange;
    tick();
}

function pausePlay() {
    state.playing = false;
    if (playTimer) { clearTimeout(playTimer); playTimer = null; }
    document.getElementById("playBtn").textContent = "Play";
    document.getElementById("playBtn").style.background = "";
}

function tick() {
    if (!state.playing) return;
    if (state.currentStep >= steps.length - 1) { pausePlay(); return; }
    goToStep(state.currentStep + 1);
    playTimer = setTimeout(tick, state.speed);
}

// ═══ Reset & rebuild ═══
function rebuild() {
    pausePlay();
    state.currentStep = 0;
    steps = buildSteps();
    render();
}

// ═══ Events ═══
function attachEvents() {
    document.getElementById("resetBtn").addEventListener("click", rebuild);
    document.getElementById("prevBtn").addEventListener("click", () => goToStep(state.currentStep - 1));
    document.getElementById("nextBtn").addEventListener("click", () => goToStep(state.currentStep + 1));
    document.getElementById("playBtn").addEventListener("click", () => { if (state.playing) pausePlay(); else startPlay(); });
    document.getElementById("stepSlider").addEventListener("input", e => { pausePlay(); goToStep(+e.target.value); });
    document.getElementById("speedSelect").addEventListener("change", e => { state.speed = +e.target.value; });
    document.getElementById("tSlider").addEventListener("input", e => {
        state.T = +e.target.value;
        document.getElementById("tVal").textContent = state.T;
        rebuild();
    });
    document.getElementById("lSlider").addEventListener("input", e => {
        state.L = +e.target.value;
        document.getElementById("lVal").textContent = state.L;
        rebuild();
    });
}

attachEvents();
rebuild();
