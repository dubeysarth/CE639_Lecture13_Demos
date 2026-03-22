// ═══════════════════════════════════════════════════════════
//  Demo 4 — Transformer Architecture Explorer
//  CE 639: AI for Civil Engineering · Lecture 13
// ═══════════════════════════════════════════════════════════
const SVG_NS = "http://www.w3.org/2000/svg";
const COLORS = { purple: "#63045a", blue: "#2d3e78", orange: "#d46e00", green: "#2d7d5a", rose: "#c84f3d", muted: "#6d6172" };

const LAYER_COLORS = {
    input: { bg: "#E6F1FB", bc: "#185FA5", tc: "#0C447C" },
    embed: { bg: "#EAF3DE", bc: "#3B6D11", tc: "#27500A" },
    attn: { bg: "#EEEDFE", bc: "#534AB7", tc: "#3C3489" },
    cattn: { bg: "#FCEBEB", bc: "#A32D2D", tc: "#791F1F" },
    ffn: { bg: "#E1F5EE", bc: "#0F6E56", tc: "#085041" },
    decomp: { bg: "#FBEAF0", bc: "#993556", tc: "#72243E" },
    output: { bg: "#FAECE7", bc: "#993C1D", tc: "#712B13" },
};

const state = { L: 96, T: 24, d: 64, H: 4, arch: "enc-dec", selectedId: null };

// ═══ Architecture definitions ═══
function makeArchitectures() {
    const { L, T, d, H } = state;
    const dk = Math.floor(d / H);
    const ff = d * 4;
    const L2 = Math.ceil(L / 2);
    const L4 = Math.ceil(L / 4);
    const u = Math.max(3, Math.round(Math.log(L) * 2));

    return {
        "enc-only": {
            label: "Encoder-Only",
            desc: `BERT-style encoder. All ${L} timesteps processed in parallel; no decoder. Pooled representation projects to ${T} forecast steps in one shot.`,
            layers: [
                {
                    id: "eo0", t: "input", badge: "IN", name: "Raw hydro input",
                    sin: `[1,${L},3]`, sout: `[1,${L},3]`,
                    f: `3 features × ${L} days\n  P   = precipitation (mm/day)\n  PET = evapotranspiration\n  SWF = soil wetness [0,1]`,
                    det: "Full historical window. Model sees all L timesteps at once — no recurrence.",
                    hydro: "SWF encodes antecedent soil moisture — the key state variable linking rain to runoff.",
                    viz: "series"
                },
                {
                    id: "eo1", t: "embed", badge: "EMBED", name: "Linear embedding + PE",
                    sin: `[1,${L},3]`, sout: `[1,${L},${d}]`,
                    f: `X = input·W_emb + PE(pos)\nW_emb : [3, ${d}]\nPE: sinusoidal, period range ~6d – >>L`,
                    det: `Projects 3-d features into ${d}-d model space; adds sinusoidal PE so positions are distinguishable.`,
                    hydro: "Sinusoidal PE naturally encodes seasonal cycles — positions 1 and 182 have similar low-frequency components.",
                    viz: "pe"
                },
                {
                    id: "eo2", t: "attn", badge: "×N", name: "Multi-head self-attention",
                    sin: `[1,${L},${d}]`, sout: `[1,${L},${d}]`,
                    f: `Q=K=V=X   each [1,${L},${d}]\nA = softmax(QKᵀ/√${dk})\n    shape: [1,${L},${L}]\nOut = A·V  [1,${L},${d}]\n(${H} heads, d_k=${dk})`,
                    det: `Full bidirectional attention — ${H} heads, each computes a ${L}×${L} similarity matrix. No causal mask.`,
                    hydro: `Freely links any two days: day 15's SWF spike can directly inform day 30's baseflow forecast.`,
                    viz: "full"
                },
                {
                    id: "eo3", t: "ffn", badge: "×N", name: "FFN + LayerNorm",
                    sin: `[1,${L},${d}]`, sout: `[1,${L},${d}]`,
                    f: `FFN(x) = ReLU(x·W₁+b₁)·W₂+b₂\nW₁:[${d},${ff}]  W₂:[${ff},${d}]\nLN: (x−μ)/σ·γ+β`,
                    det: "Pointwise nonlinear transform. Same weights at every position. LayerNorm stabilises training.",
                    hydro: "Models threshold effects: runoff generated only when SWF ≥ field capacity (~0.6).",
                    viz: "ffn"
                },
                {
                    id: "eo4", t: "output", badge: "OUT", name: "Global pool → forecast",
                    sin: `[1,${L},${d}]`, sout: `[1,${T},1]`,
                    f: `z = mean(X, dim=time)  [1,${d}]\ny = z·W_out + b\nW_out: [${d},${T}]  →  [1,${T},1]`,
                    det: `Collapses time axis via mean-pooling, decodes all T forecasts simultaneously.`,
                    hydro: `One-shot forecast: all ${T} daily streamflows produced in one pass, no autoregressive error.`,
                    viz: "output"
                },
            ]
        },
        "enc-dec": {
            label: "Enc-Decoder",
            desc: `Original Transformer. Encoder summarises ${L}-step history; decoder autoregressively generates ${T} future steps via cross-attention to encoder memory.`,
            layers: [
                {
                    id: "ed0", t: "input", badge: "ENC IN", name: "Historical input",
                    sin: `[1,${L},3]`, sout: `[1,${L},3]`,
                    f: `Historical window [t−L+1 … t]\n  P, PET, SWF → [1,${L},3]`,
                    det: "Full L-step history given to encoder. No masking.",
                    hydro: `Longer L (${L} days) captures multi-week soil moisture memory and inter-event dry periods.`,
                    viz: "series"
                },
                {
                    id: "ed1", t: "embed", badge: "EMBED", name: "Encoder embed + PE",
                    sin: `[1,${L},3]`, sout: `[1,${L},${d}]`,
                    f: `X_enc = Linear(src) + PE\n        [1,${L},${d}]`, det: "", hydro: "", viz: "pe"
                },
                {
                    id: "ed2", t: "attn", badge: "×N", name: "Encoder self-attention",
                    sin: `[1,${L},${d}]`, sout: `[1,${L},${d}]`,
                    f: `Q=K=V=X_enc\nA_enc = softmax(QKᵀ/√${dk})·V\n        → [1,${L},${d}] = memory M`,
                    det: "Full bidirectional self-attention. Produces encoder memory M.",
                    hydro: "M encodes all temporal patterns: wet/dry cycles, flood pulses, recession limbs.",
                    viz: "full"
                },
                {
                    id: "ed3", t: "ffn", badge: "×N", name: "Encoder FFN+LN → M",
                    sin: `[1,${L},${d}]`, sout: `[1,${L},${d}]`,
                    f: `FFN + LayerNorm\nEncoder memory M : [1,${L},${d}]\n(passed to decoder cross-attn)`,
                    det: "Final encoder output M is the 'memory' decoder cross-attention reads from.", hydro: "", viz: "ffn"
                },
                {
                    id: "ed4", t: "input", badge: "DEC IN", name: "Target seed + placeholders",
                    sin: `[1,${T},3]`, sout: `[1,${T},3]`,
                    f: `x_tgt[0]   = last observed step\nx_tgt[1:T] = zeros\n\nshape: [1,${T},3]`,
                    det: `Decoder seeded with 1 real observation + ${T - 1} zero placeholders.`,
                    hydro: "The seed carries the last known SWF, anchoring the forecast to current conditions.",
                    viz: "series"
                },
                {
                    id: "ed5", t: "embed", badge: "EMBED", name: "Decoder embed + PE",
                    sin: `[1,${T},3]`, sout: `[1,${T},${d}]`,
                    f: `X_dec = Linear(tgt) + PE\n        [1,${T},${d}]`, det: "", hydro: "", viz: "pe"
                },
                {
                    id: "ed6", t: "attn", badge: "×M", name: "Masked self-attention",
                    sin: `[1,${T},${d}]`, sout: `[1,${T},${d}]`,
                    f: `Causal mask: M_ij = −∞ if j>i\nA = softmax((QKᵀ/√${dk})+M)·V\n    [1,${T},${T}] (lower-triangular)`,
                    det: "Prevents decoder from peeking at future target positions.",
                    hydro: `Q(t+3) forecast cannot attend to Q(t+5) — prevents data leakage.`,
                    viz: "causal"
                },
                {
                    id: "ed7", t: "cattn", badge: "×M", name: "Cross-attention ← encoder M",
                    sin: `[1,${T},${d}]`, sout: `[1,${T},${d}]`,
                    f: `Q  = X_dec  [1,${T},${d}]  ← decoder\nK,V = M      [1,${L},${d}]  ← encoder\n\nA_cross: [1,${T},${L}]\nOut = A_cross·V  [1,${T},${d}]`,
                    det: `Each decoder query (future step) attends to all ${L} encoder keys.`,
                    hydro: "A query for day t+5 attends to the historical peak — the model learns the catchment travel time.",
                    viz: "cross"
                },
                {
                    id: "ed8", t: "ffn", badge: "×M", name: "Decoder FFN+LN",
                    sin: `[1,${T},${d}]`, sout: `[1,${T},${d}]`,
                    f: `FFN + LayerNorm  [1,${T},${d}]`, det: "", hydro: "", viz: "ffn"
                },
                {
                    id: "ed9", t: "output", badge: "OUT", name: "Output projection",
                    sin: `[1,${T},${d}]`, sout: `[1,${T},1]`,
                    f: `y = X_dec·W_out + b\nW_out: [${d},1]\ny    : [1,${T},1] → T forecasts`,
                    det: "Each decoder hidden state maps to one forecast via a shared linear layer.",
                    hydro: `Outputs Q(t+1)…Q(t+${T}) in m³/s after inverse normalisation.`,
                    viz: "output"
                },
            ]
        },
        "dec-only": {
            label: "Decoder-Only",
            desc: `GPT-style causal-only stack. One unified sequence with causal masking — no encoder. Final T positions produce the T-step forecast.`,
            layers: [
                {
                    id: "do0", t: "input", badge: "IN", name: "Causal input",
                    sin: `[1,${L},3]`, sout: `[1,${L},3]`,
                    f: `Unified sequence [t−L+1…t]\n  P, PET, SWF → [1,${L},3]\n\nFuture generated autoregressively`,
                    det: "Single stream. Each position can only attend to its past.",
                    hydro: "Model learns rainfall-runoff dynamics purely from causal sequence patterns.", viz: "series"
                },
                {
                    id: "do1", t: "embed", badge: "EMBED", name: "Token embed + PE",
                    sin: `[1,${L},3]`, sout: `[1,${L},${d}]`,
                    f: `X = Linear(input) + PE\n    [1,${L},${d}]`, det: "", hydro: "", viz: "pe"
                },
                {
                    id: "do2", t: "attn", badge: "×N", name: "Causal self-attention",
                    sin: `[1,${L},${d}]`, sout: `[1,${L},${d}]`,
                    f: `Lower-triangular mask enforces causality\nA = softmax((QKᵀ/√${dk})+M)·V\n    [1,${L},${L}] masked`,
                    det: "Each position attends only to itself and all previous positions.",
                    hydro: "Models causal chain: rainfall → saturation → runoff. Position i cannot see future drivers.",
                    viz: "causal"
                },
                {
                    id: "do3", t: "ffn", badge: "×N", name: "FFN + LayerNorm",
                    sin: `[1,${L},${d}]`, sout: `[1,${L},${d}]`,
                    f: `FFN(x) = GELU(x·W₁+b₁)·W₂+b₂\n[1,${L},${d}]→[1,${L},${ff}]→[1,${L},${d}]`,
                    det: "GELU (smoother than ReLU) common in GPT-style models.", hydro: "", viz: "ffn"
                },
                {
                    id: "do4", t: "output", badge: "OUT", name: "Last-T slice → forecast",
                    sin: `[1,${L},${d}]`, sout: `[1,${T},1]`,
                    f: `X_last = X[:, ${L - T}:${L}, :]\n         [1,${T},${d}]\ny = X_last·W_out  [1,${T},1]`,
                    det: `Last T hidden states forecast the next T days.`,
                    hydro: `No future information ever seen — causal mask guarantees clean T-step forecasts.`,
                    viz: "output"
                },
            ]
        },
        "informer": {
            label: "Informer",
            desc: `Encoder-Decoder with ProbSparse attention O(L log L) and convolutional distilling that halves the sequence length at each encoder layer.`,
            layers: [
                {
                    id: "inf0", t: "input", badge: "IN", name: "Input + time features",
                    sin: `[1,${L},5]`, sout: `[1,${L},5]`,
                    f: `P, PET, SWF + month + day-of-year\n→ [1,${L},5]  (Informer style)`,
                    det: "Informer appends calendar features — helps the model align with seasonal cycles.",
                    hydro: "Direct month encoding injects monsoon phase without inferring it from PE alone.", viz: "series"
                },
                {
                    id: "inf1", t: "embed", badge: "EMBED", name: "Data+time+position embed",
                    sin: `[1,${L},5]`, sout: `[1,${L},${d}]`,
                    f: `X = DataEmbed + TimeEmbed + PosEmbed\n    [1,${L},${d}]`,
                    det: "Three separate learned embeddings summed.", hydro: "", viz: "pe"
                },
                {
                    id: "inf2", t: "attn", badge: "×N", name: "ProbSparse self-attention",
                    sin: `[1,${L},${d}]`, sout: `[1,${L},${d}]`,
                    f: `Top-u queries selected: u = c·ln(L) ≈ ${u}\nScore: max_j(q·kᵀ) − mean_j(q·kᵀ)\nTop-u→ full attn, rest → mean(V)\nComplexity: O(L·ln L) vs O(L²)`,
                    det: `Only ${u} most informative queries compute full attention; remaining ${L - u} use mean(V).`,
                    hydro: `For L=${L}: O(L²)=${L * L} ops; ProbSparse≈${L * u} ops — ${Math.round(L / u)}× fewer.`,
                    viz: "sparse"
                },
                {
                    id: "inf3", t: "decomp", badge: "DISTIL", name: "Convolutional distilling",
                    sin: `[1,${L},${d}]`, sout: `[1,${Math.ceil(L / 2)},${d}]`,
                    f: `Conv1D(d, kernel=3) → ELU\n→ MaxPool(stride=2)\n[1,${L},${d}] → [1,${Math.ceil(L / 2)},${d}]`,
                    det: `Halves sequence length after every encoder layer. Dominant features survive max-pool.`,
                    hydro: `After distilling: ${Math.ceil(L / 4)} encoder steps. Peak flows (high values) survive max-pool; flat dry-periods compress efficiently.`,
                    viz: "distil"
                },
                {
                    id: "inf4", t: "output", badge: "OUT", name: "Decoder + output projection",
                    sin: `[1,${Math.ceil(L / 2) + T},${d}]`, sout: `[1,${T},1]`,
                    f: `Dec input = concat(X_enc[-${Math.ceil(L / 2)}:], zeros_T)\nEmbedded → MaskedAttn → CrossAttn\ny = X_dec[-${T}:]·W_out  [1,${T},1]`,
                    det: `Decoder seeded with last L/2 observations — richer context than a single seed.`,
                    hydro: `All ${T} forecasts in one pass — no autoregressive error compounding.`,
                    viz: "output"
                },
            ]
        },
        "autoformer": {
            label: "Autoformer",
            desc: `Replaces dot-product attention with FFT auto-correlation O(L log L). Series decomposition blocks explicitly separate trend and seasonal components.`,
            layers: [
                {
                    id: "af0", t: "input", badge: "IN", name: "Input series",
                    sin: `[1,${L},3]`, sout: `[1,${L},3]`,
                    f: `X_raw : [1,${L},3]\n  P, PET, SWF`, det: "", hydro: "", viz: "series"
                },
                {
                    id: "af1", t: "embed", badge: "EMBED", name: "Embedding + PE",
                    sin: `[1,${L},3]`, sout: `[1,${L},${d}]`,
                    f: `X = Linear(input) + PE\n    [1,${L},${d}]`, det: "", hydro: "", viz: "pe"
                },
                {
                    id: "af2", t: "decomp", badge: "DECOMP", name: "Series decomposition block",
                    sin: `[1,${L},${d}]`, sout: `[1,${L},${d}]`,
                    f: `X_trend   = AvgPool1D(pad(X), k=25)\nX_seasonal = X − X_trend\n\nApplied before/after every attention`,
                    det: "Moving-average kernel extracts smooth trend. Residual is the seasonal/high-freq part.",
                    hydro: "Separates slow soil-drying trend (multi-week) from daily PET-rainfall oscillations.",
                    viz: "decomp"
                },
                {
                    id: "af3", t: "attn", badge: "×N", name: "Auto-Correlation block",
                    sin: `[1,${L},${d}]`, sout: `[1,${L},${d}]`,
                    f: `S(τ) = IFFT(FFT(Q)·conj(FFT(K)))\nTop-k lags: k = c·ln(L) ≈ ${u}\nOut = Σ_τ* w_τ · Roll(V,τ)\nComplexity: O(L·log L)`,
                    det: `Finds the ${u} time-lags at which the series best auto-correlates, then aggregates V shifted by those lags.`,
                    hydro: `Automatically discovers the catchment travel time τ* (e.g., rainfall-to-peak lag of 2 days).`,
                    viz: "autocorr"
                },
                {
                    id: "af4", t: "output", badge: "OUT", name: "Decomposed output projection",
                    sin: `[1,${T},${d}]`, sout: `[1,${T},1]`,
                    f: `y_seasonal = W_s · X_s[-${T}:]\ny_trend    = W_t · trend[-${T}:]\ny = y_seasonal + y_trend`,
                    det: "Explicit sum of seasonal and trend forecasts — interpretable decomposition.",
                    hydro: "You can separately inspect trend forecast (drying catchment) vs seasonal forecast (wet-season spike).",
                    viz: "output"
                },
            ]
        }
    };
}

// ═══ SVG mini-visualizations ═══
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

function drawViz(ctx2d, W, H, type) {
    const { L, T, d, H: nh } = state;
    const dk = Math.floor(d / nh);
    const u = Math.max(3, Math.round(Math.log(L) * 2));

    ctx2d.clearRect(0, 0, W, H);
    ctx2d.fillStyle = "#f8f4fa"; ctx2d.fillRect(0, 0, W, H);

    if (type === "series") {
        const N = Math.min(L, 55), pw = (W - 40) / N, ph = H - 28;
        const data = Array.from({ length: N }, (_, i) => 0.3 + .4 * Math.sin(i * .35) + .2 * Math.sin(i * .9) + .1 * (Math.sin(i * 2.1) * .5));
        const peaks = [Math.floor(N * .28), Math.floor(N * .62)];
        peaks.forEach(p => { if (p < N) data[p] += 0.35; if (p + 1 < N) data[p + 1] += 0.18; });
        // Fill
        ctx2d.beginPath(); ctx2d.moveTo(20, 8 + ph);
        data.forEach((v, i) => ctx2d.lineTo(20 + i * pw, 8 + ph - v * ph * .85));
        ctx2d.lineTo(20 + (N - 1) * pw, 8 + ph); ctx2d.closePath();
        ctx2d.fillStyle = "rgba(55,138,221,.12)"; ctx2d.fill();
        // Line
        ctx2d.beginPath(); data.forEach((v, i) => i ? ctx2d.lineTo(20 + i * pw, 8 + ph - v * ph * .85) : ctx2d.moveTo(20 + i * pw, 8 + ph - v * ph * .85));
        ctx2d.strokeStyle = "#378ADD"; ctx2d.lineWidth = 1.8; ctx2d.stroke();
        ['P', 'PET', 'SWF'].forEach((l, i) => {
            ctx2d.fillStyle = ["#378ADD", "#1D9E75", "#EF9F27"][i]; ctx2d.fillRect(22 + i * 58, H - 12, 10, 3);
            ctx2d.fillStyle = COLORS.muted; ctx2d.font = "9px sans-serif"; ctx2d.fillText(l, 34 + i * 58, H - 7);
        });
    }
    else if (type === "pe") {
        const N = Math.min(L, 48), rows = 6, cw = (W - 20) / N, rh = (H - 16) / rows;
        for (let r = 0; r < rows; r++) for (let i = 0; i < N; i++) {
            const v = r % 2 === 0 ? Math.sin(i / Math.pow(10000, r / d)) : Math.cos(i / Math.pow(10000, (r - 1) / d));
            const col = v < 0 ? lerpColor("#2d3e78", "#f5f0f6", (v + 1)) : lerpColor("#f5f0f6", "#63045a", v);
            ctx2d.fillStyle = col; ctx2d.fillRect(10 + i * cw, r * rh + 6, Math.max(1, cw - .3), Math.max(1, rh - .3));
        }
        ctx2d.fillStyle = COLORS.muted; ctx2d.font = "9px sans-serif"; ctx2d.textAlign = "center";
        ctx2d.fillText(`PE: ${rows} of ${d} dims × ${N} positions`, W / 2, H - 2);
        ctx2d.textAlign = "left";
    }
    else if (type === "full" || type === "causal") {
        const S = Math.min(L, 20), cs = Math.min((W - 40) / S, (H - 26) / S);
        const ox = (W - cs * S) / 2, oy = 12;
        for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) {
            if (type === "causal" && j > i) { ctx2d.fillStyle = "#e8e0ee"; ctx2d.fillRect(ox + j * cs, oy + i * cs, cs - .5, cs - .5); continue; }
            const d2 = Math.abs(i - j), s = Math.exp(-d2 * .2) * .7 + (Math.sin(i * .5) * Math.sin(j * .5) + 1) / 2 * .25 + (i === j ? .25 : 0);
            ctx2d.fillStyle = lerpColor("#E6F1FB", "#63045a", Math.min(1, s));
            ctx2d.fillRect(ox + j * cs, oy + i * cs, cs - .5, cs - .5);
        }
        ctx2d.fillStyle = COLORS.muted; ctx2d.font = "9px sans-serif"; ctx2d.textAlign = "center";
        ctx2d.fillText(`Self-attn [${S}×${S} sample]` + (type === "causal" ? " — causal mask" : ""), W / 2, H - 3);
        ctx2d.textAlign = "left";
    }
    else if (type === "cross") {
        const TT = Math.min(T, 10), LL = Math.min(L, 20);
        const cw2 = (W - 30) / LL, ch2 = (H - 24) / TT;
        for (let i = 0; i < TT; i++) for (let j = 0; j < LL; j++) {
            const v = Math.exp(-Math.abs(j - i * (LL / TT)) * .25) * .8 + (j < 4 ? .3 * (i < 3 ? 1 : .3) : 0);
            ctx2d.fillStyle = lerpColor("#F1EFE8", "#A32D2D", Math.min(1, v));
            ctx2d.fillRect(15 + j * cw2, 12 + i * ch2, cw2 - .5, ch2 - .5);
        }
        ctx2d.fillStyle = COLORS.muted; ctx2d.font = "9px sans-serif"; ctx2d.textAlign = "center";
        ctx2d.fillText(`Cross-attn: ${TT} decoder × ${LL} encoder keys`, W / 2, H - 3); ctx2d.textAlign = "left";
    }
    else if (type === "ffn") {
        const bx = [W * .18, W * .5, W * .82], bH = [H * .55, H * .82, H * .55], lbl = [`d=${d}`, `4d=${d * 4}`, `d=${d}`];
        const cy = H / 2 - 4;
        bx.forEach((x, i) => {
            const h = bH[i]; ctx2d.fillStyle = "#E1F5EE"; ctx2d.fillRect(x - 18, cy - h / 2, 36, h);
            ctx2d.strokeStyle = "#0F6E56"; ctx2d.lineWidth = .8; ctx2d.strokeRect(x - 18, cy - h / 2, 36, h);
            ctx2d.fillStyle = COLORS.muted; ctx2d.font = "9px sans-serif"; ctx2d.textAlign = "center"; ctx2d.fillText(lbl[i], x, cy + 4);
        });
        ctx2d.textAlign = "left";
    }
    else if (type === "sparse") {
        const S = Math.min(L, 18), cs = Math.min((W - 40) / S, (H - 26) / S);
        const ox = (W - cs * S) / 2, oy = 12;
        const uS = Math.max(2, Math.round(Math.log(S) * 2));
        const actRows = Array.from({ length: uS }, (_, i) => Math.floor(i * S / uS));
        for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) {
            ctx2d.fillStyle = "rgba(0,0,0,.04)"; ctx2d.fillRect(ox + j * cs, oy + i * cs, cs - .5, cs - .5);
        }
        actRows.forEach(i => {
            for (let j = 0; j < S; j++) {
                const v = .3 + .6 * Math.exp(-Math.abs(j - i) * .3);
                ctx2d.fillStyle = lerpColor("#FAEEDA", "#EF9F27", Math.min(1, v));
                ctx2d.fillRect(ox + j * cs, oy + i * cs, cs - .5, cs - .5);
            }
        });
        ctx2d.fillStyle = COLORS.muted; ctx2d.font = "9px sans-serif"; ctx2d.textAlign = "center";
        ctx2d.fillText(`ProbSparse: ${uS} active queries (amber) of ${S}`, W / 2, H - 3); ctx2d.textAlign = "left";
    }
    else if (type === "distil") {
        const N1 = Math.min(L, 28), N2 = Math.ceil(N1 / 2), bh = H * .4, cw1 = (W * .38) / N1, cw2 = (W * .38) / N2, oy = (H - bh) / 2;
        const ox1 = W * .06, ox2 = W * .55;
        for (let i = 0; i < N1; i++) { const v = .3 + .45 * Math.sin(i * .4); ctx2d.fillStyle = `rgba(55,138,221,${.15 + v * .5})`; ctx2d.fillRect(ox1 + i * cw1, oy, Math.max(1, cw1 - .3), bh); }
        for (let i = 0; i < N2; i++) { const v = .4 + .45 * Math.sin(i * .8); ctx2d.fillStyle = `rgba(29,158,117,${.15 + v * .55})`; ctx2d.fillRect(ox2 + i * cw2, oy, Math.max(1, cw2 - .3), bh); }
        ctx2d.strokeStyle = "#378ADD"; ctx2d.lineWidth = .8; ctx2d.strokeRect(ox1, oy, N1 * cw1, bh);
        ctx2d.strokeStyle = "#1D9E75"; ctx2d.lineWidth = .8; ctx2d.strokeRect(ox2, oy, N2 * cw2, bh);
        ctx2d.fillStyle = COLORS.muted; ctx2d.font = "9px sans-serif"; ctx2d.textAlign = "center";
        ctx2d.fillText(`[L=${N1},d]`, ox1 + N1 * cw1 / 2, oy - 3); ctx2d.fillText(`[L/2=${N2},d]`, ox2 + N2 * cw2 / 2, oy - 3);
        ctx2d.fillText("MaxPool →", W / 2, H / 2 + 3); ctx2d.textAlign = "left";
    }
    else if (type === "decomp") {
        const N = Math.min(L, 36), cw3 = (W - 30) / N;
        const orig = Array.from({ length: N }, (_, i) => 0.5 + .3 * Math.sin(i * .35) + .18 * Math.sin(i * .9) + .05 * (Math.random() - .5));
        const k = 4, trend = orig.map((_, i) => { const s = Math.max(0, i - k), e = Math.min(N - 1, i + k); return orig.slice(s, e + 1).reduce((a, b) => a + b, 0) / (e - s + 1); });
        const seas = orig.map((v, i) => v - trend[i]);
        [[orig, "#378ADD", "orig"], [trend, "#993556", "trend"]].forEach(([data, col, lbl], ri) => {
            const ry = ri * (H - 12) / 2 + 4, rh = (H - 16) / 2;
            const mn = Math.min(...data), mx = Math.max(...data) + .001;
            ctx2d.beginPath(); data.forEach((v, i) => { const x = 15 + i * cw3, y = ry + rh * .85 - (v - mn) / (mx - mn) * rh * .75; i ? ctx2d.lineTo(x, y) : ctx2d.moveTo(x, y); });
            ctx2d.strokeStyle = col; ctx2d.lineWidth = 1.5; ctx2d.stroke();
            ctx2d.fillStyle = col; ctx2d.font = "9px sans-serif"; ctx2d.fillText(lbl, 17, ry + 9);
        });
    }
    else if (type === "autocorr") {
        const N = 22, lags = Array.from({ length: N }, (_, i) => i + 1);
        const series = Array.from({ length: 60 }, (_, i) => Math.sin(i * .4) + .25 * Math.sin(i * 1.2));
        const corrs = lags.map(lag => { let s = 0; for (let i = 0; i < 60 - lag; i++)s += series[i] * series[i + lag]; return s / (60 - lag); });
        const mx = Math.max(...corrs.map(Math.abs)) + .001;
        const bw = (W - 40) / N, cy = H / 2 + 2;
        lags.forEach((lag, i) => {
            const h = (corrs[i] / mx) * (H * .34), active = i < Math.ceil(N * .32);
            ctx2d.fillStyle = active ? "rgba(239,159,39,.85)" : "rgba(29,158,117,.5)";
            ctx2d.fillRect(20 + i * bw, cy - Math.max(0, h), bw - 1.5, Math.abs(h));
        });
        ctx2d.strokeStyle = COLORS.muted; ctx2d.lineWidth = .5;
        ctx2d.beginPath(); ctx2d.moveTo(16, cy); ctx2d.lineTo(W - 6, cy); ctx2d.stroke();
        ctx2d.fillStyle = COLORS.muted; ctx2d.font = "9px sans-serif"; ctx2d.textAlign = "center";
        ctx2d.fillText("Auto-correlation by lag τ — amber = top-k selected", W / 2, H - 3); ctx2d.textAlign = "left";
    }
    else if (type === "output") {
        const NN = Math.min(d, 18), TT = Math.min(T, 10);
        const cw4 = (W * .32) / NN, cw5 = (W * .16) / TT, oy = (H - 48) / 2 + 4;
        const ox1 = W * .06, ox2 = W * .79;
        for (let i = 0; i < NN; i++) { const v = .2 + .55 * Math.abs(Math.sin(i * .6)); ctx2d.fillStyle = `rgba(55,138,221,${v})`; ctx2d.fillRect(ox1 + i * cw4, oy, Math.max(1, cw4 - .4), 48); }
        ctx2d.strokeStyle = "#378ADD"; ctx2d.lineWidth = .8; ctx2d.strokeRect(ox1, oy, NN * cw4, 48);
        for (let j = 0; j < TT; j++) { const v = 20 - Math.abs(j - TT / 2) * 1.6; ctx2d.fillStyle = "rgba(153,60,29,.85)"; ctx2d.fillRect(ox2 + j * cw5, oy + (48 - Math.max(4, v)) / 2, Math.max(1, cw5 - .4), Math.max(4, v)); }
        ctx2d.strokeStyle = "#993C1D"; ctx2d.lineWidth = .8; ctx2d.strokeRect(ox2, oy, TT * cw5, 48);
        ctx2d.fillStyle = COLORS.muted; ctx2d.font = "9px sans-serif"; ctx2d.textAlign = "center";
        ctx2d.fillText(`[T,d=${d}]`, ox1 + NN * cw4 / 2, oy - 3); ctx2d.fillText(`[T=${T},1]`, ox2 + TT * cw5 / 2, oy - 3);
        ctx2d.fillText(`→ ${T} streamflow forecasts (m³/s)`, W / 2, H - 3); ctx2d.textAlign = "left";
    }
}

// ═══ Render flow diagram ═══
function renderFlow() {
    const arch = makeArchitectures()[state.arch];
    document.getElementById("archDesc").textContent = arch.desc;
    const container = document.getElementById("flowDiagram");
    container.innerHTML = "";

    arch.layers.forEach((layer, i) => {
        const c = LAYER_COLORS[layer.t] || LAYER_COLORS.input;
        const sel = state.selectedId === layer.id;
        const div = document.createElement("div");
        div.className = "flow-layer" + (sel ? " selected" : "");
        div.style.cssText = `background:${sel ? "rgba(99,4,90,0.06)" : c.bg};border-color:${sel ? c.bc : "rgba(0,0,0,0.07)"};`;
        div.innerHTML = `<div class="fl-top">
      <span class="fl-badge" style="background:${c.bg};color:${c.tc};border:.5px solid ${c.bc}">${layer.badge}</span>
      <span class="fl-name" style="color:${c.tc}">${layer.name}</span>
    </div>
    <div class="fl-shapes">
      <span>${layer.sin}</span>
      ${layer.sout !== layer.sin ? `<span style="opacity:.5">→</span><span style="color:${c.tc};font-weight:600">${layer.sout}</span>` : ""}
    </div>`;
        div.addEventListener("click", () => { state.selectedId = layer.id; render(); });
        container.appendChild(div);
        if (i < arch.layers.length - 1) {
            const arr = document.createElement("div");
            arr.className = "fl-arrow";
            arr.innerHTML = `<svg width=10 height=12 viewBox="0 0 10 12"><path d="M5 0v8M1 5l4 6 4-6" stroke="${COLORS.muted}" stroke-width="1.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
            container.appendChild(arr);
        }
    });
}

// ═══ Render detail ═══
function renderDetail() {
    const cont = document.getElementById("detailContent");
    if (!state.selectedId) { cont.innerHTML = '<div class="empty-hint">← Click any layer block on the left</div>'; return; }
    const arch = makeArchitectures()[state.arch];
    const layer = arch.layers.find(l => l.id === state.selectedId);
    if (!layer) return;
    const c = LAYER_COLORS[layer.t] || LAYER_COLORS.input;
    document.getElementById("detailTitle").textContent = layer.name;
    document.getElementById("detailSub").textContent = layer.det || "—";
    document.getElementById("metricLayer").textContent = layer.badge;

    const W = 360, H = 130;
    cont.innerHTML = `
    <div class="detail-formula">${layer.f}</div>
    <div class="detail-shape-row">
      <span class="shape-box">${layer.sin}</span>
      <span class="shape-arrow">→</span>
      <span class="shape-box" style="border-color:${c.bc};color:${c.tc}">${layer.sout}</span>
    </div>
    <canvas id="detailViz" class="detail-vizcanvas" width=${W * 2} height=${H * 2} style="height:${H}px"></canvas>
    ${layer.hydro ? `<div class="hydro-note"><strong>Hydrology: </strong>${layer.hydro}</div>` : ""}
  `;
    const cv = document.getElementById("detailViz");
    cv.width = W * 2; cv.height = H * 2;
    const ctx = cv.getContext("2d"); ctx.scale(2, 2);
    drawViz(ctx, W, H, layer.viz);
}

// ═══ Arch tabs ═══
function renderTabs() {
    const archs = makeArchitectures();
    const container = document.getElementById("archTabs");
    container.innerHTML = "";
    Object.entries(archs).forEach(([k, v]) => {
        const b = document.createElement("button");
        b.className = "arch-tab" + (k === state.arch ? " on" : "");
        b.textContent = v.label;
        b.addEventListener("click", () => { state.arch = k; state.selectedId = null; render(); });
        container.appendChild(b);
    });
}

// ═══ Math & Metrics ═══
function renderMath() {
    const { L, T, d, H: nh } = state;
    const dk = Math.floor(d / nh);
    const u = Math.max(3, Math.round(Math.log(L) * 2));
    document.getElementById("mathComplexity").innerHTML =
        `Standard: O(L²·d) = O(${L * L}·${d})<br>ProbSparse (Informer): O(L·ln L·d) ≈ O(${L * u}·${d})<br><strong>${Math.round((L * L) / (L * u))}× reduction with ProbSparse</strong>`;
    // Approximate params for enc-dec
    const nLayer = 2;
    const params = nLayer * (4 * d * d + 2 * d * d * 4) + d * 3 + d * 1;
    document.getElementById("mathParams").innerHTML =
        `Per layer (attn+FFN): ≈ ${(4 * d * d + 2 * d * d * 4).toLocaleString()}<br>2 layers × 2 stacks ≈ <strong>${params.toLocaleString()} params</strong><br>(d=${d}, ${nh} heads, d_k=${dk})`;
}

function updateMetrics() {
    const { L, T, d, H: nh } = state;
    const dk = Math.floor(d / nh);
    const params = 2 * (4 * d * d + 2 * d * d * 4) + d * 3 + d * 1;
    document.getElementById("metricParams").textContent = params.toLocaleString();
    document.getElementById("metricAttnOps").textContent = `${L * L}`;
    document.getElementById("metricDk").textContent = dk;
    document.getElementById("metricFFN").textContent = d * 4;
}

// ═══ Control labels ═══
function updateControlLabels() {
    document.getElementById("lValue").textContent = state.L;
    document.getElementById("tValue").textContent = state.T;
    document.getElementById("dValue").textContent = state.d;
    document.getElementById("hValue").textContent = state.H;
}

// ═══ Interpretation ═══
function updateInterpretation() {
    const { L, T, d, H: nh, arch } = state;
    const dk = Math.floor(d / nh);
    const archLabels = { "enc-only": "Encoder-Only", "enc-dec": "Encoder-Decoder", "dec-only": "Decoder-Only", "informer": "Informer", "autoformer": "Autoformer" };
    let items = [];
    items.push(`<strong>${archLabels[arch]}</strong>: history L=${L} days, forecast T=${T} days, d=${d}, ${nh} heads (d_k=${dk}).`);
    if (arch === "enc-dec") items.push(`Cross-attention is the key component — each of the ${T} decoder queries reads all ${L} encoder memory positions, allowing the model to learn the catchment's travel time.`);
    if (arch === "dec-only") items.push(`Causal masking enforces physical causality: position i only attends to j≤i. All ${T} future positions are generated autoregressively.`);
    if (arch === "informer") items.push(`ProbSparse attention reduces quadratic O(${L * L}) to O(${L * Math.max(3, Math.round(Math.log(L) * 2))}) — critical for processing multi-year streamflow records.`);
    if (arch === "autoformer") items.push(`Series decomposition explicitly separates the slow soil-moisture trend from daily rainfall oscillations — no other architecture does this by design.`);
    items.push(`Attention matrix is ${L}×${L} = ${L * L} values. For L>100, standard attention becomes the bottleneck — motivating Informer, PatchTST, and other efficient variants.`);
    document.getElementById("interpretation").innerHTML = "<ul>" + items.map(t => `<li>${t}</li>`).join("") + "</ul>";
}

// ═══ Main render ═══
function render() {
    updateControlLabels();
    renderTabs();
    renderFlow();
    renderDetail();
    renderMath();
    updateMetrics();
    updateInterpretation();
}

// ═══ Events ═══
function attachEvents() {
    [["lSlider", "L"], ["tSlider", "T"], ["dSlider", "d"], ["hSlider", "H"]].forEach(([id, key]) => {
        document.getElementById(id).addEventListener("input", e => {
            state[key] = +e.target.value;
            state.selectedId = null;
            render();
        });
    });
    document.getElementById("resetBtn").addEventListener("click", () => {
        state.L = 96; state.T = 24; state.d = 64; state.H = 4; state.arch = "enc-dec"; state.selectedId = null;
        document.getElementById("lSlider").value = 96;
        document.getElementById("tSlider").value = 24;
        document.getElementById("dSlider").value = 64;
        document.getElementById("hSlider").value = 4;
        render();
    });
}

attachEvents();
render();
