const PHASES = [
  {
    short: 'Raw',
    name: 'Raw inputs',
    title: 'Raw hydrological inputs',
    equation: `x^(t) = [rain_t, temp_t, soil_t, Q_t]`,
    shape: `[30 × 4] source history`,
    plain: 'The source sequence is a 30-day table. Each day has rainfall, temperature, soil moisture, and observed streamflow.',
    hydro: 'This is the hydrological memory presented to the model: recent forcings, antecedent wetness, and catchment response.',
    why: 'A clean source sequence defines the forecasting problem. Everything later in the architecture depends on how this history is represented.'
  },
  {
    short: 'Embed + PE',
    name: 'Embeddings + positional encoding',
    title: 'Projected tokens plus temporal position',
    equation: `x̃^(t) = x^(t) W_emb + b_emb\nz^(t) = x̃^(t) + PE(t)`,
    shape: `[30 × 6] embedded source tokens`,
    plain: 'Each 4-variable day is projected into a 6-dimensional model space, then enriched with a positional signal so the model knows where each token sits in time.',
    hydro: 'Without positional encoding, the model could see “storm” and “wet soil” patterns but would struggle to know whether they happened yesterday or three weeks ago.',
    why: 'Attention is permutation-invariant unless position is injected. Streamflow forecasting needs ordering because timing drives causality.'
  },
  {
    short: 'Enc SA',
    name: 'Encoder self-attention',
    title: 'Encoder self-attention across the 30-day history',
    equation: `Attention(Q, K, V) = softmax(QK^T / √d_k) V`,
    shape: `[30 × 30] attention scores → [30 × 6] context`,
    plain: 'Each source day can look at every other source day and compute a weighted summary of what matters for its representation.',
    hydro: 'A wet day near the forecast origin can attend backward to the older storm pulse that produced the stored catchment wetness it still reflects.',
    why: 'This is where the model escapes the single hidden-state bottleneck of an RNN.'
  },
  {
    short: 'Enc FFN',
    name: 'Encoder FFN',
    title: 'Position-wise feed-forward refinement',
    equation: `FFN(h) = W_2 ReLU(W_1 h + b_1) + b_2`,
    shape: `[30 × 6] → [30 × 6] position-wise transform`,
    plain: 'After attention mixes information across time, a feed-forward network transforms each position independently to refine the feature representation.',
    hydro: 'Think of this as sharpening each day’s latent hydrological signature after it has already gathered context from the full history.',
    why: 'Attention handles who talks to whom; the FFN improves what each position says after that interaction.'
  },
  {
    short: 'Dec shift',
    name: 'Decoder input shift',
    title: 'Shifted target sequence into the decoder',
    equation: `decoder input = [<BOS>, y_1, y_2, …, y_6]`,
    shape: `[7 × 1] target values → [7 × 6] decoder states`,
    plain: 'The decoder does not receive the future output directly. It receives a shifted version so position 3 predicts day 3 using earlier target positions only.',
    hydro: 'Forecast day 4 can condition on earlier forecasted hydrograph shape, but not on later ungenerated flows.',
    why: 'The shift aligns autoregressive prediction with parallel training.'
  },
  {
    short: 'Masked SA',
    name: 'Masked self-attention',
    title: 'Causal mask inside the decoder',
    equation: `M_{i,j} = 0  for  j ≤ i\nM_{i,j} = -∞ for  j > i`,
    shape: `[7 × 7] masked target attention`,
    plain: 'The decoder can see only current and earlier target positions. Future target positions are blocked before softmax.',
    hydro: 'Forecast day 2 may use day 1, but it must not leak information from day 5 during training.',
    why: 'This preserves causal forecasting behavior while still allowing batched training.'
  },
  {
    short: 'Cross-attn',
    name: 'Cross-attention',
    title: 'Decoder queries read encoder memory',
    equation: `CrossAttn(Q_dec, K_enc, V_enc)\nQ from decoder, K/V from encoder`,
    shape: `[7 × 30] decoder-to-source attention`,
    plain: 'Each decoder position forms a query and reads the encoded 30-day source memory to extract the most relevant forcing and state history.',
    hydro: 'Here forecast day k can selectively look back to the late storm pulse and the recent wetness state that jointly drive the upcoming hydrograph.',
    why: 'This is the bridge between what will be predicted and what has already happened.'
  },
  {
    short: 'Output',
    name: 'Output projection',
    title: 'Project decoder states to discharge',
    equation: `ŷ_k = h_k W_out + b_out`,
    shape: `[7 × 6] → [7 × 1] forecast discharge`,
    plain: 'The decoder’s hidden states are projected to scalar streamflow values, one for each forecast day.',
    hydro: 'The final hydrograph reflects both short-memory wetness and the lingering impact of the earlier rainfall pulse.',
    why: 'The architecture becomes hydrologically useful only when the latent sequence is turned into actual discharge predictions.'
  }
];

const state = {
  phase: 0,
  mode: 'training',
  showShapes: true,
  showMask: true,
  decoderStep: 1,
  playing: false,
  timer: null
};

const DATA = buildDataset();
const refs = {};

document.addEventListener('DOMContentLoaded', init);

function init() {
  refs.phaseSlider = document.getElementById('phaseSlider');
  refs.phaseTicks = document.getElementById('phaseTicks');
  refs.resetBtn = document.getElementById('resetBtn');
  refs.prevBtn = document.getElementById('prevBtn');
  refs.playBtn = document.getElementById('playBtn');
  refs.nextBtn = document.getElementById('nextBtn');
  refs.modeToggle = document.getElementById('modeToggle');
  refs.shapesToggle = document.getElementById('shapesToggle');
  refs.maskToggle = document.getElementById('maskToggle');
  refs.decoderStepSelect = document.getElementById('decoderStepSelect');

  refs.metricPhase = document.getElementById('metricPhase');
  refs.metricStep = document.getElementById('metricStep');
  refs.metricShape = document.getElementById('metricShape');
  refs.metricContext = document.getElementById('metricContext');
  refs.metricLeakage = document.getElementById('metricLeakage');
  refs.metricCost = document.getElementById('metricCost');

  refs.architecturePanel = document.getElementById('architecturePanel');
  refs.activeComputationPanel = document.getElementById('activeComputationPanel');
  refs.activePanelTitle = document.getElementById('activePanelTitle');
  refs.equationBlock = document.getElementById('equationBlock');
  refs.shapeBlock = document.getElementById('shapeBlock');
  refs.plainBlock = document.getElementById('plainBlock');
  refs.hydroBlock = document.getElementById('hydroBlock');
  refs.whyBlock = document.getElementById('whyBlock');
  refs.forecastPanel = document.getElementById('forecastPanel');
  refs.forecastCaption = document.getElementById('forecastCaption');
  refs.teacherStep = document.getElementById('teacherStep');

  setupPhaseTicks();
  setupDecoderSelector();
  bindEvents();
  render();
}

function bindEvents() {
  refs.phaseSlider.addEventListener('input', (e) => {
    state.phase = Number(e.target.value);
    stopPlay();
    render();
  });

  refs.resetBtn.addEventListener('click', () => {
    state.phase = 0;
    state.mode = 'training';
    state.showShapes = true;
    state.showMask = true;
    state.decoderStep = 1;
    stopPlay();
    syncControls();
    render();
  });

  refs.prevBtn.addEventListener('click', () => {
    stopPlay();
    state.phase = (state.phase + PHASES.length - 1) % PHASES.length;
    syncControls();
    render();
  });

  refs.nextBtn.addEventListener('click', () => {
    stopPlay();
    state.phase = (state.phase + 1) % PHASES.length;
    syncControls();
    render();
  });

  refs.playBtn.addEventListener('click', () => {
    if (state.playing) {
      stopPlay();
    } else {
      state.playing = true;
      refs.playBtn.textContent = 'Pause';
      state.timer = window.setInterval(() => {
        state.phase = (state.phase + 1) % PHASES.length;
        syncControls();
        render();
      }, 2200);
    }
  });

  refs.modeToggle.addEventListener('click', () => {
    state.mode = state.mode === 'training' ? 'inference' : 'training';
    refs.modeToggle.classList.toggle('is-on', state.mode === 'training');
    render();
  });

  refs.shapesToggle.addEventListener('click', () => {
    state.showShapes = !state.showShapes;
    refs.shapesToggle.classList.toggle('is-on', state.showShapes);
    render();
  });

  refs.maskToggle.addEventListener('click', () => {
    state.showMask = !state.showMask;
    refs.maskToggle.classList.toggle('is-on', state.showMask);
    render();
  });

  refs.decoderStepSelect.addEventListener('change', (e) => {
    state.decoderStep = Number(e.target.value);
    render();
  });
}

function setupPhaseTicks() {
  refs.phaseTicks.innerHTML = PHASES.map((phase, idx) => (
    `<div class="phase-tick ${idx === state.phase ? 'active' : ''}">${idx + 1}. ${phase.short}</div>`
  )).join('');
}

function setupDecoderSelector() {
  refs.decoderStepSelect.innerHTML = Array.from({ length: 7 }, (_, i) => (
    `<option value="${i + 1}">Forecast day ${i + 1}</option>`
  )).join('');
  refs.decoderStepSelect.value = String(state.decoderStep);
  refs.modeToggle.classList.add('is-on');
  refs.shapesToggle.classList.add('is-on');
  refs.maskToggle.classList.add('is-on');
}

function syncControls() {
  refs.phaseSlider.value = String(state.phase);
  refs.decoderStepSelect.value = String(state.decoderStep);
  refs.modeToggle.classList.toggle('is-on', state.mode === 'training');
  refs.shapesToggle.classList.toggle('is-on', state.showShapes);
  refs.maskToggle.classList.toggle('is-on', state.showMask);
  setupPhaseTicks();
}

function stopPlay() {
  state.playing = false;
  refs.playBtn.textContent = 'Play';
  if (state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
}

function render() {
  syncControls();
  renderMetrics();
  renderArchitecture();
  renderExplanation();
  renderActiveComputation();
  renderForecast();
}

function renderMetrics() {
  const phase = PHASES[state.phase];
  const step = state.decoderStep;
  const visibleContext = getVisibleContext();
  const leakage = getLeakageStatus();
  const cost = getAttentionCost();

  refs.metricPhase.textContent = phase.name;
  refs.metricStep.textContent = `${step} / 7`;
  refs.metricShape.textContent = phase.shape;
  refs.metricContext.textContent = visibleContext;
  refs.metricLeakage.textContent = leakage;
  refs.metricCost.textContent = cost;
  refs.teacherStep.textContent = String(step);
}

function getVisibleContext() {
  if (state.phase <= 3) return '30 source days';
  if (state.phase === 4) {
    return state.mode === 'training'
      ? '7 target positions shown in parallel'
      : `${state.decoderStep} generated target position(s)`;
  }
  if (state.phase === 5) {
    return state.mode === 'training'
      ? `${state.decoderStep} visible target steps for the selected row`
      : `${state.decoderStep} generated target steps are available`;
  }
  if (state.phase >= 6) {
    return state.mode === 'training'
      ? `30 source days + ${state.decoderStep} target steps`
      : `30 source days + ${state.decoderStep} generated step(s)`;
  }
  return '30 source days';
}

function getLeakageStatus() {
  if (state.phase <= 3) return 'Encoder only';
  if (state.phase === 4) return 'Prepared for causal decoding';
  return 'Yes — future blocked';
}

function getAttentionCost() {
  if (state.phase === 2) return 'O(30²) ≈ 900 scores';
  if (state.phase === 5) return 'O(7²) ≈ 49 masked scores';
  if (state.phase === 6) return '≈ 7 × 30 = 210 cross scores';
  if (state.phase === 7) return 'Reuse prior attention maps';
  return 'Attention not active';
}

function renderExplanation() {
  const phase = PHASES[state.phase];
  refs.activePanelTitle.textContent = phase.title;
  refs.equationBlock.textContent = phase.equation;
  refs.shapeBlock.textContent = phase.shape;
  refs.plainBlock.textContent = phase.plain + getModeSuffix('plain');
  refs.hydroBlock.textContent = phase.hydro + getModeSuffix('hydro');
  refs.whyBlock.textContent = phase.why;
}

function getModeSuffix(kind) {
  if (state.phase < 4) return '';
  if (kind === 'plain') {
    return state.mode === 'training'
      ? ' In training mode, all decoder positions are displayed together, but masking preserves autoregressive logic.'
      : ' In inference mode, only generated target history up to the selected forecast day is available.';
  }
  return state.mode === 'training'
    ? ' This mirrors teacher forcing: the decoder is exposed to earlier target positions during learning, but not to later ones.'
    : ' This mirrors deployment: the decoder must live with its own generated history as it marches across the forecast horizon.';
}

function renderArchitecture() {
  const active = state.phase;
  const svg = [];
  svg.push(`<svg viewBox="0 0 760 580" role="img" aria-label="Transformer encoder-decoder architecture">
    <defs>
      <marker id="arrowHead" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto">
        <path d="M0,0 L10,4 L0,8 z" fill="currentColor"></path>
      </marker>
    </defs>`);

  const boxes = [
    { key: 'raw', x: 48, y: 72, w: 148, h: 82, title: 'Raw input history', sub: '30 days × 4 variables', phase: 0 },
    { key: 'embed', x: 232, y: 58, w: 128, h: 70, title: 'Embedding', sub: 'linear projection', phase: 1 },
    { key: 'pe', x: 232, y: 140, w: 128, h: 66, title: 'Positional encoding', sub: 'time order signal', phase: 1 },
    { key: 'encsa', x: 396, y: 78, w: 146, h: 78, title: 'Encoder self-attn', sub: '30 ↔ 30', phase: 2 },
    { key: 'encffn', x: 574, y: 78, w: 138, h: 78, title: 'Encoder FFN', sub: 'position-wise', phase: 3 },
    { key: 'decshift', x: 68, y: 388, w: 148, h: 80, title: 'Decoder input shift', sub: '7 target positions', phase: 4 },
    { key: 'mask', x: 262, y: 388, w: 150, h: 80, title: 'Masked self-attn', sub: 'causal target flow', phase: 5, decoder: true },
    { key: 'cross', x: 456, y: 388, w: 150, h: 80, title: 'Cross-attention', sub: 'Q_dec to K/V_enc', phase: 6, decoder: true },
    { key: 'output', x: 632, y: 388, w: 96, h: 80, title: 'Output', sub: '7-day Q forecast', phase: 7, decoder: true }
  ];

  svg.push(`<rect x="24" y="28" width="712" height="522" fill="rgba(255,255,255,0.65)" rx="28"></rect>`);
  svg.push(`<text x="52" y="42" class="diagram-subtitle">Source sequence</text>`);
  svg.push(`<text x="52" y="372" class="diagram-subtitle">Target sequence</text>`);

  svg.push(renderSourceTokens(52, 190));
  svg.push(renderTargetTokens(68, 486));

  boxes.forEach((box) => {
    const activeClass = box.phase === active
      ? `active${box.decoder ? ' decoder-active' : ''}`
      : '';
    svg.push(`<g>
      <rect class="diagram-box ${activeClass}" x="${box.x}" y="${box.y}" width="${box.w}" height="${box.h}"></rect>
      <text x="${box.x + 16}" y="${box.y + 28}" class="diagram-title">${box.title}</text>
      <text x="${box.x + 16}" y="${box.y + 48}" class="diagram-subtitle">${box.sub}</text>
      ${state.showShapes ? renderShapeBadge(box) : ''}
    </g>`);
  });

  const arrows = [
    { x1: 196, y1: 113, x2: 232, y2: 93, phase: 1 },
    { x1: 196, y1: 113, x2: 232, y2: 173, phase: 1 },
    { x1: 360, y1: 93, x2: 396, y2: 117, phase: 2 },
    { x1: 360, y1: 173, x2: 396, y2: 117, phase: 2 },
    { x1: 542, y1: 117, x2: 574, y2: 117, phase: 3 },
    { x1: 216, y1: 428, x2: 262, y2: 428, phase: 5, decoder: true },
    { x1: 412, y1: 428, x2: 456, y2: 428, phase: 6, decoder: true },
    { x1: 606, y1: 428, x2: 632, y2: 428, phase: 7, decoder: true },
    { x1: 644, y1: 156, x2: 520, y2: 388, phase: 6, decoder: true }
  ];

  arrows.forEach((arrow) => {
    const activeClass = arrow.phase === active
      ? `active${arrow.decoder ? ' decoder-active' : ''}`
      : '';
    svg.push(`<path class="arrow-line ${activeClass}" d="M ${arrow.x1} ${arrow.y1} L ${arrow.x2} ${arrow.y2}"></path>`);
  });

  svg.push(`<text x="458" y="338" class="diagram-subtitle">Encoded source memory</text>`);
  svg.push(`<text x="619" y="506" class="diagram-subtitle">Forecast days 1…7</text>`);
  svg.push(`</svg>`);
  refs.architecturePanel.innerHTML = svg.join('');
}

function renderSourceTokens(x, y) {
  const parts = ['<g>'];
  for (let i = 0; i < 30; i++) {
    const cx = x + (i % 10) * 16;
    const cy = y + Math.floor(i / 10) * 18;
    const cls = i === 22 || i === 23 ? 'pulse' : (i >= 28 ? 'wet' : '');
    parts.push(`<rect class="token-cell ${cls}" x="${cx}" y="${cy}" width="12" height="12"></rect>`);
  }
  parts.push(`<text x="${x}" y="${y - 12}" class="source-label">30-day source tokens</text>`);
  parts.push(`<text x="${x}" y="${y + 66}" class="source-label">pulse at day 23 • wet state at day 30</text>`);
  parts.push('</g>');
  return parts.join('');
}

function renderTargetTokens(x, y) {
  const parts = ['<g>'];
  for (let i = 0; i < 7; i++) {
    const fill = i < state.decoderStep ? 'rgba(162,97,196,0.18)' : 'rgba(162,97,196,0.07)';
    parts.push(`<rect x="${x + i * 22}" y="${y}" width="16" height="16" rx="4" fill="${fill}" stroke="rgba(162,97,196,0.24)"></rect>`);
  }
  parts.push(`<text x="${x}" y="${y - 12}" class="target-label">7 target positions</text>`);
  parts.push('</g>');
  return parts.join('');
}

function renderShapeBadge(box) {
  const shapeText = {
    raw: '[30×4]',
    embed: '[30×6]',
    pe: '[30×6]',
    encsa: '[30×30]',
    encffn: '[30×6]',
    decshift: '[7×6]',
    mask: '[7×7]',
    cross: '[7×30]',
    output: '[7×1]'
  }[box.key];
  return `
    <rect class="shape-badge" x="${box.x + box.w - 68}" y="${box.y + box.h - 28}" width="56" height="18"></rect>
    <text class="badge-text" x="${box.x + box.w - 60}" y="${box.y + box.h - 15}">${shapeText}</text>
  `;
}

function renderActiveComputation() {
  switch (state.phase) {
    case 0:
      refs.activeComputationPanel.innerHTML = renderRawInputsPanel();
      break;
    case 1:
      refs.activeComputationPanel.innerHTML = renderEmbeddingPanel();
      break;
    case 2:
      refs.activeComputationPanel.innerHTML = renderEncoderAttentionPanel();
      break;
    case 3:
      refs.activeComputationPanel.innerHTML = renderFFNPanel();
      break;
    case 4:
      refs.activeComputationPanel.innerHTML = renderDecoderShiftPanel();
      break;
    case 5:
      refs.activeComputationPanel.innerHTML = renderMaskedAttentionPanel();
      break;
    case 6:
      refs.activeComputationPanel.innerHTML = renderCrossAttentionPanel();
      break;
    case 7:
      refs.activeComputationPanel.innerHTML = renderOutputProjectionPanel();
      break;
    default:
      refs.activeComputationPanel.innerHTML = '';
  }
}

function renderRawInputsPanel() {
  const width = 760;
  const height = 580;
  const left = 62;
  const right = 34;
  const top = 42;
  const chartH = 96;
  const gap = 28;
  const series = [
    { name: 'Rainfall (mm/day)', key: 'rain', color: '#3178c6', values: DATA.historyRain },
    { name: 'Temperature (°C)', key: 'temp', color: '#d67835', values: DATA.historyTemp },
    { name: 'Soil moisture (%)', key: 'soil', color: '#2d8c5e', values: DATA.historySoil },
    { name: 'Observed Q (m³/s)', key: 'flow', color: '#634fb6', values: DATA.historyFlow }
  ];

  const svg = [`<div class="legend-row">
    <span class="legend-chip" style="color:#3178c6"><span class="legend-dot"></span>Rainfall</span>
    <span class="legend-chip" style="color:#d67835"><span class="legend-dot"></span>Temperature</span>
    <span class="legend-chip" style="color:#2d8c5e"><span class="legend-dot"></span>Soil moisture</span>
    <span class="legend-chip" style="color:#634fb6"><span class="legend-dot"></span>Observed streamflow</span>
  </div>`];
  svg.push(`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Raw input chart">`);
  svg.push(`<rect x="0" y="0" width="${width}" height="${height}" fill="transparent"></rect>`);

  series.forEach((s, idx) => {
    const y0 = top + idx * (chartH + gap);
    const vals = s.values;
    const xScale = (i) => left + (i / 29) * (width - left - right);
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const yScale = (v) => y0 + chartH - ((v - minV) / (maxV - minV || 1)) * (chartH - 16);
    svg.push(`<text x="${left}" y="${y0 - 10}" class="diagram-title" style="font-size:13px">${s.name}</text>`);
    svg.push(`<line class="axis-line" x1="${left}" y1="${y0 + chartH}" x2="${width - right}" y2="${y0 + chartH}"></line>`);
    for (let g = 0; g <= 4; g++) {
      const gy = y0 + (g / 4) * chartH;
      svg.push(`<line class="grid-line" x1="${left}" y1="${gy}" x2="${width - right}" y2="${gy}"></line>`);
    }

    if (s.key === 'rain') {
      vals.forEach((v, i) => {
        const x = xScale(i) - 5;
        const barH = y0 + chartH - yScale(v);
        svg.push(`<rect x="${x}" y="${yScale(v)}" width="10" height="${barH}" fill="${s.color}" opacity="${i === 22 || i === 23 ? 0.95 : 0.55}" rx="4"></rect>`);
      });
    } else {
      const d = vals.map((v, i) => `${i === 0 ? 'M' : 'L'} ${xScale(i)} ${yScale(v)}`).join(' ');
      svg.push(`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="3" stroke-linecap="round"></path>`);
      vals.forEach((v, i) => {
        if (i % 3 === 0 || i >= 27 || i === 22) {
          svg.push(`<circle cx="${xScale(i)}" cy="${yScale(v)}" r="${i === 22 || i >= 28 ? 4.5 : 3.2}" fill="${s.color}" opacity="${i === 22 || i >= 28 ? 1 : 0.8}"></circle>`);
        }
      });
    }
  });

  const pulseX = left + (22 / 29) * (width - left - right);
  const wetX = left + (29 / 29) * (width - left - right);

  svg.push(`<line x1="${pulseX}" y1="36" x2="${pulseX}" y2="520" stroke="rgba(49,120,198,0.28)" stroke-width="2" stroke-dasharray="5 6"></line>`);
  svg.push(`<line x1="${wetX}" y1="36" x2="${wetX}" y2="520" stroke="rgba(45,140,94,0.32)" stroke-width="2" stroke-dasharray="5 6"></line>`);

  svg.push(`<rect class="annotation-card" x="470" y="40" width="246" height="92"></rect>`);
  svg.push(`<text x="490" y="66" class="annotation-title">Storyline anchors</text>`);
  svg.push(`<text x="490" y="90" class="annotation-copy">• Day 23 rainfall pulse is the older event.</text>`);
  svg.push(`<text x="490" y="108" class="annotation-copy">• Day 30 soil moisture is the recent wetness state.</text>`);
  svg.push(`<text x="490" y="126" class="annotation-copy">• The forecast should depend on both.</text>`);

  for (let i = 0; i < 30; i += 5) {
    const x = left + (i / 29) * (width - left - right);
    svg.push(`<text x="${x - 6}" y="548" class="chart-label">D${i + 1}</text>`);
  }
  svg.push(`</svg>`);
  return svg.join('');
}

function renderEmbeddingPanel() {
  const shownRows = 10;
  const dims = 6;
  const cell = 28;
  const matrixX = 188;
  const matrixY = 106;
  const peX = 434;
  const sumX = 596;
  const svg = [`<svg viewBox="0 0 760 580" role="img" aria-label="Embedding and positional encoding panel">`];

  svg.push(`<text x="44" y="48" class="diagram-title">From physical variables to model tokens</text>`);
  svg.push(`<text x="44" y="68" class="diagram-subtitle">Showing the first 10 source days only so the embedding story stays readable.</text>`);

  svg.push(`<rect class="annotation-card" x="32" y="90" width="128" height="328"></rect>`);
  svg.push(`<text x="48" y="118" class="annotation-title">Raw token slice</text>`);
  ['Day', 'rain', 'temp', 'soil', 'Q'].forEach((label, idx) => {
    svg.push(`<text x="${48 + idx * 18}" y="144" class="diagram-subtitle">${label}</text>`);
  });

  for (let r = 0; r < shownRows; r++) {
    const y = 164 + r * 24;
    const values = [
      r + 1,
      DATA.historyRain[r].toFixed(1),
      DATA.historyTemp[r].toFixed(1),
      DATA.historySoil[r].toFixed(0),
      DATA.historyFlow[r].toFixed(0)
    ];
    values.forEach((v, idx) => {
      svg.push(`<text x="${48 + idx * 18}" y="${y}" class="chart-label">${v}</text>`);
    });
  }

  svg.push(`<text x="${matrixX}" y="96" class="annotation-title">Embedded tokens</text>`);
  svg.push(renderMatrix(DATA.embeddings.slice(0, shownRows), matrixX, matrixY, cell, valueToBlue, 'e'));

  svg.push(`<text x="${peX}" y="96" class="annotation-title">Positional encoding</text>`);
  svg.push(renderMatrix(DATA.positional.slice(0, shownRows), peX, matrixY, cell, valueToTeal, 'p'));

  svg.push(`<text x="${sumX}" y="96" class="annotation-title">z = embed + PE</text>`);
  svg.push(renderMatrix(DATA.encoderInput.slice(0, shownRows), sumX, matrixY, cell, valueToPurple, 'z'));

  svg.push(`<path class="arrow-line active" d="M 160 258 L ${matrixX - 12} 258"></path>`);
  svg.push(`<path class="arrow-line active" d="M ${matrixX + cell * dims} 258 L ${peX - 14} 258"></path>`);
  svg.push(`<path class="arrow-line active" d="M ${peX + cell * dims} 258 L ${sumX - 14} 258"></path>`);

  if (state.showShapes) {
    svg.push(`<rect class="shape-badge" x="${matrixX}" y="434" width="84" height="22"></rect>`);
    svg.push(`<text class="badge-text" x="${matrixX + 10}" y="449">[10×6]</text>`);
    svg.push(`<rect class="shape-badge" x="${sumX}" y="434" width="84" height="22"></rect>`);
    svg.push(`<text class="badge-text" x="${sumX + 10}" y="449">[30×6]</text>`);
  }

  svg.push(`<rect class="annotation-card" x="36" y="450" width="688" height="92"></rect>`);
  svg.push(`<text x="54" y="478" class="annotation-title">Interpretation</text>`);
  svg.push(`<text x="54" y="500" class="annotation-copy">A day with high rain, moderate temperature, and rising wetness is now a 6-D token. Positional encoding adds a stripe-like temporal fingerprint.</text>`);
  svg.push(`<text x="54" y="520" class="annotation-copy">Now the model can compare “what happened” and “when it happened” inside the same vector space.</text>`);

  svg.push(`</svg>`);
  return svg.join('');
}

function renderEncoderAttentionPanel() {
  const start = 18;
  const end = 30;
  const subset = DATA.encoderAttention.slice(start, end).map((row) => row.slice(start, end));
  const focus = Math.max(start, 30 - state.decoderStep - 1);
  const focusLocal = focus - start;
  const cell = 28;
  const x0 = 116;
  const y0 = 92;
  const svg = [`<svg viewBox="0 0 760 580" role="img" aria-label="Encoder self-attention panel">`];

  svg.push(`<text x="42" y="48" class="diagram-title">Encoder self-attention on the last 12 source days</text>`);
  svg.push(`<text x="42" y="68" class="diagram-subtitle">A focused source day can retrieve information from any earlier source day.</text>`);
  svg.push(renderHeatmap(subset, x0, y0, cell, focusLocal, focusLocal, valueToBlue));

  for (let i = 0; i < 12; i++) {
    const label = `D${start + i + 1}`;
    svg.push(`<text x="${x0 + i * cell + 6}" y="${y0 - 10}" class="chart-label">${label}</text>`);
    svg.push(`<text x="${x0 - 34}" y="${y0 + i * cell + 18}" class="chart-label">${label}</text>`);
  }

  const focusWeights = subset[focusLocal];
  const barX = 486;
  const barY = 112;
  const barW = 218;
  const barH = 236;
  svg.push(`<rect class="annotation-card" x="${barX - 18}" y="${barY - 30}" width="252" height="298"></rect>`);
  svg.push(`<text x="${barX}" y="${barY - 6}" class="annotation-title">Focused token: D${focus + 1}</text>`);
  svg.push(`<text x="${barX}" y="${barY + 12}" class="annotation-copy">Weights from one source query over the final 12 source keys</text>`);

  focusWeights.forEach((w, i) => {
    const y = barY + 28 + i * 18;
    const width = w * barW;
    const strong = start + i === 22 || start + i >= 28;
    svg.push(`<text x="${barX}" y="${y + 10}" class="chart-label">D${start + i + 1}</text>`);
    svg.push(`<rect x="${barX + 48}" y="${y}" width="${width}" height="12" rx="6" fill="${strong ? '#3178c6' : 'rgba(54,93,157,0.34)'}"></rect>`);
  });

  svg.push(`<rect class="annotation-card" x="40" y="448" width="672" height="90"></rect>`);
  svg.push(`<text x="56" y="476" class="annotation-title">Hydrology reading</text>`);
  svg.push(`<text x="56" y="500" class="annotation-copy">The selected late-history token can still assign weight to the older storm pulse near D23–D24 while also attending to the freshest wetness state at D29–D30.</text>`);
  svg.push(`<text x="56" y="520" class="annotation-copy">That is the key pedagogical win of attention: direct access to multiple relevant memories without serial compression.</text>`);

  if (state.showShapes) {
    svg.push(`<rect class="shape-badge" x="580" y="386" width="118" height="22"></rect>`);
    svg.push(`<text class="badge-text" x="590" y="401">[30×30] scores</text>`);
  }

  svg.push(`</svg>`);
  return svg.join('');
}

function renderFFNPanel() {
  const indices = [22, 26, 29];
  const labels = ['Older storm pulse (D23)', 'Recession transition (D27)', 'Recent wet state (D30)'];
  const svg = [`<svg viewBox="0 0 760 580" role="img" aria-label="Encoder FFN panel">`];
  svg.push(`<text x="42" y="48" class="diagram-title">Position-wise feed-forward network</text>`);
  svg.push(`<text x="42" y="68" class="diagram-subtitle">Each position is refined independently after contextual mixing.</text>`);

  indices.forEach((idx, col) => {
    const x = 56 + col * 226;
    const y = 118;
    const before = DATA.encoderContext[idx];
    const after = DATA.encoderOutput[idx];
    svg.push(`<rect class="annotation-card" x="${x}" y="${y}" width="196" height="324"></rect>`);
    svg.push(`<text x="${x + 16}" y="${y + 28}" class="annotation-title">${labels[col]}</text>`);
    svg.push(`<text x="${x + 16}" y="${y + 48}" class="annotation-copy">before FFN</text>`);
    before.forEach((v, i) => {
      const yy = y + 70 + i * 20;
      svg.push(`<text x="${x + 16}" y="${yy + 10}" class="chart-label">h${i + 1}</text>`);
      svg.push(`<rect x="${x + 48}" y="${yy}" width="${Math.abs(v) * 58}" height="12" rx="6" fill="rgba(54,93,157,0.42)"></rect>`);
    });
    svg.push(`<path class="arrow-line active" d="M ${x + 78} ${y + 214} L ${x + 124} ${y + 214}"></path>`);
    svg.push(`<text x="${x + 16}" y="${y + 236}" class="annotation-copy">after FFN</text>`);
    after.forEach((v, i) => {
      const yy = y + 256 + i * 10;
      svg.push(`<rect x="${x + 48}" y="${yy}" width="${Math.abs(v) * 56}" height="8" rx="4" fill="rgba(39,143,138,0.56)"></rect>`);
    });
  });

  svg.push(`<rect class="annotation-card" x="42" y="466" width="676" height="76"></rect>`);
  svg.push(`<text x="58" y="492" class="annotation-title">What changed?</text>`);
  svg.push(`<text x="58" y="516" class="annotation-copy">Attention mixes information across time. The FFN then re-expresses each day’s latent state in a more useful basis for later decoder access.</text>`);

  if (state.showShapes) {
    svg.push(`<rect class="shape-badge" x="608" y="82" width="108" height="22"></rect>`);
    svg.push(`<text class="badge-text" x="618" y="97">[30×6] → [30×6]</text>`);
  }

  svg.push(`</svg>`);
  return svg.join('');
}

function renderDecoderShiftPanel() {
  const tokens = state.mode === 'training'
    ? [null, ...DATA.forecast.slice(0, 6)]
    : [null, ...DATA.forecast.slice(0, state.decoderStep - 1), ...Array.from({ length: 6 - (state.decoderStep - 1) }, () => null)];
  const svg = [`<svg viewBox="0 0 760 580" role="img" aria-label="Decoder shifted input panel">`];
  svg.push(`<text x="42" y="48" class="diagram-title">Shifted target sequence</text>`);
  svg.push(`<text x="42" y="68" class="diagram-subtitle">${state.mode === 'training'
    ? 'Training mode shows all seven positions in one parallel batch.'
    : 'Inference mode exposes only generated history up to the selected step.'}</text>`);

  svg.push(`<rect class="annotation-card" x="46" y="104" width="668" height="148"></rect>`);
  svg.push(`<text x="64" y="132" class="annotation-title">Decoder input tokens</text>`);
  tokens.forEach((value, i) => {
    const x = 74 + i * 92;
    const active = i <= state.decoderStep - 1 || state.mode === 'training';
    svg.push(`<rect x="${x}" y="156" width="72" height="54" rx="14" fill="${active ? 'rgba(162,97,196,0.18)' : 'rgba(162,97,196,0.06)'}" stroke="rgba(162,97,196,0.24)"></rect>`);
    svg.push(`<text x="${x + 18}" y="178" class="annotation-title">${i === 0 ? '<BOS>' : `y${i}`}</text>`);
    svg.push(`<text x="${x + 14}" y="198" class="chart-label">${value == null ? 'known later' : `${value.toFixed(1)} m³/s`}</text>`);
    if (i > 0) {
      svg.push(`<path class="arrow-line active decoder-active" d="M ${x - 18} ${183} L ${x} ${183}"></path>`);
    }
  });

  svg.push(`<rect class="annotation-card" x="46" y="286" width="668" height="210"></rect>`);
  svg.push(`<text x="64" y="314" class="annotation-title">What the shift does</text>`);
  svg.push(`<text x="64" y="340" class="annotation-copy">The model predicts forecast day k from positions up to k-1 in the decoder input.</text>`);
  svg.push(`<text x="64" y="362" class="annotation-copy">So the target sequence is offset by one step: start token first, then earlier target values.</text>`);
  svg.push(`<text x="64" y="394" class="annotation-title">${state.mode === 'training' ? 'Training mode intuition' : 'Inference mode intuition'}</text>`);
  const copy = state.mode === 'training'
    ? 'All decoder positions are processed at once, but each row is still mask-limited to its legal history.'
    : `Only forecast days 1…${state.decoderStep} exist so far. Later positions remain unavailable because the model has not generated them yet.`;
  svg.push(`<text x="64" y="418" class="annotation-copy">${copy}</text>`);
  svg.push(`<text x="64" y="456" class="annotation-copy">This is how the original Transformer supports parallel training without sacrificing autoregressive deployment.</text>`);

  if (state.showShapes) {
    svg.push(`<rect class="shape-badge" x="564" y="516" width="144" height="22"></rect>`);
    svg.push(`<text class="badge-text" x="574" y="531">[7×1] → embed → [7×6]</text>`);
  }

  svg.push(`</svg>`);
  return svg.join('');
}

function renderMaskedAttentionPanel() {
  const size = 7;
  const cell = 48;
  const x0 = 126;
  const y0 = 96;
  const weights = DATA.decoderAttention;
  const stepRow = state.decoderStep - 1;
  const svg = [`<svg viewBox="0 0 760 580" role="img" aria-label="Masked self-attention panel">`];
  svg.push(`<text x="42" y="48" class="diagram-title">Masked decoder self-attention</text>`);
  svg.push(`<text x="42" y="68" class="diagram-subtitle">Selected row: forecast day ${state.decoderStep}. Cells above the diagonal are illegal future lookups.</text>`);

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const x = x0 + c * cell;
      const y = y0 + r * cell;
      const allowed = c <= r;
      const visible = state.mode === 'training' ? true : (r < state.decoderStep && c < state.decoderStep);
      let fill = allowed ? rgbaFromHex('#278f8a', 0.10 + weights[r][c] * 0.88) : 'rgba(203,79,82,0.08)';
      if (!visible) fill = 'rgba(230,236,244,0.85)';
      svg.push(`<rect x="${x}" y="${y}" width="${cell - 4}" height="${cell - 4}" rx="10" class="${allowed ? 'mask-cell-open' : 'mask-cell-blocked'}" fill="${fill}" stroke="rgba(58,92,138,0.12)"></rect>`);
      if (state.showMask && !allowed) {
        svg.push(`<line x1="${x + 10}" y1="${y + 10}" x2="${x + cell - 14}" y2="${y + cell - 14}" stroke="rgba(203,79,82,0.38)" stroke-width="2"></line>`);
      }
    }
  }

  for (let i = 0; i < size; i++) {
    svg.push(`<text x="${x0 + i * cell + 14}" y="${y0 - 12}" class="chart-label">y${i + 1}</text>`);
    svg.push(`<text x="${x0 - 34}" y="${y0 + i * cell + 28}" class="chart-label">y${i + 1}</text>`);
  }
  svg.push(`<rect class="heat-outline" x="${x0 + stepRow * cell}" y="${y0 + stepRow * cell}" width="${cell - 4}" height="${cell - 4}" rx="10"></rect>`);

  svg.push(`<rect class="annotation-card" x="506" y="102" width="206" height="292"></rect>`);
  svg.push(`<text x="526" y="130" class="annotation-title">Selected row weights</text>`);
  weights[stepRow].forEach((w, i) => {
    const y = 158 + i * 28;
    const legal = i <= stepRow;
    const visible = state.mode === 'training' ? true : (i < state.decoderStep);
    const width = legal && visible ? w * 128 : 24;
    svg.push(`<text x="526" y="${y + 10}" class="chart-label">to y${i + 1}</text>`);
    svg.push(`<rect x="582" y="${y}" width="${width}" height="14" rx="7" fill="${legal && visible ? '#a261c4' : 'rgba(203,79,82,0.22)'}"></rect>`);
  });

  const modeCopy = state.mode === 'training'
    ? `All seven rows exist now, but row ${state.decoderStep} only sees y1…y${state.decoderStep}.`
    : `Only rows up to y${state.decoderStep} exist so far. Later rows are unavailable during generation.`;
  svg.push(`<rect class="annotation-card" x="42" y="464" width="670" height="76"></rect>`);
  svg.push(`<text x="58" y="490" class="annotation-title">Mask logic</text>`);
  svg.push(`<text x="58" y="516" class="annotation-copy">${modeCopy}</text>`);

  if (state.showShapes) {
    svg.push(`<rect class="shape-badge" x="596" y="416" width="108" height="22"></rect>`);
    svg.push(`<text class="badge-text" x="606" y="431">[7×7] mask</text>`);
  }

  svg.push(`</svg>`);
  return svg.join('');
}

function renderCrossAttentionPanel() {
  const stepIdx = state.decoderStep - 1;
  const weights = DATA.crossAttention[stepIdx];
  const x0 = 70;
  const y0 = 150;
  const barW = 18;
  const chartH = 250;
  const svg = [`<svg viewBox="0 0 760 580" role="img" aria-label="Cross-attention panel">`];
  svg.push(`<text x="42" y="48" class="diagram-title">Cross-attention from decoder day ${state.decoderStep}</text>`);
  svg.push(`<text x="42" y="68" class="diagram-subtitle">The decoder query reads the encoded 30-day source memory to pick the most relevant past information.</text>`);

  svg.push(`<rect class="annotation-card" x="42" y="102" width="142" height="318"></rect>`);
  svg.push(`<text x="62" y="130" class="annotation-title">Decoder query</text>`);
  svg.push(`<text x="62" y="154" class="annotation-copy">Forecast day ${state.decoderStep}</text>`);
  DATA.decoderStates[stepIdx].forEach((v, i) => {
    const yy = 182 + i * 26;
    svg.push(`<text x="62" y="${yy + 11}" class="chart-label">q${i + 1}</text>`);
    svg.push(`<rect x="92" y="${yy}" width="${Math.abs(v) * 64}" height="14" rx="7" fill="rgba(162,97,196,0.6)"></rect>`);
  });

  svg.push(`<path class="arrow-line active decoder-active" d="M 188 258 L 230 258"></path>`);
  svg.push(`<text x="238" y="126" class="annotation-title">Attention over encoder keys (30 days)</text>`);

  for (let i = 0; i < 30; i++) {
    const x = x0 + 170 + i * (barW + 2);
    const h = weights[i] * chartH;
    const y = y0 + chartH - h;
    const emph = i === 22 || i === 23 || i >= 28;
    svg.push(`<rect x="${x}" y="${y}" width="${barW}" height="${h}" rx="6" fill="${emph ? '#3178c6' : 'rgba(54,93,157,0.30)'}"></rect>`);
    if (i % 5 === 0 || i === 22 || i >= 28) {
      svg.push(`<text x="${x - 2}" y="${y0 + chartH + 18}" class="chart-label">D${i + 1}</text>`);
    }
  }
  svg.push(`<line class="axis-line" x1="${x0 + 160}" y1="${y0 + chartH}" x2="${x0 + 760 - 240}" y2="${y0 + chartH}"></line>`);

  svg.push(`<line x1="${x0 + 170 + 22 * (barW + 2) + barW / 2}" y1="${y0 - 8}" x2="${x0 + 170 + 22 * (barW + 2) + barW / 2}" y2="${y0 + chartH}" stroke="rgba(49,120,198,0.32)" stroke-dasharray="5 6"></line>`);
  svg.push(`<line x1="${x0 + 170 + 29 * (barW + 2) + barW / 2}" y1="${y0 - 8}" x2="${x0 + 170 + 29 * (barW + 2) + barW / 2}" y2="${y0 + chartH}" stroke="rgba(45,140,94,0.38)" stroke-dasharray="5 6"></line>`);

  const pulseWeight = weights[22] + weights[23];
  const wetWeight = weights[28] + weights[29];
  svg.push(`<rect class="annotation-card" x="238" y="436" width="474" height="104"></rect>`);
  svg.push(`<text x="258" y="464" class="annotation-title">Hydrology interpretation</text>`);
  svg.push(`<text x="258" y="488" class="annotation-copy">Day ${state.decoderStep} assigns ${(pulseWeight * 100).toFixed(1)}% of its cross-attention to the older storm pulse (D23–D24)</text>`);
  svg.push(`<text x="258" y="508" class="annotation-copy">and ${(wetWeight * 100).toFixed(1)}% to the recent wetness memory (D29–D30).</text>`);
  svg.push(`<text x="258" y="528" class="annotation-copy">That balance changes smoothly across the forecast horizon.</text>`);

  if (state.showShapes) {
    svg.push(`<rect class="shape-badge" x="608" y="102" width="100" height="22"></rect>`);
    svg.push(`<text class="badge-text" x="618" y="117">[7×30]</text>`);
  }

  svg.push(`</svg>`);
  return svg.join('');
}

function renderOutputProjectionPanel() {
  const visible = state.mode === 'training' ? 7 : state.decoderStep;
  const svg = [`<svg viewBox="0 0 760 580" role="img" aria-label="Output projection panel">`];
  svg.push(`<text x="42" y="48" class="diagram-title">Output projection to streamflow</text>`);
  svg.push(`<text x="42" y="68" class="diagram-subtitle">${state.mode === 'training'
    ? 'All seven forecast values are available from the batched decoder pass.'
    : `Only forecast days 1…${state.decoderStep} are currently generated.`}</text>`);

  svg.push(`<rect class="annotation-card" x="46" y="100" width="246" height="316"></rect>`);
  svg.push(`<text x="64" y="128" class="annotation-title">Selected hidden state h_${state.decoderStep}</text>`);
  DATA.decoderStates[state.decoderStep - 1].forEach((v, i) => {
    const yy = 156 + i * 28;
    svg.push(`<text x="64" y="${yy + 11}" class="chart-label">d${i + 1}</text>`);
    svg.push(`<rect x="96" y="${yy}" width="${Math.abs(v) * 86}" height="14" rx="7" fill="rgba(162,97,196,0.58)"></rect>`);
  });
  svg.push(`<path class="arrow-line active decoder-active" d="M 184 334 L 244 334"></path>`);
  svg.push(`<text x="68" y="366" class="annotation-copy">ŷ_${state.decoderStep} = h_${state.decoderStep} W_out + b_out</text>`);
  svg.push(`<text x="68" y="392" class="annotation-copy">Current scalar forecast: ${DATA.forecast[state.decoderStep - 1].toFixed(1)} m³/s</text>`);

  const chartX = 336;
  const chartY = 114;
  const chartW = 370;
  const chartH = 286;
  const minV = Math.min(...DATA.forecast) - 3;
  const maxV = Math.max(...DATA.forecast) + 3;
  svg.push(`<rect class="annotation-card" x="${chartX - 18}" y="${chartY - 18}" width="406" height="336"></rect>`);
  for (let g = 0; g <= 4; g++) {
    const y = chartY + (g / 4) * chartH;
    svg.push(`<line class="grid-line" x1="${chartX}" y1="${y}" x2="${chartX + chartW}" y2="${y}"></line>`);
  }
  svg.push(`<line class="axis-line" x1="${chartX}" y1="${chartY + chartH}" x2="${chartX + chartW}" y2="${chartY + chartH}"></line>`);

  const points = DATA.forecast.map((v, i) => {
    const x = chartX + (i / 6) * chartW;
    const y = chartY + chartH - ((v - minV) / (maxV - minV)) * chartH;
    return { x, y, v };
  });
  const pathVisible = points.slice(0, visible).map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
  svg.push(`<path d="${pathVisible}" fill="none" stroke="#c83f7f" stroke-width="4" stroke-linecap="round"></path>`);
  if (visible < 7) {
    const pathFuture = points.slice(visible - 1).map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
    svg.push(`<path d="${pathFuture}" fill="none" stroke="rgba(200,63,127,0.30)" stroke-width="4" stroke-dasharray="7 8"></path>`);
  }
  points.forEach((p, i) => {
    const active = i < visible;
    svg.push(`<circle cx="${p.x}" cy="${p.y}" r="${i === state.decoderStep - 1 ? 6 : 4.2}" fill="${active ? '#c83f7f' : 'rgba(200,63,127,0.32)'}"></circle>`);
    svg.push(`<text x="${p.x - 12}" y="${chartY + chartH + 24}" class="chart-label">F${i + 1}</text>`);
  });

  svg.push(`<text x="${chartX}" y="${chartY + chartH + 52}" class="annotation-copy">Projected discharge trajectory</text>`);
  svg.push(`<rect class="annotation-card" x="46" y="454" width="660" height="86"></rect>`);
  svg.push(`<text x="64" y="482" class="annotation-title">Reading the output</text>`);
  svg.push(`<text x="64" y="506" class="annotation-copy">The hydrograph stays elevated because recent wetness remains high, while the older pulse still contributes a decaying memory signature across the horizon.</text>`);

  if (state.showShapes) {
    svg.push(`<rect class="shape-badge" x="588" y="70" width="112" height="22"></rect>`);
    svg.push(`<text class="badge-text" x="598" y="85">[7×6] → [7×1]</text>`);
  }

  svg.push(`</svg>`);
  return svg.join('');
}

function renderForecast() {
  const width = 760;
  const height = 300;
  const x0 = 72;
  const y0 = 34;
  const chartW = 640;
  const chartH = 210;

  const tailFlow = DATA.historyFlow.slice(-10);
  const combined = [...tailFlow, ...DATA.forecast];
  const minV = Math.min(...combined) - 4;
  const maxV = Math.max(...combined) + 4;
  const scaleY = (v) => y0 + chartH - ((v - minV) / (maxV - minV)) * chartH;
  const scaleXHistory = (i) => x0 + (i / 16) * chartW;
  const scaleXForecast = (i) => x0 + ((10 + i) / 16) * chartW;

  const svg = [`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Forecast chart">`];
  for (let g = 0; g <= 4; g++) {
    const y = y0 + (g / 4) * chartH;
    svg.push(`<line class="grid-line" x1="${x0}" y1="${y}" x2="${x0 + chartW}" y2="${y}"></line>`);
  }
  svg.push(`<line class="axis-line" x1="${x0}" y1="${y0 + chartH}" x2="${x0 + chartW}" y2="${y0 + chartH}"></line>`);

  const histPath = tailFlow.map((v, i) => `${i === 0 ? 'M' : 'L'} ${scaleXHistory(i)} ${scaleY(v)}`).join(' ');
  svg.push(`<path d="${histPath}" fill="none" stroke="#634fb6" stroke-width="3.5" stroke-linecap="round"></path>`);
  tailFlow.forEach((v, i) => {
    svg.push(`<circle cx="${scaleXHistory(i)}" cy="${scaleY(v)}" r="3.2" fill="#634fb6"></circle>`);
  });

  const visible = state.mode === 'training' ? 7 : state.decoderStep;
  const forecastVisiblePath = DATA.forecast.slice(0, visible).map((v, i) => (
    `${i === 0 ? 'M' : 'L'} ${scaleXForecast(i)} ${scaleY(v)}`
  )).join(' ');
  svg.push(`<path d="${forecastVisiblePath}" fill="none" stroke="#c83f7f" stroke-width="4" stroke-linecap="round"></path>`);
  if (visible < 7) {
    const futurePath = DATA.forecast.slice(visible - 1).map((v, i) => (
      `${i === 0 ? 'M' : 'L'} ${scaleXForecast(i + visible - 1)} ${scaleY(v)}`
    )).join(' ');
    svg.push(`<path d="${futurePath}" fill="none" stroke="rgba(200,63,127,0.32)" stroke-width="4" stroke-dasharray="8 8"></path>`);
  }

  DATA.forecast.forEach((v, i) => {
    const active = i < visible;
    svg.push(`<circle cx="${scaleXForecast(i)}" cy="${scaleY(v)}" r="${i === state.decoderStep - 1 ? 5.8 : 4.2}" fill="${active ? '#c83f7f' : 'rgba(200,63,127,0.28)'}"></circle>`);
    svg.push(`<text x="${scaleXForecast(i) - 8}" y="${y0 + chartH + 22}" class="chart-label">F${i + 1}</text>`);
  });

  svg.push(`<line x1="${scaleXForecast(0) - 20}" y1="${y0}" x2="${scaleXForecast(0) - 20}" y2="${y0 + chartH}" stroke="rgba(58,92,138,0.24)" stroke-dasharray="5 6"></line>`);
  svg.push(`<text x="${x0}" y="${height - 18}" class="chart-label">Last 10 observed days</text>`);
  svg.push(`<text x="${scaleXForecast(0)}" y="${height - 18}" class="chart-label">7 forecast days</text>`);
  svg.push(`</svg>`);

  refs.forecastPanel.innerHTML = svg.join('');
  refs.forecastCaption.textContent = state.mode === 'training'
    ? `Training mode: all seven target positions are predicted in one masked decoder pass, but the selected day ${state.decoderStep} is highlighted.`
    : `Inference mode: the decoder has generated only days 1–${state.decoderStep}. Later values are shown as the eventual trajectory but remain unavailable to the model at this step.`;
}

function buildDataset() {
  const historyRain = [];
  const historyTemp = [];
  const historySoil = [];
  const historyFlow = [];
  let soil = 47;

  for (let t = 0; t < 30; t++) {
    const day = t + 1;
    let rain = Math.max(0, 0.35 + 0.42 * Math.sin(day / 3.1) + 0.28 * Math.cos((day + 2) / 5.4));
    if (t === 7) rain += 4.2;
    if (t === 8) rain += 1.7;
    if (t === 22) rain += 5.4;
    if (t === 23) rain += 2.2;
    if (t === 27) rain += 1.3;
    rain = round(rain, 2);

    const temp = round(27.5 + 3.3 * Math.sin(day / 5.7) - 1.2 * Math.cos(day / 8.5), 2);
    soil = clamp(soil * 0.86 + rain * 2.2 - 0.42 * (temp - 26) + 3.1, 24, 92);
    soil = round(soil, 1);

    const prevRain = t > 0 ? historyRain[t - 1] : 0;
    const olderPulse = t >= 22 ? historyRain[22] * 0.24 : 0;
    const flow = round(12 + 0.48 * soil + 1.35 * rain + 0.55 * prevRain + olderPulse, 1);

    historyRain.push(rain);
    historyTemp.push(temp);
    historySoil.push(soil);
    historyFlow.push(flow);
  }

  const futureRain = [0.8, 0.4, 0.1, 1.2, 0.6, 0.2, 0.0];
  const futureTemp = [30.5, 31.0, 31.2, 30.7, 30.1, 29.8, 29.5];
  const forecast = [];
  const truthShift = [];
  let futureSoil = historySoil[29];
  const olderPulseMemory = historyRain[22] * 1.4 + historyRain[23] * 0.8;

  for (let k = 0; k < 7; k++) {
    futureSoil = clamp(futureSoil * 0.91 + futureRain[k] * 2.0 - 0.35 * (futureTemp[k] - 25) + 1.1, 20, 100);
    const q = round(14 + 0.42 * futureSoil + (0.36 - 0.03 * k) * olderPulseMemory + 1.1 * futureRain[k], 1);
    forecast.push(q);
    truthShift.push(k === 0 ? null : forecast[k - 1]);
  }

  const sourceMatrix = [];
  for (let i = 0; i < 30; i++) {
    sourceMatrix.push([
      normalize(historyRain[i], historyRain),
      normalize(historyTemp[i], historyTemp),
      normalize(historySoil[i], historySoil),
      normalize(historyFlow[i], historyFlow)
    ]);
  }

  const Wemb = [
    [0.62, -0.18, 0.24, 0.48, 0.16, -0.12],
    [0.08, 0.55, -0.12, 0.18, -0.20, 0.42],
    [0.18, 0.14, 0.61, 0.12, 0.34, -0.08],
    [0.30, 0.10, 0.22, 0.64, 0.18, 0.12]
  ];
  const bEmb = [0.06, -0.02, 0.04, 0.08, 0.03, -0.05];
  const embeddings = sourceMatrix.map((row) => addVec(matVec(row, Wemb), bEmb));
  const positional = buildPositionalEncoding(30, 6);
  const encoderInput = embeddings.map((row, i) => addVec(row, positional[i]));

  const Wq = [
    [0.60, 0.12, -0.08, 0.18, 0.16, 0.04],
    [-0.06, 0.56, 0.08, 0.10, -0.12, 0.24],
    [0.12, 0.06, 0.62, 0.16, 0.20, -0.04],
    [0.22, 0.18, 0.12, 0.54, 0.10, 0.08],
    [0.10, -0.08, 0.24, 0.16, 0.52, 0.12],
    [0.04, 0.22, -0.06, 0.12, 0.18, 0.50]
  ];
  const Wk = [
    [0.58, 0.08, -0.04, 0.12, 0.16, 0.06],
    [0.02, 0.52, 0.14, 0.10, -0.10, 0.20],
    [0.10, 0.04, 0.60, 0.12, 0.18, 0.02],
    [0.20, 0.14, 0.10, 0.56, 0.10, 0.04],
    [0.12, -0.06, 0.18, 0.14, 0.54, 0.10],
    [0.06, 0.18, -0.02, 0.08, 0.14, 0.48]
  ];
  const Wv = [
    [0.54, 0.10, 0.08, 0.12, 0.10, 0.06],
    [0.08, 0.50, 0.06, 0.10, 0.14, 0.18],
    [0.10, 0.06, 0.56, 0.14, 0.16, 0.06],
    [0.16, 0.12, 0.08, 0.50, 0.12, 0.10],
    [0.12, 0.08, 0.16, 0.14, 0.48, 0.12],
    [0.10, 0.16, 0.06, 0.10, 0.12, 0.46]
  ];

  const encoderQ = encoderInput.map((row) => matVec(row, Wq));
  const encoderK = encoderInput.map((row) => matVec(row, Wk));
  const encoderV = encoderInput.map((row) => matVec(row, Wv));
  const encoderAttention = attentionMatrix(encoderQ, encoderK);
  const encoderContext = applyAttention(encoderAttention, encoderV);

  const W1 = [
    [0.44, 0.10, 0.08, 0.18, 0.20, 0.06, 0.10, 0.14],
    [0.08, 0.40, 0.10, 0.06, 0.12, 0.18, 0.20, 0.08],
    [0.10, 0.12, 0.42, 0.20, 0.10, 0.08, 0.14, 0.18],
    [0.18, 0.08, 0.12, 0.40, 0.14, 0.10, 0.16, 0.12],
    [0.12, 0.16, 0.20, 0.14, 0.38, 0.10, 0.08, 0.18],
    [0.10, 0.18, 0.06, 0.12, 0.14, 0.42, 0.16, 0.08]
  ];
  const b1 = [0.04, 0.02, 0.03, 0.01, 0.05, 0.02, 0.03, 0.04];
  const W2 = [
    [0.28, 0.08, 0.12, 0.16, 0.14, 0.10],
    [0.12, 0.26, 0.10, 0.08, 0.12, 0.14],
    [0.14, 0.12, 0.28, 0.10, 0.08, 0.12],
    [0.16, 0.10, 0.12, 0.26, 0.14, 0.08],
    [0.12, 0.14, 0.08, 0.12, 0.24, 0.10],
    [0.10, 0.16, 0.12, 0.08, 0.12, 0.24],
    [0.08, 0.12, 0.14, 0.10, 0.16, 0.22],
    [0.12, 0.08, 0.10, 0.14, 0.12, 0.20]
  ];
  const b2 = [0.06, 0.03, 0.05, 0.04, 0.03, 0.02];
  const encoderOutput = encoderContext.map((row) => addVec(matVec(reluVec(addVec(matVec(row, W1), b1)), W2), b2));

  const decoderInputValues = [0, ...forecast.slice(0, 6)].map((v) => (v === 0 ? 0.0 : normalize(v, forecast)));
  const decoderEmb = decoderInputValues.map((v, i) => addVec(
    [0.42 * v + 0.08, 0.16 * v + 0.04, 0.24 * v + 0.02, 0.36 * v + 0.06, 0.14 * v + 0.03, 0.20 * v + 0.04],
    buildPositionalEncoding(7, 6)[i]
  ));
  const decoderAttention = buildMaskedDecoderAttention(7);

  const decoderStates = Array.from({ length: 7 }, (_, step) => {
    const crossRow = buildCrossRow(step, historyRain, historySoil, historyFlow);
    const weighted = weightedSum(crossRow, encoderOutput);
    const dec = addVec(weighted.map((v, i) => 0.62 * v + 0.38 * decoderEmb[step][i]), [0.03, 0.02, 0.01, 0.04, 0.02, 0.03]);
    return dec.map((v) => round(v, 3));
  });
  const crossAttention = Array.from({ length: 7 }, (_, step) => buildCrossRow(step, historyRain, historySoil, historyFlow));

  return {
    historyRain, historyTemp, historySoil, historyFlow,
    futureRain, futureTemp, forecast, truthShift,
    embeddings, positional, encoderInput,
    encoderAttention, encoderContext, encoderOutput,
    decoderAttention, crossAttention, decoderStates
  };
}

function buildCrossRow(step, rain, soil, flow) {
  const row = [];
  const flowNorm = flow.map((v) => normalize(v, flow));
  const soilNorm = soil.map((v) => normalize(v, soil));
  for (let j = 0; j < 30; j++) {
    const older = gaussian(j, 22.5, 3.2);
    const recent = gaussian(j, 28.9, 1.8);
    const local = gaussian(j, Math.max(18, 28 - step), 2.6) * 0.18;
    const recWeight = Math.max(0.68 - 0.07 * step, 0.28);
    const pulseWeight = Math.min(0.32 + 0.07 * step, 0.72);
    const score = recWeight * recent + pulseWeight * older + local + 0.02 * (flowNorm[j] + soilNorm[j]);
    row.push(score);
  }
  return softmax(row);
}

function buildMaskedDecoderAttention(size) {
  const matrix = [];
  for (let r = 0; r < size; r++) {
    const raw = [];
    for (let c = 0; c < size; c++) {
      if (c > r) raw.push(-999);
      else raw.push((r + 1 - c) * 0.6 + (c + 1) * 0.12);
    }
    const clipped = raw.map((v) => v < -100 ? -1e9 : v);
    const attn = maskedSoftmax(clipped);
    matrix.push(attn.map((v, idx) => idx > r ? 0 : v));
  }
  return matrix;
}

function attentionMatrix(Q, K) {
  const dk = Math.sqrt(Q[0].length);
  return Q.map((q) => {
    const scores = K.map((k) => dot(q, k) / dk);
    return softmax(scores);
  });
}

function applyAttention(attn, V) {
  return attn.map((row) => weightedSum(row, V));
}

function buildPositionalEncoding(length, dModel) {
  const pe = [];
  for (let pos = 0; pos < length; pos++) {
    const row = [];
    for (let i = 0; i < dModel; i++) {
      const divTerm = Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
      const angle = pos / divTerm;
      row.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
    }
    pe.push(row.map((v) => round(v, 3)));
  }
  return pe;
}

function renderMatrix(matrix, x0, y0, cell, colorFn, prefix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const flat = matrix.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const parts = [`<g>`];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const value = matrix[r][c];
      const norm = (value - min) / (max - min || 1);
      parts.push(`<rect x="${x0 + c * cell}" y="${y0 + r * cell}" width="${cell - 3}" height="${cell - 3}" rx="8" fill="${colorFn(norm)}" stroke="rgba(255,255,255,0.64)"></rect>`);
      parts.push(`<text x="${x0 + c * cell + 5}" y="${y0 + r * cell + 18}" class="chart-label">${prefix}${c + 1}</text>`);
    }
  }
  for (let c = 0; c < cols; c++) {
    parts.push(`<text x="${x0 + c * cell + 8}" y="${y0 - 10}" class="chart-label">d${c + 1}</text>`);
  }
  parts.push(`</g>`);
  return parts.join('');
}

function renderHeatmap(matrix, x0, y0, cell, highlightRow, highlightCol, colorFn) {
  const flat = matrix.flat();
  const min = Math.min(...flat);
  const max = Math.max(...flat);
  const parts = ['<g>'];
  matrix.forEach((row, r) => {
    row.forEach((value, c) => {
      const norm = (value - min) / (max - min || 1);
      parts.push(`<rect class="heat-cell" x="${x0 + c * cell}" y="${y0 + r * cell}" width="${cell - 3}" height="${cell - 3}" rx="8" fill="${colorFn(norm)}"></rect>`);
    });
  });
  parts.push(`<rect class="heat-outline" x="${x0}" y="${y0 + highlightRow * cell}" width="${matrix[0].length * cell - 3}" height="${cell - 3}" rx="10"></rect>`);
  parts.push(`<rect class="heat-outline" x="${x0 + highlightCol * cell}" y="${y0}" width="${cell - 3}" height="${matrix.length * cell - 3}" rx="10"></rect>`);
  parts.push('</g>');
  return parts.join('');
}

function valueToBlue(v) {
  const alpha = 0.18 + v * 0.82;
  return `rgba(49,120,198,${alpha.toFixed(3)})`;
}

function valueToTeal(v) {
  const alpha = 0.14 + v * 0.78;
  return `rgba(39,143,138,${alpha.toFixed(3)})`;
}

function valueToPurple(v) {
  const alpha = 0.14 + v * 0.78;
  return `rgba(162,97,196,${alpha.toFixed(3)})`;
}

function rgbaFromHex(hex, alpha) {
  const clean = hex.replace('#', '');
  const bigint = parseInt(clean, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function normalize(value, arr) {
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  return (value - min) / (max - min || 1);
}

function gaussian(x, mean, sigma) {
  return Math.exp(-Math.pow(x - mean, 2) / (2 * sigma * sigma));
}

function round(value, digits = 2) {
  const factor = Math.pow(10, digits);
  return Math.round(value * factor) / factor;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function addVec(a, b) {
  return a.map((v, i) => round(v + b[i], 3));
}

function reluVec(v) {
  return v.map((x) => Math.max(0, x));
}

function matVec(row, matrix) {
  const out = Array(matrix[0].length).fill(0);
  for (let j = 0; j < matrix[0].length; j++) {
    for (let i = 0; i < row.length; i++) {
      out[j] += row[i] * matrix[i][j];
    }
  }
  return out.map((v) => round(v, 3));
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / (sum || 1));
}

function maskedSoftmax(arr) {
  const finite = arr.filter((v) => Number.isFinite(v) && v > -1e8);
  const max = Math.max(...finite);
  const exps = arr.map((v) => (v < -1e8 ? 0 : Math.exp(v - max)));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / (sum || 1));
}

function weightedSum(weights, matrix) {
  const out = Array(matrix[0].length).fill(0);
  for (let i = 0; i < weights.length; i++) {
    for (let j = 0; j < matrix[0].length; j++) {
      out[j] += weights[i] * matrix[i][j];
    }
  }
  return out.map((v) => round(v, 3));
}
