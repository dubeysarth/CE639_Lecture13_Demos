const state = {
  history: 512,
  horizon: 48,
  dataSize: 'medium',
  latency: 'balanced',
  interpretability: 60,
  regime: 20,
};

const MODEL_ORDER = ['lstm', 'transformer', 'patchtst', 'informer', 'autoformer'];

const MODEL_META = {
  lstm: {
    label: 'LSTM + attention',
    summary: 'Compact recurrent memory plus a selective readout over recent history.',
    complexity: 'O(T) recurrent pass',
    use: 'Best for short or event-heavy sequences and smaller datasets.',
    note: 'Deployment-friendly when long dense attention is unnecessary.',
  },
  transformer: {
    label: 'Original Transformer',
    summary: 'Full pairwise timestep interactions with a clean baseline interpretation.',
    complexity: 'O(T²)',
    use: 'Best as a moderate-length baseline when cost is still acceptable.',
    note: 'Useful pedagogically, but can become expensive quickly as T grows.',
  },
  patchtst: {
    label: 'PatchTST',
    summary: 'Patch tokens compress local windows before self-attention is applied.',
    complexity: 'O((T / p)²)',
    use: 'Best for long medium-data forecasting with repeated local motifs.',
    note: 'A strong default for long streamflow-style sequences without extreme context lengths.',
  },
  informer: {
    label: 'Informer',
    summary: 'ProbSparse attention retains only the most informative query interactions.',
    complexity: 'O(T log T)',
    use: 'Best for very long contexts when latency matters.',
    note: 'Designed to keep sequence growth manageable in long-horizon settings.',
  },
  autoformer: {
    label: 'Autoformer',
    summary: 'Series decomposition plus autocorrelation for long-range periodic structure.',
    complexity: 'O(T log T)',
    use: 'Best for very long, periodic or multi-season forecasting problems.',
    note: 'Especially attractive when seasonality and repeated lag structure dominate.',
  },
};

const SCENARIOS = [
  {
    key: 'urban_flash',
    title: 'Urban flash-flood nowcast',
    description: 'Short memory, event-dominant rainfall pulses, limited local training data, and strict deployment latency.',
    recommendation: 'LSTM + attention',
    settings: {
      history: 96,
      horizon: 12,
      dataSize: 'small',
      latency: 'strict',
      interpretability: 74,
      regime: -75,
    },
  },
  {
    key: 'basin_streamflow',
    title: 'Daily basin streamflow forecasting',
    description: 'Months of history, medium training scale, and a mix of seasonality and hydrometeorological event memory.',
    recommendation: 'PatchTST',
    settings: {
      history: 720,
      horizon: 60,
      dataSize: 'medium',
      latency: 'balanced',
      interpretability: 58,
      regime: 28,
    },
  },
  {
    key: 'reservoir_inflow',
    title: 'Reservoir inflow or snowmelt operations',
    description: 'Very long context, long horizon, periodic dominance, and interest in trend plus repeated seasonal structure.',
    recommendation: 'Autoformer',
    settings: {
      history: 2048,
      horizon: 168,
      dataSize: 'medium',
      latency: 'relaxed',
      interpretability: 66,
      regime: 84,
    },
  },
  {
    key: 'national_warning',
    title: 'Regional flood early-warning telemetry',
    description: 'Very long context across many sensors, large training scale, and strong pressure for efficient inference.',
    recommendation: 'Informer',
    settings: {
      history: 4096,
      horizon: 96,
      dataSize: 'large',
      latency: 'strict',
      interpretability: 44,
      regime: -10,
    },
  },
];

const el = {
  historyRange: document.getElementById('historyRange'),
  horizonRange: document.getElementById('horizonRange'),
  dataSizeSelect: document.getElementById('dataSizeSelect'),
  latencySelect: document.getElementById('latencySelect'),
  interpretRange: document.getElementById('interpretRange'),
  regimeRange: document.getElementById('regimeRange'),
  historyValue: document.getElementById('historyValue'),
  horizonValue: document.getElementById('horizonValue'),
  interpretValue: document.getElementById('interpretValue'),
  regimeValue: document.getElementById('regimeValue'),
  matrixValue: document.getElementById('matrixValue'),
  matrixSub: document.getElementById('matrixSub'),
  complexityValue: document.getElementById('complexityValue'),
  complexitySub: document.getElementById('complexitySub'),
  recommendValue: document.getElementById('recommendValue'),
  recommendSub: document.getElementById('recommendSub'),
  confidenceValue: document.getElementById('confidenceValue'),
  confidenceSub: document.getElementById('confidenceSub'),
  architectureStrip: document.getElementById('architectureStrip'),
  costBars: document.getElementById('costBars'),
  scalingCallout: document.getElementById('scalingCallout'),
  whyList: document.getElementById('whyList'),
  hydroText: document.getElementById('hydroText'),
  patchVisual: document.getElementById('patchVisual'),
  informerVisual: document.getElementById('informerVisual'),
  autoformerVisual: document.getElementById('autoformerVisual'),
  patchCaption: document.getElementById('patchCaption'),
  informerCaption: document.getElementById('informerCaption'),
  autoformerCaption: document.getElementById('autoformerCaption'),
  scenarioGrid: document.getElementById('scenarioGrid'),
};

function clamp(x, min, max) {
  return Math.min(max, Math.max(min, x));
}

function round(value, digits = 0) {
  const p = 10 ** digits;
  return Math.round(value * p) / p;
}

function formatInteger(num) {
  return new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(num);
}

function formatMaybeK(num) {
  if (num >= 1_000_000) return `${round(num / 1_000_000, 2)}M`;
  if (num >= 1_000) return `${round(num / 1_000, 1)}k`;
  return `${round(num, 0)}`;
}

function formatMB(mb) {
  if (mb >= 1024) return `${round(mb / 1024, 2)} GB`;
  return `${round(mb, 2)} MB`;
}

function regimeLabel(value) {
  if (value <= -60) return 'strongly event-dominant';
  if (value <= -20) return 'leans event-dominant';
  if (value < 20) return 'balanced or mixed';
  if (value < 60) return 'leans periodic';
  return 'strongly periodic';
}

function patchSize(T) {
  if (T >= 3072) return 64;
  if (T >= 1536) return 48;
  if (T >= 768) return 32;
  if (T >= 256) return 16;
  return 8;
}

function computeSignals() {
  const T = state.history;
  const H = state.horizon;
  const shortSignal = clamp((256 - T) / 208, 0, 1);
  const mediumSignal = clamp(1 - Math.abs(Math.log2(T) - 9) / 2.8, 0, 1);
  const longSignal = clamp((T - 256) / 1400, 0, 1);
  const veryLongSignal = clamp((T - 900) / 2400, 0, 1);
  const horizonShort = clamp((72 - H) / 66, 0, 1);
  const horizonLong = clamp((H - 48) / 240, 0, 1);
  const periodic = clamp((state.regime + 100) / 200, 0, 1);
  const eventish = 1 - periodic;
  const interpret = state.interpretability / 100;
  const sizeValue = { small: 0.2, medium: 0.6, large: 1 }[state.dataSize];
  const smallData = state.dataSize === 'small' ? 1 : 0;
  const mediumData = state.dataSize === 'medium' ? 1 : 0;
  const largeData = state.dataSize === 'large' ? 1 : 0;
  const strict = state.latency === 'strict' ? 1 : 0;
  const balanced = state.latency === 'balanced' ? 1 : 0;
  const relaxed = state.latency === 'relaxed' ? 1 : 0;

  return {
    T,
    H,
    shortSignal,
    mediumSignal,
    longSignal,
    veryLongSignal,
    horizonShort,
    horizonLong,
    periodic,
    eventish,
    interpret,
    sizeValue,
    smallData,
    mediumData,
    largeData,
    strict,
    balanced,
    relaxed,
  };
}

function scoreModels(sig) {
  const scores = {};
  scores.lstm =
    32 +
    40 * sig.shortSignal +
    18 * sig.eventish +
    14 * sig.strict +
    10 * sig.smallData +
    8 * sig.interpret +
    10 * sig.horizonShort -
    28 * sig.longSignal -
    26 * sig.veryLongSignal -
    10 * sig.periodic;

  scores.transformer =
    28 +
    18 * sig.mediumSignal +
    12 * sig.largeData +
    10 * sig.interpret +
    6 * sig.mediumData +
    8 * (1 - Math.abs(sig.periodic - 0.5) * 2) -
    20 * sig.strict * Math.max(sig.longSignal, 0.4) -
    30 * sig.veryLongSignal -
    8 * sig.smallData;

  scores.patchtst =
    36 +
    24 * sig.longSignal +
    18 * sig.mediumSignal +
    16 * (1 - sig.smallData) +
    12 * sig.horizonLong +
    10 * sig.periodic +
    6 * sig.balanced +
    4 * sig.strict -
    12 * sig.eventish * sig.veryLongSignal -
    8 * sig.shortSignal;

  scores.informer =
    32 +
    32 * sig.veryLongSignal +
    14 * sig.longSignal +
    14 * sig.horizonLong +
    16 * sig.strict +
    8 * sig.eventish +
    10 * sig.largeData +
    5 * sig.balanced -
    12 * sig.shortSignal -
    8 * sig.interpret;

  scores.autoformer =
    32 +
    30 * sig.veryLongSignal +
    14 * sig.longSignal +
    16 * sig.horizonLong +
    22 * sig.periodic +
    8 * sig.interpret +
    8 * (1 - sig.smallData) +
    5 * sig.relaxed -
    18 * sig.eventish -
    10 * sig.strict -
    12 * sig.shortSignal;

  MODEL_ORDER.forEach((key) => {
    scores[key] = round(scores[key], 1);
  });
  return scores;
}

function sortedEntries(obj) {
  return Object.entries(obj).sort((a, b) => b[1] - a[1]);
}

function describeConfidence(margin, bestKey, sig) {
  let value = 58 + margin * 1.8;
  const crispRules =
    (bestKey === 'lstm' && sig.shortSignal > 0.65) ||
    (bestKey === 'patchtst' && sig.longSignal > 0.25 && sig.veryLongSignal < 0.45) ||
    (bestKey === 'informer' && sig.veryLongSignal > 0.55 && sig.strict > 0) ||
    (bestKey === 'autoformer' && sig.veryLongSignal > 0.45 && sig.periodic > 0.65);
  if (crispRules) value += 10;
  value = clamp(value, 52, 96);
  let label = 'Moderate';
  let message = 'some neighboring architectures are still plausible';
  if (value >= 84) {
    label = 'High';
    message = 'the regime clearly favors one family';
  } else if (value < 68) {
    label = 'Low';
    message = 'this is a boundary case, so compare baselines';
  }
  return { value: Math.round(value), label, message };
}

function computeStats() {
  const sig = computeSignals();
  const scores = scoreModels(sig);
  const ranking = sortedEntries(scores);
  const [bestKey, bestScore] = ranking[0];
  const [, secondScore] = ranking[1];
  const margin = bestScore - secondScore;
  const confidence = describeConfidence(margin, bestKey, sig);
  const p = patchSize(sig.T);
  const patchTokens = Math.ceil(sig.T / p);
  const fullCells = sig.T * sig.T;
  const fullMB = (fullCells * 4) / (1024 * 1024);
  const tLogT = sig.T * Math.log2(sig.T);
  const patchCells = patchTokens * patchTokens;
  const lstmCost = 14 * sig.T + 8 * sig.H;
  const autoCost = 1.18 * tLogT;
  const costMap = {
    lstm: lstmCost,
    transformer: fullCells,
    patchtst: patchCells,
    informer: tLogT,
    autoformer: autoCost,
  };
  const maxCost = Math.max(...Object.values(costMap));
  const ratio = fullCells / tLogT;

  return {
    sig,
    scores,
    ranking,
    bestKey,
    confidence,
    patchSize: p,
    patchTokens,
    fullCells,
    fullMB,
    tLogT,
    patchCells,
    costMap,
    maxCost,
    ratio,
  };
}

function whyReasons(bestKey, stats) {
  const { sig, patchSize: p, patchTokens } = stats;
  const reasons = [];
  const densePairwise = `${formatInteger(stats.fullCells)} pairwise scores`;

  if (bestKey === 'lstm') {
    reasons.push(`T = ${sig.T} is short enough that a compact recurrent model can preserve temporal order without paying for ${densePairwise} from dense self-attention.`);
    reasons.push(`The regime is ${regimeLabel(state.regime)}, so recent storm pulses matter more than long periodic structure.`);
    reasons.push(`Your latency requirement is ${state.latency}, which favors a lighter deployment footprint than a long-sequence transformer.`);
    if (state.dataSize === 'small') {
      reasons.push('The dataset is small, so a simpler sequence model is less likely to waste capacity or overcomplicate training.');
    }
  }

  if (bestKey === 'transformer') {
    reasons.push(`The sequence is in the moderate range, where full attention remains teachable and still manageable compared with the very-long-context cases.`);
    reasons.push(`This regime does not strongly force sparse or decomposed long-sequence variants, so the original Transformer remains a reasonable reference architecture.`);
    reasons.push(`Interpretability importance is ${state.interpretability}/100, so inspecting direct timestep-to-timestep interactions can still be useful in the classroom.`);
    reasons.push(`This choice is best treated as a baseline: once T grows further, O(T²) becomes hard to justify operationally.`);
  }

  if (bestKey === 'patchtst') {
    reasons.push(`T = ${sig.T} is long enough that dense attention would require ${densePairwise}, but patching with size p = ${p} reduces the token count to ${patchTokens}.`);
    reasons.push(`The data size is ${state.dataSize}, which is a good fit for learning patch-level representations without the full O(T²) burden.`);
    reasons.push(`The regime ${state.regime >= 0 ? 'leans periodic' : 'remains mixed'}, so local hydro-meteorological motifs can be grouped before global interaction.`);
    reasons.push('This is the lecture default for long medium-data time series when you want a transformer family model without jumping all the way to very-long sparse variants.');
  }

  if (bestKey === 'informer') {
    reasons.push(`T = ${sig.T} is in the very-long range, where dense self-attention would create ${densePairwise}; sparse attention reduces that to an O(T log T) style scaling.`);
    reasons.push(`Latency is ${state.latency}, so reducing attention computations matters for deployment and repeated inference.`);
    reasons.push(`The regime is ${regimeLabel(state.regime)}, which still benefits from long context, but does not require the strong decomposition bias that Autoformer uses.`);
    reasons.push('This is the lecture choice when the sequence must stay long, the horizon is substantial, and computation cannot grow quadratically.');
  }

  if (bestKey === 'autoformer') {
    reasons.push(`The regime is ${regimeLabel(state.regime)}, so separating trend and seasonal structure is more attractive than treating the series as only irregular events.`);
    reasons.push(`T = ${sig.T} and horizon = ${sig.H} place the problem in the long-range forecasting zone, where autocorrelation-based selection is more efficient than dense attention.`);
    reasons.push(`Compared with the original Transformer, the long-sequence cost moves from O(T²) toward O(T log T), which is critical once T becomes large.`);
    reasons.push('This is the lecture choice when repeated lag structure, multi-season memory, or smooth long-horizon evolution is central to the forecasting task.');
  }

  return reasons;
}

function hydrologyExplanation(bestKey, stats) {
  const { sig } = stats;
  const lead = state.regime >= 0
    ? 'seasonal persistence, reservoir operation cycles, snowmelt memory, or repeated hydro-climatic patterns'
    : 'storm pulses, abrupt runoff response, flashy hydrographs, or transient event signatures';

  const maps = {
    lstm: {
      main: `Treat this as a short-memory civil-engineering problem where ${lead} sit mostly in the recent history. A recurrent encoder with attention can focus the forecast on a few influential antecedent timesteps without building a large pairwise attention matrix.`,
      highlight: 'Example interpretation: urban drainage response, culvert inflow, or event-driven basin runoff with limited local data and strict latency.',
    },
    transformer: {
      main: `Treat this as a moderate-length benchmarking case. Full attention gives a clean picture of pairwise temporal interactions, so it is pedagogically useful when you want to demonstrate what dense attention buys before sequence length becomes too large.`,
      highlight: 'Example interpretation: a reference baseline for daily streamflow or infrastructure-demand forecasting when T is not yet operationally extreme.',
    },
    patchtst: {
      main: `Treat this as a long streamflow-style sequence where local blocks of rainfall, PET, temperature, and antecedent discharge can be compressed into patch tokens. The model preserves transformer-style interaction, but it reasons over grouped windows instead of every single timestep.`,
      highlight: 'Example interpretation: daily or hourly basin forecasting with months of history and medium data volume, where motifs repeat but latency still matters.',
    },
    informer: {
      main: `Treat this as a very-long telemetry problem. The forecasting system still needs long context, but not every timestep pair deserves equal computational attention. Informer is preferable when the deployment setting penalizes dense attention at inference time.`,
      highlight: 'Example interpretation: regional flood early warning, large telemetry networks, or long sensor histories where edge or near-real-time inference matters.',
    },
    autoformer: {
      main: `Treat this as a periodic-dominant long-horizon forecasting problem. Trend and seasonal structure should be separated before long-range relationships are matched, which is exactly the bias Autoformer brings through decomposition and autocorrelation.`,
      highlight: 'Example interpretation: reservoir inflow, snowmelt-influenced discharge, tidal or seasonal hydrology, or long-range operation planning with repeated lag structure.',
    },
  };

  const item = maps[bestKey];
  return `
    <div>${item.main}</div>
    <div class="hydro-highlight">${item.highlight}</div>
    <div class="hydro-note">Current regime summary: T = ${sig.T}, horizon = ${sig.H}, data size = ${state.dataSize}, latency = ${state.latency}, temporal character = ${regimeLabel(state.regime)}.</div>
  `;
}

function costLabelValue(key, stats) {
  const raw = stats.costMap[key];
  if (key === 'transformer') return `${formatMaybeK(raw)} pair scores`;
  if (key === 'patchtst') return `${formatMaybeK(raw)} patch pairs`;
  if (key === 'informer' || key === 'autoformer') return `${formatMaybeK(raw)} sparse-style ops`;
  return `${formatMaybeK(raw)} recurrent units`;
}

function renderArchitectureStrip(stats) {
  el.architectureStrip.innerHTML = MODEL_ORDER.map((key) => {
    const meta = MODEL_META[key];
    const active = key === stats.bestKey ? 'active' : '';
    const score = stats.scores[key];
    const width = clamp(score, 0, 100);
    return `
      <div class="arch-card ${active}">
        <div class="arch-top">
          <div class="arch-title">${meta.label}</div>
          <span class="score-badge">${Math.round(score)}</span>
        </div>
        <div class="arch-summary">${meta.summary}</div>
        <div class="score-line">
          <div class="cost-label"><span>fit to current regime</span><span>${Math.round(score)} / 100</span></div>
          <div class="score-track"><div class="score-fill" style="width:${width}%"></div></div>
        </div>
        <div class="model-complexity">${meta.complexity}</div>
        <div class="model-use">${meta.use}</div>
        <div class="model-note">${meta.note}</div>
      </div>
    `;
  }).join('');
}

function renderCostBars(stats) {
  el.costBars.innerHTML = MODEL_ORDER.map((key) => {
    const width = (stats.costMap[key] / stats.maxCost) * 100;
    return `
      <div class="cost-row">
        <div class="cost-label"><span>${MODEL_META[key].label}</span><span>${costLabelValue(key, stats)}</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
      </div>
    `;
  }).join('');

  el.scalingCallout.innerHTML = `
    At <strong>T = ${state.history}</strong>, dense self-attention implies
    <strong>${formatInteger(stats.fullCells)}</strong> pairwise interactions, while an
    <strong>O(T log₂T)</strong> style long-sequence mechanism is about
    <strong>${formatInteger(Math.round(stats.tLogT))}</strong> operations.
    That is a rough <strong>${round(stats.ratio, 1)}×</strong> reduction before even accounting for constant factors.
  `;
}

function patchSVG(stats) {
  const seqCells = 20;
  const patchTokens = clamp(Math.ceil(stats.patchTokens / 4), 4, 16);
  const patchEvery = Math.ceil(seqCells / patchTokens);
  let bottom = '';
  for (let i = 0; i < seqCells; i += 1) {
    const x = 18 + i * 14;
    const patchIndex = Math.floor(i / patchEvery);
    const fills = ['#dcecff', '#bfd9f6', '#98c6f4', '#75b4ec', '#dcecff'];
    const fill = fills[patchIndex % fills.length];
    bottom += `<rect x="${x}" y="100" width="11" height="26" rx="4" fill="${fill}" stroke="#6f95b8" stroke-width="0.7" />`;
  }
  let top = '';
  for (let i = 0; i < patchTokens; i += 1) {
    const x = 30 + i * ((250 - 30) / Math.max(1, patchTokens - 1));
    top += `<rect x="${x - 8}" y="34" width="16" height="18" rx="5" fill="#1864ab" opacity="0.9" />`;
  }
  let arrows = '';
  for (let i = 0; i < Math.min(patchTokens, 10); i += 1) {
    const x = 30 + i * ((250 - 30) / Math.max(1, patchTokens - 1));
    const y2 = 100 + ((i * patchEvery + Math.floor(patchEvery / 2)) * 14);
    arrows += `<line x1="${x}" y1="54" x2="${18 + Math.min(seqCells - 1, i * patchEvery + Math.floor(patchEvery / 2)) * 14 + 5}" y2="98" stroke="#7aa6d1" stroke-width="1.2" />`;
  }
  return `
    <svg viewBox="0 0 310 170" xmlns="http://www.w3.org/2000/svg" aria-label="PatchTST patching schematic">
      <rect x="0" y="0" width="310" height="170" fill="#ffffff" />
      <text x="18" y="20" font-size="12" fill="#4d6b83" font-weight="700">Patch tokens</text>
      ${top}
      ${arrows}
      <text x="18" y="93" font-size="12" fill="#4d6b83" font-weight="700">Original time steps</text>
      ${bottom}
      <text x="18" y="151" font-size="11" fill="#6c859a">Sequence is grouped into local windows before attention is computed.</text>
    </svg>
  `;
}

function informerSVG(stats) {
  const rows = 8;
  const cols = 8;
  let cells = '';
  let activeRows = '';
  const activeRowIds = [1, 3, 6];
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const x = 26 + c * 28;
      const y = 24 + r * 15;
      const active = activeRowIds.includes(r);
      const fill = active && (c > 1 || r === 3) ? '#76b3ed' : '#e8eff7';
      const opacity = active ? 1 : 0.9;
      cells += `<rect x="${x}" y="${y}" width="22" height="10" rx="3" fill="${fill}" opacity="${opacity}" stroke="#c0d0df" stroke-width="0.6" />`;
    }
  }
  activeRowIds.forEach((r) => {
    const y = 24 + r * 15 + 5;
    activeRows += `<circle cx="16" cy="${y}" r="4" fill="#1864ab" />`;
  });
  return `
    <svg viewBox="0 0 310 170" xmlns="http://www.w3.org/2000/svg" aria-label="Informer ProbSparse schematic">
      <rect x="0" y="0" width="310" height="170" fill="#ffffff" />
      <text x="16" y="18" font-size="12" fill="#4d6b83" font-weight="700">Attention grid</text>
      ${cells}
      ${activeRows}
      <text x="16" y="118" font-size="11" fill="#4d6b83">Dark markers indicate the few queries kept for detailed attention.</text>
      <path d="M18 136 C70 122, 106 148, 150 138 S228 126, 286 142" fill="none" stroke="#1864ab" stroke-width="2.2" />
      <circle cx="84" cy="130" r="5" fill="#f59f00" />
      <circle cx="196" cy="134" r="5" fill="#f59f00" />
      <text x="16" y="156" font-size="11" fill="#6c859a">Focus attention budget on the most informative interactions instead of all T² pairs.</text>
    </svg>
  `;
}

function autoformerSVG(stats) {
  const periodicBias = clamp((state.regime + 100) / 200, 0, 1);
  const amplitude = 12 + periodicBias * 16;
  const trendSlope = 0.08 + stats.sig.horizonLong * 0.08;
  const seasonal = [];
  const trend = [];
  for (let x = 0; x <= 250; x += 10) {
    const sx = 24 + x;
    const wave = 58 + Math.sin(x / 24) * amplitude;
    const trendY = 60 + (x - 125) * trendSlope;
    seasonal.push(`${sx},${round(wave, 1)}`);
    trend.push(`${sx},${round(trendY, 1)}`);
  }
  const lags = [28, 60, 92, 124, 156, 188, 220, 252];
  const bars = lags.map((x, i) => {
    const peak = i % 3 === 0 || i === 5;
    const height = peak ? 26 + periodicBias * 14 : 10 + ((i % 2) * 6);
    return `<rect x="${x}" y="${148 - height}" width="14" height="${height}" rx="4" fill="${peak ? '#1864ab' : '#cad9e7'}" />`;
  }).join('');

  return `
    <svg viewBox="0 0 310 170" xmlns="http://www.w3.org/2000/svg" aria-label="Autoformer decomposition and autocorrelation schematic">
      <rect x="0" y="0" width="310" height="170" fill="#ffffff" />
      <text x="16" y="18" font-size="12" fill="#4d6b83" font-weight="700">Trend + seasonal decomposition</text>
      <polyline points="${seasonal.join(' ')}" fill="none" stroke="#75b4ec" stroke-width="2.1" />
      <polyline points="${trend.join(' ')}" fill="none" stroke="#1864ab" stroke-width="2.6" />
      <text x="16" y="92" font-size="11" fill="#4d6b83">Autocorrelation lag peaks</text>
      ${bars}
      <text x="16" y="160" font-size="11" fill="#6c859a">Blue bars mark repeated lag structure selected more strongly as periodicity rises.</text>
    </svg>
  `;
}

function renderVisuals(stats) {
  el.patchVisual.innerHTML = patchSVG(stats);
  el.informerVisual.innerHTML = informerSVG(stats);
  el.autoformerVisual.innerHTML = autoformerSVG(stats);

  el.patchCaption.textContent = `For T = ${state.history}, using patch size p = ${stats.patchSize} reduces ${state.history} raw timesteps to ${stats.patchTokens} tokens before attention.`;
  el.informerCaption.textContent = `The key teaching idea is selective focus: keep only the most informative query interactions so growth is closer to O(T log T) than O(T²).`;
  el.autoformerCaption.textContent = `As the regime becomes more periodic, decomposition and repeated lag selection become more attractive for long-horizon forecasting.`;
}

function renderScenarios() {
  el.scenarioGrid.innerHTML = SCENARIOS.map((scenario) => `
    <div class="scenario-card">
      <div class="scenario-title-row">
        <div class="scenario-title">${scenario.title}</div>
        <span class="score-badge">${scenario.recommendation}</span>
      </div>
      <p>${scenario.description}</p>
      <div class="scenario-note">Preset: T = ${scenario.settings.history}, horizon = ${scenario.settings.horizon}, data = ${scenario.settings.dataSize}, latency = ${scenario.settings.latency}, regime = ${regimeLabel(scenario.settings.regime)}.</div>
      <button class="preset-button" data-scenario="${scenario.key}">Apply preset</button>
    </div>
  `).join('');

  el.scenarioGrid.querySelectorAll('.preset-button').forEach((button) => {
    button.addEventListener('click', () => {
      const scenario = SCENARIOS.find((item) => item.key === button.dataset.scenario);
      if (!scenario) return;
      Object.assign(state, scenario.settings);
      syncControlsFromState();
      render();
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
  });
}

function syncControlsFromState() {
  el.historyRange.value = state.history;
  el.horizonRange.value = state.horizon;
  el.dataSizeSelect.value = state.dataSize;
  el.latencySelect.value = state.latency;
  el.interpretRange.value = state.interpretability;
  el.regimeRange.value = state.regime;
}

function renderHeaderValues() {
  el.historyValue.textContent = `${state.history} steps`;
  el.horizonValue.textContent = `${state.horizon} steps`;
  el.interpretValue.textContent = `${state.interpretability} / 100`;
  el.regimeValue.textContent = regimeLabel(state.regime);
}

function renderMetrics(stats) {
  const bestMeta = MODEL_META[stats.bestKey];
  el.matrixValue.textContent = `${formatInteger(stats.fullCells)} cells`;
  el.matrixSub.textContent = `${state.history} × ${state.history}, about ${formatMB(stats.fullMB)} per head in float32`;
  el.complexityValue.textContent = bestMeta.complexity;

  const complexitySubs = {
    lstm: 'Recurrent state update with a selective attention readout rather than full timestep-to-timestep self-attention',
    transformer: 'Every timestep interacts with every other timestep inside dense self-attention',
    patchtst: `Patch size p = ${stats.patchSize} gives ${stats.patchTokens} tokens instead of ${state.history} raw timesteps`,
    informer: 'ProbSparse attention keeps only the most informative query interactions',
    autoformer: 'Decomposition and autocorrelation reduce long-range matching burden for long periodic horizons',
  };
  el.complexitySub.textContent = complexitySubs[stats.bestKey];

  el.recommendValue.textContent = bestMeta.label;
  el.recommendSub.textContent = bestMeta.use;
  el.confidenceValue.textContent = `${stats.confidence.value}%`;
  el.confidenceSub.textContent = `${stats.confidence.label}: ${stats.confidence.message}`;
}

function renderExplanation(stats) {
  const reasons = whyReasons(stats.bestKey, stats);
  el.whyList.innerHTML = reasons.map((text) => `<li>${text}</li>`).join('');
  el.hydroText.innerHTML = hydrologyExplanation(stats.bestKey, stats);
}

function render() {
  renderHeaderValues();
  const stats = computeStats();
  renderMetrics(stats);
  renderArchitectureStrip(stats);
  renderCostBars(stats);
  renderExplanation(stats);
  renderVisuals(stats);
}

function bindControls() {
  el.historyRange.addEventListener('input', (e) => {
    state.history = Number(e.target.value);
    render();
  });

  el.horizonRange.addEventListener('input', (e) => {
    state.horizon = Number(e.target.value);
    render();
  });

  el.dataSizeSelect.addEventListener('change', (e) => {
    state.dataSize = e.target.value;
    render();
  });

  el.latencySelect.addEventListener('change', (e) => {
    state.latency = e.target.value;
    render();
  });

  el.interpretRange.addEventListener('input', (e) => {
    state.interpretability = Number(e.target.value);
    render();
  });

  el.regimeRange.addEventListener('input', (e) => {
    state.regime = Number(e.target.value);
    render();
  });
}

renderScenarios();
syncControlsFromState();
bindControls();
render();
