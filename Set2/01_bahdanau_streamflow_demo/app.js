const CONFIG = {
  seed: 639,
  historyDays: 30,
  forecastDays: 7,
  defaultSharpness: 1.3,
  autoplayMs: 1400,
  hydrograph: {
    width: 920,
    height: 470,
    margin: { top: 40, right: 34, bottom: 56, left: 68 }
  },
  heatmap: {
    width: 620,
    height: 290,
    margin: { top: 32, right: 18, bottom: 38, left: 54 }
  },
  attentionLinksToShow: 6
};

const state = {
  step: 1,
  sharpness: CONFIG.defaultSharpness,
  showBaseline: true,
  showHeatmap: true,
  visibleSeries: "both",
  playing: false,
  hoveredCell: null,
  timer: null
};

const DATA = buildSyntheticData();
const ELEMENTS = {};

init();

function init() {
  cacheElements();
  bindEvents();
  render();
}

function cacheElements() {
  ELEMENTS.resetBtn = document.getElementById("resetBtn");
  ELEMENTS.prevBtn = document.getElementById("prevBtn");
  ELEMENTS.playBtn = document.getElementById("playBtn");
  ELEMENTS.nextBtn = document.getElementById("nextBtn");
  ELEMENTS.stepSlider = document.getElementById("stepSlider");
  ELEMENTS.sliderReadout = document.getElementById("sliderReadout");
  ELEMENTS.baselineToggle = document.getElementById("baselineToggle");
  ELEMENTS.heatmapToggle = document.getElementById("heatmapToggle");
  ELEMENTS.sharpnessSlider = document.getElementById("sharpnessSlider");
  ELEMENTS.sharpnessReadout = document.getElementById("sharpnessReadout");
  ELEMENTS.seriesToggle = document.getElementById("seriesToggle");

  ELEMENTS.metricStep = document.getElementById("metricStep");
  ELEMENTS.metricArgmax = document.getElementById("metricArgmax");
  ELEMENTS.metricPeak = document.getElementById("metricPeak");
  ELEMENTS.metricEntropy = document.getElementById("metricEntropy");
  ELEMENTS.metricContext = document.getElementById("metricContext");
  ELEMENTS.metricVerdict = document.getElementById("metricVerdict");

  ELEMENTS.hydrographSvg = document.getElementById("hydrographSvg");
  ELEMENTS.hydroNote = document.getElementById("hydroNote");
  ELEMENTS.heatmapSvg = document.getElementById("heatmapSvg");
  ELEMENTS.heatmapCard = document.getElementById("heatmapCard");
  ELEMENTS.heatmapNote = document.getElementById("heatmapNote");
  ELEMENTS.heatmapReadout = document.getElementById("heatmapReadout");

  ELEMENTS.mathWorked = document.getElementById("mathWorked");
  ELEMENTS.interpretationText = document.getElementById("interpretationText");
  ELEMENTS.stepExplanation = document.getElementById("stepExplanation");
}

function bindEvents() {
  ELEMENTS.resetBtn.addEventListener("click", resetDemo);
  ELEMENTS.prevBtn.addEventListener("click", () => stepTo(state.step === 1 ? CONFIG.forecastDays : state.step - 1));
  ELEMENTS.nextBtn.addEventListener("click", () => stepTo(state.step === CONFIG.forecastDays ? 1 : state.step + 1));
  ELEMENTS.playBtn.addEventListener("click", togglePlay);

  ELEMENTS.stepSlider.addEventListener("input", (event) => {
    pauseAutoplay();
    stepTo(Number(event.target.value));
  });

  ELEMENTS.baselineToggle.addEventListener("change", (event) => {
    state.showBaseline = event.target.checked;
    render();
  });

  ELEMENTS.heatmapToggle.addEventListener("change", (event) => {
    state.showHeatmap = event.target.checked;
    render();
  });

  ELEMENTS.sharpnessSlider.addEventListener("input", (event) => {
    state.sharpness = Number(event.target.value);
    pauseAutoplay();
    render();
  });

  ELEMENTS.seriesToggle.addEventListener("click", (event) => {
    const btn = event.target.closest("button[data-series]");
    if (!btn) return;
    pauseAutoplay();
    state.visibleSeries = btn.dataset.series;
    Array.from(ELEMENTS.seriesToggle.querySelectorAll("button")).forEach((node) => {
      node.classList.toggle("active", node === btn);
    });
    render();
  });
}

function resetDemo() {
  pauseAutoplay();
  state.step = 1;
  state.sharpness = CONFIG.defaultSharpness;
  state.showBaseline = true;
  state.showHeatmap = true;
  state.visibleSeries = "both";
  state.hoveredCell = null;

  ELEMENTS.stepSlider.value = "1";
  ELEMENTS.baselineToggle.checked = true;
  ELEMENTS.heatmapToggle.checked = true;
  ELEMENTS.sharpnessSlider.value = String(CONFIG.defaultSharpness);
  Array.from(ELEMENTS.seriesToggle.querySelectorAll("button")).forEach((node) => {
    node.classList.toggle("active", node.dataset.series === "both");
  });

  render();
}

function togglePlay() {
  if (state.playing) {
    pauseAutoplay();
  } else {
    state.playing = true;
    ELEMENTS.playBtn.textContent = "Pause";
    state.timer = window.setInterval(() => {
      stepTo(state.step === CONFIG.forecastDays ? 1 : state.step + 1, { silentPause: true });
    }, CONFIG.autoplayMs);
  }
}

function pauseAutoplay() {
  if (state.timer) {
    window.clearInterval(state.timer);
    state.timer = null;
  }
  state.playing = false;
  if (ELEMENTS.playBtn) {
    ELEMENTS.playBtn.textContent = "Play";
  }
}

function stepTo(step, options = {}) {
  state.step = clamp(step, 1, CONFIG.forecastDays);
  if (!options.silentPause) pauseAutoplay();
  render();
}

function render() {
  const computed = computeAttentionOutputs(DATA, state.sharpness);
  const current = computed.steps[state.step - 1];

  ELEMENTS.stepSlider.value = String(state.step);
  ELEMENTS.sliderReadout.textContent = `Forecast day ${state.step} of ${CONFIG.forecastDays}`;
  ELEMENTS.sharpnessReadout.textContent = `${state.sharpness.toFixed(2)}×`;
  ELEMENTS.hydroNote.textContent = `Selected forecast day: ${CONFIG.historyDays + state.step}`;

  renderMetrics(current, computed);
  renderHydrograph(current, computed);
  renderHeatmap(computed);
  renderMath(current);
  renderInterpretation(current);
  renderStepExplanation(current);
}

function renderMetrics(current, computed) {
  const argmaxDay = current.argmaxIndex + 1;
  const peak = current.alpha[current.argmaxIndex];
  const entropy = current.entropy;
  const contextNorm = norm(current.context);
  const verdict = makeVerdict(current, computed.fixedContext);

  ELEMENTS.metricStep.textContent = String(state.step);
  ELEMENTS.metricArgmax.textContent = `Day ${argmaxDay}`;
  ELEMENTS.metricPeak.textContent = peak.toFixed(3);
  ELEMENTS.metricEntropy.textContent = entropy.toFixed(2);
  ELEMENTS.metricContext.textContent = contextNorm.toFixed(3);
  ELEMENTS.metricVerdict.textContent = verdict;
}

function renderHydrograph(current, computed) {
  const svg = ELEMENTS.hydrographSvg;
  const { width, height, margin } = CONFIG.hydrograph;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const topBandH = 112;
  const flowTop = margin.top + topBandH;
  const flowBottom = margin.top + innerH;
  const flowHeight = flowBottom - flowTop;

  const histX = (idx) => margin.left + (idx / (CONFIG.historyDays + CONFIG.forecastDays - 1)) * innerW;
  const allFlow = DATA.streamflow.concat(computed.predictionsAttention);
  const maxFlow = Math.max(...allFlow) * 1.12;
  const minFlow = 0;
  const yFlow = (value) => flowBottom - ((value - minFlow) / (maxFlow - minFlow)) * flowHeight;

  const maxRain = Math.max(...DATA.rainfall) * 1.14;
  const yRain = (value) => margin.top + ((maxRain - value) / maxRain) * (topBandH - 12);

  const forecastStartX = histX(CONFIG.historyDays - 1) + innerW / (CONFIG.historyDays + CONFIG.forecastDays - 1);
  const selectedForecastIndex = CONFIG.historyDays - 1 + state.step;
  const selectedForecastX = histX(selectedForecastIndex);
  const selectedForecastY = yFlow(computed.predictionsAttention[state.step - 1]);

  const histPath = buildPath(DATA.streamflow.map((v, i) => [histX(i), yFlow(v)]));
  const attPath = buildPath(computed.predictionsAttention.map((v, i) => [histX(CONFIG.historyDays + i), yFlow(v)]));
  const basePath = buildPath(computed.predictionsBaseline.map((v, i) => [histX(CONFIG.historyDays + i), yFlow(v)]));

  const xTicks = [0, 4, 9, 14, 19, 24, 29, 33, 36];
  const yTicks = niceTicks(0, maxFlow, 5);
  const rainTicks = niceTicks(0, maxRain, 4);

  const linkIndices = topIndices(current.alpha, CONFIG.attentionLinksToShow).sort((a, b) => current.alpha[b] - current.alpha[a]);
  const linkPaths = linkIndices.map((idx) => {
    const x0 = histX(idx);
    const y0 = yFlow(DATA.streamflow[idx]);
    const cpY = margin.top + 46 + (1 - current.alpha[idx] / current.alpha[current.argmaxIndex]) * 34;
    const path = `M ${x0.toFixed(1)} ${y0.toFixed(1)} Q ${(x0 + selectedForecastX) / 2} ${cpY.toFixed(1)} ${selectedForecastX.toFixed(1)} ${selectedForecastY.toFixed(1)}`;
    const opacity = 0.22 + 0.72 * (current.alpha[idx] / current.alpha[current.argmaxIndex]);
    const strokeWidth = 1.6 + 6.4 * current.alpha[idx];
    return `<path d="${path}" fill="none" stroke="rgba(124,79,159,${opacity.toFixed(3)})" stroke-width="${strokeWidth.toFixed(2)}" stroke-linecap="round"/>`;
  }).join("");

  const rainBars = DATA.rainfall.map((value, i) => {
    const barW = innerW / (CONFIG.historyDays + CONFIG.forecastDays - 1) * 0.62;
    const x = histX(i) - barW / 2;
    const y = yRain(value);
    const h = Math.max(0, margin.top + topBandH - y);
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" rx="4" fill="rgba(107,143,184,0.82)"/>`;
  }).join("");

  const histPoints = DATA.streamflow.map((value, i) => {
    const x = histX(i);
    const y = yFlow(value);
    const isArgmax = i === current.argmaxIndex;
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${isArgmax ? 5.2 : 2.5}" fill="${isArgmax ? "#7c4f9f" : "rgba(38,70,83,0.32)"}"/>`;
  }).join("");

  const forecastCircles = computed.predictionsAttention.map((value, i) => {
    const x = histX(CONFIG.historyDays + i);
    const y = yFlow(value);
    const selected = i === state.step - 1;
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${selected ? 6.6 : 4.2}" fill="${selected ? "#8b5cf6" : "rgba(124,79,159,0.5)"}" stroke="white" stroke-width="${selected ? 2 : 1.1}"/>`;
  }).join("");

  const baselineCircles = state.showBaseline ? computed.predictionsBaseline.map((value, i) => {
    const x = histX(CONFIG.historyDays + i);
    const y = yFlow(value);
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.2" fill="#d97706" opacity="0.82"/>`;
  }).join("") : "";

  const baselineMarker = state.showBaseline ? (() => {
    const dayMeanX = histX(computed.fixedFocusDay - 1);
    return `
      <line class="baseline-marker-line" x1="${dayMeanX.toFixed(1)}" x2="${dayMeanX.toFixed(1)}" y1="${margin.top + 8}" y2="${flowBottom.toFixed(1)}"/>
      <circle class="baseline-marker-dot" cx="${dayMeanX.toFixed(1)}" cy="${margin.top + 8}" r="6"/>
      <rect x="${(dayMeanX - 42).toFixed(1)}" y="${(margin.top - 22).toFixed(1)}" width="84" height="22" rx="11" fill="rgba(217,119,6,0.12)"/>
      <text x="${dayMeanX.toFixed(1)}" y="${(margin.top - 7).toFixed(1)}" text-anchor="middle" class="small-label" fill="#a65c00">fixed c</text>
    `;
  })() : "";

  const selectedForecastPill = `
    <rect class="selected-forecast-pill" x="${(selectedForecastX - 16).toFixed(1)}" y="${(flowTop - 24).toFixed(1)}" width="32" height="${(flowBottom - flowTop + 36).toFixed(1)}" rx="14"/>
  `;

  const forecastBoundary = `
    <line x1="${forecastStartX.toFixed(1)}" y1="${margin.top.toFixed(1)}" x2="${forecastStartX.toFixed(1)}" y2="${flowBottom.toFixed(1)}" stroke="rgba(124,79,159,0.55)" stroke-width="2.4" stroke-dasharray="8 7"/>
    <text x="${(forecastStartX + 8).toFixed(1)}" y="${(margin.top + 18).toFixed(1)}" class="axis-label" fill="#7c4f9f">forecast starts</text>
  `;

  const axesAndTicks = `
    ${yTicks.map((tick) => {
      const y = yFlow(tick);
      return `
        <line class="grid-line" x1="${margin.left}" x2="${width - margin.right}" y1="${y.toFixed(1)}" y2="${y.toFixed(1)}"/>
        <text x="${(margin.left - 12).toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="end" class="tick-label">${tick.toFixed(0)}</text>
      `;
    }).join("")}
    ${rainTicks.map((tick) => {
      const y = yRain(tick);
      return `<text x="${(width - margin.right + 8).toFixed(1)}" y="${(y + 4).toFixed(1)}" class="tick-label">${tick.toFixed(0)} mm</text>`;
    }).join("")}
    ${xTicks.map((tick) => {
      const x = histX(tick);
      return `
        <line x1="${x.toFixed(1)}" x2="${x.toFixed(1)}" y1="${flowBottom.toFixed(1)}" y2="${(flowBottom + 7).toFixed(1)}" stroke="rgba(38,70,83,0.22)" />
        <text x="${x.toFixed(1)}" y="${(flowBottom + 25).toFixed(1)}" text-anchor="middle" class="tick-label">Day ${tick + 1}</text>
      `;
    }).join("")}
    <text x="${(margin.left - 48).toFixed(1)}" y="${(flowTop - 8).toFixed(1)}" class="axis-label" transform="rotate(-90 ${margin.left - 48} ${flowTop - 8})">Streamflow</text>
    <text x="${(width - margin.right + 42).toFixed(1)}" y="${(margin.top + topBandH / 2).toFixed(1)}" class="axis-label" transform="rotate(-90 ${width - margin.right + 42} ${margin.top + topBandH / 2})">Rainfall</text>
    <text x="${(margin.left + innerW / 2).toFixed(1)}" y="${(height - 14).toFixed(1)}" text-anchor="middle" class="axis-label">Historical input days and forecast days</text>
  `;

  const legend = `
    <g transform="translate(${(margin.left + 8).toFixed(1)} ${margin.top + 12})">
      <circle cx="0" cy="0" r="5" fill="#264653"></circle>
      <text x="12" y="4" class="legend-label">streamflow</text>
      <rect x="82" y="-7" width="14" height="14" rx="3" fill="rgba(107,143,184,0.82)"></rect>
      <text x="103" y="4" class="legend-label">rainfall</text>
      <circle cx="181" cy="0" r="5" fill="#8b5cf6"></circle>
      <text x="193" y="4" class="legend-label">attention forecast</text>
      ${state.showBaseline ? `<circle cx="332" cy="0" r="5" fill="#d97706"></circle><text x="344" y="4" class="legend-label">fixed-context baseline</text>` : ""}
    </g>
  `;

  const streamflowVisible = state.visibleSeries === "both" || state.visibleSeries === "flow";
  const rainfallVisible = state.visibleSeries === "both" || state.visibleSeries === "rain";

  svg.innerHTML = `
    <defs>
      <linearGradient id="hydroBg" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="rgba(255,255,255,0.92)"/>
        <stop offset="100%" stop-color="rgba(246,248,250,0.95)"/>
      </linearGradient>
      <linearGradient id="forecastFill" x1="0" x2="1" y1="0" y2="0">
        <stop offset="0%" stop-color="rgba(124,79,159,0.07)"/>
        <stop offset="100%" stop-color="rgba(124,79,159,0.13)"/>
      </linearGradient>
    </defs>
    <rect x="0" y="0" width="${width}" height="${height}" rx="26" fill="url(#hydroBg)"/>
    <rect class="history-band" x="${margin.left}" y="${margin.top}" width="${(forecastStartX - margin.left).toFixed(1)}" height="${innerH.toFixed(1)}" rx="16"/>
    <rect class="forecast-band" x="${forecastStartX.toFixed(1)}" y="${margin.top}" width="${(width - margin.right - forecastStartX).toFixed(1)}" height="${innerH.toFixed(1)}" rx="16" fill="url(#forecastFill)"/>
    ${axesAndTicks}
    ${forecastBoundary}
    ${selectedForecastPill}
    ${rainfallVisible ? rainBars : ""}
    ${linkPaths}
    ${streamflowVisible ? `<path d="${histPath}" fill="none" stroke="#264653" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/>` : ""}
    ${streamflowVisible ? histPoints : ""}
    ${streamflowVisible ? `<path d="${attPath}" fill="none" stroke="#8b5cf6" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" stroke-dasharray="0 0"/>` : ""}
    ${streamflowVisible && state.showBaseline ? `<path d="${basePath}" fill="none" stroke="#d97706" stroke-width="3" stroke-dasharray="7 6" stroke-linecap="round"/>` : ""}
    ${streamflowVisible ? forecastCircles : ""}
    ${streamflowVisible ? baselineCircles : ""}
    ${streamflowVisible ? `<circle cx="${selectedForecastX.toFixed(1)}" cy="${selectedForecastY.toFixed(1)}" r="8.2" fill="#8b5cf6" stroke="#ffffff" stroke-width="2.4"/>` : ""}
    ${baselineMarker}
    ${legend}
  `;
}

function renderHeatmap(computed) {
  ELEMENTS.heatmapCard.classList.toggle("hidden-panel", !state.showHeatmap);
  if (!state.showHeatmap) {
    return;
  }

  const svg = ELEMENTS.heatmapSvg;
  const { width, height, margin } = CONFIG.heatmap;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const cols = CONFIG.historyDays;
  const rows = CONFIG.forecastDays;
  const cellW = innerW / cols;
  const cellH = innerH / rows;

  const cellMarkup = [];
  computed.steps.forEach((step, rowIndex) => {
    step.alpha.forEach((alpha, colIndex) => {
      const x = margin.left + colIndex * cellW;
      const y = margin.top + rowIndex * cellH;
      const isSelectedRow = rowIndex === state.step - 1;
      const isHovered = state.hoveredCell && state.hoveredCell.row === rowIndex && state.hoveredCell.col === colIndex;
      const fill = heatColor(alpha, isSelectedRow);
      const stroke = isHovered ? "rgba(35,55,73,0.9)" : (isSelectedRow ? "rgba(124,79,159,0.28)" : "rgba(255,255,255,0.18)");
      const strokeWidth = isHovered ? 2.2 : 1;
      cellMarkup.push(`<rect data-row="${rowIndex}" data-col="${colIndex}" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(cellW - 1.1).toFixed(1)}" height="${(cellH - 1.1).toFixed(1)}" rx="5" fill="${fill}" stroke="${stroke}" stroke-width="${strokeWidth}"/>`);
    });
  });

  const xTickPositions = [0, 4, 9, 14, 19, 24, 29];
  const selectedRowY = margin.top + (state.step - 1) * cellH;

  svg.innerHTML = `
    <rect x="0" y="0" width="${width}" height="${height}" rx="24" fill="rgba(255,255,255,0.94)"/>
    <rect x="${margin.left}" y="${selectedRowY.toFixed(1)}" width="${innerW.toFixed(1)}" height="${cellH.toFixed(1)}" rx="10" fill="rgba(124,79,159,0.08)" stroke="rgba(124,79,159,0.18)"/>
    ${cellMarkup.join("")}
    ${xTickPositions.map((idx) => {
      const x = margin.left + idx * cellW + cellW / 2;
      return `<text x="${x.toFixed(1)}" y="${(height - 14).toFixed(1)}" text-anchor="middle" class="tick-label">${idx + 1}</text>`;
    }).join("")}
    ${Array.from({ length: rows }, (_, r) => {
      const y = margin.top + r * cellH + cellH / 2 + 4;
      return `<text x="${(margin.left - 12).toFixed(1)}" y="${y.toFixed(1)}" text-anchor="end" class="tick-label">t=${r + 1}</text>`;
    }).join("")}
    <text x="${(margin.left + innerW / 2).toFixed(1)}" y="${(height - 2).toFixed(1)}" text-anchor="middle" class="axis-label">Input day j</text>
    <text x="${14}" y="${(margin.top + innerH / 2).toFixed(1)}" class="axis-label" transform="rotate(-90 14 ${margin.top + innerH / 2})">Forecast step t</text>
  `;

  const rects = svg.querySelectorAll("rect[data-row]");
  rects.forEach((rect) => {
    rect.addEventListener("mouseenter", onHeatCellHover);
    rect.addEventListener("mouseleave", () => {
      state.hoveredCell = null;
      ELEMENTS.heatmapReadout.textContent = "Hover over the heatmap to read a specific weight.";
      ELEMENTS.heatmapNote.innerHTML = "Hover a cell to inspect α<sub>t,j</sub>";
      renderHeatmap(computed);
    });
  });

  const currentHover = state.hoveredCell;
  if (currentHover) {
    const alpha = computed.steps[currentHover.row].alpha[currentHover.col];
    ELEMENTS.heatmapReadout.textContent = `Hovered cell: step ${currentHover.row + 1}, input day ${currentHover.col + 1}, α = ${alpha.toFixed(3)}`;
    ELEMENTS.heatmapNote.innerHTML = `Step ${currentHover.row + 1}, day ${currentHover.col + 1}`;
  }
}

function onHeatCellHover(event) {
  const target = event.currentTarget;
  state.hoveredCell = {
    row: Number(target.dataset.row),
    col: Number(target.dataset.col)
  };
  render();
}

function renderMath(current) {
  const argmax = current.argmaxIndex;
  const s = current.decoderState;
  const h = DATA.hiddenStates[argmax];
  const w1s = current.w1s;
  const w2h = current.perDay[argmax].w2h;
  const summed = addVectors(w1s, w2h).map((v, i) => v + DATA.params.b[i]);
  const z = current.perDay[argmax].z;
  const score = current.scores[argmax];
  const alpha = current.alpha[argmax];
  const context = current.context;

  const worked = [
    `Selected day j* = ${argmax + 1}`,
    `s^(t−1)  = ${formatVector(s)}`,
    `h^(j*)   = ${formatVector(h)}`,
    `W1 s     = ${formatVector(w1s)}`,
    `W2 h     = ${formatVector(w2h)}`,
    `+ b      = ${formatVector(DATA.params.b)}`,
    `sum      = ${formatVector(summed)}`,
    `tanh(.)  = ${formatVector(z)}`,
    `e_(t,j*) = v^T z = ${score.toFixed(3)}`,
    `α_(t,j*) = softmax(e_(t,j*)) = ${alpha.toFixed(3)}`,
    `c_t      = Σ α_(t,j) h^(j) = ${formatVector(context)}`,
    `||c_t||  = ${norm(context).toFixed(3)}`
  ].join("\n");

  ELEMENTS.mathWorked.textContent = worked;
}

function renderInterpretation(current) {
  const topDays = topIndices(current.alpha, 3).map((idx) => idx + 1);
  const rainSummary = topDays.map((day) => DATA.rainfall[day - 1].toFixed(1));
  const smSummary = topDays.map((day) => DATA.soilMoisture[day - 1].toFixed(2));
  const phase = interpretationPhase(state.step);
  const topDayText = topDays.map((day, i) => `day ${day} (rainfall ${rainSummary[i]} mm, soil moisture ${smSummary[i]})`).join(", ");

  let body = "";
  if (phase === "early") {
    body = `At this short horizon, the decoder still cares most about recent storm response. Attention concentrates on ${topDayText}, where rainfall and elevated flow are still fresh in the catchment memory. Hydrologically, that means the model is leaning on near-term runoff generation plus wet near-surface storage. The explanation is quasi-interpretable because the attended days coincide with the late part of the stronger storm and the beginning of the recession limb.`;
  } else if (phase === "middle") {
    body = `At this middle horizon, attention becomes more balanced. The decoder still sees recent forcing, but it also starts borrowing from slightly older days with stronger soil-moisture storage. The attended set — ${topDayText} — shows this blend clearly: some mass remains on post-storm days, while some shifts toward days that encode delayed drainage and recession behavior. This is the moment where Bahdanau attention stops acting like a recency detector and starts acting like a storage-aware summarizer.`;
  } else {
    body = `At this longer horizon, the decoder is less interested in the sharpest rainfall pulse and more interested in storage, recharge, and baseflow memory. The attended days — ${topDayText} — carry information about wet catchment state rather than only immediate rainfall. In hydrological terms, later forecast days are asking: which historical states best explain the tail of recession and continued channel support after the flood peak has passed?`;
  }

  ELEMENTS.interpretationText.textContent = body;
}

function renderStepExplanation(current) {
  const argmaxDay = current.argmaxIndex + 1;
  const focusStrength = current.alpha[current.argmaxIndex];
  const fixedDelta = Math.abs(computedFixedFocusDifference(current));

  let sentence = `Forecast step ${state.step} highlights day ${argmaxDay} most strongly, with peak attention weight ${focusStrength.toFixed(3)}. `;
  if (state.step <= 2) {
    sentence += `The decoder is still closely tied to the recent storm pulse, so it prefers a context that preserves event timing and quick runoff cues rather than compressing everything into one fixed summary.`;
  } else if (state.step <= 5) {
    sentence += `The decoder now mixes recent rainfall with wetness storage, so horizon-specific context matters more than a single bottleneck vector. This is exactly where attention gives a cleaner teaching story than a fixed encoder summary.`;
  } else {
    sentence += `The decoder is looking deeper into hydrologic memory. Older wet days and recession states become more valuable, which is hard to recover from one fixed vector but natural for attention.`;
  }
  sentence += ` Relative to the fixed baseline, the attended center shifts by about ${fixedDelta.toFixed(1)} historical days.`;

  ELEMENTS.stepExplanation.textContent = sentence;
}

function computeAttentionOutputs(data, sharpness) {
  const steps = [];
  const predictionsAttention = [];
  const predictionsBaseline = [];
  const fixedAlpha = softmax(data.fixedScores.map((v) => sharpness * v));
  const fixedContext = weightedContext(data.hiddenStates, fixedAlpha);
  const fixedFocusDay = weightedMeanDay(fixedAlpha);

  for (let t = 1; t <= CONFIG.forecastDays; t++) {
    const decoderState = data.decoderStates[t - 1];
    const w1s = matVecMul(data.params.W1, decoderState);
    const perDay = [];
    const scores = data.hiddenStates.map((h) => {
      const w2h = matVecMul(data.params.W2, h);
      const sum = addVectors(addVectors(w1s, w2h), data.params.b);
      const z = sum.map((v) => Math.tanh(v));
      perDay.push({ w2h, z });
      return dot(data.params.v, z);
    });

    const alpha = softmax(scores.map((v) => sharpness * v));
    const context = weightedContext(data.hiddenStates, alpha);
    const argmaxIndex = argmax(alpha);
    const entropy = attentionEntropy(alpha);

    const predictionAttention = makeAttentionPrediction(data, context, t);
    const predictionBaseline = makeBaselinePrediction(data, fixedContext, t);

    predictionsAttention.push(predictionAttention);
    predictionsBaseline.push(predictionBaseline);

    steps.push({
      t,
      decoderState,
      w1s,
      perDay,
      scores,
      alpha,
      context,
      argmaxIndex,
      entropy,
      predictionAttention,
      predictionBaseline
    });
  }

  return {
    steps,
    predictionsAttention,
    predictionsBaseline,
    fixedAlpha,
    fixedContext,
    fixedFocusDay
  };
}

function buildSyntheticData() {
  const rng = mulberry32(CONFIG.seed);
  const n = CONFIG.historyDays;
  const rainfall = [];
  const temperature = [];
  const soilMoisture = [];
  const streamflow = [];
  const quickKernel = [0.0, 0.18, 0.34, 0.23, 0.13, 0.06];

  for (let i = 0; i < n; i++) {
    const d = i + 1;
    const background = rng() < 0.22 ? 0.8 + 3.2 * rng() : 0;
    const event1 = 28 * gaussian(d, 9, 1.45);
    const event2 = 39 * gaussian(d, 24, 1.85);
    const tail = 6 * gaussian(d, 28, 1.7);
    rainfall.push(round1(background + event1 + event2 + tail));

    const temp = 22.2 + 3.8 * Math.sin((d - 3) / 5.2) + 1.4 * Math.cos(d / 3.8) + (rng() - 0.5) * 0.8;
    temperature.push(round1(temp));
  }

  for (let i = 0; i < n; i++) {
    const prev = i === 0 ? 0.33 : soilMoisture[i - 1];
    const evap = Math.max(0, temperature[i] - 18) * 0.0054;
    const recharge = 0.0095 * rainfall[i];
    const sm = clamp(0.90 * prev + recharge - evap + 0.018, 0.18, 0.92);
    soilMoisture.push(round3(sm));
  }

  let slowStore = 4.2;
  for (let i = 0; i < n; i++) {
    let quick = 0;
    for (let k = 1; k < quickKernel.length; k++) {
      const src = i - k;
      if (src >= 0) quick += rainfall[src] * quickKernel[k];
    }
    slowStore = 0.84 * slowStore + 0.055 * rainfall[i] + 0.70 * soilMoisture[i];
    const q = 2.6 + quick + 0.30 * slowStore + 2.1 * soilMoisture[i];
    streamflow.push(round2(q));
  }

  const rainMax = Math.max(...rainfall);
  const flowMax = Math.max(...streamflow);
  const smMin = Math.min(...soilMoisture);
  const smMax = Math.max(...soilMoisture);

  const recent3 = rainfall.map((_, i) => (
    0.6 * safeAt(rainfall, i - 1) + 0.3 * safeAt(rainfall, i - 2) + 0.1 * safeAt(rainfall, i - 3)
  ));
  const ante8 = rainfall.map((_, i) => average(Array.from({ length: 8 }, (_, k) => safeAt(rainfall, i - k))));
  const recent3Max = Math.max(...recent3, 1e-6);
  const ante8Max = Math.max(...ante8, 1e-6);

  const hiddenStates = rainfall.map((rain, i) => {
    const qN = streamflow[i] / flowMax;
    const rN = rain / rainMax;
    const sN = normalize(soilMoisture[i], smMin, smMax);
    const recentN = recent3[i] / recent3Max;
    const anteN = ante8[i] / ante8Max;
    const recency = (i + 1) / n;
    return [
      round3(0.58 * rN + 0.30 * qN + 0.18 * recentN),
      round3(0.62 * sN + 0.22 * anteN + 0.12 * qN),
      round3(recency)
    ];
  });

  const decoderStates = Array.from({ length: CONFIG.forecastDays }, (_, idx) => {
    const t = idx + 1;
    return [
      round3(1.05 - 0.10 * (t - 1)),
      round3(0.38 + 0.11 * (t - 1)),
      round3(1.00 - 0.13 * (t - 1))
    ];
  });

  const params = {
    W1: [
      [1.15, -0.35, 0.60],
      [-0.25, 1.10, -0.25],
      [0.35, 0.45, 0.95]
    ],
    W2: [
      [1.30, 0.10, 0.95],
      [0.10, 1.25, -0.55],
      [0.70, 0.35, 0.90]
    ],
    b: [-0.90, -0.55, -1.00],
    v: [1.05, 0.95, 0.75]
  };

  const fixedScores = hiddenStates.map((h) => 0.85 * h[0] + 0.70 * h[1] + 0.95 * h[2] - 1.08);

  return {
    rainfall,
    temperature,
    soilMoisture,
    streamflow,
    hiddenStates,
    decoderStates,
    params,
    fixedScores,
    lastFlow: streamflow[streamflow.length - 1]
  };
}

function makeAttentionPrediction(data, context, t) {
  const q = 0.9
    + data.lastFlow * Math.pow(0.82, t)
    + 2.8 * context[0]
    + 3.9 * context[1]
    - 1.1 * (1 - context[2])
    - 0.45 * (t - 1);
  return round2(q);
}

function makeBaselinePrediction(data, fixedContext, t) {
  const q = 0.9
    + data.lastFlow * Math.pow(0.82, t)
    + 2.4 * fixedContext[0]
    + 3.3 * fixedContext[1]
    - 0.7 * (1 - fixedContext[2])
    - 0.33 * (t - 1);
  return round2(q);
}

function makeVerdict(current, fixedContext) {
  const attendedDay = current.argmaxIndex + 1;
  const fixedDay = weightedMeanDay(softmax(DATA.fixedScores.map((v) => state.sharpness * v)));
  const shift = Math.abs(attendedDay - fixedDay);
  if (shift >= 5) return "Attention clearly shifts with horizon; the fixed bottleneck stays too generic.";
  if (norm(subtractVectors(current.context, fixedContext)) > 0.10) return "Attention adapts its summary; the fixed bottleneck looks smoother but less specific.";
  return "Attention is still more horizon-aware than the fixed bottleneck, even when the shift is modest.";
}

function interpretationPhase(step) {
  if (step <= 2) return "early";
  if (step <= 5) return "middle";
  return "late";
}

function computedFixedFocusDifference(current) {
  const fixedFocus = weightedMeanDay(softmax(DATA.fixedScores.map((v) => state.sharpness * v)));
  return (current.argmaxIndex + 1) - fixedFocus;
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a += 0x6D2B79F5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(x, mu, sigma) {
  return Math.exp(-0.5 * ((x - mu) / sigma) ** 2);
}

function round1(x) { return Math.round(x * 10) / 10; }
function round2(x) { return Math.round(x * 100) / 100; }
function round3(x) { return Math.round(x * 1000) / 1000; }

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalize(value, min, max) {
  if (max <= min) return 0;
  return (value - min) / (max - min);
}

function safeAt(array, index) {
  return index >= 0 && index < array.length ? array[index] : 0;
}

function average(arr) {
  return arr.reduce((sum, value) => sum + value, 0) / Math.max(1, arr.length);
}

function matVecMul(matrix, vector) {
  return matrix.map((row) => round3(dot(row, vector)));
}

function dot(a, b) {
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}

function addVectors(a, b) {
  return a.map((value, i) => round3(value + b[i]));
}

function subtractVectors(a, b) {
  return a.map((value, i) => value - b[i]);
}

function norm(vec) {
  return Math.sqrt(vec.reduce((sum, value) => sum + value * value, 0));
}

function softmax(values) {
  const maxValue = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - maxValue));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function weightedContext(hiddenStates, alpha) {
  const dims = hiddenStates[0].length;
  const out = Array.from({ length: dims }, () => 0);
  hiddenStates.forEach((h, idx) => {
    for (let i = 0; i < dims; i++) out[i] += alpha[idx] * h[i];
  });
  return out.map((v) => round3(v));
}

function attentionEntropy(alpha) {
  return -alpha.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
}

function argmax(array) {
  let bestIndex = 0;
  let bestValue = array[0];
  for (let i = 1; i < array.length; i++) {
    if (array[i] > bestValue) {
      bestValue = array[i];
      bestIndex = i;
    }
  }
  return bestIndex;
}

function topIndices(array, k) {
  return array
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .slice(0, k)
    .map((item) => item.index);
}

function weightedMeanDay(alpha) {
  return alpha.reduce((sum, value, idx) => sum + value * (idx + 1), 0);
}

function niceTicks(min, max, count) {
  const step = (max - min) / Math.max(1, count - 1);
  return Array.from({ length: count }, (_, i) => round1(min + step * i));
}

function buildPath(points) {
  if (!points.length) return "";
  return points.map((point, idx) => `${idx === 0 ? "M" : "L"} ${point[0].toFixed(1)} ${point[1].toFixed(1)}`).join(" ");
}

function formatVector(vector) {
  return `[${vector.map((v) => Number(v).toFixed(3)).join(", ")}]`;
}

function heatColor(alpha, selectedRow) {
  const t = clamp(alpha / 0.16, 0, 1);
  const r = Math.round(238 - 165 * t);
  const g = Math.round(244 - 138 * t);
  const b = Math.round(250 - 123 * t);
  const base = `rgb(${r}, ${g}, ${b})`;
  if (!selectedRow) return base;
  return mixRgb(base, "rgb(124, 79, 159)", 0.10 + 0.18 * t);
}

function mixRgb(rgbA, rgbB, amount) {
  const a = rgbA.match(/\d+/g).map(Number);
  const b = rgbB.match(/\d+/g).map(Number);
  const mix = a.map((v, i) => Math.round(v * (1 - amount) + b[i] * amount));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}
