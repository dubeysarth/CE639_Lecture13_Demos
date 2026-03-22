const DEFAULT_STATE = {
  useScaling: true,
  usePosition: true,
  twoHeads: false,
  swapped: false,
  dk: 8,
  temperature: 1.0,
  mode: "sequence",
  selectedToken: 3,
  activeHead: 0
};

const TOKENS = [
  { day: "Day 1", rain: 2,  soil: 0.30 },
  { day: "Day 2", rain: 18, soil: 0.38 },
  { day: "Day 3", rain: 5,  soil: 0.47 },
  { day: "Day 4", rain: 26, soil: 0.63 },
  { day: "Day 5", rain: 9,  soil: 0.57 },
  { day: "Day 6", rain: 4,  soil: 0.53 }
];

const SWAP_PAIR = [1, 4];
const state = { ...DEFAULT_STATE };

const els = {
  resetBtn: document.getElementById("resetBtn"),
  scalingToggle: document.getElementById("scalingToggle"),
  positionToggle: document.getElementById("positionToggle"),
  headToggle: document.getElementById("headToggle"),
  headToggleText: document.getElementById("headToggleText"),
  swapBtn: document.getElementById("swapBtn"),
  swapBtnText: document.getElementById("swapBtnText"),
  dkSlider: document.getElementById("dkSlider"),
  dkValue: document.getElementById("dkValue"),
  tempSlider: document.getElementById("tempSlider"),
  tempValue: document.getElementById("tempValue"),
  modeSelect: document.getElementById("modeSelect"),
  metricDk: document.getElementById("metricDk"),
  metricScore: document.getElementById("metricScore"),
  metricWeight: document.getElementById("metricWeight"),
  metricVariance: document.getElementById("metricVariance"),
  metricOrder: document.getElementById("metricOrder"),
  metricMode: document.getElementById("metricMode"),
  matrixStatus: document.getElementById("matrixStatus"),
  matrixQ: document.getElementById("matrixQ"),
  matrixK: document.getElementById("matrixK"),
  matrixV: document.getElementById("matrixV"),
  matrixS: document.getElementById("matrixS"),
  matrixScaled: document.getElementById("matrixScaled"),
  matrixA: document.getElementById("matrixA"),
  matrixO: document.getElementById("matrixO"),
  scaledScoreTitle: document.getElementById("scaledScoreTitle"),
  interactionWrap: document.getElementById("interactionWrap"),
  interactionNote: document.getElementById("interactionNote"),
  selectedTokenChip: document.getElementById("selectedTokenChip"),
  permOriginal: document.getElementById("permOriginal"),
  permSwapped: document.getElementById("permSwapped"),
  permDiffBars: document.getElementById("permDiffBars"),
  permSummaryBadge: document.getElementById("permSummaryBadge"),
  permInterpretation: document.getElementById("permInterpretation"),
  peCanvas: document.getElementById("peCanvas"),
  peRowChip: document.getElementById("peRowChip"),
  peNote: document.getElementById("peNote"),
  settingMeaning: document.getElementById("settingMeaning"),
  equationFocus: document.getElementById("equationFocus"),
  classPrompts: document.getElementById("classPrompts"),
  headTabs: document.getElementById("headTabs"),
  panelA: document.getElementById("panelA"),
  panelB: document.getElementById("panelB"),
  panelC: document.getElementById("panelC"),
  panelD: document.getElementById("panelD")
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
  return sum;
}

function variance(values) {
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  return values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
}

function transpose(matrix) {
  return matrix[0].map((_, c) => matrix.map(row => row[c]));
}

function matMul(A, B) {
  const Bt = transpose(B);
  return A.map(row => Bt.map(col => dot(row, col)));
}

function softmaxRows(matrix, temperature) {
  return matrix.map(row => {
    const adjusted = row.map(v => v / temperature);
    const maxValue = Math.max(...adjusted);
    const exps = adjusted.map(v => Math.exp(v - maxValue));
    const sum = exps.reduce((s, v) => s + v, 0);
    return exps.map(v => v / sum);
  });
}

function applyMatrix(A, V) {
  return A.map(row => {
    const out = [];
    for (let c = 0; c < V[0].length; c += 1) {
      let sum = 0;
      for (let j = 0; j < row.length; j += 1) sum += row[j] * V[j][c];
      out.push(sum);
    }
    return out;
  });
}

function swapArrayItems(arr, i, j) {
  const out = arr.slice();
  [out[i], out[j]] = [out[j], out[i]];
  return out;
}

function invertPermutation(mapping) {
  const inv = new Array(mapping.length);
  mapping.forEach((v, i) => {
    inv[v] = i;
  });
  return inv;
}

function reorderByIndices(arr, indices) {
  return indices.map(i => arr[i]);
}

function normalizeTokenFeatures(tokens) {
  return tokens.map((token, i) => {
    const rain = token.rain / 30;
    const soil = clamp((token.soil - 0.25) / 0.45, 0, 1);
    const lagRain = i === 0 ? 0.08 : tokens[i - 1].rain / 30;
    const prevSoil = i === 0 ? soil * 0.85 : clamp((tokens[i - 1].soil - 0.25) / 0.45, 0, 1);
    const recession = clamp(0.58 * prevSoil + 0.42 * soil, 0, 1);

    const quick = [
      1.35 * rain,
      0.95 * lagRain,
      0.72 * rain + 0.42 * soil,
      0.55 * Math.max(rain - lagRain, 0) + 0.22 * soil
    ];

    const slow = [
      1.18 * soil,
      0.92 * recession,
      0.52 * lagRain + 0.28 * soil,
      0.84 * ((soil + recession) / 2)
    ];

    const blend = quick.map((v, idx) => 0.65 * v + 0.35 * slow[idx]);

    return {
      ...token,
      rainNorm: rain,
      soilNorm: soil,
      lagRain,
      recession,
      quick,
      slow,
      blend
    };
  });
}

function sinusoidalPosition(pos, dModel) {
  const row = [];
  for (let i = 0; i < dModel; i += 1) {
    const denom = Math.pow(10000, (2 * Math.floor(i / 2)) / dModel);
    const angle = pos / denom;
    row.push(i % 2 === 0 ? Math.sin(angle) : Math.cos(angle));
  }
  return row;
}

function expandBase(base, dModel, variant = 0) {
  const out = [];
  for (let j = 0; j < dModel; j += 1) {
    const a = base[j % base.length];
    const b = base[(j + 1) % base.length];
    const c = base[(j + 2) % base.length];
    const sign = j % 2 === 0 ? 1 : -1;
    let value;
    if (variant === 0) {
      value =
        (1.12 + 0.022 * j) * a +
        0.46 * b +
        sign * 0.18 * c +
        0.16 * Math.sin((j + 1) * 0.63 + 1.3 * a - 0.85 * b);
    } else if (variant === 1) {
      value =
        (1.02 + 0.018 * j) * a +
        0.30 * b -
        0.12 * c +
        0.18 * Math.cos((j + 1) * 0.55 + 0.8 * a + 0.6 * c);
    } else if (variant === 2) {
      value =
        (1.06 + 0.017 * j) * a +
        0.24 * b +
        0.28 * c +
        0.10 * Math.sin((j + 1) * 0.48 + 0.7 * b);
    } else {
      value =
        (0.98 + 0.014 * j) * a +
        0.34 * b +
        sign * 0.22 * c +
        0.14 * Math.cos((j + 1) * 0.72 + 1.1 * a);
    }
    out.push(value);
  }
  return out;
}

function buildHeadMatrices(features, dk, usePosition, headType) {
  const peScale = usePosition ? 0.58 : 0;
  const Q = [];
  const K = [];
  const V = [];

  features.forEach((token, idx) => {
    const position = sinusoidalPosition(idx, dk);
    const base =
      headType === "quick"
        ? token.quick
        : headType === "slow"
          ? token.slow
          : token.blend;

    const qBase = expandBase(base, dk, headType === "quick" ? 0 : headType === "slow" ? 2 : 0)
      .map((v, j) => v + peScale * position[j]);
    const kBase = expandBase(base, dk, headType === "quick" ? 1 : headType === "slow" ? 3 : 1)
      .map((v, j) => v + peScale * position[j]);

    Q.push(qBase);
    K.push(kBase);

    const valueQuick = [
      1.08 * token.rainNorm + 0.20 * token.soilNorm,
      0.75 * token.lagRain + 0.20 * token.soilNorm,
      0.28 * token.recession + 0.22 * token.rainNorm
    ];
    const valueSlow = [
      0.26 * token.rainNorm + 1.08 * token.soilNorm,
      0.48 * token.recession + 0.52 * token.soilNorm,
      0.25 * token.lagRain + 0.70 * token.recession
    ];
    const valueBlend = valueQuick.map((v, j) => 0.56 * v + 0.44 * valueSlow[j]);

    V.push(headType === "quick" ? valueQuick : headType === "slow" ? valueSlow : valueBlend);
  });

  return { Q, K, V };
}

function computeHead(headMatrices, dk, useScaling, temperature) {
  const rawScores = matMul(headMatrices.Q, transpose(headMatrices.K));
  const scoreIntoSoftmax = rawScores.map(row =>
    row.map(v => (useScaling ? v / Math.sqrt(dk) : v))
  );
  const attention = softmaxRows(scoreIntoSoftmax, temperature);
  const output = applyMatrix(attention, headMatrices.V);

  return {
    ...headMatrices,
    rawScores,
    scoreIntoSoftmax,
    attention,
    output
  };
}

function getSequence(tokens, swapped) {
  if (!swapped) {
    return {
      tokens,
      mapping: [0, 1, 2, 3, 4, 5]
    };
  }
  const mapping = [0, 4, 2, 3, 1, 5];
  const swappedTokens = reorderByIndices(tokens, mapping);
  return { tokens: swappedTokens, mapping };
}

function buildModel(currentState) {
  const featureBase = normalizeTokenFeatures(TOKENS);
  const originalSequence = getSequence(featureBase, false);
  const workingSequence = getSequence(featureBase, currentState.swapped);

  const originalBlend = computeHead(
    buildHeadMatrices(originalSequence.tokens, currentState.dk, currentState.usePosition, "blend"),
    currentState.dk,
    currentState.useScaling,
    currentState.temperature
  );
  const originalQuick = computeHead(
    buildHeadMatrices(originalSequence.tokens, currentState.dk, currentState.usePosition, "quick"),
    currentState.dk,
    currentState.useScaling,
    currentState.temperature
  );
  const originalSlow = computeHead(
    buildHeadMatrices(originalSequence.tokens, currentState.dk, currentState.usePosition, "slow"),
    currentState.dk,
    currentState.useScaling,
    currentState.temperature
  );

  const workingBlend = computeHead(
    buildHeadMatrices(workingSequence.tokens, currentState.dk, currentState.usePosition, "blend"),
    currentState.dk,
    currentState.useScaling,
    currentState.temperature
  );
  const workingQuick = computeHead(
    buildHeadMatrices(workingSequence.tokens, currentState.dk, currentState.usePosition, "quick"),
    currentState.dk,
    currentState.useScaling,
    currentState.temperature
  );
  const workingSlow = computeHead(
    buildHeadMatrices(workingSequence.tokens, currentState.dk, currentState.usePosition, "slow"),
    currentState.dk,
    currentState.useScaling,
    currentState.temperature
  );

  const heads = currentState.twoHeads
    ? [workingQuick, workingSlow]
    : [workingBlend];
  const originalHeads = currentState.twoHeads
    ? [originalQuick, originalSlow]
    : [originalBlend];

  const displayedHeadIndex = clamp(currentState.activeHead, 0, heads.length - 1);
  const displayedHead = heads[displayedHeadIndex];

  const combinedOutput = currentState.twoHeads
    ? heads[0].output.map((row, i) => row.map((v, j) => 0.5 * (v + heads[1].output[i][j])))
    : heads[0].output.map(row => row.slice());

  const swappedSequenceForPermutation = getSequence(featureBase, true);
  const swappedBlend = computeHead(
    buildHeadMatrices(swappedSequenceForPermutation.tokens, currentState.dk, currentState.usePosition, "blend"),
    currentState.dk,
    currentState.useScaling,
    currentState.temperature
  );
  const swappedQuick = computeHead(
    buildHeadMatrices(swappedSequenceForPermutation.tokens, currentState.dk, currentState.usePosition, "quick"),
    currentState.dk,
    currentState.useScaling,
    currentState.temperature
  );
  const swappedSlow = computeHead(
    buildHeadMatrices(swappedSequenceForPermutation.tokens, currentState.dk, currentState.usePosition, "slow"),
    currentState.dk,
    currentState.useScaling,
    currentState.temperature
  );

  const permutationOriginalHeads = currentState.twoHeads ? [originalQuick, originalSlow] : [originalBlend];
  const permutationSwappedHeads = currentState.twoHeads ? [swappedQuick, swappedSlow] : [swappedBlend];

  const originalPermutationCombined = currentState.twoHeads
    ? permutationOriginalHeads[0].output.map((row, i) => row.map((v, j) => 0.5 * (v + permutationOriginalHeads[1].output[i][j])))
    : permutationOriginalHeads[0].output.map(row => row.slice());

  const swappedPermutationCombined = currentState.twoHeads
    ? permutationSwappedHeads[0].output.map((row, i) => row.map((v, j) => 0.5 * (v + permutationSwappedHeads[1].output[i][j])))
    : permutationSwappedHeads[0].output.map(row => row.slice());

  const inverseMapping = invertPermutation(swappedSequenceForPermutation.mapping);
  const unswappedOutput = inverseMapping.map(index => swappedPermutationCombined[index]);
  const permutationDiff = originalPermutationCombined.map((row, i) =>
    Math.sqrt(row.reduce((s, v, j) => s + (v - unswappedOutput[i][j]) ** 2, 0))
  );
  const maxPermutationDiff = Math.max(...permutationDiff);

  return {
    featureBase,
    workingSequence,
    originalSequence,
    heads,
    originalHeads,
    displayedHead,
    displayedHeadIndex,
    combinedOutput,
    permutationOriginalTokens: originalSequence.tokens,
    permutationSwappedTokens: swappedSequenceForPermutation.tokens,
    permutationOriginalOutput: originalPermutationCombined,
    permutationSwappedOutput: swappedPermutationCombined,
    permutationDiff,
    maxPermutationDiff
  };
}

function getModeLabel(mode) {
  return mode === "toy"
    ? "Toy 2x2 matrix example"
    : mode === "sequence"
      ? "6-day self-attention sequence"
      : "Permutation test";
}

function visibleRowIndices(model) {
  if (state.mode === "toy") return [0, 3];
  return model.workingSequence.tokens.map((_, i) => i);
}

function visibleColumnCount() {
  return state.mode === "toy" ? Math.min(state.dk, 4) : Math.min(state.dk, 6);
}

function renderMatrix(container, matrix, rowLabels, colPrefix, selectedRow, maxCols, visibleRows = null) {
  const rows = visibleRows || matrix.map((_, i) => i);
  const cols = matrix[0] ? Array.from({ length: Math.min(matrix[0].length, maxCols) }, (_, i) => i) : [];
  let html = '<table class="matrix-table"><thead><tr><th></th>';
  cols.forEach(c => {
    html += `<th>${colPrefix}${c + 1}</th>`;
  });
  if (matrix[0] && matrix[0].length > maxCols) html += "<th>…</th>";
  html += "</tr></thead><tbody>";

  rows.forEach(r => {
    html += `<tr class="${r === selectedRow ? "highlight" : ""}">`;
    html += `<th class="row-label">${rowLabels[r]}</th>`;
    cols.forEach(c => {
      html += `<td>${fmt(matrix[r][c], 2)}</td>`;
    });
    if (matrix[0] && matrix[0].length > maxCols) html += "<td>…</td>";
    html += "</tr>";
  });

  html += "</tbody></table>";
  container.innerHTML = html;
}

function renderSquareMatrix(container, matrix, rowLabels, selectedRow, visibleRows = null) {
  const rows = visibleRows || matrix.map((_, i) => i);
  let html = '<table class="matrix-table"><thead><tr><th></th>';
  rows.forEach(r => {
    html += `<th>${rowLabels[r]}</th>`;
  });
  html += "</tr></thead><tbody>";

  rows.forEach(r => {
    html += `<tr class="${r === selectedRow ? "highlight" : ""}">`;
    html += `<th class="row-label">${rowLabels[r]}</th>`;
    rows.forEach(c => {
      html += `<td>${fmt(matrix[r][c], 2)}</td>`;
    });
    html += "</tr>";
  });

  html += "</tbody></table>";
  container.innerHTML = html;
}

function tokenLabels(tokens) {
  return tokens.map(token => token.day.replace("Day ", "D"));
}

function outputIntensity(row) {
  return clamp((row[0] + row[1] + row[2]) / 2.6, 0, 1);
}

function renderMatrixPanel(model) {
  const labels = tokenLabels(model.workingSequence.tokens);
  const head = model.displayedHead;
  const selected = clamp(state.selectedToken, 0, labels.length - 1);
  const rowSubset = visibleRowIndices(model);
  const colCount = visibleColumnCount();

  renderMatrix(els.matrixQ, head.Q, labels, "d", selected, colCount, rowSubset);
  renderMatrix(els.matrixK, head.K, labels, "d", selected, colCount, rowSubset);
  renderMatrix(els.matrixV, head.V, labels, "v", selected, head.V[0].length, rowSubset);
  renderSquareMatrix(els.matrixS, head.rawScores, labels, selected, rowSubset);
  renderSquareMatrix(els.matrixScaled, head.scoreIntoSoftmax, labels, selected, rowSubset);
  renderSquareMatrix(els.matrixA, head.attention, labels, selected, rowSubset);
  renderMatrix(els.matrixO, head.output, labels, "o", selected, head.output[0].length, rowSubset);

  const bestIdx = head.attention[selected]
    .map((value, idx) => ({ value, idx }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 2);

  const selectedToken = model.workingSequence.tokens[selected];
  const bestText = bestIdx
    .map(item => `${model.workingSequence.tokens[item.idx].day} (${fmt(item.value, 2)})`)
    .join(" and ");

  const saturationNote =
    !state.useScaling && state.dk >= 10
      ? "Without scaling, the score matrix grows with dₖ, so softmax becomes peaky and one or two days dominate."
      : state.useScaling
        ? "Scaling divides the score matrix by √dₖ so the logits stay in a range where softmax can compare days more smoothly."
        : "Scaling is off, but at this dₖ and temperature the logits are not yet fully saturated.";

  els.matrixStatus.innerHTML = `
    <strong>${selectedToken.day}</strong> is the current query token.
    Its strongest links are to ${bestText}. ${saturationNote}
  `;
  els.scaledScoreTitle.innerHTML = state.useScaling
    ? 'Scores into softmax = S / √d<sub>k</sub>'
    : 'Scores into softmax = S';
}

function createTokenBox(token, index, selected, swappedIndices, outputRow) {
  const div = document.createElement("button");
  div.type = "button";
  div.className = "token-box";
  if (index === selected) div.classList.add("selected");
  if (swappedIndices.includes(index)) div.classList.add("swapped");

  div.innerHTML = `
    <div class="token-day">${token.day}</div>
    <div class="token-meta">Rain: ${token.rain} mm<br>Soil proxy: ${fmt(token.soil, 2)}</div>
    <div class="token-output">
      ${outputRow.map(value => `
        <div class="output-bar"><span style="width:${Math.round(clamp(value / 1.2, 0, 1) * 100)}%"></span></div>
      `).join("")}
    </div>
  `;

  div.addEventListener("mouseenter", () => {
    state.selectedToken = index;
    render();
  });
  div.addEventListener("click", () => {
    state.selectedToken = index;
    render();
  });

  return div;
}

function makeArcPath(x1, x2, baseY, lift) {
  const mid = (x1 + x2) / 2;
  return `M ${x1} ${baseY} Q ${mid} ${baseY - lift} ${x2} ${baseY}`;
}

function renderInteractionSVG(head, tokens, selectedIndex, headTitle, headTag) {
  const width = 760;
  const height = 290;
  const y = 200;
  const tokenWidth = 96;
  const tokenHeight = 58;
  const xPositions = tokens.map((_, i) => 74 + i * 118);
  const colors = {
    quick: "#2f5fb8",
    slow: "#1d8c77",
    blend: "#7c5cb8"
  };
  const color = colors[headTag] || colors.blend;

  const arcs = head.attention[selectedIndex].map((weight, j) => {
    const x1 = xPositions[selectedIndex];
    const x2 = xPositions[j];
    const lift = 38 + Math.abs(selectedIndex - j) * 22;
    const opacity = clamp(0.16 + 1.3 * weight, 0.12, 0.95);
    const strokeWidth = 1.5 + 14 * weight;
    return `
      <path d="${makeArcPath(x1, x2, y, lift)}"
            fill="none"
            stroke="${color}"
            stroke-opacity="${opacity}"
            stroke-width="${strokeWidth.toFixed(2)}"
            stroke-linecap="round"></path>
      <text x="${(x1 + x2) / 2}" y="${y - lift - 8}" text-anchor="middle"
            font-size="12" fill="${color}" fill-opacity="${Math.min(opacity + 0.12, 0.95)}">${fmt(weight, 2)}</text>
    `;
  }).join("");

  const boxes = tokens.map((token, i) => {
    const x = xPositions[i] - tokenWidth / 2;
    const isSelected = i === selectedIndex;
    const fill = isSelected ? "rgba(47,95,184,0.12)" : "rgba(255,255,255,0.92)";
    const stroke = isSelected ? color : "rgba(24,37,59,0.12)";
    return `
      <g class="token-node" data-index="${i}">
        <rect x="${x}" y="${y}" rx="16" ry="16" width="${tokenWidth}" height="${tokenHeight}"
              fill="${fill}" stroke="${stroke}" stroke-width="${isSelected ? 2 : 1.2}"></rect>
        <text x="${xPositions[i]}" y="${y + 20}" text-anchor="middle" font-size="13" font-weight="700" fill="#18253b">${token.day.replace("Day ", "D")}</text>
        <text x="${xPositions[i]}" y="${y + 38}" text-anchor="middle" font-size="11.5" fill="#556277">R ${token.rain} · S ${fmt(token.soil, 2)}</text>
      </g>
    `;
  }).join("");

  return `
    <div class="head-caption">
      <h3>${headTitle}</h3>
      <span class="small-pill">${headTag === "quick" ? "short-lag storm response" : headTag === "slow" ? "longer soil-memory response" : "single blended head"}</span>
    </div>
    <svg class="svg-frame" viewBox="0 0 ${width} ${height}" role="img" aria-label="${headTitle}">
      <text x="28" y="32" font-size="13" fill="#556277">Query token: ${tokens[selectedIndex].day} → attention weights over all days</text>
      ${arcs}
      ${boxes}
    </svg>
  `;
}

function bindInteractionEvents() {
  document.querySelectorAll(".token-node").forEach(node => {
    node.addEventListener("mouseenter", () => {
      state.selectedToken = Number(node.dataset.index);
      render();
    });
    node.addEventListener("click", () => {
      state.selectedToken = Number(node.dataset.index);
      render();
    });
  });
}

function renderInteractionPanel(model) {
  const selected = clamp(state.selectedToken, 0, model.workingSequence.tokens.length - 1);
  els.selectedTokenChip.textContent = `Selected query: ${model.workingSequence.tokens[selected].day}`;

  if (state.twoHeads) {
    els.interactionWrap.innerHTML = `
      ${renderInteractionSVG(model.heads[0], model.workingSequence.tokens, selected, "Head 1", "quick")}
      ${renderInteractionSVG(model.heads[1], model.workingSequence.tokens, selected, "Head 2", "slow")}
    `;
  } else {
    els.interactionWrap.innerHTML = renderInteractionSVG(
      model.heads[0],
      model.workingSequence.tokens,
      selected,
      "Single attention head",
      "blend"
    );
  }

  bindInteractionEvents();

  const selectedHead = model.displayedHead;
  const ranked = selectedHead.attention[selected]
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value);

  const top = ranked[0];
  const second = ranked[1];
  const noteParts = [];
  noteParts.push(
    `<strong>${model.workingSequence.tokens[selected].day}</strong> attends most strongly to
    <strong>${model.workingSequence.tokens[top.index].day}</strong> (${fmt(top.value, 2)}).`
  );

  if (second) {
    noteParts.push(
      `The second-strongest link is ${model.workingSequence.tokens[second.index].day} (${fmt(second.value, 2)}).`
    );
  }

  if (state.twoHeads) {
    noteParts.push(
      `In two-head mode, Head 1 emphasizes storm similarity and recent pulse behavior, while Head 2 tracks broader wetness memory.`
    );
  } else {
    noteParts.push(
      `In one-head mode, rainfall pulse and soil memory are blended into a single pattern, so the head must compromise.`
    );
  }

  els.interactionNote.innerHTML = noteParts.join(" ");
}

function renderPermutationPanel(model) {
  const selected = clamp(state.selectedToken, 0, 5);
  const originalRow = document.createElement("div");
  originalRow.className = "sequence-row";

  model.permutationOriginalTokens.forEach((token, i) => {
    originalRow.appendChild(
      createTokenBox(token, i, selected, [], model.permutationOriginalOutput[i])
    );
  });

  const swappedRow = document.createElement("div");
  swappedRow.className = "sequence-row";

  const swappedIndices = [1, 4];
  model.permutationSwappedTokens.forEach((token, i) => {
    swappedRow.appendChild(
      createTokenBox(token, i, selected, swappedIndices, model.permutationSwappedOutput[i])
    );
  });

  els.permOriginal.replaceChildren(originalRow);
  els.permSwapped.replaceChildren(swappedRow);

  const maxDiff = model.maxPermutationDiff;
  const normalizedMax = Math.max(maxDiff, 0.12);
  els.permDiffBars.innerHTML = model.permutationDiff.map((value, idx) => `
    <div class="diff-row">
      <strong>${TOKENS[idx].day.replace("Day ", "D")}</strong>
      <div class="diff-track"><span style="width:${(value / normalizedMax) * 100}%"></span></div>
      <span>${fmt(value, 3)}</span>
    </div>
  `).join("");

  if (model.maxPermutationDiff < 0.025) {
    els.permSummaryBadge.textContent = "Only a permutation";
    els.permSummaryBadge.style.background = "rgba(29, 140, 119, 0.12)";
    els.permSummaryBadge.style.color = "#1d8c77";
    els.permInterpretation.innerHTML = `
      With positional encoding off, the swapped input mostly just swaps the outputs too.
      After undoing the swap, the outputs almost line up exactly. The model has no reliable clue about true time order.
    `;
  } else {
    els.permSummaryBadge.textContent = "Outputs change meaningfully";
    els.permSummaryBadge.style.background = "rgba(211, 109, 42, 0.14)";
    els.permSummaryBadge.style.color = "#a45520";
    els.permInterpretation.innerHTML = `
      With positional encoding on, each position carries a sinusoidal signature. Swapping two days changes the encoded sequence,
      so the outputs are no longer just a simple row permutation. Order becomes recoverable.
    `;
  }
}

function drawHeatmap(model) {
  const canvas = els.peCanvas;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const rows = 6;
  const cols = state.dk;
  const leftPad = 72;
  const topPad = 34;
  const gridWidth = width - leftPad - 18;
  const gridHeight = height - topPad - 34;
  const cellW = gridWidth / cols;
  const cellH = gridHeight / rows;
  const selected = clamp(state.selectedToken, 0, rows - 1);

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const peMatrix = Array.from({ length: rows }, (_, r) => sinusoidalPosition(r, cols));
  const allValues = peMatrix.flat();
  const maxAbs = Math.max(...allValues.map(v => Math.abs(v))) || 1;

  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for (let c = 0; c < cols; c += 1) {
    ctx.fillStyle = "#556277";
    ctx.fillText(`d${c + 1}`, leftPad + c * cellW + cellW / 2, 16);
  }

  for (let r = 0; r < rows; r += 1) {
    ctx.fillStyle = "#556277";
    ctx.fillText(`Day ${r + 1}`, 34, topPad + r * cellH + cellH / 2);

    for (let c = 0; c < cols; c += 1) {
      const value = peMatrix[r][c] / maxAbs;
      const positive = value >= 0;
      const alpha = 0.12 + Math.abs(value) * 0.72;
      ctx.fillStyle = positive
        ? `rgba(47,95,184,${alpha})`
        : `rgba(211,109,42,${alpha})`;
      ctx.fillRect(leftPad + c * cellW, topPad + r * cellH, cellW - 2, cellH - 2);
    }
  }

  ctx.strokeStyle = "rgba(47,95,184,0.92)";
  ctx.lineWidth = 3;
  ctx.strokeRect(leftPad + 1, topPad + selected * cellH + 1, gridWidth - 4, cellH - 4);

  els.peRowChip.textContent = `Highlighted row: Day ${selected + 1}`;
  els.peNote.innerHTML = state.usePosition
    ? `Low-frequency columns change slowly across days and act like coarse location markers. High-frequency columns oscillate faster and separate nearby positions. The highlighted row is the positional signature currently added to the selected token before forming Q and K.`
    : `Positional encoding is currently off. The heatmap still shows the sinusoidal template that would be added to each token. Without it, content alone determines attention and time order is not explicitly encoded.`;
}

function renderTextPanels(model) {
  const selected = clamp(state.selectedToken, 0, model.workingSequence.tokens.length - 1);
  const selectedToken = model.workingSequence.tokens[selected];
  const head = model.displayedHead;
  const topLinks = head.attention[selected]
    .map((value, idx) => ({ value, idx }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 3);

  const orderRecoverable = model.maxPermutationDiff >= 0.025 ? "Yes" : "No";

  els.settingMeaning.innerHTML = `
    <p>
      <strong>${selectedToken.day}</strong> is currently asking: “Which earlier days look most useful when estimating today’s flow response?”
      The top attention targets are ${topLinks.map(item => `${model.workingSequence.tokens[item.idx].day} (${fmt(item.value, 2)})`).join(", ")}.
    </p>
    <ul>
      <li><strong>Scaling ${state.useScaling ? "on" : "off"}:</strong> ${state.useScaling ? "the logits are normalized by √dₖ so attention stays comparable as dₖ grows." : "larger dₖ increases raw score magnitude, so softmax can saturate and become overly confident."}</li>
      <li><strong>Positional encoding ${state.usePosition ? "on" : "off"}:</strong> ${state.usePosition ? "each day carries a position signature, so swapping days changes the meaning of the sequence." : "the model sees content but not absolute time order, so swapping days mostly just swaps outputs."}</li>
      <li><strong>${state.twoHeads ? "Two heads" : "One head"}:</strong> ${state.twoHeads ? "one head can lock onto short-lag storm response while the other tracks longer wetness memory." : "one blended head must express both short and long memory in a single pattern."}</li>
    </ul>
    <div class="math-card">
      Order recoverable? ${orderRecoverable}<br>
      Working mode: ${getModeLabel(state.mode)}
    </div>
  `;

  const maxLogit = Math.max(...head.scoreIntoSoftmax.flat());
  const maxWeight = Math.max(...head.attention.flat());
  els.equationFocus.innerHTML = `
    <p>The current equation in use is:</p>
    <div class="math-card">
      Attention(Q, K, V) = softmax( ${state.useScaling ? "QKᵀ / √dₖ" : "QKᵀ"} / T ) V
    </div>
    <p>
      Here, the largest score entering softmax is <strong>${fmt(maxLogit, 2)}</strong> and the largest attention
      weight after softmax is <strong>${fmt(maxWeight, 2)}</strong>. ${!state.useScaling && state.dk >= 10
        ? "That is the classic saturation story: a few large logits make softmax nearly one-hot."
        : "The logits remain in a range where several days can still share attention."}
    </p>
    <ul>
      <li><strong>Q:</strong> what the current day is searching for.</li>
      <li><strong>K:</strong> what each day offers as an address or match.</li>
      <li><strong>V:</strong> the information actually mixed together to build the output.</li>
      <li><strong>Self-attention:</strong> every token can compare itself with every other token, so the score matrix is all-to-all.</li>
    </ul>
  `;

  const prompts = [
    {
      title: "Scaling experiment",
      body: "Set dₖ to 16, turn scaling off, and lower the temperature. Ask students why one or two days suddenly dominate the softmax matrix."
    },
    {
      title: "Permutation challenge",
      body: "Turn positional encoding off, click “Swap two days,” and ask whether the model can still tell which rainfall pulse happened earlier."
    },
    {
      title: "Hydrology interpretation",
      body: state.twoHeads
        ? "Compare Head 1 and Head 2. Which one acts more like short-lag runoff response, and which one behaves more like soil-moisture memory?"
        : "Turn on two heads and ask students why one head should be free to specialize in storm response while the other handles storage memory."
    }
  ];

  els.classPrompts.innerHTML = prompts.map(prompt => `
    <article class="prompt-card">
      <h3>${prompt.title}</h3>
      <p>${prompt.body}</p>
    </article>
  `).join("");
}

function renderMetrics(model) {
  const head = model.displayedHead;
  const flattenedOutput = model.combinedOutput.flat();
  const varianceValue = variance(flattenedOutput);
  const maxScore = Math.max(...head.scoreIntoSoftmax.flat());
  const maxWeight = Math.max(...head.attention.flat());
  const orderRecoverable = model.maxPermutationDiff >= 0.025 ? "Yes" : "No";

  els.metricDk.textContent = String(state.dk);
  els.metricScore.textContent = fmt(maxScore, 2);
  els.metricWeight.textContent = fmt(maxWeight, 2);
  els.metricVariance.textContent = fmt(varianceValue, 3);
  els.metricOrder.textContent = orderRecoverable;
  els.metricMode.textContent = getModeLabel(state.mode);
}

function updateControlLabels() {
  els.scalingToggle.checked = state.useScaling;
  els.positionToggle.checked = state.usePosition;
  els.headToggle.setAttribute("aria-pressed", state.twoHeads ? "true" : "false");
  els.headToggleText.textContent = state.twoHeads ? "Show 2 heads" : "Show 1 head";
  els.swapBtn.setAttribute("aria-pressed", state.swapped ? "true" : "false");
  els.swapBtnText.textContent = state.swapped ? "Days 2 and 5 swapped" : "Swap two days";
  els.dkSlider.value = String(state.dk);
  els.dkValue.textContent = String(state.dk);
  els.tempSlider.value = fmt(state.temperature, 2);
  els.tempValue.textContent = fmt(state.temperature, 2);
  els.modeSelect.value = state.mode;

  const headButtons = els.headTabs.querySelectorAll(".mini-toggle");
  headButtons.forEach(button => {
    const headIndex = Number(button.dataset.head);
    const shouldBeVisible = state.twoHeads || headIndex === 0;
    button.style.display = shouldBeVisible ? "inline-flex" : "none";
    button.classList.toggle("active", headIndex === state.activeHead);
  });
}

function emphasizeModePanels() {
  [els.panelA, els.panelB, els.panelC, els.panelD].forEach(panel => panel.classList.remove("mode-emphasis"));
  if (state.mode === "toy") {
    els.panelA.classList.add("mode-emphasis");
    els.panelD.classList.add("mode-emphasis");
  } else if (state.mode === "sequence") {
    els.panelA.classList.add("mode-emphasis");
    els.panelB.classList.add("mode-emphasis");
  } else {
    els.panelC.classList.add("mode-emphasis");
    els.panelD.classList.add("mode-emphasis");
  }
}

function render() {
  updateControlLabels();
  const model = buildModel(state);

  state.activeHead = clamp(state.activeHead, 0, model.heads.length - 1);
  state.selectedToken = clamp(state.selectedToken, 0, model.workingSequence.tokens.length - 1);

  renderMetrics(model);
  renderMatrixPanel(model);
  renderInteractionPanel(model);
  renderPermutationPanel(model);
  drawHeatmap(model);
  renderTextPanels(model);
  emphasizeModePanels();
}

function bindEvents() {
  els.resetBtn.addEventListener("click", () => {
    Object.assign(state, DEFAULT_STATE);
    render();
  });

  els.scalingToggle.addEventListener("change", e => {
    state.useScaling = e.target.checked;
    render();
  });

  els.positionToggle.addEventListener("change", e => {
    state.usePosition = e.target.checked;
    render();
  });

  els.headToggle.addEventListener("click", () => {
    state.twoHeads = !state.twoHeads;
    state.activeHead = 0;
    render();
  });

  els.swapBtn.addEventListener("click", () => {
    state.swapped = !state.swapped;
    render();
  });

  els.dkSlider.addEventListener("input", e => {
    state.dk = Number(e.target.value);
    render();
  });

  els.tempSlider.addEventListener("input", e => {
    state.temperature = Number(e.target.value);
    render();
  });

  els.modeSelect.addEventListener("change", e => {
    state.mode = e.target.value;
    render();
  });

  els.headTabs.querySelectorAll(".mini-toggle").forEach(button => {
    button.addEventListener("click", () => {
      state.activeHead = Number(button.dataset.head);
      render();
    });
  });
}

bindEvents();
render();
