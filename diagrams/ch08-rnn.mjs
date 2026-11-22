// Slide diagrams for Chapter 8 (Recurrent Neural Networks).
// All 41 diagrams use id prefix `ch08-rnn-<concept>` and are verified
// for zero visual defects by tools/build_module_diagrams.mjs.

import { C, TOKENS, grid, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// ══════════════════════════════════════════════════════════════════════
// §8.0 index.md (4 diagrams)
// ══════════════════════════════════════════════════════════════════════

// 1. Fixed-Shape i.i.d. vs. Bounded-State Causal Sequence Model
function fixedVsSeq() {
  const W = 540, H = 264;
  let o = '';
  // Top panel: Fixed-shape i.i.d.
  o += box(16, 12, 508, 104, D.red.tint, D.red.base, 8, 2);
  o += tx(270, 32, '(a) Fixed-Shape Feedforward / CNN (Assumes Fixed Width & i.i.d.)', { fs: 14, fw: 700, fill: D.red.dark });
  o += box(30, 52, 124, 48, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(92, 68, 'Fixed Input x', { fs: 13, fw: 700, fill: C.ink });
  o += tx(92, 86, 'shape (d,)', { mono: true, fs: 12, fill: C.muted });
  o += arrow(160, 76, 198, 76, D.red.base);
  o += box(204, 52, 134, 48, '#FFFFFF', C.gray, 6, 1.8);
  o += tx(271, 68, 'Static Layer f(x)', { fs: 13, fw: 700, fill: C.ink });
  o += tx(271, 86, 'no memory across t', { fs: 11.5, fill: C.muted });
  o += arrow(344, 76, 378, 76, D.red.base);
  o += box(384, 52, 128, 48, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(448, 68, '1024²⁰ contexts', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(448, 86, '87% unseen 3-gram', { fs: 11, fill: D.red.dark });

  // Bottom panel: Bounded-state sequence model
  o += box(16, 132, 508, 118, C.lblue, C.blue, 8, 2);
  o += tx(270, 152, '(b) Latent Autoregressive Sequence Model (O(1) State Memory)', { fs: 14, fw: 700, fill: D.blue.dark });
  o += box(30, 174, 132, 56, '#FFFFFF', C.blue, 6, 1.8);
  o += tx(96, 193, 'Prefix x₁ … xₜ₋₁', { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(96, 213, 'arbitrary length t−1', { fs: 11.5, fill: C.muted });
  o += arrow(168, 202, 204, 202, C.blue);
  o += box(210, 174, 138, 56, C.lgreen, C.green, 6, 2);
  o += tx(279, 193, 'Hidden State hₜ₋₁', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(279, 213, '512 B · shared W', { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += arrow(354, 202, 386, 202, C.green);
  o += box(392, 174, 118, 56, '#FFFFFF', C.purple, 6, 1.8);
  o += tx(451, 193, 'P(xₜ | hₜ₋₁)', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(451, 213, 'score & generate', { fs: 11.5, fill: C.muted });
  return svg(W, H, o);
}

// 2. Chapter 8 6-Section Roadmap
function roadmap() {
  const W = 540, H = 268;
  let o = '';
  const bw = 152, bh = 86;
  const xs = [16, 194, 372];
  const ty = 20, by = 156;
  const items = [
    { x: xs[0], y: ty, s: '§8.1 Sequences', t: 'Autoregression', d: 'Window m vs. state hₜ', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: xs[1], y: ty, s: '§8.2 Text → Tokens', t: 'Byte-Level BPE', d: 'UTF-8 + regex merges', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: xs[2], y: ty, s: '§8.3 Lang Models', t: 'n-Grams & bpb', d: 'Kneser–Ney · PPL · bpb', f: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: xs[2], y: by, s: '§8.4 RNN LMs', t: 'Recurrent Cell', d: 'tanh state · clip θ=1', f: C.lgreen, st: C.green, dk: D.green.dark },
    { x: xs[1], y: by, s: '§8.5 BPTT', t: 'Time Gradients', d: '(W_hh^T)^k · TBPTT', f: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: xs[0], y: by, s: '§8.6 Decoding', t: 'Text Generation', d: 'Beam w · top-p · min-p', f: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  o += arrow(xs[0] + bw + 3, ty + bh / 2, xs[1] - 3, ty + bh / 2, C.blue);
  o += arrow(xs[1] + bw + 3, ty + bh / 2, xs[2] - 3, ty + bh / 2, C.amber);
  o += arrow(xs[2] + bw / 2, ty + bh + 4, xs[2] + bw / 2, by - 4, C.green);
  o += arrow(xs[2] - 3, by + bh / 2, xs[1] + bw + 3, by + bh / 2, D.red.base);
  o += arrow(xs[1] - 3, by + bh / 2, xs[0] + bw + 3, by + bh / 2, C.purple);

  items.forEach((it) => {
    o += box(it.x, it.y, bw, bh, it.f, it.st, 8, 2);
    o += tx(it.x + bw / 2, it.y + 20, it.s, { mono: true, fs: 12.5, fw: 700, fill: it.dk });
    o += tx(it.x + bw / 2, it.y + 44, it.t, { fs: 15, fw: 700, fill: C.ink });
    o += tx(it.x + bw / 2, it.y + 66, it.d, { fs: 12, fw: 600, fill: C.muted });
  });
  return svg(W, H, o);
}

// 3. Quantitative Benchmarks on Wells' The Time Machine
function wellsBenchmarks() {
  const W = 540, H = 276;
  let o = '';
  o += box(16, 12, 508, 36, D.purple.tint, C.purple, 7, 1.8);
  o += tx(270, 30, 'The Time Machine Corpus: 179,462 UTF-8 Bytes · 65,741 BPE-1024 Tokens', { fs: 13.5, fw: 700, fill: D.purple.dark });

  const cards = [
    { x: 16,  y: 60, s: '§8.1 Sine Wave AR(4)', m: '0.05 → 0.25 MSE', d: '1-step vs 64-step rollout', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 276, y: 60, s: '§8.2 Pre-Tokenized BPE', m: '2.78 bytes / token', d: '0 OOVs · matches tiktoken', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 16,  y: 132, s: '§8.3 Kneser–Ney 3-Gram', m: '1.90 bpb (BPE-1024 KN)', d: 'Word PPL 2,481 → 300', f: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 276, y: 132, s: '§8.4 Clipped RNNLM', m: '2.12 bpb · 512 B state', d: '89% params in E & W_hq', f: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 16,  y: 204, s: '§8.5 BPTT Decay Probe', m: '< 1% norm by lag 8', d: 'ρ=1.80 × tanh\' 0.27 ≈ 0.49', f: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 276, y: 204, s: '§8.6 Truncated Samplers', m: 'distinct-3 ≈ 0.99', d: 'min-p / top-p break loops', f: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  cards.forEach((c) => {
    o += box(c.x, c.y, 248, 60, c.f, c.st, 7, 1.8);
    o += tx(c.x + 124, c.y + 16, c.s, { fs: 12.5, fw: 700, fill: c.dk });
    o += tx(c.x + 124, c.y + 34, c.m, { mono: true, fs: 13.5, fw: 700, fill: C.ink });
    o += tx(c.x + 124, c.y + 50, c.d, { fs: 11.5, fill: C.muted });
  });
  return svg(W, H, o);
}

// 4. End-to-End Sequence Pipeline Summary
function pipelineSummary() {
  const W = 540, H = 272;
  let o = '';
  const stages = [
    { y: 14,  t: '1. Byte-Level BPE (§8.2)', eq: 'UTF-8 bytes → regex chunks → rank-ordered merges (|V| = 1,027)', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 66,  t: '2. Chain-Rule Factorization (§§8.1, 8.3)', eq: 'P(x_{1:T}) = ∏ P(xₜ | x_{<t})   ·   bpb = −(1/B) ∑ log₂ P(xₜ | x_{<t})', f: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 118, t: '3. Shared-Weight Recurrent Cell (§8.4)', eq: 'Hₜ = tanh(Xₜ W_xh + Hₜ₋₁ W_hh + b_h)   ·   Oₜ = Hₜ W_hq + b_q', f: C.lgreen, st: C.green, dk: D.green.dark },
    { y: 170, t: '4. Truncated BPTT + Clipping (§8.5)', eq: 'Detach across T₀ = 32 windows   ·   g ← min(1, θ / ||g||) g', f: D.red.tint, st: D.red.base, dk: D.red.dark },
    { y: 222, t: '5. Autoregressive Decoding (§8.6)', eq: 'Beam search (constrained)  or  temperature τ + top-p / min-p', f: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  stages.forEach((s, i) => {
    o += box(20, s.y, 500, 40, s.f, s.st, 7, 1.8);
    o += tx(270, s.y + 13, s.t, { fs: 13, fw: 700, fill: s.dk });
    o += tx(270, s.y + 29, s.eq, { mono: true, fs: 11.5, fw: 600, fill: C.ink });
    if (i < stages.length - 1) {
      o += arrow(270, s.y + 41, 270, stages[i + 1].y - 1, s.st);
    }
  });
  return svg(W, H, o);
}

// ══════════════════════════════════════════════════════════════════════
// §8.1 sequence.md (8 diagrams)
// ══════════════════════════════════════════════════════════════════════

// 5. i.i.d. Tabular Data vs. Temporal Sequence Data
function iidVsTemporal() {
  const W = 540, H = 258;
  let o = '';
  // Top: i.i.d.
  o += box(16, 14, 508, 98, N.panel, C.gray, 8, 1.8);
  o += tx(270, 32, '(a) i.i.d. Tabular / Image Examples: Exchangeable Order, Fixed Width d', { fs: 14, fw: 700, fill: C.ink });
  const exs = ['Ex 1: x⁽¹⁾ ∈ ℝᵈ', 'Ex 2: x⁽²⁾ ∈ ℝᵈ', 'Ex 3: x⁽³⁾ ∈ ℝᵈ'];
  exs.forEach((lab, i) => {
    const x = 34 + i * 164;
    o += box(x, 50, 144, 44, '#FFFFFF', C.gray, 6, 1.6);
    o += tx(x + 72, 66, lab, { mono: true, fs: 12.5, fw: 700, fill: C.ink });
    o += tx(x + 72, 83, 'independent draw', { fs: 11.5, fill: C.muted });
  });

  // Bottom: Sequential
  o += box(16, 128, 508, 116, C.lblue, C.blue, 8, 2);
  o += tx(270, 148, '(b) Sequential Data: Strict Causal Order & Growing Prefix x_{<t}', { fs: 14, fw: 700, fill: D.blue.dark });
  const toks = ['"Thank"', '"you"', '"for"', '"your"', '"help"'];
  toks.forEach((w, i) => {
    const x = 28 + i * 100;
    const isTarget = i === 4;
    o += box(x, 168, 82, 40, isTarget ? C.lgreen : '#FFFFFF', isTarget ? C.green : C.blue, 6, 1.8);
    o += tx(x + 41, 188, w, { mono: true, fs: 13.5, fw: 700, fill: isTarget ? D.green.dark : C.ink });
    if (i < 4) o += arrow(x + 84, 188, x + 98, 188, C.blue);
  });
  o += tx(270, 226, 'Permuting order ("your thank for you") destroys predictive meaning!', { fs: 12.5, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

// 6. Taxonomy of Sequence Learning Regimes
function taskTaxonomy() {
  const W = 540, H = 266;
  let o = '';
  const panels = [
    { x: 16,  y: 14,  t: '1. Sequence-to-Label', sub: '(x₁, …, x_T) → class y', ex: 'Sentiment · ECG classification', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 276, y: 14,  t: '2. Fixed-to-Sequence', sub: 'image x → (y₁, …, y_T)', ex: 'Image captioning · synthesis', f: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 16,  y: 142, t: '3. Sequence-to-Sequence', sub: '(x_{1:T}) → (y_{1:T\'})  (T ≠ T\')', ex: 'Translation · ASR transcription', f: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 276, y: 142, t: '4. Density Estimation (LM)', sub: 'x_{1:T} → ∏ P(xₜ | x_{<t})', ex: '0 external labels · score & sample', f: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  panels.forEach((p) => {
    o += box(p.x, p.y, 248, 112, p.f, p.st, 8, 2);
    o += tx(p.x + 124, p.y + 24, p.t, { fs: 14.5, fw: 700, fill: p.dk });
    o += box(p.x + 12, p.y + 40, 224, 34, '#FFFFFF', p.st, 6, 1.5);
    o += tx(p.x + 124, p.y + 57, p.sub, { mono: true, fs: 12, fw: 700, fill: C.ink });
    o += tx(p.x + 124, p.y + 92, p.ex, { fs: 12, fw: 600, fill: C.ink });
  });
  return svg(W, H, o);
}

// 7. Growing Prefixes Funneling into Fixed-Size Predictor
function growingPrefix() {
  const W = 540, H = 256;
  let o = '';
  const rows = [
    { y: 24,  step: 't = 2', pre: ['x₁'], target: 'x̂₂' },
    { y: 92,  step: 't = 3', pre: ['x₁', 'x₂'], target: 'x̂₃' },
    { y: 160, step: 't = 4', pre: ['x₁', 'x₂', 'x₃'], target: 'x̂₄' },
  ];
  rows.forEach((r) => {
    o += tx(38, r.y + 20, r.step, { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
    o += grid([r.pre], 72, r.y, 38, 4, { fs: 13.5, fill: () => [C.lblue, C.blue, false, C.ink] });
    const endX = 72 + r.pre.length * 42;
    o += arrow(endX + 4, r.y + 19, 232, 112, C.blue);
  });

  // Fixed-size bottleneck box
  o += box(238, 60, 148, 104, C.lamber, C.amber, 8, 2);
  o += tx(312, 86, 'History Compressor', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(312, 110, '1. Window x_{t−m:t−1}', { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += tx(312, 130, '2. Latent state hₜ₋₁', { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  o += tx(312, 150, '(Fixed Width!)', { fs: 12, fw: 700, fill: D.orange.dark });

  o += arrow(392, 112, 424, 112, C.green);
  o += box(430, 74, 94, 76, C.lgreen, C.green, 8, 2);
  o += tx(477, 100, 'Predictor f', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(477, 124, 'E[xₜ | x_{<t}]', { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += tx(270, 236, 'Prefix x_{<t} grows with t, so every model must compress it to fixed size.', { fs: 13, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 8. Chain Rule Factorization & Log-Domain Sum
function chainRule() {
  const W = 540, H = 254;
  let o = '';
  const cards = [
    { x: 18,  title: 'P("the")', val: '0.05', logv: '−3.00 nats', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, title: 'P("cat" | "the")', val: '0.01', logv: '−4.61 nats', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 370, title: 'P("sat"|"the","cat")', val: '0.20', logv: '−1.61 nats', f: C.lblue, st: C.blue, dk: D.blue.dark },
  ];
  cards.forEach((c, i) => {
    o += box(c.x, 20, 152, 86, c.f, c.st, 8, 2);
    o += tx(c.x + 76, 42, c.title, { mono: true, fs: 11.5, fw: 700, fill: c.dk });
    o += tx(c.x + 76, 66, `p = ${c.val}`, { mono: true, fs: 14.5, fw: 700, fill: C.ink });
    o += tx(c.x + 76, 88, `log p = ${c.logv}`, { mono: true, fs: 12, fill: C.muted });
    if (i < 2) o += tx(c.x + 164, 63, '×', { fs: 22, fw: 700, fill: C.ink });
  });

  // Direct product vs Log sum
  o += box(18, 128, 238, 106, D.red.tint, D.red.base, 8, 2);
  o += tx(137, 150, 'Direct Product ∏ P(xₜ | x_{<t})', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(137, 178, '0.05 × 0.01 × 0.20 = 10⁻⁴', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(137, 206, 'Underflows float32 to 0.0 at T > 100!', { fs: 12, fw: 600, fill: D.red.dark });

  o += box(284, 128, 238, 106, C.lgreen, C.green, 8, 2);
  o += tx(403, 150, 'Log-Domain Sum ∑ log P(xₜ | x_{<t})', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(403, 178, '−3.00 − 4.61 − 1.61 = −9.21', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(403, 206, 'Numerically stable at any length T', { fs: 12, fw: 600, fill: D.green.dark });
  return svg(W, H, o);
}

// 9. Markov Condition: m-Step Memory Horizon
function markovHorizon() {
  const W = 540, H = 246;
  let o = '';
  o += tx(270, 22, 'm-th Order Markov Condition (m = 3): Future ⊥ Distant Past | Last m Steps', { fs: 13.5, fw: 700, fill: C.ink });

  // Distant past (discarded)
  o += box(16, 46, 170, 88, N.panel, C.gray, 8, 1.8, true);
  o += tx(101, 66, 'Distant Past (Ignored)', { fs: 12.5, fw: 700, fill: C.muted });
  o += grid([['x₁', '…', 'xₜ₋₄']], 26, 80, 46, 6, {
    fs: 13,
    fill: () => ['#FFFFFF', C.gray, true, C.muted],
  });

  // Active Markov window
  o += box(200, 46, 198, 88, C.lblue, C.blue, 8, 2);
  o += tx(299, 66, 'Active Markov Window (m = 3)', { fs: 12.5, fw: 700, fill: D.blue.dark });
  o += grid([['xₜ₋₃', 'xₜ₋₂', 'xₜ₋₁']], 220, 80, 48, 7, {
    fs: 13,
    fill: () => ['#FFFFFF', C.blue, false, D.blue.dark],
  });

  o += arrow(404, 98, 436, 98, C.green);
  o += box(442, 46, 82, 88, C.lgreen, C.green, 8, 2);
  o += tx(483, 66, 'Target', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += grid([['xₜ']], 459, 80, 48, 0, {
    fs: 14,
    fill: () => ['#FFFFFF', C.green, false, D.green.dark],
  });

  o += box(28, 156, 484, 72, C.lamber, C.amber, 8, 1.8);
  o += tx(270, 178, 'P(xₜ | x₁, …, xₜ₋₁)  =  P(xₜ | xₜ₋ₘ, …, xₜ₋₁)', { mono: true, fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(270, 204, 'Tractable fixed-width context, but blind to dependencies > m steps back.', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 10. Stationarity: Sliding One Shared Predictor Across Time
function stationarityPooling() {
  const W = 540, H = 252;
  let o = '';
  o += tx(270, 20, 'Time-Homogeneous (Stationary) Transition Law P(xₜ | x_{t−m:t−1})', { fs: 14, fw: 700, fill: D.blue.dark });

  const s = 44, g = 5, x0 = 26, y0 = 42;
  const cells = ['x₁', 'x₂', 'x₃', 'x₄', 'x₅', '…', 'x₉₉₇', 'x₉₉₈', 'x₉₉₉', 'x₁₀₀₀'];
  o += grid([cells], x0, y0, s, g, {
    fs: 12,
    fill: (r, c) => c === 5 ? [N.panel, C.gray, true, C.muted] : [C.lblue, C.blue, false, C.ink],
  });

  // Two sliding windows converging on the same shared model f_theta
  o += box(x0, y0 + s + 10, 4 * s + 3 * g, 26, D.purple.tint, C.purple, 5, 1.6);
  o += tx(x0 + (4 * s + 3 * g) / 2, y0 + s + 23, 'Early window → target x₅', { fs: 11.5, fw: 700, fill: D.purple.dark });

  const rx = x0 + 6 * (s + g);
  o += box(rx, y0 + s + 10, 4 * s + 3 * g, 26, D.purple.tint, C.purple, 5, 1.6);
  o += tx(rx + (4 * s + 3 * g) / 2, y0 + s + 23, 'Late window → target x₁₀₀₀', { fs: 11.5, fw: 700, fill: D.purple.dark });

  o += arrow(120, y0 + s + 40, 210, 174, C.purple);
  o += arrow(420, y0 + s + 40, 330, 174, C.purple);

  o += box(60, 178, 420, 58, C.lgreen, C.green, 8, 2);
  o += tx(270, 198, 'One Shared Predictor f_θ (Pooled Dataset)', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(270, 220, 'T − m = 996 training pairs share identical dynamics', { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 11. Sliding Lag Feature Matrix (996 x 4) and Train/Val Split
function lagMatrix() {
  const W = 540, H = 258;
  let o = '';
  o += tx(270, 20, 'Supervised Lag Matrix Construction (T = 1000, Window m = 4 → 996 Rows)', { fs: 13.5, fw: 700, fill: C.ink });

  // Feature matrix X (996 x 4)
  o += box(18, 38, 286, 206, C.lblue, C.blue, 8, 2);
  o += tx(161, 56, 'Lag Feature Matrix X ∈ ℝ⁹⁹⁶ˣ⁴', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  const rows = [
    ['x₁', 'x₂', 'x₃', 'x₄'],
    ['x₂', 'x₃', 'x₄', 'x₅'],
    ['x₆₀₀', 'x₆₀₁', 'x₆₀₂', 'x₆₀₃'],
    ['x₆₀₁', 'x₆₀₂', 'x₆₀₃', 'x₆₀₄'],
  ];
  o += grid(rows, 76, 72, 36, 4, {
    fs: 11.5,
    fill: (r) => r < 3 ? ['#FFFFFF', C.blue, false, C.ink] : [C.lamber, C.amber, false, D.orange.dark],
  });

  o += arrow(310, 148, 344, 148, C.purple);

  // Target column y (996 x 1)
  o += box(350, 38, 172, 206, C.lgreen, C.green, 8, 2);
  o += tx(436, 56, 'Targets y ∈ ℝ⁹⁹⁶ˣ¹', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  const targs = [['x₅'], ['x₆'], ['x₆₀₄'], ['x₆₀₅']];
  o += grid(targs, 376, 72, 36, 4, {
    fs: 11.5,
    fill: (r) => r < 3 ? ['#FFFFFF', C.green, false, D.green.dark] : [C.lamber, C.amber, false, D.orange.dark],
  });
  o += tx(464, 108, 'Train', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(464, 128, '1..600', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(464, 196, 'Val', { fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(464, 214, '601..996', { mono: true, fs: 11.5, fill: D.orange.dark });
  return svg(W, H, o);
}

// 12. Noiseless Oscillator vs. Noisy Moving-Average Fit
function averagingVsOscillator() {
  const W = 540, H = 266;
  let o = '';
  // Left panel: True 2-lag oscillator
  o += box(16, 14, 246, 236, C.lblue, C.blue, 8, 2);
  o += tx(139, 36, 'Exact 2-Lag Oscillator', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(139, 58, 'xₜ = 2cos(0.01)xₜ₋₁ − xₜ₋₂', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += box(28, 76, 222, 54, '#FFFFFF', C.blue, 6, 1.6);
  o += tx(139, 95, 'Weights: w ≈ (−1.00, +2.00)', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(139, 116, 'Captures true wave curvature', { fs: 11.5, fill: C.muted });
  o += box(28, 144, 222, 88, D.red.tint, D.red.base, 6, 1.6);
  o += tx(139, 166, 'Under Noise σ² = 0.04:', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(139, 188, 'Var = 5σ² = 0.20 (5× worse)', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(139, 210, 'Rejected by 1-step MSE!', { fs: 12, fw: 700, fill: D.red.dark });

  // Right panel: Learned 4-lag moving average
  o += box(278, 14, 246, 236, C.lamber, C.amber, 8, 2);
  o += tx(401, 36, 'Learned 4-Lag Moving Avg', { fs: 14.5, fw: 700, fill: D.orange.dark });
  o += tx(401, 58, 'x̂ₜ = 0.25(xₜ₋₄ + … + xₜ₋₁)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += box(290, 76, 222, 54, C.lgreen, C.green, 6, 1.6);
  o += tx(401, 95, '1-Step Noise Var = σ²/4 = 0.01', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(401, 116, 'Total 1-step MSE ≈ 0.05 ★', { fs: 12, fw: 600, fill: D.green.dark });
  o += box(290, 144, 222, 88, '#FFFFFF', C.amber, 6, 1.6);
  o += tx(401, 166, 'During Multistep Rollout:', { fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(401, 188, 'No curvature → flat decay to', { fs: 12, fill: C.ink });
  o += tx(401, 210, 'x* = b / (1 − ∑wᵢ)', { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

// ══════════════════════════════════════════════════════════════════════
// §8.2 text-sequence.md (5 diagrams)
// ══════════════════════════════════════════════════════════════════════

// 13. How Tokenizer Granularity Controls Model Footprint & Horizon
function tokenizerImpact() {
  const W = 540, H = 264;
  let o = '';
  o += box(170, 14, 200, 52, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 32, 'Tokenizer Choice', { fs: 15, fw: 700, fill: D.purple.dark });
  o += tx(270, 52, 'Vocab |V| & Segmentation', { mono: true, fs: 12.5, fw: 600, fill: C.ink });

  const cols = [
    { x: 16,  t: '1. Embedding E', eq: '|V| × d params', sub: '65,728 weights (29.5%)', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 192, t: '2. Sequence Length T', eq: '173k → 65k → 32k', sub: 'Steps per document', f: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 368, t: '3. Softmax Head W_hq', eq: 'h × |V| params', sub: '132,483 weights (59.4%)', f: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  cols.forEach((c) => {
    o += arrow(270, 68, c.x + 78, 108, C.purple);
    o += box(c.x, 112, 156, 134, c.f, c.st, 8, 2);
    o += tx(c.x + 78, 138, c.t, { fs: 13.5, fw: 700, fill: c.dk });
    o += box(c.x + 10, 156, 136, 36, '#FFFFFF', c.st, 6, 1.5);
    o += tx(c.x + 78, 174, c.eq, { mono: true, fs: 12, fw: 700, fill: C.ink });
    o += tx(c.x + 78, 218, c.sub, { fs: 12, fw: 600, fill: C.ink });
  });
  return svg(W, H, o);
}

// 14. Classical Vocab Frequency Table & <unk> Collapse
function vocabUnk() {
  const W = 540, H = 258;
  let o = '';
  // Left: Vocab table
  o += box(16, 14, 246, 228, C.lblue, C.blue, 8, 2);
  o += tx(139, 36, 'Classical Vocab (Sorted by Freq)', { fs: 13.5, fw: 700, fill: D.blue.dark });
  const entries = [
    ['idx 0', '<unk>', 'reserved fallback'],
    ['idx 1', '"the"', '2,261 occurrences'],
    ['idx 2', '"i"',   '1,267 occurrences'],
    ['idx 3', '"and"', '1,245 occurrences'],
  ];
  entries.forEach((e, i) => {
    const y = 54 + i * 44;
    const isUnk = i === 0;
    o += box(28, y, 222, 36, isUnk ? D.red.tint : '#FFFFFF', isUnk ? D.red.base : C.blue, 5, 1.5);
    o += tx(139, y + 18, `${e[0]}: ${e[1]} (${e[2]})`, { mono: true, fs: 11.5, fw: 700, fill: isUnk ? D.red.dark : C.ink });
  });

  // Right: Unseen / rare words collapsing into idx 0
  o += box(284, 14, 240, 228, D.red.tint, D.red.base, 8, 2);
  o += tx(404, 36, 'The <unk> Information Collapse', { fs: 13.5, fw: 700, fill: D.red.dark });
  const unseen = ['"chronometer" (OOV)', '"recondite" (< min_freq)', '"traveller\'s" (OOV)'];
  unseen.forEach((w, i) => {
    const y = 58 + i * 38;
    o += box(302, y, 204, 30, '#FFFFFF', D.red.base, 5, 1.5);
    o += tx(404, y + 15, w, { mono: true, fs: 12, fw: 600, fill: C.ink });
  });
  o += arrow(404, 174, 404, 196, D.red.base);
  o += box(302, 200, 204, 32, '#FFFFFF', D.red.base, 6, 2);
  o += tx(404, 216, 'All map to idx 0 (<unk>)!', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

// 15. UTF-8 Variable-Width Byte Decomposition
function utf8Bytes() {
  const W = 540, H = 262;
  let o = '';
  o += tx(270, 20, 'UTF-8 Encodes All ~150,000 Unicode Characters into 256 Base Bytes (0..255)', { fs: 13.5, fw: 700, fill: C.ink });

  const examples = [
    { y: 40,  ch: "'a' (ASCII)",   bytes: ['97'], note: '1 byte / char', f: C.lblue, st: C.blue },
    { y: 90,  ch: "'é' (Latin/Greek)", bytes: ['195', '169'], note: '2 bytes / char', f: C.lgreen, st: C.green },
    { y: 140, ch: 'U+4F60 (CJK)',    bytes: ['228', '189', '160'], note: '3 bytes / char', f: C.lamber, st: C.amber },
    { y: 190, ch: 'U+1F680 (Emoji)', bytes: ['240', '159', '154', '128'], note: '4 bytes / char', f: D.purple.tint, st: C.purple },
  ];
  examples.forEach((ex) => {
    o += box(18, ex.y, 138, 38, '#FFFFFF', ex.st, 6, 1.8);
    o += tx(87, ex.y + 19, ex.ch, { mono: true, fs: 12.5, fw: 700, fill: C.ink });
    o += arrow(162, ex.y + 19, 190, ex.y + 19, ex.st);
    o += grid([ex.bytes], 196, ex.y, 44, 5, {
      fs: 12.5,
      fill: () => [ex.f, ex.st, false, C.ink],
    });
    o += tx(410, ex.y + 19, ex.note, { fs: 12.5, fw: 700, fill: C.ink, anchor: 'start' });
  });
  o += tx(270, 246, 'Zero out-of-vocabulary tokens on any text, code, or language!', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 16. BPE Iterative Greedy Merge Loop
function bpeGreedyLoop() {
  const W = 540, H = 268;
  let o = '';
  const steps = [
    { y: 14,  step: 'Step 0 (Init)', rule: '256 raw UTF-8 byte tokens (IDs 0..255)', len: 'len = 32', f: N.panel, st: C.gray, dk: C.ink },
    { y: 76,  step: 'Merge 1 (ID 256)', rule: "('u', 'g') → 'ug'   (count = 6)", len: 'len = 26', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 138, step: 'Merge 2 (ID 257)', rule: "('h', 'ug') → 'hug'  (count = 4)", len: 'len = 22', f: C.lgreen, st: C.green, dk: D.green.dark },
    { y: 200, step: 'Merge 3 (ID 258)', rule: "('hug', ' ') → 'hug ' (count = 3)", len: 'len = 19', f: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  steps.forEach((s, i) => {
    o += box(20, s.y, 376, 46, s.f, s.st, 7, 1.8);
    o += tx(208, s.y + 15, s.step, { fs: 13, fw: 700, fill: s.dk });
    o += tx(208, s.y + 33, s.rule, { mono: true, fs: 12.5, fw: 600, fill: C.ink });
    o += box(408, s.y, 112, 46, '#FFFFFF', s.st, 7, 1.8);
    o += tx(464, s.y + 23, s.len, { mono: true, fs: 13.5, fw: 700, fill: s.dk });
    if (i < steps.length - 1) {
      o += arrow(208, s.y + 47, 208, steps[i + 1].y - 1, s.st);
    }
  });
  return svg(W, H, o);
}

// 17. Special Tokens Isolation & Prompt-Injection Defense
function specialTokenSafety() {
  const W = 540, H = 260;
  let o = '';
  o += tx(270, 20, 'Token ID Partitioning & Prompt-Injection Guard (allow_special)', { fs: 13.5, fw: 700, fill: C.ink });

  // ID range bar
  o += box(18, 40, 148, 56, C.lblue, C.blue, 7, 2);
  o += tx(92, 60, 'IDs 0 .. 255', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(92, 80, 'Raw UTF-8 Bytes', { fs: 12, fill: C.ink });

  o += box(174, 40, 178, 56, C.lgreen, C.green, 7, 2);
  o += tx(263, 60, 'IDs 256 .. 1023', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(263, 80, '768 BPE Merges', { fs: 12, fill: C.ink });

  o += box(360, 40, 162, 56, D.purple.tint, C.purple, 7, 2);
  o += tx(441, 60, 'IDs 1024 .. 1026', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(441, 80, '<pad> <bos> <eos>', { mono: true, fs: 12, fw: 700, fill: D.purple.dark });

  // Untrusted vs Trusted encoding of "<bos>"
  o += box(18, 116, 244, 126, C.lamber, C.amber, 8, 2);
  o += tx(140, 138, 'Untrusted User Text', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(140, 158, 'encode("<bos>", False)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += box(32, 174, 216, 52, '#FFFFFF', C.amber, 6, 1.5);
  o += tx(140, 192, '[60, 98, 111, 115, 62]', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(140, 212, '5 literal ASCII bytes (Safe!)', { fs: 11.5, fill: C.ink });

  o += box(278, 116, 244, 126, D.purple.tint, C.purple, 8, 2);
  o += tx(400, 138, 'System Control Code', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(400, 158, 'encode("<bos>", True)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += box(292, 174, 216, 52, '#FFFFFF', C.purple, 6, 1.5);
  o += tx(400, 192, '[1025]', { mono: true, fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(400, 212, 'Dedicated <bos> control ID', { fs: 11.5, fill: C.ink });
  return svg(W, H, o);
}

// ══════════════════════════════════════════════════════════════════════
// §8.3 language-model.md (5 diagrams)
// ══════════════════════════════════════════════════════════════════════

// 18. One Next-Token Conditional Distribution Powers Scoring & Generation
function lmScoringGeneration() {
  const W = 540, H = 264;
  let o = '';
  o += box(46, 12, 448, 54, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 31, 'Next-Token Conditional Model P(xₜ | x_{<t})', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(270, 51, 'Exact chain rule: log P(x_{1:T}) = ∑ log P(xₜ | x_{<t})', { mono: true, fs: 11.5, fw: 600, fill: C.ink });

  o += arrow(200, 68, 136, 104, C.blue);
  o += arrow(340, 68, 404, 104, C.green);

  // Left: Scoring
  o += box(16, 108, 244, 140, C.lblue, C.blue, 8, 2);
  o += tx(138, 130, '1. Sequence Scoring (ASR / MT)', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += box(26, 146, 224, 40, '#FFFFFF', C.green, 6, 1.8);
  o += tx(138, 166, '"to recognize speech" ✓', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += box(26, 196, 224, 40, '#FFFFFF', D.red.base, 6, 1.8);
  o += tx(138, 216, '"to wreck a nice beach" ✗', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });

  // Right: Generation
  o += box(280, 108, 244, 140, C.lgreen, C.green, 8, 2);
  o += tx(402, 130, '2. Ancestral Text Generation', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += box(292, 146, 220, 88, '#FFFFFF', C.green, 6, 1.8);
  o += tx(402, 168, 'x₁ ∼ P(x₁)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(402, 192, 'x₂ ∼ P(x₂ | x₁)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(402, 216, 'x₃ ∼ P(x₃ | x₁, x₂) …', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 19. Perplexity Calibration Scale (Effective Branching Factor)
function perplexityScale() {
  const W = 540, H = 250;
  let o = '';
  o += tx(270, 22, 'Perplexity PPL = exp(−(1/T) ∑ log P(xₜ | x_{<t})) = Effective Branching Factor', { fs: 13.5, fw: 700, fill: C.ink });

  const anchors = [
    { x: 16,  ppl: 'PPL = 1',   sub: 'Perfect Oracle', d: 'P(true) = 1.0 always', f: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 148, ppl: 'PPL = k',   sub: 'Fair k-Sided Die', d: 'P(true) = 1/k average', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 280, ppl: 'PPL = |V|', sub: 'Uniform Guessing', d: 'P(x) = 1/|V| baseline', f: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 412, ppl: 'PPL = ∞',   sub: 'Unsmoothed MLE', d: 'Any test token P = 0', f: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];
  anchors.forEach((a, i) => {
    o += box(a.x, 48, 114, 116, a.f, a.st, 8, 2);
    o += tx(a.x + 57, 74, a.ppl, { mono: true, fs: 15, fw: 700, fill: a.dk });
    o += tx(a.x + 57, 104, a.sub, { fs: 12.5, fw: 700, fill: C.ink });
    o += tx(a.x + 57, 134, a.d, { fs: 11, fill: C.muted });
    if (i < 3) o += arrow(a.x + 116, 106, anchors[i + 1].x - 2, 106, C.gray);
  });

  o += box(16, 184, 508, 50, D.purple.tint, C.purple, 7, 1.8);
  o += tx(270, 209, 'Lower PPL = sharper predictions; any P(xₜ) = 0 on test text sends PPL → ∞.', { fs: 12, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// 20. Interpolated Kneser–Ney Discounting & Continuation Counts
function kneserNeyFlow() {
  const W = 540, H = 268;
  let o = '';
  o += box(16, 12, 508, 48, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 28, 'P_KN(w|u) = max(N(u,w) − D, 0) / N(u)  +  λ(u) · P_KN(w|u\')', { mono: true, fs: 12, fw: 700, fill: D.purple.dark });
  o += tx(270, 48, 'Closed-form absolute discount: D = c₁ / (c₁ + 2c₂) ≈ 0.93 on word trigrams', { fs: 12, fw: 600, fill: C.ink });

  // Left: Discounted highest order
  o += box(16, 76, 246, 176, C.lblue, C.blue, 8, 2);
  o += tx(139, 98, '1. Absolute Discounting (D ≈ 0.93)', { fs: 13, fw: 700, fill: D.blue.dark });
  o += box(26, 114, 226, 56, '#FFFFFF', C.blue, 6, 1.5);
  o += tx(139, 132, 'Seen: "the time traveller"', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(139, 152, '(47 − 0.93) / 87 ≈ 0.53', { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });
  o += box(26, 180, 226, 56, '#FFFFFF', C.blue, 6, 1.5);
  o += tx(139, 198, 'Freed mass λ(u) = D·S(u)/N(u)', { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });
  o += tx(139, 218, 'Passed recursively to suffix u\'', { fs: 11.5, fill: C.muted });

  // Right: Lower-order continuation counts
  o += box(278, 76, 246, 176, C.lgreen, C.green, 8, 2);
  o += tx(401, 98, '2. Continuation Counts at u\'', { fs: 13, fw: 700, fill: D.green.dark });
  o += box(288, 114, 226, 56, '#FFFFFF', C.green, 6, 1.5);
  o += tx(401, 132, '"said": 48 predecessors', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(401, 152, 'P_KN("said" | u) = 0.0018', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += box(288, 180, 226, 56, D.red.tint, D.red.base, 6, 1.5);
  o += tx(401, 198, '"filby": few predecessors', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(401, 218, 'P_KN("filby" | u) = 2.3·10⁻⁵', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

// 21. Perplexity vs. Bits per Byte Across Tokenizers
function pplVsBpb() {
  const W = 540, H = 258;
  let o = '';
  // Left: Incomparable Perplexity
  o += box(16, 14, 244, 174, D.red.tint, D.red.base, 8, 2);
  o += tx(138, 36, 'Raw Perplexity (Per Token T)', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(138, 56, 'Incomparable across granularities!', { fs: 12, fw: 600, fill: D.red.dark });
  const ppls = [
    ['Character (|V|=27)', 'PPL ≈ 6.4'],
    ['BPE-1024 (|V|=1024)', 'PPL ≈ 63'],
    ['Word (|V|=4316)', 'PPL ≈ 300'],
  ];
  ppls.forEach((p, i) => {
    const y = 72 + i * 36;
    o += box(26, y, 224, 28, '#FFFFFF', D.red.base, 5, 1.5);
    o += tx(138, y + 14, `${p[0]}: ${p[1]}`, { mono: true, fs: 11, fw: 700, fill: C.ink });
  });

  // Right: Invariant Bits per Byte
  o += box(280, 14, 244, 174, C.lgreen, C.green, 8, 2);
  o += tx(402, 36, 'Bits per Byte (Per UTF-8 Byte B)', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(402, 56, 'bpb = (T / B) log₂ PPL', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  const bpbs = [
    ['Character (1.0 B/tok)', '2.67 bpb'],
    ['BPE-1024 (3.1 B/tok)', '1.90 bpb ★'],
    ['Word (+ <unk> spell)', '2.22 bpb'],
  ];
  bpbs.forEach((p, i) => {
    const y = 72 + i * 36;
    o += box(290, y, 224, 28, '#FFFFFF', C.green, 5, 1.5);
    o += tx(402, y + 14, `${p[0]}: ${p[1]}`, { mono: true, fs: 11, fw: 700, fill: i === 1 ? D.green.dark : C.ink });
  });

  o += box(16, 202, 508, 42, C.lblue, C.blue, 7, 1.8);
  o += tx(270, 223, 'Operational meaning: bpb / 8 = lossless arithmetic-coding compression ratio.', { fs: 12.5, fw: 700, fill: D.blue.dark });
  return svg(W, H, o);
}

// 22. Train–Validation Boundary Leakage Guard
function windowLeakageGuard() {
  const W = 540, H = 250;
  let o = '';
  o += tx(270, 22, 'Why TimeMachine Drops num_steps = T Boundary Windows Between Splits', { fs: 13.5, fw: 700, fill: C.ink });

  // Train windows
  o += box(18, 48, 166, 98, C.lblue, C.blue, 8, 2);
  o += tx(101, 70, 'Training Windows', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(101, 92, 'windows[:num_train]', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(101, 114, 'Last row touches token', { fs: 11.5, fill: C.muted });
  o += tx(101, 130, 'num_train + T − 1', { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });

  // Dropped buffer
  o += box(194, 48, 152, 98, D.red.tint, D.red.base, 8, 2, true);
  o += tx(270, 70, 'Dropped Buffer (T)', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(270, 92, 'num_train : +T', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(270, 114, 'Straddles boundary!', { fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(270, 130, '(Targets seen in train)', { fs: 11, fill: D.red.dark });

  // Val windows
  o += box(356, 48, 166, 98, C.lgreen, C.green, 8, 2);
  o += tx(439, 70, 'Validation Windows', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(439, 92, 'windows[num_train+T:]', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(439, 114, 'Starts at token', { fs: 11.5, fill: C.muted });
  o += tx(439, 130, 'num_train + T (0 leak)', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });

  o += box(16, 166, 508, 66, C.lamber, C.amber, 8, 1.8);
  o += tx(270, 188, 'Adjacent length-T windows overlap in T − 1 tokens.', { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(270, 212, 'Skipping T windows guarantees zero validation tokens appear as training targets.', { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// ══════════════════════════════════════════════════════════════════════
// §8.4 rnn.md (5 diagrams)
// ══════════════════════════════════════════════════════════════════════

// 23. Fixed-Window Neural m-Gram vs. Shared-Weight Recurrent State
function windowVsRecurrentParams() {
  const W = 540, H = 258;
  let o = '';
  // Left: Fixed window neural m-gram
  o += box(16, 14, 244, 228, D.red.tint, D.red.base, 8, 2);
  o += tx(138, 36, 'Fixed-Window Neural m-Gram', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(138, 56, '(Bengio et al., 2003)', { fs: 12, fill: C.muted });
  o += box(30, 74, 216, 56, '#FFFFFF', D.red.base, 6, 1.5);
  o += tx(138, 94, 'Concat [Xₜ₋ₘ, …, Xₜ₋₁]', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(138, 114, 'W_xh shape: (m·d) × h', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += box(30, 144, 216, 82, '#FFFFFF', D.red.base, 6, 1.5);
  o += tx(138, 166, '• Params grow linearly with m', { fs: 12, fw: 600, fill: D.red.dark });
  o += tx(138, 188, '• +8,192 weights per extra lag', { fs: 12, fill: C.ink });
  o += tx(138, 210, '• Hard blind spot at lag m + 1', { fs: 12, fw: 700, fill: D.red.dark });

  // Right: Recurrent Neural Network
  o += box(280, 14, 244, 228, C.lgreen, C.green, 8, 2);
  o += tx(402, 36, 'Recurrent Neural Network', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(402, 56, '(Elman, 1990)', { fs: 12, fill: C.muted });
  o += box(294, 74, 216, 56, '#FFFFFF', C.green, 6, 1.5);
  o += tx(402, 94, 'Hₜ = tanh(XₜW_xh + Hₜ₋₁W_hh)', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(402, 114, 'Fixed shape: (d×h) + (h×h)', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += box(294, 144, 216, 82, '#FFFFFF', C.green, 6, 1.5);
  o += tx(402, 166, '• O(1) params independent of T', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(402, 188, '• 24,704 cell weights total', { fs: 12, fill: C.ink });
  o += tx(402, 210, '• Summarizes arbitrary prefix x_{<t}', { fs: 12, fw: 600, fill: D.green.dark });
  return svg(W, H, o);
}

// 24. Single Recurrent Step Matrix Dimensions
function cellDimensions() {
  const W = 540, H = 264;
  let o = '';
  o += tx(270, 20, 'Minibatch Tensor Shapes Inside One Recurrent Step (b, d=64, h=128, q=1027)', { fs: 13, fw: 700, fill: C.ink });

  // Left inputs
  o += box(16, 44, 152, 68, C.lblue, C.blue, 7, 2);
  o += tx(92, 66, 'Input Xₜ · W_xh', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(92, 90, '(b, 64) × (64, 128)', { mono: true, fs: 11.5, fill: C.ink });

  o += box(16, 140, 152, 68, D.purple.tint, C.purple, 7, 2);
  o += tx(92, 162, 'State Hₜ₋₁ · W_hh', { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });
  o += tx(92, 186, '(b, 128) × (128, 128)', { mono: true, fs: 11.5, fill: C.ink });

  o += arrow(172, 78, 206, 114, C.blue);
  o += arrow(172, 174, 206, 138, C.purple);

  // Center tanh state
  o += box(212, 86, 132, 80, C.lgreen, C.green, 8, 2);
  o += tx(278, 110, 'Hₜ = tanh(… + b_h)', { mono: true, fs: 11, fw: 700, fill: D.green.dark });
  o += tx(278, 134, 'shape: (b, 128)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(278, 152, '512 B per seq', { fs: 11.5, fill: C.muted });

  o += arrow(348, 126, 378, 126, C.green);

  // Right output
  o += box(384, 86, 140, 80, C.lamber, C.amber, 8, 2);
  o += tx(454, 110, 'Logits Oₜ', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(454, 132, 'Hₜ · W_hq + b_q', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(454, 152, 'shape: (b, 1027)', { mono: true, fs: 12, fw: 700, fill: D.orange.dark });

  o += tx(270, 238, 'All 5 parameter tensors (W_xh, W_hh, b_h, W_hq, b_q) are shared across all t = 1..T.', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 25. Batched Input Projection GEMM Outside the Sequential Loop
function batchedInputGemm() {
  const W = 540, H = 258;
  let o = '';
  // Top: Parallel input GEMM
  o += box(16, 14, 508, 102, C.lgreen, C.green, 8, 2);
  o += tx(270, 34, '1. Hoisted Outside Time Loop: 1 Parallel GEMM Across All T Steps', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += box(32, 50, 210, 50, '#FFFFFF', C.green, 6, 1.6);
  o += tx(137, 68, 'All Window Inputs X_{1:T}', { fs: 13, fw: 700, fill: C.ink });
  o += tx(137, 88, 'shape: (T·b, d)', { mono: true, fs: 12, fill: C.muted });
  o += arrow(248, 75, 286, 75, C.green);
  o += box(292, 50, 216, 50, '#FFFFFF', C.green, 6, 1.6);
  o += tx(400, 68, 'U_{1:T} = X_{1:T} · W_xh', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(400, 88, 'saturates GPU Tensor Cores', { fs: 12, fill: C.ink });

  // Bottom: Sequential recurrent loop
  o += box(16, 132, 508, 112, C.lamber, C.amber, 8, 2);
  o += tx(270, 152, '2. Inside Sequential Time Loop: Only Hₜ₋₁ · W_hh Waits on Step t − 1', { fs: 13.5, fw: 700, fill: D.orange.dark });
  const steps = [
    ['Step t = 1', 'tanh(U₁ + H₀W_hh)'],
    ['Step t = 2', 'tanh(U₂ + H₁W_hh)'],
    ['Step t = T', 'tanh(U_T+H_{T−1}W_hh)'],
  ];
  steps.forEach((s, i) => {
    const x = 26 + i * 168;
    o += box(x, 172, 156, 54, '#FFFFFF', C.amber, 6, 1.6);
    o += tx(x + 78, 190, s[0], { fs: 12, fw: 700, fill: D.orange.dark });
    o += tx(x + 78, 210, s[1], { mono: true, fs: 11, fw: 700, fill: C.ink });
    if (i < 2) o += arrow(x + 158, 199, x + 166, 199, D.orange.base);
  });
  return svg(W, H, o);
}

// 26. Parameter & FLOP Breakdown (222,915 Params)
function paramFlopBreakdown() {
  const W = 540, H = 260;
  let o = '';
  o += tx(270, 20, 'Where the 222,915 Parameters and 156,032 MACs/Token Live (d=64, h=128, |V|=1027)', { fs: 12.5, fw: 700, fill: C.ink });

  // Bar 1: Parameters
  o += tx(18, 48, 'Parameters (222,915 total):', { fs: 13, fw: 700, fill: C.ink, anchor: 'start' });
  o += box(18, 62, 148, 52, C.lblue, C.blue, 6, 2);
  o += tx(92, 80, 'Embedding E', { fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(92, 98, '65,728 (29.5%)', { mono: true, fs: 11.5, fill: C.ink });

  o += box(172, 62, 68, 52, C.lgreen, C.green, 6, 2);
  o += tx(206, 80, 'RNN Cell', { fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(206, 98, '11.1%', { mono: true, fs: 11.5, fw: 700, fill: C.ink });

  o += box(246, 62, 276, 52, C.lamber, C.amber, 6, 2);
  o += tx(384, 80, 'Output Projection W_hq + b_q', { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(384, 98, '132,483 params (59.4%)', { mono: true, fs: 12, fw: 700, fill: C.ink });

  // Bar 2: Per-Token MACs
  o += tx(18, 142, 'Compute per Token (156,032 MACs):', { fs: 13, fw: 700, fill: C.ink, anchor: 'start' });
  o += box(18, 156, 96, 52, C.lgreen, C.green, 6, 2);
  o += tx(66, 174, 'RNN Cell', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(66, 192, '24.6k (15.8%)', { mono: true, fs: 11, fill: C.ink });

  o += box(120, 156, 402, 52, C.lamber, C.amber, 6, 2);
  o += tx(321, 174, 'Output Vocabulary Projection Hₜ · W_hq', { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(321, 192, '131,456 MACs (84.2% of all FLOPs!)', { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += tx(270, 236, '89% of parameters and 84% of FLOPs sit in the vocabulary layers, not the recurrence!', { fs: 12.5, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// 27. Global Gradient Norm Clipping onto Ball of Radius theta
function gradClippingBall() {
  const W = 540, H = 256;
  let o = '';
  // Left: Geometric diagram
  o += box(16, 14, 248, 228, N.panel, C.gray, 8, 1.8);
  o += tx(140, 34, 'Projection onto Radius-θ Ball', { fs: 13.5, fw: 700, fill: C.ink });
  const cx = 92, cy = 148, r = 50;
  o += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${C.lgreen}" stroke="${C.green}" stroke-width="2" stroke-dasharray="5 4"/>`;
  // Unclipped exploding vector
  o += arrow(cx, cy, 220, 68, D.red.base, true);
  o += tx(174, 54, 'unclipped g (||g|| ≫ θ)', { mono: true, fs: 11, fw: 700, fill: D.red.dark });
  // Clipped vector stopping at circle boundary: dx=128, dy=-80, len=150.9; scale 50/150.9 = 0.331 -> (92+42.4, 148-26.5) = (134, 121)
  o += arrow(cx, cy, 134, 121, C.green);
  o += tx(196, 136, 'clipped g', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(196, 154, '(||g|| = θ)', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(cx, cy + 68, 'ball of radius θ = 1', { mono: true, fs: 12, fw: 700, fill: D.green.dark });

  // Right: Formula and effect
  o += box(280, 14, 244, 228, C.lblue, C.blue, 8, 2);
  o += tx(402, 36, 'Global Norm Clipping Rule', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += box(294, 54, 216, 48, '#FFFFFF', C.blue, 6, 1.6);
  o += tx(402, 78, 'g ← min(1, θ / ||g||) · g', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += box(294, 114, 216, 112, '#FFFFFF', C.green, 6, 1.6);
  o += tx(402, 136, '• Preserves exact direction of g', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(402, 160, '• Caps max parameter step at η·θ', { fs: 12, fw: 600, fill: C.ink });
  o += tx(402, 184, '• Computed jointly over all params', { fs: 12, fill: C.ink });
  o += tx(402, 208, '• Enables fast SGD at η = 4', { fs: 12, fw: 700, fill: D.blue.dark });
  return svg(W, H, o);
}

// ══════════════════════════════════════════════════════════════════════
// §8.5 bptt.md (9 diagrams)
// ══════════════════════════════════════════════════════════════════════

// 28. Scalar 40-Step Power Scaling: Vanishing vs. Exploding
function scalarBpttPowers() {
  const W = 540, H = 258;
  let o = '';
  o += tx(270, 20, '40-Step Sensitivity ∂hₜ / ∂hₜ₋₄₀ = w_h⁴⁰ Across Three Regimes', { fs: 14, fw: 700, fill: C.ink });

  const regimes = [
    { y: 42,  w: 'w_h = 1.1 (Exploding)', math: '1.1⁴⁰ ≈ 45.3×', note: 'Gradient overflows; destabilizes SGD', bw: 290, f: D.red.tint, st: D.red.base, dk: D.red.dark },
    { y: 112, w: 'w_h = 1.0 (Unit Margin)', math: '1.0⁴⁰ = 1.00×', note: 'Razor\'s edge: preserves exact norm', bw: 110, f: C.lgreen, st: C.green, dk: D.green.dark },
    { y: 182, w: 'w_h = 0.9 (Vanishing)', math: '0.9⁴⁰ ≈ 0.015×', note: '98.5% of distant gradient signal lost!', bw: 18,  f: C.lblue, st: C.blue, dk: D.blue.dark },
  ];
  regimes.forEach((r) => {
    o += box(18, r.y, 196, 54, '#FFFFFF', r.st, 7, 1.8);
    o += tx(116, r.y + 18, r.w, { mono: true, fs: 12.5, fw: 700, fill: r.dk });
    o += tx(116, r.y + 38, r.math, { mono: true, fs: 13.5, fw: 700, fill: C.ink });
    o += box(228, r.y + 6, r.bw, 20, r.f, r.st, 4, 1.8);
    o += tx(228, r.y + 42, r.note, { fs: 12, fw: 600, fill: r.dk, anchor: 'start' });
  });
  return svg(W, H, o);
}

// 29. Shared-Weight Unrolled Graph & Total Gradient Sum
function sharedWeightAccumulation() {
  const W = 540, H = 264;
  let o = '';
  // Top: Unrolled states h1, h2, h3
  const xs = [46, 214, 382];
  xs.forEach((x, i) => {
    const t = i + 1;
    o += box(x, 18, 112, 42, C.lgreen, C.green, 6, 1.8);
    o += tx(x + 56, 39, `Loss l_${t}(o_${t})`, { mono: true, fs: 12, fw: 700, fill: D.green.dark });
    o += arrow(x + 56, 86, x + 56, 62, C.gray);
    o += box(x, 88, 112, 44, C.lblue, C.blue, 6, 2);
    o += tx(x + 56, 110, `State h_${t}`, { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
    if (i < 2) o += arrow(x + 114, 110, xs[i + 1] - 2, 110, C.blue);
  });

  // Shared parameter node W_h at bottom
  o += box(136, 188, 268, 58, D.red.tint, D.red.base, 8, 2);
  o += tx(270, 208, 'Shared Parameter w_h', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(270, 228, '∂L/∂w_h = (1/T) ∑ₜ (∂lₜ/∂hₜ)(∂hₜ/∂w_h)', { mono: true, fs: 11.5, fw: 700, fill: C.ink });

  // Red backward arrows from h_1, h_2, h_3 to shared w_h
  o += arrow(102, 134, 184, 186, D.red.base);
  o += arrow(270, 134, 270, 186, D.red.base);
  o += arrow(438, 134, 356, 186, D.red.base);
  return svg(W, H, o);
}

// 30. O(T^2) Forward Pairwise Products vs. O(T) Reverse-Mode BPTT
function forwardVsReverseBptt() {
  const W = 540, H = 252;
  let o = '';
  // Left: Naive Forward Expansion O(T^2)
  o += box(16, 14, 244, 224, D.red.tint, D.red.base, 8, 2);
  o += tx(138, 36, 'Naive Pairwise Sums: O(T²)', { fs: 14, fw: 700, fill: D.red.dark });
  const pairs = ['t = 1: u₁', 't = 2: u₂ + c₂u₁', 't = 3: u₃ + c₃u₂ + c₃c₂u₁', 't = T: ∑ᵢ (∏ⱼ cⱼ) uᵢ'];
  pairs.forEach((p, i) => {
    const y = 56 + i * 36;
    o += box(26, y, 224, 28, '#FFFFFF', D.red.base, 5, 1.4);
    o += tx(138, y + 14, p, { mono: true, fs: 11.5, fw: 600, fill: C.ink });
  });
  o += tx(138, 216, 'T(T−1)/2 redundant Jacobian products', { fs: 11.5, fw: 700, fill: D.red.dark });

  // Right: Reverse-Mode BPTT O(T)
  o += box(280, 14, 244, 224, C.lgreen, C.green, 8, 2);
  o += tx(402, 36, 'Reverse-Mode BPTT: O(T)', { fs: 14, fw: 700, fill: D.green.dark });
  const rev = ['G_T = (∂L/∂O_T) W_hq^T', 'G_{T−1} = loc + G_T W_hh^T', 'G_t = loc + G_{t+1} W_hh^T', 'Single O(T) backward sweep'];
  rev.forEach((p, i) => {
    const y = 56 + i * 36;
    o += box(290, y, 224, 28, '#FFFFFF', C.green, 5, 1.4);
    o += tx(402, y + 14, p, { mono: true, fs: 11.5, fw: 700, fill: i === 3 ? D.green.dark : C.ink });
  });
  o += tx(402, 216, 'Exactly 1 matrix multiply by W_hh^T / step', { fs: 11.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 31. Adjoint Node Recursion at Hidden State H_t
function adjointNodeRecursion() {
  const W = 540, H = 256;
  let o = '';
  o += tx(270, 20, 'Two Incoming Gradient Paths Summing at Hidden State Adjoint Gₜ = ∂L / ∂Hₜ', { fs: 13.5, fw: 700, fill: C.ink });

  // Top node O_t
  o += box(164, 38, 152, 48, C.lamber, C.amber, 7, 2);
  o += tx(240, 56, 'Output Logits Oₜ', { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(240, 74, 'local grad ∂L / ∂Oₜ', { mono: true, fs: 11.5, fill: C.ink });

  // Downward red arrow from O_t to H_t
  o += arrow(240, 88, 240, 128, D.red.base);
  o += tx(154, 108, '(∂L/∂Oₜ) W_hq^T', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });

  // Right node H_{t+1}
  o += box(400, 132, 122, 56, C.lblue, C.blue, 7, 2);
  o += tx(461, 152, 'Next State Hₜ₊₁', { fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(461, 172, 'adjoint Gₜ₊₁', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });

  // Leftward red arrow from H_{t+1} to H_t (gap is 306..400 = 94px!)
  o += arrow(396, 160, 310, 160, D.red.base);
  o += tx(353, 140, 'Gₜ₊₁ W_hh^T', { mono: true, fs: 11, fw: 700, fill: D.red.dark });

  // Center node H_t
  o += box(174, 132, 132, 56, D.red.tint, D.red.base, 8, 2.2);
  o += tx(240, 152, 'State Hₜ', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(240, 172, 'Gₜ = sum of 2', { mono: true, fs: 11.5, fw: 700, fill: C.ink });

  // Leftward red arrow to H_{t-1}
  o += arrow(170, 160, 138, 160, D.red.base);
  o += box(16, 132, 118, 56, C.lblue, C.blue, 7, 2);
  o += tx(75, 152, 'Prev State Hₜ₋₁', { fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(75, 172, '← Gₜ W_hh^T', { mono: true, fs: 11, fill: C.ink });

  o += box(24, 206, 492, 36, C.lgreen, C.green, 6, 1.6);
  o += tx(270, 224, 'Gₜ = (∂L/∂Oₜ) W_hq^T + Gₜ₊₁ W_hh^T   (boundary G_{T+1} = 0)', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 32. Unrolled Lag-k Matrix Power Expansion of G_t
function lagMatrixPowers() {
  const W = 540, H = 258;
  let o = '';
  o += tx(270, 20, 'Unrolling Gₜ = ∑_{j=t}^{T} (∂L/∂Oⱼ) W_hq^T (W_hh^T)^{j−t} Across Future Steps j', { fs: 13, fw: 700, fill: C.ink });

  const lags = [
    { y: 40,  lag: 'Lag k = 0 (j = t)',   op: '∇Oₜ L · W_hq^T · I', note: 'Direct step-t loss', f: C.lgreen, st: C.green, dk: D.green.dark },
    { y: 90,  lag: 'Lag k = 1 (j = t+1)', op: '∇Oₜ₊₁ L · W_hq^T · (W_hh^T)¹', note: '1 transpose GEMM', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 140, lag: 'Lag k = 2 (j = t+2)', op: '∇Oₜ₊₂ L · W_hq^T · (W_hh^T)²', note: '2 transpose GEMMs', f: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 190, lag: 'Lag k = T−t (j = T)', op: '∇O_T L · W_hq^T · (W_hh^T)^{T−t}', note: 'Scales as ρ^{T−t}', f: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];
  lags.forEach((l) => {
    o += box(16, l.y, 144, 38, l.f, l.st, 6, 1.8);
    o += tx(88, l.y + 19, l.lag, { mono: true, fs: 11, fw: 700, fill: l.dk });
    o += box(166, l.y, 232, 38, '#FFFFFF', l.st, 6, 1.6);
    o += tx(282, l.y + 19, l.op, { mono: true, fs: 11, fw: 700, fill: C.ink });
    o += box(404, l.y, 120, 38, l.f, l.st, 6, 1.5);
    o += tx(464, l.y + 19, l.note, { fs: 11.5, fw: 700, fill: l.dk });
  });
  o += tx(270, 244, 'Parameter gradients accumulate outer products: ∂L/∂W_hh = ∑ₜ Hₜ₋₁ᵀ Gₜ', { mono: true, fs: 12, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// 33. Spectral Radius Regimes Around the Complex Unit Circle
function spectralRadiusRegimes() {
  const W = 540, H = 256;
  let o = '';
  // Left: Unit disk in complex plane
  o += box(16, 14, 232, 228, N.panel, C.gray, 8, 1.8);
  o += tx(132, 34, 'Eigenvalues λᵢ of W_hh in ℂ', { fs: 13.5, fw: 700, fill: C.ink });
  o += tx(132, 56, 'Outside disk: ρ > 1 (explode)', { mono: true, fs: 11, fw: 700, fill: D.red.dark });
  const cx = 132, cy = 138, r = 52;
  o += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${C.lblue}" stroke="${C.green}" stroke-width="2.5"/>`;
  o += tx(cx, cy - 10, 'ρ < 1', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(cx, cy + 12, '(vanish)', { fs: 11.5, fw: 600, fill: D.blue.dark });
  o += tx(132, 214, 'Boundary |λ| = 1: unit norm', { mono: true, fs: 11, fw: 700, fill: D.green.dark });

  // Right: Three regime cards
  const cards = [
    { y: 14,  t: '1. Vanishing (ρ < 1)', d: '|| (W_hh^T)^k || ∼ ρ^k → 0 exponentially', f: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 92,  t: '2. Critical Unit Circle (ρ = 1)', d: '|λᵢ| = 1 preserves norm across arbitrary lags k', f: C.lgreen, st: C.green, dk: D.green.dark },
    { y: 170, t: '3. Exploding (ρ > 1) & Non-Normal', d: 'Grows as ρ^k → ∞ (or transient hump)', f: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];
  cards.forEach((c) => {
    o += box(264, c.y, 260, 68, c.f, c.st, 7, 1.8);
    o += tx(394, c.y + 22, c.t, { fs: 13.5, fw: 700, fill: c.dk });
    o += tx(394, c.y + 46, c.d, { fs: 11.5, fw: 600, fill: C.ink });
  });
  return svg(W, H, o);
}

// 34. Nonlinear tanh Contraction vs. Spectral Radius
function tanhContraction() {
  const W = 540, H = 258;
  let o = '';
  o += tx(270, 20, 'Why Our Trained RNN Vanishes Despite Spectral Radius ρ(W_hh) ≈ 1.80', { fs: 13.5, fw: 700, fill: C.ink });

  // Factor 1: tanh'(z)
  o += box(16, 44, 152, 114, C.lblue, C.blue, 8, 2);
  o += tx(92, 68, 'Local tanh Derivative', { fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(92, 94, '1 − Hₜ² ∈ (0, 1]', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(92, 118, 'Measured mean:', { fs: 11.5, fill: C.muted });
  o += tx(92, 138, '0.27 ×', { mono: true, fs: 15, fw: 700, fill: D.blue.dark });

  o += tx(180, 101, '×', { fs: 24, fw: 700, fill: C.ink });

  // Factor 2: W_hh^T
  o += box(194, 44, 152, 114, C.lamber, C.amber, 8, 2);
  o += tx(270, 68, 'Recurrent W_hh^T', { fs: 12, fw: 700, fill: D.orange.dark });
  o += tx(270, 94, 'Spectral radius ρ', { fs: 12, fw: 600, fill: C.ink });
  o += tx(270, 118, 'Measured value:', { fs: 11.5, fill: C.muted });
  o += tx(270, 138, '1.80 ×', { mono: true, fs: 15, fw: 700, fill: D.orange.dark });

  o += tx(358, 101, '≈', { fs: 24, fw: 700, fill: C.ink });

  // Product: Effective Jacobian multiplier
  o += box(372, 44, 152, 114, D.red.tint, D.red.base, 8, 2);
  o += tx(448, 68, 'Net Step Multiplier', { fs: 12, fw: 700, fill: D.red.dark });
  o += tx(448, 94, '0.27 × 1.80 ≈ 0.49', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(448, 118, 'Halves every step!', { fs: 11.5, fw: 700, fill: C.ink });
  o += tx(448, 138, '0.49⁸ ≈ 0.003 (<1%)', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });

  o += box(16, 176, 508, 66, D.purple.tint, C.purple, 8, 1.8);
  o += tx(270, 198, 'G_{t−1} = (∂L/∂O_{t−1}) W_hq^T  +  [ Gₜ ⊙ (1 − Hₜ²) ] W_hh^T', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(270, 222, 'Saturated hidden units (Hₜ → ±1) choke off backward gradient flow within ~8 steps.', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 35. Why Global Clipping Leaves Lag Ratios Unchanged
function clippingLagRatios() {
  const W = 540, H = 258;
  let o = '';
  o += tx(270, 20, 'Uniform Scalar Clipping Rescales Total Norm ||g|| Without Changing Lag Ratios', { fs: 13, fw: 700, fill: C.ink });

  // Left: Before clipping
  o += box(16, 40, 242, 198, D.red.tint, D.red.base, 8, 2);
  o += tx(137, 62, 'Before Clipping (||g|| ≈ 10)', { fs: 13.5, fw: 700, fill: D.red.dark });
  const bars1 = [
    { y: 84,  lab: 'Lag 0:', w: 130, val: '9.8' },
    { y: 122, lab: 'Lag 2:', w: 31,  val: '2.35' },
    { y: 160, lab: 'Lag 8:', w: 4,   val: '0.033' },
  ];
  bars1.forEach((b) => {
    o += tx(30, b.y + 12, b.lab, { mono: true, fs: 12, fw: 700, fill: C.ink, anchor: 'start' });
    o += box(84, b.y, b.w, 24, '#FFFFFF', D.red.base, 4, 1.6);
    o += tx(84 + b.w + 8, b.y + 12, b.val, { mono: true, fs: 12, fw: 700, fill: D.red.dark, anchor: 'start' });
  });
  o += tx(137, 214, 'Lag 0 : Lag 8 ratio ≈ 300 : 1', { mono: true, fs: 12, fw: 700, fill: D.red.dark });

  // Right: After clipping
  o += box(282, 40, 242, 198, C.lgreen, C.green, 8, 2);
  o += tx(403, 62, 'After Clipping by 0.1 (||g|| = 1)', { fs: 13.5, fw: 700, fill: D.green.dark });
  const bars2 = [
    { y: 84,  lab: 'Lag 0:', w: 80, val: '0.98' },
    { y: 122, lab: 'Lag 2:', w: 19, val: '0.235' },
    { y: 160, lab: 'Lag 8:', w: 4,  val: '0.0033' },
  ];
  bars2.forEach((b) => {
    o += tx(296, b.y + 12, b.lab, { mono: true, fs: 12, fw: 700, fill: C.ink, anchor: 'start' });
    o += box(350, b.y, b.w, 24, '#FFFFFF', C.green, 4, 1.6);
    o += tx(350 + b.w + 8, b.y + 12, b.val, { mono: true, fs: 12, fw: 700, fill: D.green.dark, anchor: 'start' });
  });
  o += tx(403, 214, 'Same ≈ 300 : 1 lag ratio!', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 36. Full-Sequence BPTT Memory Wall vs. Truncated Segments
function fullBpttMemoryWall() {
  const W = 540, H = 256;
  let o = '';
  // Top: Full-corpus BPTT
  o += box(16, 14, 508, 102, D.red.tint, D.red.base, 8, 2);
  o += tx(270, 34, '(a) Full-Corpus BPTT (T = 65,741 Tokens): Prohibitive Memory & Latency', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += box(32, 52, 232, 48, '#FFFFFF', D.red.base, 6, 1.6);
  o += tx(148, 70, 'Must Retain All 65,741 States H_{1:T}', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(148, 88, 'O(T·b·h) VRAM for ∂L/∂W_hh', { mono: true, fs: 11.5, fill: C.ink });
  o += box(276, 52, 232, 48, '#FFFFFF', D.red.base, 6, 1.6);
  o += tx(392, 70, 'Only 1 SGD Update per Book!', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(392, 88, '65,741 sequential steps / update', { fs: 11.5, fill: C.ink });

  // Bottom: Truncated BPTT
  o += box(16, 132, 508, 110, C.lgreen, C.green, 8, 2);
  o += tx(270, 152, '(b) Truncated BPTT (T₀ = 32 Token Windows with detach / stop_gradient)', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += box(32, 172, 232, 52, '#FFFFFF', C.green, 6, 1.6);
  o += tx(148, 190, 'Bounded O(T₀·b·h) VRAM', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(148, 210, '2,054× smaller activation graph!', { fs: 11.5, fw: 600, fill: C.ink });
  o += box(276, 172, 232, 52, '#FFFFFF', C.green, 6, 1.6);
  o += tx(392, 190, 'Frequent Minibatch Updates', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(392, 210, 'Graph freed after every 32 steps', { fs: 11.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// ══════════════════════════════════════════════════════════════════════
// §8.6 decoding.md (5 diagrams)
// ══════════════════════════════════════════════════════════════════════

// 37. Combinatorial Search Space Explosion (1027^50 > 10^150)
function searchSpaceExplosion() {
  const W = 540, H = 258;
  let o = '';
  o += box(16, 12, 508, 52, D.purple.tint, C.purple, 8, 2);
  o += tx(270, 30, 'Exhaustive Search Over T = 50 Tokens: |V|ᵀ = 1,027⁵⁰ > 10¹⁵⁰ Candidates', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(270, 50, 'Goal: Convert next-token conditionals P(xₜ | x_{<t}) into fluent text in O(T) steps', { fs: 12, fw: 600, fill: C.ink });

  // Left hazard: Pure Maximization
  o += box(16, 80, 244, 164, D.red.tint, D.red.base, 8, 2);
  o += tx(138, 102, 'Hazard 1: Naive Argmax', { fs: 14, fw: 700, fill: D.red.dark });
  o += box(30, 120, 216, 52, '#FFFFFF', D.red.base, 6, 1.5);
  o += tx(138, 138, 'xₜ = argmax P(x | x_{<t})', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(138, 158, '100% deterministic trajectory', { fs: 11.5, fill: C.muted });
  o += tx(138, 194, '• Myopic local commitment', { fs: 12, fw: 600, fill: D.red.dark });
  o += tx(138, 216, '• Locks into repetition loops!', { fs: 12, fw: 700, fill: D.red.dark });

  // Right hazard: Unpruned Sampling
  o += box(280, 80, 244, 164, C.lamber, C.amber, 8, 2);
  o += tx(402, 102, 'Hazard 2: Unpruned Sampling', { fs: 14, fw: 700, fill: D.orange.dark });
  o += box(294, 120, 216, 52, '#FFFFFF', C.amber, 6, 1.5);
  o += tx(402, 138, 'xₜ ∼ P(x | x_{<t}) over all |V|', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(402, 158, '~900 tail tokens hold ~18% mass', { fs: 11.5, fill: C.muted });
  o += tx(402, 194, '• Draws incoherent tail tokens', { fs: 12, fw: 600, fill: D.orange.dark });
  o += tx(402, 216, '• Exposure bias derails rollout!', { fs: 12, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

// 38. Greedy Decoding Myopia Tree (0.15 vs 0.36)
function greedyMyopiaTree() {
  const W = 540, H = 256;
  let o = '';
  o += tx(270, 20, 'Why Greedy Search Misses the Globally Optimal Sequence (Myopia)', { fs: 14, fw: 700, fill: C.ink });

  // Root
  o += box(18, 104, 86, 48, N.panel, C.gray, 7, 2);
  o += tx(61, 128, 'Prompt', { fs: 13.5, fw: 700, fill: C.ink });

  // Top branch: Path A (Greedy picks at step 1)
  o += arrow(108, 114, 168, 72, D.red.base);
  o += box(174, 44, 124, 52, D.red.tint, D.red.base, 7, 2);
  o += tx(236, 62, 'Token A (Greedy)', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(236, 82, 'P(A) = 0.50 ★', { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += arrow(302, 70, 350, 70, D.red.base);
  o += box(356, 44, 168, 52, D.red.tint, D.red.base, 7, 2);
  o += tx(440, 62, 'Best P(x₂|A) = 0.30', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(440, 82, 'Joint: 0.5×0.3 = 0.15', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });

  // Bottom branch: Path B (Pruned by greedy, but 2.4x higher joint P)
  o += arrow(108, 142, 168, 184, C.green);
  o += box(174, 160, 124, 52, C.lgreen, C.green, 7, 2);
  o += tx(236, 178, 'Token B (Pruned)', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(236, 198, 'P(B) = 0.40', { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += arrow(302, 186, 350, 186, C.green);
  o += box(356, 160, 168, 52, C.lgreen, C.green, 7, 2);
  o += tx(440, 178, 'Best P(x₂|B) = 0.90', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(440, 198, 'Joint: 0.4×0.9 = 0.36', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });

  o += tx(270, 238, 'Beam search (w ≥ 2) keeps both A and B alive at step 1 and recovers the 0.36 winner!', { fs: 12.5, fw: 700, fill: D.blue.dark });
  return svg(W, H, o);
}

// 39. Length Normalization in Beam Search (1 / T^beta)
function lengthNormCurves() {
  const W = 540, H = 258;
  let o = '';
  o += tx(270, 20, 'Comparing Variable-Length Hypotheses: Score = (1 / T^β) ∑ log P(xₜ | x_{<t})', { fs: 13, fw: 700, fill: C.ink });

  const cards = [
    { x: 16,  b: 'β = 0 (Unnormalized)', eq: '∑ log P ≤ 0', eff: 'Monotonically decreases with T; unfairly favors premature <eos>!', f: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 194, b: 'β = 0.75 (Standard)',  eq: '(∑ log P) / T^{0.75}', eff: 'Wu et al. (2016) sweet spot for MT & ASR beam ranking.', f: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 372, b: 'β = 1.0 (Per-Token Avg)', eq: '(1 / T) ∑ log P', eff: 'Removes length penalty completely; can over-favor long outputs.', f: C.lblue, st: C.blue, dk: D.blue.dark },
  ];
  cards.forEach((c) => {
    o += box(c.x, 42, 152, 134, c.f, c.st, 8, 2);
    o += tx(c.x + 76, 64, c.b, { fs: 12, fw: 700, fill: c.dk });
    o += box(c.x + 8, 80, 136, 30, '#FFFFFF', c.st, 5, 1.5);
    o += tx(c.x + 76, 95, c.eq, { mono: true, fs: 11, fw: 700, fill: C.ink });
    const lines = c.eff.match(/.{1,24}(\s|$)/g) || [c.eff];
    lines.slice(0, 3).forEach((ln, idx) => {
      o += tx(c.x + 76, 126 + idx * 16, ln.trim(), { fs: 11, fill: C.ink });
    });
  });

  o += box(16, 192, 508, 50, D.purple.tint, C.purple, 7, 1.8);
  o += tx(270, 217, '10-tok (−15) vs 25-tok (−28): β=0 picks 10-tok; β=0.75 picks 25-tok (−2.50 > −2.67).', { fs: 11.5, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// 40. The Unreliable Tail Problem in Ancestral Sampling
function unreliableTailRollout() {
  const W = 540, H = 256;
  let o = '';
  o += tx(270, 20, 'Why Raw Ancestral Sampling Derails Over 50 Steps (Holtzman et al., 2020)', { fs: 13.5, fw: 700, fill: C.ink });

  // Head vs Tail distribution bar
  o += box(18, 44, 240, 82, C.lgreen, C.green, 8, 2);
  o += tx(138, 66, 'Calibrated Head (~50 Tokens)', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(138, 88, '82% of probability mass', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(138, 108, 'Plausible continuations', { fs: 12, fill: C.muted });

  o += box(272, 44, 250, 82, D.red.tint, D.red.base, 8, 2);
  o += tx(397, 66, 'Unreliable Tail (~900 Tokens)', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(397, 88, '18% collective mass per step!', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(397, 108, 'Individually tiny, collectively huge', { fs: 12, fill: C.ink });

  // 50-step compounding box
  o += box(16, 144, 508, 96, C.lamber, C.amber, 8, 2);
  o += tx(270, 168, '50-Token Rollout Survival Probability Without Truncation:', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(270, 194, 'P(0 tail draws in 50 steps) = (1 − 0.18)⁵⁰ ≈ 0.00005 (0.005%)', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(270, 220, 'Remedy: Truncate the tail (top-k, top-p, min-p) and renormalize!', { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 41. Stateless O(T^2) Re-Evaluation vs. Stateful O(T) Recurrent Decoding
function statelessVsStatefulDecode() {
  const W = 540, H = 264;
  let o = '';
  // Top: Stateless O(T^2)
  o += box(16, 12, 508, 106, D.red.tint, D.red.base, 8, 2);
  o += tx(270, 32, '(a) Stateless step_fn(ids): Re-Reads Growing Prefix at Every Step → O(T²)', { fs: 13.5, fw: 700, fill: D.red.dark });
  const stSteps = ['Step 1: read x₁ (1)', 'Step 2: read x₁:₂ (2)', 'Step T: read x_{1:T} (T)'];
  stSteps.forEach((s, i) => {
    const x = 28 + i * 166;
    o += box(x, 48, 152, 34, '#FFFFFF', D.red.base, 5, 1.5);
    o += tx(x + 76, 65, s, { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  });
  o += tx(270, 100, 'Total cell updates = 1 + 2 + … + T = T(T + 1) / 2 = O(T²) wasted work!', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });

  // Bottom: Stateful O(T)
  o += box(16, 134, 508, 116, C.lgreen, C.green, 8, 2);
  o += tx(270, 154, '(b) Stateful Recurrent Carry (predict): Carries 512-Byte State Hₜ → O(T)', { fs: 13.5, fw: 700, fill: D.green.dark });
  const cSteps = ['H₁ = g(H₀, x₁)', 'H₂ = g(H₁, x₂)', 'H_T = g(H_{T−1}, x_T)'];
  cSteps.forEach((s, i) => {
    const x = 28 + i * 166;
    o += box(x, 172, 152, 36, '#FFFFFF', C.green, 5, 1.8);
    o += tx(x + 76, 190, s, { mono: true, fs: 11, fw: 700, fill: D.green.dark });
    if (i < 2) o += arrow(x + 154, 190, x + 164, 190, C.green);
  });
  o += tx(270, 230, '1 cell update per token (O(T) total) · constant 512-byte state memory!', { fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

export const diagrams = {
  // §8.0 index.md
  'ch08-rnn-fixed-vs-seq':             fixedVsSeq,
  'ch08-rnn-roadmap':                  roadmap,
  'ch08-rnn-wells-benchmarks':         wellsBenchmarks,
  'ch08-rnn-pipeline-summary':         pipelineSummary,
  // §8.1 sequence.md
  'ch08-rnn-iid-vs-temporal':          iidVsTemporal,
  'ch08-rnn-task-taxonomy':            taskTaxonomy,
  'ch08-rnn-growing-prefix':           growingPrefix,
  'ch08-rnn-chain-rule':               chainRule,
  'ch08-rnn-markov-horizon':           markovHorizon,
  'ch08-rnn-stationarity-pooling':     stationarityPooling,
  'ch08-rnn-lag-matrix':               lagMatrix,
  'ch08-rnn-averaging-vs-oscillator':  averagingVsOscillator,
  // §8.2 text-sequence.md
  'ch08-rnn-tokenizer-impact':         tokenizerImpact,
  'ch08-rnn-vocab-unk':                vocabUnk,
  'ch08-rnn-utf8-bytes':               utf8Bytes,
  'ch08-rnn-bpe-greedy-loop':          bpeGreedyLoop,
  'ch08-rnn-special-token-safety':     specialTokenSafety,
  // §8.3 language-model.md
  'ch08-rnn-lm-scoring-generation':    lmScoringGeneration,
  'ch08-rnn-perplexity-scale':         perplexityScale,
  'ch08-rnn-kneser-ney-flow':          kneserNeyFlow,
  'ch08-rnn-ppl-vs-bpb':               pplVsBpb,
  'ch08-rnn-window-leakage-guard':     windowLeakageGuard,
  // §8.4 rnn.md
  'ch08-rnn-window-vs-recurrent-params': windowVsRecurrentParams,
  'ch08-rnn-cell-dimensions':          cellDimensions,
  'ch08-rnn-batched-input-gemm':       batchedInputGemm,
  'ch08-rnn-param-flop-breakdown':     paramFlopBreakdown,
  'ch08-rnn-grad-clipping-ball':       gradClippingBall,
  // §8.5 bptt.md
  'ch08-rnn-scalar-bptt-powers':       scalarBpttPowers,
  'ch08-rnn-shared-weight-accumulation': sharedWeightAccumulation,
  'ch08-rnn-forward-vs-reverse-bptt':  forwardVsReverseBptt,
  'ch08-rnn-adjoint-node-recursion':   adjointNodeRecursion,
  'ch08-rnn-lag-matrix-powers':        lagMatrixPowers,
  'ch08-rnn-spectral-radius-regimes':  spectralRadiusRegimes,
  'ch08-rnn-tanh-contraction':         tanhContraction,
  'ch08-rnn-clipping-lag-ratios':      clippingLagRatios,
  'ch08-rnn-full-bptt-memory-wall':    fullBpttMemoryWall,
  // §8.6 decoding.md
  'ch08-rnn-search-space-explosion':   searchSpaceExplosion,
  'ch08-rnn-greedy-myopia-tree':       greedyMyopiaTree,
  'ch08-rnn-length-norm-curves':       lengthNormCurves,
  'ch08-rnn-unreliable-tail-rollout':  unreliableTailRollout,
  'ch08-rnn-stateless-vs-stateful-decode': statelessVsStatefulDecode,
};
