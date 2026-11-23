// Slide diagrams for Chapter 2: Linear Neural Networks for Regression
// (chapter_linear-regression/*.md).
//
// Exports 60 conceptual, geometric, and systems diagrams keyed by stable
// `ch02-lr-<concept>` IDs.

import { C, TOKENS, grid, tx, arrow, svg } from './engine.mjs';
import { sec, num } from './sections.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// Helper: clean centered stacked cards (zero badge collision, generous padding)
function stackCards(W, H, title, cards) {
  let o = '';
  const Wc = 560;
  const Hc = Math.max(H, cards.length === 4 ? 312 : 284);
  if (title) {
    o += tx(Wc / 2, 20, title, { fs: 15, fw: 700, fill: C.ink });
  }
  const topY = title ? 38 : 16;
  const availH = Hc - topY - 12;
  const gap = 10;
  const ch = Math.floor((availH - gap * (cards.length - 1)) / cards.length);
  cards.forEach((c, idx) => {
    const y = topY + idx * (ch + gap);
    o += box(14, y, Wc - 28, ch, c.fill, c.stroke, 8, 2);
    const headerText = c.badge ? `${c.head}  [${c.badge}]` : c.head;
    o += tx(Wc / 2, y + Math.round(ch * 0.34), headerText, { fs: 13.5, fw: 700, fill: c.dark });
    o += tx(Wc / 2, y + Math.round(ch * 0.72), c.sub, { fs: 12.5, fw: 600, fill: C.ink, mono: !!c.mono });
  });
  return svg(Wc, Hc, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2.0 index.md (3 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function lrMiniature() {
  const W = 560, H = 286;
  let o = '';
  o += tx(280, 20, 'Linear Regression: Deep Learning in Miniature', { fs: 15.5, fw: 700, fill: C.ink });

  const bw = 160, bh = 88, y1 = 42;
  const xs = [16, 200, 384];
  o += arrow(xs[0] + bw + 2, y1 + bh / 2, xs[1] - 2, y1 + bh / 2, C.blue);
  o += arrow(xs[1] + bw + 2, y1 + bh / 2, xs[2] - 2, y1 + bh / 2, D.orange.base);

  o += box(xs[0], y1, bw, bh, C.lblue, C.blue);
  o += tx(xs[0] + bw / 2, y1 + 22, '1. Affine Forward', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(xs[0] + bw / 2, y1 + 46, 'ŷ = Xw + b', { mono: true, fs: 13.5, fw: 700, fill: C.ink });
  o += tx(xs[0] + bw / 2, y1 + 68, '1-neuron layer', { fs: 12.5, fill: C.muted });

  o += box(xs[1], y1, bw, bh, C.lamber, C.amber);
  o += tx(xs[1] + bw / 2, y1 + 22, '2. Squared Loss', { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(xs[1] + bw / 2, y1 + 46, '‖ŷ − y‖² / 2n', { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(xs[1] + bw / 2, y1 + 68, 'Gaussian MLE', { fs: 12.5, fill: C.muted });

  o += box(xs[2], y1, bw, bh, C.lgreen, C.green);
  o += tx(xs[2] + bw / 2, y1 + 22, '3. Optimization', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(xs[2] + bw / 2, y1 + 46, 'lstsq vs. SGD', { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(xs[2] + bw / 2, y1 + 68, 'O(nd²) vs O(Bd)', { mono: true, fs: 12, fill: C.muted });

  const y2 = 152, hw = 254, hh = 114;
  o += box(16, y2, hw, hh, D.red.tint, D.red.base);
  o += tx(16 + hw / 2, y2 + 22, 'Two Classic Failure Modes', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(16 + hw / 2, y2 + 48, '• Shape bug: (B,)−(B,1) → (B,B)', { mono: true, fs: 12, fw: 600, fill: C.ink });
  o += tx(16 + hw / 2, y2 + 72, '• Overfitting: deg-19 on 20 pts', { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(16 + hw / 2, y2 + 94, 'Both still lower training loss!', { fs: 12.5, fw: 700, fill: D.red.dark });

  o += box(290, y2, hw, hh, D.purple.tint, C.purple);
  o += tx(290 + hw / 2, y2 + 22, 'Exact Analytical Oracles', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(290 + hw / 2, y2 + 48, '• Projection: Xᵀ(Xŵ − y) = 0', { mono: true, fs: 12, fw: 600, fill: C.ink });
  o += tx(290 + hw / 2, y2 + 72, '• Noise floor: E[L(w*)] = σ²/2', { mono: true, fs: 12, fw: 600, fill: C.ink });
  o += tx(290 + hw / 2, y2 + 94, '• Recovery SE: σ / √n', { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });

  return svg(W, H, o);
}

function lrRoadmap() {
  const W = 560, H = 296;
  let o = '';
  o += tx(280, 20, 'Chapter Progression: Theory → Systems → Generalization', { fs: 15, fw: 700, fill: C.ink });

  const stages = [
    { x: 16,  y: 44,  w: 162, h: 68, sec: 'Theory', sub: 'Xw+b, lstsq, SGD', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 199, y: 44,  w: 162, h: 68, sec: 'Synthetic Data', sub: 'w*=(2,-3.4), σ=0.01', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 382, y: 44,  w: 162, h: 68, sec: 'Scratch', sub: 'Autograd & loop', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 382, y: 136, w: 162, h: 68, sec: 'OO Design', sub: 'Module · Trainer', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 199, y: 136, w: 162, h: 68, sec: 'Concise', sub: 'Dense layer & MSE', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 16,  y: 136, w: 162, h: 68, sec: 'Generalization', sub: 'Bias² + Var + CV', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  o += arrow(180, 78, 197, 78, C.blue);
  o += arrow(363, 78, 380, 78, D.orange.base);
  o += arrow(463, 114, 463, 134, C.purple);
  o += arrow(380, 170, 363, 170, C.purple);
  o += arrow(197, 170, 180, 170, C.green);
  o += arrow(97, 206, 97, 224, D.red.base);

  stages.forEach(s => {
    o += box(s.x, s.y, s.w, s.h, s.fill, s.st, 8, 2);
    o += tx(s.x + s.w / 2, s.y + 23, s.sec, { fs: 14, fw: 700, fill: s.dk });
    o += tx(s.x + s.w / 2, s.y + 47, s.sub, { mono: true, fs: 12, fw: 600, fill: C.ink });
  });

  o += box(16, 226, 528, 54, D.red.tint, D.red.base, 8, 2);
  o += tx(280, 244, 'Weight Decay (Ridge Regularization): L(w,b) + (λ/2)‖w‖₂²', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(280, 266, 'Step (1−ηλ)w  ·  MAP prior λ = σ²/(nτ²)  ·  Spectral filter μ/(μ+λ)', { fs: 12, fw: 600, fill: C.ink });

  return svg(W, H, o);
}

function lrBenchmarks() {
  return stackCards(560, 312, 'Quantitative Running Benchmarks Across the Chapter', [
    { head: '3-Point Projection', badge: 'ŵ = (7/6, -1/2)', sub: 'Points (0,1),(1,1),(2,0) → orthogonal residual Xᵀr = 0', fill: C.lblue, stroke: C.blue, dark: D.blue.dark },
    { head: 'Synthetic 2D Testbed', badge: 'floor 5e-5', sub: 'n = 1,000, w* = (2, −3.4), b* = 4.2, σ = 0.01 → SE ≈ 3.2e-4', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark },
    { head: 'Polynomial Curve Fitting', badge: 'deg 1 vs 3 vs 19', sub: 'n = 20 on cubic f*(x), σ = 0.1 → deg 3 hits floor σ² = 0.01', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
    { head: 'High-Dimensional Ridge', badge: 'λ = 1/3, df ≈ 127', sub: 'd = 200, n = 300, w* = 0.1·1, σ = 1.0 → quashes noisy directions', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
  ]);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2.1 linear-regression.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function affinePlane() {
  const W = 560, H = 268;
  let o = '';
  o += tx(280, 20, 'Affine Pricing Rule: Weighted Features + Intercept b', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 46, 160, 62, C.lblue, C.blue);
  o += tx(96, 68, 'x₁ = 150 m²', { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(96, 90, 'w₁ = +$3,000/m²', { mono: true, fs: 12, fill: C.ink });

  o += box(16, 122, 160, 62, C.lblue, C.blue);
  o += tx(96, 144, 'x₂ = 20 yrs', { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(96, 166, 'w₂ = −$1,000/yr', { mono: true, fs: 12, fill: C.ink });

  o += box(16, 198, 160, 52, C.lamber, C.amber);
  o += tx(96, 218, 'bias b = +$50k', { mono: true, fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(96, 238, 'frees origin (0,0)', { fs: 12, fill: C.muted });

  o += arrow(178, 77, 248, 134, C.blue);
  o += arrow(178, 153, 248, 148, C.blue);
  o += arrow(178, 224, 248, 162, D.orange.base);

  o += box(252, 110, 102, 76, D.purple.tint, C.purple, 10, 2);
  o += tx(303, 136, 'wᵀx + b', { mono: true, fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(303, 162, 'affine sum', { fs: 12.5, fill: C.muted });

  o += arrow(356, 148, 390, 148, C.green);

  o += box(394, 74, 150, 148, C.lgreen, C.green, 10, 2);
  o += tx(469, 100, 'Predicted ŷ', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(469, 128, '+$450,000', { mono: true, fs: 12.5, fill: C.ink });
  o += tx(469, 148, '− $20,000', { mono: true, fs: 12.5, fill: C.ink });
  o += tx(469, 168, '+ $50,000', { mono: true, fs: 12.5, fill: D.orange.dark });
  o += line(408, 182, 530, 182, C.green, 1.5);
  o += tx(469, 202, '= $480,000', { mono: true, fs: 14.5, fw: 700, fill: D.green.dark });

  return svg(W, H, o);
}

function designMatrix() {
  const W = 560, H = 272;
  let o = '';
  o += tx(280, 20, 'Batched Matrix Formulation: ŷ = Xw + b = X̃w̃', { fs: 15, fw: 700, fill: C.ink });

  const cy = 114;
  o += box(18, cy - 54, 116, 108, C.lblue, C.blue);
  o += tx(76, cy - 22, 'X', { fs: 20, fw: 700, fill: D.blue.dark });
  o += tx(76, cy + 4, 'shape (n, d)', { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  o += tx(76, cy + 28, 'n rows (x⁽ⁱ⁾)ᵀ', { fs: 12, fill: C.muted });

  o += tx(148, cy, '·', { fs: 28, fw: 700, fill: C.ink });

  o += box(164, cy - 42, 72, 84, N.panel, C.gray);
  o += tx(200, cy - 12, 'w', { fs: 19, fw: 700, fill: C.ink });
  o += tx(200, cy + 14, '(d, 1)', { mono: true, fs: 12.5, fill: C.ink });

  o += tx(252, cy, '+', { fs: 22, fw: 700, fill: C.ink });

  o += box(268, cy - 24, 80, 48, C.lamber, C.amber);
  o += tx(308, cy - 6, 'b', { fs: 18, fw: 700, fill: D.orange.dark });
  o += tx(308, cy + 14, 'broadcast', { fs: 11.5, fill: D.orange.dark });

  o += tx(366, cy, '=', { fs: 22, fw: 700, fill: C.ink });

  o += box(384, cy - 54, 158, 108, C.lgreen, C.green);
  o += tx(463, cy - 22, 'ŷ', { fs: 20, fw: 700, fill: D.green.dark });
  o += tx(463, cy + 4, 'shape (n, 1)', { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  o += tx(463, cy + 28, 'all n predictions', { fs: 12, fill: C.muted });

  o += box(18, 192, 524, 64, D.purple.tint, C.purple);
  o += tx(280, 214, 'Augmented Design Matrix: X̃ = [X, 1] ∈ ℝⁿˣ⁽ᵈ⁺¹⁾,  w̃ = (wᵀ, b)ᵀ', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(280, 238, 'Absorbs bias b into 1 fused matrix-vector product ŷ = X̃w̃', { fs: 13, fw: 600, fill: C.ink });

  return svg(W, H, o);
}

function outlierPull() {
  const W = 560, H = 266;
  let o = '';
  o += tx(280, 20, 'Why One Outlier Yanks Squared Loss (ℓ₂) But Not MAE (ℓ₁)', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 42, 254, 204, D.red.tint, D.red.base);
  o += tx(143, 66, 'Squared Loss: ½(ŷ − y)²', { fs: 14.5, fw: 700, fill: D.red.dark });
  o += tx(143, 92, 'Gradient pull ∝ residual r', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += box(28, 110, 230, 56, '#FFFFFF', D.red.base, 6, 1.5);
  o += tx(143, 128, '19 clean pts: r ≈ 0', { mono: true, fs: 12, fill: C.muted });
  o += tx(143, 150, '1 outlier (10,000): r ≈ 10⁴', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(143, 190, 'Fitted slope w_sq = 22.88', { mono: true, fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(143, 216, '(pulled 11× off true w = 2.00!)', { fs: 12.5, fw: 600, fill: D.red.dark });

  o += box(290, 42, 254, 204, C.lgreen, C.green);
  o += tx(417, 66, 'Absolute Loss (MAE): |ŷ − y|', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(417, 92, 'Subgradient = sign(r) = ±1', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += box(302, 110, 230, 56, '#FFFFFF', C.green, 6, 1.5);
  o += tx(417, 128, '19 clean pts: 19 votes of ±1', { mono: true, fs: 12, fill: D.green.dark });
  o += tx(417, 150, '1 outlier: capped at 1 vote', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(417, 190, 'Fitted slope w_mae = 2.00', { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(417, 216, '(exact recovery despite typo!)', { fs: 12.5, fw: 600, fill: D.green.dark });

  return svg(W, H, o);
}

function qrVsGram() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Solving Least Squares: Explicit Inverse vs. QR/SVD lstsq', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 44, 528, 92, D.red.tint, D.red.base);
  o += tx(280, 66, 'Anti-Pattern: inv(X.T @ X) @ X.T @ y', { mono: true, fs: 14, fw: 700, fill: D.red.dark });
  o += tx(280, 92, 'Squares condition number: κ(XᵀX) = κ(X)² → doubles float32 precision loss', { fs: 13, fw: 600, fill: C.ink });
  o += tx(280, 116, 'Fails completely when columns are collinear (d > n or duplicate features)', { fs: 12.5, fill: D.red.dark });

  o += box(16, 152, 528, 96, C.lgreen, C.green);
  o += tx(280, 174, 'Production Standard: linalg.lstsq(X, y)  (QR / SVD)', { mono: true, fs: 14, fw: 700, fill: D.green.dark });
  o += tx(280, 200, 'Works directly on X with condition number κ(X) in O(nd²) flops', { fs: 13, fw: 600, fill: C.ink });
  o += tx(280, 224, '3-point check: ŵ = (7/6, −1/2)ᵀ, orthogonality max |Xᵀr| ≤ 10⁻⁷', { mono: true, fs: 12, fw: 700, fill: D.green.dark });

  return svg(W, H, o);
}

function scalingCost() {
  return stackCards(560, 284, 'Why Deep Learning Favors Minibatch SGD Over Closed Forms', [
    { head: '1. Analytic Normal Equations / lstsq', badge: 'O(nd² + d³) flops', sub: 'At d = 10⁸ params, a d×d Gram matrix needs 40 PB of RAM!', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark },
    { head: '2. Full-Batch Gradient Descent', badge: 'O(nd) per step', sub: 'Reads all n = 10⁷ examples just to take 1 parameter update', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
    { head: '3. Minibatch Stochastic Gradient Descent', badge: 'O(|B|d) per step', sub: 'Updates in O(d) memory & O(|B|d) time, independent of n!', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
  ]);
}

function minibatchStep() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'One Minibatch SGD Iteration (Step t, Batch Size |B|)', { fs: 15, fw: 700, fill: C.ink });

  const y = 52, bh = 114, bw = 158;
  const xs = [16, 201, 386];
  o += box(xs[0], y, bw, bh, C.lblue, C.blue);
  o += tx(xs[0] + bw / 2, y + 24, '1. Sample Bₜ', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(xs[0] + bw / 2, y + 50, '|B| random rows', { fs: 12.5, fill: C.ink });
  o += tx(xs[0] + bw / 2, y + 74, 'X_B: (|B|, d)', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(xs[0] + bw / 2, y + 96, 'y_B: (|B|, 1)', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });

  o += arrow(xs[0] + bw + 2, y + bh / 2, xs[1] - 2, y + bh / 2, C.blue);

  o += box(xs[1], y, bw, bh, C.lamber, C.amber);
  o += tx(xs[1] + bw / 2, y + 24, '2. Batch Gradient', { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(xs[1] + bw / 2, y + 50, 'r = X_B w + b − y_B', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(xs[1] + bw / 2, y + 74, 'g_w = X_Bᵀ r / |B|', { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  o += tx(xs[1] + bw / 2, y + 96, 'E[g_w] = ∇_w L', { mono: true, fs: 12, fill: C.muted });

  o += arrow(xs[1] + bw + 2, y + bh / 2, xs[2] - 2, y + bh / 2, D.red.base);

  o += box(xs[2], y, bw, bh, C.lgreen, C.green);
  o += tx(xs[2] + bw / 2, y + 24, '3. Parameter Step', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(xs[2] + bw / 2, y + 50, 'w ← w − η g_w', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(xs[2] + bw / 2, y + 74, 'b ← b − η g_b', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(xs[2] + bw / 2, y + 96, 'SE(g) ∝ 1/√|B|', { mono: true, fs: 12, fill: D.green.dark });

  o += box(16, 186, 528, 60, D.purple.tint, C.purple);
  o += tx(280, 206, 'Learned Parameters vs. Tuned Hyperparameters', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 228, 'Parameters (w, b) updated by SGD  ·  Hyperparameters (η, |B|) tuned on val set', { fs: 12.5, fw: 600, fill: C.ink });

  return svg(W, H, o);
}

function loopVsGemm() {
  const W = 560, H = 266;
  let o = '';
  o += tx(280, 20, 'Why Vectorized Matrix Kernels Run 100×+ Faster Than Python Loops', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 44, 528, 92, D.red.tint, D.red.base);
  o += tx(32, 66, 'Python Loop: [X[i:i+1] @ w + b for i in range(1000)]', { mono: true, fs: 13, fw: 700, fill: D.red.dark, anchor: 'start' });
  for (let i = 0; i < 5; i++) {
    const bx = 32 + i * 74;
    o += box(bx, 84, 56, 34, '#FFFFFF', D.red.base, 5, 1.5);
    o += tx(bx + 28, 101, `row ${i}`, { mono: true, fs: 11.5, fill: D.red.dark });
    if (i < 4) o += arrow(bx + 58, 101, bx + 72, 101, D.red.base);
  }
  o += tx(408, 101, '1,000 kernel calls!', { fs: 12.5, fw: 700, fill: D.red.dark, anchor: 'start' });

  o += box(16, 152, 528, 94, C.lgreen, C.green);
  o += tx(32, 174, 'Vectorized Call: X @ w + b   (1 BLAS / GPU Kernel Launch)', { mono: true, fs: 13, fw: 700, fill: D.green.dark, anchor: 'start' });
  o += box(32, 192, 276, 38, '#FFFFFF', C.green, 6, 1.8);
  o += tx(170, 211, 'All 1,000 rows in SIMD / Tensor Cores', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += arrow(312, 211, 348, 211, C.green);
  o += box(352, 192, 174, 38, '#FFFFFF', C.green, 6, 1.8);
  o += tx(439, 211, '100×–1,000× speedup', { fs: 13, fw: 700, fill: D.green.dark });

  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2.2 synthetic-regression-data.md (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function synthOracle() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Real-World Ambiguity vs. Synthetic Ground-Truth Oracle', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 44, 254, 202, D.red.tint, D.red.base);
  o += tx(143, 68, 'Real Dataset (No Answer Key)', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(143, 98, '• True w*, b* unknown', { fs: 13, fill: C.ink });
  o += tx(143, 124, '• Noise level σ² unknown', { fs: 13, fill: C.ink });
  o += tx(143, 150, '• Linear assumption approximate', { fs: 13, fill: C.ink });
  o += box(30, 174, 226, 54, '#FFFFFF', D.red.base, 6, 1.5);
  o += tx(143, 192, 'If loss plateaus high:', { fs: 12, fw: 700, fill: D.red.dark });
  o += tx(143, 212, 'Code bug or bad model?', { fs: 12.5, fw: 700, fill: D.red.dark });

  o += box(290, 44, 254, 202, C.lgreen, C.green);
  o += tx(417, 68, `Synthetic Benchmark (${sec('sec_synthetic-regression-data')})`, { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(417, 98, '• w* = (2, −3.4)ᵀ, b* = 4.2', { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  o += tx(417, 124, '• Noise σ = 0.01 (floor 5e-5)', { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  o += tx(417, 150, '• Linear model is exact truth', { fs: 13, fill: C.ink });
  o += box(304, 174, 226, 54, '#FFFFFF', C.green, 6, 1.5);
  o += tx(417, 192, 'Pass/Fail Unit Test:', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(417, 212, '‖ŵ − w*‖∞ ≲ 3σ/√n ≈ 10⁻³', { mono: true, fs: 12, fw: 700, fill: D.green.dark });

  return svg(W, H, o);
}

function synthDgp() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Synthetic Data-Generating Process (2,000 Examples)', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 46, 180, 72, C.lblue, C.blue);
  o += tx(106, 68, 'Features X ~ N(0, I₂)', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(106, 94, 'Signal std ≈ 3.94', { fs: 12.5, fill: C.ink });

  o += box(16, 134, 180, 72, C.lamber, C.amber);
  o += tx(106, 156, 'Noise ε ~ N(0, 0.01²)', { mono: true, fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(106, 182, '400× smaller than signal', { fs: 12.5, fill: C.ink });

  o += arrow(198, 82, 234, 114, C.blue);
  o += arrow(198, 170, 234, 136, D.orange.base);

  o += box(238, 86, 144, 80, D.purple.tint, C.purple);
  o += tx(310, 114, 'y = Xw* + b* + ε', { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });
  o += tx(310, 140, 'w*=(2,−3.4), b*=4.2', { mono: true, fs: 11, fill: C.ink });

  o += arrow(384, 110, 412, 82, C.green);
  o += arrow(384, 142, 412, 170, C.purple);

  o += box(416, 46, 128, 72, C.lgreen, C.green);
  o += tx(480, 70, 'Train Split', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(480, 94, 'n = 1,000', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += box(416, 134, 128, 72, D.purple.tint, C.purple);
  o += tx(480, 158, 'Val Split', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(480, 182, 'n\' = 1,000', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += tx(280, 234, 'Both splits come from the exact same i.i.d. distribution P(x, y)', { fs: 13, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

function synthShapes() {
  const W = 560, H = 250;
  let o = '';
  o += tx(280, 20, 'Shape Alignment in synthetic_data(w, b, noise, n)', { fs: 15, fw: 700, fill: C.ink });

  const cy = 114;
  o += box(18, cy - 46, 96, 92, C.lblue, C.blue);
  o += tx(66, cy - 14, 'X', { mono: true, fs: 18, fw: 700, fill: D.blue.dark });
  o += tx(66, cy + 14, '(n, 2)', { mono: true, fs: 13, fill: C.ink });

  o += tx(128, cy, '@', { mono: true, fs: 19, fw: 700, fill: C.ink });

  o += box(144, cy - 46, 128, 92, D.purple.tint, C.purple);
  o += tx(208, cy - 14, 'w.reshape(-1,1)', { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });
  o += tx(208, cy + 14, '(2, 1) col!', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += tx(286, cy, '+', { fs: 20, fw: 700, fill: C.ink });

  o += box(302, cy - 46, 98, 92, C.lamber, C.amber);
  o += tx(351, cy - 14, 'b + eps', { mono: true, fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(351, cy + 14, '(n, 1)', { mono: true, fs: 13, fill: C.ink });

  o += tx(416, cy, '=', { fs: 20, fw: 700, fill: C.ink });

  o += box(432, cy - 46, 110, 92, C.lgreen, C.green);
  o += tx(487, cy - 14, 'y', { mono: true, fs: 18, fw: 700, fill: D.green.dark });
  o += tx(487, cy + 14, '(n, 1)', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += box(18, 184, 524, 48, D.red.tint, D.red.base);
  o += tx(280, 208, 'Without w.reshape(-1, 1): (n,) + (n, 1) broadcasts to (n, n)!', { mono: true, fs: 13, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

function jaxKeySplit() {
  const W = 560, H = 250;
  let o = '';
  o += tx(280, 20, 'Stateless PRNG Key Splitting in JAX', { fs: 15, fw: 700, fill: C.ink });

  o += box(20, 90, 126, 64, D.purple.tint, C.purple);
  o += tx(83, 112, 'input key', { mono: true, fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(83, 134, 'uint32[2]', { mono: true, fs: 12, fill: C.muted });

  o += arrow(148, 122, 190, 122, C.purple);

  o += box(194, 82, 142, 80, N.panel, C.gray);
  o += tx(265, 110, 'random.split', { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(265, 136, '2 subkeys', { fs: 12, fill: C.muted });

  o += arrow(338, 102, 378, 74, C.blue);
  o += arrow(338, 142, 378, 170, D.orange.base);

  o += box(382, 42, 158, 64, C.lblue, C.blue);
  o += tx(461, 64, 'key1 → X', { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(461, 86, 'normal(key1,(n,d))', { mono: true, fs: 11.5, fill: C.ink });

  o += box(382, 138, 158, 64, C.lamber, C.amber);
  o += tx(461, 160, 'key2 → eps', { mono: true, fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(461, 182, 'normal(key2,(n,1))', { mono: true, fs: 11.5, fill: C.ink });

  o += tx(280, 228, 'Reusing one key would make eps repeat the first n draws of X (row-major)!', { fs: 12.5, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

function noiseFloor() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'The Irreducible Loss Floor at the True Parameters (w*, b*)', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 46, 246, 90, D.red.tint, D.red.base);
  o += tx(139, 70, 'At Initialization (w=0, b=0)', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(139, 96, 'E[L(0, 0)] = ½(‖w*‖² + b*² + σ²)', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(139, 118, '≈ 16.6', { mono: true, fs: 14.5, fw: 700, fill: D.red.dark });

  o += arrow(264, 91, 294, 91, C.green);

  o += box(298, 46, 246, 90, C.lgreen, C.green);
  o += tx(421, 70, 'At True Oracle (w*, b*)', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(421, 96, 'E[L(w*, b*)] = ½ E[ε²] = ½σ²', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(421, 118, '= 5 × 10⁻⁵', { mono: true, fs: 14.5, fw: 700, fill: D.green.dark });

  o += box(16, 156, 528, 84, C.lblue, C.blue);
  o += tx(280, 180, 'Why No Predictor Can Beat 5 × 10⁻⁵ on Unseen Data', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(280, 204, 'Future noise ε ~ N(0, σ²) is independent of x; on n\' = 1,000 val points,', { fs: 12.5, fill: C.ink });
  o += tx(280, 224, 'val loss fluctuates around 5×10⁻⁵ with SE = σ²/√(2n\') ≈ 2.2×10⁻⁶', { mono: true, fs: 12, fw: 600, fill: D.blue.dark });

  return svg(W, H, o);
}

function recoveryDecomp() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Exact Least-Squares Decomposition: Signal Cancels, Noise Remains', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 44, 528, 56, C.lblue, C.blue);
  o += tx(280, 64, '1. True labels:  y = X̃w̃* + ε', { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(280, 84, 'Plug into normal equations: w̃̂ = (X̃ᵀX̃)⁻¹ X̃ᵀ (X̃w̃* + ε)', { mono: true, fs: 12.5, fill: C.ink });

  o += arrow(280, 102, 280, 122, C.purple);

  o += box(16, 124, 528, 56, D.purple.tint, C.purple);
  o += tx(280, 144, '2. Signal cancels:  (X̃ᵀX̃)⁻¹(X̃ᵀX̃)w̃* = w̃*', { mono: true, fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 164, 'Estimation error:   w̃̂ − w̃* = (X̃ᵀX̃)⁻¹ X̃ᵀ ε', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += arrow(280, 182, 280, 202, C.green);

  o += box(16, 204, 528, 42, C.lgreen, C.green);
  o += tx(280, 225, 'Linear in ε → unbiased E[w̃̂] = w̃*, exact recovery when σ = 0!', { fs: 13, fw: 700, fill: D.green.dark });

  return svg(W, H, o);
}

function gramConcentration() {
  const W = 560, H = 262;
  let o = '';
  o += tx(280, 20, 'Isotropic Features: Sample Gram Matrix Concentrates at I₃', { fs: 15, fw: 700, fill: C.ink });

  const s = 44, g = 4, gx = 34, gy = 66;
  o += tx(gx + 1.5 * s + g, gy - 18, 'X̃ᵀX̃ / n  (n = 1,000)', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  const G = [['1.02', '−0.01', '0.02'], ['−0.01', '0.98', '−0.02'], ['0.02', '−0.02', '1.00']];
  o += grid(G, gx, gy, s, g, {
    fs: 12,
    fill: (r, c) => r === c ? [C.lblue, C.blue, false, D.blue.dark] : [N.panel, N.hairline, false, C.muted],
  });

  o += tx(206, gy + 1.5 * s + g, '≈', { fs: 28, fw: 700, fill: C.ink });

  const gx2 = 236;
  o += tx(gx2 + 1.5 * s + g, gy - 18, 'Population E[x̃x̃ᵀ] = I₃', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  const I = [['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1']];
  o += grid(I, gx2, gy, s, g, {
    fs: 14,
    fill: (r, c) => r === c ? [C.lgreen, C.green, false, D.green.dark] : ['#FFFFFF', N.hairline, false, C.muted],
  });

  o += box(398, 54, 144, 152, D.purple.tint, C.purple);
  o += tx(470, 80, 'Decoupled', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(470, 102, 'Error:', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(470, 132, 'w̃̂ − w̃* ≈', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(470, 156, '(1/n) X̃ᵀε', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(470, 184, '3 independent', { fs: 12, fill: C.muted });

  o += tx(280, 234, 'Diagonal entries average xⱼ² → 1; off-diagonals average xⱼxₖ → 0', { fs: 13, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function seScaling() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'The σ / √n Parameter Recovery Law (SE ≈ 3.16 × 10⁻⁴)', { fs: 15, fw: 700, fill: C.ink });

  const bars = [
    { label: 'n = 10',     se: '3.2 × 10⁻³', w: 252, col: D.red.base, tint: D.red.tint, dk: D.red.dark },
    { label: 'n = 100',    se: '1.0 × 10⁻³', w: 192, col: C.amber,    tint: C.lamber,   dk: D.orange.dark },
    { label: 'n = 1,000',  se: '3.2 × 10⁻⁴', w: 132, col: C.blue,     tint: C.lblue,    dk: D.blue.dark },
    { label: 'n = 10,000', se: '1.0 × 10⁻⁴', w: 72,  col: C.green,    tint: C.lgreen,   dk: D.green.dark },
  ];

  o += tx(544, 38, 'bar length: log scale', { fs: 11.5, fill: C.muted, anchor: 'end' });
  bars.forEach((b, i) => {
    const y = 46 + i * 42;
    o += tx(110, y + 15, b.label, { mono: true, fs: 13.5, fw: 700, fill: C.ink, anchor: 'end' });
    o += box(122, y, b.w, 30, b.tint, b.col, 6, 2);
    o += tx(122 + b.w + 12, y + 15, `SE = ${b.se}`, { mono: true, fs: 13, fw: 700, fill: b.dk, anchor: 'start' });
  });

  o += box(16, 216, 528, 36, D.purple.tint, C.purple);
  o += tx(280, 234, '10× smaller error requires 100× more training examples (Cov(w̃̂) = σ²(X̃ᵀX̃)⁻¹)', { fs: 12.5, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2.3 linear-regression-scratch.md (10 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function broadcastTrap() {
  const W = 560, H = 272;
  let o = '';
  o += tx(280, 20, 'Silent Shape Bug: Why Loss Can Drop While Weights Fail', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 44, 254, 210, D.red.tint, D.red.base);
  o += tx(143, 66, 'Bug: y\\_hat(32,1) − y(32,)', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(143, 92, 'Broadcasts to (32, 32) matrix', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(143, 114, 'of all 1,024 pairwise diffs!', { fs: 12.5, fill: C.ink });
  o += box(28, 132, 230, 56, '#FFFFFF', D.red.base, 6, 1.5);
  o += tx(143, 150, 'Minimizing mean(32×32) pulls', { fs: 12, fill: C.ink });
  o += tx(143, 170, 'ŷᵢ → ȳ (mean) and w → 0!', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(143, 212, 'Loss drops smoothly to ~7.8,', { fs: 12.5, fw: 600, fill: D.red.dark });
  o += tx(143, 234, 'hiding the bug without oracle!', { fs: 12, fw: 700, fill: D.red.dark });

  o += box(290, 44, 254, 210, C.lgreen, C.green);
  o += tx(417, 66, 'Fix: y.reshape((-1, 1))', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(417, 92, '(32, 1) − (32, 1) → (32, 1)', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(417, 114, '32 aligned per-example residuals', { fs: 12.5, fill: C.ink });
  o += box(302, 132, 230, 56, '#FFFFFF', C.green, 6, 1.5);
  o += tx(417, 150, 'Val loss hits 5 × 10⁻⁵', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(417, 170, 'ŵ → (2.0, −3.4), b̂ → 4.2', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(417, 212, 'Dual verification: check BOTH', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(417, 234, 'loss floor and weight recovery!', { fs: 12, fw: 600, fill: C.ink });

  return svg(W, H, o);
}

function paramInit() {
  return stackCards(560, 284, 'Allocating Trainable Parameters for Autograd', [
    { head: 'Weights w ~ N(0, 0.01²) · shape (2, 1)', badge: 'b = zeros(1)', sub: 'Small random Gaussian weights + zero bias (breaks symmetry in deeper nets)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark },
    { head: 'Differentiable Parameters', badge: '∂L/∂w, ∂L/∂b', sub: 'Created so that the backward pass can return the gradient of the loss for both', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark },
  ]);
}

function defensiveReshape() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Forward Pass & Defensive Reshape in squared_loss', { fs: 15, fw: 700, fill: C.ink });

  const cy = 104;
  o += box(16, cy - 38, 140, 76, C.lblue, C.blue);
  o += tx(86, cy - 12, 'linreg(X,w,b)', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(86, cy + 14, 'y\\_hat: (32, 1)', { mono: true, fs: 12.5, fill: C.ink });

  o += arrow(158, cy, 188, cy, C.blue);

  o += box(192, cy - 38, 176, 76, C.lamber, C.amber);
  o += tx(280, cy - 12, 'reshape(y, (-1, 1))', { mono: true, fs: 12, fw: 700, fill: D.orange.dark });
  o += tx(280, cy + 14, '(32,) → (32, 1) guard', { mono: true, fs: 12, fill: C.ink });

  o += arrow(370, cy, 400, cy, C.green);

  o += box(404, cy - 38, 140, 76, C.lgreen, C.green);
  o += tx(474, cy - 12, 'reduce_mean', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(474, cy + 14, '(r ** 2) / 2', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += box(16, 172, 528, 64, D.purple.tint, C.purple);
  o += tx(280, 194, 'Contract: y_hat and y.reshape((-1, 1)) both have shape (|B|, 1)', { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 218, 'Guarantees elementwise residual vector r ∈ ℝ^{|B|×1} before squaring', { fs: 12.5, fill: C.ink });

  return svg(W, H, o);
}

function gradVerify() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Cross-Checking Autograd Against Analytical Matrix Calculus', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 48, 232, 118, C.lblue, C.blue);
  o += tx(132, 72, 'Analytical Formula', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(132, 98, 'r = Xb @ w + b − yb', { mono: true, fs: 12, fill: C.ink });
  o += tx(132, 122, '∇_w L = Xbᵀ r / 32', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(132, 144, '∂_{b} L = Σ r / 32', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });

  o += box(312, 48, 232, 118, D.purple.tint, C.purple);
  o += tx(428, 72, 'Reverse-Mode Autograd', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(428, 98, 'backward sweep', { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });
  o += tx(428, 122, 'returns ∂L/∂w, ∂L/∂b', { mono: true, fs: 12, fill: C.ink });

  o += arrow(250, 107, 310, 107, C.green);
  o += arrow(310, 107, 250, 107, C.green);

  o += box(16, 186, 528, 54, C.lgreen, C.green);
  o += tx(280, 206, 'Max Entrywise Discrepancy: |by_hand − by_autograd| ~ 10⁻⁷', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(280, 226, 'Exact match up to float32 machine epsilon on a 32-example minibatch', { fs: 12.5, fill: C.ink });

  return svg(W, H, o);
}

function reshuffleBias() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Why Reshuffling Every Epoch Preserves Unbiased Gradients', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 44, 528, 94, C.lgreen, C.green);
  o += tx(32, 66, 'Random Permutation (Unbiased: E[∇L_B] = ∇L)', { fs: 14, fw: 700, fill: D.green.dark, anchor: 'start' });
  o += tx(32, 92, 'Batch mean label ȳ_B ≈ 4.2 ± 0.7  →  ∂_{b} L_B ≈ −4.2', { mono: true, fs: 12.5, fw: 600, fill: C.ink, anchor: 'start' });
  o += tx(32, 118, 'Every step pushes b steadily upward from 0 toward b* = 4.2', { fs: 12.5, fw: 700, fill: D.green.dark, anchor: 'start' });

  o += box(16, 154, 528, 94, D.red.tint, D.red.base);
  o += tx(32, 176, 'Sorted by Label Without Reshuffling (Biased Trajectory!)', { fs: 14, fw: 700, fill: D.red.dark, anchor: 'start' });
  o += tx(32, 202, 'Batch 1 (32 smallest y): ȳ_B ≈ −4.6  →  ∂_{b} L_B ≈ +4.6', { mono: true, fs: 12.5, fw: 600, fill: C.ink, anchor: 'start' });
  o += tx(32, 228, 'Steps b downward away from b* = 4.2, cycling wildly every epoch!', { fs: 12.5, fw: 700, fill: D.red.dark, anchor: 'start' });

  return svg(W, H, o);
}

function genExhaustion() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, '1,000 Examples at batch_size=32 & Single-Pass Generator Pitfall', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 44, 528, 86, C.lblue, C.blue);
  o += tx(280, 66, 'Pass 1: ⌈1000 / 32⌉ = 32 Minibatches (31 × 32 + 1 × 8)', { fs: 13.5, fw: 700, fill: D.blue.dark });
  const bx0 = 32, by = 80;
  for (let i = 0; i < 4; i++) {
    o += box(bx0 + i * 96, by, 84, 36, '#FFFFFF', C.blue, 6, 1.5);
    o += tx(bx0 + i * 96 + 42, by + 18, `B${'₁₂₃₄'[i]} (32)`, { mono: true, fs: 12, fw: 700, fill: C.ink });
  }
  o += tx(430, by + 18, '···', { fs: 18, fw: 700, fill: C.ink });
  o += box(452, by, 76, 36, C.lamber, C.amber, 6, 1.8);
  o += tx(490, by + 18, 'B₃₂ (8)', { mono: true, fs: 12, fw: 700, fill: D.orange.dark });

  o += box(16, 146, 254, 96, D.red.tint, D.red.base);
  o += tx(143, 170, 'Reusing loader = minibatches()', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(143, 196, 'Pass 2 yields 0 batches!', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(143, 220, 'Training stops after epoch 1', { fs: 12, fill: C.ink });

  o += box(290, 146, 254, 96, C.lgreen, C.green);
  o += tx(417, 170, 'Call minibatches() Per Epoch', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(417, 196, 'Fresh generator & shuffle', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(417, 220, 'Or a reusable data loader', { fs: 12, fill: C.ink });

  return svg(W, H, o);
}

function fourStepLoop() {
  const W = 560, H = 266;
  let o = '';
  o += tx(280, 20, 'The Four-Step Minibatch Update Cycle', { fs: 15, fw: 700, fill: C.ink });

  const bw = 248, bh = 86;
  o += box(16, 42, bw, bh, C.lblue, C.blue);
  o += tx(16 + bw / 2, 64, '1. Forward Pass', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(16 + bw / 2, 88, 'y_hat = linreg(Xb, w, b)', { mono: true, fs: 12, fill: C.ink });
  o += tx(16 + bw / 2, 108, 'recorded for differentiation', { fs: 12, fill: C.ink });

  o += arrow(266, 85, 294, 85, C.blue);

  o += box(296, 42, bw, bh, C.lamber, C.amber);
  o += tx(296 + bw / 2, 64, '2. Loss', { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(296 + bw / 2, 88, 'loss = squared_loss(y_hat, yb)', { mono: true, fs: 12, fill: C.ink });
  o += tx(296 + bw / 2, 108, 'one scalar per minibatch', { fs: 12, fill: C.ink });

  o += arrow(420, 130, 420, 154, D.red.base);

  o += box(296, 158, bw, bh, D.purple.tint, C.purple);
  o += tx(296 + bw / 2, 180, '3. Backward Pass', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(296 + bw / 2, 204, 'gradients of loss', { mono: true, fs: 12, fw: 700, fill: D.purple.dark });
  o += tx(296 + bw / 2, 224, 'with respect to w and b', { fs: 12, fill: C.ink });

  o += arrow(294, 201, 266, 201, D.orange.base);

  o += box(16, 158, bw, bh, C.lgreen, C.green);
  o += tx(16 + bw / 2, 180, '4. Update', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(16 + bw / 2, 204, 'param ← param − lr * grad', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(16 + bw / 2, 224, 'not recorded; next minibatch', { fs: 12, fill: C.ink });

  o += arrow(140, 156, 140, 130, C.blue);

  return svg(W, H, o);
}

function contractionRate() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Theoretical Excess-Loss Contraction: 7× Drop Per Epoch (η = 0.03)', { fs: 14.5, fw: 700, fill: C.ink });

  const bw = 160, bh = 104, y = 46;
  const xs = [16, 200, 384];

  o += box(xs[0], y, bw, bh, C.lblue, C.blue);
  o += tx(xs[0] + bw / 2, y + 24, 'Epoch 0 (Init)', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(xs[0] + bw / 2, y + 52, 'Excess loss ≈ 16.6', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(xs[0] + bw / 2, y + 78, '½(‖w*‖² + b*²)', { mono: true, fs: 12, fill: C.muted });

  o += arrow(xs[0] + bw + 2, y + bh / 2, xs[1] - 2, y + bh / 2, C.purple);

  o += box(xs[1], y, bw, bh, D.purple.tint, C.purple);
  o += tx(xs[1] + bw / 2, y + 24, '1 Step (η = 0.03)', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(xs[1] + bw / 2, y + 52, 'Error × (1 − η)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(xs[1] + bw / 2, y + 78, 'Loss × 0.97²', { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });

  o += arrow(xs[1] + bw + 2, y + bh / 2, xs[2] - 2, y + bh / 2, C.green);

  o += box(xs[2], y, bw, bh, C.lgreen, C.green);
  o += tx(xs[2] + bw / 2, y + 24, '1 Epoch (32 steps)', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(xs[2] + bw / 2, y + 52, '0.97⁶⁴ ≈ 0.142', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(xs[2] + bw / 2, y + 78, '≈ 1/7 per epoch!', { fs: 13, fw: 700, fill: C.ink });

  o += box(16, 170, 528, 74, C.lamber, C.amber);
  o += tx(280, 194, 'After 7 Epochs: 16.6 × 0.142⁷ ≈ 2×10⁻⁵ < Floor 5×10⁻⁵', { mono: true, fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(280, 220, 'Straight line on log-loss plot until hitting the irreducible floor at epoch 7–8', { fs: 12.5, fill: C.ink });

  return svg(W, H, o);
}

function paramRecovery() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Why Checking Loss Alone Is Blind to 10⁻³ Parameter Drift', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 46, 254, 190, C.lblue, C.blue);
  o += tx(143, 70, 'Validation Loss Lens', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(143, 100, 'Noise floor: ½σ² = 5 × 10⁻⁵', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(143, 128, 'Param drift δ = 5 × 10⁻⁴ adds:', { fs: 12.5, fill: C.ink });
  o += tx(143, 154, 'ΔL = ½ · 3 · δ² ≈ 3.8 × 10⁻⁷', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(143, 186, 'Only 0.7% of the noise floor!', { fs: 13, fw: 700, fill: D.red.dark });
  o += tx(143, 210, '(Invisible on a loss plot)', { fs: 12, fill: C.muted });

  o += box(290, 46, 254, 190, C.lgreen, C.green);
  o += tx(417, 70, 'Parameter Error Lens', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(417, 100, 'Compare (ŵ, b̂) vs (w*, b*)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(417, 128, '|ŵ₁ − 2.0|   ~ 3 × 10⁻⁴', { mono: true, fs: 12.5, fill: D.green.dark });
  o += tx(417, 154, '|ŵ₂ + 3.4|   ~ 3 × 10⁻⁴', { mono: true, fs: 12.5, fill: D.green.dark });
  o += tx(417, 186, 'Matches SE = σ/√n ≈ 3.2×10⁻⁴', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(417, 210, '(Direct proof of correctness)', { fs: 12, fill: C.ink });

  return svg(W, H, o);
}

function lrRegimes() {
  return stackCards(560, 286, 'Three Learning-Rate Regimes on a Quadratic Bowl (μ_max ≈ 1)', [
    { head: 'Too Small: η = 0.003 (Crawling)', badge: 'val loss ≈ 2–3', sub: 'Loss contracts by 0.997²/step → after 320 steps, 16.6 × 0.997⁶⁴⁰ ≈ 2.4', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
    { head: 'Well Tuned: η = 0.03 to 0.3', badge: 'val loss ≈ 5e-5', sub: 'Reaches the irreducible noise floor σ²/2 = 5 × 10⁻⁵ in 1–7 epochs', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
    { head: 'Unstable: η = 3 > 2 / μ_max', badge: 'val loss → NaN', sub: 'Multiplies error by (1 − ημ) ≈ −2 per step → oscillates & diverges!', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark },
  ]);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2.4 oo-design.md (9 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function ooDecouple() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Why Factor Training Into Three Orthogonal Abstractions?', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 46, 246, 198, D.red.tint, D.red.base);
  o += tx(139, 70, 'Monolithic Loop', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(139, 98, 'Model + Data + Loop coupled', { fs: 12.5, fill: C.ink });
  o += box(30, 116, 218, 62, '#FFFFFF', D.red.base, 6, 1.5);
  o += tx(139, 138, '30 Models × 10 Datasets', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(139, 160, '= 300 custom train loops!', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(139, 206, 'Adding grad clipping or GPUs', { fs: 12, fill: C.ink });
  o += tx(139, 226, 'means editing every file', { fs: 12, fill: D.red.dark });

  o += arrow(264, 145, 294, 145, C.green);

  o += box(298, 46, 246, 198, C.lgreen, C.green);
  o += tx(421, 70, 'd2l OO Harness', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(421, 98, 'trainer.fit(model, data)', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += box(312, 116, 218, 62, '#FFFFFF', C.green, 6, 1.5);
  o += tx(421, 138, '30 Modules + 10 DataModules', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(421, 160, '+ 1 universal Trainer!', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(421, 206, 'Swap models or datasets without', { fs: 12, fill: C.ink });
  o += tx(421, 226, 'touching a line of Trainer', { fs: 12, fill: D.green.dark });

  return svg(W, H, o);
}

function ooUtilities() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Two Ergonomic Utilities: @add_to_class & save_hyperparameters', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 46, 254, 194, C.lblue, C.blue);
  o += tx(143, 70, '@d2l.add_to_class(Class)', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(143, 98, 'setattr(Class, fn.__name__, fn)', { mono: true, fs: 11, fill: C.ink });
  o += box(30, 116, 226, 60, '#FFFFFF', C.blue, 6, 1.5);
  o += tx(143, 138, 'Attaches methods across cells', { fs: 12, fw: 600, fill: C.ink });
  o += tx(143, 158, 'Live instances gain fn()!', { fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(143, 206, 'Avoids 60-line monolithic', { fs: 12, fill: C.muted });
  o += tx(143, 224, 'class blocks in notebooks', { fs: 12, fill: C.muted });

  o += box(290, 46, 254, 194, D.purple.tint, C.purple);
  o += tx(417, 70, 'self.save_hyperparameters()', { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });
  o += tx(417, 98, 'Inspects __init__ stack frame', { fs: 12, fill: C.ink });
  o += box(304, 116, 226, 60, '#FFFFFF', C.purple, 6, 1.5);
  o += tx(417, 138, 'Binds args → self.lr, self.w', { mono: true, fs: 11.5, fw: 700, fill: D.purple.dark });
  o += tx(417, 158, 'Supports ignore=[...]', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(417, 206, 'Eliminates repetitive', { fs: 12, fill: C.muted });
  o += tx(417, 224, 'self.x = x boilerplate', { mono: true, fs: 12, fill: C.muted });

  return svg(W, H, o);
}

function moduleContract() {
  const W = 560, H = 282;
  let o = '';
  o += tx(280, 20, 'The d2l.Module Pure Return Contract (Zero Trainer Coupling)', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 50, 144, 82, C.lblue, C.blue);
  o += tx(88, 74, 'batch tuple', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(88, 98, 'batch[:-1] = (X,)', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(88, 118, 'batch[-1]  = y', { mono: true, fs: 11.5, fill: C.ink });

  o += arrow(162, 91, 194, 91, C.blue);

  o += box(198, 42, 168, 98, D.purple.tint, C.purple);
  o += tx(282, 64, 'd2l.Module', { mono: true, fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(282, 88, 'ŷ = self(*batch[:-1])', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(282, 108, 'l = self.loss(ŷ, y)', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(282, 126, 'training / val_step', { mono: true, fs: 11, fill: D.purple.dark });

  o += arrow(368, 91, 400, 91, C.green);

  o += box(404, 50, 140, 82, C.lgreen, C.green);
  o += tx(474, 74, 'as_metrics(out)', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(474, 98, '{\'loss\': l}', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(474, 118, '(+ \'acc\' for classifiers)', { fs: 11.5, fill: C.muted });

  o += box(16, 158, 528, 106, C.lamber, C.amber);
  o += tx(280, 182, 'configure_optimizers() → Returns Optimizer (+ Scheduler)', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(280, 208, '• Module never prints, plots, or mutates Trainer state directly', { fs: 13, fw: 600, fill: C.ink });
  o += tx(280, 232, '• Any custom loop or Trainer can drive Module without side effects', { fs: 13, fw: 600, fill: C.ink });

  return svg(W, H, o);
}

function scratchModule() {
  return stackCards(560, 284, 'LinearRegressionScratch(d2l.Module) Anatomy', [
    { head: '1. __init__(num_inputs, lr, sigma=0.01)', badge: 'registered params', sub: 'Registers self.w ~ N(0, σ²) of shape (d, 1) and self.b = zeros(1)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark },
    { head: '2. forward(self, X)', badge: 'matmul(X, w) + b', sub: 'Computes affine prediction ŷ ∈ ℝ^{|B|×1} from registered parameters', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark },
    { head: '3. loss(self, y\\_hat, y)', badge: 'reduce_mean(r²/2)', sub: 'Reshapes y to (-1, 1) and returns scalar half mean squared error', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
  ]);
}

function sgdClass() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Wrapping From-Scratch SGD in the Optimizer Interface', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 46, 254, 192, C.lblue, C.blue);
  o += tx(143, 70, 'The Update Rule', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(143, 98, 'For every registered parameter:', { fs: 12, fill: C.ink });
  o += box(30, 114, 226, 64, '#FFFFFF', C.blue, 6, 1.5);
  o += tx(143, 136, 'param ← param − lr · grad', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(143, 158, 'lr saved as a hyperparameter', { fs: 12, fill: C.ink });
  o += tx(143, 206, 'No state beyond lr', { fs: 12, fill: C.muted });

  o += box(290, 46, 254, 192, C.lgreen, C.green);
  o += tx(417, 70, 'The Optimizer Interface', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(417, 98, 'Same calls as built-in optimizers:', { fs: 12, fill: C.ink });
  o += box(304, 114, 226, 64, '#FFFFFF', C.green, 6, 1.5);
  o += tx(417, 136, 'Trainer applies one update', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(417, 158, 'per minibatch', { fs: 12, fw: 700, fill: D.green.dark });
  o += tx(417, 206, 'Swap in a built-in SGD later', { fs: 12, fill: C.muted });

  return svg(W, H, o);
}

function reusableLoader() {
  const W = 560, H = 262;
  let o = '';
  o += tx(280, 20, 'Single-Use Generator vs. Reusable Multi-Epoch Loader', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 46, 254, 196, D.red.tint, D.red.base);
  o += tx(143, 70, 'Python Generator (yield)', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(143, 100, '• Epoch 1: yields 32 batches', { fs: 12.5, fill: C.ink });
  o += tx(143, 126, '• Epoch 2: yields 0 batches!', { fs: 13, fw: 700, fill: D.red.dark });
  o += tx(143, 152, '• len(gen) → TypeError', { mono: true, fs: 12, fill: D.red.dark });
  o += tx(143, 180, '• Single-threaded in-RAM only', { fs: 12.5, fill: C.muted });

  o += box(290, 46, 254, 196, C.lgreen, C.green);
  o += tx(417, 70, 'Reusable Data Loader', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(417, 100, '• __iter__() reshuffles each pass', { mono: true, fs: 11.5, fw: 700, fill: D.green.dark });
  o += tx(417, 126, '• Epoch 1, 2, ...: all 32 batches', { fs: 12.5, fill: C.ink });
  o += tx(417, 152, '• len(loader) == 32 in O(1)', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(417, 180, '• One loader serves every epoch', { fs: 12.5, fill: C.ink });

  return svg(W, H, o);
}

function averagerWeights() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Why Epoch Means Must Weight Each Batch by Its Size n_b', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 44, 356, 62, C.lblue, C.blue);
  o += tx(194, 66, 'Batches 1..31: n_b = 32 each (992 examples)', { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(194, 88, 'Suppose mean loss L̄_b = 0.010', { mono: true, fs: 12, fill: C.ink });

  o += box(386, 44, 158, 62, C.lamber, C.amber);
  o += tx(465, 66, 'Batch 32: n₃₂ = 8', { fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(465, 88, 'L̄₃₂ = 0.050', { mono: true, fs: 12, fw: 700, fill: D.orange.dark });

  o += box(16, 124, 254, 122, D.red.tint, D.red.base);
  o += tx(143, 148, 'Unweighted Mean of 32 Batches', { fs: 13, fw: 700, fill: D.red.dark });
  o += tx(143, 174, '(31×0.010 + 1×0.050)/32', { mono: true, fs: 12, fill: C.ink });
  o += tx(143, 198, '= 0.01125  (+9% error!)', { mono: true, fs: 13, fw: 700, fill: D.red.dark });
  o += tx(143, 222, 'Last 8 pts weighted 4× too much', { fs: 12, fill: D.red.dark });

  o += box(290, 124, 254, 122, C.lgreen, C.green);
  o += tx(417, 148, 'd2l.Averager (Size-Weighted)', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(417, 174, '(992×0.010 + 8×0.050)/1000', { mono: true, fs: 12, fill: C.ink });
  o += tx(417, 198, '= 0.01032  (Exact mean!)', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(417, 222, 'Every example gets weight 1/1000', { fs: 12, fill: D.green.dark });

  return svg(W, H, o);
}

function trainerTimeline() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Trainer Epoch Timeline: Mid-Epoch Train vs. End-of-Epoch Val', { fs: 14.5, fw: 700, fill: C.ink });

  const y = 110;
  o += line(44, y, 516, y, C.gray, 3);

  o += line(64, y - 12, 64, y + 12, C.ink, 2.5);
  o += tx(64, y + 32, 'epoch e', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += box(150, 42, 260, 48, C.lblue, C.blue);
  o += tx(280, 60, 'record("train", fit_epoch())', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(280, 78, 'averages all steps → x = e + 0.5', { mono: true, fs: 11.5, fill: C.ink });
  o += arrow(280, 92, 280, y - 4, C.blue);

  o += line(496, y - 12, 496, y + 12, C.green, 2.5);
  o += tx(496, y - 26, 'x = e + 1.0', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += box(284, 152, 250, 56, C.lgreen, C.green);
  o += tx(409, 172, 'record("val", validate())', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(409, 192, 'evaluates end-of-epoch weights', { fs: 12, fill: C.ink });
  o += arrow(496, 150, 496, y + 14, C.green);

  o += tx(280, 232, 'Explains why epoch-1 train loss (x=0.5) often sits above val loss (x=1.0)!', { fs: 12.5, fw: 600, fill: D.purple.dark });
  return svg(W, H, o);
}

function detachGraph() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Why Trainer Detaches Logged Metrics', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 46, 254, 194, D.red.tint, D.red.base);
  o += tx(143, 70, 'Without .detach() in Averager', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(143, 98, 'sums += out["loss"] * n', { mono: true, fs: 12, fill: C.ink });
  o += box(30, 116, 226, 60, '#FFFFFF', D.red.base, 6, 1.5);
  o += tx(143, 138, 'Chains all 32 autograd DAGs', { fs: 12, fw: 700, fill: D.red.dark });
  o += tx(143, 158, 'alive in GPU RAM → OOM!', { fs: 12, fw: 700, fill: D.red.dark });
  o += tx(143, 206, 'Fix: v.detach() frees each batch\'s', { fs: 12, fill: C.ink });
  o += tx(143, 224, 'compute graph immediately', { fs: 12, fw: 700, fill: D.green.dark });

  o += box(290, 46, 254, 194, C.lblue, C.blue);
  o += tx(417, 70, 'Sums Stay on the Device', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(417, 98, 'sums[name] += v.detach() * n', { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });
  o += box(304, 116, 226, 60, '#FFFFFF', C.blue, 6, 1.5);
  o += tx(417, 138, 'No host synchronization', { fs: 12, fill: C.ink });
  o += tx(417, 158, 'inside the minibatch loop', { fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(417, 206, '1 host sync per epoch via', { fs: 12, fill: C.ink });
  o += tx(417, 224, 'float(self.sums[name]) / self.n', { mono: true, fs: 11, fill: D.blue.dark });

  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2.5 linear-regression-concise.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function scratchToConcise() {
  const W = 560, H = 276;
  let o = '';
  o += tx(280, 20, 'Swapping Hand-Written Primitives for Framework Components', { fs: 15, fw: 700, fill: C.ink });

  const rows = [
    ['1. Model Layer', 'w, b + matmul(X, w) + b', 'built-in dense layer'],
    ['2. Loss Function', 'reduce_mean((ŷ − y)² / 2)', 'built-in MSE loss'],
    ['3. Optimizer', 'for p in params: p -= lr*g', 'built-in SGD'],
  ];

  rows.forEach(([title, sc, co], i) => {
    const y = 44 + i * 72;
    o += box(16, y, 244, 58, C.lamber, C.amber);
    o += tx(138, y + 20, `${title} (Scratch)`, { fs: 13, fw: 700, fill: D.orange.dark });
    o += tx(138, y + 42, sc, { mono: true, fs: 11.5, fill: C.ink });

    o += arrow(262, y + 29, 294, y + 29, C.green);

    o += box(296, y, 248, 58, C.lgreen, C.green);
    o += tx(420, y + 20, `${title} (Concise)`, { fs: 13, fw: 700, fill: D.green.dark });
    o += tx(420, y + 42, co, { fs: 13, fw: 700, fill: C.ink });
  });

  return svg(W, H, o);
}

function lazyLinear() {
  const W = 560, H = 254;
  let o = '';
  o += tx(280, 20, 'How the Dense Layer Infers Input Dimension d', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 48, 160, 92, C.lamber, C.amber);
  o += tx(96, 72, '1. Construction', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(96, 96, 'dense, 1 output', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(96, 118, 'd unknown (uninit)', { fs: 12, fill: C.muted });

  o += arrow(178, 94, 206, 94, C.blue);

  o += box(210, 48, 152, 92, C.lblue, C.blue);
  o += tx(286, 72, '2. First Batch X', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(286, 96, 'X.shape = (32, 2)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(286, 118, 'infers d = 2!', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });

  o += arrow(364, 94, 392, 94, C.green);

  o += box(396, 48, 148, 92, C.lgreen, C.green);
  o += tx(470, 72, '3. Materialized', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(470, 96, 'weights: 2', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(470, 118, 'bias:    1', { mono: true, fs: 12, fw: 700, fill: C.ink });

  o += box(16, 162, 528, 72, D.purple.tint, C.purple);
  o += tx(280, 186, 'Inherited forward: d2l.Module.forward(X) → self.net(X)', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(280, 212, 'No custom forward() method needed on LinearRegression!', { fs: 13, fill: C.ink });

  return svg(W, H, o);
}

function lazyVsEager(fw = 'pytorch') {
  const title = 'Initializing N(0, 0.01²) Weights and a Zero Bias';
  const cards = {
    pytorch: [
      { head: '1. Construction: nn.LazyLinear(1)', badge: 'shapes unknown', sub: 'A dummy call model(data.X[:1]) creates the parameters', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
      { head: '2. Initialization: net.apply(init)', badge: 'normal_, zeros_', sub: 'weight (1, 2) drawn from N(0, 0.01²); bias set to 0', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
    ],
    tensorflow: [
      { head: '1. Construction: Dense(1, kernel_initializer=…)', badge: 'RandomNormal', sub: 'The initializer is attached; shapes are still unknown', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
      { head: '2. First Minibatch', badge: 'deferred', sub: 'kernel (2, 1) drawn from N(0, 0.01²); bias set to 0', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
    ],
    mxnet: [
      { head: '1. Construction: initialize(init.Normal(0.01))', badge: 'deferred', sub: 'The initializer is attached; shapes are still unknown', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
      { head: '2. First Minibatch', badge: 'shapes inferred', sub: 'weight (1, 2) drawn from N(0, 0.01²); bias set to 0', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
    ],
    jax: [
      { head: '1. Construction: nnx.Linear(2, 1, …)', badge: 'eager', sub: 'num_inputs = 2, kernel_init = normal(0.01), rngs', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
      { head: '2. Parameters Exist Immediately', badge: 'no dummy call', sub: 'kernel (2, 1) drawn from N(0, 0.01²); bias set to 0', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
    ],
  };
  return stackCards(560, 200, title, cards[fw]);
}

function mseHalving() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Loss Scaling Invariant: Why Concise MSE Halves η to 0.015', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 46, 242, 134, C.lblue, C.blue);
  o += tx(137, 70, 'Scratch Loss (with ½)', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(137, 96, 'L = ½ MSE  →  grad = g', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(137, 122, 'Floor: ½σ² = 5 × 10⁻⁵', { mono: true, fs: 12.5, fill: D.blue.dark });
  o += tx(137, 148, 'Learning rate: η = 0.03', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  o += tx(280, 113, '≡', { fs: 28, fw: 700, fill: C.purple });

  o += box(302, 46, 242, 134, C.lgreen, C.green);
  o += tx(423, 70, 'Framework MSE (no ½)', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(423, 96, 'L_fw = 2L  →  grad = 2g', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(423, 122, 'Floor: σ² = 10⁻⁴', { mono: true, fs: 12.5, fill: D.green.dark });
  o += tx(423, 148, 'Learning rate: η = 0.015', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });

  o += box(16, 196, 528, 52, D.purple.tint, C.purple);
  o += tx(280, 222, 'Identical Update Step: −η_fw · (2g) = −0.015(2g) = −0.03g!', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });

  return svg(W, H, o);
}

function paramDiscovery() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Automatic Parameter Tree Discovery in configure_optimizers', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(156, 44, 248, 48, D.purple.tint, C.purple);
  o += tx(280, 68, 'LinearRegression (Module)', { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });

  o += arrow(220, 94, 134, 126, C.blue);
  o += arrow(340, 94, 426, 126, C.blue);

  o += box(34, 130, 200, 52, C.lblue, C.blue);
  o += tx(134, 150, 'dense-layer weight', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(134, 168, '2 weights', { fs: 11.5, fill: C.muted });

  o += box(326, 130, 200, 52, C.lblue, C.blue);
  o += tx(426, 150, 'dense-layer bias', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += tx(426, 168, '1 intercept', { fs: 11.5, fill: C.muted });

  o += arrow(134, 184, 210, 204, C.green);
  o += arrow(426, 184, 350, 204, C.green);

  o += box(34, 204, 492, 40, C.lgreen, C.green);
  o += tx(280, 224, 'all registered parameters → SGD(lr=0.015) (scales to 1,000 layers!)', { mono: true, fs: 12, fw: 700, fill: D.green.dark });

  return svg(W, H, o);
}

function getWb() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Extracting Learned Weights and Bias via model.get_w_b()', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 52, 150, 92, C.lblue, C.blue);
  o += tx(91, 80, 'Trained Layer', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(91, 106, 'LinearRegression', { mono: true, fs: 12, fill: C.ink });
  o += tx(91, 126, '.net', { mono: true, fs: 12, fill: C.ink });
  o += arrow(168, 98, 200, 98, C.blue);

  o += box(204, 52, 150, 92, D.purple.tint, C.purple);
  o += tx(279, 80, 'get_w_b()', { mono: true, fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(279, 106, 'returns weight', { fs: 12, fill: C.ink });
  o += tx(279, 126, 'and bias values', { fs: 12, fill: C.ink });
  o += arrow(356, 98, 388, 98, C.green);

  o += box(392, 52, 152, 92, C.lgreen, C.green);
  o += tx(468, 80, 'Compare to Truth', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(468, 106, 'w* = (2, −3.4)', { mono: true, fs: 12, fill: C.ink });
  o += tx(468, 126, 'b* = 4.2', { mono: true, fs: 12, fill: C.ink });

  o += box(16, 166, 528, 66, C.lamber, C.amber);
  o += tx(280, 190, 'Errors of order 10⁻⁴ to 10⁻³: a few σ / √n ≈ 3 × 10⁻⁴', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(280, 214, 'Framework layers recover the same parameters as the scratch model', { fs: 12.5, fill: C.ink });

  return svg(W, H, o);
}

function threeTiers() {
  return stackCards(560, 286, 'Pedagogical Progression: Three Implementation Tiers', [
    { head: 'Tier 1: Bare Tensor Functions', badge: 'linreg + train()', sub: 'Raw tensors w, b, manual backward pass, manual loop & generator', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
    { head: 'Tier 2: OO Scratch Module', badge: 'LinearRegressionScratch', sub: 'Packages raw w, b & custom SGD into reusable d2l.Module + Trainer', fill: C.lblue, stroke: C.blue, dark: D.blue.dark },
    { head: 'Tier 3: Concise Framework API', badge: 'LinearRegression', sub: 'Plugs a built-in dense layer, MSE loss, and SGD into the same Trainer!', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
  ]);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2.6 generalization.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function empVsPopRisk() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Empirical Risk R_emp[f] vs. Population Risk R[f]', { fs: 15, fw: 700, fill: C.ink });

  o += box(140, 42, 280, 56, D.purple.tint, C.purple);
  o += tx(280, 62, 'Population Distribution P(x, y)', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(280, 82, 'i.i.d. data-generating process', { fs: 12, fill: C.muted });

  o += arrow(210, 100, 140, 132, C.blue);
  o += arrow(350, 100, 420, 132, C.green);

  o += box(16, 136, 254, 110, C.lblue, C.blue);
  o += tx(143, 160, 'Training Sample D (size n)', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(143, 186, 'R_emp[f] = (1/n) Σ l(f(xᵢ), yᵢ)', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(143, 212, 'Minimized during training', { fs: 12, fill: D.blue.dark });
  o += tx(143, 230, '(Optimistically biased for f̂!)', { fs: 11.5, fw: 700, fill: D.red.dark });

  o += box(290, 136, 254, 110, C.lgreen, C.green);
  o += tx(417, 160, 'True Population Risk R[f]', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(417, 186, 'R[f] = E_{(x,y)~P}[l(f(x), y)]', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(417, 212, 'Governs deployment accuracy', { fs: 12, fill: D.green.dark });
  o += tx(417, 230, 'Estimated on held-out D\'', { fs: 12, fw: 700, fill: D.green.dark });

  return svg(W, H, o);
}

function ermOptimism() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Why Training Error Underestimates True Risk by 2σ²p/n', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 48, 254, 128, C.lblue, C.blue);
  o += tx(143, 72, 'Expected Training MSE', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(143, 100, 'E[R_emp] = σ² (1 − p/n)', { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(143, 128, 'Each parameter absorbs', { fs: 12.5, fill: C.muted });
  o += tx(143, 148, 'σ²/n of training noise!', { fs: 12.5, fw: 700, fill: D.blue.dark });

  o += box(290, 48, 254, 128, C.lamber, C.amber);
  o += tx(417, 72, 'Expected In-Sample Test MSE', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(417, 100, 'E[R_in]  = σ² (1 + p/n)', { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(417, 128, 'Fitted noise hurts on', { fs: 12.5, fill: C.muted });
  o += tx(417, 148, 'fresh labels y\' by +σ²p/n!', { fs: 12.5, fw: 700, fill: D.orange.dark });

  o += box(16, 192, 528, 56, D.red.tint, D.red.base);
  o += tx(280, 212, 'At p = n = 20 (Degree 19 on 20 Points):', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(280, 234, 'E[R_emp] = σ²(1 − 20/20) = 0  even when fitting pure noise!', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  return svg(W, H, o);
}

function vandermonde() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'Polynomial Curve Fitting as Linear Regression on Monomials', { fs: 15, fw: 700, fill: C.ink });

  const cy = 108;
  o += box(16, cy - 36, 94, 72, C.lblue, C.blue);
  o += tx(63, cy - 10, 'Scalar x', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(63, cy + 14, 'in [−1, 1]', { mono: true, fs: 12.5, fill: C.ink });

  o += arrow(112, cy, 146, cy, C.purple);

  o += box(150, cy - 44, 224, 88, D.purple.tint, C.purple);
  o += tx(262, cy - 18, 'Vandermonde Basis ϕ_d(x)', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(262, cy + 8, '[1, x, x², ..., xᵈ]', { mono: true, fs: 13.5, fw: 700, fill: C.ink });
  o += tx(262, cy + 30, 'X ∈ ℝ^{n × (d+1)}', { mono: true, fs: 12, fill: C.muted });

  o += arrow(376, cy, 410, cy, C.green);

  o += box(414, cy - 36, 130, 72, C.lgreen, C.green);
  o += tx(479, cy - 10, 'Linear Fit', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(479, cy + 14, 'ŷ = Xw', { mono: true, fs: 13.5, fw: 700, fill: C.ink });

  o += box(16, 174, 528, 66, C.lamber, C.amber);
  o += tx(280, 196, 'Nonlinear in x, but 100% Linear in Weights w = (w₀, ..., w_d)ᵀ!', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(280, 220, 'Solved exactly via lstsq(features(x, d), y) for any degree d = 1..19', { mono: true, fs: 12, fill: C.ink });

  return svg(W, H, o);
}

function weightExplosion() {
  return stackCards(560, 286, 'Three Regimes on n = 20 Points of a Noisy Cubic (σ² = 0.01)', [
    { head: 'Degree 1 (Underfitting · 2 Params)', badge: 'max |ŵ| ≈ 4', sub: 'Train MSE ≈ 1.5, Test MSE ≈ 1.8 (180× noise floor; line misses cubic)', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
    { head: 'Degree 3 (Well-Specified · 4 Params)', badge: 'max |ŵ| ≈ 5.6', sub: 'Train MSE ≈ 0.008, Test MSE ≈ 0.011 (hits σ² = 0.01 floor; matches f*)', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
    { head: 'Degree 19 (Overfitting · 20 Params)', badge: 'max |ŵ| > 10⁶!', sub: 'Train MSE ≈ 0, Test MSE > 10⁴ (collinear x¹⁷ ≈ x¹⁹ cancel via huge ±w_j)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark },
  ]);
}

function biasVarBullseye() {
  const W = 560, H = 266;
  let o = '';
  o += tx(280, 20, 'Expected Test MSE = Bias²(x) + Variance(x) + σ²', { fs: 15, fw: 700, fill: C.ink });

  const bw = 164, bh = 136, y = 46;
  const xs = [16, 198, 380];

  o += box(xs[0], y, bw, bh, C.lamber, C.amber);
  o += tx(xs[0] + bw / 2, y + 24, '1. Squared Bias', { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(xs[0] + bw / 2, y + 52, '(f̄(x) − f*(x))²', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(xs[0] + bw / 2, y + 82, 'Rigid model misses', { fs: 12, fill: C.ink });
  o += tx(xs[0] + bw / 2, y + 102, 'truth even on ∞ data', { fs: 12, fill: C.ink });
  o += tx(xs[0] + bw / 2, y + 122, '(dominates d = 1, 2)', { fs: 11.5, fw: 700, fill: D.orange.dark });

  o += box(xs[1], y, bw, bh, D.red.tint, D.red.base);
  o += tx(xs[1] + bw / 2, y + 24, '2. Variance', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(xs[1] + bw / 2, y + 52, 'E_D[(f̂_D(x)−f̄(x))²]', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  o += tx(xs[1] + bw / 2, y + 82, 'Sensitivity of fit to', { fs: 12, fill: C.ink });
  o += tx(xs[1] + bw / 2, y + 102, 'random sample noise', { fs: 12, fill: C.ink });
  o += tx(xs[1] + bw / 2, y + 122, '(scales ≈ σ²p/n)', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });

  o += box(xs[2], y, bw, bh, C.lblue, C.blue);
  o += tx(xs[2] + bw / 2, y + 24, '3. Noise Floor', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(xs[2] + bw / 2, y + 52, 'Var(ε) = σ²', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(xs[2] + bw / 2, y + 82, 'Irreducible label', { fs: 12, fill: C.ink });
  o += tx(xs[2] + bw / 2, y + 102, 'uncertainty around f*', { fs: 12, fill: C.ink });
  o += tx(xs[2] + bw / 2, y + 122, '(σ² = 0.01 here)', { mono: true, fs: 11.5, fw: 700, fill: D.blue.dark });

  o += box(16, 198, 528, 52, C.lgreen, C.green);
  o += tx(280, 224, 'Sweet Spot (d = 3): Bias² collapses to 0 and Variance is only 4σ²/20 = 0.002!', { fs: 12.5, fw: 700, fill: D.green.dark });

  return svg(W, H, o);
}

function sampleScale() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Growing Dataset Size n = 20 → 1,000: Variance Collapses, Bias Stays', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 46, 254, 196, C.lamber, C.amber);
  o += tx(143, 70, 'Degree 1 (High Bias)', { fs: 14.5, fw: 700, fill: D.orange.dark });
  o += tx(143, 102, 'n = 20:    Test MSE ≈ 1.8–2.1', { mono: true, fs: 12.5, fill: C.ink });
  o += tx(143, 128, 'n = 100:   Test MSE ≈ 1.7–2.0', { mono: true, fs: 12.5, fill: C.ink });
  o += tx(143, 154, 'n = 1,000: Test MSE ≈ 1.7–1.9', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += box(30, 176, 226, 48, '#FFFFFF', C.amber, 6, 1.5);
  o += tx(143, 200, 'More data NEVER cures bias!', { fs: 12.5, fw: 700, fill: D.orange.dark });

  o += box(290, 46, 254, 196, C.lgreen, C.green);
  o += tx(417, 70, 'Degree 19 (High Variance)', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(417, 102, 'n = 20:    Test MSE > 10⁸', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(417, 128, 'n = 100:   Test MSE ≈ 0.01–0.2', { mono: true, fs: 12.5, fill: C.ink });
  o += tx(417, 154, 'n = 1,000: Test MSE ≈ 0.010', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += box(304, 176, 226, 48, '#FFFFFF', C.green, 6, 1.5);
  o += tx(417, 200, 'Variance ∝ p/n → matches cubic!', { fs: 12, fw: 700, fill: D.green.dark });

  return svg(W, H, o);
}

function threeWaySplit() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'The Three-Way Data Partition Discipline', { fs: 15, fw: 700, fill: C.ink });

  const bw = 160, bh = 136, y = 46;
  const xs = [16, 200, 384];

  o += box(xs[0], y, bw, bh, C.lblue, C.blue);
  o += tx(xs[0] + bw / 2, y + 24, '1. Training Set', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(xs[0] + bw / 2, y + 54, 'Fits parameters', { fs: 13, fill: C.ink });
  o += tx(xs[0] + bw / 2, y + 76, '(w, b) via SGD/lstsq', { mono: true, fs: 11.5, fill: C.ink });
  o += tx(xs[0] + bw / 2, y + 110, 'Biased downward', { fs: 12, fw: 700, fill: D.red.dark });

  o += arrow(xs[0] + bw + 2, y + bh / 2, xs[1] - 2, y + bh / 2, C.blue);

  o += box(xs[1], y, bw, bh, C.lamber, C.amber);
  o += tx(xs[1] + bw / 2, y + 24, '2. Validation Set', { fs: 14.5, fw: 700, fill: D.orange.dark });
  o += tx(xs[1] + bw / 2, y + 54, 'Selects hyperparams', { fs: 13, fill: C.ink });
  o += tx(xs[1] + bw / 2, y + 76, '(degree d, η, λ)', { mono: true, fs: 12, fill: C.ink });
  o += tx(xs[1] + bw / 2, y + 110, 'Or K-fold CV if small', { fs: 11.5, fw: 700, fill: D.orange.dark });

  o += arrow(xs[1] + bw + 2, y + bh / 2, xs[2] - 2, y + bh / 2, C.green);

  o += box(xs[2], y, bw, bh, C.lgreen, C.green);
  o += tx(xs[2] + bw / 2, y + 24, '3. Test Vault', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(xs[2] + bw / 2, y + 54, 'Evaluated ONCE', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(xs[2] + bw / 2, y + 76, 'on final winner', { fs: 12.5, fill: C.ink });
  o += tx(xs[2] + bw / 2, y + 110, 'Unbiased risk report', { fs: 12, fw: 700, fill: D.green.dark });

  o += box(16, 198, 528, 50, D.red.tint, D.red.base);
  o += tx(280, 223, 'Rule: Tuning hyperparameters on the Test Set leaks test noise into model choice!', { fs: 12.5, fw: 700, fill: D.red.dark });

  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §2.7 weight-decay.md (9 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function twoThermometers() {
  const W = 560, H = 266;
  let o = '';
  o += tx(280, 20, 'The Two-Thermometer Collinearity Pathology (x₁ ≈ x₂)', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 44, 254, 204, C.lgreen, C.green);
  o += tx(143, 68, 'Model A: w = (1, 1)ᵀ', { mono: true, fs: 14, fw: 700, fill: D.green.dark });
  o += tx(143, 96, 'ŷ = x₁ + x₂  (sums the sensors)', { mono: true, fs: 12, fill: C.ink });
  o += box(30, 114, 226, 56, '#FFFFFF', C.green, 6, 1.5);
  o += tx(143, 134, 'Norm: ‖w‖₂² = 2', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(143, 156, 'Half the noise variance of 2x₁', { fs: 12, fill: C.ink });
  o += tx(143, 198, 'When sensor drift x₁ − x₂ = 2:', { fs: 12, fill: C.muted });
  o += tx(143, 220, 'Perturbation = 1(2) = 2 (stable)', { fs: 12.5, fw: 700, fill: D.green.dark });

  o += box(290, 44, 254, 204, D.red.tint, D.red.base);
  o += tx(417, 68, 'Model B: w = (501, −499)ᵀ', { mono: true, fs: 14, fw: 700, fill: D.red.dark });
  o += tx(417, 96, 'ŷ = (x₁+x₂) + 500(x₁−x₂)', { mono: true, fs: 12, fill: C.ink });
  o += box(304, 114, 226, 56, '#FFFFFF', D.red.base, 6, 1.5);
  o += tx(417, 134, 'Norm: ‖w‖₂² = 500,002!', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(417, 156, 'Fits tiny noise in x₁ − x₂', { fs: 12, fill: C.ink });
  o += tx(417, 198, 'When sensor drift x₁ − x₂ = 2:', { fs: 12, fill: C.muted });
  o += tx(417, 220, 'Prediction explodes by +1,000!', { fs: 12.5, fw: 700, fill: D.red.dark });

  return svg(W, H, o);
}

function penalizedLoss() {
  const W = 560, H = 258;
  let o = '';
  o += tx(280, 20, 'The Ridge Objective: L_λ(w, b) = L(w, b) + (λ/2)‖w‖₂²', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 48, 228, 98, C.lblue, C.blue);
  o += tx(130, 72, 'Data Misfit L(w, b)', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(130, 98, '‖Xw + b − y‖₂² / (2n)', { mono: true, fs: 12, fw: 700, fill: C.ink });
  o += tx(130, 122, 'Fits observed targets', { fs: 12, fill: C.muted });

  o += tx(262, 97, '+', { fs: 26, fw: 700, fill: C.ink });

  o += box(280, 48, 264, 98, D.purple.tint, C.purple);
  o += tx(412, 72, 'ℓ₂ Norm Penalty (λ/2)‖w‖₂²', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(412, 98, 'Bounds Lipschitz slope ‖w‖₂', { fs: 12.5, fw: 600, fill: C.ink });
  o += tx(412, 122, 'Bias b is NEVER penalized!', { fs: 12.5, fw: 700, fill: D.purple.dark });

  o += box(16, 166, 528, 74, C.lamber, C.amber);
  o += tx(280, 190, 'Why Leave b Unpenalized & Standardize Features First?', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(280, 214, '• b only shifts baseline units (Kelvin vs Celsius), not input sensitivity', { fs: 12, fill: C.ink });
  o += tx(280, 232, '• Isotropic Σ wⱼ² requires features on comparable unit-variance scales', { fs: 12, fill: C.ink });

  return svg(W, H, o);
}

function multiplicativeDecay() {
  const W = 560, H = 256;
  let o = '';
  o += tx(280, 20, 'Why ℓ₂ Regularization Is Called Weight Decay in SGD', { fs: 15, fw: 700, fill: C.ink });

  const cy = 104;
  o += box(16, cy - 42, 142, 84, C.lblue, C.blue);
  o += tx(87, cy - 14, 'Current wₜ', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(87, cy + 14, 'before step', { fs: 12.5, fill: C.muted });

  o += arrow(160, cy, 192, cy, D.purple.base);

  o += box(196, cy - 42, 174, 84, D.purple.tint, C.purple);
  o += tx(283, cy - 14, '1. Shrinkage', { fs: 14, fw: 700, fill: D.purple.dark });
  o += tx(283, cy + 12, '(1 − ηλ) wₜ', { mono: true, fs: 13.5, fw: 700, fill: C.ink });
  o += tx(283, cy + 32, 'factor < 1', { mono: true, fs: 11.5, fill: D.purple.dark });

  o += arrow(372, cy, 404, cy, C.green);

  o += box(408, cy - 42, 136, 84, C.lgreen, C.green);
  o += tx(476, cy - 14, '2. Data Step', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(476, cy + 14, '− η ∇_w L_B', { mono: true, fs: 13, fw: 700, fill: C.ink });

  o += box(16, 168, 528, 70, C.lamber, C.amber);
  o += tx(280, 192, 'Algebraic Identity: w − η(∇L + λw) = (1 − ηλ)w − η∇L', { mono: true, fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(280, 218, 'Unreinforced directions (∇L = 0) decay exponentially as (1 − ηλ)ᵗ → 0', { fs: 12.5, fill: C.ink });

  return svg(W, H, o);
}

function gaussianPriorMap() {
  const W = 560, H = 262;
  let o = '';
  o += tx(280, 20, 'Bayesian MAP View: Gaussian Prior Yields λ = σ² / (nτ²)', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 48, 160, 92, D.purple.tint, C.purple);
  o += tx(96, 72, 'Gaussian Prior', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(96, 98, 'wⱼ ~ N(0, τ²)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(96, 120, '−log p(w) = ‖w‖²/2τ²', { mono: true, fs: 11, fill: D.purple.dark });

  o += tx(190, 94, '+', { fs: 24, fw: 700, fill: C.ink });

  o += box(204, 48, 166, 92, C.lblue, C.blue);
  o += tx(287, 72, 'Gaussian Likelihood', { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(287, 98, 'ε⁽ⁱ⁾ ~ N(0, σ²)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(287, 120, '−log p(y|X,w) = nL/σ²', { mono: true, fs: 11, fill: D.blue.dark });

  o += arrow(372, 94, 400, 94, C.green);

  o += box(404, 48, 140, 92, C.lgreen, C.green);
  o += tx(474, 72, 'MAP Penalty', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(474, 98, 'λ = σ²/(nτ²)', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(474, 120, '= 1/3 on testbed', { mono: true, fs: 11.5, fill: C.ink });

  o += box(16, 160, 528, 82, C.lamber, C.amber);
  o += tx(280, 184, 'Three Scaling Laws of Optimal Regularization λ = σ² / (nτ²)', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(280, 208, '• Noisier labels (σ² ↑) or smaller true weights (τ² ↓)  ⇒  stronger penalty λ ↑', { fs: 12.5, fill: C.ink });
  o += tx(280, 228, '• Abundant training data (n → ∞)  ⇒  penalty vanishes as λ ∝ 1/n → 0', { fs: 12.5, fill: C.ink });

  return svg(W, H, o);
}

function highdimTestbed() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'High-Dimensional Benchmark (d = 200 Features, n = 300 Examples)', { fs: 14.5, fw: 700, fill: C.ink });

  const bw = 160, bh = 112, y = 46;
  const xs = [16, 200, 384];

  o += box(xs[0], y, bw, bh, C.lgreen, C.green);
  o += tx(xs[0] + bw / 2, y + 24, 'Oracle (w*, b*)', { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(xs[0] + bw / 2, y + 52, 'Val Loss = 0.50', { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(xs[0] + bw / 2, y + 78, '½σ² = ½(1.0)²', { mono: true, fs: 12, fill: C.ink });
  o += tx(xs[0] + bw / 2, y + 98, '‖w*‖₂ = √2 ≈ 1.41', { mono: true, fs: 11.5, fill: C.muted });

  o += box(xs[1], y, bw, bh, C.lblue, C.blue);
  o += tx(xs[1] + bw / 2, y + 24, 'Ridge (λ = 1/3)', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(xs[1] + bw / 2, y + 52, 'Val ≈ 0.75–1.05', { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(xs[1] + bw / 2, y + 78, '‖ŵ‖₂ ≈ 1.2', { mono: true, fs: 12, fill: C.ink });
  o += tx(xs[1] + bw / 2, y + 98, 'Stable for 50+ epochs', { fs: 11.5, fill: D.blue.dark });

  o += box(xs[2], y, bw, bh, D.red.tint, D.red.base);
  o += tx(xs[2] + bw / 2, y + 24, 'Unpenalized (λ = 0)', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(xs[2] + bw / 2, y + 52, 'Val Loss ≈ 1.1–1.9', { mono: true, fs: 13, fw: 700, fill: D.red.dark });
  o += tx(xs[2] + bw / 2, y + 78, 'Train loss ≈ 0.25!', { mono: true, fs: 12, fill: D.red.dark });
  o += tx(xs[2] + bw / 2, y + 98, 'No better than w=0 (1.50)', { fs: 11.5, fw: 700, fill: D.red.dark });

  o += box(16, 176, 528, 66, D.purple.tint, C.purple);
  o += tx(280, 198, 'Why validation_step Never Includes the Penalty Term (λ/2)‖w‖₂²', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(280, 222, 'Validation loss measures pure predictive MSE so all λ values are comparable!', { fs: 12.5, fill: C.ink });

  return svg(W, H, o);
}

function eigenFilter() {
  return stackCards(560, 286, 'Spectral Filtering in the Eigenbasis of S = (1/n)XᵀX', [
    { head: 'Well-Determined Directions (μⱼ ≫ λ)', badge: 'μⱼ/(μⱼ+λ) ≈ 1', sub: 'Rich data spread pins down the slope → ridge leaves vⱼᵀŵ₀ untouched', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
    { head: 'Poorly Determined Directions (μⱼ ≪ λ)', badge: 'μⱼ/(μⱼ+λ) ≪ 1', sub: 'Least squares amplifies noise as σ/√(nμⱼ) → ridge quashes noisy coordinate!', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
    { head: 'Unobserved Null-Space Directions (μⱼ = 0)', badge: 'factor = 0', sub: 'Shifted matrix (S + λI) is strictly invertible and zeroes null-space weights', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark },
  ]);
}

function errorSpectrum() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Squared Parameter Error ‖ŵ − w*‖₂² Split Across 200 Eigenvectors', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 46, 254, 146, D.red.tint, D.red.base);
  o += tx(143, 70, 'k ≈ 54–55 Ill-Conditioned Dirs', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(143, 92, '(eigenvalues μⱼ < λ = 1/3)', { mono: true, fs: 12, fill: C.muted });
  o += tx(143, 122, 'λ = 0:    error ≈ 0.8–2.2', { mono: true, fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(143, 148, 'λ = 1/3:  error ≈ 0.3–0.4', { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(143, 174, '2–6× surgical error reduction!', { fs: 12.5, fw: 700, fill: D.green.dark });

  o += box(290, 46, 254, 146, C.lblue, C.blue);
  o += tx(417, 70, '~145 Well-Determined Dirs', { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(417, 92, '(eigenvalues μⱼ ≥ 1/3 up to 3.2)', { mono: true, fs: 12, fill: C.muted });
  o += tx(417, 122, 'λ = 0:    error ≈ 0.5–0.6', { mono: true, fs: 13.5, fw: 700, fill: C.ink });
  o += tx(417, 148, 'λ = 1/3:  error ≈ 0.3–0.45', { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(417, 174, 'Signal preserved (df ≈ 127)', { fs: 12.5, fw: 700, fill: D.blue.dark });

  o += box(16, 206, 528, 44, C.lgreen, C.green);
  o += tx(280, 228, 'Expected Val Loss = ½‖ŵ − w*‖₂² + 0.50: drops from ~1.1–1.8 (λ=0) to ~0.75–1.05 (λ=1/3)!', { fs: 12.5, fw: 700, fill: D.green.dark });

  return svg(W, H, o);
}

function earlyStoppingFilter() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Why Unpenalized SGD Peaked at Epoch 2: Early Stopping as a Filter', { fs: 14.5, fw: 700, fill: C.ink });

  o += box(16, 46, 254, 134, C.lgreen, C.green);
  o += tx(143, 70, 'Large-Eigenvalue Dir (μ = 3.2)', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(143, 98, '1 Epoch (15 steps, η = 0.05):', { fs: 12, fill: C.ink });
  o += tx(143, 124, '1 − (1 − 0.05×3.2)¹⁵ = 93%', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(143, 152, 'Signal learned in 1–2 epochs!', { fs: 12.5, fw: 700, fill: D.green.dark });

  o += box(290, 46, 254, 134, D.red.tint, D.red.base);
  o += tx(417, 70, 'Small-Eigenvalue Dir (μ = 0.04)', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(417, 98, '1 Epoch (15 steps, η = 0.05):', { fs: 12, fill: C.ink });
  o += tx(417, 124, '1 − (1 − 0.05×0.04)¹⁵ = 3%', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(417, 152, 'Soaks up noise over epochs 5–50!', { fs: 12.5, fw: 700, fill: D.red.dark });

  o += box(16, 194, 528, 52, D.purple.tint, C.purple);
  o += tx(280, 220, 'Stopping at T steps: filter 1 − (1 − ημⱼ)ᵀ ≈ μⱼ / (μⱼ + 1/(ηT))', { mono: true, fs: 12, fw: 700, fill: D.purple.dark });

  return svg(W, H, o);
}

function paramGroupsWd() {
  return stackCards(560, 286, 'Exempting Bias b From Optimizer-Level Weight Decay', [
    { head: 'Weights w: Decayed Every Step', badge: 'weight decay = wd', sub: 'w ← w − η(∇L + wd · w): the optimizer adds wd · w to the gradient', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true },
    { head: 'Bias b: Exempt From Decay', badge: 'decay = 0', sub: 'b ← b − η ∂L/∂b: the intercept is left unpenalized', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true },
  ]);
}

function lstsqReference() {
  const W = 560, H = 260;
  let o = '';
  o += tx(280, 20, 'Direct lstsq Reference Fit vs. 3-SE Pass/Fail Tolerance', { fs: 15, fw: 700, fill: C.ink });

  o += box(16, 46, 256, 134, C.lblue, C.blue);
  o += tx(144, 70, 'Exact QR Solve: lstsq(X̃, y)', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(144, 98, 'ŵ ≈ (2, −3.4)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(144, 124, 'b̂ ≈ 4.2', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(144, 152, 'Errors ~10⁻⁴ ≈ σ / √n', { mono: true, fs: 12, fill: D.blue.dark });

  o += box(288, 46, 256, 134, C.lgreen, C.green);
  o += tx(416, 70, '3-SE Tolerance Band', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(416, 98, '3 × (σ / √n) ≈ 10⁻³', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(416, 124, 'Any converged SGD in this chapter', { fs: 12, fill: C.ink });
  o += tx(416, 152, 'must land within ±10⁻³ of w*, b*', { fs: 12, fw: 700, fill: D.green.dark });

  o += box(16, 196, 528, 48, D.purple.tint, C.purple);
  o += tx(280, 220, 'Separates statistical sampling noise (~3×10⁻⁴) from optimization bugs!', { fs: 13, fw: 700, fill: D.purple.dark });

  return svg(W, H, o);
}

export const diagrams = {
  // §2.0 index.md
  'ch02-lr-miniature':             lrMiniature,
  'ch02-lr-roadmap':               lrRoadmap,
  'ch02-lr-benchmarks':            lrBenchmarks,
  // §2.1 linear-regression.md
  'ch02-lr-affine-plane':          affinePlane,
  'ch02-lr-design-matrix':         designMatrix,
  'ch02-lr-outlier-pull':          outlierPull,
  'ch02-lr-qr-vs-gram':            qrVsGram,
  'ch02-lr-scaling-cost':          scalingCost,
  'ch02-lr-minibatch-step':        minibatchStep,
  'ch02-lr-loop-vs-gemm':          loopVsGemm,
  // §2.2 synthetic-regression-data.md
  'ch02-lr-synth-oracle':          synthOracle,
  'ch02-lr-synth-dgp':             synthDgp,
  'ch02-lr-synth-shapes':          synthShapes,
  'ch02-lr-jax-key-split':         jaxKeySplit,
  'ch02-lr-noise-floor':           noiseFloor,
  'ch02-lr-recovery-decomp':       recoveryDecomp,
  'ch02-lr-gram-concentration':    gramConcentration,
  'ch02-lr-se-scaling':            seScaling,
  'ch02-lr-lstsq-reference':       lstsqReference,
  // §2.3 linear-regression-scratch.md
  'ch02-lr-broadcast-trap':        broadcastTrap,
  'ch02-lr-param-init':            paramInit,
  'ch02-lr-defensive-reshape':     defensiveReshape,
  'ch02-lr-grad-verify':           gradVerify,
  'ch02-lr-reshuffle-bias':        reshuffleBias,
  'ch02-lr-gen-exhaustion':        genExhaustion,
  'ch02-lr-four-step-loop':        fourStepLoop,
  'ch02-lr-contraction-rate':      contractionRate,
  'ch02-lr-param-recovery':        paramRecovery,
  'ch02-lr-lr-regimes':            lrRegimes,
  // §2.4 oo-design.md
  'ch02-lr-oo-decouple':           ooDecouple,
  'ch02-lr-oo-utilities':          ooUtilities,
  'ch02-lr-module-contract':       moduleContract,
  'ch02-lr-scratch-module':        scratchModule,
  'ch02-lr-sgd-class':             sgdClass,
  'ch02-lr-reusable-loader':       reusableLoader,
  'ch02-lr-averager-weights':      averagerWeights,
  'ch02-lr-trainer-timeline':      trainerTimeline,
  'ch02-lr-detach-graph':          detachGraph,
  // §2.5 linear-regression-concise.md
  'ch02-lr-scratch-to-concise':    scratchToConcise,
  'ch02-lr-lazy-linear':           lazyLinear,
  'ch02-lr-lazy-vs-eager':         lazyVsEager,
  'ch02-lr-lazy-vs-eager-tensorflow': () => lazyVsEager('tensorflow'),
  'ch02-lr-lazy-vs-eager-mxnet':   () => lazyVsEager('mxnet'),
  'ch02-lr-lazy-vs-eager-jax':     () => lazyVsEager('jax'),
  'ch02-lr-mse-halving':           mseHalving,
  'ch02-lr-param-discovery':       paramDiscovery,
  'ch02-lr-get-wb':                getWb,
  'ch02-lr-three-tiers':           threeTiers,
  // §2.6 generalization.md
  'ch02-lr-emp-vs-pop-risk':       empVsPopRisk,
  'ch02-lr-erm-optimism':          ermOptimism,
  'ch02-lr-vandermonde':           vandermonde,
  'ch02-lr-weight-explosion':      weightExplosion,
  'ch02-lr-bias-var-bullseye':     biasVarBullseye,
  'ch02-lr-sample-scale':          sampleScale,
  'ch02-lr-three-way-split':       threeWaySplit,
  // §2.7 weight-decay.md
  'ch02-lr-two-thermometers':      twoThermometers,
  'ch02-lr-penalized-loss':        penalizedLoss,
  'ch02-lr-multiplicative-decay':  multiplicativeDecay,
  'ch02-lr-gaussian-prior-map':    gaussianPriorMap,
  'ch02-lr-highdim-testbed':       highdimTestbed,
  'ch02-lr-eigen-filter':          eigenFilter,
  'ch02-lr-error-spectrum':        errorSpectrum,
  'ch02-lr-early-stopping-filter': earlyStoppingFilter,
  'ch02-lr-param-groups-wd':       paramGroupsWd,
};
