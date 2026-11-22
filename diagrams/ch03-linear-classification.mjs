// Slide diagrams for Chapter 3: Linear Neural Networks for Classification
// (chapter_linear-classification/*.md).
//
// Exports 64 conceptual, geometric, and systems diagrams keyed by stable
// `ch03-cls-<concept>` IDs.

import { C, TOKENS, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

function stackCards(W, H, title, cards) {
  let o = '';
  const Wc = 560;
  const Hc = Math.max(H, cards.length === 4 ? 312 : 284);
  if (title) o += tx(Wc / 2, 20, title, { fs: 15, fw: 700, fill: C.ink });
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

function twoCols(W, H, title, left, right, foot) {
  const Wc = 560;
  let o = '';
  o += tx(Wc / 2, 20, title, { fs: 14.5, fw: 700, fill: C.ink });
  const y = 44, ch = foot ? 156 : 204, cw = 256;
  [left, right].forEach((p, idx) => {
    const x = idx === 0 ? 14 : 290;
    o += box(x, y, cw, ch, p.fill, p.stroke, 8, 2);
    o += tx(x + cw / 2, y + 24, p.head, { fs: 13.5, fw: 700, fill: p.dark });
    p.lines.forEach((ln, li) => {
      o += tx(x + cw / 2, y + 54 + li * 26, ln.t, {
        fs: ln.fs || 12,
        fw: ln.fw || 600,
        fill: ln.col || C.ink,
        mono: !!ln.mono,
      });
    });
  });
  if (foot) {
    o += box(14, 212, 532, 44, foot.fill, foot.stroke, 8, 2);
    o += tx(280, 234, foot.t, { fs: 12, fw: 700, fill: foot.dark, mono: !!foot.mono });
  }
  return svg(Wc, H, o);
}

function threeStages(W, H, title, s1, s2, s3, foot) {
  const Wc = 560;
  let o = '';
  o += tx(Wc / 2, 20, title, { fs: 14.5, fw: 700, fill: C.ink });
  const bw = 162, bh = 122, y = 46, xs = [14, 199, 384];
  o += arrow(xs[0] + bw + 2, y + bh / 2, xs[1] - 2, y + bh / 2, s1.stroke);
  o += arrow(xs[1] + bw + 2, y + bh / 2, xs[2] - 2, y + bh / 2, s2.stroke);
  [s1, s2, s3].forEach((s, i) => {
    const x = xs[i];
    o += box(x, y, bw, bh, s.fill, s.stroke, 8, 2);
    o += tx(x + bw / 2, y + 24, s.head, { fs: 13, fw: 700, fill: s.dark });
    s.lines.forEach((ln, li) => {
      o += tx(x + bw / 2, y + 52 + li * 24, ln, { fs: 11.5, fw: 600, fill: C.ink, mono: s.mono });
    });
  });
  if (foot) {
    o += box(14, 184, 532, 62, foot.fill, foot.stroke, 8, 2);
    o += tx(280, 206, foot.t1, { fs: 12.5, fw: 700, fill: foot.dark });
    o += tx(280, 228, foot.t2, { fs: 11.5, fw: 600, fill: C.ink, mono: !!foot.mono });
  }
  return svg(Wc, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §3.0 index.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const whyClassification = () => stackCards(560, 284, 'Why Classification Needs Its Own Loss and Diagnostics', [
  { head: '1. Unordered Categories Break Scalar Geometry', badge: 'y ∈ {1..q}', sub: 'Numbering 1..10 puts 5.5 between coat & sandal and penalizes 1→10 by 81×', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark },
  { head: '2. Raw Accuracy Hides Rare-Class & Cost Failures', badge: '1% prior', sub: 'Always predicting "healthy" scores 99% accuracy while catching 0% of cases', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
  { head: '3. Test Error Bars & Deployment Shift', badge: 'p_S ≠ p_T', sub: 'Finite test sets have ±1/√n noise and vouch only for the source distribution', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark },
]);

const regVsCls = () => twoCols(560, 270, 'From Linear Regression (§2) to Softmax Classification (§3)',
  {
    head: 'Linear Regression (§2)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: 'Target: scalar y ∈ ℝ', mono: true },
      { t: 'Model: ŷ = wᵀx + b (1 output)', mono: true },
      { t: 'Noise: Gaussian N(ŷ, σ²)' },
      { t: 'Loss: ½(ŷ − y)²  (MSE)', mono: true },
    ],
  },
  {
    head: 'Softmax Regression (§3)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'Target: one-hot y, q entries', mono: true },
      { t: 'Model: ŷ = softmax(Wᵀx + b)', mono: true },
      { t: 'Likelihood: Categorical(ŷ)' },
      { t: 'Loss: −log ŷ_y  (Cross-Entropy)', mono: true },
    ],
  },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t: 'Shared Invariant: Logit / Output Gradient Is Prediction Minus Truth (ŷ − y)!' }
);

const clsRoadmap = () => stackCards(560, 312, 'Chapter 3 Progression: From Logits to Trusted Decisions', [
  { head: '§3.1 Softmax & Cross-Entropy · §3.2 Fashion-MNIST', badge: 'o → ŷ → l', sub: 'One-hot vectors, shift invariance, ∇_o l = ŷ − y, 60k+10k 28×28 garments', fill: C.lblue, stroke: C.blue, dark: D.blue.dark },
  { head: '§3.3 Implementation (Scratch to LogSumExp)', badge: '7,850 params', sub: 'Classifier base class, 10×10 confusion matrix, 28×28 templates, LogSumExp', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark },
  { head: '§3.4 Measuring Classifiers · §3.5 Generalization', badge: 'ECE · VC', sub: 'Cost thresholds, reliability diagrams, temperature T, Hoeffding & VC bounds', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
  { head: '§3.6 Environment and Distribution Shift', badge: 'β(x) & C⁻¹μ', sub: 'Covariate shift domain classifier, confusion-matrix label shift, feedback loops', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
]);

const clsBenchmarks = () => stackCards(560, 312, 'Quantitative Running Benchmarks Across Chapter 3', [
  { head: '§3.1 3-Class Toy Image & §§3.2–3.3 Fashion-MNIST', badge: '83% val acc', sub: 'Logits (1.0, 2.2, 0.3) → ŷ=(0.21, 0.69, 0.10); LogSumExp handles (1000, 0, 0)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark },
  { head: '§3.4 Clinical Screening (1% Prior, 100:1 Cost)', badge: '90% rec, 15% prec', sub: 'Bayes threshold drops from 0.50 to 1/101 ≈ 0.01; Fashion-MNIST ECE < 0.03', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
  { head: '§3.5 Finite Test Sets (n = 10,000) & Reuse', badge: '±0.7% CLT', sub: 'Majority vote of 1,000 filtered coin-flippers jumps from 50% to ~70% acc!', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark },
  { head: '§3.6 Domain Shift Reweighting & 50% Shirt Surge', badge: '55% → 95% tgt', sub: 'Domain logit h(x)≈2x₁−2 tilts boundary; C⁻¹μ recovers 0.50 shirt prior', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
]);

// ═══════════════════════════════════════════════════════════════════════════
// §3.1 softmax-regression.md (11 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const ordinalTrap = () => twoCols(560, 266, 'Why Numbering Garments 1..10 Breaks Squared-Error Regression',
  {
    head: 'False Metric Ordering', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: '1: T-shirt   2: Trouser', mono: true },
      { t: '5: Coat      6: Sandal', mono: true },
      { t: 'Prediction ŷ = 5.5 sits halfway', col: D.red.dark, fw: 700 },
      { t: 'between a coat and a sandal!' },
    ],
  },
  {
    head: '81× Quadratic Penalty Distortion', fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    lines: [
      { t: 'T-shirt (1) → Trouser (2):', mono: true },
      { t: '  error² = (2 − 1)² = 1', mono: true },
      { t: 'T-shirt (1) → Boot (10):', mono: true },
      { t: '  error² = (10 − 1)² = 81×!', mono: true, col: D.red.dark, fw: 700 },
    ],
  },
  { fill: C.lgreen, stroke: C.green, dark: D.green.dark, t: 'Solution: Represent unordered classes symmetrically on the probability simplex' }
);

const onehotSimplex = () => threeStages(560, 262, 'One-Hot Encoding: Symmetric Vertices of the Probability Simplex',
  { head: 'Cat (y = 1)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['y = (1, 0, 0)', '100% on class 1', '‖e₁ − e₂‖₂ = √2'] },
  { head: 'Chicken (y = 2)', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['y = (0, 1, 0)', '100% on class 2', '‖e₂ − e₃‖₂ = √2'] },
  { head: 'Dog (y = 3)', fill: C.lamber, stroke: C.amber, dark: D.orange.dark, mono: true, lines: ['y = (0, 0, 1)', '100% on class 3', '‖e₃ − e₁‖₂ = √2'] },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t1: 'Every Pair of Distinct Classes Sits at Identical Distance √2', t2: 'Math uses a one-hot y of length q; code stores 0-based integer index y ∈ {0..q−1}' }
);

const softmaxPipe = () => threeStages(560, 262, 'The Softmax Operator: From Real Logits to Probabilities',
  { head: '1. Raw Logits o', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['o₁ = 1.0 (cat)', 'o₂ = 2.2 (chicken)', 'o₃ = 0.3 (dog)'] },
  { head: '2. Exponentials eᵒ', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['e¹·⁰ = 2.72', 'e²·² = 9.03', 'e⁰·³ = 1.35 (Σ=13.1)'] },
  { head: '3. Softmax ŷ', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['ŷ₁ = 0.21 (21%)', 'ŷ₂ = 0.69 (69%)', 'ŷ₃ = 0.10 (10%)'] },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t1: 'Strictly Positive (ŷⱼ > 0), Normalized (Σ ŷⱼ = 1), Order-Preserving', t2: 'argmax_{j} ŷ_j = argmax_{j} o_j  (inference can skip exp() altogether!)' }
);

const shiftInvariance = () => twoCols(560, 266, 'Shift Invariance & Log-Odds Ratios of the Softmax',
  {
    head: 'Additive Shift Cancels Out', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: 'softmax(o + c·1) = softmax(o)', mono: true, fw: 700, col: D.blue.dark },
      { t: 'exp(oⱼ + c) = exp(c)·exp(oⱼ)', mono: true },
      { t: 'Common factor exp(c) cancels' },
      { t: 'in numerator and denominator!' },
    ],
  },
  {
    head: 'Logit Differences = Log-Odds', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'oᵢ − oⱼ = log(ŷᵢ / ŷⱼ)', mono: true, fw: 700, col: D.green.dark },
      { t: 'o₂ − o₁ = 2.2 − 1.0 = 1.2', mono: true },
      { t: 'ŷ₂ / ŷ₁ = e¹·² ≈ 3.32×', mono: true },
      { t: 'Only pairwise differences matter!' },
    ],
  },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t: 'Numerical Safeguard: Subtracting c = max_{k} o_k keeps all exponents ≤ 0' }
);

const twoClassSigmoid = () => twoCols(560, 260, 'Two-Class Softmax (q = 2) Reduces to Logistic Regression',
  {
    head: '2-Class Softmax (2 columns)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: 'o₁ = w₁ᵀx + b₁', mono: true },
      { t: 'o₂ = w₂ᵀx + b₂', mono: true },
      { t: 'ŷ₁ = e^{o₁} / (e^{o₁} + e^{o₂})', mono: true, col: D.blue.dark, fw: 700 },
      { t: 'Divide top & bottom by e^{o₁}:' },
    ],
  },
  {
    head: 'Binary Logistic Sigmoid σ(o)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'o = o₁ − o₂ = (w₁−w₂)ᵀx + (b₁−b₂)', mono: true, fs: 11 },
      { t: 'ŷ₁ = 1 / (1 + e^{−o}) = σ(o)', mono: true, col: D.green.dark, fw: 700 },
      { t: 'ŷ₂ = 1 − ŷ₁ = σ(−o)', mono: true },
      { t: 'o = log(ŷ₁ / ŷ₂)  ("logit")', mono: true },
    ],
  },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t: 'One parameter vector (w₁ − w₂) determines all binary class probabilities!' }
);

function batchedSoftmax() {
  const W = 560, H = 264;
  let o = '';
  o += tx(280, 20, 'Minibatch Vectorization on Fashion-MNIST (n = 256, d = 784, q = 10)', { fs: 14.5, fw: 700, fill: C.ink });
  const cy = 108;
  o += box(16, cy - 46, 106, 92, C.lblue, C.blue);
  o += tx(69, cy - 16, 'X', { fs: 19, fw: 700, fill: D.blue.dark });
  o += tx(69, cy + 8, '256 × 784', { mono: true, fs: 12, fill: C.ink });
  o += tx(69, cy + 28, 'batch images', { fs: 11.5, fill: C.muted });

  o += tx(134, cy, '·', { fs: 26, fw: 700, fill: C.ink });

  o += box(146, cy - 46, 102, 92, N.panel, C.gray);
  o += tx(197, cy - 16, 'W', { fs: 19, fw: 700, fill: C.ink });
  o += tx(197, cy + 8, '784 × 10', { mono: true, fs: 12, fill: C.ink });
  o += tx(197, cy + 28, '7,840 weights', { fs: 11.5, fill: C.muted });

  o += tx(260, cy, '+', { fs: 20, fw: 700, fill: C.ink });

  o += box(274, cy - 28, 78, 56, C.lamber, C.amber);
  o += tx(313, cy - 8, 'b', { fs: 17, fw: 700, fill: D.orange.dark });
  o += tx(313, cy + 14, '(1, 10)', { mono: true, fs: 11.5, fill: D.orange.dark });

  o += arrow(356, cy, 388, cy, C.green);

  o += box(392, cy - 46, 152, 92, C.lgreen, C.green);
  o += tx(468, cy - 16, 'Ŷ = softmax(O)', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(468, cy + 8, '256 × 10', { mono: true, fs: 12.5, fill: C.ink });
  o += tx(468, cy + 28, 'each row Σ = 1', { fs: 11.5, fill: D.green.dark });

  o += box(16, 180, 528, 64, D.purple.tint, C.purple);
  o += tx(280, 202, '1 BLAS Matrix Multiply: 256 × 784 × 10 ≈ 2.0 × 10⁶ MACs (~4 MFLOPs)', { mono: true, fs: 12, fw: 700, fill: D.purple.dark });
  o += tx(280, 226, 'Rowwise softmax normalizes along axis=1 with keepdims=True', { fs: 12.5, fill: C.ink });
  return svg(W, H, o);
}

const ceSurprisal = () => twoCols(560, 264, 'Cross-Entropy Loss l(y, ŷ) = −log ŷ_y on ŷ = (0.21, 0.69, 0.10)',
  {
    head: 'True Label = Chicken (y = 2)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'Predicted prob: ŷ₂ = 0.69', mono: true, fw: 700 },
      { t: 'l = −log(0.69) ≈ 0.37 nats', mono: true, col: D.green.dark, fw: 700 },
      { t: 'Low surprisal: model assigned' },
      { t: '69% mass to the true class' },
    ],
  },
  {
    head: 'True Label = Dog (y = 3)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: 'Predicted prob: ŷ₃ = 0.10', mono: true, fw: 700 },
      { t: 'l = −log(0.10) ≈ 2.30 nats', mono: true, col: D.red.dark, fw: 700 },
      { t: '6.2× higher penalty! Diverges' },
      { t: 'to +∞ as ŷ_y → 0 (blunder)' },
    ],
  },
  { fill: C.lblue, stroke: C.blue, dark: D.blue.dark, t: 'Minimizing Mean Cross-Entropy = Maximizing Dataset Log-Likelihood' }
);

const mseVsCeGrad = () => twoCols(560, 266, 'Why Squared Error on Softmax Probabilities Stalls on Blunders',
  {
    head: 'Squared Loss ½‖ŷ − y‖₂²', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: '∂ŷₖ/∂oⱼ = ŷₖ(1_{j=k} − ŷⱼ)', mono: true, fs: 11.5 },
      { t: 'Slope wrt margin m = o_y − o_other:' },
      { t: '−2σ(m)(1 − σ(m))² → 0 as m → −∞', mono: true, fs: 11, col: D.red.dark, fw: 700 },
      { t: 'Vanishing gradient on blunders!', col: D.red.dark, fw: 700 },
    ],
  },
  {
    head: 'Cross-Entropy −log ŷ_y', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'log(exp(o_y)) cancels exponential!' },
      { t: 'Slope wrt margin m = o_y − o_other:' },
      { t: '−(1 − σ(m)) → −1 as m → −∞', mono: true, fs: 11.5, col: D.green.dark, fw: 700 },
      { t: 'Full unit gradient on blunders!', col: D.green.dark, fw: 700 },
    ],
  },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t: 'Logarithm undoes the softmax exponential so bad errors get full gradients' }
);

const residualGrad = () => threeStages(560, 262, 'Logit Gradient Is Prediction Minus Truth: ∇_o l = ŷ − y',
  { head: 'Predicted ŷ', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['cat:     0.21', 'chicken: 0.69', 'dog:     0.10'] },
  { head: 'True One-Hot y', fill: C.lamber, stroke: C.amber, dark: D.orange.dark, mono: true, lines: ['cat:     0', 'chicken: 1', 'dog:     0'] },
  { head: '∇_o l = ŷ − y', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['∂o₁ = +0.21 (↓)', '∂o₂ = −0.31 (↑)', '∂o₃ = +0.10 (↓)'] },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t1: 'Step −η∇_o l Boosts Chicken (+0.31η) and Suppresses Rivals', t2: 'Each rival class is pushed down in proportion to the probability mass it stole!' }
);

const templateNudge = () => twoCols(560, 266, 'Parameter Gradients & Global Convexity of Softmax Regression',
  {
    head: 'Rank-1 Template Update', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: '∇_W L = (1/n) Xᵀ(Ŷ − Y)', mono: true, fw: 700, col: D.blue.dark },
      { t: 'For example x with true class y:' },
      { t: 'w_y ← w_y + η(1 − ŷ_y) x  (pull)', mono: true, fs: 11.5, col: D.green.dark },
      { t: 'w_j ← w_j − η ŷ_j x       (push)', mono: true, fs: 11.5, col: D.red.dark },
    ],
  },
  {
    head: 'PSD Hessian ⇒ Global Convexity', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: '∇²_{o} l = diag(ŷ) − ŷŷᵀ', mono: true, fw: 700, col: D.green.dark },
      { t: '= Cov(e_Y) for Y ~ Cat(ŷ)', mono: true, fs: 12 },
      { t: 'vᵀ(∇²_{o} l)v = Var(v_Y) ≥ 0', mono: true, fs: 12 },
      { t: 'No spurious local minima!' },
    ],
  },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t: 'Every SGD step nudges the true class template toward x and rivals away' }
);

const entropyKl = () => threeStages(560, 262, 'Shannon Information Theory: Cross-Entropy = Entropy + KL Divergence',
  { head: 'Entropy H(p)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['−Σ pⱼ log pⱼ', 'Irreducible label', 'uncertainty (≥ 0)'] },
  { head: 'KL Div D_KL(p‖p̂)', fill: C.lamber, stroke: C.amber, dark: D.orange.dark, mono: true, lines: ['Σ pⱼ log(pⱼ/p̂ⱼ)', 'Extra nats from', 'wrong model (≥ 0)'] },
  { head: 'Cross-Ent H(p,p̂)', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['H(p) + D_KL(p‖p̂)', 'Uniquely min at', 'p̂ = p (Gibbs!)'] },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t1: 'Why Cross-Entropy Rewards Honest Calibrated Probabilities', t2: 'On a 60%/40% photo, predicting (0.6, 0.4) beats (0.9, 0.1) (0.67 vs 0.98 nats)' }
);

// ═══════════════════════════════════════════════════════════════════════════
// §3.2 image-classification-dataset.md (10 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const benchmarkSweetspot = () => stackCards(560, 286, 'Why Fashion-MNIST Is the Goldilocks Vision Benchmark', [
  { head: 'MNIST Handwritten Digits (1998) — Too Easy', badge: 'Linear ~92%', sub: '70k 28×28 digits: trivial models saturate >99% with no headroom', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
  { head: 'Fashion-MNIST Apparel (2017) — Sweet Spot', badge: 'Linear ~83%', sub: '70k 28×28 garments across 10 classes: trains in seconds, CNNs >90%', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
  { head: 'ImageNet-1K Natural Photos (2009) — Heavy', badge: '1.2M × 224²', sub: '1,000 categories (~150 GB): requires hours of multi-GPU compute', fill: C.lblue, stroke: C.blue, dark: D.blue.dark },
]);

const garmentTaxonomy = () => stackCards(560, 286, '10 Balanced Fashion-MNIST Classes Grouped by Silhouette', [
  { head: 'Distinct Silhouettes', badge: '1, 3, 8', sub: 'Trouser (1), dress (3), bag (8): leg gap, long single body, handles', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
  { head: 'Footwear Cluster (Linear Recall ~88%–95%)', badge: '5, 7, 9: shoes', sub: 'Sandal (5), sneaker (7), ankle boot (9): distinct side-profile shoes', fill: C.lblue, stroke: C.blue, dark: D.blue.dark },
  { head: 'Boxy Upper-Body Tops (Bottleneck)', badge: '0, 2, 4, 6', sub: 'T-shirt, pullover, coat, shirt overlap; shirt recall is ~55%!', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark },
]);

const pixelRescale = () => threeStages(560, 256, 'FashionMNIST DataModule Preprocessing Pipeline',
  { head: '1. Raw IDX Bytes', fill: N.panel, stroke: C.gray, dark: C.ink, mono: true, lines: ['uint8 in [0..255]', '1 byte / pixel', '0 = dark bg'] },
  { head: '2. Resize & Cast', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['resize=(28, 28)', 'astype(float32)', 'divide by 255.0'] },
  { head: '3. Tensor Split', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['float32 in [0, 1]', '4 bytes / pixel', 'ready for X @ W'] },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t1: 'Why Rescale Pixels to [0, 1] Before Linear Layer?', t2: 'Keeps initial logits Wᵀx near 0 so softmax starts near uniform (1/10)' }
);

const datasetSplits = () => twoCols(560, 260, 'Fashion-MNIST Splits & Memory Footprint (784 Pixels/Image)',
  {
    head: 'Training Split (data.train)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: '60,000 images (6,000 / class)', fw: 700 },
      { t: 'uint8:   47.0 MB', mono: true },
      { t: 'float32: 188.2 MB', mono: true, col: D.blue.dark, fw: 700 },
      { t: 'Reshuffled every epoch' },
    ],
  },
  {
    head: 'Validation Split (data.val)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: '10,000 images (1,000 / class)', fw: 700 },
      { t: 'uint8:   7.8 MB', mono: true },
      { t: 'float32: 31.4 MB', mono: true, col: D.green.dark, fw: 700 },
      { t: 'Deterministic order (no shuffle)' },
    ],
  },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t: 'Entire dataset (~220 MB in float32) fits comfortably in host RAM & VRAM' }
);

const idxHeader = () => twoCols(560, 256, 'Parsing Official IDX Binary Files in MXNet',
  {
    head: 'Images (*-images-idx3-ubyte.gz)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: 'Bytes 0..15: 16-byte header', mono: true, fw: 700 },
      { t: '(magic, count, 28, 28)', mono: true, fs: 11.5 },
      { t: 'Bytes 16..: raw uint8 pixels', mono: true },
      { t: 'reshape(-1, 28, 28, 1)', mono: true, col: D.blue.dark },
    ],
  },
  {
    head: 'Labels (*-labels-idx1-ubyte.gz)', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    lines: [
      { t: 'Bytes 0..7: 8-byte header', mono: true, fw: 700 },
      { t: '(magic, count)', mono: true, fs: 11.5 },
      { t: 'Bytes 8..: 1 byte per class', mono: true },
      { t: 'astype(int32) ∈ {0..9}', mono: true, col: D.purple.dark },
    ],
  },
  { fill: C.lgreen, stroke: C.green, dark: D.green.dark, t: 'Yields the standard 60,000 / 10,000 train / validation split' }
);

const epochBatches = () => threeStages(560, 258, 'One Training Epoch at batch_size = 64 (60,000 Images)',
  { head: '1. Permute 60k', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['shuffle=True', 'fresh order', 'every epoch'] },
  { head: '2. 937 Full Batches', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['937 × 64', '= 59,968 imgs', '200 KB / batch'] },
  { head: '3. Partial Batch 938', fill: C.lamber, stroke: C.amber, dark: D.orange.dark, mono: true, lines: ['1 × 32 images', 'retained so all', '60k are visited!'] },
  { fill: C.lgreen, stroke: C.green, dark: D.green.dark, t1: 'At batch_size = 256 (§3.3): 60,000 = 234 × 256 + 96 → 235 Batches/Epoch', t2: 'd2l.Averager weights the final partial batch by its exact size (32 or 96)' }
);

const nchwVsNhwc = () => twoCols(560, 260, 'Two Conventions for 4D Minibatch Image Tensors',
  {
    head: 'Channel-First: NCHW', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: 'Channel axis before rows, cols', fw: 700 },
      { t: 'Single image: (1, 28, 28)', mono: true },
      { t: 'Minibatch X:  (64, 1, 28, 28)', mono: true, col: D.blue.dark, fw: 700 },
      { t: 'Each channel plane contiguous' },
    ],
  },
  {
    head: 'Channel-Last: NHWC', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'Channel axis after rows, cols', fw: 700 },
      { t: 'Single image: (28, 28, 1)', mono: true },
      { t: 'Minibatch X:  (64, 28, 28, 1)', mono: true, col: D.green.dark, fw: 700 },
      { t: 'Channels at pixel (r,c) contiguous' },
    ],
  },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t: 'Both flatten to (64, 784) via X.reshape((-1, 784)) for softmax regression' }
);

const minibatchFlatten = () => threeStages(560, 256, 'From 4D Spatial Image Batch to 2D Design Matrix',
  { head: '4D Image Batch X', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['(64, 1, 28, 28)', 'or (64, 28, 28, 1)', 'float32 (200 KB)'] },
  { head: 'Flatten Spatial Axes', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['reshape(-1, 784)', 'O(1) zero-copy view', '28 × 28 = 784'] },
  { head: '2D Matrix + 1D y', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['X\\_flat: (64, 784)', 'y: (64,) int32/64', 'ready for X @ W'] },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t1: 'Note: Flattening Discards 2D Spatial Neighborhood Structure', t2: 'Linear models treat pixels as unordered coordinates; CNNs (Ch 6) keep 2D grid' }
);

const producerConsumer = () => twoCols(560, 264, 'Producer–Consumer Pipeline: T_epoch ≥ max(T_load, T_compute)',
  {
    head: 'Data-Starved (T_load > T_comp)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: '• Small batch_size (1 or 16):', fw: 700 },
      { t: '  per-batch dispatch dominates' },
      { t: '• A single loading process:' },
      { t: '  GPU sits idle waiting for CPU!' },
    ],
  },
  {
    head: 'Compute-Bound (T_load ≤ T_comp)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: '• Batch size 64–256 amortizes IPC', fw: 700 },
      { t: '• Parallel loading workers' },
      { t: '  prefetch batch b+1 while GPU' },
      { t: '  trains on batch b (full overlap!)' },
    ],
  },
  { fill: C.lblue, stroke: C.blue, dark: D.blue.dark, t: 'Always benchmark a standalone pass over train_dataloader() before training!' }
);

const visualDebug = () => stackCards(560, 284, 'What Visual Inspection (data.visualize) Catches vs. Misses', [
  { head: '1. Label Offsets & Spatial Transposes', badge: 'Caught by eye', sub: 'Off-by-one class names ("trouser" on a boot) or sideways rotated garments', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
  { head: '2. Channel Scrambling or Sign Inversion', badge: 'Caught by eye', sub: 'White background (1 − X) instead of dark background (0) is obvious', fill: C.lblue, stroke: C.blue, dark: D.blue.dark },
  { head: '3. Unnormalized [0, 255] Float Scale', badge: 'Needs X.max() check', sub: 'imshow auto-scales [vmin, vmax] while unscaled pixels blow up logits 255×!', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark },
]);

// ═══════════════════════════════════════════════════════════════════════════
// §3.3 softmax-regression-scratch.md (9 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const fp32Hazards = () => twoCols(560, 266, 'Twin Float32 Numerical Hazards of Naive Softmax + Log',
  {
    head: '1. Overflow (o_k > 88.7)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: 'float32 max ≈ 3.4 × 10³⁸', mono: true },
      { t: 'exp(1000) → inf', mono: true, col: D.red.dark, fw: 700 },
      { t: 'inf / (inf + 1 + 1) → NaN!', mono: true, col: D.red.dark, fw: 700 },
      { t: 'Fixed by shifting o − o_max' },
    ],
  },
  {
    head: '2. Underflow (o_y − o_max < −104)', fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    lines: [
      { t: 'exp(−200) → 0.0 in float32', mono: true, col: D.orange.dark, fw: 700 },
      { t: '−log(0.0) → +inf!', mono: true, col: D.red.dark, fw: 700 },
      { t: 'Clamping p ≥ 1e-12 caps loss at' },
      { t: '27.6 but ZEROES the gradient!' },
    ],
  },
  { fill: C.lgreen, stroke: C.green, dark: D.green.dark, t: 'Permanent Fix (§3.3 Concise): Fuse log and softmax via LogSumExp on raw logits' }
);

const rowwiseBroadcast = () => threeStages(560, 258, 'Rowwise Softmax Normalization With keepdims=True',
  { head: '1. O\\_exp = exp(O)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['shape: (n, q)', 'e.g. (256, 10)', 'positive entries'] },
  { head: '2. Row Partition Sum', fill: C.lamber, stroke: C.amber, dark: D.orange.dark, mono: true, lines: ['reduce_sum(axis=1,', '  keepdims=True)', 'shape: (n, 1)!'] },
  { head: '3. Broadcast Divide', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['(n, q) / (n, 1)', '→ Ŷ: (n, q)', 'each row sums to 1'] },
  { fill: D.red.tint, stroke: D.red.base, dark: D.red.dark, t1: 'Why keepdims=True Is Mandatory', t2: 'Without keepdims=True, row sums have shape (n,), and (n, q)/(n,) fails or misaligns!' }
);

const maxShiftRange = () => threeStages(560, 256, 'Max-Shift Invariance Traps Exponentials Inside (0, 1]',
  { head: 'Extreme Logits o', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark, mono: true, lines: ['[1000.0, 0.0, 0.0]', 'o_max = 1000.0', 'naive exp → inf'] },
  { head: 'Shifted o − o_max', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['[0.0, −1000, −1000]', 'all entries ≤ 0', 'max entry == 0.0'] },
  { head: 'Safe Softmax', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['exp: [1.0, 0.0, 0.0]', 'sum = 1.0 ∈ [1, q]', 'ŷ = [1.0, 0.0, 0.0]'] },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t1: 'Since max(o_k − o_max) = 0, Partition Sum Σ exp(o_k − o_max) Lies in [1, q]!', t2: 'Division by 0 and overflow to ∞ are both mathematically impossible' }
);

const classifierArch = () => twoCols(560, 260, 'The d2l.Classifier Base Class: Dual-Metric Validation',
  {
    head: 'training_step(batch)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: 'Inherited from d2l.Module:' },
      { t: 'Y_hat = self(*batch[:-1])', mono: true, fs: 11.5 },
      { t: 'return self.loss(Y_hat, Y)', mono: true, fs: 11.5, col: D.blue.dark, fw: 700 },
      { t: 'Differentiable cross-entropy' },
    ],
  },
  {
    head: 'validation_step(batch)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'Overridden in d2l.Classifier:' },
      { t: '{\'loss\': self.loss(Y_hat, Y),', mono: true, fs: 11.5, col: D.green.dark, fw: 700 },
      { t: ' \'acc\':  self.accuracy(Y_hat, Y)}', mono: true, fs: 11.5, col: D.green.dark, fw: 700 },
      { t: 'Logs both val_loss & val_acc!' },
    ],
  },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t: 'accuracy() compares argmax(Y\\_hat, axis=1) == Y on raw logits OR probabilities' }
);

const scratchParams = () => threeStages(560, 258, 'SoftmaxRegressionScratch: 7,850 Parameters & Initial Loss Check',
  { head: 'Flatten Input X', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['(256, 1, 28, 28)', '→ (256, 784)', 'd = 784 pixels'] },
  { head: 'Parameters (W, b)', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['W: (784,10) σ=0.01', 'b: (10,) zeros', '7,840 + 10 = 7,850'] },
  { head: 'Init Sanity Check', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['O ≈ 0 ⇒ ŷⱼ ≈ 1/10', 'L₀ ≈ −log(0.1)', '= log 10 ≈ 2.30'] },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t1: 'Always Check Epoch-0 Loss Against log(q)!', t2: 'For q = 10 classes, initial cross-entropy must start near log(10) ≈ 2.3026 nats' }
);

const gatherVsOnehot = () => twoCols(560, 260, 'Selecting True-Class Probabilities: O(n) Gather vs. O(nq) One-Hot',
  {
    head: 'Integer Gather: y\\_hat[range(n), y]', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'y\\_hat = [[0.1, 0.3, 0.6],', mono: true, fs: 11.5 },
      { t: '         [0.3, 0.2, 0.5]]', mono: true, fs: 11.5 },
      { t: 'y = [0, 2] → picks [0.1, 0.5]', mono: true, fs: 11.5, col: D.green.dark, fw: 700 },
      { t: 'O(n) time & O(n) memory!' },
    ],
  },
  {
    head: 'Dense One-Hot: −Σ Y_ij log Ŷ_{ij}', fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    lines: [
      { t: 'Allocates n × q one-hot matrix Y' },
      { t: 'Multiplies (q − 1) zeros per row' },
      { t: 'At q = 10⁵ vocab & n = 256:' },
      { t: 'wastes 100 MB per minibatch!', col: D.red.dark, fw: 700 },
    ],
  },
  { fill: C.lblue, stroke: C.blue, dark: D.blue.dark, t: 'One gather call plucks 1 prob per row; no n × q matrix is built' }
);

const confusionAccum = () => threeStages(560, 258, 'Vectorized 10 × 10 Confusion Matrix Accumulation: N += PᵀY',
  { head: 'Predicted One-Hot P', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['ĉ = argmax(O, 1)', 'P = one_hot(ĉ, 10)', 'shape: (n, 10)'] },
  { head: 'True One-Hot Y', fill: C.lamber, stroke: C.amber, dark: D.orange.dark, mono: true, lines: ['true labels y', 'Y = one_hot(y, 10)', 'shape: (n, 10)'] },
  { head: 'Count Matrix PᵀY', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['(10, n) @ (n, 10)', '→ N: (10, 10)', 'Σ N_ij = 10,000'] },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t1: 'Column Normalization: C = N / N.sum(axis=0, keepdims=True)', t2: 'Entry C_{ij} = P(predicted = i | true = j) so every column j sums to 1.0' }
);

const rawLogitsContract = () => twoCols(560, 264, 'Why Concise SoftmaxRegression.forward Returns Raw Logits O',
  {
    head: 'Correct: forward(X) → Logits O', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'net = Flatten → linear(q = 10)', mono: true, fs: 11 },
      { t: '(no softmax layer at the end)', fs: 11 },
      { t: '• accuracy() uses argmax(O, 1)' },
      { t: '• Built-in CE fuses log_softmax', col: D.green.dark, fw: 700 },
    ],
  },
  {
    head: 'Bug: Putting softmax in forward', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: 'Passes probabilities p ∈ [0, 1]' },
      { t: 'into the built-in cross-entropy:' },
      { t: 'Applies softmax TWICE!', col: D.red.dark, fw: 700 },
      { t: 'Loss trapped ≥ log(e+9)−1 ≈ 1.46!', mono: true, fs: 11, col: D.red.dark, fw: 700 },
    ],
  },
  { fill: C.lblue, stroke: C.blue, dark: D.blue.dark, t: 'Rule: Never place a softmax layer before a framework cross-entropy loss!' }
);

const logsumexpFusion = () => threeStages(560, 262, 'The Fused LogSumExp Cross-Entropy Identity',
  { head: '1. Shift by o_max', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['o_max = max_{k} o_k', 'd_k = o_k − o_max', 'all d_k ≤ 0'] },
  { head: '2. Safe LogSumExp', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['S = Σ exp(d_k)', 'o_max + log(S)', 'never overflows!'] },
  { head: '3. Subtract o_y', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['l = LSE(o) − o_y', 'linear −o_y term', 'never underflows!'] },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t1: 'On o = (1000, 0, 0): True Class 0 Gives l = 0.0; True Class 1 Gives l = 1000.0', t2: 'Gradient ∇_o l = softmax(o) − y stays finite and nonzero for any logit magnitude' }
);

// ═══════════════════════════════════════════════════════════════════════════
// §3.4 classification.md (10 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const accuracyLimits = () => stackCards(560, 284, 'Three Real-World Conditions That Break Raw Accuracy', [
  { head: '1. Class Imbalance (Rare Positives, π = 1%)', badge: '99% trivial acc', sub: 'True negatives (TN) swamp the denominator while all positives are missed', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark },
  { head: '2. Asymmetric Error Costs (κ_FN ≫ κ_FP)', badge: '100:1 cost ratio', sub: 'Missing a cancer diagnosis costs far more than ordering a follow-up lab test', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
  { head: '3. Uncalibrated Confidence Scores', badge: 'ECE = Σ(n_b/n)|a_b−c̄_b|', sub: 'Cost thresholds act directly on p̂, failing if the model is overconfident', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark },
]);

function imbalanceMatrix() {
  const W = 560, H = 266;
  let o = '';
  o += tx(280, 20, '1% Disease Screening (1,000 Sick, 99,000 Healthy): Accuracy Paradox', { fs: 14.5, fw: 700, fill: C.ink });

  o += tx(168, 46, 'True Sick (1k)', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(304, 46, 'True Healthy (99k)', { fs: 12.5, fw: 700, fill: D.blue.dark });

  o += tx(94, 92, 'Pred Sick', { fs: 12.5, fw: 700, fill: C.ink, anchor: 'end' });
  o += box(106, 60, 124, 64, C.lgreen, C.green);
  o += tx(168, 82, 'TP = 900', { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(168, 104, '(90% caught)', { fs: 11.5, fill: C.ink });

  o += box(242, 60, 124, 64, C.lamber, C.amber);
  o += tx(304, 82, 'FP = 4,950', { mono: true, fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(304, 104, '(5% false alarm)', { fs: 11.5, fill: C.ink });

  o += tx(94, 168, 'Pred Healthy', { fs: 12.5, fw: 700, fill: C.ink, anchor: 'end' });
  o += box(106, 136, 124, 64, D.red.tint, D.red.base);
  o += tx(168, 158, 'FN = 100', { mono: true, fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(168, 180, '(10% missed)', { fs: 11.5, fill: C.ink });

  o += box(242, 136, 124, 64, C.lblue, C.blue);
  o += tx(304, 158, 'TN = 94,050', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(304, 180, '(95% cleared)', { fs: 11.5, fill: C.ink });

  o += box(382, 60, 162, 140, D.purple.tint, C.purple);
  o += tx(463, 84, 'Raw Accuracy:', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += tx(463, 112, 'Always Healthy:', { fs: 12, fill: C.ink });
  o += tx(463, 132, '99.0% (0 TP!)', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(463, 160, '90%-Recall Test:', { fs: 12, fill: C.ink });
  o += tx(463, 180, '95.0% (900 TP)', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });

  o += box(16, 214, 528, 38, C.lgreen, C.green);
  o += tx(280, 233, 'TN = 94,050 dominates accuracy; we need ratios that ignore TN!', { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

const recallVsPrecision = () => twoCols(560, 262, 'Isolating Positive Detection: Recall (Column) vs. Precision (Row)',
  {
    head: 'Recall = TP / (TP + FN)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'Reads down True Sick column:' },
      { t: '900 / (900 + 100) = 90.0%', mono: true, col: D.green.dark, fw: 700 },
      { t: 'P(flagged | truly sick)', mono: true },
      { t: 'Invariant to disease prevalence!' },
    ],
  },
  {
    head: 'Precision = TP / (TP + FP)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: 'Reads across Pred Sick row:' },
      { t: '900 / (900 + 4,950) = 15.4%', mono: true, col: D.blue.dark, fw: 700 },
      { t: 'P(truly sick | flagged)', mono: true },
      { t: 'Depends on population prevalence!' },
    ],
  },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t: 'Two-stage screening catches 90% and sends 5.85% to confirmatory testing' }
);

const prevalenceBayes = () => stackCards(560, 284, 'Same Test (90% Recall, 5% FPR) Across Population Prevalences π', [
  { head: 'Rare Screening: π = 0.1% Sick → Precision = 1.8%', badge: 'Acc = 95.0%', sub: '90 TP vs 4,995 FP from the 99.9% healthy majority (55:1 false alarms)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark },
  { head: 'General Clinic: π = 1.0% Sick → Precision = 15.4%', badge: 'Acc = 95.0%', sub: '900 TP vs 4,950 FP (1 in 6.5 flagged patients is genuinely sick)', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
  { head: 'Symptomatic Ward: π = 10% Sick → Precision = 66.7%', badge: 'At 50%: 94.7%', sub: 'Precision = (recall·π) / (recall·π + FPR·(1−π)) climbs steeply with π!', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
]);

const asymmetricCost = () => twoCols(560, 260, 'Expected Cost Per Person Under κ_FN = 100 and κ_FP = 1',
  {
    head: '"Always Healthy" Rule (99% Acc)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: 'FN = 1,000 missed sick patients', fw: 700 },
      { t: 'FP = 0 false alarms' },
      { t: 'Total = 100 × 1,000 = 100,000', mono: true, fs: 11.5 },
      { t: 'Expected cost = 1.00 / person', mono: true, col: D.red.dark, fw: 700 },
    ],
  },
  {
    head: 'Screening Test (95% Acc)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'FN = 100 missed, FP = 4,950', fw: 700 },
      { t: 'Total = 100(100) + 4,950 = 14,950', mono: true, fs: 11.5 },
      { t: 'Expected cost = 0.15 / person', mono: true, col: D.green.dark, fw: 700 },
      { t: '6.7× lower cost than trivial rule!', col: D.green.dark, fw: 700 },
    ],
  },
  { fill: C.lblue, stroke: C.blue, dark: D.blue.dark, t: 'Expected Cost = (κ_FN · FN + κ_FP · FP) / n aligns evaluation with real stakes' }
);

const costThreshold = () => threeStages(560, 260, 'Deriving the Bayes-Optimal Decision Threshold τ*',
  { head: 'Predict Healthy', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark, mono: true, lines: ['Misses if sick (p)', 'Expected cost:', 'C₀ = p · κ_FN'] },
  { head: 'Predict Sick', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['False alarm (1−p)', 'Expected cost:', 'C₁ = (1 − p) · κ_FP'] },
  { head: 'Optimal Rule C₁ < C₀', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['p > τ* =', 'κ_FP / (κ_FP + κ_FN)', '= 1 / 101 ≈ 0.01'] },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t1: 'Equal Costs (κ_FN = κ_FP) Give τ* = 0.50; 100:1 Cost Drops τ* to 1%!', t2: 'Requires calibrated probabilities p = P(sick | x) so numerical thresholds are valid' }
);

const thresholdSweep = () => twoCols(560, 264, 'Sweeping Threshold τ ∈ [0, 1]: Trade-Offs & Summary Curves',
  {
    head: 'Lowering Threshold τ ↓', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: 'Flags more examples as positive:' },
      { t: '• Recall (TPR) ↑ monotonically', col: D.green.dark, fw: 700 },
      { t: '• Precision ↓ (more false alarms)', col: D.orange.dark, fw: 700 },
      { t: '• F1 = 2·Prec·Rec / (Prec + Rec)', mono: true, fs: 11.5 },
    ],
  },
  {
    head: 'ROC vs. Precision–Recall Curve', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark,
    lines: [
      { t: '• ROC (TPR vs FPR): invariant to π;' },
      { t: '  AUC = P(s_pos > s_neg)', mono: true, fs: 11.5, col: D.purple.dark, fw: 700 },
      { t: '• PR Curve (Precision vs Recall):' },
      { t: '  exposes false alarms when π ≪ 1!', fw: 700 },
    ],
  },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t: 'Under 1% prevalence, ROC can look great (90% TPR, 5% FPR) while precision is 15%' }
);

const calibrationMeaning = () => twoCols(560, 264, 'Accuracy and Calibration Are Orthogonal Properties',
  {
    head: 'Calibrated But Useless (10% Acc)', fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    lines: [
      { t: 'Always predicts ŷ = (0.1, ..., 0.1)', mono: true, fs: 11.5 },
      { t: 'on 10 balanced classes:' },
      { t: 'Confidence c̄ = 0.10, Acc = 0.10', mono: true, fw: 700 },
      { t: 'ECE = 0.000, zero discrimination!' },
    ],
  },
  {
    head: 'Accurate & Overconfident (95%)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: 'Right on 95% of examples, but' },
      { t: 'always reports confidence c̄ = 1.00:', mono: true, fs: 11.5 },
      { t: 'ECE = |0.95 − 1.00| = 0.050', mono: true, col: D.red.dark, fw: 700 },
      { t: 'Dangerous for cost thresholds!' },
    ],
  },
  { fill: C.lgreen, stroke: C.green, dark: D.green.dark, t: 'Goal: High accuracy AND honest confidence P(correct | p̂ = α) = α' }
);

const eceBins = () => threeStages(560, 258, 'Computing Expected Calibration Error (ECE) Across B = 10 Bins',
  { head: '1. Per-Example Stats', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['conf p̂ᵢ = max_{j} ŷᵢⱼ', 'corr cᵢ = (ĉᵢ == yᵢ)', 'i = 1..10,000'] },
  { head: '2. Bin Averages', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['10 bins of width 0.1', 'c̄_b = mean(p̂ in I_b)', 'a_b = mean(c in I_b)'] },
  { head: '3. Weighted Gap', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['ECE =', 'Σ (n_b/n)|a_b − c̄_b|', '< 0.03 on Fashion'] },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t1: 'Pigeonhole Lower Bound: max_{j} ŷ_j ≥ 1/q = 0.10 for q = 10 Classes', t2: 'Bin [0, 0.1) is always empty (n₁ = 0); filter to bins with n_b ≥ 100 when plotting' }
);

const tempRecalibration = () => threeStages(560, 260, 'Post-Hoc Temperature Scaling: ŷ(T) = softmax(o / T)',
  { head: '1. Freeze Weights W', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['Collect val logits O', 'Split val in half:', '5k fit / 5k eval'] },
  { head: '2. 1D Convex Fit T*', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['T* = argmin_{T}', '  NLL(O_half / T, y)', 'T* ≈ 0.92 on Fashion'] },
  { head: '3. Recalibrated ŷ(T*)', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['T > 1: softens', 'T < 1: sharpens', 'Accuracy 100% same!'] },
  { fill: D.red.tint, stroke: D.red.base, dark: D.red.dark, t1: 'Never Fit Temperature T on the Training Set!', t2: 'Training margins are optimistically wide; fit T on held-out validation data' }
);

// ═══════════════════════════════════════════════════════════════════════════
// §3.5 generalization-classification.md (11 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const genObstacles = () => twoCols(560, 262, 'Two Obstacles Between Measured Accuracy and True Population Risk',
  {
    head: '1. Finite Test Sample & Reuse', fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    lines: [
      { t: 'Even n = 10,000 test images has' },
      { t: '±0.7% sampling uncertainty (95%)', fw: 700, col: D.orange.dark },
      { t: 'Reusing the test set across k models' },
      { t: 'cherry-picks noise & erodes bounds!' },
    ],
  },
  {
    head: '2. Memorization vs. Learning', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: 'Lookup-table memorizer achieves' },
      { t: '0% training error, 90% test error!', fw: 700, col: D.red.dark },
      { t: 'Why does low training error certify' },
      { t: 'a linear model (7,850 params)?' },
    ],
  },
  { fill: C.lblue, stroke: C.blue, dark: D.blue.dark, t: 'Answered by Concentration Bounds (CLT / Hoeffding) & VC Uniform Convergence' }
);

const bernoulliSe = () => threeStages(560, 258, 'Test Error of a Fixed Classifier f as a Bernoulli Average',
  { head: '1. Fixed Classifier f', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['Chosen BEFORE', 'drawing test set D', 'true error ε(f)'] },
  { head: '2. n i.i.d. Coin Flips', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['eᵢ = (f(xᵢ) ≠ yᵢ)', 'eᵢ ~ Bernoulli(ε)', 'Var(eᵢ) = ε(1−ε) ≤ ¼'] },
  { head: '3. Test Error ε_D(f)', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['E[ε_D] = ε(f)', 'SE = √(ε(1−ε)/n)', '≤ 0.5 / √n'] },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t1: 'On Fashion-MNIST (ε ≈ 0.15, n = 10,000): SE = √(0.15 × 0.85 / 10,000) ≈ 0.36%', t2: 'Square-root law: cutting the error bar in half requires 4× as many test examples!' }
);

const clt10k = () => twoCols(560, 260, 'Why Standard Vision Benchmarks Use n = 10,000 Test Examples',
  {
    head: '95% Normal (CLT) Interval', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: 'ε_D(f) ± 1.96 √(ε(1 − ε) / n)', mono: true, fw: 700, col: D.blue.dark },
      { t: 'At ε = 0.15, n = 10,000:' },
      { t: 'radius = 1.96 × 0.36% = ±0.70%', mono: true, fw: 700 },
      { t: 'Worst-case variance is at ε = 0.50' },
    ],
  },
  {
    head: 'Sample Size for ±1% Margin', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'Demand worst-case radius ≤ 0.01:' },
      { t: '1.96 × (0.5 / √n) ≤ 0.01', mono: true },
      { t: 'n ≥ (0.98 / 0.01)² = 9,604!', mono: true, col: D.green.dark, fw: 700 },
      { t: 'Rounded up → n = 10,000 benchmark!' },
    ],
  },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t: 'MNIST, Fashion-MNIST, and CIFAR-10 all ship with n = 10,000 test images' }
);

const hoeffdingVsClt = () => twoCols(560, 266, 'Asymptotic Normal Interval vs. Finite-Sample Hoeffding Bound',
  {
    head: 'Plug-In Normal (CLT)', fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    lines: [
      { t: 'r = 1.96 √(ε_D(1 − ε_D) / n)', mono: true, fs: 11.5, fw: 700 },
      { t: 'n ≥ 9,604 for worst-case ±1%', mono: true, fs: 11.5 },
      { t: 'Failure mode: when ε_D = 0 errors', col: D.red.dark, fw: 700 },
      { t: 'on small n, radius collapses to 0!' },
    ],
  },
  {
    head: 'Hoeffding Bound (Exact ∀ n, ε)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'P(|ε_D − ε| ≥ t) ≤ 2 exp(−2nt²)', mono: true, fs: 11, fw: 700 },
      { t: 'r = √(log(2/δ) / (2n))', mono: true, fs: 12, col: D.green.dark, fw: 700 },
      { t: 'At 95% (δ=0.05): r ≈ 1.36 / √n', mono: true, fs: 11.5 },
      { t: 'n ≥ 18,445 for ±1% (exact!)' },
    ],
  },
  { fill: C.lblue, stroke: C.blue, dark: D.blue.dark, t: 'When ε_D = 0, the "Rule of Three" gives 95% upper bound [0, 3/n]' }
);

const unionBound = () => threeStages(560, 260, 'Evaluating k Fixed Classifiers on One Test Set: The Union Bound',
  { head: '1 Model (k = 1)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['P(bad₁) ≤ δ', 'r = √(log(40)/2n)', '±1.36% at n=10k'] },
  { head: 'Union Over k Models', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['P(∪ bad) ≤ Σ P(bad)', '≤ 2k exp(−2nt²)', 'set RHS = δ = 0.05'] },
  { head: 'Grows as √log k', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['r = √(log(2k/δ)/2n)', 'k = 10³ → ±2.30%', 'k = 10⁶ → ±2.96%'] },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t1: 'Non-Adaptive Comparisons Pay Only a Logarithmic Penalty √(log k)', t2: 'Requires all k candidate models to be fixed BEFORE observing any test-set score!' }
);

const adaptiveOverfit = () => twoCols(560, 266, 'Non-Adaptive Selection vs. Adaptive Test-Set Overfitting (n = 1,000)',
  {
    head: 'Non-Adaptive Best of k = 1,000', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: '1,000 independent fair-coin guessers' },
      { t: '(true population accuracy = 50.0%)' },
      { t: 'Best single guesser on test set:' },
      { t: 'acc ≈ 55–56%  (0.5 + √(log k/2n))', mono: true, fs: 11, col: D.blue.dark, fw: 700 },
    ],
  },
  {
    head: 'Adaptive Majority Vote (k = 1k)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: 'Keep guessers with test acc > 50%' },
      { t: 'and take their majority vote:' },
      { t: 'Test accuracy jumps to ~70–72%!', mono: true, col: D.red.dark, fw: 700 },
      { t: 'Population accuracy is STILL 50.0%!', col: D.red.dark, fw: 700 },
    ],
  },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t: 'Once test feedback guides model design, the √(log k) union bound breaks!' }
);

const uniformConv = () => threeStages(560, 258, 'Why Training Selection Requires Uniform Convergence Over F',
  { head: 'Fixed f Before S', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['E[ε_S(f)] = ε(f)', 'Single Hoeffding', 'bound applies'] },
  { head: 'Trained f_S ∈ F', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark, mono: true, lines: ['Chosen AFTER seeing', 'training set S!', 'ε_S(f_S) biased ↓'] },
  { head: 'Uniform Bound ∀f∈F', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['Bound sup_{f∈F}', '|ε_S(f) − ε(f)|', 'covers f_S too!'] },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t1: 'For Finite Hypothesis Class |F| < ∞: ε(f) ≤ ε_S(f) + √(log(2|F|/δ) / (2n))', t2: 'What if |F| = ∞ (continuous weights)? Count distinct labelings on n points instead!' }
);

const dichotomies = () => twoCols(560, 262, 'From Infinite Continuous Weights to Finite Binary Labelings',
  {
    head: 'Continuous Parameter Space', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: 'f(x) = sign(wᵀx + b),  w ∈ ℝᵈ', mono: true, fs: 11.5 },
      { t: 'Uncountably many weight vectors:' },
      { t: '|F| = ∞  ⇒  log|F| = ∞ (vacuous)', mono: true, fs: 11.5, col: D.red.dark, fw: 700 },
    ],
  },
  {
    head: 'Equivalence on n Sample Points', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'On x⁽¹⁾, ..., x⁽ⁿ⁾, any f outputs' },
      { t: 'one binary vector in {−1, +1}ⁿ:' },
      { t: 'At most 2ⁿ distinct labelings!', col: D.green.dark, fw: 700 },
      { t: 'Shattering m pts = realizing all 2ᵐ' },
    ],
  },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t: 'VC Dimension v = VC(F) is the largest m that F can shatter' }
);

const radonVc = () => twoCols(560, 264, 'VC Dimension of Linear Halfspaces in ℝᵈ Equals d + 1',
  {
    head: 'Lower Bound: VC ≥ d + 1', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'Pick origin 0 and basis e₁, ..., e_d:' },
      { t: 'Given any labels y₀, y₁, ..., y_d ∈ {±1}:', fs: 11.5 },
      { t: 'Set b = y₀ and wᵢ = yᵢ − y₀', mono: true, col: D.green.dark, fw: 700 },
      { t: 'Realizes all 2^{d+1} labelings!' },
    ],
  },
  {
    head: 'Upper Bound: VC < d + 2 (Radon)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: 'Any d+2 vectors [xᵢ, 1] ∈ ℝ^{d+1}' },
      { t: 'are dependent: Σ aᵢ[xᵢ, 1] = 0', mono: true, fs: 11.5 },
      { t: 'Convex hulls of {aᵢ>0} & {aᵢ<0} cross!', col: D.red.dark, fw: 700 },
      { t: 'No hyperplane can separate them' },
    ],
  },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t: 'Caution: VC(F) ≠ param count in general (1-param sin(ωx) has VC = ∞!)' }
);

const sauerGrowth = () => twoCols(560, 264, 'Sauer\'s Lemma: Polynomial O(nᵛ) vs. Exponential 2ⁿ Growth',
  {
    head: 'Unrestricted Memorizer (v = ∞)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: 'Realizes all 2ⁿ labelings on n pts', fw: 700 },
      { t: 'At n = 100 points in ℝ²:' },
      { t: '2¹⁰⁰ ≈ 1.27 × 10³⁰ labelings', mono: true, col: D.red.dark, fw: 700 },
      { t: 'log(2ⁿ)/n = log 2 (never vanishes!)' },
    ],
  },
  {
    head: 'Linear Halfplanes in ℝ² (v = 3)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'Sauer: Σ_{i≤v} C(n,i) ≤ (en/v)ᵛ', mono: true, fs: 11.5, fw: 700 },
      { t: 'At n = 100 points in ℝ²:' },
      { t: 'At most 9,902 labelings (O(n²))', mono: true, fs: 11.5, col: D.green.dark, fw: 700 },
      { t: 'VC gap: O(√(v / n)) → 0 as n → ∞!' },
    ],
  },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t: 'VC Bound: ε(f) ≤ ε_S(f) + c √((v + log(1/δ)) / n) holds ∀ f ∈ F' }
);

const vcVsHeldout = () => twoCols(560, 264, 'A Priori VC Bounds vs. Post-Training Held-Out Test Bounds',
  {
    head: 'A Priori VC Bound (Uniform ∀f∈F)', fill: C.lamber, stroke: C.amber, dark: D.orange.dark,
    lines: [
      { t: 'Linear model (v = 785, n = 60,000):' },
      { t: '√(v/n) ≈ ±11.4% (actual gap ~1%)', mono: true, fs: 11.5, col: D.orange.dark, fw: 700 },
      { t: 'Overparameterized deep net (v ≥ n):' },
      { t: '√(v/n) ≥ 100% (completely vacuous!)' },
    ],
  },
  {
    head: 'Held-Out Test Bound (1 Fixed f_S)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'Evaluates ONLY the final trained f_S' },
      { t: 'on n_test = 10,000 untouched images:' },
      { t: '95% CLT radius = ±0.70%!', mono: true, col: D.green.dark, fw: 700 },
      { t: 'Independent of parameter count!' },
    ],
  },
  { fill: C.lblue, stroke: C.blue, dark: D.blue.dark, t: 'VC theory explains why simple models cannot memorize; test sets certify deep nets' }
);

// ═══════════════════════════════════════════════════════════════════════════
// §3.6 environment-and-distribution-shift.md (9 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const labelSwapTrap = () => twoCols(560, 260, 'Why Unlabeled Domain Adaptation Needs Structural Invariance',
  {
    head: 'Source Domain p_S(x, y)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark,
    lines: [
      { t: 'Marginal inputs: p_S(x)', mono: true },
      { t: 'Cat photos → y = cat (100%)', col: D.blue.dark, fw: 700 },
      { t: 'Dog photos → y = dog (100%)', col: D.blue.dark, fw: 700 },
    ],
  },
  {
    head: 'Adversarial Target p_T(x, y)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: 'Same inputs: p_T(x) = p_S(x)', mono: true, fw: 700 },
      { t: 'Labels secretly swap overnight:' },
      { t: 'Cat photos → y = dog (0% acc!)', col: D.red.dark, fw: 700 },
    ],
  },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t: 'No unlabeled test can detect the swap unless p(y|x) OR p(x|y) stays invariant!' }
);

const causalFactorizations = () => stackCards(560, 286, 'Two Joint Factorizations Define Three Shift Regimes', [
  { head: '1. Covariate Shift: p(x) Drifts, p(y|x) Fixed', badge: 'Causal: x → y', sub: 'p(x,y) = p(x)p(y|x): photo vs cartoon style changes, but cat pixels mean cat', fill: C.lblue, stroke: C.blue, dark: D.blue.dark },
  { head: '2. Label Shift: Class Priors p(y) Drift, p(x|y) Fixed', badge: 'Causal: y → x', sub: 'p(x,y) = p(y)p(x|y): winter flu surge raises p(flu) while symptoms stay fixed', fill: C.lgreen, stroke: C.green, dark: D.green.dark },
  { head: '3. Concept Shift: Conditional Rule p(y|x) Drifts', badge: 'Needs fresh labels', sub: '"Soda" vs "pop" across geography or evolving diagnostic criteria over time', fill: C.lamber, stroke: C.amber, dark: D.orange.dark },
]);

const importanceReweight = () => threeStages(560, 258, 'Covariate Shift Identity: Reweighting Source Into Target Risk',
  { head: 'Target Risk R_T[f]', fill: C.lamber, stroke: C.amber, dark: D.orange.dark, mono: true, lines: ['∫ l(f(x),y) p(y|x)', '  × p_T(x) dx', '(no target labels!)'] },
  { head: 'Multiply by p_S/p_S', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['p_T(x) = p_S(x)', '  × [p_T(x)/p_S(x)]', 'β(x) = p_T / p_S'] },
  { head: 'Weighted Source ERM', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['(1/n) Σ β(x⁽ⁱ⁾)', '  × l(f(x⁽ⁱ⁾), y⁽ⁱ⁾)', 'unbiased for R_T!'] },
  { fill: D.red.tint, stroke: D.red.base, dark: D.red.dark, t1: 'Non-Negotiable Requirement: Support Overlap (p_S(x) > 0 Wherever p_T(x) > 0)', t2: 'If source has zero elderly controls (p_S = 0), no finite weight β can create them!' }
);

const domainClassifier = () => threeStages(560, 260, 'Estimating β(x) = p_T(x)/p_S(x) via a Logistic Domain Classifier',
  { head: '1. Pool Inputs Only', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['n source: z = 0', 'm target: z = 1', '(zero y labels!)'] },
  { head: '2. Fit Logistic h(x)', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['P(z=1|x) = σ(h(x))', 'odds P(1)/P(0) =', '(m/n) p_T/p_S = eʰ'] },
  { head: '3. Extract Weights', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['βᵢ = exp(h(x⁽ⁱ⁾))', '(or clip at c = 5)', 'train weighted ERM'] },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t1: 'Bypasses High-Dimensional Density Estimation of p_S(x) and p_T(x)!', t2: 'Directly estimates the 1D log-density ratio h(x) = log(p_T(x)/p_S(x)) + log(m/n)' }
);

const essClipping = () => twoCols(560, 262, 'Right-Tail Weight Explosion & Effective Sample Size n_eff',
  {
    head: 'Unclipped βᵢ = exp(2x₁ − 2)', fill: D.red.tint, stroke: D.red.base, dark: D.red.dark,
    lines: [
      { t: 'Source N(0,1) → Target N(2,1):' },
      { t: 'Max weight β_max ≈ 30–160!', mono: true, col: D.red.dark, fw: 700 },
      { t: 'n_eff = (Σ βᵢ)² / Σ βᵢ² ≈ 30–120', mono: true, fs: 11.5, col: D.red.dark, fw: 700 },
      { t: '88–97% of the sample wasted!' },
    ],
  },
  {
    head: 'Clipped Weights min(βᵢ, c = 5)', fill: C.lgreen, stroke: C.green, dark: D.green.dark,
    lines: [
      { t: 'Caps extreme right-tail ratios at 5:' },
      { t: 'Max weight β̃_{max} = 5.0', mono: true, col: D.green.dark, fw: 700 },
      { t: 'n_eff jumps to ~200 of 1,000!', mono: true, col: D.green.dark, fw: 700 },
      { t: 'Target accuracy reaches ~93–95%!' },
    ],
  },
  { fill: C.lblue, stroke: C.blue, dark: D.blue.dark, t: 'Theoretical large-n ESS for shift distance s is n·exp(−s²) (1000·e⁻⁴ ≈ 18)' }
);

const weightedGrad = () => threeStages(560, 256, 'Weighted Logistic Regression Step Inside fit_logreg',
  { head: 'Unweighted Residual', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['p̂ᵢ = σ(wᵀx⁽ⁱ⁾ + b)', 'raw error:', 'eᵢ = p̂ᵢ − y⁽ⁱ⁾'] },
  { head: 'Multiply by vᵢ ∝ βᵢ', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['v = β / mean(β)', 'weighted error:', 'rᵢ = vᵢ(p̂ᵢ − y⁽ⁱ⁾)'] },
  { head: 'Tilted Boundary', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['w ← w − η Xᵀr / n', 'Target acc jumps', 'from ~55% → ~95%!'] },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t1: 'Why Reweighting Works on a Misspecified Linear Model', t2: 'Focuses the linear boundary on the right arm of the parabola where p_T lives' }
);

const bbsePipeline = () => threeStages(560, 260, 'Black-Box Label Shift Estimation: C p_T = μ',
  { head: '1. Source Matrix C', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['On source val split:', 'C_ij = P(ĉ=i|y=j)', 'columns sum to 1'] },
  { head: '2. Target Freqs μ', fill: C.lamber, stroke: C.amber, dark: D.orange.dark, mono: true, lines: ['On unlabeled target:', 'μ_i = P_T(ĉ(x)=i)', '(needs NO labels!)'] },
  { head: '3. Solve p̂_T = C⁻¹μ', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['np.linalg.solve(C,μ)', 'recovers true target', 'class priors p_T!'] },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t1: 'Why C Is Identical on Source and Target Under Label Shift', t2: 'Because ĉ(x) depends only on x and p_T(x | y) = p_S(x | y) is invariant!' }
);

const clinicalBbse = () => threeStages(560, 258, 'Worked Clinical Label-Shift Inversion (90% Sensitivity, 5% FPR)',
  { head: 'Confusion Matrix C', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['[0.95  0.10]', '[0.05  0.90]', 'det(C) = 0.85 ≠ 0'] },
  { head: 'Target Flag Rate μ', fill: C.lamber, stroke: C.amber, dark: D.orange.dark, mono: true, lines: ['13.5% flagged sick:', 'μ = (0.865, 0.135)ᵀ', '0.05(1−p)+0.90p=.135'] },
  { head: 'Solved Prior p_T', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['0.85 p = 0.085', '⇒ p_T(sick) = 10.0%!', '(4.5% FP, 1% FN)'] },
  { fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, t1: 'Invertibility Guarantee: Strict Diagonal Dominance (Recall C_{jj} > 1/2)', t2: 'In 2-class screening, estimation SE scales as 1 / |C₁₁ − C₁₀| = 1 / (TPR − FPR)' }
);

const logitPriorShift = () => threeStages(560, 260, 'Instant Zero-Retraining Bayes Logit Adaptation Under Label Shift',
  { head: 'Source Logits o(x)', fill: C.lblue, stroke: C.blue, dark: D.blue.dark, mono: true, lines: ['Trained on balanced', 'p_S(y) = 0.10', 'ŷ_S = softmax(o)'] },
  { head: 'Add Log-Prior Ratio', fill: D.purple.tint, stroke: C.purple, dark: D.purple.dark, mono: true, lines: ['o\'_j = o_j +', '  log(p̂_T(j)/p_S(j))', 'shirt: +log(0.5/0.1)'] },
  { head: 'Adapted Target Pred', fill: C.lgreen, stroke: C.green, dark: D.green.dark, mono: true, lines: ['ŷ_T = softmax(o\')', '+6–13 pts target acc', 'in 0 grad steps!'] },
  { fill: C.lamber, stroke: C.amber, dark: D.orange.dark, t1: 'Follows From Bayes\' Theorem: p_T(y|x) ∝ p_S(y|x) · [p_T(y) / p_S(y)]', t2: 'Taking logs turns the multiplicative prior ratio into an additive logit offset!' }
);

export const diagrams = {
  // §3.0 index.md
  'ch03-cls-why-classification':    whyClassification,
  'ch03-cls-reg-vs-cls':            regVsCls,
  'ch03-cls-roadmap':               clsRoadmap,
  'ch03-cls-benchmarks':            clsBenchmarks,
  // §3.1 softmax-regression.md
  'ch03-cls-ordinal-trap':          ordinalTrap,
  'ch03-cls-onehot-simplex':        onehotSimplex,
  'ch03-cls-softmax-pipe':          softmaxPipe,
  'ch03-cls-shift-invariance':      shiftInvariance,
  'ch03-cls-two-class-sigmoid':     twoClassSigmoid,
  'ch03-cls-batched-softmax':       batchedSoftmax,
  'ch03-cls-ce-surprisal':          ceSurprisal,
  'ch03-cls-mse-vs-ce-grad':        mseVsCeGrad,
  'ch03-cls-residual-grad':         residualGrad,
  'ch03-cls-template-nudge':        templateNudge,
  'ch03-cls-entropy-kl':            entropyKl,
  // §3.2 image-classification-dataset.md
  'ch03-cls-benchmark-sweetspot':   benchmarkSweetspot,
  'ch03-cls-garment-taxonomy':      garmentTaxonomy,
  'ch03-cls-pixel-rescale':         pixelRescale,
  'ch03-cls-dataset-splits':        datasetSplits,
  'ch03-cls-idx-header':            idxHeader,
  'ch03-cls-epoch-batches':         epochBatches,
  'ch03-cls-nchw-vs-nhwc':          nchwVsNhwc,
  'ch03-cls-minibatch-flatten':     minibatchFlatten,
  'ch03-cls-producer-consumer':     producerConsumer,
  'ch03-cls-visual-debug':          visualDebug,
  // §3.3 softmax-regression-scratch.md
  'ch03-cls-fp32-hazards':          fp32Hazards,
  'ch03-cls-rowwise-broadcast':     rowwiseBroadcast,
  'ch03-cls-max-shift-range':       maxShiftRange,
  'ch03-cls-classifier-arch':       classifierArch,
  'ch03-cls-scratch-params':        scratchParams,
  'ch03-cls-gather-vs-onehot':      gatherVsOnehot,
  'ch03-cls-confusion-accum':       confusionAccum,
  'ch03-cls-raw-logits-contract':   rawLogitsContract,
  'ch03-cls-logsumexp-fusion':      logsumexpFusion,
  // §3.4 classification.md
  'ch03-cls-accuracy-limits':       accuracyLimits,
  'ch03-cls-imbalance-matrix':      imbalanceMatrix,
  'ch03-cls-recall-vs-precision':   recallVsPrecision,
  'ch03-cls-prevalence-bayes':      prevalenceBayes,
  'ch03-cls-asymmetric-cost':       asymmetricCost,
  'ch03-cls-cost-threshold':        costThreshold,
  'ch03-cls-threshold-sweep':       thresholdSweep,
  'ch03-cls-calibration-meaning':   calibrationMeaning,
  'ch03-cls-ece-bins':              eceBins,
  'ch03-cls-temp-recalibration':    tempRecalibration,
  // §3.5 generalization-classification.md
  'ch03-cls-gen-obstacles':         genObstacles,
  'ch03-cls-bernoulli-se':          bernoulliSe,
  'ch03-cls-clt-10k':               clt10k,
  'ch03-cls-hoeffding-vs-clt':      hoeffdingVsClt,
  'ch03-cls-union-bound':           unionBound,
  'ch03-cls-adaptive-overfit':      adaptiveOverfit,
  'ch03-cls-uniform-conv':          uniformConv,
  'ch03-cls-dichotomies':           dichotomies,
  'ch03-cls-radon-vc':              radonVc,
  'ch03-cls-sauer-growth':          sauerGrowth,
  'ch03-cls-vc-vs-heldout':         vcVsHeldout,
  // §3.6 environment-and-distribution-shift.md
  'ch03-cls-label-swap-trap':       labelSwapTrap,
  'ch03-cls-causal-factorizations': causalFactorizations,
  'ch03-cls-importance-reweight':   importanceReweight,
  'ch03-cls-domain-classifier':     domainClassifier,
  'ch03-cls-ess-clipping':          essClipping,
  'ch03-cls-weighted-grad':         weightedGrad,
  'ch03-cls-bbse-pipeline':         bbsePipeline,
  'ch03-cls-clinical-bbse':         clinicalBbse,
  'ch03-cls-logit-prior-shift':     logitPriorShift,
};
