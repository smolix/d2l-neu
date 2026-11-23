// Slide diagrams for ch34-mdl-calc.mjs (§34 Calculus & Automatic Differentiation)
import { C, TOKENS, tx as rawTx, arrow, svg as rawSvg } from "./engine.mjs";
import { sec, num } from "./sections.mjs";
import { textWidth } from "./mdl-textwidth.mjs";

const D = TOKENS.accents;
const N = TOKENS.neutrals;

let _boxes = [];
// Text that does not fit its box is shrunk, but never below MIN_FS; a string
// still too wide at MIN_FS is wrapped onto two lines at the space nearest its
// middle. Widths come from the real font metrics (mdl-textwidth.mjs).
const MIN_FS = 11;
export const _fitLog = [];
function split2(s) {
  // Prefer a gap of two or more spaces (between clauses) near the middle,
  // else the space nearest the middle.
  const mid = s.length / 2;
  let best = -1, bestScore = Infinity;
  for (let i = 1; i < s.length - 1; i++) {
    if (s[i] !== ' ') continue;
    const wide = s[i - 1] === ' ' || s[i + 1] === ' ';
    const score = Math.abs(i - mid) - (wide ? 0.3 * s.length : 0);
    if (score < bestScore) { best = i; bestScore = score; }
  }
  return best < 0 ? null : [s.slice(0, best).trimEnd(), s.slice(best + 1).trimStart()];
}
const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) => {
  if (w >= 40 && h >= 28) _boxes.push({ x, y, w, h });
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
};
// { lines: [one or two strings], fs } for text s in width maxW.
function fitText(s, maxW, opts) {
  const baseFs = opts.fs || 14;
  const mono = !!opts.mono, bold = (opts.fw || 500) >= 600;
  const w = textWidth(s, baseFs, mono, bold);
  if (w <= maxW) return { lines: [s], fs: baseFs };
  const fit = +(baseFs * maxW / w).toFixed(2);
  if (fit >= MIN_FS) return { lines: [s], fs: fit };
  const parts = split2(s);
  if (parts) {
    const w2 = Math.max(...parts.map(p => textWidth(p, baseFs, mono, bold)));
    const fs2 = Math.max(MIN_FS, Math.min(baseFs, +(baseFs * maxW / w2).toFixed(2)));
    _fitLog.push(['wrap', s, fs2, w2 * fs2 / baseFs > maxW]);
    return { lines: parts, fs: fs2 };
  }
  _fitLog.push(['over', s, MIN_FS]);
  return { lines: [s], fs: MIN_FS };
}
const tx = (x, y, text, opts = {}) => {
  const s = String(text ?? '');
  if (!s) return '';
  let maxW = 2 * Math.min(x, 540 - x) - 14;
  let yy = y;
  const inside = _boxes.filter(b => x >= b.x && x <= b.x + b.w && y >= b.y && y <= b.y + b.h);
  if (inside.length > 0) {
    const b = inside.reduce((a, c) => (c.w * c.h < a.w * a.h ? c : a));
    maxW = 2 * Math.min(x - b.x, b.x + b.w - x) - 16;
    yy = Math.max(b.y + 12, Math.min(b.y + b.h - 12, y));
  }
  const { lines, fs } = fitText(s, maxW, opts);
  if (lines.length === 1) return rawTx(x, yy, lines[0], { ...opts, fs });
  const dy = 0.6 * fs;
  return rawTx(x, yy - dy, lines[0], { ...opts, fs }) + rawTx(x, yy + dy, lines[1], { ...opts, fs });
};
const svg = (W, H, body) => {
  _boxes = [];
  return rawSvg(W, H, body);
};
const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
const hline = (x1, x2, y) => line(x1, y, x2, y, N.hairline, 1.5, true);
// A titled card: title, a mono line and a muted line at fixed offsets. If
// any of them wraps, the lines flow down from the title instead.
const card = (x, y, w, h, fill, st, title, sub1, sub2, dk) => {
  const compact = h <= 76;
  const ys = compact ? [y + 16, y + 36, y + 54] : [y + 18, y + 39, y + 59];
  const items = [
    [title, { fs: compact ? 14 : 14.5, fw: 700, fill: dk }],
    [sub1, { mono: true, fs: 12.5, fw: 600, fill: C.ink }],
    [sub2, { fs: 12.5, fill: C.muted }],
  ];
  let o = box(x, y, w, h, fill, st);
  const fitted = items.map(([t, op]) => (t ? fitText(String(t), w - 16, op) : null));
  if (!fitted.some(f => f && f.lines.length > 1)) {
    items.forEach(([t, op], i) => { if (t) o += tx(x + w / 2, ys[i], t, op); });
    return o;
  }
  // Line centres: 1.15 fs inside a wrapped item, `gap` between items; the
  // gaps shrink (to no less than 1.25 fs) when the stack would leave the card.
  const live = fitted.map((f, i) => [f, i]).filter(([f]) => f);
  const inner = live.reduce((a, [f]) => a + (f.lines.length - 1) * 1.15 * f.fs, 0);
  let gap = compact ? 19 : 20.5;
  const room = y + h - 11 - ys[0] - inner;
  if (live.length > 1 && gap * (live.length - 1) > room) {
    gap = Math.max(1.25 * Math.max(...live.map(([f]) => f.fs)), room / (live.length - 1));
    _fitLog.push(['squeeze', String(title), +gap.toFixed(1), h]);
  }
  let cy = ys[0];
  live.forEach(([f, i]) => {
    f.lines.forEach((ln, j) => {
      if (j > 0) cy += 1.15 * f.fs;
      o += rawTx(x + w / 2, +cy.toFixed(2), ln, { ...items[i][1], fs: f.fs });
    });
    cy += gap;
  });
  return o;
};

// ── index.md (4 diagrams) ──
function twoQuestions() {
  let o = card(18, 16, 504, 52, D.purple.tint, C.purple, 'Every Training Step: w ← w − η ∇ℓ(w)', `Raises two fundamental mathematical questions answered by Chapter ${num('chap_mdl-calculus')}`, '', D.purple.dark);
  o += card(18, 84, 242, 162, C.lamber, C.amber, '1. Safe Step Size η?', 'Gradient is a local linear model', 'Curvature |f″| ≤ L bends loss away', D.orange.dark);
  o += tx(139, 182, `Descent Lemma (§§${num('sec_mdl-single_variable_calculus')}–${num('sec_mdl-multivariable_calculus')}):`, { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(139, 206, '0 < η < 2/L  (optimal η = 1/L)', { mono: true, fs: 13, fw: 700, fill: C.ink });
  o += tx(139, 228, 'H_f classifies min vs saddle', { fs: 13, fill: C.muted });
  o += card(280, 84, 242, 162, C.lgreen, C.green, '2. Why Is ∇ℓ So Cheap?', '10⁸ weights: 12 days via finite diff', 'vs 30 ms in 1 reverse VJP sweep!', D.green.dark);
  o += tx(401, 182, `Reverse Autodiff (§§${num('sec_mdl-chain-rule-graphs')}–${num('sec_mdl-matrix-calculus-autodiff')}):`, { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(401, 206, '((uᵀJ_K)J_{K−1})⋯J₁  (~3× fwd)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(401, 228, `+ §§${num('sec_mdl-integral_calculus')}–${num('sec_mdl-expectation-gradients')} expectation grads`, { fs: 13, fill: C.muted });
  return svg(540, 260, o);
}

function threePillars() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, `Pillar 1: Taylor Remainder & Descent Lemma (§§${num('sec_mdl-single_variable_calculus')}–${num('sec_mdl-multivariable_calculus')})`, 'f(x − ηf′(x)) ≤ f(x) − η(1 − Lη/2) f′(x)²   from   |f″(ξ)| ≤ L', 'Certifies safe step sizes 0 < η < 2/L and classifies critical points via Hessian H_f', D.blue.dark);
  o += card(20, 98, 500, 72, C.lamber, C.amber, `Pillar 2: Reverse-Mode Jacobian Products (§§${num('sec_mdl-chain-rule-graphs')}–${num('sec_mdl-matrix-calculus-autodiff')})`, 'uᵀ J = ((uᵀ J_K) J_{K−1} ⋯) J₁   and   df = ⟨∇f, dX⟩ = tr((∇f)ᵀ dX)', 'Propagates 1 row adjoint backward across DAGs in ~3 forward-pass times', D.orange.dark);
  o += card(20, 182, 500, 72, C.lgreen, C.green, `Pillar 3: Differentiating Expectations (§§${num('sec_mdl-integral_calculus')}–${num('sec_mdl-expectation-gradients')})`, '∇_θ E[g(X)] = E[(g(X)−b) ∇_θ log p_θ(X)]  vs  E_z[∇_θ g(T_θ(z))]', 'Trades generality of REINFORCE against low variance of pathwise reparameterization', D.green.dark);
  return svg(540, 268, o);
}

function roadmapThreeParts() {
  let o = card(16, 16, 160, 224, C.lblue, C.blue, 'Part I: Local Models', `${sec('sec_mdl-single_variable_calculus')} 1D Calculus`, 'Pₙ(x) + Rₙ(x), η < 2/L', D.blue.dark);
  o += tx(96, 118, 'Newton & subgradients', { fs: 13, fill: C.ink });
  o += hline(32, 160, 138);
  o += tx(96, 160, `${sec('sec_mdl-multivariable_calculus')} Multivariable`, { fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(96, 184, '∇f ⊥ contours', { mono: true, fs: 13, fill: C.ink });
  o += tx(96, 208, 'Symmetric Hessian H_f', { fs: 13, fill: C.muted });
  o += arrow(179, 128, 189, 128, C.blue);
  o += card(192, 16, 160, 224, C.lamber, C.amber, 'Part II: Chain & AD', `${sec('sec_mdl-chain-rule-graphs')} Graph DAGs`, 'Path sum & adjoints n̄', D.orange.dark);
  o += tx(272, 112, `${sec('sec_mdl-matrix-calculus')} Matrix Calc`, { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(272, 134, 'df = ⟨∇f, dX⟩, p − y', { mono: true, fs: 12.5, fill: C.ink });
  o += hline(208, 336, 152);
  o += tx(272, 174, `${sec('sec_mdl-matrix-calculus-autodiff')} Autodiff`, { fs: 14, fw: 700, fill: D.orange.dark });
  o += tx(272, 198, 'Duals, Tape, 2√T mem', { fs: 13, fill: C.ink });
  o += tx(272, 220, 'HVPs & Implicit Diff', { fs: 13, fill: C.muted });
  o += arrow(355, 128, 365, 128, C.amber);
  o += card(368, 16, 156, 224, C.lgreen, C.green, 'Part III: Integrals', `${sec('sec_mdl-integral_calculus')} Integration`, 'Riemann, FTC, Fubini', D.green.dark);
  o += tx(446, 118, '|det J_φ| & √(2π)', { mono: true, fs: 13, fill: C.ink });
  o += hline(384, 508, 138);
  o += tx(446, 160, `${sec('sec_mdl-expectation-gradients')} Expectations`, { fs: 14, fw: 700, fill: D.green.dark });
  o += tx(446, 184, 'Leibniz boundary rule', { fs: 13, fill: C.ink });
  o += tx(446, 208, 'Score vs Pathwise', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 256, o);
}

function runningBenchmarks() {
  let o = card(16, 14, 246, 70, C.lblue, C.blue, `${sec('sec_mdl-single_variable_calculus')} Logistic Loss ℓ(w)=log(1+e⁻ʷ)`, 'Gap 0.00125 = ½ℓ″(0)(0.1)²', 'L = ¼ ⇒ η < 8 safe, η = 4 optimal', D.blue.dark);
  o += card(278, 14, 246, 70, C.lblue, C.blue, `${sec('sec_mdl-multivariable_calculus')} LSE f(x,y)=log(eˣ+eʸ)`, '∇f(0,log 2) = (⅓,⅔)ᵀ, softmax', 'H_f = diag(p)−ppᵀ has λ = 0, 4/9', D.blue.dark);
  o += card(16, 96, 246, 70, C.lamber, C.amber, `${sec('sec_mdl-chain-rule-graphs')} 4-Input Squaring DAG`, 'Integer adjoints: w̄ = 73,472', '12 mults in reverse vs 24 forward', D.orange.dark);
  o += card(278, 96, 246, 70, C.lamber, C.amber, `${sec('sec_mdl-matrix-calculus')} 2-2-1 ReLU Network`, 'J_g = w₂ᵀ diag(m) W₁ = (2, 4)', 'Left-to-right VJP: 667× fewer mults', D.orange.dark);
  o += card(16, 178, 246, 70, D.purple.tint, C.purple, `${sec('sec_mdl-matrix-calculus-autodiff')} Dual & Var Engines`, '(∑ xⱼ²)², n = 200: 802 vs 80,200', 'Checkpointing: 401 → 40 states (2√T)', D.purple.dark);
  o += card(278, 178, 246, 70, C.lgreen, C.green, `§§${num('sec_mdl-integral_calculus')}–${num('sec_mdl-expectation-gradients')} Normal & E[X²]`, '∫ e^{−z²/2} dz = √(2π) (polar)', 'Pathwise σ=2.0 vs REINFORCE σ=7.18', D.green.dark);
  return svg(540, 262, o);
}

// ── §34.1 mdl-single-variable-calculus.md (6 diagrams) ──
function logisticCurvatureGap() {
  let o = card(20, 14, 500, 70, C.lblue, C.blue, '1D Benchmark: Logistic Loss ℓ(w) = log(1 + e⁻ʷ) = s(−w) at w = 0', 'ℓ(0) = log 2 ≈ 0.693147,  ℓ′(0) = −½,  ℓ″(0) = ¼', 'Step to w = 0.1 (ϵ = 0.1) and compare tangent line against true curved loss:', D.blue.dark);
  o += card(20, 100, 236, 140, C.lamber, C.amber, 'Tangent Line Prediction', 'ℓ(0) + 0.1 ℓ′(0)', '0.693147 − 0.05 = 0.643147', D.orange.dark);
  o += tx(138, 212, 'Ignores upward bend ℓ″ > 0', { fs: 13, fill: C.muted });
  o += arrow(260, 170, 280, 170, C.green);
  o += card(284, 100, 236, 140, C.lgreen, C.green, 'True Loss & Curvature Gap', 'ℓ(0.1) ≈ 0.644397', 'Gap = +0.001250 ≈ ½ℓ″(0)(0.1)²', D.green.dark);
  o += tx(402, 212, '½(¼)(0.01) = 0.00125 exact!', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 256, o);
}

function littleOLinear() {
  let o = card(20, 14, 500, 74, D.purple.tint, C.purple, 'Local Perturbation Form of the Derivative', 'f(x + ϵ) = f(x) + ϵ f′(x) + r(ϵ),  r(ϵ)/ϵ → 0 as ϵ → 0', 'Proof: (f(x+ϵ) − f(x))/ϵ − f′(x) = r(ϵ)/ϵ → 0  ⇔  difference quotient → f′(x)', D.purple.dark);
  o += card(20, 104, 154, 136, C.lblue, C.blue, '1. Base Value', 'f(x) = O(1)', 'Height at x', D.blue.dark);
  o += tx(97, 210, 'ϵ → 0: f(x+ϵ) → f(x)', { mono: true, fs: 12, fw: 700, fill: D.blue.dark });
  o += card(193, 104, 154, 136, C.lgreen, C.green, '2. Linear Term', 'ϵ f′(x) = O(ϵ)', 'Tangent line shift', D.green.dark);
  o += tx(270, 210, 'Halving ϵ halves shift', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += card(366, 104, 154, 136, C.lamber, C.amber, '3. Remainder r(ϵ)', 'o(ϵ)  (typically O(ϵ²))', 'Curvature gap', D.orange.dark);
  o += tx(443, 210, 'Halving ϵ quarters gap', { fs: 12.5, fw: 700, fill: D.orange.dark });
  return svg(540, 256, o);
}

function elementaryRules() {
  let o = card(16, 14, 246, 70, C.lblue, C.blue, '1. Constants & Linear', 'd/dx (c) = 0,   d/dx (ax) = a', 'Zero curvature everywhere on ℝ', D.blue.dark);
  o += card(278, 14, 246, 70, C.lblue, C.blue, '2. Power Rule (All Real n)', 'd/dx (xⁿ) = n xⁿ⁻¹', 'Holds on x > 0 via xⁿ = e^{n log x}', D.blue.dark);
  o += card(16, 96, 246, 70, C.lgreen, C.green, '3. Exponential & Base e', '(eˣ)′ = eˣ,  (aˣ)′ = (log a) aˣ', 'Unique base e where slope at 0 is 1', D.green.dark);
  o += card(278, 96, 246, 70, C.lamber, C.amber, '4. Natural Logarithm', 'd/dx (log x) = 1 / x  (x > 0)', 'Inverse of eˣ: 1 / e^{log x} = 1/x', D.orange.dark);
  o += card(16, 178, 508, 68, D.purple.tint, C.purple, '5. Trigonometric Cycle: d/dx (sin x) = cos x,   d/dx (cos x) = −sin x', 'Why aˣ has slope proportional to height:   (a^{x+ϵ} − aˣ)/ϵ = aˣ · ((a^{ϵ} − 1)/ϵ) → (log a) aˣ', '', D.purple.dark);
  return svg(540, 260, o);
}

function sumProductChain() {
  let o = tx(270, 20, 'Proving Combination Rules via Local Linear Expansions g(x+ϵ) = g(x) + ϵg′(x) + o(ϵ)', { fs: 14.5, fw: 700, fill: C.ink });
  o += card(16, 38, 160, 198, C.lblue, C.blue, '1. Sum Rule', '(g + h)′ = g′ + h′', 'Add linear models:', D.blue.dark);
  o += tx(96, 148, 'ϵ(g′ + h′) + o(ϵ)', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(96, 180, 'Two o(ϵ) remainders', { fs: 13, fill: C.ink });
  o += tx(96, 204, 'add to o(ϵ)', { fs: 13, fill: C.muted });
  o += card(190, 38, 160, 198, C.lamber, C.amber, '2. Product Rule', '(gh)′ = g h′ + g′ h', 'Multiply expansions:', D.orange.dark);
  o += tx(270, 148, 'ϵ(gh′ + g′h) + O(ϵ²)', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(270, 180, 'Corner area ϵ²g′h′', { fs: 13, fill: C.ink });
  o += tx(270, 204, 'is o(ϵ) and vanishes', { fs: 13, fill: C.muted });
  o += card(364, 38, 160, 198, C.lgreen, C.green, '3. Chain Rule', '(g∘h)′ = g′(h) h′', 'Step t = ϵh′(x) + o(ϵ):', D.green.dark);
  o += tx(444, 148, 'g(y+t)−g(y) ≈ g′(y)t', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(444, 180, '= ϵ g′(h)h′ + o(ϵ)', { mono: true, fs: 12, fill: C.ink });
  o += tx(444, 204, 'Multiplies local slopes!', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 252, o);
}

function softplusSigmoidLogit() {
  let o = card(14, 16, 144, 114, C.lblue, C.blue, 'Softplus s(x)', 'log(1 + eˣ)', 'Smooth ReLU (s > 0)', D.blue.dark);
  o += arrow(162, 73, 194, 73, C.blue);
  o += tx(178, 55, 'd/dx', { mono: true, fs: 11, fw: 700, fill: D.blue.dark });
  o += card(198, 16, 144, 114, C.lgreen, C.green, 'Sigmoid σ(x)', 's′(x) = 1/(1 + e⁻ˣ)', 'σ(0) = ½, range (0,1)', D.green.dark);
  o += arrow(346, 73, 378, 73, C.green);
  o += tx(362, 55, 'd/dx', { mono: true, fs: 11, fw: 700, fill: D.green.dark });
  o += card(382, 16, 144, 114, C.lamber, C.amber, 'Curvature σ′(x)', 'σ(x)(1 − σ(x))', 'Peaks at x=0: L = ¼', D.orange.dark);
  o += card(14, 146, 512, 98, D.purple.tint, C.purple, 'Inverse-Function Rule: Logit(p) = σ⁻¹(p) = log(p / (1 − p))  for p ∈ (0, 1)', 'g(f(x)) = x  ⇒  g′(y) = 1 / f′(g(y))  ⇒  logit′(p) = 1 / (σ′(x)) = 1 / (p(1 − p))', 'At p = ½ (x = 0): logit′(½) = 1 / (¼) = 4  (reciprocal of sigmoid slope ¼)', D.purple.dark);
  return svg(540, 258, o);
}

function lagrangeRemainder() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, 'Taylor Polynomial Pₙ(x) + Exact Lagrange Remainder Rₙ(x)', 'f(x) = Pₙ(x) + f^{(n+1)}(ξ)(x − x₀)^{n+1}/(n+1)!', 'Holds for some interior point ξ strictly between x₀ and x', D.blue.dark);
  o += card(20, 104, 238, 140, D.purple.tint, C.purple, "Proof via Rolle's Theorem", 'G(x₀) = G(x) = 0  ⇒  G′(ξ) = 0', 'Product rule telescopes sum!', D.purple.dark);
  o += tx(139, 214, 'Leaves M = f^{(n+1)}(ξ)', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += card(282, 104, 238, 140, C.lgreen, C.green, 'Key Optimization Cases', 'n = 0: Mean Value Theorem', 'n = 1: f(x₀)+f′(x₀)u + ½f″(ξ)u²', D.green.dark);
  o += tx(401, 214, 'Engine of Descent Lemma!', { fs: 13.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

// ── §34.2 mdl-multivariable-calculus.md (11 diagrams) ──
function jointPerturbation() {
  let o = card(20, 14, 500, 66, D.purple.tint, C.purple, 'From 1D Coordinate Slices to Joint N-Dimensional Moves w + ϵ', 'Every gradient descent step updates all N = 10⁶ weights simultaneously!', '', D.purple.dark);
  o += card(20, 96, 238, 148, C.lblue, C.blue, 'First-Order Vector: Gradient ∇ℓ', '∇_w ℓ = [∂ℓ/∂w₁, …, ∂ℓ/∂w_N]ᵀ', 'Δℓ ≈ ϵ · ∇_w ℓ(w)', D.blue.dark);
  o += tx(139, 214, 'Points uphill (steepest ascent)', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += card(282, 96, 238, 148, C.lamber, C.amber, 'Second-Order Matrix: Hessian H_ℓ', '[H_ℓ]_{ij} = ∂²ℓ / (∂wᵢ ∂wⱼ)', 'Curvature: ½ ϵᵀ H_ℓ ϵ', D.orange.dark);
  o += tx(401, 214, 'Eigenvalues classify min vs saddle', { fs: 13, fw: 700, fill: D.orange.dark });
  return svg(540, 258, o);
}

function lsePartials() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, 'Running Benchmark: Log-Sum-Exp f(x, y) = log(eˣ + eʸ)', '∂f/∂x = eˣ/(eˣ + eʸ) = p₁,   ∂f/∂y = eʸ/(eˣ + eʸ) = p₂', 'Partial derivatives are positive and sum to 1: ∇f(x, y) = softmax(x, y)!', D.blue.dark);
  o += card(20, 102, 238, 140, C.lgreen, C.green, 'Evaluated at (0, log 2)', 'eˣ = 1,  eʸ = 2  ⇒  f = log 3', '∂f/∂x = 1/3,   ∂f/∂y = 2/3', D.green.dark);
  o += tx(139, 212, 'y raises f twice as fast as x', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += card(282, 102, 238, 140, C.lamber, C.amber, 'Joint Step ϵ = (0.01, −0.03)', 'Δf ≈ (⅓)(0.01) + (⅔)(−0.03)', 'Predicted shift: −0.016667', D.orange.dark);
  o += tx(401, 212, 'f ≈ log 3 − 0.016667 = 1.08195', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  return svg(540, 256, o);
}

function multivarGradient() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, 'Multivariable Differentiability: Genuine Linear Map Λ(ϵ)', 'ℓ(w + ϵ) = ℓ(w) + Λ(ϵ) + o(∥ϵ∥),  o(∥ϵ∥)/∥ϵ∥ → 0', 'Probing along axis eᵢ (ϵ = h eᵢ) forces Λ(eᵢ) = ∂ℓ/∂wᵢ', D.blue.dark);
  o += card(20, 104, 500, 138, C.lgreen, C.green, 'Uniqueness of the Gradient Dot Product', 'Λ(∑ᵢ ϵᵢ eᵢ) = ∑ᵢ ϵᵢ (∂ℓ/∂wᵢ) = ϵ · ∇_w ℓ(w)', 'Coordinate changes add up cleanly ONLY when ℓ is differentiable at w!', D.green.dark);
  o += tx(270, 216, 'Column gradient ∇_w ℓ ∈ ℝ^{N} matches the exact shape of parameter vector w ∈ ℝ^{N}', { fs: 13.5, fw: 700, fill: D.green.dark });
  return svg(540, 256, o);
}

function ridgeCounterexample() {
  let o = card(20, 14, 238, 136, D.red.tint, D.red.base, 'Pathological Ridge at (0, 0)', 'f = 2xy/(x² + y²), f(0,0) = 0', 'On axes: f(x,0) = f(0,y) = 0', D.red.dark);
  o += tx(139, 102, '∂f/∂x(0,0) = ∂f/∂y(0,0) = 0', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(139, 128, 'Yet on diagonal y=x: f(x,x) = 1!', { fs: 13, fw: 700, fill: D.red.dark });
  o += card(282, 14, 238, 136, C.lgreen, C.green, 'C¹ Sufficient Condition', 'All ∂ℓ/∂wᵢ exist near w', 'AND are continuous at w', D.green.dark);
  o += tx(401, 102, 'Guarantees differentiability!', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(401, 128, 'Smooth loss → O(∥ϵ∥²) gap', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += card(20, 166, 500, 78, C.lamber, C.amber, 'Lesson: Partial Derivatives Along Axes Do Not Guarantee a Tangent Plane', 'A tangent plane must fit the surface along EVERY ray approaching w, not just the axes', '', D.orange.dark);
  return svg(540, 258, o);
}

function directionalRays() {
  let o = card(20, 14, 500, 64, C.lblue, C.blue, 'Directional Derivative Along Unit Vector u (∥u∥₂ = 1)', 'D_u ℓ(w) = lim_{h→0} (ℓ(w + hu) − ℓ(w))/h = u · ∇_w ℓ(w) = ∥∇ℓ∥₂ cos θ', '', D.blue.dark);
  o += card(16, 94, 160, 150, C.lamber, C.amber, '1. Shift Ray (1, 1)/√2', 'u · ∇f = (⅓+⅔)/√2', '= +1/√2 ≈ +0.707', D.orange.dark);
  o += tx(96, 200, 'f(x+t, y+t) = f + t', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(96, 222, 'Exact linear growth', { fs: 12.5, fill: C.muted });
  o += card(190, 94, 160, 150, C.lgreen, C.green, '2. Steepest Descent', 'u = −(1, 2)/√5', 'u · ∇f = −√5/3 ≈ −0.745', D.green.dark);
  o += tx(270, 200, 'Opposite to ∇f', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(270, 222, 'Minimum possible slope', { fs: 12.5, fill: C.muted });
  o += card(364, 94, 160, 150, D.purple.tint, C.purple, '3. Level-Set Tangent', 'u = (2, −1)/√5', 'u · ∇f = (⅔ − ⅔)/√5 = 0', D.purple.dark);
  o += tx(444, 200, 'u ⊥ ∇f (90°)', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += tx(444, 222, 'Tangent to eˣ + eʸ = 3', { fs: 12.5, fill: C.muted });
  return svg(540, 258, o);
}

function levelSetOrthogonality() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, 'Why Gradients Stand Orthogonal to Level Sets {x : ℓ(x) = c}', 'Any curve γ(t) in {ℓ = c} with γ(0) = x, velocity v = γ′(0):', '0 = d/dt ℓ(γ(t))|_{t=0} = v · ∇ℓ(x)   ⇒   ∇ℓ(x) ⊥ tangent vector v!', D.blue.dark);
  o += card(20, 104, 238, 140, C.lgreen, C.green, 'LSE Contour at (0, log 2)', 'Level curve: eˣ + eʸ = 3', 'Tangent v = (eʸ, −eˣ) = (2, −1)', D.green.dark);
  o += tx(139, 214, 'v · ∇f = 2(⅓) − 1(⅔) = 0!', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += card(282, 104, 238, 140, C.lamber, C.amber, 'Contour Packing Density', '∥∇ℓ∥₂ = rate of change ⊥ contour', 'Tightly packed contours ⇒ large ∥∇ℓ∥', D.orange.dark);
  o += tx(401, 214, 'Widely spaced ⇒ flat plateau', { fs: 13, fw: 700, fill: D.orange.dark });
  return svg(540, 258, o);
}

function criticalDiscBound() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, 'Critical Points of ψ(x, y) = x e^{−x² − y²}: Setting ∇ψ(x, y) = 0', '∇ψ = [1 − 2x²,  −2xy]ᵀ e^{−x²−y²} = 0   ⇒   y = 0  and  x = ±1/√2', 'Two critical points: (−1/√2, 0) with ψ ≈ −0.4289  and  (+1/√2, 0) with ψ ≈ +0.4289', D.blue.dark);
  o += card(20, 102, 238, 142, D.red.tint, D.red.base, 'Why ∇f = 0 Alone Is Not Enough', 'f(x) = x³ has unique f′(0) = 0', 'Yet no global min or max exists!', D.red.dark);
  o += tx(139, 214, 'Must check boundary / ∞ behavior', { fs: 13, fw: 700, fill: D.red.dark });
  o += card(282, 102, 238, 142, C.lgreen, C.green, 'Compact Disc Bound (R = 2)', '∥(x,y)∥ > 2 ⇒ |ψ| ≤ 2e⁻⁴ < 0.04', 'Inside disc: ψ(−1/√2, 0) ≈ −0.43', D.green.dark);
  o += tx(401, 214, 'Extreme Value Thm ⇒ Global Min!', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function hessianLse() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, 'The Hessian Matrix H_f(x) = ∇²f(x): [H_f]_{ij} = ∂²f / (∂xᵢ ∂xⱼ)', 'LSE f(x) = log ∑_{k} e^{x_k} has gradient p = softmax(x):', '∂pᵢ/∂xⱼ = pᵢ δ_{ij} − pᵢ pⱼ   ⇒   H_f(x) = diag(p) − p pᵀ', D.blue.dark);
  o += card(20, 104, 238, 140, C.lamber, C.amber, 'At (0, log 2): p = (⅓, ⅔)ᵀ', 'H_f = (2/9) [1, −1; −1, 1]', 'tr(H_f) = 4/9,   det(H_f) = 0', D.orange.dark);
  o += tx(139, 214, 'Positive semidefinite (H_f ⪰ 0)', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += card(282, 104, 238, 140, C.lgreen, C.green, 'Eigendirections of H_f', 'λ₁ = 0 along q₁ = (1, 1)ᵀ (flat shift)', 'λ₂ = 4/9 along q₂ = (1, −1)ᵀ (curved)', D.green.dark);
  o += tx(401, 214, 'Curves upward orthogonal to (1,1)ᵀ', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function clairautSchwarz() {
  let o = card(20, 14, 500, 72, D.purple.tint, C.purple, 'Clairaut–Schwarz Theorem: Continuous Mixed Partials Commute', '∂²f/∂xᵢ∂xⱼ = ∂²f/∂xⱼ∂xᵢ  ⇒  H_f(x) = H_f(x)ᵀ (symmetric)', 'Both orders are limits of the same symmetric 4-corner second difference Δ_h:', D.purple.dark);
  o += card(20, 102, 250, 142, C.lblue, C.blue, 'Symmetric Second Difference Δ_h', '[f(x+heᵢ+heⱼ) − f(x+heᵢ)', '', D.blue.dark);
  o += tx(145, 160, '− f(x+heⱼ) + f(x)] / h²', { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  o += tx(145, 184, 'Unchanged when swapping i ↔ j!', { fs: 12.5, fill: C.muted });
  o += tx(145, 214, '2× 1D MVT ⇒ Δ_h = ∂²_{ij} f(ξ_h)', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += arrow(274, 173, 290, 173, C.green);
  o += card(294, 102, 226, 142, C.lgreen, C.green, 'Spectral Theorem Payoff', 'Symmetric H_f = Q Λ Qᵀ', 'All n eigenvalues λᵢ are REAL', D.green.dark);
  o += tx(407, 214, 'Orthonormal principal axes qᵢ!', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function secondOrderTaylor() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, '1D Slice Reduction: φ(t) = f(x₀ + tδ) for t ∈ [0, 1]  (δ = x − x₀)', 'φ′(t) = ∇f(x₀ + tδ) · δ     and     φ″(t) = δᵀ H_f(x₀ + tδ) δ', 'Applying 1D Taylor with Lagrange remainder at t = 1 yields:', D.blue.dark);
  o += card(20, 104, 238, 140, C.lamber, C.amber, 'Exact Remainder Form', 'f(x₀+δ) = f(x₀) + ∇f(x₀)·δ', '+ ½ δᵀ H_f(ξ) δ   (ξ on segment)', D.orange.dark);
  o += tx(139, 214, 'Exact for some τ ∈ (0, 1)', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += card(282, 104, 238, 140, C.lgreen, C.green, 'Local Quadratic Bowl at x₀', 'f(x₀+δ) = f(x₀) + ∇f(x₀)·δ', '+ ½ δᵀ H_f(x₀) δ + o(∥δ∥²)', D.green.dark);
  o += tx(401, 214, 'C³ smooth ⇒ O(∥δ∥³) cubic gap', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function highdSaddles() {
  let o = card(20, 14, 500, 70, D.purple.tint, C.purple, 'Why Saddle Points Dominate High-Dimensional Loss Landscapes', 'A local minimum requires ALL n eigenvalues of H_f to be nonnegative (λ₁, …, λₙ ≥ 0)', '', D.purple.dark);
  o += card(20, 100, 238, 144, D.red.tint, D.red.base, 'Eigenvalue Repulsion (GOE)', 'Coin-flip heuristic: 2⁻ⁿ', 'True random symmetric H_f:', D.red.dark);
  o += tx(139, 192, 'P(H ≻ 0) ~ exp(−(ln 3 / 4) n²)', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(139, 218, 'Decays quadratically in n² in exp!', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(282, 100, 238, 144, C.lgreen, C.green, 'Loss vs. Index (Dauphin et al.)', 'Index = fraction of negative λᵢ', 'High loss ⇒ high index (saddles)', D.green.dark);
  o += tx(401, 192, 'Index → 0 near global minimum', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(401, 218, 'Local minima sit near low loss floor', { fs: 12.5, fill: C.ink });
  return svg(540, 258, o);
}

// ── §34.3 mdl-chain-rule-graphs.md (9 diagrams) ──
function graphSharing() {
  let o = card(20, 14, 238, 118, D.red.tint, D.red.base, 'Naive Perturbation (N = 10⁸)', 'Nudge wⱼ → wⱼ + ϵ one by one', 'Requires N = 10⁸ forward passes!', D.red.dark);
  o += tx(139, 108, 'Recomputes downstream layers 10⁸×', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(282, 14, 238, 118, C.lgreen, C.green, 'Computation Graph (DAG)', 'Each intermediate node stored once', 'All early weights share upper path!', D.green.dark);
  o += tx(401, 108, '1 backward sweep reuses upper adjoints', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += card(20, 148, 500, 96, D.purple.tint, C.purple, 'Backpropagation = Factoring the Path-Sum Chain Rule from the Output Backward', 'Every parameter w₁, …, w_N shares the downstream sensitivity df/du and df/dv;', 'computing adjoints top-down visits each directed edge in the DAG only ONCE!', D.purple.dark);
  return svg(540, 258, o);
}

function parallelBranches() {
  let o = card(20, 48, 96, 76, C.lblue, C.blue, 'Node a', 'a + ϵ', '', D.blue.dark);
  o += arrow(120, 72, 196, 44, C.blue);
  o += tx(156, 38, '∂u/∂a', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += arrow(120, 100, 196, 128, C.amber);
  o += tx(156, 134, '∂v/∂a', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += card(200, 14, 134, 58, C.lblue, C.blue, 'Child u(a, b)', 'Δu ≈ ϵ ∂u/∂a', '', D.blue.dark);
  o += card(200, 100, 134, 58, C.lamber, C.amber, 'Child v(a, b)', 'Δv ≈ ϵ ∂v/∂a', '', D.orange.dark);
  o += arrow(338, 44, 412, 72, C.blue);
  o += tx(376, 38, '∂f/∂u', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += arrow(338, 128, 412, 100, C.amber);
  o += tx(376, 134, '∂f/∂v', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += card(416, 48, 104, 76, C.lgreen, C.green, 'Output f', 'f(u, v)', '', D.green.dark);
  o += card(20, 174, 500, 70, D.purple.tint, C.purple, 'Multivariable Chain Rule Across Parallel Branches', 'df/da = (∂f/∂u)(∂u/∂a) + (∂f/∂v)(∂v/∂a)   (multiply along branches, add at merge)', '', D.purple.dark);
  return svg(540, 258, o);
}

function edgeVsTotal() {
  let o = card(20, 14, 238, 138, C.lblue, C.blue, 'Edge Derivative: ∂c / ∂n', 'Local partial of child c formula', 'Holds ALL OTHER parents of c fixed!', D.blue.dark);
  o += tx(139, 124, 'Single directed edge n → c', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += card(282, 14, 238, 138, C.lgreen, C.green, 'Total Derivative: df / dn = n̄', 'Adjoint of node n w.r.t. output f', 'ALL downstream nodes respond!', D.green.dark);
  o += tx(401, 124, 'Sums over ALL paths n → ⋯ → f', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += card(20, 168, 500, 76, C.lamber, C.amber, 'Skip-Edge Example: f(r, p, s) = r + p + s  with  r = p²  at p = 3', 'Local skip edge: ∂f/∂p = 1 (holding r fixed)   vs   Total adjoint: df/dp = 2p + 1 = 7!', '', D.orange.dark);
  return svg(540, 258, o);
}

function pathSumRule() {
  let o = card(20, 14, 500, 74, D.purple.tint, C.purple, 'Path-Sum Chain Rule on Any Differentiable Computation DAG', 'df/dx = ∑_{paths P: x → f} ∏_{(u→v) ∈ P} ∂v/∂u', 'Each path multiplies its edge slopes; parallel paths add together at merge nodes', D.purple.dark);
  o += card(20, 104, 238, 140, D.red.tint, D.red.base, 'Explicit Path Enumeration', 'm stages × 2 branches = 2ᵐ paths', 'At m = 50: 2⁵⁰ ≈ 10¹⁵ paths!', D.red.dark);
  o += tx(139, 214, 'Exponential O(2ᵐ) — intractable!', { fs: 13, fw: 700, fill: D.red.dark });
  o += arrow(262, 174, 278, 174, C.green);
  o += card(282, 104, 238, 140, C.lgreen, C.green, 'Dynamic Programming (AD)', 'Factor shared sub-paths at nodes', 'n̄ = ∑_{c ∈ children(n)} c̄ (∂c/∂n)', D.green.dark);
  o += tx(401, 214, 'Linear O(|Edges|) time!', { fs: 13.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function resnetPaths() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, 'Residual Stack: x_{l+1} = x_l + g_l(x_l)  Across l = 0, …, m − 1', 'dx_{m}/dx₀ = ∏_{l} (1 + g_l′(x_l)) = ∑_{S ⊆ {0..m−1}} ∏_{l∈S} g_l′(x_l)', 'Each block branches into an identity skip highway (slope 1) + residual branch g_l′', D.blue.dark);
  o += card(20, 104, 238, 140, C.lgreen, C.green, '2ᵐ Total Paths Ensemble', 'C(m, k) paths visit k blocks', 'and skip m − k blocks with slope 1', D.green.dark);
  o += tx(139, 214, 'Unbroken gradient highway to x₀!', { fs: 13, fw: 700, fill: D.green.dark });
  o += card(282, 104, 238, 140, C.lamber, C.amber, 'Veit et al. (m = 54 ResNet)', '2⁵⁴ ≈ 1.8 × 10¹⁶ paths total', 'Most gradient flows via k ∈ [5, 17]', D.orange.dark);
  o += tx(401, 214, 'Short effective paths prevent vanishing', { fs: 12.5, fw: 700, fill: D.orange.dark });
  return svg(540, 258, o);
}

function forwardEdgeTable() {
  const m = (x, y, t, fill = C.ink) => tx(x, y, t, { mono: true, fs: 11.5, fw: 700, fill });
  let o = tx(270, 20, 'Forward Pass & Local Edge Slopes at (w, x, y, z) = (1, 1, −2, 1)', { fs: 15.5, fw: 700, fill: C.ink });
  o += card(16, 38, 156, 110, C.lblue, C.blue, 'Inputs (Layer 0)', '', '', D.blue.dark);
  o += m(94, 78, 'w = 1, x = 1');
  o += m(94, 96, 'y = −2, z = 1');
  o += m(94, 114, 'w+x+y+z = 1', D.blue.dark);
  o += m(94, 132, 'w+x−y−z = 3', D.blue.dark);
  o += arrow(175, 93, 189, 93, C.blue);
  o += card(192, 38, 156, 110, C.lamber, C.amber, 'Layer 1 (a, b)', '', '', D.orange.dark);
  o += m(270, 78, 'a = 1² = 1');
  o += m(270, 96, 'b = 3² = 9');
  o += m(270, 114, '∂a/∂(w,x,y,z) = 2', D.orange.dark);
  o += m(270, 132, '∂b/∂(w,x;y,z) = ±6', D.orange.dark);
  o += arrow(351, 93, 365, 93, C.green);
  o += card(368, 38, 156, 110, C.lgreen, C.green, 'Layer 2 & Output', '', '', D.green.dark);
  o += m(446, 78, 'u = 100, v = 64');
  o += m(446, 96, 'f = 164² = 26,896');
  o += m(446, 114, '∂u/∂(a,b) = 20', D.green.dark);
  o += m(446, 132, '∂v/∂(a,b) = ∓16', D.green.dark);
  o += card(16, 164, 508, 80, D.purple.tint, C.purple, 'All 14 Local Edge Derivatives Are Known After 1 Forward Pass', 'Top edges: ∂f/∂u = ∂f/∂v = 2(u + v) = 328', 'Each edge derivative ∂c/∂n depends ONLY on node c and its immediate parents!', D.purple.dark);
  return svg(540, 258, o);
}

function forwardModeTrace() {
  let o = tx(270, 20, 'Forward-Mode Sweep Seeded at w (dw/dw = 1, dx/dw = dy/dw = dz/dw = 0)', { fs: 15, fw: 700, fill: C.ink });
  o += card(16, 38, 156, 108, C.lblue, C.blue, '1. Layer 1 Tangents', 'da/dw = ∂a/∂w = 2', 'db/dw = ∂b/∂w = 6', D.blue.dark);
  o += arrow(175, 92, 189, 92, C.blue);
  o += card(192, 38, 156, 108, C.lamber, C.amber, '2. Layer 2 Tangents', 'du/dw = 20(2)+20(6) = 160', 'dv/dw = −16(2)+16(6) = 64', D.orange.dark);
  o += arrow(351, 92, 365, 92, C.green);
  o += card(368, 38, 156, 108, C.lgreen, C.green, '3. Output Tangent', 'df/dw = 328(160)+328(64)', '= 73,472  (6 mults)', D.green.dark);
  o += card(16, 162, 508, 82, D.red.tint, D.red.base, 'The Forward-Mode Bottleneck for N Inputs', 'du/dw = 160, dv/dw = 64 depend on w: no reuse for y or z!', 'Finding all 4 input derivatives takes 4 × 6 = 24 multiplications (N sweeps for N weights)', D.red.dark);
  return svg(540, 258, o);
}

function backwardModeTrace() {
  let o = tx(270, 20, 'Reverse-Mode Sweep Seeded at Output f̄ = df/df = 1 (Top-Down)', { fs: 15.5, fw: 700, fill: C.ink });
  o += card(16, 38, 156, 108, C.lgreen, C.green, '1. Top Layer (u, v)', 'f̄ = 1', 'ū = v̄ = 1 · 328 = 328', D.green.dark);
  o += arrow(175, 92, 189, 92, C.green);
  o += card(192, 38, 156, 108, C.lamber, C.amber, '2. Layer 1 (a, b)', 'ā = 328(20−16) = 1,312', 'b̄ = 328(20+16) = 11,808', D.orange.dark);
  o += arrow(351, 92, 365, 92, C.blue);
  o += card(368, 38, 156, 108, C.lblue, C.blue, '3. All 4 Inputs!', 'w̄ = x̄ = 1312(2)+11808(6)', 'ȳ = z̄ = 1312(2)−11808(6)', D.blue.dark);
  o += card(16, 162, 508, 82, D.purple.tint, C.purple, 'One Reverse Sweep Returns All 4 Gradients in 12 Multiplications (Half of 24)', 'w̄ = x̄ = +73,472   and   ȳ = z̄ = −68,224', 'Because ā and b̄ summarize the upper graph once, each input costs only 2 mults!', D.purple.dark);
  return svg(540, 258, o);
}

function gradCheckPrecision() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, 'Central-Difference Error = Truncation O(h²) + Roundoff O(ε_mach / h)', 'D₀f = (f(w+h) − f(w−h))/(2h) ≈ f′(w) + ⅙f‴(ξ)h² ± ε_mach|f|/h', 'Balancing h² ~ ε_mach / h gives optimal step h* ∝ ε_mach^{1/3} and min error O(ε_mach^{2/3})', D.blue.dark);
  o += card(20, 102, 238, 142, C.lgreen, C.green, 'Float64 (ε_mach ≈ 2.2 × 10⁻¹⁶)', 'Optimal step: h* ≈ 10⁻⁵', 'Min relative error: ~10⁻¹¹', D.green.dark);
  o += tx(139, 214, 'Cleanly catches subtle grad bugs!', { fs: 13, fw: 700, fill: D.green.dark });
  o += card(282, 102, 238, 142, D.red.tint, D.red.base, 'Float32 (ε_mach ≈ 1.2 × 10⁻⁷)', 'Optimal step: h* ≈ 10⁻²', 'Min relative error: ~10⁻⁴', D.red.dark);
  o += tx(401, 186, 'At h ≤ 10⁻⁸: w + h == w → grad = 0!', { fs: 12.5, fw: 600, fill: D.red.dark });
  o += tx(401, 214, 'Always run grad checks in float64!', { fs: 13, fw: 700, fill: D.red.dark });
  return svg(540, 258, o);
}

// ── §34.4 mdl-matrix-calculus.md (13 diagrams) ──
function threeDerivShapes() {
  let o = card(20, 14, 500, 68, C.lblue, C.blue, '2-2-1 ReLU Network at x = (1, 1)ᵀ: g(x) = w₂ᵀ φ(W₁x + b)', 'Forward pass: a = W₁x + b = (3, −2)ᵀ  →  h = φ(a) = (3, 0)ᵀ (unit 2 inactive)  →  g = 6', '', D.blue.dark);
  o += card(16, 98, 160, 146, C.lgreen, C.green, '1. Input Jacobian', '∂g / ∂x ∈ ℝ¹ˣ²', '2δ₁ + 4δ₂ → [2, 4]', D.green.dark);
  o += tx(96, 212, '∇ₓg = [2,4]ᵀ ∈ ℝ²ˣ¹', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += card(190, 98, 160, 146, C.lamber, C.amber, '2. Weight Gradient', '∇_{W₁} g ∈ ℝ²ˣ²', '[2, 2;  0, 0]', D.orange.dark);
  o += tx(270, 212, 'Matches W₁ shape (2×2)', { fs: 12.5, fw: 700, fill: D.orange.dark });
  o += card(364, 98, 160, 146, D.purple.tint, C.purple, '3. Hidden Jacobian', '∂h / ∂x ∈ ℝ²ˣ²', 'diag(1, 0) W₁ = [1,2;0,0]', D.purple.dark);
  o += tx(444, 212, 'Maps ℝ² → ℝ² locally', { fs: 12.5, fw: 700, fill: D.purple.dark });
  return svg(540, 258, o);
}

function jacobianGrid() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, 'The Jacobian Matrix J_f(x) ∈ ℝᵐˣⁿ for f : ℝⁿ → ℝᵐ', 'f(x + δ) = f(x) + J_f(x) δ + o(∥δ∥),       [J_f(x)]_{ij} = ∂fᵢ / ∂xⱼ (x)', '', D.blue.dark);
  o += card(20, 102, 238, 142, C.lgreen, C.green, 'Row View (m Rows)', 'Row i = (∇fᵢ(x))ᵀ ∈ ℝ¹ˣⁿ', 'Linear model of output coordinate fᵢ', D.green.dark);
  o += tx(139, 214, 'm outputs → m rows', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += card(282, 102, 238, 142, C.lamber, C.amber, 'Column View (n Columns)', 'Col j = ∂f / ∂xⱼ ∈ ℝᵐˣ¹', 'Velocity of f when input xⱼ moves', D.orange.dark);
  o += tx(401, 214, 'n inputs → n columns', { fs: 13.5, fw: 700, fill: D.orange.dark });
  return svg(540, 258, o);
}

function shapeConventions() {
  let o = card(16, 14, 246, 106, C.lblue, C.blue, '1. Jacobian J_f = ∂f/∂x ∈ ℝᵐˣⁿ', 'Numerator layout: out × in', 'Composes directly: J_{g∘f} = J_g J_f', D.blue.dark);
  o += tx(139, 100, 'For scalar f (m=1): 1 × n row', { fs: 13, fill: D.blue.dark });
  o += card(278, 14, 246, 106, C.lgreen, C.green, '2. Vector Gradient ∇_x f ∈ ℝⁿˣ¹', '∇_x f = (∂f/∂x)ᵀ (column)', 'Matches shape of x for x ← x − η∇f', D.green.dark);
  o += tx(401, 100, 'df = ⟨∇_x f, dx⟩ = (∇_x f)ᵀ dx', { mono: true, fs: 12.5, fill: D.green.dark });
  o += card(16, 136, 246, 106, C.lamber, C.amber, '3. Matrix Gradient ∇_W f ∈ ℝ@@', '[∇_W f]_{ij} = ∂f / ∂W_{ij}', 'Exact same p × q shape as W!', D.orange.dark)
    .replace('ℝ@@<', 'ℝ<tspan baseline-shift="super" font-size="70%">p×q</tspan><');
  o += tx(139, 222, 'df = tr((∇_W f)ᵀ dW)', { mono: true, fs: 12, fill: D.orange.dark });
  o += card(278, 136, 246, 106, D.purple.tint, C.purple, '4. Hessian Matrix H_f ∈ ℝⁿˣⁿ', 'H_f(x) = J_{∇f}(x)', 'Jacobian of gradient map ∇f : ℝⁿ→ℝⁿ', D.purple.dark);
  o += tx(401, 222, 'Symmetric: H_f = H_fᵀ when f ∈ C²', { fs: 13, fill: D.purple.dark });
  return svg(540, 256, o);
}

function jacobianChainProof() {
  let o = card(16, 16, 152, 106, C.lblue, C.blue, '1. Perturb Input x', 'x ↦ x + δ  (∈ ℝⁿ)', 'A = J_f(x) ∈ ℝᵖˣⁿ', D.blue.dark);
  o += arrow(172, 69, 190, 69, C.blue);
  o += card(194, 16, 152, 106, C.lamber, C.amber, '2. Intermediate Shift', 'y = f(x) ↦ y + η', 'η = Aδ + o(∥δ∥) ∈ ℝᵖ', D.orange.dark);
  o += arrow(350, 69, 368, 69, C.green);
  o += card(372, 16, 152, 106, C.lgreen, C.green, '3. Outer Expansion', 'g(y+η) = g(y)+Bη+o(∥η∥)', '= g(y) + BAδ + o(∥δ∥)', D.green.dark);
  o += card(16, 140, 508, 104, D.purple.tint, C.purple, 'Matrix Chain Rule: J_{g ∘ f}(x) = J_g(f(x)) · J_f(x) = B A ∈ ℝᵐˣⁿ', 'Entry (i, k): [BA]_{ik} = ∑_{j=1}^{p} (∂gᵢ / ∂yⱼ)(∂fⱼ / ∂xₖ)  (sums over all p intermediate coordinates yⱼ)', `Unifies multivariable path sums (${sec('sec_mdl-chain-rule-graphs')}) with matrix multiplication (${sec('sec_mdl-linear-maps')})!`, D.purple.dark);
  return svg(540, 258, o);
}

function deepJacobianNorms() {
  let o = card(20, 14, 500, 68, C.lblue, C.blue, 'Deep Network Input–Output Jacobian: J = J_K J_{K−1} ⋯ J₁', 'For repeated weight matrix W = VΛV⁻¹ across K = 50 steps:   W^{K} = V Λ^{K} V⁻¹ ~ |λ_max|^{K}', '', D.blue.dark);
  o += card(20, 98, 238, 146, C.lamber, C.amber, 'Vanishing Gradients (|λ| < 1)', 'At |λ_max| = 0.9, depth K = 50:', '0.9⁵⁰ ≈ 0.005  (200× shrinkage)', D.orange.dark);
  o += tx(139, 214, 'At 0.5⁵⁰ ≈ 8.9 × 10⁻¹⁶ (stalls!)', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += card(282, 98, 238, 146, D.red.tint, D.red.base, 'Exploding Gradients (|λ| > 1)', 'At |λ_max| = 1.1, depth K = 50:', '1.1⁵⁰ ≈ 117.4  (exponential blowup)', D.red.dark);
  o += tx(401, 214, 'Plus non-normal transient spikes!', { fs: 13, fw: 700, fill: D.red.dark });
  return svg(540, 258, o);
}

function differentialRecipe() {
  let o = tx(270, 20, 'The Differential Method: One 2-Step Recipe Replaces Derivative Tables', { fs: 15.5, fw: 700, fill: C.ink });
  o += card(20, 38, 238, 144, C.lblue, C.blue, 'Step 1: Expand Differential df', '• d(aU + bV) = a dU + b dV', '• d(AB) = (dA)B + A(dB)', D.blue.dark);
  o += tx(139, 128, '• d(Aᵀ) = (dA)ᵀ', { mono: true, fs: 13, fw: 600, fill: C.ink });
  o += tx(139, 152, '• dφ(X) = φ′(X) ⊙ dX', { mono: true, fs: 13, fw: 600, fill: C.ink });
  o += arrow(262, 110, 278, 110, C.green);
  o += card(282, 38, 238, 144, C.lgreen, C.green, 'Step 2: Isolate dX via Trace', 'Rearrange into canonical form:', 'df = ⟨M, dX⟩ = tr(Mᵀ dX)', D.green.dark);
  o += tx(401, 134, 'Read off gradient directly:', { fs: 13.5, fw: 600, fill: C.ink });
  o += tx(401, 160, '∇_X f = M  (same shape as X!)', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += box(20, 196, 500, 48, D.purple.tint, C.purple);
  o += tx(270, 220, 'Frobenius Inner Product: ⟨A, B⟩ = tr(AᵀB) = ∑_{i,j} A_{ij} B_{ij}', { mono: true, fs: 14, fw: 700, fill: D.purple.dark });
  return svg(540, 258, o);
}

function traceMoveRule() {
  let o = card(20, 14, 500, 68, D.purple.tint, C.purple, 'The Trace Move Rule (From Cyclic Invariance tr(UV) = tr(VU))', 'Moves multipliers off dX onto the adjoint side of ⟨·, ·⟩ with automatic transposes', '', D.purple.dark);
  o += card(16, 98, 160, 146, C.lblue, C.blue, '1. Left Factor B', '⟨A, B dX⟩', '= ⟨Bᵀ A, dX⟩', D.blue.dark);
  o += tx(96, 198, 'Left B moves across as', { fs: 12.5, fill: C.ink });
  o += tx(96, 220, 'left Bᵀ!', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += card(190, 98, 160, 146, C.lamber, C.amber, '2. Right Factor C', '⟨A, (dX) C⟩', '= ⟨A Cᵀ, dX⟩', D.orange.dark);
  o += tx(270, 198, 'Right C moves across as', { fs: 12.5, fill: C.ink });
  o += tx(270, 220, 'right Cᵀ!', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += card(364, 98, 160, 146, C.lgreen, C.green, '3. Hadamard ⊙', '⟨A, B ⊙ dX⟩', '= ⟨A ⊙ B, dX⟩', D.green.dark);
  o += tx(444, 198, 'Elementwise mask B', { fs: 12.5, fill: C.ink });
  o += tx(444, 220, 'moves unchanged!', { fs: 13.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function workedMatrixGrads() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, '1. Linear Form: f(x) = aᵀx', 'df = aᵀ dx = ⟨a, dx⟩     ⇒     ∇_x (aᵀx) = a', '', D.blue.dark);
  o += card(20, 98, 500, 72, C.lamber, C.amber, '2. Quadratic Form: f(x) = xᵀAx   (General Square A ∈ ℝⁿˣⁿ)', 'df = (dx)ᵀAx + xᵀA dx = ⟨Ax, dx⟩ + ⟨Aᵀx, dx⟩ = ⟨(A + Aᵀ)x, dx⟩', '⇒  ∇_x (xᵀAx) = (A + Aᵀ)x     (equals 2Ax ONLY when A = Aᵀ!)', D.orange.dark);
  o += card(20, 182, 500, 72, C.lgreen, C.green, '3. Least Squares: ℓ(W) = ∥Wx − y∥₂² = ⟨r, r⟩   (r = Wx − y)', 'dℓ = 2⟨r, (dW)x⟩ = ⟨2r xᵀ, dW⟩     ⇒     ∇_W ℓ = 2(Wx − y)xᵀ', 'Outer product of residual column r ∈ ℝᵐˣ¹ and input row xᵀ ∈ ℝ¹ˣⁿ', D.green.dark);
  return svg(540, 268, o);
}

function threeSanityChecks() {
  let o = tx(270, 20, 'Three Sanity Checks for Any Hand-Derived Matrix Gradient', { fs: 16, fw: 700, fill: C.ink });
  o += card(16, 38, 160, 134, C.lblue, C.blue, '1. Shape Check', 'dim(∇_X f) == dim(X)', 'Catches rectangular', D.blue.dark);
  o += tx(96, 116, 'transpose bugs (p ≠ q)', { fs: 12.5, fill: C.muted });
  o += tx(96, 146, 'Misses: square p = q!', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(190, 38, 160, 134, C.lamber, C.amber, '2. 1×1 Scalar Check', 'All matrices 1×1', 'Catches missing 2s,', D.orange.dark);
  o += tx(270, 116, 'signs & dropped terms', { fs: 12.5, fill: C.muted });
  o += tx(270, 146, 'Misses: all transposes!', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(364, 38, 160, 134, C.lgreen, C.green, '3. Nonsymmetric Test', 'Random square A ≠ Aᵀ', 'Compare vs autograd', D.green.dark);
  o += tx(444, 116, 'or central differences', { fs: 12.5, fill: C.muted });
  o += tx(444, 146, 'Catches EVERYTHING!', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += card(16, 186, 508, 60, D.red.tint, D.red.base, 'Square Trap: For A = [0,1;0,0] and x = (1,2)ᵀ, xᵀAx = x₁x₂ has ∇f = (2,1)ᵀ = (A+Aᵀ)x', 'Naive 2Ax = (4,0)ᵀ passes checks 1 and 2, yet is wrong!', '', D.red.dark);
  return svg(540, 260, o);
}

function layerVjpDuality() {
  let o = card(16, 20, 170, 108, C.lblue, C.blue, 'Layer Input X', 'Forward: X ↦ Y = F(X)', 'Differential: dX ↦ dY', D.blue.dark);
  o += arrow(190, 52, 346, 52, C.blue);
  o += tx(268, 36, 'Forward Pass: Y = F(X)', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += arrow(346, 98, 190, 98, D.red.base);
  o += tx(268, 118, 'Backward VJP: Ȳ ↦ X̄', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += card(350, 20, 174, 108, C.lgreen, C.green, 'Layer Output Y', 'Upstream adjoint:', 'Ȳ = ∇_Y ℓ', D.green.dark);
  o += card(16, 146, 508, 98, D.purple.tint, C.purple, 'Adjoint Duality Identity: ⟨X̄, dX⟩ = ⟨Ȳ, dY⟩  for Every Perturbation dX', 'Never forms the 10¹²-entry tensor ∂Y/∂X! For vector layer dy = J dx:', '⟨ȳ, J dx⟩ = ⟨Jᵀ ȳ, dx⟩   ⇒   x̄ = Jᵀ ȳ   (or row form x̄ᵀ = ȳᵀ J)', D.purple.dark);
  return svg(540, 258, o);
}

function batchedMatmulVjp() {
  let o = card(20, 14, 500, 68, C.lblue, C.blue, 'Batched Linear Layer: Y = XW   (X ∈ ℝ@1,  W ∈ ℝ@2  ⇒  Y ∈ ℝ@3)', '⟨Ȳ, dY⟩ = ⟨Ȳ, (dX)W + X(dW)⟩ = ⟨Ȳ Wᵀ, dX⟩ + ⟨Xᵀ Ȳ, dW⟩', '', D.blue.dark)
    .replace('ℝ@1', 'ℝ<tspan baseline-shift="super" font-size="70%">n×d</tspan>').replace('ℝ@2', 'ℝ<tspan baseline-shift="super" font-size="70%">d×q</tspan>').replace('ℝ@3', 'ℝ<tspan baseline-shift="super" font-size="70%">n×q</tspan>');
  o += card(20, 98, 238, 146, C.lgreen, C.green, '1. Input Adjoint X̄ ∈ ℝ@@', 'X̄ = Ȳ Wᵀ   (ndq mults)', '(n × q) · (q × d) → (n × d)', D.green.dark)
    .replace('ℝ@@', 'ℝ<tspan baseline-shift="super" font-size="70%">n×d</tspan>');
  o += tx(139, 214, 'Needs W saved from forward pass', { fs: 13, fw: 700, fill: D.green.dark });
  o += card(282, 98, 238, 146, C.lamber, C.amber, '2. Weight Adjoint W̄ ∈ ℝ@@', 'W̄ = Xᵀ Ȳ   (ndq mults)', '(d × n) · (n × q) → (d × q)', D.orange.dark)
    .replace('ℝ@@', 'ℝ<tspan baseline-shift="super" font-size="70%">d×q</tspan>');
  o += tx(401, 214, 'Needs X saved on forward tape!', { fs: 13, fw: 700, fill: D.orange.dark });
  return svg(540, 258, o);
}

function softmaxJacobianVar() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, 'Softmax Jacobian = Hessian of Log-Sum-Exp', 'p = softmax(z)  ⇒  J_softmax(z) = diag(p) − p pᵀ ∈ ℝ^{C×C}', 'Shift invariance p(z + c1) = p(z)  ⇒  J_softmax 1 = p − p(pᵀ1) = 0  (null vector 1)', D.blue.dark);
  o += card(20, 102, 500, 140, C.lgreen, C.green, 'Statistical Proof of Global Convexity of Log-Sum-Exp', 'vᵀ J_softmax v = ∑ᵢ pᵢ vᵢ² − (∑ᵢ pᵢ vᵢ)² = E_{I~p}[v_I²] − (E_{I~p}[v_I])² = Var_{I~p}(v_I) ≥ 0', 'Because variance is never negative, H_LSE = J_softmax ⪰ 0 everywhere!', D.green.dark);
  o += tx(270, 216, 'Fast O(C) VJP without forming C×C matrix: J_softmaxᵀ u = p ⊙ (u − (pᵀu) 1)', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 256, o);
}

function softmaxCeCancel() {
  let o = card(16, 16, 152, 108, C.lblue, C.blue, '1. Logits z ∈ ℝ^{C}', 'p = softmax(z)', 'J_p = diag(p) − ppᵀ', D.blue.dark);
  o += arrow(172, 70, 190, 70, C.blue);
  o += card(194, 16, 152, 108, C.lamber, C.amber, '2. Cross-Entropy ℓ', 'ℓ(p) = −∑ᵢ yᵢ log pᵢ', '∂ℓ/∂pᵢ = −yᵢ / pᵢ', D.orange.dark);
  o += arrow(350, 70, 368, 70, C.green);
  o += card(372, 16, 152, 108, C.lgreen, C.green, '3. 1/pᵢ Cancels!', '−∑ᵢ (yᵢ/pᵢ) pᵢ(δᵢⱼ−pⱼ)', '= (∑ᵢ yᵢ) pⱼ − yⱼ', D.green.dark);
  o += card(16, 140, 508, 104, D.purple.tint, C.purple, 'Clean Logit Gradient: ∇_z ℓ = p − y   (Whenever ∑_{i=1}^{C} yᵢ = 1)', 'Avoids dividing by tiny pᵢ ≈ 0 (numerical stability!) and gives prediction − target directly', 'Caution: if target y is unnormalized (e.g. two-hot ∑ yᵢ = 2), gradient is (∑ yᵢ)p − y = 2p − y!', D.purple.dark);
  return svg(540, 258, o);
}

// ── §34.5 mdl-matrix-calculus-autodiff.md (12 diagrams) ──
function perturbationVsAd() {
  let o = card(20, 14, 238, 128, D.red.tint, D.red.base, 'Finite Differences / Fwd Mode', '10⁸ parameters → 10⁸ passes', 'At 10 ms/pass: 11.6 DAYS / step!', D.red.dark);
  o += tx(139, 116, 'Propagates 1 input column J eⱼ', { fs: 13, fw: 700, fill: D.red.dark });
  o += card(282, 14, 238, 128, C.lgreen, C.green, 'Reverse-Mode Autodiff (VJP)', '1 scalar loss → 1 backward sweep', 'Takes ~3 forward passes (~30 ms)!', D.green.dark);
  o += tx(401, 116, 'Propagates 1 output row uᵀ J', { fs: 13, fw: 700, fill: D.green.dark });
  o += card(20, 158, 500, 86, D.purple.tint, C.purple, 'Where the 10⁸ Speedup Comes From: Matrix Multiplication Associativity', 'Forward mode: J_K(⋯(J₂(J₁ v)))  (1 column per sweep → n = 10⁸ sweeps)', 'Reverse mode: (((uᵀ J_K) J_{K−1}) ⋯) J₁  (1 row per sweep → m = 1 sweep!)', D.purple.dark);
  return svg(540, 258, o);
}

function dualNumberAlgebra() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, 'Dual Numbers a + bε with ε² = 0: Truncating Taylor Series Algebraically', '(a + bε) + (c + dε) = (a + c) + (b + d)ε', '(a + bε)(c + dε) = ac + (ad + bc)ε + bdε² = ac + (ad + bc)ε   (Product Rule!)', D.blue.dark);
  o += card(20, 104, 238, 140, C.lgreen, C.green, 'Exact Powers at x = 3 (3 + 1ε)', '(3 + ε)² = 9 + 6ε  (f=9, f′=6)', '(3 + ε)³ = 27 + 27ε (f=27, f′=27)', D.green.dark);
  o += tx(139, 214, 'Zero truncation or roundoff step h!', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += card(282, 104, 238, 140, D.purple.tint, C.purple, '2×2 Matrix Representation', 'a + bε  ↔  [a, b;  0, a]', 'ε = [0, 1; 0, 0]  satisfies  ε² = 0!', D.purple.dark);
  o += tx(401, 214, 'Ordinary 2×2 matrix arithmetic', { fs: 13, fw: 700, fill: D.purple.dark });
  return svg(540, 258, o);
}

function dualInduction() {
  let o = card(20, 14, 500, 68, D.purple.tint, C.purple, 'Structural Induction: Evaluating f(x + 1ε) = f(x) + f′(x)ε', 'Every primitive φ (sin, exp, log) is overloaded via φ(a + bε) = φ(a) + φ′(a) b ε', '', D.purple.dark);
  o += card(16, 98, 160, 146, C.lblue, C.blue, '1. Base Cases', 'Constant c ↦ c + 0ε', 'Seeded input x ↦ x + 1ε', D.blue.dark);
  o += tx(96, 214, 'Exact slopes 0 and 1', { fs: 13, fw: 700, fill: D.blue.dark });
  o += card(190, 98, 160, 146, C.lamber, C.amber, '2. Inductive Step', 'g + h ↦ (g′ + h′)ε', 'g · h ↦ (gh′ + g′h)ε', D.orange.dark);
  o += tx(270, 214, 'φ(g) ↦ φ′(g(x))g′(x)ε', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += card(364, 98, 160, 146, C.lgreen, C.green, '3. Control Flow', 'Loops & if/else branches', 'execute concrete ops', D.green.dark);
  o += tx(444, 214, 'Differentiates active path!', { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function fwdLsePasses() {
  let o = tx(270, 20, 'Forward Mode on ℓ(x₁, x₂, x₃) = log(e^{x₁} + e^{x₂} + e^{x₃}): 1 Pass per Input!', { fs: 15, fw: 700, fill: C.ink });
  o += card(16, 38, 160, 122, C.lblue, C.blue, 'Pass 1: Seed e₁', 'x + (1, 0, 0)ε', 'Returns ℓ & ∂ℓ/∂x₁ = p₁', D.blue.dark);
  o += tx(96, 138, 'p₁ = 0.2537', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += card(190, 38, 160, 122, C.lamber, C.amber, 'Pass 2: Seed e₂', 'x + (0, 1, 0)ε', 'Returns ℓ & ∂ℓ/∂x₂ = p₂', D.orange.dark);
  o += tx(270, 138, 'p₂ = 0.0566', { mono: true, fs: 13, fw: 700, fill: D.orange.dark });
  o += card(364, 38, 160, 122, C.lgreen, C.green, 'Pass 3: Seed e₃', 'x + (0, 0, 1)ε', 'Returns ℓ & ∂ℓ/∂x₃ = p₃', D.green.dark);
  o += tx(444, 138, 'p₃ = 0.6897', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += card(16, 174, 508, 70, D.red.tint, D.red.base, 'Redundant Work: Recomputes ℓ(x) = 2.371539 Three Separate Times', 'For n = 10⁸ parameters, forward mode requires 10⁸ separate passes!', '', D.red.dark);
  return svg(540, 258, o);
}

function jvpVectorTangent() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, 'Jacobian–Vector Product (JVP): Seeding Forward Mode with Direction v ∈ ℝⁿ', 'f(x + vε) = f(x) + (J_f(x) v) ε  (one pass: ∑ⱼ vⱼ · column j of J)', 'Setting v = eⱼ extracts column j of J_f(x) in O(1) memory', D.blue.dark);
  o += card(20, 102, 238, 142, C.lamber, C.amber, 'Scalar Tangent (b ∈ ℝ)', '1 directional derivative Jv', 'O(1) arithmetic per primitive', D.orange.dark);
  o += tx(139, 214, 'Ideal when n ≪ m (few inputs)', { fs: 13, fw: 700, fill: D.orange.dark });
  o += card(282, 102, 238, 142, D.red.tint, D.red.base, 'Vector Tangent (B = Iₙ ∈ ℝⁿˣⁿ)', 'Propagates length-n arrays in Dual', 'Every + and * does O(n) vector ops!', D.red.dark);
  o += tx(401, 214, 'Same O(n) total arithmetic cost!', { fs: 13, fw: 700, fill: D.red.dark });
  return svg(540, 258, o);
}

function varTapeAnatomy() {
  const code = (x, y, t, fill = C.ink, fw = 600) => rawTx(x, y, t, { mono: true, fs: 11.5, fw, fill, anchor: 'start' });
  let o = card(16, 14, 244, 132, C.lblue, C.blue, '1. Var Node State on Tape', '', '', D.blue.dark);
  // Two columns: SVG collapses runs of spaces, so align with x, not padding.
  const field = (y, name, desc, fill, fw) => code(32, y, name, fill, fw) + code(110, y, ': ' + desc, fill, fw);
  o += field(54, '.value', 'forward float');
  o += field(76, '.grad', 'adjoint, 0.0');
  o += field(98, '._parents', 'parent Vars');
  o += field(120, '._backward', 'VJP closure', D.blue.dark, 700);
  o += card(280, 14, 244, 132, C.lamber, C.amber, '2. Local VJP Closures', '', '', D.orange.dark);
  o += code(294, 54, 'Add: a.grad += g; b.grad += g');
  o += code(294, 76, 'Mul: a.grad += b.value * g');
  o += code(331, 98, 'b.grad += a.value * g');
  o += tx(402, 124, 'Captures forward values on tape!', { fs: 12.5, fw: 700, fill: D.orange.dark });
  o += card(16, 162, 508, 82, C.lgreen, C.green, '3. Post-Order Topological Sort & Reversed Replay (backprop)', '', '', D.green.dark);
  o += tx(270, 202, 'DFS post-order places every parent before its children; iterating over reversed(order)', { fs: 12.5, fill: C.ink });
  o += tx(270, 224, 'guarantees v.grad holds ALL child contributions before v._backward(v.grad) runs!', { fs: 12.5, fill: C.ink });
  return svg(540, 258, o);
}

function threeForwardBudget() {
  let o = tx(270, 20, 'Arithmetic Budget of One Linear Layer hₖ = Wₖ h_{k−1}  (Wₖ ∈ ℝ^{d_k × d_{k−1}})', { fs: 15, fw: 700, fill: C.ink });
  o += card(16, 38, 160, 124, C.lblue, C.blue, '1. Forward Pass', 'hₖ = Wₖ h_{k−1}', '1 × (d_k d_{k−1}) mults', D.blue.dark);
  o += tx(96, 140, 'Saves h_{k−1} on tape', { fs: 12.5, fw: 700, fill: D.blue.dark });
  o += card(190, 38, 160, 124, C.lamber, C.amber, '2. Weight Gradient', '∇_{Wₖ} ℓ = h̄ₖ h_{k−1}ᵀ', '1 × (d_k d_{k−1}) mults', D.orange.dark);
  o += tx(270, 140, 'Outer product VJP', { fs: 12.5, fw: 700, fill: D.orange.dark });
  o += card(364, 38, 160, 124, C.lgreen, C.green, '3. Adjoint Pushdown', 'h̄_{k−1} = Wₖᵀ h̄ₖ', '1 × (d_k d_{k−1}) mults', D.green.dark);
  o += tx(444, 140, 'Feeds layer k − 1', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += card(16, 178, 508, 66, D.purple.tint, C.purple, 'Total Forward + Backward Cost = 3 × (d_k d_{k−1}) Multiplications (~3× Forward Pass!)', 'Baur–Strassen Theorem: Value + ALL partial derivatives never costs more than 3–5× forward evaluation', '', D.purple.dark);
  return svg(540, 258, o);
}

function opsCountGap() {
  let o = card(20, 14, 500, 66, C.lblue, C.blue, 'Benchmark: f(x) = (∑_{j=1}^{200} xⱼ²)²   (n = 200 Inputs, 2n + 1 = 401 Ops per Eval)', 'Measured primitive + and * operation counts in our Dual and Var engines:', '', D.blue.dark);
  o += card(20, 96, 238, 148, D.red.tint, D.red.base, 'Forward Mode (200 Dual Runs)', 'n(2n + 1) = 80,200 ops', '0 tape nodes stored (O(1) mem)', D.red.dark);
  o += tx(139, 214, '100× more arithmetic (n/2)!', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += card(282, 96, 238, 148, C.lgreen, C.green, 'Reverse Mode (1 Var Tape Sweep)', '2(2n + 1) = 802 ops (401+401)', '401 intermediate nodes on tape', D.green.dark);
  o += tx(401, 214, '100× faster, trades O(T) memory!', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function pearlmutterHvp() {
  let o = card(20, 14, 500, 72, D.red.tint, D.red.base, 'The Second-Order Memory Wall: Full Hessian H ∈ ℝ^{10⁶ × 10⁶}', '10¹² entries = 4 TB in float32: cannot store or invert', 'Newton–CG & curvature probes only need the Hessian–vector product H v ∈ ℝⁿ:', D.red.dark);
  o += card(20, 102, 500, 142, C.lgreen, C.green, "Pearlmutter's Trick (1994): Forward-Over-Reverse HVP in O(n) Time & Memory", 'H v = J_{∇ℓ}(x) v = d/dε ∇ℓ(x + ε v) at ε = 0', 'Seed x with Dual(xⱼ, vⱼ) and run reverse-mode grad(ℓ) once (~4–6× forward cost)!', D.green.dark);
  o += tx(270, 216, 'Carries dual tangents alongside backward closures without recording a second tape', { fs: 13.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function implicitDiffSolve() {
  let o = card(16, 16, 152, 108, C.lblue, C.blue, '1. Equilibrium Root', 'g(x*, θ) = 0', 'Any solver (K steps)', D.blue.dark);
  o += tx(92, 104, 'O(1) tape memory!', { fs: 12.5, fw: 700, fill: D.blue.dark });
  o += arrow(172, 70, 190, 70, C.blue);
  o += card(194, 16, 152, 108, C.lamber, C.amber, '2. Implicit Theorem', '(∂g/∂x)(∂x*/∂θ) + ∂g/∂θ = 0', '∂x*/∂θ = −(∂g/∂x)⁻¹ ∂g/∂θ', D.orange.dark);
  o += arrow(350, 70, 368, 70, C.green);
  o += card(372, 16, 152, 108, C.lgreen, C.green, '3. 1 Adjoint Solve', '(∂g/∂x)ᵀ a = u', 'uᵀ(∂x*/∂θ) = −aᵀ(∂g/∂θ)', D.green.dark);
  o += card(16, 140, 508, 104, D.purple.tint, C.purple, 'Why Adjoint Implicit Differentiation Beats Unrolling the Solver', '1. Memory is O(1) independent of solver iterations K (no K-step forward tape stored)', '2. Solves ONE d × d linear system (∂g/∂x)ᵀ a = u instead of p separate solves for p parameters θ!', D.purple.dark);
  return svg(540, 258, o);
}

function implicitFailures() {
  let o = tx(270, 20, 'Two Hypotheses Required for Implicit Differentiation on g(x*, θ) = x*² − θ = 0', { fs: 15, fw: 700, fill: C.ink });
  o += card(20, 38, 238, 204, D.red.tint, D.red.base, 'Failure 1: Singular ∂g/∂x', 'At θ = 0 (x* = 0): ∂g/∂x = 2x* = 0', 'Formula 1/(2x*) divides by zero!', D.red.dark);
  o += tx(139, 162, 'x*(θ) = √θ has vertical tangent', { fs: 13, fill: C.ink });
  o += tx(139, 188, 'at θ = 0 (not differentiable)', { fs: 13, fill: C.ink });
  o += tx(139, 218, 'Requires invertible ∂g/∂x!', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += card(282, 38, 238, 204, C.lamber, C.amber, 'Failure 2: Unconverged Solver', 'At θ = 4 (x* = 2, true dx*/dθ = 0.25):', 'Stop Newton at k = 1 (x̃₁ = 2.5):', D.orange.dark);
  o += tx(401, 162, 'Implicit formula gives 0.2000', { mono: true, fs: 12.5, fill: C.ink });
  o += tx(401, 188, 'Unrolled 1-step gives 0.5000', { mono: true, fs: 12.5, fill: C.ink });
  o += tx(401, 218, 'Both wrong until x̃ₖ → x* (k ≥ 3)!', { fs: 13, fw: 700, fill: D.orange.dark });
  return svg(540, 256, o);
}

function autodiffVsConditioning() {
  let o = card(20, 14, 238, 124, C.lgreen, C.green, 'What Autodiff Solves', '• Exact chain rule to machine eps', '• O(1) sweeps (~3× forward cost)', D.green.dark);
  o += tx(139, 114, 'No finite-difference truncation error', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += card(282, 14, 238, 124, D.red.tint, D.red.base, 'What Autodiff CANNOT Fix', 'If 50 layers each shrink by 0.5×:', 'True gradient is 0.5⁵⁰ ≈ 8.9 × 10⁻¹⁶!', D.red.dark);
  o += tx(401, 114, 'AD computes 8.9e-16 fast, training stalls!', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(20, 154, 500, 90, D.purple.tint, C.purple, 'Architectural Remedies Keep Singular Values of J_K ⋯ J₁ Near 1', '1. Variance-preserving initialization (Xavier / He fan-in scaling)', '2. Normalization layers (LayerNorm / BatchNorm)  &  3. Residual skip highways (I + J_l)', D.purple.dark);
  return svg(540, 258, o);
}

// ── §34.6 mdl-integral-calculus.md (5 diagrams) ──
function densityAreaParadox() {
  let o = card(20, 14, 500, 70, C.lblue, C.blue, 'Standard Normal Density p(z) = (1/√(2π)) e^{−z²/2}: Height vs. Area', 'p(0) ≈ 0.3989 is the peak height, yet P(Z = 0) = ∫₀⁰ p(z) dz = 0!', 'Probability is area under p(z): P(−1 ≤ Z ≤ 1) = ∫_{−1}¹ p(z) dz ≈ 0.6827', D.blue.dark);
  o += card(20, 100, 238, 144, C.lamber, C.amber, 'Question 1: Definite Integrals', 'How to sum continuous height f(x)', 'across [a, b] without infinite sums?', D.orange.dark);
  o += tx(139, 214, 'Answer: FTC Antiderivatives G(b)−G(a)', { fs: 12.5, fw: 700, fill: D.orange.dark });
  o += card(282, 100, 238, 144, C.lgreen, C.green, 'Question 2: Normalizer √(2π)', 'e^{−z²/2} has NO elementary G(z)!', 'Why is ∫_{−∞}^{∞} e^{−z²/2} dz = √(2π)?', D.green.dark);
  o += tx(401, 214, 'Answer: 2D Fubini + Polar Jacobian r', { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function ftcEvalPitfall() {
  let o = card(20, 14, 500, 88, C.lgreen, C.green, 'Fundamental Theorem Evaluation Rule (FTC Part 2)', 'If f is continuous on [a, b] and G′(x) = f(x) for ALL x ∈ [a, b], then ∫_{a}^{b} f(x) dx = G(b) − G(a)', 'Running example: ∫₀² x/(1+x²) dx = [½ log(1+x²)]₀² = ½ log 5 ≈ 0.804719', D.green.dark);
  o += card(20, 118, 500, 126, D.red.tint, D.red.base, 'Singular Antiderivative Trap: Why G′(x) = f(x) Must Hold at EVERY Point of [a, b]', 'False: ∫_{−1}¹ x⁻² dx = [−1/x]_{−1}¹ = (−1) − (+1) = −2 < 0,  though x⁻² > 0!', 'Why it fails: G(x) = −1/x blows up at x = 0 ∈ [−1, 1]; both ∫_{−1}⁰ x⁻² dx and ∫₀¹ x⁻² dx diverge to +∞!', D.red.dark);
  return svg(540, 258, o);
}

function improperPowerLaws() {
  let o = card(20, 14, 238, 118, C.lblue, C.blue, 'Unbounded Tail: ∫₁^{∞} x⁻ˢ dx', 'lim_{b→∞} (b^{1−s} − 1)/(1 − s)', 'Converges to 1/(s−1) iff s > 1!', D.blue.dark);
  o += tx(139, 108, 'Diverges for s ≤ 1 (harmonic 1/x)', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(282, 14, 238, 118, C.lamber, C.amber, 'Origin Blow-Up: ∫₀¹ x⁻ˢ dx', 'lim_{a→0⁺} (1 − a^{1−s})/(1 − s)', 'Converges to 1/(1−s) iff s < 1!', D.orange.dark);
  o += tx(401, 108, 'Diverges for s ≥ 1', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(20, 148, 500, 96, D.purple.tint, C.purple, 'Heavy-Tailed Probability Densities p(x) ∝ x⁻ˢ as x → ∞', '• Normalizable density (∫ p dx < ∞) requires s > 1   (e.g. Cauchy 1/(π(1+x²)) has s = 2)', '• Finite mean (∫ x p dx < ∞) requires s > 2   |   Finite variance (∫ x² p dx < ∞) requires s > 3', D.purple.dark);
  return svg(540, 258, o);
}

function ibpStein() {
  let o = card(20, 14, 500, 70, C.lblue, C.blue, 'Integration by Parts (Reversing the Product Rule (uv)′ = u′v + uv′)', '∫_{a}^{b} u(x) v′(x) dx  =  [u(x) v(x)]_{a}^{b}  −  ∫_{a}^{b} u′(x) v(x) dx', 'Shifts a derivative off v′ onto u′ at the cost of a boundary term [uv]_{a}^{b}', D.blue.dark);
  o += card(20, 100, 500, 144, C.lgreen, C.green, "Stein's Identity for the Gaussian Bell w(z) = e^{−z²/2}  (where w′(z) = −z w(z))", '∫_{−∞}^{∞} g′(z) w(z) dz = [g(z)w(z)]_{−∞}^{∞} − ∫_{−∞}^{∞} g(z)(−z w(z)) dz = ∫_{−∞}^{∞} z g(z) w(z) dz', 'For Z ~ N(0, 1): E[Z g(Z)] = E[g′(Z)]   ⇒   g(z)=z gives E[Z²]=1;  g(z)=z³ gives E[Z⁴]=3E[Z²]=3!', D.green.dark);
  return svg(540, 258, o);
}

function flowDensityTransform() {
  let o = card(16, 16, 152, 108, C.lblue, C.blue, 'Base Noise X ~ p_X', 'Isotropic Gaussian', 'Region Ω ⊆ ℝⁿ', D.blue.dark);
  o += arrow(172, 70, 190, 70, C.blue);
  o += card(194, 16, 152, 108, C.lamber, C.amber, 'Invertible Map φ', 'Y = φ(X)  (diffeo)', 'Vol scales by |det J_φ|', D.orange.dark);
  o += arrow(350, 70, 368, 70, C.green);
  o += card(372, 16, 152, 108, C.lgreen, C.green, 'Target Y ~ p_Y', 'p_Y(φ(x)) |det J_φ|   = p_X(x)', 'Mass conserved!', D.green.dark);
  o += card(16, 140, 508, 104, D.purple.tint, C.purple, 'Exact Normalizing Flow Log-Likelihood Formula', 'log p_Y(φ(x)) = log p_X(x) − log |det J_φ(x)|', 'Expanding volume (|det J_φ| > 1) dilutes density; triangular coupling layers compute log |det J_φ| in O(n)!', D.purple.dark);
  return svg(540, 258, o);
}

// ── §34.7 mdl-expectation-gradients.md (11 diagrams) ──
function paramDependentSampling() {
  let o = card(20, 14, 500, 72, D.red.tint, D.red.base, 'The Core Obstacle: Parameters θ Govern the Distribution p_θ(x) Itself', 'ℓ(θ) = E_{X~p_θ}[g(X)]  (VAE: z ~ N(μ_θ, σ_θ²); RL: a ~ π_θ(a|s))', 'Once a raw sample xᵢ ~ p_θ is drawn, it carries no computation graph back to θ!', D.red.dark);
  o += card(20, 102, 500, 104, C.lblue, C.blue, 'Running Analytical Benchmark: X ~ N(θ, 1)  (X = θ + Z,  Z ~ N(0, 1))', 'ℓ(θ) = E[X²] = E[(θ + Z)²] = θ² + 1     ⇒     True derivative: ℓ′(θ) = 2θ', 'At θ = 1.5: ℓ(1.5) = 3.25 and exact slope is ℓ′(1.5) = 3.000 — our yardstick for every Monte Carlo estimator!', D.blue.dark);
  return svg(540, 220, o);
}

function mcRmseScaling() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, 'Expectation as Definite Integral & Monte Carlo Average', 'E[g(X)] = ∫ g(x) p(x) dx ≈ μ̂ₙ = (1/n) ∑ᵢ g(xᵢ),  xᵢ ~ i.i.d. p', 'Unbiased (E[μ̂ₙ] = E[g(X)]) with variance Var(μ̂ₙ) = σ² / n  where σ² = Var(g(X))', D.blue.dark);
  o += card(20, 104, 238, 140, C.lgreen, C.green, 'Dimension-Free Rate: σ / √n', 'RMSE(μ̂ₙ) = σ / √n  in ANY ℝᵈ!', 'Beats O(N^{−2/d}) grid quadrature', D.green.dark);
  o += tx(139, 214, '100× more samples → 10× smaller err', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += card(282, 104, 238, 140, C.lamber, C.amber, 'Single-Sample SD σ Rules All', 'To cut RMSE by k× with fewer samples,', 'reduce single-sample variance σ²!', D.orange.dark);
  o += tx(401, 214, '3.6× smaller σ ⇒ 13× fewer samples!', { fs: 12.5, fw: 700, fill: D.orange.dark });
  return svg(540, 258, o);
}

function integralSwap() {
  let o = card(20, 14, 500, 72, D.purple.tint, C.purple, 'Differentiating Under the Integral Sign ("The Swap" on Fixed Support X)', 'd/dθ ∫_{X} f(x, θ) dx  =  ∫_{X} (∂f / ∂θ)(x, θ) dx', 'Exchanges limit of difference quotients (ϵ → 0) with limit of Riemann sums', D.purple.dark);
  o += card(20, 102, 238, 142, C.lblue, C.blue, '1. Bounded Closed Interval [a, b]', 'Support endpoints a, b fixed (no θ)', 'Joint continuity of ∂_{θ} f(x, θ)', D.blue.dark);
  o += tx(139, 214, 'Sufficient for exact swap!', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += card(282, 102, 238, 142, C.lgreen, C.green, '2. Unbounded Domain (e.g. ℝ)', 'Dominated Convergence bound:', '|∂_{θ} f(x, θ′)| ≤ M(x) with ∫ M dx < ∞', D.green.dark);
  o += tx(401, 214, 'Gaussian tails e^{−x²/2} always qualify!', { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function exponentialTilting() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, 'Log-Integral-Exp (Log-Partition Function): Continuous Softmax', 'd/dθ log ∫ e^{θx} p(x) dx = E_{X~p̃_θ}[X]', 'where p̃_θ(x) ∝ e^{θx} p(x) is the exponentially tilted density', D.blue.dark);
  o += card(20, 104, 238, 140, C.lamber, C.amber, `Discrete Log-Sum-Exp (${sec('sec_mdl-multivariable_calculus')})`, '∇_z log ∑ᵢ e^{zᵢ} = softmax(z)', 'Returns discrete probability vector p', D.orange.dark);
  o += tx(139, 214, 'H_LSE = Var_{I~p}', { mono: true, fs: 13, fw: 700, fill: D.orange.dark });
  o += card(282, 104, 238, 140, C.lgreen, C.green, 'Standard Normal Check p = N(0,1)', 'log E[e^{θZ}] = ½ θ²   (tilted p̃_θ = N(θ,1))', 'd/dθ (½ θ²) = θ = E_{N(θ,1)}[X]!', D.green.dark);
  o += tx(401, 214, 'Completing the square matches swap', { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function movingSupportSign() {
  let o = card(20, 14, 500, 68, C.lblue, C.blue, 'Moving-Support Pitfall: X ~ Uniform(0, θ) with Density p_θ(x) = 1/θ on [0, θ]', 'True mean: E[X] = ∫₀^{θ} (x/θ) dx = θ/2     ⇒     True derivative: d/dθ E[X] = +1/2', '', D.blue.dark);
  o += card(20, 98, 238, 146, D.red.tint, D.red.base, 'Naive Interior Swap Alone', '∫₀^{θ} ∂_{θ}(x/θ) dx = ∫₀^{θ} (−x/θ²) dx', '= −1/2   (WRONG SIGN!)', D.red.dark);
  o += tx(139, 214, 'Misses moving upper wall b(θ) = θ', { fs: 13, fw: 700, fill: D.red.dark });
  o += arrow(262, 171, 278, 171, C.green);
  o += card(282, 98, 238, 146, C.lgreen, C.green, 'Full Leibniz Rule', 'Interior (−½) + Boundary f(θ,θ) b′(θ)', '= −1/2 + (θ/θ)(1) = −1/2 + 1 = +1/2!', D.green.dark);
  o += tx(401, 214, 'Boundary sweep supplies +1!', { fs: 13.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function twoEscapeRoutes() {
  let o = card(20, 14, 500, 68, D.red.tint, D.red.base, 'Why Raw Swapped Integral ∫ g(x) ∇_θ p_θ(x) dx Cannot Be Sampled Directly', '∇_θ p_θ(x) is NOT a probability density: it takes negative values and integrates to ∇_θ 1 = 0!', '', D.red.dark);
  o += card(20, 98, 238, 146, C.lamber, C.amber, 'Route 1: Score Function', 'Multiply & divide by p_θ(x):', '∇_θ p_θ = p_θ · ∇_θ log p_θ', D.orange.dark);
  o += tx(139, 192, 'E_{p_θ}[g(X) ∇_θ log p_θ(X)]', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(139, 218, 'Works for discrete X & step g!', { fs: 12.5, fw: 700, fill: C.ink });
  o += card(282, 98, 238, 146, C.lgreen, C.green, 'Route 2: Pathwise (Reparam)', 'Substitute X = T_θ(z), z ~ q(z):', 'Removes θ from density q(z)!', D.green.dark);
  o += tx(401, 192, 'E_{z~q}[∇_θ g(T_θ(z))]', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(401, 218, 'Low variance when g is smooth!', { fs: 12.5, fw: 700, fill: C.ink });
  return svg(540, 258, o);
}

function scoreLogDeriv() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, 'The Log-Derivative Trick: ∇_θ p_θ(x) = p_θ(x) ∇_θ log p_θ(x)', '∇_θ E[g(X)] = ∫ g(x) p_θ(x) ∇_θ log p_θ(x) dx = E[g(X) s_θ(X)]', 'where s_θ(x) = ∇_θ log p_θ(x) is the score function (REINFORCE, Williams 1992)', D.blue.dark);
  o += card(20, 102, 238, 142, C.lgreen, C.green, 'Zero-Mean Score Property', 'Set g(x) ≡ 1 on fixed support:', 'E[s_θ(X)] = ∫ ∇_θ p_θ dx = ∇_θ 1 = 0', D.green.dark);
  o += tx(139, 214, 'For N(θ, 1): s_θ(X) = X − θ = Z', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += card(282, 102, 238, 142, C.lamber, C.amber, 'Never Differentiates g(x)!', 'Treats g(X) as a scalar weight', 'Handles black-box RL rewards', D.orange.dark);
  o += tx(401, 214, '& discrete actions a ~ π_θ(a|s)', { fs: 13, fw: 700, fill: D.orange.dark });
  return svg(540, 258, o);
}

function baselineVariance() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, 'Control-Variate Baseline Subtraction: (g(X) − b) ∇_θ log p_θ(X)', 'E[(g(X) − b) s_θ(X)] = E[g(X) s_θ(X)] − b · 0', 'Unbiased for ANY constant b independent of X because E[s_θ(X)] = 0!', D.blue.dark);
  o += card(20, 104, 238, 140, D.purple.tint, C.purple, 'Optimal Constant Baseline b*', 'Min Var((g(X) − b) s_θ(X)):', 'b* = E[g(X) s_θ(X)²] / E[s_θ(X)²]', D.purple.dark);
  o += tx(139, 214, 'For g(X)=X² on N(θ,1): b* = θ² + 3', { mono: true, fs: 12, fw: 700, fill: D.purple.dark });
  o += card(282, 104, 238, 140, C.lgreen, C.green, 'Variance Drop on E[X²]', 'Plain REINFORCE: θ⁴ + 14θ² + 15', 'With baseline b*: 8θ² + 6', D.green.dark);
  o += tx(401, 214, 'Cuts O(θ⁴) explosion down to O(θ²)!', { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function pathwiseReparam() {
  let o = card(16, 16, 152, 108, C.lblue, C.blue, '1. Base Noise z ~ q', 'z ~ N(0, I) or Unif(0,1)', 'No θ in density q(z)!', D.blue.dark);
  o += arrow(172, 70, 190, 70, C.blue);
  o += card(194, 16, 152, 108, C.lamber, C.amber, '2. Deterministic Map', 'X = T_θ(z) = μ + σ ⊙ z', 'Differentiable path!', D.orange.dark);
  o += arrow(350, 70, 368, 70, C.green);
  o += card(372, 16, 152, 108, C.lgreen, C.green, '3. Chain Rule VJP', '∇_θ g(T_θ(z))', '= J_{T_θ}ᵀ ∇_x g(X)', D.green.dark);
  o += card(16, 140, 508, 104, D.purple.tint, C.purple, 'Pathwise (Reparameterization) Trick (Kingma & Welling, 2014)', '∇_θ E_{X~p_θ}[g(X)] = ∇_θ ∫ g(T_θ(z)) q(z) dz = E_{z~q}[∇_θ g(T_θ(z))]', 'Uses direct gradient ∇_x g(X) of the loss surface — standard engine of VAEs!', D.purple.dark);
  return svg(540, 258, o);
}

function stepLossFailure() {
  let o = card(20, 14, 500, 70, C.lblue, C.blue, 'Step-Function Benchmark: g(x) = 1(x > 0) with X = θ + Z (Z ~ N(0, 1)) at θ = 1.5', 'True expectation: E[1(θ + Z > 0)] = P(Z > −θ) = Φ(θ)   ⇒   True slope: φ(1.5) ≈ 0.1295', '', D.blue.dark);
  o += card(20, 100, 238, 144, D.red.tint, D.red.base, 'Pathwise Fails (Biased = 0!)', 'd/dθ 1(θ + z > 0) = 0 for all z ≠ −θ', 'Jump occurs at single point z = −θ', D.red.dark);
  o += tx(139, 214, 'Returns 0 with probability 1 ≠ 0.1295!', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(282, 100, 238, 144, C.lgreen, C.green, 'Score Function Succeeds!', '1(θ + Z > 0) · Z  never diffs g(x)', 'E[1(Z > −θ) Z] = ∫_{−θ}^{∞} z φ(z) dz', D.green.dark);
  o += tx(401, 214, '= [−φ(z)]_{−θ}^{∞} = φ(θ) ≈ 0.1295!', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function estimatorVarianceBars() {
  let o = tx(270, 20, 'Single-Sample Standard Deviation σ on ℓ(θ) = E[X²] at θ = 1.5 (True Slope = 3.0)', { fs: 14.5, fw: 700, fill: C.ink });
  o += card(20, 40, 152, 134, C.lgreen, C.green, '1. Pathwise 2X', 'Var = 4 (const!)', 'σ = 2.000', D.green.dark);
  o += tx(96, 150, '1× baseline samples', { fs: 12.5, fw: 700, fill: D.green.dark });
  o += card(194, 40, 152, 134, C.lamber, C.amber, '2. Score + Baseline', 'Var = 8θ² + 6 = 24', 'σ = 4.899', D.orange.dark);
  o += tx(270, 150, '6.0× more samples', { fs: 12.5, fw: 700, fill: D.orange.dark });
  o += card(368, 40, 152, 134, D.red.tint, D.red.base, '3. Plain REINFORCE', 'Var = θ⁴+14θ²+15', 'σ = 7.181', D.red.dark);
  o += tx(444, 150, '12.9× more samples!', { fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(20, 188, 500, 58, D.purple.tint, C.purple, 'Why Pathwise Wins on Smooth Losses: |g′(X)| = O(θ) vs. |g(X)| = O(θ²)', 'No baseline closes the gap: 8θ² + 6 > 4 for every θ', '', D.purple.dark);
  return svg(540, 258, o);
}

export const diagrams = {
  'ch34-calc-two-questions': twoQuestions,
  'ch34-calc-three-pillars': threePillars,
  'ch34-calc-roadmap-three-parts': roadmapThreeParts,
  'ch34-calc-running-benchmarks': runningBenchmarks,
  'ch34-calc-logistic-curvature-gap': logisticCurvatureGap,
  'ch34-calc-little-o-linear': littleOLinear,
  'ch34-calc-elementary-rules': elementaryRules,
  'ch34-calc-sum-product-chain': sumProductChain,
  'ch34-calc-softplus-sigmoid-logit': softplusSigmoidLogit,
  'ch34-calc-lagrange-remainder': lagrangeRemainder,
  'ch34-calc-joint-perturbation': jointPerturbation,
  'ch34-calc-lse-partials': lsePartials,
  'ch34-calc-multivar-gradient': multivarGradient,
  'ch34-calc-ridge-counterexample': ridgeCounterexample,
  'ch34-calc-directional-rays': directionalRays,
  'ch34-calc-level-set-orthogonality': levelSetOrthogonality,
  'ch34-calc-critical-disc-bound': criticalDiscBound,
  'ch34-calc-hessian-lse': hessianLse,
  'ch34-calc-clairaut-schwarz': clairautSchwarz,
  'ch34-calc-second-order-taylor': secondOrderTaylor,
  'ch34-calc-highd-saddles': highdSaddles,
  'ch34-calc-graph-sharing': graphSharing,
  'ch34-calc-parallel-branches': parallelBranches,
  'ch34-calc-edge-vs-total': edgeVsTotal,
  'ch34-calc-path-sum-rule': pathSumRule,
  'ch34-calc-resnet-paths': resnetPaths,
  'ch34-calc-forward-edge-table': forwardEdgeTable,
  'ch34-calc-forward-mode-trace': forwardModeTrace,
  'ch34-calc-backward-mode-trace': backwardModeTrace,
  'ch34-calc-grad-check-precision': gradCheckPrecision,
  'ch34-calc-three-deriv-shapes': threeDerivShapes,
  'ch34-calc-jacobian-grid': jacobianGrid,
  'ch34-calc-shape-conventions': shapeConventions,
  'ch34-calc-jacobian-chain-proof': jacobianChainProof,
  'ch34-calc-deep-jacobian-norms': deepJacobianNorms,
  'ch34-calc-differential-recipe': differentialRecipe,
  'ch34-calc-trace-move-rule': traceMoveRule,
  'ch34-calc-worked-matrix-grads': workedMatrixGrads,
  'ch34-calc-three-sanity-checks': threeSanityChecks,
  'ch34-calc-layer-vjp-duality': layerVjpDuality,
  'ch34-calc-batched-matmul-vjp': batchedMatmulVjp,
  'ch34-calc-softmax-jacobian-var': softmaxJacobianVar,
  'ch34-calc-softmax-ce-cancel': softmaxCeCancel,
  'ch34-calc-perturbation-vs-ad': perturbationVsAd,
  'ch34-calc-dual-number-algebra': dualNumberAlgebra,
  'ch34-calc-dual-induction': dualInduction,
  'ch34-calc-fwd-lse-passes': fwdLsePasses,
  'ch34-calc-jvp-vector-tangent': jvpVectorTangent,
  'ch34-calc-var-tape-anatomy': varTapeAnatomy,
  'ch34-calc-three-forward-budget': threeForwardBudget,
  'ch34-calc-ops-count-gap': opsCountGap,
  'ch34-calc-pearlmutter-hvp': pearlmutterHvp,
  'ch34-calc-implicit-diff-solve': implicitDiffSolve,
  'ch34-calc-implicit-failures': implicitFailures,
  'ch34-calc-autodiff-vs-conditioning': autodiffVsConditioning,
  'ch34-calc-density-area-paradox': densityAreaParadox,
  'ch34-calc-ftc-eval-pitfall': ftcEvalPitfall,
  'ch34-calc-improper-power-laws': improperPowerLaws,
  'ch34-calc-ibp-stein': ibpStein,
  'ch34-calc-flow-density-transform': flowDensityTransform,
  'ch34-calc-param-dependent-sampling': paramDependentSampling,
  'ch34-calc-mc-rmse-scaling': mcRmseScaling,
  'ch34-calc-integral-swap': integralSwap,
  'ch34-calc-exponential-tilting': exponentialTilting,
  'ch34-calc-moving-support-sign': movingSupportSign,
  'ch34-calc-two-escape-routes': twoEscapeRoutes,
  'ch34-calc-score-log-deriv': scoreLogDeriv,
  'ch34-calc-baseline-variance': baselineVariance,
  'ch34-calc-pathwise-reparam': pathwiseReparam,
  'ch34-calc-step-loss-failure': stepLossFailure,
  'ch34-calc-estimator-variance-bars': estimatorVarianceBars,
};
