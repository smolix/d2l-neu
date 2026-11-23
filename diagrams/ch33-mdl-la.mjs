// Slide diagrams for ch33-mdl-la.mjs (§33 Linear Algebra)
import { C, TOKENS, grid, tx as rawTx, arrow, svg as rawSvg } from "./engine.mjs";
import { sec, num } from "./sections.mjs";
import { textWidth } from "./mdl-textwidth.mjs";

const D = TOKENS.accents;
const N = TOKENS.neutrals;
const BLUE = [C.lblue, C.blue, false, C.ink];
const GREEN = [C.lgreen, C.green, false, C.ink];
const AMBER = [C.lamber, C.amber, false, C.ink];
const RED = [D.red.tint, D.red.base, false, D.red.dark];
const PURPLE = [D.purple.tint, C.purple, false, D.purple.dark];

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
function motivationAxes() {
  let o = card(18, 16, 504, 54, D.purple.tint, C.purple, `Geometric Linear Algebra in ℝᵈ (${sec('chap_mdl-linear-algebra')})`, 'Right coordinates turn matrix products into transparent geometry', '', D.purple.dark);
  o += card(18, 90, 156, 156, C.lblue, C.blue, '1. Subspace Axes', 'QᵀQ = Iₖ', 'Lengths & Projections', D.blue.dark);
  o += tx(96, 184, 'P = QQᵀ', { mono: true, fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(96, 212, `${sec('sec_mdl-geometry-linear-algebraic-ops')}–${num('sec_mdl-linear-maps')}`, { mono: true, fs: 13, fill: C.muted });
  o += card(192, 90, 156, 156, C.lamber, C.amber, '2. Eigen-Axes', 'Avᵢ = λᵢvᵢ', 'Repeated Square Maps', D.orange.dark);
  o += tx(270, 184, 'Aᵏx = ∑ cᵢλᵢᵏvᵢ', { mono: true, fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(270, 212, `${sec('sec_mdl-eigendecompositions')}–${num('sec_mdl-spectra-in-practice')}`, { mono: true, fs: 13, fill: C.muted });
  o += card(366, 90, 156, 156, C.lgreen, C.green, '3. Singular Axes', 'Avᵢ = σᵢuᵢ', 'Any m × n Layer', D.green.dark);
  o += tx(444, 184, 'A = UΣVᵀ', { mono: true, fs: 14, fw: 700, fill: D.green.dark });
  o += tx(444, 212, `${sec('sec_mdl-svd-low-rank')}`, { mono: true, fs: 13, fill: C.muted });
  return svg(540, 262, o);
}

function roadmapTwoStage() {
  let o = tx(270, 20, 'Stage 1: Single Map in Standard Coordinates', { fs: 15.5, fw: 700, fill: D.blue.dark });
  o += card(24, 36, 230, 74, C.lblue, C.blue, `${sec('sec_mdl-geometry-linear-algebraic-ops')} Vectors & Projections`, 'P = QQᵀ · A = QR · 1/√d', 'Attention & Least Squares', D.blue.dark);
  o += arrow(258, 73, 282, 73, C.blue);
  o += card(286, 36, 230, 74, C.lblue, C.blue, `${sec('sec_mdl-linear-maps')} Linear Maps & Volume`, 'rank(A) + dim ker(A) = n', 'Inverses & det(AB)', D.blue.dark);
  o += hline(24, 516, 128);
  o += tx(270, 148, 'Stage 2: Natural Coordinates Tailored to the Matrix', { fs: 15.5, fw: 700, fill: D.purple.dark });
  o += card(16, 166, 156, 86, C.lamber, C.amber, `${sec('sec_mdl-eigendecompositions')} Eigen`, 'A = VΛV⁻¹', 'Sym: VΛVᵀ', D.orange.dark);
  o += arrow(175, 209, 189, 209, C.amber);
  o += card(192, 166, 156, 86, D.red.tint, D.red.base, `${sec('sec_mdl-spectra-in-practice')} Spectra`, 'Power Iter', 'Non-Normal Mᵏ', D.red.dark);
  o += arrow(351, 209, 365, 209, C.green);
  o += card(368, 166, 156, 86, C.lgreen, C.green, `${sec('sec_mdl-svd-low-rank')} SVD`, 'A = UΣVᵀ', 'LoRA · PCA · κ', D.green.dark);
  return svg(540, 268, o);
}

function threeTheorems() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, '1. Orthogonal Projection (P = QQᵀ)', 'y − Py ⊥ S   ⇒   ∥y − Py∥ ≤ ∥y − s∥ for all s ∈ S', 'Closest point & right-triangle least squares', D.blue.dark);
  o += card(20, 98, 500, 72, C.lamber, C.amber, '2. Spectral Theorem (A = Aᵀ ⇒ A = VΛVᵀ)', 'xᵀAx = ∑ λᵢ (vᵢᵀx)²   with real λᵢ and orthonormal V', 'Positive definiteness, Rayleigh bounds & curvature bowls', D.orange.dark);
  o += card(20, 182, 500, 72, C.lgreen, C.green, '3. Eckart–Young–Mirsky (Aₖ = ∑_{i≤k} σᵢuᵢvᵢᵀ)', 'min_{rank(B)≤k} ∥A − B∥₂ = σₖ₊₁   and   ∥A − Aₖ∥_{F}² = ∑_{i>k} σᵢ²', 'Optimal low-rank compression, PCA & LoRA adapters', D.green.dark);
  return svg(540, 268, o);
}

function runningBenchmarks() {
  let o = card(16, 16, 160, 108, C.lblue, C.blue, 'Running Matrix A', '[1, 2; −1, 3]', 'det = 5 · λ = 2 ± i', D.blue.dark);
  o += tx(96, 104, 'σ = 3.62, 1.38', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += card(190, 16, 160, 108, C.lamber, C.amber, 'Defective Shear T', '[1, 1; 0, 1]', 'λ = 1 (g=1 < a=2)', D.orange.dark);
  o += tx(270, 104, 'Tᵏ=[1,k;0,1]  σ=φ,1/φ', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += card(364, 16, 160, 108, D.red.tint, D.red.base, 'Singular Matrix B', '[2, 1; 4, 2]', 'rank = 1 · det = 0', D.red.dark);
  o += tx(444, 104, 'ker = span{(1, −2)}', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(16, 140, 244, 104, D.purple.tint, C.purple, 'Fashion-MNIST (ℝ⁷⁸⁴)', '70,000 images (28×28)', 'Random cos ~ 1/√784 ≈ 0.036', D.purple.dark);
  o += tx(138, 224, '92% nearest-mean hyperplane', { fs: 13, fw: 700, fill: D.purple.dark });
  o += card(280, 140, 244, 104, C.lgreen, C.green, 'Cat Photo (400 × 500)', '200,000 pixels · σᵢ ∝ i@@', 'Rank-20 SVD: 9% params, 8% err', D.green.dark)
    .replace('i@@<', 'i<tspan baseline-shift="super" font-size="70%">−1.1</tspan><');
  o += tx(402, 224, 'Eckart–Young vs Gaussian bulk', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 260, o);
}

// ── §33.1 mdl-geometry-linear-algebraic-ops.md (10 diagrams) ──
function pythagorasSplit() {
  let o = card(20, 16, 500, 68, C.lblue, C.blue, 'Orthogonal Decomposition: v = proj_{w}(v) + r', 'proj_{w}(v) = ((v·w)/(w·w)) w = (2, 1);   r = v − proj_{w}(v) = (−1, 2)', 'Orthogonality check: r · w = (−1)(2) + (2)(1) = 0', D.blue.dark);
  o += card(20, 102, 146, 86, D.purple.tint, C.purple, 'Hypotenuse ∥v∥²', 'v = (1, 3)', '1² + 3² = 10', D.purple.dark);
  o += tx(184, 145, '=', { fs: 24, fw: 700, fill: C.ink });
  o += card(202, 102, 146, 86, C.lgreen, C.green, 'Shadow ∥proj∥²', 'proj = (2, 1)', '2² + 1² = 5', D.green.dark);
  o += tx(362, 145, '+', { fs: 24, fw: 700, fill: C.ink });
  o += card(376, 102, 144, 86, C.lamber, C.amber, 'Residual ∥r∥²', 'r = (−1, 2)', '(−1)² + 2² = 5', D.orange.dark);
  o += box(20, 204, 500, 44, N.panel, C.gray);
  o += tx(270, 226, 'Signed shadow length = (v · w) / ∥w∥ = 5 / √5 = +√5', { mono: true, fs: 14, fw: 700, fill: C.ink });
  return svg(540, 262, o);
}

function cauchySchwarz() {
  let o = card(20, 16, 500, 74, C.lblue, C.blue, 'Right-Triangle Proof of Cauchy–Schwarz in ℝᵈ', '∥v∥² = ∥proj_{w}(v)∥² + ∥r∥²  ≥  ∥proj_{w}(v)∥² = (v · w)² / ∥w∥²', 'Dropping ∥r∥² ≥ 0 and taking square roots gives |v · w| ≤ ∥v∥ ∥w∥', D.blue.dark);
  o += card(20, 106, 240, 134, C.lgreen, C.green, 'Strict Inequality (r ≠ 0)', '|v · w| < ∥v∥ ∥w∥', 'cos θ = (v·w)/(∥v∥∥w∥) ∈ (−1, 1)', D.green.dark);
  o += tx(140, 214, 'θ ∈ (0, π) non-collinear', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += card(280, 106, 240, 134, C.lamber, C.amber, 'Equality Case (r = 0)', '|v · w| = ∥v∥ ∥w∥', 'v = t* w  (collinear vectors)', D.orange.dark);
  o += tx(400, 214, 'θ = 0 (cos=1) or θ = π (cos=−1)', { fs: 13.5, fw: 700, fill: D.orange.dark });
  return svg(540, 256, o);
}

function angleClip() {
  let o = card(20, 16, 500, 70, C.lblue, C.blue, 'Exact Arithmetic: Cauchy–Schwarz Bounds Cosine', 'cos θ = (v·w)/(∥v∥ ∥w∥) ∈ [−1, 1]  ⇒  θ = arccos(cos θ) ∈ [0, π]', 'Worked pair v=(1,3), w=(2,1): cos θ = 5/(√10√5) = 1/√2 ⇒ θ = π/4', D.blue.dark);
  o += card(20, 104, 232, 132, D.red.tint, D.red.base, 'Float64 Roundoff Hazard', 'w = 3v, v = (10⁸+1, 3, 7)', 'cos = 1 + 2.2×10⁻¹⁶ > 1', D.red.dark);
  o += tx(136, 210, 'np.arccos(cos) → NaN!', { mono: true, fs: 13.5, fw: 700, fill: D.red.dark });
  o += arrow(256, 170, 282, 170, C.green);
  o += card(286, 104, 234, 132, C.lgreen, C.green, 'Safe Clipping Guard', 'np.clip(cos, -1.0, 1.0)', 'Clamps 1 + 2.2e-16 → 1.0', D.green.dark);
  o += tx(403, 210, 'np.arccos(1.0) → 0.0 rad', { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  return svg(540, 252, o);
}

function projectorQQT() {
  let o = card(20, 16, 236, 106, C.lblue, C.blue, 'Orthonormal Basis Q ∈ ℝⁿˣᵏ', 'Columns q₁, …, qₖ span S', 'QᵀQ = Iₖ  (qᵢ · qⱼ = δᵢⱼ)', D.blue.dark);
  o += arrow(260, 69, 280, 69, C.purple);
  o += card(284, 16, 236, 106, D.purple.tint, C.purple, 'Subspace Projector P = QQᵀ', 'Px = Q(Qᵀx) ∈ S', 'Qᵀ(x − Px) = 0  (r ⊥ S)', D.purple.dark);
  o += card(20, 140, 236, 104, C.lgreen, C.green, 'Idempotent: P² = P', 'P² = Q(QᵀQ)Qᵀ = QQᵀ = P', 'Projecting a 2nd time does nothing', D.green.dark);
  o += card(284, 140, 236, 104, C.lamber, C.amber, 'Symmetric: Pᵀ = P', '(QQᵀ)ᵀ = QQᵀ = P', 'Orthogonal (closest) vs oblique', D.orange.dark);
  return svg(540, 260, o);
}

function normalVsQr() {
  let o = card(20, 14, 500, 52, D.purple.tint, C.purple, 'Least Squares: min_{x} ∥Ax − y∥²  (A = [1, t], y = (6, 0, 0))', 'Both routes: x̂ = (5, −3), Py = (5, 2, −1), r = (1, −2, 1)', '', D.purple.dark);
  o += card(20, 82, 238, 160, D.red.tint, D.red.base, 'Route 1: Normal Equations', 'AᵀA x̂ = Aᵀy', '[3, 3; 3, 5] x̂ = [6; 0]', D.red.dark);
  o += tx(139, 190, 'Condition number SQUARES:', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(139, 216, 'κ(AᵀA) = κ(A)²  (roundoff loss)', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(282, 82, 238, 160, C.lgreen, C.green, 'Route 2: Orthonormal QR', 'A = QR   ⇒   R x̂ = Qᵀy', 'Py = QQᵀy  (QᵀQ = I)', D.green.dark);
  o += tx(401, 190, 'Preserves conditioning:', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(401, 216, 'κ(R) = κ(A)  (numerically stable)', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function gramSchmidtQr() {
  let o = card(16, 16, 152, 108, C.lblue, C.blue, 'Step 1: Normalize a₁', 'a₁ = (3, 4), ∥a₁∥ = 5', 'q₁ = a₁/5 = (0.6, 0.8)', D.blue.dark);
  o += arrow(172, 70, 190, 70, C.blue);
  o += card(194, 16, 152, 108, C.lamber, C.amber, 'Step 2: Subtract proj', 'a₂ = (1, 0), q₁·a₂ = 0.6', 'v₂ = a₂ − 0.6q₁', D.orange.dark);
  o += arrow(350, 70, 368, 70, C.amber);
  o += card(372, 16, 152, 108, C.lgreen, C.green, 'Step 3: Normalize v₂', 'v₂ = (0.64, −0.48)', 'q₂ = v₂/0.8 = (0.8, −0.6)', D.green.dark);
  o += card(16, 142, 508, 104, D.purple.tint, C.purple, 'Matrix Form: A = QR  (Upper-Triangular R)', '[a₁  a₂] = [q₁  q₂] [5,  0.6;  0,  0.8]     where rᵢⱼ = qᵢ · aⱼ', 'r₂₁ = 0 because a₁ depends only on q₁ (each aⱼ ∈ span{q₁, …, qⱼ})', D.purple.dark);
  return svg(540, 260, o);
}

function cosineScaleInvariance() {
  let o = card(20, 16, 238, 118, C.lblue, C.blue, 'Cosine Similarity (Normalized)', 'cos θ = (v · w) / (∥v∥ ∥w∥)', 'cos(c v, w) = cos(v, w) for c > 0', D.blue.dark);
  o += tx(139, 110, 'Image & 10% dimmed copy: cos = 1', { fs: 13, fw: 700, fill: D.blue.dark });
  o += card(282, 16, 238, 118, C.lamber, C.amber, 'Scaled Dot Product (Attention)', 's(q, k) = (q · k) / √d', 'Retains lengths ∥q∥ and ∥k∥', D.orange.dark);
  o += tx(401, 110, 'Magnitude modulates confidence', { fs: 13, fw: 700, fill: D.orange.dark });
  o += card(20, 150, 500, 94, C.lgreen, C.green, 'Where Each Metric Is Used in Deep Learning', 'Cosine: SimCLR / CLIP contrastive losses & k-NN vector retrieval', 'Scaled Dot Product: Transformer self-attention QKᵀ / √d', D.green.dark);
  return svg(540, 258, o);
}

function isotropicVsData() {
  let o = tx(270, 20, 'Cosine Similarity of Random Pairs in ℝ⁷⁸⁴', { fs: 16, fw: 700, fill: C.ink });
  o += card(16, 38, 160, 134, C.lblue, C.blue, '1. Isotropic Sphere', 'u, v ~ Unif(S⁷⁸³)', 'Mean = 0.00', D.blue.dark);
  o += tx(96, 132, 'Std = 1/√784 = 0.036', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(96, 154, 'Full d = 784 space', { fs: 13, fill: C.muted });
  o += card(190, 38, 160, 134, C.lamber, C.amber, '2. Raw Fashion-MNIST', 'Pixels x ∈ [0, 1]⁷⁸⁴', 'Mean = +0.60', D.orange.dark);
  o += tx(270, 132, 'Std = 0.18', { mono: true, fs: 13, fw: 700, fill: D.orange.dark });
  o += tx(270, 154, 'Positive orthant cone', { fs: 13, fill: C.muted });
  o += card(364, 38, 160, 134, C.lgreen, C.green, '3. Centered Images', 'x − x̄_train', 'Mean = 0.00', D.green.dark);
  o += tx(444, 132, 'Std = 0.34 (10× wider)', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += tx(444, 154, 'd_eff = 1/0.34² ≈ 8.6!', { fs: 13, fw: 700, fill: D.green.dark });
  o += card(16, 188, 508, 58, D.purple.tint, C.purple, 'Key Takeaway: Effective Low Dimensionality', 'Centered 784-pixel images concentrate near an ~9-D subspace!', '', D.purple.dark);
  return svg(540, 260, o);
}

function attentionSqrtD() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, 'Why q · k Grows as √d  (qᵢ, kᵢ ~ i.i.d. mean 0, var 1)', 'Var(q · k) = ∑ᵢ₌₁ᵈ Var(qᵢ kᵢ) = d   ⇒   Std(q · k) = √d', 'Geometric view: ∥q∥ ≈ √d, ∥k∥ ≈ √d, cos θ ≈ 1/√d  ⇒  |q·k| ≈ √d', D.blue.dark);
  o += card(20, 102, 236, 140, D.red.tint, D.red.base, 'Unscaled q · k (d = 64)', 'Std = √64 = 8.0', '1σ gap → logits (8, 0)', D.red.dark);
  o += tx(138, 198, 'softmax: (0.9997, 0.0003)', { mono: true, fs: 13, fw: 700, fill: D.red.dark });
  o += tx(138, 222, 'Saturated! Gradients vanish', { fs: 13, fw: 700, fill: D.red.dark });
  o += arrow(260, 172, 280, 172, C.green);
  o += card(284, 102, 236, 140, C.lgreen, C.green, 'Scaled q · k / √d (d = 64)', 'Divide by √64 = 8 → Std = 1.0', '1σ gap → logits (1, 0)', D.green.dark);
  o += tx(402, 198, 'softmax: (0.731, 0.269)', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(402, 222, 'Healthy gradient flow!', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function nearestMeanBisector() {
  let o = card(20, 16, 500, 74, C.lblue, C.blue, 'Nearest-Mean Rule: ∥x − x̄₁∥² < ∥x − x̄₀∥²', '∥x∥² − 2x·x̄₁ + ∥x̄₁∥²  <  ∥x∥² − 2x·x̄₀ + ∥x̄₀∥²   (∥x∥² cancels!)', 'Reduces to a single hyperplane test: w · x + b > 0', D.blue.dark);
  o += card(20, 106, 238, 134, C.lgreen, C.green, 'Normal Vector w', 'w = x̄₁ − x̄₀', 'Points from mean 0 to mean 1', D.green.dark);
  o += tx(139, 212, '1 dot product per test image', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += card(282, 106, 238, 134, C.lamber, C.amber, 'Midpoint Offset b', 'm = ½(x̄₀ + x̄₁)', 'b = −w · m = −½(∥x̄₁∥² − ∥x̄₀∥²)', D.orange.dark);
  o += tx(401, 212, 'Bisects the segment [x̄₀, x̄₁]', { fs: 13.5, fw: 700, fill: D.orange.dark });
  return svg(540, 256, o);
}

// ── §33.2 mdl-linear-maps.md (10 diagrams) ──
function rowVsColView() {
  let o = card(20, 14, 238, 112, C.lblue, C.blue, 'Row View: m Hyperplane Tests', '(Ax)ᵢ = aᵢᵀ · x', '1 input → m signed scores', D.blue.dark);
  o += tx(139, 104, 'Hides global space collapse', { fs: 13, fill: C.muted });
  o += card(282, 14, 238, 112, C.lgreen, C.green, 'Column View: Axis Images', 'Ax = x₁ a₁ + ⋯ + xₙ aₙ', 'aⱼ = A eⱼ  (where axis j lands)', D.green.dark);
  o += tx(401, 104, 'Reveals rank, kernel & volume', { fs: 13, fw: 700, fill: D.green.dark });
  o += card(20, 142, 500, 102, C.lamber, C.amber, 'Bottleneck Layer Example: W ∈ ℝ²⁵⁶ˣ⁷⁸⁴ on Fashion-MNIST', '784 input dimensions  →  at most 256 output dimensions (rank ≤ 256)', 'At least 784 − 256 = 528 independent directions collapse to 0!', D.orange.dark);
  return svg(540, 258, o);
}

function orthogonalIsometry() {
  let o = card(20, 14, 500, 72, C.lgreen, C.green, 'Orthogonal Matrix Q ∈ ℝⁿˣⁿ: Orthonormal Columns (QᵀQ = I)', '(Qx) · (Qy) = xᵀQᵀQy = xᵀy = x · y   ⇒   ∥Qx∥ = ∥x∥ and ∠ preserved', 'Square Q satisfies Q⁻¹ = Qᵀ  (so QQᵀ = I as well: rows are orthonormal)', D.green.dark);
  o += card(20, 102, 238, 140, C.lblue, C.blue, '2D Orthogonal Maps', 'Rotations R_φ (det = +1)', 'Reflections R_φ diag(1,−1) (det = −1)', D.blue.dark);
  o += tx(139, 212, 'Rigid motion: all cosines fixed', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += card(282, 102, 238, 140, D.red.tint, D.red.base, 'Counterexample: diag(1, 10)', 'Columns ⊥, but lengths 1 ≠ 10!', 'u=(1,1) ⊥ v=(1,−1) (90°)', D.red.dark);
  o += tx(401, 212, 'Images (1,10) & (1,−10): 169°!', { fs: 13.5, fw: 700, fill: D.red.dark });
  return svg(540, 258, o);
}

function einsumUnification() {
  let o = tx(270, 20, 'Einstein Summation: Sum Over Repeated Indices Omitted After ->', { fs: 15.5, fw: 700, fill: D.purple.dark });
  o += card(18, 38, 242, 66, C.lblue, C.blue, '1. Dot Product: v · w', "einsum('i,i->', v, w)", '∑ᵢ vᵢ wᵢ  (i summed)', D.blue.dark);
  o += card(280, 38, 242, 66, C.lblue, C.blue, '2. Matrix–Vector: Av', "einsum('ij,j->i', A, v)", '∑ⱼ aᵢⱼ vⱼ  (i kept, j summed)', D.blue.dark);
  o += card(18, 116, 242, 66, C.lgreen, C.green, '3. Matrix Product: AB', "einsum('ij,jk->ik', A, B)", '∑ⱼ aᵢⱼ bⱼₖ  (i,k kept, j summed)', D.green.dark);
  o += card(280, 116, 242, 66, C.lamber, C.amber, '4. Trace: tr(A)', "einsum('ii->', A)", '∑ᵢ aᵢᵢ  (diagonal i summed)', D.orange.dark);
  o += box(18, 196, 504, 48, D.purple.tint, C.purple);
  o += tx(270, 220, 'Cyclic Trace Proof: tr(AB) = aᵢⱼ bⱼᵢ = bⱼᵢ aᵢⱼ = tr(BA)  ⇒  tr(VAV⁻¹) = tr(A)', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  return svg(540, 258, o);
}

function einsumAttention() {
  let o = card(20, 14, 500, 88, C.lblue, C.blue, "Multi-Head Attention Contraction: einsum('bhij,bhtj->bhit', Q, K)", 'Q: (b=2, h=3, i=4, j=8) · K: (b=2, h=3, t=5, j=8)   →   S: (2, 3, 4, 5)', 'Kept axes: b (batch), h (head), i (query), t (key)  |  Summed axis: j (width=8)', D.blue.dark);
  o += card(20, 118, 238, 124, D.red.tint, D.red.base, 'Bad Order: (AB)v', 'Matrix–Matrix first: O(n³)', 'n = 1000  ⇒  10⁹ mult-adds', D.red.dark);
  o += tx(139, 214, '500× slower!', { fs: 14, fw: 700, fill: D.red.dark });
  o += arrow(262, 180, 278, 180, C.green);
  o += card(282, 118, 238, 124, C.lgreen, C.green, 'Good Order: A(Bv)', 'Two Matrix–Vectors: 2n²', 'n = 1000  ⇒  2×10⁶ mult-adds', D.green.dark);
  o += tx(401, 214, 'With v on the right: go right-to-left!', { fs: 13.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function rankNullitySplit() {
  let o = card(20, 14, 500, 68, D.purple.tint, C.purple, 'Rank–Nullity Theorem for Any A ∈ ℝᵐˣⁿ', 'rank(A) + dim ker(A) = n   (surviving + collapsed = input dimension)', '', D.purple.dark);
  o += card(20, 98, 186, 138, C.lblue, C.blue, 'Input Space ℝⁿ', 'Basis w₁..wₖ of ker(A)', '+ u₁..uₙ₋ₖ complement', D.blue.dark);
  o += tx(113, 208, 'dim = k + (n − k) = n', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += arrow(212, 134, 294, 134, C.green);
  o += tx(253, 118, 'Auᵢ indep', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  o += arrow(212, 198, 294, 198, D.red.base);
  o += tx(253, 182, 'Awⱼ = 0', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += card(300, 98, 220, 64, C.lgreen, C.green, 'col(A) ⊆ ℝᵐ (Survives)', 'dim col(A) = rank(A) = n − k', '', D.green.dark);
  o += card(300, 172, 220, 64, D.red.tint, D.red.base, 'Collapsed to 0 ∈ ℝᵐ', 'dim ker(A) = k', '', D.red.dark);
  return svg(540, 252, o);
}

function bottleneckGh() {
  let o = tx(270, 20, 'Rank-r Factorization A = G H: Row Rank = Column Rank = r', { fs: 15.5, fw: 700, fill: C.ink });
  o += box(32, 54, 110, 136, C.lblue, C.blue);
  o += tx(87, 110, 'A', { fs: 22, fw: 700, fill: D.blue.dark });
  o += tx(87, 136, 'm × n', { mono: true, fs: 14, fw: 600, fill: C.ink });
  o += tx(87, 212, 'mn entries', { fs: 13.5, fill: C.muted });
  o += tx(168, 122, '=', { fs: 24, fw: 700, fill: C.ink });
  o += box(194, 54, 62, 136, C.lgreen, C.green);
  o += tx(225, 110, 'G', { fs: 20, fw: 700, fill: D.green.dark });
  o += tx(225, 136, 'm × r', { mono: true, fs: 13.5, fw: 600, fill: C.ink });
  o += tx(225, 212, 'col basis', { fs: 13.5, fill: D.green.dark });
  o += tx(272, 122, '·', { fs: 28, fw: 700, fill: C.ink });
  o += box(288, 92, 136, 60, C.lamber, C.amber);
  o += tx(356, 114, 'H', { fs: 20, fw: 700, fill: D.orange.dark });
  o += tx(356, 136, 'r × n', { mono: true, fs: 13.5, fw: 600, fill: C.ink });
  o += tx(356, 212, 'row basis', { fs: 13.5, fill: D.orange.dark });
  o += card(438, 54, 86, 136, D.purple.tint, C.purple, 'Width r', 'x ∈ ℝⁿ', '↓ Hx ∈ ℝʳ', D.purple.dark);
  o += tx(481, 166, 'r(m+n)', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  return svg(540, 240, o);
}

function inverse2x2Wheel() {
  let o = card(20, 14, 500, 66, C.lblue, C.blue, 'Explicit 2×2 Inverse & Five Equivalent Conditions for Square A ∈ ℝⁿˣⁿ', 'A⁻¹ = (1/(ps − qr)) [s, −q; −r, p]     (for A = [1, 2; −1, 3]: det = 5, A⁻¹ = ⅕[3, −2; 1, 1])', '', D.blue.dark);
  o += card(16, 96, 156, 66, C.lgreen, C.green, '1. Invertible', 'A⁻¹A = AA⁻¹ = I', '', D.green.dark);
  o += card(192, 96, 156, 66, C.lgreen, C.green, '2. Indep. Columns', '∑ xⱼaⱼ = 0 ⇒ x = 0', '', D.green.dark);
  o += card(368, 96, 156, 66, C.lgreen, C.green, '3. Trivial Kernel', 'ker(A) = {0}', '', D.green.dark);
  o += card(96, 178, 160, 66, C.lamber, C.amber, '4. Full Rank', 'rank(A) = n (onto ℝⁿ)', '', D.orange.dark);
  o += card(284, 178, 160, 66, D.purple.tint, C.purple, '5. Nonzero Volume', 'det(A) ≠ 0', '', D.purple.dark);
  return svg(540, 258, o);
}

function detAxioms() {
  let o = tx(270, 20, 'Three Geometric Volume Axioms Uniquely Fix det(A)', { fs: 16, fw: 700, fill: C.ink });
  o += card(16, 38, 160, 114, C.lblue, C.blue, '1. Multilinear', 'Linear in each aⱼ', 'det(…,ca+b,…) =', D.blue.dark);
  o += tx(96, 128, 'c det(a) + det(b)', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += card(190, 38, 160, 114, C.lamber, C.amber, '2. Alternating', 'aᵢ = aⱼ ⇒ det = 0', 'Swap flips sign:', D.orange.dark);
  o += tx(270, 120, 'det(…,u,w,…)', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(270, 139, '= −det(…,w,u,…)', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += card(364, 38, 160, 114, C.lgreen, C.green, '3. Normalized', 'Unit cube volume = 1', 'det(Iₙ) = 1', D.green.dark);
  o += tx(444, 128, 'sgn(σ) = ±1', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += card(16, 168, 508, 76, D.purple.tint, C.purple, 'Unique Consequence: The n! Permutation Formula', 'det(A) = ∑_{σ ∈ Sₙ} sgn(σ) a_{σ(1),1} a_{σ(2),2} ⋯ a_{σ(n),n}', 'Any multilinear alternating f satisfies f(A) = f(I) det(A)', D.purple.dark);
  return svg(540, 258, o);
}

function triangularFlowDet() {
  let o = card(20, 16, 240, 154, C.lblue, C.blue, 'Upper-Triangular U', 'det(U) = u₁₁ u₂₂ ⋯ uₙₙ', '', D.blue.dark);
  const U = [['u₁₁', '*', '*'], ['0', 'u₂₂', '*'], ['0', '0', 'u₃₃']];
  o += grid(U, 91, 66, 30, 4, { fs: 12, fill: (r, c) => r === c ? GREEN : (r > c ? [N.panel, N.hairline, false, C.muted] : BLUE) });
  o += card(280, 16, 240, 154, C.lamber, C.amber, 'Block-Triangular Matrix', 'det = det(D₁) · det(D₂)', '', D.orange.dark);
  const B = [['D₁', '*'], ['0', 'D₂']];
  o += grid(B, 362, 76, 36, 4, { fs: 13, fill: (r, c) => r === c ? AMBER : [N.panel, N.hairline, false, C.muted] });
  o += card(20, 182, 500, 64, C.lgreen, C.green, 'Normalizing Flows: log |det J| = ∑ᵢ₌₁ⁿ log |Jᵢᵢ| in O(n) Time', 'Avoids the ≈ n³/3 LU determinant at every training step!', '', D.green.dark);
  return svg(540, 260, o);
}

function detMultiplicativity() {
  let o = card(16, 16, 144, 88, C.lblue, C.blue, 'Region Ω (Vol V)', 'Unit cube V = 1', '', D.blue.dark);
  o += arrow(164, 60, 190, 60, C.blue);
  o += tx(177, 44, 'B', { mono: true, fs: 14, fw: 700, fill: D.blue.dark });
  o += card(194, 16, 152, 88, C.lamber, C.amber, 'Image B(Ω)', 'Vol = |det B| · V', '', D.orange.dark);
  o += arrow(350, 60, 376, 60, C.amber);
  o += tx(363, 44, 'A', { mono: true, fs: 14, fw: 700, fill: D.orange.dark });
  o += card(380, 16, 144, 88, C.lgreen, C.green, 'Image A(B(Ω))', 'Vol = |det A| ·', '', D.green.dark);
  o += tx(452, 76, '|det B| · V', { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  o += card(16, 120, 246, 60, D.purple.tint, C.purple, '1. Scalar Commutativity', 'det(AB) = det(A)det(B) = det(BA)', '', D.purple.dark);
  o += card(278, 120, 246, 60, D.purple.tint, C.purple, '2. Similarity Invariance', 'det(VAV⁻¹) = det(A)', '', D.purple.dark);
  o += card(16, 190, 246, 60, C.lblue, C.blue, '3. Inverse Volume', 'det(A⁻¹) = 1 / det(A)', '', D.blue.dark);
  o += card(278, 190, 246, 60, C.lgreen, C.green, '4. Orthogonal Maps', 'det(QᵀQ)=1 ⇒ det(Q) = ±1', '', D.green.dark);
  return svg(540, 262, o);
}

// ── §33.3 mdl-eigendecomposition.md (10 diagrams) ──
function schurQrLoop() {
  let o = card(20, 14, 500, 66, D.red.tint, D.red.base, 'Why Eigensolvers Never Solve p(λ) = det(A − λI) = 0 Directly', 'No radical formula for n ≥ 5 (Abel), and polynomial roots are hyper-sensitive to roundoff', '', D.red.dark);
  o += card(20, 96, 146, 94, C.lblue, C.blue, '1. Factor QR', 'Aₖ − μI = QₖRₖ', 'Qₖ orthogonal', D.blue.dark);
  o += arrow(170, 143, 194, 143, C.blue);
  o += card(198, 96, 152, 94, C.lamber, C.amber, '2. Reverse Multiply', 'Aₖ₊₁ = RₖQₖ + μI', '= Qₖᵀ Aₖ Qₖ (similar!)', D.orange.dark);
  o += arrow(354, 143, 378, 143, C.green);
  o += card(382, 96, 138, 94, C.lgreen, C.green, '3. Schur Form U', 'Subdiagonal → 0', 'diag(U) = λ₁..λₙ', D.green.dark);
  o += box(20, 204, 500, 44, D.purple.tint, C.purple);
  o += tx(270, 226, 'det(U − λI) = ∏ (uᵢᵢ − λ) = 0  ⇒  Eigenvalues sit directly on the diagonal!', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  return svg(540, 260, o);
}

function diagPowers() {
  let o = tx(270, 20, 'Diagonalization A = VΛV⁻¹ Decouples Matrix Powers Aᵏx', { fs: 16, fw: 700, fill: C.ink });
  o += card(16, 38, 152, 94, C.lblue, C.blue, '1. To Eigenbasis', 'c = V⁻¹x', 'x = ∑ cᵢ vᵢ', D.blue.dark);
  o += arrow(172, 85, 190, 85, C.blue);
  o += card(194, 38, 152, 94, C.lamber, C.amber, '2. Scalar Powers', 'Λᵏc = (λᵢᵏ cᵢ)', 'V⁻¹V = I cancels!', D.orange.dark);
  o += arrow(350, 85, 368, 85, C.amber);
  o += card(372, 38, 152, 94, C.lgreen, C.green, '3. Back to ℝⁿ', 'Aᵏx = V(Λᵏc)', '= ∑ cᵢ λᵢᵏ vᵢ', D.green.dark);
  o += card(16, 150, 508, 96, D.purple.tint, C.purple, 'Two-Page Surfer Trajectory: Pᵏp₀ = ⅓(1, 2) + ⅔(¼)ᵏ(1, −1)', 'λ₁ = 1 keeps stationary state π = (⅓, ⅔);  λ₂ = ¼ shrinks error 4× per click!', 'At k = 50: (¼)⁵⁰ < 10⁻³⁰  (converged to machine precision)', D.purple.dark);
  return svg(540, 260, o);
}

function similarityInvariants() {
  let o = card(20, 14, 500, 74, D.purple.tint, C.purple, 'Similar Matrices A′ = BAB⁻¹ Share the Characteristic Polynomial', 'det(BAB⁻¹ − λI) = det(B) det(A − λI) det(B⁻¹) = ∏(λᵢ − λ)', 'Same linear map in different coordinates ⇒ identical eigenvalues & multiplicities', D.purple.dark);
  o += card(20, 104, 238, 140, C.lblue, C.blue, 'Trace = Sum of λᵢ', 'tr(A) = tr(VΛV⁻¹) = tr(Λ)', 'tr(A) = λ₁ + λ₂ + ⋯ + λₙ', D.blue.dark);
  o += tx(139, 216, 'Surfer P: ½ + ¾ = 5/4 = 1 + ¼', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += card(282, 104, 238, 140, C.lgreen, C.green, 'Determinant = Product of λᵢ', 'det(A) = det(V)det(Λ)det(V⁻¹)', 'det(A) = λ₁ · λ₂ ⋯ λₙ', D.green.dark);
  o += tx(401, 216, 'Surfer P: ⅜ − ⅛ = 1/4 = 1 · ¼', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function multiplicityGap() {
  let o = card(20, 14, 500, 86, C.lblue, C.blue, 'Algebraic vs. Geometric Multiplicity: 1 ≤ g(λ) ≤ a(λ)', 'a(λ) = root multiplicity in det(A − λI)     g(λ) = dim ker(A − λI) (# indep. eigenvectors)', 'Block triangular proof: p(t) = (λ − t)ᵍ det(Y − tI)  ⇒  g(λ) ≤ a(λ)', D.blue.dark);
  o += card(20, 116, 238, 142, C.lgreen, C.green, 'Diagonalizable (g = a ∀λ)', 'Full eigenbasis V ∈ ℂⁿˣⁿ', 'Always holds if λ₁..λₙ distinct!', D.green.dark);
  o += tx(139, 228, 'A = VΛV⁻¹  (pure λᵢᵏ powers)', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += card(282, 116, 238, 142, D.red.tint, D.red.base, 'Defective (g < a for some λ)', 'Missing eigenvectors!', 'Shear T = [1,1; 0,1]: a(1)=2, g(1)=1', D.red.dark);
  o += tx(401, 228, 'Needs Jordan blocks Jₘ(λ)', { mono: true, fs: 13, fw: 700, fill: D.red.dark });
  return svg(540, 272, o);
}

function jordanTransient() {
  let o = card(20, 14, 500, 72, C.lamber, C.amber, 'Jordan Block Power Formula: J₂(λ) = λI + N  (with Nilpotent N² = 0)', 'J₂(λ)ᵏ = (λI + N)ᵏ = λᵏ I + k λᵏ⁻¹ N = [λᵏ,  k λᵏ⁻¹;  0,  λᵏ]', 'Defective eigenvalue multiplies exponential λᵏ by polynomial k!', D.orange.dark);
  o += card(20, 102, 238, 142, D.red.tint, D.red.base, 'Transient Growth (λ = 0.99)', 'Corner entry: k · (0.99)ᵏ⁻¹', 'k=1: 1.0  →  k=100: 37.0 (peak!)', D.red.dark);
  o += tx(139, 214, 'Grows 37× despite |λ| < 1!', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += card(282, 102, 238, 142, C.lgreen, C.green, 'Asymptotic Decay (k ≫ 100)', 'Peak at k ≈ −1 / ln|λ| = 100', 'k=200: 27.1  →  k=800: 0.26', D.green.dark);
  o += tx(401, 214, 'Exponential 0.99ᵏ wins as k→∞', { fs: 13.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function complexRotationBlock() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, 'Complex Eigenpair in a Real Matrix: A(u + iv) = (a + ib)(u + iv)', 'Real & imaginary parts: Au = au − bv   and   Av = bu + av', 'In basis B = [v, u], A acts as a 2D rotation–scaling block:', D.blue.dark);
  o += card(20, 104, 238, 140, D.purple.tint, C.purple, 'Canonical 2×2 Block', 'B⁻¹AB = [a, −b; b, a] = r R_θ', 'r = √(a²+b²),  θ = arg(a+ib)', D.purple.dark);
  o += tx(139, 214, 'Scales by r, rotates by θ', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += card(282, 104, 238, 140, C.lamber, C.amber, 'Running Matrix A = [1,2;−1,3]', 'λ = 2 ± i  (r = √5 ≈ 2.24)', 'B⁻¹AB = [2, −1; 1, 2], θ = 26.6°', D.orange.dark);
  o += tx(401, 214, 'Elliptical spiral in ℝ² (period 13.6)', { fs: 13, fw: 700, fill: D.orange.dark });
  return svg(540, 258, o);
}

function realJordanCases() {
  let o = tx(270, 20, 'Real Jordan Form: Three Elementary Building Blocks of Aᵏ', { fs: 16, fw: 700, fill: C.ink });
  o += card(16, 38, 160, 198, C.lgreen, C.green, '1. Distinct Real λ', '[λ₁, 0; 0, λ₂]', 'Pure axis scaling', D.green.dark);
  o += tx(96, 152, 'Aᵏ → λᵢᵏ along vᵢ', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(96, 184, 'Ex: Surfer P (1, ¼)', { fs: 13, fill: C.ink });
  o += tx(96, 210, 'Symmetric S (3, 1)', { fs: 13, fill: C.muted });
  o += card(190, 38, 160, 198, C.lamber, C.amber, '2. Defective Jordan', '[λ, 1; 0, λ]', 'g(λ) < a(λ) shortage', D.orange.dark);
  o += tx(270, 152, 'Aᵏ → k λᵏ⁻¹  polynomial', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(270, 184, 'Ex: Shear T = J₂(1)', { fs: 13, fill: C.ink });
  o += tx(270, 210, 'J₂(0.99) peaks at 37', { fs: 13, fill: C.muted });
  o += card(364, 38, 160, 198, C.lblue, C.blue, '3. Complex Pair', '[a,−b; b,a] = r R_θ', 'λ = a ± ib (b ≠ 0)', D.blue.dark);
  o += tx(444, 152, 'Aᵏ → rᵏ R_{kθ}', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(444, 184, 'Ex: Running A (2±i)', { fs: 13, fill: C.ink });
  o += tx(444, 210, 'r = √5, θ = 26.6°', { fs: 13, fill: C.muted });
  return svg(540, 252, o);
}

function spectralTheoremProof() {
  let o = card(16, 14, 508, 54, D.purple.tint, C.purple, 'Spectral Theorem: Real Symmetric A = Aᵀ  ⇒  A = VΛVᵀ (VᵀV = I)', 'Eliminates both complex eigenvalues and defective Jordan blocks', '', D.purple.dark);
  o += card(16, 82, 160, 160, C.lblue, C.blue, 'Part (i): Real λ', 'v*Av = λ∥v∥²', '= (Av)*v = λ̄∥v∥²', D.blue.dark);
  o += tx(96, 192, '⇒  λ = λ̄ ∈ ℝ', { mono: true, fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(96, 218, 'Real eigenvectors', { fs: 13, fill: C.muted });
  o += card(190, 82, 160, 160, C.lamber, C.amber, 'Part (ii): Ortho vᵢ', 'λ(uᵀv) = (Au)ᵀv', '= uᵀ(Av) = μ(uᵀv)', D.orange.dark);
  o += tx(270, 192, 'λ ≠ μ ⇒ uᵀv = 0', { mono: true, fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(270, 218, 'Distinct eigenspaces ⊥', { fs: 13, fill: C.muted });
  o += card(364, 82, 160, 160, C.lgreen, C.green, 'Part (iii): Induction', 'x ⊥ v₁ ⇒ Ax ⊥ v₁', 'C = QᵀAQ symmetric', D.green.dark);
  o += tx(444, 192, 'V = [v₁, Q V′]', { mono: true, fs: 14, fw: 700, fill: D.green.dark });
  o += tx(444, 218, 'Full orthonormal basis', { fs: 13, fill: C.muted });
  return svg(540, 256, o);
}

function rayleighQuotient() {
  let o = card(20, 14, 500, 74, C.lblue, C.blue, 'Rayleigh Quotient R(x) = (xᵀAx)/(xᵀx) = ∑ᵢ₌₁ⁿ λᵢ yᵢ²  (∥y∥² = 1)', 'Convex weighted average of eigenvalues:  λ_min ≤ R(x) ≤ λ_max', 'Attains λ_max at top eigenvector v₁ and λ_min at bottom eigenvector vₙ', D.blue.dark);
  o += box(40, 114, 460, 32, N.panel, C.gray, 16);
  o += box(110, 114, 320, 32, C.lgreen, C.green, 16);
  o += tx(110, 102, 'λₙ = λ_min (at vₙ)', { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(270, 130, 'R(x) ∈ [λ_min, λ_max]', { mono: true, fs: 14, fw: 700, fill: D.green.dark });
  o += tx(430, 102, 'λ₁ = λ_max (at v₁)', { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  o += card(20, 164, 500, 80, C.lamber, C.amber, 'Two Core Deep Learning Consequences', '1. Spectral norm: max_{∥x∥=1} ∥Ax∥² = λ_max(AᵀA) = σ₁²', '2. Gradient descent on ½xᵀAx converges iff |1 − ηλᵢ| < 1  ⇔  0 < η < 2/λ_max', D.orange.dark);
  return svg(540, 258, o);
}

function marchenkoPastur() {
  let o = card(20, 14, 500, 70, C.lblue, C.blue, 'Sample Covariance of Pure Noise: C = (1/n) XᵀX  (Xᵢⱼ ~ N(0, 1), c = d/n)', 'Eigenvalues spread over [(1 − √c)², (1 + √c)²], not at 1!', 'For c = d/n = 1000/4000 = 1/4: noise bulk spans [0.25, 2.25]', D.blue.dark);
  o += card(20, 100, 310, 142, C.lamber, C.amber, 'Marchenko–Pastur Noise Bulk', 'λ ∈ [(1 − √c)², (1 + √c)²] = [0.25, 2.25]', 'Indistinguishable from random noise!', D.orange.dark);
  o += tx(175, 214, 'Even at 4 samples/dim, max noise λ = 2.25', { fs: 13, fw: 700, fill: D.orange.dark });
  o += card(346, 100, 174, 142, C.lgreen, C.green, 'Signal Outliers', 'λ > (1 + √c)²', 'Escapes above 2.25', D.green.dark);
  o += tx(433, 214, 'True data structure!', { fs: 13.5, fw: 700, fill: D.green.dark });
  return svg(540, 256, o);
}

// ── §33.4 mdl-spectra-in-practice.md (9 diagrams) ──
function threeChallenges() {
  let o = tx(270, 20, 'Three Real-World Challenges Where Textbook Diagonalization Fails', { fs: 15.5, fw: 700, fill: C.ink });
  o += card(16, 38, 160, 198, C.lblue, C.blue, '1. Billion-Row Scale', 'O(n³) eig impossible', 'Only Av products', D.blue.dark);
  o += tx(96, 152, 'Web graph / LLM', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += tx(96, 182, 'Tool: Gershgorin &', { fs: 13, fill: C.ink });
  o += tx(96, 206, 'Power Iteration', { fs: 13, fw: 700, fill: D.blue.dark });
  o += card(190, 38, 160, 198, C.lamber, C.amber, '2. Finite Steps k', 'ρ(M) < 1 only gives', 'k → ∞ decay!', D.orange.dark);
  o += tx(270, 152, 'M = [0.9, 4; 0, 0.8]', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(270, 182, 'Grows 10.8× at k=6', { fs: 13, fill: C.ink });
  o += tx(270, 206, 'Tool: ∥Aᵏ∥₂ & SVD', { fs: 13, fw: 700, fill: D.orange.dark });
  o += card(364, 38, 160, 198, D.red.tint, D.red.base, '3. Changing Factors', 'BPTT multiplies T', 'different Jacobians', D.red.dark);
  o += tx(444, 152, 'D₁ᵀ D₂ᵀ ⋯ D_Tᵀ', { mono: true, fs: 13, fw: 700, fill: D.red.dark });
  o += tx(444, 182, 'ρ(Dₜ)=0 can explode!', { fs: 13, fill: C.ink });
  o += tx(444, 206, 'Tool: Lyapunov μ', { fs: 13, fw: 700, fill: D.red.dark });
  return svg(540, 252, o);
}

function pagerankDamping() {
  let o = card(16, 16, 236, 114, C.lblue, C.blue, '3-Page Web Graph & Matrix P', '1 → 2,   2 → {1, 3},   3 → dead end', 'P = [0, ½, ⅓;  1, 0, ⅓;  0, ½, ⅓]', D.blue.dark);
  o += tx(134, 110, 'Stationary π = (0.3, 0.4, 0.3)', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });
  o += arrow(256, 73, 280, 73, C.purple);
  o += card(284, 16, 240, 114, D.purple.tint, C.purple, 'Damped Google Matrix G', 'G = αP + (1−α)(1/n) 11ᵀ', 'α = 0.85 (follow link) + 0.15 (jump)', D.purple.dark);
  o += tx(404, 110, 'Strictly positive ⇒ Perron–Frobenius', { fs: 12.5, fw: 700, fill: D.purple.dark });
  o += card(16, 146, 508, 98, C.lgreen, C.green, 'Haveliwala–Kamvar Bound: |λ₂(G)| ≤ α = 0.85 on ANY Web Graph', 'Matrix-free step: Gp = α(Pp) + ((1−α)/n)1  (1 sparse matvec per sweep)', '50 sweeps cut error by 0.85⁵⁰ ≈ 3 × 10⁻⁴ regardless of graph size n!', D.green.dark);
  return svg(540, 258, o);
}

function gelfandVsFinite() {
  let o = card(20, 14, 500, 68, D.purple.tint, C.purple, "Gelfand's Formula Governs Only the Distant Limit k → ∞", 'ρ(A) = maxᵢ |λᵢ| = lim_{k→∞} ∥Aᵏ∥^{1/k}', 'and ρ(A)ᵏ ≤ ∥Aᵏ∥₂ for all k ≥ 1', D.purple.dark);
  o += card(20, 98, 238, 144, C.lamber, C.amber, 'Counterexample 1: Shear T', 'T = [1, 1; 0, 1]  has ρ(T) = 1', 'Yet Tᵏ = [1, k; 0, 1]', D.orange.dark);
  o += tx(139, 212, '∥Tᵏ∥₂ ≈ k → ∞ (unbounded!)', { mono: true, fs: 13, fw: 700, fill: D.orange.dark });
  o += card(282, 98, 238, 144, D.red.tint, D.red.base, 'Counterexample 2: J₂(0.99)', 'ρ(J₂(0.99)) = 0.99 < 1', 'Yet ∥J₂(0.99)ᵏ∥₂ ≈ 37 at k=100', D.red.dark);
  o += tx(401, 212, 'Transient blowup before decay!', { fs: 13.5, fw: 700, fill: D.red.dark });
  return svg(540, 258, o);
}

function normalVsNonnormal() {
  let o = card(20, 16, 238, 134, C.lgreen, C.green, 'Normal Matrices (AᵀA = AAᵀ)', 'Unitary eigenbasis A = UΛU*', 'Symmetric, orthogonal, r R_θ', D.green.dark);
  o += tx(139, 122, '∥Aᵏ∥₂ = ρ(A)ᵏ  for ALL k ≥ 1', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += card(282, 16, 238, 134, D.red.tint, D.red.base, 'Non-Normal (AᵀA ≠ AAᵀ)', 'Non-orthogonal V: κ₂(V) > 1', 'A = VΛV⁻¹  (nearly parallel vᵢ)', D.red.dark);
  o += tx(401, 122, '∥Aᵏ∥₂ ≤ κ₂(V) ρ(A)ᵏ', { mono: true, fs: 13.5, fw: 700, fill: D.red.dark });
  o += card(20, 166, 500, 80, C.lamber, C.amber, 'Geometric Cause of Transient Amplification: Unraveling Cancellation', 'At ∠(v₁, v₂) = 1.4° (κ₂(V) = 80), e₂ = 40v₁ + v₂ cancels huge parts;', 'they decay at 0.9ᵏ vs 0.8ᵏ, cancellation breaks and ∥M⁶e₂∥ grows 10.8×!', D.orange.dark);
  return svg(540, 260, o);
}

function bpttNilpotent() {
  let o = card(20, 14, 500, 82, C.lblue, C.blue, 'Backpropagation Through Time (BPTT) Multiplies Changing Jacobians', '∇_{h₀} L = D₁ᵀ D₂ᵀ ⋯ D_Tᵀ ∇_{h_T} L     where Dₜ = ∂hₜ / ∂hₜ₋₁ changes at every step t', '', D.blue.dark);
  o += card(16, 112, 152, 86, C.lamber, C.amber, 'Factor D₁', '[0, 2; 0, 0]', 'λ = 0,0  (ρ = 0)', D.orange.dark);
  o += tx(178, 155, '·', { fs: 28, fw: 700, fill: C.ink });
  o += card(190, 112, 152, 86, C.lamber, C.amber, 'Factor D₂', '[0, 0; 2, 0]', 'λ = 0,0  (ρ = 0)', D.orange.dark);
  o += tx(354, 155, '=', { fs: 24, fw: 700, fill: C.ink });
  o += card(368, 112, 156, 86, D.red.tint, D.red.base, 'Product D₂D₁', '[0, 0; 0, 4]', 'ρ(D₂D₁) = 4!', D.red.dark);
  o += box(16, 212, 508, 48, D.red.tint, D.red.base);
  o += tx(270, 236, 'Both factors have ρ = 0 (D₁² = D₂² = 0), yet alternating (D₂D₁)^{T/2} explodes as 4^{T/2}!', { fs: 13.5, fw: 700, fill: D.red.dark });
  return svg(540, 274, o);
}

function stabilityHierarchy() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, '1. Spectral Radius: ρ(A) = maxᵢ |λᵢ| = lim_{k→∞} ∥Aᵏ∥^{1/k}', 'Governs: Asymptotic (k → ∞) rate of powers of ONE FIXED matrix A', 'Blind to finite-step non-normal transient growth and changing Jacobian factors', D.blue.dark);
  o += card(20, 98, 500, 72, C.lgreen, C.green, '2. Spectral Norm: ∥A∥₂ = σ₁(A) = max_{∥x∥=1} ∥Ax∥', 'Governs: worst-case stretch, ∥D_T⋯D₁∥₂ ≤ ∏ₜ ∥Dₜ∥₂', 'DL Remedies: Orthogonal initialization (∥W∥₂ = 1), spectral normalization, gradient clipping', D.green.dark);
  o += card(20, 182, 500, 72, D.purple.tint, C.purple, '3. Top Lyapunov Exponent: μ = lim_{T→∞} (1/T) log ∥D_T⋯D₁ x₀∥', 'Governs: long-run growth (μ > 0) / decay (μ < 0), CHANGING Dₜ', 'DL Remedies: Gated recurrences (LSTM / GRU) pass selected directions with factors ≈ 1', D.purple.dark);
  return svg(540, 268, o);
}

function randomMatrixGap() {
  let o = card(20, 14, 238, 118, C.lblue, C.blue, 'Circular Law: Eigenvalues', 'Aᵢⱼ ~ N(0, 1)  (n × n Ginibre)', 'λᵢ / √n fill unit disc in ℂ', D.blue.dark);
  o += tx(139, 108, 'ρ(A) ≈ 1.0 √n', { mono: true, fs: 14, fw: 700, fill: D.blue.dark });
  o += card(282, 14, 238, 118, C.lamber, C.amber, 'Marchenko–Pastur: Stretch', 'Max eigenvalue of (1/n)AᵀA → 4', 'σ₁(A) = √(4n) = 2√n', D.orange.dark);
  o += tx(401, 108, '∥A∥₂ ≈ 2.0 √n  (2× larger!)', { mono: true, fs: 14, fw: 700, fill: D.orange.dark });
  o += card(20, 148, 500, 96, D.purple.tint, C.purple, 'The Deep Product Paradox at Fan-In Initialization Dᵢⱼ ~ N(0, 1/n)', '1 layer preserves mean squared length: E[∥Dx∥²] = ∥x∥²  (log E[∥Dx∥²] = 0)', "By Jensen's inequality on concave log: μ = ½ E[log ∥Dx∥²] ≈ −1/(2n) < 0  (deep products shrink!)", D.purple.dark);
  return svg(540, 258, o);
}

function eigenToSvdBridge() {
  let o = card(16, 16, 238, 224, D.red.tint, D.red.base, 'Where Eigenvalues Fall Short', '', '', D.red.dark);
  o += tx(135, 66, '1. Rectangular m × n layers', { fs: 13.5, fw: 600, fill: C.ink });
  o += tx(135, 92, '2. Defective matrices (T)', { fs: 13.5, fw: 600, fill: C.ink });
  o += tx(135, 118, '3. Non-normal 10× growth', { fs: 13.5, fw: 600, fill: C.ink });
  o += tx(135, 144, '4. Random matrix 2× gap', { fs: 13.5, fw: 600, fill: C.ink });
  o += tx(135, 184, 'Single basis V cannot stay', { fs: 13, fill: C.muted });
  o += tx(135, 206, 'orthonormal for input & output', { fs: 13, fill: C.muted });
  o += arrow(258, 128, 282, 128, C.green);
  o += card(286, 16, 238, 224, C.lgreen, C.green, `The SVD Solution (${sec('sec_mdl-svd-low-rank')})`, 'Two orthonormal bases V, U', 'A vᵢ = σᵢ uᵢ  (σ₁ ≥ σ₂ ≥ ⋯ ≥ 0)', D.green.dark);
  o += tx(405, 118, '• Exists for ANY m × n matrix', { fs: 13.5, fw: 600, fill: C.ink });
  o += tx(405, 144, '• Measures 1-step stretch σ₁', { fs: 13.5, fw: 600, fill: C.ink });
  o += tx(405, 170, '• Gives optimal rank-k Aₖ,', { fs: 13.5, fw: 600, fill: C.ink });
  o += tx(405, 194, 'pseudoinverse A⁺ & κ(A)', { fs: 13.5, fw: 600, fill: C.ink });
  return svg(540, 256, o);
}

function spectraSummary() {
  let o = card(16, 14, 246, 70, C.lblue, C.blue, '1. Gershgorin Discs', '|λ − aᵢᵢ| ≤ rᵢ = ∑_{j≠i} |aᵢⱼ|', 'O(n²) entry bound; diag dominance', D.blue.dark);
  o += card(278, 14, 246, 70, C.lblue, C.blue, '2. Power Iteration', 'v_{k+1} = Avₖ / ∥Avₖ∥', 'Dir rate |λ₂/λ₁|ᵏ · Rayleigh |λ₂/λ₁|²ᵏ', D.blue.dark);
  o += card(16, 96, 246, 70, D.purple.tint, C.purple, '3. Damped PageRank', 'G = αP + (1−α)(1/n)11ᵀ', 'α = 0.85 bounds |λ₂(G)| ≤ 0.85', D.purple.dark);
  o += card(278, 96, 246, 70, C.lamber, C.amber, '4. Non-Normal Growth', '∥Aᵏ∥₂ ≤ κ₂(V) ρ(A)ᵏ', 'Normal (AᵀA=AAᵀ): ∥Aᵏ∥₂ = ρ(A)ᵏ', D.orange.dark);
  o += card(16, 178, 246, 70, D.red.tint, D.red.base, '5. BPTT Jacobians', '∥D_T⋯D₁∥₂ ≤ ∏ ∥Dₜ∥₂', 'Controlled by σ₁, gates & clipping', D.red.dark);
  o += card(278, 178, 246, 70, C.lgreen, C.green, '6. Random Gaussian A', 'ρ(A) ≈ √n  vs  ∥A∥₂ ≈ 2√n', 'Max stretch is 2× spectral radius', D.green.dark);
  return svg(540, 262, o);
}

// ── §33.5 mdl-svd-low-rank.md (8 diagrams) ──
function svdTwoBases() {
  let o = card(16, 16, 152, 110, C.lblue, C.blue, 'Input Basis V', 'v₁, …, vₙ ∈ ℝⁿ', 'VᵀV = Iₙ (ortho)', D.blue.dark);
  o += arrow(172, 71, 190, 71, C.blue);
  o += card(194, 16, 152, 110, C.lamber, C.amber, 'Diagonal Stretch Σ', 'A vᵢ = σᵢ uᵢ', 'σ₁ ≥ σ₂ ≥ ⋯ ≥ 0', D.orange.dark);
  o += arrow(350, 71, 368, 71, C.green);
  o += card(372, 16, 152, 110, C.lgreen, C.green, 'Output Basis U', 'u₁, …, uₘ ∈ ℝᵐ', 'UᵀU = Iₘ (ortho)', D.green.dark);
  o += card(16, 142, 508, 102, D.purple.tint, C.purple, 'Low-Rank Cat Photo Benchmark: 400 × 500 = 200,000 Pixels', 'Keeping k = 20 dyads ∑ᵢ₌₁²⁰ σᵢ uᵢ vᵢᵀ stores 20(400 + 500 + 1) = 18,020 numbers (9.0%)', 'Reconstructs the photograph within 8.0% relative Frobenius error!', D.purple.dark);
  return svg(540, 258, o);
}

function svdFourForms() {
  let o = card(16, 14, 246, 108, C.lblue, C.blue, '1. Full SVD: U Σ Vᵀ', 'U:(m×m)  Σ:(m×n)  Vᵀ:(n×n)', 'Includes full null-space bases', D.blue.dark);
  o += tx(139, 102, 'Orthogonal square U and V', { fs: 13, fill: C.muted });
  o += card(278, 14, 246, 108, C.lamber, C.amber, '2. Thin SVD (p = min(m,n))', 'U:(m×p)  Σ:(p×p)  Vᵀ:(p×n)', 'Drops unreached rows/cols', D.orange.dark);
  o += tx(401, 102, 'svd(A, full_matrices=False)', { mono: true, fs: 12.5, fill: D.orange.dark });
  o += card(16, 136, 246, 108, C.lgreen, C.green, '3. Compact SVD (r = rank A)', 'Uᵣ:(m×r)  Σᵣ:(r×r)  Vᵣᵀ:(r×n)', 'Keeps only σ₁ ≥ ⋯ ≥ σᵣ > 0', D.green.dark);
  o += tx(139, 224, 'Zero-free minimal factorization', { fs: 13, fill: D.green.dark });
  o += card(278, 136, 246, 108, D.purple.tint, C.purple, '4. Dyadic Sum of Rank-1s', 'A = ∑ᵢ₌₁ʳ σᵢ uᵢ vᵢᵀ', 'Sorted heaviest to lightest', D.purple.dark);
  o += tx(401, 224, 'Truncate at k ≤ r for Aₖ!', { fs: 13, fw: 700, fill: D.purple.dark });
  return svg(540, 258, o);
}

function svdExistenceProof() {
  let o = tx(270, 20, 'Existence of the SVD from the Spectral Theorem on AᵀA', { fs: 16, fw: 700, fill: C.ink });
  o += card(16, 38, 156, 102, C.lblue, C.blue, '1. Gram Matrix AᵀA', 'Symmetric & PSD:', 'xᵀAᵀAx = ∥Ax∥² ≥ 0', D.blue.dark);
  o += arrow(175, 89, 189, 89, C.blue);
  o += card(192, 38, 156, 102, C.lamber, C.amber, '2. Eigenbasis of AᵀA', 'AᵀA vᵢ = λᵢ vᵢ', 'σᵢ = √λᵢ  (r: σᵢ > 0)', D.orange.dark);
  o += arrow(351, 89, 365, 89, C.green);
  o += card(368, 38, 156, 102, C.lgreen, C.green, '3. Push Forward uᵢ', 'uᵢ = Avᵢ/σᵢ (i ≤ r)', 'Complete u_{r+1..m}', D.green.dark);
  o += card(16, 156, 508, 88, D.purple.tint, C.purple, 'Crucial Check: Why Output Directions u₁, …, uᵣ Are Automatically Orthonormal', 'uᵢᵀ uⱼ = (vᵢᵀ AᵀA vⱼ) / (σᵢ σⱼ) = (σⱼ² vᵢᵀ vⱼ) / (σᵢ σⱼ) = δᵢⱼ', 'And for i > r: ∥Avᵢ∥² = λᵢ = 0  ⇒  Avᵢ = 0, so AV = UΣ  ⇒  A = UΣVᵀ', D.purple.dark);
  return svg(540, 258, o);
}

function svdVsEigen() {
  let o = card(20, 14, 500, 68, C.lblue, C.blue, 'Variational Norms & Gram Decompositions', '∥A∥₂ = max_{∥x∥=1} ∥Ax∥ = σ₁,   ∥A∥_{F}² = tr(AᵀA) = ∑ᵢ σᵢ²', 'AᵀA = V(ΣᵀΣ)Vᵀ   and   AAᵀ = U(ΣΣᵀ)Uᵀ   share nonzero eigenvalues σᵢ²', D.blue.dark);
  o += card(20, 98, 238, 144, C.lamber, C.amber, 'Running Matrix A = [1,2;−1,3]', 'Eigenvalues: λ = 2 ± i', '|λ₁| = |λ₂| = √5 ≈ 2.236', D.orange.dark);
  o += tx(139, 212, 'Governs asymptotic Aᵏ rate', { fs: 13, fw: 700, fill: D.orange.dark });
  o += card(282, 98, 238, 144, C.lgreen, C.green, 'Singular Values of Same A', 'AᵀA = [2, −1; −1, 13]', 'σ₁ ≈ 3.618,  σ₂ ≈ 1.382 ≠ √5!', D.green.dark);
  o += tx(401, 212, 'Product match: σ₁σ₂ = |λ₁λ₂| = 5', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(540, 256, o);
}

function eckartYoungBounds() {
  let o = tx(270, 16, 'Dyadic Energy Split: A = Aₖ (Kept Rank k) + (A − Aₖ) (Dropped Tail)', { fs: 14.5, fw: 700, fill: C.ink });
  const bars = [
    ['σ₁', 92, C.lgreen, C.green, D.green.dark],
    ['σ₂', 72, C.lgreen, C.green, D.green.dark],
    ['σₖ', 50, C.lgreen, C.green, D.green.dark],
    ['σₖ₊₁', 32, D.red.tint, D.red.base, D.red.dark],
    ['σₖ₊₂', 20, D.red.tint, D.red.base, D.red.dark],
    ['σᵣ', 12, D.red.tint, D.red.base, D.red.dark],
  ];
  bars.forEach(([lab, h, f, st, dk], i) => {
    const x = 42 + i * 78, y = 136 - h;
    o += box(x, y, 54, h, f, st, 5);
    o += tx(x + 27, 152, lab, { mono: true, fs: 13.5, fw: 700, fill: dk });
  });
  o += line(264, 34, 264, 162, C.purple, 2, true);
  o += tx(148, 36, 'Kept in Aₖ (rank k)', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(392, 36, 'Tail A − Aₖ (error)', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += card(20, 174, 238, 70, C.lblue, C.blue, 'Spectral Norm Error', 'min_{rank(B)≤k} ∥A − B∥₂ = σₖ₊₁', '', D.blue.dark);
  o += card(282, 174, 238, 70, C.lamber, C.amber, 'Frobenius Norm Error', 'min_{rank(B)≤k}  ∥A − B∥_{F}² = ∑_{i>k} σᵢ²', '', D.orange.dark);
  return svg(540, 258, o);
}

function weylFrobenius() {
  let o = card(20, 14, 500, 72, D.purple.tint, C.purple, "Weyl's Inequality for Singular Values (from Spectral Eckart–Young)", 'σ_{i+j−1}(X + Y) ≤ ∥(X − X_{i−1}) + (Y − Y_{j−1})∥₂ ≤ σᵢ(X) + σⱼ(Y)', 'Proof: X_{i−1} + Y_{j−1} has rank ≤ i + j − 2', D.purple.dark);
  o += card(20, 102, 238, 142, C.lblue, C.blue, '1. Perturbation Bound (j = 1)', '|σᵢ(A + E) − σᵢ(A)| ≤ σ₁(E) = ∥E∥₂', 'Singular values are 1-Lipschitz!', D.blue.dark);
  o += tx(139, 214, 'Roundoff shifts σᵢ by ≤ ∥E∥₂', { fs: 13, fw: 700, fill: D.blue.dark });
  o += card(282, 102, 238, 142, C.lgreen, C.green, '2. Frobenius Eckart–Young', 'X = A−B, Y = B, j = k+1  (σ_{k+1}(B)=0)', '⇒  σᵢ(A − B) ≥ σ_{k+i}(A)  ∀i ≥ 1', D.green.dark);
  o += tx(401, 214, 'Sum squares: ∥A−B∥_{F}² ≥ ∑_{i>k} σᵢ²', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

function condErrorBound() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, 'Condition Number κ(A) = σ₁ / σᵣ ≥ 1: Ratio of Extreme Stretches', 'Perturbed system A(x + δx) = b + δb   ⇒   δx = A⁻¹ δb', '∥δx∥ ≤ σᵣ⁻¹ ∥δb∥   and   ∥b∥ = ∥Ax∥ ≤ σ₁ ∥x∥   (⇒ 1/∥x∥ ≤ σ₁/∥b∥)', D.blue.dark);
  o += card(20, 102, 238, 142, D.red.tint, D.red.base, 'Relative Error Bound', '∥δx∥/∥x∥ ≤ κ(A) · (∥δb∥/∥b∥)', 'Worst case: b ∥ u₁,  δb ∥ uᵣ', D.red.dark);
  o += tx(139, 214, 'Loses up to log₁₀ κ(A) digits', { fs: 13, fw: 700, fill: D.red.dark });
  o += card(282, 102, 238, 142, C.lamber, C.amber, 'Gram Squaring Hazard', 'σᵢ(AᵀA) = σᵢ(A)²', '⇒  κ(AᵀA) = (σ₁/σᵣ)² = κ(A)²', D.orange.dark);
  o += tx(401, 214, 'Why QR/SVD beat normal eqs!', { fs: 13, fw: 700, fill: D.orange.dark });
  return svg(540, 258, o);
}

function muonNewtonSchulz() {
  let o = card(20, 14, 500, 72, C.lblue, C.blue, 'Polar Decomposition: A = UΣVᵀ = (UVᵀ)(VΣVᵀ) = Q P', 'Q = UVᵀ sets every σᵢ → 1: the closest orthogonal matrix', 'Muon optimizer orthogonalizes momentum M so all active directions update at unit scale', D.blue.dark);
  o += card(20, 102, 240, 142, D.purple.tint, C.purple, 'Newton–Schulz Iteration', 'X_{t+1} = ½ (3Xₜ − Xₜ Xₜᵀ Xₜ)', 'Start: X₀ = M / ∥M∥_{F}  (σᵢ ∈ (0, 1])', D.purple.dark);
  o += tx(140, 214, 'Pure GPU matmuls — no SVD call!', { fs: 13, fw: 700, fill: D.purple.dark });
  o += arrow(264, 173, 282, 173, C.green);
  o += card(286, 102, 234, 142, C.lgreen, C.green, 'Cubic Singular Map p(σ)', 'Xₜ = U p(Σ) Vᵀ', 'p(σ) = ½(3σ − σ³) → 1', D.green.dark);
  o += tx(403, 214, '×1.5 near 0, then error squares', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(540, 258, o);
}

export const diagrams = {
  'ch33-la-motivation-axes': motivationAxes,
  'ch33-la-roadmap-two-stage': roadmapTwoStage,
  'ch33-la-three-theorems': threeTheorems,
  'ch33-la-running-benchmarks': runningBenchmarks,
  'ch33-la-pythagoras-split': pythagorasSplit,
  'ch33-la-cauchy-schwarz': cauchySchwarz,
  'ch33-la-angle-clip': angleClip,
  'ch33-la-projector-qqt': projectorQQT,
  'ch33-la-normal-vs-qr': normalVsQr,
  'ch33-la-gram-schmidt-qr': gramSchmidtQr,
  'ch33-la-cosine-scale-invariance': cosineScaleInvariance,
  'ch33-la-isotropic-vs-data': isotropicVsData,
  'ch33-la-attention-sqrt-d': attentionSqrtD,
  'ch33-la-nearest-mean-bisector': nearestMeanBisector,
  'ch33-la-row-vs-col-view': rowVsColView,
  'ch33-la-orthogonal-isometry': orthogonalIsometry,
  'ch33-la-einsum-unification': einsumUnification,
  'ch33-la-einsum-attention': einsumAttention,
  'ch33-la-rank-nullity-split': rankNullitySplit,
  'ch33-la-bottleneck-gh': bottleneckGh,
  'ch33-la-inverse-2x2-wheel': inverse2x2Wheel,
  'ch33-la-det-axioms': detAxioms,
  'ch33-la-triangular-flow-det': triangularFlowDet,
  'ch33-la-det-multiplicativity': detMultiplicativity,
  'ch33-la-schur-qr-loop': schurQrLoop,
  'ch33-la-diag-powers': diagPowers,
  'ch33-la-similarity-invariants': similarityInvariants,
  'ch33-la-multiplicity-gap': multiplicityGap,
  'ch33-la-jordan-transient': jordanTransient,
  'ch33-la-complex-rotation-block': complexRotationBlock,
  'ch33-la-real-jordan-cases': realJordanCases,
  'ch33-la-spectral-theorem-proof': spectralTheoremProof,
  'ch33-la-rayleigh-quotient': rayleighQuotient,
  'ch33-la-marchenko-pastur': marchenkoPastur,
  'ch33-la-three-challenges': threeChallenges,
  'ch33-la-pagerank-damping': pagerankDamping,
  'ch33-la-gelfand-vs-finite': gelfandVsFinite,
  'ch33-la-normal-vs-nonnormal': normalVsNonnormal,
  'ch33-la-bptt-nilpotent': bpttNilpotent,
  'ch33-la-stability-hierarchy': stabilityHierarchy,
  'ch33-la-random-matrix-gap': randomMatrixGap,
  'ch33-la-eigen-to-svd-bridge': eigenToSvdBridge,
  'ch33-la-spectra-summary': spectraSummary,
  'ch33-la-svd-two-bases': svdTwoBases,
  'ch33-la-svd-four-forms': svdFourForms,
  'ch33-la-svd-existence-proof': svdExistenceProof,
  'ch33-la-svd-vs-eigen': svdVsEigen,
  'ch33-la-eckart-young-bounds': eckartYoungBounds,
  'ch33-la-weyl-frobenius': weylFrobenius,
  'ch33-la-cond-error-bound': condErrorBound,
  'ch33-la-muon-newton-schulz': muonNewtonSchulz,
};
