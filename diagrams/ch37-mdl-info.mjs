// Slide diagrams for Chapter 37 (chapter_mdl-information-theory).
// Rendered and audited by tools/build_module_diagrams.mjs with 0 visual defects.

import { C, TOKENS, tx, arrow, svg } from './engine.mjs';
import { textWidth } from './mdl-textwidth.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const rect = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, sw = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const circle = (cx, cy, r, fill, stroke, sw = 2, dash = false) =>
  `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="4 3"' : ''}/>`;

const path = (d, stroke, fill = 'none', sw = 2.5, dash = false) =>
  `<path d="${d}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// Text that does not fit is shrunk, but never below MIN_FS. A mono string
// still too wide at MIN_FS is set in the (narrower) sans face instead; a
// string still too wide after that is wrapped onto two lines at the space
// nearest its middle. Widths come from the real font metrics
// (mdl-textwidth.mjs).
const MIN_FS = 11;
export const _fitLog = [];
function split2(s) {
  const mid = s.length / 2;
  let best = -1;
  for (let i = 1; i < s.length - 1; i++) {
    if (s[i] === ' ' && (best < 0 || Math.abs(i - mid) < Math.abs(best - mid))) best = i;
  }
  return best < 0 ? null : [s.slice(0, best).trimEnd(), s.slice(best + 1).trimStart()];
}

// { lines: [one or two strings], fs, mono } for text t in a box of width maxW.
function fitLines(t, maxW, o = {}) {
  const str = String(t);
  const avail = maxW - 22;
  const base = o.fs || 14.5;
  const bold = (o.fw || 500) >= 600;
  const fit1 = (mono) => {
    const w = textWidth(str, base, mono, bold);
    return w <= avail ? base : Math.floor((base * avail / w) * 10) / 10;
  };
  let mono = !!o.mono;
  let fs = fit1(mono);
  if (fs < MIN_FS && mono) { mono = false; fs = fit1(false); }
  if (fs >= MIN_FS) return { lines: [str], fs, mono };
  const parts = split2(str);
  if (!parts) {
    _fitLog.push(['over', str, MIN_FS]);
    return { lines: [str], fs: MIN_FS, mono };
  }
  const w2 = Math.max(...parts.map(p => textWidth(p, base, mono, bold)));
  fs = Math.max(MIN_FS, Math.min(base, Math.floor((base * avail / w2) * 10) / 10));
  _fitLog.push(['wrap', str, fs, w2 * fs / base > avail]);
  return { lines: parts, fs, mono };
}

// Centred text fitted to width maxW (one line, or two around y).
function fitTx(x, y, t, maxW, o = {}) {
  const { lines, fs, mono } = fitLines(t, maxW, o);
  if (lines.length === 1) return tx(x, y, lines[0], { ...o, fs, mono });
  const dy = 0.6 * fs;
  return tx(x, y - dy, lines[0], { ...o, fs, mono }) + tx(x, y + dy, lines[1], { ...o, fs, mono });
}

// Stack fitted items: a wrapped item adds 1.2 fs of height between its two
// lines; sp separates items. Returns the markup.
function stackLines(cx, cy, items, maxW, sp) {
  const fitted = items.map(it => ({ it, ...fitLines(it.t, maxW, it) }));
  const extra = fitted.map(f => (f.lines.length > 1 ? 1.2 * f.fs : 0));
  const total = (fitted.length - 1) * sp + extra.reduce((a, b) => a + b, 0);
  let y = cy - total / 2;
  let o = '';
  fitted.forEach((f, i) => {
    f.lines.forEach((ln, j) => {
      o += tx(cx, y + j * 1.2 * f.fs, ln, { ...f.it, fs: f.fs, mono: f.mono });
    });
    y += extra[i] + sp;
  });
  return o;
}

function wrapExtra(items, maxW) {
  return items.reduce((a, it) => {
    const f = fitLines(it.t, maxW, it);
    return a + (f.lines.length > 1 ? 1.2 * f.fs : 0);
  }, 0);
}

// High-level clean card with lines of text guaranteed inside borders
function card(x, y, w, h, fill, stroke, lines, opts = {}) {
  let o = rect(x, y, w, h, fill, stroke, opts.rx || 8, opts.sw || 2, opts.dash || false);
  const n = lines.length;
  const items = lines.map((ln, i) => {
    const item = typeof ln === 'string' ? { t: ln } : ln;
    return {
      t: item.t,
      fs: item.fs || (i === 0 ? 15 : 13.5),
      fw: item.fw || (i === 0 ? 700 : 500),
      mono: item.mono || false,
      fill: item.fill || (i === 0 ? C.ink : C.muted),
    };
  });
  const ex = wrapExtra(items, w);
  const spacing = opts.spacing || (n <= 1 ? 0 : Math.min(22, (h - 24 - ex) / (n - 1)));
  return o + stackLines(x + w / 2, y + h / 2, items, w, spacing);
}

// Header bar + body card
function titledPanel(x, y, w, h, title, accent, bodyLines, opts = {}) {
  let o = rect(x, y, w, h, opts.fill || '#FFFFFF', accent.base, 10, 2);
  o += rect(x + 6, y + 6, w - 12, 30, accent.tint, 'none', 6, 0);
  o += fitTx(x + w / 2, y + 21, title, w - 12, { fs: 14.5, fw: 700, fill: accent.dark });
  const bodyTop = y + 40;
  const bodyH = h - 44;
  const n = bodyLines.length;
  const items = bodyLines.map(ln => {
    const item = typeof ln === 'string' ? { t: ln } : ln;
    return {
      t: item.t,
      fs: item.fs || 13.5,
      fw: item.fw || 600,
      mono: item.mono || false,
      fill: item.fill || C.ink,
    };
  });
  const ex = wrapExtra(items, w);
  const sp = opts.spacing || (n <= 1 ? 0 : Math.min(22, (bodyH - 20 - ex) / (n - 1)));
  return o + stackLines(x + w / 2, bodyTop + bodyH / 2, items, w, sp);
}

// ═══════════════════════════════════════════════════════════════════════════
// §37.0 index.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function ceDecomposition() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 44, D.purple.tint, C.purple, [
    { t: 'Fundamental Loss Identity:   CE(P, Q)  =  H(P)  +  D_KL(P ‖ Q)  ≥  H(P)', mono: true, fs: 14, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 70, 240, 134, '1. Irreducible Data Floor H(P)', D.blue, [
    { t: 'H(P) = −E_P[log p(x)]', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Intrinsic label uncertainty /', fs: 13, fill: C.ink },
    { t: 'minimum lossless file size', fs: 13, fill: C.ink },
    { t: '(Independent of model Q)', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 70, 240, 134, '2. Model Mismatch D_KL(P ‖ Q)', D.green, [
    { t: 'D_KL(P ‖ Q) = E_P[log(p/q)] ≥ 0', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Extra surprisal / wasted bits', fs: 13, fill: C.ink },
    { t: 'from using Q instead of P', fs: 13, fill: C.ink },
    { t: '(Vanishes iff Q = P)', fs: 12.5, fill: D.green.dark },
  ]);

  o += card(20, 218, 500, 50, C.lamber, C.amber, [
    { t: 'Physical Meaning: Arithmetic coding stores n symbols in n · CE₂(P, Q) + 2 bits!', fs: 13.5, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function threePillars() {
  const W = 540, H = 286;
  let o = '';
  o += titledPanel(20, 14, 156, 192, '1. Gibbs Floor (§37.1)', D.blue, [
    { t: 'D_KL(P ‖ Q) ≥ 0', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'CE(P, Q) ≥ H(P)', mono: true, fs: 13, fill: C.ink },
    { t: 'No classifier beats', fs: 12.5, fill: C.ink },
    { t: 'conditional label', fs: 12.5, fill: C.ink },
    { t: 'entropy H(Y | X)', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(192, 14, 156, 192, '2. Coding (§37.2)', D.green, [
    { t: '∑ 2⁻ℓᵢ ≤ 1 (Kraft)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Arithmetic stream:', fs: 12.5, fill: C.ink },
    { t: 'ℓ < −log₂ q(x₁:ₙ) + 2', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'Any causal LLM is', fs: 12.5, fill: C.ink },
    { t: 'a lossless codec!', fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(364, 14, 156, 192, '3. Critics (§§37.3–7)', D.purple, [
    { t: 'f-GAN & InfoNCE', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Bound D_f and I(X;Y)', fs: 12.5, fill: C.ink },
    { t: 'from samples alone,', fs: 12.5, fill: C.ink },
    { t: 'capped at log N nats', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'for batch size N', fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 220, 500, 52, D.red.tint, D.red.base, [
    { t: 'Density-Ratio Blind Spot: All f-divergences freeze on disjoint supports → OT, MMD, Scores', fs: 13, fw: 700, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

function chapterRoadmap() {
  const W = 540, H = 292;
  let o = '';
  o += card(20, 12, 500, 74, C.lblue, C.blue, [
    { t: 'Track 1 · Log-Loss, Coding & Perplexity (§§37.1–37.2)', fs: 15, fw: 700, fill: D.blue.dark },
    { t: '§37.1 Surprisal, H(P), Gibbs, smoothing, distillation  ·  §37.2 Kraft, arithmetic coding, MDL, AEP, PPL', fs: 12.5, fill: C.ink },
  ]);
  o += arrow(270, 88, 270, 108, C.blue);

  o += card(20, 112, 500, 74, C.lamber, C.amber, [
    { t: 'Track 2 · Divergences, Transport & Score Fields (§§37.3–37.5)', fs: 15, fw: 700, fill: D.orange.dark },
    { t: '§37.3 f-divergences, Pinsker, f-GAN  ·  §37.4 IPMs, MMD, W₁, Sinkhorn  ·  §37.5 Stein, Score Matching, KSD', fs: 12.5, fill: C.ink },
  ]);
  o += arrow(270, 188, 270, 208, D.orange.base);

  o += card(20, 212, 500, 68, C.lgreen, C.green, [
    { t: 'Track 3 · Dependence, Bottlenecks & Contrastive Bounds (§§37.6–37.7)', fs: 15, fw: 700, fill: D.green.dark },
    { t: '§37.6 Mutual Info, DPI, Fano, Capacity, IB  ·  §37.7 KSG neighbors, McAllester–Stratos, DV/NWJ, InfoNCE', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function accessDecisionTree() {
  const W = 540, H = 286;
  let o = '';
  o += titledPanel(20, 12, 240, 122, '1. Tractable Density q_θ(x) > 0', D.blue, [
    { t: 'Forward KL: D_KL(P ‖ Q_θ)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Maximum likelihood (mode-covering)', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 12, 240, 122, '2. Unnormalized Target p̃(z, x)', D.orange, [
    { t: 'Reverse KL: D_KL(Q_θ ‖ P)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Variational inference / ELBO (mode-seeking)', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(20, 148, 240, 124, '3. Implicit Sample Generator Q_θ', D.green, [
    { t: 'Kernel MMD  or  Wasserstein W₁', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Sample test functions (valid when disjoint)', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 148, 240, 124, '4. Energy Model q̃_θ = exp(−E_θ)', D.purple, [
    { t: 'Fisher Divergence D_F & KSD', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Matches score s_Q = −∇_x E_θ without Z', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §37.1 mdl-information-theory.md (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function lossCurveLandmarks() {
  const W = 540, H = 286;
  let o = '';
  o += fitTx(270, 20, '10-Class Cross-Entropy Landmarks (in Nats)', 500, { fs: 15.5, fw: 700, fill: C.ink });

  const bars = [
    { name: 'Untrained Uniform (q = 1/10)', val: 'ln 10 = 2.303 nats', w: 290, fill: N.panel, st: C.gray, dk: C.ink },
    { name: 'Overconfident (0.99 on 10% noise)', val: '0.689 nats (tail penalty!)', w: 185, fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { name: 'Plateau Alert (0.600 nats)', val: '0.600 nats (near 10% floor)', w: 165, fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { name: 'Bayes Floor (10% Label Noise)', val: 'H(Y|X) = 0.545 nats', w: 152, fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { name: 'Clean Deterministic Floor', val: 'H(Y|X) = 0.000 nats', w: 132, fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  bars.forEach((b, i) => {
    const y = 38 + i * 38;
    o += card(16, y, 210, 30, '#FFFFFF', b.st, [
      { t: b.name, fs: 12.5, fw: 700, fill: C.ink },
    ], { rx: 6, sw: 1.5 });
    o += card(234, y, b.w, 30, b.fill, b.st, [
      { t: b.val, mono: true, fs: 12.5, fw: 700, fill: b.dk },
    ], { rx: 6, sw: 2 });
  });

  o += card(16, 234, 508, 40, D.purple.tint, C.purple, [
    { t: 'Symmetric Noise Floor (rate ρ, k classes):  H_b(ρ) + ρ log(k − 1)', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function shannonEntropyAxioms() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Shannon Entropy:   H(X) = H(P) = −E_{x~P}[log p(x)] = −∑ᵢ pᵢ log pᵢ', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 70, 156, 134, '1. Zero-Prob Rule', D.blue, [
    { t: '0 log 0 = 0', mono: true, fs: 15, fw: 700, fill: D.blue.dark },
    { t: 'lim_{t↓0} t log t = 0', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Impossible events add', fs: 12.5, fill: C.muted },
    { t: 'zero average surprise', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(192, 70, 156, 134, '2. Non-Negativity', D.green, [
    { t: 'H(P) ≥ 0', mono: true, fs: 15, fw: 700, fill: D.green.dark },
    { t: '−pᵢ log pᵢ ≥ 0 on [0,1]', mono: true, fs: 12, fill: C.ink },
    { t: 'H(P) = 0 iff one', fs: 12.5, fill: C.ink },
    { t: 'outcome has pᵢ = 1', fs: 12.5, fill: D.green.dark },
  ]);

  o += titledPanel(364, 70, 156, 134, '3. Label Invariance', D.purple, [
    { t: 'H(p₁, …, pₖ)', mono: true, fs: 14.5, fw: 700, fill: D.purple.dark },
    { t: 'Depends only on', fs: 12.5, fill: C.ink },
    { t: 'probabilities pᵢ,', fs: 12.5, fill: C.ink },
    { t: 'never on symbol values', fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 218, 500, 50, C.lamber, C.amber, [
    { t: 'Maximum Entropy over k Outcomes:  0 ≤ H(P) ≤ log k  (attained uniquely by uniform pᵢ = 1/k)', fs: 13, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function entropyChainRule() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 44, D.purple.tint, C.purple, [
    { t: 'Chain Rule for Entropy:   H(X, Y) = H(X) + H(Y | X)   (from log p(x,y) = log p(x) + log p(y|x))', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 72, 220, 118, C.lblue, C.blue, [
    { t: 'Step 1: Observe X', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'Question: "Is it a?"', fs: 13, fill: C.ink },
    { t: 'P(X=1) = ½, P(X=0) = ½', mono: true, fs: 12.5, fill: C.ink },
    { t: 'H₂(X) = 1.00 bit', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
  ]);

  o += tx(260, 131, '+', { fs: 24, fw: 700, fill: C.ink });

  o += card(280, 72, 240, 118, C.lgreen, C.green, [
    { t: 'Step 2: Remaining Y | X', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '• X=1 ("yes"): H₂(Y|1) = 0 bits', mono: true, fs: 12.5, fill: C.ink },
    { t: '• X=0 ("no"): H₂(½,¼,¼) = 1.5 bits', mono: true, fs: 12.5, fill: C.ink },
    { t: 'H₂(Y | X) = ½(0) + ½(1.5) = 0.75 bits', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 206, 500, 62, C.lamber, C.amber, [
    { t: 'Sequence Chain Rule: H(X₁, …, Xₙ) = ∑_{i=1}^{n} H(Xᵢ | X₁, …, Xᵢ₋₁)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Total: H₂(X, Y) = 1.00 + 0.75 = 1.75 bits = H₂(½, ¼, ⅛, ⅛)', mono: true, fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function gibbsJensenProof() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 14, 500, 68, C.lblue, C.blue, [
    { t: 'Step 1 · Apply Jensen’s Inequality to Strictly Concave log on supp(P)', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'D_KL(P ‖ Q) = −E_{x~P}[log(q(x)/p(x))]  ≥  −log E_{x~P}[q(x)/p(x)]', mono: true, fs: 13, fill: C.ink },
  ]);
  o += arrow(270, 84, 270, 104, C.blue);

  o += card(20, 108, 500, 68, D.purple.tint, C.purple, [
    { t: 'Step 2 · Cancel p(x) Across the Support of P', fs: 14.5, fw: 700, fill: D.purple.dark },
    { t: '−log ∑_{x ∈ supp P} p(x) (q(x)/p(x))  =  −log Q(supp P)  ≥  −log 1 = 0', mono: true, fs: 13, fw: 700, fill: C.ink },
  ]);
  o += arrow(270, 178, 270, 198, C.green);

  o += card(20, 202, 500, 68, C.lgreen, C.green, [
    { t: 'Step 3 · Strict Concavity Equality Certificate', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'D_KL(P ‖ Q) = 0  ⇔  q(x)/p(x) = 1 on supp(P) and Q(supp P) = 1  ⇔  P = Q', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function irreducibleFloorSplit() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Population Test NLL:   E_{(x,y)}[−log q(y | x)]  =  H(Y | X)  +  E_x[D_KL(p(·|x) ‖ q(·|x))]', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 70, 240, 134, 'Aleatoric Floor H(Y | X)', D.blue, [
    { t: 'E_x[H(p(· | x))]', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'Inherent label ambiguity /', fs: 13, fill: C.ink },
    { t: 'recording noise in dataset', fs: 13, fill: C.ink },
    { t: 'No model can go lower!', fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(280, 70, 240, 134, 'Reducible Excess E_x[D_KL]', D.green, [
    { t: 'E_x[D_KL(p(·|x) ‖ q(·|x))] ≥ 0', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Approximation + estimation', fs: 13, fill: C.ink },
    { t: 'gap of classifier q(· | x)', fs: 13, fill: C.ink },
    { t: '0 iff q(· | x) = p(· | x) a.e.', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 218, 500, 50, D.purple.tint, C.purple, [
    { t: 'Overparameterized Warning: Training NLL can hit 0 by memorizing noise, while Test NLL ≥ H(Y|X)', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function properScoringCalibration() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'Log Score: S_log(q, y) = −log q_y', D.blue, [
    { t: 'E_{y~p}[S_log] = H(p) + D_KL(p ‖ q)', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Excess over truth: D_KL(p ‖ q) ≥ 0', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Penalty → +∞ as q_y → 0:', fs: 13, fill: D.red.dark },
    { t: 'Heavily punishes confident errors', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Brier Score: ∑_{j} (q_j − 1_{j=y})²', D.green, [
    { t: 'E_{y~p}[S_Brier] = (1−‖p‖₂²) + ‖q−p‖₂²', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Excess over truth: ‖q − p‖₂² ≥ 0', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Bounded ≤ 2 even when q_y = 0:', fs: 13, fill: C.ink },
    { t: 'Tolerates rare tail outliers', fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 216, 500, 52, C.lamber, C.amber, [
    { t: 'Strictly Proper Guarantee: Both are uniquely minimized at q = p (honest calibrated probabilities!)', fs: 13, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §37.2 mdl-coding-compression.md (9 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function timeMachineBpc() {
  const W = 540, H = 286;
  let o = '';
  o += fitTx(270, 20, 'Compressing The Time Machine (173,428 chars, 27 symbols)', 500, { fs: 15, fw: 700, fill: C.ink });

  const codecs = [
    { name: 'Raw 8-Bit ASCII Bytes', val: '8.00 bits/char (173.4 KB)', w: 280, fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { name: 'Fixed 27-Symbol Code (log₂ 27)', val: '4.75 bits/char (103.1 KB)', w: 210, fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { name: 'Lempel–Ziv + Huffman (zlib -9)', val: '2.84 bits/char (61.6 KB)', w: 168, fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { name: 'Adaptive Order-3 Predictor', val: '2.55 bits/char (55.3 KB)', w: 152, fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  codecs.forEach((c, i) => {
    const y = 40 + i * 44;
    o += card(16, y, 212, 34, N.panel, C.gray, [
      { t: c.name, fs: 12.5, fw: 700, fill: C.ink },
    ], { rx: 6, sw: 1.5 });
    o += card(236, y, c.w, 34, c.fill, c.st, [
      { t: c.val, mono: true, fs: 12.5, fw: 700, fill: c.dk },
    ], { rx: 6, sw: 2 });
  });

  o += card(16, 224, 508, 48, D.purple.tint, C.purple, [
    { t: 'Prediction = Compression:  File length ℓ < −∑_{i=1}^{n} log₂ q(xᵢ | x_{<i}) + 2 bits', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function kraftDyadicIntervals() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Kraft’s Inequality via Dyadic Intervals: ∑_{i=1}^{k} 2⁻ℓᵢ ≤ 1', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  // Unit interval bar [0, 1) split into 1/2, 1/4, 1/8, 1/8
  const bx = 20, by = 72, bh = 64;
  o += card(bx, by, 250, bh, C.lblue, C.blue, [
    { t: 'c₁ = "0"  (ℓ₁ = 1)', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: '[0, 1/2) · width 2⁻¹ = 0.50', mono: true, fs: 12, fill: C.ink },
  ]);
  o += card(bx + 250, by, 125, bh, C.lgreen, C.green, [
    { t: 'c₂="10" (ℓ₂=2)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: '[1/2, 3/4) · 2⁻²', mono: true, fs: 11.5, fill: C.ink },
  ]);
  o += card(bx + 375, by, 62, bh, C.lamber, C.amber, [
    { t: '110', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: '2⁻³', mono: true, fs: 11.5, fill: C.ink },
  ]);
  o += card(bx + 437, by, 63, bh, D.purple.tint, C.purple, [
    { t: '111', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    { t: '2⁻³', mono: true, fs: 11.5, fill: C.ink },
  ]);

  o += titledPanel(20, 152, 240, 116, 'Necessity (Prefix-Free ⇒ ≤ 1)', D.blue, [
    { t: 'Codeword c ↔ [0.c, 0.c + 2⁻ℓ)', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Prefix-free ⇔ intervals disjoint', fs: 12.5, fill: C.ink },
    { t: 'inside [0, 1) ⇒ total width ≤ 1', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 152, 240, 116, 'Sufficiency (≤ 1 ⇒ Construct)', D.green, [
    { t: 'Sort ℓ₁ ≤ ⋯ ≤ ℓₖ & pack left-to-right', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Start sᵢ = ∑_{m<i} 2⁻ℓₘ', mono: true, fs: 12.5, fill: C.ink },
    { t: 'is a multiple of 2⁻ℓᵢ ⇒ valid codeword', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function entropyWasteSplit() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 44, D.purple.tint, C.purple, [
    { t: 'Prefix-Code Length Decomposition (Z = ∑ 2⁻ℓᵢ ≤ 1, qᵢ = 2⁻ℓᵢ/Z)', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 70, 156, 126, C.lblue, C.blue, [
    { t: '1. True Entropy', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'H₂(P)', mono: true, fs: 16, fw: 700, fill: D.blue.dark },
    { t: 'Irreducible floor', fs: 12.5, fill: C.ink },
    { t: '1.295 bits', mono: true, fs: 13.5, fw: 700, fill: C.ink },
  ]);

  o += tx(184, 133, '+', { fs: 22, fw: 700, fill: C.ink });

  o += card(192, 70, 156, 126, C.lamber, C.amber, [
    { t: '2. Shape Mismatch', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'D_KL⁽²⁾(P ‖ Q) ≥ 0', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Rounding pᵢ → qᵢ', fs: 12.5, fill: C.ink },
    { t: '+0.005 bits', mono: true, fs: 13.5, fw: 700, fill: C.ink },
  ]);

  o += tx(356, 133, '+', { fs: 22, fw: 700, fill: C.ink });

  o += card(364, 70, 156, 126, D.red.tint, D.red.base, [
    { t: '3. Unspent Kraft', fs: 14, fw: 700, fill: D.red.dark },
    { t: '−log₂ Z ≥ 0', mono: true, fs: 14, fw: 700, fill: D.red.dark },
    { t: 'Unused leaves (Z<1)', fs: 12.5, fill: C.ink },
    { t: '+0.300 bits', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 210, 500, 56, C.lgreen, C.green, [
    { t: 'Shannon Code on P = (0.6, 0.3, 0.1):  ℓ = ⌈log₂(1/pᵢ)⌉ = (1, 2, 4)  ⇒  E[ℓ] = 1.600 bits', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Always satisfies the Shannon Sandwich:   H₂(P) ≤ E_P[ℓ] < H₂(P) + 1', mono: true, fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function shannonVsHuffman() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'Shannon Code: ℓᵢ = ⌈−log₂ pᵢ⌉', D.orange, [
    { t: 'On P = (0.6, 0.3, 0.1):', fs: 13, fw: 700, fill: C.ink },
    { t: 'Surprisals (0.74, 1.74, 3.32) → (1, 2, 4)', mono: true, fs: 12, fill: C.ink },
    { t: 'Kraft sum Z = ½ + ¼ + 1/16 = 0.8125', mono: true, fs: 12, fill: D.red.dark },
    { t: 'Expected length: E[ℓ] = 1.60 bits', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Huffman Code: Bottom-Up Merge', D.green, [
    { t: 'Merges two rarest symbols (0.3, 0.1):', fs: 13, fw: 700, fill: C.ink },
    { t: 'Assigns lengths (1, 2, 2)', mono: true, fs: 13, fill: C.ink },
    { t: 'Kraft sum Z = ½ + ¼ + ¼ = 1.000 (Full!)', mono: true, fs: 12, fill: D.green.dark },
    { t: 'Expected length: E[ℓ] = 1.40 bits', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 216, 500, 52, D.red.tint, D.red.base, [
    { t: 'The 1-Bit Symbol Wall: On P(a) = 0.9 (H₂ = 0.469 bits), ANY symbol code spends ≥ 1.00 bit/symbol!', fs: 12.5, fw: 700, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

function arithTwoBitOverhead() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Arithmetic Coding Message Interval [lo, hi) of Width  w = ∏_{i=1}^{n} q(xᵢ | x_{<i})', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  // Visual interval diagram
  o += rect(20, 72, 500, 72, N.panel, C.gray, 8, 1.5);
  o += rect(90, 82, 360, 52, C.lblue, C.blue, 6, 2);
  o += rect(150, 90, 200, 36, C.lgreen, C.green, 5, 2.5);
  o += fitTx(120, 108, 'lo', 50, { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
  o += fitTx(400, 108, 'hi = lo + w', 84, { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += fitTx(250, 108, 'Dyadic [m2⁻ℓ, (m+1)2⁻ℓ)', 192, { mono: true, fs: 12, fw: 700, fill: D.green.dark });

  o += card(20, 160, 240, 106, C.lgreen, C.green, [
    { t: 'Choose Length ℓ = ⌈log₂(1/w)⌉ + 1', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Guarantees dyadic step 2⁻ℓ ≤ w / 2', mono: true, fs: 12.5, fill: C.ink },
    { t: 'so one multiple fits inside [lo, hi)', fs: 12.5, fill: C.ink },
  ]);

  o += card(280, 160, 240, 106, D.purple.tint, C.purple, [
    { t: 'Amortized 2-Bit Overhead', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'ℓ < −∑_{i=1}^{n} log₂ q(xᵢ | x_{<i}) + 2', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
    { t: 'Overhead 2/n → 0 bits/symbol!', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function llmUniversalCompressor() {
  const W = 540, H = 284;
  let o = '';
  o += card(20, 12, 500, 40, D.purple.tint, C.purple, [
    { t: 'Frozen 70B LLM + Arithmetic Coder vs. Domain Codecs (1 GB Raw Datasets)', fs: 14, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 66, 240, 140, 'ImageNet Patches (1 GB)', D.blue, [
    { t: 'PNG Codec:  58.5% of raw', mono: true, fs: 13, fill: C.ink },
    { t: '70B LLM + Arith:  43.4%!', mono: true, fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'Beats image codec on raw pixels', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 66, 240, 140, 'LibriSpeech Audio (1 GB)', D.green, [
    { t: 'FLAC Codec:  30.3% of raw', mono: true, fs: 13, fill: C.ink },
    { t: '70B LLM + Arith:  16.4%!', mono: true, fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'Nearly halves lossless audio size', fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 220, 500, 50, D.red.tint, D.red.base, [
    { t: 'Two-Part Catch: Including 140 GB of LLM weights (2 B/param) inflates 1 GB to >14,000%!', fs: 13, fw: 700, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

function mdlTwoPartCode() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Minimum Description Length (MDL):   M̂ = argmin_{M} [ L(M) + L(D | M) ]', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 70, 240, 134, 'Model Cost L(M): BIC Rule', D.purple, [
    { t: 'Empirical precision ~ 1 / √n', mono: true, fs: 13, fill: C.ink },
    { t: 'Grid of √n points costs:', fs: 12.5, fill: C.muted },
    { t: '½ log₂ n  bits / parameter', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: '(~5 bits at n = 1,000 flips)', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 70, 240, 134, 'Data Cost L(D | M): Log-Loss', D.green, [
    { t: 'Fair Coin M₀: 0 + 1000 = 1000 bits', fs: 12.5, fill: C.ink },
    { t: '• 600 heads: 5 + 971 = 976 bits (M₁✓)', fs: 12, fw: 700, fill: D.green.dark },
    { t: '• 520 heads: 5 + 998.8 = 1003.8 (M₀✓)', fs: 12, fw: 700, fill: D.red.dark },
    { t: 'Automatic Occam’s Razor!', fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 218, 500, 50, C.lamber, C.amber, [
    { t: 'Adaptive Prequential Bonus: Pays only for contexts that actually occur in the stream!', fs: 13, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function englishCompressionLadder() {
  const W = 540, H = 294;
  let o = '';
  o += fitTx(270, 18, '27-Symbol English Compression & Perplexity Ladder (PPL = 2^bpc)', 500, { fs: 14.5, fw: 700, fill: C.ink });

  const rungs = [
    { name: 'Uniform 27-Symbol Guessing', bpc: '4.75 bpc', ppl: 'PPL = 27.00', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { name: 'Unigram Letter Frequencies (Order 0)', bpc: '4.08 bpc', ppl: 'PPL = 16.91', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { name: 'Lempel–Ziv + Huffman (zlib -9)', bpc: '2.84 bpc', ppl: 'PPL = 7.16', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { name: 'Adaptive Order-3 Character Model', bpc: '2.55 bpc', ppl: 'PPL = 5.84', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { name: 'Neural Language Model (1.2 nats)', bpc: '1.73 bpc', ppl: 'PPL = 3.32', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { name: 'Shannon Human Estimate (H_rate)', bpc: '~1.00 bpc', ppl: 'PPL ≈ 2.00', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  rungs.forEach((r, i) => {
    const y = 34 + i * 42;
    o += card(16, y, 254, 34, '#FFFFFF', r.st, [
      { t: r.name, fs: 12.5, fw: 700, fill: C.ink },
    ], { rx: 6, sw: 1.5 });
    o += card(278, y, 114, 34, r.fill, r.st, [
      { t: r.bpc, mono: true, fs: 13, fw: 700, fill: r.dk },
    ], { rx: 6, sw: 2 });
    o += card(400, y, 124, 34, r.fill, r.st, [
      { t: r.ppl, mono: true, fs: 13, fw: 700, fill: r.dk },
    ], { rx: 6, sw: 2 });
  });
  return svg(W, H, o);
}

function tokenizerBpcNormalization() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, D.purple.tint, C.purple, [
    { t: 'Tokenizer-Invariant Metric:   bpc = (n_tok / n_char) · log₂(PPL)   (10⁶-Char Corpus)', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, 'Model A: Coarse Subwords', D.green, [
    { t: 'n_tok = 2 × 10⁵  (5 chars/tok)', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Per-token PPL = 32  (5 bits/tok)', mono: true, fs: 12.5, fill: D.red.dark },
    { t: 'bpc = 0.2 × 5 = 1.00 bit/char!', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Total file: 1.00 Mb (Winner!)', fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(280, 68, 240, 138, 'Model B: Fine Character Tokens', D.orange, [
    { t: 'n_tok = 5 × 10⁵  (2 chars/tok)', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Per-token PPL = 8  (3 bits/tok)', mono: true, fs: 12.5, fill: D.green.dark },
    { t: 'bpc = 0.5 × 3 = 1.50 bits/char', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Total file: 1.50 Mb (+50% larger!)', fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 220, 500, 48, C.lblue, C.blue, [
    { t: 'Rule: Never compare raw token PPL across different vocabularies; always normalize to bpc or bpb', fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §37.3 mdl-divergences-distances.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function disjointPointMasses() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 40, D.red.tint, D.red.base, [
    { t: 'Disjoint Point Masses P = δ₀ vs. Q = δ_d  (Separated by Distance d > 0)', mono: true, fs: 14, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(20, 66, 240, 146, 'f-Divergences (Density Ratio p/q)', D.red, [
    { t: 'D_f(δ₀ ‖ δ_d) = f(0) + f\'(∞)', fs: 12.5, fw: 700, fill: D.red.dark },
    { t: '• KL(P ‖ Q) = ∞  (undefined slope)', fs: 12, fill: C.ink },
    { t: '• JS(P, Q) = log 2 ≈ 0.693  (∂_{d} = 0)', fs: 12, fill: C.ink },
    { t: '• TV(P, Q) = 1.000  (∂_{d} = 0)', fs: 12, fill: C.ink },
  ]);

  o += titledPanel(280, 66, 240, 146, 'Geometry-Aware Metrics', D.green, [
    { t: 'Compare locations via test functions:', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '• RBF MMD² = 2(1 − e^{−d²/2ℓ²})', mono: true, fs: 12, fill: C.ink },
    { t: '  (Nonzero local gradient for d ~ ℓ)', fs: 12, fill: C.muted },
    { t: '• Wasserstein W₁ = |d|  (∂_{d} = ±1!)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 224, 500, 48, C.lamber, C.amber, [
    { t: 'Why Vanilla GANs Saturate: When thin generator manifolds miss data, JS/TV gradients vanish!', fs: 13, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function divergenceVsMetric() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'General Divergence D(P, Q)', D.blue, [
    { t: '1. Non-negativity: D(P, Q) ≥ 0', fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: '2. Positivity: D(P, Q) = 0 ⇔ P = Q', fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'KL fails symmetry:', fs: 13, fw: 700, fill: D.red.dark },
    { t: 'KL(N(0,1)‖N(0,4)) = 0.318 ≠ 0.807', fs: 11.5, fill: C.ink },
    { t: 'KL fails triangle: 2 > ½ + ½', fs: 12, fill: D.red.dark },
  ]);

  o += titledPanel(280, 14, 240, 188, 'True Probability Metric d(P, Q)', D.green, [
    { t: '1 & 2. Divergence axioms hold', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '3. Symmetry: d(P, Q) = d(Q, P)', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '4. Triangle: d(P,R) ≤ d(P,Q)+d(Q,R)', fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Examples: TV, Hellinger d_H, √JS,', fs: 12.5, fill: C.ink },
    { t: 'RBF-MMD, Wasserstein W₁ & W₂', fs: 12.5, fill: C.ink },
  ]);

  o += card(20, 216, 500, 52, D.purple.tint, C.purple, [
    { t: 'Gaussian Intuition: D_KL(N(μ₁,1) ‖ N(μ₂,1)) = ½(μ₁ − μ₂)² behaves like a SQUARED distance', fs: 13, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function fDivergenceCatalog() {
  const W = 540, H = 288;
  let o = '';
  o += card(20, 10, 500, 36, D.purple.tint, C.purple, [
    { t: 'Catalog of f-Divergences:  D_f(P ‖ Q) = E_Q[f(u)] ≥ f(E_Q[u]) = f(1) = 0   (u = p/q)', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 56, 156, 106, C.lblue, C.blue, [
    { t: 'Forward KL', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'f(u) = u log u', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'f\'\'(1) = 1 · Asym', mono: true, fs: 12, fill: C.muted },
    { t: 'Zero-avoiding (MLE)', fs: 12, fill: D.blue.dark },
  ]);

  o += card(192, 56, 156, 106, C.lamber, C.amber, [
    { t: 'Reverse KL', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'f(u) = −log u', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'f\'\'(1) = 1 · Asym', mono: true, fs: 12, fill: C.muted },
    { t: 'Mode-seeking (VI)', fs: 12, fill: D.orange.dark },
  ]);

  o += card(364, 56, 156, 106, D.red.tint, D.red.base, [
    { t: 'Pearson χ²', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'f(u) = (u − 1)²', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'f\'\'(1) = 2 · Asym', mono: true, fs: 12, fill: C.muted },
    { t: 'Quadratic tail weight', fs: 12, fill: D.red.dark },
  ]);

  o += card(20, 172, 156, 106, C.lgreen, C.green, [
    { t: 'Sq. Hellinger d_H²', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'f(u) = (√u − 1)²', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'f\'\'(1) = 1/2 · Sym', mono: true, fs: 12, fill: C.muted },
    { t: 'd_H is a true metric', fs: 12, fill: D.green.dark },
  ]);

  o += card(192, 172, 156, 106, D.teal.tint, D.teal.base, [
    { t: 'Total Variation TV', fs: 14, fw: 700, fill: D.teal.dark },
    { t: 'f(u) = ½ |u − 1|', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'V-Corner · Sym', mono: true, fs: 12, fill: C.muted },
    { t: 'True metric in [0, 1]', fs: 12, fill: D.teal.dark },
  ]);

  o += card(364, 172, 156, 106, D.purple.tint, C.purple, [
    { t: 'Jensen–Shannon JS', fs: 14, fw: 700, fill: D.purple.dark },
    { t: '½KL(P‖M)+½KL(Q‖M)', mono: true, fs: 11.5, fw: 700, fill: C.ink },
    { t: 'f\'\'(1) = 1/4 · Sym', mono: true, fs: 12, fill: C.muted },
    { t: '√JS metric ≤ √log 2', fs: 12, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function localChi2Collapse() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: '2nd-Order Taylor Expansion Around u = p/q = 1:   f(u) ≈ f(1) + f\'(1)(u−1) + ½f\'\'(1)(u−1)²', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 70, 152, 124, N.panel, C.gray, [
    { t: 'First-Order Term', fs: 14, fw: 700, fill: C.ink },
    { t: 'f\'(1) E_Q[u − 1]', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: '= f\'(1)(1 − 1) = 0!', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Slope drops out', fs: 12.5, fill: C.muted },
  ]);
  o += arrow(176, 132, 198, 132, C.blue);

  o += card(202, 70, 156, 124, C.lblue, C.blue, [
    { t: 'Local χ² Universal Law', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'D_f(P ‖ Q) ≈', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: '½ f\'\'(1) χ²(P ‖ Q)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: '2·KL ≈ 4·d_H² ≈ 8·JS', mono: true, fs: 12, fill: C.ink },
  ]);
  o += arrow(362, 132, 384, 132, C.purple);

  o += card(388, 70, 132, 124, D.purple.tint, C.purple, [
    { t: 'Fisher Metric', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'D_f(q_{θ+h} ‖ q_θ)', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: '≈ ½ f\'\'(1) hᵀI(θ)h', mono: true, fs: 12, fw: 700, fill: D.purple.dark },
    { t: 'Natural Gradient', fs: 12, fill: C.muted },
  ]);

  o += card(20, 208, 500, 58, C.lgreen, C.green, [
    { t: 'Key Consequence: Locally near P = Q, every smooth f-divergence defines the exact same', fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Riemannian geometry (the Fisher information matrix I(θ)), differing only by scalar f\'\'(1)!', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function forwardVsReverseKl() {
  const W = 540, H = 284;
  let o = '';
  o += titledPanel(20, 14, 240, 192, 'Forward KL: D_KL(P ‖ Q_θ)', D.blue, [
    { t: 'E_{x~P}[log(p(x) / q_θ(x))]', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'Blows up when q_θ(x) → 0 on data', fs: 12.5, fill: D.red.dark },
    { t: 'Zero-Avoiding / Mode-Covering:', fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'Stretches across all modes of P', fs: 12.5, fill: C.ink },
    { t: '(Maximum Likelihood / LLMs)', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 14, 240, 192, 'Reverse KL: D_KL(Q_θ ‖ P)', D.orange, [
    { t: 'E_{x~Q_θ}[log(q_θ(x) / p(x))]', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Blows up when p(x) → 0 on model', fs: 12.5, fill: D.red.dark },
    { t: 'Zero-Forcing / Mode-Seeking:', fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Locks onto single mode of P (cost −log wₖ)', fs: 12, fill: C.ink },
    { t: '(Variational Inference / RLHF)', fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 220, 500, 50, D.purple.tint, C.purple, [
    { t: 'On 0.7 N(−2, 0.6²) + 0.3 N(+2, 0.6²): Forward KL gives σ* = 1.93; Reverse KL gives σ* = 0.60', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function pinskerCoarseningProof() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, C.lgreen, C.green, [
    { t: 'Pinsker’s Inequality:   TV(P, Q) ≤ √(½ D_KL(P ‖ Q))   ⇔   D_KL(P ‖ Q) ≥ 2 TV(P, Q)²', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, 'Step 1: Coarsen to A* = {p > q}', D.blue, [
    { t: 'a = P(A*),  b = Q(A*),  a−b = TV', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'By the log-sum inequality:', fs: 12.5, fill: C.ink },
    { t: 'D_KL(P ‖ Q) ≥ D_KL(Ber(a) ‖ Ber(b))', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: '(2-bucket binary channel DPI)', fs: 12, fill: C.muted },
  ]);

  o += titledPanel(280, 68, 240, 138, 'Step 2: Bernoulli Curvature ≥ 4', D.purple, [
    { t: 'ρ(a) = D_KL(Ber(a)‖Ber(b)) − 2(a−b)²', mono: true, fs: 11.5, fw: 700, fill: D.purple.dark },
    { t: 'ρ(b) = ρ\'(b) = 0, and:', mono: true, fs: 12.5, fill: C.ink },
    { t: 'ρ\'\'(a) = 1/(a(1−a)) − 4 ≥ 0', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: '⇒  D_KL(Ber(a)‖Ber(b)) ≥ 2(a−b)²', mono: true, fs: 12, fw: 700, fill: C.ink },
  ]);

  o += card(20, 220, 500, 48, C.lamber, C.amber, [
    { t: 'Detector Ceiling: Any 1-sample classifier on a 50/50 mix has accuracy ≤ ½(1 + √(D_KL / 2))', fs: 12.5, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function fganVariationalGame() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 44, D.purple.tint, C.purple, [
    { t: 'Fenchel–Young Variational Bound:   D_f(P ‖ Q) ≥ sup_{T} ( E_{x~P}[T(x)] − E_{x~Q}[f*(T(x))] )', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 70, 152, 126, C.lblue, C.blue, [
    { t: 'Real Data x ~ P', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '+ E_P[T(x)]', mono: true, fs: 14.5, fw: 700, fill: C.ink },
    { t: 'Critic pushes T(x)', fs: 12.5, fill: C.muted },
    { t: 'up on real data', fs: 12.5, fill: C.muted },
  ]);

  o += card(192, 70, 156, 126, D.red.tint, D.red.base, [
    { t: 'Model Samples x ~ Q', fs: 14, fw: 700, fill: D.red.dark },
    { t: '− E_Q[f*(T(x))]', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'Conjugate penalty', fs: 12.5, fill: C.muted },
    { t: 'caps critic growth', fs: 12.5, fill: C.muted },
  ]);

  o += card(368, 70, 152, 126, C.lgreen, C.green, [
    { t: 'Optimal Critic T*', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'T*(x) = f\'(p(x)/q(x))', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: '• KL: 1 + log(p/q)', mono: true, fs: 12, fill: C.ink },
    { t: '• χ²: 2(p/q − 1)', mono: true, fs: 12, fill: C.ink },
  ]);

  o += card(20, 210, 500, 58, C.lamber, C.amber, [
    { t: 'Original GAN (Goodfellow 2014):  T(x) = log D(x),  −f*(T) = log(1 − D(x))', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'At Bayes discriminator D*(x) = p / (p + q), objective equals 2 · JS(P, Q) − log 4', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §37.4 mdl-test-functions-transport.md (9 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function shiftedGaussianSlopes() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 40, D.red.tint, D.red.base, [
    { t: 'Gradient ∂D/∂d on Overlapping Shifted Gaussians N(0, 1) vs. N(d, 1) at d = 8.0', fs: 14, fw: 700, fill: D.red.dark },
  ]);

  const rows = [
    { name: 'Jensen–Shannon ∂JS/∂d', d2: 'd=2: 0.22', d8: 'd=8: 2 × 10⁻⁴ (Flat!)', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { name: 'Total Variation ∂TV/∂d', d2: 'd=2: 0.24', d8: 'd=8: 1 × 10⁻⁴ (Flat!)', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { name: 'RBF Kernel ∂MMD²/∂d (ℓ=1)', d2: 'd=2: 0.40', d8: 'd=8: 7 × 10⁻⁵ (Flat!)', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { name: 'Wasserstein-1 ∂W₁/∂d', d2: 'd=2: 1.00', d8: 'd=8: 1.00 (Steady!)', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  rows.forEach((r, i) => {
    const y = 64 + i * 42;
    o += card(16, y, 196, 34, '#FFFFFF', r.st, [
      { t: r.name, fs: 12.5, fw: 700, fill: C.ink },
    ], { rx: 6, sw: 1.5 });
    o += card(220, y, 114, 34, N.panel, C.gray, [
      { t: r.d2, mono: true, fs: 12.5, fw: 600, fill: C.ink },
    ], { rx: 6, sw: 1.5 });
    o += card(342, y, 182, 34, r.fill, r.st, [
      { t: r.d8, mono: true, fs: 12.5, fw: 700, fill: r.dk },
    ], { rx: 6, sw: 2 });
  });

  o += card(16, 236, 508, 38, C.lblue, C.blue, [
    { t: 'Smooth Gaussian tails do NOT rescue JS, TV, or fixed-ℓ MMD once separation d ≫ 1!', fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function ipmAuditorClasses() {
  const W = 540, H = 284;
  let o = '';
  o += card(20, 12, 500, 42, D.purple.tint, C.purple, [
    { t: 'Integral Probability Metric:  IPM_{G}(P, Q) = sup_{g ∈ G} (E_P[g(x)] − E_Q[g(y)])', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 68, 156, 146, '1. Bounded |g| ≤ ½', D.orange, [
    { t: 'Yields TV(P, Q)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'g*(x) = ½ sign(p−q)', mono: true, fs: 12, fill: C.ink },
    { t: 'Too rich on samples', fs: 12, fill: D.red.dark },
    { t: '(memorizes points!)', fs: 12, fill: D.red.dark },
  ]);

  o += titledPanel(192, 68, 156, 146, '2. RKHS Ball ‖w‖₂≤1', D.blue, [
    { t: 'Yields MMD(P, Q)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'g(x) = ⟨w, φ(x)⟩', mono: true, fs: 12.5, fill: C.ink },
    { t: 'No training needed!', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Closed-form kernel', fs: 12, fill: C.ink },
  ]);

  o += titledPanel(364, 68, 156, 146, '3. 1-Lipschitz Lip₁', D.green, [
    { t: 'Yields W₁(P, Q)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: '|g(x)−g(y)| ≤ ‖x−y‖', mono: true, fs: 11.5, fill: C.ink },
    { t: 'Grows linearly with', fs: 12, fill: C.ink },
    { t: 'distance (WGAN)', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 226, 500, 44, C.lblue, C.blue, [
    { t: 'Key Difference from f-GAN: Objective is linear in g, so constraint g ∈ G is what keeps it finite', fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function testClassGoldilocks() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 190, 'Failure 1: G Too Rich / Spiky', D.red, [
    { t: 'Class: All functions |g(x)| ≤ ½', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Draw X₁:N, Y₁:N ~ N(0, 1) (P = Q!)', mono: true, fs: 12, fill: C.ink },
    { t: 'Continuous samples never overlap,', fs: 12.5, fill: C.ink },
    { t: 'so g(x) = 1_{x∈{Xᵢ}} − ½ achieves:', mono: true, fs: 12, fill: D.red.dark },
    { t: 'Empirical TV(P̂_N, Q̂_N) = 1.00!', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(280, 14, 240, 190, 'Failure 2: G Bounded in Height', D.orange, [
    { t: 'If |g(x)| ≤ M everywhere:', fs: 13, fw: 700, fill: C.ink },
    { t: 'E_P[g] − E_Q[g] ≤ 2M (saturates)', mono: true, fs: 12.5, fill: D.orange.dark },
    { t: 'On N(0,1) vs N(d,1) at d = 8:', fs: 12.5, fill: C.ink },
    { t: 'TV = 2Φ(d/2) − 1 = 0.99994', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Zero gradient to pull far samples!', fs: 12.5, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 218, 500, 50, C.lgreen, C.green, [
    { t: '1-Lipschitz Resolution: Bounded slope |∇g| ≤ 1 prevents sample memorization AND grows with d', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function mmdMeanEmbedding() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Mean Feature Embedding:   m_P = E_{x~P}[φ(x)],   MMD(P, Q) = sup_{‖w‖₂≤1} wᵀ(m_P − m_Q)', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 70, 152, 124, N.panel, C.gray, [
    { t: 'Linear φ(x) = x', fs: 14, fw: 700, fill: C.ink },
    { t: 'On N(0,1) vs N(0,4):', mono: true, fs: 12, fill: C.ink },
    { t: '|E[X] − E[Y]| = 0', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Blind to variance!', fs: 12.5, fill: D.red.dark },
  ]);
  o += arrow(176, 132, 198, 132, C.blue);

  o += card(202, 70, 156, 124, C.lblue, C.blue, [
    { t: 'Quadratic φ = (x, x²)', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'm_P = (0, 1)', mono: true, fs: 13, fill: C.ink },
    { t: 'm_Q = (0, 4)', mono: true, fs: 13, fill: C.ink },
    { t: '‖m_P − m_Q‖₂ = 3', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);
  o += arrow(362, 132, 384, 132, C.green);

  o += card(388, 70, 132, 124, C.lgreen, C.green, [
    { t: 'Cauchy–Schwarz', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'w* = (m_P−m_Q)/‖·‖', mono: true, fs: 11.5, fill: C.ink },
    { t: 'Solves sup_{w} in', fs: 12.5, fill: C.ink },
    { t: 'closed form!', fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 208, 500, 58, D.purple.tint, C.purple, [
    { t: 'Characteristic RBF Kernel k(x,x\') = exp(−‖x−x\'‖² / 2ℓ²):', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Embeds all moments j = 0..∞ simultaneously ⇒ MMD(P, Q) = 0 if and only if P = Q', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function mmdKernelExpansion() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 44, D.purple.tint, C.purple, [
    { t: 'Expanding ‖m_P − m_Q‖₂² via the Kernel Trick  k(x, x\') = ⟨φ(x), φ(x\')⟩', mono: true, fs: 14, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 72, 148, 118, C.lblue, C.blue, [
    { t: 'Within-P Similarity', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: '+ E_{x,x\'~P}[k(x,x\')]', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Average kernel on', fs: 12.5, fill: C.muted },
    { t: 'pairs from P', fs: 12.5, fill: C.muted },
  ]);

  o += tx(180, 131, '+', { fs: 22, fw: 700, fill: C.ink });

  o += card(192, 72, 148, 118, C.lgreen, C.green, [
    { t: 'Within-Q Similarity', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: '+ E_{y,y\'~Q}[k(y,y\')]', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Average kernel on', fs: 12.5, fill: C.muted },
    { t: 'pairs from Q', fs: 12.5, fill: C.muted },
  ]);

  o += tx(354, 131, '−', { fs: 24, fw: 700, fill: D.red.dark });

  o += card(368, 72, 152, 118, D.red.tint, D.red.base, [
    { t: 'Cross Similarity', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: '2 E_{x~P,y~Q}[k(x,y)]', fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Cancels within-group', fs: 12, fill: C.ink },
    { t: 'terms iff P = Q', fs: 12.5, fill: C.ink },
  ]);

  o += card(20, 206, 500, 60, C.lamber, C.amber, [
    { t: 'Witness Function (Optimal Auditor):  g*(x) ∝ E_{X~P}[k(x, X)] − E_{Y~Q}[k(x, Y)]', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Unbiased U-Statistic drops diagonal self-pairs i = j (which add upward bias k(xᵢ,xᵢ)/n = 1/n)', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function krDualityBridge() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 14, 194, 174, C.lblue, C.blue, [
    { t: 'Primal Optimal Transport', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'W₁(P, Q) =', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'inf_{γ ∈ Π(P,Q)} E_γ[‖x − y‖]', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
    { t: 'Minimum mass × distance', fs: 12.5, fill: C.ink },
    { t: 'work over couplings γ', fs: 12.5, fill: C.muted },
  ]);

  o += arrow(220, 84, 320, 84, C.purple);
  o += fitTx(270, 64, 'LP Strong Duality', 96, { fs: 12.5, fw: 700, fill: D.purple.dark });
  o += arrow(320, 122, 220, 122, C.green);
  o += fitTx(270, 142, 'Kantorovich 1958', 96, { fs: 12.5, fw: 700, fill: D.green.dark });

  o += card(326, 14, 194, 174, C.lgreen, C.green, [
    { t: 'Dual 1-Lipschitz IPM', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'W₁(P, Q) =', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'sup_{‖g‖_{Lip} ≤ 1} (E_P[g] − E_Q[g])', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    { t: '1-Lipschitz WGAN critic', fs: 12.5, fill: C.ink },
    { t: '|g(x) − g(y)| ≤ ‖x − y‖', mono: true, fs: 12, fill: C.muted },
  ]);

  o += card(20, 204, 500, 60, C.lamber, C.amber, [
    { t: 'Exact 1D Collapse:  W₁(P, Q) = ∫_{−∞}^{∞} |F_P(t) − F_Q(t)| dt   (Area between CDFs)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Attained by monotone quantile coupling (F_P⁻¹(U), F_Q⁻¹(U)) and critic slope g\'(t) = sign(F_Q − F_P)', fs: 12, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function w2UncrossingFid() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, '1D Uncrossing Under W₂² Cost', D.blue, [
    { t: 'For x < x\' and y < y\':', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Crossing: (x−y\')² + (x\'−y)²', mono: true, fs: 12.5, fill: D.red.dark },
    { t: 'Monotone: (x−y)² + (x\'−y\')²', mono: true, fs: 12.5, fill: D.green.dark },
    { t: 'Savings = 2(x\' − x)(y\' − y) > 0!', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Quantile map is strictly unique', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Gaussian W₂² & FID Metric', D.green, [
    { t: '1D Gaussians N(μ₁,σ₁²), N(μ₂,σ₂²):', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'W₂² = (μ₁ − μ₂)² + (σ₁ − σ₂)²', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'In ℝⁿ (Fréchet Inception Dist.):', fs: 12.5, fill: C.ink },
    { t: '‖μ₁−μ₂‖₂² + tr(Σ₁+Σ₂−2(Σ₁¹/²Σ₂Σ₁¹/²)¹/²)', mono: true, fs: 10.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 216, 500, 52, D.purple.tint, C.purple, [
    { t: 'Why W₂ Matters for Flows (§38.7): Strictly penalizes crossing trajectories in space-time!', fs: 13, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function sinkhornGibbsScaling() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Entropic Optimal Transport:   min_{γ ∈ Π(p,q)} ⟨γ, C⟩ + ε ∑_{i,j} γ_{ij}(log γ_{ij} − 1)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 70, 152, 124, D.purple.tint, C.purple, [
    { t: '1. Gibbs Kernel K', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'K_{ij} = exp(−C_{ij}/ε)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'KKT form:', fs: 12.5, fill: C.muted },
    { t: 'γ_ε = diag(a) K diag(b)', mono: true, fs: 11.5, fw: 700, fill: D.purple.dark },
  ]);
  o += arrow(176, 132, 198, 132, C.purple);

  o += card(202, 70, 156, 124, C.lblue, C.blue, [
    { t: '2. Sinkhorn–Knopp', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'b ← q / (Kᵀ a)', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'a ← p / (K b)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'O(N²) GPU matmuls', fs: 12.5, fill: C.muted },
  ]);
  o += arrow(362, 132, 384, 132, C.green);

  o += card(388, 70, 132, 124, C.lgreen, C.green, [
    { t: '3. ε Trade-Off', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Large ε: fast, blurry', fs: 12, fill: C.ink },
    { t: 'Small ε → 0: exact', fs: 12, fill: C.ink },
    { t: 'LP, O(1/ε) steps', mono: true, fs: 12, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 208, 500, 58, C.lamber, C.amber, [
    { t: 'End-to-End Differentiable: Unrolled Sinkhorn matrix-vector divisions backpropagate cleanly,', fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'using log-domain logsumexp updates whenever ε is small enough that exp(−C_{ij}/ε) underflows', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function mmdVsW1Dimension() {
  const W = 540, H = 284;
  let o = '';
  o += titledPanel(20, 14, 240, 192, 'Fixed-Kernel MMD (RBF)', D.blue, [
    { t: 'Compute: Closed-form O(N²) U-stat', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Sample Rate: O(N⁻¹/²) in ANY dim n!', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Tail Gradient: Vanishes for d ≫ ℓ', fs: 12.5, fill: D.red.dark },
    { t: 'Best for: Two-sample tests, KID,', fs: 12.5, fill: C.ink },
    { t: 'multi-bandwidth kernel losses', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 14, 240, 192, 'Wasserstein-1 Distance (W₁)', D.green, [
    { t: 'Compute: OT LP / Sinkhorn / Critic', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Empirical Rate: O(N⁻¹/ⁿ) for n > 2!', mono: true, fs: 12, fw: 700, fill: D.red.dark },
    { t: 'Tail Gradient: Unit slope ‖∇W₁‖ = 1', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Best for: WGAN (neural critic G),', fs: 12.5, fill: C.ink },
    { t: 'low-D or regularized minibatch OT', fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 220, 500, 50, D.red.tint, D.red.base, [
    { t: 'Curse of Dimension in Empirical W₁: In n = 100 dims, halving sample W₁ error takes 2¹⁰⁰ ≈ 10³⁰× data!', fs: 12.5, fw: 700, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §37.5 mdl-scores.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function ebmScoreEscape() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 44, D.red.tint, D.red.base, [
    { t: 'Energy-Based Model (EBM):   q_θ(x) = exp(−E_θ(x)) / Z(θ),   Z(θ) = ∫_{ℝⁿ} exp(−E_θ(x)) dx', mono: true, fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 70, 156, 126, D.red.tint, D.red.base, [
    { t: 'MLE Bottleneck', fs: 14, fw: 700, fill: D.red.dark },
    { t: '∇_θ log Z(θ)', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'Intractable ℝⁿ', fs: 12.5, fill: D.red.dark },
    { t: 'integral (n = 784+)', fs: 12.5, fill: C.muted },
  ]);

  o += card(192, 70, 156, 126, C.lamber, C.amber, [
    { t: 'GAN / MMD / W₁', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'Needs x ~ Q_θ', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'Requires expensive', fs: 12.5, fill: D.orange.dark },
    { t: 'Langevin MCMC loop', fs: 12, fill: C.muted },
  ]);

  o += card(364, 70, 156, 126, C.lgreen, C.green, [
    { t: 'Score Escape Hatch', fs: 14, fw: 700, fill: D.green.dark },
    { t: 's_Q(x) = ∇_x log q_θ', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '= −∇_x E_θ(x)', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: '∇_x log Z(θ) = 0!', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 210, 500, 58, C.lblue, C.blue, [
    { t: 'Fisher Divergence:   D_F(P ‖ Q) = ½ E_{x~P}[ ‖s_P(x) − s_Q(x)‖₂² ]', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'Compares data and model slope fields using only 1 backward pass wrt input x (zero Z needed!)', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function fisherModeBlindness() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, D.red.tint, D.red.base, [
    { t: 'Mode-Weight Blindness on P_w = w N(−m, 1) + (1 − w) N(+m, 1)   (w = 0.7 vs. 0.3, m = 5)', fs: 13.5, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(20, 68, 156, 138, 'Left Mode (x ≈ −5)', D.blue, [
    { t: 'r₁(x) ≈ 1 for both w', fs: 12.5, fill: C.ink },
    { t: 's_{0.7}(x) ≈ −(x + 5)', fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 's_{0.3}(x) ≈ −(x + 5)', fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Scores identical!', fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(192, 68, 156, 138, 'Valley (x ≈ 0)', D.red, [
    { t: 'Zero crossing shifts', fs: 12.5, fill: C.ink },
    { t: 'to x = c_w/m = ±0.085', fs: 11.5, fw: 700, fill: D.red.dark },
    { t: 'But p(0) ~ e^{−m²/2}', fs: 12.5, fill: C.ink },
    { t: '≈ 1.5 × 10⁻⁶ (Empty!)', fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(364, 68, 156, 138, 'Right Mode (x ≈ +5)', D.blue, [
    { t: 'r₂(x) ≈ 1 for both w', fs: 12.5, fill: C.ink },
    { t: 's_{0.7}(x) ≈ −(x − 5)', fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 's_{0.3}(x) ≈ −(x − 5)', fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Scores identical!', fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 220, 500, 48, C.lamber, C.amber, [
    { t: 'At m = 5:  D_KL(P_{0.7} ‖ P_{0.3}) = 0.339 nats,  but  D_F(P_{0.7} ‖ P_{0.3}) ≈ 4×10⁻⁶ ≈ 0!', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function multiscaleNoiseBridge() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'Clean Mixture (σ_noise = 0)', D.red, [
    { t: 'Total variance: σ_tot² = 1.00', mono: true, fs: 13, fill: C.ink },
    { t: 'Mode separation: 2m / σ_tot = 10.0 σ', mono: true, fs: 12.5, fill: D.red.dark },
    { t: 'Valley density p(0) = 1.5 × 10⁻⁶', mono: true, fs: 12.5, fill: D.red.dark },
    { t: 'D_F ≈ 4 × 10⁻⁶ (Blind to w!)', mono: true, fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Noised Mixture (σ_noise = 2)', D.green, [
    { t: 'Total variance: σ_tot² = 1 + 4 = 5.00', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Mode separation: 10 / √5 = 4.47 σ', mono: true, fs: 12.5, fill: D.green.dark },
    { t: 'Valley density p(0) = 1.5 × 10⁻²', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '10,000× higher valley visibility!', fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 216, 500, 52, D.purple.tint, C.purple, [
    { t: 'Diffusion Recipe (Song & Ermon 2019): High noise learns global mode weights; low noise refines shapes', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function steinIbpIdentity() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Stein’s Identity:   E_{x~p}[ g\'(x) + g(x) s_p(x) ] = 0   (for any smooth test function g)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 70, 152, 124, D.red.tint, D.red.base, [
    { t: 'Unknown Cross Term', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: '−E_p[g(x) s_p(x)]', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: '= −∫ g(x) p\'(x) dx', mono: true, fs: 12.5, fill: C.ink },
    { t: '(since p\' = p s_p)', fs: 12, fill: C.muted },
  ]);
  o += arrow(176, 132, 198, 132, C.blue);

  o += card(202, 70, 156, 124, C.lblue, C.blue, [
    { t: 'Integrate by Parts', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: '−[g(x)p(x)]_{−∞}^{∞}', mono: true, fs: 13, fill: C.muted },
    { t: '+ ∫ g\'(x) p(x) dx', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: '(Boundary vanishes)', fs: 12, fill: C.ink },
  ]);
  o += arrow(362, 132, 384, 132, C.green);

  o += card(388, 70, 132, 124, C.lgreen, C.green, [
    { t: 'Sample Average!', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: '+ E_p[g\'(x)]', mono: true, fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'No formula for', fs: 12, fill: C.ink },
    { t: 's_p(x) needed!', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 208, 500, 58, D.purple.tint, C.purple, [
    { t: 'Gaussian Special Case (Stein’s Lemma, p = N(0, 1), s_p(x) = −x):   E[X g(X)] = E[g\'(X)]', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Choosing g(x) = s_Q(x) turns the unknown cross term −E_P[s_P s_Q] into +E_P[s_Q\'(x)]!', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function hyvarinenTwoTerms() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 44, D.purple.tint, C.purple, [
    { t: 'Hyvärinen Score Matching (2005):   J(Q) = E_{x~P}[ ½ ‖s_Q(x)‖₂² + tr(∇_x s_Q(x)) ]', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 70, 240, 136, 'Term 1: ½ ‖s_Q(x)‖₂² → 0', D.blue, [
    { t: 'Penalizes non-zero slope at data xᵢ:', fs: 12.5, fw: 700, fill: C.ink },
    { t: '∇_x log q(xᵢ) ≈ 0', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'Forces data points to sit at', fs: 12.5, fill: C.ink },
    { t: 'stationary points of log q(x)', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 70, 240, 136, 'Term 2: tr(∇_x s_Q(x)) < 0', D.green, [
    { t: 'Minimizes Laplacian Δ log q(xᵢ):', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'tr(∇²_{x} log q(xᵢ)) ≪ 0', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Forces log-density to curve downward', fs: 12.5, fill: C.ink },
    { t: '(a sharp peak, not a minimum!)', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 220, 500, 48, C.lamber, C.amber, [
    { t: 'Exact Shift:  J(Q) = D_F(P ‖ Q) − ½ E_P[‖s_P(x)‖₂²]   (Requires neither Z nor s_P!)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function ksdSteinKernel() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Kernel Stein Discrepancy (KSD):   KSD(P, Q) = sup_{‖g‖_{H} ≤ 1} E_{x~P}[ g\'(x) + s_Q(x) g(x) ]', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 68, 236, 130, D.purple.tint, C.purple, [
    { t: 'Closed-Form Stein Kernel k_Q', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'KSD²(P, Q) = E_{x,x\'~P}[k_Q(x, x\')]', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'Uses only model score s_Q(xᵢ)', fs: 12.5, fill: C.ink },
    { t: 'and kernel derivatives ∂k(xᵢ, xⱼ)', fs: 12.5, fill: C.muted },
  ]);

  o += arrow(260, 133, 280, 133, C.purple);

  o += card(284, 68, 236, 130, C.lgreen, C.green, [
    { t: 'Smoothed Fisher Identity', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'KSD²(P, Q) = E_{x,x\'~P}[', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '(s_Q−s_P)(x) k(x,x\') (s_Q−s_P)(x\') ]', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    { t: 'KSD(P, Q) = 0 ⇔ P = Q', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 212, 500, 54, C.lamber, C.amber, [
    { t: 'Goodness-of-Fit Without Sampling Q: Tests an unnormalized model q̃(x) against data {xᵢ} ~ P', fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'in O(N²) time via the off-diagonal U-statistic (1 / N(N−1)) ∑_{i≠j} k_Q(xᵢ, xⱼ)', mono: true, fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function masterLossGuide() {
  const W = 540, H = 288;
  let o = '';
  o += card(20, 10, 500, 36, D.purple.tint, C.purple, [
    { t: 'Master Decision Guide: Matching Training Objectives to Model Access', fs: 14.5, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 56, 156, 106, C.lblue, C.blue, [
    { t: '1. MLE (Forward KL)', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Needs: q_θ(x) > 0', mono: true, fs: 12, fill: C.ink },
    { t: 'δ₀ vs δ_d: ∞', mono: true, fs: 12, fill: D.red.dark },
    { t: 'LLMs, Autoregressive', fs: 12, fill: C.muted },
  ]);

  o += card(192, 56, 156, 106, C.lamber, C.amber, [
    { t: '2. VI / ELBO (Rev KL)', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Needs: x~q_θ, p̃(x)', mono: true, fs: 12, fill: C.ink },
    { t: 'δ₀ vs δ_d: ∞', mono: true, fs: 12, fill: D.red.dark },
    { t: 'Mode-seeking fit', fs: 12, fill: C.muted },
  ]);

  o += card(364, 56, 156, 106, D.red.tint, D.red.base, [
    { t: '3. f-GAN / JS Critic', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Needs: Samples + T', fs: 12, fill: C.ink },
    { t: 'δ₀ vs δ_d: Constant', mono: true, fs: 12, fill: D.red.dark },
    { t: 'Zero grad if disjoint', fs: 12, fill: D.red.dark },
  ]);

  o += card(20, 172, 156, 106, D.teal.tint, D.teal.base, [
    { t: '4. Kernel MMD / KID', fs: 13.5, fw: 700, fill: D.teal.dark },
    { t: 'Needs: Samples + k', fs: 12, fill: C.ink },
    { t: 'δ₀ vs δ_d: √(2−2e⁻ᵈ²), 2ℓ² = 1', mono: true, fs: 11, fill: C.ink },
    { t: 'O(N⁻¹/²) in any dim', fs: 12, fill: D.teal.dark },
  ]);

  o += card(192, 172, 156, 106, C.lgreen, C.green, [
    { t: '5. WGAN / Sinkhorn', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Needs: Samples + Lip₁', fs: 12, fill: C.ink },
    { t: 'δ₀ vs δ_d: |d| (Linear)', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    { t: 'Steady unit gradient', fs: 12, fill: D.green.dark },
  ]);

  o += card(364, 172, 156, 106, D.purple.tint, C.purple, [
    { t: '6. Score / KSD (D_F)', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Needs: s_Q = ∇log q̃', mono: true, fs: 12, fill: C.ink },
    { t: 'No Z or Q-samples!', fs: 12, fw: 700, fill: D.purple.dark },
    { t: 'Needs multi-scale σ', fs: 12, fill: C.muted },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §37.6 mdl-mutual-information.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function bscFlippedBit() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 40, C.lblue, C.blue, [
    { t: 'Binary Symmetric Channel (Flip Rate δ = 0.1):   I(X; Y) = D_KL(P_{X,Y} ‖ P_X ⊗ P_Y)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 66, 240, 140, 'True Joint P_{X,Y} (δ = 0.1)', D.blue, [
    { t: '[ 0.45   0.05 ]   (X = 0)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: '[ 0.05   0.45 ]   (X = 1)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'H(X, Y) = 1.018 nats', mono: true, fs: 13, fill: C.ink },
    { t: 'H(Y | X) = H_b(0.1) = 0.325 nats', mono: true, fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 66, 240, 140, 'Independent Product P_X ⊗ P_Y', D.orange, [
    { t: '[ 0.25   0.25 ]   (p_X = ½)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: '[ 0.25   0.25 ]   (p_Y = ½)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Ratio p_{X,Y} / (p_X p_Y):', fs: 12.5, fill: C.ink },
    { t: '1.8 on diagonal,  0.2 off-diagonal', mono: true, fs: 12, fw: 700, fill: C.ink },
  ]);

  o += card(20, 220, 500, 48, C.lgreen, C.green, [
    { t: 'I(X; Y) = H(Y) − H(Y|X) = 0.693 − 0.325 = 0.368 nats  (0.9 ln 1.8 + 0.1 ln 0.2 = 0.368)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function bivariateGaussianMi() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 42, D.purple.tint, C.purple, [
    { t: 'Bivariate Gaussian Mutual Information:   I(X; Y) = −½ log(1 − ρ²)   (Var(Y|X) = σ_Y²(1 − ρ²))', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);

  const cols = [
    { rho: 'ρ = 0.0', var: '1−ρ² = 1.00', mi: '0.000 nats', fill: N.panel, st: C.gray, dk: C.ink },
    { rho: 'ρ = 0.5', var: '1−ρ² = 0.75', mi: '0.144 nats', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { rho: 'ρ = 0.9', var: '1−ρ² = 0.19', mi: '0.830 nats', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { rho: 'ρ = 0.99', var: '1−ρ² = 0.02', mi: '1.959 nats', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { rho: 'ρ → ±1', var: '1−ρ² → 0', mi: 'I → ∞', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  cols.forEach((c, i) => {
    const x = 20 + i * 102;
    o += card(x, 70, 92, 124, c.fill, c.st, [
      { t: c.rho, mono: true, fs: 14, fw: 700, fill: c.dk },
      { t: c.var, mono: true, fs: 11.5, fill: C.ink },
      { t: c.mi, mono: true, fs: 13, fw: 700, fill: c.dk },
    ]);
  });

  o += card(20, 210, 500, 58, C.lgreen, C.green, [
    { t: 'Bijection Invariance:   I(φ(X); ψ(Y)) = I(X; Y)   for all smooth bijections φ, ψ', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Jacobians |det Dφ| and |det Dψ| cancel identically between p_{X,Y} and p_X p_Y!', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function dpiUsableInfo() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Data-Processing Inequality along Y → X → Z₁ → Z₂:   I(Y; Z₂) ≤ I(Y; Z₁) ≤ I(Y; X)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, 'Raw Input X: Concentric Circles', D.red, [
    { t: 'Radius r = 1 (Y=0) vs r = 2 (Y=1)', mono: true, fs: 12.5, fill: C.ink },
    { t: 'True Shannon MI: I(Y; X) = ln 2', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'Linear half-plane accuracy ≤ 2/3:', fs: 12.5, fill: C.ink },
    { t: 'Linearly Usable Info = 0 nats!', fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(280, 68, 240, 138, 'Feature Z₁ = ‖X‖₂² ∈ {1, 4}', D.green, [
    { t: 'Creates NO new Shannon info:', fs: 12.5, fill: C.ink },
    { t: 'I(Y; Z₁) = ln 2 = I(Y; X)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Untangles circles into 1D points:', fs: 12.5, fill: C.ink },
    { t: 'Linearly Usable Info = ln 2 (100%)!', fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 220, 500, 48, D.purple.tint, C.purple, [
    { t: 'Deep Learning Takeaway: Hidden layers cannot create label info; they make it linearly decodable!', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function ibRateDistortion() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 44, D.purple.tint, C.purple, [
    { t: 'Information Bottleneck (Y → X → Z):   min_{p(z|x)} [ I(X; Z) − β I(Y; Z) ]', mono: true, fs: 14, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 70, 156, 126, C.lblue, C.blue, [
    { t: 'Compression Rate', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'R = I(X; Z)', mono: true, fs: 14.5, fw: 700, fill: C.ink },
    { t: 'Discards nuisance', fs: 12.5, fill: C.muted },
    { t: 'details of input X', fs: 12.5, fill: C.muted },
  ]);

  o += card(192, 70, 156, 126, C.lgreen, C.green, [
    { t: 'KL Distortion d_IB', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'I(X;Y) − I(Y;Z) =', fs: 13, fw: 700, fill: C.ink },
    { t: 'E[D_KL(p(y|x)‖p(y|z))]', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Exact RD problem!', fs: 12.5, fill: D.green.dark },
  ]);

  o += card(364, 70, 156, 126, C.lamber, C.amber, [
    { t: 'Phase Transition', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'β ≤ 1/ρ² ⇒ I(X;Z)=0', mono: true, fs: 12, fw: 700, fill: D.red.dark },
    { t: 'Gaussian frontier:', fs: 12.5, fill: C.ink },
    { t: 'β_c = 1 / 0.81 = 1.235', mono: true, fs: 12, fw: 700, fill: D.orange.dark },
  ]);

  o += card(20, 210, 500, 58, C.lblue, C.blue, [
    { t: 'Variational IB & VAEs:  E_x[D_KL(p(z|x) ‖ r(z))] = I(X; Z) + D_KL(p(z) ‖ r(z)) ≥ I(X; Z)', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'Replacing marginal p(z) with prior r(z) = N(0, I) gives the exact KL regularizer of a VAE!', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §37.7 mdl-mi-estimation.md (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function mcallesterStratosWall() {
  const W = 540, H = 284;
  let o = '';
  o += card(20, 12, 500, 42, D.red.tint, D.red.base, [
    { t: 'McAllester–Stratos Theorem (2020): Any 95% Distribution-Free Lower Bound B(Sₙ) ≤ O(log n)', fs: 13.5, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 68, 236, 62, C.lblue, C.blue, [
    { t: '1. Birthday Collision Bound', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Pr(collision in n draws of M) ≤ n²/(2M)', mono: true, fs: 11.5, fill: C.ink },
  ]);
  o += arrow(260, 99, 280, 99, C.blue);

  o += card(284, 68, 236, 62, C.lblue, C.blue, [
    { t: '2. Test on M = n² Uniform Symbols', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'H(X) = 2 log n  ⇒  Pr(B ≤ 2 log n) ≥ 0.95', mono: true, fs: 11.5, fill: C.ink },
  ]);

  o += card(20, 142, 236, 62, D.purple.tint, C.purple, [
    { t: '3. All-Distinct Overlap (> 50%)', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: '0.95 + 0.5 > 1 ⇒ B ≤ 2 log n when distinct', mono: true, fs: 11.5, fill: C.ink },
  ]);
  o += arrow(260, 173, 280, 173, C.purple);

  o += card(284, 142, 236, 62, D.red.tint, D.red.base, [
    { t: '4. Label-Invariance Ceiling', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Outputs ≤ 2 log n on ANY distinct sample!', fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 218, 500, 50, C.lamber, C.amber, [
    { t: 'At n = 10⁶ samples: 2 ln n = 27.6 nats max certificate, even when true H = 177 nats (256-bit keys)!', fs: 12.5, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function tishbyTanhBinning() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'True Deterministic Layer Z = f(X)', D.blue, [
    { t: 'On 4,096 finite inputs X:', fs: 13, fw: 700, fill: C.ink },
    { t: 'H(Z | X) = 0  (no noise injected)', mono: true, fs: 12.5, fill: D.blue.dark },
    { t: 'I(X; Z) = H(Z) = H(X) = 8.32 nats', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Injective weights never lose info', fs: 12.5, fill: C.ink },
    { t: '(On continuous X: I(X; Z) = ∞)', fs: 12, fill: C.muted },
  ]);

  o += titledPanel(280, 14, 240, 188, '30-Bin Histogram Z_bin (tanh vs ReLU)', D.orange, [
    { t: 'I(X; Z_bin) = H(Z_bin) − 0 = H(Z_bin)', mono: true, fs: 12, fw: 700, fill: D.orange.dark },
    { t: '• tanh: Large weights saturate ±1,', fs: 12.5, fill: C.ink },
    { t: '  merging inputs into 2 end bins ↓', fs: 12.5, fill: D.red.dark },
    { t: '• ReLU: Unbounded positive axis,', fs: 12.5, fill: C.ink },
    { t: '  no bin collapse, same test acc!', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 216, 500, 52, D.purple.tint, C.purple, [
    { t: 'Resolution (Saxe et al. 2018): The "compression phase" measured tanh bin clustering, not true I(X; Z)', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function variationalMiBounds() {
  const W = 540, H = 286;
  let o = '';
  o += titledPanel(20, 12, 240, 122, '1. Barber–Agakov (Decoder q)', D.blue, [
    { t: 'I(X;Y) ≥ H(X) + E_{P_{X,Y}}[log q(x|y)]', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
    { t: 'Requires known H(X) & normalized q', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 12, 240, 122, '2. Donsker–Varadhan (DV / MINE)', D.orange, [
    { t: 'E_{P_{X,Y}}[T] − log E_{P_X⊗P_Y}[e^T]', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark },
    { t: 'T* = pmi + c · Biased on minibatches', fs: 12.5, fill: D.red.dark },
  ]);

  o += titledPanel(20, 148, 240, 124, '3. NWJ (f-GAN KL Dual)', D.purple, [
    { t: 'E_{P_{X,Y}}[T] − e⁻¹ E_{P_X⊗P_Y}[e^T]', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
    { t: 'T* = 1 + pmi · Unbiased, Var = e^{2I}−1', fs: 12, fill: D.red.dark },
  ]);

  o += titledPanel(280, 148, 240, 124, '4. InfoNCE (Contrastive)', D.green, [
    { t: 'log N − L_NCE(T, N) ≤ log N', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'T* = pmi + c(x) · Low variance workhorse!', fs: 12, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function dvNwjTilting() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Exponential Tilting Proof:   q̃(x) = q(x) e^{T(x)} / Z,   Z = E_Q[e^T]   (P = P_{X,Y}, Q = P_X⊗P_Y)', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 70, 240, 136, 'Donsker–Varadhan (Exact KL Slack)', D.blue, [
    { t: 'D_KL(P ‖ Q) − (E_P[T] − log Z)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '= E_P[log(p / q̃)] = D_KL(P ‖ Q̃) ≥ 0', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
    { t: 'Tight iff q̃ = p  ⇔  T* = pmi + c', mono: true, fs: 12, fill: D.green.dark },
    { t: 'Batch −log(mean e^T) is biased high!', fs: 12, fill: D.red.dark },
  ]);

  o += titledPanel(280, 70, 240, 136, 'NWJ Linearization (log Z ≤ Z / e)', D.purple, [
    { t: 'Tangent bound at Z = e gives:', fs: 12.5, fill: C.ink },
    { t: '−log Z ≥ −Z/e  (drops log!)', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
    { t: 'Bound: E_P[T] − e⁻¹ E_Q[e^T]', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Tight at Z = e  ⇔  T* = 1 + pmi', mono: true, fs: 12, fill: D.green.dark },
  ]);

  o += card(20, 220, 500, 48, C.lamber, C.amber, [
    { t: 'Trade-off: NWJ removes DV’s log-mean bias on minibatches, but pins the shift c = 1', fs: 13, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function infonceBagProof() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 14, 500, 68, C.lblue, C.blue, [
    { t: 'Step 1 · Form Candidate Bag V = (Y₁, Y₂, …, Y_N) with Y₁ = Y⁺ and Y_{2:N} ~ P_Y^{⊗(N−1)}', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Chain Rule:  I(X; V) = I(X; Y₁) + I(X; Y_{2:N} | Y₁) = I(X; Y) + 0 = I(X; Y)', mono: true, fs: 13, fill: C.ink },
  ]);
  o += arrow(270, 84, 270, 104, C.blue);

  o += card(20, 108, 500, 68, D.purple.tint, C.purple, [
    { t: 'Step 2 · Rescaled Softmax γ(x, v) = N e^{T(x,y₁)} / ∑_{j} e^{T(x,y_j)} Normalizes q(v | x)', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'By permutation symmetry of p(v) = ∏ p_Y(y_j):   E_{p(v)}[γ(x, V)] = 1  ⇒  q(v|x) = p(v)γ(x,v)', mono: true, fs: 12, fill: C.ink },
  ]);
  o += arrow(270, 178, 270, 198, C.green);

  o += card(20, 202, 500, 68, C.lgreen, C.green, [
    { t: 'Step 3 · Apply Barber–Agakov Ratio Bound to (X, V)', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'I(X; Y) = I(X; V) ≥ E[log(q(V|X) / p(V))] = E[log γ(X, V)] = log N − L_NCE', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function starvedVsHitBatches() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, D.red.tint, D.red.base, [
    { t: 'The Rare-Pair Lottery Under P_X ⊗ P_Y When True I > log N  (Expected Hits = N e⁻ᴵ < 1)', fs: 13.5, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, 'Starved Batches (> 50% of Draws)', D.red, [
    { t: 'Zero plausible pairs land in batch:', fs: 12.5, fw: 700, fill: C.ink },
    { t: '(1/N) ∑ e^{pmi(xᵢ, yⱼ)} ≪ 1', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: '• NWJ reports ≈ I + 1 (overshoots!)', fs: 12.5, fill: C.ink },
    { t: '• DV −log(mean) explodes upward!', fs: 12.5, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(280, 68, 240, 138, 'Rare Hit Batches (Prob ~ N e⁻ᴵ)', D.orange, [
    { t: 'One lucky pair has e^{pmi} ≥ eᴵ ≫ N:', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Pulls population mean E_Q[e^{pmi}] = 1', mono: true, fs: 12, fill: C.ink },
    { t: 'Batch SD = √((e^{2I} − 1)/N)', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: '= 36 nats at I = 6, N = 128!', fs: 12.5, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 220, 500, 48, C.lgreen, C.green, [
    { t: 'Why InfoNCE Wins: Places positive e^{T(x,y⁺)} inside its own denominator → bounded ≤ log N!', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

export const diagrams = {
  // index.md (4)
  'ch37-info-ce-decomposition':          ceDecomposition,
  'ch37-info-three-pillars':             threePillars,
  'ch37-info-chapter-roadmap':           chapterRoadmap,
  'ch37-info-access-decision-tree':      accessDecisionTree,
  // mdl-information-theory.md (6)
  'ch37-info-loss-curve-landmarks':      lossCurveLandmarks,
  'ch37-info-shannon-entropy-axioms':    shannonEntropyAxioms,
  'ch37-info-entropy-chain-rule':        entropyChainRule,
  'ch37-info-gibbs-jensen-proof':        gibbsJensenProof,
  'ch37-info-irreducible-floor-split':   irreducibleFloorSplit,
  'ch37-info-proper-scoring-calibration': properScoringCalibration,
  // mdl-coding-compression.md (9)
  'ch37-info-time-machine-bpc':          timeMachineBpc,
  'ch37-info-kraft-dyadic-intervals':    kraftDyadicIntervals,
  'ch37-info-entropy-waste-split':       entropyWasteSplit,
  'ch37-info-shannon-vs-huffman':        shannonVsHuffman,
  'ch37-info-arith-two-bit-overhead':    arithTwoBitOverhead,
  'ch37-info-llm-universal-compressor':  llmUniversalCompressor,
  'ch37-info-mdl-two-part-code':         mdlTwoPartCode,
  'ch37-info-english-compression-ladder': englishCompressionLadder,
  'ch37-info-tokenizer-bpc-normalization': tokenizerBpcNormalization,
  // mdl-divergences-distances.md (7)
  'ch37-info-disjoint-point-masses':     disjointPointMasses,
  'ch37-info-divergence-vs-metric':      divergenceVsMetric,
  'ch37-info-f-divergence-catalog':      fDivergenceCatalog,
  'ch37-info-local-chi2-collapse':       localChi2Collapse,
  'ch37-info-forward-vs-reverse-kl':     forwardVsReverseKl,
  'ch37-info-pinsker-coarsening-proof':  pinskerCoarseningProof,
  'ch37-info-fgan-variational-game':     fganVariationalGame,
  // mdl-test-functions-transport.md (9)
  'ch37-info-shifted-gaussian-slopes':   shiftedGaussianSlopes,
  'ch37-info-ipm-auditor-classes':       ipmAuditorClasses,
  'ch37-info-test-class-goldilocks':     testClassGoldilocks,
  'ch37-info-mmd-mean-embedding':        mmdMeanEmbedding,
  'ch37-info-mmd-kernel-expansion':      mmdKernelExpansion,
  'ch37-info-kr-duality-bridge':         krDualityBridge,
  'ch37-info-w2-uncrossing-fid':         w2UncrossingFid,
  'ch37-info-sinkhorn-gibbs-scaling':    sinkhornGibbsScaling,
  'ch37-info-mmd-vs-w1-dimension':       mmdVsW1Dimension,
  // mdl-scores.md (7)
  'ch37-info-ebm-score-escape':          ebmScoreEscape,
  'ch37-info-fisher-mode-blindness':     fisherModeBlindness,
  'ch37-info-multiscale-noise-bridge':   multiscaleNoiseBridge,
  'ch37-info-stein-ibp-identity':        steinIbpIdentity,
  'ch37-info-hyvarinen-two-terms':       hyvarinenTwoTerms,
  'ch37-info-ksd-stein-kernel':          ksdSteinKernel,
  'ch37-info-master-loss-guide':         masterLossGuide,
  // mdl-mutual-information.md (4)
  'ch37-info-bsc-flipped-bit':           bscFlippedBit,
  'ch37-info-bivariate-gaussian-mi':     bivariateGaussianMi,
  'ch37-info-dpi-usable-info':           dpiUsableInfo,
  'ch37-info-ib-rate-distortion':        ibRateDistortion,
  // mdl-mi-estimation.md (6)
  'ch37-info-mcallester-stratos-wall':   mcallesterStratosWall,
  'ch37-info-tishby-tanh-binning':       tishbyTanhBinning,
  'ch37-info-variational-mi-bounds':     variationalMiBounds,
  'ch37-info-dv-nwj-tilting':            dvNwjTilting,
  'ch37-info-infonce-bag-proof':         infonceBagProof,
  'ch37-info-starved-vs-hit-batches':    starvedVsHitBatches,
};
