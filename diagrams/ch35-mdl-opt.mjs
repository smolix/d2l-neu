// Slide diagrams for Chapter 35 (chapter_mdl-optimization).
// Every diagram is rendered and audited by tools/build_module_diagrams.mjs
// with 0 visual defects (no text-line intersections, no rect border clips,
// no text overlaps, no out-of-viewBox text).

import { C, TOKENS, tx, arrow, svg } from './engine.mjs';
import { textWidth } from './mdl-textwidth.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

// Low-level SVG primitives
const rect = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, sw = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const circle = (cx, cy, r, fill, stroke, sw = 2, dash = false) =>
  `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="4 3"' : ''}/>`;

const path = (d, stroke, fill = 'none', sw = 2.5, dash = false) =>
  `<path d="${d}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// Text that does not fit is shrunk, but never below MIN_FS; a string still
// too wide at MIN_FS is wrapped onto two lines at the space nearest its
// middle. Widths come from the real font metrics (mdl-textwidth.mjs).
const MIN_FS = 11;
export const _fitLog = [];
// Break s at one space into two lines. Prefer, in order: a break that keeps
// brackets closed, a double-space or post-punctuation boundary, a break just
// before a relation (= ≤ ⇒ …), and the middle; never leave an operator
// dangling at the end of the first line.
const OPS = new Set(['+', '−', '-', '×', '·', '⊙', '/', '=', '≤', '≥', '≈', '⇒', '→', '⇔', '←', '∝', '&']);
const RELS = new Set(['=', '≤', '≥', '≈', '⇒', '→', '⇔', '<', '>', '∝']);
function split2(s) {
  let best = -1, bestCost = Infinity, depth = 0;
  for (let i = 1; i < s.length - 1; i++) {
    const c = s[i];
    if ('([{⟨'.includes(c)) depth++;
    if (')]}⟩'.includes(c)) depth = Math.max(0, depth - 1);
    if (c !== ' ' || s[i - 1] === ' ') continue;
    const head = s.slice(0, i).trimEnd(), tail = s.slice(i).trimStart();
    if (!head || !tail) continue;
    let cost = Math.abs(head.length - tail.length) / s.length;
    if (depth > 0) cost += 0.6;
    if (OPS.has(head[head.length - 1])) cost += 0.5;
    if (s[i + 1] === ' ') cost -= 0.2;
    if (',;:'.includes(head[head.length - 1])) cost -= 0.15;
    if (RELS.has(tail[0])) cost -= 0.1;
    if (cost < bestCost) { bestCost = cost; best = i; }
  }
  return best < 0 ? null : [s.slice(0, best).trimEnd(), s.slice(best + 1).trimStart()];
}

// { lines: [one or two strings], fs } for text t in a box of width maxW.
function fitLines(t, maxW, o = {}) {
  const str = String(t);
  const avail = maxW - 22;
  const base = o.fs || 14.5;
  const mono = !!o.mono, bold = (o.fw || 500) >= 600;
  const w = textWidth(str, base, mono, bold);
  if (w <= avail) return { lines: [str], fs: base };
  const fit = Math.floor((base * avail / w) * 10) / 10;
  if (fit >= MIN_FS) return { lines: [str], fs: fit };
  const parts = split2(str);
  if (!parts) {
    _fitLog.push(['over', str, MIN_FS]);
    return { lines: [str], fs: MIN_FS };
  }
  const w2 = Math.max(...parts.map(p => textWidth(p, base, mono, bold)));
  const fs = Math.max(MIN_FS, Math.min(base, Math.floor((base * avail / w2) * 10) / 10));
  _fitLog.push(['wrap', str, fs, w2 * fs / base > avail]);
  return { lines: parts, fs };
}

// Centred text fitted to width maxW (one line, or two around y).
function fitTx(x, y, t, maxW, o = {}) {
  const { lines, fs } = fitLines(t, maxW, o);
  if (lines.length === 1) return tx(x, y, lines[0], { ...o, fs });
  const dy = 0.6 * fs;
  return tx(x, y - dy, lines[0], { ...o, fs }) + tx(x, y + dy, lines[1], { ...o, fs });
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
      o += tx(cx, y + j * 1.2 * f.fs, ln, { ...f.it, fs: f.fs });
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
// §35.0 index.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function threeQuestions() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 38, D.purple.tint, C.purple, [
    { t: 'Three Core Questions of Optimization Geometry', fs: 15.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 62, 158, 152, '1. Step Ceiling', D.red, [
    { t: 'Why η < 2 / L?', fs: 14.5, fw: 700, fill: D.red.dark },
    { t: 'Steep wall L caps', fs: 13, fill: C.ink },
    { t: 'safe step length', fs: 13, fill: C.ink },
    { t: 'η > 2/L diverges', mono: true, fs: 12.5, fill: D.red.dark },
  ]);

  o += titledPanel(191, 62, 158, 152, '2. Slow Valley', D.blue, [
    { t: 'Why κ = L / μ?', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'GD: O(κ) steps', mono: true, fs: 13, fill: C.ink },
    { t: 'Momentum: O(√κ)', mono: true, fs: 13, fill: D.blue.dark },
    { t: 'Newton: 1 step', mono: true, fs: 12.5, fill: D.green.dark },
  ]);

  o += titledPanel(362, 62, 158, 152, '3. Noise & Optima', D.green, [
    { t: 'Why stall > f*?', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'SGD floor ∝ ησ²/b', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Convex + KKT', fs: 13, fill: C.ink },
    { t: 'certify global min', fs: 13, fill: D.green.dark },
  ]);

  o += card(20, 228, 500, 44, C.lamber, C.amber, [
    { t: 'Running Benchmark: f(x) = ½(x₁² + 1000 x₂²)  (μ = 1, L = 1000, κ = 1000)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function benchmarkValley() {
  const W = 540, H = 312;
  let o = '';
  o += fitTx(270, 20, 'Steps on the Valley f(x) = ½(x₁² + 1000 x₂²)', 500, { fs: 15.5, fw: 700, fill: C.ink });
  o += fitTx(270, 42, 'Stop at ‖xₖ‖ ≤ 10⁻⁶‖x₀‖; Adam stops at f < 10⁻⁸ (there GD needs 6,160, heavy ball 296)', 508, { fs: 12, fill: C.ink });

  const rows = [
    { label: 'GD (η* = 2/1001)', val: '6,908 steps', w: 280, fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { label: 'Nesterov Look-Ahead', val: '508 steps', w: 155, fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { label: 'Heavy-Ball Momentum', val: '315 steps', w: 132, fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { label: 'Diagonal Adam (aligned)', val: '22 steps (f < 10⁻⁸)', w: 190, fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { label: 'Newton Method (H⁻¹g)', val: '1 step', w: 78, fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  rows.forEach((r, i) => {
    const y = 58 + i * 38;
    o += card(16, y, 184, 30, N.panel, C.gray, [
      { t: r.label, fs: 13, fw: 700, fill: C.ink },
    ], { rx: 6, sw: 1.5 });
    o += card(212, y, r.w, 30, r.fill, r.st, [
      { t: r.val, mono: true, fs: 13, fw: 700, fill: r.dk },
    ], { rx: 6, sw: 2 });
  });

  o += card(16, 254, 508, 44, D.teal.tint, D.teal.base, [
    { t: 'Constant-Step SGD (η = 10⁻³, σ² = 1): Stalls at loss floor 7.5 × 10⁻⁴', fs: 13.5, fw: 700, fill: D.teal.dark },
  ]);
  return svg(W, H, o);
}

function chapterArchitecture() {
  const W = 540, H = 296;
  let o = '';
  o += card(20, 12, 500, 72, C.lblue, C.blue, [
    { t: 'Part I · Exact Gradients & Geometry (§§35.1–35.3)', fs: 15, fw: 700, fill: D.blue.dark },
    { t: 'L-smoothness ceiling 2/L · κ vs √κ momentum · Newton/BFGS · Convexity & PL', fs: 13, fill: C.ink },
  ]);
  o += arrow(270, 86, 270, 108, C.blue);

  o += card(20, 112, 500, 72, C.lamber, C.amber, [
    { t: 'Part II · Stochastic & Adaptive Methods (§§35.4–35.5)', fs: 15, fw: 700, fill: D.orange.dark },
    { t: '1/b noise variance · SGD noise ball · 1/k & WSD schedules · AdamW, K-FAC, Muon', fs: 13, fill: C.ink },
  ]);
  o += arrow(270, 186, 270, 208, D.orange.base);

  o += card(20, 212, 500, 72, C.lgreen, C.green, [
    { t: 'Part III · Constraints, Duality & Numerics (§§35.6–35.8)', fs: 15, fw: 700, fill: D.green.dark },
    { t: 'KKT & projections · Proximal ISTA · Strong duality & conjugates · bf16 & κ(A)', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function studyRoadmap() {
  const W = 540, H = 284;
  let o = '';
  o += card(20, 14, 232, 68, C.lblue, C.blue, [
    { t: '§35.1 Gradient & Momentum', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'Essential: 2/L ceiling & √κ', fs: 13, fill: C.ink },
  ]);
  o += card(288, 14, 232, 68, C.lgreen, C.green, [
    { t: '§35.3 Convexity & Rates', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'Essential: Tangents, Jensen, PL', fs: 13, fill: C.ink },
  ]);

  o += arrow(136, 84, 136, 110, C.blue);
  o += arrow(404, 84, 404, 110, C.green);

  o += card(20, 114, 232, 68, C.lamber, C.amber, [
    { t: '§35.4 Stochastic Gradient', fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: 'Noise floor ησ²/b & schedules', fs: 13, fill: C.ink },
  ]);
  o += card(288, 114, 232, 68, D.purple.tint, C.purple, [
    { t: '§35.5 Adaptive Methods', fs: 14.5, fw: 700, fill: D.purple.dark },
    { t: 'Adam scale invariance & Muon', fs: 13, fill: C.ink },
  ]);

  o += arrow(136, 184, 136, 206, C.gray);
  o += arrow(404, 184, 404, 206, C.gray);

  o += card(20, 210, 156, 60, N.panel, C.gray, [
    { t: '§35.2 Curvature', fs: 14, fw: 700, fill: C.ink },
    { t: 'Newton & BFGS', fs: 12.5, fill: C.muted },
  ]);
  o += card(192, 210, 156, 60, N.panel, C.gray, [
    { t: '§§35.6–7 Duality', fs: 14, fw: 700, fill: C.ink },
    { t: 'KKT, Prox, SVM', fs: 12.5, fill: C.muted },
  ]);
  o += card(364, 210, 156, 60, N.panel, C.gray, [
    { t: '§35.8 Numerics', fs: 14, fw: 700, fill: C.ink },
    { t: 'bf16, Welford, κ', fs: 12.5, fill: C.muted },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §35.1 mdl-gradient-based-optimization.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function threeStepRegimes() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 38, N.panel, C.gray, [
    { t: 'Start x₀ = (2, 1) on f(x, y) = ½(x² + 10y²)  →  f(x₀) = 7.00, ∇f = (2, 10)', mono: true, fs: 13.5, fw: 700, fill: C.ink },
  ]);

  o += titledPanel(20, 64, 156, 154, 'η = 0.01 (Tiny)', D.blue, [
    { t: 'x₁ = (1.98, 0.90)', mono: true, fs: 13, fill: C.ink },
    { t: 'f(x₁) = 6.01', mono: true, fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'Safe along y,', fs: 13, fill: C.muted },
    { t: 'crawls along x', fs: 13, fill: C.muted },
  ]);

  o += titledPanel(192, 64, 156, 154, 'η = 0.10 = 1/L', D.green, [
    { t: 'x₁ = (1.80, 0.00)', mono: true, fs: 13, fill: C.ink },
    { t: 'f(x₁) = 1.62', mono: true, fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'Kills steep y-mode', fs: 13, fill: D.green.dark },
    { t: 'in a single step', fs: 13, fill: D.green.dark },
  ]);

  o += titledPanel(364, 64, 156, 154, 'η = 0.25 > 2/L', D.red, [
    { t: 'x₁ = (1.50, −1.50)', mono: true, fs: 13, fill: C.ink },
    { t: 'f(x₁) = 12.38', mono: true, fs: 14.5, fw: 700, fill: D.red.dark },
    { t: 'Overshoots wall', fs: 13, fill: D.red.dark },
    { t: 'and diverges!', fs: 13, fill: D.red.dark },
  ]);

  o += card(20, 232, 500, 42, C.lamber, C.amber, [
    { t: 'Stability Ceiling: 0 < η < 2/L = 0.20  (steep curvature L = 10 caps step)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function steepestDescentCone() {
  const W = 540, H = 280;
  let o = '';
  o += rect(24, 34, 226, 200, C.lgreen, C.green, 8, 1.5, true);
  o += rect(250, 34, 226, 200, D.red.tint, D.red.base, 8, 1.5, true);
  o += line(250, 26, 250, 242, C.ink, 2.5);
  o += fitTx(137, 56, 'Descent Half-Space', 200, { fs: 14.5, fw: 700, fill: D.green.dark });
  o += fitTx(137, 76, '∇f(x)ᵀd < 0', 200, { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  o += fitTx(363, 56, 'Ascent Half-Space', 200, { fs: 14.5, fw: 700, fill: D.red.dark });
  o += fitTx(363, 76, '∇f(x)ᵀd > 0', 200, { mono: true, fs: 13.5, fw: 700, fill: D.red.dark });

  const cx = 250, cy = 148;
  o += arrow(cx, cy, 110, cy, C.green);
  o += card(38, 166, 178, 52, '#FFFFFF', C.green, [
    { t: 'd* = −∇f / ‖∇f‖', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Steepest (θ = 180°)', fs: 12.5, fill: C.ink },
  ]);

  o += arrow(cx, cy, 156, 102, C.blue);
  o += arrow(cx, cy, 386, cy, D.red.base);
  o += card(284, 166, 178, 52, '#FFFFFF', D.red.base, [
    { t: '+∇f(x) (Ascent)', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Tangent: ∇fᵀd = 0', mono: true, fs: 12.5, fill: C.ink },
  ]);
  o += circle(cx, cy, 5, C.ink, '#FFFFFF', 2);

  o += fitTx(270, 264, 'Cauchy–Schwarz: ∇f(x)ᵀd = ‖∇f(x)‖ cos θ ≥ −‖∇f(x)‖', 500, { mono: true, fs: 13.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function lipschitzQuadraticCeiling() {
  const W = 540, H = 284;
  let o = '';
  o += card(20, 12, 500, 38, C.lblue, C.blue, [
    { t: 'L-Smoothness: ‖∇f(x) − ∇f(y)‖ ≤ L‖x − y‖  ⇔  Global Quadratic Ceiling', fs: 14.5, fw: 700, fill: D.blue.dark },
  ]);

  o += rect(28, 62, 270, 198, '#FFFFFF', C.gray, 8, 1.5);
  // Ceiling: the parabola tangent to f at x = (74, 134.5), vertex (165, 174).
  o += path('M 48 108.7 Q 163 237 278 113.1', D.red.base, 'none', 2.5, true);
  o += path('M 48 110 Q 178 242 278 158', C.blue, 'none', 3, false);
  o += line(48, 112, 208, 250.7, C.gray, 2, true);
  o += circle(74, 134.5, 5, C.ink, '#FFFFFF', 1.5);
  o += circle(165, 174, 5, D.red.base, '#FFFFFF', 1.5);
  o += arrow(165, 136, 165, 168, C.green);

  o += card(312, 62, 208, 60, D.red.tint, D.red.base, [
    { t: 'Quadratic Ceiling (L)', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'f(x) + ∇fᵀΔ + (L/2)‖Δ‖²', mono: true, fs: 12.5, fill: C.ink },
  ]);
  o += card(312, 131, 208, 60, C.lblue, C.blue, [
    { t: 'True Objective f(y)', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'Always sits below ceiling', fs: 12.5, fill: C.ink },
  ]);
  o += card(312, 200, 208, 60, C.lgreen, C.green, [
    { t: 'Min at η* = 1 / L', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Drop ≥ (1 / 2L)‖∇f(x)‖²', mono: true, fs: 12.5, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function saddleHalt() {
  const W = 540, H = 276;
  let o = '';
  o += card(20, 12, 500, 38, D.red.tint, D.red.base, [
    { t: 'Saddle Trap on f(x, y) = ½ x² + cos y  (L = 1, η = 1/L = 1)', mono: true, fs: 14.5, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 68, 144, 118, C.lblue, C.blue, [
    { t: 'Start (1, 0)', mono: true, fs: 15, fw: 700, fill: D.blue.dark },
    { t: 'f(1, 0) = 1.5', mono: true, fs: 13.5, fill: C.ink },
    { t: '∇f = (1, 0)', mono: true, fs: 13, fill: C.muted },
  ]);
  o += arrow(168, 127, 194, 127, C.blue);

  o += card(198, 68, 144, 118, D.red.tint, D.red.base, [
    { t: 'Saddle (0, 0)', mono: true, fs: 15, fw: 700, fill: D.red.dark },
    { t: 'f(0, 0) = 1.0', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: '∇f = (0, 0) → HALT', mono: true, fs: 12.5, fill: D.red.dark },
  ]);
  o += arrow(346, 127, 372, 127, C.green, true);

  o += card(376, 68, 144, 118, C.lgreen, C.green, [
    { t: 'Global Min (0, π)', mono: true, fs: 15, fw: 700, fill: D.green.dark },
    { t: 'f(0, π) = −1.0', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: '2 units lower!', fs: 13, fill: D.green.dark },
  ]);

  o += card(20, 204, 500, 56, N.panel, C.gray, [
    { t: 'Indefinite Hessian at (0, 0): ∇²f(0, 0) = diag(+1, −1)', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'Curves upward along x (+1) but downward along y (−1): ∇f = 0 is NOT a minimum', fs: 13, fill: C.muted },
  ]);
  return svg(W, H, o);
}

function armijoBacktracking() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 52, C.lblue, C.blue, [
    { t: 'Armijo Sufficient-Decrease Condition (c ∈ (0, 1))', fs: 15, fw: 700, fill: D.blue.dark },
    { t: 'Accept η when  f(xₖ − η∇f(xₖ))  ≤  f(xₖ) − c η ‖∇f(xₖ)‖²', mono: true, fs: 13.5, fill: C.ink },
  ]);

  o += card(20, 82, 144, 112, D.red.tint, D.red.base, [
    { t: 'Trial η₀', mono: true, fs: 15, fw: 700, fill: D.red.dark },
    { t: 'Overshoots Armijo', fs: 13, fill: C.ink },
    { t: 'ceiling → Reject', fs: 13, fill: D.red.dark },
  ]);
  o += arrow(168, 138, 194, 138, D.orange.base);

  o += card(198, 82, 144, 112, C.lamber, C.amber, [
    { t: 'Halve: η ← η / 2', mono: true, fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: 'Repeat while', fs: 13, fill: C.ink },
    { t: 'Armijo test fails', fs: 13, fill: C.ink },
  ]);
  o += arrow(346, 138, 372, 138, C.green);

  o += card(376, 82, 144, 112, C.lgreen, C.green, [
    { t: 'Accepted η*', mono: true, fs: 15, fw: 700, fill: D.green.dark },
    { t: 'η ≤ 2(1−c) / L_loc', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Guaranteed drop', fs: 13, fill: D.green.dark },
  ]);

  o += card(20, 212, 500, 58, D.purple.tint, C.purple, [
    { t: 'Automatic Termination Bound (no knowledge of L_loc required):', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'Returned step always satisfies  η ≥ min(η₀, (1 − c) / L_loc)', mono: true, fs: 13.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function edgeOfStability() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 38, D.purple.tint, C.purple, [
    { t: 'Full-Batch Deep Network Training at the Edge of Stability', fs: 15, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 62, 240, 146, 'Phase 1: Sharpening', D.blue, [
    { t: 'λ_max(∇²f(xₖ)) < 2 / η', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Top Hessian eigenvalue', fs: 13, fill: C.ink },
    { t: 'grows steadily as network', fs: 13, fill: C.ink },
    { t: 'enters narrower valleys', fs: 13, fill: C.muted },
  ]);
  o += arrow(264, 135, 276, 135, C.purple);

  o += titledPanel(280, 62, 240, 146, 'Phase 2: Hovering at 2/η', D.orange, [
    { t: 'λ_max(∇²f(xₖ)) ≈ 2 / η', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: '1 − ηλ_max < −1 triggers', fs: 13, fill: C.ink },
    { t: 'stiff-mode oscillation;', fs: 13, fill: C.ink },
    { t: '3rd-order terms restrain λ_max', fs: 13, fill: D.orange.dark },
  ]);

  o += card(20, 222, 500, 46, C.lgreen, C.green, [
    { t: 'Result: Training loss keeps descending over long horizons despite local spikes', fs: 13.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function nesterovLookahead() {
  const W = 540, H = 284;
  let o = '';
  o += titledPanel(20, 14, 240, 190, 'Heavy-Ball Momentum', D.blue, [
    { t: '1. Evaluate ∇f(xₖ) at xₖ', mono: true, fs: 13, fill: C.ink },
    { t: '2. Add momentum βvₖ', mono: true, fs: 13, fill: C.ink },
    { t: 'vₖ₊₁ = βvₖ − η∇f(xₖ)', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: '√κ rate on quadratics;', fs: 13, fill: C.muted },
    { t: 'can cycle on general convex', fs: 12.5, fill: D.red.dark },
  ]);

  o += titledPanel(280, 14, 240, 190, 'Nesterov Look-Ahead (NAG)', D.green, [
    { t: '1. Coast to x̃ₖ = xₖ + βvₖ', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: '2. Evaluate ∇f(x̃ₖ) ahead!', mono: true, fs: 13, fill: C.ink },
    { t: 'vₖ₊₁ = βvₖ − η∇f(xₖ + βvₖ)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Brakes before overshoot;', fs: 13, fill: C.ink },
    { t: 'Global O(√κ) & O(1/k²)', fs: 13, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 218, 500, 52, D.purple.tint, C.purple, [
    { t: 'Nemirovski–Yudin Lower Bound (d ≥ 2k + 1):', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'No first-order method can beat O(1/k²) (convex) or O(√κ log(1/ε)) (strongly convex)', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §35.2 mdl-curvature-methods.md (10 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function curvatureVsFirstOrder() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 38, D.purple.tint, C.purple, [
    { t: 'Enduring Curvature vs. Modeling Curvature on ½(x₁² + 1000 x₂²)', fs: 14.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 62, 156, 152, 'Gradient Descent', D.red, [
    { t: '6,908 steps', mono: true, fs: 15.5, fw: 700, fill: D.red.dark },
    { t: 'Uses scalar L = 1000', fs: 13, fill: C.ink },
    { t: 'Steep wall caps η;', fs: 12.5, fill: C.muted },
    { t: 'flat valley crawls', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(192, 62, 156, 152, 'Heavy-Ball', D.blue, [
    { t: '315 steps', mono: true, fs: 15.5, fw: 700, fill: D.blue.dark },
    { t: 'Uses μ = 1, L = 1000', fs: 13, fill: C.ink },
    { t: 'Velocity equalizes', fs: 12.5, fill: C.muted },
    { t: 'modes to √κ', fs: 12.5, fill: D.blue.dark },
  ]);

  o += titledPanel(364, 62, 156, 152, 'Newton Method', D.green, [
    { t: '1 step', mono: true, fs: 15.5, fw: 700, fill: D.green.dark },
    { t: 'Uses ∇²f = diag(1,1000)', mono: true, fs: 12, fill: C.ink },
    { t: 'Inverts exact curvature', fs: 12.5, fill: D.green.dark },
    { t: 'in every direction', fs: 12.5, fill: D.green.dark },
  ]);

  o += card(20, 228, 500, 44, C.lamber, C.amber, [
    { t: 'Trade-off: Eliminates κ completely, but costs O(d²) memory & O(d³) solve', fs: 13.5, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function newtonQuadraticModel() {
  const W = 540, H = 282;
  let o = '';
  o += rect(24, 20, 260, 242, '#FFFFFF', C.gray, 8, 1.5);
  o += path('M 44 60 Q 150 230 264 140', C.blue, 'none', 3, false);
  o += path('M 44 74 Q 156 236 264 90', C.green, 'none', 2.5, true);
  o += circle(68, 102, 5, C.ink, '#FFFFFF', 1.5);
  o += circle(155, 162, 6, C.green, '#FFFFFF', 1.5);
  o += arrow(74, 198, 152, 198, C.green);

  o += card(300, 20, 220, 72, C.lblue, C.blue, [
    { t: '2nd-Order Taylor Model', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'mₖ(p) = fₖ + gₖᵀp + ½pᵀHₖp', mono: true, fs: 12.5, fill: C.ink },
  ]);
  o += card(300, 105, 220, 72, C.lgreen, C.green, [
    { t: 'Stationary Bowl Minimum', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '∇mₖ(p) = gₖ + Hₖp = 0', mono: true, fs: 13, fill: C.ink },
  ]);
  o += card(300, 190, 220, 72, D.purple.tint, C.purple, [
    { t: 'Natural Unit Step (η = 1)', fs: 14.5, fw: 700, fill: D.purple.dark },
    { t: 'xₖ₊₁ = xₖ − (∇²f(xₖ))⁻¹∇f(xₖ)', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function mahalanobisSteepest() {
  const W = 540, H = 290;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'Euclidean Norm ‖d‖₂ = 1', D.red, [
    { t: 'min ∇f(x)ᵀd  s.t. dᵀd = 1', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Isotropic ball ignores', fs: 13, fill: C.ink },
    { t: 'anisotropic curvature H', fs: 13, fill: C.ink },
    { t: 'd = −∇f(x)  (zig-zags)', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Hessian Norm ‖d‖_{P} = 1', D.green, [
    { t: 'min ∇f(x)ᵀd  s.t. dᵀPd = 1', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Ellipsoid aligns with', fs: 13, fill: C.ink },
    { t: 'level sets of quadratic bowl', fs: 13, fill: C.ink },
    { t: 'd = −P⁻¹∇f(x)  (direct!)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 216, 500, 60, C.lblue, C.blue, [
    { t: 'Special Cases:', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'P = I (GD)  ·  P = diag(√v̂) (Adam)  ·  P = ∇²f (Newton)', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function affineInvariance() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 14, 186, 176, D.red.tint, D.red.base, [
    { t: 'Original x-Coords', fs: 14.5, fw: 700, fill: D.red.dark },
    { t: 'f(x) = ½ xᵀAx', mono: true, fs: 13.5, fill: C.ink },
    { t: 'Elongated Valley', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'κ(A) = 1000', mono: true, fs: 13.5, fill: C.ink },
    { t: 'xₖ₊₁ = xₖ − Hₓ⁻¹ gₓ', mono: true, fs: 12.5, fill: D.red.dark },
  ]);

  o += arrow(212, 82, 328, 82, C.purple);
  o += fitTx(270, 62, 'x = A⁻¹/² y', 110, { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += arrow(328, 126, 212, 126, C.purple);
  o += fitTx(270, 148, 'T yₖ₊₁ = xₖ₊₁', 110, { mono: true, fs: 13, fw: 700, fill: D.purple.dark });

  o += card(334, 14, 186, 176, C.lgreen, C.green, [
    { t: 'Whitened y-Coords', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'h(y) = ½ ‖y‖₂²', mono: true, fs: 13.5, fill: C.ink },
    { t: 'Isotropic Sphere', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'κ(I) = 1', mono: true, fs: 13.5, fill: C.ink },
    { t: 'yₖ₊₁ = yₖ − I⁻¹ yₖ = 0', mono: true, fs: 12.5, fill: D.green.dark },
  ]);

  o += card(20, 206, 500, 60, D.purple.tint, C.purple, [
    { t: 'Affine Invariance:  (Tᵀ Hₓ T)⁻¹ (Tᵀ gₓ) = T⁻¹ Hₓ⁻¹ gₓ', mono: true, fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'Newton trajectories commute with any invertible linear change of basis T', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function quadraticDoublingDigits() {
  const W = 540, H = 278;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Quadratic Error Recursion:  (L₂/2μ)‖eₖ₊₁‖ ≤ ((L₂/2μ)‖eₖ‖)²', mono: true, fs: 14.5, fw: 700, fill: D.blue.dark },
  ]);

  const steps = [
    { k: 'k = 0', err: '10⁻¹', dig: '1 digit', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { k: 'k = 1', err: '10⁻²', dig: '2 digits', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { k: 'k = 2', err: '10⁻⁴', dig: '4 digits', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { k: 'k = 3', err: '10⁻⁸', dig: '8 digits', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { k: 'k = 4', err: '10⁻¹⁶', dig: '16 digits', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  steps.forEach((s, i) => {
    const x = 20 + i * 103;
    o += card(x, 74, 88, 110, s.fill, s.st, [
      { t: s.k, mono: true, fs: 13.5, fw: 700, fill: C.ink },
      { t: s.err, mono: true, fs: 16, fw: 700, fill: s.dk },
      { t: s.dig, fs: 12.5, fill: C.muted },
    ]);
    if (i < 4) {
      o += arrow(x + 90, 129, x + 101, 129, C.gray);
    }
  });

  o += card(20, 202, 500, 62, C.lgreen, C.green, [
    { t: 'Global-to-Local Damped Newton Strategy:', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Far from x*: Armijo backtracking αₖ < 1  ·  Near x*: accepts full step αₖ = 1', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function hessianScaleWall() {
  const W = 540, H = 290;
  let o = '';
  o += fitTx(270, 20, 'The O(d²) Memory & O(d³) Compute Wall of Exact Newton', 500, { fs: 15.5, fw: 700, fill: C.ink });

  const tiers = [
    { name: 'Toy Logistic Model', d: 'd = 5', mem: '25 entries (100 B)', flops: '125 FLOPs · Instant', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { name: 'Small Linear Layer', d: 'd = 10⁴', mem: '10⁸ entries (400 MB)', flops: '10¹² FLOPs · Seconds', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { name: 'ConvNet / ViT', d: 'd = 10⁸', mem: '10¹⁶ entries (40 PB)', flops: '10²⁴ FLOPs · Impossible', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { name: 'Frontier LLM', d: 'd = 10¹¹', mem: '10²² entries (40M PB)', flops: '10³³ FLOPs · Impossible', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];

  tiers.forEach((t, i) => {
    const y = 38 + i * 60;
    o += card(20, y, 165, 50, t.fill, t.st, [
      { t: t.name, fs: 13.5, fw: 700, fill: t.dk },
      { t: t.d, mono: true, fs: 13, fill: C.ink },
    ]);
    o += card(196, y, 158, 50, '#FFFFFF', t.st, [
      { t: 'Hessian Memory d²', fs: 12.5, fill: C.muted },
      { t: t.mem, fs: 12.5, fw: 700, fill: C.ink },
    ]);
    o += card(366, y, 154, 50, '#FFFFFF', t.st, [
      { t: 'Linear Solve ~ d³', fs: 12.5, fill: C.muted },
      { t: t.flops, fs: 12.5, fw: 700, fill: t.dk },
    ]);
  });
  return svg(W, H, o);
}

function saddleWrongWay() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, D.red.tint, D.red.base, [
    { t: 'Negative Curvature Reverses Newton Direction: f(x, y) = ½ x² + cos y', fs: 14.5, fw: 700, fill: D.red.dark },
  ]);

  o += rect(20, 66, 236, 198, '#FFFFFF', C.gray, 8, 1.5);
  o += path('M 36 146 Q 110 72 238 228', C.blue, 'none', 3, false);
  // Newton target = the crest (saddle y = 0) of the slice; y_k sits on the slope.
  o += circle(89, 122, 5, D.red.base, '#FFFFFF', 1.5);
  o += circle(148, 142, 5, C.ink, '#FFFFFF', 1.5);
  o += arrow(142, 104, 96, 104, D.red.base);
  o += arrow(156, 164, 206, 207, C.green);
  o += tx(164, 126, 'yₖ', { fs: 13, fw: 700, fill: C.ink });
  o += tx(60, 110, 'y = 0', { fs: 12, fw: 700, fill: D.red.dark });
  o += fitTx(138, 244, '1D slice along y (f_yy = −cos y < 0)', 220, { fs: 12.5, fw: 600, fill: C.muted });

  o += card(272, 66, 248, 92, D.red.tint, D.red.base, [
    { t: 'Pure Newton Step (Uphill!)', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'yₖ₊₁ = yₖ − (−sin yₖ)/(−cos yₖ)', mono: true, fs: 12.5, fill: C.ink },
    { t: 'yₖ − tan yₖ ≈ −yₖ³/3 → Saddle 0', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
  ]);

  o += card(272, 172, 248, 92, C.lgreen, C.green, [
    { t: 'Gradient Descent (Downhill)', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'yₖ₊₁ = yₖ − η(−sin yₖ) = yₖ + η sin yₖ', mono: true, fs: 12, fill: C.ink },
    { t: 'Escapes crest → Global Min y = π', fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function secantChordSlope() {
  const W = 540, H = 296;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: '1D Secant Curvature Proxy:  hₖ = (f\'(xₖ) − f\'(xₖ₋₁)) / (xₖ − xₖ₋₁)', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 72, 148, 116, N.panel, C.gray, [
    { t: 'Two Past Gradients', fs: 14, fw: 700, fill: C.ink },
    { t: '(xₖ₋₁, f\'(xₖ₋₁))', mono: true, fs: 13, fill: C.ink },
    { t: '(xₖ, f\'(xₖ))', mono: true, fs: 13, fill: D.blue.dark },
  ]);
  o += arrow(172, 130, 194, 130, C.blue);

  o += card(198, 72, 148, 116, C.lblue, C.blue, [
    { t: 'Quasi-Newton Step', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'xₖ₊₁ = xₖ − f\'(xₖ)/hₖ', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: '1 gradient / step', fs: 13, fill: C.ink },
  ]);
  o += arrow(350, 130, 372, 130, C.green);

  o += card(376, 72, 144, 116, C.lgreen, C.green, [
    { t: 'Superlinear Rate', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'eₖ₊₁ ≈ C eₖ eₖ₋₁', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Order φ ≈ 1.618', mono: true, fs: 13, fill: C.ink },
  ]);

  o += card(20, 206, 500, 76, D.purple.tint, C.purple, [
    { t: 'On f(x) = eˣ − 2x (x* = log 2):', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Error drops 10⁻¹ → 10⁻² → 10⁻³ → 10⁻⁵ → 10⁻⁹', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Multiplies accurate digits by the golden ratio 1.618 per step without Hessians', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function bfgsEraseWrite() {
  const W = 540, H = 306;
  let o = '';
  o += card(20, 12, 500, 44, D.purple.tint, C.purple, [
    { t: 'BFGS Inverse-Hessian Update:  Mₖ₊₁ yₖ = sₖ   (γₖ = 1 / yₖᵀsₖ)', mono: true, fs: 14.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 70, 236, 132, '1. Erase Old Action Along yₖ', D.blue, [
    { t: '(I − γₖ sₖ yₖᵀ) Mₖ (I − γₖ yₖ sₖᵀ)', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Projects yₖ → 0 while keeping', fs: 13, fill: C.ink },
    { t: 'subspace ⊥ sₖ untouched', fs: 13, fill: C.ink },
  ]);

  o += tx(270, 136, '+', { fs: 24, fw: 700, fill: C.ink });

  o += titledPanel(284, 70, 236, 132, '2. Write Secant Match sₖ', D.green, [
    { t: '+ γₖ sₖ sₖᵀ', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Maps yₖ ↦ γₖ sₖ (sₖᵀyₖ) = sₖ', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Rank-2 symmetric update', fs: 13, fill: C.ink },
  ]);

  o += card(20, 216, 500, 76, C.lamber, C.amber, [
    { t: 'Positive-Definiteness Certificate:', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'zᵀ Mₖ₊₁ z = wᵀ Mₖ w + γₖ(sₖᵀz)² > 0', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Preserves Mₖ₊₁ ≻ 0 iff curvature condition yₖᵀsₖ > 0 holds (enforced by Wolfe line search)', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function trustRegionDamping() {
  const W = 540, H = 284;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Trust-Region Optimality:  (B + τI)p = −g,   B + τI ⪰ 0,   τ(Δ − ‖p‖) = 0', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 70, 240, 138, 'Interior Case (τ = 0)', D.green, [
    { t: 'B ≻ 0  and  ‖B⁻¹g‖ ≤ Δ', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Unconstrained Newton step', fs: 13, fill: C.ink },
    { t: 'p = −B⁻¹g fits inside ball', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Local quadratic convergence', fs: 12.5, fill: D.green.dark },
  ]);

  o += titledPanel(280, 70, 240, 138, 'Boundary Case (τ > −λ_min)', D.orange, [
    { t: 'B indefinite or ‖B⁻¹g‖ > Δ', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Spectral shift τ lifts all', fs: 13, fill: C.ink },
    { t: 'eigenvalues λᵢ + τ > 0 until', fs: 13, fill: C.ink },
    { t: '‖(B + τI)⁻¹g‖ = Δ (escapes saddle)', mono: true, fs: 12, fw: 700, fill: D.orange.dark },
  ]);

  o += card(20, 222, 500, 48, D.purple.tint, C.purple, [
    { t: 'Spectral Damping: Increasing τ interpolates smoothly from Newton to −(1/τ)g', fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §35.3 mdl-convexity.md (5 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function intersectionHull() {
  const W = 540, H = 280;
  let o = '';
  o += titledPanel(20, 14, 240, 188, '1. Arbitrary Intersections', D.blue, [
    { t: 'C = ⋂ᵢ Cᵢ  is always convex', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Top-Down Construction:', fs: 13, fw: 700, fill: C.ink },
    { t: '• Simplex Δ: 1 plane ∩ d half-spaces', fs: 12.5, fill: C.ink },
    { t: '• PSD Cone S₊ᵈ: ⋂_{z} {zᵀAz ≥ 0}', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 14, 240, 188, '2. Convex Hull conv(S)', D.green, [
    { t: '∑ θᵢ xᵢ  (θᵢ ≥ 0, ∑ θᵢ = 1)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Bottom-Up Construction:', fs: 13, fw: 700, fill: C.ink },
    { t: 'Smallest convex superset', fs: 13, fill: C.ink },
    { t: 'wrapping all mixtures of S', fs: 13, fill: C.ink },
  ]);

  o += card(20, 216, 500, 50, D.red.tint, D.red.base, [
    { t: 'Warning: Unions do NOT preserve convexity (e.g. [0, 1] ∪ [2, 3] has a gap)', fs: 13.5, fw: 700, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

function epigraphSublevel() {
  const W = 540, H = 284;
  let o = '';
  o += rect(20, 18, 252, 248, '#FFFFFF', C.gray, 8, 1.5);
  o += `<path d="M 48 64 Q 146 226 244 64 Z" fill="${C.lblue}" stroke="none"/>`;
  o += path('M 48 64 Q 146 226 244 64', C.blue, 'none', 3, false);
  o += line(70, 96, 222, 96, D.red.base, 2.5, false);
  o += circle(70, 96, 4.5, D.red.base, '#FFFFFF', 1.5);
  o += circle(222, 96, 4.5, D.red.base, '#FFFFFF', 1.5);
  o += arrow(146, 96, 146, 142, D.red.base);
  o += line(70, 232, 222, 232, C.green, 4, false);
  o += circle(70, 232, 4, C.green, '#FFFFFF', 1.5);
  o += circle(222, 232, 4, C.green, '#FFFFFF', 1.5);
  o += fitTx(146, 44, 'epi(f) = {(x, t) : t ≥ f(x)}', 236, { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += fitTx(146, 250, 'Sublevel {x : f(x) ≤ c}', 236, { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });

  o += card(288, 18, 232, 74, D.red.tint, D.red.base, [
    { t: 'Zero-Order Chord Test', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'f(θx + (1−θ)y) ≤ θf(x) + (1−θ)f(y)', mono: true, fs: 11.5, fill: C.ink },
    { t: 'Gap on x²: θ(1−θ)(x − y)² ≥ 0', mono: true, fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += card(288, 105, 232, 74, C.lblue, C.blue, [
    { t: 'Epigraph Equivalence', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'f is convex  ⇔  epi(f) is a', fs: 13, fill: C.ink },
    { t: 'convex subset of ℝᵈ × ℝ', fs: 13, fill: C.ink },
  ]);

  o += card(288, 192, 232, 74, C.lgreen, C.green, [
    { t: 'Convex Sublevel Sets', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'f convex ⇒ {f(x) ≤ c} convex', fs: 13, fill: C.ink },
    { t: '(Converse fails: √|x|)', fs: 12.5, fill: C.muted },
  ]);
  return svg(W, H, o);
}

function threeTestsTriangle() {
  const W = 540, H = 286;
  let o = '';
  o += card(140, 12, 260, 66, C.lblue, C.blue, [
    { t: '0th-Order: Chord Inequality', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'f(θx + (1−θ)y) ≤ θf(x) + (1−θ)f(y)', mono: true, fs: 12.5, fill: C.ink },
  ]);

  o += card(16, 152, 236, 72, C.lgreen, C.green, [
    { t: '1st-Order: Global Tangent', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'f(y) ≥ f(x) + ∇f(x)ᵀ(y − x)', mono: true, fs: 12.5, fill: C.ink },
  ]);

  o += card(288, 152, 236, 72, D.purple.tint, C.purple, [
    { t: '2nd-Order: PSD Hessian', fs: 14.5, fw: 700, fill: D.purple.dark },
    { t: '∇²f(x) ⪰ 0   (∀x ∈ open C)', mono: true, fs: 12.5, fill: C.ink },
  ]);

  o += arrow(175, 82, 130, 148, C.blue);
  o += arrow(150, 148, 195, 82, C.green);
  o += fitTx(80, 114, 'θ ↓ 0 / z-cancel', 140, { mono: true, fs: 12, fw: 700, fill: D.blue.dark });

  o += arrow(256, 188, 284, 188, C.purple);
  o += arrow(284, 174, 256, 174, C.green);

  o += card(20, 238, 500, 38, C.lamber, C.amber, [
    { t: '1st ⇔ 2nd Order via 1D Taylor:  g(1) = g(0) + g\'(0) + ½(y−x)ᵀ∇²f(z_τ)(y−x)', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function jensenGapVariance() {
  const W = 540, H = 282;
  let o = '';
  o += rect(20, 16, 248, 250, '#FFFFFF', C.gray, 8, 1.5);
  o += path('M 44 216 Q 144 202 240 60', C.blue, 'none', 3, false);
  // Supporting line = tangent to the curve at m (142, 170.8), slope −0.789.
  o += line(60, 235.5, 240, 93.5, C.green, 2, true);
  o += line(44, 216, 240, 60, D.red.base, 2.5, false);
  o += circle(142, 138, 5.5, D.red.base, '#FFFFFF', 1.5);
  o += circle(142, 170.8, 5.5, C.blue, '#FFFFFF', 1.5);
  o += arrow(142, 166, 142, 142, D.red.base);
  o += fitTx(144, 38, 'E[f(Z)] ≥ f(E[Z])', 230, { mono: true, fs: 14, fw: 700, fill: D.red.dark });
  o += fitTx(144, 248, 'm = E[Z]  (linear term vanishes)', 230, { mono: true, fs: 12.5, fw: 600, fill: D.green.dark });

  o += card(284, 16, 236, 76, C.lgreen, C.green, [
    { t: '1. Subgradient Bound at m', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'f(Z) ≥ f(m) + gᵀ(Z − m)', mono: true, fs: 13, fill: C.ink },
    { t: 'for g ∈ ∂f(m), every outcome Z', fs: 12.5, fill: C.muted },
  ]);

  o += card(284, 103, 236, 76, C.lblue, C.blue, [
    { t: '2. Take Expectation E[·]', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'E[gᵀ(Z − m)] = gᵀ(m − m) = 0', mono: true, fs: 12.5, fill: C.ink },
    { t: '⇒  E[f(Z)] ≥ f(E[Z])', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(284, 190, 236, 76, D.red.tint, D.red.base, [
    { t: '3. Strict Convexity Gap', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'x²: E[Z²] − (E[Z])² = Var(Z)', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Equality ⇔ Z = m almost surely', fs: 12.5, fw: 700, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

function closureRules() {
  const W = 540, H = 284;
  let o = '';
  o += titledPanel(20, 14, 240, 120, '1. Nonnegative Sums', D.blue, [
    { t: '∑ᵢ wᵢ fᵢ(x)   (wᵢ ≥ 0)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Empirical risk + regularizers', fs: 13, fill: C.ink },
  ]);

  o += titledPanel(280, 14, 240, 120, '2. Affine Pre-Composition', D.green, [
    { t: 'w ↦ f(Xw + b)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Convex in predictions ⇒ in w', fs: 13, fill: C.ink },
  ]);

  o += titledPanel(20, 148, 240, 122, '3. Pointwise Max / Supremum', D.purple, [
    { t: 'maxᵢ fᵢ(x)   (epi = ⋂ᵢ epi fᵢ)', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    { t: 'Certifies hinge & ‖w‖₁ = ∑|wᵢ|', fs: 13, fill: C.ink },
  ]);

  o += titledPanel(280, 148, 240, 122, '4. Monotone Composition', D.orange, [
    { t: 'h(g(x)): g convex, h convex ↑', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark },
    { t: 'Without h ↑ fails: (x² − 1)²', fs: 13, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §35.4 mdl-stochastic-gradient-methods.md (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function minibatchVarianceLaw() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 14, 152, 174, N.panel, C.gray, [
    { t: 'Single Example i', fs: 14, fw: 700, fill: C.ink },
    { t: '∇fᵢ(x)', mono: true, fs: 15, fw: 700, fill: D.red.dark },
    { t: 'Mean: ∇f(x)', mono: true, fs: 13, fill: C.ink },
    { t: 'Cov: Σ', mono: true, fs: 13, fill: C.ink },
    { t: 'Var: tr(Σ)', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
  ]);

  o += arrow(176, 101, 200, 101, C.blue);

  o += card(204, 14, 152, 174, C.lblue, C.blue, [
    { t: 'Minibatch Average', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'ĝ_b = (1/b) ∑ ∇fᵢ', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'E[ĝ_b] = ∇f(x)', mono: true, fs: 13, fill: D.green.dark },
    { t: 'Power: tr(Σ) / b', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'RMS: √(tr Σ) / √b', mono: true, fs: 13, fill: C.ink },
  ]);

  o += arrow(360, 101, 384, 101, C.green);

  o += card(388, 14, 132, 174, C.lgreen, C.green, [
    { t: 'Noise Scale', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'b_noise =', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'tr(Σ) / ‖∇f‖²', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Signal = Noise', fs: 12.5, fill: C.muted },
    { t: 'crossover size', fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 204, 500, 62, C.lamber, C.amber, [
    { t: 'Square-Root Amplitude Law: 10× lower RMS gradient noise requires 100× larger b', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Below b_noise: increasing b speeds training linearly · Above b_noise: diminishing returns', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function secondMomentBalance() {
  const W = 540, H = 278;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Exact Second-Moment Recursion:  eₖ₊₁ = (1 − ηλ)² eₖ + η²σ²   (eₖ = E[xₖ²])', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 74, 148, 114, C.lgreen, C.green, [
    { t: 'Signal Contraction', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'ρ² eₖ = (1 − ηλ)² eₖ', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Pulls variance down', fs: 12.5, fill: C.muted },
    { t: 'toward 0 each step', fs: 12.5, fill: C.muted },
  ]);

  o += tx(183, 131, '+', { fs: 22, fw: 700, fill: C.ink });

  o += card(198, 74, 144, 114, D.red.tint, D.red.base, [
    { t: 'Noise Injection', fs: 14, fw: 700, fill: D.red.dark },
    { t: '+ η² σ²', mono: true, fs: 15, fw: 700, fill: D.red.dark },
    { t: 'Cross term E[xₖξₖ]=0', mono: true, fs: 12, fill: C.ink },
    { t: 'by martingales', fs: 12.5, fill: C.muted },
  ]);

  o += arrow(346, 131, 370, 131, C.purple);

  o += card(374, 74, 146, 114, D.purple.tint, C.purple, [
    { t: 'Stationary Floor V', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'V = η²σ² / (1 − ρ²)', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
    { t: '≈ ησ² / (2λ)', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'Loss: ½λV ≈ ησ²/4', mono: true, fs: 12.5, fill: D.purple.dark },
  ]);

  o += card(20, 206, 500, 56, C.lamber, C.amber, [
    { t: 'Linear Batch-Size Scaling Rule:  E[f(x_{∞})] ≈ η tr(Σ) / (4b)', mono: true, fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'Halving learning rate η and doubling batch size b cut the loss floor by the same factor', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function valleyNoisePenalty() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 40, D.purple.tint, C.purple, [
    { t: 'Double Penalty of κ = 1000 Under Noise (η = 1/L = 10⁻³, σ₁² = σ₂² = 1)', fs: 14, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 64, 240, 146, 'Flat Mode x₁ (λ₁ = 1)', D.blue, [
    { t: 'Contraction: ρ₁ = 0.999', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Transient: ~3,450 steps', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Variance V₁ = 5.0 × 10⁻⁴', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Loss floor ½λ₁V₁ = 2.5×10⁻⁴ (33%)', mono: true, fs: 12, fill: C.ink },
  ]);

  o += titledPanel(280, 64, 240, 146, 'Steep Mode x₂ (λ₂ = 1000)', D.red, [
    { t: 'Contraction: ρ₂ = 0.000', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Transient: 1 step', fs: 13.5, fill: C.ink },
    { t: '2 − ηλ₂ = 1 (doubles floor!)', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Loss floor ½λ₂V₂ = 5.0×10⁻⁴ (67%)', mono: true, fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 224, 500, 48, C.lamber, C.amber, [
    { t: 'Total Expected Loss Floor:  2.5×10⁻⁴ + 5.0×10⁻⁴ = 7.5 × 10⁻⁴  (never reaches 0!)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function robbinsMonroBalance() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 40, C.lblue, C.blue, [
    { t: 'Robbins–Monro Conditions (1951):  ∑ ηₖ = ∞   and   ∑ ηₖ² < ∞', mono: true, fs: 14.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 66, 156, 146, 'Too Slow: ηₖ = η', D.red, [
    { t: '∑ ηₖ = ∞  (✓)', mono: true, fs: 13, fill: D.green.dark },
    { t: '∑ ηₖ² = ∞  (✗)', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Infinite noise power;', fs: 12.5, fill: C.ink },
    { t: 'trapped on floor V', fs: 12.5, fill: D.red.dark },
  ]);

  o += titledPanel(192, 66, 156, 146, 'Goldilocks: c/(k+k₀)', D.green, [
    { t: '∑ 1/k = ∞  (✓)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: '∑ 1/k² = π²/6 < ∞ (✓)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Infinite travel reach,', fs: 12.5, fill: C.ink },
    { t: 'finite total variance', fs: 12.5, fill: D.green.dark },
  ]);

  o += titledPanel(364, 66, 156, 146, 'Too Fast: ηₖ = 1/k²', D.orange, [
    { t: '∑ 1/k² < ∞  (✗)', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: '∑ 1/k⁴ < ∞  (✓)', mono: true, fs: 13, fill: D.green.dark },
    { t: 'Bounded travel;', fs: 12.5, fill: C.ink },
    { t: 'freezes before x*', fs: 12.5, fill: D.orange.dark },
  ]);

  o += card(20, 226, 500, 44, D.purple.tint, C.purple, [
    { t: 'Polynomial Window:  ηₖ ∝ k⁻α  satisfies both conditions iff  ½ < α ≤ 1', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function oneOverKThreshold() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, D.purple.tint, C.purple, [
    { t: 'Asymptotic Matching for ηₖ = c / (k + k₀):  eₖ ≈ A / k  with  A = c²σ² / (2cμ − 1)', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 68, 156, 144, 'Sub-Critical: c < 1/2μ', D.red, [
    { t: '2cμ < 1', mono: true, fs: 14, fw: 700, fill: D.red.dark },
    { t: 'Step decays faster', fs: 12.5, fill: C.ink },
    { t: 'than contraction:', fs: 12.5, fill: C.ink },
    { t: 'eₖ = Θ(k⁻²ᶜᵘ)  (Slow!)', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(192, 68, 156, 144, 'Optimal: c* = 1 / μ', D.green, [
    { t: '2c*μ = 2 > 1', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Minimizes A(c):', fs: 12.5, fill: C.ink },
    { t: 'eₖ ≈ σ² / (μ² k)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Minimax O(1/k) rate', fs: 12.5, fill: D.green.dark },
  ]);

  o += titledPanel(364, 68, 156, 144, 'Overshot: c = 2 / μ', D.blue, [
    { t: '2cμ = 4 > 1', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'Keeps O(1/k) exponent!', fs: 12.5, fill: C.ink },
    { t: 'A = (4/3) σ² / μ²', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: '(Only +33% penalty)', fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 224, 500, 46, C.lamber, C.amber, [
    { t: 'Asymmetric Risk: Underestimating c < 1/(2μ) ruins the power-law exponent!', fs: 13.5, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

function convexSgdTelescoping() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 14, 500, 68, C.lblue, C.blue, [
    { t: 'Step 1 · Expand Distance & Apply Subgradient Inequality', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'E[‖xₖ₊₁ − x*‖² | xₖ]  ≤  ‖xₖ − x*‖² − 2η(f(xₖ) − f*) + η²G²', mono: true, fs: 13, fill: C.ink },
  ]);
  o += arrow(270, 84, 270, 104, C.blue);

  o += card(20, 108, 500, 68, D.purple.tint, C.purple, [
    { t: 'Step 2 · Telescope Over k = 0..K−1 & Apply Jensen to x̄_K', fs: 14.5, fw: 700, fill: D.purple.dark },
    { t: 'E[f(x̄_K)] − f*  ≤  D² / (2ηK)  +  ηG² / 2', mono: true, fs: 13.5, fw: 700, fill: C.ink },
  ]);
  o += arrow(270, 178, 270, 198, C.green);

  o += card(20, 202, 500, 68, C.lgreen, C.green, [
    { t: 'Step 3 · Balance Transient and Noise Floor for Budget K', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'η* = D / (G√K)   ⇒   E[f(x̄_K)] − f*  ≤  D G / √K', mono: true, fs: 14, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function ghadimiLanRandomStop() {
  const W = 540, H = 298;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Ghadimi–Lan Nonconvex Stationarity Theorem (L-Smooth, Noise σ²)', fs: 14.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 70, 236, 122, D.purple.tint, C.purple, [
    { t: 'Random Stopping Index R', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'R ~ Uniform{0, …, K−1}', mono: true, fs: 13, fill: C.ink },
    { t: 'Converts average gradient norm', fs: 12.5, fill: C.ink },
    { t: 'into E[‖∇f(x_R)‖²] certificate', fs: 12.5, fill: D.purple.dark },
  ]);

  o += arrow(260, 131, 280, 131, C.purple);

  o += card(284, 70, 236, 122, C.lgreen, C.green, [
    { t: 'Optimal Budget Step η*', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'η* = min(1/L, √(2Δ / Lσ²K))', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'E[‖∇f(x_R)‖²] ≤', mono: true, fs: 12.5, fill: C.ink },
    { t: '2LΔ/K + 2σ√(2LΔ/K)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
  ]);

  o += card(20, 208, 500, 76, D.red.tint, D.red.base, [
    { t: 'O(1/ε⁴) Sample Complexity:', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'E[‖∇f(x_R)‖²] ≤ ε²  requires  K = Θ(σ²LΔ / ε⁴)', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Cutting RMS gradient norm √(E‖∇f‖²) by 10× demands 10,000× more iterations!', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function svrgControlVariate() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 44, D.purple.tint, C.purple, [
    { t: 'SVRG Control Variate:  g̃ᵢ(x) = ∇fᵢ(x) − ∇fᵢ(x̃) + ∇f(x̃)', mono: true, fs: 14.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 70, 156, 132, '1. Epoch Snapshot', D.blue, [
    { t: 'Save x̃ once/epoch', fs: 13, fill: C.ink },
    { t: 'Compute full mean', fs: 12.5, fill: C.muted },
    { t: '∇f(x̃) = (1/n)∑∇fᵢ(x̃)', mono: true, fs: 11.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(192, 70, 156, 132, '2. Zero-Mean Shift', D.orange, [
    { t: 'Subtract correlated', fs: 12.5, fill: C.ink },
    { t: '∇fᵢ(x̃) − ∇f(x̃)', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Eᵢ[g̃ᵢ(x)] = ∇f(x)', mono: true, fs: 12.5, fill: D.green.dark },
  ]);

  o += titledPanel(364, 70, 156, 132, '3. Vanishing Noise', D.green, [
    { t: 'Var(g̃ᵢ) ≤', mono: true, fs: 13, fill: C.ink },
    { t: 'L_max² ‖x − x̃‖² → 0', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'as x, x̃ → x*', fs: 13, fill: D.green.dark },
  ]);

  o += card(20, 216, 500, 52, C.lgreen, C.green, [
    { t: 'Result: Restores geometric (linear) convergence at constant η with zero noise floor!', fs: 13.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §35.5 mdl-adaptive-stochastic-methods.md (9 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function diagonalPrecondProxy() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 14, 152, 174, D.red.tint, D.red.base, [
    { t: 'Raw Gradient g', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'gᵢ = hᵢ xᵢ', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'h = (1, 1000)', mono: true, fs: 13, fill: C.ink },
    { t: '1000 : 1 skew', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'across axes', fs: 12.5, fill: C.muted },
  ]);

  o += arrow(176, 101, 200, 101, C.blue);

  o += card(204, 14, 152, 174, C.lblue, C.blue, [
    { t: 'RMS Gradient Proxy', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '√vᵢ ≈ |gᵢ| = hᵢ|xᵢ|', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'Steep axis has', fs: 13, fill: C.ink },
    { t: '1000× larger RMS', fs: 13, fw: 700, fill: C.ink },
    { t: 'at comparable |xᵢ|', fs: 12.5, fill: C.muted },
  ]);

  o += arrow(360, 101, 384, 101, C.green);

  o += card(388, 14, 132, 174, C.lgreen, C.green, [
    { t: 'Normalized Step', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Δxᵢ = −η gᵢ/√vᵢ', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Cancels hᵢ!', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Mimics ideal', fs: 12.5, fill: C.ink },
    { t: 'ηᵢ = 1 / hᵢ', mono: true, fs: 13, fill: C.ink },
  ]);

  o += card(20, 204, 500, 62, D.purple.tint, C.purple, [
    { t: 'Key Insight: On an axis-aligned bowl, gradient magnitude reveals local curvature', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Dividing by √(running mean square) automatically assigns smaller steps to steeper axes', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function adagradRareFrequent() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'AdaGrad Update:  sₜ = sₜ₋₁ + gₜ ⊙ gₜ,   xₜ₊₁ = xₜ − (η / (√sₜ + ε)) gₜ', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, 'Frequent Feature (Coord 1)', D.blue, [
    { t: 'Active every step: g_{t,1} = ±1', mono: true, fs: 13, fill: C.ink },
    { t: 'At t = 10⁴:  s_{t,1} = 10⁴', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Step η / √10⁴ = η / 100', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Automatically damped against noise', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 68, 240, 138, 'Rare Token Embedding (Coord 2)', D.green, [
    { t: 'Once in 10⁴ steps: g_{t,2} = ±1', mono: true, fs: 12.5, fill: C.ink },
    { t: 'At t = 10⁴:  s_{t,2} = 1', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Step η / √1 = full η!', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: '100× larger step when token appears', fs: 12.5, fill: C.ink },
  ]);

  o += card(20, 220, 500, 48, D.red.tint, D.red.base, [
    { t: 'Deep Learning Limitation: Persistent noise makes sₜ ≈ σ²t grow forever → η/√sₜ → 0', fs: 13, fw: 700, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

function rmspropEmaWindow() {
  const W = 540, H = 282;
  let o = '';
  o += card(20, 12, 500, 42, D.purple.tint, C.purple, [
    { t: 'RMSProp EMA Preconditioner:  vₜ = β₂ vₜ₋₁ + (1 − β₂) gₜ²  =  (1−β₂) ∑ β₂ᵏ g_{t−k}²', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 68, 240, 138, 'Short Window: β₂ = 0.9', D.orange, [
    { t: 'Half-life k₁/₂ = log(2)/(1−β₂)', mono: true, fs: 12.5, fill: C.ink },
    { t: 'k₁/₂ ≈ 7 steps', mono: true, fs: 16, fw: 700, fill: D.orange.dark },
    { t: 'Fast adaptation to local curvature,', fs: 12.5, fill: C.ink },
    { t: 'forgets gradient spikes quickly', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 68, 240, 138, 'Standard Window: β₂ = 0.999', D.green, [
    { t: 'Effective horizon 1/(1−β₂) = 1000', mono: true, fs: 12.5, fill: C.ink },
    { t: 'k₁/₂ ≈ 693 steps', mono: true, fs: 16, fw: 700, fill: D.green.dark },
    { t: 'Stable second-moment scale;', fs: 12.5, fill: C.ink },
    { t: 'step size never vanishes under noise', fs: 12.5, fill: D.green.dark },
  ]);

  o += card(20, 220, 500, 48, C.lblue, C.blue, [
    { t: 'Under stationary noise E[gₜ²] = σ²:  E[vₜ] → σ²  (steady-state step ≈ η / σ)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function adamDualMomentFlow() {
  const W = 540, H = 284;
  let o = '';
  o += card(16, 88, 92, 68, N.panel, C.gray, [
    { t: 'Gradient', fs: 13.5, fw: 700, fill: C.ink },
    { t: 'gₜ', mono: true, fs: 16, fw: 700, fill: D.red.dark },
  ]);

  o += arrow(110, 104, 146, 56, C.blue);
  o += arrow(110, 140, 146, 188, D.orange.base);

  o += card(150, 18, 206, 74, C.lblue, C.blue, [
    { t: '1st Moment (β₁ = 0.9)', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'mₜ = β₁mₜ₋₁ + (1−β₁)gₜ', mono: true, fs: 12.5, fill: C.ink },
    { t: 'm̂ₜ = mₜ / (1 − β₁ᵗ)', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);

  o += card(150, 152, 206, 74, C.lamber, C.amber, [
    { t: '2nd Moment (β₂ = 0.999)', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²', mono: true, fs: 12.5, fill: C.ink },
    { t: 'v̂ₜ = vₜ / (1 − β₂ᵗ)', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
  ]);

  o += arrow(358, 56, 392, 102, C.blue);
  o += arrow(358, 188, 392, 142, D.orange.base);

  o += card(396, 74, 128, 96, C.lgreen, C.green, [
    { t: 'Adam Step', fs: 14, fw: 700, fill: D.green.dark },
    { t: '−η m̂ₜ', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: '────────', mono: true, fs: 12, fill: C.muted },
    { t: '√v̂ₜ + ε', mono: true, fs: 14, fw: 700, fill: D.green.dark },
  ]);

  o += card(16, 240, 508, 34, D.purple.tint, C.purple, [
    { t: 'm̂ₜ smooths high-frequency minibatch oscillations · √v̂ₜ normalizes per-coordinate scale', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function adamScaleInvariance() {
  const W = 540, H = 308;
  let o = '';
  o += card(20, 12, 500, 54, C.lgreen, C.green, [
    { t: 'Diagonal Scale Invariance (ε = 0):', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '∇f̃(x) = diag(c₁,…,c_d) ∇f(x)  ⇒  x̃ₜ = xₜ', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
  ], { spacing: 20 });

  o += card(20, 80, 152, 130, C.lblue, C.blue, [
    { t: 'Rescaled Gradient', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'g̃ᵢ = cᵢ gᵢ', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: '(cᵢ > 0)', mono: true, fs: 13, fill: C.muted },
  ]);
  o += arrow(176, 145, 198, 145, C.blue);

  o += card(202, 80, 156, 130, D.purple.tint, C.purple, [
    { t: 'Linear EMA Scaling', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'm̂ᵢ → cᵢ m̂ᵢ', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'v̂ᵢ → cᵢ² v̂ᵢ', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: '√v̂ᵢ → cᵢ √v̂ᵢ', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);
  o += arrow(362, 145, 384, 145, C.green);

  o += card(388, 80, 132, 130, C.lgreen, C.green, [
    { t: 'Exact Cancel', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'cᵢ m̂ᵢ', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: '────────', mono: true, fs: 12, fill: C.muted },
    { t: 'cᵢ √v̂ᵢ', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 224, 500, 70, C.lamber, C.amber, [
    { t: 'Consequence: On ½(x₁² + 1000 x₂²) from (1, 1), Adam follows x_{t,1} = x_{t,2}', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Identical trajectory to the isotropic bowl ½(x₁² + x₂²): iteration count scales with ‖x₀‖_{∞} / η', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function rotatedValleyFailure() {
  const W = 540, H = 286;
  let o = '';
  o += titledPanel(20, 14, 240, 192, 'Aligned Valley (θ = 0°)', D.green, [
    { t: 'A₀ = diag(1, 1000)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: '√v̂₁ = 1   vs   √v̂₂ = 1000', mono: true, fs: 13, fill: C.ink },
    { t: 'Adam (β₁=0): 22 steps!', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Adam (β₁=0.9): 165–205 steps', fs: 13, fill: C.ink },
    { t: '(Robust across 100× in η)', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 14, 240, 192, 'Rotated Valley (θ = 45°)', D.red, [
    { t: 'A_{π/4} = [500.5, −499.5; …]', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Steep mode (−1, 1)/√2 projects', fs: 12.5, fill: C.ink },
    { t: 'equally onto both axes: v̂₁ ≈ v̂₂!', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Adam (β₁=0): Never (2-cycle)', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Adam (β₁=0.9): 559–12,862 steps', fs: 12.5, fill: C.ink },
  ]);

  o += card(20, 220, 500, 52, D.purple.tint, C.purple, [
    { t: 'Why 45° Rotation Defeats Diagonal Scaling:  diag(√v̂)⁻¹ ≈ (1/σ) I', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'When v̂₁ ≈ v̂₂, diagonal preconditioning collapses to a scalar learning rate!', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function reddiCycleDrift() {
  const W = 540, H = 320;
  let o = '';
  o += card(20, 12, 500, 42, D.red.tint, D.red.base, [
    { t: 'Reddi Counterexample (β₁ = 0, β₂ = 1/17):  gₜ Cycles (+4, −1, −1) on [−1, 1]', fs: 14, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 68, 156, 132, C.lblue, C.blue, [
    { t: 'Step 1: g₁ = +4', mono: true, fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'v₁ = 15.12, √v₁ = 3.89', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Update −η g₁/√v₁:', fs: 12.5, fill: C.muted },
    { t: 'Δx₁ = −1.03η (left)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(192, 68, 156, 132, C.lamber, C.amber, [
    { t: 'Step 2: g₂ = −1', mono: true, fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: 'v₂ = 1.83, √v₂ = 1.35', mono: true, fs: 12.5, fill: C.ink },
    { t: 'vₜ forgets spike fast:', fs: 12.5, fill: C.muted },
    { t: 'Δx₂ = +0.74η (right)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
  ]);

  o += card(364, 68, 156, 132, D.red.tint, D.red.base, [
    { t: 'Step 3: g₃ = −1', mono: true, fs: 14.5, fw: 700, fill: D.red.dark },
    { t: 'v₃ = 1.05, √v₃ = 1.02', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Full right step:', fs: 12.5, fill: C.muted },
    { t: 'Δx₃ = +0.98η (right)', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 214, 500, 92, C.lgreen, C.green, [
    { t: 'Net 3-Step Drift:  (−1.03 + 0.74 + 0.98)η = +0.69 η > 0', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: '→  locks onto worst x = +1!', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'AMSGrad Fix:  ṽₜ = max(ṽₜ₋₁, vₜ) = 15.12', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '⇒  net drift −0.51 η → true optimum x* = −1', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function adamwLinfBox() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 190, 'Coupled Adam (ℓ₂ in gₜ)', D.blue, [
    { t: 'u = ∇ℓ(w) + λw', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Decay scaled by 1 / √v̂_{t,i}:', fs: 13, fill: C.ink },
    { t: 'Noisy weights barely decay;', fs: 12.5, fill: D.red.dark },
    { t: 'quiet weights shrink 100× faster', fs: 12.5, fill: D.red.dark },
    { t: 'Fixed point: ∇ℓ(w*) + λw* = 0', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(280, 14, 240, 190, 'Decoupled AdamW', D.green, [
    { t: 'wₜ₊₁ = (1 − ηλ)wₜ − η m̂ₜ / (√v̂ₜ + ε)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Uniform shrinkage across all', fs: 13, fill: C.ink },
    { t: 'gradient noise scales!', fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Exact-gradient fixed point (ε→0):', fs: 12.5, fill: C.ink },
    { t: 'ℓ_{∞} Box: ‖w*‖_{∞} ≤ 1 / λ', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 218, 500, 50, D.purple.tint, C.purple, [
    { t: 'AdamW Stationarity:  λ wᵢ (|gᵢ| + ε) = −gᵢ   ⇒   |wᵢ| = |gᵢ| / (λ(|gᵢ| + ε)) < 1/λ', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function preconditioningLadder() {
  const W = 540, H = 286;
  let o = '';
  o += card(20, 12, 500, 40, D.purple.tint, C.purple, [
    { t: 'Two-Sided Matrix Preconditioning on m × n Layer Gradient G = U Σ Vᵀ', fs: 14.5, fw: 700, fill: D.purple.dark },
  ]);

  o += titledPanel(20, 64, 156, 152, 'K-FAC (a = 1)', D.blue, [
    { t: 'S_out⁻¹ G S_in⁻¹', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'S_in = E[aaᵀ]', mono: true, fs: 12.5, fill: C.ink },
    { t: 'S_out = E[δδᵀ]', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Kronecker Fisher /', fs: 12.5, fill: C.muted },
    { t: 'Natural Gradient', fs: 12.5, fill: D.blue.dark },
  ]);

  o += titledPanel(192, 64, 156, 152, 'Shampoo (a = 1/4)', D.orange, [
    { t: 'P_l⁻¹/⁴ G P_r⁻¹/⁴', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'P_l = ∑ G_τ G_τᵀ', mono: true, fs: 12.5, fill: C.ink },
    { t: 'P_r = ∑ G_τᵀ G_τ', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Full-matrix AdaGrad', fs: 12.5, fill: C.muted },
    { t: 'along rows & cols', fs: 12.5, fill: D.orange.dark },
  ]);

  o += titledPanel(364, 64, 156, 152, 'Muon (Orthogonal)', D.green, [
    { t: '(GGᵀ)⁻¹/⁴G(GᵀG)⁻¹/⁴', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    { t: '= U Vᵀ  (Polar Factor)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'All singular values = 1', fs: 12.5, fill: C.ink },
    { t: 'Fast Newton–Schulz;', fs: 12.5, fill: C.ink },
    { t: 'rotation-equivariant!', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 230, 500, 44, C.lgreen, C.green, [
    { t: 'Muon Polar Update: Equalizes spectral conditioning without storing m×m or n×n factors', fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §35.6 mdl-constrained-optimization-duality.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function boundaryForceBalance() {
  const W = 540, H = 282;
  let o = '';
  o += rect(20, 16, 236, 250, '#FFFFFF', C.gray, 8, 1.5);
  o += circle(104, 168, 62, C.lblue, C.blue, 2.5, false);
  o += line(104, 168, 204, 38, C.gray, 1.5, true);
  o += arrow(141, 118, 175, 74, C.green);
  o += arrow(141, 118, 107, 162, D.red.base);
  o += circle(141, 118, 5.5, C.ink, '#FFFFFF', 1.5);
  o += circle(204, 38, 5, D.blue.base, '#FFFFFF', 1.5);
  o += tx(140, 70, '∇g(x*)', { mono: true, fs: 12, fw: 700, fill: D.green.dark });
  o += tx(140, 168, '∇f(x*)', { mono: true, fs: 12, fw: 700, fill: D.red.dark });
  o += tx(222, 50, 'y', { fs: 14, fw: 700, fill: D.blue.dark });
  o += fitTx(138, 246, 'Unit Disk ‖x‖₂ ≤ 1,  x* = (0.6, 0.8)', 220, { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });

  o += card(272, 16, 248, 76, C.lblue, C.blue, [
    { t: 'Nonzero Gradient at Optimum', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'f(x) = ½‖x − y‖²,  y = (3, 4)', mono: true, fs: 12.5, fill: C.ink },
    { t: '∇f(x*) = x* − y = (−2.4, −3.2) ≠ 0', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
  ]);

  o += card(272, 103, 248, 76, C.lgreen, C.green, [
    { t: 'Outward Wall Normal ∇g(x*)', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'g(x) = ½(‖x‖² − 1) ≤ 0', mono: true, fs: 12.5, fill: C.ink },
    { t: '∇g(x*) = x* = (0.6, 0.8)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(272, 190, 248, 76, D.red.tint, D.red.base, [
    { t: 'Exact Force Balance (λ* = 4)', fs: 14, fw: 700, fill: D.red.dark },
    { t: '∇f(x*) + 4 ∇g(x*) = 0', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'No feasible move enters disk', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function hyperplaneShadowPrice() {
  const W = 540, H = 298;
  let o = '';
  o += card(20, 12, 500, 44, C.lblue, C.blue, [
    { t: 'Closest Point on Hyperplane:  min ½‖x‖₂²   s.t.   aᵀx − b = 0', mono: true, fs: 14.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 70, 236, 124, D.purple.tint, C.purple, [
    { t: 'Closed-Form Optimum', fs: 14, fw: 700, fill: D.purple.dark },
    { t: '∇ₓL = x + νa = 0  ⇒  x = −νa', mono: true, fs: 12.5, fill: C.ink },
    { t: 'x* = (b / ‖a‖²) a', mono: true, fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'ν* = −b / ‖a‖²', mono: true, fs: 13.5, fw: 700, fill: C.ink },
  ]);

  o += arrow(260, 132, 280, 132, C.purple);

  o += card(284, 70, 236, 124, C.lgreen, C.green, [
    { t: 'Multiplier = Shadow Price', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'p*(b) = f(x*) = b² / (2‖a‖²)', mono: true, fs: 12.5, fill: C.ink },
    { t: '∂p*(b) / ∂b = b/‖a‖² = −ν*', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Exact marginal cost of threshold b', fs: 12, fill: C.muted },
  ]);

  o += card(20, 208, 500, 76, C.lamber, C.amber, [
    { t: 'Example a = (3, 4), b = 5:', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'x* = (0.6, 0.8),   p*(5) = 0.50,   ν* = −0.20', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Shifting hyperplane threshold b: 5.0 → 5.1 raises minimum cost by −ν*(0.1) = +0.02', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function activeVsInactiveBall() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 190, 'Regime 1: Inactive (‖y‖ < r)', D.blue, [
    { t: 'g(x*) = ½(‖y‖² − r²) < 0', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'Unconstrained minimum x* = y', fs: 13, fill: C.ink },
    { t: 'already lies inside the ball', fs: 13, fill: C.ink },
    { t: 'Wall exerts zero force:', fs: 12.5, fill: C.muted },
    { t: 'λ* = 0', mono: true, fs: 15, fw: 700, fill: D.blue.dark },
  ]);

  o += titledPanel(280, 14, 240, 190, 'Regime 2: Active (‖y‖ ≥ r)', D.green, [
    { t: 'g(x*) = 0   (‖x*‖ = r)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Clipped onto boundary sphere:', fs: 13, fill: C.ink },
    { t: 'x* = r y / ‖y‖', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'Wall pushes inward:', fs: 12.5, fill: C.muted },
    { t: 'λ* = ‖y‖ / r − 1 ≥ 0', mono: true, fs: 14, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 218, 500, 50, D.purple.tint, C.purple, [
    { t: 'Unified Clipping Formula:  x* = y / max(1, ‖y‖/r),   λ* = max(0, ‖y‖/r − 1)', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function kktFourConditions() {
  const W = 540, H = 284;
  let o = '';
  o += titledPanel(20, 14, 240, 120, '1. Stationarity (Force Balance)', D.blue, [
    { t: '∇f(x*) + ∑ λᵢ*∇gᵢ + ∑ νⱼ*∇hⱼ = 0', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
    { t: '−∇f lies in active normal cone', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 14, 240, 120, '2. Primal Feasibility', D.green, [
    { t: 'gᵢ(x*) ≤ 0,   hⱼ(x*) = 0', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Candidate x* obeys all constraints', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(20, 148, 240, 122, '3. Dual Feasibility (One-Way)', D.orange, [
    { t: 'λᵢ* ≥ 0   (for all i = 1,…,m)', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Walls only push inward, never pull', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 148, 240, 122, '4. Complementary Slackness', D.purple, [
    { t: 'λᵢ* gᵢ(x*) = 0   (∀i = 1,…,m)', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    { t: 'Inactive wall (gᵢ < 0) ⇒ λᵢ* = 0', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function projectionNonexpansive() {
  const W = 540, H = 280;
  let o = '';
  o += rect(20, 16, 232, 248, '#FFFFFF', C.gray, 8, 1.5);
  o += rect(40, 146, 192, 92, C.lgreen, C.green, 24, 2);
  o += line(76, 56, 196, 40, D.red.base, 2.5, true);
  o += arrow(76, 56, 76, 142, C.blue);
  o += arrow(196, 40, 196, 142, C.blue);
  o += line(76, 146, 196, 146, C.green, 3.5, false);
  o += circle(76, 56, 4.5, D.red.base, '#FFFFFF', 1.5);
  o += circle(196, 40, 4.5, D.red.base, '#FFFFFF', 1.5);
  o += circle(76, 146, 5, C.green, '#FFFFFF', 1.5);
  o += circle(196, 146, 5, C.green, '#FFFFFF', 1.5);
  o += tx(60, 54, 'x', { fs: 15, fw: 700, fill: D.red.dark });
  o += tx(214, 40, 'y', { fs: 15, fw: 700, fill: D.red.dark });
  o += tx(76, 168, 'u', { fs: 15, fw: 700, fill: D.green.dark });
  o += tx(196, 168, 'v', { fs: 15, fw: 700, fill: D.green.dark });
  o += fitTx(136, 196, 'Closed Convex Set C', 180, { fs: 14, fw: 700, fill: D.green.dark });
  o += fitTx(136, 220, '‖u − v‖₂ ≤ ‖x − y‖₂', 180, { mono: true, fs: 13, fw: 700, fill: D.green.dark });

  o += card(268, 16, 252, 74, C.lblue, C.blue, [
    { t: '1. Two Obtuse Angles (≥ 90°)', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '(x − u)ᵀ(v − u) ≤ 0', mono: true, fs: 13, fill: C.ink },
    { t: '(y − v)ᵀ(u − v) ≤ 0', mono: true, fs: 13, fill: C.ink },
  ]);

  o += card(268, 103, 252, 74, D.purple.tint, C.purple, [
    { t: '2. Add & Cauchy–Schwarz', fs: 14, fw: 700, fill: D.purple.dark },
    { t: '‖u − v‖₂² ≤ (x − y)ᵀ(u − v)', mono: true, fs: 12.5, fill: C.ink },
    { t: '≤ ‖x − y‖₂ ‖u − v‖₂', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);

  o += card(268, 190, 252, 74, C.lgreen, C.green, [
    { t: '3. Divide by ‖u − v‖₂', fs: 14, fw: 700, fill: D.green.dark },
    { t: '‖Π_C(x) − Π_C(y)‖₂ ≤ ‖x − y‖₂', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: '1-Lipschitz contractive map', fs: 12.5, fill: C.muted },
  ]);
  return svg(W, H, o);
}

function pgdStepCycle() {
  const W = 540, H = 298;
  let o = '';
  o += card(20, 12, 500, 42, C.lblue, C.blue, [
    { t: 'Projected Gradient Descent (PGD):   xₖ₊₁ = Π_C(xₖ − η ∇f(xₖ))', mono: true, fs: 14.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 70, 148, 122, N.panel, C.gray, [
    { t: 'Feasible xₖ ∈ C', fs: 14, fw: 700, fill: C.ink },
    { t: 'Evaluate gradient', fs: 12.5, fill: C.muted },
    { t: '∇f(xₖ)', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
  ]);
  o += arrow(172, 131, 194, 131, C.blue);

  o += card(198, 70, 148, 122, C.lamber, C.amber, [
    { t: '1. Gradient Step', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'yₖ = xₖ − η∇f(xₖ)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '(May exit C)', fs: 12.5, fill: D.orange.dark },
  ]);
  o += arrow(350, 131, 372, 131, C.green);

  o += card(376, 70, 144, 122, C.lgreen, C.green, [
    { t: '2. Project Π_C', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'xₖ₊₁ = Π_C(yₖ)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Closest point in C', fs: 12.5, fill: C.ink },
  ]);

  o += card(20, 208, 500, 76, D.purple.tint, C.purple, [
    { t: 'Fixed-Point Optimality:  x* = Π_C(x* − η∇f(x*))', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    { t: '⇔  ∇f(x*)ᵀ(x − x*) ≥ 0  ∀x ∈ C', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    { t: 'By nonexpansiveness of Π_C, PGD contracts at least as fast as GD: rate ρ = max(|1−ημ|, |1−ηL|)', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function softThresholdingProx() {
  const W = 540, H = 282;
  let o = '';
  o += rect(20, 16, 244, 250, '#FFFFFF', C.gray, 8, 1.5);
  o += line(36, 141, 248, 141, C.ink, 1.5, false);
  o += line(142, 32, 142, 248, C.ink, 1.5, false);
  o += line(52, 231, 232, 51, C.gray, 1.5, true);
  o += path('M 52 196 L 107 141 L 177 141 L 232 86', C.green, 'none', 3.5, false);
  o += circle(107, 141, 4.5, D.red.base, '#FFFFFF', 1.5);
  o += circle(177, 141, 4.5, D.red.base, '#FFFFFF', 1.5);
  o += tx(244, 154, 'z', { fs: 14, fw: 700, fill: C.ink });
  o += tx(154, 38, 'x', { fs: 14, fw: 700, fill: C.ink });
  o += tx(107, 156, '−λ', { fs: 13, fw: 700, fill: D.red.dark });
  o += tx(177, 156, '+λ', { fs: 13, fw: 700, fill: D.red.dark });
  o += fitTx(142, 256, 'Dead Zone [−λ, +λ] → Exact 0', 228, { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });

  o += card(280, 16, 240, 76, C.lblue, C.blue, [
    { t: 'Proximal Operator Definition', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'prox_{ψ}(z) = argmin_{x} (ψ(x) + ½‖x−z‖²)', mono: true, fs: 11.5, fill: C.ink },
    { t: 'Balances kinked ψ against tether', fs: 12.5, fill: C.muted },
  ]);

  o += card(280, 103, 240, 76, C.lgreen, C.green, [
    { t: 'Soft-Thresholding (ψ = λ|x|)', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'sign(z) max(|z| − λ, 0)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Shrinks by λ; zeroes |z| ≤ λ', fs: 12.5, fill: C.ink },
  ]);

  o += card(280, 190, 240, 76, D.purple.tint, C.purple, [
    { t: 'Unifies Projection & Lasso', fs: 14, fw: 700, fill: D.purple.dark },
    { t: '• ψ = I_C  ⇒  prox_{ψ} = Π_C', mono: true, fs: 12.5, fill: C.ink },
    { t: '• ψ = λ‖w‖₁ ⇒  ISTA exact zeros', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §35.7 mdl-lagrangian-duality.md (9 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function dualLowerBoundCert() {
  const W = 540, H = 302;
  let o = '';
  o += card(20, 12, 500, 42, D.purple.tint, C.purple, [
    { t: 'Bracketing the Unknown Optimum p* Without Zero-Gradient Stationarity', fs: 14.5, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 68, 152, 130, C.lgreen, C.green, [
    { t: 'Dual Lower Bound', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Any λ ≥ 0 gives', fs: 13, fill: C.ink },
    { t: 'q(λ) = inf_{x} L(x,λ)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'q(3) = 7.875 ≤ p*', mono: true, fs: 13, fw: 700, fill: C.ink },
  ]);

  o += arrow(176, 133, 196, 133, C.green);

  o += card(200, 68, 140, 130, C.lamber, C.amber, [
    { t: 'True Optimum p*', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'q(λ) ≤ d* ≤ p* ≤ f(x)', mono: true, fs: 11.5, fw: 700, fill: C.ink },
    { t: 'p* = d* = 8.00', mono: true, fs: 14, fw: 700, fill: D.orange.dark },
    { t: '(at λ* = 4)', fs: 12.5, fill: C.muted },
  ]);

  o += arrow(364, 133, 344, 133, C.blue);

  o += card(368, 68, 152, 130, C.lblue, C.blue, [
    { t: 'Primal Upper Bound', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'Any feasible x ∈ C', fs: 13, fill: C.ink },
    { t: 'g(x) ≤ 0 gives', mono: true, fs: 12.5, fill: C.ink },
    { t: 'p* ≤ f(x) = 8.01', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);

  o += card(20, 212, 500, 76, C.lgreen, C.green, [
    { t: 'Stopping Certificate:  f(x) − p*  ≤  f(x) − q(λ)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: '=  8.01 − 7.875 = 0.135  (or 0.01 at λ*=4)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Every dual-feasible multiplier λ ≥ 0 certifies an immediate suboptimality bound!', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function concaveInfimumEnvelope() {
  const W = 540, H = 282;
  let o = '';
  o += rect(20, 16, 244, 250, '#FFFFFF', C.gray, 8, 1.5);
  // Each affine line is tangent to the curve, so the curve is their lower envelope.
  o += line(36, 199, 189.5, 52, C.gray, 1.5, true);
  o += line(36, 173, 248, 91.3, C.gray, 1.5, true);
  o += line(36, 75.6, 248, 187.8, C.gray, 1.5, true);
  o += line(100.2, 52, 248, 213, C.gray, 1.5, true);
  o += path('M 36 202 Q 142 72 248 216', C.green, 'none', 3.5, false);
  o += circle(136.6, 140.3, 5.5, D.orange.base, '#FFFFFF', 1.5);
  o += fitTx(142, 38, 'Affine family λ ↦ f(x) + λg(x)', 228, { fs: 12.5, fw: 600, fill: C.muted });
  o += fitTx(142, 246, 'Lower envelope q(λ) = inf_{x} L(x, λ)', 228, { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });

  o += card(280, 16, 240, 76, C.lgreen, C.green, [
    { t: 'Universal Concavity of q', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'For fixed x: (λ,ν) ↦ L(x,λ,ν)', mono: true, fs: 12, fill: C.ink },
    { t: 'is affine ⇒ inf_{x} is ALWAYS concave', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);

  o += card(280, 103, 240, 76, C.lblue, C.blue, [
    { t: 'Holds Even for Nonconvex f!', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'Hypograph {(λ, t) : t ≤ q(λ)} is', fs: 12.5, fill: C.ink },
    { t: 'an intersection of half-spaces', fs: 12.5, fill: C.ink },
  ]);

  o += card(280, 190, 240, 76, C.lamber, C.amber, [
    { t: 'Convex Dual Problem', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'd* = sup_{λ⪰0,ν} q(λ,ν) ≤ p*', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark },
    { t: 'Maximizes a concave function', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function slaterInteriorPoint() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 12, 500, 42, C.lgreen, C.green, [
    { t: 'Slater’s Condition  ⇒  Strong Duality (d* = p*) & Attained Dual Multipliers', fs: 14.5, fw: 700, fill: D.green.dark },
  ]);

  o += titledPanel(20, 68, 240, 140, '1. Convex Primal Problem', D.blue, [
    { t: 'f(x), g₁(x), …, gₘ(x) convex', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'Equality constraints affine:', fs: 13, fill: C.ink },
    { t: 'hⱼ(x) = aⱼᵀx − bⱼ = 0', mono: true, fs: 13, fill: C.ink },
    { t: 'Epigraph set A is convex', fs: 12.5, fill: C.muted },
  ]);

  o += titledPanel(280, 68, 240, 140, '2. Strictly Feasible Point x̄', D.green, [
    { t: '∃ x̄ ∈ relint(D) such that:', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: '• gᵢ(x̄) < 0  (non-affine gᵢ)', mono: true, fs: 12.5, fill: C.ink },
    { t: '• gᵢ(x̄) ≤ 0  (affine gᵢ)', mono: true, fs: 12.5, fill: C.ink },
    { t: '• hⱼ(x̄) = 0  (all equalities)', mono: true, fs: 12.5, fill: C.ink },
  ]);

  o += card(20, 222, 500, 46, D.purple.tint, C.purple, [
    { t: 'Bonus: Checked by testing a single interior point x̄ (e.g. x̄ = 0 in a ball) without knowing x*!', fs: 13, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function saddleMinmaxGame() {
  const W = 540, H = 320;
  let o = '';
  o += card(20, 12, 500, 44, D.purple.tint, C.purple, [
    { t: 'Min–Max Swapability:  d* = sup_{λ⪰0} inf_{x} L(x,λ)  ≤  inf_{x} sup_{λ⪰0} L(x,λ) = p*', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
  ]);

  o += card(20, 70, 152, 130, C.lgreen, C.green, [
    { t: 'Dual Max (λ ⪰ 0)', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'L(x*, λ) ≤ L(x*, λ*)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Concave downward', fs: 12.5, fill: C.muted },
    { t: 'along multipliers λ', fs: 12.5, fill: C.muted },
  ]);

  o += arrow(176, 135, 198, 135, C.green);

  o += card(202, 70, 136, 130, C.lamber, C.amber, [
    { t: 'Saddle Point', fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: '(x*, λ*, ν*)', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'd* = L* = p*', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Zero gap!', fs: 13, fill: D.green.dark },
  ]);

  o += arrow(362, 135, 340, 135, C.blue);

  o += card(366, 70, 154, 130, C.lblue, C.blue, [
    { t: 'Primal Min (x ∈ D)', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'L(x*, λ*) ≤ L(x, λ*)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Convex upward', fs: 12.5, fill: C.muted },
    { t: 'along parameters x', fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 214, 500, 92, C.lblue, C.blue, [
    { t: 'Saddle Inequality Proves KKT:', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '∇ₓL(x*, λ*, ν*) = 0   and   ∑ λᵢ* gᵢ(x*) = 0', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Strong duality + attained optima holds if and only if (x*, λ*, ν*) is a Lagrangian saddle point', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function bilinearGdaSpiral() {
  const W = 540, H = 280;
  let o = '';
  o += titledPanel(20, 14, 240, 188, 'Simultaneous GDA on min_{x} max_{y} xy', D.red, [
    { t: 'xₖ₊₁ = xₖ − η yₖ', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'yₖ₊₁ = yₖ + η xₖ', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'rₖ₊₁² = (1 + η²) rₖ²', mono: true, fs: 14, fw: 700, fill: D.red.dark },
    { t: 'Spirals outward for ANY η > 0!', fs: 13, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(280, 14, 240, 188, 'Stabilizing Min–Max Games', D.green, [
    { t: '• Extragradient (Korpelevich):', fs: 13, fw: 700, fill: D.green.dark },
    { t: '  Look-ahead step damps rotation', fs: 12.5, fill: C.ink },
    { t: '• Optimistic GDA / Damping:', fs: 13, fw: 700, fill: D.green.dark },
    { t: '  Contracts inward to saddle (0, 0)', fs: 12.5, fill: C.ink },
  ]);

  o += card(20, 216, 500, 50, D.purple.tint, C.purple, [
    { t: 'Three ML Roles of Duality:  (1) Primal Recovery x(λ*)  ·  (2) n ≪ d Duals  ·  (3) GAN/Robust Games', fs: 13, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function weightDecayNormBall() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 14, 186, 174, C.lblue, C.blue, [
    { t: 'Penalized Ridge Form', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'min_{w}  ℓ(w) + (λ/2)‖w‖₂²', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Fix weight decay λ > 0', fs: 13, fill: C.ink },
    { t: 'Minimizer w_λ defines radius', fs: 12.5, fill: C.muted },
    { t: 'r = ‖w_λ‖₂', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
  ]);

  o += arrow(212, 82, 328, 82, C.purple);
  o += fitTx(270, 62, 'r = ‖w_λ‖₂', 110, { mono: true, fs: 13, fw: 700, fill: D.purple.dark });
  o += arrow(328, 124, 212, 124, C.purple);
  o += fitTx(270, 146, 'λ* = −∂p*/∂(½r²)', 110, { mono: true, fs: 12, fw: 700, fill: D.purple.dark });

  o += card(334, 14, 186, 174, C.lgreen, C.green, [
    { t: 'Constrained Norm Ball', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'min_{w} ℓ(w)  s.t. ½‖w‖₂² ≤ ½r²', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'Fix norm budget r > 0', fs: 13, fill: C.ink },
    { t: 'Slater holds at w̄ = 0;', fs: 12.5, fill: C.muted },
    { t: 'recovery λ* = λ', mono: true, fs: 14, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 204, 500, 60, C.lamber, C.amber, [
    { t: 'Shared KKT Stationarity Condition:   ∇ℓ(w*) + λ w* = 0', mono: true, fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: 'For convex losses ℓ(w), sweeping λ ∈ (0, ∞) traces the exact same path as sweeping r', fs: 13, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function fenchelConjugatePairs() {
  const W = 540, H = 286;
  let o = '';
  o += titledPanel(20, 12, 240, 122, '1. Isotropic Quadratic', D.blue, [
    { t: 'f(x) = ½‖x‖₂²  ↔  f*(y) = ½‖y‖₂²', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Self-conjugate; maximizer x* = y', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 12, 240, 122, '2. Elliptical Quadratic (A ≻ 0)', D.green, [
    { t: '½ xᵀAx  ↔  ½ yᵀA⁻¹y', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Inverts metric; maximizer x* = A⁻¹y', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(20, 148, 240, 124, '3. Exponential & Entropy', D.orange, [
    { t: 'f(x) = eˣ  ↔  f*(y) = y log y − y', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark },
    { t: 'Domain y > 0; maximizer x* = log y', fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 148, 240, 124, '4. ℓ₁ Penalty & ℓ_{∞} Box', D.purple, [
    { t: 'λ‖x‖₁  ↔  I_{‖y‖_{∞} ≤ λ}(y)', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    { t: '0 inside box ‖y‖_{∞} ≤ λ, +∞ outside', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function lseEntropyDuality() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 14, 186, 174, C.lblue, C.blue, [
    { t: 'Logit Space x ∈ ℝⁿ', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'f(x) = lse(x) = log ∑ e^{xᵢ}', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Smooth convex envelope', fs: 13, fill: C.ink },
    { t: 'of maxᵢ xᵢ', mono: true, fs: 13, fill: D.blue.dark },
  ]);

  o += arrow(212, 82, 328, 82, C.purple);
  o += fitTx(270, 62, 'p* = softmax(x)', 110, { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });
  o += arrow(328, 124, 212, 124, C.green);
  o += fitTx(270, 146, 'f** = f', 110, { mono: true, fs: 13, fw: 700, fill: D.green.dark });

  o += card(334, 14, 186, 174, C.lgreen, C.green, [
    { t: 'Simplex p ∈ Δ', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'lse*(p) = ∑ pᵢ log pᵢ', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: '= −H(p)  (Neg. Entropy)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '+∞ outside simplex Δ', fs: 12.5, fill: D.red.dark },
  ]);

  o += card(20, 204, 500, 60, D.purple.tint, C.purple, [
    { t: 'Variational Principle:   lse(x) = max_{p ∈ Δ} (pᵀx + H(p))', mono: true, fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'Softmax is the unique distribution balancing expected score pᵀx against Shannon entropy H(p)', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function svmMarginSupport() {
  const W = 540, H = 282;
  let o = '';
  o += rect(20, 16, 240, 250, '#FFFFFF', C.gray, 8, 1.5);
  o += line(44, 196, 196, 44, C.blue, 2, true);
  o += line(64, 216, 216, 64, C.ink, 2.5, false);
  o += line(84, 236, 236, 84, D.red.base, 2, true);
  o += circle(94, 146, 7, C.lblue, C.blue, 2.5);
  o += circle(156, 84, 7, C.lblue, C.blue, 2.5);
  o += circle(144, 176, 7, D.red.tint, D.red.base, 2.5);
  o += circle(62, 92, 4.5, C.blue, '#FFFFFF', 1.5);
  o += circle(108, 60, 4.5, C.blue, '#FFFFFF', 1.5);
  o += circle(192, 186, 4.5, D.red.base, '#FFFFFF', 1.5);
  o += circle(212, 146, 4.5, D.red.base, '#FFFFFF', 1.5);
  o += fitTx(140, 252, 'Active SVs (αᵢ* > 0) vs Interior (αᵢ* = 0)', 224, { fs: 12, fw: 700, fill: C.ink });

  o += card(276, 16, 244, 76, C.lblue, C.blue, [
    { t: 'Complementary Slackness', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'αᵢ* (1 − yᵢ(w*ᵀxᵢ + b*)) = 0', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'αᵢ* > 0 ⇒ point sits on margin ±1', fs: 12.5, fill: D.blue.dark },
  ]);

  o += card(276, 103, 244, 76, C.lgreen, C.green, [
    { t: 'Sparse Dual Expansion', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'w* = ∑_{i ∈ SV} αᵢ* yᵢ xᵢ', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Interior points can be deleted!', fs: 12.5, fill: C.ink },
  ]);

  o += card(276, 190, 244, 76, D.purple.tint, C.purple, [
    { t: 'The Kernel Trick', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'Dual & w*ᵀx use only xᵢᵀxⱼ', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Replace xᵢᵀxⱼ ↦ k(xᵢ, xⱼ)', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §35.8 mdl-numerical-stability-conditioning.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function nanVsSilentDrift() {
  const W = 540, H = 284;
  let o = '';
  o += titledPanel(20, 14, 240, 192, 'Mode 1: Unstable Algorithm', D.red, [
    { t: 'Problem: Softmax on (1000,1001,1002)', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Well-conditioned problem (κ ≈ 1)', fs: 12.5, fill: D.green.dark },
    { t: 'Naive exp(1000) → ∞ → NaN!', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Cure: Rewrite Algorithm', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Max-shift z − max(z) is exact', mono: true, fs: 12.5, fill: C.ink },
  ]);

  o += titledPanel(280, 14, 240, 192, 'Mode 2: Ill-Conditioned Problem', D.orange, [
    { t: 'Problem: 12×12 Hilbert Solve Hx = b', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Backward-stable solver (ω ≈ 10⁻¹⁷)', fs: 12.5, fill: D.green.dark },
    { t: 'κ(H) ≈ 10¹⁶ ⇒ 30% Silent Drift!', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Cure: Regularize Problem', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Add ridge λI to lift spectral floor', fs: 12.5, fill: C.ink },
  ]);

  o += card(20, 220, 500, 50, C.lblue, C.blue, [
    { t: 'Diagnostic Rule: Small residual ω ≈ u acquits the solver and blames problem conditioning κ', fs: 13, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function log1pDetour() {
  const W = 540, H = 302;
  let o = '';
  o += card(20, 12, 500, 54, C.lgreen, C.green, [
    { t: 'Direct Route:  log1p(x) = x − ½x² + ⋯   at x = 10⁻⁸', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: '(κ_f → 1, 0% Error in fp32)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ], { spacing: 20 });

  o += card(20, 80, 144, 132, C.lblue, C.blue, [
    { t: 'Input x', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'x = 10⁻⁸', mono: true, fs: 14.5, fw: 700, fill: C.ink },
    { t: '(Below fp32', fs: 12.5, fill: C.muted },
    { t: 'u ≈ 6×10⁻⁸)', mono: true, fs: 12, fill: C.muted },
  ]);
  o += arrow(168, 146, 194, 146, D.orange.base);

  o += card(198, 80, 144, 132, C.lamber, C.amber, [
    { t: 'Step 1', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'y = fl(1+x)', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Rounds to y = 1.0', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Tiny rel. error 10⁻⁸', fs: 12.5, fill: C.ink },
    { t: '(κ₊ ≈ 1)', mono: true, fs: 12.5, fill: D.green.dark },
  ]);
  o += arrow(346, 146, 372, 146, D.red.base);

  o += card(376, 80, 144, 132, D.red.tint, D.red.base, [
    { t: 'Step 2', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'z = log(y)', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'κ_log(y) = 1/|log y|', mono: true, fs: 12, fill: C.ink },
    { t: '≈ 1/x = 10⁸!', mono: true, fs: 14, fw: 700, fill: D.red.dark },
    { t: 'z = 0 (100% err)', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
  ]);

  o += card(20, 226, 500, 62, D.purple.tint, C.purple, [
    { t: 'Golden Rule of Numerical Stability:', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'An unstable algorithm routes a well-conditioned problem through an ill-conditioned intermediate step', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function welfordVsNaive() {
  const W = 540, H = 282;
  let o = '';
  o += titledPanel(20, 14, 240, 190, 'Naive One-Pass: E[x²] − m²', D.red, [
    { t: 'Data: m = 10⁹,  s = 1 (fp64)', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Forms (1/N)∑xᵢ² ≈ 10¹⁸ and m² ≈ 10¹⁸', mono: true, fs: 11.5, fill: C.ink },
    { t: 'Grid spacing at 10¹⁸ is 128!', fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Squares condition number:', fs: 12.5, fill: C.ink },
    { t: 'κ_sub = 2m²/s² = 2×10¹⁸ (0 digits)', mono: true, fs: 12, fw: 700, fill: D.red.dark },
  ]);

  o += titledPanel(280, 14, 240, 190, 'Welford Streaming Recursion', D.green, [
    { t: 'dₖ = xₖ − mₖ₋₁', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'mₖ = mₖ₋₁ + dₖ / k', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'Mₖ = Mₖ₋₁ + dₖ (xₖ − mₖ)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Accumulates O(1) centered squares;', fs: 12.5, fill: C.ink },
    { t: 'κ_var ≈ 2|m|/s = 2×10⁹ (7+ digits)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
  ]);

  o += card(20, 218, 500, 50, C.lblue, C.blue, [
    { t: 'Exact Algebraic Identity:  ∑_{i=1}^{k} (xᵢ − mₖ)²  =  Mₖ₋₁ + (xₖ − mₖ₋₁)(xₖ − mₖ)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function backwardForwardError() {
  const W = 540, H = 280;
  let o = '';
  o += card(20, 14, 194, 174, C.lgreen, C.green, [
    { t: 'Backward Error ω', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '(A + ΔA)x̂ = b', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'ω = ‖b − Ax̂‖₂ / (‖A‖₂‖x̂‖₂)', mono: true, fs: 12, fill: C.ink },
    { t: 'Stable solver: ω ≈ u', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: '(Measures Algorithm)', fs: 12.5, fill: C.muted },
  ]);

  o += arrow(220, 101, 320, 101, D.red.base);
  o += fitTx(270, 78, '× κ(A)', 96, { mono: true, fs: 15, fw: 700, fill: D.red.dark });
  o += fitTx(270, 124, 'σ₁ / σ_d', 96, { mono: true, fs: 13, fw: 600, fill: C.muted });

  o += card(326, 14, 194, 174, D.red.tint, D.red.base, [
    { t: 'Forward Error in x̂', fs: 14.5, fw: 700, fill: D.red.dark },
    { t: '‖x̂ − x‖₂ / ‖x̂‖₂ ≤ κ(A) ω', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Amplifies data & rounding', fs: 12.5, fill: C.ink },
    { t: 'noise by κ(A) = ‖A‖₂‖A⁻¹‖₂', mono: true, fs: 12, fill: C.ink },
    { t: '(Measures Problem + Alg)', fs: 12.5, fill: C.muted },
  ]);

  o += card(20, 204, 500, 60, D.purple.tint, C.purple, [
    { t: 'Lost-Digits Rule of Thumb:   Accurate Digits ≈ log₁₀(1 / u) − log₁₀ κ(A)', mono: true, fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'fp32 (~7 digits) loses 5 digits at κ = 10⁵  ·  fp64 (~16 digits) loses all 16 digits at κ = 10¹⁶', fs: 12.5, fill: C.ink },
  ]);
  return svg(W, H, o);
}

export const diagrams = {
  // index.md (4)
  'ch35-opt-three-questions':          threeQuestions,
  'ch35-opt-benchmark-valley':         benchmarkValley,
  'ch35-opt-chapter-architecture':     chapterArchitecture,
  'ch35-opt-study-roadmap':            studyRoadmap,
  // mdl-gradient-based-optimization.md (7)
  'ch35-opt-three-step-regimes':       threeStepRegimes,
  'ch35-opt-steepest-descent-cone':    steepestDescentCone,
  'ch35-opt-lipschitz-quadratic-ceiling': lipschitzQuadraticCeiling,
  'ch35-opt-saddle-halt':              saddleHalt,
  'ch35-opt-armijo-backtracking':      armijoBacktracking,
  'ch35-opt-edge-of-stability':        edgeOfStability,
  'ch35-opt-nesterov-lookahead':       nesterovLookahead,
  // mdl-curvature-methods.md (10)
  'ch35-opt-curvature-vs-first-order': curvatureVsFirstOrder,
  'ch35-opt-newton-quadratic-model':   newtonQuadraticModel,
  'ch35-opt-mahalanobis-steepest':     mahalanobisSteepest,
  'ch35-opt-affine-invariance':        affineInvariance,
  'ch35-opt-quadratic-doubling-digits': quadraticDoublingDigits,
  'ch35-opt-hessian-scale-wall':       hessianScaleWall,
  'ch35-opt-saddle-wrong-way':         saddleWrongWay,
  'ch35-opt-secant-chord-slope':       secantChordSlope,
  'ch35-opt-bfgs-erase-write':         bfgsEraseWrite,
  'ch35-opt-trust-region-damping':     trustRegionDamping,
  // mdl-convexity.md (5)
  'ch35-opt-intersection-hull':        intersectionHull,
  'ch35-opt-epigraph-sublevel':        epigraphSublevel,
  'ch35-opt-three-tests-triangle':     threeTestsTriangle,
  'ch35-opt-jensen-gap-variance':      jensenGapVariance,
  'ch35-opt-closure-rules':            closureRules,
  // mdl-stochastic-gradient-methods.md (8)
  'ch35-opt-minibatch-variance-law':   minibatchVarianceLaw,
  'ch35-opt-second-moment-balance':    secondMomentBalance,
  'ch35-opt-valley-noise-penalty':     valleyNoisePenalty,
  'ch35-opt-robbins-monro-balance':    robbinsMonroBalance,
  'ch35-opt-one-over-k-threshold':     oneOverKThreshold,
  'ch35-opt-convex-sgd-telescoping':   convexSgdTelescoping,
  'ch35-opt-ghadimi-lan-random-stop':  ghadimiLanRandomStop,
  'ch35-opt-svrg-control-variate':     svrgControlVariate,
  // mdl-adaptive-stochastic-methods.md (9)
  'ch35-opt-diagonal-precond-proxy':   diagonalPrecondProxy,
  'ch35-opt-adagrad-rare-frequent':    adagradRareFrequent,
  'ch35-opt-rmsprop-ema-window':       rmspropEmaWindow,
  'ch35-opt-adam-dual-moment-flow':    adamDualMomentFlow,
  'ch35-opt-adam-scale-invariance':    adamScaleInvariance,
  'ch35-opt-rotated-valley-failure':   rotatedValleyFailure,
  'ch35-opt-reddi-cycle-drift':        reddiCycleDrift,
  'ch35-opt-adamw-linf-box':           adamwLinfBox,
  'ch35-opt-preconditioning-ladder':   preconditioningLadder,
  // mdl-constrained-optimization-duality.md (7)
  'ch35-opt-boundary-force-balance':   boundaryForceBalance,
  'ch35-opt-hyperplane-shadow-price':  hyperplaneShadowPrice,
  'ch35-opt-active-vs-inactive-ball':  activeVsInactiveBall,
  'ch35-opt-kkt-four-conditions':      kktFourConditions,
  'ch35-opt-projection-nonexpansive':  projectionNonexpansive,
  'ch35-opt-pgd-step-cycle':           pgdStepCycle,
  'ch35-opt-soft-thresholding-prox':   softThresholdingProx,
  // mdl-lagrangian-duality.md (9)
  'ch35-opt-dual-lower-bound-cert':    dualLowerBoundCert,
  'ch35-opt-concave-infimum-envelope': concaveInfimumEnvelope,
  'ch35-opt-slater-interior-point':    slaterInteriorPoint,
  'ch35-opt-saddle-minmax-game':       saddleMinmaxGame,
  'ch35-opt-bilinear-gda-spiral':      bilinearGdaSpiral,
  'ch35-opt-weight-decay-norm-ball':   weightDecayNormBall,
  'ch35-opt-fenchel-conjugate-pairs':  fenchelConjugatePairs,
  'ch35-opt-lse-entropy-duality':      lseEntropyDuality,
  'ch35-opt-svm-margin-support':       svmMarginSupport,
  // mdl-numerical-stability-conditioning.md (4)
  'ch35-opt-nan-vs-silent-drift':      nanVsSilentDrift,
  'ch35-opt-log1p-detour':             log1pDetour,
  'ch35-opt-welford-vs-naive':         welfordVsNaive,
  'ch35-opt-backward-forward-error':   backwardForwardError,
};
