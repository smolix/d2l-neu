// Slide diagrams for Chapter 4: Multilayer Perceptrons (ch04-mlp.mjs).
// Every diagram is pure DOM-free SVG built with engine.mjs helpers and tokens,
// designed for zero text-overlap, zero border-clipping, zero line-text
// intersection, and zero viewBox overflow.

import { C, TOKENS, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const PAL = {
  blue:   { fill: C.lblue,        stroke: C.blue,        dark: D.blue.dark,   rail: C.blue },
  green:  { fill: C.lgreen,       stroke: C.green,       dark: D.green.dark,  rail: C.green },
  orange: { fill: C.lamber,       stroke: C.amber,       dark: D.orange.dark, rail: C.amber },
  purple: { fill: D.purple.tint,  stroke: C.purple,      dark: D.purple.dark, rail: C.purple },
  red:    { fill: D.red.tint,     stroke: D.red.base,    dark: D.red.dark,    rail: D.red.base },
  teal:   { fill: D.teal.tint,    stroke: D.teal.base,   dark: D.teal.dark,   rail: D.teal.base },
  gold:   { fill: D.gold.tint,    stroke: D.gold.base,   dark: D.gold.dark,   rail: D.gold.base },
  slate:  { fill: N.panel,        stroke: C.gray,        dark: C.ink,         rail: C.gray },
  white:  { fill: '#FFFFFF',      stroke: C.gray,        dark: C.ink,         rail: C.gray },
};

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const fillRect = (x, y, w, h, fill, rx = 4) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}"/>`;

const hline = (x1, x2, y, color = N.hairline, dash = true) =>
  `<line x1="${x1}" y1="${y}" x2="${x2}" y2="${y}" stroke="${color}" stroke-width="1.5"${dash ? ' stroke-dasharray="4 4"' : ''}/>`;

const vline = (x, y1, y2, color = N.hairline, dash = true) =>
  `<line x1="${x}" y1="${y1}" x2="${x}" y2="${y2}" stroke="${color}" stroke-width="1.5"${dash ? ' stroke-dasharray="4 4"' : ''}/>`;

// Text fitting. Widths are the bundled faces' advances (Source Sans 3
// Regular / Bold, chars 32-126, in 1/100 em; weight 500 is measured as
// Regular plus 3%), and 0.60 em for JetBrains Mono, so a fit estimate
// matches the outlined SVG. A label is shrunk no further than FIT_FLOOR px:
// a card line that would need more is wrapped onto two lines instead.
const ADV = {
  r: '2029435050826125303042502531253550505050505050505050252550505042855459576253496265264858497365665766575354645279514854303530505054505546565029505425255026835554565635423454477245474230243050',
  b: '2034545353866730343446533033303453535353535353535353303053535346905760586455526467305161527666686068615656665681575254343434535056535747575234535728285529865756575740443857527851524634273453',
};
const FIT_FLOOR = 11;
const FIT_PAD = 10;

function emWidth(s, o = {}) {
  let w = 0;
  const bold = (o.fw || 500) >= 600;
  for (const ch of String(s)) {
    const c = ch.codePointAt(0);
    if (c >= 0x300 && c <= 0x36f) continue;          // combining mark
    if (o.mono) { w += 0.6; continue; }
    if (c >= 32 && c < 127) {
      const t = bold ? ADV.b : ADV.r;
      w += +t.substr((c - 32) * 2, 2) / 100;
    } else {
      w += 0.58;
    }
  }
  return o.mono || bold ? w : w * 1.03;
}

// Largest font size (<= o.fs) at which str fits in maxW (no floor).
function fitSize(str, maxW, o = {}) {
  const base = o.fs || 13.5;
  const em = emWidth(str, o);
  if (!maxW || em * base <= maxW - FIT_PAD) return base;
  return +((maxW - FIT_PAD) / em).toFixed(2);
}

// Split str at a space into two lines: the longer half as short as possible,
// preferring breaks after punctuation and separators, and avoiding breaks
// inside brackets or in front of an operator.
function splitTwo(str, o = {}) {
  const s = String(str);
  let best = null, depth = 0;
  for (let i = 1; i < s.length - 1; i++) {
    const ch = s[i];
    if ('([{'.includes(ch)) depth++;
    if (')]}'.includes(ch)) depth = Math.max(0, depth - 1);
    if (ch !== ' ') continue;
    const a = s.slice(0, i).trimEnd(), b = s.slice(i + 1).trimStart();
    if (!a || !b) continue;
    const prev = a[a.length - 1], next = b[0];
    let pen = 0.12;
    if (',;:·|→—–'.includes(prev)) pen = 0;
    if ('·=+×→⇒|−-–—)]}≈≤≥<>'.includes(next)) pen += 0.5;
    if (depth > 0) pen += 0.5;
    const m = Math.max(emWidth(a, o), emWidth(b, o)) * (1 + pen);
    if (!best || m < best.m) best = { a, b, m };
  }
  return best ? [best.a, best.b] : null;
}

function fitWarn(msg) {
  if (globalThis.__fitLog) globalThis.__fitLog.push(msg);
}

// One label, shrunk to fit maxW but never below FIT_FLOOR.
function fitTx(x, y, str, maxW, o = {}) {
  const s = String(str);
  const f = fitSize(s, maxW, o);
  if (f < FIT_FLOOR) fitWarn(`"${s}" needs ${f}px in ${maxW}`);
  return tx(x, y, s, { ...o, fs: Math.max(FIT_FLOOR, f) });
}

// Multi-line card with guaranteed internal padding so text never touches borders.
// lines: array of { t, fs, fw, fill, mono } or plain strings. A line that
// would shrink below FIT_FLOOR (or by more than 15%, when the card has the
// height for it) is wrapped onto two lines; a '\n' in a line sets the break.
function card(x, y, w, h, themeKey, lines, opts = {}) {
  const p = PAL[themeKey] || PAL.slate;
  const fill = opts.white ? '#FFFFFF' : p.fill;
  const stroke = p.stroke;
  let o = box(x, y, w, h, fill, stroke, opts.rx || 8, opts.sw || 2, opts.dash || false);
  if (opts.rail) {
    o += fillRect(x + 1, y + 2, 5, h - 4, p.rail, 3);
  }
  const n = lines.length;
  if (n === 0) return o;
  const cx = opts.anchor === 'start' ? x + (opts.rail ? 14 : 12) : x + w / 2;
  const availW = w - (opts.rail ? 20 : 14);
  const items = lines.map((ln, i) => {
    const obj = typeof ln === 'string' ? {} : ln;
    const st = {
      fs: obj.fs || (i === 0 ? 14 : 12.8),
      fw: obj.fw || (i === 0 ? 700 : 500),
      fill: obj.fill || (i === 0 ? p.dark : C.ink),
      anchor: obj.anchor || opts.anchor || 'middle',
    };
    if (obj.mono) st.mono = true;
    const t = String(typeof ln === 'string' ? ln : ln.t);
    if (t.includes('\n')) {   // an explicit line break
      const two = t.split('\n', 2);
      const fs = Math.min(...two.map((s) => fitSize(s, availW, st)));
      return { t: two.join(' '), st, f: fs, parts: { two, fs }, fixed: true };
    }
    return { t, st, f: fitSize(t, availW, st), parts: null };
  });
  // Wrapping plan: lines under the floor must wrap; lines shrunk by >15% wrap
  // when the card is tall enough to take the extra row.
  const wrapOf = (it) => {
    const two = splitTwo(it.t, it.st);
    if (!two) return null;
    let fs = Math.min(...two.map((s) => fitSize(s, availW, it.st)));
    if (n === 1) fs = Math.min(fs, +((h - 5) / 2.18).toFixed(2));   // two rows must fit the height
    return fs > it.f ? { two, fs } : null;
  };
  const layout = () => {
    const rows = [];
    items.forEach((it, i) => {
      if (it.parts) {
        rows.push({ t: it.parts.two[0], st: { ...it.st, fs: Math.max(FIT_FLOOR, it.parts.fs) }, gap: i ? null : 0 });
        rows.push({ t: it.parts.two[1], st: { ...it.st, fs: Math.max(FIT_FLOOR, it.parts.fs) }, gap: +(Math.max(FIT_FLOOR, it.parts.fs) * 1.18).toFixed(2) });
      } else {
        rows.push({ t: it.t, st: { ...it.st, fs: Math.max(FIT_FLOOR, it.f) }, gap: i ? null : 0 });
      }
    });
    const wrapSum = rows.reduce((a, r) => a + (r.gap || 0), 0);
    const maxStep = n <= 2 ? 19 : 20;
    const itemStep = n <= 1 ? 0 : Math.min(maxStep, (h - 22 - wrapSum) / (n - 1));
    const gapOf = (r) => (r.gap === null ? itemStep : r.gap);
    const ok = rows.every((r, i) => !i || gapOf(r) >= 0.575 * (r.st.fs + rows[i - 1].st.fs))
      && wrapSum + (n - 1) * itemStep + (rows[0].st.fs + rows[rows.length - 1].st.fs) / 2 <= h - 4.5;
    return { rows, itemStep, wrapSum, ok };
  };
  for (const it of items) {
    if (it.fixed) {
      if (it.f < FIT_FLOOR) fitWarn(`"${it.t}" needs ${it.f}px even wrapped, in ${availW}`);
    } else if (it.f < FIT_FLOOR) {
      it.parts = wrapOf(it);
      if (!it.parts) fitWarn(`"${it.t}" needs ${it.f}px in ${availW} and has no space to wrap`);
      else if (it.parts.fs < FIT_FLOOR) fitWarn(`"${it.t}" needs ${it.parts.fs}px even wrapped, in ${availW}`);
    }
  }
  for (const it of items) {
    if (!it.parts && it.f < 0.85 * it.st.fs) {
      it.parts = wrapOf(it);
      if (it.parts && !layout().ok) it.parts = null;
    }
  }
  if (!items.some((it) => it.parts)) {
    // No wrapping: the original even spacing.
    const maxStep = n <= 2 ? 19 : 20;
    const span = n <= 1 ? 0 : Math.min((n - 1) * maxStep, h - 22);
    const y0 = y + h / 2 - span / 2;
    const step = n <= 1 ? 0 : span / (n - 1);
    items.forEach((it, i) => {
      const ly = n === 1 ? y + h / 2 : y0 + i * step;
      o += fitTx(cx, ly, it.t, availW, it.st);
    });
    return o;
  }
  const L = layout();
  if (!L.ok) fitWarn(`card at (${x},${y}) ${w}x${h} too short for wrapped text "${items.map((it) => it.t).join(' | ')}"`);
  const span = L.wrapSum + (n - 1) * L.itemStep;
  let ly = y + h / 2 - span / 2;
  L.rows.forEach((r, i) => {
    if (i) ly += r.gap === null ? L.itemStep : r.gap;
    o += tx(cx, +ly.toFixed(2), r.t, r.st);
  });
  return o;
}

// Labeled horizontal arrow with label safely above and sub-label safely below.
function hArrow(x1, x2, y, color, topLabel, botLabel) {
  let o = arrow(x1, y, x2, y, color);
  const mx = (x1 + x2) / 2;
  const span = Math.abs(x2 - x1);
  if (topLabel) o += fitTx(mx, y - 15, topLabel, span + 24, { fs: 12.5, fw: 700, fill: color, mono: true });
  if (botLabel) o += fitTx(mx, y + 17, botLabel, span + 24, { fs: 12, fw: 600, fill: C.muted });
  return o;
}

// Labeled vertical arrow with label safely to the right (or left).
function vArrow(x, y1, y2, color, rightLabel, side = 'right') {
  let o = arrow(x, y1, x, y2, color);
  const my = (y1 + y2) / 2;
  if (rightLabel) {
    const lx = side === 'right' ? x + 10 : x - 10;
    const anc = side === 'right' ? 'start' : 'end';
    o += tx(lx, my, rightLabel, { fs: 12.5, fw: 700, fill: color, mono: true, anchor: anc });
  }
  return o;
}

// ═══════════════════════════════════════════════════════════════════════════
// §4.0 INDEX.MD DIAGRAMS (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 1. ch04-mlp-xor-ceiling
function xorCeiling() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 12, 244, 34, 'red', ['Linear Classifier Ceiling (~84%)']);
  o += card(280, 12, 244, 34, 'green', ['MLP Hidden Features (~87%)']);

  // Left: two failure boxes
  o += card(16, 56, 244, 88, 'red', [
    '1. No Feature Interactions (XOR)',
    { t: 'o = w₁x₁ + w₂x₂ + b', mono: true, fs: 13, fill: D.red.dark },
    'Single hyperplane fails on {0,1}²',
  ], { white: true, rail: true });

  o += card(16, 156, 244, 92, 'orange', [
    '2. Context-Blind Pixel Weights',
    'Fixed weight w_ij per pixel:',
    'cannot tell collar vs. sleeve apart',
  ], { white: true, rail: true });

  // Right: MLP pipeline
  o += card(280, 56, 244, 54, 'blue', [
    { t: 'Input X : (256, 784)', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    'Flattened 28×28 garment batch',
  ]);
  o += vArrow(340, 113, 133, C.blue, 'W⁽¹⁾ (784×256) + ReLU');
  o += card(280, 136, 244, 54, 'purple', [
    { t: 'Hidden H = φ(XW⁽¹⁾ + b⁽¹⁾)', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    '256 learned garment-part detectors',
  ]);
  o += vArrow(340, 193, 213, C.green, 'W⁽²⁾ (256×10) readout');
  o += card(280, 216, 244, 34, 'green', [
    { t: 'Logits O : (256, 10) → ~87% acc', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

// 2. ch04-mlp-three-challenges
function threeChallenges() {
  const W = 540, H = 268;
  let o = '';
  o += card(16, 12, 508, 36, 'purple', ['Stacking Nonlinear Hidden Layers Introduces Three Core Challenges']);

  const cols = [
    {
      x: 16, theme: 'blue',
      head: 'Backpropagation',
      lines: ['203,530 Gradients', { t: '1 reverse VJP sweep', mono: true, fill: D.blue.dark }, 'Cost: ~2× fwd FLOPs', { t: 'RAM: O(n·K·h) cache', mono: true }],
    },
    {
      x: 190, theme: 'orange',
      head: 'Numerical Stability',
      lines: ['K-Layer Matrix Product', { t: 'h·τ²·E[φ\'(z)²] = 1', mono: true, fill: D.orange.dark }, 'Xavier: 2/(n_in+n_out)', { t: 'He (ReLU): 2 / n_in', mono: true }],
    },
    {
      x: 364, theme: 'green',
      head: 'Generalization',
      lines: ['p = 203.5K > n = 60K', { t: 'Memorizes random y', mono: true, fill: D.green.dark }, 'Min-norm SGD bias', { t: 'Dropout: ξ·h/(1−p)', mono: true }],
    },
  ];

  cols.forEach((c) => {
    o += card(c.x, 60, 160, 34, c.theme, [c.head]);
    o += card(c.x, 102, 160, 114, c.theme, c.lines, { white: true });
  });

  o += card(16, 226, 508, 32, 'slate', [
    'All three converge in End-to-End Kaggle House Price Prediction',
  ]);
  return svg(W, H, o);
}

// 3. ch04-mlp-benchmark-ladder
function benchmarkLadder() {
  const W = 540, H = 276;
  let o = '';
  o += card(16, 10, 508, 32, 'slate', ['Fashion-MNIST Running Benchmark (d=784, q=10, B=256, η=0.1)']);

  const rows = [
    { y: 50,  theme: 'slate',  arch: 'Softmax (784→10)',             params: '7.85K params · 0.8 MB', acc: '~84% val', w: 126 },
    { y: 94,  theme: 'blue',   arch: '1-Hidden MLP (784→256→10)',    params: '203.5K params · 1.1 MB', acc: '~87% val', w: 130 },
    { y: 138, theme: 'purple', arch: '2-Hidden Dropout (256→256)',   params: '269.3K params · 1.3 MB', acc: '~87% val', w: 130 },
    { y: 182, theme: 'red',    arch: '10-Layer MLP · N(0, 0.01²)',   params: '795.7K params · 3.4 MB', acc: '10% stuck!', w: 15 },
    { y: 226, theme: 'green',  arch: '10-Layer MLP · He Init (2/h)', params: '795.7K params · 3.4 MB', acc: '~85% val', w: 128 },
  ];

  rows.forEach((r) => {
    o += card(16, r.y, 194, 36, r.theme, [{ t: r.arch, fs: 13, fw: 700 }]);
    o += card(218, r.y, 144, 36, r.theme, [{ t: r.params, mono: true, fs: 12, fill: C.ink }], { white: true });
    // Accuracy bar box on right
    const p = PAL[r.theme];
    o += box(370, r.y, 154, 36, '#FFFFFF', p.stroke, 6, 1.8);
    o += fillRect(372, r.y + 2, r.w, 32, p.fill, 4);
    o += fitTx(447, r.y + 18, r.acc, 144, { mono: true, fs: 13, fw: 700, fill: p.dark });
  });
  return svg(W, H, o);
}

// 4. ch04-mlp-chapter-roadmap
function chapterRoadmap() {
  const W = 540, H = 264;
  let o = '';
  // Top row: §§4.1 -> 4.2 -> 4.3 -> 4.4
  const top = [
    { x: 14,  theme: 'blue',   sec: 'MLP & φ',     sub: '2^m depth folds' },
    { x: 146, theme: 'blue',   sec: 'Fashion MLP', sub: '203.5K → 87%' },
    { x: 278, theme: 'red',    sec: 'Backprop',    sub: '3 matrix VJP rules' },
    { x: 410, theme: 'orange', sec: 'He / Xavier', sub: 'h·τ²·E[φ\'²] = 1' },
  ];
  top.forEach((b, i) => {
    o += card(b.x, 18, 116, 74, b.theme, [
      { t: b.sec, fs: 13.5, fw: 700 },
      { t: b.sub, mono: true, fs: 12, fill: C.ink },
    ]);
    if (i < 3) o += arrow(b.x + 118, 55, top[i + 1].x - 2, 55, C.blue);
  });

  // Down connector from §4.4 to bottom row
  o += arrow(468, 96, 468, 142, D.orange.base);

  // Bottom row (right to left): §4.5 -> §4.6 -> §4.7
  const bot = [
    { x: 368, theme: 'purple', sec: 'Generalization', sub: 'Double descent & bias' },
    { x: 192, theme: 'purple', sec: 'Dropout Mask',   sub: 'h\' = ξ·h / (1 − p)' },
    { x: 14,  theme: 'green',  sec: 'Kaggle Houses',  sub: '5-fold CV & log-RMSE' },
  ];
  bot.forEach((b, i) => {
    o += card(b.x, 146, 158, 76, b.theme, [
      { t: b.sec, fs: 14, fw: 700 },
      { t: b.sub, mono: true, fs: 12.5, fill: C.ink },
    ]);
    if (i < 2) o += arrow(b.x - 2, 184, bot[i + 1].x + 160, 184, C.purple);
  });

  o += card(14, 232, 512, 24, 'slate', [
    { t: 'Representation  →  Optimization  →  Generalization', fs: 12.5, fw: 600 },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4.1 MLP.MD DIAGRAMS (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 5. ch04-mlp-monotonicity-limits
function monotonicityLimits() {
  const W = 540, H = 260;
  let o = '';
  // Left panel: Monotonicity failure
  o += card(14, 12, 248, 34, 'red', ['Failure 1: Forced Monotonicity']);
  o += box(14, 52, 248, 136, '#FFFFFF', D.red.base, 8, 1.8);
  // Axes inside left plot: x in [44..240], y in [70..158]
  o += `<line x1="44" y1="154" x2="240" y2="154" stroke="${C.gray}" stroke-width="1.5"/>`;
  o += `<line x1="44" y1="66" x2="44" y2="154" stroke="${C.gray}" stroke-width="1.5"/>`;
  // True U-curve: risk vs temperature (min at x=142)
  o += `<path d="M 56 76 Q 142 184 228 76" fill="none" stroke="${C.green}" stroke-width="2.5"/>`;
  // Monotonic linear line
  o += `<line x1="56" y1="142" x2="228" y2="82" stroke="${D.red.base}" stroke-width="2" stroke-dasharray="5 4"/>`;
  o += tx(142, 66, 'True U-risk vs. Linear w·T+b', { fs: 12, fw: 700, fill: C.ink });
  o += tx(142, 172, '37°C optimum (non-monotonic)', { fs: 12, fw: 600, fill: D.green.dark });

  o += card(14, 196, 248, 52, 'red', [
    'Linear o_j rises or falls everywhere;',
    'cannot reverse slope at 37°C',
  ], { white: true });

  // Right panel: Context-blind features
  o += card(278, 12, 248, 34, 'orange', ['Failure 2: Zero Feature Interactions']);
  o += card(278, 52, 248, 88, 'orange', [
    { t: 'Linear: ∂o_j/∂x_i = w_ij (fixed)', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    'Pixel x_i adds the exact same vote',
    'regardless of neighboring pixels x_k',
  ], { white: true, rail: true });
  o += card(278, 152, 248, 96, 'green', [
    'Hidden Layer Solution:',
    { t: 'h_m = φ(w₁x₁ + w₂x₂ + … + b)', mono: true, fs: 12.5, fill: D.green.dark },
    'Jointly detects multi-pixel patterns',
    '(collar cutout, sleeve, strap)',
  ], { white: true, rail: true });
  return svg(W, H, o);
}

// 6. ch04-mlp-xor-impossibility
function xorImpossibility() {
  const W = 540, H = 264;
  let o = '';
  // Left: Geometric square {0,1}^2
  o += card(14, 12, 236, 34, 'purple', ['XOR Geometry on {0, 1}²']);
  o += box(14, 52, 236, 200, '#FFFFFF', C.purple, 8, 1.8);

  // Square corners at (64, 194) = (0,0), (194, 194) = (1,0), (64, 84) = (0,1), (194, 84) = (1,1)
  const x0 = 68, x1 = 192, y0 = 200, y1 = 98;
  o += box(x0, y1, x1 - x0, y0 - y1, N.panel, C.gray, 2, 1.5, true);
  // Diagonals crossing at midpoint (130, 140)
  o += `<line x1="${x0 + 12}" y1="${y0 - 12}" x2="${x1 - 12}" y2="${y1 + 12}" stroke="${C.blue}" stroke-width="2"/>`;
  o += `<line x1="${x0 + 12}" y1="${y1 + 12}" x2="${x1 - 12}" y2="${y0 - 12}" stroke="${D.red.base}" stroke-width="2"/>`;

  // 4 corner badges
  o += card(x0 - 22, y0 - 14, 44, 28, 'blue', [{ t: '0', mono: true, fs: 14, fw: 700 }]);
  o += card(x1 - 22, y1 - 14, 44, 28, 'blue', [{ t: '0', mono: true, fs: 14, fw: 700 }]);
  o += card(x0 - 22, y1 - 14, 44, 28, 'red',  [{ t: '1', mono: true, fs: 14, fw: 700 }]);
  o += card(x1 - 22, y0 - 14, 44, 28, 'red',  [{ t: '1', mono: true, fs: 14, fw: 700 }]);

  o += tx(x0, y0 + 28, '(0,0)', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(x1, y0 + 28, '(1,0)', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(x0, y1 - 26, '(0,1)', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += tx(x1, y1 - 26, '(1,1)', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });

  // Right: Algebraic contradiction proof
  o += card(266, 12, 260, 34, 'red', ['Algebraic Contradiction Proof']);
  o += card(266, 54, 260, 60, 'red', [
    'Class 1 corners (0,1) & (1,0):',
    { t: 'w₂+b > 0, w₁+b > 0 ⇒ w₁+w₂+2b > 0', mono: true, fs: 12, fill: D.red.dark },
  ], { white: true, rail: true });
  o += card(266, 122, 260, 60, 'blue', [
    'Class 0 corners (0,0) & (1,1):',
    { t: 'b ≤ 0, w₁+w₂+b ≤ 0 ⇒ w₁+w₂+2b ≤ 0', mono: true, fs: 12, fill: D.blue.dark },
  ], { white: true, rail: true });
  o += card(266, 190, 260, 62, 'purple', [
    'Shared Diagonal Midpoint (½, ½):',
    'Both diagonals average to ½(w₁+w₂)+b',
    '⇒ No hyperplane separates XOR! ∎',
  ]);
  return svg(W, H, o);
}

// 7. ch04-mlp-linear-collapse
function linearCollapse() {
  const W = 540, H = 264;
  let o = '';
  // Top panel: Without phi (Linear collapse)
  o += card(14, 12, 512, 30, 'red', ['(a) Without Nonlinearity φ(z) = z : Matrix Associativity Collapses Depth']);
  o += card(14, 50, 96, 54, 'blue', [{ t: 'X', fs: 16, fw: 700 }, { t: '(n × d)', mono: true, fs: 12.5 }]);
  o += hArrow(114, 152, 77, D.red.base, '× W⁽¹⁾');
  o += card(156, 50, 108, 54, 'slate', [{ t: 'H = XW⁽¹⁾', mono: true, fs: 13.5, fw: 700 }, { t: '(n × h)', mono: true, fs: 12.5 }]);
  o += hArrow(268, 306, 77, D.red.base, '× W⁽²⁾');
  o += card(310, 50, 216, 54, 'red', [
    { t: 'O = X(W⁽¹⁾W⁽²⁾) = XW_{eff}', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    'Single affine map (rank ≤ min(d,h,q))',
  ]);

  o += hline(14, 526, 124);

  // Bottom panel: With ReLU phi (Nonlinearity blocks collapse)
  o += card(14, 138, 512, 30, 'green', ['(b) With Elementwise ReLU φ(z) = max(z, 0) : Bends & Folds Representation']);
  o += card(14, 176, 86, 68, 'blue', [{ t: 'X', fs: 16, fw: 700 }, { t: '(n × d)', mono: true, fs: 12.5 }]);
  o += arrow(104, 210, 132, 210, C.blue);
  o += card(136, 176, 114, 68, 'orange', [
    { t: 'Z = XW⁽¹⁾+b⁽¹⁾', mono: true, fs: 12.5, fw: 700 },
    'Pre-activations',
  ]);
  o += arrow(254, 210, 282, 210, C.purple);
  o += card(286, 176, 106, 68, 'purple', [
    { t: 'H = ReLU(Z)', mono: true, fs: 13, fw: 700 },
    'Blocks factoring!',
  ]);
  o += arrow(396, 210, 424, 210, C.green);
  o += card(428, 176, 98, 68, 'green', [
    { t: 'O = HW⁽²⁾+b⁽²⁾', mono: true, fs: 12.5, fw: 700 },
    'Separates XOR',
  ]);
  return svg(W, H, o);
}

// 8. ch04-mlp-xor-two-solutions
function xorTwoSolutions() {
  const W = 540, H = 266;
  let o = '';
  o += card(14, 12, 248, 86, 'blue', [
    'Construction A: Sum & Fold',
    { t: 'h₁ = ReLU(x₁ + x₂)', mono: true, fs: 12.5, fill: D.blue.dark },
    { t: 'h₂ = ReLU(x₁ + x₂ − 1)', mono: true, fs: 12.5, fill: D.blue.dark },
    { t: 'o = h₁ − 2h₂  (merges (0,1)&(1,0))', mono: true, fs: 12, fill: C.ink },
  ], { white: true, rail: true });

  o += card(278, 12, 248, 86, 'green', [
    'Construction B: Asymmetric Differences',
    { t: 'h₁ = ReLU(x₁ − x₂)', mono: true, fs: 12.5, fill: D.green.dark },
    { t: 'h₂ = ReLU(x₂ − x₁)', mono: true, fs: 12.5, fill: D.green.dark },
    { t: 'o = h₁ + h₂  (merges (0,0)&(1,1))', mono: true, fs: 12, fill: C.ink },
  ], { white: true, rail: true });

  // Middle table mapping all 4 corners under Construction B
  o += card(14, 108, 512, 82, 'purple', [
    'Construction B Hidden Mapping: (x₁, x₂) ↦ (h₁, h₂) ↦ o = h₁ + h₂',
    { t: '(0,0) ↦ (0,0) → 0   |   (1,0) ↦ (1,0) → 1   |   (0,1) ↦ (0,1) → 1   |   (1,1) ↦ (0,0) → 0', mono: true, fs: 12, fill: C.ink },
    'Both class-0 corners collapse to the origin (0, 0) in hidden space!',
  ], { white: true });

  // Bottom warning: Why h=1 fails
  o += card(14, 200, 512, 54, 'red', [
    'Why h = 1 ReLU Unit Cannot Solve XOR (h ≥ 2 is minimal):',
    'h₁ = ReLU(w₁x₁ + w₂x₂ + b) is monotonic along (w₁, w₂) → carves only 1 halfspace',
  ]);
  return svg(W, H, o);
}

// 9. ch04-mlp-param-scaling
function paramScaling() {
  const W = 540, H = 268;
  let o = '';
  o += card(14, 10, 512, 32, 'blue', ['Where Live the Parameters in a Fashion-MNIST MLP? (d=784, h=256, q=10)']);

  // Proportional bar for 1-hidden-layer MLP (203,530 total)
  o += card(14, 52, 512, 84, 'blue', [
    '1-Hidden-Layer MLP: 203,530 Total Parameters (203,264 MACs / image)',
    { t: 'Layer 1: W⁽¹⁾ (784×256) + b⁽¹⁾ (256) = 200,960 params (98.7% of total!)', mono: true, fs: 12.5, fill: D.blue.dark },
    { t: 'Layer 2: W⁽²⁾ (256×10)  + b⁽²⁾ (10)  =   2,570 params (1.3% of total)', mono: true, fs: 12.5, fill: D.green.dark },
  ], { white: true, rail: true });

  // 3 architecture comparison cards below
  const archs = [
    { x: 14,  theme: 'slate',  title: 'Softmax (K = 1)',      eq: '784×10 + 10',       tot: '7,850 params' },
    { x: 188, theme: 'blue',   title: '1-Hidden (h = 256)',   eq: '(d+1)h + (h+1)q',   tot: '203,530 (26×)' },
    { x: 362, theme: 'purple', title: '2-Hidden (256→256)',   eq: '+ (256×256 + 256)', tot: '269,322 (34×)' },
  ];
  archs.forEach((a) => {
    o += card(a.x, 148, 164, 106, a.theme, [
      a.title,
      { t: a.eq, mono: true, fs: 12.5, fill: C.ink },
      { t: a.tot, mono: true, fs: 13.5, fw: 700, fill: PAL[a.theme].dark },
    ]);
  });
  return svg(W, H, o);
}

// 10. ch04-mlp-uat-curse
function uatCurse() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 12, 512, 52, 'purple', [
    'Universal Approximation Theorem (Cybenko 1989; Leshno et al. 1993)',
    { t: 'Any continuous f on compact K ⊂ ℝᵈ can be approximated to ε > 0 by 1 hidden layer', fs: 13, fill: C.ink },
  ]);

  // Two limitations side by side
  o += card(14, 76, 248, 118, 'red', [
    'Limit 1: Exponential Width Curse',
    { t: 'Shallow (K = 2): h = Ω(ε⁻ᵈ) units', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    '1-layer hinges add segments one by one',
    '(h + 1 pieces); high-d oscillations',
    'require astronomical width h',
  ], { white: true, rail: true });

  o += card(278, 76, 248, 118, 'orange', [
    'Limit 2: Existence ≠ Learnability',
    { t: 'UAT proves ∃ weights W⁽¹⁾, W⁽²⁾', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    'Does NOT guarantee minibatch SGD',
    'finds them from random init or',
    'generalizes from n finite samples',
  ], { white: true, rail: true });

  o += card(14, 206, 512, 46, 'green', [
    'Why We Go Deep: Composing m layers multiplies pieces (2^m vs. 2m + 1)',
  ]);
  return svg(W, H, o);
}

// 11. ch04-mlp-rectifier-vs-squashing
function rectifierVsSquashing() {
  const W = 540, H = 272;
  let o = '';
  o += card(14, 10, 512, 32, 'slate', ['Activation Derivatives φ\'(z) Gate Every Backward Gradient']);

  const acts = [
    { y: 50,  theme: 'green',  name: 'ReLU max(z,0)',       zNeg: 'z=−4: 0.000 (dead)', z0: 'z=0: 0.00', zPos: 'z=+4: 1.000 (100%)' },
    { y: 94,  theme: 'teal',   name: 'Leaky ReLU (0.01z)',  zNeg: 'z=−4: 0.010 (alive)', z0: 'z=0: 0.01', zPos: 'z=+4: 1.000 (100%)' },
    { y: 138, theme: 'blue',   name: 'GELU z·Φ(z)',         zNeg: 'z=−4: −0.0005',       z0: 'z=0: 0.50', zPos: 'z=+4: 1.000 (100%)' },
    { y: 182, theme: 'orange', name: 'Sigmoid σ(z)',        zNeg: 'z=−4: 0.018 (sat!)',  z0: 'z=0: 0.25', zPos: 'z=+4: 0.018 (sat!)' },
    { y: 226, theme: 'red',    name: 'Tanh 2σ(2z)−1',       zNeg: 'z=−4: 0.0013 (sat!)', z0: 'z=0: 1.00', zPos: 'z=+4: 0.0013 (sat!)' },
  ];

  acts.forEach((a) => {
    o += card(14, a.y, 148, 36, a.theme, [{ t: a.name, fs: 13, fw: 700 }]);
    o += card(168, a.y, 124, 36, a.theme, [{ t: a.zNeg, mono: true, fs: 11.5, fill: C.ink }], { white: true });
    o += card(298, a.y, 94,  36, a.theme, [{ t: a.z0,   mono: true, fs: 12,   fill: C.ink }], { white: true });
    o += card(398, a.y, 128, 36, a.theme, [{ t: a.zPos, mono: true, fs: 11.5, fw: 700, fill: PAL[a.theme].dark }]);
  });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4.2 MLP-IMPLEMENTATION.MD DIAGRAMS (5 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 12. ch04-mlp-impl-goals
function implGoals() {
  const W = 540, H = 256;
  let o = '';
  o += card(14, 12, 240, 94, 'slate', [
    'Softmax Regression Baseline',
    { t: '784 → 10 (7,850 params)', mono: true, fs: 13 },
    'Global whole-garment templates',
    'Plateaus at ~84% val accuracy',
  ], { white: true, rail: true });

  o += hArrow(258, 282, 59, C.blue);

  o += card(286, 12, 240, 94, 'blue', [
    '1-Hidden-Layer ReLU MLP',
    { t: '784 → 256 → 10 (203,530 params)', mono: true, fs: 12.5, fill: D.blue.dark },
    '256 localized edge/part detectors',
    'Reaches ~87% (−20% error rate!)',
  ], { white: true, rail: true });

  // Bottom 4 engineering goals
  const goals = [
    { x: 14,  theme: 'blue',   t1: '1. Scratch & API', t2: 'Params vs Sequential' },
    { x: 144, theme: 'purple', t1: '2. Sanity Checks', t2: 'Shapes, log 10, 50%' },
    { x: 274, theme: 'green',  t1: '3. GPU Timing',    t2: '26× FLOPs ≈ 1× time' },
    { x: 404, theme: 'orange', t1: '4. Weight Filters', t2: 'Inspect W⁽¹⁾ columns' },
  ];
  goals.forEach((g) => {
    o += card(g.x, 126, 122, 114, g.theme, [
      { t: g.t1, fs: 13.5, fw: 700 },
      { t: g.t2, fs: 12.5, fill: C.ink },
    ]);
  });
  return svg(W, H, o);
}

// 13. ch04-mlp-symmetry-breaking
function symmetryBreaking() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 12, 248, 178, 'red', [
    'Constant / Equal Init (Broken)',
    { t: 'w₁ = w₂ = … = w₂₅₆ = c', mono: true, fs: 13, fill: D.red.dark },
    'All 256 hidden units compute',
    'identical activations h₁ = … = h₂₅₆',
    'and receive identical gradients:',
    { t: '256 units collapse into 1!', fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  o += card(278, 12, 248, 178, 'green', [
    'Small Random Init N(0, 0.01²)',
    { t: 'W⁽ᵏ⁾ ~ N(0, 0.01²),  b⁽ᵏ⁾ = 0', mono: true, fs: 13, fill: D.green.dark },
    '1. Breaks permutation symmetry',
    '2. Keeps initial logits O ≈ 0',
    '3. Uniform softmax p̂_j ≈ 1/10',
    { t: 'Initial loss ≈ log(10) ≈ 2.303', mono: true, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 202, 512, 50, 'orange', [
    'Pitfall: every weight tensor needs its own random draw',
    '(two layers that replay the same random stream start with identical weights)',
  ]);
  return svg(W, H, o);
}

// 14. ch04-mlp-forward-dataflow
function forwardDataflow() {
  const W = 540, H = 250;
  let o = '';
  o += card(14, 12, 512, 32, 'blue', ['Two-Line Forward Pass & Pre-Training Diagnostic Invariants (n = 8)']);

  // 4 stages left to right
  const boxes = [
    { x: 14,  theme: 'slate',  t1: 'Input X',       t2: '(8, 784)', t3: 'flattened pixels' },
    { x: 146, theme: 'blue',   t1: 'Z = XW⁽¹⁾+b⁽¹⁾', t2: '(8, 256)', t3: '~50% Z_ij > 0' },
    { x: 278, theme: 'purple', t1: 'H = Z*(Z > 0)', t2: '(8, 256)', t3: 'active ≈ 0.50' },
    { x: 410, theme: 'green',  t1: 'O = HW⁽²⁾+b⁽²⁾', t2: '(8, 10)',  t3: 'raw class logits' },
  ];
  boxes.forEach((b, i) => {
    o += card(b.x, 58, 116, 98, b.theme, [
      { t: b.t1, mono: true, fs: 13, fw: 700 },
      { t: b.t2, mono: true, fs: 13.5, fw: 700, fill: C.ink },
      { t: b.t3, fs: 12, fill: C.muted },
    ]);
    if (i < 3) o += arrow(b.x + 118, 107, boxes[i + 1].x - 2, 107, C.blue);
  });

  o += card(14, 172, 512, 64, 'green', [
    'Three Millisecond Pre-Training Sanity Checks on Untrained Network:',
    { t: '✓ H.shape == (8, 256) & O.shape == (8, 10)\n✓ loss ≈ log 10 ≈ 2.303   ✓ (H > 0).mean ≈ 0.50', mono: true, fs: 12, fill: D.green.dark },
  ], { white: true, rail: true });
  return svg(W, H, o);
}

// 15. ch04-mlp-framework-inits
function frameworkInits() {
  const W = 540, H = 266;
  let o = '';
  o += card(14, 10, 512, 32, 'orange', ['Framework Default Linear Initializers Differ—Override for Fair Benchmarks!']);

  o += card(14, 54, 238, 124, 'slate', [
    'Built-In Default (Dense Layer)',
    'Fan-in-scaled uniform or normal draw,',
    'chosen differently by each framework',
    { t: 'σ ≈ 0.02–0.04 at n_in = 784', mono: true, fs: 12.5, fw: 700, fill: C.ink },
  ], { white: true, rail: true });
  o += arrow(256, 116, 284, 116, C.green);
  o += card(288, 54, 238, 124, 'green', [
    'Explicit Override',
    { t: 'W ~ N(0, 0.01²),  b = 0', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    'Same draw as MLPScratch',
    'in every layer',
  ], { white: true, rail: true });

  o += card(14, 190, 512, 34, 'green', [
    { t: 'Identical starting scale ⇒ MLP matches MLPScratch trajectory-for-trajectory', fs: 13, fw: 700, fill: D.green.dark },
  ]);
  o += fitTx(270, 246, 'He initialization σ = √(2/n_in) later replaces σ = 0.01 for deep ReLU stacks.', 512, { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 16. ch04-mlp-impl-summary
function implSummary() {
  const W = 540, H = 266;
  let o = '';
  const items = [
    { x: 14,  y: 14,  theme: 'blue',   t1: '1. Parameter Budget',  t2: '203,530 floats (0.81 MB)', t3: '98.6% in W⁽¹⁾ (784×256)' },
    { x: 276, y: 14,  theme: 'purple', t1: '2. Raw Logits Forward', t2: 'H = relu(XW⁽¹⁾+b⁽¹⁾)',   t3: 'Return O (no softmax!)' },
    { x: 14,  y: 98,  theme: 'green',  t1: '3. Pre-Train Checks',  t2: 'loss ≈ 2.303, 50% active', t3: 'Catches bugs in < 1 ms' },
    { x: 276, y: 98,  theme: 'orange', t1: '4. Explicit Init',     t2: 'Override framework defaults', t3: 'Avoid epoch-1 confounding' },
    { x: 14,  y: 182, theme: 'teal',   t1: '5. GPU Throughput',    t2: '26× FLOPs ≈ 1× wall time', t3: 'Data & launch dominate' },
    { x: 276, y: 182, theme: 'red',    t1: '6. Learned W⁽¹⁾ Lobes', t2: 'Collar & sleeve detectors', t3: '87% acc (−20% errors)' },
  ];
  items.forEach((it) => {
    o += card(it.x, it.y, 250, 72, it.theme, [
      it.t1,
      { t: it.t2, mono: true, fs: 12.5, fill: PAL[it.theme].dark },
      { t: it.t3, fs: 12.5, fill: C.ink },
    ], { white: true, rail: true });
  });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4.3 BACKPROP.MD DIAGRAMS (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 17. ch04-mlp-backprop-efficiency
function backpropEfficiency() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 12, 248, 114, 'red', [
    'Finite Differences / Forward Mode',
    { t: '(J(w+ε) − J(w−ε)) / 2ε', mono: true, fs: 12.5, fill: D.red.dark },
    'Perturbs 1 scalar weight at a time:',
    { t: '203,530 forward sweeps / step!', fw: 700, fill: D.red.dark },
    '(~4 × 10¹⁰ MACs per example)',
  ], { white: true, rail: true });

  o += card(278, 12, 248, 114, 'green', [
    'Reverse-Mode Backpropagation',
    { t: 'Seed ∂J/∂J = 1 at scalar loss', mono: true, fs: 12.5, fill: D.green.dark },
    'Reuses shared upstream ∂J/∂Z⁽ᵏ⁾:',
    { t: '1 backward sweep → ALL 203,530!', fw: 700, fill: D.green.dark },
    '(fwd + bwd ≈ 4 × 10⁵ MACs per example)',
  ], { white: true, rail: true });

  o += card(14, 140, 512, 110, 'blue', [
    'The Fundamental Time–Memory Trade-Off of Training:',
    { t: 'Forward Pass: X ──► H⁽¹⁾ ──► … ──► H⁽ᴷ⁻¹⁾ ──► O ──► J   (caches H⁽ᵏ⁾ in GPU RAM)', mono: true, fs: 12, fill: D.blue.dark },
    { t: 'Backward Pass: ∂J/∂W⁽ᵏ⁾ = (H⁽ᵏ⁻¹⁾)ᵀ (∂J/∂Z⁽ᵏ⁾)   (reads cached H⁽ᵏ⁻¹⁾; O(n·K·h) RAM)', mono: true, fs: 12, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

// 18. ch04-mlp-local-matmul-rule
function localMatmulRule() {
  const W = 540, H = 266;
  let o = '';
  o += card(14, 12, 512, 46, 'blue', [
    'Forward Matrix Multiply:  A = X W',
    { t: 'X : (n × d)    ·    W : (d × h)    =    A : (n × h)   [caches X and W]', mono: true, fs: 13, fill: C.ink },
  ]);

  o += vArrow(138, 62, 92, D.red.base, 'upstream G = ∂J/∂A : (n × h)');
  o += vArrow(402, 62, 92, D.red.base);

  o += card(14, 96, 248, 108, 'red', [
    '1. Parameter Gradient ∂J/∂W',
    { t: '∂J/∂W = Xᵀ G', mono: true, fs: 16, fw: 700, fill: D.red.dark },
    { t: '(d × n) · (n × h) = (d × h)', mono: true, fs: 12.5, fill: C.ink },
    'Contracts across batch axis n;',
    'requires cached forward input X!',
  ], { white: true, rail: true });

  o += card(278, 96, 248, 108, 'purple', [
    '2. Input Gradient ∂J/∂X',
    { t: '∂J/∂X = G Wᵀ', mono: true, fs: 16, fw: 700, fill: D.purple.dark },
    { t: '(n × h) · (h × d) = (n × d)', mono: true, fs: 12.5, fill: C.ink },
    'Contracts across output width h;',
    'propagates error to layer below',
  ], { white: true, rail: true });

  o += card(14, 214, 512, 40, 'green', [
    'Shape-Lock Mnemonic: XᵀG is the ONLY way (n×d) and (n×h) multiply to (d×h)!',
  ]);
  return svg(W, H, o);
}

// 19. ch04-mlp-local-bias-act-rules
function localBiasActRules() {
  const W = 540, H = 264;
  let o = '';
  // Left: Broadcasted bias rule
  o += card(14, 12, 248, 240, 'orange', [
    'Rule 2: Broadcast Bias Z = A + b',
    { t: 'b : (1 × h) added to all n rows', mono: true, fs: 12.5, fill: C.ink },
    { t: '∂J/∂A = G   (copied unchanged)', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: '∂J/∂b = Gᵀ 1_{n} = G.sum(axis=0)', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    'Because b_j branches into all n',
    'examples, backward sums over n',
    '(zero forward cache needed!)',
  ], { white: true, rail: true });

  // Right: Elementwise activation rule
  o += card(278, 12, 248, 240, 'purple', [
    'Rule 3: Activation H = φ(Z)',
    { t: 'h_ij = φ(z_ij) acts entrywise', mono: true, fs: 12.5, fill: C.ink },
    { t: '∂J/∂Z = (∂J/∂H) ⊙ φ\'(Z)', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'ReLU: G_Z = G_H * (Z > 0)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    'Diagonal Jacobian gates each',
    'entry by local slope φ\'(z_ij):',
    '100% if z_ij > 0,  0 if z_ij ≤ 0',
  ], { white: true, rail: true });
  return svg(W, H, o);
}

// 20. ch04-mlp-five-line-sweep
function fiveLineSweep() {
  const W = 540, H = 278;
  let o = '';
  const steps = [
    { y: 10,  theme: 'red',    step: 'Line 1 (Output Logits)', eq: 'G_O = ∂J/∂O = (1/n)(Ŷ − Y)',                          shape: '(n × q)' },
    { y: 62,  theme: 'orange', step: 'Line 2 (Layer 2 Params)', eq: '∂J/∂W⁽²⁾ = HᵀG_O + λW⁽²⁾,   ∂J/∂b⁽²⁾ = G_Oᵀ 1_{n}',    shape: '(h × q), (q,)' },
    { y: 114, theme: 'purple', step: 'Line 3 (Hidden Acts)',    eq: '∂J/∂H = G_O (W⁽²⁾)ᵀ   [credit assignment to units]', shape: '(n × h)' },
    { y: 166, theme: 'blue',   step: 'Line 4 (ReLU Gate)',      eq: 'G_Z = ∂J/∂Z = (∂J/∂H) ⊙ φ\'(Z)',                      shape: '(n × h)' },
    { y: 218, theme: 'green',  step: 'Line 5 (Layer 1 Params)', eq: '∂J/∂W⁽¹⁾ = XᵀG_Z + λW⁽¹⁾,   ∂J/∂b⁽¹⁾ = G_Zᵀ 1_{n}',    shape: '(d × h), (h,)' },
  ];

  steps.forEach((s, i) => {
    o += card(14, s.y, 152, 42, s.theme, [{ t: s.step, fs: 12.5, fw: 700 }]);
    o += card(172, s.y, 260, 42, s.theme, [{ t: s.eq, mono: true, fs: 12, fw: 700, fill: C.ink }], { white: true });
    o += card(438, s.y, 88,  42, 'slate', [{ t: s.shape, mono: true, fs: 12, fw: 600 }]);
    if (i < 4) o += arrow(302, s.y + 43, 302, steps[i + 1].y - 1, D.red.base);
  });
  return svg(W, H, o);
}

// 21. ch04-mlp-k-layer-recurrence
function kLayerRecurrence() {
  const W = 540, H = 252;
  let o = '';
  o += card(14, 12, 512, 34, 'purple', ['Deep K-Layer Backward Module (for k = K−1 down to 1)']);

  // Upstream G_{Z^(k+1)} on right -> multiply by W^(k+1)^T ->odot phi'(Z^(k)) -> G_{Z^(k)} on left
  o += card(384, 64, 142, 64, 'red', [
    { t: 'G_Z⁽ᵏ⁺¹⁾', mono: true, fs: 15, fw: 700 },
    'Upstream layer k+1',
  ]);
  o += arrow(380, 96, 348, 96, D.red.base);

  o += card(206, 64, 138, 64, 'purple', [
    { t: '× (W⁽ᵏ⁺¹⁾)ᵀ ⊙ φ\'(Z⁽ᵏ⁾)', mono: true, fs: 12, fw: 700 },
    'Project & gate slope',
  ]);
  o += arrow(202, 96, 170, 96, D.red.base);

  o += card(14, 64, 152, 64, 'red', [
    { t: 'G_Z⁽ᵏ⁾ = ∂J/∂Z⁽ᵏ⁾', mono: true, fs: 13.5, fw: 700 },
    'Layer k sensitivity',
  ]);

  // Branch down from G_Z^(k) to parameter gradients
  o += vArrow(90, 132, 166, C.green, 'branch to params');

  o += card(14, 170, 512, 68, 'green', [
    'Read Off Layer-k Parameter Gradients Using Cached Forward H⁽ᵏ⁻¹⁾:',
    { t: '∂J/∂W⁽ᵏ⁾ = (H⁽ᵏ⁻¹⁾)ᵀ G_Z⁽ᵏ⁾ + λW⁽ᵏ⁾          ∂J/∂b⁽ᵏ⁾ = (G_Z⁽ᵏ⁾)ᵀ 1_{n}', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });
  return svg(W, H, o);
}

// 22. ch04-mlp-worked-numbers
function workedNumbers() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 10, 512, 32, 'blue', ['Worked 2→2→1 Hand Trace: X = [1, 2],  y = 0,  L = ½(o − y)²']);

  // Forward row
  o += card(14, 52, 158, 78, 'blue', [
    '1. Forward Pre-Act',
    { t: 'Z = X W⁽¹⁾ = [−1, 2]', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    'Unit 1 negative (z₁=−1)',
  ], { white: true });
  o += arrow(176, 91, 194, 91, C.blue);
  o += card(198, 52, 154, 78, 'purple', [
    '2. ReLU Gate',
    { t: 'H = ReLU(Z) = [0, 2]', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    { t: 'mask φ\'(Z) = [0, 1]', mono: true, fs: 12 },
  ], { white: true });
  o += arrow(356, 91, 374, 91, C.blue);
  o += card(378, 52, 148, 78, 'green', [
    '3. Output & Loss',
    { t: 'o = HW⁽²⁾ = −2', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'L = 2, ∂L/∂o = −2', mono: true, fs: 12.5, fill: D.red.dark },
  ], { white: true });

  // Backward row
  o += card(378, 150, 148, 98, 'orange', [
    '4. Output Grads',
    { t: '∂L/∂W⁽²⁾ = Hᵀ(−2)', mono: true, fs: 12 },
    { t: '= [0, −4]ᵀ', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
  ], { white: true });
  o += arrow(374, 199, 356, 199, D.red.base);
  o += card(198, 150, 154, 98, 'red', [
    '5. Dead ReLU Gate!',
    { t: '∂L/∂H = [−4, 2]', mono: true, fs: 12.5 },
    { t: '∂L/∂Z = [−4,2]⊙[0,1]', mono: true, fs: 12 },
    { t: '= [0, 2]  (z₁ killed!)', mono: true, fs: 13, fw: 700, fill: D.red.dark },
  ], { white: true });
  o += arrow(194, 199, 176, 199, D.red.base);
  o += card(14, 150, 158, 98, 'red', [
    '6. Layer-1 Weight Grad',
    { t: '∂L/∂W⁽¹⁾ = Xᵀ(∂L/∂Z)', mono: true, fs: 12 },
    { t: 'Col 1: [0,0]ᵀ frozen', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Col 2: [2,4]ᵀ learns', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ], { white: true });
  return svg(W, H, o);
}

// 23. ch04-mlp-activation-memory
function activationMemory() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 12, 248, 96, 'green', [
    'Inference Forward Pass',
    'Buffer ping-pong: overwrites H⁽ᵏ⁻¹⁾',
    'as soon as H⁽ᵏ⁾ is computed',
    { t: 'Peak RAM: O(n · h)  (no depth K!)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(278, 12, 248, 96, 'red', [
    'Training Forward + Backward',
    'Must retain X and every H⁽ᵏ⁾',
    'until backward computes (H⁽ᵏ⁻¹⁾)ᵀG',
    { t: 'Peak RAM: O(n · K · h)  (× depth K!)', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  // Memory bar comparison at n = 256
  o += card(14, 120, 512, 64, 'blue', [
    '1-Hidden-Layer MLP Cached Activations at Batch Size n = 256 (1.08 MB Total):',
    { t: 'X (256×784) = 0.80 MB   +   H (256×256) = 0.26 MB   +   Ŷ (256×10) = 0.01 MB', mono: true, fs: 12.5, fill: D.blue.dark },
  ]);

  o += card(14, 194, 512, 56, 'orange', [
    'Static Model Parameters (W⁽¹⁾, b⁽¹⁾, W⁽²⁾, b⁽²⁾): 203,530 floats = 0.81 MB',
    'Even with K = 1 hidden layer, cached batch activations (1.08 MB) > parameters (0.81 MB)!',
  ]);
  return svg(W, H, o);
}

// 24. ch04-mlp-flop-ratio
function flopRatio() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 512, 34, 'purple', ['Why a Training Step Costs ~3× Forward FLOPs (Interior Layers)']);

  // 1x Forward vs 2x Backward
  o += card(14, 58, 248, 124, 'blue', [
    'Forward Pass: 1× Matmul / Layer',
    { t: 'Z⁽ᵏ⁾ = H⁽ᵏ⁻¹⁾ W⁽ᵏ⁾', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    'Layer 1 (784×256): 200,704 MACs',
    'Layer 2 (256×10):    2,560 MACs',
    { t: 'Total Forward = 203,264 MACs/img', mono: true, fs: 12, fw: 700 },
  ], { white: true, rail: true });

  o += card(278, 58, 248, 124, 'red', [
    'Backward Pass: 2× Matmuls / Layer',
    { t: '1. Weight grad: (H⁽ᵏ⁻¹⁾)ᵀ G_Z⁽ᵏ⁾', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: '2. Input grad:  G_Z⁽ᵏ⁾ (W⁽ᵏ⁾)ᵀ', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    'Each has the EXACT same n·d_in·d_out',
    'FLOP count as the forward matmul!',
  ], { white: true, rail: true });

  o += card(14, 194, 512, 54, 'green', [
    'Rule of Thumb for Deep Networks (K Interior Layers):',
    { t: 'FLOPs(backward) ≈ 2 × FLOPs(forward)   ⇒   FLOPs(train step) ≈ 3 × FLOPs(inference)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4.4 NUMERICAL-STABILITY-AND-INIT.MD DIAGRAMS (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 25. ch04-mlp-depth-regimes
function depthRegimes() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 10, 512, 34, 'slate', ['10-Layer ReLU MLP (Width h = 256): Signal & Gradient Scaling Across Depth']);

  const regimes = [
    {
      x: 14, theme: 'red', title: 'Too Small: τ = 0.01',
      lines: [
        { t: 'Multiplier: 0.0128× / layer', mono: true, fs: 12, fill: D.red.dark },
        { t: '9 hops → 10⁻¹⁷ (vanished!)', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
        'Early layers get 0 gradient;',
        'loss frozen at log 10 (10% acc)',
      ],
    },
    {
      x: 188, theme: 'orange', title: 'Too Large: τ = 1.0',
      lines: [
        { t: 'Multiplier: 128× / layer', mono: true, fs: 12, fill: D.orange.dark },
        { t: '9 hops → 10¹⁹ (exploded!)', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark },
        'Deeper stacks overflow float32',
        '(>3.4×10³⁸) to inf / NaN',
      ],
    },
    {
      x: 362, theme: 'green', title: 'Balanced: He τ = √(2/h)',
      lines: [
        { t: 'Multiplier: 1.00× / layer', mono: true, fs: 12, fill: D.green.dark },
        { t: '9 hops → O(1) preserved!', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
        'Healthy activations & grads',
        'across all 10 layers (~85% acc)',
      ],
    },
  ];

  regimes.forEach((r) => {
    o += card(r.x, 54, 164, 34, r.theme, [r.title]);
    o += card(r.x, 94, 164, 156, r.theme, r.lines, { white: true });
  });
  return svg(W, H, o);
}

// 26. ch04-mlp-product-chain
function productChain() {
  const W = 540, H = 256;
  let o = '';
  o += card(14, 12, 512, 34, 'purple', ['Unrolling the Backward Chain Across t = K − 1 − k Layers']);

  // Chain of 4 boxes from right (Layer K-1) to left (Layer k)
  const nodes = [
    { x: 426, w: 100, theme: 'green',  t1: '∂L/∂H⁽ᴷ⁻¹⁾', t2: 'top hidden grad' },
    { x: 276, w: 136, theme: 'orange', t1: '⊙φ\'(Z⁽ᴷ⁻¹⁾)W⁽ᴷ⁻¹⁾ᵀ', t2: 'layer K−1 factor' },
    { x: 127, w: 136, theme: 'orange', t1: '⊙φ\'(Z⁽ᵏ⁺¹⁾)W⁽ᵏ⁺¹⁾ᵀ', t2: '⋯ t matrix factors' },
    { x: 14,  w: 100, theme: 'red',    t1: '∂L/∂H⁽ᵏ⁾', t2: 'early layer grad' },
  ];
  nodes.forEach((n) => {
    o += card(n.x, 58, n.w, 68, n.theme, [
      { t: n.t1, mono: true, fs: 12, fw: 700 },
      { t: n.t2, fs: 12, fill: C.muted },
    ]);
  });
  o += arrow(424, 92, 414, 92, D.red.base);
  o += arrow(274, 92, 265, 92, D.red.base);
  o += arrow(125, 92, 116, 92, D.red.base);

  // Exponential drift comparison at depth t = 50
  o += card(14, 142, 164, 98, 'red', [
    'Sub-Unity Drift (0.9×)',
    { t: '0.9⁵⁰ ≈ 0.005', mono: true, fs: 15, fw: 700, fill: D.red.dark },
    'Vanishes exponentially',
  ], { white: true, rail: true });
  o += card(188, 142, 164, 98, 'green', [
    'Unit Gain (1.0×)',
    { t: '1.0⁵⁰ = 1.000', mono: true, fs: 15, fw: 700, fill: D.green.dark },
    'Depth-invariant scale!',
  ], { white: true, rail: true });
  o += card(362, 142, 164, 98, 'orange', [
    'Super-Unity Drift (1.1×)',
    { t: '1.1⁵⁰ ≈ 117.4', mono: true, fs: 15, fw: 700, fill: D.orange.dark },
    'Explodes exponentially',
  ], { white: true, rail: true });
  return svg(W, H, o);
}

// 27. ch04-mlp-variance-multiplier
function varianceMultiplier() {
  const W = 540, H = 266;
  let o = '';
  o += card(14, 12, 512, 54, 'blue', [
    'Analytical Per-Layer Variance Multiplier (Width h, Weight Variance τ²)',
    { t: 'E[g_i²]  =  ( h  ·  τ²  ·  E[φ\'(z)²] )  ·  E[(g\'_j)²]', mono: true, fs: 14.5, fw: 700, fill: D.blue.dark },
  ]);

  const rows = [
    { y: 78,  theme: 'red',    name: 'ReLU, τ = 0.01',     ephi: 'E[φ\'²] = 1/2', mult: '256 · 10⁻⁴ · ½ = 0.0128',   hops: '0.0128⁹ ≈ 9.2×10⁻¹⁸' },
    { y: 126, theme: 'orange', name: 'ReLU, τ = 1.00',     ephi: 'E[φ\'²] = 1/2', mult: '256 · 1 · ½ = 128.0',       hops: '128⁹ ≈ 9.2×10¹⁸' },
    { y: 174, theme: 'purple', name: 'Sigmoid, τ = 1/16',  ephi: 'E[σ\'²] ≤ 1/16', mult: '256 · (1/256) · 1/16 ≤ 1/16', hops: '16⁻⁹ ≈ 1.5×10⁻¹¹' },
    { y: 222, theme: 'green',  name: 'ReLU He, τ = √(2/h)', ephi: 'E[φ\'²] = 1/2', mult: '256 · (2/256) · ½ = 1.000', hops: '1.0⁹ = 1.00 (flat!)' },
  ];

  rows.forEach((r) => {
    o += card(14,  r.y, 134, 38, r.theme, [{ t: r.name, fs: 12.5, fw: 700 }]);
    o += card(152, r.y, 96,  38, r.theme, [{ t: r.ephi, mono: true, fs: 11.5, fill: C.ink }], { white: true });
    o += card(252, r.y, 144, 38, r.theme, [{ t: r.mult, mono: true, fs: 11.5, fill: C.ink }], { white: true });
    o += card(400, r.y, 126, 38, r.theme, [{ t: r.hops, mono: true, fs: 11.5, fw: 700, fill: PAL[r.theme].dark }]);
  });
  return svg(W, H, o);
}

// 28. ch04-mlp-sigmoid-dilemma
function sigmoidDilemma() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 512, 34, 'red', ['Why Weight Scaling Alone Cannot Rescue Deep Sigmoid Networks']);

  o += card(14, 56, 248, 126, 'orange', [
    'Option A: Keep Pre-Acts Near z ≈ 0',
    { t: 'Set h·τ² = 1  (Xavier scale)', mono: true, fs: 13, fill: D.orange.dark },
    'Max slope is σ\'(0) = 1/4 ⇒ σ\'(0)² = 1/16',
    { t: 'Gradient variance shrinks ≥ 16× / layer!', fw: 700, fill: D.red.dark },
    '(10 layers → 16⁻⁹ ≈ 1.5 × 10⁻¹¹)',
  ], { white: true, rail: true });

  o += card(278, 56, 248, 126, 'red', [
    'Option B: Inflate Weights (h·τ² ≥ 16)',
    { t: 'Try to cancel the 1/16 slope factor', mono: true, fs: 12.5, fill: D.red.dark },
    'Pushes pre-activations into |z| ≫ 3',
    'saturated tails where σ\'(±4) ≈ 0.018:',
    { t: 'Gradients vanish even faster!', fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  o += card(14, 194, 512, 52, 'green', [
    'Resolution: Replace squashing hidden activations with non-saturating rectifiers',
    '(ReLU / GELU) or zero-centered tanh(z) with unit slope tanh\'(0) = 1 at the origin',
  ]);
  return svg(W, H, o);
}

// 29. ch04-mlp-xavier-balance
function xavierBalance() {
  const W = 540, H = 260;
  let o = '';
  o += card(14, 12, 252, 96, 'blue', [
    'Forward Variance Condition',
    { t: 'o_j = Σ_{i=1}^{n_in} x_i w_ij', mono: true, fs: 12.5 },
    { t: 'Var[o_j] = (n_in · τ²) Var[x_i]', mono: true, fs: 12.5, fill: D.blue.dark },
    { t: 'Requires:  n_in · τ² = 1', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ], { white: true, rail: true });

  o += card(274, 12, 252, 96, 'red', [
    'Backward Variance Condition',
    { t: 'g_i = Σ_{j=1}^{n_out} w_ij g\'_j', mono: true, fs: 12.5 },
    { t: 'Var[g_i] = (n_out · τ²) Var[g\'_j]', mono: true, fs: 12.5, fill: D.red.dark },
    { t: 'Requires:  n_out · τ² = 1', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  o += arrow(130, 112, 210, 144, C.blue);
  o += arrow(410, 112, 330, 144, D.red.base);

  o += card(14, 148, 512, 98, 'purple', [
    'Xavier / Glorot Compromise (Glorot & Bengio, 2010): Average Fan-In and Fan-Out',
    { t: '½(n_in + n_out) τ² = 1   ⇒   τ² = 2 / (n_in + n_out)', mono: true, fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'Gaussian: N(0, 2/(n_in+n_out))\nUniform: U(−√(6/(n_in+n_out)), +√(6/(n_in+n_out)))', mono: true, fs: 12, fill: C.ink },
  ]);
  return svg(W, H, o);
}

// 30. ch04-mlp-he-relu-halving
function heReluHalving() {
  const W = 540, H = 258;
  let o = '';
  // 3-step pipeline: Pre-activation z -> ReLU clips negative half -> He doubles variance
  o += card(14, 14, 156, 134, 'blue', [
    '1. Pre-Activation z',
    { t: 'z = Σ x_i w_ij', mono: true, fs: 13 },
    'Symmetric zero-mean',
    { t: 'E[z²] =\nn_in · τ² · v⁽ᵏ⁻¹⁾', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
  ], { white: true, rail: true });

  o += arrow(174, 81, 190, 81, C.purple);

  o += card(194, 14, 156, 134, 'orange', [
    '2. ReLU Half-Line',
    { t: 'h = max(z, 0)', mono: true, fs: 13 },
    'Zeroes out z < 0 half!',
    { t: 'E[h²] = ½ E[z²]', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
  ], { white: true, rail: true });

  o += arrow(354, 81, 370, 81, C.green);

  o += card(374, 14, 152, 134, 'green', [
    '3. He / Kaiming Rule',
    { t: '½ · n_in · τ² = 1', mono: true, fs: 13 },
    'Doubles Xavier scale:',
    { t: 'τ² = 2 / n_in', mono: true, fs: 14.5, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 164, 512, 82, 'purple', [
    'Why Xavier Decays by 2⁻¹⁰ ≈ 10⁻³ Across 10 ReLU Layers While He Stays at 1.0:',
    { t: 'Xavier (τ² = 1/h):  v⁽ᵏ⁾ = ½ · h · (1/h) v⁽ᵏ⁻¹⁾ = ½ v⁽ᵏ⁻¹⁾\n(loses 50% energy per layer)', mono: true, fs: 12, fill: D.red.dark },
    { t: 'He (τ² = 2/h):  v⁽ᵏ⁾ = ½ · h · (2/h) v⁽ᵏ⁻¹⁾ = 1 · v⁽ᵏ⁻¹⁾\n(preserves energy at any depth)', mono: true, fs: 12, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

// 31. ch04-mlp-init-limits
function initLimits() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 12, 512, 34, 'orange', ['Three Boundaries of Step-0 Variance-Preserving Initialization']);

  const cards = [
    {
      x: 14, theme: 'blue', title: '1. Post-Step-1 Drift',
      lines: ['Var[o] = n_in·τ²·γ²', 'assumes W ⊥ X at step 0.', 'After SGD steps, weights', 'correlate with inputs &', 'gains can drift from 1.'],
    },
    {
      x: 188, theme: 'purple', title: '2. ReLU Scale Symmetry',
      lines: ['ReLU(αz) = α ReLU(z)', 'for any α > 0:', 'scaling W⁽¹⁾ by α and', 'W⁽²⁾ by 1/α keeps f(x)', 'but skews layer grads!'],
    },
    {
      x: 362, theme: 'green', title: '3. Depth K ≥ 50 Stack',
      lines: ['Very deep networks pair', 'He / Xavier init with:', '• BatchNorm / LayerNorm', '• Residual skips x + f(x)', '(in later chapters)'],
    },
  ];
  cards.forEach((c) => {
    o += card(c.x, 56, 164, 34, c.theme, [c.title]);
    o += card(c.x, 96, 164, 154, c.theme, c.lines, { white: true });
  });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4.5 GENERALIZATION-DEEP.MD DIAGRAMS (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 32. ch04-mlp-overparam-paradox
function overparamParadox() {
  const W = 540, H = 260;
  let o = '';
  o += card(14, 12, 512, 34, 'purple', ['The Overparameterization Paradox: p ≥ n Interpolators']);

  o += card(14, 56, 164, 134, 'red', [
    'Classical Polynomial',
    { t: 'Degree 19 on n = 20', mono: true, fs: 12.5 },
    { t: 'p / n = 1.0 (threshold!)', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    'Unique interpolator;',
    'wild oscillations:',
    { t: 'Test MSE ≥ 10⁵ (cubic: 0.01)', fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  o += card(188, 56, 164, 134, 'blue', [
    'MLP on Full Fashion',
    { t: 'p = 203.5K, n = 60K', mono: true, fs: 12.5 },
    { t: 'p / n = 3.4 params/img', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    'Train acc: 87–90%',
    'Test acc:  ~87%',
    { t: '1–3 point gap', fw: 700, fill: D.blue.dark },
  ], { white: true, rail: true });

  o += card(362, 56, 164, 134, 'green', [
    'MLP on 1K Subset',
    { t: 'p = 203.5K, n = 1,000', mono: true, fs: 12.5 },
    { t: 'p / n = 203.5 params/img', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    'Interpolates 100% train,',
    'yet no catastrophic blowup:',
    { t: 'Test acc ≈ 80%!', fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 202, 512, 46, 'slate', [
    'Key Question: Why does p/n = 203.5 generalize when p/n = 1.0 explodes? → Conditioning & Implicit Bias',
  ]);
  return svg(W, H, o);
}

// 33. ch04-mlp-vacuous-vc-bound
function vacuousVcBound() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 512, 48, 'purple', [
    'Distribution-Free Uniform Convergence Bound (VC Dimension / Rademacher)',
    { t: 'ε(f) ≤ ε_S(f) + B(ℱ, n, δ)   [B depends only on ℱ and n]', mono: true, fs: 12.5, fill: C.ink },
  ]);

  o += card(14, 72, 248, 114, 'red', [
    '1. Test on Shuffled Random Labels',
    { t: 'Same MLP class ℱ, same n = 1,000', mono: true, fs: 12 },
    'Hits 100% train acc: ε_S(f_rand) = 0.00',
    'True test error is chance: ε(f_rand) = 0.90',
    { t: '⇒ Forces B(ℱ, n, δ) ≥ 0.90!', mono: true, fs: 13, fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  o += hArrow(264, 276, 129, D.red.base);

  o += card(278, 72, 248, 114, 'orange', [
    '2. Apply Bound to True Labels',
    { t: 'Because B(ℱ, n, δ) is label-blind:', mono: true, fs: 12 },
    'ε(f_true) ≤ 0.00 + B(ℱ, n, δ) ≥ 0.90',
    'Cannot guarantee better than 90% error',
    { t: 'even though real test error is < 20%!', fw: 700, fill: D.orange.dark },
  ], { white: true, rail: true });

  o += card(14, 198, 512, 52, 'green', [
    'Takeaway: Overparameterized generalization cannot be explained by raw capacity ceilings |ℱ|;',
    'it depends on which specific interpolator SGD selects on structured vs. noisy data.',
  ]);
  return svg(W, H, o);
}

// 34. ch04-mlp-inductive-bias-map
function inductiveBiasMap() {
  const W = 540, H = 266;
  let o = '';
  o += card(14, 10, 512, 34, 'blue', ['No Free Lunch Theorem (Wolpert 1996): Generalization Requires Inductive Bias']);

  const items = [
    { x: 14,  y: 54,  theme: 'slate',  t1: 'Linear / Softmax',      t2: 'Interpolates only if d ≥ n', t3: 'Bias: flat linear halfspaces' },
    { x: 276, y: 54,  theme: 'blue',   t1: 'Ridge / Weight Decay',  t2: 'Does not interpolate (λ > 0)', t3: 'Bias: small ‖w‖₂ smooth fits' },
    { x: 14,  y: 144, theme: 'teal',   t1: '1-Nearest Neighbor',    t2: '100% train accuracy always!', t3: 'Bias: Voronoi cells (asympt. ≤ 2ε*)' },
    { x: 276, y: 144, theme: 'green',  t1: 'Overparam MLP + SGD',   t2: '100% train accuracy (p ≫ n)', t3: 'Bias: min-norm / max-margin path' },
  ];
  items.forEach((it) => {
    o += card(it.x, it.y, 250, 78, it.theme, [
      it.t1,
      { t: it.t2, mono: true, fs: 12, fill: PAL[it.theme].dark },
      { t: it.t3, fs: 12.5, fill: C.ink },
    ], { white: true, rail: true });
  });

  o += card(14, 232, 512, 24, 'purple', [
    { t: '1-NN proves 100% training accuracy (interpolation) is compatible with strong test generalization!', fs: 12.5, fw: 700 },
  ]);
  return svg(W, H, o);
}

// 35. ch04-mlp-gen-summary
function genSummary() {
  const W = 540, H = 268;
  let o = '';
  const rows = [
    { y: 12,  theme: 'red',    t1: '1. Random-Label Memorization', t2: '100% train acc on noise ⇒ class-only VC bounds B(ℱ,n) ≥ 0.9 are vacuous' },
    { y: 62,  theme: 'orange', t1: '2. Double Descent at h = n',   t2: 'Square H has s_min ≈ 10⁻⁵–10⁻³ (explodes ‖W‖); h ≫ n heals s_min & shrinks norm' },
    { y: 112, theme: 'blue',   t1: '3. Min-Norm Implicit Bias',    t2: 'Squared-loss GD from w₀ = 0 stays in row(H) ⇒ converges to min-‖w‖₂ fit' },
    { y: 162, theme: 'purple', t1: '4. Max-Margin Logistic Bias',  t2: 'Cross-entropy GD sends ‖w_t‖ → ∞ while w_t/‖w_t‖ → hard-margin SVM' },
    { y: 212, theme: 'green',  t1: '5. Clean-First & Early Stop',  t2: 'SGD fits coherent clean structure in ~20 epochs before memorizing noise' },
  ];
  rows.forEach((r) => {
    o += card(14, r.y, 512, 42, r.theme, [
      { t: r.t1, fs: 13.5, fw: 700 },
      { t: r.t2, fs: 12, fill: C.ink },
    ], { white: true, rail: true });
  });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4.6 DROPOUT.MD DIAGRAMS (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 36. ch04-mlp-coadaptation
function coadaptation() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 248, 178, 'red', [
    'Without Dropout (p = 0)',
    'All hidden units always present:',
    { t: 'h₁ + h₂ − h₃ cancels errors', mono: true, fs: 12.5, fill: D.red.dark },
    'Multi-unit conspiracies memorize',
    'isolated noisy training labels',
    { t: '(fragile co-adaptation!)', fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  o += card(278, 12, 248, 178, 'green', [
    'With Dropout (p = 0.5)',
    'Each unit dropped with prob p:',
    { t: 'ξ_j ~ Bernoulli(1 − p)', mono: true, fs: 12.5, fill: D.green.dark },
    'Any pair co-occurs only (1−p)²=25%;',
    'every unit must learn features useful',
    { t: 'with random subsets of partners!', fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 202, 512, 44, 'purple', [
    'Three Complementary Regularizers:\nL₂ Weight Decay (‖W‖²)  ·  Early Stopping  ·  Dropout Masking',
  ]);
  return svg(W, H, o);
}

// 37. ch04-mlp-input-noise-ridge
function inputNoiseRidge() {
  const W = 540, H = 256;
  let o = '';
  o += card(14, 12, 512, 48, 'blue', [
    'Bishop (1995): Linear Predictor Perturbed by Zero-Mean Input Noise ε (Var[ε_j] = s²)',
    { t: 'E_ε [ ( wᵀ(x + ε) + b − y )² ]', mono: true, fs: 14.5, fw: 700, fill: D.blue.dark },
  ]);

  o += vArrow(100, 64, 92, C.blue);
  o += vArrow(270, 64, 92, C.gray);
  o += vArrow(440, 64, 92, C.green);

  o += card(14, 96, 164, 92, 'blue', [
    '1. Clean Squared Loss',
    { t: '(wᵀx + b − y)²', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    'Standard unregularized',
    'empirical risk',
  ], { white: true });

  o += card(188, 96, 164, 92, 'slate', [
    '2. Cross Term Vanishes',
    { t: '2(wᵀx+b−y)wᵀE[ε]', mono: true, fs: 12, fw: 700 },
    { t: '= 0   (since E[ε] = 0)', mono: true, fs: 12.5, fill: C.muted },
  ], { white: true });

  o += card(362, 96, 164, 92, 'green', [
    '3. Exact L₂ Penalty!',
    { t: 'E[(wᵀε)²] = s²‖w‖₂²', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    'Noise variance s² acts',
    'as ridge λ = s²!',
  ], { white: true });

  o += card(14, 202, 512, 42, 'purple', [
    'Dropout generalizes additive input jitter to multiplicative binary noise across every hidden layer.',
  ]);
  return svg(W, H, o);
}

// 38. ch04-mlp-inverted-dropout
function invertedDropout() {
  const W = 540, H = 264;
  let o = '';
  // Top: Training mode
  o += card(14, 12, 512, 30, 'orange', ['(a) Training Mode: Mask & Rescale by 1 / (1 − p)']);
  o += card(14, 50, 160, 68, 'blue', [
    'Activations h',
    { t: '[0.8, 1.2, 0.4, 2.0]', mono: true, fs: 11.5, fw: 700 },
  ], { white: true });
  o += arrow(178, 84, 196, 84, C.amber);
  o += card(200, 50, 140, 68, 'orange', [
    'Gate ξ ~ Bern(0.5)',
    { t: '[1, 0, 0, 1]', mono: true, fs: 12, fw: 700, fill: D.orange.dark },
    { t: 'scale 1/(1−p) = 2', mono: true, fs: 11.5 },
  ], { white: true });
  o += arrow(344, 84, 362, 84, C.amber);
  o += card(366, 50, 160, 68, 'green', [
    'Output h\' = ξh/(1−p)',
    { t: '[1.6, 0.0, 0.0, 4.0]', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    { t: 'E[h\'] = h (unbiased)', fs: 11.5 },
  ], { white: true });

  o += hline(14, 526, 134);

  // Bottom: Evaluation / Test mode
  o += card(14, 146, 512, 30, 'green', ['(b) Evaluation Mode: Exact Identity Pass']);
  o += card(14, 184, 212, 66, 'blue', [
    'Activations h',
    { t: '[ 0.8, 1.2, 0.4, 2.0 ]', mono: true, fs: 12, fw: 700 },
  ], { white: true });
  o += hArrow(232, 296, 217, C.green, 'h\' = h');
  o += card(300, 184, 226, 66, 'green', [
    'No Mask, No Scaling at Test Time!',
    'Expected scale already matched during training',
  ]);
  return svg(W, H, o);
}

// 39. ch04-mlp-dropout-variance
function dropoutVariance() {
  const W = 540, H = 268;
  let o = '';
  o += card(14, 10, 512, 34, 'purple', ['Inverted Dropout Preserves Mean E[h\'] = h but Inflates Variance by p / (1 − p)']);

  const rows = [
    { y: 52,  theme: 'green',  p: 'p = 0.0 (Off)',   surv: '100% survive', scale: 'scale: 1.00×', varStr: 'Var = 0.00 h² (std = 0.0|h|)' },
    { y: 100, theme: 'blue',   p: 'p = 0.2 (Light)', surv: '80% survive',  scale: 'scale: 1.25×', varStr: 'Var = 0.25 h² (std = 0.5|h|)' },
    { y: 148, theme: 'orange', p: 'p = 0.5 (Std)',   surv: '50% survive',  scale: 'scale: 2.00×', varStr: 'Var = 1.00 h² (std = 1.0|h|)' },
    { y: 196, theme: 'red',    p: 'p = 0.8 (Heavy)', surv: '20% survive',  scale: 'scale: 5.00×', varStr: 'Var = 4.00 h² (std = 2.0|h|)' },
  ];

  rows.forEach((r) => {
    o += card(14,  r.y, 118, 38, r.theme, [{ t: r.p, fs: 12.5, fw: 700 }]);
    o += card(138, r.y, 104, 38, r.theme, [{ t: r.surv, mono: true, fs: 12, fill: C.ink }], { white: true });
    o += card(248, r.y, 104, 38, r.theme, [{ t: r.scale, mono: true, fs: 12, fill: C.ink }], { white: true });
    o += card(358, r.y, 168, 38, r.theme, [{ t: r.varStr, mono: true, fs: 11.5, fw: 700, fill: PAL[r.theme].dark }]);
  });

  o += fitTx(270, 252, 'Second-moment inflation E[(h\')²] = h²/(1−p) explains why BatchNorm stats shift after Dropout.', 512, { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// 40. ch04-mlp-dropout-three-views
function dropoutThreeViews() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 512, 34, 'purple', ['Three Complementary Theoretical Views of Dropout']);

  const views = [
    {
      x: 14, theme: 'blue', title: '1. Adaptive Ridge',
      lines: [
        { t: 'Penalty: (p/(1−p)) Σ x_j² w_j²', mono: true, fs: 11.5, fill: D.blue.dark },
        'Exact on linear regression;',
        'penalizes weights on high-',
        'energy features most',
      ],
    },
    {
      x: 188, theme: 'orange', title: '2. Anti-Co-Adaptation',
      lines: [
        { t: 'Pair survival: (1 − p)²', mono: true, fs: 12, fill: D.orange.dark },
        'At p = 0.5, pairs co-occur',
        'only 25% of steps (triples',
        '12.5%), breaking conspiracies',
      ],
    },
    {
      x: 362, theme: 'green', title: '3. 2^h Subnetworks',
      lines: [
        { t: '2⁵¹² ≈ 10¹⁵⁴ thinned nets', mono: true, fs: 12, fill: D.green.dark },
        'Weight-sharing ensemble;',
        'test pass approximates',
        'geometric mean of models',
      ],
    },
  ];
  views.forEach((v) => {
    o += card(v.x, 56, 164, 34, v.theme, [v.title]);
    o += card(v.x, 96, 164, 152, v.theme, v.lines, { white: true });
  });
  return svg(W, H, o);
}

// 41. ch04-mlp-dropout-guidelines
function dropoutGuidelines() {
  const W = 540, H = 266;
  let o = '';
  const items = [
    { x: 14,  y: 14,  theme: 'green',  t1: 'Dense / Tabular / Small n', t2: 'Use p ∈ [0.2, 0.5]',     t3: 'High p/n ratio & multi-epoch noise risk' },
    { x: 276, y: 14,  theme: 'blue',   t1: 'Multi-Epoch Transformers',  t2: 'Use p ≈ 0.1',            t3: 'Light regularization on residual paths' },
    { x: 14,  y: 140, theme: 'slate',  t1: 'Single-Epoch LLM Pretrain', t2: 'Set p = 0.0 (Off)',      t3: 'Fresh tokens every step; no epoch overfit' },
    { x: 276, y: 140, theme: 'red',    t1: 'Directly Before BatchNorm', t2: 'Avoid (p = 0.0)!',       t3: '1/(1−p) variance inflation shifts BN stats' },
  ];
  items.forEach((it) => {
    o += card(it.x, it.y, 250, 110, it.theme, [
      it.t1,
      { t: it.t2, mono: true, fs: 13.5, fw: 700, fill: PAL[it.theme].dark },
      { t: it.t3, fs: 12.5, fill: C.ink },
    ], { white: true, rail: true });
  });
  return svg(W, H, o);
}

// 42. ch04-mlp-dropout-summary
function dropoutSummary() {
  const W = 540, H = 266;
  let o = '';
  const rows = [
    { y: 12,  theme: 'blue',   t1: 'Inverted Mask',     t2: 'h\'_j = ξ_j h_j / (1 − p)  with ξ_j ~ Bernoulli(1 − p)  ⇒  E[h\'_j] = h_j' },
    { y: 62,  theme: 'purple', t1: 'Test-Time Identity', t2: 'Disable mask at eval (h\'_j = h_j): zero test FLOP or memory overhead' },
    { y: 112, theme: 'orange', t1: 'Injected Variance',  t2: 'Var[h\'_j] = (p / (1 − p)) h_j²   (std = |h_j| at p = 0.5; 2|h_j| at p = 0.8)' },
    { y: 162, theme: 'green',  t1: 'Noisy-Label Gain',   t2: 'On 40%-corrupted Fashion-MNIST (n=1,000), p=0.5 boosts val acc by +3 to +5 pts' },
    { y: 212, theme: 'red',    t1: 'Mode-Switch Guard',  t2: 'Always switch to evaluation mode for validation so it is deterministic' },
  ];
  rows.forEach((r) => {
    o += card(14, r.y, 148, 42, r.theme, [{ t: r.t1, fs: 13, fw: 700 }]);
    o += card(168, r.y, 358, 42, r.theme, [{ t: r.t2, mono: true, fs: 11.5, fill: C.ink }], { white: true });
  });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §4.7 KAGGLE-HOUSE-PRICE.MD DIAGRAMS (5 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 43. ch04-mlp-images-vs-tabular
function imagesVsTabular() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 12, 248, 182, 'blue', [
    'Fashion-MNIST Images',
    { t: 'n = 60,000 labeled images', mono: true, fs: 12.5, fill: D.blue.dark },
    '• 784 homogeneous pixel columns',
    '• Identical [0, 1] scale, 0 NaNs',
    '• 10 balanced classes (accuracy)',
    '• p/n = 3.4 params per example',
  ], { white: true, rail: true });

  o += card(278, 12, 248, 182, 'orange', [
    'Kaggle Ames Houses',
    { t: 'n = 1,460 train (+1,459 test)', mono: true, fs: 12.5, fill: D.orange.dark },
    '• 79 heterogeneous columns',
    '  (35 numeric, 44 categorical)',
    '• Scales 10⁰..10⁵; NaNs in 19 cols',
    '• Skewed $34.9k–$755k (log-RMSE)',
  ], { white: true, rail: true });

  o += card(14, 206, 512, 46, 'purple', [
    'Tabular Priority: Leakage-Free Preprocessing + Shared 5-Fold CV + L₂ Weight Decay',
  ]);
  return svg(W, H, o);
}

// 44. ch04-mlp-log-rmse-metric
function logRmseMetric() {
  const W = 540, H = 260;
  let o = '';
  o += card(14, 12, 248, 108, 'red', [
    'Why Dollar RMSE Fails',
    '±$50k on $60k home = 83% error!',
    '±$50k on $650k home = 7.7% error',
    { t: 'Dollar MSE only cares about mansions', fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  o += card(278, 12, 248, 108, 'green', [
    'Kaggle Log-RMSE Metric',
    { t: '√( (1/n) Σ (log y_i − log ŷ_i)² )', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    'log(ŷ/y) ≈ (ŷ − y)/y measures',
    { t: 'relative % error across all brackets!', fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  // Pipeline below: y -> z = log y -> MSE training -> exp(z_hat) = y_hat
  o += card(14, 136, 512, 110, 'blue', [
    'Direct Loss Alignment: Train on Log-Transformed Targets Under Standard MSE',
    { t: 'Raw Price y ($)  ──log(y)──►  Target z = log y  ──MSE(z, ẑ)──►  Predict ẑ  ──exp(ẑ)──►  ŷ ($)', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
    'Log-RMSE of 0.135  ⇔  typical prediction within exp(±0.135) ≈ ±14% of true sale price',
  ]);
  return svg(W, H, o);
}

// 45. ch04-mlp-leakage-free-std
function leakageFreeStd() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 512, 34, 'blue', ['Steps 1 & 2: Leakage-Free Numerical Standardization and Mean Imputation']);

  o += card(14, 58, 156, 126, 'blue', [
    '1. Fit on Train ONLY',
    { t: 'μ̂_j = train[j].mean()', mono: true, fs: 11.5 },
    { t: 'ŝ_j = train[j].std()', mono: true, fs: 11.5 },
    'Never touch val/test',
    'when computing μ̂, ŝ!',
  ], { white: true, rail: true });

  o += arrow(174, 121, 190, 121, C.blue);

  o += card(194, 58, 156, 126, 'purple', [
    '2. Transform All Splits',
    { t: 'x_ij ← (x_ij − μ̂_j)/ŝ_j', mono: true, fs: 12, fw: 700, fill: D.purple.dark },
    'Applies train μ̂_j, ŝ_j',
    'to train, val, and test',
    '(unit variance = 1)',
  ], { white: true, rail: true });

  o += arrow(354, 121, 370, 121, C.green);

  o += card(374, 58, 152, 126, 'green', [
    '3. Impute fillna(0)',
    { t: 'NaN  ←  0.0', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    'Since 0 is the exact',
    'standardized train mean,',
    '0 = mean imputation!',
  ], { white: true, rail: true });

  o += card(14, 198, 512, 52, 'orange', [
    'Why Standardize Before Weight Decay? Raw LotArea (10⁴) vs. Bedrooms (10⁰) would',
    'force tiny weights on LotArea and huge weights on Bedrooms, distorting L₂ penalty ½λ‖W‖².',
  ]);
  return svg(W, H, o);
}

// 46. ch04-mlp-onehot-nan
// Pad a mono table row to a fixed width with no-break spaces.
const nbRow = (s) => s.padEnd(37).replace(/ /g, '\u00a0');

function onehotNan() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 512, 34, 'purple', ['Steps 3 & 4: Categorical One-Hot Encoding With Informative Missingness']);

  // Left: raw categorical column
  o += card(14, 58, 146, 124, 'orange', [
    'Raw Categorical',
    { t: 'Row 1: "RL"', mono: true, fs: 12.5 },
    { t: 'Row 2: "RM"', mono: true, fs: 12.5 },
    { t: 'Row 3: NaN', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
  ], { white: true });

  o += arrow(164, 120, 206, 120, C.purple);

  // Right: Expanded binary columns including _nan
  o += card(210, 58, 316, 124, 'green', [
    'One-Hot + Explicit _nan Indicator (345 cols)',
    // No-break spaces keep the columns aligned (SVG collapses plain spaces).
    { t: nbRow('       _RL _RM _FV _nan'), mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: nbRow('Row 1:  1   0   0   0'), mono: true, fs: 12 },
    { t: nbRow('Row 2:  0   1   0   0'), mono: true, fs: 12 },
    { t: nbRow('Row 3:  0   0   0   1  (MSZoning=NaN)'), mono: true, fs: 12, fw: 700, fill: D.purple.dark },
  ], { white: true });

  o += card(14, 196, 512, 54, 'blue', [
    'Vocabulary Lock via reindex(columns=train_cols, fill_value=0):',
    'Drops unseen test categories & zero-fills absent categories so train/val/test all have 345 columns.',
  ]);
  return svg(W, H, o);
}

// 47. ch04-mlp-target-centering
function targetCentering() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 248, 124, 'red', [
    'Uncentered Log Price (log y ≈ 12.02)',
    { t: 'Initial output ẑ₀ ≈ 0.0', mono: true, fs: 12.5 },
    { t: 'Step-0 MSE ≈ (12.02 − 0)² ≈ 144!', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    'Early SGD epochs waste updates',
    'shifting output bias from 0 to 12',
  ], { white: true, rail: true });

  o += card(278, 12, 248, 124, 'green', [
    'Centered Target (z − offset)',
    { t: 'offset = mean(log y\\_train)\n≈ 12.02', mono: true, fs: 12 },
    { t: 'Step-0 RMSE ≈ std(log y) ≈ 0.40!', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    'Gradients immediately shape',
    'feature weights from epoch 1',
  ], { white: true, rail: true });

  o += card(14, 152, 512, 92, 'blue', [
    'End-to-End Target Centering & Un-Centering Round-Trip in KaggleHouse:',
    { t: 'Train target:  z\\_centered = log(SalePrice) − self.offset     (mean = 0.0, std ≈ 0.40)', mono: true, fs: 12, fill: D.blue.dark },
    { t: 'Submission:    SalePrice_hat = exp( pred_centered + self.offset )   (restores $ scale)', mono: true, fs: 12, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

export const diagrams = {
  // §4.0 index.md
  'ch04-mlp-xor-ceiling':            xorCeiling,
  'ch04-mlp-three-challenges':       threeChallenges,
  'ch04-mlp-benchmark-ladder':       benchmarkLadder,
  'ch04-mlp-chapter-roadmap':        chapterRoadmap,
  // §4.1 mlp.md
  'ch04-mlp-monotonicity-limits':    monotonicityLimits,
  'ch04-mlp-xor-impossibility':      xorImpossibility,
  'ch04-mlp-linear-collapse':        linearCollapse,
  'ch04-mlp-xor-two-solutions':      xorTwoSolutions,
  'ch04-mlp-param-scaling':          paramScaling,
  'ch04-mlp-uat-curse':              uatCurse,
  'ch04-mlp-rectifier-vs-squashing': rectifierVsSquashing,
  // §4.2 mlp-implementation.md
  'ch04-mlp-impl-goals':             implGoals,
  'ch04-mlp-symmetry-breaking':      symmetryBreaking,
  'ch04-mlp-forward-dataflow':       forwardDataflow,
  'ch04-mlp-framework-inits':        frameworkInits,
  'ch04-mlp-impl-summary':           implSummary,
  // §4.3 backprop.md
  'ch04-mlp-backprop-efficiency':    backpropEfficiency,
  'ch04-mlp-local-matmul-rule':      localMatmulRule,
  'ch04-mlp-local-bias-act-rules':   localBiasActRules,
  'ch04-mlp-five-line-sweep':        fiveLineSweep,
  'ch04-mlp-k-layer-recurrence':     kLayerRecurrence,
  'ch04-mlp-worked-numbers':         workedNumbers,
  'ch04-mlp-activation-memory':      activationMemory,
  'ch04-mlp-flop-ratio':             flopRatio,
  // §4.4 numerical-stability-and-init.md
  'ch04-mlp-depth-regimes':          depthRegimes,
  'ch04-mlp-product-chain':          productChain,
  'ch04-mlp-variance-multiplier':    varianceMultiplier,
  'ch04-mlp-sigmoid-dilemma':        sigmoidDilemma,
  'ch04-mlp-xavier-balance':         xavierBalance,
  'ch04-mlp-he-relu-halving':        heReluHalving,
  'ch04-mlp-init-limits':            initLimits,
  // §4.5 generalization-deep.md
  'ch04-mlp-overparam-paradox':      overparamParadox,
  'ch04-mlp-vacuous-vc-bound':       vacuousVcBound,
  'ch04-mlp-inductive-bias-map':     inductiveBiasMap,
  'ch04-mlp-gen-summary':            genSummary,
  // §4.6 dropout.md
  'ch04-mlp-coadaptation':           coadaptation,
  'ch04-mlp-input-noise-ridge':      inputNoiseRidge,
  'ch04-mlp-inverted-dropout':       invertedDropout,
  'ch04-mlp-dropout-variance':       dropoutVariance,
  'ch04-mlp-dropout-three-views':    dropoutThreeViews,
  'ch04-mlp-dropout-guidelines':     dropoutGuidelines,
  'ch04-mlp-dropout-summary':        dropoutSummary,
  // §4.7 kaggle-house-price.md
  'ch04-mlp-images-vs-tabular':      imagesVsTabular,
  'ch04-mlp-log-rmse-metric':        logRmseMetric,
  'ch04-mlp-leakage-free-std':       leakageFreeStd,
  'ch04-mlp-onehot-nan':             onehotNan,
  'ch04-mlp-target-centering':       targetCentering,
};
