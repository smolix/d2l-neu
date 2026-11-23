// Slide diagrams for Chapter 5: Deep Learning Computation (ch05-builders-guide.mjs).
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
};

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const fillRect = (x, y, w, h, fill, rx = 4) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}"/>`;

const hline = (x1, x2, y, color = N.hairline, dash = true) =>
  `<line x1="${x1}" y1="${y}" x2="${x2}" y2="${y}" stroke="${color}" stroke-width="1.5"${dash ? ' stroke-dasharray="4 4"' : ''}/>`;

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

function hArrow(x1, x2, y, color, topLabel, botLabel) {
  let o = arrow(x1, y, x2, y, color);
  const mx = (x1 + x2) / 2;
  const span = Math.abs(x2 - x1);
  if (topLabel) o += fitTx(mx, y - 15, topLabel, span + 24, { fs: 12.5, fw: 700, fill: color, mono: true });
  if (botLabel) o += fitTx(mx, y + 17, botLabel, span + 24, { fs: 12, fw: 600, fill: C.muted });
  return o;
}

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

// Reusable 2x2 quadrant diagram builder
function quadDiagram(titleTheme, titleText, q1, q2, q3, q4, footTheme, footText) {
  const W = 540, H = footText ? 268 : 244;
  let o = '';
  o += card(14, 10, 512, 32, titleTheme, [titleText]);
  const qs = [
    { x: 14,  y: 50,  ...q1 },
    { x: 276, y: 50,  ...q2 },
    { x: 14,  y: 138, ...q3 },
    { x: 276, y: 138, ...q4 },
  ];
  qs.forEach((q) => {
    o += card(q.x, q.y, 250, 78, q.theme, q.lines, { white: true, rail: true });
  });
  if (footText) o += card(14, 226, 512, 32, footTheme || 'slate', [footText]);
  return svg(W, H, o);
}

// Reusable 3-column comparison diagram builder
function threeColDiagram(titleTheme, titleText, c1, c2, c3, footTheme, footText) {
  const W = 540, H = footText ? 268 : 246;
  let o = '';
  o += card(14, 10, 512, 32, titleTheme, [titleText]);
  [c1, c2, c3].forEach((c, i) => {
    const x = 14 + i * 174;
    o += card(x, 52, 164, 34, c.theme, [c.head]);
    o += card(x, 92, 164, 124, c.theme, c.lines, { white: true });
  });
  if (footText) o += card(14, 226, 512, 32, footTheme || 'slate', [footText]);
  return svg(W, H, o);
}

// Reusable 5-row summary table diagram builder
function fiveRowDiagram(rows) {
  const W = 540, H = 266;
  let o = '';
  rows.forEach((r, i) => {
    const y = 12 + i * 50;
    o += card(14,  y, 152, 42, r.theme, [{ t: r.k, fs: 13, fw: 700 }]);
    o += card(172, y, 354, 42, r.theme, [{ t: r.v, mono: false, fs: 12.2, fill: C.ink }], { white: true });
  });
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §5.0 INDEX.MD DIAGRAMS (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const fourBottlenecks = () => quadDiagram(
  'purple', 'Four Production Systems Bottlenecks at 8B-Parameter Scale',
  { theme: 'blue',   lines: ['1. Memory Capacity (§5.2, §5.7)', { t: '16 B/param → 128 GB fp32 Adam', mono: true, fill: D.blue.dark }, 'Exceeds 80 GB GPU before batch 1'] },
  { theme: 'orange', lines: ['2. Numerical Stability (§5.3, §5.5)', { t: 'Unscaled ResNet: 2³²× variance!', mono: true, fill: D.orange.dark }, '16-bit weights round small Δw to 0'] },
  { theme: 'red',    lines: ['3. Fault Tolerance & Security (§5.6)', { t: '1 crash / 3 hrs on 16,384 GPUs', mono: true, fill: D.red.dark }, '96 GB atomic ckpt; safetensors vs RCE'] },
  { theme: 'green',  lines: ['4. Async GPUs & Debugging (§5.7–5.8)', { t: 'Async queues + atomic float adds', mono: true, fill: D.green.dark }, 'Synced timers & non-invasive hooks'] },
  'slate', 'Every subsystem in Chapter 5 operates by traversing the recursive Module Tree.'
);

function moduleTreeHub() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 10, 512, 34, 'purple', ['The Unifying Abstraction: One Registered Module Tree, Six Operations']);
  // Center tree box
  o += card(172, 86, 196, 86, 'purple', [
    'Registered Module Tree',
    { t: 'net → blocks → layers → tensors', fs: 12, fw: 700, fill: D.purple.dark },
    'Parameters + Buffers',
  ]);
  // 6 surrounding subsystem badges
  const ops = [
    { x: 14,  y: 52,  theme: 'blue',   t1: '1. Optimizer',   t2: 'collect params' },
    { x: 14,  y: 112, theme: 'orange', t1: '2. Initializer', t2: 'init every weight' },
    { x: 14,  y: 172, theme: 'green',  t1: '3. Checkpoint',  t2: 'save / load state' },
    { x: 394, y: 52,  theme: 'teal',   t1: '4. Dtype Cast',  t2: 'cast all tensors' },
    { x: 394, y: 112, theme: 'blue',   t1: '5. GPU Move',    t2: 'move to device' },
    { x: 394, y: 172, theme: 'purple', t1: '6. Hooks/Modes', t2: 'train/eval & hooks' },
  ];
  ops.forEach((op) => {
    o += card(op.x, op.y, 132, 48, op.theme, [
      { t: op.t1, fs: 13, fw: 700 },
      { t: op.t2, fs: 12, fill: C.ink },
    ], { white: true });
  });
  o += card(14, 228, 512, 26, 'red', [
    { t: 'Invariant: Unregistered attributes run in the forward pass but are invisible to all 6 walks!', fs: 12.5, fw: 700 },
  ]);
  return svg(W, H, o);
}

function systemsRoadmap() {
  const W = 540, H = 262;
  let o = '';
  const top = [
    { x: 14,  theme: 'blue',   s: '§5.1 Modules',     sub: 'Tree & __setattr__' },
    { x: 146, theme: 'blue',   s: '§5.2 State & RAM', sub: '16 B/param & tying' },
    { x: 278, theme: 'orange', s: '§5.3 Init Scale',  sub: '(1+g²)^N & 1/√N' },
    { x: 410, theme: 'purple', s: '§5.4 Custom Ops',  sub: 'RMSNorm & STE' },
  ];
  top.forEach((b, i) => {
    o += card(b.x, 14, 116, 74, b.theme, [{ t: b.s, fs: 13.5, fw: 700 }, { t: b.sub, mono: true, fs: 11.5, fill: C.ink }]);
    if (i < 3) o += arrow(b.x + 118, 51, top[i + 1].x - 2, 51, C.blue);
  });
  o += arrow(468, 92, 468, 134, C.purple);
  const bot = [
    { x: 410, theme: 'orange', s: '§5.5 Numerics',    sub: 'bf16 + fp32 master' },
    { x: 278, theme: 'green',  s: '§5.6 Checkpoints', sub: 'safetensors & 5-part' },
    { x: 146, theme: 'teal',   s: '§5.7 GPUs & HBM',  sub: 'Async & √K ckpt' },
    { x: 14,  theme: 'red',    s: '§5.8 Repro/Hooks', sub: 'Seeds & NaN traps' },
  ];
  bot.forEach((b, i) => {
    o += card(b.x, 138, 116, 74, b.theme, [{ t: b.s, fs: 13.5, fw: 700 }, { t: b.sub, mono: true, fs: 11.5, fill: C.ink }]);
    if (i < 3) o += arrow(b.x - 2, 175, bot[i + 1].x + 118, 175, C.purple);
  });
  o += card(14, 224, 512, 28, 'slate', [
    { t: 'Top Row: Model Structure & State (§§5.1–5.4)   →   Bottom Row: Precision, I/O & Hardware (§§5.5–5.8)', fs: 12.5, fw: 600 },
  ]);
  return svg(W, H, o);
}

const eightBBenchmarks = () => fiveRowDiagram([
  { theme: 'blue',   k: '§5.1–5.2 Inventory', v: 'Toy 4-Block MLP: 729,866 params  |  Llama 3 8B: 32×9 + 3 = 291 tensors' },
  { theme: 'purple', k: '§5.2 Static Floor',  v: 'Toy 2-Block: 0.28 MiB fp32 Adam  |  8B Model: 16 GB bf16 inf / 128 GB Adam' },
  { theme: 'orange', k: '§5.3 Residual Gain', v: '32 blocks unscaled g=1: 65,536× std  |  Scaled g=1/√N: capped at √e ≈ 1.65×' },
  { theme: 'green',  k: '§5.6 Checkpoints',   v: '200-step resume: 0.0 bit diff  |  8B disk: 32 GB w + 64 GB m,v = 96 GB' },
  { theme: 'teal',   k: '§5.7 Act Checkpt',   v: '16 blocks (B=8192): 1.1 → 0.45 GiB  |  Segment s=√K cuts O(K) → O(√K) RAM' },
]);

// ═══════════════════════════════════════════════════════════════════════════
// §5.1 MODEL-CONSTRUCTION.MD DIAGRAMS (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const scaleBookkeeping = () => threeColDiagram(
  'blue', 'Why Production Architectures Require Recursive Module Trees',
  { theme: 'slate', head: 'Chapter 4 MLP', lines: ['2 Linear Layers', { t: 'W1, b1, W2, b2', mono: true }, '4 parameter tensors:', 'easy to pass by hand'] },
  { theme: 'blue',  head: 'Llama 3 8B Model', lines: ['32 Transformer Blocks', { t: '32 × 9 + 3 = 291', mono: true, fill: D.blue.dark }, '291 distinct tensors', 'across nested blocks'] },
  { theme: 'red',   head: 'Unregistered Hazard', lines: ['Plain Python attribute', { t: 'runs in forward(X)', mono: true, fill: D.red.dark }, 'Skipped by optimizer,', 'checkpoint, & .to(gpu)!'] },
  'purple', 'Solution: Every module registers its children & parameters into a single addressable tree.'
);

function setattrRegistry() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 512, 32, 'blue', ['How Module.__setattr__ Builds the Registry & How net(X) Executes']);
  o += card(14, 54, 156, 82, 'blue', [
    '1. Assignment in __init__',
    { t: 'self.hidden = Linear()', mono: true, fs: 11.5 },
    { t: 'self.out = Linear()', mono: true, fs: 11.5 },
  ], { white: true });
  o += arrow(174, 95, 194, 95, C.blue);
  o += card(198, 54, 152, 82, 'purple', [
    '2. __setattr__ Hook',
    'Detects Module / Param',
    '& inserts into dict:',
    { t: 'child "hidden"', mono: true, fs: 11.5, fill: D.purple.dark },
  ], { white: true });
  o += arrow(354, 95, 374, 95, C.purple);
  o += card(378, 54, 148, 82, 'green', [
    '3. Tree Addressable',
    { t: 'hidden.weight', mono: true, fs: 12, fill: D.green.dark },
    { t: 'out.weight', mono: true, fs: 12, fill: D.green.dark },
    'Visible to all walks!',
  ], { white: true });

  o += card(14, 152, 512, 92, 'orange', [
    'Why You Must Call net(X) (__call__) Instead of net.forward(X):',
    { t: 'net(X)  ──►  1. Pre-Hooks & Lazy Shape Build  ──►  2. forward(X)  ──►  3. Forward Hooks', mono: true, fs: 12, fw: 700, fill: D.orange.dark },
    'Calling net.forward(X) directly bypasses hooks, tracing, and deferred parameter initialization!',
  ]);
  return svg(W, H, o);
}

const treeWalkOps = () => fiveRowDiagram([
  { theme: 'blue',   k: '1. Optimizer Setup',  v: 'Collects registered params → an unregistered layer never gets updates' },
  { theme: 'green',  k: '2. Checkpointing',    v: 'Saves registered state → an unregistered layer is lost & reloads random' },
  { theme: 'teal',   k: '3. Device Placement', v: 'Moves registered tensors → an unregistered layer stays on the CPU' },
  { theme: 'orange', k: '4. Initialization',   v: 'Re-inits registered layers → an unregistered layer keeps default weights' },
  { theme: 'purple', k: '5. Modes & Hooks',    v: 'Toggles train/eval → an unregistered Dropout/BN stays in train mode' },
]);

function sequentialLoop() {
  const W = 540, H = 252;
  let o = '';
  o += card(14, 12, 512, 32, 'purple', ['Anatomy of MySequential: Register in __init__, Chain in forward(X)']);
  o += card(14, 56, 512, 68, 'blue', [
    '1. Construction (__init__ / add): Register Each Module in _modules by Index Key',
    { t: '_modules = { "0": Linear(20→256),   "1": ReLU(),   "2": Linear(256→10) }', mono: true, fs: 12.5, fill: D.blue.dark },
  ], { white: true, rail: true });

  // Chain in forward
  const nodes = [
    { x: 14,  theme: 'slate',  t: 'X : (B, 20)' },
    { x: 138, theme: 'blue',   t: 'block "0"' },
    { x: 262, theme: 'purple', t: 'block "1"' },
    { x: 386, theme: 'green',  t: 'block "2" → O' },
  ];
  nodes.forEach((n, i) => {
    o += card(n.x, 146, i === 3 ? 140 : 104, 54, n.theme, [{ t: n.t, mono: true, fs: 13, fw: 700 }]);
    if (i < 3) o += arrow(n.x + 106, 173, nodes[i + 1].x - 2, 173, C.purple);
  });
  o += tx(270, 226, 'for block in self._modules.values():  X = block(X)', { mono: true, fs: 13.5, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// One variant per framework: each deck shows only its own container rules.
const PLAIN_LIST = {
  pytorch: {
    bad: ['Broken: Plain Python List', { t: 'self.layers = [L1, L2]', mono: true, fs: 13, fill: D.red.dark },
      'forward() still runs, but', { t: 'net.parameters() → 0 params', mono: true, fs: 12, fw: 700, fill: D.red.dark },
      'The optimizer never sees L1, L2'],
    good: ['Fixed: nn.ModuleList', { t: 'nn.ModuleList([L1, L2])', mono: true, fs: 12.5, fill: D.green.dark },
      'All 20×256+256+256×10+10 =', { t: '7,946 parameters registered!', fw: 700, fill: D.green.dark }],
    foot: ['Silent Bug: self.first & self.last as attributes, self.blocks as a plain list:',
      'gradients flow through self.blocks, but its weights stay permanently frozen!'],
  },
  mxnet: {
    bad: ['Broken: Plain Python List', { t: 'self.layers = [L1, L2]', mono: true, fs: 13, fill: D.red.dark },
      'forward() still runs, but', { t: 'collect_params() → 0 params', mono: true, fs: 12, fw: 700, fill: D.red.dark },
      'Gluon only warns about the list'],
    good: ['Fixed: register_child', { t: 'self.register_child(blk)', mono: true, fs: 12.5, fill: D.green.dark },
      'All 20×256+256+256×10+10 =', { t: '7,946 parameters registered!', fw: 700, fill: D.green.dark }],
    foot: ['Silent Bug: self.first & self.last as attributes, self.blocks as a plain list:',
      'gradients flow through self.blocks, but its weights stay permanently frozen!'],
  },
  tensorflow: {
    good: ['Keras Tracks Python Lists', { t: 'self.layers = [L1, L2]', mono: true, fs: 13, fill: D.green.dark },
      'Attribute assignment scans', 'lists, tuples and dicts:', { t: '7,946 parameters tracked', fw: 700, fill: D.green.dark }],
    bad: ['But: No New Layers After Build', { t: 'net.extra = Dense(10)', mono: true, fs: 12.5, fill: D.red.dark },
      'after the variables exist', { t: '→ ValueError', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
      'cannot add state to a built layer'],
    foot: ['Keras locks the layer tree once its variables are built,',
      'so declare every sublayer in __init__ before the first call.'],
  },
  jax: {
    bad: ['Rejected: Plain Python List', { t: 'self.layers = [L1, L2]', mono: true, fs: 13, fill: D.red.dark },
      'NNX raises at construction:', { t: '→ ValueError', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
      'list assigned to a static attribute'],
    good: ['Fixed: nnx.List', { t: 'nnx.List([L1, L2])', mono: true, fs: 12.5, fill: D.green.dark },
      'All 7,946 parameters appear in', { t: 'nnx.state(net, nnx.Param)', mono: true, fs: 12, fw: 700, fill: D.green.dark }],
    foot: ['NNX fails loudly at construction, so a plain list',
      'can never silently hide layers from the optimizer.'],
  },
};
function plainListTrap(fw) {
  const W = 540, H = 264;
  const v = PLAIN_LIST[fw];
  const left = fw === 'tensorflow' ? ['green', v.good] : ['red', v.bad];
  const right = fw === 'tensorflow' ? ['red', v.bad] : ['green', v.good];
  let o = '';
  o += card(14, 12, 248, 174, left[0], left[1], { white: true, rail: true });
  o += card(278, 12, 248, 174, right[0], right[1], { white: true, rail: true });
  o += card(14, 198, 512, 54, 'orange', v.foot);
  return svg(W, H, o);
}

function eagerVsTrace() {
  const W = 540, H = 260;
  let o = '';
  o += card(14, 12, 248, 168, 'blue', [
    'Eager Execution (Default)',
    { t: 'Runs line-by-line on concrete values', fs: 12.5, fill: D.blue.dark },
    '• Data-dependent Python if / while',
    '  (while norm(X) > 1:) works!',
    '• Interactive print() & pdb work',
    '• Python dispatch overhead per op',
  ], { white: true, rail: true });

  o += card(278, 12, 248, 168, 'purple', [
    'Compiled Graph Tracing',
    { t: 'jit-compile the step', fs: 12.5, fill: D.purple.dark },
    '• Traces forward() ONCE on abstract',
    '  shape/dtype symbols (no values!)',
    '• Static structure loops (for m in',
    '  self.blocks) unroll cleanly',
  ], { white: true, rail: true });

  o += card(14, 192, 512, 56, 'orange', [
    'Tracing Rule: Loops over static model structure always compile;',
    'loops conditioned on runtime tensor values must become symbolic control flow\n(by hand or by the compiler).',
  ]);
  return svg(W, H, o);
}

function lazyDryRun() {
  const W = 540, H = 256;
  let o = '';
  o += card(14, 12, 512, 32, 'blue', ['Three-Stage Lifecycle of Lazy Shape Inference']);

  o += card(14, 56, 156, 122, 'orange', [
    '1. Construct Lazy',
    { t: 'n_out = 256', mono: true, fs: 12.5 },
    { t: 'n_in  = ?', mono: true, fs: 12.5, fill: D.orange.dark },
    'Weight shape unknown;',
    '0 bytes allocated yet',
  ], { white: true, rail: true });
  o += arrow(174, 117, 190, 117, C.blue);
  o += card(194, 56, 156, 122, 'blue', [
    '2. Dry Run net(X)',
    { t: 'X : (2, 20)', mono: true, fs: 13 },
    'Infers n_in = 20;',
    'allocates 20 × 256 W',
    '& runs the initializer',
  ], { white: true, rail: true });
  o += arrow(354, 117, 370, 117, C.green);
  o += card(374, 56, 152, 122, 'green', [
    '3. Custom Init',
    'Apply He init',
    'Now n_in = 20 is known',
    'so σ = √(2/n_in) can',
    'be computed!',
  ], { white: true, rail: true });

  o += card(14, 192, 512, 52, 'purple', [
    'Dry-Run Rule: Variance-scaling initializers need n_in, so push one dummy batch',
    'through a lazy model before applying a custom initializer.',
  ]);
  return svg(W, H, o);
}

function lazyDryRunJax() {
  const W = 540, H = 256;
  let o = '';
  o += card(14, 12, 512, 32, 'blue', ['Explicit Shapes: Parameters Exist as Soon as the Constructor Returns']);

  o += card(14, 56, 156, 122, 'orange', [
    '1. Construct',
    { t: 'Linear(20, 256)', mono: true, fs: 12.5 },
    'n_in and n_out given;',
    'weights allocated',
    'right away',
  ], { white: true, rail: true });
  o += arrow(174, 117, 190, 117, C.blue);
  o += card(194, 56, 156, 122, 'blue', [
    '2. Inspect',
    { t: 'kernel : (20, 256)', mono: true, fs: 12.5 },
    'Shapes and bytes',
    'are known before',
    'any data arrives',
  ], { white: true, rail: true });
  o += arrow(354, 117, 370, 117, C.green);
  o += card(374, 56, 152, 122, 'green', [
    '3. Custom Init',
    'Apply He init',
    'n_in = 20 known',
    'so σ = √(2/n_in)',
    'applies at once',
  ], { white: true, rail: true });

  o += card(14, 192, 512, 52, 'purple', [
    'No Dry Run Needed: every parameter is allocated and inspectable',
    'immediately, so custom initialization can follow construction directly.',
  ]);
  return svg(W, H, o);
}

function configBuilder() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 512, 32, 'purple', ['Reconstructing a Model Requires Three Separate Artifacts']);

  o += card(14, 56, 156, 118, 'blue', [
    '1. Config (@dataclass)',
    { t: 'num_inputs: 784', mono: true, fs: 11.5 },
    { t: 'num_hiddens: 256', mono: true, fs: 11.5 },
    { t: 'num_blocks: 4', mono: true, fs: 11.5 },
    'Saved as config.json',
  ], { white: true, rail: true });

  o += card(192, 56, 156, 118, 'purple', [
    '2. Builder Function',
    { t: 'build_mlp(cfg)', mono: true, fs: 13, fw: 700 },
    'Executable Python code',
    'wiring ResidualBlocks',
    '& ReLU activations',
  ], { white: true, rail: true });

  o += card(370, 56, 156, 118, 'green', [
    '3. State Dictionary',
    { t: '729,866 float32s', mono: true, fs: 12.5, fw: 700 },
    '20 weight/bias tensors',
    'keyed by tree path in',
    'model.safetensors',
  ], { white: true, rail: true });

  o += card(14, 188, 512, 58, 'slate', [
    'Analytical Count Check: Input (784×256+256 = 200,960) + 4 Blocks (8×(256²+256) = 526,336)',
    '+ Output Head (256×10+10 = 2,570) = 729,866 Total Parameters across 10 Linear layers.',
  ]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §5.2 PARAMETERS-STATE-MEMORY.MD DIAGRAMS (9 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function adam16bFloor() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 10, 512, 34, 'red', ['The 16-Byte-per-Parameter Static Floor of fp32 Adam Training (8B Model)']);

  const parts = [
    { x: 14,  theme: 'blue',   t1: 'Weights w',      t2: 'fp32 · 4 B/param', t3: '32 GB' },
    { x: 144, theme: 'red',    t1: 'Gradients g',    t2: 'fp32 · 4 B/param', t3: '32 GB' },
    { x: 274, theme: 'orange', t1: '1st Moment m',   t2: 'fp32 · 4 B/param', t3: '32 GB' },
    { x: 404, theme: 'purple', t1: '2nd Moment v',   t2: 'fp32 · 4 B/param', t3: '32 GB' },
  ];
  parts.forEach((p) => {
    o += card(p.x, 56, 122, 112, p.theme, [
      p.t1,
      { t: p.t2, mono: true, fs: 11.5, fill: C.ink },
      { t: p.t3, mono: true, fs: 18, fw: 700, fill: PAL[p.theme].dark },
    ]);
  });

  o += card(14, 182, 512, 70, 'red', [
    'Total Persistent State = 16 B × 8 × 10⁹ = 128 GB (vs. 80 GB Single-GPU HBM Limit!)',
    { t: 'Exceeds an 80 GB A100/H100 before loading a single token or computing a single activation.', fs: 12.5, fill: C.ink },
  ], { white: true, rail: true });
  return svg(W, H, o);
}

function stateTreePaths() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 10, 512, 32, 'blue', ['Two-Block Residual MLP Testbed: 12 Leaf Tensors (18,634 Parameters)']);

  const cols = [
    { x: 14,  theme: 'blue',   k: 'Child "0" (Input)',   sh: '20 → 64 Linear',   p: 'weight 20×64 + bias 64', c: '1,344 params' },
    { x: 188, theme: 'purple', k: 'Children "1" & "2"',  sh: '2 × ResidualBlock', p: '2 blocks × 2 layers × (W, b)', c: '16,640 params' },
    { x: 362, theme: 'green',  k: 'Child "3" (Output)',  sh: '64 → 10 Linear',   p: 'weight 64×10 + bias 10', c: '650 params' },
  ];
  cols.forEach((c) => {
    o += card(c.x, 52, 164, 134, c.theme, [
      c.k,
      { t: c.sh, fs: 12.5, fw: 600 },
      { t: c.p, fs: 12, fill: C.ink },
      { t: c.c, mono: true, fs: 13.5, fw: 700, fill: PAL[c.theme].dark },
    ], { white: true, rail: true });
  });

  o += card(14, 198, 512, 52, 'slate', [
    'Path Addressing Mirrors Attribute Traversal:',
    { t: 'block 1 → body → layer 0 → weight   (18,634 params × 16 B = 0.28 MiB)', fs: 12.5, fw: 600, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function fourTensorKinds() {
  const W = 540, H = 268;
  let o = '';
  o += card(14, 10, 512, 32, 'purple', ['Where Each Kind of Value Inside a Module Goes']);

  const rows = [
    { y: 50,  theme: 'green',  kind: '1. Parameter',       flags: 'Opt: YES\nCkpt: YES · GPU: YES', ex: 'Weights W, biases b, RMSNorm gain g' },
    { y: 104, theme: 'blue',   kind: '2. Registered Buffer', flags: 'Opt: NO\nCkpt: YES · GPU: YES', ex: 'Feature μ, σ, RandomFeatures W, BN stats' },
    { y: 158, theme: 'orange', kind: '3. Plain Attribute', flags: 'Opt: NO\nCkpt: NO · GPU: NO!', ex: 'Python floats/ints (eps=1e-6); NEVER tensors!' },
    { y: 212, theme: 'slate',  kind: '4. Local Variable',  flags: 'Opt: NO\nCkpt: NO · GPU: Live', ex: 'Forward activations cached for 1 backward' },
  ];
  rows.forEach((r) => {
    o += card(14,  r.y, 146, 44, r.theme, [{ t: r.kind, fs: 13, fw: 700 }]);
    o += card(166, r.y, 178, 44, r.theme, [{ t: r.flags, mono: true, fs: 11.2, fw: 700, fill: PAL[r.theme].dark }], { white: true });
    o += card(350, r.y, 176, 44, r.theme, [{ t: r.ex, fs: 11.8, fill: C.ink }], { white: true });
  });
  return svg(W, H, o);
}

function standardizerBuffer() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 512, 56, 'blue', [
    'Sequential( Standardizer(μ, σ),  Linear(W, b) )   —   Full State Tree',
    { t: 'Buffers: mean, std (non-trainable)\nParameters: weight, bias (trainable)', mono: true, fs: 12, fill: C.ink },
  ]);

  o += vArrow(138, 72, 102, C.green, 'all 4 tensors');
  o += vArrow(402, 72, 102, C.purple, 'trainable only');

  o += card(14, 106, 248, 138, 'green', [
    'Checkpoints & Device Moves',
    { t: 'saved state → 4 tensors:', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '• mean, std (buffers)', mono: true, fs: 12 },
    { t: '• weight, bias (params)', mono: true, fs: 12 },
    'Preprocessing travels with model',
    'to disk and to GPU HBM!',
  ], { white: true, rail: true });

  o += card(278, 106, 248, 138, 'purple', [
    'Optimizer (SGD / Adam)',
    { t: 'trainable → 2 tensors:', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
    { t: '• weight, bias ONLY', mono: true, fs: 12 },
    'Skips mean & std so',
    'gradient descent never',
    'corrupts training-set stats!',
  ], { white: true, rail: true });
  return svg(W, H, o);
}

function lazyAdamMoments(fw) {
  const W = 540, H = 254;
  const eager = fw === 'jax';
  let o = '';
  o += card(14, 12, 512, 32, 'orange', [eager
    ? 'Adam Allocates Its Two Moment Buffers (m, v) When It Is Constructed'
    : 'Adam Allocates Its Two Moment Buffers (m, v) Lazily on Step 1']);

  o += card(14, 56, 230, 122, 'blue', eager ? [
    'Before the Optimizer Exists',
    { t: 'Model params: 18,634 floats', mono: true, fs: 12.5, fill: D.blue.dark },
    { t: 'Optimizer state: none yet', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    'Only weights (and later',
    'gradients) occupy memory',
  ] : [
    'Before the First Update',
    { t: 'Model params: 18,634 floats', mono: true, fs: 12.5, fill: D.blue.dark },
    { t: 'Optimizer state: 0 floats!', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    'Adam holds no moments',
    'right after construction',
  ], { white: true, rail: true });

  o += hArrow(248, 292, 117, C.purple, eager ? 'init' : 'step 1');

  o += card(296, 56, 230, 122, 'purple', [
    eager ? 'After Constructing Adam' : 'After the First Update',
    { t: 'm (1st moment): 18,634 floats', mono: true, fs: 12, fill: D.purple.dark },
    { t: 'v (2nd moment): 18,634 floats', mono: true, fs: 12, fill: D.purple.dark },
    { t: 'Optimizer state = 37,268 floats', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    '(2× full model copies added!)',
  ], { white: true, rail: true });

  o += card(14, 192, 512, 50, 'slate', eager ? [
    'Systems Consequence: Adam adds +8 B/param (64 GB for an 8B model) the moment it is',
    'constructed, so budget for it before the first forward pass ever runs!',
  ] : [
    'Systems Consequence: A training job that fits during the forward and backward pass can still OOM',
    'on the very first optimizer step, when Adam suddenly allocates +8 B/param (64 GB for 8B)!',
  ]);
  return svg(W, H, o);
}

function metaShapeTrace() {
  const W = 540, H = 256;
  let o = '';
  o += card(14, 12, 248, 124, 'red', [
    'Eager CPU Allocation',
    { t: 'Linear(4096, 128256)', mono: true, fs: 13 },
    'Allocates & randomly fills',
    '525,336,576 float32 numbers:',
    { t: '2.10 GB RAM consumed!', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  o += card(278, 12, 248, 124, 'green', [
    'Abstract Shape Tracing',
    { t: 'Trace constructor on shapes only', fs: 12, fw: 700, fill: D.green.dark },
    'Records shape (128256, 4096)',
    'and dtype float32 in header:',
    { t: '0 Bytes RAM allocated!', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 150, 512, 92, 'blue', [
    'Abstract Tracing Records Only (shape, dtype) Pairs:',
    { t: '525,336,576 × 4 B = 2.10 GB counted   ──►   0 B allocated', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
    'Audit a model\'s exact memory budget before touching real hardware!',
  ]);
  return svg(W, H, o);
}

function tiedDedupGrad() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 512, 32, 'purple', ['Weight Tying in ToyLM (|V|=100, d=16): Deduplication & Gradient Sum']);

  o += card(14, 56, 248, 126, 'blue', [
    '1. Storage Deduplication',
    { t: 'head reuses the embedding W', fs: 12, fw: 700, fill: D.blue.dark },
    'Untied: 1600 + 272 + 1600 = 3,472',
    'Tied:   1600 + 272        = 1,872',
    { t: 'Tree walk counts shared W once!', fw: 700, fill: D.blue.dark },
  ], { white: true, rail: true });

  o += card(278, 56, 248, 126, 'red', [
    '2. Multivariate Chain Rule',
    { t: 'W used at lookup AND logits:', mono: true, fs: 12 },
    { t: '∂L/∂W = ∂L/∂W_emb + ∂L/∂W_head', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    'Autograd sums gradients from',
    'both call sites into one grad!',
  ], { white: true, rail: true });

  o += card(14, 194, 512, 54, 'green', [
    'GPT-2 Small (|V| = 50,257, d = 768): tying saves 38.6M of 124M parameters (31%)',
    'and makes input embeddings and output logits share one geometric space.',
  ]);
  return svg(W, H, o);
}

const freezingPitfalls = () => quadDiagram(
  'red', 'Two Silent Systems Pitfalls When Freezing Backbone Parameters',
  { theme: 'red',   lines: ['Pitfall 1: Stale Adam Moments', { t: 'Optimizer built before freezing', fw: 700, fill: D.red.dark }, '64 GB of m, v stay allocated in RAM!'] },
  { theme: 'green', lines: ['Fix 1: Filter Optimizer Upfront', { t: 'Adam over trainable params only', fw: 700, fill: D.green.dark }, 'Allocates moments ONLY for head (4.2 GB)'] },
  { theme: 'orange', lines: ['Pitfall 2: Drifting BatchNorm Buffers', { t: 'Frozen weights, still in train mode', fw: 700, fill: D.orange.dark }, 'Running mean/var buffers still mutate!'] },
  { theme: 'green', lines: ['Fix 2: Lock Evaluation Mode Too', { t: 'Eval mode + frozen weights', fw: 700, fill: D.green.dark }, 'Freezes both weights AND running stats'] },
  'slate', 'Head-only 8B fine-tuning fits in 38.3 GB (32 GB weights + 6.3 GB head grad/Adam) on one 80 GB GPU!'
);

const stateSummary = () => fiveRowDiagram([
  { theme: 'green',  k: 'Parameters vs Buffers', v: 'Both saved in checkpoints & moved to GPU; only Parameters get gradients' },
  { theme: 'red',    k: '16 B/param Adam Floor', v: 'fp32 w (4 B) + g (4 B) + m (4 B) + v (4 B) = 128 GB for 8B model' },
  { theme: 'blue',   k: 'Abstract Counting',     v: 'Abstract shape tracing audits exact byte budgets with 0 B physical RAM' },
  { theme: 'purple', k: 'Weight Tying',          v: 'Aliases embedding & head to one tensor; deduplicated in count, grads sum' },
  { theme: 'orange', k: 'Safe Freezing',         v: 'Build optimizer on unfrozen subset only + frozen BatchNorm in eval mode' },
]);

// ═══════════════════════════════════════════════════════════════════════════
// §5.3 INIT.MD DIAGRAMS (10 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function resExplosion() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 248, 122, 'blue', [
    'Plain Feedforward Layer',
    { t: 'x_{k+1} = φ(W_k x_k)', mono: true, fs: 13 },
    'He init (σ = √(2/n_in)) gives:',
    { t: 'Var[x_{k+1}] = 1.0 × Var[x_k]', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    'Stays flat across depth',
  ], { white: true, rail: true });

  o += card(278, 12, 248, 122, 'red', [
    'Unscaled Residual Block (g = 1)',
    { t: 'x_{k+1} = x_k + f_k(x_k)', mono: true, fs: 13 },
    'Uncorrelated skip + body add:',
    { t: 'Var[x_{k+1}] = (1 + 1) Var[x_k] = 2×!', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    'Doubles variance at EVERY block!',
  ], { white: true, rail: true });

  o += card(14, 148, 512, 100, 'orange', [
    'Exponential Compounding Across N = 32 Residual Blocks at Step 0:',
    { t: 'Variance: m₃₂ = 2³² m₀ ≈ 4.29 × 10⁹ m₀     ⇒     Activation Std: √(2³²) = 2¹⁶ = 65,536 × std(x₀)!', mono: true, fs: 12, fw: 700, fill: D.red.dark },
    'Overflows fp16 (max 65,504) on block 32 before a single gradient step is taken!',
  ]);
  return svg(W, H, o);
}

const defaultInitDecay = () => fiveRowDiagram([
  { theme: 'red',    k: 'Uniform ±1/√n_in',   v: 'σ² = 1/(3n_{in}) → ReLU mult ×1/6 (10 layers → 1.7×10⁻⁸)' },
  { theme: 'orange', k: 'LeCun Normal',       v: 'N(0, 1/n_in) → σ² = 1/n_in → ReLU mult ×1/2 (10 layers → 9.8×10⁻⁴)' },
  { theme: 'orange', k: 'Glorot Uniform',     v: 'n_in = n_out → σ² = 1/n_in → ReLU mult ×1/2 (10 layers → 9.8×10⁻⁴)' },
  { theme: 'red',    k: 'Uniform ±0.07',      v: 'σ² ≈ 0.00163 → ReLU mult ×0.21 (10 layers → 1.6×10⁻⁷)' },
  { theme: 'green',  k: 'Explicit He Normal', v: 'N(0, 2/n_in) → σ² = 2/n_in → ReLU mult ×1 (10 layers → 1.00 preserved!)' },
]);

function treeApplyInit() {
  const W = 540, H = 254;
  let o = '';
  o += card(14, 12, 512, 32, 'blue', ['Tree-Wide Initialization: He Init Applied to Every Submodule']);
  o += card(14, 56, 156, 118, 'red', [
    '1. Default Stack',
    '10 ReLU Linear layers',
    'with default init:',
    { t: 'E[y²] ≈ 10⁻⁸–10⁻³', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    '(signal collapsed!)',
  ], { white: true, rail: true });
  o += arrow(174, 115, 190, 115, C.blue);
  o += card(194, 56, 156, 118, 'purple', [
    '2. Recursive Walk',
    'Visits every Linear',
    'layer and redraws its',
    'weight in place',
    'with σ = √(2/n_in)',
  ], { white: true, rail: true });
  o += arrow(354, 115, 370, 115, C.green);
  o += card(374, 56, 152, 118, 'green', [
    '3. Restored Scale',
    'Parameter references',
    'remain intact;',
    { t: 'E[y²] ≈ 0.6–1.1', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 188, 512, 54, 'orange', [
    'In-Place Invariant: Overwrite each weight in place so existing references',
    'and tied weights keep pointing to the same parameter.',
  ]);
  return svg(W, H, o);
}

function truncatedNormal() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 248, 154, 'red', [
    'Standard Normal N(0, σ²)',
    'Unbounded tails (−∞, +∞):',
    'Across 10⁸ foundation-model',
    'weights, ~60 entries exceed ±5σ',
    { t: 'max |w| ≈ 5.6σ  (extreme spikes!)', mono: true, fs: 12, fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  o += card(278, 12, 248, 154, 'green', [
    'Truncated Normal (±2σ Cutoff)',
    'Rejects & redraws |w| > 2σ:',
    'Strictly bounds every initial weight',
    'inside [−2σ, +2σ] (BERT / ViT):',
    { t: 'max |w| ≤ 2.0σ  (zero outliers!)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 180, 512, 66, 'purple', [
    'Variance Shrinkage Factor of ±2σ Truncation:',
    { t: 'std shrinks by 0.8796×  (σ = 0.02 → std ≈ 0.0176)', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
    'Variance-compensated initializers divide by 0.8796 to restore σ.',
  ]);
  return svg(W, H, o);
}

function weightShapeConventions(layout) {
  const W = 540, H = 256;
  const outIn = layout === 'out-in';
  const nm = outIn ? 'W' : 'K';
  const [ai, ao] = outIn ? [1, 0] : [0, 1];
  let o = '';
  o += card(14, 12, 512, 32, 'blue', ['Finding Fan-In n_in for Custom Initializers σ = g / √n_in']);

  o += card(14, 56, 248, 128, 'blue', [
    'Weight Layout',
    { t: outIn ? 'y = x Wᵀ + b' : 'y = x K + b', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: outIn ? 'W.shape == (n_out, n_in)' : 'K.shape == (n_in, n_out)', mono: true, fs: 13, fw: 700 },
    `Fan-in is axis ${ai}:  n_in = ${nm}.shape[${ai}]`,
    `Fan-out is axis ${ao}: n_out = ${nm}.shape[${ao}]`,
  ], { white: true, rail: true });

  o += card(278, 56, 248, 128, 'purple', [
    'Example: d → 4d Expansion',
    { t: outIn ? 'W.shape == (4d, d)' : 'K.shape == (d, 4d)', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    { t: `n_in = ${nm}.shape[${ai}] = d`, mono: true, fs: 13, fw: 700 },
    'so σ = g / √d',
  ], { white: true, rail: true });

  o += card(14, 196, 512, 48, 'red', [
    `Bug Alert: Reading shape[${ao}] as n_in on a d → 4d expansion layer`,
    'uses 4d instead of d, making initial weights 2× too small (4× too little variance)!',
  ]);
  return svg(W, H, o);
}

function resVarianceDerivation() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 10, 512, 32, 'purple', ['Mean-Square Propagation Through One d → 4d → d Residual Block']);

  const steps = [
    { x: 14,  theme: 'slate',  t1: 'Input x_k',       t2: 'width d',    t3: 'E[x²] = m_k' },
    { x: 146, theme: 'blue',   t1: 'Linear d→4d',     t2: 'He g = √2',  t3: 'E[z²] = 2 m_k' },
    { x: 278, theme: 'purple', t1: 'ReLU(z)',         t2: 'halves E[·²]', t3: 'E[h²] = m_k' },
    { x: 410, theme: 'orange', t1: 'Proj 4d→d',       t2: 'gain g',     t3: 'E[f²] = g² m_k' },
  ];
  steps.forEach((s, i) => {
    o += card(s.x, 54, 116, 86, s.theme, [
      { t: s.t1, fs: 13, fw: 700 },
      { t: s.t2, mono: true, fs: 12 },
      { t: s.t3, mono: true, fs: 13, fw: 700, fill: PAL[s.theme].dark },
    ]);
    if (i < 3) o += arrow(s.x + 118, 97, steps[i + 1].x - 2, 97, C.blue);
  });

  o += card(14, 156, 512, 92, 'green', [
    'Add Uncorrelated Skip Connection:  x_{k+1} = x_k + f_k(x_k)   (with E[x_kᵀ f_k(x_k)] = 0)',
    { t: 'm_{k+1} = E[x_k²] + E[f_k(x_k)²] = m_k + g² m_k = (1 + g²) m_k', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: '⇒   m_N = (1 + g²)^N m_0', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });
  return svg(W, H, o);
}

const threeGainSchedules = () => threeColDiagram(
  'purple', 'Three Output-Projection Gain Schedules Across N = 32 Residual Blocks',
  { theme: 'red',   head: '1. Unscaled (g = 1)',     lines: ['Per-block: ×2.0', { t: 'm_N = 2^N m_0', mono: true, fill: D.red.dark }, 'At N = 32 blocks:', { t: 'std = 65,536×!', mono: true, fw: 700, fill: D.red.dark }] },
  { theme: 'blue',  head: '2. GPT-2 (g = 1/√N)',     lines: ['Per-block: ×(1+1/N)', { t: 'm_N = (1+1/N)^N ≤ e', mono: true, fill: D.blue.dark }, 'At ANY depth N:', { t: 'std ≤ √e ≈ 1.65×', mono: true, fw: 700, fill: D.blue.dark }] },
  { theme: 'green', head: '3. Fixup Zero (g = 0)',   lines: ['Per-block: ×1.0', { t: 'm_N = 1^N m_0 = m_0', mono: true, fill: D.green.dark }, 'Starts as identity:', { t: 'std = 1.00× exact', mono: true, fw: 700, fill: D.green.dark }] },
  'slate', 'Both g = 1/√N and g = 0 prevent depth explosion across hundreds of transformer blocks.'
);

function gpt2SqrtNBound() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 512, 48, 'blue', [
    'Why Output-Projection Scaling g = 1 / √N Bounds Variance at Euler\'s Number e',
    { t: 'm_N / m_0  =  ( 1 + g² )^N  =  ( 1 + 1/N )^N  ↗  e ≈ 2.71828   as N → ∞', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);

  const depths = [
    { x: 14,  n: 'N = 2 Blocks',    m: '(1.5)² = 2.25',   s: 'std = 1.50×' },
    { x: 146, n: 'N = 8 Blocks',    m: '(1.125)⁸ = 2.57', s: 'std = 1.60×' },
    { x: 278, n: 'N = 32 Blocks',   m: '(1+1/32)³² = 2.68', s: 'std = 1.64×' },
    { x: 410, n: 'N = 1000 Blocks', m: '→ e ≈ 2.718',     s: 'std = 1.65×' },
  ];
  depths.forEach((d) => {
    o += card(d.x, 74, 116, 102, 'green', [
      { t: d.n, fs: 13, fw: 700 },
      { t: d.m, mono: true, fs: 12, fill: C.ink },
      { t: d.s, mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    ], { white: true });
  });

  o += card(14, 190, 512, 54, 'purple', [
    'The Same 1/√N Scaling in GPT-2 / GPT-3: Count total residual branches N_branches',
    '(2 per Transformer block: 1 Attention + 1 FFN) and scale each output projection\'s init by 1/√N_branches.',
  ]);
  return svg(W, H, o);
}

function zeroOutProjWakeup() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 512, 32, 'green', ['Why Zeroing ONLY the Residual Output Projection (g = 0) Breaks Symmetry']);

  o += card(14, 54, 158, 128, 'blue', [
    'Step 0 Forward',
    { t: 'W_in ~ N(0, 2/d) (He!)', mono: true, fs: 11.5, fill: D.blue.dark },
    { t: 'h_k = ReLU(x_k W_in) ≠ 0', mono: true, fs: 11.5, fw: 700 },
    { t: 'W_out = 0 ⇒ f_k(x_k) = 0', mono: true, fs: 11.5, fill: D.orange.dark },
    'Block is exact identity!',
  ], { white: true, rail: true });

  o += arrow(176, 118, 190, 118, C.purple);

  o += card(194, 54, 156, 128, 'purple', [
    'Step 1 Backward',
    'Since h_k ≠ 0 has distinct',
    'random features:',
    { t: '∂L/∂W_out = h_kᵀ G ≠ 0!', mono: true, fs: 12, fw: 700, fill: D.purple.dark },
    'W_out wakes up on step 1!',
  ], { white: true, rail: true });

  o += arrow(354, 118, 368, 118, C.green);

  o += card(372, 54, 154, 128, 'green', [
    'Step 2+ Training',
    'Now W_out ≠ 0, so',
    { t: '∂L/∂h_k = G W_outᵀ ≠ 0', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    'Both W_in and W_out',
    'train with broken symmetry!',
  ], { white: true, rail: true });

  o += card(14, 194, 512, 54, 'red', [
    'Crucial Distinction: Never zero BOTH W_in and W_out! If W_in = 0 too, then h_k = 0',
    'and ∂L/∂W_out = 0 forever—locking the entire residual branch at zero permanently.',
  ]);
  return svg(W, H, o);
}

const initSummary = () => fiveRowDiagram([
  { theme: 'blue',   k: 'Single-Layer Scale', v: 'σ = g / √n_in  (g = √2 for ReLU, g = 1 linear); defaults decay deep stacks' },
  { theme: 'purple', k: 'Tree-Wide Apply',    v: 'net.apply(fn) overwrites leaf weights in place while preserving references' },
  { theme: 'orange', k: 'Truncated Normal',   v: 'Redraws |w| > 2σ (shrinks std by 0.88×) to eliminate >5σ outliers at 10⁸ scale' },
  { theme: 'red',    k: 'Residual Variance',  v: 'm_{k+1} = (1 + g²) m_k ⇒ unscaled g=1 explodes std by 2^{N/2} (65,536× at N=32)' },
  { theme: 'green',  k: 'Branch Scaling',     v: 'Scale output projection by g = 1/√N (std ≤ √e ≈ 1.65×) or g = 0 (identity start)' },
]);

// ═══════════════════════════════════════════════════════════════════════════
// §5.4 CUSTOM-LAYERS.MD DIAGRAMS (10 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const customLayerNeeds = () => threeColDiagram(
  'purple', 'Three Reasons Every Deep Learning Engineer Writes Custom Modules',
  { theme: 'blue',   head: '1. Research Velocity', lines: ['RMSNorm (2019) drove', 'Llama & Gopher years', 'before reaching', { t: 'major libraries', fw: 700, fill: D.blue.dark }] },
  { theme: 'green',  head: '2. Tree Composability', lines: ['Custom modules auto-', 'integrate with:', '• Optimizers (Param)', '• Checkpoints (Buffer)', '• Device moves'] },
  { theme: 'orange', head: '3. Surrogate Grads',   lines: ['4-bit quantization', { t: 'y = round(w)', mono: true, fill: D.orange.dark }, 'has ∂y/∂w = 0; needs', 'custom STE backward!'] },
  'slate', 'Subclass the base module and register parameters and buffers to make custom layers first-class.'
);

function statelessCentered() {
  const W = 540, H = 252;
  let o = '';
  o += card(14, 12, 512, 32, 'blue', ['Stateless Custom Module: CenteredLayer Inside a Sequential Pipeline']);

  o += card(14, 58, 148, 86, 'slate', [
    'Input X',
    { t: 'shape (4, 8)', mono: true, fs: 13 },
    'Arbitrary mean',
  ], { white: true });
  o += arrow(166, 101, 186, 101, C.blue);
  o += card(190, 58, 152, 86, 'blue', [
    'Linear(8 → 128)',
    { t: '1,152 params', mono: true, fs: 12.5, fill: D.blue.dark },
    'Output Y : (4, 128)',
  ], { white: true });
  o += arrow(346, 101, 366, 101, C.purple);
  o += card(370, 58, 156, 86, 'purple', [
    'CenteredLayer()',
    { t: 'Y − Y.mean()', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    '0 params, 0 buffers',
  ]);

  o += card(14, 160, 512, 78, 'green', [
    'Verification Invariant: Output Mean Vanishes to Float32 Machine Epsilon',
    { t: 'net(X).mean() ≈ ±10⁻⁸   (within float32 eps = 2⁻²³ ≈ 1.19 × 10⁻⁷ of exact 0.0)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });
  return svg(W, H, o);
}

function rmsnormFormula() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 10, 512, 48, 'purple', [
    'RMSNorm (Zhang & Sennrich, 2019)',
    { t: 'RMS(x) = √((1/d) Σ_{i=1}^{d} x_i² + ε)  ⇒  RMSNorm(x) = (x / RMS(x)) ⊙ g', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ]);

  o += card(14, 70, 248, 116, 'blue', [
    'Exact Scale Invariance (c·x ↦ same)',
    { t: 'x = (3, 4) ⇒ RMS = √12.5 ≈ 3.536', mono: true, fs: 12 },
    { t: 'x / RMS = (0.849, 1.131)', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: '100·x = (300, 400) ⇒ (0.849, 1.131)!', mono: true, fs: 12, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(278, 70, 248, 116, 'green', [
    'Why Foundation Models Prefer It',
    '• Drops mean-centering (x − μ)',
    '• Drops additive bias β (only gain g ∈ ℝᵈ)',
    '• Saves 1 reduction pass across all',
    '  65 norm sites in Llama 3 8B!',
  ], { white: true, rail: true });

  o += card(14, 198, 512, 52, 'orange', [
    'Role of ε ≈ 10⁻⁶ Inside the Square Root: Prevents division by 0/0 = NaN if x = 0.',
  ]);
  return svg(W, H, o);
}

function rmsnormComposable() {
  const W = 540, H = 256;
  let o = '';
  o += card(14, 12, 512, 32, 'green', ['Registering the Gain as a Parameter Guarantees Full Composability']);

  o += card(14, 56, 158, 118, 'purple', [
    '1. Register Gain g',
    { t: 'gain = Parameter(ones(d))', mono: true, fs: 11 },
    'registered as trainable',
    'Initialized to 1 ∈ ℝᵈ',
  ], { white: true, rail: true });
  o += arrow(176, 115, 190, 115, C.blue);
  o += card(194, 56, 156, 118, 'blue', [
    '2. Optimizer Discovers',
    { t: 'net = Seq(RMSNorm, Lin)', mono: true, fs: 11 },
    '1 SGD step updates',
    { t: 'gain ≠ [1, 1, …, 1]!', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ], { white: true, rail: true });
  o += arrow(354, 115, 368, 115, C.green);
  o += card(372, 56, 154, 118, 'green', [
    '3. Checkpoint Clone',
    'Load saved state into clone',
    'Restores trained gain;',
    { t: 'clone(X) == net(X)!', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 188, 512, 54, 'slate', [
    'Llama 3 8B Footprint: 65 RMSNorm layers × 4,096 floats = 266,240 parameters (0.0033% of 8B).',
  ]);
  return svg(W, H, o);
}

const customLayerChecks = () => threeColDiagram(
  'blue', 'Three-Point Verification Protocol for Any Custom Layer',
  { theme: 'orange', head: '1. Non-Trivial Weights', lines: ['Never test at g = 1!', { t: 'g = linspace(0.5, 1.5)', mono: true, fill: D.orange.dark }, 'Catches forgotten', '⊙ g multiplication!'] },
  { theme: 'blue',   head: '2. Forward & Grad Parity', lines: ['Compare vs. native or', 'hand-calculated values:', { t: 'max |y − y_ref| < 10⁻⁶', mono: true, fill: D.blue.dark }, '& verify ∂L/∂g ≠ 0'] },
  { theme: 'green',  head: '3. State Round-Trip',    lines: ['Save the state, load', 'into fresh clone:', { t: 'clone(X) == net(X)', mono: true, fill: D.green.dark }, 'Catches missing buffers!'] },
  'purple', 'Run all three checks whenever a custom layer reimplements or extends an op.'
);

function randomFourierFeatures() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 512, 46, 'blue', [
    'Random Fourier Features (Rahimi & Recht, 2007): Fixed Random Buffers',
    { t: 'Z = √(2/D) · cos(XW + b),  fixed W ~ N(0, 1)^{d×D},  b ~ U(0, 2π)^D', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);

  o += card(14, 72, 152, 108, 'slate', [
    '1D Input X',
    { t: 'shape (n, 1)', mono: true, fs: 13 },
    'x ∈ [−2, 2]',
  ], { white: true });
  o += arrow(170, 126, 188, 126, C.blue);
  o += card(192, 72, 158, 108, 'blue', [
    'RandomFeatures(1, 500)',
    { t: 'Buffers: W, b (frozen!)', mono: true, fs: 11.5, fw: 700, fill: D.blue.dark },
    '500 cosine basis waves',
    'Output Z : (n, 500)',
  ], { white: true, rail: true });
  o += arrow(354, 126, 372, 126, C.green);
  o += card(376, 72, 150, 108, 'green', [
    'Linear(500 → 1)',
    { t: 'Params: w_head, b_head', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    'Fits sin(3x) to',
    { t: 'MSE < 0.01', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 194, 512, 52, 'purple', [
    'Why W and b Must Be Registered Buffers: Optimizer must NOT train them, yet checkpoints',
    'MUST save them so the trained linear head receives the exact same 500 cosine basis waves!',
  ]);
  return svg(W, H, o);
}

function missingBufferDisaster() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 248, 168, 'red', [
    'Bug: Save Only Trainable Head',
    { t: 'Reload head into fresh RandomFeatures', fs: 12, fill: D.red.dark },
    'Fresh model draws new random W\' ≠ W',
    'and b\' ≠ b; trained head weights now',
    'multiply completely wrong features:',
    { t: 'MSE >100× worse than full state', fs: 12, fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  o += card(278, 12, 248, 168, 'green', [
    'Fix: Save Full State (Params + Buffers)',
    { t: 'W and b registered as buffers', fs: 12, fill: D.green.dark },
    'The saved state holds both the',
    'frozen random basis (W, b) and',
    'the trained linear head:',
    { t: 'Trained fit restored exactly', fs: 12, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 192, 512, 52, 'orange', [
    'Diagnostic Habit: Always test checkpoint round-trips by loading into a brand-new model',
    'built with a different random seed!',
  ]);
  return svg(W, H, o);
}

function quantizationZeroGrad() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 512, 32, 'red', ['Why Standard Autograd Freezes Quantization-Aware Training']);

  o += card(14, 56, 248, 128, 'orange', [
    'Forward Pass: y = round(w)',
    'Rounds continuous fp32 weights w',
    'to nearest integer Grid {-7..+7}',
    'for fast 4-bit integer matmuls:',
    { t: 'Piecewise-constant staircase!', fw: 700, fill: D.orange.dark },
  ], { white: true, rail: true });

  o += card(278, 56, 248, 128, 'red', [
    'True Derivative: ∂round(w)/∂w',
    'Flat on every open interval',
    '(k − 0.5, k + 0.5) and undefined',
    'at half-integer jumps:',
    { t: '∇_w L = (∇_y L) · 0 = 0 everywhere!', mono: true, fs: 12, fw: 700, fill: D.red.dark },
  ], { white: true, rail: true });

  o += card(14, 196, 512, 50, 'purple', [
    'To train weights behind round(), sign(), or argmax(), we must replace the zero backward',
    'derivative with a smooth surrogate Jacobian (the Straight-Through Estimator, ∂y/∂w ← 1).',
  ]);
  return svg(W, H, o);
}

function stopGradientTrick() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 10, 512, 48, 'purple', [
    'The Stop-Gradient One-Liner for Identity Surrogates',
    { t: 'y  =  x  +  stop_gradient( round(x) − x )', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);

  o += card(14, 70, 248, 118, 'blue', [
    '1. Forward Value Evaluation',
    { t: 'stop_gradient(u) returns u unchanged:', fs: 12 },
    { t: 'y = x + (round(x) − x)', mono: true, fs: 13, fill: D.blue.dark },
    { t: '  = round(x)   (exact quantized value!)', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
  ], { white: true, rail: true });

  o += card(278, 70, 248, 118, 'green', [
    '2. Backward Gradient Evaluation',
    { t: 'stop_gradient(...) has derivative 0:', fs: 12 },
    { t: '∂y/∂x = ∂x/∂x + 0', mono: true, fs: 13, fill: D.green.dark },
    { t: '      = 1   (exact STE identity pass!)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 200, 512, 50, 'slate', [
    'The same identity works for any discrete forward op f(x): x + stop_gradient(f(x) − x)',
    'evaluates to f(x) forward and passes the gradient through unchanged backward.',
  ]);
  return svg(W, H, o);
}

const customLayersSummary = () => fiveRowDiagram([
  { theme: 'blue',   k: 'Stateless Module',     v: 'CenteredLayer: Y − Y.mean(); subclasses Module, 0 params, 0 buffers' },
  { theme: 'purple', k: 'Parameterized Layer',  v: 'RMSNorm: (x / RMS(x)) ⊙ g; registers gain g as trainable Parameter' },
  { theme: 'green',  k: 'Fixed-Buffer Layer',   v: 'RandomFeatures: √(2/D)cos(XW+b); registers W, b as non-trainable buffers' },
  { theme: 'orange', k: 'Custom VJP / STE',     v: 'RoundSTE: forward evaluates round(x), backward passes upstream grad 1×' },
  { theme: 'teal',   k: 'Stop-Gradient Trick',  v: 'x + stop_gradient(f(x) − x) implements the STE in one line' },
]);

// ═══════════════════════════════════════════════════════════════════════════
// §5.6 SAVING-LOADING.MD DIAGRAMS (10 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const ioThreeStakes = () => threeColDiagram(
  'red', 'Why Serialization & Checkpointing Govern Production Survival',
  { theme: 'orange', head: '1. Hardware Crashes', lines: ['Llama 3 405B run:', '16,384 GPUs × 54 days', { t: '419 interruptions!', mono: true, fw: 700, fill: D.orange.dark }, '(~1 crash every 3 hrs)'] },
  { theme: 'blue',   head: '2. Stateful Adam',    lines: ['Weights alone fail:', 'restarting Adam with', { t: 'm = 0, v = 0', mono: true, fw: 700, fill: D.blue.dark }, 'spikes loss >100×!'] },
  { theme: 'red',    head: '3. Pickle Security',  lines: ['Untrusted .pt / .pkl', 'files execute arbitrary', { t: '__reduce__ bytecode', mono: true, fw: 700, fill: D.red.dark }, 'on load (RCE hazard!)'] },
  'green', 'Production rule: Use safetensors for weight sharing and 5-part atomic checkpoints for training.'
);

function stateNotCode() {
  const W = 540, H = 256;
  let o = '';
  o += card(14, 12, 512, 32, 'blue', ['What Gets Saved to Disk: The State Dictionary, Never the Python Class']);

  o += card(14, 56, 232, 122, 'purple', [
    'Live Model in Python Memory',
    '1. Python Code: forward(X), ReLU',
    '2. State Tree: parameters + buffers',
    { t: 'hidden: weight, bias', mono: true, fs: 11.5, fill: D.purple.dark },
    { t: 'output: weight, bias', mono: true, fs: 11.5, fill: D.purple.dark },
  ], { white: true, rail: true });

  o += hArrow(250, 290, 117, C.green, 'save');

  o += card(294, 56, 232, 122, 'green', [
    'Saved File on Disk',
    'Stores ONLY path → tensor bytes:',
    { t: '{ "hidden…weight": fp32 bytes, … }', mono: true, fs: 11.5, fill: D.green.dark },
    'Contains ZERO Python code and',
    'zero record of ReLU or wiring!',
  ], { white: true, rail: true });

  o += card(14, 192, 512, 52, 'orange', [
    'Reconstruction Contract: To load weights, you must first instantiate matching Python',
    'architecture code with identical attribute names (renaming self.output → self.head breaks keys!).',
  ]);
  return svg(W, H, o);
}

function pickleRceVsSafe() {
  const W = 540, H = 260;
  let o = '';
  o += card(14, 12, 248, 174, 'red', [
    'Python Pickle Files',
    'Stack-based virtual machine!',
    'Executes arbitrary callables in:',
    { t: '__reduce__ → (os.system, ("sh",))', mono: true, fs: 11.5, fw: 700, fill: D.red.dark },
    '• RCE the instant load() runs',
    '• Even restricted unpicklers',
    '  still parse VM opcodes',
  ], { white: true, rail: true });

  o += card(278, 12, 248, 174, 'green', [
    'safetensors (.safetensors)',
    'Strictly declarative data format:',
    { t: '[8B len] + [JSON] + [raw bytes]', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    '• Cannot represent code/callables',
    '• Validates bounds before reading',
    '• Supports instant zero-copy mmap',
    '  straight from disk to RAM/GPU!',
  ], { white: true, rail: true });

  o += card(14, 198, 512, 50, 'purple', [
    'Security Rule: Never enable unrestricted unpickling for files from the internet.',
  ]);
  return svg(W, H, o);
}

function safetensorsAudit() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 512, 32, 'blue', ['Auditing mlp.safetensors (7,946 float32 Params = 31,784 Data Bytes)']);

  o += card(14, 56, 124, 118, 'purple', [
    'Bytes 0..8',
    { t: 'struct "<Q"', mono: true, fs: 12.5, fw: 700 },
    '64-bit uint:',
    { t: 'n ≈ 300 B', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
  ], { white: true });
  o += arrow(142, 115, 158, 115, C.blue);
  o += card(162, 56, 188, 118, 'blue', [
    'Bytes 8..8+n (JSON Header)',
    { t: '"…hidden…bias": {', mono: true, fs: 11.5 },
    { t: '  "dtype": "F32",', mono: true, fs: 11.5 },
    { t: '  "shape": [256],', mono: true, fs: 11.5 },
    { t: ' "data_offsets":[0,1024]}', mono: true, fs: 11, fill: D.blue.dark },
  ], { white: true });
  o += arrow(354, 115, 370, 115, C.green);
  o += card(374, 56, 152, 118, 'green', [
    'Bytes 8+n .. end',
    { t: '31,784 Raw Bytes', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    '4 contiguous fp32',
    'buffers (zero-copy',
    'mmap-ready!)',
  ], { white: true });

  o += card(14, 188, 512, 56, 'slate', [
    'Random Access: every tensor\'s [begin, end] byte offset sits in the short header,',
    'so a loader can mmap one layer out of a 100 GB file without reading the rest!',
  ]);
  return svg(W, H, o);
}

function freshAdamShock() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 512, 32, 'red', ['Why Resuming With a Fresh Adam Spikes the Loss by Orders of Magnitude']);

  o += card(14, 54, 248, 128, 'red', [
    'Fresh Adam on Step 201 (m₀ = v₀ = 0)',
    'Step-1 bias correction (1 − β¹) cancels:',
    { t: 'm̂₁ = g,    v̂₁ = g ⊙ g', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Δw_i = −η · g_i / (|g_i| + ε) ≈ ±η!', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    'Every weight jumps by full ±0.05!',
  ], { white: true, rail: true });

  o += card(278, 54, 248, 128, 'green', [
    'Restored Adam State (m₂₀₀, v₂₀₀)',
    'Preserves 200 steps of curvature',
    'history v₂₀₀ and momentum m₂₀₀:',
    { t: 'Δw_i = −η · m̂₂₀₁ / (√v̂₂₀₁ + ε)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    'Takes smooth, small step in basin!',
  ], { white: true, rail: true });

  o += card(14, 194, 512, 54, 'orange', [
    'Euclidean Shock Magnitude: Across p = 1,409 parameters at η = 0.05, a fresh Adam step',
    'moves the weight vector by ‖Δw‖₂ ≈ 0.05 × √1409 ≈ 1.88—throwing the model out of its minimum!',
  ]);
  return svg(W, H, o);
}

function checkpoint96gbBudget() {
  const W = 540, H = 260;
  let o = '';
  o += card(14, 12, 512, 32, 'purple', ['Disk Footprint for an 8B-Parameter Model: Inference vs. Training Checkpoint']);

  o += card(14, 56, 200, 126, 'blue', [
    'Inference Weights File',
    { t: 'model.safetensors (bf16)', mono: true, fs: 11.5 },
    { t: '2 B/param = 16 GB', mono: true, fs: 16, fw: 700, fill: D.blue.dark },
    'Stores weights w only',
    '(no m, v, or gradients)',
  ], { white: true, rail: true });

  o += card(228, 56, 298, 126, 'orange', [
    'Resumable Training Checkpoint (fp32)',
    { t: 'w (32 GB) + Adam m (32 GB) + Adam v (32 GB)', mono: true, fs: 11.5 },
    { t: '12 B/param = 96 GB  (6× larger!)', mono: true, fs: 15, fw: 700, fill: D.orange.dark },
    'Gradients g (32 GB in RAM) are NOT saved',
    'to disk—they are recomputed on next step',
  ], { white: true, rail: true });

  o += card(14, 194, 512, 52, 'slate', [
    'Why 12 B/param on Disk vs. 16 B/param in RAM: Gradients g exist only between backward()',
    'and step(), so checkpoints save w + m + v (12 B/param) plus tiny RNG & step metadata.',
  ]);
  return svg(W, H, o);
}

function atomicWriteReplace() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 512, 32, 'red', ['Never Overwrite a Checkpoint File Directly: Use Atomic os.replace']);

  o += card(14, 56, 248, 118, 'red', [
    'Unsafe: Direct Write to ckpt.pt',
    { t: 'save(state, "ckpt.pt")', mono: true, fs: 12.5, fill: D.red.dark },
    'Immediately truncates existing file;',
    'if job crashes mid-write (96 GB takes',
    'seconds), BOTH old & new are lost!',
  ], { white: true, rail: true });

  o += card(278, 56, 248, 118, 'green', [
    'Crash-Safe: Write .tmp + os.replace',
    { t: '1. save(state, "ckpt.pt.tmp")', mono: true, fs: 12, fill: D.green.dark },
    { t: '2. os.replace(tmp, "ckpt.pt")', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    'POSIX atomic directory rename:',
    'ckpt.pt is ALWAYS complete!',
  ], { white: true, rail: true });

  o += card(14, 188, 512, 56, 'blue', [
    'Five Mandatory Compartments Inside ckpt.pt:',
    { t: '1. Model State (w + buffers)   2. Optimizer State (m, v)\n3. PRNG States   4. Step t   5. Config', mono: true, fs: 11.8, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

const bitwiseResumeTest = () => threeColDiagram(
  'green', 'Resuming at Step 200 → Step 205: What Each Compartment Controls',
  { theme: 'green',  head: 'Full 5-Part Resume', lines: ['Restores w, Adam m/v,', 'PRNG state, & step 200:', { t: 'max |w − w_ref| = 0.0', mono: true, fw: 700, fill: D.green.dark }, 'Exact bitwise match!'] },
  { theme: 'orange', head: 'Omit PRNG State',     lines: ['Restores w & Adam m/v,', 'but resets RNG seed:', { t: 'Replays step-1 batches!', fw: 700, fill: D.orange.dark }, 'Trajectory drifts (~0.01)'] },
  { theme: 'red',    head: 'Omit Adam State',     lines: ['Restores w & RNG,', 'but fresh m=0, v=0:', { t: 'Loss ~1000× from step 202!', mono: true, fw: 700, fill: D.red.dark }, 'Destroys convergence'] },
  'slate', 'Bit-for-bit resumption proves no hidden state was left outside your checkpoint dictionary.'
);

function transferKeyAudit() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 512, 32, 'purple', ['Auditing Missing & Unexpected Keys on Pretrained Transfer Loads']);

  o += card(14, 56, 248, 124, 'blue', [
    '1. Expected Head Diff (OK)',
    'Replace 1000-way ImageNet head',
    'with a new 10-way head & load:',
    { t: 'missing: head weight + bias', mono: true, fs: 11.5, fill: D.blue.dark },
    { t: 'unexpected: [] (filtered out)', mono: true, fs: 11.5, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(278, 56, 248, 124, 'red', [
    '2. Silent Backbone Gap (BUG!)',
    'If the last stage was renamed or',
    'dropped under a non-strict load:',
    { t: 'missing: last stage (50–60 keys)', mono: true, fs: 11.5, fw: 700, fill: D.red.dark },
    'Whole stage stays random with no error!',
  ], { white: true, rail: true });

  o += card(14, 194, 512, 54, 'green', [
    'Mandatory Assertion After Any Non-Strict Load:',
    { t: 'assert set(missing) == head_keys and not unexpected', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

const ioSummary = () => fiveRowDiagram([
  { theme: 'blue',   k: 'State Dict Only',     v: 'Serializes path → tensor bytes; requires matching Python code to rebuild' },
  { theme: 'green',  k: 'safetensors Format',  v: '8-byte length + JSON header + aligned raw bytes; zero RCE, zero-copy mmap' },
  { theme: 'purple', k: '5-Part Checkpoint',   v: 'Model state + Optimizer m,v + PRNG states + step t + config (12 B/param)' },
  { theme: 'orange', k: 'Atomic Commit',       v: 'Write to path + ".tmp" then os.replace() so crashes never corrupt ckpt' },
  { theme: 'red',    k: 'Strict Key Audit',    v: 'Always assert the missing and unexpected keys match only replaced layers' },
]);

// ═══════════════════════════════════════════════════════════════════════════
// §5.7 GPUS-DEVICES-MEMORY.MD DIAGRAMS (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

const gpuThreeConstraints = () => threeColDiagram(
  'blue', 'Three Hardware Laws Governing GPU Deep Learning Systems',
  { theme: 'green',  head: '1. Compute Speed',   lines: ['4096 × 4096 matmul:', 'CPU fp32: 0.1–2 s', { t: 'GPU TensorCore: < 2 ms', mono: true, fw: 700, fill: D.green.dark }, '(~100–1000× speedup!)'] },
  { theme: 'red',    head: '2. PCIe Bottleneck', lines: ['Copying 64 MiB matrix', 'CPU RAM → GPU HBM', { t: 'takes 3–5× LONGER', mono: true, fw: 700, fill: D.red.dark }, 'than the 4096³ matmul!'] },
  { theme: 'purple', head: '3. Async & HBM Cap', lines: ['Python queues kernels', 'without waiting; HBM', 'is fixed (24 / 80 GB)', { t: 'vs 128 GB fp32 Adam', mono: true, fill: D.purple.dark }] },
  'slate', 'Golden Rule: Move the model once, copy each batch once at the boundary, and never sync per step.'
);

function deviceMismatchGuard() {
  const W = 540, H = 256;
  let o = '';
  o += card(14, 12, 512, 32, 'red', ['Rejecting Cross-Device Operations Instead of Auto-Copying']);

  o += card(14, 56, 168, 114, 'blue', [
    'Tensor X',
    { t: 'device = GPU 0', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    'Lives in GPU 0 HBM',
  ], { white: true });

  o += card(196, 80, 148, 66, 'red', [
    { t: 'X + Y  →  error!', mono: true, fs: 12.5, fw: 700 },
    'Refuses implicit PCIe copy',
  ]);

  o += card(358, 56, 168, 114, 'orange', [
    'Tensor Y',
    { t: 'device = CPU (or GPU 1)', mono: true, fs: 12, fw: 700, fill: D.orange.dark },
    'Lives in Host RAM',
  ], { white: true });

  o += card(14, 184, 512, 58, 'green', [
    'Why Failing Fast Saves Throughput: A single silent CPU↔GPU copy inside an inner',
    'loop costs milliseconds over PCIe, destroying GPU utilization without any warning.',
  ]);
  return svg(W, H, o);
}

function modelOptimizerPlacement() {
  const W = 540, H = 254;
  let o = '';
  o += card(14, 12, 512, 32, 'blue', ['Mandatory Device Initialization Order for Models, Optimizers, and Batches']);

  const steps = [
    { x: 14,  theme: 'blue',   t1: '1. Move Model First',  t2: 'model → GPU',          t3: 'All params & buffers on GPU' },
    { x: 188, theme: 'purple', t1: '2. Build Optimizer',   t2: 'Adam on GPU params',   t3: 'Allocates m, v on GPU HBM' },
    { x: 362, theme: 'green',  t1: '3. Move Each Batch',   t2: 'X, y → GPU',           t3: 'Co-locates inputs with net' },
  ];
  steps.forEach((s, i) => {
    o += card(s.x, 56, 164, 118, s.theme, [
      s.t1,
      { t: s.t2, fs: 12.5, fw: 700, fill: PAL[s.theme].dark },
      { t: s.t3, fs: 12, fill: C.ink },
    ], { white: true, rail: true });
    if (i < 2) o += arrow(s.x + 166, 115, steps[i + 1].x - 2, 115, C.blue);
  });

  o += card(14, 188, 512, 52, 'red', [
    'Inside forward(X): Never allocate new tensors on the default device;',
    "create temporaries on X's device (e.g. zeros_like(X)) so they match the input!",
  ]);
  return svg(W, H, o);
}

function benchmarkFourRules() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 512, 32, 'purple', ['Four Mandatory Rules for Benchmarking Asynchronous GPU Kernels']);

  const rules = [
    { x: 14,  theme: 'orange', t1: '1. Warmup',     t2: 'Run 3 untimed steps', t3: 'absorbs compile & init' },
    { x: 146, theme: 'blue',   t1: '2. Pre-Sync',   t2: 'device sync',         t3: 'drain prior queue' },
    { x: 278, theme: 'purple', t1: '3. Timed Loop', t2: 't0 = time()',        t3: 'enqueue N kernel runs' },
    { x: 410, theme: 'green',  t1: '4. Post-Sync!', t2: 'sync → t1',           t3: 'wait for GPU finish!' },
  ];
  rules.forEach((r, i) => {
    o += card(r.x, 56, 116, 116, r.theme, [
      { t: r.t1, fs: 13.5, fw: 700 },
      { t: r.t2, mono: true, fs: 11.2, fw: 700, fill: PAL[r.theme].dark },
      { t: r.t3, fs: 11.8, fill: C.ink },
    ], { white: true });
    if (i < 3) o += arrow(r.x + 118, 114, rules[i + 1].x - 2, 114, C.purple);
  });

  o += card(14, 186, 512, 58, 'red', [
    'Without Rule 4 (a final device sync): Your timer stops as soon as',
    'Python finishes pushing commands into the queue (~0.02 ms), underreporting GPU time by 100×!',
  ]);
  return svg(W, H, o);
}

function stepMemoryTimeline() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 10, 512, 32, 'blue', ['GPU Memory Lifecycle Across Step 1 (3×4096 MLP, 4N = 144 MiB, B = 4096)']);

  const stages = [
    { x: 14,  theme: 'blue',   t1: '1. Model Init',   m: '144 MiB (4N)',  sub: 'Weights w only' },
    { x: 146, theme: 'orange', t1: '2. Forward Peak', m: '336 MiB',       sub: '+192 MiB acts (3×64)' },
    { x: 278, theme: 'red',    t1: '3. Backward End', m: '288 MiB (8N)',  sub: 'Acts freed; +144 MiB g' },
    { x: 410, theme: 'purple', t1: '4. Adam Step 1',  m: '576 MiB (16N)', sub: '+288 MiB moments m,v' },
  ];
  stages.forEach((s, i) => {
    o += card(s.x, 54, 116, 122, s.theme, [
      { t: s.t1, fs: 13, fw: 700 },
      { t: s.m, mono: true, fs: 13.5, fw: 700, fill: PAL[s.theme].dark },
      { t: s.sub, fs: 11.5, fill: C.ink },
    ], { white: true });
    if (i < 3) o += arrow(s.x + 118, 115, stages[i + 1].x - 2, 115, C.blue);
  });

  o += card(14, 190, 512, 60, 'green', [
    'Step 2+ Peak Memory = Static 16N (576 MiB) + Forward Activations (192 MiB) = 768 MiB:',
    'Peak always occurs at the end of forward() right before backward() starts freeing layer activations!',
  ]);
  return svg(W, H, o);
}

const eightBGpuFitMatrix = () => fiveRowDiagram([
  { theme: 'green',  k: 'bf16 Inference',      v: '2 B/param = 16 GB static  →  24 GB GPU: YES (8 GB KV) | 80 GB: YES' },
  { theme: 'blue',   k: 'fp32 Inference',      v: '4 B/param = 32 GB static  →  24 GB GPU: NO (OOM)      | 80 GB: YES' },
  { theme: 'purple', k: 'Head Fine-Tuning',    v: '32 GB w + 6.3 GB head Adam = 38.3 GB  →  24 GB: NO     | 80 GB: YES' },
  { theme: 'red',    k: 'Full fp32 Adam',      v: '16 B/param = 128 GB static →  24 GB: NO | 80 GB: NO (needs ≥2 GPUs)' },
  { theme: 'orange', k: 'Activation OOM Fix',  v: 'Activations scale as O(K·B·T·d) → halve micro-batch B or use √K ckpt' },
]);

function sqrtKSegments() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 10, 512, 34, 'purple', ['Activation Checkpointing on K = 16 Blocks With Segment Size s = √16 = 4']);

  o += card(14, 54, 248, 126, 'red', [
    'Standard Backprop (No Checkpointing)',
    'Caches all K = 16 block inputs',
    'simultaneously during forward():',
    { t: 'Peak = 16 activation tensors', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Memory grows linearly in K', fs: 12.5, fill: D.red.dark },
  ], { white: true, rail: true });

  o += card(278, 54, 248, 126, 'green', [
    'Segmented Checkpointing (s = 4)',
    'Caches K/s = 4 boundaries and',
    'recomputes one 4-block segment:',
    { t: 'Peak = K/s + s = 4 + 4 = 8 tensors!', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Half the stored tensors of K = 16', fs: 12.5, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 192, 512, 56, 'blue', [
    'Optimal Segment Size: peak stored tensors K/s + s is smallest at s = √K (2√K tensors),',
    'at the cost of one extra forward pass per segment (roughly 20–30% extra step time).',
  ]);
  return svg(W, H, o);
}

const gpuSummary = () => fiveRowDiagram([
  { theme: 'blue',   k: 'Boundary Transfers',  v: 'Model to GPU before building the optimizer; copy each batch once per step' },
  { theme: 'orange', k: 'Async Command Queue', v: 'Avoid per-step scalar readouts, prints and Python ifs on tensors' },
  { theme: 'purple', k: 'Synced Benchmarking', v: '3 warmup steps + device sync before t0 + device sync before t1' },
  { theme: 'teal',   k: '3 Memory Views',      v: 'Live allocated ⊆ Caching allocator pool (reserved) ⊆ CUDA driver (nvidia-smi)' },
  { theme: 'green',  k: '√K Checkpointing',    v: 'Save every s = √K boundary & recompute interior: O(K) → O(√K) act RAM' },
]);

// ═══════════════════════════════════════════════════════════════════════════
// §5.5 NUMERICS.MD DIAGRAMS (10 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function sixteenBitTwoHazards() {
  const W = 540, H = 260;
  let o = '';
  o += card(14, 12, 512, 32, 'red', ['Why Naive 16-Bit Training Fails: Range Overflow vs. Precision Rounding']);

  o += card(14, 56, 248, 134, 'orange', [
    'Hazard 1: fp16 Range Overflow',
    { t: 'fp16(300.0) ** 2  →  inf!', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    '5-bit exponent caps max finite',
    'value at 65,504 (and underflows',
    'gradients < 6×10⁻⁵ to subnormal/0)',
  ], { white: true, rail: true });

  o += card(278, 56, 248, 134, 'purple', [
    'Hazard 2: bf16 Update Swamping',
    { t: 'bf16(1.0) + bf16(1e-3)  →  1.0!', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    '8-bit exponent keeps 300² = 90,112',
    'finite, but 7-bit mantissa has gap',
    '2⁻⁷ ≈ 0.0078 at 1.0—freezing weights!',
  ], { white: true, rail: true });

  o += card(14, 202, 512, 46, 'green', [
    'Mixed-Precision Solution: Compute matmuls in 16 bits, but accumulate weight updates in fp32!',
  ]);
  return svg(W, H, o);
}

const rangeVsPrecision = () => threeColDiagram(
  'blue', 'IEEE Floating-Point Bit Allocation: x = (−1)^s · 2^e · (1 + f)',
  { theme: 'blue',   head: 'float32 (32 bits)',  lines: ['1 sign + 8 exp + 23 mant', { t: 'max: 3.40 × 10³⁸', mono: true }, { t: 'tiny: 1.18 × 10⁻³⁸', mono: true }, { t: 'eps: 1.19 × 10⁻⁷', mono: true, fw: 700, fill: D.blue.dark }] },
  { theme: 'green',  head: 'bfloat16 (16 bits)', lines: ['1 sign + 8 exp + 7 mant',  { t: 'max: 3.39 × 10³⁸ (!)', mono: true, fill: D.green.dark }, { t: 'tiny: 1.18 × 10⁻³⁸', mono: true }, { t: 'eps: 7.81 × 10⁻³', mono: true, fw: 700, fill: D.orange.dark }] },
  { theme: 'orange', head: 'float16 (16 bits)',  lines: ['1 sign + 5 exp + 10 mant', { t: 'max: 6.55 × 10⁴ (!)', mono: true, fill: D.red.dark }, { t: 'tiny: 6.10 × 10⁻⁵', mono: true }, { t: 'eps: 9.77 × 10⁻⁴', mono: true, fw: 700, fill: D.blue.dark }] },
  'purple', 'Exponent bits govern dynamic range (overflow/underflow); mantissa bits govern relative spacing eps = 2⁻ᵗ.'
);

function updateSwamping() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 512, 32, 'red', ['The Parameter Update Swamping Trap on the bfloat16 Number Line']);

  // Visual number line around 1.0000 in bf16
  o += box(14, 54, 512, 114, '#FFFFFF', C.purple, 8, 1.8);
  o += tx(270, 72, 'bfloat16 representable grid near w = 1.0  (spacing eps = 2⁻⁷ = 0.0078125)', { fs: 13, fw: 700, fill: C.ink });

  // Dead zone highlight [1.0, 1.0 + 0.0039]
  o += fillRect(80, 96, 170, 28, D.red.tint, 4);
  o += `<line x1="60" y1="110" x2="480" y2="110" stroke="${C.ink}" stroke-width="2"/>`;
  // Ticks at x=80 (1.0000000), x=250 (midpoint +0.003906), x=420 (1.0078125)
  o += `<line x1="80" y1="94" x2="80" y2="126" stroke="${C.blue}" stroke-width="3"/>`;
  o += `<line x1="250" y1="98" x2="250" y2="122" stroke="${D.red.base}" stroke-width="2" stroke-dasharray="3 3"/>`;
  o += `<line x1="420" y1="94" x2="420" y2="126" stroke="${C.green}" stroke-width="3"/>`;

  o += tx(80, 144, '1.0000000', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(250, 144, '+½eps = 0.0039 (round cutoff)', { mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  o += tx(430, 144, '1.0078125', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });

  o += card(14, 180, 512, 70, 'red', [
    'Any update below half the local bf16 spacing (η·g < 0.0039 upward from w = 1.0) rounds back to w!',
    { t: 'In Fashion-MNIST MLP (η = 0.01), ~90% of pure-bf16 updates vanish—fp32 master weights fix this completely.', fs: 12.2, fill: C.ink },
  ], { white: true, rail: true });
  return svg(W, H, o);
}

function dtypePromotion(fw) {
  const W = 540, H = 258;
  let o = '';
  if (fw === 'tensorflow') {
    o += card(14, 12, 248, 168, 'orange', [
      'Strict Dtype Policy',
      'Refuses implicit cross-dtype',
      'promotion between tensors:',
      { t: 'x16 + y32 → InvalidArgumentError', mono: true, fs: 11.2, fw: 700, fill: D.red.dark },
      'Requires an explicit',
      { t: 'tf.cast(x, dtype)', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark },
    ], { white: true, rail: true });
    o += card(278, 12, 248, 168, 'green', [
      'Python Literals Adapt',
      'A literal constant takes on',
      "the tensor's dtype:",
      { t: 'x16 + 1.0  →  fp16', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
      'Only tensor-tensor mixes',
      'must be cast by hand',
    ], { white: true, rail: true });
    o += card(14, 192, 512, 54, 'red', [
      'Upside of Strictness: a stray fp32 bias or buffer inside a 16-bit forward pass fails loudly',
      'instead of silently promoting every downstream activation back to fp32 (2× RAM).',
    ]);
    return svg(W, H, o);
  }
  o += card(14, 12, 248, 168, 'blue', [
    'Mixed Float Widths Promote',
    'Mixed operands promote to the',
    'smallest format covering both:',
    { t: 'fp16 + fp32  →  fp32', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'bf16 + fp32  →  fp32', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'fp16 + bf16  →  fp32 (needs both!)', mono: true, fs: 12, fw: 700, fill: D.purple.dark },
  ], { white: true, rail: true });
  o += card(278, 12, 248, 168, 'green', [
    'Scalars and Ints Do Not Widen',
    'A Python float or an int tensor',
    "keeps the fp16 tensor's dtype:",
    { t: 'x16 + 1.0  →  fp16', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'x16 + int  →  fp16', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    'Only float-float mixes promote',
  ], { white: true, rail: true });
  o += card(14, 192, 512, 54, 'red', [
    'Stray fp32 Buffer Contamination Hazard: Leaving a single bias or buffer in fp32 inside a',
    '16-bit forward pass silently promotes every downstream activation tensor back to fp32 (2× RAM)!',
  ]);
  return svg(W, H, o);
}

function modelCastHalving() {
  const W = 540, H = 252;
  let o = '';
  o += card(14, 12, 512, 32, 'blue', ['Whole-Model Dtype Casting on Fashion-MNIST MLP (203,530 Parameters)']);

  o += card(14, 56, 226, 118, 'blue', [
    'Default fp32 Model',
    { t: '203,530 × 4 Bytes', mono: true, fs: 13 },
    { t: '= 814,120 Bytes (0.78 MiB)', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    '8B scale: 32.0 GB weights',
  ], { white: true, rail: true });

  o += hArrow(244, 296, 115, C.green, 'cast');

  o += card(300, 56, 226, 118, 'green', [
    'Cast 16-Bit Model',
    { t: '203,530 × 2 Bytes', mono: true, fs: 13 },
    { t: '= 407,060 Bytes (0.39 MiB)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    '8B scale: 16.0 GB weights (2×!)',
  ], { white: true, rail: true });

  o += card(14, 188, 512, 52, 'purple', [
    'Inference vs. Training Rule: Cast the whole model to 16 bits for pure inference;',
    'for training, keep weights in fp32 and let autocast / mixed-precision policy cast on the fly.',
  ]);
  return svg(W, H, o);
}

const ampMemoryAudit = () => threeColDiagram(
  'purple', 'Where Mixed Precision Saves Memory on an 8B Model (and Where It Does Not)',
  { theme: 'blue',   head: 'Pure fp32 Training',      lines: ['fp32 weights: 32 GB', 'fp32 grads:   32 GB', 'fp32 Adam m,v: 64 GB', { t: 'Static = 128 GB + 1× Acts', mono: true, fw: 700, fill: D.blue.dark }] },
  { theme: 'purple', head: 'Mixed Precision (AMP)',   lines: ['fp32 master w: 32 GB', 'fp32 grads:   32 GB', 'fp32 Adam m,v: 64 GB', { t: 'Static = 128 GB + ½× Acts', mono: true, fw: 700, fill: D.orange.dark }] },
  { theme: 'green',  head: 'Pure bf16 Inference',     lines: ['bf16 weights: 16 GB', 'No grads:      0 GB', 'No Adam m,v:   0 GB', { t: 'Static = 16 GB (fits 24G!)', mono: true, fw: 700, fill: D.green.dark }] },
  'slate', 'AMP keeps the 128 GB static parameter floor, but halves activation memory and roughly doubles matmul speed.'
);

function fp16LossScaling() {
  const W = 540, H = 264;
  let o = '';
  o += card(14, 10, 512, 32, 'orange', ['Dynamic Loss Scaling Rescues fp16 Gradients From Underflow (< 6.1×10⁻⁵)']);

  const steps = [
    { x: 14,  theme: 'red',    t1: '1. Scale Loss',     t2: 'L_scaled = S · L',       t3: 'S = 2¹⁶ = 65,536' },
    { x: 146, theme: 'orange', t1: '2. Backward (fp16)', t2: '∇(S·L) = S · ∇L',        t3: 'lifts ~32% subnormals!' },
    { x: 278, theme: 'blue',   t1: '3. Unscale (fp32)', t2: 'g_fp32 = (S·∇L) / S',    t3: 'check for inf / NaN' },
    { x: 410, theme: 'green',  t1: '4. Step or Backoff', t2: 'finite → Adam.step()',   t3: 'inf → skip & S ← S/2' },
  ];
  steps.forEach((s, i) => {
    o += card(s.x, 54, 116, 118, s.theme, [
      { t: s.t1, fs: 13, fw: 700 },
      { t: s.t2, mono: true, fs: 11.2, fw: 700, fill: PAL[s.theme].dark },
      { t: s.t3, fs: 11.5, fill: C.ink },
    ], { white: true });
    if (i < 3) o += arrow(s.x + 118, 113, steps[i + 1].x - 2, 113, C.amber);
  });

  o += card(14, 186, 512, 64, 'purple', [
    'Why Dynamic Backoff Must Skip optimizer.step() Completely When an inf Appears:',
    'Allowing a single inf gradient into Adam turns m and v into permanent NaNs across all future steps!',
  ]);
  return svg(W, H, o);
}

const precisionSelector = () => threeColDiagram(
  'green', 'Choosing the Right Accelerator Precision Mode for Training',
  { theme: 'green',  head: '1. bf16 Mixed Precision', lines: ['Ampere / Hopper / TPU:', 'fp32 master + bf16 ops', { t: 'No GradScaler needed!', fw: 700, fill: D.green.dark }, '(8-bit exp = fp32 range)'] },
  { theme: 'orange', head: '2. fp16 + GradScaler',    lines: ['Older Volta / Turing:', 'fp32 master + fp16 ops', { t: 'Dynamic loss scaling S', fw: 700, fill: D.orange.dark }, 'prevents 5-bit underflow'] },
  { theme: 'blue',   head: '3. TF32 TensorCore Math', lines: ['Zero-code fp32 speedup:', 'stores 32-bit tensors,', 'rounds matmul inputs', { t: 'to 10-bit mantissa', mono: true, fill: D.blue.dark }] },
  'slate', 'Default recommendation on modern hardware: bfloat16 autocast with fp32 master weights & reductions.'
);

function logsumexpAndSums() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 12, 248, 238, 'red', [
    'Hazard A: Log-Sum-Exp Overflow',
    { t: 'x = [12, 11, 10] in fp16 (max 65,504)', mono: true, fs: 11.5 },
    { t: 'Naive: exp(12) ≈ 162,755 → inf!', mono: true, fs: 12, fw: 700, fill: D.red.dark },
    { t: 'Fix: Shift by m = max_{i} x_i = 12', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'm + log Σ exp(x_i − m)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    'Exponents are ≤ 0, so exp(x_i−m) ∈ (0,1]',
    'and never overflows in any format!',
  ], { white: true, rail: true });

  o += card(278, 12, 248, 238, 'orange', [
    'Hazard B: 16-Bit Running Sum Stall',
    { t: 'Add 2⁻¹⁰ ≈ 0.000977 for 4,096 steps', mono: true, fs: 11.5 },
    { t: 'fp16 accumulator freezes at 2.000!', mono: true, fs: 12, fw: 700, fill: D.red.dark },
    'Because fp16 spacing on [2, 4) is 2⁻⁹,',
    'every +2⁻¹⁰ increment rounds to +0!',
    { t: 'Fix: Accumulate sums/means in fp32', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'fp32 accumulator → 4.0 (exact)', mono: true, fs: 12.5, fill: D.green.dark },
  ], { white: true, rail: true });
  return svg(W, H, o);
}

const numericsSummary = () => fiveRowDiagram([
  { theme: 'blue',   k: 'float32 (1+8+23)',    v: 'eps ≈ 1.2×10⁻⁷, max 3.4×10³⁸ → Master weights, Adam m/v, reductions' },
  { theme: 'green',  k: 'bfloat16 (1+8+7)',    v: 'eps ≈ 7.8×10⁻³, max 3.4×10³⁸ → Preferred 16-bit compute (no loss scaling)' },
  { theme: 'orange', k: 'float16 (1+5+10)',    v: 'eps ≈ 9.8×10⁻⁴, max 65,504 → Requires dynamic loss scaling S·L' },
  { theme: 'purple', k: 'AMP Training Loop',   v: 'Cast fp32 master w → 16b for matmuls/acts; apply grads to fp32 master w' },
  { theme: 'red',    k: 'Stable Reductions',   v: 'Subtract max m in logsumexp; always accumulate running sums in fp32' },
]);

// ═══════════════════════════════════════════════════════════════════════════
// §5.8 REPRODUCIBILITY-INSPECTION.MD DIAGRAMS (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

function reproVsDebug() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 248, 174, 'blue', [
    '1. Statistical Reproducibility',
    'Is a +0.3% accuracy gain real?',
    { t: '5 seeds on Fashion-MNIST:', mono: true, fs: 12.5, fill: D.blue.dark },
    { t: 'span 1.0–4.0 percentage points!', fw: 700, fill: D.red.dark },
    'Requires isolated PRNG streams &',
    'reporting mean ± std across N ≥ 5 seeds',
  ], { white: true, rail: true });

  o += card(278, 12, 248, 174, 'purple', [
    '2. Bitwise Debugging & Inspection',
    'Why did loss turn NaN at step 40,000?',
    { t: 'Deterministic replay + Layer hooks:', mono: true, fs: 12, fill: D.purple.dark },
    '• Replay exact step-by-step trajectory',
    '• Profile layer activation/grad stds',
    '• Trap the exact first NaN submodule',
  ], { white: true, rail: true });

  o += card(14, 198, 512, 48, 'slate', [
    'Controlling randomness makes bugs repeatable; execution hooks expose internal tensors without editing model code.',
  ]);
  return svg(W, H, o);
}

const fiveRngSources = () => fiveRowDiagram([
  { theme: 'blue',   k: '1. Weight Init (W₀)',   v: 'Controls starting basin; unseeded runs diverge from step 0' },
  { theme: 'purple', k: '2. Stochastic Layers',  v: 'Dropout binary masks ξ_j; forward(X) differs on identical inputs' },
  { theme: 'green',  k: '3. DataLoader Shuffle', v: 'Epoch minibatch permutation; changes gradient update trajectory' },
  { theme: 'orange', k: '4. Data Augmentation',  v: 'Random crops, flips, jitter inside Dataset.__getitem__' },
  { theme: 'red',    k: '5. Forked Workers!',    v: 'OS fork clones identical NumPy RNG state across workers unless re-seeded!' },
]);

function privateGenerators() {
  const W = 540, H = 260;
  let o = '';
  o += card(14, 12, 512, 32, 'red', ['Why Global PRNG State Is Fragile vs. Private Generator / Key Isolation']);

  o += card(14, 54, 248, 134, 'red', [
    'Fragile: Shared Global PRNG',
    { t: 'One global seed → one shared stream', fs: 12, fw: 700, fill: D.red.dark },
    'Adding 1 extra randn() call in logging',
    'or changing num_batches shifts the',
    'entire downstream random sequence for',
    'every subsequent layer and epoch!',
  ], { white: true, rail: true });

  o += card(278, 54, 248, 134, 'green', [
    'Robust: Isolated Stream Objects',
    { t: 'Private generator or split key', fs: 12, fw: 700, fill: D.green.dark },
    '• one stream for shuffling',
    '• one stream for model init',
    '• one stream per data worker',
    'Each advances independently!',
  ], { white: true, rail: true });

  o += card(14, 200, 512, 48, 'blue', [
    'Full-Stack Seeding Checklist: framework PRNG + NumPy + Python random + every data worker.',
  ]);
  return svg(W, H, o);
}

function floatNonassociative() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 512, 32, 'orange', ['Floating-Point Addition Is Not Associative: (a + b) + c ≠ a + (b + c)']);

  o += card(14, 56, 248, 118, 'green', [
    'Grouping 1: (10⁸ + (−10⁸)) + 1.0',
    { t: 'Step 1: 10⁸ + (−10⁸) = 0.0  (exact)', mono: true, fs: 12 },
    { t: 'Step 2: 0.0 + 1.0    = 1.0!', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    'Large opposite terms cancel first,',
    'preserving the small +1.0 increment',
  ], { white: true, rail: true });

  o += card(278, 56, 248, 118, 'red', [
    'Grouping 2: 10⁸ + ((−10⁸) + 1.0)',
    { t: 'At 10⁸, float32 grid spacing is 8.0!', mono: true, fs: 11.5, fill: D.red.dark },
    { t: 'Step 1: −10⁸ + 1.0 rounds to −10⁸!', mono: true, fs: 12 },
    { t: 'Step 2: 10⁸ + (−10⁸) = 0.0!', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    'The +1.0 is completely swamped!',
  ], { white: true, rail: true });

  o += card(14, 188, 512, 58, 'purple', [
    'Why Seeded GPU Runs Still Diverge: Parallel GPU threads accumulate partial sums via',
    'non-deterministic atomicAdd arrival orders unless deterministic kernels are enabled.',
  ]);
  return svg(W, H, o);
}

function determinismTiers() {
  const W = 540, H = 258;
  let o = '';
  o += card(14, 12, 248, 174, 'purple', [
    'Tier 1: Bitwise Determinism',
    { t: 'Deterministic kernels switched on', fs: 12, fw: 700, fill: D.purple.dark },
    '• Same machine & GPU architecture',
    '• Same CUDA / framework version',
    '• Same batch/world size',
    { t: 'Role: Local bug & NaN bisection', fw: 700, fill: D.purple.dark },
  ], { white: true, rail: true });

  o += card(278, 12, 248, 174, 'green', [
    'Tier 2: Scientific Reproducibility',
    { t: 'N ≥ 5 independent random seeds', mono: true, fs: 11.5, fill: D.green.dark },
    '• Survives A100 vs. H100 vs. TPU',
    '• Survives compiler & library updates',
    '• Reports mean ± standard error',
    { t: 'Role: Validating real ML gains!', fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });

  o += card(14, 198, 512, 48, 'slate', [
    'Never confuse the two: Bitwise equality is for debugging code; multi-seed statistics are for evaluating models.',
  ]);
  return svg(W, H, o);
}

function xavierRes15x() {
  const W = 540, H = 262;
  let o = '';
  o += card(14, 10, 512, 32, 'blue', ['Why Each 64→64 Xavier Residual Block Multiplies Variance by 1.5×']);

  const steps = [
    { x: 14,  theme: 'slate',  t1: 'Block Input x_k', t2: 'width d = 64',  t3: 'Var = v_k' },
    { x: 146, theme: 'blue',   t1: 'Xavier 64→64',    t2: 'τ² = 2/(64+64)', t3: 'Var = 1.0 v_k' },
    { x: 278, theme: 'purple', t1: 'ReLU Gate',       t2: 'halves 2nd mom', t3: 'Var = 0.5 v_k' },
    { x: 410, theme: 'blue',   t1: 'Xavier 64→64',    t2: 'τ² = 1/64',      t3: 'Var(body) = 0.5 v_k' },
  ];
  steps.forEach((s, i) => {
    o += card(s.x, 54, 116, 86, s.theme, [
      { t: s.t1, fs: 12.5, fw: 700 },
      { t: s.t2, mono: true, fs: 11.5 },
      { t: s.t3, mono: true, fs: 12.5, fw: 700, fill: PAL[s.theme].dark },
    ]);
    if (i < 3) o += arrow(s.x + 118, 97, steps[i + 1].x - 2, 97, C.blue);
  });

  o += card(14, 154, 512, 94, 'green', [
    'Combine Skip + Body:  Var[h_k] = Var[x_k] + Var[body(x_k)] = v_k + 0.5 v_k = 1.5 v_k',
    { t: 'Per-block std multiplier = √1.5 ≈ 1.225×   ⇒   After 8 blocks: 0.69 × (1.5)⁴ ≈ 3.5 (hooks: 3.2–4.5)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
  ], { white: true, rail: true });
  return svg(W, H, o);
}

function hookMemoryLeak() {
  const W = 540, H = 260;
  let o = '';
  o += card(14, 12, 248, 174, 'red', [
    'Bug: Storing Live Hook Tensors',
    { t: 'def hook(m, inp, out):', mono: true, fs: 11.5 },
    { t: '    history.append(out)  # LEAK!', mono: true, fs: 11.5, fw: 700, fill: D.red.dark },
    'out still points into the ENTIRE',
    'recorded backward graph;',
    'retains all step activations → OOM!',
  ], { white: true, rail: true });

  o += card(278, 12, 248, 174, 'green', [
    'Fix: Detach + Remove Handle',
    { t: 'def hook(m, inp, out):', mono: true, fs: 11.5 },
    { t: '    s = out.detach().std()', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    { t: '    history.append(float(s))', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    'Remove the hook handle when done:',
    'severs the graph & unregisters!',
  ], { white: true, rail: true });

  o += card(14, 198, 512, 48, 'purple', [
    'Automated NaN Trap: Check isfinite(out).all() inside a forward hook to catch the exact offending layer!',
  ]);
  return svg(W, H, o);
}

const reproSummary = () => fiveRowDiagram([
  { theme: 'blue',   k: 'Full-Stack Seeding',  v: 'Seed the framework, NumPy, Python random and every data worker' },
  { theme: 'green',  k: 'Statistical Claims',  v: 'Single-seed deltas (<1–4%) are noise; report mean ± std over N ≥ 5 seeds' },
  { theme: 'purple', k: 'Deterministic Ops',   v: 'Deterministic kernels fix the parallel GPU atomicAdd order' },
  { theme: 'orange', k: 'Forward Hooks',       v: 'Profile activation stds & trap first NaN layer without editing model code' },
  { theme: 'red',    k: 'Leak-Free Cleanup',   v: 'Always detach tensors saved in hooks and remove the hook handle after use' },
]);

export const diagrams = {
  // §5.0 index.md
  'ch05-bg-four-bottlenecks':         fourBottlenecks,
  'ch05-bg-module-tree-hub':          moduleTreeHub,
  'ch05-bg-systems-roadmap':          systemsRoadmap,
  'ch05-bg-8b-benchmarks':            eightBBenchmarks,
  // §5.1 model-construction.md
  'ch05-bg-scale-bookkeeping':        scaleBookkeeping,
  'ch05-bg-setattr-registry':         setattrRegistry,
  'ch05-bg-tree-walk-ops':            treeWalkOps,
  'ch05-bg-sequential-loop':          sequentialLoop,
  'ch05-bg-plain-list-trap-pytorch':    () => plainListTrap('pytorch'),
  'ch05-bg-plain-list-trap-tensorflow': () => plainListTrap('tensorflow'),
  'ch05-bg-plain-list-trap-jax':        () => plainListTrap('jax'),
  'ch05-bg-plain-list-trap-mxnet':      () => plainListTrap('mxnet'),
  'ch05-bg-eager-vs-trace':           eagerVsTrace,
  'ch05-bg-lazy-dry-run':             lazyDryRun,
  'ch05-bg-lazy-dry-run-jax':         lazyDryRunJax,
  'ch05-bg-config-builder':           configBuilder,
  // §5.2 parameters-state-memory.md
  'ch05-bg-adam-16b-floor':           adam16bFloor,
  'ch05-bg-state-tree-paths':         stateTreePaths,
  'ch05-bg-four-tensor-kinds':        fourTensorKinds,
  'ch05-bg-standardizer-buffer':      standardizerBuffer,
  'ch05-bg-lazy-adam-moments':        () => lazyAdamMoments('lazy'),
  'ch05-bg-lazy-adam-moments-jax':    () => lazyAdamMoments('jax'),
  'ch05-bg-meta-shape-trace':         metaShapeTrace,
  'ch05-bg-tied-dedup-grad':          tiedDedupGrad,
  'ch05-bg-freezing-pitfalls':        freezingPitfalls,
  'ch05-bg-state-summary':            stateSummary,
  // §5.3 init.md
  'ch05-bg-res-explosion':            resExplosion,
  'ch05-bg-default-init-decay':       defaultInitDecay,
  'ch05-bg-tree-apply-init':          treeApplyInit,
  'ch05-bg-truncated-normal':         truncatedNormal,
  'ch05-bg-weight-shape-conventions': () => weightShapeConventions('out-in'),
  'ch05-bg-weight-shape-conventions-in-out': () => weightShapeConventions('in-out'),
  'ch05-bg-res-variance-derivation':  resVarianceDerivation,
  'ch05-bg-three-gain-schedules':     threeGainSchedules,
  'ch05-bg-gpt2-sqrt-n-bound':        gpt2SqrtNBound,
  'ch05-bg-zero-out-proj-wakeup':     zeroOutProjWakeup,
  'ch05-bg-init-summary':             initSummary,
  // §5.4 custom-layers.md
  'ch05-bg-custom-layer-needs':       customLayerNeeds,
  'ch05-bg-stateless-centered':       statelessCentered,
  'ch05-bg-rmsnorm-formula':          rmsnormFormula,
  'ch05-bg-rmsnorm-composable':       rmsnormComposable,
  'ch05-bg-custom-layer-checks':      customLayerChecks,
  'ch05-bg-random-fourier-features':  randomFourierFeatures,
  'ch05-bg-missing-buffer-disaster':  missingBufferDisaster,
  'ch05-bg-quantization-zero-grad':   quantizationZeroGrad,
  'ch05-bg-stop-gradient-trick':      stopGradientTrick,
  'ch05-bg-custom-layers-summary':    customLayersSummary,
  // §5.6 saving-loading.md
  'ch05-bg-io-three-stakes':          ioThreeStakes,
  'ch05-bg-state-not-code':           stateNotCode,
  'ch05-bg-pickle-rce-vs-safe':       pickleRceVsSafe,
  'ch05-bg-safetensors-audit':        safetensorsAudit,
  'ch05-bg-fresh-adam-shock':         freshAdamShock,
  'ch05-bg-checkpoint-96gb-budget':   checkpoint96gbBudget,
  'ch05-bg-atomic-write-replace':     atomicWriteReplace,
  'ch05-bg-bitwise-resume-test':      bitwiseResumeTest,
  'ch05-bg-transfer-key-audit':       transferKeyAudit,
  'ch05-bg-io-summary':               ioSummary,
  // §5.7 gpus-devices-memory.md
  'ch05-bg-gpu-three-constraints':    gpuThreeConstraints,
  'ch05-bg-device-mismatch-guard':    deviceMismatchGuard,
  'ch05-bg-model-optimizer-placement': modelOptimizerPlacement,
  'ch05-bg-benchmark-four-rules':     benchmarkFourRules,
  'ch05-bg-step-memory-timeline':     stepMemoryTimeline,
  'ch05-bg-8b-gpu-fit-matrix':        eightBGpuFitMatrix,
  'ch05-bg-sqrt-k-segments':          sqrtKSegments,
  'ch05-bg-gpu-summary':              gpuSummary,
  // §5.5 numerics.md
  'ch05-bg-16bit-two-hazards':        sixteenBitTwoHazards,
  'ch05-bg-range-vs-precision':       rangeVsPrecision,
  'ch05-bg-update-swamping':          updateSwamping,
  'ch05-bg-dtype-promotion':          () => dtypePromotion('promote'),
  'ch05-bg-dtype-promotion-tensorflow': () => dtypePromotion('tensorflow'),
  'ch05-bg-model-cast-halving':       modelCastHalving,
  'ch05-bg-amp-memory-audit':         ampMemoryAudit,
  'ch05-bg-fp16-loss-scaling':        fp16LossScaling,
  'ch05-bg-precision-selector':       precisionSelector,
  'ch05-bg-logsumexp-and-sums':       logsumexpAndSums,
  'ch05-bg-numerics-summary':         numericsSummary,
  // §5.8 reproducibility-inspection.md
  'ch05-bg-repro-vs-debug':           reproVsDebug,
  'ch05-bg-five-rng-sources':         fiveRngSources,
  'ch05-bg-private-generators':       privateGenerators,
  'ch05-bg-float-nonassociative':     floatNonassociative,
  'ch05-bg-determinism-tiers':        determinismTiers,
  'ch05-bg-xavier-res-15x':           xavierRes15x,
  'ch05-bg-hook-memory-leak':         hookMemoryLeak,
  'ch05-bg-repro-summary':            reproSummary,
};
