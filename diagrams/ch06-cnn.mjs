// Slide diagrams for chapter_convolutional-neural-networks (Chapter 6: CNNs)
import { C, TOKENS, FS, FM, grid, tx, baselineY, arrow, block, chip, rc, svg } from './engine.mjs';
import { sec, num } from './sections.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;
const BLUE = [C.lblue, C.blue, false, C.ink];
const GREEN = [C.lgreen, C.green, false, C.ink];
const AMBER = [C.lamber, C.amber, false, C.ink];
const PURPLE = [D.purple.tint, C.purple, false, D.purple.dark];
const RED = [D.red.tint, D.red.base, false, D.red.dark];
const GRAY = [N.panel, N.hairline, false, C.muted];

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// Width-safe text: realistic glyph widths (Source Sans 3 ≈ 0.50 em, a little
// more in bold; mono ≈ 0.60 em), a modest shrink (never below FIT_MIN_FS),
// and a two-line wrap at the space nearest the middle when the text still
// does not fit. fitLines() returns { fs, lines }.
const FIT_MIN_FS = 11;
const fitCharW = (o) => (o.mono ? 0.60 : (o.fw >= 700 ? 0.52 : 0.50));
function fitLines(t, maxW, o = {}) {
  const str = String(t);
  const fs0 = o.fs || 13.5;
  const avail = maxW - 24;
  const cw = fitCharW(o);
  const fsFor = (s) => avail / (Math.max(1, s.length) * cw);
  const floor = Math.max(FIT_MIN_FS, Math.min(fs0, 0.85 * fs0));
  if (fsFor(str) >= floor) return { fs: +Math.min(fs0, fsFor(str)).toFixed(2), lines: [str] };
  // two lines: split at the space that minimizes the longer half
  // Prefer a break after punctuation or before a separator (·, →, +, …)
  // when it costs at most ~20% more width than the most balanced break.
  const cands = [];
  for (let i = 0; i < str.length; i++) {
    if (str[i] !== ' ') continue;
    const a = str.slice(0, i).trimEnd(), b = str.slice(i + 1).trimStart();
    if (!a || !b) continue;
    const nice = /[,;:·)]$/.test(a) || /^[·→⇒(]/.test(b);
    const bad = /=$/.test(a) || /^=/.test(b);
    cands.push({ a, b, m: Math.max(a.length, b.length), nice, bad });
  }
  let best = null;
  const mMin = Math.min(...cands.map((c) => c.m));
  for (const c of cands) {
    const score = c.m - (c.nice && c.m <= 1.2 * mMin ? 0.2 * mMin : 0) + (c.bad ? 0.3 * mMin : 0);
    if (!best || score < best.score) best = { ...c, score };
  }
  if (!best) return { fs: +Math.max(FIT_MIN_FS, Math.min(fs0, fsFor(str))).toFixed(2), lines: [str] };
  const fs = Math.max(FIT_MIN_FS, Math.min(fs0, avail / (best.m * cw)));
  return { fs: +fs.toFixed(2), lines: [best.a, best.b] };
}

function fitTx(cx, cy, t, maxW, o = {}) {
  const { fs, lines } = fitLines(t, maxW, o);
  if (lines.length === 1) return tx(cx, cy, lines[0], { ...o, fs });
  const d = 0.6 * fs;
  return tx(cx, cy - d, lines[0], { ...o, fs }) + tx(cx, cy + d, lines[1], { ...o, fs });
}

function card(x, y, w, h, fill, stroke, lines) {
  let o = box(x, y, w, h, fill, stroke, 8, 2);
  const opts = (l) => ({ fs: l.fs || 13.5, fw: l.fw || 600, mono: !!l.mono, fill: l.fill || C.ink });
  const fitted = lines.map((l) => ({ ...fitLines(l.t, w, opts(l)), o: opts(l) }));
  if (fitted.every((f) => f.lines.length === 1)) {
    const step = h / (lines.length + 1);
    fitted.forEach((f, idx) => { o += tx(x + w / 2, y + step * (idx + 1), f.lines[0], { ...f.o, fs: f.fs }); });
    return o;
  }
  // A line wrapped: stack all rows at 1.2 × their size, spread the slack evenly.
  const rows = fitted.flatMap((f) => f.lines.map((s) => ({ s, fs: f.fs, o: f.o })));
  const used = rows.reduce((a, r) => a + 1.2 * r.fs, 0);
  const gap = Math.max(0, (h - used) / (rows.length + 1));
  let cy = y + gap;
  rows.forEach((r) => { cy += 0.6 * r.fs; o += tx(x + w / 2, cy, r.s, { ...r.o, fs: r.fs }); cy += 0.6 * r.fs + gap; });
  return o;
}

// ── index.md diagrams (4) ─────────────────────────────────────────────────

function flattenVsGrid() {
  const W = 540, H = 270;
  let o = '';
  // Left panel: Flattened MLP
  o += box(16, 14, 244, 242, D.red.tint, D.red.base, 10, 2);
  o += tx(138, 34, 'MLP: Flatten 2D → 1D Vector', { fs: 14.5, fw: 700, fill: D.red.dark });
  o += tx(138, 54, 'Destroys 2D spatial neighbors', { fs: 12.5, fw: 600, fill: C.muted });
  o += grid([[1, 1, 0, 1, 0, 0, 1]], 29, 74, 27, 4, { fs: 12.5, fill: () => RED });
  o += tx(138, 120, 'All-to-all dense weights', { fs: 12.5, fw: 600, fill: D.red.dark });
  o += card(30, 138, 216, 52, '#FFFFFF', D.red.base, [
    { t: '1MP RGB → 36 TB weights', fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Permutation blind (~87%)', fs: 12, fw: 500, fill: C.muted },
  ]);
  o += tx(138, 224, '9 × 10¹² free parameters', { mono: true, fs: 13, fw: 700, fill: D.red.dark });

  // Right panel: 2D ConvNet
  o += box(280, 14, 244, 242, C.lgreen, C.green, 10, 2);
  o += tx(402, 34, 'CNN: 2D Shared Local Kernel', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(402, 54, 'Preserves locality & shift symmetry', { fs: 12.5, fw: 600, fill: C.muted });
  const G = [[1, 1, 0], [1, 0, 0], [0, 1, 1]];
  o += grid(G, 304, 72, 26, 3, {
    fs: 12,
    fill: (r, c) => (r < 2 && c < 2) ? AMBER : BLUE,
  });
  o += arrow(394, 114, 424, 114, C.green);
  o += grid([['K']], 432, 96, 36, 3, { fs: 14, fill: () => GREEN });
  o += card(294, 166, 216, 48, '#FFFFFF', C.green, [
    { t: 'Shared k×k filter bank', fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Independent of H × W!', fs: 12, fw: 600, fill: C.ink },
  ]);
  o += tx(402, 234, 'Hundreds of parameters', { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function roadmapPipeline() {
  const W = 540, H = 276;
  let o = '';
  const stages = [
    { x: 16,  y: 18, t: `${sec('sec_why-conv')} Symmetry`, s1: 'Equivariance + Locality', s2: 'V_{i,j,a,b} = K_{a,b}', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, y: 18, t: `${sec('sec_conv_layer')} Conv Layer`, s1: '2D Cross-Corr & im2col', s2: 'Y = X ★ K, RF r = 2L+1', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 372, y: 18, t: `${sec('sec_padding')} Pad & Stride`, s1: 'Borders, Stride, Dilation', s2: '⌊(n+2p−d(k−1)−1)/s⌋+1', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 372, y: 158, t: `${sec('sec_channels')} Channels`, s1: '4D Kernels & 1×1 Conv', s2: 'c_o c_i k_h k_w + c_o', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 194, y: 158, t: `${sec('sec_pooling')} Pooling`, s1: 'Max / Avg & Global Pool', s2: '0 params, shift tolerance', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 16,  y: 158, t: `${sec('sec_lenet')} LeNet`, s1: 'Full CNN vs. MLP Test', s2: '61.7K params, 416K MACs', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  const bw = 152, bh = 96;
  o += arrow(16 + bw + 4, 66, 194 - 4, 66, C.blue);
  o += arrow(194 + bw + 4, 66, 372 - 4, 66, C.purple);
  o += arrow(372 + bw / 2, 18 + bh + 4, 372 + bw / 2, 158 - 4, C.amber);
  o += arrow(372 - 4, 206, 194 + bw + 4, 206, C.amber);
  o += arrow(194 - 4, 206, 16 + bw + 4, 206, C.green);
  stages.forEach((s) => {
    o += card(s.x, s.y, bw, bh, s.fill, s.st, [
      { t: s.t, fs: 14.5, fw: 700, fill: s.dk },
      { t: s.s1, fs: 12.5, fw: 600, fill: C.ink },
      { t: s.s2, mono: true, fs: 11.5, fw: 600, fill: C.muted },
    ]);
  });
  return svg(W, H, o);
}

function benchmarksOverview() {
  const W = 540, H = 276;
  let o = '';
  const rows = [
    { sec: sec('sec_why-conv'), label: '3×3 RGB→64ch Kernel', metric: '1,792 params at any H×W', col: C.blue, bg: C.lblue, dk: D.blue.dark },
    { sec: sec('sec_conv_layer'), label: 'Learned Edge [1, -1]', metric: 'Loss 12.0 → 0.004 in 10 steps', col: C.blue, bg: C.lblue, dk: D.blue.dark },
    { sec: sec('sec_padding'), label: 'Strided / Dilated 3×3×3', metric: 'Receptive field r = 7 → 15 px', col: C.purple, bg: D.purple.tint, dk: D.purple.dark },
    { sec: sec('sec_channels'), label: '1×1 Bottleneck (256→64→256)', metric: '589K → 69.6K weights (8.5× cut)', col: C.amber, bg: C.lamber, dk: D.orange.dark },
    { sec: sec('sec_pooling'), label: '2×2 Stride-2 MaxPool', metric: '50% 1-px shift absorbed, −∞ pad', col: C.amber, bg: C.lamber, dk: D.orange.dark },
    { sec: sec('sec_lenet'), label: 'LeNet on Fashion-MNIST', metric: '4% params do 86% of MACs', col: C.green, bg: C.lgreen, dk: D.green.dark },
  ];
  rows.forEach((r, i) => {
    const y = 14 + i * 42;
    o += card(16, y, 58, 34, r.bg, r.col, [{ t: r.sec, mono: true, fs: 13.5, fw: 700, fill: r.dk }]);
    o += card(84, y, 210, 34, '#FFFFFF', r.col, [{ t: r.label, fs: 13, fw: 700, fill: C.ink }]);
    o += card(304, y, 220, 34, r.bg, r.col, [{ t: r.metric, mono: true, fs: 12, fw: 600, fill: r.dk }]);
  });
  return svg(W, H, o);
}

function resolutionScaling() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 16, 236, 102, C.lblue, C.blue, [
    { t: 'Input: 224 × 224 RGB', fs: 15, fw: 700, fill: D.blue.dark },
    { t: '3×3 Conv (3 → 64 ch)', fs: 13, fw: 600, fill: C.ink },
    { t: 'Params: 1,792  ·  MACs: 86.7M', mono: true, fs: 12, fw: 600, fill: C.muted },
  ]);
  o += arrow(258, 67, 282, 67, C.purple);
  o += card(288, 16, 236, 102, C.lamber, C.amber, [
    { t: '2× Resolution: 448 × 448', fs: 15, fw: 700, fill: D.orange.dark },
    { t: 'Same 3×3 Conv (3 → 64 ch)', fs: 13, fw: 600, fill: C.ink },
    { t: 'Params: 1,792  ·  MACs: 346.8M', mono: true, fs: 12, fw: 600, fill: D.orange.dark },
  ]);
  o += card(16, 138, 246, 104, C.lgreen, C.green, [
    { t: 'Parameters: 1× (Unchanged)', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'c_o · c_i · k_h · k_w + c_o', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Independent of H and W', fs: 12.5, fw: 500, fill: C.muted },
  ]);
  o += card(278, 138, 246, 104, D.purple.tint, C.purple, [
    { t: 'MACs & Activations: 4×', fs: 14.5, fw: 700, fill: D.purple.dark },
    { t: 'Scales with output area H · W', fs: 13, fw: 600, fill: C.ink },
    { t: '4× spatial positions evaluated', fs: 12.5, fw: 500, fill: C.muted },
  ]);
  return svg(W, H, o);
}

// ── why-conv.md diagrams (8) ─────────────────────────────────────────────

function denseParamExplosion() {
  const W = 540, H = 264;
  let o = '';
  const bars = [
    { y: 20,  title: 'Dense 1MP → 1MP Grid', weights: '10¹² weights', mem: '4 TB (fp32)', w: 280, fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { y: 80,  title: 'Dense 1MP → 1K Hidden', weights: '10⁹ weights', mem: '4 GB (fp32)', w: 210, fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 140, title: 'Equivariant 1MP (Toeplitz)', weights: '4 × 10⁶ weights', mem: '16 MB (fp32)', w: 130, fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 200, title: 'Equivariant + Local (3×3)', weights: '9 weights', mem: '36 Bytes!', w: 36, fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  bars.forEach((b) => {
    o += card(16, b.y, 210, 44, '#FFFFFF', b.st, [{ t: b.title, fs: 13.5, fw: 700, fill: C.ink }]);
    o += box(238, b.y, b.w, 44, b.fill, b.st, 6, 2);
    if (b.w < 100) {
      o += tx(238 + b.w + 12, b.y + 22, `${b.weights} · ${b.mem}`, { mono: true, fs: 13, fw: 700, fill: b.dk, anchor: 'start' });
    } else {
      o += fitTx(238 + b.w / 2, b.y + 22, b.weights, b.w, { mono: true, fs: 12.5, fw: 700, fill: b.dk });
      if (b.w < 160) {
        o += tx(238 + b.w + 12, b.y + 22, b.mem, { mono: true, fs: 13, fw: 700, fill: b.dk, anchor: 'start' });
      }
    }
  });
  return svg(W, H, o);
}

function equivarianceVsInvariance() {
  const W = 540, H = 266;
  let o = '';
  // Top: Equivariance
  o += box(16, 14, 508, 112, C.lblue, C.blue, 10, 2);
  o += tx(270, 34, 'Translation Equivariance: f(T_v X) = T_v f(X)', { mono: true, fs: 14, fw: 700, fill: D.blue.dark });
  o += card(32, 52, 136, 58, '#FFFFFF', C.blue, [
    { t: 'Shifted Input T_v X', fs: 13, fw: 700, fill: C.ink },
    { t: 'Target at (i+v₁, j+v₂)', mono: true, fs: 11.5, fill: C.muted },
  ]);
  o += arrow(176, 81, 214, 81, C.blue);
  o += card(220, 52, 112, 58, D.purple.tint, C.purple, [
    { t: 'Conv Filter f', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Shared Kernel K', fs: 11.5, fill: C.ink },
  ]);
  o += arrow(340, 81, 378, 81, C.blue);
  o += card(384, 52, 124, 58, '#FFFFFF', C.blue, [
    { t: 'Output T_v f(X)', fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'Peak shifts by v!', fs: 12, fw: 700, fill: D.blue.dark },
  ]);

  // Bottom: Invariance
  o += box(16, 140, 508, 112, C.lgreen, C.green, 10, 2);
  o += tx(270, 160, 'Translation Invariance: g(T_v X) = g(X)', { mono: true, fs: 14, fw: 700, fill: D.green.dark });
  o += card(32, 178, 136, 58, '#FFFFFF', C.green, [
    { t: 'Shifted Map T_v H', fs: 13, fw: 700, fill: C.ink },
    { t: 'Peak anywhere in H×W', fs: 11.5, fill: C.muted },
  ]);
  o += arrow(176, 207, 214, 207, C.green);
  o += card(220, 178, 112, 58, C.lamber, C.amber, [
    { t: 'Global Pool g', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'max / mean(H,W)', mono: true, fs: 11.5, fill: C.ink },
  ]);
  o += arrow(340, 207, 378, 207, C.green);
  o += card(384, 178, 124, 58, '#FFFFFF', C.green, [
    { t: 'Class Score y', fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Unchanged by v!', fs: 12, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function offsetReindexing() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 20, 224, 96, C.lblue, C.blue, [
    { t: 'Absolute Coordinates (k, l)', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'W_{i, j, k, l} · X_{k, l}', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'Depends on 4 absolute indices', fs: 12.5, fill: C.muted },
  ]);
  o += arrow(248, 68, 292, 68, C.purple);
  o += tx(270, 48, 'k = i+a', { mono: true, fs: 12, fw: 700, fill: D.purple.dark });
  o += tx(270, 90, 'l = j+b', { mono: true, fs: 12, fw: 700, fill: D.purple.dark });
  o += card(300, 20, 224, 96, C.lgreen, C.green, [
    { t: 'Relative Offsets (a, b)', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'V_{i, j, a, b} · X_{i+a, j+b}', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: '(a, b) = offset from (i, j)', fs: 12.5, fill: C.muted },
  ]);
  o += card(16, 136, 508, 106, C.lamber, C.amber, [
    { t: 'Why Re-Index by Offset (a, b) = (k − i, l − j)?', fs: 15, fw: 700, fill: D.orange.dark },
    { t: '1. Equivariance drops output index (i, j):  V_{i,j,a,b} = K_{a,b}', mono: true, fs: 13, fw: 600, fill: C.ink },
    { t: '2. Locality bounds relative offset magnitude:  |a| ≤ Δ, |b| ≤ Δ', mono: true, fs: 13, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function infiniteGridShift() {
  const W = 540, H = 260;
  let o = '';
  // Left: original support on infinite grid
  o += box(16, 18, 226, 224, N.panel, C.gray, 8, 2, true);
  o += tx(129, 38, 'Infinite Zero-Padded Grid ℤ²', { fs: 13.5, fw: 700, fill: C.ink });
  const G1 = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]];
  o += grid(G1, 51, 56, 36, 4, {
    fs: 13.5,
    fill: (r, c, v) => v ? BLUE : GRAY,
  });
  o += tx(129, 224, 'Support at (i, j)', { mono: true, fs: 13, fw: 700, fill: D.blue.dark });

  o += arrow(250, 130, 290, 130, C.purple);
  o += tx(270, 108, 'T_v', { mono: true, fs: 15, fw: 700, fill: D.purple.dark });

  // Right: shifted support on infinite grid
  o += box(298, 18, 226, 224, N.panel, C.gray, 8, 2, true);
  o += tx(411, 38, 'Shifted by v = (+1, +1)', { fs: 13.5, fw: 700, fill: D.green.dark });
  const G2 = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]];
  o += grid(G2, 333, 56, 36, 4, {
    fs: 13.5,
    fill: (r, c, v) => v ? GREEN : GRAY,
  });
  o += tx(411, 224, 'No boundary clipping!', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function impulseProbing() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 16, 152, 98, C.lblue, C.blue, [
    { t: '1. Zero Input X = 0', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'T_v(0) = 0', mono: true, fs: 13, fw: 600, fill: C.ink },
    { t: '⇒ U_{i,j} = u', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);
  o += arrow(174, 65, 190, 65, C.purple);
  o += card(194, 16, 152, 98, C.lamber, C.amber, [
    { t: '2. Unit Impulse δ', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: '1 at (k, l), else 0', fs: 12.5, fw: 600, fill: C.ink },
    { t: 'Probes W_{i,j,k,l}', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark },
  ]);
  o += arrow(352, 65, 368, 65, C.purple);
  o += card(372, 16, 152, 98, C.lgreen, C.green, [
    { t: '3. Shift Impulse', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'v = (i, j)', mono: true, fs: 13, fw: 600, fill: C.ink },
    { t: '⇒ V_{i,j,a,b} = K_{a,b}', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 134, 508, 112, D.purple.tint, C.purple, [
    { t: 'Necessity Proof Summary (Unit Impulse Probing)', fs: 15, fw: 700, fill: D.purple.dark },
    { t: 'W_{i,j,i+a,j+b} = W_{0,0,a,b} ≡ K_{a,b}  for all locations (i, j)', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'Collapses 10¹² dense weights (4 TB) → 1999² ≈ 4×10⁶ shared offset weights', fs: 13, fw: 600, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function localityWindow() {
  const W = 540, H = 266;
  let o = '';
  const s = 34, g = 4, gx = 30, gy = 38;
  o += tx(125, 22, '5×5 Offsets Around (i, j)', { fs: 14, fw: 700, fill: C.ink });
  const M = [
    ['0', '0', '0', '0', '0'],
    ['0', 'K', 'K', 'K', '0'],
    ['0', 'K', 'K₀', 'K', '0'],
    ['0', 'K', 'K', 'K', '0'],
    ['0', '0', '0', '0', '0'],
  ];
  o += grid(M, gx, gy, s, g, {
    fs: 13,
    fill: (r, c) => (r === 2 && c === 2) ? AMBER : (r >= 1 && r <= 3 && c >= 1 && c <= 3) ? GREEN : GRAY,
  });
  o += tx(125, 244, 'Δ = 1 ⇒ 3×3 window (9 weights)', { mono: true, fs: 12.5, fw: 700, fill: D.green.dark });

  o += card(246, 36, 278, 92, C.lgreen, C.green, [
    { t: 'Active Window: |a|, |b| ≤ Δ', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '(2Δ + 1) × (2Δ + 1) nonzeros', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Δ=1 → 9 weights · Δ=5 → 121 weights', fs: 12.5, fill: C.muted },
  ]);
  o += card(246, 144, 278, 92, N.panel, C.gray, [
    { t: 'Pruned Exterior: |a| > Δ or |b| > Δ', fs: 14, fw: 700, fill: C.muted },
    { t: 'K_{a, b} = 0  (distant pixels ignored)', mono: true, fs: 13, fw: 600, fill: C.ink },
    { t: '4 × 10⁶ weights → 9 weights (36 B)', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function inductiveBias() {
  const W = 540, H = 264;
  let o = '';
  const cols = [
    { x: 16,  t: 'Layer 1 (Local)', s1: '3×3 Receptive Field', s2: 'Edges & Gradients', s3: 'Position-blind locally', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: 'Mid Layers', s1: 'Widening RF + ReLU', s2: 'Corners & Textures', s3: 'Combines local parts', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 372, t: 'Deep Layers', s1: 'Global Receptive Field', s2: 'Whole Objects', s3: 'Recovers layout context', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  cols.forEach((c) => {
    o += card(c.x, 18, 152, 124, c.fill, c.st, [
      { t: c.t, fs: 14.5, fw: 700, fill: c.dk },
      { t: c.s1, mono: true, fs: 12, fw: 700, fill: C.ink },
      { t: c.s2, fs: 13, fw: 600, fill: C.ink },
      { t: c.s3, fs: 11.5, fill: C.muted },
    ]);
  });
  o += arrow(172, 80, 190, 80, C.purple);
  o += arrow(350, 80, 368, 80, C.green);
  o += card(16, 160, 508, 88, C.lamber, C.amber, [
    { t: 'Inductive Bias Trade-Off', fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: 'Gain: Pools statistical evidence across all H×W pixels (huge sample efficiency)', fs: 12.5, fw: 600, fill: C.ink },
    { t: 'Cost: Single layer cannot tell eye (top) from nostril (middle) by (i, j) alone', fs: 12.5, fw: 600, fill: C.muted },
  ]);
  return svg(W, H, o);
}

function convVsXcorr() {
  const W = 540, H = 264;
  let o = '';
  // Left: Cross-correlation
  o += box(16, 16, 236, 232, C.lblue, C.blue, 10, 2);
  o += tx(134, 38, 'Cross-Correlation (X ★ K)', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(134, 60, 'Unflipped indices (i+a, j+b)', { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  o += grid([['a', 'b'], ['c', 'd']], 88, 82, 42, 6, { fs: 15, fill: () => BLUE });
  o += tx(134, 194, 'Framework conv layer default', { fs: 13, fw: 700, fill: D.blue.dark });
  o += tx(134, 218, 'Σ K_{a,b} X_{i+a, j+b}', { mono: true, fs: 12.5, fill: C.ink });

  o += tx(270, 132, '⇔', { fs: 26, fw: 700, fill: D.purple.dark });

  // Right: Mathematical convolution
  o += box(288, 16, 236, 232, C.lgreen, C.green, 10, 2);
  o += tx(406, 38, "Math Convolution (K' * X)", { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(406, 60, 'Flipped indices (i−a, j−b)', { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  o += grid([['d', 'c'], ['b', 'a']], 360, 82, 42, 6, { fs: 15, fill: () => GREEN });
  o += tx(406, 194, "K'_{a,b} = K_{-a,-b}", { mono: true, fs: 13, fw: 700, fill: D.green.dark });
  o += tx(406, 218, "Same learned function family!", { fs: 12.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function kernel4dTensor() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 28, 142, 114, C.lblue, C.blue, [
    { t: 'Input Tensor X', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '(c_i, H, W)', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'e.g. RGB c_i = 3', fs: 12.5, fill: C.muted },
  ]);
  o += tx(174, 85, '★', { fs: 22, fw: 700, fill: D.purple.dark });
  o += card(190, 28, 164, 114, D.purple.tint, C.purple, [
    { t: '4D Kernel Bank K', fs: 14, fw: 700, fill: D.purple.dark },
    { t: '(c_o, c_i, k_h, k_w)', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: '+ bias u ∈ ℝ^{c_o}', mono: true, fs: 12, fill: C.muted },
  ]);
  o += arrow(360, 85, 384, 85, C.green);
  o += card(390, 28, 134, 114, C.lgreen, C.green, [
    { t: 'Output Maps H', fs: 14, fw: 700, fill: D.green.dark },
    { t: '(c_o, H, W)', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'e.g. c_o = 64 maps', fs: 12.5, fill: C.muted },
  ]);
  o += card(16, 162, 508, 82, C.lamber, C.amber, [
    { t: 'Parameter Count:  c_o · c_i · k_h · k_w + c_o', mono: true, fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: 'RGB (3) → 64 channels with 3×3 kernel:  64 × 3 × 9 + 64 = 1,792 params (at any H×W!)', fs: 13, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function modalities1d2d3d() {
  const W = 540, H = 260;
  let o = '';
  const mods = [
    { x: 16,  t: '1D Convolution', ax: 'Time / Sequence (T)', sh: 'X: (c_i, T)', k: 'K: (c_o, c_i, k_t)', ex: 'Audio, ECG, Text', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: '2D Convolution', ax: 'Height × Width (H, W)', sh: 'X: (c_i, H, W)', k: 'K: (c_o, c_i, k_h, k_w)', ex: 'Photos, Medical X-Ray', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 372, t: '3D Convolution', ax: 'Depth × H × W (D,H,W)', sh: 'X: (c_i, D, H, W)', k: 'K: (c_o,c_i,k_d,k_h,k_w)', ex: 'Video, Volumetric MRI', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  mods.forEach((m) => {
    o += card(m.x, 16, 152, 154, m.fill, m.st, [
      { t: m.t, fs: 14.5, fw: 700, fill: m.dk },
      { t: m.ax, fs: 12, fw: 600, fill: C.ink },
      { t: m.sh, mono: true, fs: 12, fw: 700, fill: m.dk },
      { t: m.k, mono: true, fs: 11, fw: 600, fill: C.ink },
      { t: m.ex, fs: 12, fill: C.muted },
    ]);
  });
  o += card(16, 184, 508, 62, C.lamber, C.amber, [
    { t: 'Spectrogram Caveat: Time axis is shift-equivariant; frequency requires log-mel scale!', fs: 13, fw: 700, fill: D.orange.dark },
  ]);
  return svg(W, H, o);
}

// ── conv-layer.md diagrams (10) ──────────────────────────────────────────

function verticalEdgeMotiv() {
  const W = 540, H = 260;
  let o = '';
  o += tx(92, 22, '6×8 Stripe Image X', { fs: 14, fw: 700, fill: D.blue.dark });
  const X = [
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
  ];
  o += grid(X, 16, 38, 16, 3, {
    fs: 11.5,
    fill: (r, c, v) => v === 1 ? BLUE : GRAY,
  });
  o += tx(226, 95, '★', { fs: 22, fw: 700, fill: D.purple.dark });
  o += tx(276, 62, 'Kernel K', { fs: 13.5, fw: 700, fill: D.purple.dark });
  o += grid([[1, -1]], 246, 78, 28, 3, { fs: 12.5, fill: () => PURPLE });
  o += arrow(314, 94, 342, 94, C.green);

  o += tx(418, 22, 'Edge Map Y (6×7)', { fs: 14, fw: 700, fill: D.green.dark });
  const Y = [
    [0, 1, 0, 0, 0, -1, 0],
    [0, 1, 0, 0, 0, -1, 0],
    [0, 1, 0, 0, 0, -1, 0],
    [0, 1, 0, 0, 0, -1, 0],
    [0, 1, 0, 0, 0, -1, 0],
    [0, 1, 0, 0, 0, -1, 0],
  ];
  o += grid(Y, 352, 38, 16, 3, {
    fs: 11,
    fill: (r, c, v) => v === 1 ? GREEN : v === -1 ? AMBER : [N.panel, N.hairline, false, C.muted],
  });

  o += card(16, 162, 508, 86, C.lgreen, C.green, [
    { t: 'Horizontal Finite Difference:  Y_{i,j} = X_{i,j} − X_{i,j+1}', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    { t: '+1 at white→black edge (col 1)  ·  −1 at black→white edge (col 5)  ·  0 on flat regions', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function validWindowShrinkage() {
  const W = 540, H = 266;
  let o = '';
  const cards = [
    { x: 16,  y: 16, t: 'Top-Left (0,0)', eq: '0·0+1·1+3·2+4·3', val: '= 19', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 278, y: 16, t: 'Top-Right (0,1)', eq: '1·0+2·1+4·2+5·3', val: '= 25', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 16,  y: 96, t: 'Bot-Left (1,0)', eq: '3·0+4·1+6·2+7·3', val: '= 37', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 278, y: 96, t: 'Bot-Right (1,1)', eq: '4·0+5·1+7·2+8·3', val: '= 43', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  cards.forEach((c) => {
    o += card(c.x, c.y, 246, 68, c.fill, c.st, [
      { t: `${c.t}:  ${c.eq} ${c.val}`, mono: true, fs: 12.5, fw: 700, fill: c.dk },
    ]);
  });
  o += card(16, 178, 508, 74, C.lamber, C.amber, [
    { t: 'Valid Window Count Along Axis:  n − k + 1', mono: true, fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: '3×3 ★ 2×2 → 2×2   ·   6×8 ★ 1×2 → 6×7   ·   28×28 ★ 5×5 → 24×24', mono: true, fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function flippedKernelEquiv() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 18, 236, 134, C.lblue, C.blue, [
    { t: 'Cross-Correlation Kernel K', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '[0, 1;  2, 3]', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'Indexes X_{i+a, j+b}', mono: true, fs: 12.5, fill: C.muted },
  ]);
  o += card(288, 18, 236, 134, D.purple.tint, C.purple, [
    { t: "Flipped Conv Kernel K'", fs: 14, fw: 700, fill: D.purple.dark },
    { t: '[3, 2;  1, 0]', mono: true, fs: 14, fw: 700, fill: C.ink },
    { t: 'Indexes X_{i−a, j−b}', mono: true, fs: 12.5, fill: C.muted },
  ]);
  o += card(16, 168, 508, 72, C.lgreen, C.green, [
    { t: 'Identical Output on X:  [[19, 25], [37, 43]]', mono: true, fs: 14.5, fw: 700, fill: D.green.dark },
    { t: "Learning K under cross-correlation ≡ learning K' under mathematical convolution", fs: 13, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function resolutionIndependence() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 20, 146, 92, C.lblue, C.blue, [
    { t: 'Input 3 × 3', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '→ Output 2 × 2', mono: true, fs: 13, fw: 600, fill: C.ink },
  ]);
  o += card(16, 136, 146, 92, C.lblue, C.blue, [
    { t: 'Input 6 × 8', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '→ Output 5 × 7', mono: true, fs: 13, fw: 600, fill: C.ink },
  ]);
  o += arrow(168, 66, 206, 110, C.purple);
  o += arrow(168, 182, 206, 138, C.purple);
  o += card(212, 66, 152, 116, D.purple.tint, C.purple, [
    { t: 'Same 2×2 conv', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: '4 weights + 1 bias', fs: 13, fw: 700, fill: C.ink },
    { t: '= 5 parameters', fs: 13, fw: 700, fill: D.purple.dark },
  ]);
  o += arrow(370, 124, 394, 124, C.green);
  o += card(400, 36, 124, 176, C.lgreen, C.green, [
    { t: 'Works at', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'ANY H × W!', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '(Dense layer', fs: 12, fill: C.muted },
    { t: 'crashes if', fs: 12, fill: C.muted },
    { t: 'H×W changes)', fs: 12, fill: C.muted },
  ]);
  return svg(W, H, o);
}

function orientationSelectivity() {
  const W = 540, H = 264;
  let o = '';
  // Left: Vertical stripe X with [1, -1]
  o += box(16, 16, 242, 232, C.lgreen, C.green, 10, 2);
  o += tx(137, 38, 'Vertical Edges: X ★ [1, −1]', { fs: 14, fw: 700, fill: D.green.dark });
  o += grid([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]], 26, 62, 23, 3, {
    fs: 11.5,
    fill: (r, c, v) => v ? BLUE : GRAY,
  });
  o += arrow(130, 99, 152, 99, C.green);
  o += grid([[1, 0, -1], [1, 0, -1], [1, 0, -1]], 158, 62, 24, 3, {
    fs: 11,
    fill: (r, c, v) => v === 1 ? GREEN : v === -1 ? AMBER : GRAY,
  });
  o += tx(137, 186, 'Fires ±1 at vertical jumps', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(137, 214, 'Horizontal neighbor diff ≠ 0', { fs: 12, fill: C.ink });

  // Right: Horizontal stripe X.T with [1, -1]
  o += box(282, 16, 242, 232, D.red.tint, D.red.base, 10, 2);
  o += tx(403, 38, 'Horizontal Stripe: Xᵀ ★ [1, −1]', { fs: 14, fw: 700, fill: D.red.dark });
  o += grid([[1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1]], 316, 60, 23, 3, {
    fs: 11.5,
    fill: (r, c, v) => v ? BLUE : GRAY,
  });
  o += arrow(398, 111, 420, 111, D.red.base);
  o += grid([[0, 0], [0, 0], [0, 0], [0, 0]], 426, 60, 24, 3, {
    fs: 11,
    fill: () => GRAY,
  });
  o += tx(403, 186, 'Outputs ALL ZEROS (0)!', { fs: 13, fw: 700, fill: D.red.dark });
  o += tx(403, 214, 'Needs [1, −1]ᵀ for horiz edges', { fs: 12, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function kernelGdConvergence() {
  const W = 540, H = 260;
  let o = '';
  const steps = [
    { x: 16,  t: 'Step 0 (Init)', k: 'K ≈ [0.00, 0.00]', l: 'Loss ≈ 12.0', sub: 'Misses twelve ±1 edges', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 194, t: 'Step 3 (GD)', k: 'K ≈ [0.74, −0.74]', l: 'Loss ≈ 0.83', sub: 'η = 0.03 updates K', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 372, t: 'Step 9 (Converged)', k: 'K = [0.99, −0.99]', l: 'Loss ≈ 0.004', sub: 'Recovers [1, −1]!', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  steps.forEach((s) => {
    o += card(s.x, 18, 152, 128, s.fill, s.st, [
      { t: s.t, fs: 14, fw: 700, fill: s.dk },
      { t: s.k, mono: true, fs: 12.5, fw: 700, fill: C.ink },
      { t: s.l, mono: true, fs: 13, fw: 700, fill: s.dk },
      { t: s.sub, fs: 11.5, fill: C.muted },
    ]);
  });
  o += arrow(172, 82, 190, 82, C.amber);
  o += arrow(350, 82, 368, 82, C.green);
  o += card(16, 164, 508, 80, C.lblue, C.blue, [
    { t: 'Why Initial Loss Equals 12.0', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: '6 rows × (one +1 edge + one −1 edge) = 12 nonzero targets × (±1 − 0)² = 12.0', fs: 13, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function im2colGemm() {
  const W = 540, H = 270;
  let o = '';
  o += tx(68, 22, 'Input X (3×3)', { fs: 13.5, fw: 700, fill: D.blue.dark });
  o += grid([[0, 1, 2], [3, 4, 5], [6, 7, 8]], 26, 40, 24, 3, { fs: 12, fill: () => BLUE });
  o += arrow(110, 82, 148, 82, C.purple);
  o += tx(129, 66, 'im2col', { fs: 12, fw: 700, fill: D.purple.dark });

  o += tx(214, 22, 'Patches (4×4)', { fs: 13.5, fw: 700, fill: D.purple.dark });
  const P = [[0, 1, 3, 4], [1, 2, 4, 5], [3, 4, 6, 7], [4, 5, 7, 8]];
  o += grid(P, 156, 40, 26, 3, { fs: 11.5, fill: () => PURPLE });

  o += tx(284, 82, '×', { fs: 22, fw: 700, fill: C.ink });
  o += tx(318, 22, 'vec(K)', { mono: true, fs: 13.5, fw: 700, fill: D.orange.dark });
  o += grid([[0], [1], [2], [3]], 304, 40, 26, 3, { fs: 11.5, fill: () => AMBER });

  o += tx(348, 82, '=', { fs: 22, fw: 700, fill: C.ink });
  o += tx(442, 22, 'vec(Y) → (2×2)', { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  o += grid([[19], [25], [37], [43]], 368, 40, 28, 3, { fs: 11.5, fill: () => GREEN });
  o += arrow(406, 82, 434, 82, C.green);
  o += grid([[19, 25], [37, 43]], 442, 54, 34, 4, { fs: 13, fill: () => GREEN });

  o += card(16, 172, 508, 82, C.lamber, C.amber, [
    { t: 'Systems Trade-Off:  P ∈ ℝ^{(h_o·w_o)×(k_h·k_w)}  ×  vec(K) ∈ ℝ^{(k_h·k_w)×1}', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Duplicates interior pixels k_h·k_w times in RAM to run at peak GPU GEMM Tensor Core speed', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function stacked3x3Vs7x7() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 226, D.red.tint, D.red.base, [
    { t: 'Single 7×7 Conv Layer', fs: 15, fw: 700, fill: D.red.dark },
    { t: 'Receptive Field: 7 × 7', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'Weights: 7² c² = 49 c²', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Nonlinearities: 1 ReLU', fs: 13, fw: 600, fill: C.ink },
    { t: 'Heavy & shallow representation', fs: 12, fill: C.muted },
  ]);
  o += card(282, 18, 242, 226, C.lgreen, C.green, [
    { t: 'Three Stacked 3×3 Layers', fs: 15, fw: 700, fill: D.green.dark },
    { t: 'Receptive Field: 1+3(2) = 7 × 7', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'Weights: 3 × 3² c² = 27 c²', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Nonlinearities: 3 ReLUs', fs: 13, fw: 600, fill: C.ink },
    { t: '45% fewer weights + deeper!', fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function effectiveRfGaussian() {
  const W = 540, H = 264;
  let o = '';
  o += tx(270, 22, 'Theoretical RF (Uniform Box) vs. Effective RF (Gaussian Bell Curve)', { fs: 14, fw: 700, fill: C.ink });
  // Bar histogram showing path counts for L=3 1D kernel size 3: [1, 3, 6, 7, 6, 3, 1]
  const counts = [1, 3, 6, 7, 6, 3, 1];
  const offs = ['-3', '-2', '-1', '0', '+1', '+2', '+3'];
  const bx0 = 42, bw = 54, gap = 12, base = 186;
  counts.forEach((cnt, i) => {
    const x = bx0 + i * (bw + gap);
    const bh = cnt * 16;
    const y = base - bh;
    const isCenter = i >= 2 && i <= 4;
    o += box(x, y, bw, bh, isCenter ? C.lgreen : C.lblue, isCenter ? C.green : C.blue, 5, 2);
    o += tx(x + bw / 2, y - 12, `${cnt}p`, { mono: true, fs: 12.5, fw: 700, fill: isCenter ? D.green.dark : D.blue.dark });
    o += tx(x + bw / 2, base + 16, offs[i], { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  });
  o += line(26, base, 514, base, C.gray, 2);
  o += tx(270, 226, 'Center pixel (0) has 7× more paths than edges (±3) after 3 layers', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(270, 248, 'Theoretical width grows as O(L), but effective Gaussian σ grows as O(√L)', { fs: 12.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

function learningEdgeLoop() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 20, 146, 96, C.lblue, C.blue, [
    { t: 'Forward Pass', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'Ŷ = conv2d(X)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '6 × 7 map', mono: true, fs: 11.5, fill: C.muted },
  ]);
  o += arrow(168, 68, 192, 68, C.amber);
  o += card(198, 20, 146, 96, C.lamber, C.amber, [
    { t: 'Squared Error', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'L = Σ (Ŷ − Y)²', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '42 spatial entries', fs: 11.5, fill: C.muted },
  ]);
  o += arrow(350, 68, 374, 68, D.red.base);
  o += card(380, 20, 144, 96, D.red.tint, D.red.base, [
    { t: 'GD Step', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'K ← K − η ∇_K L', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'η = 0.03, 10 steps', mono: true, fs: 11.5, fill: C.muted },
  ]);
  o += card(16, 138, 508, 100, C.lgreen, C.green, [
    { t: 'Automatic Filter Discovery via Backpropagation', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'Random N(0, 0.01²) weights → [1.0, −1.0] vertical edge detector from data alone', fs: 13, fw: 600, fill: C.ink },
    { t: 'In deep CNNs, end-to-end task loss supervises thousands of kernels simultaneously', fs: 12.5, fill: C.muted },
  ]);
  return svg(W, H, o);
}

// ── padding-and-strides.md diagrams (7) ──────────────────────────────────

function borderErosion() {
  const W = 540, H = 264;
  let o = '';
  // Left nested squares showing 240x240 shrinking to 200x200
  o += box(24, 24, 180, 180, D.red.tint, D.red.base, 6, 2, true);
  o += tx(114, 42, '240 × 240 Input', { mono: true, fs: 12.5, fw: 700, fill: D.red.dark });
  o += box(48, 56, 132, 132, C.lblue, C.blue, 6, 2);
  o += tx(114, 110, '200 × 200', { mono: true, fs: 14.5, fw: 700, fill: D.blue.dark });
  o += tx(114, 132, 'after 10 × (5×5)', { fs: 12, fw: 600, fill: C.ink });
  o += tx(114, 228, '30% border area lost!', { fs: 13.5, fw: 700, fill: D.red.dark });

  // Right 3 controls
  const ctrls = [
    { y: 20,  t: '1. Padding (p)', desc: 'Preserves H×W & samples borders equally', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 98,  t: '2. Stride (s)', desc: 'Downsamples grid & grows RF geometrically', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { y: 176, t: '3. Dilation (d)', desc: 'Spreads k×k taps over d(k−1)+1 at full res', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  ctrls.forEach((c) => {
    o += card(228, c.y, 296, 64, c.fill, c.st, [
      { t: c.t, fs: 14, fw: 700, fill: c.dk },
      { t: c.desc, fs: 12.5, fw: 600, fill: C.ink },
    ]);
  });
  return svg(W, H, o);
}

function oddVsEvenKernel() {
  const W = 540, H = 260;
  let o = '';
  // Left: Odd kernel k=3
  o += box(16, 16, 242, 228, C.lgreen, C.green, 10, 2);
  o += tx(137, 38, 'Odd Kernel (k = 3, 5, 7)', { fs: 14.5, fw: 700, fill: D.green.dark });
  o += tx(137, 60, 'p = (k − 1)/2 is an integer!', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += grid([['p=1', 'x−1', 'x₀', 'x+1', 'p=1']], 29, 86, 39, 4, {
    fs: 12,
    fill: (r, c) => c === 2 ? AMBER : (c === 0 || c === 4) ? GRAY : GREEN,
  });
  o += tx(137, 162, 'Symmetric padding (1 left, 1 right)', { fs: 12.5, fw: 600, fill: D.green.dark });
  o += tx(137, 188, 'Output Y[i,j] centered on X[i,j]', { fs: 12.5, fw: 700, fill: C.ink });

  // Right: Even kernel k=2
  o += box(282, 16, 242, 228, D.red.tint, D.red.base, 10, 2);
  o += tx(403, 38, 'Even Kernel (k = 2, 4)', { fs: 14.5, fw: 700, fill: D.red.dark });
  o += tx(403, 60, 'k − 1 is odd (no center pixel)', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += grid([['x₀', 'x+1', 'p=1']], 338, 86, 40, 5, {
    fs: 12,
    fill: (r, c) => c === 2 ? GRAY : RED,
  });
  o += tx(403, 162, 'Requires asymmetric padding', { fs: 12.5, fw: 600, fill: D.red.dark });
  o += tx(403, 188, 'Half-pixel coordinate shift!', { fs: 12.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

function paddingDarkFrame() {
  const W = 540, H = 264;
  let o = '';
  // Left: Zero padding output on 5x5 ones
  o += box(16, 16, 242, 232, C.lamber, C.amber, 10, 2);
  o += tx(137, 38, 'Zero Padding (5×5 Ones ★ 3×3)', { fs: 13.5, fw: 700, fill: D.orange.dark });
  const Z = [
    [4, 6, 6, 6, 4],
    [6, 9, 9, 9, 6],
    [6, 9, 9, 9, 6],
    [6, 9, 9, 9, 6],
    [4, 6, 6, 6, 4],
  ];
  o += grid(Z, 58, 54, 29, 3, {
    fs: 12,
    fill: (r, c, v) => v === 9 ? BLUE : AMBER,
  });
  o += tx(137, 224, 'Dark border frame (4, 6 vs. 9)', { fs: 12.5, fw: 700, fill: D.orange.dark });

  // Right: Reflect/Replicate padding
  o += box(282, 16, 242, 232, C.lgreen, C.green, 10, 2);
  o += tx(403, 38, 'Reflect / Replicate Padding', { fs: 13.5, fw: 700, fill: D.green.dark });
  const R = [
    [9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9],
    [9, 9, 9, 9, 9],
  ];
  o += grid(R, 324, 54, 29, 3, {
    fs: 12,
    fill: () => GREEN,
  });
  o += tx(403, 224, 'Uniform 9s (no border artifact!)', { fs: 12.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function strideStepDerivation() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 18, 508, 74, C.lblue, C.blue, [
    { t: 'Padded Input Axis Length:  n + 2p', mono: true, fs: 15, fw: 700, fill: D.blue.dark },
    { t: 'Place first width-k window at 0 ⇒ (n + 2p − k) pixels remain for steps of size s', fs: 13, fw: 600, fill: C.ink },
  ]);
  o += card(16, 108, 244, 128, D.purple.tint, C.purple, [
    { t: 'General Strided Formula', fs: 14.5, fw: 700, fill: D.purple.dark },
    { t: 'h_o = ⌊(n + 2p − k) / s⌋ + 1', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: '(+1 counts initial placement)', fs: 12, fill: C.muted },
  ]);
  o += card(280, 108, 244, 128, C.lgreen, C.green, [
    { t: 'With "Same" Padding (2p = k−1)', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'h_o = ⌊(n − 1) / s⌋ + 1 = ⌈n / s⌉', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Divides resolution by s!', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function threeStrideRegimes() {
  const W = 540, H = 260;
  let o = '';
  const regimes = [
    { x: 16,  t: '1. Preserve ("Same")', cfg: 'k=3, p=1, s=1', out: 'n × n → n × n', role: 'Constant-res feature stage', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: '2. Halve (Downsample)', cfg: 'k=3, p=1, s=2', out: 'n × n → ⌈n/2⌉ × ⌈n/2⌉', role: '4× fewer pixels, 2× channels', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 372, t: '3. Patchify (Tokens)', cfg: 'k=4, p=0, s=4', out: 'n × n → (n/4) × (n/4)', role: 'Disjoint tiles (ViT / ConvNeXt)', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  regimes.forEach((r) => {
    o += card(r.x, 20, 152, 220, r.fill, r.st, [
      { t: r.t, fs: 13.5, fw: 700, fill: r.dk },
      { t: r.cfg, mono: true, fs: 13, fw: 700, fill: C.ink },
      { t: r.out, mono: true, fs: 11.5, fw: 700, fill: r.dk },
      { t: r.role, fs: 12, fill: C.muted },
    ]);
  });
  return svg(W, H, o);
}

function geometricRfStride() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 132, C.lblue, C.blue, [
    { t: 'Unit Strides s = (1, 1, 1)', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'Three 3×3 Conv Layers', fs: 12.5, fw: 600, fill: C.ink },
    { t: 'r = 1 + 2(1) + 2(1) + 2(1) = 7', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Slow linear growth: 2L + 1', fs: 12, fill: C.muted },
  ]);
  o += card(282, 18, 242, 132, C.lgreen, C.green, [
    { t: 'Strided Tower s = (2, 2, 1)', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'Same Three 3×3 Conv Layers', fs: 12.5, fw: 600, fill: C.ink },
    { t: 'r = 1 + 2(1) + 2(2) + 2(4) = 15', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Geometric growth (>2× wider!)', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 166, 508, 78, D.purple.tint, C.purple, [
    { t: 'Master Receptive-Field Equation:  r = 1 + Σ_{i=1}^{L} (k_i − 1) Π_{j=1}^{i-1} s_j', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Early stride s_j multiplies the spatial reach of EVERY layer above it (independent of padding p)', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function dilationGridding() {
  const W = 540, H = 264;
  let o = '';
  // Left: Constant dilation d=(2,2) gridding
  o += box(16, 16, 242, 232, D.red.tint, D.red.base, 10, 2);
  o += tx(137, 38, 'Constant Dilation d = (4, 4, 4, 4)', { fs: 13.5, fw: 700, fill: D.red.dark });
  const G1 = [
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 0, 0],
    [1, 0, 1, 0, 1],
  ];
  o += grid(G1, 66, 56, 26, 3, {
    fs: 11.5,
    fill: (r, c, v) => v ? RED : GRAY,
  });
  o += tx(137, 212, 'Gridding Artifact (7.4% sampled!)', { fs: 13, fw: 700, fill: D.red.dark });
  o += tx(137, 232, 'Blind to thin objects in gaps', { fs: 12, fill: C.ink });

  // Right: Exponential dilation d=(1,2,4,8)
  o += box(282, 16, 242, 232, C.lgreen, C.green, 10, 2);
  o += tx(403, 38, 'Exponential Schedule d = (1, 2, 4, 8)', { fs: 13.5, fw: 700, fill: D.green.dark });
  const G2 = [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
  ];
  o += grid(G2, 332, 56, 26, 3, {
    fs: 11.5,
    fill: () => GREEN,
  });
  o += tx(403, 212, '100% Dense Coverage (r = 31)', { fs: 13, fw: 700, fill: D.green.dark });
  o += tx(403, 232, 'Finer layers bridge coarser taps', { fs: 12, fill: C.ink });
  return svg(W, H, o);
}

// ── channels.md diagrams (9) ─────────────────────────────────────────────

function channelScalingWall() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 20, 236, 118, C.lblue, C.blue, [
    { t: 'Early Stem (3 → 64 ch)', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: '3×3 kernel: 9 · 3 · 64 + 64', mono: true, fs: 12.5, fill: C.ink },
    { t: '= 1,792 parameters', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
  ]);
  o += arrow(258, 79, 282, 79, D.red.base);
  o += card(288, 20, 236, 118, D.red.tint, D.red.base, [
    { t: 'Deep Stage (256 → 256 ch)', fs: 14.5, fw: 700, fill: D.red.dark },
    { t: '3×3 kernel: 9 · 256 · 256 + 256', mono: true, fs: 12.5, fill: C.ink },
    { t: '= 590,080 parameters!', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
  ]);
  o += card(16, 156, 508, 86, C.lamber, C.amber, [
    { t: 'The Quadratic Channel Wall:  Cost ∝ c_i × c_o', mono: true, fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: 'At 56×56 resolution, one 256→256 3×3 layer executes 1.85 × 10⁹ multiply-adds per image', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function colorOpponency() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 20, 146, 86, D.red.tint, D.red.base, [
    { t: 'Red Channel X_R', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Weight K_R = +1', mono: true, fs: 13, fw: 700, fill: C.ink },
  ]);
  o += card(16, 124, 146, 86, C.lgreen, C.green, [
    { t: 'Green Channel X_G', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Weight K_G = −1', mono: true, fs: 13, fw: 700, fill: C.ink },
  ]);
  o += arrow(168, 63, 210, 102, D.red.base);
  o += arrow(168, 167, 210, 128, C.green);
  o += card(216, 66, 118, 98, D.purple.tint, C.purple, [
    { t: 'Channel Sum', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Y = X_R − X_G', mono: true, fs: 13, fw: 700, fill: C.ink },
  ]);
  o += arrow(340, 115, 366, 115, C.blue);
  o += card(372, 24, 152, 182, C.lblue, C.blue, [
    { t: 'Opponent Output', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '+1 in Red region', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: '−1 in Green region', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '0 on neutral gray', mono: true, fs: 12.5, fill: C.muted },
    { t: 'Detects isoluminant', fs: 12, fw: 600, fill: C.ink },
    { t: 'color edges!', fs: 12, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function multiOut4d() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 62, 128, 124, C.lblue, C.blue, [
    { t: '3D Input X', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '(c_i, n_h, n_w)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'All c_i channels', fs: 12, fill: C.muted },
  ]);
  const banks = [
    { y: 18,  t: 'K₁: (c_i, k_h, k_w)', out: 'Y₁: (h_o, w_o)', col: C.blue, bg: C.lblue, dk: D.blue.dark },
    { y: 98,  t: 'K₂: (c_i, k_h, k_w)', out: 'Y₂: (h_o, w_o)', col: C.purple, bg: D.purple.tint, dk: D.purple.dark },
    { y: 178, t: 'K_{c_o}: (c_i, k_h, k_w)', out: 'Y_{c_o}: (h_o, w_o)', col: C.green, bg: C.lgreen, dk: D.green.dark },
  ];
  banks.forEach((b) => {
    o += arrow(150, 124, 178, b.y + 30, b.col);
    o += card(186, b.y, 176, 60, b.bg, b.col, [
      { t: b.t, mono: true, fs: 12, fw: 700, fill: b.dk },
    ]);
    o += arrow(368, b.y + 30, 392, b.y + 30, b.col);
    o += card(400, b.y, 124, 60, '#FFFFFF', b.col, [
      { t: b.out, mono: true, fs: 12, fw: 700, fill: b.dk },
    ]);
  });
  return svg(W, H, o);
}

function kernelShiftLinearity() {
  const W = 540, H = 256;
  let o = '';
  const chs = [
    { x: 16,  t: 'Channel 0: K', eq: 'X ★ K', v: 'Y₀[0,0] = 56', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: 'Channel 1: K + 1', eq: '56 + 1 × (sum=20)', v: 'Y₁[0,0] = 76', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 372, t: 'Channel 2: K + 2', eq: '56 + 2 × (sum=20)', v: 'Y₂[0,0] = 96', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  chs.forEach((c) => {
    o += card(c.x, 18, 152, 122, c.fill, c.st, [
      { t: c.t, fs: 14, fw: 700, fill: c.dk },
      { t: c.eq, mono: true, fs: 12, fw: 600, fill: C.ink },
      { t: c.v, mono: true, fs: 14, fw: 700, fill: c.dk },
    ]);
  });
  o += card(16, 158, 508, 82, D.purple.tint, C.purple, [
    { t: 'Linearity Check:  X ★ (K + c) = (X ★ K) + c · Σ_{(a,b)} X_{window}', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Top-left 2×2 window sum across both input channels = (0+1+3+4) + (1+2+4+5) = 20', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function threeResourceBudgets() {
  const W = 540, H = 260;
  let o = '';
  const b = [
    { x: 16,  t: '1. Parameter Memory', eq: 'c_o c_i k_h k_w + c_o', sc: 'Independent of H × W', peak: 'Peaks in deep wide layers', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: '2. Arithmetic (MACs)', eq: 'h_o w_o · c_o c_i k_h k_w', sc: 'Scales with area × c_o c_i', peak: 'Uniform across stages', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 372, t: '3. Activation RAM', eq: 'B · h_o w_o c_o', sc: 'Cached for backward pass', peak: 'Peaks in early high-res stem!', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  b.forEach((item) => {
    o += card(item.x, 20, 152, 220, item.fill, item.st, [
      { t: item.t, fs: 13.5, fw: 700, fill: item.dk },
      { t: item.eq, mono: true, fs: 11.5, fw: 700, fill: C.ink },
      { t: item.sc, fs: 12, fw: 600, fill: C.muted },
      { t: item.peak, fs: 12, fw: 700, fill: item.dk },
    ]);
  });
  return svg(W, H, o);
}

function memoryVsParamsStages() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 18, 242, 134, C.lblue, C.blue, [
    { t: 'Early Stem (3→64 @ 224×224)', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Params: 1,792 (7 KB)', mono: true, fs: 12.5, fw: 600, fill: C.ink },
    { t: 'Activations: 3.21M (12.8 MB!)', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: '1,792× more activations than params', fs: 11.5, fw: 700, fill: D.blue.dark },
  ]);
  o += card(282, 18, 242, 134, D.purple.tint, C.purple, [
    { t: 'Deep Stage (256→256 @ 56×56)', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Params: 590,080 (2.36 MB!)', mono: true, fs: 12.5, fw: 700, fill: D.purple.dark },
    { t: 'Activations: 0.80M (3.2 MB)', mono: true, fs: 12.5, fw: 600, fill: C.ink },
    { t: '330× more params than stem', fs: 11.5, fw: 700, fill: D.purple.dark },
  ]);
  o += card(16, 168, 508, 78, D.red.tint, D.red.base, [
    { t: 'Training GPU OOM Rule:  VRAM is dominated by Batch Size × Early High-Res Feature Maps!', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Both layers execute ~10⁸–10⁹ MACs, but parameter memory and activation RAM live at opposite ends', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function matmul1x1Reshape() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 28, 126, 98, C.lblue, C.blue, [
    { t: '3D Input X', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '(c_i, H, W)', mono: true, fs: 13, fw: 700, fill: C.ink },
  ]);
  o += arrow(148, 77, 174, 77, C.purple);
  o += card(180, 28, 168, 98, D.purple.tint, C.purple, [
    { t: 'Zero-Copy Reshape', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'K: (c_o, c_i)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '× X_flat: (c_i, H·W)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
  ]);
  o += arrow(354, 77, 380, 77, C.green);
  o += card(386, 28, 138, 98, C.lgreen, C.green, [
    { t: 'Reshape Output', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Y: (c_o, H, W)', mono: true, fs: 13, fw: 700, fill: C.ink },
  ]);
  o += card(16, 148, 508, 90, C.lamber, C.amber, [
    { t: 'Why 1×1 Convolutions Need Zero im2col Workspace Duplication', fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: '1×1 windows never overlap neighbors ⇒ single dense GEMM across all H·W pixels simultaneously', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function bottleneck1x1Savings() {
  const W = 540, H = 264;
  let o = '';
  // Top: Direct 3x3
  o += box(16, 16, 508, 96, D.red.tint, D.red.base, 10, 2);
  o += tx(270, 38, 'Direct 3×3 Convolution (256 → 256 ch at 56×56)', { fs: 14.5, fw: 700, fill: D.red.dark });
  o += card(42, 52, 456, 46, '#FFFFFF', D.red.base, [
    { t: '256 × 256 × 9 = 589,824 weights   ·   1.85 × 10⁹ MACs', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
  ]);

  // Bottom: 1x1 Bottleneck
  o += box(16, 128, 508, 122, C.lgreen, C.green, 10, 2);
  o += tx(270, 148, '1×1 Bottleneck Stack: 69,632 weights · 2.18×10⁸ MACs (8.5× reduction!)', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += card(28, 166, 144, 68, '#FFFFFF', C.blue, [
    { t: '1×1 Reduce', fs: 13, fw: 700, fill: D.blue.dark },
    { t: '256 → 64 + ReLU', mono: true, fs: 11.5, fill: C.ink },
    { t: '16,384 weights', mono: true, fs: 11.5, fill: C.muted },
  ]);
  o += arrow(176, 200, 194, 200, C.green);
  o += card(198, 166, 144, 68, '#FFFFFF', C.green, [
    { t: '3×3 Spatial', fs: 13, fw: 700, fill: D.green.dark },
    { t: '64 → 64 + ReLU', mono: true, fs: 11.5, fill: C.ink },
    { t: '36,864 weights', mono: true, fs: 11.5, fill: C.muted },
  ]);
  o += arrow(346, 200, 364, 200, C.green);
  o += card(368, 166, 144, 68, '#FFFFFF', C.blue, [
    { t: '1×1 Expand', fs: 13, fw: 700, fill: D.blue.dark },
    { t: '64 → 256 + ReLU', mono: true, fs: 11.5, fill: C.ink },
    { t: '16,384 weights', mono: true, fs: 11.5, fill: C.muted },
  ]);
  return svg(W, H, o);
}

function groupedDepthwisePreview() {
  const W = 540, H = 260;
  let o = '';
  const items = [
    { x: 16,  t: 'Standard Conv', w: 'c_o · c_i · k²', s: '1× baseline', role: 'All-to-all channels', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: 'Grouped (g groups)', w: '(c_o·c_i·k²)/g', s: 'g× cheaper', role: 'Block-diagonal (ResNeXt)', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 372, t: 'Depthwise-Separable', w: 'c_i·k² + c_o·c_i', s: '~8–9× cheaper (3×3)', role: 'Space ⊥ Channel (MobileNet)', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  items.forEach((it) => {
    o += card(it.x, 20, 152, 220, it.fill, it.st, [
      { t: it.t, fs: 13.5, fw: 700, fill: it.dk },
      { t: it.w, mono: true, fs: 12, fw: 700, fill: C.ink },
      { t: it.s, fs: 13, fw: 700, fill: it.dk },
      { t: it.role, fs: 11.5, fill: C.muted },
    ]);
  });
  return svg(W, H, o);
}

// ── pooling.md diagrams (9) ──────────────────────────────────────────────

function jitterSensitivity() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 20, 236, 114, D.red.tint, D.red.base, [
    { t: 'Equivariant Conv Feature Map', fs: 14, fw: 700, fill: D.red.dark },
    { t: '1-pixel camera jitter shifts', fs: 12.5, fw: 600, fill: C.ink },
    { t: 'EVERY edge coordinate by 1 px', fs: 12.5, fw: 700, fill: D.red.dark },
  ]);
  o += arrow(258, 77, 282, 77, C.green);
  o += card(288, 20, 236, 114, C.lgreen, C.green, [
    { t: 'Local / Global Pooling Layer', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Summarizes k×k patch by max/mean', fs: 12.5, fw: 600, fill: C.ink },
    { t: '0 learnable parameters!', fs: 13, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 152, 508, 86, C.lblue, C.blue, [
    { t: 'Two Complementary Roles of Pooling', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: '1. Local pooling (k×k, stride k): absorbs small spatial jitters & halves resolution', fs: 12.5, fw: 600, fill: C.ink },
    { t: '2. Global pooling (H×W): collapses spatial coordinates into invariant channel vector', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function maxVsAvgPool() {
  const W = 540, H = 264;
  let o = '';
  o += tx(78, 24, 'Input X (3×3)', { fs: 14, fw: 700, fill: D.blue.dark });
  o += grid([[0, 1, 2], [3, 4, 5], [6, 7, 8]], 28, 42, 30, 4, {
    fs: 13,
    fill: (r, c) => (r >= 1 && c >= 1) ? AMBER : BLUE,
  });
  o += tx(78, 166, '5 non-max entries', { fs: 12, fill: C.muted });

  o += card(164, 24, 172, 216, C.lgreen, C.green, [
    { t: '2×2 Max-Pool', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '[[4, 5], [7, 8]]', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'Keeps peak amplitude', fs: 12, fw: 600, fill: D.green.dark },
    { t: 'Backprop: 100% grad', fs: 12, fw: 600, fill: C.ink },
    { t: 'to winning pixel only', fs: 11.5, fill: C.muted },
  ]);
  o += card(352, 24, 172, 216, D.purple.tint, C.purple, [
    { t: '2×2 Avg-Pool', fs: 14.5, fw: 700, fill: D.purple.dark },
    { t: '[[2, 3], [5, 6]]', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'Measures area coverage', fs: 12, fw: 600, fill: D.purple.dark },
    { t: 'Backprop: splits grad', fs: 12, fw: 600, fill: C.ink },
    { t: 'equally (1/4 each)', fs: 11.5, fill: C.muted },
  ]);
  return svg(W, H, o);
}

function blockBoundaryShift() {
  const W = 540, H = 260;
  let o = '';
  const cases = [
    { x: 16,  t: 'Peak at Col j = 0', blk: 'Left 2×2 Block [1, 0]', out: 'MaxPool → [1, 0]', status: 'Baseline position', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: 'Shift j = 0 → 1', blk: 'Still Left Block [0, 1]', out: 'MaxPool → [1, 0]', status: '100% Absorbed!', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 372, t: 'Shift j = 1 → 2', blk: 'Crosses to Right Block', out: 'MaxPool → [0, 1]', status: 'Jumps 1 cell!', fill: C.lamber, st: C.amber, dk: D.orange.dark },
  ];
  cases.forEach((c) => {
    o += card(c.x, 18, 152, 134, c.fill, c.st, [
      { t: c.t, fs: 14, fw: 700, fill: c.dk },
      { t: c.blk, fs: 12, fw: 600, fill: C.ink },
      { t: c.out, mono: true, fs: 12.5, fw: 700, fill: C.ink },
      { t: c.status, fs: 13, fw: 700, fill: c.dk },
    ]);
  });
  o += card(16, 168, 508, 76, D.purple.tint, C.purple, [
    { t: 'Shift-Absorption Fraction for k×k Stride-k Pool:  (k − 1) / k  per axis', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'k = 2 absorbs 50% of 1-px horizontal shifts;  k = 3 absorbs 67% (4/9 ≈ 44.4% of diagonal shifts)', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function subsamplingAliasing() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 16, 242, 132, D.red.tint, D.red.base, [
    { t: 'Naive Stride-2 Subsampling', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'x = (1,0,1,0,1,0) → (1, 1, 1)', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'T₁x = (0,1,0,1,0,1) → (0, 0, 0)!', mono: true, fs: 12, fw: 700, fill: D.red.dark },
    { t: '1-px shift flips all-1s to all-0s', fs: 12, fill: C.muted },
  ]);
  o += card(282, 16, 242, 132, C.lgreen, C.green, [
    { t: 'Anti-Aliased BlurPool [½, ½]', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'x = (1,0,1,0,1,0) → (0.5, 0.5, 0.5)', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'T₁x = (0,1,0,1,0,1) → (0.5, 0.5, 0.5)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Coincides with 2×2 AvgPool!', fs: 12, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 164, 508, 82, C.lamber, C.amber, [
    { t: 'Nyquist Aliasing Hazard (Zhang, 2019)', fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: 'Low-pass smoothing before stride-2 downsampling stabilizes high-frequency textures under shift', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function globalPoolingCollapse() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 26, 152, 110, C.lblue, C.blue, [
    { t: 'Feature Maps X', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '(c_o, H, W)', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: 'H×W spatial grid', fs: 12, fill: C.muted },
  ]);
  o += arrow(174, 81, 206, 81, C.purple);
  o += card(212, 26, 152, 110, D.purple.tint, C.purple, [
    { t: 'Global Pool', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'Window = H × W', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '0 parameters', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  o += arrow(370, 81, 402, 81, C.green);
  o += card(408, 26, 116, 110, C.lgreen, C.green, [
    { t: 'Vector z', fs: 14, fw: 700, fill: D.green.dark },
    { t: '(c_o, 1, 1)', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: '1 scalar/ch', fs: 12, fill: C.muted },
  ]);
  o += card(16, 154, 508, 88, C.lamber, C.amber, [
    { t: 'Boundary Caveat with Zero-Padded Preceding Convs', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'Interior impulse participates in 9 conv windows (mean 9/16) vs. border impulse in 6 (mean 6/16)', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function nonoverlappingDefaultStride() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 20, 242, 220, C.lblue, C.blue, [
    { t: 'Default Pool Stride: s = k', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: '3×3 max-pool on 4×4 input', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'k=3, p=0, s=3 (non-overlapping)', mono: true, fs: 12, fill: C.muted },
    { t: '⌊(4−3)/3⌋ + 1 = 1 × 1 output', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Output: [[10.]]', mono: true, fs: 13.5, fw: 700, fill: C.ink },
  ]);
  o += card(282, 20, 242, 220, C.lgreen, C.green, [
    { t: 'Custom Pool: p = 1, s = 2', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '3×3 max-pool, pad 1, stride 2', mono: true, fs: 11.5, fw: 700, fill: C.ink },
    { t: 'Overlapping 3×3 windows', fs: 12, fill: C.muted },
    { t: '⌊(4+2−3)/2⌋ + 1 = 2 × 2 output', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Output: [[5, 7], [13, 15]]', mono: true, fs: 13, fw: 700, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function negInfPadding() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 16, 242, 134, C.lgreen, C.green, [
    { t: 'Correct: Pad MaxPool with −∞', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Input X − 16 ∈ [−16, −1]', mono: true, fs: 12.5, fw: 600, fill: C.ink },
    { t: 'max(−∞, −11) = −11', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Output: [[−11, −9], [−3, −1]]', mono: true, fs: 12.5, fw: 700, fill: C.ink },
  ]);
  o += card(282, 16, 242, 134, D.red.tint, D.red.base, [
    { t: 'Bug: Pad MaxPool with 0', fs: 14, fw: 700, fill: D.red.dark },
    { t: '0 beats all negative numbers!', fs: 12.5, fw: 600, fill: C.ink },
    { t: 'max(0, −11) = 0  (Corrupted!)', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Breaks shift equivariance in value', fs: 12, fill: C.muted },
  ]);
  o += card(16, 166, 508, 80, C.lamber, C.amber, [
    { t: 'Algebraic Neutral Elements:  0 for Sum (+)   vs.   −∞ for Max (max)', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'For average pooling, divide border windows only by their real pixel count', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function poolingChannelIndep() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 20, 236, 102, C.lblue, C.blue, [
    { t: 'Channel 0: X', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '→ Pool(X) = [[5, 7], [13, 15]]', mono: true, fs: 12.5, fw: 700, fill: C.ink },
  ]);
  o += card(16, 138, 236, 102, D.purple.tint, C.purple, [
    { t: 'Channel 1: X + 1', fs: 14, fw: 700, fill: D.purple.dark },
    { t: '→ Pool(X+1) = [[6, 8], [14, 16]]', mono: true, fs: 12.5, fw: 700, fill: C.ink },
  ]);
  o += card(276, 20, 248, 220, C.lgreen, C.green, [
    { t: 'Pooling vs. Convolution', fs: 15, fw: 700, fill: D.green.dark },
    { t: 'Convolution: sums over c_i', fs: 13, fw: 600, fill: C.ink },
    { t: '⇒ changes width c_i → c_o', mono: true, fs: 12.5, fill: C.muted },
    { t: 'Pooling: per-channel only', fs: 13, fw: 700, fill: D.green.dark },
    { t: '⇒ always keeps c_o = c_i', mono: true, fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function classicalVsModernPooling() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 224, C.lamber, C.amber, [
    { t: 'Classical CNNs (LeNet / VGG)', fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: '• Local Pool after every Conv stage', fs: 12.5, fw: 600, fill: C.ink },
    { t: '• Fixed max/avg downsampling rule', fs: 12.5, fill: C.muted },
    { t: '• Flatten + 3 huge Dense layers', fs: 12.5, fw: 700, fill: D.red.dark },
    { t: '(≈90–96% of all parameters!)', fs: 12, fill: D.red.dark },
  ]);
  o += card(282, 18, 242, 224, C.lgreen, C.green, [
    { t: 'Modern CNNs (ResNet / ConvNeXt)', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '• Stride-2 Convs between stages', fs: 12.5, fw: 600, fill: C.ink },
    { t: '• Learns downsampling weights', fs: 12.5, fill: C.muted },
    { t: '• Global Avg Pool + 1 Linear layer', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '(Eliminates >75% of parameters!)', fs: 12, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

// ── lenet.md diagrams (11) ───────────────────────────────────────────────

function mlpVsLenetParadigm() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 224, D.red.tint, D.red.base, [
    { t: `Chapter ${num('chap_perceptrons')} MLP Baseline`, fs: 14.5, fw: 700, fill: D.red.dark },
    { t: 'Flatten 28×28 → 784 → 256 → 10', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'Parameters: 203,530', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Multiply-Adds: ~203,000', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Val Accuracy: ~87% (30 epochs)', fs: 13, fw: 700, fill: C.ink },
  ]);
  o += card(282, 18, 242, 224, C.lgreen, C.green, [
    { t: `LeNet-5 Spatial CNN (${sec('sec_lenet')})`, fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '2D Conv+Pool Encoder → Dense Head', fs: 12, fw: 600, fill: C.ink },
    { t: 'Parameters: 61,706 (3.3× fewer!)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Multiply-Adds: 416,520', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Val Accuracy: ~88–90% (30 epochs)', fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function lenet51998VsModern() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 16, 242, 232, C.lamber, C.amber, [
    { t: 'Historical LeNet-5 (1998)', fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: '• 32×32 input, unpadded Conv1', fs: 12.5, fw: 600, fill: C.ink },
    { t: '• Scaled tanh: 1.7159 tanh(2x/3)', mono: true, fs: 11.5, fill: C.ink },
    { t: '• Trainable pool scale + bias', fs: 12.5, fill: C.muted },
    { t: '• Sparse 6→16 Conv2 table', fs: 12.5, fill: C.muted },
    { t: '• Euclidean RBF output units', fs: 12.5, fill: C.ink },
  ]);
  o += card(282, 16, 242, 232, C.lblue, C.blue, [
    { t: 'Our Pedagogical LeNet', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: '• 28×28 input + p=2 on Conv1', fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: '• Standard Sigmoid σ(x)', mono: true, fs: 12, fill: C.ink },
    { t: '• 0-param 2×2 AvgPool (s=2)', fs: 12.5, fill: C.ink },
    { t: '• Dense 6→16 Conv2 tensor', fs: 12.5, fill: C.ink },
    { t: '• Linear logits + Cross-Entropy', fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);
  return svg(W, H, o);
}

function lenetParamFlopAsymmetry() {
  const W = 540, H = 264;
  let o = '';
  o += tx(270, 24, 'LeNet Resource Split: 61,706 Params vs. 416,520 MACs', { fs: 14.5, fw: 700, fill: C.ink });
  // Bar 1: Parameters
  o += tx(16, 56, 'Parameters (Memory):', { fs: 13.5, fw: 700, fill: C.ink, anchor: 'start' });
  o += box(16, 68, 36, 42, C.lblue, C.blue, 5, 2);
  o += box(56, 68, 468, 42, D.red.tint, D.red.base, 5, 2);
  o += tx(290, 89, 'Dense Head: 59,134 params (95.8%! — Dense1 alone is 78%)', { fs: 13, fw: 700, fill: D.red.dark });
  o += tx(16, 124, 'Conv1+2: 2,572 params (4.2%)', { fs: 12.5, fw: 700, fill: D.blue.dark, anchor: 'start' });

  // Bar 2: MACs
  o += tx(16, 160, 'Multiply-Adds (Compute):', { fs: 13.5, fw: 700, fill: C.ink, anchor: 'start' });
  o += box(16, 172, 430, 42, C.lgreen, C.green, 5, 2);
  o += tx(231, 193, 'Conv1+2: 357,600 MACs (85.9%! — reused across H×W)', { fs: 13, fw: 700, fill: D.green.dark });
  o += box(450, 172, 74, 42, C.lamber, C.amber, 5, 2);
  o += tx(487, 193, '14.1%', { mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });
  o += tx(524, 232, 'Dense Head: 58,920 MACs (14.1%)', { fs: 12.5, fw: 700, fill: D.orange.dark, anchor: 'end' });
  return svg(W, H, o);
}

function sigmoidGradientAttenuation() {
  const W = 540, H = 260;
  let o = '';
  const layers = ['Dense2 (σ′≤¼)', 'Dense1 (σ′≤¼)', 'Conv2 (σ′≤¼)', 'Conv1 (σ′≤¼)'];
  layers.forEach((l, i) => {
    const x = 392 - i * 124;
    o += card(x, 28, 108, 82, i === 3 ? D.red.tint : C.lamber, i === 3 ? D.red.base : C.amber, [
      { t: l, fs: 12.5, fw: 700, fill: i === 3 ? D.red.dark : D.orange.dark },
      { t: `∏ σ′ ≤ 1/${4 ** (i + 1)}`, fs: 14, fw: 700, fill: C.ink },
    ]);
    if (i < 3) {
      o += arrow(x - 3, 69, x - 13, 69, D.red.base);
    }
  });
  o += card(16, 132, 508, 110, D.red.tint, D.red.base, [
    { t: 'Why Classical LeNet Stalls at lr = 0.1 and Requires lr = 2.0', fs: 14.5, fw: 700, fill: D.red.dark },
    { t: '1. Conv1’s gradient carries four σ′ factors; their product is ≤ (1/4)⁴ = 1/256', fs: 12.5, fw: 600, fill: C.ink },
    { t: '2. Forward activations cluster near 0.5, pinning initial loss at ln(10) ≈ 2.30 plateau', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function lenetTrainingTrajectory() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 20, 152, 116, D.red.tint, D.red.base, [
    { t: 'Early epochs (if any)', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'Sigmoid Plateau', fs: 13, fw: 700, fill: C.ink },
    { t: 'Loss ≈ ln 10 = 2.30', mono: true, fs: 12, fill: D.red.dark },
    { t: 'Val Acc ≈ 10%', mono: true, fs: 12, fill: C.muted },
  ]);
  o += arrow(172, 78, 190, 78, C.amber);
  o += card(194, 20, 152, 116, C.lamber, C.amber, [
    { t: 'By epoch ~5', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'Rapid Escape', fs: 13, fw: 700, fill: C.ink },
    { t: 'Weights break symmetry', fs: 11.5, fill: C.muted },
    { t: 'Val Acc → 82%', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark },
  ]);
  o += arrow(350, 78, 368, 78, C.green);
  o += card(372, 20, 152, 116, C.lgreen, C.green, [
    { t: 'By epoch 30', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Convergence', fs: 13, fw: 700, fill: C.ink },
    { t: 'Val Acc ≈ 88–90%', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Beats MLP (~87%)!', fs: 12, fw: 700, fill: C.ink },
  ]);
  o += card(16, 156, 508, 86, C.lblue, C.blue, [
    { t: '3.3× Fewer Parameters, Higher Generalization Accuracy', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: '61,706 params (LeNet) vs. 203,530 params (MLP) under identical 30-epoch SGD budget', fs: 13, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function lenetFilterFamilies() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 224, C.lblue, C.blue, [
    { t: 'Family 1: Silhouette Detectors', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: '• Smooth 5×5 intensity averagers', fs: 12.5, fw: 600, fill: C.ink },
    { t: '• Highlight filled foreground', fs: 12.5, fill: C.ink },
    { t: '  garment shape (or background)', fs: 12.5, fill: C.muted },
    { t: '• Separates sneaker vs. shirt mass', fs: 12.5, fw: 700, fill: D.blue.dark },
  ]);
  o += card(282, 18, 242, 224, C.lgreen, C.green, [
    { t: 'Family 2: Contour / Edge Filters', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '• Bright columns beside dark cols', fs: 12.5, fw: 600, fill: C.ink },
    { t: '• Learned 5×5 relatives of our', fs: 12.5, fill: C.ink },
    { t: '  hand-crafted [1, −1] kernel!', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '• Discovered from labels alone', fs: 12.5, fill: C.muted },
  ]);
  return svg(W, H, o);
}

function pixelShuffleInvariance() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 16, 242, 232, C.lblue, C.blue, [
    { t: 'MLP on Shuffled Pixels (X·P)', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '(X P) W⁽¹⁾ = X (P W⁽¹⁾)', mono: true, fs: 13.5, fw: 700, fill: C.ink },
    { t: '• Merely permutes rows of W⁽¹⁾', fs: 12.5, fw: 600, fill: C.ink },
    { t: '• i.i.d. init ⇒ identical SGD distribution', fs: 12, fill: C.muted },
    { t: '• Accuracy unchanged: ≈87%', fs: 13, fw: 700, fill: D.blue.dark },
  ]);
  o += card(282, 16, 242, 232, D.red.tint, D.red.base, [
    { t: 'LeNet on Shuffled Pixels (X·P)', fs: 14, fw: 700, fill: D.red.dark },
    { t: '5×5 window reads 25 random pixels', fs: 12.5, fw: 700, fill: D.red.dark },
    { t: '• Destroys 2D spatial locality', fs: 12.5, fw: 600, fill: C.ink },
    { t: '• Shared local filters become useless', fs: 12, fill: C.muted },
    { t: '• Accuracy drops to ~85–86%!', fs: 13, fw: 700, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

function orderedVsShuffledAcc() {
  const W = 540, H = 260;
  let o = '';
  const bars = [
    { y: 24,  label: 'LeNet (Ordered Pixels)', acc: '≈89–90%', w: 300, fill: C.lgreen, st: C.green, dk: D.green.dark },
    { y: 90,  label: 'MLP (Ordered = Shuffled)', acc: '≈87%', w: 230, fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 156, label: 'LeNet (Shuffled Pixels)', acc: '≈85–86%', w: 180, fill: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];
  bars.forEach((b) => {
    o += box(16, b.y, 196, 46, '#FFFFFF', b.st, 6, 1.8);
    o += tx(114, b.y + 23, b.label, { fs: 13, fw: 700, fill: C.ink });
    o += box(222, b.y, b.w, 46, b.fill, b.st, 6, 2);
    o += tx(222 + b.w / 2, b.y + 23, b.acc, { mono: true, fs: 14, fw: 700, fill: b.dk });
  });
  o += tx(270, 232, 'Shuffling pixels erases LeNet’s advantage over the MLP', { fs: 13.5, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

function lenetShiftBreakers() {
  const W = 540, H = 260;
  let o = '';
  const brks = [
    { x: 16,  t: '1. Padded Borders', s1: 'Conv1 uses p = 2 zeros', s2: 'Edge shift changes 0s', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 194, t: '2. Pool Boundaries', s1: '2×2 AvgPool (stride 2)', s2: 'Jumps at 2×2 boundary', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 372, t: '3. Flattened Head', s1: '16×5×5 → 400 → 120', s2: 'Unique weights / cell', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];
  brks.forEach((b) => {
    o += card(b.x, 18, 152, 132, b.fill, b.st, [
      { t: b.t, fs: 14, fw: 700, fill: b.dk },
      { t: b.s1, mono: true, fs: 11.5, fw: 700, fill: C.ink },
      { t: b.s2, fs: 12, fill: C.muted },
    ]);
  });
  o += card(16, 166, 508, 78, C.lgreen, C.green, [
    { t: 'Modern Fix: Replace Flatten+Dense with Global Average Pooling (16×5×5 → 16 → 10)', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Enforces exact spatial invariance across the 5×5 grid while cutting parameters 22.5×', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function lenetToModernEvolution() {
  const W = 540, H = 268;
  let o = '';
  const rows = [
    { comp: 'Activation', old: 'Sigmoid (σ′ ≤ 1/4)', mod: 'ReLU / GELU (slope 1)' },
    { comp: 'Initialization', old: 'Xavier Uniform', mod: 'He (Kaiming) 2/n_in' },
    { comp: 'Downsampling', old: '2×2 Average Pooling', mod: 'MaxPool / Stride-2 Conv' },
    { comp: 'Stabilization', old: 'None (requires lr=2.0)', mod: 'BatchNorm / LayerNorm + Skip' },
    { comp: 'Classifier Head', old: 'Flatten + 3 Dense (96% params)', mod: 'Global Avg Pool + 1 Linear' },
  ];
  rows.forEach((r, i) => {
    const y = 14 + i * 49;
    o += card(16, y, 122, 40, N.panel, C.gray, [{ t: r.comp, fs: 13, fw: 700, fill: C.ink }]);
    o += card(146, y, 174, 40, D.red.tint, D.red.base, [{ t: r.old, fs: 12, fw: 600, fill: D.red.dark }]);
    o += arrow(324, y + 20, 342, y + 20, C.green);
    o += card(346, y, 178, 40, C.lgreen, C.green, [{ t: r.mod, fs: 12, fw: 700, fill: D.green.dark }]);
  });
  return svg(W, H, o);
}

function lenetVsGapHead() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 224, D.red.tint, D.red.base, [
    { t: 'Classical LeNet Dense Head', fs: 14.5, fw: 700, fill: D.red.dark },
    { t: 'Conv Encoder: 2,572 params', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Dense 400→120→84→10:', mono: true, fs: 12.5, fill: C.ink },
    { t: '59,134 head parameters', mono: true, fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'Total: 61,706 parameters', mono: true, fs: 13.5, fw: 700, fill: C.ink },
  ]);
  o += card(282, 18, 242, 224, C.lgreen, C.green, [
    { t: 'LeNet + Global Avg Pool Head', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'Conv Encoder: 2,572 params', mono: true, fs: 12.5, fill: C.ink },
    { t: 'GAP (16×5×5→16) + Linear(16→10):', mono: true, fs: 11.5, fill: C.ink },
    { t: 'Only 170 head parameters!', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Total: 2,742 params (22.5× smaller)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function convLayerSummary() {
  const W = 540, H = 264;
  let o = '';
  const items = [
    { x: 16,  y: 16,  t: '1. 2D Cross-Correlation', s1: 'Y_{i,j} = Σ_{a,b} K_{a,b} X_{i+a,j+b}', s2: 'Output: (n_h−k_h+1) × (n_w−k_w+1)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 278, y: 16,  t: '2. Learned Edge Filters', s1: 'K = [1, −1] finite difference', s2: 'Recovered by gradient descent in 10 steps', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 16,  y: 140, t: '3. im2col Matrix Multiply', s1: 'vec(Y) = P_im2col · vec(K)', s2: 'k_h k_w× patch RAM for GPU GEMM', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 278, y: 140, t: '4. Stacked Receptive Field', s1: 'r = 1 + Σ (k_i − 1) = 2L + 1', s2: '3×(3×3) saves 45% vs 7×7', fill: C.lamber, st: C.amber, dk: D.orange.dark },
  ];
  items.forEach((it) => {
    o += card(it.x, it.y, 246, 108, it.fill, it.st, [
      { t: it.t, fs: 14, fw: 700, fill: it.dk },
      { t: it.s1, mono: true, fs: 11.5, fw: 700, fill: C.ink },
      { t: it.s2, fs: 11.5, fill: C.muted },
    ]);
  });
  return svg(W, H, o);
}

export const diagrams = {
  // index.md (4)
  'ch06-cnn-flatten-vs-grid':          flattenVsGrid,
  'ch06-cnn-roadmap-pipeline':         roadmapPipeline,
  'ch06-cnn-benchmarks-overview':      benchmarksOverview,
  'ch06-cnn-resolution-scaling':       resolutionScaling,
  // why-conv.md (9)
  'ch06-cnn-dense-param-explosion':    denseParamExplosion,
  'ch06-cnn-equivariance-vs-invariance': equivarianceVsInvariance,
  'ch06-cnn-offset-reindexing':        offsetReindexing,
  'ch06-cnn-infinite-grid-shift':      infiniteGridShift,
  'ch06-cnn-impulse-probing':          impulseProbing,
  'ch06-cnn-locality-window':          localityWindow,
  'ch06-cnn-inductive-bias':           inductiveBias,
  'ch06-cnn-conv-vs-xcorr':            convVsXcorr,
  'ch06-cnn-4d-kernel-tensor':         kernel4dTensor,
  'ch06-cnn-1d-2d-3d-modalities':      modalities1d2d3d,
  // conv-layer.md (10)
  'ch06-cnn-vertical-edge-motiv':      verticalEdgeMotiv,
  'ch06-cnn-valid-window-shrinkage':   validWindowShrinkage,
  'ch06-cnn-flipped-kernel-equiv':     flippedKernelEquiv,
  'ch06-cnn-resolution-independence':  resolutionIndependence,
  'ch06-cnn-orientation-selectivity':  orientationSelectivity,
  'ch06-cnn-learning-edge-loop':       learningEdgeLoop,
  'ch06-cnn-kernel-gd-convergence':    kernelGdConvergence,
  'ch06-cnn-im2col-gemm':              im2colGemm,
  'ch06-cnn-stacked-3x3-vs-7x7':       stacked3x3Vs7x7,
  'ch06-cnn-effective-rf-gaussian':    effectiveRfGaussian,
  'ch06-cnn-conv-layer-summary':       convLayerSummary,
  // padding-and-strides.md (7)
  'ch06-cnn-border-erosion':           borderErosion,
  'ch06-cnn-odd-vs-even-kernel':       oddVsEvenKernel,
  'ch06-cnn-padding-dark-frame':       paddingDarkFrame,
  'ch06-cnn-stride-step-derivation':   strideStepDerivation,
  'ch06-cnn-three-stride-regimes':     threeStrideRegimes,
  'ch06-cnn-geometric-rf-stride':      geometricRfStride,
  'ch06-cnn-dilation-gridding':        dilationGridding,
  // channels.md (9)
  'ch06-cnn-channel-scaling-wall':     channelScalingWall,
  'ch06-cnn-color-opponency':          colorOpponency,
  'ch06-cnn-multi-out-4d':             multiOut4d,
  'ch06-cnn-kernel-shift-linearity':   kernelShiftLinearity,
  'ch06-cnn-three-resource-budgets':   threeResourceBudgets,
  'ch06-cnn-memory-vs-params-stages':  memoryVsParamsStages,
  'ch06-cnn-1x1-matmul-reshape':       matmul1x1Reshape,
  'ch06-cnn-1x1-bottleneck-savings':   bottleneck1x1Savings,
  'ch06-cnn-grouped-depthwise-preview': groupedDepthwisePreview,
  // pooling.md (9)
  'ch06-cnn-jitter-sensitivity':       jitterSensitivity,
  'ch06-cnn-max-vs-avg-pool':          maxVsAvgPool,
  'ch06-cnn-block-boundary-shift':     blockBoundaryShift,
  'ch06-cnn-subsampling-aliasing':     subsamplingAliasing,
  'ch06-cnn-global-pooling-collapse':  globalPoolingCollapse,
  'ch06-cnn-nonoverlapping-default-stride': nonoverlappingDefaultStride,
  'ch06-cnn-neg-inf-padding':          negInfPadding,
  'ch06-cnn-pooling-channel-indep':    poolingChannelIndep,
  'ch06-cnn-classical-vs-modern-pooling': classicalVsModernPooling,
  // lenet.md (11)
  'ch06-cnn-mlp-vs-lenet-paradigm':    mlpVsLenetParadigm,
  'ch06-cnn-lenet5-1998-vs-modern':    lenet51998VsModern,
  'ch06-cnn-lenet-param-flop-asymmetry': lenetParamFlopAsymmetry,
  'ch06-cnn-sigmoid-gradient-attenuation': sigmoidGradientAttenuation,
  'ch06-cnn-lenet-training-trajectory': lenetTrainingTrajectory,
  'ch06-cnn-lenet-filter-families':    lenetFilterFamilies,
  'ch06-cnn-pixel-shuffle-invariance': pixelShuffleInvariance,
  'ch06-cnn-ordered-vs-shuffled-acc':  orderedVsShuffledAcc,
  'ch06-cnn-lenet-shift-breakers':     lenetShiftBreakers,
  'ch06-cnn-lenet-to-modern-evolution': lenetToModernEvolution,
  'ch06-cnn-lenet-vs-gap-head':        lenetVsGapHead,
};
