// Slide diagrams for chapter_convolutional-modern (Chapter 7: Modern CNNs)
import { C, TOKENS, FS, FM, grid, tx, baselineY, arrow, block, chip, rc, svg } from './engine.mjs';

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

function fourScalingBarriers() {
  const W = 540, H = 264;
  let o = '';
  const b = [
    { x: 16,  y: 16,  t: '1. Signal & Degradation Drift', s1: 'Sigmoids shrink grads ≤ 1/4^L', s2: 'Plain-56 loss > Plain-20 (0.28>0.20)', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 278, y: 16,  t: '2. 92% Dense-Head Bloat', s1: 'AlexNet/VGG: >92% params in head', s2: 'Contributes <5% of total MACs', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 16,  y: 140, t: '3. FLOPs ≠ GPU Latency', s1: 'Depthwise convs cut MACs 8–9×', s2: 'Only 9 MACs/read → memory-bound', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 278, y: 140, t: '4. Confounded Recipes', s1: 'AdamW + Mixup + smoothing + EMA', s2: 'Adds +4.3% to unchanged ResNet-50', fill: C.lblue, st: C.blue, dk: D.blue.dark },
  ];
  b.forEach((it) => {
    o += card(it.x, it.y, 246, 108, it.fill, it.st, [
      { t: it.t, fs: 14, fw: 700, fill: it.dk },
      { t: it.s1, fs: 12.5, fw: 600, fill: C.ink },
      { t: it.s2, mono: true, fs: 11.5, fw: 600, fill: C.muted },
    ]);
  });
  return svg(W, H, o);
}

function decadeEvolutionArc() {
  const W = 540, H = 272;
  let o = '';
  const stages = [
    { x: 16,  y: 18, t: '§7.1 AlexNet (2012)', s1: 'ReLU, Dropout, GPUs', s2: '15.3% top-5 error', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, y: 18, t: '§7.2 VGG/NiN/GoogLeNet', s1: '3×3 Blocks, 1×1, GAP', s2: 'Stem–Body–Head', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 372, y: 18, t: '§7.3 Norm Layers', s1: 'BatchNorm, LayerNorm', s2: 'GroupNorm, 300× LR', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 372, y: 152, t: '§7.4 ResNet / ResNeXt', s1: 'Skip x + g(x), Groups', s2: '152+ layers, DenseNet', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 194, y: 152, t: '§§7.5–7.6 Mobile & Recipes', s1: 'Depthwise, RepVGG, SE', s2: 'AdamW, Mixup, EMA', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 16,  y: 152, t: '§§7.7–7.8 ConvNeXt/RegNet', s1: '7×7 DW, 1 GELU/LN', s2: 'Design-space CDFs', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  const bw = 152, bh = 98;
  o += arrow(16 + bw + 4, 67, 194 - 4, 67, C.blue);
  o += arrow(194 + bw + 4, 67, 372 - 4, 67, C.purple);
  o += arrow(372 + bw / 2, 18 + bh + 4, 372 + bw / 2, 152 - 4, C.amber);
  o += arrow(372 - 4, 201, 194 + bw + 4, 201, C.amber);
  o += arrow(194 - 4, 201, 16 + bw + 4, 201, C.green);
  stages.forEach((s) => {
    o += card(s.x, s.y, bw, bh, s.fill, s.st, [
      { t: s.t, fs: 13, fw: 700, fill: s.dk },
      { t: s.s1, fs: 12, fw: 600, fill: C.ink },
      { t: s.s2, mono: true, fs: 11.5, fill: C.muted },
    ]);
  });
  return svg(W, H, o);
}

function threeUnifyingInvariants() {
  const W = 540, H = 260;
  let o = '';
  const invs = [
    { x: 16,  t: '1. RF per FLOP & GAP', s1: '2×(3×3) = 18c² vs 25c²', s2: '3×(3×3) = 27c² vs 49c²', s3: 'GAP removes 43M head', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: '2. Scale & Nesting', s1: 'BN(a wᵀ h) = BN(wᵀ h)', s2: 'f(x) = x + g(x) (g=0 → I)', s3: 'Nested classes F ⊆ F′', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 372, t: '3. Factorization Ratio', s1: 'DWS / Dense = 1/c_o + 1/k²', s2: '≈ 1/9 MACs for 3×3', s3: '9 vs 9c_o MACs/read', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  invs.forEach((it) => {
    o += card(it.x, 18, 152, 224, it.fill, it.st, [
      { t: it.t, fs: 13.5, fw: 700, fill: it.dk },
      { t: it.s1, mono: true, fs: 11.5, fw: 700, fill: C.ink },
      { t: it.s2, mono: true, fs: 11.5, fw: 600, fill: C.ink },
      { t: it.s3, fs: 12, fill: C.muted },
    ]);
  });
  return svg(W, H, o);
}

function paretoScoreboard() {
  const W = 540, H = 268;
  let o = '';
  const rows = [
    { m: 'AlexNet (§7.1)', p: '46.8M params', c: '938M MACs', a: '91% (10 ep)', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { m: 'Quarter-VGG (§7.2)', p: '43.1M params', c: '512M MACs', a: '93% (10 ep)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { m: 'GoogLeNet (§7.2)', p: '6.0M params', c: '276M MACs', a: '91% (10 ep)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { m: 'ResNet-18 (§§7.4, 7.6)', p: '11.2M params', c: '319M MACs', a: '93–94%', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { m: 'ConvNeXt-Atto (§7.7)', p: '3.38M params', c: '46M MACs', a: '94% (ResNet stem)', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { m: 'RegNetX32 (§7.8)', p: '0.52M params!', c: '112M MACs', a: '94% (<5% weights!)', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  rows.forEach((r, i) => {
    const y = 12 + i * 41;
    o += card(16, y, 166, 34, '#FFFFFF', r.st, [{ t: r.m, fs: 12.5, fw: 700, fill: C.ink }]);
    o += card(190, y, 108, 34, r.fill, r.st, [{ t: r.p, mono: true, fs: 11.5, fw: 700, fill: r.dk }]);
    o += card(306, y, 96, 34, N.panel, C.gray, [{ t: r.c, mono: true, fs: 11.5, fw: 600, fill: C.ink }]);
    o += card(410, y, 114, 34, r.fill, r.st, [{ t: r.a, fs: 11.5, fw: 700, fill: r.dk }]);
  });
  return svg(W, H, o);
}

// ── alexnet.md diagrams (7) ──────────────────────────────────────────────

function handcraftedVsE2e() {
  const W = 540, H = 260;
  let o = '';
  // Top: Hand-crafted
  o += box(16, 16, 508, 104, D.red.tint, D.red.base, 10, 2);
  o += tx(270, 36, 'Pre-2012 Pipeline: Fixed Hand-Crafted Features (26.2% Top-5 Error)', { fs: 13.5, fw: 700, fill: D.red.dark });
  o += card(28, 52, 134, 54, '#FFFFFF', C.gray, [{ t: 'Raw Pixels X', fs: 13, fw: 700 }]);
  o += arrow(168, 79, 194, 79, C.gray);
  o += card(200, 52, 156, 54, N.panel, C.gray, [
    { t: 'Frozen SIFT / HOG', fs: 13, fw: 700, fill: C.muted },
    { t: 'No gradient updates!', fs: 11.5, fill: D.red.dark },
  ]);
  o += arrow(362, 79, 388, 79, D.red.base);
  o += card(394, 52, 118, 54, '#FFFFFF', D.red.base, [{ t: 'Trainable SVM', fs: 13, fw: 700, fill: D.red.dark }]);

  // Bottom: AlexNet E2E
  o += box(16, 136, 508, 108, C.lgreen, C.green, 10, 2);
  o += tx(270, 156, 'AlexNet (2012): End-to-End Learned Hierarchy (15.3% Top-5 Error!)', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += card(28, 174, 134, 54, '#FFFFFF', C.blue, [{ t: '224×224 RGB', fs: 13, fw: 700, fill: D.blue.dark }]);
  o += arrow(168, 201, 194, 201, C.green);
  o += card(200, 174, 156, 54, '#FFFFFF', C.green, [
    { t: '5 Conv Stages', fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Edges → Parts → Objects', fs: 11.5, fill: C.ink },
  ]);
  o += arrow(362, 201, 388, 201, C.green);
  o += card(394, 174, 118, 54, '#FFFFFF', C.green, [
    { t: '3 Dense + Loss', fs: 12.5, fw: 700, fill: D.green.dark },
    { t: '← ∇L updates all', mono: true, fs: 11, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

function imagenetAugmentation() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 20, 152, 116, C.lblue, C.blue, [
    { t: '1.2M ImageNet Photos', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: '1,000 categories', fs: 12.5, fw: 600, fill: C.ink },
    { t: '256 × 256 base frame', mono: true, fs: 12, fill: C.muted },
  ]);
  o += arrow(174, 78, 194, 78, C.purple);
  o += card(200, 20, 156, 116, D.purple.tint, C.purple, [
    { t: 'Geometric Views', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: '32×32 crops (224×224)', mono: true, fs: 11.5, fw: 700, fill: C.ink },
    { t: '× 2 horiz reflections', fs: 12, fill: C.ink },
    { t: '= 2,048 views / image!', fs: 12.5, fw: 700, fill: D.purple.dark },
  ]);
  o += arrow(362, 78, 382, 78, C.green);
  o += card(388, 20, 136, 116, C.lgreen, C.green, [
    { t: 'PCA Color Jitter', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Perturbs RGB', fs: 12.5, fill: C.ink },
    { t: 'principal axes', fs: 12.5, fill: C.ink },
    { t: 'Lighting invariant', fs: 12, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 154, 508, 86, C.lamber, C.amber, [
    { t: 'Why Augmentation + Dropout Were Mandatory for 60M Parameters', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'Without 2,048× crop/flip diversity and Dropout(0.5), AlexNet memorizes 1.2M images within epochs', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function gpuVsCpuBudget() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 16, 508, 72, D.purple.tint, C.purple, [
    { t: '90-Epoch ImageNet Training Arithmetic Budget:  6.5 × 10¹⁷ FLOPs', mono: true, fs: 14, fw: 700, fill: D.purple.dark },
    { t: '10⁹ MACs/img × 3 (fwd+bwd) × 2 FLOPs/MAC × 1.2×10⁶ images × 90 epochs', mono: true, fs: 12, fw: 600, fill: C.ink },
  ]);
  o += card(16, 104, 242, 138, D.red.tint, D.red.base, [
    { t: '2012 Quad-Core CPU', fs: 14.5, fw: 700, fill: D.red.dark },
    { t: 'Peak: ~0.2 TFLOPS', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'DRAM Bandwidth: ~20 GB/s', mono: true, fs: 12, fill: C.muted },
    { t: 'Wall-Clock: > 45 Days!', fs: 13.5, fw: 700, fill: D.red.dark },
  ]);
  o += card(282, 104, 242, 138, C.lgreen, C.green, [
    { t: '2× NVIDIA GTX 580 GPUs', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'Peak: 2 × 1.6 = 3.2 TFLOPS', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'VRAM Bandwidth: 2 × 192 GB/s', mono: true, fs: 12, fill: C.muted },
    { t: 'Wall-Clock: 5–6 Days (16× peak FLOPS)', fs: 13.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function lenetVsAlexnetScale() {
  const W = 540, H = 264;
  let o = '';
  const rows = [
    { dim: 'Input & Trunk', l: '1×28×28 · 2 Convs (6–16ch)', a: '224×224 · 5 Convs (96–384ch)' },
    { dim: 'Activation', l: 'Sigmoid σ(x) (σ′ ≤ 1/4)', a: 'ReLU max(x,0) (6× faster)' },
    { dim: 'Pooling', l: '2×2 AvgPool (stride 2)', a: '3×3 Overlapping MaxPool (s=2)' },
    { dim: 'Dense Head', l: '120 → 84 → 10 (no dropout)', a: '4096 → 4096 → 10 + Dropout(0.5)' },
    { dim: 'Total Cost', l: '61.7K params · 0.42M MACs', a: '46.8M params (760×) · 938M MACs' },
  ];
  rows.forEach((r, i) => {
    const y = 14 + i * 48;
    o += card(16, y, 114, 38, N.panel, C.gray, [{ t: r.dim, fs: 12.5, fw: 700, fill: C.ink }]);
    o += card(136, y, 182, 38, C.lblue, C.blue, [{ t: r.l, fs: 11.5, fw: 600, fill: D.blue.dark }]);
    o += card(324, y, 200, 38, C.lgreen, C.green, [{ t: r.a, fs: 11.5, fw: 700, fill: D.green.dark }]);
  });
  return svg(W, H, o);
}

function reluHeDropoutTriad() {
  const W = 540, H = 260;
  let o = '';
  const cols = [
    { x: 16,  t: '1. ReLU Activation', s1: 'max(x, 0)', s2: 'Slope = 1 for x > 0', s3: 'Stops 4^{-K} decay', s4: '6× faster than tanh', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: '2. He Initialization', s1: 'W ~ N(0, 2/n_in)', s2: 'n_in = c_i · k²', s3: 'Compensates for ReLU', s4: 'zeroing half of inputs', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 372, t: '3. Dropout (p = 0.5)', s1: 'Zeroes 50% of 4096', s2: 'Scales survivors ×2', s3: 'Prevents 43M head', s4: 'from memorizing data', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  cols.forEach((c) => {
    o += card(c.x, 18, 152, 224, c.fill, c.st, [
      { t: c.t, fs: 13.5, fw: 700, fill: c.dk },
      { t: c.s1, mono: true, fs: 12.5, fw: 700, fill: C.ink },
      { t: c.s2, mono: true, fs: 11.5, fw: 600, fill: c.dk },
      { t: c.s3, fs: 12, fill: C.ink },
      { t: c.s4, fs: 12, fill: C.muted },
    ]);
  });
  return svg(W, H, o);
}

function alexnetParamMacSplit() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 16, 242, 134, C.lgreen, C.green, [
    { t: '5 Convolutional Layers (Trunk)', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Parameters: 3.72M (8.0%)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Multiply-Adds: 895.1M (95.4%!)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Conv1 Intensity: 724 MACs / byte', mono: true, fs: 11.5, fill: C.muted },
  ]);
  o += card(282, 16, 242, 134, D.red.tint, D.red.base, [
    { t: '3 Fully Connected Layers (Head)', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'Parameters: 43.04M (92.0%!)', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Multiply-Adds: 43.0M (4.6%)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Dense1 Intensity: 0.25 MACs / byte!', mono: true, fs: 11.5, fill: D.red.dark },
  ]);
  o += card(16, 166, 508, 82, C.lamber, C.amber, [
    { t: 'Batch-1 Roofline: 172 MB of Dense Weights Streamed for 43M MACs', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Dense1 (6400×4096 = 26.2M params) holds 56% of the entire network just to read the 5×5 grid', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function cosineWarmupCurve() {
  const W = 540, H = 260;
  let o = '';
  // Draw clean axes and schedule curve with no text intersecting the curve!
  const x0 = 56, y0 = 186, w = 440, h = 136;
  const xw = x0 + 64; // warmup end t_w = 0.1 T
  o += line(x0, y0 - h, x0, y0, C.gray, 2);
  o += line(x0, y0, x0 + w, y0, C.gray, 2);
  o += line(xw, y0 - h, xw, y0, N.hairline, 1.5, true);

  // Polyline for linear warmup + cosine decay
  let pts = `${x0},${y0} ${xw},${y0 - h}`;
  for (let i = 1; i <= 30; i++) {
    const p = i / 30;
    const px = xw + p * (w - 64);
    const py = y0 - h * 0.5 * (1 + Math.cos(Math.PI * p));
    pts += ` ${px.toFixed(1)},${py.toFixed(1)}`;
  }
  o += `<polyline points="${pts}" fill="none" stroke="${C.blue}" stroke-width="3"/>`;

  // Labels placed well clear of lines and curve
  o += tx(x0 - 10, y0 - h, 'η₀', { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark, anchor: 'end' });
  o += tx(x0 - 10, y0, '0', { mono: true, fs: 13, fill: C.muted, anchor: 'end' });
  o += tx(x0, y0 + 18, '0', { mono: true, fs: 12.5, fill: C.ink });
  o += tx(xw, y0 + 18, 't_w = T/10', { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });
  o += tx(x0 + w, y0 + 18, 'T = 4,690', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx(xw + 150, y0 - h - 16, 'Half-Cosine Decay: ½ η₀ (1 + cos(π(t−t_w)/(T−t_w)))', { mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  o += tx(270, 238, 'Warmup prevents early step-0 divergence; cosine decay settles iterates smoothly', { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

// ── blocks.md diagrams (6) ───────────────────────────────────────────────

function blocksThreeSolutions() {
  const W = 540, H = 260;
  let o = '';
  const sols = [
    { x: 16,  t: '1. VGG (2014)', prob: 'Fixes ad-hoc layer design', mech: 'Repeatable 3×3 blocks', leg: 'Topology as data (`arch`)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: '2. NiN (2013)', prob: 'Fixes 43M dense head', mech: '1×1 MLP + Global Avg Pool', leg: '46.8M → 3.4M params (14×)', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 372, t: '3. GoogLeNet (2014)', prob: 'Fixes single-scale cost', mech: '4 branches + 1×1 bottlenecks', leg: 'Stem–Body–Head template', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  sols.forEach((s) => {
    o += card(s.x, 18, 152, 224, s.fill, s.st, [
      { t: s.t, fs: 14, fw: 700, fill: s.dk },
      { t: s.prob, fs: 12, fw: 600, fill: C.muted },
      { t: s.mech, fs: 12.5, fw: 700, fill: C.ink },
      { t: s.leg, mono: true, fs: 11.5, fw: 700, fill: s.dk },
    ]);
  });
  return svg(W, H, o);
}

function vggRfEfficiency() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 16, 242, 112, C.lblue, C.blue, [
    { t: '5×5 Receptive Field', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: '1×(5×5): 25c² (1 ReLU)', mono: true, fs: 12, fill: C.ink },
    { t: '2×(3×3): 18c² (2 ReLUs)', mono: true, fs: 12, fw: 700, fill: D.blue.dark },
    { t: '28% fewer weights & FLOPs!', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  o += card(282, 16, 242, 112, C.lgreen, C.green, [
    { t: '7×7 Receptive Field', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '1×(7×7): 49c² (1 ReLU)', mono: true, fs: 12, fill: C.ink },
    { t: '3×(3×3): 27c² (3 ReLUs)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: '45% fewer weights & FLOPs!', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 146, 508, 100, D.purple.tint, C.purple, [
    { t: 'Constant Per-Stage Compute Invariant in VGG', fs: 14.5, fw: 700, fill: D.purple.dark },
    { t: 'Each stride-2 max-pool quarters the area H·W while the channels double (c² × 4)', fs: 12.5, fw: 600, fill: C.ink },
    { t: 'Per-layer MACs (H/2)(W/2) · 9(2c)² = HW · 9c² stay constant!', mono: true, fs: 12, fw: 700, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function ninPerPixelMlp() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 32, 142, 104, C.lblue, C.blue, [
    { t: 'k×k Spatial Conv', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: '+ ReLU', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Local patch filter', fs: 12, fill: C.muted },
  ]);
  o += arrow(164, 84, 190, 84, C.purple);
  o += card(196, 32, 146, 104, D.purple.tint, C.purple, [
    { t: '1×1 Conv + ReLU', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'c_o × c_o per pixel', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'Hidden MLP layer 1', fs: 12, fill: C.muted },
  ]);
  o += arrow(348, 84, 374, 84, C.green);
  o += card(380, 32, 144, 104, C.lgreen, C.green, [
    { t: '1×1 Conv (+ ReLU)', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: '(No ReLU on last', fs: 12, fw: 700, fill: D.red.dark },
    { t: 'K-class logit block!)', fs: 11.5, fw: 700, fill: D.red.dark },
  ]);
  o += card(16, 156, 508, 82, C.lamber, C.amber, [
    { t: 'Network in Network (NiN Block):  Shared 2-Layer MLP Evaluated at Every Pixel (x, y)', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Enriches local nonlinear feature interaction across channels without widening spatial kernel k', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function vggVsPooledVgg() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 224, D.red.tint, D.red.base, [
    { t: 'Quarter-VGG (Dense Head)', fs: 14.5, fw: 700, fill: D.red.dark },
    { t: 'Conv Trunk: 0.6M params', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Dense Head (6272→4096→4096→10):', mono: true, fs: 11.5, fill: C.ink },
    { t: '42.5M head params (98.6%!)', mono: true, fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Val Acc (η₀=0.03): ~92.0%', fs: 13, fw: 700, fill: C.ink },
  ]);
  o += card(282, 18, 242, 224, C.lgreen, C.green, [
    { t: 'PooledVGG (Global Avg Pool)', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'Same Conv Trunk: 0.6M params', mono: true, fs: 12.5, fill: C.ink },
    { t: 'GAP (128×7×7→128) + Linear(128→10):', mono: true, fs: 11, fill: C.ink },
    { t: 'Only 1,290 head params (99% cut!)', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Val Acc (η₀=0.03): ~89.5–90%', fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function inceptionBottleneckMath() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 134, D.red.tint, D.red.base, [
    { t: 'Direct 5×5 Branch (No Bottleneck)', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: '192 ch → 32 ch at 28×28', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Weights: 25·192·32 = 153,600', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'MACs: 28² · 153,600 = 120.4M', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
  ]);
  o += card(282, 18, 242, 134, C.lgreen, C.green, [
    { t: 'With 16-ch 1×1 Bottleneck', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: '192 → 16 (1×1) → 32 (5×5)', mono: true, fs: 12.5, fill: C.ink },
    { t: 'Weights: 3,072 + 12,800 = 15,872', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'MACs: 28² · 15,872 = 12.4M', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 168, 508, 76, C.lamber, C.amber, [
    { t: '9.7× Reduction in Both Parameters and Multiply-Adds!', fs: 14.5, fw: 700, fill: D.orange.dark },
    { t: 'Compressing channels via 1×1 projection before 3×3 and 5×5 filters makes multi-branch blocks cheap', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function stemBodyHeadTemplate() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 24, 152, 118, C.lblue, C.blue, [
    { t: '1. Stem', fs: 15, fw: 700, fill: D.blue.dark },
    { t: '7×7_s2 + MaxPool', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'Shrinks H×W 4×–8×', fs: 12, fill: C.ink },
    { t: 'Cuts activation RAM', fs: 11.5, fill: C.muted },
  ]);
  o += arrow(172, 83, 190, 83, C.purple);
  o += card(194, 24, 152, 118, D.purple.tint, C.purple, [
    { t: '2. Stage Body', fs: 15, fw: 700, fill: D.purple.dark },
    { t: '3–4 Stages of Blocks', fs: 12, fw: 700, fill: C.ink },
    { t: 'Halve H×W, double C', fs: 12, fill: C.ink },
    { t: '90%+ of total MACs', fs: 11.5, fill: C.muted },
  ]);
  o += arrow(350, 83, 368, 83, C.green);
  o += card(372, 24, 152, 118, C.lgreen, C.green, [
    { t: '3. Pooled Head', fs: 15, fw: 700, fill: D.green.dark },
    { t: 'Global Avg Pool', fs: 12, fw: 700, fill: C.ink },
    { t: '+ 1 Linear Layer', fs: 12, fill: C.ink },
    { t: '<1% of parameters!', fs: 11.5, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 160, 508, 80, C.lamber, C.amber, [
    { t: 'Universal Macro Blueprint Inherited by ResNet, MobileNet, ConvNeXt, and RegNet', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Every modern backbone varies only the stem granularity and the internal stage block topology', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ── batch-norm.md diagrams (10) ──────────────────────────────────────────

function unnormalizedScaleDrift() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 224, D.red.tint, D.red.base, [
    { t: 'Unnormalized Deep Stack', fs: 14.5, fw: 700, fill: D.red.dark },
    { t: '• Step 0: Xavier/He balances σ²', fs: 12.5, fw: 600, fill: C.ink },
    { t: '• Step t > 0: weight updates shift', fs: 12.5, fill: C.ink },
    { t: '  each layer’s mean & variance', fs: 12.5, fill: C.ink },
    { t: '• Narrow usable LR band (~3×)', fs: 13, fw: 700, fill: D.red.dark },
  ]);
  o += card(282, 18, 242, 224, C.lgreen, C.green, [
    { t: 'Normalized Stack (BatchNorm)', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: '• Enforces μ = 0, σ² = 1 at every step', fs: 12.5, fw: 600, fill: C.ink },
    { t: '• Learnable γ, β restore flexibility', fs: 12.5, fill: C.ink },
    { t: '• Scale-invariant weights: L(aw)=L(w)', mono: true, fs: 11.5, fill: C.ink },
    { t: '• 300× usable LR band [0.03, 10]!', fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function bnForwardPipeline() {
  const W = 540, H = 256;
  let o = '';
  const steps = [
    { x: 16,  t: '1. Minibatch B', s1: '{x₁, …, x_m}', s2: 'Pre-activations', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 146, t: '2. Moments', s1: 'μ̂_B = mean(x)', s2: 'σ̂²_B = var(x)', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 276, t: '3. Standardize', s1: 'x̂ = (x − μ̂_B)', s2: '÷ √(σ̂²_B + ε)', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 406, t: '4. Scale & Shift', s1: 'y = γ x̂ + β', s2: 'Learned γ, β', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  steps.forEach((s, i) => {
    o += card(s.x, 26, 118, 106, s.fill, s.st, [
      { t: s.t, fs: 13, fw: 700, fill: s.dk },
      { t: s.s1, mono: true, fs: 11.5, fw: 700, fill: C.ink },
      { t: s.s2, mono: true, fs: 11.5, fill: C.muted },
    ]);
    if (i < 3) o += arrow(s.x + 120, 79, s.x + 128, 79, C.gray);
  });
  o += card(16, 152, 508, 86, C.lblue, C.blue, [
    { t: 'Placement:  h = φ( BN(W x) )   — Right Before Nonlinearity φ', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'ε = 10⁻⁵ prevents division by zero;  setting γ = σ̂_B, β = μ̂_B recovers the unnormalized feature', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function bnBiasCancellation() {
  const W = 540, H = 260;
  let o = '';
  const inputs = [
    { y: 18,  t: 'x = (2, 4, 6, 8)', st: 'μ̂ = 5, σ̂ = √5', col: C.blue, bg: C.lblue, dk: D.blue.dark },
    { y: 84,  t: '10x = (20, 40, 60, 80)', st: 'μ̂ = 50, σ̂ = 10√5', col: C.purple, bg: D.purple.tint, dk: D.purple.dark },
    { y: 150, t: 'x + 100 = (102, 104, 106, 108)', st: 'μ̂ = 105, σ̂ = √5', col: C.amber, bg: C.lamber, dk: D.orange.dark },
  ];
  inputs.forEach((inp) => {
    o += card(16, inp.y, 236, 54, inp.bg, inp.col, [
      { t: inp.t, mono: true, fs: 12, fw: 700, fill: inp.dk },
      { t: inp.st, mono: true, fs: 11.5, fill: C.ink },
    ]);
    o += arrow(258, inp.y + 27, 294, 111, inp.col);
  });
  o += card(300, 44, 224, 134, C.lgreen, C.green, [
    { t: 'Identical Standardized x̂', fs: 14, fw: 700, fill: D.green.dark },
    { t: '(−1.34, −0.45, 0.45, 1.34)', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Any pre-BN bias +b cancels', fs: 12, fw: 600, fill: D.red.dark },
    { t: 'in (x_i + b) − (μ̂_B + b)!', mono: true, fs: 12, fw: 700, fill: D.red.dark },
  ]);
  o += tx(270, 234, 'Always disable the bias of layers immediately preceding BatchNorm', { fs: 13, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

function bnOrthogonalGradient() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 132, C.lblue, C.blue, [
    { t: '1. Scale Invariance', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'BN(a wᵀ h) = BN(wᵀ h)', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: '⇒ ∇_w L(a w) = (1/a) ∇_w L(w)', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'Larger weight norm ⇒ smaller grad', fs: 12, fill: C.muted },
  ]);
  o += card(282, 18, 242, 132, C.lgreen, C.green, [
    { t: '2. Orthogonal Gradient g ⊥ w', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'wᵀ g = 0  along constant ray', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '‖w − η g‖² = ‖w‖² + η² ‖g‖²', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Every SGD step grows ‖w‖!', fs: 12, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 166, 508, 78, D.purple.tint, C.purple, [
    { t: 'Self-Stabilizing Feedback Loop (Arora et al., 2019)', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'Too-large LR η rapidly inflates ‖w‖, which automatically shrinks effective step size ∝ η / ‖w‖', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function bnDenseVsConv() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 224, C.lblue, C.blue, [
    { t: 'BatchNorm on a Dense Layer', fs: 14.5, fw: 700, fill: D.blue.dark },
    { t: 'Input shape: (m, d)', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Reduce over the batch axis', mono: true, fs: 12.5, fill: D.blue.dark },
    { t: '1 mean & var per unit j ∈ {1..d}', fs: 12.5, fill: C.ink },
    { t: 'Pools m = 128 values / feature', mono: true, fs: 12, fill: C.muted },
  ]);
  o += card(282, 18, 242, 224, C.lgreen, C.green, [
    { t: 'BatchNorm on a Conv Layer', fs: 14.5, fw: 700, fill: D.green.dark },
    { t: 'Input: m × (c channels, H × W)', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Reduce over batch, H and W', mono: true, fs: 12.5, fill: D.green.dark },
    { t: '1 mean, var, γ, β per channel c', fs: 12.5, fill: C.ink },
    { t: 'Pools m·H·W = 100,352 values!', mono: true, fs: 12, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function bnTrainVsEvalMode() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 16, 242, 142, C.lblue, C.blue, [
    { t: 'Training Mode (training=True)', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '1. Normalize by batch (μ̂_B, σ̂²_B)', fs: 12.5, fw: 600, fill: C.ink },
    { t: '2. Update running EMA buffers:', fs: 12, fill: C.muted },
    { t: 'μ̂ ← ρ μ̂ + (1−ρ) μ̂_B  (ρ = 0.9)', mono: true, fs: 11.5, fw: 700, fill: D.blue.dark },
    { t: 'Effective window: 1/(1−ρ) = 10 steps', fs: 11.5, fill: C.ink },
  ]);
  o += card(282, 16, 242, 142, C.lgreen, C.green, [
    { t: 'Prediction Mode (training=False)', fs: 14, fw: 700, fill: D.green.dark },
    { t: '1. Freeze running (μ̂, σ̂²)', fs: 12.5, fw: 600, fill: C.ink },
    { t: '2. Deterministic affine map:', fs: 12, fill: C.muted },
    { t: 'y = γ (x − μ̂)/√(σ̂² + ε) + β', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    { t: 'Works at batch size m = 1!', fs: 12, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 172, 508, 76, D.red.tint, D.red.base, [
    { t: 'API pitfall: a momentum of 0.1 means 1 − ρ in some frameworks; 0.9 means ρ in others', fs: 13, fw: 700, fill: D.red.dark },
    { t: 'Cosine LR decay to 0 lets weights settle at the end of training so running stats lock on cleanly', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function bnDistributionShift() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 28, 152, 108, C.lamber, C.amber, [
    { t: 'Input Batch X', fs: 14, fw: 700, fill: D.orange.dark },
    { t: 'X ~ N(2.0, 3.0²)', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: '128 × 6 ch × 28×28', mono: true, fs: 12, fill: C.muted },
  ]);
  o += arrow(172, 82, 190, 82, C.blue);
  o += card(194, 28, 152, 108, C.lblue, C.blue, [
    { t: 'BatchNorm(6)', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'γ = 1, β = 0', mono: true, fs: 12.5, fw: 600, fill: C.ink },
    { t: 'ρ = 0.9, ε = 10⁻⁵', mono: true, fs: 12, fill: C.muted },
  ]);
  o += arrow(350, 82, 368, 82, C.green);
  o += card(372, 28, 152, 108, C.lgreen, C.green, [
    { t: 'Normalized Y', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'mean = 0.000', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'var  = 1.000', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 154, 508, 84, D.purple.tint, C.purple, [
    { t: 'After 1 Step:  moving_mean = 0.9(0) + 0.1(2.0) = 0.20', mono: true, fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'Matches the framework BatchNorm layer within 10⁻⁶ single-precision floating-point roundoff', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function bnSmallBatchCollapse() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 132, C.lblue, C.blue, [
    { t: 'Large Batch (32 images / GPU)', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'BatchNorm Top-1 Error: 23.6%', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark },
    { t: 'GroupNorm Top-1 Error: 24.1%', mono: true, fs: 12.5, fw: 600, fill: C.ink },
    { t: 'Both estimate moments accurately', fs: 12, fill: C.muted },
  ]);
  o += card(282, 18, 242, 132, D.red.tint, D.red.base, [
    { t: 'Micro-Batch (2 images / GPU)', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'BatchNorm: 34.7%! (+11%)', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'GroupNorm Holds Steady: 24.1%!', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'At N=1, Dense BN outputs ≡ β!', fs: 12, fw: 700, fill: D.red.dark },
  ]);
  o += card(16, 166, 508, 78, C.lgreen, C.green, [
    { t: 'Why Single-Example Normalizers (LN / GN) Rule Detection, Segmentation & Transformers', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Computing (μ̂, σ̂²) within each example n removes batch coupling and train/eval mode mismatch', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function nfnetWeightStd() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 20, 242, 126, D.purple.tint, C.purple, [
    { t: '1. Scaled Weight Standardization', fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Ŵ = γ (W − μ_W)/(√N_in σ_W)', mono: true, fs: 11.5, fw: 700, fill: C.ink },
    { t: 'Normalizes filter weights directly', fs: 12, fill: C.ink },
    { t: 'Preserves variance at every step', fs: 12, fill: C.muted },
  ]);
  o += card(282, 20, 242, 126, C.lamber, C.amber, [
    { t: '2. Adaptive Gradient Clipping', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Clips ‖G_l‖_F / ‖W_l‖_F ≤ λ', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Replaces BN’s self-stabilizing', fs: 12, fill: C.ink },
    { t: 'large-learning-rate feedback loop', fs: 12, fill: C.muted },
  ]);
  o += card(16, 162, 508, 78, C.lgreen, C.green, [
    { t: 'NFNet Takeaway (Brock et al., 2021):  Scale Control Is the Core Job of Normalization', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Controlling variance through weights instead of activations trains deep ResNets without BatchNorm', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function normalizerDecisionTree() {
  const W = 540, H = 264;
  let o = '';
  const items = [
    { x: 16,  y: 16,  t: 'BatchNorm (BN)', axes: 'Axes: (N, H, W) per C', use: 'Standard CNNs (N ≥ 16); folds into conv', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 278, y: 16,  t: 'GroupNorm (GN, G=32)', axes: 'Axes: (C/G, H, W) per N', use: 'Detection & seg (N = 1–2 images/GPU)', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 16,  y: 140, t: 'Per-Token / Pixel LN', axes: 'Axes: (C,) at each (N,H,W)', use: 'ViT & ConvNeXt (no buffers, train = eval)', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 278, y: 140, t: 'InstanceNorm (G = C)', axes: 'Axes: (H, W) per (N, C)', use: 'Style transfer & gen (per-map contrast)', fill: C.lamber, st: C.amber, dk: D.orange.dark },
  ];
  items.forEach((it) => {
    o += card(it.x, it.y, 246, 108, it.fill, it.st, [
      { t: it.t, fs: 14, fw: 700, fill: it.dk },
      { t: it.axes, mono: true, fs: 12, fw: 700, fill: C.ink },
      { t: it.use, fs: 11.5, fill: C.muted },
    ]);
  });
  return svg(W, H, o);
}

// ── resnet.md diagrams (5) ───────────────────────────────────────────────

function degradationParadox() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 134, D.red.tint, D.red.base, [
    { t: 'Empirical Plain-Net Paradox', fs: 14, fw: 700, fill: D.red.dark },
    { t: '20-Layer Plain Train Loss: 0.20', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: '56-Layer Plain Train Loss: ≈0.28!', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Worse on TRAINING set (not overfitting)', fs: 11.5, fw: 700, fill: D.red.dark },
  ]);
  o += card(282, 18, 242, 134, C.lgreen, C.green, [
    { t: 'Constructive Existence Proof', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'First 20 layers: copy 20-layer net', fs: 12.5, fw: 600, fill: C.ink },
    { t: 'Remaining 36 layers: identity f(x)=x', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: '⇒ 56-layer loss MUST be ≤ 0.20!', fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 168, 508, 76, C.lamber, C.amber, [
    { t: 'Why Plain SGD Fails:  Identity f(x) = x Is Hard to Approximate Across 36 Nonlinear Layers', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Reparameterizing f(x) = x + g(x) places the exact identity at the trivial zero origin g(x) = 0', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function residualGradientHighway() {
  const W = 540, H = 264;
  let o = '';
  // Top forward recursion
  o += box(16, 16, 508, 108, C.lblue, C.blue, 10, 2);
  o += tx(270, 38, 'Forward Telescoping Sum:  x_L = x_ℓ + Σ_{k=ℓ+1}^L g_k(x_{k-1})', { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });
  o += card(32, 54, 128, 54, '#FFFFFF', C.blue, [{ t: 'Early Feature x_ℓ', mono: true, fs: 12.5, fw: 700 }]);
  o += arrow(166, 81, 198, 81, C.blue);
  o += card(204, 54, 146, 54, '#FFFFFF', C.purple, [{ t: '+ Σ Residual Branches', fs: 12.5, fw: 700, fill: D.purple.dark }]);
  o += arrow(356, 81, 388, 81, C.blue);
  o += card(394, 54, 114, 54, '#FFFFFF', C.blue, [{ t: 'Deep Output x_L', mono: true, fs: 12.5, fw: 700 }]);

  // Bottom backward highway
  o += card(16, 140, 508, 108, C.lgreen, C.green, [
    { t: 'Backward Identity Highway:  ∂L/∂x_ℓ = (∂L/∂x_L) · (I + ∂/∂x_ℓ Σ g_k)', mono: true, fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Additive identity I routes upstream gradient ∂L/∂x_L straight to x_ℓ', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'without multiplying through L − ℓ intermediate weight matrices!', fs: 12, fw: 600, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function residualShapeTransition() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 18, 242, 220, C.lblue, C.blue, [
    { t: 'Identity Block: Residual(3)', mono: true, fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'use_1x1conv=False, strides=1', mono: true, fs: 12, fill: C.muted },
    { t: 'Input:  4 × 3 ch × 6×6', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Shortcut: direct X (0 params)', fs: 12.5, fw: 600, fill: D.blue.dark },
    { t: 'Output: 4 × 3 ch × 6×6', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
  ]);
  o += card(282, 18, 242, 220, C.lgreen, C.green, [
    { t: 'Projection: Residual(6, s=2)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'use_1x1conv=True, strides=2', mono: true, fs: 12, fill: C.muted },
    { t: 'Input:  4 × 3 ch × 6×6', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'Shortcut: 1×1_s2 + BN', mono: true, fs: 12, fw: 600, fill: D.green.dark },
    { t: 'Output: 4 × 6 ch × 3×3', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function postVsPreactivation() {
  const W = 540, H = 264;
  let o = '';
  const cols = [
    { x: 16,  t: 'Post-Activation (2015)', s1: 'Conv→BN→ReLU (×2)', s2: 'y = ReLU(x + g(x))', s3: 'ReLU sits ON highway', s4: 'Jacobian gated by ReLU′', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: 'Pre-Activation (2016b)', s1: 'BN→ReLU→Conv (×2)', s2: 'y = x + g(x) (no ReLU)', s3: 'Pure identity highway', s4: 'Exact I in ∂x_L / ∂x_0', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 372, t: 'ResNet-D Shortcut', s1: 'Replaces 1×1_s2 skip', s2: '2×2 AvgPool + 1×1_s1', s3: 'Reads 100% of pixels', s4: '(vs 25% in 1×1_s2!)', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  cols.forEach((c) => {
    o += card(c.x, 18, 152, 228, c.fill, c.st, [
      { t: c.t, fs: 13.5, fw: 700, fill: c.dk },
      { t: c.s1, mono: true, fs: 11.5, fw: 600, fill: C.ink },
      { t: c.s2, mono: true, fs: 11.5, fw: 700, fill: c.dk },
      { t: c.s3, fs: 12, fw: 600, fill: C.ink },
      { t: c.s4, fs: 11.5, fill: C.muted },
    ]);
  });
  return svg(W, H, o);
}

function skipTopologyComparison() {
  const W = 540, H = 260;
  let o = '';
  const tops = [
    { x: 16,  t: 'ResNet (§7.4)', eq: 'x_k = x + g_k(x)', w: 'Constant width c', mem: 'O(n · c) activation RAM', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: 'ResNeXt (§7.4)', eq: 'x_k = x + Σ G_i(x)', w: '2cb + 9b²/g weights', mem: '2× wider b at same cost', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 372, t: 'DenseNet (§7.4)', eq: 'x_k = [x, f_k(x)]', w: 'Width c + k·r (growth r)', mem: 'O(n² · r) concat RAM', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  tops.forEach((t) => {
    o += card(t.x, 18, 152, 224, t.fill, t.st, [
      { t: t.t, fs: 14, fw: 700, fill: t.dk },
      { t: t.eq, mono: true, fs: 12, fw: 700, fill: C.ink },
      { t: t.w, fs: 12.5, fw: 600, fill: C.ink },
      { t: t.mem, fs: 12, fw: 700, fill: t.dk },
    ]);
  });
  return svg(W, H, o);
}

// ── efficient-convnets.md diagrams (5) ───────────────────────────────────

function edgeBudgetGap() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 134, D.red.tint, D.red.base, [
    { t: 'VGG-16 (Datacenter Backbone)', fs: 14, fw: 700, fill: D.red.dark },
    { t: '15.3B MACs / frame (460B/s @ 30fps)', mono: true, fs: 11.5, fw: 700, fill: D.red.dark },
    { t: '138M parameters (550 MB fp32)', mono: true, fs: 12, fill: C.ink },
    { t: 'ImageNet Top-1: 71.5%', fs: 13, fw: 700, fill: C.ink },
  ]);
  o += card(282, 18, 242, 134, C.lgreen, C.green, [
    { t: 'MobileNetV1 (Edge Backbone)', fs: 14, fw: 700, fill: D.green.dark },
    { t: '0.57B MACs / frame (27× fewer!)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: '4.2M parameters (33× fewer!)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'ImageNet Top-1: 70.6% (−0.9%)', fs: 13, fw: 700, fill: C.ink },
  ]);
  o += card(16, 168, 508, 76, C.lamber, C.amber, [
    { t: 'Three Pillars of Efficient ConvNets', fs: 14, fw: 700, fill: D.orange.dark },
    { t: '1. Depthwise-Separable Factorization   ·   2. Compound Scaling & SE   ·   3. RepVGG Branch Fusion', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function miniMobilenetPipeline() {
  const W = 540, H = 256;
  let o = '';
  const st = [
    { x: 16,  t: 'Stem 3×3_s2', s1: '1 → 32 ch', s2: '48 × 48', fill: C.lblue, col: C.blue, dk: D.blue.dark },
    { x: 146, t: '7 DWS Blocks', s1: '64 → 128 → 256', s2: '→ 512 @ 6×6', fill: D.purple.tint, col: C.purple, dk: D.purple.dark },
    { x: 276, t: 'Global Avg Pool', s1: '512 × 6 × 6', s2: '→ 512 vector', fill: C.lamber, col: C.amber, dk: D.orange.dark },
    { x: 406, t: 'Linear Head', s1: '512 → 10', s2: '5,130 params', fill: C.lgreen, col: C.green, dk: D.green.dark },
  ];
  st.forEach((s, i) => {
    o += card(s.x, 24, 118, 104, s.fill, s.col, [
      { t: s.t, fs: 13, fw: 700, fill: s.dk },
      { t: s.s1, mono: true, fs: 11.5, fw: 700, fill: C.ink },
      { t: s.s2, mono: true, fs: 11.5, fill: C.muted },
    ]);
    if (i < 3) o += arrow(s.x + 120, 76, s.x + 128, 76, C.gray);
  });
  o += card(16, 148, 508, 90, C.lgreen, C.green, [
    { t: 'MiniMobileNet Total:  542,474 Parameters  ·  50.3M Multiply-Adds', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    { t: '7 separable blocks use 530,528 weights vs. 4,663,296 for dense 3×3 blocks (8.8× reduction!)', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function arithmeticIntensityRoofline() {
  const W = 540, H = 264;
  let o = '';
  o += card(16, 16, 242, 134, C.lgreen, C.green, [
    { t: 'Dense 3×3 Conv (High Intensity)', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'MACs per input pixel read: 9 · c_o', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'At c_o = 128: 1,152 MACs / read!', mono: true, fs: 12.5, fw: 700, fill: D.green.dark },
    { t: 'Compute-bound (saturates GPU)', fs: 12, fill: C.ink },
  ]);
  o += card(282, 16, 242, 134, D.red.tint, D.red.base, [
    { t: 'Depthwise 3×3 (Low Intensity)', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: 'MACs per input pixel read: ONLY 9!', mono: true, fs: 12, fw: 700, fill: D.red.dark },
    { t: '128× lower data reuse in SRAM', fs: 12.5, fw: 600, fill: C.ink },
    { t: 'HBM memory-bandwidth bound!', fs: 12, fw: 700, fill: D.red.dark },
  ]);
  o += card(16, 166, 508, 82, C.lamber, C.amber, [
    { t: 'Why 7.7× Fewer MACs Yield Only 2–3.5× GPU Speedup (and Narrow Dense Wins at 50M MACs)', fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Motivated EfficientNetV2 to replace early high-res depthwise layers with fused dense 3×3 convs', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function compoundScalingCube() {
  const W = 540, H = 256;
  let o = '';
  const axes = [
    { x: 16,  t: 'Depth Scaling (d)', eq: 'd = 1.2^φ', cost: 'MACs ∝ d (linear)', role: 'Wider receptive field', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: 'Width Scaling (w)', eq: 'w = 1.1^φ', cost: 'MACs ∝ w² (quadratic)', role: 'Finer feature channels', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 372, t: 'Resolution (r)', eq: 'r = 1.15^φ', cost: 'MACs ∝ r² (quadratic)', role: 'Sharper input detail', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  axes.forEach((a) => {
    o += card(a.x, 18, 152, 126, a.fill, a.st, [
      { t: a.t, fs: 13.5, fw: 700, fill: a.dk },
      { t: a.eq, mono: true, fs: 14, fw: 700, fill: C.ink },
      { t: a.cost, mono: true, fs: 11.5, fw: 600, fill: a.dk },
      { t: a.role, fs: 12, fill: C.muted },
    ]);
  });
  o += card(16, 162, 508, 78, C.lamber, C.amber, [
    { t: 'Compound Step φ:  Total MACs × (1.2 × 1.1² × 1.15²)^φ ≈ 1.92^φ  (~2× compute per step)', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Jointly scaling depth, width, and resolution avoids single-axis saturation (EfficientNet B0→B7)', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function seChannelGating() {
  const W = 540, H = 256;
  let o = '';
  const st = [
    { x: 16,  t: 'Feature Map X', s1: '(C, H, W)', s2: 'Local conv out', fill: C.lblue, col: C.blue, dk: D.blue.dark },
    { x: 146, t: '1. Squeeze (GAP)', s1: 'z = mean(H,W)', s2: 'Shape (C, 1, 1)', fill: D.purple.tint, col: C.purple, dk: D.purple.dark },
    { x: 276, t: '2. Excite MLP', s1: 'C → C/r → C', s2: 's = σ(W₂ReLU(W₁z))', fill: C.lamber, col: C.amber, dk: D.orange.dark },
    { x: 406, t: '3. Rescale ⊙', s1: 'Y_c = s_c X_c', s2: 'Shape (C, H, W)', fill: C.lgreen, col: C.green, dk: D.green.dark },
  ];
  st.forEach((s, i) => {
    o += card(s.x, 24, 118, 104, s.fill, s.col, [
      { t: s.t, fs: 13, fw: 700, fill: s.dk },
      { t: s.s1, mono: true, fs: 11.5, fw: 700, fill: C.ink },
      { t: s.s2, mono: true, fs: 11, fill: C.muted },
    ]);
    if (i < 3) o += arrow(s.x + 120, 76, s.x + 128, 76, C.gray);
  });
  o += card(16, 148, 508, 90, C.lgreen, C.green, [
    { t: 'Global Channel Attention for Only 2C²/r Parameters and <0.03% Extra MACs!', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'Because the 2-layer MLP operates on a 1×1 pooled vector z, its arithmetic cost is independent of H×W', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ── training-recipes.md diagrams (7) ─────────────────────────────────────

function resnet50StrikesBack() {
  const W = 540, H = 260;
  let o = '';
  const bars = [
    { y: 20,  t: '2015 Baseline (90 ep, SGD+step)', acc: '76.1%', w: 180, fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 84,  t: 'Modern A3 (100 ep, LAMB+Mixup)', acc: '78.1% (+2.0%)', w: 230, fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { y: 148, t: 'Modern A1 (600 ep, Full Recipe)', acc: '80.4% (+4.3%!)', w: 290, fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  bars.forEach((b) => {
    o += card(16, b.y, 212, 46, '#FFFFFF', b.st, [{ t: b.t, fs: 12.5, fw: 700, fill: C.ink }]);
    o += card(234, b.y, b.w, 46, b.fill, b.st, [{ t: b.acc, mono: true, fs: 13.5, fw: 700, fill: b.dk }]);
  });
  o += tx(270, 224, 'Zero architectural changes to 2015 ResNet-50 (Wightman et al., 2021)!', { fs: 13.5, fw: 700, fill: D.green.dark });
  o += tx(270, 244, 'Always compare architectures under matched training recipes', { fs: 12.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

function recipePipeline20152022() {
  const W = 540, H = 264;
  let o = '';
  const stages = [
    { x: 16,  y: 16, t: '1. Data Augmentation', old: '2015: Crop + Flip', mod: '2022: +RandAug, Mixup, CutMix', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 278, y: 16, t: '2. Forward & Regularize', old: '2015: L2 weight decay', mod: '2022: +Stochastic Depth (drop_path)', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 16,  y: 102, t: '3. Loss Objective', old: '2015: 1-hot Cross-Entropy', mod: '2022: Label Smoothing (ε = 0.1)', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 278, y: 102, t: '4. Optimizer & Schedule', old: '2015: SGD + Step Drops (90 ep)', mod: '2022: AdamW + Cosine Warmup (300 ep)', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  stages.forEach((s) => {
    o += card(s.x, s.y, 246, 74, s.fill, s.st, [
      { t: s.t, fs: 13.5, fw: 700, fill: s.dk },
      { t: s.old, fs: 11.5, fill: C.muted },
      { t: s.mod, fs: 11.5, fw: 700, fill: C.ink },
    ]);
  });
  o += card(16, 188, 508, 62, C.lgreen, C.green, [
    { t: '5. Evaluation Checkpoint:  2015 Final Iterate θ_T   →   2022 Shadow EMA θ̄_T = ρ θ̄ + (1−ρ) θ', fs: 13, fw: 700, fill: D.green.dark },
    { t: 'Stronger augmentation + regularization requires longer training horizons (300–600 epochs)', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function adamwDecoupledDecay() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 134, D.red.tint, D.red.base, [
    { t: 'Classic Adam + Coupled L2', fs: 14, fw: 700, fill: D.red.dark },
    { t: 'g_t ← ∇L(θ_t) + λ θ_t', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Divides λ θ_t by √(v̂_t) + ε!', mono: true, fs: 12.5, fw: 700, fill: D.red.dark },
    { t: 'Weights with large grads decay LESS', fs: 11.5, fill: C.muted },
  ]);
  o += card(282, 18, 242, 134, C.lgreen, C.green, [
    { t: 'AdamW: Decoupled Weight Decay', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'θ_{t+1} = (1 − η_t λ) θ_t − η_t m̂_t/√v̂_t', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    { t: 'Uniform fractional shrinkage η_t λ', fs: 12.5, fw: 700, fill: C.ink },
    { t: 'Matches tuned SGD generalization!', fs: 11.5, fill: D.green.dark },
  ]);
  o += card(16, 168, 508, 76, C.lamber, C.amber, [
    { t: 'Comparing Decay Across Optimizers:  Compare Product  η₀ × λ', mono: true, fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'AdamW (η₀=0.002, λ=0.05) ⇒ 10⁻⁴/step   vs.   SGD (η₀=0.05, λ=5·10⁻⁴, mom=0.9) ⇒ 2.5·10⁻⁵/step', mono: true, fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function mixupVsCutmix() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 136, C.lblue, C.blue, [
    { t: 'Mixup (Global Alpha Blend)', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'x̃ = λ x_i + (1 − λ) x_j', mono: true, fs: 13, fw: 700, fill: C.ink },
    { t: 'ỹ = λ y_i + (1 − λ) y_j', mono: true, fs: 13, fw: 700, fill: D.blue.dark },
    { t: 'Superposes whole images (λ ~ Beta)', fs: 11.5, fill: C.muted },
  ]);
  o += card(282, 18, 242, 136, C.lgreen, C.green, [
    { t: 'CutMix (Spatial Patch Paste)', fs: 14, fw: 700, fill: D.green.dark },
    { t: 'x̃ = M ⊙ x_i + (1 − M) ⊙ x_j', mono: true, fs: 12.5, fw: 700, fill: C.ink },
    { t: 'ỹ = λy_i + (1−λ)y_j, λ = area', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Locally natural pixel statistics!', fs: 12, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 170, 508, 74, D.purple.tint, C.purple, [
    { t: 'Linearity of Cross-Entropy in Target y:  L(p, ỹ) = λ L(p, y_i) + (1 − λ) L(p, y_j)', mono: true, fs: 13, fw: 700, fill: D.purple.dark },
    { t: 'Enforces smooth linear probability transitions between classes outside the training manifold', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function stochasticDepthGate() {
  const W = 540, H = 256;
  let o = '';
  o += card(16, 48, 118, 92, C.lblue, C.blue, [
    { t: 'Block Input', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'x_ℓ', mono: true, fs: 15, fw: 700, fill: C.ink },
  ]);
  // Top shortcut path
  o += arrow(140, 56, 378, 56, C.green);
  o += tx(259, 24, 'Always-On Identity Shortcut (100% flow)', { fs: 12.5, fw: 700, fill: D.green.dark });
  // Bottom branch with Bernoulli gate
  o += arrow(140, 110, 174, 110, C.purple);
  o += card(180, 78, 184, 62, D.purple.tint, C.purple, [
    { t: 'Residual Branch f(x_ℓ)', fs: 12.5, fw: 700, fill: D.purple.dark },
    { t: '× b / (1 − p_ℓ), b~Bern(1−p_ℓ)', mono: true, fs: 11, fw: 700, fill: C.ink },
  ]);
  o += arrow(370, 110, 384, 98, C.purple);
  o += card(388, 48, 136, 92, C.lgreen, C.green, [
    { t: 'Output x_{ℓ+1}', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: '𝔼[x_{ℓ+1}] = x_ℓ+f(x_ℓ)', mono: true, fs: 11, fw: 700, fill: C.ink },
    { t: 'Eval: b ≡ 1', mono: true, fs: 12, fill: D.green.dark },
  ]);
  o += card(16, 154, 508, 86, C.lamber, C.amber, [
    { t: 'Linear Depth Schedule:  p_ℓ = (ℓ / L) · p_max   (0 at stem → 0.1–0.5 at top block)', mono: true, fs: 13, fw: 700, fill: D.orange.dark },
    { t: 'Trains an implicit ensemble of shallower subnetworks; scaling 1/(1−p) keeps train/eval expectations equal', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function classicVsModernSetup() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 224, C.lblue, C.blue, [
    { t: 'ClassicRecipe (2015 Baseline)', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '• Same ResNet-18 (11.2M params)', fs: 12, fw: 600, fill: C.ink },
    { t: '• SGD (η₀=0.05, mom=0.9, wd=5e−4)', mono: true, fs: 11, fill: C.ink },
    { t: '• 2-ep warmup + ÷10 step drops', fs: 12, fill: C.muted },
    { t: '• 30 epochs, 1-hot Cross-Entropy', fs: 12, fill: C.ink },
    { t: 'FashionMNIST10k: 89.0–90.5%', fs: 13, fw: 700, fill: D.blue.dark },
  ]);
  o += card(282, 18, 242, 224, C.lgreen, C.green, [
    { t: 'ModernRecipe (Compact 2022)', fs: 14, fw: 700, fill: D.green.dark },
    { t: '• Exact Same ResNet-18 (11.2M)', fs: 12, fw: 600, fill: C.ink },
    { t: '• AdamW (η₀=0.002, decoupled wd=0.05)', mono: true, fs: 10.5, fill: C.ink },
    { t: '• 3-ep warmup + half-cosine decay', fs: 12, fill: C.muted },
    { t: '• 45 epochs, Mixup(0.2) + Smooth(0.1)', fs: 11.5, fill: C.ink },
    { t: 'FashionMNIST10k: 90.5–91.0% (+1 pt)', fs: 13, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function leaderboardMethodology() {
  const W = 540, H = 260;
  let o = '';
  const rules = [
    { x: 16,  t: '1. Retune LR on Swap', s1: 'SGD η₀ = 0.05 vs.', s2: 'AdamW η₀ = 0.002 (25×)', s3: 'Never swap optimizer', s4: 'at fixed learning rate!', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 194, t: '2. ImageNet-V2 Shift', s1: 'Recht et al. (2019)', s2: '−11 to −14% absolute drop', s3: 'Preserves relative model', s4: 'rankings linearly', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 372, t: '3. Downstream Transfer', s1: 'ImageNet top-1 saturates', s2: 'near multi-label floor', s3: 'Test backbones on COCO', s4: 'detection & segmentation', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  rules.forEach((r) => {
    o += card(r.x, 18, 152, 224, r.fill, r.st, [
      { t: r.t, fs: 13.5, fw: 700, fill: r.dk },
      { t: r.s1, mono: true, fs: 11.5, fw: 700, fill: C.ink },
      { t: r.s2, mono: true, fs: 11.5, fw: 700, fill: r.dk },
      { t: r.s3, fs: 12, fill: C.ink },
      { t: r.s4, fs: 12, fill: C.muted },
    ]);
  });
  return svg(W, H, o);
}

// ── convnext.md diagrams (8) ─────────────────────────────────────────────

function resnetVsSwinGap() {
  const W = 540, H = 260;
  let o = '';
  const cols = [
    { x: 16,  t: '2015 ResNet-50', s1: '4.1G MACs · 25.6M', s2: '90-ep 2015 recipe', s3: 'Dense 3×3 bottleneck', s4: 'Top-1: 76.1%', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 194, t: '2021 Swin-T (ViT)', s1: '4.5G MACs · 28.3M', s2: '300-ep AdamW recipe', s3: '7×7 Window Attention', s4: 'Top-1: 81.3% (+5.2%)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 372, t: '2022 ConvNeXt-T', s1: '4.5G MACs · 28.6M', s2: '300-ep AdamW recipe', s3: '7×7 DWConv (0 Attn!)', s4: 'Top-1: 82.0% (+5.9%!)', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  cols.forEach((c) => {
    o += card(c.x, 18, 152, 224, c.fill, c.st, [
      { t: c.t, fs: 14, fw: 700, fill: c.dk },
      { t: c.s1, mono: true, fs: 11.5, fw: 600, fill: C.ink },
      { t: c.s2, fs: 12, fill: C.muted },
      { t: c.s3, fs: 12, fw: 700, fill: C.ink },
      { t: c.s4, mono: true, fs: 13, fw: 700, fill: c.dk },
    ]);
  });
  return svg(W, H, o);
}

function convnextWaterfall() {
  const W = 540, H = 264;
  let o = '';
  const steps = [
    { y: 14,  t: '0. ResNet-50 (2015 recipe)', acc: '76.1%', w: 150, fill: N.panel, st: C.gray, dk: C.ink },
    { y: 64,  t: '1. + Modern 300-ep Recipe (+2.7%)', acc: '78.8%', w: 205, fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 114, t: '2. + (3,3,9,3) Stages & Patch (+0.7%)', acc: '79.5%', w: 225, fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 164, t: '3. + Depthwise, InvBot & 7×7 (+1.1%)', acc: '80.6%', w: 255, fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { y: 214, t: '4. + 1 GELU/LN & Downsample (+1.4%)', acc: '82.0%!', w: 295, fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  steps.forEach((s) => {
    o += card(16, s.y, 218, 38, '#FFFFFF', s.st, [{ t: s.t, fs: 11.5, fw: 700, fill: C.ink }]);
    o += card(240, s.y, s.w - 15, 38, s.fill, s.st, [{ t: s.acc, mono: true, fs: 13, fw: 700, fill: s.dk }]);
  });
  return svg(W, H, o);
}

function resnetVsPatchifyStem() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 134, C.lblue, C.blue, [
    { t: 'ResNet Overlapping Stem', fs: 14, fw: 700, fill: D.blue.dark },
    { t: '7×7 Conv (s=2, p=3) + BN + ReLU', mono: true, fs: 11.5, fw: 700, fill: C.ink },
    { t: '+ 3×3 MaxPool (s=2, p=1) → H/4 × W/4', mono: true, fs: 11.5, fill: C.ink },
    { t: 'Preserves fine detail on 64×64!', fs: 12, fw: 700, fill: D.blue.dark },
  ]);
  o += card(282, 18, 242, 134, D.purple.tint, C.purple, [
    { t: 'ConvNeXt / ViT Patchify Stem', fs: 14, fw: 700, fill: D.purple.dark },
    { t: '4×4 Conv (s=4, p=0) + LayerNorm', mono: true, fs: 11.5, fw: 700, fill: C.ink },
    { t: 'Non-overlapping 4×4 tiles → H/4 × W/4', mono: true, fs: 11.5, fill: C.ink },
    { t: '+0.1% on 224² (but ≈ −2 pts on 64²!)', fs: 12, fw: 700, fill: D.red.dark },
  ]);
  o += card(16, 168, 508, 76, C.lgreen, C.green, [
    { t: 'Stage Depth Allocation:  (3, 4, 6, 3)  →  (3, 3, 9, 3)  (1 : 1 : 3 : 1 Ratio)', mono: true, fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Concentrates 50% of blocks in Stage 3 (14×14 grid on ImageNet), adding +0.6% top-1 accuracy', fs: 12.5, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function convnextBlockReorder() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 134, C.lamber, C.amber, [
    { t: 'MobileNetV2 Order (DW in Middle)', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: '1×1 (c→4c) → DW k×k (on 4c!) → 1×1', mono: true, fs: 11.5, fw: 700, fill: C.ink },
    { t: '7×7 DW on 4c costs 4·49c = 196c MACs', mono: true, fs: 11.5, fill: D.red.dark },
    { t: '4× heavier spatial filtering', fs: 12, fill: C.muted },
  ]);
  o += card(282, 18, 242, 134, C.lgreen, C.green, [
    { t: 'ConvNeXt Order (DW at Front!)', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'DW 7×7 (on c!) → LN → 1×1(4c) → 1×1(c)', mono: true, fs: 11, fw: 700, fill: C.ink },
    { t: '7×7 DW on c costs ONLY 49c MACs!', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Mirrors ViT: Spatial_c → MLP_{c→4c→c}', mono: true, fs: 11.5, fill: C.ink },
  ]);
  o += card(16, 168, 508, 76, C.lblue, C.blue, [
    { t: 'Per-Pixel Block Compute:  49c (7×7 Depthwise)  +  8c² (Two 1×1 Layers)', mono: true, fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'At c = 96, the 7×7 spatial kernel is only 6.0% of block MACs (1.6% at c = 384)!', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function convnextBlockDataflow() {
  const W = 540, H = 256;
  let o = '';
  const st = [
    { x: 16,  t: 'DWConv 7×7', s1: 'c → c (groups=c)', s2: '+ LayerNorm(c)', fill: C.lblue, col: C.blue, dk: D.blue.dark },
    { x: 146, t: '1×1 Expand', s1: 'c → 4c', s2: '+ Single GELU', fill: D.purple.tint, col: C.purple, dk: D.purple.dark },
    { x: 276, t: '1×1 Project', s1: '4c → c (linear)', s2: '× LayerScale γ', fill: C.lamber, col: C.amber, dk: D.orange.dark },
    { x: 406, t: 'DropPath + X', s1: 'X + drop(γ⊙Y)', s2: 'Init γ = 10⁻⁶', fill: C.lgreen, col: C.green, dk: D.green.dark },
  ];
  st.forEach((s, i) => {
    o += card(s.x, 26, 118, 104, s.fill, s.col, [
      { t: s.t, fs: 13, fw: 700, fill: s.dk },
      { t: s.s1, mono: true, fs: 11, fw: 700, fill: C.ink },
      { t: s.s2, mono: true, fs: 11, fill: s.dk },
    ]);
    if (i < 3) o += arrow(s.x + 120, 78, s.x + 128, 78, C.gray);
  });
  o += card(16, 150, 508, 88, C.lgreen, C.green, [
    { t: 'Only 1 Activation (GELU) and 1 Normalization (LN) per Block (+0.9% on ImageNet!)', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'In channels-last layout, LayerNorm and both 1×1 convolutions run as standard last-axis linear layers', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function convnextAttoPyramid() {
  const W = 540, H = 260;
  let o = '';
  const stages = [
    { x: 16,  t: 'Stage 1 (d=2)', c: '40 ch @ 16×16', sub: 'After 4×4_s4 stem', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 146, t: 'Stage 2 (d=2)', c: '80 ch @ 8×8', sub: 'After LN+2×2_s2', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 276, t: 'Stage 3 (d=6)', c: '160 ch @ 4×4', sub: 'Deepest stage (3×)', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 406, t: 'Stage 4 (d=2)', c: '320 ch @ 2×2', sub: '→ GAP + LN → 10', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  stages.forEach((s, i) => {
    o += card(s.x, 22, 118, 108, s.fill, s.st, [
      { t: s.t, fs: 13, fw: 700, fill: s.dk },
      { t: s.c, mono: true, fs: 11.5, fw: 700, fill: C.ink },
      { t: s.sub, fs: 11, fill: C.muted },
    ]);
    if (i < 3) o += arrow(s.x + 120, 76, s.x + 128, 76, C.gray);
  });
  o += card(16, 150, 508, 92, C.lamber, C.amber, [
    { t: 'ConvNeXt-Atto Total:  3,376,450 Parameters  (1/3 of ResNet-18’s 11.2M)', mono: true, fs: 14, fw: 700, fill: D.orange.dark },
    { t: '>95% of parameters sit in 1×1 pointwise linear layers & downsamplers; <3% in 7×7 depthwise kernels', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function localAblationSteps() {
  const W = 540, H = 264;
  let o = '';
  const rows = [
    { t: '0. Compact ResNet-18 (base=35, BN)', p: '3.35M', m: '43.2M MACs', a: '≈ 94.0%', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { t: '1. + ConvNeXt Body & Stages (with BN)', p: '3.38M', m: '46.1M MACs', a: '≈ 94.0%', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { t: '2. + Swap BatchNorm → LayerNorm', p: '3.38M', m: '46.1M MACs', a: '≈ 94.0%', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { t: '3. + Swap Stem → 4×4_s4 Patchify', p: '3.38M', m: '44.3M MACs', a: '≈ 91.7% (−2 pts!)', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
  ];
  rows.forEach((r, i) => {
    const y = 16 + i * 58;
    o += card(16, y, 250, 46, '#FFFFFF', r.st, [{ t: r.t, fs: 12.5, fw: 700, fill: C.ink }]);
    o += card(274, y, 114, 46, N.panel, C.gray, [{ t: `${r.p} · ${r.m}`, mono: true, fs: 11, fw: 600, fill: C.ink }]);
    o += card(396, y, 128, 46, r.fill, r.st, [{ t: r.a, mono: true, fs: 12.5, fw: 700, fill: r.dk }]);
  });
  return svg(W, H, o);
}

function convnextv2Grn() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 128, C.lblue, C.blue, [
    { t: '1. Sparse Masked Autoencoder', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Mask 75% of image patches', fs: 12.5, fw: 600, fill: C.ink },
    { t: 'Submanifold sparse convolutions', fs: 12, fill: C.ink },
    { t: 'prevent zero-bleeding at borders', fs: 11.5, fill: C.muted },
  ]);
  o += card(282, 18, 242, 128, C.lgreen, C.green, [
    { t: '2. Global Response Norm (GRN)', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'g_c = ‖X_c‖₂ ,  g̅ = mean_c(g_c)', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'Replaces LayerScale after GELU', fs: 12, fill: C.ink },
    { t: 'Prevents dead/collapsed channels', fs: 11.5, fw: 700, fill: D.green.dark },
  ]);
  o += card(16, 162, 508, 80, D.purple.tint, C.purple, [
    { t: 'GRN Formula:  GRN(X)_c = γ_c · (g_c / (g̅ + ε)) · X_c + β_c + X_c', mono: true, fs: 13.5, fw: 700, fill: D.purple.dark },
    { t: 'Normalizing each channel’s spatial L2 norm against the layer mean enforces inter-channel competition', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function layerscaleVsDroppath() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 134, C.lblue, C.blue, [
    { t: 'Layer Scale γ (init 10⁻⁶)', fs: 14, fw: 700, fill: D.blue.dark },
    { t: 'Deterministic vector γ ∈ ℝ^c', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'Starts near identity at step 0', fs: 12, fill: C.ink },
    { t: 'Remains active at eval time', fs: 12, fw: 700, fill: D.blue.dark },
  ]);
  o += card(282, 18, 242, 134, D.purple.tint, C.purple, [
    { t: 'Stochastic Depth (DropPath)', fs: 14, fw: 700, fill: D.purple.dark },
    { t: 'Random gate b/(1−p) per example', mono: true, fs: 11.5, fw: 700, fill: C.ink },
    { t: 'Regularizes co-adaptation in train', fs: 12, fill: C.ink },
    { t: 'Turns off (b ≡ 1) at eval time', fs: 12, fw: 700, fill: D.purple.dark },
  ]);
  o += card(16, 168, 508, 76, C.lgreen, C.green, [
    { t: 'Complementary Roles in Deep ConvNeXt & Vision Transformer Blocks', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'LayerScale stabilizes early gradient flow; DropPath trains an implicit ensemble of subnetworks', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ── cnn-design.md diagrams (5) ───────────────────────────────────────────

function pointSearchVsDesignSpace() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 224, D.red.tint, D.red.base, [
    { t: 'Single-Model Search (Manual / NAS)', fs: 13.5, fw: 700, fill: D.red.dark },
    { t: '• Searches 1.8 × 10¹⁸ haystack', fs: 12.5, fw: 600, fill: C.ink },
    { t: '  for 1 point model at 1 FLOP budget', fs: 12, fill: C.muted },
    { t: '• Yields opaque magic numbers:', fs: 12.5, fill: C.ink },
    { t: '  (3,4,6,3) vs. (3,3,9,3) depths', mono: true, fs: 11.5, fill: D.red.dark },
    { t: '• No scientific rule for new budgets', fs: 12, fw: 700, fill: D.red.dark },
  ]);
  o += card(282, 18, 242, 224, C.lgreen, C.green, [
    { t: 'Design-Space Distillation (RegNet)', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: '• Sample n=500 models per space', fs: 12.5, fw: 600, fill: C.ink },
    { t: '• Compare empirical error CDFs F̂(e)', mono: true, fs: 11.5, fw: 700, fill: D.green.dark },
    { t: '• Progressively constrain 16 knobs:', fs: 12.5, fill: C.ink },
    { t: '  AnyNetX_A → B → C → D → E → RegNetX', mono: true, fs: 11, fw: 700, fill: D.green.dark },
    { t: '• Generalizes across all FLOP regimes!', fs: 12, fw: 700, fill: D.green.dark },
  ]);
  return svg(W, H, o);
}

function empiricalCdfDominance() {
  const W = 540, H = 260;
  let o = '';
  const x0 = 64, y0 = 192, w = 420, h = 136;
  o += line(x0, y0 - h, x0, y0, C.gray, 2);
  o += line(x0, y0, x0 + w, y0, C.gray, 2);

  // Two step/S-curves: Good space (left/upper, green) vs Bad space (right/lower, red)
  const goodPts = `${x0+30},${y0} ${x0+70},${y0-35} ${x0+110},${y0-95} ${x0+160},${y0-124} ${x0+230},${y0-h} ${x0+w},${y0-h}`;
  const badPts  = `${x0+120},${y0} ${x0+190},${y0-28} ${x0+260},${y0-75} ${x0+330},${y0-112} ${x0+390},${y0-h}`;
  o += `<polyline points="${badPts}" fill="none" stroke="${D.red.base}" stroke-width="3" stroke-dasharray="6 4"/>`;
  o += `<polyline points="${goodPts}" fill="none" stroke="${C.green}" stroke-width="3"/>`;

  // Labels placed well away from curves
  o += tx(x0 - 12, y0 - h, '1.0', { mono: true, fs: 12.5, fw: 700, fill: C.ink, anchor: 'end' });
  o += tx(x0 - 12, y0, '0.0', { mono: true, fs: 12.5, fill: C.muted, anchor: 'end' });
  o += tx(x0 + 20, y0 - h - 18, 'Better Space F̂₁(e) (Growing Widths: 43% mean err)', { fs: 12.5, fw: 700, fill: D.green.dark, anchor: 'start' });
  o += tx(x0 + 250, y0 - 28, 'Worse Space F̂₂(e) (63% err)', { fs: 12.5, fw: 700, fill: D.red.dark, anchor: 'start' });
  o += tx(x0 + w / 2, y0 + 22, 'Validation Error Threshold e  →  (left & higher = better)', { fs: 13, fw: 700, fill: C.ink });
  o += fitTx(270, 242, 'F̂(e) = (1/n) Σ 1(e_j ≤ e) measures stochastic dominance across the population', 516, { mono: true, fs: 11.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

function anynetStageMacBreakdown() {
  const W = 540, H = 260;
  let o = '';
  const parts = [
    { x: 16,  t: '1. Stem (3×3_s2)', eq: 'r² · 9 · c_0', sub: 'Halves r → r/2 (c_0=32)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 194, t: '2. 1×1 Convs + Skip', eq: 'r² (2 c_in c + (2d−1)c²)', sub: 'Quadratic in stage width c', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 372, t: '3. Grouped 3×3 (b=1)', eq: 'r² · 9 · g · c · d', sub: 'Linear in c at fixed g!', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  parts.forEach((p) => {
    o += card(p.x, 20, 152, 128, p.fill, p.st, [
      { t: p.t, fs: 13.5, fw: 700, fill: p.dk },
      { t: p.eq, mono: true, fs: 11.5, fw: 700, fill: C.ink },
      { t: p.sub, fs: 12, fill: C.muted },
    ]);
  });
  o += card(16, 164, 508, 80, C.lamber, C.amber, [
    { t: 'Why Grouped 3×3 Cost Is Linear in Width c When Group Width g Is Fixed', fs: 13.5, fw: 700, fill: D.orange.dark },
    { t: 'Number of groups = c / g  ⇒  MACs = r² · 9 · c² / (c / g) = 9 r² g c   per block', mono: true, fs: 12.5, fw: 700, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function regnetLinearStaircase() {
  const W = 540, H = 264;
  let o = '';
  // Draw a 4-stage staircase of bars representing quantized stage widths
  const stages = [
    { x: 36,  w: 86,  h: 32,  t: 'Stage 1', c: 'c₁ = 24', d: 'd₁ = 1' },
    { x: 142, w: 96,  h: 56,  t: 'Stage 2', c: 'c₂ = 56', d: 'd₂ = 1' },
    { x: 258, w: 116, h: 88,  t: 'Stage 3', c: 'c₃ = 152', d: 'd₃ = 4' },
    { x: 394, w: 116, h: 122, t: 'Stage 4', c: 'c₄ = 368', d: 'd₄ = 7' },
  ];
  const base = 192;
  stages.forEach((s, i) => {
    const y = base - s.h;
    const cols = [
      [C.lblue, C.blue, D.blue.dark],
      [C.lblue, C.blue, D.blue.dark],
      [D.purple.tint, C.purple, D.purple.dark],
      [C.lgreen, C.green, D.green.dark],
    ][i];
    o += box(s.x, y, s.w, s.h, cols[0], cols[1], 6, 2);
    o += tx(s.x + s.w / 2, y - 14, `${s.c} (${s.d})`, { mono: true, fs: 12, fw: 700, fill: cols[2] });
    o += tx(s.x + s.w / 2, base + 18, s.t, { fs: 13, fw: 700, fill: C.ink });
  });
  o += line(24, base, 520, base, C.gray, 2);
  o += fitTx(270, 22, 'Linear Ramp: u_j = u₀ + a·j  →  Quantized Staircase: ũ_j = u₀ m^{s_j}', 500, { mono: true, fs: 12.5, fw: 700, fill: D.purple.dark });
  o += fitTx(270, 242, 'Collapses 16 arbitrary stage hyperparameters into (d, u₀, a, m, g) with bottleneck b = 1', 512, { fs: 12.5, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

function regnetVsResnetTradeoffs() {
  const W = 540, H = 260;
  let o = '';
  o += card(16, 18, 242, 134, C.lgreen, C.green, [
    { t: 'RegNetX32 (2 Stages, g=16, b=1)', fs: 13.5, fw: 700, fill: D.green.dark },
    { t: 'Parameters: 519,658 (21× fewer!)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Multiply-Adds: 112M (3× fewer!)', mono: true, fs: 12, fw: 700, fill: D.green.dark },
    { t: 'Val Acc: ~94.0% (ties ResNet-18!)', fs: 12.5, fw: 700, fill: C.ink },
  ]);
  o += card(282, 18, 242, 134, C.lblue, C.blue, [
    { t: 'ResNet-18 (4 Stages, Dense 3×3)', fs: 13.5, fw: 700, fill: D.blue.dark },
    { t: 'Parameters: 11,175,370', mono: true, fs: 12, fw: 700, fill: C.ink },
    { t: 'Multiply-Adds: 319M', mono: true, fs: 12, fill: C.ink },
    { t: 'Val Acc: ~94.0% (1.5× faster GPU ep)', fs: 12, fw: 700, fill: D.blue.dark },
  ]);
  o += card(16, 168, 508, 76, C.lamber, C.amber, [
    { t: 'Three Calibration Nuances:  1. Momentum+WD Coupling  ·  2. 94% Ceiling  ·  3. GPU Latency', fs: 12.5, fw: 700, fill: D.orange.dark },
    { t: 'RegNetX runs 5× faster than depthwise EfficientNets on GPUs at equal FLOPs', fs: 12, fw: 600, fill: C.ink },
  ]);
  return svg(W, H, o);
}

export const diagrams = {
  // index.md (4)
  'ch07-mcnn-four-scaling-barriers':     fourScalingBarriers,
  'ch07-mcnn-decade-evolution-arc':      decadeEvolutionArc,
  'ch07-mcnn-three-unifying-invariants': threeUnifyingInvariants,
  'ch07-mcnn-pareto-scoreboard':         paretoScoreboard,
  // alexnet.md (7)
  'ch07-mcnn-handcrafted-vs-e2e':        handcraftedVsE2e,
  'ch07-mcnn-imagenet-augmentation':     imagenetAugmentation,
  'ch07-mcnn-gpu-vs-cpu-budget':         gpuVsCpuBudget,
  'ch07-mcnn-lenet-vs-alexnet-scale':    lenetVsAlexnetScale,
  'ch07-mcnn-relu-he-dropout-triad':     reluHeDropoutTriad,
  'ch07-mcnn-alexnet-param-mac-split':   alexnetParamMacSplit,
  'ch07-mcnn-cosine-warmup-curve':       cosineWarmupCurve,
  // blocks.md (6)
  'ch07-mcnn-blocks-three-solutions':    blocksThreeSolutions,
  'ch07-mcnn-vgg-rf-efficiency':         vggRfEfficiency,
  'ch07-mcnn-nin-per-pixel-mlp':         ninPerPixelMlp,
  'ch07-mcnn-vgg-vs-pooled-vgg':         vggVsPooledVgg,
  'ch07-mcnn-inception-bottleneck-math': inceptionBottleneckMath,
  'ch07-mcnn-stem-body-head-template':   stemBodyHeadTemplate,
  // batch-norm.md (10)
  'ch07-mcnn-unnormalized-scale-drift':  unnormalizedScaleDrift,
  'ch07-mcnn-bn-forward-pipeline':       bnForwardPipeline,
  'ch07-mcnn-bn-bias-cancellation':      bnBiasCancellation,
  'ch07-mcnn-bn-orthogonal-gradient':    bnOrthogonalGradient,
  'ch07-mcnn-bn-dense-vs-conv':          bnDenseVsConv,
  'ch07-mcnn-bn-train-vs-eval-mode':     bnTrainVsEvalMode,
  'ch07-mcnn-bn-distribution-shift':     bnDistributionShift,
  'ch07-mcnn-bn-small-batch-collapse':   bnSmallBatchCollapse,
  'ch07-mcnn-nfnet-weight-std':          nfnetWeightStd,
  'ch07-mcnn-normalizer-decision-tree':  normalizerDecisionTree,
  // resnet.md (5)
  'ch07-mcnn-degradation-paradox':       degradationParadox,
  'ch07-mcnn-residual-gradient-highway': residualGradientHighway,
  'ch07-mcnn-residual-shape-transition': residualShapeTransition,
  'ch07-mcnn-post-vs-preactivation':     postVsPreactivation,
  'ch07-mcnn-skip-topology-comparison':  skipTopologyComparison,
  // efficient-convnets.md (5)
  'ch07-mcnn-edge-budget-gap':           edgeBudgetGap,
  'ch07-mcnn-mini-mobilenet-pipeline':   miniMobilenetPipeline,
  'ch07-mcnn-arithmetic-intensity-roofline': arithmeticIntensityRoofline,
  'ch07-mcnn-compound-scaling-cube':     compoundScalingCube,
  'ch07-mcnn-se-channel-gating':         seChannelGating,
  // training-recipes.md (7)
  'ch07-mcnn-resnet50-strikes-back':     resnet50StrikesBack,
  'ch07-mcnn-recipe-pipeline-2015-2022': recipePipeline20152022,
  'ch07-mcnn-adamw-decoupled-decay':     adamwDecoupledDecay,
  'ch07-mcnn-mixup-vs-cutmix':           mixupVsCutmix,
  'ch07-mcnn-stochastic-depth-gate':     stochasticDepthGate,
  'ch07-mcnn-classic-vs-modern-setup':   classicVsModernSetup,
  'ch07-mcnn-leaderboard-methodology':   leaderboardMethodology,
  // convnext.md (8)
  'ch07-mcnn-resnet-vs-swin-gap':        resnetVsSwinGap,
  'ch07-mcnn-convnext-waterfall':        convnextWaterfall,
  'ch07-mcnn-resnet-vs-patchify-stem':   resnetVsPatchifyStem,
  'ch07-mcnn-convnext-block-reorder':    convnextBlockReorder,
  'ch07-mcnn-convnext-block-dataflow':   convnextBlockDataflow,
  'ch07-mcnn-convnext-atto-pyramid':     convnextAttoPyramid,
  'ch07-mcnn-local-ablation-steps':      localAblationSteps,
  'ch07-mcnn-convnextv2-grn':            convnextv2Grn,
  'ch07-mcnn-layerscale-vs-droppath':    layerscaleVsDroppath,
  // cnn-design.md (5)
  'ch07-mcnn-point-search-vs-design-space': pointSearchVsDesignSpace,
  'ch07-mcnn-empirical-cdf-dominance':   empiricalCdfDominance,
  'ch07-mcnn-anynet-stage-mac-breakdown': anynetStageMacBreakdown,
  'ch07-mcnn-regnet-linear-staircase':   regnetLinearStaircase,
  'ch07-mcnn-regnet-vs-resnet-tradeoffs': regnetVsResnetTradeoffs,
};
