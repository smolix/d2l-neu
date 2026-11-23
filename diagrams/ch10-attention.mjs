// Slide diagrams for Chapter 10 (Attention Mechanisms): chapter_attention/*.md
import { C, TOKENS, grid, tx as rawTx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const hline = (x1, x2, y) => line(x1, y, x2, y, N.hairline, 1.5, true);

// Text fitting. Labels shrink to fit their slot, but never below FS_FLOOR:
// a label that would need a smaller size wraps onto two lines instead.
// Character widths are realistic averages in em (Source Sans 3 / JetBrains Mono).
const FS_FLOOR = 11;
const charW = (mono, fw) => (mono ? 0.60 : (fw >= 600 ? 0.53 : 0.50));

const fitFs = (t, maxFs, availW, factor = 0.53) =>
  !t ? maxFs : Math.max(FS_FLOOR, Math.min(maxFs, +(((availW - 16) / (String(t).length * factor))).toFixed(1)));

// Candidate two-line splits, best first: at a lone " · " list separator, then
// before "→"/"⇒", after ","/";", or before "+"/"=" near the middle, then at the space
// nearest the middle.
function splitCandidates(t) {
  const s = String(t), mid = s.length / 2, out = [];
  const depth = [];  // bracket depth: never break inside (…) or […]
  for (let i = 0, d = 0; i < s.length; i++) { if ('([{'.includes(s[i])) d++; if (')]}'.includes(s[i])) d--; depth.push(d); }
  const dot = s.indexOf(' · ');
  if (dot > 0 && depth[dot] === 0 && s.indexOf(' · ', dot + 1) < 0) out.push([s.slice(0, dot), s.slice(dot + 3)]);
  const kinds = [i => s[i + 1] === '→' || s[i + 1] === '⇒', i => s[i - 1] === ',' || s[i - 1] === ';', i => s[i + 1] === '+', i => s[i + 1] === '='];
  kinds.push(() => true);  // any top-level space near the middle
  for (const kind of kinds) {
    let bp = -1;
    for (let i = 1; i < s.length - 1; i++) {
      if (s[i] === ' ' && depth[i] === 0 && kind(i) && Math.abs(i - mid) < 0.25 * s.length && (bp < 0 || Math.abs(i - mid) < Math.abs(bp - mid))) bp = i;
    }
    if (bp > 0) out.push([s.slice(0, bp), s.slice(bp + 1)]);
  }
  let best = -1;
  for (let i = 0; i < s.length; i++) if (s[i] === ' ' && (best < 0 || Math.abs(i - mid) < Math.abs(best - mid))) best = i;
  if (best > 0) out.push([s.slice(0, best), s.slice(best + 1)]);
  return out;
}

// Fit one label into availW: {fs, lines}. One line when it fits at FS_FLOOR or
// larger, else two lines (wrapped lines are capped at 13 px).
function fitLines(text, maxFs, availW, mono, fw, nowrap = false) {
  const factor = charW(mono, fw || 500);
  const t = String(text ?? '');
  const fits = u => u.length * factor * FS_FLOOR <= availW - 16;
  if (!nowrap && t && !fits(t)) {
    const cands = splitCandidates(t);
    if (cands.length) {
      const longest = p => (p[0].length > p[1].length ? p[0] : p[1]);
      const parts = cands.find(p => fits(longest(p))) || cands[cands.length - 1];
      return { fs: fitFs(longest(parts), Math.min(maxFs, 13), availW, factor), lines: parts };
    }
  }
  return { fs: fitFs(t, maxFs, availW, factor), lines: [t] };
}

function tx(x, y, text, opts = {}) {
  const maxFs = opts.fs || 15;
  const anchor = opts.anchor || 'middle';
  let availW = opts.availW;
  if (!availW) {
    if (anchor === 'middle') availW = 2 * Math.min(x, 540 - x);
    else if (anchor === 'start') availW = 540 - x;
    else if (anchor === 'end') availW = x;
  }
  const { availW: _ignore, nowrap, ...rest } = opts;
  const { fs, lines } = fitLines(text, maxFs, availW, opts.mono, opts.fw, nowrap);
  if (lines.length === 1) return rawTx(x, y, text, { ...rest, fs });
  const dy = +(fs * 0.62).toFixed(1);
  return rawTx(x, y - dy, lines[0], { ...rest, fs }) + rawTx(x, y + dy, lines[1], { ...rest, fs });
}

// Helper: titled card with 1-4 labels (title, mono l1, muted l2, optional
// extra {t, mono, fs, fw, fill}), each
// fitted to the box width and wrapped onto two lines when needed; the text
// block is centred vertically in the box.
function card(x, y, w, h, fill, stroke, title, l1, l2, tColor = C.ink, fsTitle = 14.5, fsSub = 12.5, extra = null) {
  let o = box(x, y, w, h, fill, stroke, 8, 2);
  const cx = x + w / 2;
  const items = [
    { t: title, mono: false, fw: 700, fill: tColor, fs: fsTitle },
    { t: l1, mono: true, fw: 600, fill: C.ink, fs: fsSub },
    { t: l2, mono: false, fw: 500, fill: C.muted, fs: fsSub },
    extra && { mono: false, fw: 700, fs: fsSub, ...extra },
  ].filter(it => it && it.t);
  const GAP = 7;
  items.forEach(it => { Object.assign(it, fitLines(it.t, it.fs, w, it.mono, it.fw)); it.fs = it.fs; it.lh = it.fs * 1.24; });
  const total = items.reduce((s, it) => s + it.lines.length * it.lh, 0) + GAP * (items.length - 1);
  let cy = y + (h - total) / 2;
  items.forEach(it => {
    it.lines.forEach(ln => {
      o += rawTx(cx, +(cy + it.lh / 2).toFixed(1), ln, { mono: it.mono, fs: it.fs, fw: it.fw, fill: it.fill });
      cy += it.lh;
    });
    cy += GAP;
  });
  return o;
}

// ═══════════════════════════════════════════════════════════════════════════
// §10.0 index.md (3 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 1. ch10-attn-rnn-vs-lookup
function rnnVsLookup() {
  const W = 540, H = 280;
  let o = '';
  // Top panel: RNN Fixed-Width Bottleneck
  o += tx(20, 20, '(a) Recurrent Bottleneck: O(n) steps into 1 fixed vector', { fs: 15, fw: 700, fill: D.red.dark, anchor: 'start' });
  const xs = [24, 126, 228];
  const labs = ['h₁', 'h₂', 'hₙ'];
  xs.forEach((x, i) => {
    o += box(x, 38, 66, 42, D.red.tint, D.red.base, 6, 2);
    o += tx(x + 33, 59, labs[i], { mono: true, fs: 14, fw: 700, fill: D.red.dark });
  });
  o += arrow(94, 59, 122, 59, D.red.base);
  o += arrow(196, 59, 224, 59, D.red.base, true);
  o += arrow(298, 59, 332, 59, D.red.base);
  o += card(336, 36, 184, 46, '#FFFFFF', D.red.base, '768 floats only', '786K nums squeezed', null, D.red.dark, 13.5, 12);
  o += tx(270, 102, 'Early details compressed before knowing the final query', { fs: 13, fw: 600, fill: C.muted });

  o += hline(20, 520, 124);

  // Bottom panel: Attention Direct Content Lookup
  o += tx(20, 146, '(b) Attention Lookup: O(1) direct retrieval over all m positions', { fs: 15, fw: 700, fill: D.green.dark, anchor: 'start' });
  o += card(24, 166, 152, 62, C.lblue, C.blue, 'Stored Keys & Values', '(k₁,v₁) … (kₘ,vₘ)', null, D.blue.dark, 13.5, 12);
  o += arrow(180, 197, 214, 197, C.blue);
  o += card(218, 166, 138, 62, C.lamber, C.amber, 'Query Score', 'αᵢ ∝ exp a(q, kᵢ)', null, D.orange.dark, 13.5, 12);
  o += arrow(360, 197, 394, 197, C.green);
  o += card(398, 166, 122, 62, C.lgreen, C.green, 'Convex Sum', '∑ αᵢ vᵢ', null, D.green.dark, 13.5, 13);
  o += tx(270, 252, 'Differentiable soft retrieval with 1-hop gradient path to every token', { fs: 13, fw: 600, fill: D.green.dark });
  return svg(W, H, o);
}

// 2. ch10-attn-three-challenges
function threeChallenges() {
  const W = 540, H = 286;
  let o = '';
  o += card(140, 12, 260, 44, D.purple.tint, C.purple, 'All-Pairs Content Lookup', '∑ α(q, kᵢ) vᵢ  (O(1) path length)', null, D.purple.dark, 14.5, 12.5);

  const cols = [20, 194, 368];
  const cw = 152;
  cols.forEach((x) => {
    o += arrow(270, 60, x + cw / 2, 86, C.purple);
  });

  // 3 challenge pillars
  o += card(cols[0], 90, cw, 78, C.lamber, C.amber, '1. Order Blind', 'f(ΠX) = Π f(X)', 'ignores sequence order', D.orange.dark, 14, 12);
  o += card(cols[1], 90, cw, 78, D.red.tint, D.red.base, '2. O(n²) Cost', 'n² scores & memory', '64 GiB at n=128K', D.red.dark, 14, 12);
  o += card(cols[2], 90, cw, 78, C.lblue, C.blue, '3. Opacity', 'weights ≠ function', 'where vs. what writes', D.blue.dark, 14, 12);

  cols.forEach((x) => {
    o += arrow(x + cw / 2, 172, x + cw / 2, 198, C.green);
  });

  o += card(cols[0], 202, cw, 68, C.lgreen, C.green, '§10.4 Positions', 'RoPE & ALiBi', 'relative offsets', D.green.dark, 13.5, 12);
  o += card(cols[1], 202, cw, 68, C.lgreen, C.green, '§§10.5–10.6 Scale', 'FlashAttention', 'windows & linear', D.green.dark, 13.5, 12);
  o += card(cols[2], 202, cw, 68, C.lgreen, C.green, '§10.7 Circuits', 'QK / OV & Stream', '2-layer induction', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 3. ch10-attn-roadmap
function roadmap() {
  const W = 540, H = 276;
  let o = '';
  const stages = [
    { x: 20,  y: 22,  w: 236, h: 96, fill: C.lblue, st: C.blue, dk: D.blue.dark, title: 'Part 1: Build the Layer', s1: '§10.1 Q, K, V & Kernel Pooling', s2: '§10.2 Scaled Dot & Masking · §10.3 MHA' },
    { x: 284, y: 22,  w: 236, h: 96, fill: C.lamber, st: C.amber, dk: D.orange.dark, title: 'Part 2: Sequence Order', s1: '§10.4 Positional Information', s2: 'Sinusoidal · RoPE · ALiBi · YaRN' },
    { x: 284, y: 156, w: 236, h: 96, fill: C.lgreen, st: C.green, dk: D.green.dark, title: 'Part 3: Scale & Speed', s1: '§10.5 Online Softmax & FlashAttn', s2: '§10.6 Sliding Windows & Linear Attn' },
    { x: 20,  y: 156, w: 236, h: 96, fill: D.purple.tint, st: C.purple, dk: D.purple.dark, title: 'Part 4: Interpretation', s1: '§10.7 Attention Circuits', s2: 'Residual Stream · QK/OV · Induction' },
  ];
  stages.forEach((s) => {
    o += card(s.x, s.y, s.w, s.h, s.fill, s.st, s.title, s.s1, s.s2, s.dk, 14.5, 12);
  });
  o += arrow(260, 70, 280, 70, C.blue);
  o += arrow(402, 122, 402, 152, C.amber);
  o += arrow(280, 204, 260, 204, C.green);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §10.1 queries-keys-values.md (9 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 4. ch10-attn-qkv-state-vs-db
function qkvStateVsDb() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 20, 232, 102, D.red.tint, D.red.base, 'Recurrent State h_t', '768 floats total', 'compresses 1,024 tokens early', D.red.dark, 15, 12.5);
  o += card(288, 20, 232, 102, C.lblue, C.blue, 'Key–Value Memory D', '1,024 × 768 = 786,432 floats', 'stores all m records verbatim', D.blue.dark, 15, 12.5);

  o += arrow(136, 126, 136, 158, D.red.base);
  o += arrow(404, 126, 404, 158, C.blue);

  o += card(20, 162, 232, 84, '#FFFFFF', D.red.base, 'O(1) Memory, Lossy Recall', 'O(m) gradient hops to token 1', 'information bottleneck', D.red.dark, 13.5, 12);
  o += card(288, 162, 232, 84, C.lgreen, C.green, 'O(md) Memory, Exact Access', 'O(1) direct query–key score', 'retrieves context on demand', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 5. ch10-attn-qkv-hard-vs-soft
function qkvHardVsSoft() {
  const W = 540, H = 266;
  let o = '';
  // Left: Hard Lookup
  o += card(20, 18, 234, 104, D.red.tint, D.red.base, 'Hard Database Lookup', 'q = 5 → 20,  q = 4 → miss!', 'piecewise-constant step', D.red.dark, 14.5, 12.5);
  o += arrow(137, 126, 137, 156, D.red.base);
  o += card(20, 160, 234, 84, '#FFFFFF', D.red.base, 'Zero Gradient Everywhere', '∂Lookup / ∂q = 0 a.e.', 'cannot train with backprop', D.red.dark, 14, 12.5);

  // Right: Soft Lookup
  o += card(286, 18, 234, 104, C.lblue, C.blue, 'Soft Differentiable Lookup', 'a(q, kᵢ) = −(q − kᵢ)²', 'q = 4 → 0.05·10 + 0.95·20 = 19.5', D.blue.dark, 14.5, 12.5);
  o += arrow(403, 126, 403, 156, C.green);
  o += card(286, 160, 234, 84, C.lgreen, C.green, 'Smooth Nonzero Gradients', '∂Attention / ∂(q, kᵢ, vᵢ) ≠ 0', 'end-to-end trainable via SGD', D.green.dark, 14, 12.5);
  return svg(W, H, o);
}

// 6. ch10-attn-qkv-scalar-trace
function qkvScalarTrace() {
  const W = 540, H = 276;
  let o = '';
  o += card(20, 88, 96, 84, D.purple.tint, C.purple, 'Query', 'q = 4', null, D.purple.dark, 15, 14);
  o += arrow(120, 108, 156, 70, C.purple);
  o += arrow(120, 152, 156, 190, C.purple);

  o += card(160, 26, 176, 82, C.lblue, C.blue, 'Record 1: (k₁=2, v₁=10)', 'a₁ = −(4−2)² = −4', 'α₁ = e⁻⁴/(e⁻⁴+e⁻¹) = 0.047', D.blue.dark, 13.5, 11.5);
  o += card(160, 152, 176, 82, C.lamber, C.amber, 'Record 2: (k₂=5, v₂=20)', 'a₂ = −(4−5)² = −1', 'α₂ = e⁻¹/(e⁻⁴+e⁻¹) = 0.953', D.orange.dark, 13.5, 11.5);

  o += arrow(340, 67, 376, 108, C.blue);
  o += arrow(340, 193, 376, 152, C.amber);

  o += card(380, 82, 140, 96, C.lgreen, C.green, 'Weighted Sum', '0.047(10)+0.953(20)', 'Output = 19.53', D.green.dark, 14, 12);
  o += tx(270, 256, 'Midpoint q = 3.5 scores (−2.25, −2.25) → weights (0.5, 0.5) → output 15.0', { fs: 13, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// 7. ch10-attn-qkv-simplex-spectrum
function qkvSimplexSpectrum() {
  const W = 540, H = 264;
  let o = '';
  o += tx(270, 22, 'The Attention Interpolation Spectrum on the Probability Simplex', { fs: 15.5, fw: 700, fill: C.ink });
  const cw = 154, y = 46, ch = 128;
  o += card(20, y, cw, ch, C.lamber, C.amber, 'Sharp Limit (σ → 0)', 'α = [0, …, 1, …, 0]', '1-NN / Argmax Lookup', D.orange.dark, 14, 12);
  o += card(193, y, cw, ch, C.lblue, C.blue, 'Soft Attention (0 < σ < ∞)', 'αᵢ ≥ 0,  ∑ αᵢ = 1', 'Convex Value Mixture', D.blue.dark, 14, 12);
  o += card(366, y, cw, ch, C.lgreen, C.green, 'Smooth Limit (σ → ∞)', 'α = [1/m, …, 1/m]', 'Global Average Pool', D.green.dark, 14, 12);

  o += arrow(178, y + ch / 2, 189, y + ch / 2, C.gray);
  o += arrow(351, y + ch / 2, 362, y + ch / 2, C.gray);

  o += card(20, 194, 500, 52, D.purple.tint, C.purple, 'Decoupled Spaces: Queries & Keys in ℝᵈ  ·  Values & Output in ℝᵈᵛ', 'Attention(q, D) = ∑ᵢ α(q, kᵢ) vᵢ ∈ ConvexHull(v₁, …, vₘ)', null, D.purple.dark, 13.5, 12.5);
  return svg(W, H, o);
}

// 8. ch10-attn-qkv-shift-invariance
function qkvShiftInvariance() {
  const W = 540, H = 260;
  let o = '';
  o += card(20, 24, 144, 88, C.lblue, C.blue, 'Raw Scores a', 'a = [102, 100, 99]', 'risk of exp overflow', D.blue.dark, 14, 12);
  o += arrow(168, 68, 194, 68, C.purple);
  o += card(198, 24, 144, 88, D.purple.tint, C.purple, 'Subtract Max M', 'a − 102 = [0, −2, −3]', 'all exponents ≤ 0', D.purple.dark, 14, 12);
  o += arrow(346, 68, 372, 68, C.green);
  o += card(376, 24, 144, 88, C.lgreen, C.green, 'Softmax Weights', 'α = [0.84, 0.11, 0.04]', 'exact same simplex!', D.green.dark, 14, 12);

  o += card(20, 142, 500, 96, C.lamber, C.amber, 'Why Shift Invariance Holds & Enables End-to-End Backprop', 'αᵢ = exp(aᵢ + c) / ∑ⱼ exp(aⱼ + c) = 1 / (1 + ∑ⱼ≠ᵢ exp(aⱼ − aᵢ))', 'Only pairwise score differences (aⱼ − aᵢ) affect weights and gradients', D.orange.dark, 14, 12.5);
  return svg(W, H, o);
}

// 9. ch10-attn-qkv-vector-trace
function qkvVectorTrace() {
  const W = 540, H = 278;
  let o = '';
  o += tx(270, 20, '2D Vector Lookup for Query q = (1, 1) with Score a = −‖q − kᵢ‖²', { fs: 15, fw: 700, fill: C.ink });
  o += card(20, 38, 152, 86, C.lblue, C.blue, '1. Exact Key Match', 'k₁=(1,1) → s₁=0', 'α₁ = 0.67 · v₁=(1,3)', D.blue.dark, 13.5, 12);
  o += card(194, 38, 152, 86, D.red.tint, D.red.base, '2. Zero-Value Dilution', 'k₂=(2,1) → s₂=−1', 'α₂ = 0.24 · v₂=(0,0)!', D.red.dark, 13.5, 12);
  o += card(368, 38, 152, 86, C.lamber, C.amber, '3. Farther Key', 'k₃=(0,0) → s₃=−2', 'α₃ = 0.09 · v₃=(4,1)', D.orange.dark, 13.5, 12);

  o += arrow(96, 128, 220, 166, C.blue);
  o += arrow(270, 128, 270, 166, D.red.base);
  o += arrow(444, 128, 320, 166, C.amber);

  o += card(90, 170, 360, 88, C.lgreen, C.green, 'Blended Output: o = (1.03, 2.09)', '0.67(1,3) + 0.24(0,0) + 0.09(4,1) ≈ (1.03, 2.09)', 'v₂=(0,0) contributes 0 yet pulls o toward the origin by 24%', D.green.dark, 14.5, 12.5);
  return svg(W, H, o);
}

// 10. ch10-attn-qkv-memory-scaling
function qkvMemoryScaling() {
  const W = 540, H = 266;
  let o = '';
  o += tx(270, 22, 'Stored Floats at Context Length m = 1,024 (GPT-2 Small, d = 768)', { fs: 15, fw: 700, fill: C.ink });
  o += card(20, 44, 154, 114, N.panel, C.gray, 'Recurrent State', 'h_t ∈ ℝ⁷⁶⁸', '768 floats (O(1))', C.ink, 14, 12.5);
  o += card(193, 44, 154, 114, C.lblue, C.blue, '1 Attention Head', 'K, V ∈ ℝ¹⁰²⁴×⁶⁴', '131,072 floats', D.blue.dark, 14, 12.5);
  o += card(366, 44, 154, 114, D.purple.tint, C.purple, '12-Head Layer', '12 × 2 × 1024 × 64', '1,572,864 floats', D.purple.dark, 14, 12.5);

  o += card(20, 180, 500, 68, C.lgreen, C.green, 'Architectural Trade-Off: Zero Compression Bottleneck', 'Parameter count is independent of m, while KV storage grows linearly as O(md)', null, D.green.dark, 14, 12.5);
  return svg(W, H, o);
}

// 11. ch10-attn-qkv-temperature-dial
function qkvTemperatureDial() {
  const W = 540, H = 266;
  let o = '';
  o += card(20, 16, 500, 56, D.purple.tint, C.purple, 'Gaussian Kernel as Softmax Temperature: β = 1 / σ²', 'αᵢ(q) = softmaxᵢ( (1 / σ²) · (−½ ‖q − kᵢ‖²) )', null, D.purple.dark, 14.5, 13);

  o += card(20, 92, 236, 152, C.lamber, C.amber, 'Narrow Bandwidth (σ = 0.1)', 'β = 1/σ² = 100 (400× gap vs σ=2)', 'Low bias · High noise variance', D.orange.dark, 14, 12, { t: 'n_eff → 1  (1-NN interpolation)', mono: true, fs: 12.5, fw: 700, fill: D.orange.dark });

  o += card(284, 92, 236, 152, C.lblue, C.blue, 'Wide Bandwidth (σ = 2.0)', 'β = 1/σ² = 0.25 (compressed gaps)', 'High bias · Low variance (s²/n)', D.blue.dark, 14, 12, { t: 'n_eff → n  (global mean ȳ)', mono: true, fs: 12.5, fw: 700, fill: D.blue.dark });
  return svg(W, H, o);
}

// 12. ch10-attn-qkv-underflow-pitfall
function qkvUnderflowPitfall() {
  const W = 540, H = 276;
  let o = '';
  o += tx(270, 20, 'Out-of-Range Query q = 15 (Training Keys in [0, 5), σ = 0.5)', { fs: 15, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 102, D.red.tint, D.red.base, 'Naive Kernel Division', '−(q − xᵢ)²/(2σ²) < −200', 'float32 exp(−200) → 0.0 / 0.0 = NaN!', D.red.dark, 14, 12);
  o += card(284, 42, 236, 102, C.lgreen, C.green, 'Stable Softmax on Log-Scores', 'subtracts M = maxⱼ aⱼ first', 'largest exponent is e⁰ = 1 (finite!)', D.green.dark, 14, 12);

  o += card(20, 166, 500, 90, C.lamber, C.amber, 'Statistical Limit as q → ∞: Convex-Hull Boundary Clamping', 'Closest boundary key x_max wins 100% of softmax weight: f(q) → y_max', 'No convex combination of stored labels can extrapolate a linear trend!', D.orange.dark, 14, 12.5);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §10.2 attention-scoring.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 13. ch10-attn-score-two-reqs
function scoreTwoReqs() {
  const W = 540, H = 266;
  let o = '';
  o += card(20, 20, 236, 110, C.lblue, C.blue, '1. Hardware-Native Scoring', 'n = 1,024 → n² = 1,048,576 pairs', 'must run as 1 Tensor Core GEMM', D.blue.dark, 14.5, 12);
  o += card(284, 20, 236, 110, C.lamber, C.amber, '2. Boolean Pair Masking', 'exclude <pad> & 523,776 future pairs', 'zero weight without float16 NaN', D.orange.dark, 14.5, 12);

  o += arrow(138, 134, 210, 168, C.blue);
  o += arrow(402, 134, 330, 168, C.amber);

  o += card(50, 172, 440, 76, C.lgreen, C.green, 'Scaled Masked Dot-Product Attention', 'masked_softmax( Q Kᵀ / √d ,  padding_mask & causal_mask ) V', null, D.green.dark, 15, 13);
  return svg(W, H, o);
}

// 14. ch10-attn-score-gaussian-to-dot
function scoreGaussianToDot() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 18, 500, 56, C.lblue, C.blue, 'Gaussian Expansion:  −½ ‖q − kᵢ‖² = qᵀkᵢ − ½‖kᵢ‖² − ½‖q‖²', 'Three terms inside the softmax exponent:', null, D.blue.dark, 14.5, 12.5);

  const cw = 154, y = 96, ch = 152;
  o += card(20, y, cw, ch, C.lgreen, C.green, 'Term 1: qᵀkᵢ', 'Bilinear Alignment', 'kept as the dot score', D.green.dark, 14, 12);
  o += card(193, y, cw, ch, C.lamber, C.amber, 'Term 2: −½‖kᵢ‖²', 'Key Norm Penalty', 'dropped → ‖kᵢ‖ is prior', D.orange.dark, 14, 12);
  o += card(366, y, cw, ch, N.panel, C.gray, 'Term 3: −½‖q‖²', 'Constant Across i', 'cancels in softmax!', C.ink, 14, 12);
  return svg(W, H, o);
}

// 15. ch10-attn-score-additive-vs-dot
function scoreAdditiveVsDot() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 20, 236, 112, D.red.tint, D.red.base, 'Additive (MLP) Attention', 'wᵥᵀ tanh(W_q q + W_k k)', 'materializes 3D (n × m × r) tensor', D.red.dark, 14.5, 12);
  o += card(284, 20, 236, 112, C.lgreen, C.green, 'Scaled Dot-Product', '(W_q q)ᵀ (W_k k) / √d', 'single 2D (n × m) GEMM QKᵀ', D.green.dark, 14.5, 12);

  o += card(20, 152, 236, 98, '#FFFFFF', D.red.base, 'Small-Init Linear Trap', 'near 0: tanh(u) ≈ u is additive', 'W_q q cancels across keys!', D.red.dark, 13.5, 12);
  o += card(284, 152, 236, 98, C.lblue, C.blue, 'Bilinear at All Scales', 'couples q and k multiplicatively', 'high arithmetic intensity on GPUs', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 16. ch10-attn-score-variance-scaling
function scoreVarianceScaling() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 16, 500, 56, D.purple.tint, C.purple, 'Why Divide by √d?  (qⱼ, kⱼ ~ N(0, 1) independent)', 'E[qᵀk] = 0,   Var(qᵀk) = ∑ⱼ Var(qⱼ kⱼ) = d   ⇒   Std(qᵀk) = √d', null, D.purple.dark, 14, 12.5);

  o += card(20, 92, 236, 156, D.red.tint, D.red.base, 'Unscaled: qᵀk', 'Std = 8 at d=64; 22.6 at d=512', 'Softmax saturates to one-hot (H → 0)', D.red.dark, 14, 12, { t: 'Jacobian ∂α/∂a → 0 (vanishing grad)', fs: 12, fw: 700, fill: D.red.dark });

  o += card(284, 92, 236, 156, C.lgreen, C.green, 'Scaled: qᵀk / √d', 'Var(qᵀk / √d) = d / d = 1 always', 'Entropy H ≈ log m − ½ across all d', D.green.dark, 14, 12, { t: 'Well-conditioned gradients at init', fs: 12, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// 17. ch10-attn-score-worked-trace
function scoreWorkedTrace() {
  const W = 540, H = 272;
  let o = '';
  o += tx(270, 20, 'Worked Trace (d = 4, √d = 2): 2 Queries × 3 Key–Value Pairs', { fs: 15, fw: 700, fill: C.ink });

  const stages = [
    { x: 18,  w: 112, fill: C.lblue, st: C.blue, dk: D.blue.dark, t: '1. Raw qᵀk', l1: 'q₁: (4, 0, 0)', l2: 'q₂: (0, 2, 2)' },
    { x: 146, w: 112, fill: D.purple.tint, st: C.purple, dk: D.purple.dark, t: '2. Scale ÷2', l1: 'q₁: (2, 0, 0)', l2: 'q₂: (0, 1, 1)' },
    { x: 274, w: 120, fill: C.lamber, st: C.amber, dk: D.orange.dark, t: '3. Softmax α', l1: '(.79, .11, .11)', l2: '(.16, .42, .42)' },
    { x: 410, w: 112, fill: C.lgreen, st: C.green, dk: D.green.dark, t: '4. Mix αV', l1: 'o₁=(.89, .21)', l2: 'o₂=(.58, .85)' },
  ];
  stages.forEach((s, i) => {
    o += card(s.x, 42, s.w, 108, s.fill, s.st, s.t, s.l1, s.l2, s.dk, 13.5, 11.5);
    if (i < 3) o += arrow(s.x + s.w + 3, 96, stages[i + 1].x - 3, 96, C.gray);
  });

  o += card(18, 172, 504, 80, N.panel, C.gray, 'Two Key Insights from the Hand Calculation', '1. Orthogonal keys (score 0) still get e⁰ = 1 → 10.7% weight', '2. Tied active coordinates (k₂, k₃ under q₂) split remaining mass equally (0.422 each)', C.ink, 13.5, 12);
  return svg(W, H, o);
}

// 18. ch10-attn-score-jacobian-saturation
function scoreJacobianSaturation() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 16, 500, 54, D.purple.tint, C.purple, 'Softmax Jacobian:  J = ∂α / ∂a = diag(α) − α αᵀ', 'Entry (i, j):  ∂αᵢ / ∂aⱼ = αᵢ (δᵢⱼ − αⱼ)', null, D.purple.dark, 14.5, 12.5);

  o += card(20, 90, 236, 156, C.lgreen, C.green, 'Balanced Regime (Scaled)', 'αᵢ ≈ 1/m  ⇒  ‖J‖_{F} ≈ 0.18', 'Strong cross-key gradient coupling', D.green.dark, 14, 12, { t: 'Peak ‖J‖_{F} = ½ at k=2 split', mono: true, fs: 12, fw: 700, fill: D.green.dark });

  o += card(284, 90, 236, 156, D.red.tint, D.red.base, 'One-Hot Corner (Unscaled)', 'α → e_i*  ⇒  αᵢ(1 − αᵢ) → 0', 'diag(e_i*) − e_i* e_i*ᵀ = 0 matrix!', D.red.dark, 14, 12, { t: '∂L/∂a = J (∂L/∂α) → 0 (frozen!)', mono: true, fs: 12, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

// 19. ch10-attn-score-mask-fill-hazards
function scoreMaskFillHazards() {
  const W = 540, H = 276;
  let o = '';
  o += tx(270, 20, 'Three Candidate Mask Fill Values Across Precision & Empty Rows', { fs: 15, fw: 700, fill: C.ink });
  const cw = 156, y = 40, ch = 128;
  o += card(18, y, cw, ch, D.red.tint, D.red.base, 'Hardcoded −1e6', 'Overflows float16!', '(max finite = 65,504)', D.red.dark, 14, 12);
  o += card(192, y, cw, ch, C.lamber, C.amber, 'Literal −inf', 'All-False row max = −inf', '−inf − (−inf) = NaN!', D.orange.dark, 14, 12);
  o += card(366, y, cw, ch, C.lgreen, C.green, 'finfo(dtype).min', 'Finite in fp16/bf16/fp32', '+ where(mask.any(-1))', D.green.dark, 14, 12);

  o += card(18, 186, 504, 72, D.red.tint, D.red.base, 'Why One NaN Padding Row Corrupts Every Token in Layer 2', 'In Layer 2 weights × values, 0.0 weight × NaN value = NaN infects every valid token!', null, D.red.dark, 13.5, 12);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §10.3 multihead-attention.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 20. ch10-attn-mha-multi-arg
function mhaMultiArg() {
  const W = 540, H = 266;
  let o = '';
  o += tx(270, 20, '"Alice gave Bob the book" — Verb "gave" Needs 3 Distinct Roles', { fs: 15, fw: 700, fill: C.ink });
  o += card(20, 44, 152, 96, C.lblue, C.blue, 'Head 1: Giver', 'attends to "Alice"', 'α₁ ≈ [0.92, 0, …]', D.blue.dark, 14, 12);
  o += card(194, 44, 152, 96, C.lamber, C.amber, 'Head 2: Recipient', 'attends to "Bob"', 'α₂ ≈ [0, 0, 0.90, …]', D.orange.dark, 14, 12);
  o += card(368, 44, 152, 96, C.lgreen, C.green, 'Head 3: Object', 'attends to "book"', 'α₃ ≈ […, 0.94]', D.green.dark, 14, 12);

  o += arrow(96, 144, 210, 176, C.blue);
  o += arrow(270, 144, 270, 176, C.amber);
  o += arrow(444, 144, 330, 176, C.green);

  o += card(50, 180, 440, 68, D.purple.tint, C.purple, 'Concatenate & Project: [Z₁, Z₂, …, Z_H] W_o', 'Preserves all retrieved arguments in separate subspaces without averaging!', null, D.purple.dark, 14, 12.5);
  return svg(W, H, o);
}

// 21. ch10-attn-mha-qkv-projections
function mhaQkvProjections() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 88, 114, 88, N.panel, C.gray, 'Input X', 'shape (n, d)', '1 token stream', C.ink, 15, 12.5);

  o += arrow(138, 104, 182, 54, C.blue);
  o += arrow(138, 132, 182, 132, C.amber);
  o += arrow(138, 160, 182, 210, C.green);

  o += card(186, 18, 156, 68, C.lblue, C.blue, 'Queries Q = X W_q', '(n, d_h) · what it seeks', null, D.blue.dark, 13.5, 11.5);
  o += card(186, 98, 156, 68, C.lamber, C.amber, 'Keys K = X W_k', '(n, d_h) · what it offers', null, D.orange.dark, 13.5, 11.5);
  o += card(186, 178, 156, 68, C.lgreen, C.green, 'Values V = X W_v', '(n, d_v) · content sent', null, D.green.dark, 13.5, 11.5);

  o += arrow(346, 52, 384, 106, C.blue);
  o += arrow(346, 132, 384, 126, C.amber);
  o += arrow(346, 212, 384, 156, C.green);

  o += card(388, 84, 132, 96, D.purple.tint, C.purple, 'Head Output', 'softmax(QKᵀ/√d_h)V', 'shape (n, d_v)', D.purple.dark, 13.5, 11.5);
  return svg(W, H, o);
}

// 22. ch10-attn-mha-asymmetry
function mhaAsymmetry() {
  const W = 540, H = 266;
  let o = '';
  o += card(20, 20, 236, 112, D.red.tint, D.red.base, 'Tied Map: W_q = W_k = W', 'uᵢ = xᵢ W  ⇒  sᵢⱼ = uᵢᵀuⱼ / √d_h', 'Symmetric PSD matrix (sᵢⱼ = sⱼᵢ)', D.red.dark, 14, 12);
  o += card(284, 20, 236, 112, C.lgreen, C.green, 'Separate Maps: W_q ≠ W_k', 'qᵢ = xᵢ W_q,  kⱼ = xⱼ W_k', 'Asymmetric score matrix (sᵢⱼ ≠ sⱼᵢ)', D.green.dark, 14, 12);

  o += card(20, 152, 236, 94, '#FFFFFF', D.red.base, 'Cauchy–Schwarz Bound', 'sᵢⱼ ≤ √(sᵢᵢ sⱼⱼ) ≤ max(sᵢᵢ, sⱼⱼ)', 'verb and noun cannot each prefer the other over itself', D.red.dark, 13.5, 12);
  o += card(284, 152, 236, 94, C.lblue, C.blue, 'Directed Syntactic Lookup', 'verb query aligns with noun key', 'while self-score sᵢᵢ can be negative', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 23. ch10-attn-mha-copy-both-bound
function mhaCopyBothBound() {
  const W = 540, H = 276;
  let o = '';
  o += tx(270, 20, 'Copy-Both Lower Bound: Target t = [v₁; v₂] ∈ ℝ²ᵈᵛ (Var = 2 per coord)', { fs: 14.5, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 116, D.red.tint, D.red.base, '1 Head: 1 Mixture m ∈ ℝᵈᵛ', 'm = α₁ v₁ + α₂ v₂', 'Bayes MSE(v₁|m) = α₂²/(α₁²+α₂²)', D.red.dark, 14, 12);
  o += card(284, 42, 236, 116, C.lgreen, C.green, '2 Heads: 2 Mixtures [m₁; m₂]', 'm₁ = α v₁ + (1−α)v₂, m₂ = (1−α)v₁ + α v₂', 'Rank-2 mixing matrix A is invertible!', D.green.dark, 14, 11.5);

  o += card(20, 172, 236, 92, '#FFFFFF', D.red.base, 'Rel. Error = 1/2 for EVERY (α₁, α₂)', 'α₂²/(α₁²+α₂²) + α₁²/(α₁²+α₂²) = 1 (of variance 2)', 'RMS relative error = √(1/2) ≈ 0.707', D.red.dark, 13, 11.5);
  o += card(284, 172, 236, 92, C.lblue, C.blue, 'Error → 0 (10⁻⁷ in fp32)', 'General k-value bound: 1 − rank(A)/k', 'Needs rank(A) = k AND H d_h ≥ k d_v', D.blue.dark, 13, 11.5);
  return svg(W, H, o);
}

// 24. ch10-attn-mha-flop-invariance
function mhaFlopInvariance() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 16, 500, 56, D.purple.tint, C.purple, 'Why Splitting d into H Heads of Width d_h = d/H Is Compute-Free', 'H slices of shape (d × d/H) pack into one (d × d) matrix for Q, K, V, O', null, D.purple.dark, 14, 12.5);

  o += card(20, 92, 236, 102, C.lblue, C.blue, 'Projections: 8 n d² FLOPs', '4 matrices × 2 n d² FLOPs', '4 d² parameters (2.36M at d=768)', D.blue.dark, 14, 12);
  o += card(284, 92, 236, 102, C.lamber, C.amber, 'Scores & Mixing: 4 n² d FLOPs', 'H heads × (2 n² d_h + 2 n² d_h)', '= 4 n² (H d_h) = 4 n² d · independent of H!', D.orange.dark, 14, 11.5);

  o += card(20, 210, 500, 44, C.lgreen, C.green, 'Only the O(H n²) softmax overhead depends on H  ·  Standard d_h ∈ {64, 128}', null, null, D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 25. ch10-attn-mha-reshape-bug
function mhaReshapeBug() {
  const W = 540, H = 272;
  let o = '';
  o += tx(270, 20, 'Splitting Projected Tensor X of Shape (B, n, H · d_h) into Heads', { fs: 15, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 124, C.lgreen, C.green, 'Correct: reshape → swap axes', 'reshape(B, n, H, d_h) → swap axes 1, 2', 'Slices each token’s d features into H heads', D.green.dark, 13.5, 11.5);
  o += card(284, 42, 236, 124, D.red.tint, D.red.base, 'Silent Bug: direct reshape', 'X.reshape(B, H, n, d_h)', 'Packs first n/H tokens into Head 0!', D.red.dark, 13.5, 11.5);

  o += card(20, 184, 500, 70, C.lamber, C.amber, 'Why No Compiler or Shape Assertion Catches the Bug', 'Both expressions return the exact same shape (B, H, n, d_h)!', 'Direct reshape mixes different sequence positions across feature channels.', D.orange.dark, 14, 12);
  return svg(W, H, o);
}

// 26. ch10-attn-mha-self-vs-cross
function mhaSelfVsCross() {
  const W = 540, H = 266;
  let o = '';
  o += card(20, 20, 236, 144, C.lblue, C.blue, 'Self-Attention', 'attention(X, X, X)', 'Q, K, V all from X ∈ ℝⁿˣᵈ', D.blue.dark, 15, 12.5, { t: 'n × n score matrix', mono: true, fs: 13, fw: 700, fill: D.blue.dark });

  o += card(284, 20, 236, 144, C.lamber, C.amber, 'Cross-Attention', 'attention(A, B, B)', 'Q from target A (n) · K,V from B (m)', D.orange.dark, 15, 12.5, { t: 'n × m score matrix', mono: true, fs: 13, fw: 700, fill: D.orange.dark });

  o += card(20, 184, 500, 64, D.purple.tint, C.purple, 'Permutation Property of Unmasked Cross-Attention', 'Permuting source rows B ↦ ΠB leaves every output row of A unchanged!', null, D.purple.dark, 14, 12.5);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §10.4 positional-information.md (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 27. ch10-attn-pos-order-blindness
function posOrderBlindness() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 20, 236, 98, C.lblue, C.blue, '"dog bites man"', 'X = [e_dog; e_bites; e_man]', 'Unmasked Self-Attention → Y', D.blue.dark, 14.5, 12);
  o += card(284, 20, 236, 98, C.lamber, C.amber, '"man bites dog"', 'ΠX = [e_man; e_bites; e_dog]', 'Unmasked Self-Attention → ΠY', D.orange.dark, 14.5, 12);

  o += arrow(138, 122, 220, 162, C.blue);
  o += arrow(402, 122, 320, 162, C.amber);

  o += card(40, 166, 460, 82, D.red.tint, D.red.base, 'Identical Set of Output Vectors (Order Blindness)', 'Mean/Sum Pooling gives the exact same vector for both sentences!', 'Position must be injected into embeddings or attention scores.', D.red.dark, 14, 12);
  return svg(W, H, o);
}

// 28. ch10-attn-pos-equivariance
function posEquivariance() {
  const W = 540, H = 270;
  let o = '';
  o += tx(270, 20, 'Four-Step Proof of Permutation Equivariance: f(ΠX) = Π f(X)', { fs: 15, fw: 700, fill: C.ink });
  const steps = [
    { x: 20,  y: 42,  title: '1. Row-Wise Projections', s1: '(ΠX)W = Π(XW) for Q, K, V', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 284, y: 42,  title: '2. Permuted Score Matrix', s1: 'S(ΠX) = Π S Πᵀ', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 284, y: 154, title: '3. Softmax Commutes', s1: 'softmax(ΠSΠᵀ) = Π A Πᵀ', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 20,  y: 154, title: '4. Orthogonal Cancellation', s1: '(ΠAΠᵀ)(ΠV) = ΠAV (ΠᵀΠ = I)', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  steps.forEach((s) => {
    o += card(s.x, s.y, 236, 86, s.fill, s.st, s.title, s.s1, null, s.dk, 14, 12.5);
  });
  o += arrow(260, 85, 280, 85, C.blue);
  o += arrow(402, 132, 402, 150, C.purple);
  o += arrow(280, 197, 260, 197, C.amber);
  return svg(W, H, o);
}

// 29. ch10-attn-pos-learned-table
function posLearnedTable() {
  const W = 540, H = 266;
  let o = '';
  o += tx(270, 20, 'Learned Position Embedding Table P ∈ ℝ^{L_max×d} (Trained at L = 128)', { fs: 14.5, fw: 700, fill: C.ink });
  o += card(20, 44, 236, 114, C.lgreen, C.green, 'Rows 0 … 127 (In-Range)', 'p_i updated on every batch', 'In-range PPL ≈ 9.6 on WikiText-103', D.green.dark, 14, 12);
  o += card(284, 44, 236, 114, D.red.tint, D.red.base, 'Rows 128 … 511 (Unseen)', '∇_(p_i) L = 0 throughout training!', 'Stuck at random N(0, 0.02²) noise', D.red.dark, 14, 12);

  o += card(20, 178, 500, 70, C.lamber, C.amber, 'Extrapolation Failure at Context Length n = 512', 'Positions ≥ 128 add untrained random vectors to token embeddings (PPL jumps to 20–25)', null, D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 30. ch10-attn-pos-planar-rotation
function posPlanarRotation() {
  const W = 540, H = 266;
  let o = '';
  o += card(20, 32, 146, 98, C.lblue, C.blue, 'Code at Pos i', '[sin(i ω_m)]', '[cos(i ω_m)]', D.blue.dark, 14, 13);
  o += arrow(170, 81, 196, 81, C.purple);
  o += card(200, 22, 146, 118, D.purple.tint, C.purple, '2×2 Offset Matrix', 'R(δ ω_m) depends', 'ONLY on offset δ!', D.purple.dark, 14, 12.5);
  o += arrow(350, 81, 376, 81, C.green);
  o += card(380, 32, 140, 98, C.lgreen, C.green, 'Code at i + δ', '[sin((i+δ)ω_m)]', '[cos((i+δ)ω_m)]', D.green.dark, 14, 12.5);

  o += card(20, 164, 500, 82, C.lamber, C.amber, 'Offset-Invariant Sinusoidal Inner Product', 'p_iᵀ p_(i+δ) = ∑_{m} (sin(iω_{m})sin((i+δ)ω_m) + cos(iω_{m})cos((i+δ)ω_m)) = ∑_{m} cos(δ ω_m)', 'Depends purely on relative distance δ — before W_q and W_k mix coordinates!', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 31. ch10-attn-pos-four-terms
function posFourTerms() {
  const W = 540, H = 276;
  let o = '';
  o += tx(270, 20, 'Expanding the Additive Score: (xᵢ + pᵢ)ᵀ A (xⱼ + pⱼ)  with A = W_q W_kᵀ / √d_h', { fs: 14, fw: 700, fill: C.ink });

  o += card(20, 40, 236, 92, C.lgreen, C.green, '1. Content–Content', 'xᵢᵀ A xⱼ', 'Pure semantic match (shift-invariant)', D.green.dark, 14, 12);
  o += card(284, 40, 236, 92, D.red.tint, D.red.base, '2. Content–Position', 'xᵢᵀ A pⱼ', 'Ties query content to ABSOLUTE j!', D.red.dark, 14, 12);
  o += card(20, 152, 236, 92, D.red.tint, D.red.base, '3. Position–Content', 'pᵢᵀ A xⱼ', 'Ties key content to ABSOLUTE i!', D.red.dark, 14, 12);
  o += card(284, 152, 236, 92, C.lamber, C.amber, '4. Position–Position', 'pᵢᵀ A pⱼ', 'Relative only if A is 2×2 block-rot', D.orange.dark, 14, 12);
  return svg(W, H, o);
}

// 32. ch10-attn-pos-alibi-slopes
function posAlibiSlopes() {
  const W = 540, H = 266;
  let o = '';
  o += card(20, 14, 500, 70, C.lblue, C.blue, 'ALiBi Score Rule:  score_{ij}^{(h)} = qᵢᵀkⱼ / √d_h − m_h · (i − j)', 'Geometric head slopes m_h = 2^(−8h/H) · for H=4:  1/4,  1/16,  1/64,  1/256', null, D.blue.dark, 14, 12.5);

  o += card(20, 100, 236, 90, C.lamber, C.amber, 'Steep Heads (m₁ = 1/4)', 'At distance 40: penalty −10 (e⁻¹⁰)', 'Strict local recency focus', D.orange.dark, 14, 12);
  o += card(284, 100, 236, 90, C.lgreen, C.green, 'Gentle Heads (m₄ = 1/256)', 'At distance 128: penalty −0.5', 'Broad context window average', D.green.dark, 14, 12);

  o += card(20, 206, 500, 46, N.panel, C.gray, 'NoPE (No Positional Encoding): relies solely on causal mask prefix length (i + 1)', null, null, C.ink, 13.5, 12);
  return svg(W, H, o);
}

// 33. ch10-attn-pos-five-schemes
function posFiveSchemes() {
  const W = 540, H = 278;
  let o = '';
  o += tx(270, 20, 'Where the Five Positional Schemes Act in TinyCharLM', { fs: 15, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 92, C.lblue, C.blue, 'Input Embedding Layer', 'Learned: H = E(X) + P_learn[0:T]', 'Sinusoidal: H = E(X) + P_sin[0:T]/√d', D.blue.dark, 14, 12);
  o += card(284, 42, 236, 92, D.purple.tint, C.purple, 'Every Attention Block', 'RoPE: rotate q ↦ Rᵢq,  k ↦ Rⱼk', 'ALiBi: add −m_h(i − j) to scores', D.purple.dark, 14, 12);

  o += card(20, 150, 500, 116, C.lgreen, C.green, 'WikiText-103 Perplexity Benchmark (Trained at n = 128 → Tested at n = 512)', 'In-Range (128):  RoPE 7.3  <  Learned 9.6  <  Sinusoidal ≈9.7  <  ALiBi ≈10  <  NoPE ≈11.7', 'Extrap (512):    Slow-Scaled RoPE ≈9  <  ALiBi 9.9 (windowed!)  <  Sin ≈15.5  <  RoPE ≈27', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 34. ch10-attn-pos-fast-slow-yarn
function posFastSlowYarn() {
  const W = 540, H = 274;
  let o = '';
  o += tx(270, 20, '16 RoPE Coordinate Pairs at d_h = 32 (Periods 2π · 10000^(m/16))', { fs: 15, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 108, C.lgreen, C.green, '6 Fast Pairs (m = 0 … 5)', 'Periods 6.3 to 112 tokens (< 128)', 'Already completed full 2π rotations!', D.green.dark, 14, 12);
  o += card(284, 42, 236, 108, D.red.tint, D.red.base, '10 Slow Pairs (m = 6 … 15)', 'Periods 199 to 35,000 tokens (> 128)', 'Rotate into unseen angles at n=512!', D.red.dark, 14, 12);

  o += arrow(138, 154, 138, 178, C.green);
  o += arrow(402, 154, 402, 178, C.purple);

  o += card(20, 182, 236, 72, C.lblue, C.blue, 'Keep Scale = 1.0', 'Preserves local byte resolution', null, D.blue.dark, 13.5, 12);
  o += card(284, 182, 236, 72, D.purple.tint, C.purple, 'Scale by 128/512 = 0.25', 'Maps angles back to trained arc (PPL ≈ 9)', null, D.purple.dark, 13.5, 11.5);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §10.5 attention-at-scale.md (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 35. ch10-attn-scale-memory-wall
function scaleMemoryWall() {
  const W = 540, H = 270;
  let o = '';
  o += tx(270, 20, 'Quadratic Score + Weight Scratch Memory (8n² Bytes/Head in FP32)', { fs: 14.5, fw: 700, fill: C.ink });
  const cw = 154, y = 42, ch = 98;
  o += card(20, y, cw, ch, C.lgreen, C.green, 'n = 1,024', '1.05M score pairs', '8.0 MiB / head', D.green.dark, 14.5, 12);
  o += card(193, y, cw, ch, C.lamber, C.amber, 'n = 8,192', '67.1M score pairs', '512 MiB / head', D.orange.dark, 14.5, 12);
  o += card(366, y, cw, ch, D.red.tint, D.red.base, 'n = 131,072', '17.2B score pairs', '128 GiB / head!', D.red.dark, 14.5, 12);

  o += card(20, 156, 500, 100, D.purple.tint, C.purple, 'The Breakthrough: Exact Attention Never Needs the n × n Matrix', 'Online Softmax (Milakov & Gimelshein, 2018) + FlashAttention (Dao et al., 2022)', 'Computes exact softmax(QKᵀ/√d_h)V in O(n) memory via on-chip SRAM tiling!', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 36. ch10-attn-scale-flop-crossover
function scaleFlopCrossover() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 16, 500, 54, C.lblue, C.blue, 'Self-Attention Layer FLOPs:  8 n d² (projections) + 4 n² d (scores & mixing)', 'Crossover Ratio:  (4 n² d) / (8 n d²) = n / (2d)', null, D.blue.dark, 14, 12.5);

  o += card(20, 90, 236, 154, C.lgreen, C.green, 'Short Context (n < 2d)', 'GPT-2 Small: d=768, n=1,024', 'n / (2d) = 0.67 → Projections dominate', D.green.dark, 14, 12, { t: '4.8 GFLOPs proj vs 3.2 GFLOPs attn', mono: true, fs: 11.5, fw: 700, fill: D.green.dark });

  o += card(284, 90, 236, 154, D.red.tint, D.red.base, 'Long Context (n ≫ 2d)', 'Frontier LLM: d=4,096, n=131,072', 'n / (2d) = 16 → Attention is 16× proj!', D.red.dark, 14, 12, { t: '4× more FLOPs per 2× context length', mono: true, fs: 11.5, fw: 700, fill: D.red.dark });
  return svg(W, H, o);
}

// 37. ch10-attn-scale-two-pass-vs-online
function scaleTwoPassVsOnline() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 20, 236, 110, D.red.tint, D.red.base, 'Naive Two-Pass Softmax', 'Pass 1: m = maxⱼ aⱼ,  s = ∑ exp(aⱼ−m)', 'Pass 2: o = ∑ (exp(aⱼ−m)/s) vⱼ', D.red.dark, 14, 11.5);
  o += card(284, 20, 236, 110, C.lgreen, C.green, 'One-Pass Online Softmax', 'Streams key blocks of size c', 'Updates (m, s, o) on the fly!', D.green.dark, 14, 12);

  o += card(20, 150, 236, 98, '#FFFFFF', D.red.base, 'Requires All n Scores First', 'Materializes n×n matrix in HBM', 'O(n²) memory allocation', D.red.dark, 13.5, 12);
  o += card(284, 150, 236, 98, C.lblue, C.blue, 'Exact Rescaling Repair', 'Rescales (s, o) by exp(m − m′)', 'O(nc) stripe memory (linear in n)', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 38. ch10-attn-scale-online-recurrence
function scaleOnlineRecurrence() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 24, 142, 92, C.lblue, C.blue, 'Prior State', '(m,  s,  o)', 'relative to old m', D.blue.dark, 14.5, 12.5);
  o += arrow(166, 70, 192, 70, C.purple);
  o += card(196, 18, 160, 104, D.purple.tint, C.purple, 'New Block Arrives', 'm′ = max(m, maxⱼ aⱼ)', 'scale = exp(m − m′) ≤ 1', D.purple.dark, 13.5, 12);
  o += arrow(360, 70, 386, 70, C.green);
  o += card(390, 24, 130, 92, C.lgreen, C.green, 'Updated State', '(m′,  s′,  o′)', 'exact to m′!', D.green.dark, 14, 12.5);

  o += box(20, 146, 500, 106, C.lamber, C.amber, 8, 2);
  o += tx(270, 166, 'Multiplicative Reference-Shift Identity', { fs: 13.5, fw: 700, fill: D.orange.dark });
  o += tx(270, 189, 's′ = s · exp(m − m′) + ∑ⱼ exp(aⱼ − m′)', { availW: 500, mono: true, fs: 12, fw: 600, fill: C.ink });
  o += tx(270, 209, 'o′ = o · exp(m − m′) + ∑ⱼ exp(aⱼ − m′) vⱼ', { availW: 500, mono: true, fs: 12, fw: 600, fill: C.ink });
  o += tx(270, 232, 'After final key block:  Output = o / s = softmax(a) V  (exact to machine precision)', { availW: 500, fs: 12, fw: 500, fill: C.muted });
  return svg(W, H, o);
}

// 39. ch10-attn-scale-online-trace
function scaleOnlineTrace() {
  const W = 540, H = 274;
  let o = '';
  o += tx(270, 20, 'Numerical Trace: 4 Keys in Two Blocks of c = 2  (Scores [1, 3] then [5, 2])', { fs: 14, fw: 700, fill: C.ink });

  o += card(20, 42, 152, 114, C.lblue, C.blue, 'After Block 1: (1, 3)', 'm₁ = 3', 's₁ = e⁻² + 1 ≈ 1.1353', D.blue.dark, 13.5, 12);
  o += arrow(176, 99, 194, 99, C.amber);
  o += card(198, 42, 152, 114, C.lamber, C.amber, 'Block 2: (5, 2)', 'm₂ = max(3, 5) = 5', 'rescale e³⁻⁵ = e⁻²', D.orange.dark, 13.5, 12);
  o += arrow(354, 99, 372, 99, C.green);
  o += card(376, 42, 144, 114, C.lgreen, C.green, 'Merged Sum s₂', 's₁·e⁻² + (1 + e⁻³)', 's₂ ≈ 1.2034 (exact!)', D.green.dark, 13.5, 11.5);

  o += card(20, 178, 500, 74, D.purple.tint, C.purple, 'Associative & Commutative Reduction', 's₂ = (e⁻² + 1)e⁻² + 1 + e⁻³ = e⁻⁴ + e⁻² + 1 + e⁻³ ≈ 1.2034', 'Key blocks can be processed in any order or in parallel tree reductions (FlashDecoding)!', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 40. ch10-attn-scale-sram-hierarchy
function scaleSramHierarchy() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 20, 236, 106, C.lgreen, C.green, 'On-Chip SRAM (Registers, L1, L2)', '~0.1 GB  ·  tens of TB/s', 'Holds tiles, never the n × n matrix', D.green.dark, 14, 12);
  o += card(284, 20, 236, 106, D.red.tint, D.red.base, 'Off-Chip Device Memory', 'HBM: 80 GB at 3.35 TB/s (H100)', 'GDDR6X: 24 GB at 1.0 TB/s (RTX 4090)', D.red.dark, 14, 12);

  o += card(20, 148, 500, 100, C.lblue, C.blue, 'The Compute vs. Bandwidth Gap (165 TFLOP/s vs. 1.0 TB/s)', 'RTX 4090 Tensor Cores execute ~164 FP16 operations in the time memory transfers 1 byte', 'Fused kernels keep active tiles (Qᵢ, Kⱼ, Vⱼ, mᵢ, sᵢ, Oᵢ) inside SRAM.', D.blue.dark, 14, 12.5);
  return svg(W, H, o);
}

// 41. ch10-attn-scale-hbm-passes
function scaleHbmPasses() {
  const W = 540, H = 272;
  let o = '';
  o += tx(270, 20, 'Naive Attention Moves 4n² Floats Across Slow HBM', { fs: 15, fw: 700, fill: D.red.dark });
  const passes = [
    { x: 18,  t: 'Pass 1: Write S', s: 'QKᵀ/√d_h → HBM (n²)' },
    { x: 146, t: 'Pass 2: Read S',  s: 'HBM → softmax (n²)' },
    { x: 274, t: 'Pass 3: Write A', s: 'weights → HBM (n²)' },
    { x: 402, t: 'Pass 4: Read A',  s: 'HBM → A V GEMM (n²)' },
  ];
  passes.forEach((p, i) => {
    o += card(p.x, 42, 118, 84, D.red.tint, D.red.base, p.t, p.s, null, D.red.dark, 13, 11);
    if (i < 3) o += arrow(p.x + 120, 84, passes[i + 1].x - 2, 84, D.red.base);
  });

  o += card(18, 142, 504, 120, C.lamber, C.amber, 'Low Arithmetic Intensity:  2n²(d_h + d_v) FLOPs / (4n² · bytes) → (d_h + d_v) / 4 (FP16)', 'At d_h = d_v = 64 in FP16:  Intensity = 32 FLOPs/byte  (vs. ~164 FLOPs/byte RTX 4090 ridge point)', 'Naive attention is severely memory-bandwidth bound at every sequence length n!', D.orange.dark, 13, 12);
  return svg(W, H, o);
}

// 42. ch10-attn-scale-flash-tiling
function scaleFlashTiling() {
  const W = 540, H = 276;
  let o = '';
  o += card(20, 20, 236, 114, C.lgreen, C.green, 'On-Chip SRAM Tile Loop', 'Hold Qᵢ, mᵢ, sᵢ, Oᵢ (b × d_h) in SRAM', 'Stream Kⱼ, Vⱼ tiles from HBM', D.green.dark, 14, 12);
  o += arrow(260, 77, 280, 77, C.green);
  o += card(284, 20, 236, 114, C.lblue, C.blue, 'Single HBM Write per Tile', 'Never writes n × n scores to HBM!', 'Traffic: Θ(n² d_h² / M) floats', D.blue.dark, 14, 12);

  o += card(20, 156, 500, 98, D.purple.tint, C.purple, 'O(n) Backward Pass via Log-Sum-Exp Recomputation', 'Forward pass saves only 1 scalar per query:  ℓᵢ = mᵢ + log sᵢ ∈ ℝⁿ', 'Backward pass recomputes αᵢⱼ = exp(aᵢⱼ − ℓᵢ) on chip in SRAM — zero n × n buffer!', D.purple.dark, 14, 12);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §10.6 windowed-linear-attention.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 43. ch10-attn-win-two-routes
function winTwoRoutes() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 20, 236, 114, C.lblue, C.blue, 'Route 1: Sliding Window', 'Attend only to i − w < j ≤ i', 'O(n w d) FLOPs · O(w d) KV cache', D.blue.dark, 14.5, 12);
  o += card(284, 20, 236, 114, C.lgreen, C.green, 'Route 2: Linear Attention', 'Kernel K(q,k) = φ(q)ᵀ φ(k)', 'O(n d d_h) FLOPs · O(d d_h) state', D.green.dark, 14.5, 12);

  o += card(20, 154, 236, 94, '#FFFFFF', C.blue, 'Preserves Softmax Sharpness', 'Exact local retrieval within window w', 'Needs multi-hop or global layers', D.blue.dark, 13.5, 12);
  o += card(284, 154, 236, 94, '#FFFFFF', C.green, 'Infinite Horizon, Fixed State', 'S_t ∈ ℝ^{d_h×d_v} associative matrix', 'Lacks softmax temperature sharpening', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 44. ch10-attn-win-receptive-field
function winReceptiveField() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 12, 500, 72, C.lblue, C.blue, 'Multi-Hop Receptive Field Across L Stacked Window Layers:  1 + L(w − 1)', 'Mistral 7B (w = 4,096, L = 32 layers) · spans 1 + 32 × 4,095 = 131,041 tokens', null, D.blue.dark, 14, 12.5);

  o += card(20, 98, 236, 156, C.lgreen, C.green, '1 Global Layer (Full Causal)', 'Direct 1-hop score α(q_t, k_s)', 'Uniform weight at Δ=12 is 1/13', D.green.dark, 14, 12, { t: 'Direct content addressing', fs: 12.5, fw: 700, fill: D.green.dark });

  o += card(284, 98, 236, 156, C.lamber, C.amber, 'L=4 Window Layers (w = 4)', 'Requires ⌈Δ/(w−1)⌉ = 4 sequential hops', 'Boundary weight at Δ=12 is (1/4)⁴ = 1/256!', D.orange.dark, 14, 11.5, { t: 'Exponential boundary attenuation', fs: 12.5, fw: 700, fill: D.orange.dark });
  return svg(W, H, o);
}

// 45. ch10-attn-win-hybrid-schedule
function winHybridSchedule() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'Hybrid Interleaving: r = 5 Windowed Layers per 1 Global Layer (Gemma 3)', { fs: 14.5, fw: 700, fill: C.ink });

  const bx = 20, bw = 76, gap = 8, y = 44, bh = 74;
  for (let i = 0; i < 6; i++) {
    const isGlobal = i === 5;
    const x = bx + i * (bw + gap);
    o += card(x, y, bw, bh, isGlobal ? D.purple.tint : C.lblue, isGlobal ? C.purple : C.blue,
              isGlobal ? 'Layer 6' : `Layer ${i + 1}`,
              isGlobal ? 'GLOBAL' : 'Window w', null,
              isGlobal ? D.purple.dark : D.blue.dark, 13, 11.5);
  }

  o += card(20, 142, 500, 104, C.lgreen, C.green, 'KV-Cache Fraction:  (r · min(n, w) + n) / ((r + 1) n)  →  1 / (r + 1)', 'At w = 1,024, r = 5:  Cache drops to 27.1% at n=8K and 16.7% (1/6) asymptotically', 'Every 6th layer restores direct 1-hop global retrieval across the full context!', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 46. ch10-attn-win-associativity
function winAssociativity() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 20, 236, 118, D.red.tint, D.red.base, 'Standard Bracketing: O(n² d)', '( φ(Q) φ(K)ᵀ ) · V', 'Forms n × n score matrix first', D.red.dark, 14, 12.5);
  o += arrow(260, 79, 280, 79, C.green);
  o += card(284, 20, 236, 118, C.lgreen, C.green, 'Linear Bracketing: O(n d_h d_v)', 'φ(Q) · ( φ(K)ᵀ V )', 'Forms d_h × d_v matrix first!', D.green.dark, 14, 12.5);

  o += card(20, 158, 500, 92, C.lblue, C.blue, 'Matrix Multiplication Associativity Removes the n² Factor', 'Feature map φ(x) = elu(x) + 1 > 0 applied elementwise to Q and K', '2 n d_h d_v FLOPs to accumulate φ(K)ᵀV  +  2 n d_h d_v FLOPs to multiply by φ(Q)', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 47. ch10-attn-win-fast-weight-rnn
function winFastWeightRnn() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 24, 152, 98, C.lblue, C.blue, '1. Outer-Product Write', 'S_t = S_(t−1) + φ(k_t)v_tᵀ', 'z_t = z_(t−1) + φ(k_t)', D.blue.dark, 13.5, 11.5);
  o += arrow(176, 73, 194, 73, C.purple);
  o += card(198, 24, 144, 98, D.purple.tint, C.purple, '2. Fixed State Matrix', 'S_t ∈ ℝ^{d_h×d_v}', '16.25 KiB at d_h=64', D.purple.dark, 13.5, 12);
  o += arrow(346, 73, 364, 73, C.green);
  o += card(368, 24, 152, 98, C.lgreen, C.green, '3. Memory Readout', 'o_tᵀ = φ(q_t)ᵀ S_t /', '(φ(q_t)ᵀ z_t + ε)', D.green.dark, 13.5, 11.5);

  o += card(20, 146, 500, 104, C.lamber, C.amber, 'Causal Linear Attention Is an RNN with Matrix-Valued Hidden State', 'State size (d_h d_v + d_h = 4,160 floats) is constant for any sequence length n!', 'A softmax KV cache (128 floats/token) exceeds the entire linear state after 33 tokens.', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 48. ch10-attn-win-chunked-gemm
function winChunkedGemm() {
  const W = 540, H = 272;
  let o = '';
  o += tx(270, 20, 'Chunked Linear Attention (Chunk Size c = 64): Tensor Core Training', { fs: 14.5, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 104, C.lblue, C.blue, 'Inter-Chunk State Term', 'Q_p @ S_(chunk−1)', 'Reads pre-chunk d_h × d_v summary', D.blue.dark, 14, 12);
  o += card(284, 42, 236, 104, C.lamber, C.amber, 'Intra-Chunk Causal Term', 'tril(Q_p @ K_pᵀ) @ V_c', 'c × c local GEMM within current chunk', D.orange.dark, 14, 12);

  o += arrow(138, 150, 220, 178, C.blue);
  o += arrow(402, 150, 320, 178, C.amber);

  o += card(40, 182, 460, 72, C.lgreen, C.green, 'Sum & Normalize:  (Q_p S + A V_c) / (Q_p z + A 1 + ε)', 'Runs entirely on Tensor Core GEMMs while storing only n/c chunk states!', null, D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 49. ch10-attn-win-three-modes
function winThreeModes() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'Three Mathematically Equivalent Execution Modes of Linear Attention', { fs: 14.5, fw: 700, fill: C.ink });
  const cw = 154, y = 42, ch = 116;
  o += card(20, y, cw, ch, D.red.tint, D.red.base, '1. Parallel cumsum', 'O(n d_h d_v) memory', '256 MiB/head at 16K', D.red.dark, 13.5, 11.5);
  o += card(193, y, cw, ch, C.lblue, C.blue, '2. Recurrent Loop', 'O(d_h d_v) = 16 KiB', 'n sequential steps', D.blue.dark, 13.5, 11.5);
  o += card(366, y, cw, ch, C.lgreen, C.green, '3. Chunked (c = 64)', 'O((n/c)d_h d_v + nc)', 'Tensor Core GEMMs', D.green.dark, 13.5, 11.5);

  o += card(20, 176, 500, 80, D.purple.tint, C.purple, 'Best Practice: Train & Prefill with Chunked Form · Decode Token-by-Token with Recurrent Form', 'All three match to ~10⁻⁷ in float32!', null, D.purple.dark, 13, 12);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §10.7 what-attention-computes.md (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 50. ch10-attn-circ-shortcut-illusion
function circShortcutIllusion() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'Why Behavioral Accuracy & Heatmaps Alone Can Mislead', { fs: 15, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 96, C.lgreen, C.green, 'Trained on Fixed Period L = 32', '1-Block model gets 100% accuracy!', 'Attends sharply to offset j − i = −31', D.green.dark, 14, 12);
  o += card(284, 42, 236, 96, D.red.tint, D.red.base, 'Probed on Period L = 24', 'Accuracy collapses to 1.6% (1/64)!', 'Still stares blindly at offset −31!', D.red.dark, 14, 12);

  o += card(20, 154, 500, 104, C.lamber, C.amber, 'Mechanistic Lesson: Position Shortcut ≠ Content-Based Induction', 'A 1-block model with bias queries/keys under RoPE can hard-code a fixed offset −31,', 'but content-based in-context lookup [A][B]…[A] → [B] requires at least 2 layers!', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 51. ch10-attn-circ-additive-heads
function circAdditiveHeads() {
  const W = 540, H = 266;
  let o = '';
  o += card(20, 22, 226, 112, C.lblue, C.blue, 'Concatenate Form', '[u₁; u₂; …; u_H] ∈ ℝ^{H·d_h}', 'multiplied by W_o ∈ ℝ^{d×H·d_h}', D.blue.dark, 14, 12);
  o += tx(270, 78, '=', { fs: 26, fw: 700, fill: D.purple.dark });
  o += card(294, 22, 226, 112, C.lgreen, C.green, 'Independent Sum Form', '∑_{m=1}^{H}  W_o^m u_m', '= ∑_{m} ∑_{j≤i} α_ij^m (W_o^m W_v^m) h_j', D.green.dark, 14, 11.5);

  o += card(20, 156, 500, 90, D.purple.tint, C.purple, 'Every Head Acts Independently & Additively on the Residual Stream', 'Partitioning W_o = [W_o¹ | … | W_o^H] into H column slices of shape (d × d_h)', 'shows that each head m contributes ∑_{j≤i} α_ij^m W_OV^m h_j directly to the stream!', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 52. ch10-attn-circ-qkov-factorization
function circQkovFactorization() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 20, 236, 116, C.lblue, C.blue, 'QK Circuit: WHERE to Attend', 'W_QK = W_qᵀ W_k ∈ ℝ^{d×d}', 's_ij = h_iᵀ W_QK h_j / √d_h', D.blue.dark, 14, 12);
  o += card(284, 20, 236, 116, C.lgreen, C.green, 'OV Circuit: WHAT to Write', 'W_OV = W_o W_v ∈ ℝ^{d×d}', 'writes ∑_{j} α_ij (W_OV h_j) to stream i', D.green.dark, 14, 12);

  o += card(20, 156, 500, 94, C.lamber, C.amber, 'Rank-d_h Bottleneck & Gauge Invariance', 'Both W_QK and W_OV factor through width d_h = d/H, so rank(W_QK), rank(W_OV) ≤ d_h.', 'Replacing (W_o, W_v) ↦ (W_o G⁻¹, G W_v) leaves W_OV identical for any invertible G!', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 53. ch10-attn-circ-depth-hierarchy
function circDepthHierarchy() {
  const W = 540, H = 272;
  let o = '';
  o += tx(270, 20, 'Expressivity Hierarchy of Attention-Only Networks by Depth N', { fs: 15, fw: 700, fill: C.ink });
  const cw = 154, y = 42, ch = 118;
  o += card(20, y, cw, ch, N.panel, C.gray, 'N = 0 Blocks', 'z_i = E e_(x_i)', 'Symmetric Bigrams', C.ink, 14, 12);
  o += card(193, y, cw, ch, C.lblue, C.blue, 'N = 1 Block', '+ ∑ α_ij E W_OV e_(x_j)', 'Skip-Trigrams & Copy', D.blue.dark, 14, 11.5);
  o += card(366, y, cw, ch, C.lgreen, C.green, 'N = 2 Blocks', 'K-Composition', 'Induction [A][B]…[A]→[B]', D.green.dark, 14, 11.5);

  o += arrow(178, y + ch / 2, 189, y + ch / 2, C.gray);
  o += arrow(351, y + ch / 2, 362, y + ch / 2, C.blue);

  o += card(20, 178, 500, 80, D.purple.tint, C.purple, 'Why N = 1 Cannot Perform Content-Based Induction', 'At N = 1, key k_j at [B]’s position depends only on e_{[B]} and has no record that x_(j−1) = [A]!', null, D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 54. ch10-attn-circ-copy-matrix-probe
function circCopyMatrixProbe() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 24, 142, 92, C.lblue, C.blue, 'Mean Stream h̄_a', 'entering Block 2', 'at token a positions', D.blue.dark, 14, 12);
  o += arrow(166, 70, 192, 70, C.purple);
  o += card(196, 24, 148, 92, D.purple.tint, C.purple, 'Summed OV Map', 'W_OV = ∑_{m} W_o^m W_v^m', 'transforms h̄_a', D.purple.dark, 13.5, 11.5);
  o += arrow(348, 70, 374, 70, C.green);
  o += card(378, 24, 142, 92, C.lgreen, C.green, 'Unembed E', 'C_ab = e_bᵀ W_OV h̄_a', '64 × 64 Copy Matrix', D.green.dark, 14, 11.5);

  o += card(20, 142, 500, 104, C.lamber, C.amber, 'Diagonal Dominance Test:  arg max_{b} C_ab = a  (98–100% vs. 1.6% chance)', 'When Block 2 attends to token a at position j, its OV circuit boosts logit b = a!', 'Caution: The 1-block model ALSO learns a diagonal C, but lacks the QK circuit to find j.', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 55. ch10-attn-circ-icl-gd
function circIclGd() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'One In-Context Gradient Descent Step as One Linear Attention Head', { fs: 14.5, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 108, C.lblue, C.blue, '1 GD Step on ½∑(yᵢ − wᵀxᵢ)²', 'From w₀ = 0:  w₁ = η ∑ᵢ yᵢ xᵢ', 'Prediction:  ŷ_q = w₁ᵀ x_q = ∑ᵢ (x_qᵀxᵢ)(η yᵢ)', D.blue.dark, 13.5, 11.5);
  o += card(284, 42, 236, 108, C.lgreen, C.green, 'Linear Attention Head', 'Tokens zᵢ = [xᵢ; yᵢ],  Query z_q = [x_q; 0]', 'q = x_q,  kᵢ = xᵢ,  vᵢ = η yᵢ  ⇒  ∑ (qᵀkᵢ)vᵢ', D.green.dark, 13.5, 11.5);

  o += card(20, 170, 500, 78, D.purple.tint, C.purple, 'Fast In-Context Learning Without Updating Model Weights', 'A single forward pass computes the exact gradient-descent regression prediction on the fly!', null, D.purple.dark, 14, 12.5);
  return svg(W, H, o);
}

// 56. ch10-attn-qkv-summary
function qkvSummary() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 18, 236, 108, C.lblue, C.blue, '1. Soft Convex Pooling', 'Attention(q, D) = ∑ αᵢ vᵢ', 'α ∈ Δ^{m−1}: αᵢ ≥ 0, ∑ αᵢ = 1', D.blue.dark, 14, 12);
  o += card(284, 18, 236, 108, C.lamber, C.amber, '2. Temperature Bandwidth σ', 'a(q, kᵢ) = −‖q − kᵢ‖² / 2σ²', 'σ → 0: 1-NN · σ → ∞: uniform mean', D.orange.dark, 14, 12);
  o += card(20, 146, 500, 106, C.lgreen, C.green, '3. Matrix Form, Effective Keys n_eff, and Max-Subtract Softmax', 'Batched evaluation: softmax(A) V  (n×m weights times m×d_v values = n×d_v outputs);', 'n_eff = 1 / ∑ αᵢ² ∈ [1, m]; subtract m(q) = maxⱼ a(q, kⱼ) so leading term is exp(0) = 1!', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 57. ch10-attn-score-summary
function scoreSummary() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 18, 236, 108, C.lblue, C.blue, '1. Scaled Dot-Product', 'a(q, k) = (qᵀ k) / √d', 'Restores Var(a) = 1 across width d', D.blue.dark, 14, 12);
  o += card(284, 18, 236, 108, C.lamber, C.amber, '2. Additive (Bahdanau)', 'a(q, k) = w_vᵀ tanh(W_q q + W_k k)', 'Handles d_q ≠ d_k but O(n m h) memory', D.orange.dark, 14, 12);
  o += card(20, 146, 500, 106, C.lgreen, C.green, '3. Safe Boolean Masking & Softmax Jacobian Conditioning', 'Fill forbidden positions with finfo(dtype).min BEFORE softmax (and guard all-masked rows with any(-1));', '1/√d scaling prevents one-hot saturation where Jacobian J(α) = diag(α) − ααᵀ → 0!', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 58. ch10-attn-scale-solved-vs-open
function scaleSolvedVsOpen() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 20, 236, 114, C.lgreen, C.green, 'What Fused Kernels Solve', 'HBM scratch memory: O(n²) → O(n)', 'HBM traffic: Θ(Hn²) → Θ(n²d_h²/M)', D.green.dark, 14, 12);
  o += card(284, 20, 236, 114, D.red.tint, D.red.base, 'What Remains O(n²)', 'Arithmetic FLOPs: still Θ(n² d)', 'Generation KV cache: still O(n d)', D.red.dark, 14, 12);
  o += card(20, 154, 500, 94, C.lblue, C.blue, 'Why §10.6 Introduces Sliding Windows and Linear Attention', 'FlashAttention is an exact IO-aware reorganization of dense softmax attention;', 'breaking the Θ(n² d) arithmetic floor requires changing the attention operator itself!', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 59. ch10-attn-scale-summary
function scaleSummary() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 18, 236, 108, D.red.tint, D.red.base, '1. The O(n²) HBM Wall', '8 n² bytes/head scratch (128 GiB @ 128K)', 'Block FLOP crossover (with FFN): n ≈ 6d', D.red.dark, 13.5, 12);
  o += card(284, 18, 236, 108, C.lblue, C.blue, '2. Online Softmax Recurrence', 'Running (m, s, o) in SRAM', 'Rescales by exp(m − m′) ≤ 1', D.blue.dark, 13.5, 12);
  o += card(20, 146, 500, 106, C.lgreen, C.green, '3. FlashAttention Tiled Fusion & Causal Block Skipping', 'Tiles Q, K, V into on-chip SRAM so S and P never touch HBM (O(nd) memory);', 'Skips strictly upper-triangular causal tiles (j > i) to cut arithmetic FLOPs in half!', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 60. ch10-attn-win-tile-wall
function winTileWall() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'Comparing 128×128 Tile Visits & Cache at n = 32,768 (T = 256 Tile Rows)', { fs: 14, fw: 700, fill: C.ink });
  const cw = 154, y = 42, ch = 128;
  o += card(20, y, cw, ch, D.red.tint, D.red.base, 'Dense Causal', '32,896 tiles visited', '32,768 KV tokens cached', D.red.dark, 14, 12);
  o += card(193, y, cw, ch, C.lblue, C.blue, 'Window (w = 128)', '511 tiles (64× fewer!)', '128 KV tokens cached', D.blue.dark, 14, 12);
  o += card(366, y, cw, ch, C.lgreen, C.green, 'Linear Attention', 'No n×n tiles!', 'Fixed d_h × d_v matrix', D.green.dark, 14, 12);
  o += card(20, 188, 500, 62, D.purple.tint, C.purple, 'Two Routes Below Θ(n² d): Score Fewer Pairs (Windows) or Factorize the Kernel (Linear)', null, null, D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 61. ch10-attn-win-decision-guide
function winDecisionGuide() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 18, 236, 108, C.lblue, C.blue, 'Short/Medium (n ≤ 32K)', 'Exact Fused FlashAttention', 'Full 1-hop recall · Θ(n²d) time', D.blue.dark, 13.5, 12);
  o += card(284, 18, 236, 108, C.lgreen, C.green, 'Long Context (n = 128K–1M)', 'Interleaved Window + Global (5:1)', '17% KV cache · 1-hop via global layers', D.green.dark, 13.5, 12);
  o += card(20, 146, 500, 106, C.lamber, C.amber, 'Ultra-Long / Constant-Memory Streaming: Gated Linear + Hybrid Softmax', 'Plain linear attention (elu+1) lacks temperature sharpening and holds ≤ d_h orthogonal keys;', 'Production hybrids (MiniMax-01, Qwen3-Next, Kimi Linear) interleave gated linear layers with softmax layers', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

export const diagrams = {
  'ch10-attn-rnn-vs-lookup': rnnVsLookup,
  'ch10-attn-three-challenges': threeChallenges,
  'ch10-attn-roadmap': roadmap,
  'ch10-attn-qkv-state-vs-db': qkvStateVsDb,
  'ch10-attn-qkv-hard-vs-soft': qkvHardVsSoft,
  'ch10-attn-qkv-scalar-trace': qkvScalarTrace,
  'ch10-attn-qkv-simplex-spectrum': qkvSimplexSpectrum,
  'ch10-attn-qkv-shift-invariance': qkvShiftInvariance,
  'ch10-attn-qkv-vector-trace': qkvVectorTrace,
  'ch10-attn-qkv-memory-scaling': qkvMemoryScaling,
  'ch10-attn-qkv-temperature-dial': qkvTemperatureDial,
  'ch10-attn-qkv-underflow-pitfall': qkvUnderflowPitfall,
  'ch10-attn-qkv-summary': qkvSummary,
  'ch10-attn-score-two-reqs': scoreTwoReqs,
  'ch10-attn-score-gaussian-to-dot': scoreGaussianToDot,
  'ch10-attn-score-additive-vs-dot': scoreAdditiveVsDot,
  'ch10-attn-score-variance-scaling': scoreVarianceScaling,
  'ch10-attn-score-worked-trace': scoreWorkedTrace,
  'ch10-attn-score-jacobian-saturation': scoreJacobianSaturation,
  'ch10-attn-score-mask-fill-hazards': scoreMaskFillHazards,
  'ch10-attn-score-summary': scoreSummary,
  'ch10-attn-mha-multi-arg': mhaMultiArg,
  'ch10-attn-mha-qkv-projections': mhaQkvProjections,
  'ch10-attn-mha-asymmetry': mhaAsymmetry,
  'ch10-attn-mha-copy-both-bound': mhaCopyBothBound,
  'ch10-attn-mha-flop-invariance': mhaFlopInvariance,
  'ch10-attn-mha-reshape-bug': mhaReshapeBug,
  'ch10-attn-mha-self-vs-cross': mhaSelfVsCross,
  'ch10-attn-pos-order-blindness': posOrderBlindness,
  'ch10-attn-pos-equivariance': posEquivariance,
  'ch10-attn-pos-learned-table': posLearnedTable,
  'ch10-attn-pos-planar-rotation': posPlanarRotation,
  'ch10-attn-pos-four-terms': posFourTerms,
  'ch10-attn-pos-alibi-slopes': posAlibiSlopes,
  'ch10-attn-pos-five-schemes': posFiveSchemes,
  'ch10-attn-pos-fast-slow-yarn': posFastSlowYarn,
  'ch10-attn-scale-memory-wall': scaleMemoryWall,
  'ch10-attn-scale-flop-crossover': scaleFlopCrossover,
  'ch10-attn-scale-two-pass-vs-online': scaleTwoPassVsOnline,
  'ch10-attn-scale-online-recurrence': scaleOnlineRecurrence,
  'ch10-attn-scale-online-trace': scaleOnlineTrace,
  'ch10-attn-scale-sram-hierarchy': scaleSramHierarchy,
  'ch10-attn-scale-hbm-passes': scaleHbmPasses,
  'ch10-attn-scale-flash-tiling': scaleFlashTiling,
  'ch10-attn-scale-solved-vs-open': scaleSolvedVsOpen,
  'ch10-attn-scale-summary': scaleSummary,
  'ch10-attn-win-two-routes': winTwoRoutes,
  'ch10-attn-win-receptive-field': winReceptiveField,
  'ch10-attn-win-hybrid-schedule': winHybridSchedule,
  'ch10-attn-win-associativity': winAssociativity,
  'ch10-attn-win-fast-weight-rnn': winFastWeightRnn,
  'ch10-attn-win-chunked-gemm': winChunkedGemm,
  'ch10-attn-win-three-modes': winThreeModes,
  'ch10-attn-win-tile-wall': winTileWall,
  'ch10-attn-win-decision-guide': winDecisionGuide,
  'ch10-attn-circ-shortcut-illusion': circShortcutIllusion,
  'ch10-attn-circ-additive-heads': circAdditiveHeads,
  'ch10-attn-circ-qkov-factorization': circQkovFactorization,
  'ch10-attn-circ-depth-hierarchy': circDepthHierarchy,
  'ch10-attn-circ-copy-matrix-probe': circCopyMatrixProbe,
  'ch10-attn-circ-icl-gd': circIclGd,
};
