// Slide diagrams for Chapter 11 (Transformers): chapter_transformers/*.md
import { C, TOKENS, grid, tx as rawTx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const hline = (x1, x2, y) => line(x1, y, x2, y, N.hairline, 1.5, true);

// Text fitting. Realistic advance widths (em per character): Source Sans 3
// ≈ 0.46 (0.49 semibold/bold), outlined mono ≈ 0.61. A line that would need less than
// FLOOR px is wrapped into two lines instead of being shrunk further.
const FLOOR = 11;
const advance = (o) => (o.mono ? 0.61 : ((o.fw || 500) >= 600 ? 0.49 : 0.46));
const fitOne = (t, maxFs, availW, f) => Math.min(maxFs, +(availW / (String(t).length * f)).toFixed(1));

// Split at the space nearest the middle, outside brackets, preferring a break after · ; , :
function splitTwo(t) {
  const s = String(t);
  const mid = s.length / 2;
  let best = -1, bestD = Infinity;
  let depth = 0;
  for (let i = 1; i < s.length - 1; i++) {
    if ('([{'.includes(s[i - 1])) depth++;
    if (')]}'.includes(s[i - 1])) depth = Math.max(0, depth - 1);
    if (s[i] !== ' ' || s[i - 1] === ' ' || '≈=×→+−<>≤≥'.includes(s[i - 1])) continue;
    // never break inside brackets; prefer a break after a separator
    const d = Math.abs(i - mid) - (/[·;,:]/.test(s[i - 1]) ? 6 : 0) + (depth > 0 ? 1000 : 0);
    if (d < bestD) { bestD = d; best = i; }
  }
  if (best < 0) return [s];
  return [s.slice(0, best).trim(), s.slice(best + 1).trim().replace(/^·\s*/, '')];
}

// -> [lines, fs]: one line at the largest size ≤ maxFs that fits, or two lines.
function fitLines(t, maxFs, availW, o) {
  const f = advance(o);
  const fs = fitOne(t, maxFs, availW, f);
  if (fs >= FLOOR) return [[String(t)], fs];
  const parts = splitTwo(t);
  if (parts.length < 2) return [parts, FLOOR];
  return [parts, Math.max(FLOOR, Math.min(...parts.map((q) => fitOne(q, maxFs, availW, f))))];
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
  const { availW: _ignore, ...rest } = opts;
  const [lines, fs] = fitLines(text, maxFs, availW - 16, opts);
  if (lines.length === 1) return rawTx(x, y, lines[0], { ...rest, fs });
  const lh = fs * 1.22;
  return rawTx(x, y - lh / 2, lines[0], { ...rest, fs }) + rawTx(x, y + lh / 2, lines[1], { ...rest, fs });
}

// Titled card: a bold title, then up to two lines (l1 mono in narrow cards,
// sans in wide ones; l2 muted sans) plus optional `extra` lines
// ({t, mono, fw, fill, fs}). Every line is fitted to the card width (wrapping
// to two lines rather than shrinking below FLOOR) and the whole stack is
// centred vertically in the card.
function card(x, y, w, h, fill, stroke, title, l1, l2, tColor = C.ink, fsTitle = 14.5, fsSub = 12.5, extra = []) {
  let o = box(x, y, w, h, fill, stroke, 8, 2);
  const specs = [{ t: title, fs: fsTitle, fw: 700, fill: tColor }];
  if (l1) specs.push({ t: l1, mono: w < 400, fs: fsSub, fw: 600, fill: C.ink });
  if (l2) specs.push({ t: l2, fs: fsSub, fw: 500, fill: C.muted });
  extra.forEach((e) => specs.push({ fs: fsSub, fw: 600, fill: C.ink, ...e }));
  const rows = [];
  specs.forEach((sp, i) => {
    const [parts, fs] = fitLines(sp.t, sp.fs, w - 22, sp);
    parts.forEach((q, j) => rows.push({ q, fs, sp, gap: j > 0 ? 1 : (i === 0 ? 0 : (i === 1 ? 7 : 5)) }));
  });
  const total = rows.reduce((a, r) => a + r.gap + r.fs * 1.22, 0);
  let cy = y + (h - total) / 2;
  const cx = x + w / 2;
  rows.forEach((r) => {
    cy += r.gap;
    o += rawTx(cx, cy + r.fs * 0.61, r.q, { mono: r.sp.mono, fs: r.fs, fw: r.sp.fw, fill: r.sp.fill });
    cy += r.fs * 1.22;
  });
  return o;
}

// ═══════════════════════════════════════════════════════════════════════════
// §11.0 index.md (4 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 1. ch11-tf-why-block
function whyBlock() {
  const W = 540, H = 276;
  let o = '';
  o += card(20, 20, 236, 110, D.red.tint, D.red.base, 'Attention Alone Fails', 'Linear projections around ∑ αᵢ vᵢ', 'No per-token nonlinear features', D.red.dark, 14.5, 12);
  o += card(284, 20, 236, 110, C.lgreen, C.green, 'The Transformer Block', 'Attention (4d²) + SwiGLU FFN (8d²)', 'Pre-Norm Residual Stream (12d²)', D.green.dark, 14.5, 12);

  o += card(20, 150, 500, 104, C.lblue, C.blue, 'GPT-2 Small (124.4M Parameters, L = 12, d = 768) Census', '12 Blocks × 12 d² = 85.0M params (28.3M Attention + 56.6M FFN)', 'Tied Token Embeddings (50,257 × 768 = 38.6M) + 1,024 Learned Positions (0.8M)', D.blue.dark, 14, 12);
  return svg(W, H, o);
}

// 2. ch11-tf-four-pillars
function fourPillars() {
  const W = 540, H = 276;
  let o = '';
  const pillars = [
    { x: 20,  y: 22,  fill: C.lblue, st: C.blue, dk: D.blue.dark, t: '1. Build & Train (§§11.1–11.2)', s1: 'Pre-Norm · RMSNorm · QK-Norm', s2: 'SwiGLU · 1/√(2L) Init · GPT' },
    { x: 284, y: 22,  fill: C.lamber, st: C.amber, dk: D.orange.dark, t: '2. Serve & Compress (§11.3)', s1: 'KV Cache · Hardware Roofline', s2: 'GQA Uptraining · MLA · Sinks' },
    { x: 20,  y: 150, fill: D.purple.tint, st: C.purple, dk: D.purple.dark, t: '3. Rewire Modalities (§§11.4–11.5)', s1: 'Encoders (MLM) · Cross-Attn', s2: 'Perceiver Latents · Vision (ViT)' },
    { x: 284, y: 150, fill: C.lgreen, st: C.green, dk: D.green.dark, t: '4. Scale Capacity (§§11.6–11.7)', s1: 'N ≈ 12Ld² · C ≈ 6ND · Chinchilla', s2: 'Sparse Mixture of Experts (MoE)' },
  ];
  pillars.forEach((p) => {
    o += card(p.x, p.y, 236, 102, p.fill, p.st, p.t, p.s1, p.s2, p.dk, 14, 12);
  });
  o += arrow(260, 73, 280, 73, C.blue);
  o += arrow(402, 128, 138, 146, C.amber);
  o += arrow(260, 201, 280, 201, C.purple);
  return svg(W, H, o);
}

// 3. ch11-tf-benchmarks
function benchmarks() {
  const W = 540, H = 278;
  let o = '';
  o += tx(270, 20, 'Executable Single-GPU Benchmarks Across Chapter 11', { fs: 15, fw: 700, fill: C.ink });
  const items = [
    { x: 18,  y: 42,  t: '§11.1 Signal Audit', s1: '32-layer token spread & grad', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 192, y: 42,  t: '§11.2 Byte GPT & GPT-2', s1: '4.8M vs RNN + 124M weights', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 366, y: 42,  t: '§11.3 Roofline & KV', s1: '20× speedup · GQA · SVD · Sinks', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 18,  y: 156, t: '§11.4 MLM & Seq2Seq', s1: 'U-curve · reversal · Perceiver', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
    { x: 192, y: 156, t: '§11.5 ViT vs ResNet', s1: '6.5M on 96×96 Fashion-MNIST', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 366, y: 156, t: '§§11.6–11.7 IsoFLOP', s1: '15-model grid · 8-expert MoE', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  items.forEach((it) => {
    o += card(it.x, it.y, 156, 94, it.fill, it.st, it.t, it.s1, null, it.dk, 13.5, 11.5);
  });
  return svg(W, H, o);
}

// 4. ch11-tf-modern-recipe
function modernRecipe() {
  const W = 540, H = 276;
  let o = '';
  const base = ['Post-Norm LayerNorm + Bias', 'Learned / Sinusoidal Pos', '2-Matrix GELU (4d)', 'Full MHA (H_kv = H)', 'Dense FFN (N_act = N)'];
  o += card(20, 20, 224, 236, N.panel, C.gray, '2017 / GPT-2 Baseline', null, null, C.ink, 15, 12.5,
    base.map((t) => ({ t, mono: true, fw: 600, fill: C.ink })));

  o += arrow(248, 138, 290, 138, C.green);

  const modern = ['Pre-Norm RMSNorm + QK-Norm', 'Rotary Embeddings (RoPE)', '3-Matrix SwiGLU (8d/3)', 'GQA / MLA / Sinks', 'Fine-Grained + Shared MoE'];
  o += card(296, 20, 224, 236, C.lgreen, C.green, 'Modern Frontier Recipe', null, null, D.green.dark, 15, 12.5,
    modern.map((t) => ({ t, mono: true, fw: 700, fill: D.green.dark })));
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §11.1 transformer-block.md (8 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 5. ch11-tf-blk-failure-modes
function blkFailureModes() {
  const W = 540, H = 270;
  let o = '';
  o += tx(270, 20, 'Three Pathologies When Stacking 32 Unstabilized Attention Layers', { fs: 14.5, fw: 700, fill: C.ink });
  const cw = 154, y = 42, ch = 128;
  o += card(20, y, cw, ch, D.red.tint, D.red.base, '1. Linear Collapse', 'No per-token MLP', 'W_v W_o products collapse', D.red.dark, 13.5, 11.5);
  o += card(193, y, cw, ch, D.red.tint, D.red.base, '2. Token Collapse', 'Post-Norm + Xavier', 'Spread → 10⁻⁴ at L=32', D.red.dark, 13.5, 11.5);
  o += card(366, y, cw, ch, D.red.tint, D.red.base, '3. Starved Grads', '2L Norm divisions', '‖∇_(W_q¹) L‖ → 10⁻⁹', D.red.dark, 13.5, 11.5);

  o += card(20, 188, 500, 62, C.lgreen, C.green, 'Remedy: Pre-Norm Residual Stream + 1/√(2L) Init + QK-Norm + SwiGLU FFN', 'Preserves O(1) token diversity and uniform gradient norms across arbitrary depth!', null, D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 6. ch11-tf-blk-post-vs-pre-eq
function blkPostVsPreEq() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 20, 236, 116, D.red.tint, D.red.base, 'Post-Norm (2017 / BERT)', 'H = Norm₁(X + Attn(X))', 'Y = Norm₂(H + FFN(H))', D.red.dark, 14.5, 12.5);
  o += card(284, 20, 236, 116, C.lgreen, C.green, 'Pre-Norm (Modern LLMs)', 'H = X + Attn(Norm₁(X))', 'Y = H + FFN(Norm₂(H))', D.green.dark, 14.5, 12.5);

  o += card(20, 156, 236, 92, '#FFFFFF', D.red.base, 'Norm Sits on Main Stream', 'Divides residual sum at every step', 'Attenuates early gradients by r^(2L)', D.red.dark, 13.5, 11.5);
  o += card(284, 156, 236, 92, C.lblue, C.blue, 'Uninterrupted Identity Highway', 'Norm sits only on branch inputs', '∂X^{(L)}/∂X^{(ℓ)} = I + … (flat grads!)', D.blue.dark, 13.5, 11.5);
  return svg(W, H, o);
}

// 7. ch11-tf-blk-token-collapse
function blkTokenCollapse() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 12, 500, 68, C.lblue, C.blue, 'Decompose Token Vector:  xᵢ = μ + δᵢ   (Mean μ = (1/n)∑ xⱼ,  Deviation δᵢ)', 'At init, uniform attention averages across tokens:  ∑ⱼ αᵢⱼ xⱼ ≈ μ  (deviations δᵢ cancel!)', null, D.blue.dark, 13.5, 12);

  o += card(20, 96, 236, 152, C.lamber, C.amber, '1. Shared Mean μ Amplifies', 'Attention adds W_o W_v μ to every i', 'Common energy grows by (1 + g⁴)×', D.orange.dark, 14, 12,
    [{ t: 'LayerNorm is across d, not across n!', fw: 700, fill: D.orange.dark }]);

  o += card(284, 96, 236, 152, D.red.tint, D.red.base, '2. Post-Norm Shrinks δᵢ', 'Divides (xᵢ + uᵢ) by √(1 + σ²)', 'Spread s_k ≈ (1 + (1+g⁴)^k / 63)^(-½)', D.red.dark, 14, 12,
    [{ t: 'At k=32 (g=1): spread → 1.2 × 10⁻⁴', mono: true, fw: 700, fill: D.red.dark }]);
  return svg(W, H, o);
}

// 8. ch11-tf-blk-width-pitfall
function blkWidthPitfall() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'Why Fixed Init Std σ = 0.02 Fails at Frontier Widths (Gain g² = d σ²)', { fs: 14.5, fw: 700, fill: C.ink });
  const cw = 154, y = 42, ch = 124;
  o += card(20, y, cw, ch, C.lgreen, C.green, 'd = 256 (Small)', 'g² = 256(0.02²) = 0.10', '1 + g⁴ = 1.01 (stable)', D.green.dark, 13.5, 11.5);
  o += card(193, y, cw, ch, C.lamber, C.amber, 'd = 1,024 (Medium)', 'g² = 1024(0.02²) = 0.41', '1 + g⁴ = 1.17 (decays)', D.orange.dark, 13.5, 11.5);
  o += card(366, y, cw, ch, D.red.tint, D.red.base, 'd = 4,096 (Llama 8B)', 'g² = 4096(0.02²) = 1.64', '1 + g⁴ = 3.68 (collapses!)', D.red.dark, 13.5, 11.5);

  o += card(20, 184, 500, 66, D.purple.tint, C.purple, 'Lesson: Fixed Numeric σ = 0.02 Is Only Small When d ≪ 1/σ² = 2,500', 'Pre-norm avoids post-norm division altogether and pairs with 1/√(2L) depth scaling.', null, D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 9. ch11-tf-blk-unrolled-highway
function blkUnrolledHighway() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 20, 500, 62, C.lblue, C.blue, 'Unrolled Pre-Norm Residual Stream Across L Blocks (2L Sublayers)', 'X^{(L)} = X^{(0)} + ∑_{ℓ=1}^{L} [ Attn_{ℓ}(Norm(X^{(ℓ−1)})) + FFN_{ℓ}(Norm(H^{(ℓ)})) ]', null, D.blue.dark, 14, 12.5);

  o += card(20, 102, 236, 146, D.red.tint, D.red.base, 'Without Depth Scaling', '2L branches each add variance σ²', 'Stream RMS grows as √(1 + 2L σ²)', D.red.dark, 14, 12,
    [{ t: 'At L = 32: stream RMS ≈ 5.0', mono: true, fw: 700, fill: D.red.dark }]);

  o += card(284, 102, 236, 146, C.lgreen, C.green, 'With 1/√(2L) Scaling (GPT-2)', 'Scale W_o and W_2 init std by 1/√(2L)', 'Total added var: ∑_{m=1}^{2L} σ²/(2L) = σ²', D.green.dark, 14, 12,
    [{ t: 'Stream RMS stays O(1) at any L!', mono: true, fw: 700, fill: D.green.dark }]);
  return svg(W, H, o);
}

// 10. ch11-tf-blk-qknorm-bound
function blkQknormBound() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 20, 236, 114, D.red.tint, D.red.base, 'Without QK-Norm', 'Q, K scale with ‖W_q‖, ‖W_k‖ = λ', 'Logits qᵀk/√d_h blow up as λ²!', D.red.dark, 14, 12);
  o += card(284, 20, 236, 114, C.lgreen, C.green, 'With Per-Head QK-Norm', 'q̂ = RMSNorm(q),  k̂ = RMSNorm(k)', '‖q̂‖₂ = ‖k̂‖₂ = √d_h (at γ = 1)', D.green.dark, 14, 12);

  o += card(20, 154, 500, 94, C.lblue, C.blue, 'Hard Cauchy–Schwarz Ceiling on Attention Logits', '|q̂ᵢᵀ k̂ⱼ| / √d_h  ≤  (‖q̂ᵢ‖₂ · ‖k̂ⱼ‖₂) / √d_h  =  (√d_h · √d_h) / √d_h  =  √d_h', 'At λ = 16: unnormalized entropy collapses to ≈ 0.02 nats; QK-Norm holds entropy at ≈ 3.7 nats!', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 11. ch11-tf-blk-ffn-kv-memory
function blkFfnKvMemory() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'The Position-Wise FFN as a Content-Addressed Key–Value Memory (Geva et al.)', { fs: 14, fw: 700, fill: C.ink });

  o += card(20, 44, 144, 96, C.lblue, C.blue, '1. Detector Keys', 'Rows w_(1,j) of W₁', '4d pattern detectors', D.blue.dark, 13.5, 12);
  o += arrow(168, 92, 192, 92, C.amber);
  o += card(196, 44, 148, 96, C.lamber, C.amber, '2. Sparse Gate', 'aⱼ = GELU(xᵀ w_(1,j))', 'fires on matching x', D.orange.dark, 13.5, 12);
  o += arrow(348, 92, 372, 92, C.green);
  o += card(376, 44, 144, 96, C.lgreen, C.green, '3. Feature Values', 'Cols w_(2,j) of W₂', 'writes ∑ aⱼ w_(2,j)', D.green.dark, 13.5, 12);

  o += card(20, 162, 500, 86, D.purple.tint, C.purple, 'Two-Thirds of Every Dense Block Lives in the FFN (8d² of 12d²)', 'In GPT-2 Small (d = 768), each block’s FFN holds 4d = 3,072 memory slots (4.72M weights)', 'acting independently on every token vector xᵢ ∈ ℝᵈ to synthesize factual & syntactic features.', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 12. ch11-tf-blk-gelu-vs-swiglu
function blkGeluVsSwiglu() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 20, 236, 120, C.lblue, C.blue, '2-Matrix GELU MLP', 'GELU(x W₁) W₂', 'Width d_ff = 4d  →  2 · d · 4d = 8d²', D.blue.dark, 14.5, 12);
  o += card(284, 20, 236, 120, C.lgreen, C.green, '3-Matrix Gated SwiGLU', '(SiLU(x W_g) ⊙ (x W₁)) W₂', 'Width d_ff = (8/3)d  →  3 · d · (8/3)d = 8d²', D.green.dark, 14.5, 11.5);

  o += card(20, 160, 500, 88, C.lamber, C.amber, 'Matched Parameter & FLOP Budget, Richer Bilinear Channel Interactions', 'At d = 256: GELU uses 2 × 256 × 1024 = 524,288 params; SwiGLU uses 3 × 256 × 683 = 524,544.', 'Multiplicative gating (SiLU(x W_g) ⊙ x W₁) beats GELU by ~0.06 nats/byte at identical cost!', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §11.2 gpt.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 13. ch11-tf-gpt-parallel-prefixes
function gptParallelPrefixes() {
  const W = 540, H = 272;
  let o = '';
  o += tx(270, 20, 'One Window of Length T Trains T Prefix Prediction Tasks Simultaneously', { fs: 14.5, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 96, C.lblue, C.blue, 'Input Window X[:, 0:T]', 'x₁,  x₂,  x₃,  …,  x_T', ' Causal Mask blocks s > t', D.blue.dark, 14, 12.5);
  o += arrow(260, 90, 280, 90, C.green);
  o += card(284, 42, 236, 96, C.lgreen, C.green, 'Shifted Targets Y = X[:, 1:T+1]', 'x₂,  x₃,  x₄,  …,  x_(T+1)', '100% of tokens supervised!', D.green.dark, 14, 12.5);

  o += card(20, 154, 500, 98, D.purple.tint, C.purple, 'Chain-Rule Factorization:  log P(x_(1:T+1)) = ∑_{t=1}^{T} log P(x_(t+1) | x_(1:t))', 'Row t of logits Z_t depends ONLY on prefix x_(1:t) because upper-triangular scores are −∞.', 'At step 0 (random weights), uniform p ≈ 1/V gives initial loss ln(256) ≈ 5.55 nats/byte.', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 14. ch11-tf-gpt-fused-qkv-gqa
function gptFusedQkvGqa() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 24, 140, 96, N.panel, C.gray, 'Normed Stream', 'X ∈ ℝ^{B×T×d}', 'single GEMM W\\_qkv', C.ink, 14, 12);
  o += arrow(164, 72, 188, 72, C.blue);
  o += card(192, 18, 164, 108, C.lblue, C.blue, 'Split & RoPE Rotate', 'Q: H heads (d)', 'K, V: H_kv heads (2 H_kv d_h)', D.blue.dark, 13.5, 11.5);
  o += arrow(360, 72, 384, 72, C.green);
  o += card(388, 24, 132, 96, C.lgreen, C.green, 'Causal SDPA', 'is_causal = True', 'grouped KV heads', D.green.dark, 13.5, 12);

  o += card(20, 148, 500, 100, C.lamber, C.amber, 'Fused Projection Size:  d × (d + 2 H_kv d_h)', 'When H_kv = H (standard MHA): W\\_qkv has shape (d, 3d) → 3d² parameters.', 'When H_kv = H / G (GQA): each KV head is shared across G query heads, shrinking KV cache by G×!', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 15. ch11-tf-gpt-learned-vs-rope
function gptLearnedVsRope() {
  const W = 540, H = 266;
  let o = '';
  o += card(20, 20, 236, 118, C.lamber, C.amber, 'pos = "learned" (GPT-2)', 'H₀ = token_emb(X) + pos_emb(0:T)', 'Added ONCE at bottom of stack', D.orange.dark, 14.5, 12);
  o += card(284, 20, 236, 118, C.lgreen, C.green, 'pos = "rope" (Modern Default)', 'H₀ = token_emb(X) (no pos table!)', 'Rotates Q, K in EVERY block', D.green.dark, 14.5, 12);

  o += card(20, 156, 236, 92, D.red.tint, D.red.base, 'Hard Context Ceiling T_max', '1,024 × 768 = 0.79M extra params', 'Cannot extrapolate past T_max', D.red.dark, 13.5, 12);
  o += card(284, 156, 236, 92, C.lblue, C.blue, 'Zero Parameters, Relative Score', 'rope_tables(T, d_h) shared by all L', 'Extends to 32K–1M tokens via YaRN', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 16. ch11-tf-gpt-lm-optimizer
function gptLmOptimizer() {
  const W = 540, H = 276;
  let o = '';
  o += tx(270, 20, 'The Four Pillars of LMOptimizer (Modern LLM Pretraining Recipe)', { fs: 15, fw: 700, fill: C.ink });
  const items = [
    { x: 20,  y: 42,  t: '1. Selective Weight Decay', s1: 'λ = 0.1 on 2D matrices ONLY', s2: 'λ = 0.0 on *emb*, norms, biases', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 284, y: 42,  t: '2. Linear Warmup (t < t_w)', s1: 'η_t = η_max · (t / t_w)', s2: 'Stabilizes early Adam v_t moments', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 20,  y: 154, t: '3. Cosine Decay (t ≥ t_w)', s1: '½ η_max (1 + cos(π(t−t_w)/(S−t_w)))', s2: 'Anneals smoothly to minimum at S', fill: C.lgreen, st: C.green, dk: D.green.dark },
    { x: 284, y: 154, t: '4. Global Grad Clipping', s1: '‖∇L‖₂ ≤ max_norm = 1.0', s2: 'Prevents rare batch loss spikes', fill: D.purple.tint, st: C.purple, dk: D.purple.dark },
  ];
  items.forEach((it) => {
    o += card(it.x, it.y, 236, 94, it.fill, it.st, it.t, it.s1, it.s2, it.dk, 14, 11.5);
  });
  return svg(W, H, o);
}

// 17. ch11-tf-gpt-sampling-pipeline
function gptSamplingPipeline() {
  const W = 540, H = 268;
  let o = '';
  const stages = [
    { x: 18,  w: 114, fill: C.lblue, st: C.blue, dk: D.blue.dark, t: '1. Last Logits', l1: 'z_t = model(X)', l2: 'row −1 ∈ ℝ^{V}' },
    { x: 148, w: 114, fill: C.lamber, st: C.amber, dk: D.orange.dark, t: '2. Temperature', l1: 'z_t ← z_t / τ', l2: 'τ<1 sharpens' },
    { x: 278, w: 114, fill: D.purple.tint, st: C.purple, dk: D.purple.dark, t: '3. Top-k Mask', l1: 'keep top k logits', l2: 'rest ← −∞' },
    { x: 408, w: 114, fill: C.lgreen, st: C.green, dk: D.green.dark, t: '4. Sample', l1: 'p = softmax(z_t)', l2: 'sample v ~ p' },
  ];
  stages.forEach((s, i) => {
    o += card(s.x, 20, s.w, 104, s.fill, s.st, s.t, s.l1, s.l2, s.dk, 13.5, 11.5);
    if (i < 3) o += arrow(s.x + s.w + 3, 72, stages[i + 1].x - 3, 72, C.gray);
  });

  o += card(18, 142, 504, 106, N.panel, C.gray, 'Controlling the Diversity vs. Coherence Spectrum', 'Greedy (top_k = 1 or τ → 0): deterministic argmax  ·  Balanced (τ = 0.7–0.8, top_k = 8–50): fluent text', 'High Temperature (τ = 2.0): flattens distribution toward uniform 1/V, sampling noisy tail tokens.', C.ink, 13.5, 12);
  return svg(W, H, o);
}

// 18. ch11-tf-gpt-checkpoint-map
function gptCheckpointMap() {
  const W = 540, H = 274;
  let o = '';
  o += tx(270, 20, 'Mapping OpenAI GPT-2 Small (124.4M) safetensors to d2l.GPT', { fs: 14.5, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 98, C.lblue, C.blue, 'Embeddings & Final Norm', 'wte (50257×768) → token_emb', 'wpe (1024×768) → pos_emb · ln_f → norm', D.blue.dark, 13.5, 11.5);
  o += card(284, 42, 236, 98, C.lgreen, C.green, '12 × TransformerBlock (h.0…11)', 'ln_1, ln_2 → norm1, norm2', 'c\\_attn, c\\_proj → W\\_qkv, W\\_o · mlp → W\\_1, W\\_2', D.green.dark, 13.5, 11.5);

  o += card(20, 160, 500, 94, D.red.tint, D.red.base, 'Layout Check: GPT-2 Stores Its Conv1D Weights as (in, out)', 'A dense layer that stores (out, in) must transpose every 2D Conv1D weight (W.T);', 'one that stores (in, out) copies the Conv1D weights directly, without a transpose.', D.red.dark, 13, 12);
  return svg(W, H, o);
}

// 19. ch11-tf-gpt-frontier-evolution
function gptFrontierEvolution() {
  const W = 540, H = 272;
  let o = '';
  o += tx(270, 20, 'Shared Backbone & Differentiators Across 2023–2025 Open-Weights LLMs', { fs: 14.5, fw: 700, fill: C.ink });

  o += card(20, 42, 500, 66, C.lblue, C.blue, 'Universal Core Backbone (Mistral, Llama 3, Qwen3, OLMo 2, DeepSeek-V3, Gemma 3, gpt-oss)', 'Pre-Norm RMSNorm (bias=False)  +  Rotary Position Embeddings (RoPE)  +  Gated SwiGLU / GeGLU', null, D.blue.dark, 13.5, 12);

  o += card(20, 128, 236, 124, C.lamber, C.amber, 'Differentiator 1: KV Cache', 'GQA 32:8 or 64:8 (Llama 3, Qwen3)', '5:1 Window + Global (Gemma 3) · MLA (DeepSeek)', D.orange.dark, 13.5, 11.5);
  o += card(284, 128, 236, 124, C.lgreen, C.green, 'Differentiator 2: FFN Sparsity', 'Dense SwiGLU (Llama 3 8B–405B)', 'Fine-Grained MoE 128–256E (Qwen3, DeepSeek-V3)', D.green.dark, 13.5, 11.5);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §11.3 kv-cache.md (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 20. ch11-tf-kv-recompute-vs-cache
function kvRecomputeVsCache() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 14, 236, 104, D.red.tint, D.red.base, 'Naive Generation: O(T²) FLOPs', 'Re-runs full prefix x_(1:t) at step t', '3584 + 3585 + … + 3839 = 950,144 tok', D.red.dark, 14, 12);
  o += card(284, 14, 236, 104, C.lgreen, C.green, 'Cached Generation: O(T) FLOPs', 'Prefill 3584 once + 256×1 decode steps', 'Total processed = 3,840 tokens (250× less!)', D.green.dark, 14, 12);

  o += card(20, 134, 500, 116, C.lblue, C.blue, 'Why Causality Makes Past Keys & Values Immutable', 'Causal attention is lower-triangular: (k_s, v_s) for s < t never changes when x_t arrives.', 'Measured on GPT-2 Small (4K context): 256 tokens in ≈ 0.5 s (550–650 tok/s) vs. 8–15 s (17–32 tok/s) naive!', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 21. ch11-tf-kv-static-ring-buffer
function kvStaticRingBuffer() {
  const W = 540, H = 270;
  let o = '';
  o += tx(270, 20, 'Static Pre-Allocated Cache Buffer (B, H_kv, T_max, d_h) per Block', { fs: 14.5, fw: 700, fill: C.ink });

  // Draw a segmented buffer bar
  o += box(20, 44, 210, 64, C.lblue, C.blue, 6, 2);
  o += tx(125, 66, 'Filled Prefix [0 … t−1]', { fs: 13.5, fw: 700, fill: D.blue.dark, availW: 210 });
  o += tx(125, 88, 'read-only cached K, V', { mono: true, fs: 12, fill: C.ink, availW: 210 });

  o += box(234, 44, 112, 64, C.lgreen, C.green, 6, 2.5);
  o += tx(290, 66, 'Slot [t]', { fs: 14, fw: 700, fill: D.green.dark, availW: 112 });
  o += tx(290, 88, 'write k_t, v_t', { mono: true, fs: 12, fill: D.green.dark, availW: 112 });

  o += box(350, 44, 170, 64, N.panel, C.gray, 6, 1.5, true);
  o += tx(435, 66, 'Unwritten [t+1 … T_max]', { fs: 13, fw: 600, fill: C.muted, availW: 170 });
  o += tx(435, 88, 'masked out (False)', { mono: true, fs: 12, fill: C.muted, availW: 170 });

  o += card(20, 132, 500, 118, D.purple.tint, C.purple, 'Why Static Shapes + In-Place Writes Are Mandatory for Fast Compilation', '1. Dynamic concatenation inside a token loop reallocates memory and forces recompilation every step.', '2. Fixed buffer address + RoPE offset t + boolean mask (idx ≤ t) lets the compiler build the step once!', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 22. ch11-tf-kv-bandwidth-floor
function kvBandwidthFloor() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 14, 236, 100, C.lblue, C.blue, 'Compute Roof (83 TFLOP/s)', '2N = 2.5 × 10⁸ FLOPs per token', 'Theoretical compute time: 0.003 ms!', D.blue.dark, 14, 12);
  o += card(284, 14, 236, 100, D.red.tint, D.red.base, 'HBM Bandwidth Floor (1 TB/s)', 'Streams 498 MB weights + KV cache', 'Physical transfer floor: ≥ 0.50 ms!', D.red.dark, 14, 12);

  o += card(20, 130, 500, 120, C.lamber, C.amber, 'Single-Stream Decode (B = 1) Is 160× Slower Than Compute Peak', 'Every generated token streams all 124M weights from HBM just to do 1 matrix–vector multiply!', 'Arithmetic intensity I ≈ 0.38 FLOPs/byte at 2K context (vs. 82 FLOPs/byte ridge). Measured latency: ~1.0 ms.', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 23. ch11-tf-kv-three-axes
function kvThreeAxes() {
  const W = 540, H = 274;
  let o = '';
  o += card(20, 14, 500, 50, D.purple.tint, C.purple, 'Per-Sequence KV Cache Footprint:  M_KV(n, 1) = 2 b L · (H_kv · d_h · n)', null, null, D.purple.dark, 14.5, 12);

  const cw = 154, y = 84, ch = 166;
  o += card(20, y, cw, ch, C.lblue, C.blue, 'Axis 1: Heads H_kv', 'GQA & MQA', 'H_kv = H / G (4–8×)', D.blue.dark, 14, 12,
    [{ t: 'Uptrain in 5–10% steps', fill: D.blue.dark }]);

  o += card(193, y, cw, ch, C.lgreen, C.green, 'Axis 2: Width 2H_{kv} d_h', 'Low-Rank SVD / MLA', '2d → r (6–16× smaller)', D.green.dark, 14, 12,
    [{ t: 'Absorb B_k, B_v in W_q, W_o', fill: D.green.dark }]);

  o += card(366, y, cw, ch, C.lamber, C.amber, 'Axis 3: Length n', 'Window w + s Sinks', 'n → w + s (constant!)', D.orange.dark, 14, 12,
    [{ t: 's ≥ 1 prevents collapse', fill: D.orange.dark }]);
  return svg(W, H, o);
}

// 24. ch11-tf-kv-mla-absorption
function kvMlaAbsorption() {
  const W = 540, H = 270;
  let o = '';
  o += tx(270, 20, 'Low-Rank KV Compression & Multi-Head Latent Attention (MLA)', { fs: 14.5, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 100, C.lblue, C.blue, '1. Compress Token to Latent c_t', 'c_t = x_t W_DKV ∈ ℝʳ (r ≪ 2d)', 'Cache stores ONLY c_t (16× smaller!)', D.blue.dark, 13.5, 12);
  o += arrow(260, 92, 280, 92, C.green);
  o += card(284, 42, 236, 100, C.lgreen, C.green, '2. Matrix Absorption at Decode', 'q_t k_jᵀ = (x_t W_q B_kᵀ) c_jᵀ', 'Absorb B_kᵀ into W_q & B_v into W_o!', D.green.dark, 13.5, 12);

  o += card(20, 158, 500, 92, C.lamber, C.amber, 'Why GPT-2’s 1,536-Dim [k_t; v_t] Compresses 6× Without Retraining', 'Because [k_t; v_t] = Norm(h_t) W_kv + b_kv is an affine image of ℝ⁷⁶⁸, its true rank is ≤ 768;', 'projecting onto the top r = 256 SVD directions preserves perplexity (37 vs. 36)!', D.orange.dark, 13, 12);
  return svg(W, H, o);
}

// 25. ch11-tf-kv-sink-window
function kvSinkWindow() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'Why a Sliding Window Needs Attention Sinks (StreamingLLM)', { fs: 15, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 106, D.red.tint, D.red.base, 'Pure Window (w = 256, s = 0)', 'Evicts Token 0 once t ≥ 256', 'GPT-2 Perplexity explodes: 36 → ≈ 3,840!', D.red.dark, 13.5, 12);
  o += card(284, 42, 236, 106, C.lgreen, C.green, 'Window + Sinks (w = 256, s = 4)', 'Keeps slots [0…3] + last 256 tokens', 'Perplexity restored to 43 (zero-shot!)', D.green.dark, 13.5, 12);

  o += card(20, 168, 500, 82, C.lblue, C.blue, 'Why Heads Dump 30–57% of Weight on Token 0', 'Softmax weights must sum to 1 even when a head finds no match in the local context;', 'Token 0 is visible to every causal query, acting as a default no-op sink!', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §11.4 encoders-decoders.md (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 26. ch11-tf-enc-bidirectional-cloze
function encBidirectionalCloze() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'Disambiguating a Missing Byte: "… known as _he European lobster"', { fs: 14.5, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 106, C.lamber, C.amber, 'Left Context Only (Causal)', '"Homarus gammarus, known as _"', 'Hundreds of valid next letters', D.orange.dark, 13.5, 12);
  o += card(284, 42, 236, 106, C.lgreen, C.green, 'Left + Right Context (Encoder)', '"… known as [_] he European …"', 'Right context pins "_" → "t" (96–97%)', D.green.dark, 13.5, 11.5);

  o += card(20, 168, 500, 80, C.lblue, C.blue, 'Value of the Second Side of Context on WikiText-103', 'Giving an encoder both left and right context cuts reconstruction loss by ~0.85 nats/byte', 'and makes every token embedding a summary of the entire document!', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 27. ch11-tf-enc-multihop-leakage
function encMultihopLeakage() {
  const W = 540, H = 270;
  let o = '';
  o += tx(270, 20, 'Why an Unmasked Multi-Layer Encoder Cannot Train on Next-Token Loss', { fs: 14, fw: 700, fill: D.red.dark });

  o += card(20, 44, 236, 104, D.red.tint, D.red.base, 'Layer 1 Direct Leakage', 'Position t attends to key t + 1', 'Copies target x_(t+1) directly into h_t^{(1)}!', D.red.dark, 13.5, 12);
  o += card(284, 44, 236, 104, C.lamber, C.amber, 'Layer 2 Multi-Hop Leakage', 'Position t attends to key t + 2 (or t − 1)', 'which already read x_(t+1) in Layer 1!', D.orange.dark, 13.5, 12);

  o += card(20, 168, 500, 82, C.lgreen, C.green, 'The MLM Rule (BERT): Remove Target Tokens from the Input Before Layer 1', 'Replace 15% of positions i ∈ M with <mask> (ID 256) and set unmasked labels to −100', 'so the network must reconstruct xᵢ purely from surrounding context!', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 28. ch11-tf-enc-bathtub-curve
function encBathtubCurve() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'Per-Position Validation Loss Across a 64-Byte Window', { fs: 15, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 114, C.lblue, C.blue, 'Causal GPT (Monotonic Drop)', 'Pos 0: 2.47 nats (no left context)', 'Pos 64: ≈ 1.33 nats (long left prefix)', D.blue.dark, 14, 12);
  o += card(284, 42, 236, 114, C.lgreen, C.green, 'Bidirectional MLM (U-Curve)', 'Edges (Pos 0 & 63): ≈ 2.4 nats (1-sided)', 'Interior (Pos 16…47): ≈ 1.55 nats (2-sided!)', D.green.dark, 14, 11.5);

  o += card(20, 174, 500, 74, D.purple.tint, C.purple, 'Why the Bathtub Curve Is Symmetric', 'Interior masked bytes read both prefix and suffix bytes; only positions near the two', 'window boundaries (t ≈ 0 or t ≈ 63) lose one side of their context window.', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 29. ch11-tf-enc-decoder-block
function encDecoderBlock() {
  const W = 540, H = 276;
  let o = '';
  o += tx(270, 20, 'Three Pre-Norm Sublayers Inside an Encoder–Decoder DecoderBlock', { fs: 14.5, fw: 700, fill: C.ink });

  const stages = [
    { x: 18,  w: 154, fill: C.lblue, st: C.blue, dk: D.blue.dark, t: '1. Causal Self-Attn', l1: 'Q, K, V from target X', l2: 'causal_mask(T_dec)' },
    { x: 193, w: 154, fill: C.lamber, st: C.amber, dk: D.orange.dark, t: '2. Cross-Attention', l1: 'Q from X · K,V from enc', l2: 'NO causal mask!' },
    { x: 368, w: 154, fill: C.lgreen, st: C.green, dk: D.green.dark, t: '3. SwiGLU FFN', l1: 'Position-wise on X', l2: 'channel mixing (8d²)' },
  ];
  stages.forEach((s, i) => {
    o += card(s.x, 42, s.w, 112, s.fill, s.st, s.t, s.l1, s.l2, s.dk, 13.5, 11.5);
    if (i < 2) o += arrow(s.x + s.w + 3, 98, stages[i + 1].x - 3, 98, C.gray);
  });

  o += card(18, 174, 504, 82, D.purple.tint, C.purple, 'Inference Efficiency: Encode Once, Decode T_dec Steps', 'Because H_enc = encoder(src) depends only on the source, the encoder and cross-attention', 'key–value projections run ONCE per source sequence and are reused at every target step!', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 30. ch11-tf-enc-three-masks
function encThreeMasks() {
  const W = 540, H = 272;
  let o = '';
  o += tx(270, 20, 'The Three Masks of a Ragged Sequence-to-Sequence Minibatch', { fs: 15, fw: 700, fill: C.ink });
  const cw = 154, y = 42, ch = 128;
  o += card(20, y, cw, ch, C.lblue, C.blue, '1. Encoder Self', 'Shape (B, 1, S, S)', 'Source pad (j < s_b)', D.blue.dark, 14, 12);
  o += card(193, y, cw, ch, D.purple.tint, C.purple, '2. Decoder Self', 'Shape (B, 1, T, T)', 'Causal ∧ Target pad', D.purple.dark, 14, 12);
  o += card(366, y, cw, ch, C.lgreen, C.green, '3. Cross-Attn', 'Shape (B, 1, T, S)', 'Source pad ONLY!', D.green.dark, 14, 12);

  o += card(20, 188, 500, 66, D.red.tint, D.red.base, 'Critical Rule: Never Apply a Causal Mask to Cross-Attention!', 'Target step t = 0 must be free to attend to the LAST source token j = S − 1 immediately.', null, D.red.dark, 13.5, 12);
  return svg(W, H, o);
}

// 31. ch11-tf-enc-wiring-decision
function encWiringDecision() {
  const W = 540, H = 272;
  let o = '';
  o += tx(270, 20, 'Selecting the Right Transformer Wiring for Your Task', { fs: 15, fw: 700, fill: C.ink });
  const cw = 154, y = 42, ch = 136;
  o += card(20, y, cw, ch, C.lblue, C.blue, 'Encoder-Only', 'Full 2-sided context', '1 parallel pass (no KV)', D.blue.dark, 14, 12,
    [{ t: 'Embeddings · Rerank · ViT', fw: 700, fill: D.blue.dark }]);

  o += card(193, y, cw, ch, C.lgreen, C.green, 'Decoder-Only', '100% tokens scored', '1 stack + 1 KV cache', D.green.dark, 14, 12,
    [{ t: 'General LLMs · Code · Chat', fw: 700, fill: D.green.dark }]);

  o += card(366, y, cw, ch, C.lamber, C.amber, 'Encoder–Decoder', 'Encode source once', 'Cross-attend from tgt', D.orange.dark, 14, 12,
    [{ t: 'Whisper Speech · Perceiver', fw: 700, fill: D.orange.dark }]);

  o += card(20, 194, 500, 60, D.purple.tint, C.purple, 'All three wirings are built from the exact same configurable d2l.TransformerBlock!', null, null, D.purple.dark, 14, 12);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §11.5 vision-transformer.md (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 32. ch11-tf-vit-pixel-vs-patch
function vitPixelVsPatch() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 20, 236, 114, D.red.tint, D.red.base, 'Pixel Tokens (96 × 96)', 'm = 9,216 tokens per image', 'm² = 84.9M pairs / head / layer!', D.red.dark, 14, 12);
  o += arrow(260, 77, 280, 77, C.green);
  o += card(284, 20, 236, 114, C.lgreen, C.green, '16 × 16 Patch Tokens (ViT)', 'm = (96/16)² = 36 patch tokens', 'm² = 1,296 pairs (65,536× fewer!)', D.green.dark, 14, 12);

  o += card(20, 154, 500, 94, C.lblue, C.blue, 'Quadratic Reduction by p⁴  (At p = 16:  p² = 256× fewer tokens,  p⁴ = 65,536× fewer scores)', 'On a 224 × 224 ImageNet image: pixel attention requires 2.52 billion pairs/head,', 'whereas p = 16 patches produce m = 14² = 196 tokens (only 38,416 pairs/head)!', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 33. ch11-tf-vit-cnn-vs-vit-bias
function vitCnnVsVitBias() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 20, 236, 116, C.lblue, C.blue, 'Convolutional Net (ResNet)', 'Hard-coded 2D k × k locality', 'Built-in translation equivariance', D.blue.dark, 14.5, 12);
  o += card(284, 20, 236, 116, D.purple.tint, C.purple, 'Vision Transformer (ViT)', 'Global all-pairs mixing at Layer 1', 'Zero built-in 2D grid topology', D.purple.dark, 14.5, 12);

  o += card(20, 156, 236, 94, C.lgreen, C.green, 'Wins in Small-Data Regime', '≈94% on 60K Fashion-MNIST', 'Strong spatial prior prevents overfit', D.green.dark, 13.5, 12);
  o += card(284, 156, 236, 94, C.lamber, C.amber, 'Wins at Large Pretraining Scale', 'Surpasses CNNs at 30M–300M+ images', 'From scratch on 60K images: only ≈89%', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 34. ch11-tf-vit-strided-conv-stem
function vitStridedConvStem() {
  const W = 540, H = 266;
  let o = '';
  o += card(20, 30, 142, 98, C.lblue, C.blue, 'Input Image X', '96 × 96 × C', 'non-overlapping tiles', D.blue.dark, 14, 12);
  o += arrow(166, 79, 192, 79, C.purple);
  o += card(196, 24, 152, 110, D.purple.tint, C.purple, 'p × p Conv, Stride p', '16 × 16, stride 16', 'output: 6 × 6 grid, d = 512', D.purple.dark, 13.5, 12);
  o += arrow(352, 79, 378, 79, C.green);
  o += card(382, 30, 138, 98, C.lgreen, C.green, 'Flatten the Grid', '6 × 6 → 36 tokens', 'shape (B, 36, 512)', D.green.dark, 13.5, 12);

  o += card(20, 158, 500, 88, C.lamber, C.amber, 'Why Strided Convolution Equals Patch Slicing + Linear Projection W_E ∈ ℝ^{Cp²×d}', 'Each of the d = 512 conv filters has weights of shape (C, 16, 16) = 256C parameters,', 'computing the exact inner product of each 16 × 16 patch with one column of W_E in 1 fused kernel!', D.orange.dark, 13, 12);
  return svg(W, H, o);
}

// 35. ch11-tf-vit-2d-pos-schemes
function vit2dPosSchemes() {
  const W = 540, H = 270;
  let o = '';
  o += tx(270, 20, 'Three Ways to Inject 2D Grid Coordinates (r, c) into Patch Tokens', { fs: 14.5, fw: 700, fill: C.ink });
  const cw = 154, y = 42, ch = 132;
  o += card(20, y, cw, ch, C.lblue, C.blue, '1. Learned 1D Table', 'p_k ∈ ℝ^{d} (k = 0…m)', 'Discovers 2D from data', D.blue.dark, 13.5, 11.5);
  o += card(193, y, cw, ch, C.lamber, C.amber, '2. 2D Sinusoidal', '[sin/cos(r); sin/cos(c)]', 'Hard-codes grid at t=0', D.orange.dark, 13.5, 11.5);
  o += card(366, y, cw, ch, C.lgreen, C.green, '3. 2D Axial RoPE', 'Rotate ½ by r, ½ by c', 'Relative 2D shift inv.', D.green.dark, 13.5, 11.5);

  o += card(20, 192, 500, 58, D.purple.tint, C.purple, 'Without pos_embedding:  ViT(shuffle_patches(X, perm)) == ViT(X)  (exact permutation invariance!)', null, null, D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 36. ch11-tf-vit-adamw-recipe
function vitAdamwRecipe() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 20, 236, 110, D.red.tint, D.red.base, 'Why Plain SGD Struggles on ViT', 'Q/K projections & MLPs have wildly', 'different gradient scales (>7% drop)', D.red.dark, 13.5, 12);
  o += card(284, 20, 236, 110, C.lgreen, C.green, 'Why AdamW + Warmup Succeeds', 'Per-coordinate second-moment scaling', '+ 10% linear warmup + cosine decay', D.green.dark, 13.5, 12);

  o += card(20, 150, 500, 98, C.lblue, C.blue, 'Selective Weight Decay Rule in ViT.configure_optimizers (λ = 0.05)', 'Apply weight decay (λ = 0.05) to 2D linear matrices and 4D conv kernels (p.ndim ≥ 2),', 'while EXEMPTING cls_token, pos_embedding, 1D LayerNorm gains, and biases (λ = 0.0)!', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 37. ch11-tf-vit-scale-crossover
function vitScaleCrossover() {
  const W = 540, H = 270;
  let o = '';
  o += tx(270, 20, 'The Inductive Bias vs. Dataset Scale Crossover (Dosovitskiy et al., 2021)', { fs: 14.5, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 112, C.lblue, C.blue, 'Small Data (< 1M–10M Images)', 'ResNet > ViT (+5% on Fashion-MNIST)', 'Hard-coded 2D locality wins', D.blue.dark, 13.5, 12);
  o += arrow(260, 98, 280, 98, C.green);
  o += card(284, 42, 236, 112, C.lgreen, C.green, 'Foundation Scale (30M–300M+)', 'ViT > ResNet (CLIP, SigLIP, DINOv2)', 'Flexible global attention outgrows CNN', D.green.dark, 13.5, 12);

  o += card(20, 174, 500, 76, D.purple.tint, C.purple, 'Unified Multimodal Interface', 'Because ViT outputs standard token vectors (B, m, d), visual patches plug directly into', 'Perceiver resamplers (§11.4) and decoder-only LLMs (§11.2) with zero architectural friction!', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §11.6 scaling-laws.md (7 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 38. ch11-tf-scale-budget-tradeoff
function scaleBudgetTradeoff() {
  const W = 540, H = 270;
  let o = '';
  o += card(130, 10, 280, 58, D.purple.tint, C.purple, 'Fixed Compute Budget: C ≈ 6 N D', 'How to split FLOPs between N and D?', null, D.purple.dark, 14.5, 12.5);

  o += arrow(210, 70, 138, 100, D.red.base);
  o += arrow(330, 70, 402, 100, D.red.base);
  o += arrow(270, 70, 270, 180, C.green);

  o += card(20, 104, 204, 68, D.red.tint, D.red.base, 'Too Large N, Small D', 'Under-trained (data-starved)', null, D.red.dark, 13.5, 11.5);
  o += card(316, 104, 204, 68, D.red.tint, D.red.base, 'Too Small N, Huge D', 'Capacity-bottlenecked', null, D.red.dark, 13.5, 11.5);

  o += card(70, 184, 400, 70, C.lgreen, C.green, 'Chinchilla Pretraining Optimum:  N* ∝ C^{0.5},  D* ∝ C^{0.5}', 'Scale parameters N and tokens D equally (~20 tokens per parameter)', null, D.green.dark, 14, 12.5);
  return svg(W, H, o);
}

// 39. ch11-tf-scale-six-nd
function scaleSixNd() {
  const W = 540, H = 270;
  let o = '';
  o += tx(270, 20, 'Why Every Parameter Matrix W (P Weights) Costs 6P FLOPs per Token', { fs: 14.5, fw: 700, fill: C.ink });
  const cw = 154, y = 42, ch = 128;
  o += card(20, y, cw, ch, C.lblue, C.blue, '1. Forward Pass', 'Y = X W', '2 P FLOPs / token', D.blue.dark, 14, 12.5);
  o += card(193, y, cw, ch, C.lamber, C.amber, '2. Backward Act', '∇_X L = (∇_Y L) Wᵀ', '2 P FLOPs / token', D.orange.dark, 14, 12.5);
  o += card(366, y, cw, ch, D.purple.tint, C.purple, '3. Backward Grad', '∇_W L = Xᵀ (∇_Y L)', '2 P FLOPs / token', D.purple.dark, 14, 12.5);

  o += card(20, 188, 500, 64, C.lgreen, C.green, 'Total Training Compute Over N Parameters and D Tokens:  C ≈ (2 + 2 + 2) N D = 6 N D', 'Backward pass ≈ 2× the forward pass (checked with FLOP counters)!', null, D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 40. ch11-tf-scale-attn-crossover
function scaleAttnCrossover() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 20, 236, 114, C.lblue, C.blue, 'Linear Projections (12d²/blk)', 'Forward: 24 d² FLOPs / token', 'Dominates when T < 6d', D.blue.dark, 14, 12);
  o += card(284, 20, 236, 114, C.lamber, C.amber, 'Attention QKᵀ & AV', 'Forward: 4 T d FLOPs / token', 'Ratio to projections = T / (6d)', D.orange.dark, 14, 12);

  o += card(20, 154, 500, 94, C.lgreen, C.green, 'Complete Per-Token Training Formula:  C = 6 N D · (1 + T / (6d))', 'At T = 128, d = 256 (our byte GPT): attention adds only T/(6d) ≈ +8% to 6ND.', 'At d = 4,096 (Llama 3 8B): attention FLOPs equal linear projections at T = 6d = 24,576 tokens.', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 41. ch11-tf-scale-cosine-horizon
function scaleCosineHorizon() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'Why Reading Early Checkpoints of a Long Schedule Biases Scaling Laws', { fs: 14, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 104, D.red.tint, D.red.base, 'Un-Annealed Checkpoint (s=500)', 'Read at step 500 of a 2000-step cosine', 'LR η_500 ≈ 0.9 η_max (still hot!)', D.red.dark, 13.5, 12);
  o += card(284, 42, 236, 104, C.lgreen, C.green, 'Matched Cosine Horizon (s=500)', 'Cosine decays fully to 0 at step 500', 'Weights settle cleanly into minimum', D.green.dark, 13.5, 12);

  o += card(20, 162, 500, 88, C.lamber, C.amber, 'Hoffmann et al. (2022) Correction to Kaplan et al. (2020)', 'Evaluating un-annealed intermediate steps overestimates loss at small token budgets D,', 'artificially tilting the estimated scaling exponent toward larger N (Kaplan 0.73 vs. corrected 0.50)!', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 42. ch11-tf-scale-chinchilla-surface
function scaleChinchillaSurface() {
  const W = 540, H = 270;
  let o = '';
  o += card(20, 16, 500, 50, D.purple.tint, C.purple, 'Chinchilla Additive Loss Decomposition:  L(N, D) = L_{∞} + A / N^α + B / D^β', null, null, D.purple.dark, 14.5, 12);

  const cw = 154, y = 84, ch = 164;
  o += card(20, y, cw, ch, N.panel, C.gray, '1. Entropy Floor L_{∞}', '≈ 1.69–1.82 nats/tok', 'Irreducible Shannon noise of natural language', C.ink, 13.5, 11.5);

  o += card(193, y, cw, ch, C.lblue, C.blue, '2. Capacity Error', 'A / N^α  (α ≈ 0.35)', 'Finite-width/depth approximation gap', D.blue.dark, 13.5, 11.5);

  o += card(366, y, cw, ch, C.lgreen, C.green, '3. Data Error', 'B / D^β  (β ≈ 0.37)', 'Finite-sample SGD estimation gap', D.green.dark, 13.5, 11.5);
  return svg(W, H, o);
}

// 43. ch11-tf-scale-kaplan-vs-chinchilla
function scaleKaplanVsChinchilla() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 20, 236, 120, C.lamber, C.amber, 'Kaplan et al. (2020)', 'N* ∝ C^{0.73},   D* ∝ C^{0.27}', '100× compute → 29× larger N, 3.5× D', D.orange.dark, 14.5, 12);
  o += card(284, 20, 236, 120, C.lgreen, C.green, 'Chinchilla (2022) & Refits', 'N* ∝ C^{0.50},   D* ∝ C^{0.50}', '100× compute → 10× larger N, 10× D', D.green.dark, 14.5, 12);

  o += card(20, 158, 500, 92, C.lblue, C.blue, 'Three Methodological Reasons the Exponent Moved from 0.73 to 0.50', '1. Decaying cosine LR to minimum at the end of every budget D  ·  2. Accounting for embedding FLOPs', '3. Tuning warmup & width-scaled peak learning rate (η ∝ 1/d) on small models.', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 44. ch11-tf-scale-lifetime-tco
function scaleLifetimeTco() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 16, 500, 52, D.purple.tint, C.purple, 'Total Lifetime Compute:  C_lifetime(N, D) = 6 N D (train) + 2 N D_inf (inference)', 'Sliding along constant-loss contour L(N, D) = L_target trades upfront D for smaller N:', null, D.purple.dark, 13.5, 12);

  o += card(20, 88, 236, 160, C.lamber, C.amber, 'Chinchilla-Allocated (~34B)', 'D ≈ 0.64T tokens (20 tok/param)', 'Cheap pretrain: 1.3 × 10²³ FLOPs', D.orange.dark, 14, 12,
    [{ t: 'Expensive serving: 68 GFLOPs/tok', mono: true, fw: 700, fill: D.orange.dark }]);

  o += card(284, 88, 236, 160, C.lgreen, C.green, 'Over-Trained (Llama 3 8B)', 'D = 15T tokens (1,875 tok/param!)', 'Higher pretrain: 7.2 × 10²³ FLOPs', D.green.dark, 14, 12,
    [{ t: '4.2× cheaper serving: 16 GFLOPs/tok!', mono: true, fw: 700, fill: D.green.dark }]);
  return svg(W, H, o);
}

// ═══════════════════════════════════════════════════════════════════════════
// §11.7 moe.md (6 diagrams)
// ═══════════════════════════════════════════════════════════════════════════

// 45. ch11-tf-moe-dense-vs-sparse
function moeDenseVsSparse() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 16, 236, 106, C.lblue, C.blue, 'Dense FFN Block', '1 FFN of width (8/3)d (8d² params)', 'N_active = N_total (100% active)', D.blue.dark, 14.5, 12);
  o += card(284, 16, 236, 106, C.lgreen, C.green, 'Sparse MoE Block', 'E Expert FFNs · Router picks top k', 'N_active ≪ N_total (k / E active!)', D.green.dark, 14.5, 12);

  o += card(20, 140, 500, 108, D.purple.tint, C.purple, 'Why the FFN Is Ideal for Conditional Computation', 'The FFN holds 2/3 of a dense block’s parameters (8d² of 12d²) and acts position-wise on each x_t.', 'DeepSeek-V3 stores 671B parameters in HBM while executing only 37B active parameters per token!', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 46. ch11-tf-moe-router-gradient
function moeRouterGradient() {
  const W = 540, H = 272;
  let o = '';
  o += tx(270, 20, 'How Router Weights W_r Learn Through Discrete Top-k Selection', { fs: 14.5, fw: 700, fill: C.ink });

  o += card(20, 42, 236, 104, C.lgreen, C.green, 'Unnormalized Gate: g_e = p_e(x)', 'MoE(x) = ∑_{e∈T_k} p_e(x) FFN_{e}(x)', '∂L/∂p_e = (∇_MoE L)ᵀ FFN_{e}(x) ≠ 0!', D.green.dark, 13.5, 11.5);
  o += card(284, 42, 236, 104, D.red.tint, D.red.base, 'k = 1 Renormalization Trap!', 'If g_e = p_e / ∑_{j∈T₁} p_j = p_e / p_e ≡ 1', 'Gate is constant 1 ⇒ ∂MoE/∂W_r = 0!', D.red.dark, 13.5, 11.5);

  o += card(20, 162, 500, 88, C.lamber, C.amber, 'Rule: Never Renormalize Top-1 Router Probabilities', 'In Top-1 routing (Switch Transformer), multiplying by the raw softmax probability p_(e*)(x)', 'is the ONLY path for task-loss gradients to update the router matrix W_r!', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 47. ch11-tf-moe-capacity-buffer
function moeCapacityBuffer() {
  const W = 540, H = 270;
  let o = '';
  o += tx(270, 20, 'Fixed-Capacity Expert Buffers:  S = ⌈γ · n k / E⌉ Slots per Expert (γ = 1.25)', { fs: 14, fw: 700, fill: C.ink });

  o += card(20, 44, 236, 106, C.lgreen, C.green, 'Queue Rank < S (Accepted)', 'Dispatched to slot (e · S + rank)', 'Processed in parallel (E, S, d) bmm', D.green.dark, 13.5, 12);
  o += card(284, 44, 236, 106, D.red.tint, D.red.base, 'Queue Rank ≥ S (Overflow!)', 'Routed to dummy sink slot (E · S)', 'Returns 0 vector (dropped token!)', D.red.dark, 13.5, 12);

  o += card(20, 170, 500, 80, C.lblue, C.blue, 'What Happens to a Dropped Token?', 'Because the MoE layer sits inside a pre-norm residual branch Y = H + MoE(Norm(H)),', 'a dropped token receives 0 from the FFN and passes unchanged along the residual stream H!', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 48. ch11-tf-moe-collapse-loop
function moeCollapseLoop() {
  const W = 540, H = 274;
  let o = '';
  o += tx(270, 20, 'The Rich-Get-Richer Positive Feedback Loop of Routing Collapse', { fs: 14.5, fw: 700, fill: D.red.dark });

  const steps = [
    { x: 20,  y: 42,  t: '1. Initial Skew (t = 0)', s1: 'Frequent bytes (" " = 19%)', s2: 'favor Expert e at random init' },
    { x: 284, y: 42,  t: '2. More Gradient Steps', s1: 'Expert e sees 3× more tokens', s2: 'and lowers loss faster' },
    { x: 284, y: 158, t: '3. Router Reward', s1: '∇_(W_r) L boosts p_e(x) higher', s2: 'when Expert e predicts well' },
    { x: 20,  y: 158, t: '4. Routing Collapse!', s1: 'Top experts monopolize 100%', s2: 'while E − k experts go dead!' },
  ];
  steps.forEach((s, i) => {
    o += card(s.x, s.y, 236, 94, i === 3 ? D.red.tint : C.lamber, i === 3 ? D.red.base : C.amber, s.t, s.s1, s.s2, i === 3 ? D.red.dark : D.orange.dark, 14, 12);
  });
  o += arrow(260, 89, 280, 89, C.amber);
  o += arrow(402, 140, 402, 154, C.amber);
  o += arrow(280, 205, 260, 205, D.red.base);
  return svg(W, H, o);
}

// 49. ch11-tf-moe-aux-vs-bias
function moeAuxVsBias() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 20, 236, 118, C.lamber, C.amber, 'Switch Auxiliary Loss', 'L_aux = E ∑_{e} f_e p̄_e  (min = 1)', 'Adds α ∇L_aux to router gradients', D.orange.dark, 14, 12);
  o += card(284, 20, 236, 118, C.lgreen, C.green, 'Loss-Free Bias Control (DeepSeek)', 'Select top_k(p(x) + b), gate by p(x)', 'b_e ← b_e + η_b sign(1/E − f_e)', D.green.dark, 14, 11.5);

  o += card(20, 158, 236, 94, '#FFFFFF', C.amber, 'Gradient Interference', '∇L_aux pushes p(x) toward uniform', 'competing with task gradient ∇L_LM', D.orange.dark, 13.5, 11.5);
  o += card(284, 158, 236, 94, C.lblue, C.blue, 'Zero Gradient Interference', 'Bias b shifts selection threshold only;', 'gating weights p_e(x) stay pure!', D.blue.dark, 13.5, 11.5);
  return svg(W, H, o);
}

// 50. ch11-tf-moe-frontier-spectrum
function moeFrontierSpectrum() {
  const W = 540, H = 270;
  let o = '';
  o += tx(270, 20, 'Evolution from Coarse Top-2 MoEs to Fine-Grained + Shared Experts', { fs: 14.5, fw: 700, fill: C.ink });
  const cw = 154, y = 42, ch = 128;
  o += card(20, y, cw, ch, C.lblue, C.blue, 'Mixtral 8x7B', 'E = 8,  k = 2', '46.7B / 12.9B (3.6×)', D.blue.dark, 14, 12);
  o += card(193, y, cw, ch, C.lamber, C.amber, 'Qwen3-235B-A22B', 'E = 128,  k = 8', '235B / 22B (10.7×)', D.orange.dark, 14, 12);
  o += card(366, y, cw, ch, C.lgreen, C.green, 'DeepSeek-V3', 'E = 256 + 1,  k = 8', '671B / 37B (18.1×)', D.green.dark, 14, 12);

  o += arrow(178, y + ch / 2, 189, y + ch / 2, C.gray);
  o += arrow(351, y + ch / 2, 362, y + ch / 2, C.gray);

  o += card(20, 188, 500, 64, D.purple.tint, C.purple, 'Combinatorial Expressivity:  C(8, 2) = 28 mixtures  →  C(256, 8) ≈ 4.1 × 10¹⁴ mixtures!', null, null, D.purple.dark, 14, 12);
  return svg(W, H, o);
}

// 51. ch11-tf-blk-collapse-law
function blkCollapseLaw() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 12, 500, 72, D.purple.tint, C.purple, 'Analytical Token Spread Law:  s_k ≈ (1 + (1 + g⁴)^k / (n − 1))^(−1/2)', 'Per-projection variance gain:  g² = d σ²   (amplifies common ratio ρ ↦ (1 + g⁴) ρ per block)', null, D.purple.dark, 14, 12);
  o += card(20, 102, 236, 146, D.red.tint, D.red.base, 'Xavier Init (g = 1, g⁴ = 1)', 'ρ doubles every post-norm block', 'k=8: 0.44 · k=16: 0.031 · k=32: 1.2×10⁻⁴', D.red.dark, 14, 12);
  o += card(284, 102, 236, 146, C.lgreen, C.green, 'Small Init σ=0.02 at d=256', 'g² = 0.10  ⇒  1 + g⁴ ≈ 1.01', 'k=8: 0.99 · k=16: 0.99 · k=32: 0.99', D.green.dark, 14, 12);
  return svg(W, H, o);
}

// 52. ch11-tf-blk-summary
function blkSummary() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 18, 236, 108, C.lblue, C.blue, '1. Attention (4d²) & FFN (8d²)', 'Token mixing + SwiGLU channel mixing', 'Total = 12d² parameters per block', D.blue.dark, 13.5, 12);
  o += card(284, 18, 236, 108, C.lgreen, C.green, '2. Pre-Norm + 1/√(2L) Init', 'X ← X + f(RMSNorm(X))', 'Preserves token spread & flat gradients', D.green.dark, 13.5, 12);
  o += card(20, 146, 500, 106, C.lamber, C.amber, '3. RMSNorm & Per-Head QK-Norm Stability', 'RMSNorm drops mean-centering μ and bias β to save 1 HBM reduction pass per token;', 'Per-head QK-Norm bounds attention logits by √d_h regardless of weight growth λ!', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 53. ch11-tf-gpt-depth-init
function gptDepthInit() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 20, 236, 112, C.lblue, C.blue, 'Standard Projections', 'token_emb, W\\_qkv, W\\_g, W\\_1', 'Initialized from N(0, 0.02²)', D.blue.dark, 14, 12);
  o += card(284, 20, 236, 112, C.lgreen, C.green, 'Residual Write Matrices', 'W\\_o (attention) & W\\_2 (FFN)', 'Scaled by 1/√(2L): σ = 0.02 / √(2L)', D.green.dark, 14, 12);
  o += card(20, 152, 500, 96, D.purple.tint, C.purple, 'Why 2L Appears in the Denominator', 'Each of the L blocks adds TWO residual branches (1 Attention + 1 FFN = 2L writes);', 'scaling each write by 1/√(2L) keeps total added variance ∑ (σ² / 2L) = σ² = O(1)!', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 54. ch11-tf-gpt-summary
function gptSummary() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 18, 236, 108, C.lblue, C.blue, '1. Architecture & Tied Head', 'Shared E ∈ ℝ^{V×d}, RoPE, Causal Mask', 'Saves Vd params (38.6M in GPT-2)', D.blue.dark, 13.5, 12);
  o += card(284, 18, 236, 108, C.lgreen, C.green, '2. LMOptimizer & SwiGLU', 'AdamW (2D decay) + warmup/cosine', '1.34 nats/byte (beats GELU & RNNLM)', D.green.dark, 13.5, 12);
  o += card(20, 146, 500, 106, C.lamber, C.amber, '3. Controlled Baselines & Pretrained GPT-2 Compatibility', 'SwiGLU beats GELU by 0.06 nats (5× seed spread); GPT beats RNNLM for context t ≥ 16;', 'Loads official 124.4M OpenAI GPT-2 checkpoint directly (perplexity ≈ 49 on Time Machine).', D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 55. ch11-tf-kv-footprint-formula
function kvFootprintFormula() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 16, 500, 62, D.purple.tint, C.purple, 'KV Cache Footprint:  M_KV(n, B) = 2 · b · B · L · n · H_kv · d_h  bytes', 'Factor 2 = Keys + Values · b = bytes/elem (4 FP32, 2 BF16, 1 FP8)', null, D.purple.dark, 14, 12);
  o += card(20, 98, 236, 148, C.lblue, C.blue, 'GPT-2 Small (B = 1, n = 4,096)', '72 KiB / token in FP32', '288 MiB cache vs. 471 MiB weights!', D.blue.dark, 13.5, 12);
  o += card(284, 98, 236, 148, D.red.tint, D.red.base, 'Batched Serving (B = 16, n = 4,096)', '16 × 288 MiB = 4.5 GiB cache', ' nearly 10× the model weight footprint!', D.red.dark, 13.5, 12);
  return svg(W, H, o);
}

// 56. ch11-tf-kv-ttft-vs-itl
function kvTtftVsItl() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 20, 236, 114, C.lgreen, C.green, 'Time to First Token (TTFT)', 'Governed by Parallel Prefill (n = T_p)', 'Compute-bound (I ≈ 780 FLOPs/B)', D.green.dark, 14, 12);
  o += card(284, 20, 236, 114, C.lamber, C.amber, 'Inter-Token Latency (ITL)', 'Governed by Sequential Decode (n = 1)', 'Bandwidth-bound (I ≈ 0.38 FLOPs/B)', D.orange.dark, 14, 12);
  o += card(20, 154, 500, 94, C.lblue, C.blue, '100× Throughput Gap on the Same GPU and Model (Context n = 2,048)', 'Parallel prefill achieves > 100,000 tokens/s by reusing weights across all 2,048 prompt tokens,', 'whereas single-stream decode achieves ~1,000 tokens/s (~1 ms/token HBM weight read floor).', D.blue.dark, 13.5, 12);
  return svg(W, H, o);
}

// 57. ch11-tf-enc-string-reversal
function encStringReversal() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'Sequence-to-Sequence String Reversal Benchmark (Length T = 12)', { fs: 14.5, fw: 700, fill: C.ink });
  o += card(20, 42, 236, 108, C.lblue, C.blue, 'Source Input (src)', '(x₁, x₂, …, x_T)  over a…p', 'Encoded bidirectionally by Encoder', D.blue.dark, 14, 12);
  o += arrow(260, 96, 280, 96, C.green);
  o += card(284, 42, 236, 108, C.lgreen, C.green, 'Shifted Target (dec_in → tgt)', 'dec_in = (<bos>, x_T, …, x₂)', 'Predicts tgt = (x_T, x_(T−1), …, x₁)', D.green.dark, 13.5, 12);
  o += card(20, 170, 500, 78, D.purple.tint, C.purple, 'Known Ground-Truth Cross-Attention Alignment: Anti-Diagonal (t, T − 1 − t)', 'To emit reversed target position t, the decoder must retrieve source slot j = T − 1 − t!', null, D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 58. ch11-tf-enc-perceiver-flow
function encPerceiverFlow() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 22, 146, 102, C.lblue, C.blue, 'Long Input X', 'Shape (B, N, d)', 'e.g. N = 8,192 tokens', D.blue.dark, 14, 12);
  o += arrow(170, 73, 192, 73, C.amber);
  o += card(196, 22, 148, 102, C.lamber, C.amber, 'Cross-Attn Read', 'M Latents Z₀ ∈ ℝ^{M×d}', 'O(M N d) cost (M = 64)', D.orange.dark, 13.5, 11.5);
  o += arrow(348, 73, 370, 73, C.green);
  o += card(374, 22, 146, 102, C.lgreen, C.green, 'L Latent Blocks', 'Self-Attn on (B, M, d)', 'O(L M² d) cost!', D.green.dark, 14, 12);
  o += card(20, 146, 500, 102, D.purple.tint, C.purple, '734× Fewer Attention Score Pairs at N = 8,192, M = 64, L = 6', 'Full 6-layer encoder: 6 × 8192² = 402.7M pairs/head  vs.  Perceiver: 64×8192 + 6×64² = 0.55M pairs!', 'Used by Flamingo & BLIP-2 Q-Former to compress visual tokens into 32–64 LLM prefix tokens.', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 59. ch11-tf-enc-summary
function encSummary() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 18, 236, 108, C.lblue, C.blue, '1. Bidirectional Encoder + MLM', 'Unmasked self-attention (mask=None)', '15% <mask> avoids multi-hop leakage', D.blue.dark, 13.5, 12);
  o += card(284, 18, 236, 108, C.lgreen, C.green, '2. Encoder–Decoder Cross-Attn', 'Q from target, K & V from H_enc', 'Encode source once; decode T_dec steps', D.green.dark, 13.5, 12);
  o += card(20, 146, 500, 106, D.purple.tint, C.purple, '3. Attention Attribution Pitfall & Perceiver Latent Bottlenecks', 'Multi-layer encoder self-attention can permute tokens before cross-attention reads them;', 'M ≪ N learned latent queries decouple attention cost from input length: O(MNd + LM²d).', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

// 60. ch11-tf-vit-matched-budget
function vitMatchedBudget() {
  const W = 540, H = 268;
  let o = '';
  o += tx(270, 20, 'Controlled Single-GPU Benchmark: ViT vs. CompactResNet on Fashion-MNIST', { fs: 14, fw: 700, fill: C.ink });
  o += card(20, 42, 236, 114, C.lblue, C.blue, 'ViT (L = 2, d = 512, p = 16)', '6.46M params · 0.48 GFLOPs/img', '36 patch tokens + 1 <cls> token', D.blue.dark, 13.5, 12);
  o += card(284, 42, 236, 114, C.lgreen, C.green, 'CompactResNet (7 ResBlocks)', '6.45M params · 0.55 GFLOPs/img', '3×3 conv pyramid + global avg pool', D.green.dark, 13.5, 12);
  o += card(20, 174, 500, 74, C.lamber, C.amber, 'Matched Across All Four Axes: Params, FLOPs, Data (96×96, 10 Epochs), and AdamW', 'Isolates architectural inductive bias (2D conv locality vs. global patch attention) fairly!', null, D.orange.dark, 13.5, 12);
  return svg(W, H, o);
}

// 61. ch11-tf-vit-summary
function vitSummary() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 18, 236, 108, C.lblue, C.blue, '1. Stride-p Patch Stem', '(C, H, W) → m = HW / p² tokens', '12Ld² block params independent of m', D.blue.dark, 13.5, 12);
  o += card(284, 18, 236, 108, C.lamber, C.amber, '2. Learned 2D Geometry', 'Permutation-invariant without pos_emb', 'Shuffling patches drops acc 89% → 31–45%', D.orange.dark, 13.5, 12);
  o += card(20, 146, 500, 106, C.lgreen, C.green, '3. Inductive Bias Crossover: Small Data vs. Foundation Scale', 'Matched ResNet leads by 5 points from scratch on small data (≈94% vs. ≈89%);', 'Pretrained ViTs overtake CNNs at 30M–300M images and provide a unified multimodal token bus.', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 62. ch11-tf-moe-switch-aux
function moeSwitchAux() {
  const W = 540, H = 268;
  let o = '';
  o += card(20, 12, 500, 72, C.lamber, C.amber, 'Switch Auxiliary Load-Balancing Loss:  L_aux = E ∑_{e=1}^{E} f_e · p̄_e', 'f_e = actual token fraction routed to expert e (stop-grad) · p̄_e = mean router probability', null, D.orange.dark, 13.5, 12);
  o += card(20, 102, 236, 146, C.lblue, C.blue, 'Gradient: ∂L_aux / ∂p̄_e = E f_e', 'Overloaded experts (f_e > 1/E)', 'receive steeper downward push!', D.blue.dark, 13.5, 12);
  o += card(284, 102, 236, 146, C.lgreen, C.green, 'Uniform Minimum: L_aux = 1', 'When f_e = 1/E for all E experts:', 'E ∑ (1/E) p̄_e = ∑ p̄_e = 1  (vs. E collapsed)', D.green.dark, 13.5, 12);
  return svg(W, H, o);
}

// 63. ch11-tf-moe-summary
function moeSummary() {
  const W = 540, H = 272;
  let o = '';
  o += card(20, 18, 236, 108, C.lblue, C.blue, '1. Top-k Conditional Compute', 'MoE(x) = ∑_{e∈T_k} p_e(x) FFN_{e}(x)', 'Decouples N_total (VRAM) from N_active', D.blue.dark, 13.5, 12);
  o += card(284, 18, 236, 108, C.lgreen, C.green, '2. Static Buffers & Balancing', 'S = ⌈γ n k / E⌉ slots per expert', 'Aux loss (α=0.01) or loss-free bias b_e', D.green.dark, 13.5, 12);
  o += card(20, 146, 500, 106, D.purple.tint, C.purple, '3. IsoFLOP Victory & Fine-Grained + Shared Expert Scaling', '8-expert Top-1 MoE beats dense GPT by 0.05 nats/byte at identical active FLOPs;', 'Splitting experts (E → mE, k → mk) + shared experts powers DeepSeek-V3 (671B / 37B active).', D.purple.dark, 13.5, 12);
  return svg(W, H, o);
}

export const diagrams = {
  'ch11-tf-why-block': whyBlock,
  'ch11-tf-four-pillars': fourPillars,
  'ch11-tf-benchmarks': benchmarks,
  'ch11-tf-modern-recipe': modernRecipe,
  'ch11-tf-blk-failure-modes': blkFailureModes,
  'ch11-tf-blk-post-vs-pre-eq': blkPostVsPreEq,
  'ch11-tf-blk-token-collapse': blkTokenCollapse,
  'ch11-tf-blk-width-pitfall': blkWidthPitfall,
  'ch11-tf-blk-unrolled-highway': blkUnrolledHighway,
  'ch11-tf-blk-qknorm-bound': blkQknormBound,
  'ch11-tf-blk-ffn-kv-memory': blkFfnKvMemory,
  'ch11-tf-blk-gelu-vs-swiglu': blkGeluVsSwiglu,
  'ch11-tf-blk-collapse-law': blkCollapseLaw,
  'ch11-tf-blk-summary': blkSummary,
  'ch11-tf-gpt-parallel-prefixes': gptParallelPrefixes,
  'ch11-tf-gpt-fused-qkv-gqa': gptFusedQkvGqa,
  'ch11-tf-gpt-learned-vs-rope': gptLearnedVsRope,
  'ch11-tf-gpt-lm-optimizer': gptLmOptimizer,
  'ch11-tf-gpt-sampling-pipeline': gptSamplingPipeline,
  'ch11-tf-gpt-checkpoint-map': gptCheckpointMap,
  'ch11-tf-gpt-frontier-evolution': gptFrontierEvolution,
  'ch11-tf-gpt-depth-init': gptDepthInit,
  'ch11-tf-gpt-summary': gptSummary,
  'ch11-tf-kv-recompute-vs-cache': kvRecomputeVsCache,
  'ch11-tf-kv-static-ring-buffer': kvStaticRingBuffer,
  'ch11-tf-kv-bandwidth-floor': kvBandwidthFloor,
  'ch11-tf-kv-three-axes': kvThreeAxes,
  'ch11-tf-kv-mla-absorption': kvMlaAbsorption,
  'ch11-tf-kv-sink-window': kvSinkWindow,
  'ch11-tf-kv-footprint-formula': kvFootprintFormula,
  'ch11-tf-kv-ttft-vs-itl': kvTtftVsItl,
  'ch11-tf-enc-bidirectional-cloze': encBidirectionalCloze,
  'ch11-tf-enc-multihop-leakage': encMultihopLeakage,
  'ch11-tf-enc-bathtub-curve': encBathtubCurve,
  'ch11-tf-enc-decoder-block': encDecoderBlock,
  'ch11-tf-enc-three-masks': encThreeMasks,
  'ch11-tf-enc-wiring-decision': encWiringDecision,
  'ch11-tf-enc-string-reversal': encStringReversal,
  'ch11-tf-enc-perceiver-flow': encPerceiverFlow,
  'ch11-tf-enc-summary': encSummary,
  'ch11-tf-vit-pixel-vs-patch': vitPixelVsPatch,
  'ch11-tf-vit-cnn-vs-vit-bias': vitCnnVsVitBias,
  'ch11-tf-vit-strided-conv-stem': vitStridedConvStem,
  'ch11-tf-vit-2d-pos-schemes': vit2dPosSchemes,
  'ch11-tf-vit-adamw-recipe': vitAdamwRecipe,
  'ch11-tf-vit-scale-crossover': vitScaleCrossover,
  'ch11-tf-vit-matched-budget': vitMatchedBudget,
  'ch11-tf-vit-summary': vitSummary,
  'ch11-tf-scale-budget-tradeoff': scaleBudgetTradeoff,
  'ch11-tf-scale-six-nd': scaleSixNd,
  'ch11-tf-scale-attn-crossover': scaleAttnCrossover,
  'ch11-tf-scale-cosine-horizon': scaleCosineHorizon,
  'ch11-tf-scale-chinchilla-surface': scaleChinchillaSurface,
  'ch11-tf-scale-kaplan-vs-chinchilla': scaleKaplanVsChinchilla,
  'ch11-tf-scale-lifetime-tco': scaleLifetimeTco,
  'ch11-tf-moe-dense-vs-sparse': moeDenseVsSparse,
  'ch11-tf-moe-router-gradient': moeRouterGradient,
  'ch11-tf-moe-capacity-buffer': moeCapacityBuffer,
  'ch11-tf-moe-collapse-loop': moeCollapseLoop,
  'ch11-tf-moe-aux-vs-bias': moeAuxVsBias,
  'ch11-tf-moe-frontier-spectrum': moeFrontierSpectrum,
  'ch11-tf-moe-switch-aux': moeSwitchAux,
  'ch11-tf-moe-summary': moeSummary,
};
