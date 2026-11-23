import { C as EC, TOKENS, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;
const C = {
  text: EC.ink,
  muted: EC.muted,
  blue:   { bg: EC.lblue,      bd: EC.blue,      tx: D.blue.dark },
  green:  { bg: EC.lgreen,     bd: EC.green,     tx: D.green.dark },
  amber:  { bg: EC.lamber,     bd: EC.amber,     tx: D.orange.dark },
  coral:  { bg: D.red.tint,    bd: D.red.base,   tx: D.red.dark },
  purple: { bg: D.purple.tint, bd: EC.purple,    tx: D.purple.dark },
  slate:  { bg: N.panel,       bd: EC.gray,      tx: EC.ink },
};

function rbox(x, y, w, h, fill, stroke, rx = 8, sw = 1.6) {
  return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;
}

// ============================================================================
// index.md (4 diagrams)
// ============================================================================

function kvVsState() {
  const W = 540, H = 256;
  let body = '';
  body += tx(W / 2, 18, 'Per-Layer Inference State: O(T) KV Cache vs. O(1) Recurrent State', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 246, 174, '#FFF8F8', C.coral.bd);
  body += tx(139, 54, 'Transformer GQA KV Cache', { fs: 12, fw: 700, fill: C.coral.tx });
  body += tx(139, 72, '8 KV heads × d=128 = 4 KB/tok', { fs: 10, fill: C.muted, mono: true });
  const kvRows = [
    ['T = 512', '2 MB / layer'],
    ['T = 8,192', '32 MB / layer'],
    ['T = 32,768', '128 MB / layer'],
    ['T = 131,072', '512 MB / layer'],
  ];
  kvRows.forEach(([lbl, val], i) => {
    const ry = 88 + i * 28;
    body += tx(26, ry + 10, lbl, { fs: 10, fw: 600, anchor: 'start', fill: C.coral.tx, mono: true });
    body += rbox(106, ry, 144, 18, C.coral.bg, C.coral.bd, 3, 1.2);
    body += tx(178, ry + 10, val, { fs: 9.5, fw: 700, fill: C.coral.tx });
  });

  body += rbox(278, 36, 246, 174, '#F5FBF7', C.green.bd);
  body += tx(401, 54, 'Mamba-2 / GatedDeltaNet State', { fs: 12, fw: 700, fill: C.green.tx });
  body += tx(401, 72, 'Fixed matrix state S_t (constant)', { fs: 10, fill: C.muted });
  const stRows = [
    ['T = 512', '2 MB (crossover)'],
    ['T = 8,192', '2 MB (16× smaller)'],
    ['T = 32,768', '2 MB (64× smaller)'],
    ['T = 131,072', '2 MB (256× smaller)'],
  ];
  stRows.forEach(([lbl, val], i) => {
    const ry = 88 + i * 28;
    body += tx(288, ry + 10, lbl, { fs: 10, fw: 600, anchor: 'start', fill: C.green.tx, mono: true });
    body += rbox(368, ry, 144, 18, C.green.bg, C.green.bd, 3, 1.2);
    body += tx(440, ry + 10, val, { fs: 9.5, fw: 700, fill: C.green.tx });
  });

  body += rbox(24, 220, 492, 26, C.amber.bg, C.amber.bd, 6);
  body += tx(W / 2, 233, 'Crossover at T = 512 tokens: fixed 2 MB state gives O(1) memory per decoded token', { fs: 10.5, fw: 700, fill: C.amber.tx });
  return svg(W, H, body);
}

function twoObstacles() {
  const W = 540, H = 256;
  let body = '';
  body += tx(W / 2, 18, 'Two Classical Obstacles and Their Modern Solutions', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 246, 94, C.coral.bg, C.coral.bd);
  body += tx(139, 54, 'Obstacle 1: Sequential Training', { fs: 12, fw: 700, fill: C.coral.tx });
  body += tx(139, 74, 'h_t = tanh(W_xh x_t + W_hh h_{t-1})', { fs: 10, fill: C.text, mono: true });
  body += tx(139, 94, 'Nonlinearity blocks GPU parallelism', { fs: 10, fill: C.muted });
  body += tx(139, 112, 'O(T) sequential kernel launches', { fs: 10, fw: 700, fill: C.coral.tx });

  body += rbox(278, 36, 246, 94, C.coral.bg, C.coral.bd);
  body += tx(401, 54, 'Obstacle 2: Fixed-Vector Decay', { fs: 12, fw: 700, fill: C.coral.tx });
  body += tx(401, 74, 'Single vector h_t ∈ R^d overwrites', { fs: 10, fill: C.text, mono: true });
  body += tx(401, 94, 'Uniform decay forgets early facts', { fs: 10, fill: C.muted });
  body += tx(401, 112, 'Cannot rebind or look up 64+ keys', { fs: 10, fw: 700, fill: C.coral.tx });

  body += arrow(139, 132, 139, 150);
  body += arrow(401, 132, 401, 150);

  body += rbox(16, 152, 246, 92, C.green.bg, C.green.bd);
  body += tx(139, 170, 'Solution 1: Linear Scan & Chunks', { fs: 12, fw: 700, fill: C.green.tx });
  body += tx(139, 190, 'h_t = a_t ⊙ h_{t-1} + b_t (§12.3–12.5)', { fs: 10, fill: C.text, mono: true });
  body += tx(139, 210, 'Associative scan in O(log₂ T) depth', { fs: 10, fill: C.muted });
  body += tx(139, 228, 'Chunked Tensor-Core GEMMs (§12.6–12.7)', { fs: 10, fw: 700, fill: C.green.tx });

  body += rbox(278, 152, 246, 92, C.blue.bg, C.blue.bd);
  body += tx(401, 170, 'Solution 2: Structured Matrix State', { fs: 12, fw: 700, fill: C.blue.tx });
  body += tx(401, 190, 'Multi-scale Δ_t + S_t ∈ R^{d_v×d_k}', { fs: 10, fill: C.text, mono: true });
  body += tx(401, 210, 'Delta rule erases old key value (§12.7)', { fs: 10, fill: C.muted });
  body += tx(401, 228, '3:1 Hybrid adds exact lookup (§12.8)', { fs: 10, fw: 700, fill: C.blue.tx });
  return svg(W, H, body);
}

function roadmap() {
  const W = 540, H = 256;
  let body = '';
  body += tx(W / 2, 18, 'Chapter 12 Progression: From Classical Gates to Hybrid Blocks', { fs: 13.5, fw: 700 });

  const row1 = [
    ['§12.1 LSTM/GRU', 'Gated vector h_t', 'Sequential O(T)', C.slate],
    ['§12.2 Seq2Seq', 'Encoder → c → Dec', 'Vector bottleneck', C.slate],
    ['§12.3 minGRU', 'h_t = a_t⊙h + b_t', 'Parallel scan', C.blue],
    ['§12.4 S4D SSM', 'N real decay rates', 'O(T log T) FFT', C.blue],
  ];
  row1.forEach(([title, sub1, sub2, col], i) => {
    const x = 14 + i * 130;
    body += rbox(x, 38, 118, 78, col.bg, col.bd);
    body += tx(x + 59, 58, title, { fs: 11, fw: 700, fill: col.tx });
    body += tx(x + 59, 78, sub1, { fs: 10, fill: C.text, mono: true });
    body += tx(x + 59, 98, sub2, { fs: 9.5, fill: C.muted });
    if (i < 3) body += arrow(x + 119, 77, x + 129, 77);
  });

  const row2 = [
    ['§12.5 Mamba', 'Token Δ_t, B_t, C_t', 'Adjoint 4D scan', C.purple],
    ['§12.6 GLA Matrix', 'S_t = a_t S + v kᵀ', 'Chunked SSD dual', C.purple],
    ['§12.7 DeltaNet', 'Read-before-write', 'WY lower-tri solve', C.green],
    ['§12.8 Hybrids', '3:1 Recurrent+Attn', 'Welch bound escape', C.amber],
  ];
  row2.forEach(([title, sub1, sub2, col], i) => {
    const x = 14 + i * 130;
    body += rbox(x, 148, 118, 78, col.bg, col.bd);
    body += tx(x + 59, 168, title, { fs: 11, fw: 700, fill: col.tx });
    body += tx(x + 59, 188, sub1, { fs: 10, fill: C.text, mono: true });
    body += tx(x + 59, 208, sub2, { fs: 9.5, fill: C.muted });
    if (i < 3) body += arrow(x + 119, 187, x + 129, 187);
  });

  body += arrow(270, 118, 270, 146);
  body += tx(W / 2, 242, 'Every mixer plugs into the same 4-block pre-norm SeqLM backbone on WikiText-103', { fs: 10.5, fw: 600, fill: C.muted });
  return svg(W, H, body);
}

function stressTests() {
  const W = 540, H = 256;
  let body = '';
  body += tx(W / 2, 18, 'Five Controlled Capability Probes + One Language Benchmark', { fs: 13.5, fw: 700 });

  const cards = [
    ['1. Delayed Recall', 'Remember 1st token', 'after L filler steps', 'Tests gate half-life', C.blue],
    ['2. String Reversal', 'Reverse length-T', 'random 16-char string', 'Tests c bit capacity', C.purple],
    ['3. Selective Copy', 'Copy 4 marked symbols', 'ignore variable noise', 'Tests token Δ_t gate', C.green],
    ['4. Assoc. Recall', 'Store m (k_i, v_i) pairs', 'query k_q at end', 'Tests d_v×d_k matrix', C.amber],
    ['5. Key Rebinding', 'Update existing key k', 'with new value v_new', 'Tests delta erasure', C.coral],
    ['6. WikiText-103 LM', '4-block SeqLM stack', 'd=256, byte-level', 'Tests nats/byte & tok/s', C.slate],
  ];
  cards.forEach(([title, l1, l2, tag, col], idx) => {
    const colIdx = idx % 3;
    const rowIdx = Math.floor(idx / 3);
    const x = 14 + colIdx * 174;
    const y = 36 + rowIdx * 108;
    body += rbox(x, y, 164, 96, col.bg, col.bd);
    body += tx(x + 82, y + 20, title, { fs: 11.5, fw: 700, fill: col.tx });
    body += tx(x + 82, y + 40, l1, { fs: 10, fill: C.text });
    body += tx(x + 82, y + 56, l2, { fs: 10, fill: C.text });
    body += rbox(x + 12, y + 68, 140, 20, '#FFFFFF', col.bd, 4, 1.0);
    body += tx(x + 82, y + 78, tag, { fs: 9.5, fw: 700, fill: col.tx });
  });
  return svg(W, H, body);
}

// ============================================================================
// lstm.md (8 diagrams)
// ============================================================================

function grammarDecay() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Long-Range Grammatical Agreement vs. Exponential Signal Decay', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 64, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 56, '"The keys to the cabinet in the back office ... are on the table."', { fs: 11.5, fw: 700, fill: C.blue.tx });
  body += tx(W / 2, 78, 'Plural subject "keys" must govern verb "are" across 30+ intervening tokens', { fs: 10.5, fill: C.text });

  const bars = [
    ['Lag k = 1', '0.9¹ = 90.0%', 220, C.green],
    ['Lag k = 10', '0.9¹⁰ = 34.9%', 85, C.blue],
    ['Lag k = 30', '0.9³⁰ = 4.2%', 10, C.amber],
    ['Lag k = 100', '0.9¹⁰⁰ = 0.003%', 2, C.coral],
  ];
  bars.forEach(([lbl, val, bw, col], i) => {
    const y = 114 + i * 30;
    body += tx(24, y + 11, lbl, { fs: 11, fw: 700, anchor: 'start', fill: C.text, mono: true });
    body += rbox(126, y, bw, 20, col.bg, col.bd, 4, 1.4);
    body += tx(136 + bw, y + 11, val, { fs: 11, fw: 700, anchor: 'start', fill: col.tx, mono: true });
  });
  return svg(W, H, body);
}

function jacobianChain() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Backward Jacobian Product Chain: ∂h_t / ∂h_{t-k} = ∏ J_s', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 52, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 54, 'Step Jacobian: J_s = diag(1 − h_s²) W_hh   with   ||diag(1 − h_s²)||₂ ≤ 1', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(W / 2, 74, 'Product norm after k steps scales as (σ_max(J))^k across lag k = t − s', { fs: 10.5, fill: C.muted });

  const regimes = [
    ['Contractive (σ < 1)', 'σ=0.85 → 0.85⁵⁰ = 3×10⁻⁴', 'Vanishing long-range grad', 'Recent steps dominate', C.coral],
    ['Borderline (σ ≈ 1)', 'Needs h_s ≈ 0 (linear)', 'Saturating tanh kills 1−h_s²', 'Fragile under SGD updates', C.amber],
    ['Expansive (σ > 1)', 'σ=1.15 → 1.15⁵⁰ = 1083×', 'Exploding gradient norm', 'Clipping rescales all lags', C.purple],
  ];
  regimes.forEach(([title, l1, l2, l3, col], i) => {
    const x = 16 + i * 172;
    body += rbox(x, 102, 164, 136, col.bg, col.bd);
    body += tx(x + 82, 122, title, { fs: 11, fw: 700, fill: col.tx });
    body += tx(x + 82, 146, l1, { fs: 9.5, fw: 600, fill: C.text, mono: true });
    body += tx(x + 82, 172, l2, { fs: 9.5, fill: C.text });
    body += tx(x + 82, 194, l3, { fs: 9.5, fill: C.muted });
  });
  return svg(W, H, body);
}

function gateHalflife() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Gated Linear Memory: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 46, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 52, 'Cell Jacobian: ∂c_t / ∂c_{t-1} = diag(f_t)   (no W_hh matrix product!)', { fs: 10.5, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(W / 2, 70, 'Half-life formula (steps to decay to 50%):   τ_{1/2} = ln 2 / ln(1 / f) ≈ 0.693 / (1 − f)', { fs: 10, fill: C.text });

  const rows = [
    ['f = 0.50 (b_f = 0)', 'τ_{1/2} = 1.0 step', C.coral],
    ['f = 0.90 (b_f = 2.2)', 'τ_{1/2} = 6.6 steps', C.amber],
    ['f = 0.99 (b_f = 4.6)', 'τ_{1/2} = 69 steps', C.blue],
    ['f = 0.999 (b_f = 6.9)', 'τ_{1/2} = 693 steps', C.green],
  ];
  rows.forEach(([lbl, val, col], i) => {
    const y = 96 + i * 36;
    body += tx(24, y + 13, lbl, { fs: 10.5, fw: 700, anchor: 'start', fill: C.text, mono: true });
    body += rbox(196, y, 240, 24, col.bg, col.bd, 5, 1.4);
    body += tx(316, y + 13, val, { fs: 10.5, fw: 700, fill: col.tx });
  });
  return svg(W, H, body);
}

function silentStorage() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Decoupling Internal Cell State c_t from Exposed Hidden State h_t', { fs: 13.5, fw: 700 });

  body += rbox(18, 44, 144, 82, C.blue.bg, C.blue.bd);
  body += tx(90, 66, 'Previous Cell c_{t-1}', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(90, 88, 'c_{t-1} = 0.80', { fs: 11, fw: 700, fill: C.text, mono: true });
  body += tx(90, 108, 'Stored long-range fact', { fs: 9.5, fill: C.muted });

  body += arrow(164, 85, 198, 85);

  body += rbox(200, 44, 152, 82, C.green.bg, C.green.bd);
  body += tx(276, 66, 'Protected Cell Update', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(276, 86, 'f_t = 0.95, i_t = 0.10', { fs: 10, fill: C.text, mono: true });
  body += tx(276, 108, 'c_t = 0.71 (retained!)', { fs: 10.5, fw: 700, fill: C.green.tx, mono: true });

  body += arrow(354, 85, 388, 85);

  body += rbox(390, 44, 134, 82, C.amber.bg, C.amber.bd);
  body += tx(457, 66, 'Closed Output Gate', { fs: 11, fw: 700, fill: C.amber.tx });
  body += tx(457, 86, 'o_t = 0.20 (muted)', { fs: 10, fill: C.text, mono: true });
  body += tx(457, 108, 'h_t = 0.12 (muted)', { fs: 10, fw: 700, fill: C.amber.tx, mono: true });

  body += rbox(18, 146, 506, 90, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 168, 'Why Separate c_t and h_t?', { fs: 12, fw: 700, fill: C.text });
  body += tx(W / 2, 190, '1. When o_t ≈ 0, stored coordinates in c_t do not perturb downstream layers at step t', { fs: 10, fill: C.text });
  body += tx(W / 2, 212, '2. Immediate ∂L_t / ∂c_t = o_t(1 − tanh² c_t)(∂L_t / ∂h_t) is muted until o_{t+k} opens', { fs: 10, fill: C.blue.tx });
  return svg(W, H, body);
}

function lstmVsGru() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Structural Comparison: LSTM (1997) vs. GRU (2014)', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 198, C.blue.bg, C.blue.bd);
  body += tx(139, 58, 'LSTM Cell (Hochreiter & Schmidhuber)', { fs: 11, fw: 700, fill: C.blue.tx });
  body += rbox(30, 72, 218, 34, '#FFFFFF', C.blue.bd, 5, 1.1);
  body += tx(139, 90, '2 States: Cell c_t + Hidden h_t', { fs: 10, fw: 700, fill: C.blue.tx });
  body += tx(139, 124, '• 3 independent gates: f_t, i_t, o_t', { fs: 10, fill: C.text });
  body += tx(139, 146, '• c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t', { fs: 10, fill: C.text, mono: true });
  body += tx(139, 168, '• Can saturate f_t ≈ 1 and i_t ≈ 1', { fs: 10, fill: C.muted });
  body += rbox(30, 188, 218, 34, C.slate.bg, C.slate.bd, 5, 1.1);
  body += tx(139, 206, '4 weight blocks: 4d(p + d) ≈ 8d²', { fs: 9.5, fw: 700, fill: C.text, mono: true });

  body += rbox(278, 38, 246, 198, C.green.bg, C.green.bd);
  body += tx(401, 58, 'GRU Cell (Cho et al.)', { fs: 11, fw: 700, fill: C.green.tx });
  body += rbox(292, 72, 218, 34, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(401, 90, '1 State: Exposed Hidden h_t only', { fs: 10, fw: 700, fill: C.green.tx });
  body += tx(401, 124, '• 2 gates: keep z_t, reset r_t', { fs: 10, fill: C.text });
  body += tx(401, 146, '• h_t = z_t⊙h_{t-1} + (1−z_t)⊙h̃_t', { fs: 10, fill: C.text, mono: true });
  body += tx(401, 168, '• Convex keep/write: h_t ∈ [-1, 1]^d', { fs: 10, fill: C.muted });
  body += rbox(292, 188, 218, 34, C.slate.bg, C.slate.bd, 5, 1.1);
  body += tx(401, 206, '3 weight blocks: 3d(p + d) ≈ 6d²', { fs: 9.5, fw: 700, fill: C.text, mono: true });
  return svg(W, H, body);
}

function zeroBiasUnderflow() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Pitfall: Zero Gate Bias Starts with a 1-Step Memory Horizon', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 48, C.coral.bg, C.coral.bd);
  body += tx(W / 2, 54, 'At initialization W ≈ 0, b_f = 0  ⇒  f_t = σ(0) = 0.5  (τ_{1/2} = 1 step!)', { fs: 11, fw: 700, fill: C.coral.tx, mono: true });
  body += tx(W / 2, 72, 'After T − 1 steps, gradient ∂c_T / ∂c_1 = (0.5)^{T-1} vanishes before learning starts', { fs: 10, fill: C.text });

  const cards = [
    ['Lag T = 10', '2⁻⁹ = 1.95 × 10⁻³', '0.2% gradient left', C.amber],
    ['Lag T = 20', '2⁻¹⁹ = 1.91 × 10⁻⁶', 'Near noise floor', C.coral],
    ['Lag T = 50', '2⁻⁴⁹ = 1.78 × 10⁻¹⁵', 'Swamped below float32 ε', C.coral],
  ];
  cards.forEach(([title, math, sub, col], i) => {
    const x = 16 + i * 172;
    body += rbox(x, 98, 164, 86, col.bg, col.bd);
    body += tx(x + 82, 120, title, { fs: 11.5, fw: 700, fill: col.tx });
    body += tx(x + 82, 144, math, { fs: 10.5, fw: 700, fill: C.text, mono: true });
    body += tx(x + 82, 166, sub, { fs: 10, fill: C.muted });
  });

  body += rbox(16, 198, 508, 40, C.green.bg, C.green.bd);
  body += tx(W / 2, 218, 'Fix: Initialize b_f > 0 (or log-uniform in [1, T_max]) so gates start open at Step 0', { fs: 10.5, fw: 700, fill: C.green.tx });
  return svg(W, H, body);
}

function keepGateInit() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Keep-Gate Initialization & Cell-Magnitude Bounding', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 146, C.coral.bg, C.coral.bd);
  body += tx(139, 58, 'Unpaired Forget-Only Bias', { fs: 11.5, fw: 700, fill: C.coral.tx });
  body += tx(139, 80, 'b_f = +5 (f ≈ 0.993), b_i = 0', { fs: 10, fill: C.text, mono: true });
  body += tx(139, 104, 'Steady-state cell bound:', { fs: 10, fill: C.muted });
  body += tx(139, 126, '|c_t| ≤ i / (1 − f) ≈ 75', { fs: 10.5, fw: 700, fill: C.coral.tx, mono: true });
  body += tx(139, 152, 'Saturates tanh(c_t) & kills ∂h_t/∂c_t!', { fs: 10, fw: 700, fill: C.coral.tx });

  body += rbox(278, 38, 246, 146, C.green.bg, C.green.bd);
  body += tx(401, 58, 'Paired Keep/Write (init_keep_gate)', { fs: 11.5, fw: 700, fill: C.green.tx });
  body += tx(401, 80, 'b_f = +5,  b_i = −5  (i = 1 − f)', { fs: 10, fill: C.text, mono: true });
  body += tx(401, 104, 'Matches GRU convex combination:', { fs: 10, fill: C.muted });
  body += tx(401, 126, 'i = σ(−b_f) = 1 − f  ⇒  |c_t| ≤ 1', { fs: 10.5, fw: 700, fill: C.green.tx, mono: true });
  body += tx(401, 152, 'τ_{1/2} ≈ 103 steps without saturation!', { fs: 10, fw: 700, fill: C.green.tx });

  body += rbox(16, 198, 508, 40, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 218, 'Delayed Recall (T = 50–200): zero-bias GRU stalls at chance (0.12); b_z = +5 reaches 100%', { fs: 10.5, fw: 700, fill: C.blue.tx });
  return svg(W, H, body);
}

function seqlmBackbone() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Shared Pre-Norm Residual Language Model Backbone (SeqLM)', { fs: 13.5, fw: 700 });

  const stages = [
    ['Byte Tokens', 'x_{1:T}', 'Embed d=256', C.slate, 14, 82],
    ['Sub-layer 1', 'H ← H +', 'Mixer(LN(H))', C.blue, 108, 116],
    ['Sub-layer 2', 'H ← H +', 'MLP_{4d}(LN(H))', C.purple, 236, 118],
    ['Repeat ×4', '4 Residual', 'Blocks', C.green, 366, 76],
    ['LM Head', 'LN(H) W_out', 'Nats / Byte', C.amber, 454, 72],
  ];
  stages.forEach(([title, l1, l2, col, x, w], i) => {
    body += rbox(x, 46, w, 86, col.bg, col.bd);
    body += tx(x + w / 2, 68, title, { fs: 10.5, fw: 700, fill: col.tx });
    body += tx(x + w / 2, 90, l1, { fs: 9.5, fill: C.text, mono: true });
    body += tx(x + w / 2, 110, l2, { fs: 9.5, fw: 600, fill: col.tx, mono: true });
    if (i < stages.length - 1) {
      const nx = stages[i + 1][4];
      body += arrow(x + w + 2, 89, nx - 2, 89);
    }
  });

  body += rbox(16, 150, 508, 82, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 170, 'Apples-to-Apples Mixer Benchmark Protocol across §§12.1–12.8', { fs: 11.5, fw: 700, fill: C.text });
  body += tx(W / 2, 192, '• Only Mixer(·) changes; on text: RNN, LSTM, GRU, minGRU, Mamba, Gated DeltaNet, attention', { fs: 9.5, fill: C.text });
  body += tx(W / 2, 212, '• Residual highway carries ∂H_L/∂H_l = I + ... while the mixer supplies causal cross-time transport', { fs: 9.5, fill: C.blue.tx });
  return svg(W, H, body);
}

// ============================================================================
// seq2seq.md (8 diagrams)
// ============================================================================

function transductionBottleneck() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Sequence-to-Sequence Transduction via a Fixed-Size Context Vector c', { fs: 13.5, fw: 700 });

  body += rbox(16, 46, 156, 86, C.blue.bg, C.blue.bd);
  body += tx(94, 68, 'Source Stream x_{1:T}', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(94, 88, '"Look" "at" "us" <eos>', { fs: 9.5, fill: C.text, mono: true });
  body += tx(94, 110, 'Length T = 4 tokens', { fs: 9.5, fill: C.muted });

  body += arrow(174, 89, 204, 89);

  body += rbox(206, 46, 128, 86, C.amber.bg, C.amber.bd);
  body += tx(270, 68, 'Context Bottleneck', { fs: 11, fw: 700, fill: C.amber.tx });
  body += tx(270, 90, 'c = h_T ∈ R^d', { fs: 11, fw: 700, fill: C.text, mono: true });
  body += tx(270, 110, 'Fixed d floats for ANY T', { fs: 9.5, fill: C.amber.tx });

  body += arrow(336, 89, 366, 89);

  body += rbox(368, 46, 156, 86, C.green.bg, C.green.bd);
  body += tx(446, 68, 'Target Stream y_{1:T\'}', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(446, 88, '"Regardez-nous" "." <eos>', { fs: 9.5, fill: C.text, mono: true });
  body += tx(446, 110, 'Length T\' = 3 (T\' ≠ T)', { fs: 9.5, fill: C.muted });

  body += rbox(16, 150, 508, 82, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 170, 'Why Single-Stream Language Modeling Is Not Enough', { fs: 11.5, fw: 700, fill: C.text });
  body += tx(W / 2, 192, '• Source length T and target length T\' differ with non-monotonic word reordering', { fs: 10, fill: C.text });
  body += tx(W / 2, 212, '• Entire source meaning must survive inside one vector c = h_T before y_1 is emitted', { fs: 10, fill: C.coral.tx });
  return svg(W, H, body);
}

function conditionedDecoder() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Conditioned Autoregressive Decoder: Training vs. Inference', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 44, C.purple.bg, C.purple.bd);
  body += tx(W / 2, 53, 'P(y | x) = ∏ P(y_{t\'} | y_{<t\'}, c),   s_{t\'} = f([e(y_{t\'-1}); c], s_{t\'-1})', { fs: 9.5, fw: 700, fill: C.purple.tx, mono: true });
  body += tx(W / 2, 70, 'Injecting c at BOTH s_0 = c and every input [e(y_{t\'-1}); c] shortens gradient path ∂L_{t\'}/∂c to 1 hop', { fs: 9.5, fill: C.text });

  body += rbox(16, 94, 246, 142, C.blue.bg, C.blue.bd);
  body += tx(139, 114, 'Training: Teacher Forcing', { fs: 11.5, fw: 700, fill: C.blue.tx });
  body += tx(139, 138, 'Feeds ground-truth prefix:', { fs: 10, fill: C.muted });
  body += tx(139, 158, '[<bos>, y_1, ..., y_{T\'-1}]', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(139, 182, '• All T\' target losses in 1 pass', { fs: 10, fill: C.text });
  body += tx(139, 202, '• Masks <pad> tokens in loss', { fs: 10, fill: C.text });

  body += rbox(278, 94, 246, 142, C.green.bg, C.green.bd);
  body += tx(401, 114, 'Inference: Autoregressive Rollout', { fs: 11.5, fw: 700, fill: C.green.tx });
  body += tx(401, 138, 'Feeds own previous prediction:', { fs: 10, fill: C.muted });
  body += tx(401, 158, 'ŷ_{t\'} = argmax P(y | ŷ_{<t\'}, c)', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(401, 182, '• Starts from <bos> and s_0 = c', { fs: 10, fill: C.text });
  body += tx(401, 202, '• Halts as soon as ŷ_{t\'} == <eos>', { fs: 10, fill: C.text });
  return svg(W, H, body);
}

function sutskeverReversal() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Why Reversing the Source Sentence Boosted 2014 NMT (Sutskever et al.)', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 508, 86, C.coral.bg, C.coral.bd);
  body += tx(W / 2, 56, 'Forward Source Order:  x_1  →  x_2  →  x_3  → [ c ] →  y_1  →  y_2  →  y_3', { fs: 11, fw: 700, fill: C.coral.tx, mono: true });
  body += tx(W / 2, 78, 'First word x_1 is T steps away from first output y_1 (average lag = T + 1 steps)', { fs: 10, fill: C.text });
  body += tx(W / 2, 98, 'Cho et al. (2014): BLEU collapses when source length T exceeds 20–30 words', { fs: 10, fw: 600, fill: C.coral.tx });

  body += rbox(16, 138, 508, 98, C.green.bg, C.green.bd);
  body += tx(W / 2, 158, 'Reversed Source Order: x_3  →  x_2  →  x_1  → [ c ] →  y_1  →  y_2  →  y_3', { fs: 11, fw: 700, fill: C.green.tx, mono: true });
  body += tx(W / 2, 180, 'x_1 sits immediately adjacent to c and y_1 (lag = 2 steps instead of T + 1!)', { fs: 10, fill: C.text });
  body += tx(W / 2, 200, 'Early target predictions y_1, y_2 anchor on fresh x_1, x_2 before error compounds', { fs: 10, fill: C.text });
  body += tx(W / 2, 220, 'An ensemble of five 4-layer LSTMs reached 34.8 BLEU on WMT\'14 English→French', { fs: 10, fw: 700, fill: C.green.tx });
  return svg(W, H, body);
}

function chrfPipeline() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Character N-Gram F-Score (chrF) Evaluation Pipeline', { fs: 13.5, fw: 700 });

  const steps = [
    ['1. Strip Spaces', 'pred, ref strings', 'without whitespace', C.slate, 14, 118],
    ['2. Char n-grams', 'Extract all n-grams', 'for n = 1, 2, ..., 6', C.blue, 146, 122],
    ['3. Clipped Overlap', 'Count(p & r) divides', 'by |p| (P_n) or |r| (R_n)', C.purple, 282, 126],
    ['4. Recall F_{β=2}', '(1+β²)PR/(β²P+R)', 'weights recall 4×', C.green, 422, 104],
  ];
  steps.forEach(([title, l1, l2, col, x, w], i) => {
    body += rbox(x, 42, w, 92, col.bg, col.bd);
    body += tx(x + w / 2, 64, title, { fs: 10.5, fw: 700, fill: col.tx });
    body += tx(x + w / 2, 86, l1, { fs: 9.5, fill: C.text, mono: true });
    body += tx(x + w / 2, 106, l2, { fs: 9.5, fill: C.muted });
    if (i < steps.length - 1) {
      const nx = steps[i + 1][4];
      body += arrow(x + w + 2, 88, nx - 2, 88);
    }
  });

  body += rbox(16, 152, 508, 80, C.amber.bg, C.amber.bd);
  body += tx(W / 2, 172, 'Why chrF Is Preferred Over Word BLEU for Modern Sequence Models', { fs: 11.5, fw: 700, fill: C.amber.tx });
  body += tx(W / 2, 194, '• Rewards partial stem/morpheme matches ("running" vs "runner") even when exact words differ', { fs: 10, fill: C.text });
  body += tx(W / 2, 214, '• Tokenization-agnostic and non-zero on short sentences without arbitrary smoothing hacks', { fs: 10, fill: C.text });
  return svg(W, H, body);
}

function bleuFormula() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'BLEU Score Decomposition: Brevity Penalty × Geometric Precision', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 114, C.blue.bg, C.blue.bd);
  body += tx(139, 58, 'Factor 1: Brevity Penalty (BP)', { fs: 11.5, fw: 700, fill: C.blue.tx });
  body += tx(139, 82, 'exp(min(0, 1 − l_ref / l_cand))', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(139, 106, 'Penalizes overly short outputs', { fs: 10, fill: C.text });
  body += tx(139, 126, 'that game 1-gram precision p_1 = 1', { fs: 10, fill: C.muted });

  body += rbox(278, 38, 246, 114, C.purple.bg, C.purple.bd);
  body += tx(401, 58, 'Factor 2: Clipped N-Gram Product', { fs: 11.5, fw: 700, fill: C.purple.tx });
  body += tx(401, 82, '∏ (p_n)^{1/2ⁿ}   for n = 1..k', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(401, 106, 'p_1: lexical adequacy; p_4: fluency', { fs: 10, fill: C.text });
  body += tx(401, 126, 'Clips counts at max reference count', { fs: 10, fill: C.muted });

  body += rbox(16, 166, 508, 68, C.coral.bg, C.coral.bd);
  body += tx(W / 2, 186, 'Sensitivity Pitfall: Any Zero Higher-Order Match Collapses Sentence BLEU to 0', { fs: 11, fw: 700, fill: C.coral.tx });
  body += tx(W / 2, 208, 'If p_4 = 0 (no exact 4-word match), ∏ p_n^{w_n} = 0 regardless of how high p_1, p_2, p_3 are!', { fs: 10, fill: C.text });
  return svg(W, H, body);
}

function reversalTask() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Controlled Bottleneck Benchmark: Random String Reversal', { fs: 13.5, fw: 700 });

  body += rbox(16, 42, 162, 86, C.blue.bg, C.blue.bd);
  body += tx(97, 64, 'Uniform Random Input', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(97, 86, 'x = "d k a p m c" <eos>', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(97, 108, '16 letters = 4 bits / char', { fs: 9.5, fill: C.muted });

  body += arrow(180, 85, 208, 85);

  body += rbox(210, 42, 120, 86, C.amber.bg, C.amber.bd);
  body += tx(270, 64, 'Context Vector', { fs: 11, fw: 700, fill: C.amber.tx });
  body += tx(270, 86, 'c = h_T ∈ R^d', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(270, 108, 'Must pack 4T bits', { fs: 9.5, fill: C.amber.tx });

  body += arrow(332, 85, 360, 85);

  body += rbox(362, 42, 162, 86, C.green.bg, C.green.bd);
  body += tx(443, 64, 'Exact Reversed Target', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(443, 86, 'y = "c m p a k d" <eos>', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(443, 108, 'First char x_1 emitted LAST', { fs: 9.5, fill: C.muted });

  body += rbox(16, 144, 508, 88, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 164, 'Why Random String Reversal Isolates Bottleneck Capacity', { fs: 11.5, fw: 700, fill: C.text });
  body += tx(W / 2, 186, '• Zero LM shortcut: i.i.d. uniform letters prevent guessing y_{t\'} from y_{<t\'}', { fs: 10, fill: C.text });
  body += tx(W / 2, 208, '• Worst-case lag for x_1: encoded at step 1, decoded at step 2T (survives 2T updates)', { fs: 10, fill: C.coral.tx });
  return svg(W, H, body);
}

function bitBudgetBars() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Theoretical Float32 Bit Budget vs. Practical GRU Packing Limit', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 46, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 53, 'Raw Hardware Bit Budget (d = 64 float32):   64 × 32 bits = 2,048 bits', { fs: 11, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(W / 2, 71, 'At just 1 bit per float (T = 16, 64 bits), d = 64 already loses a quarter of the letters', { fs: 10, fill: C.text });

  const rows = [
    ['T = 8  (32 bits)', 'd=64: 0.99 | d=256: 1.00 acc', C.green],
    ['T = 16 (64 bits)', 'd=64: 0.73 | d=256: 0.99 acc', C.blue],
    ['T = 32 (128 bits)', 'd=64: 0.41 | d=256: 0.80 acc', C.amber],
    ['T = 64 (256 bits)', 'd=64: 0.26 | d=256: 0.51 acc', C.coral],
  ];
  rows.forEach(([lbl, val, col], i) => {
    const y = 96 + i * 36;
    body += tx(22, y + 13, lbl, { fs: 10.5, fw: 700, anchor: 'start', fill: C.text, mono: true });
    body += rbox(170, y, 344, 24, col.bg, col.bd, 4, 1.3);
    body += tx(342, y + 13, val, { fs: 10, fw: 700, fill: col.tx, mono: true });
  });
  return svg(W, H, body);
}

function twoEscapeRoutes() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Two Architectural Escape Routes from the Single-Vector Bottleneck', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 198, C.amber.bg, C.amber.bd);
  body += tx(139, 58, 'Route 1: Cross-Attention', { fs: 11.5, fw: 700, fill: C.amber.tx });
  body += rbox(30, 74, 218, 38, '#FFFFFF', C.amber.bd, 5, 1.1);
  body += tx(139, 94, 'Keep ALL T states [h_1, ..., h_T]', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(139, 130, '• Dynamic query c_{t\'} = ∑ α_{t\',s} h_s', { fs: 10, fill: C.text });
  body += tx(139, 152, '• Exact random-access lookup', { fs: 10, fill: C.green.tx });
  body += tx(139, 174, '• O(T) KV cache memory cost', { fs: 10, fill: C.coral.tx });
  body += tx(139, 196, '• O(T) compute per decoded token', { fs: 10, fill: C.coral.tx });

  body += rbox(278, 38, 246, 198, C.blue.bg, C.blue.bd);
  body += tx(401, 58, 'Route 2: Structured State (§§12.3–12.8)', { fs: 11.5, fw: 700, fill: C.blue.tx });
  body += rbox(292, 74, 218, 38, '#FFFFFF', C.blue.bd, 5, 1.1);
  body += tx(401, 94, 'Upgrade O(1) State Geometry & Rule', { fs: 10, fw: 700, fill: C.text });
  body += tx(401, 130, '• §§12.3–12.5: Linear scan + Δ_t', { fs: 10, fill: C.text });
  body += tx(401, 152, '• §12.6: Matrix state S_t ∈ R^{d_v×d_k}', { fs: 10, fill: C.text });
  body += tx(401, 174, '• §12.7: Delta rule key overwrite', { fs: 10, fill: C.text });
  body += tx(401, 196, '• §12.8: 3:1 Hybrid with attention', { fs: 10, fw: 700, fill: C.blue.tx });
  return svg(W, H, body);
}

// ============================================================================
// linear-recurrence.md (7 diagrams)
// ============================================================================

function seqVsParallelGpu() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Why Nonlinear Recurrence Underutilizes GPUs during Training', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 194, C.coral.bg, C.coral.bd);
  body += tx(139, 58, 'Nonlinear GRU: O(T) Serial Depth', { fs: 11, fw: 700, fill: C.coral.tx });
  body += tx(139, 80, 'h_t = (1−z_t)⊙tanh(W h_{t-1}) + ...', { fs: 9.5, fill: C.text, mono: true });
  body += rbox(32, 96, 214, 44, '#FFFFFF', C.coral.bd, 5, 1.1);
  body += tx(139, 114, 'T = 4,096 ⇒ 4,096 serial steps', { fs: 10, fw: 700, fill: C.coral.tx });
  body += tx(139, 130, 'Step t waits for h_{t-1} inside tanh', { fs: 9.5, fill: C.muted });
  body += tx(139, 162, '• Thousands of tiny kernel launches', { fs: 10, fill: C.text });
  body += tx(139, 182, '• Low arithmetic intensity (HBM-bound)', { fs: 10, fill: C.text });

  body += rbox(278, 38, 246, 194, C.green.bg, C.green.bd);
  body += tx(401, 58, 'Linear Recurrence: O(log₂ T) Scan', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(401, 80, 'h_t = a_t ⊙ h_{t-1} + b_t', { fs: 10, fill: C.text, mono: true });
  body += rbox(294, 96, 214, 44, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(401, 114, 'T = 4,096 ⇒ log₂ T = 12 rounds!', { fs: 10, fw: 700, fill: C.green.tx });
  body += tx(401, 130, '340× reduction in sequential depth', { fs: 9.5, fill: C.muted });
  body += tx(401, 162, '• Gates a_{1:T}, b_{1:T} fused in 1 GEMM', { fs: 10, fill: C.text });
  body += tx(401, 182, '• Keeps O(1) state per step at decode', { fs: 10, fill: C.text });
  return svg(W, H, body);
}

function scalarEmaImpulse() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Scalar Linear Recurrence: Closed-Form Moving Sum & Gradient', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 48, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 53, 'Recurrence: h_t = a h_{t-1} + x_t    ⇔    Closed Form: h_t = ∑_{s=1}^{t} a^{t-s} x_s', { fs: 10.5, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(W / 2, 71, 'Exact Backward Sensitivity:  ∂h_t / ∂x_s = a^{t-s}   (identical to impulse response w_{t-s}!)', { fs: 10, fill: C.text });

  const lags = [
    ['s = t (lag 0)', 'w_0 = 1.00', 92, C.green],
    ['s = t−1 (lag 1)', 'w_1 = 0.85', 78, C.blue],
    ['s = t−2 (lag 2)', 'w_2 = 0.72', 66, C.blue],
    ['s = t−5 (lag 5)', 'w_5 = 0.44', 40, C.purple],
    ['s = t−10 (lag 10)', 'w_{10} = 0.20', 18, C.amber],
  ];
  lags.forEach(([lbl, val, bh, col], i) => {
    const x = 22 + i * 102;
    const yTop = 208 - bh;
    body += rbox(x, yTop, 86, bh, col.bg, col.bd, 4, 1.3);
    body += tx(x + 43, yTop - 12, val, { fs: 9.5, fw: 700, fill: col.tx, mono: true });
    body += tx(x + 43, 224, lbl, { fs: 9.5, fw: 600, fill: C.text });
  });
  return svg(W, H, body);
}

function affinePrecompute() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Input-Dependent Affine Recurrence: Parallel Projection + Scan', { fs: 13.5, fw: 700 });

  body += rbox(16, 48, 126, 84, C.slate.bg, C.slate.bd);
  body += tx(79, 72, 'Input Sequence', { fs: 11, fw: 700, fill: C.text });
  body += tx(79, 92, 'X ∈ R^{B×T×p}', { fs: 10, fw: 700, fill: C.slate.tx, mono: true });
  body += tx(79, 112, 'All T tokens ready', { fs: 9.5, fill: C.muted });

  body += arrow(144, 90, 174, 90);

  body += rbox(176, 48, 168, 84, C.blue.bg, C.blue.bd);
  body += tx(260, 68, '1. Batched GEMM (O(1) depth)', { fs: 10, fw: 700, fill: C.blue.tx });
  body += tx(260, 88, 'a_t = σ(W_z x_t + b_z)', { fs: 9.5, fill: C.text, mono: true });
  body += tx(260, 106, 'b_t = (1 − a_t) ⊙ (W_h x_t)', { fs: 9.5, fill: C.text, mono: true });

  body += arrow(346, 90, 376, 90);

  body += rbox(378, 48, 146, 84, C.green.bg, C.green.bd);
  body += tx(451, 68, '2. Parallel Scan', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(451, 88, 'h_t = a_t⊙h_{t-1} + b_t', { fs: 9.5, fill: C.text, mono: true });
  body += tx(451, 108, 'O(log₂ T) depth', { fs: 10, fw: 700, fill: C.green.tx });

  body += rbox(16, 150, 508, 82, C.amber.bg, C.amber.bd);
  body += tx(W / 2, 170, 'Selective Clock Gating via Input-Driven Survival Factor a_t ∈ (0, 1)^d', { fs: 11.5, fw: 700, fill: C.amber.tx });
  body += tx(W / 2, 192, '• Informative token x_t → a_{t,i} ≈ 0: flushes stale coordinate i and writes fresh candidate b_{t,i}', { fs: 10, fill: C.text });
  body += tx(W / 2, 212, '• Uninformative filler x_t → a_{t,i} ≈ 1: freezes coordinate i (h_{t,i} ≈ h_{t-1,i}) with zero decay', { fs: 10, fill: C.text });
  return svg(W, H, body);
}

function gruToMingru() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Simplifying Standard GRU to Minimal GRU (minGRU, Feng et al. 2024)', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 236, 198, C.coral.bg, C.coral.bd);
  body += tx(134, 58, 'Standard GRU (Sequential)', { fs: 11, fw: 700, fill: C.coral.tx });
  body += tx(134, 82, 'z_t = σ(W_xz x_t + W_hz h_{t-1})', { fs: 9.5, fill: C.text, mono: true });
  body += tx(134, 102, 'r_t = σ(W_xr x_t + W_hr h_{t-1})', { fs: 9.5, fill: C.text, mono: true });
  body += tx(134, 122, 'h̃_t = tanh(W_xh x_t + W_hh(r_t⊙h))', { fs: 9.5, fill: C.text, mono: true });
  body += tx(134, 146, 'h_t = z_t ⊙ h_{t-1} + (1−z_t) ⊙ h̃_t', { fs: 9.5, fill: C.text, mono: true });
  body += rbox(30, 166, 208, 54, '#FFFFFF', C.coral.bd, 5, 1.1);
  body += tx(134, 184, '3 recurrent GEMMs on h_{t-1}', { fs: 10, fw: 700, fill: C.coral.tx });
  body += tx(134, 202, '3d(p + d) ≈ 6d² parameters', { fs: 9.5, fill: C.muted, mono: true });

  body += arrow(254, 137, 286, 137);

  body += rbox(288, 38, 236, 198, C.green.bg, C.green.bd);
  body += tx(406, 58, 'minGRU (Parallel Scan Ready)', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(406, 82, 'z_t = σ(W_z x_t + b_z)  [drop W_hz]', { fs: 9.5, fill: C.text, mono: true });
  body += tx(406, 102, 'r_t removed           [drop W_hr]', { fs: 9.5, fill: C.muted, mono: true });
  body += tx(406, 122, 'h̃_t = W_h x_t + b_h     [drop W_hh]', { fs: 9.5, fill: C.text, mono: true });
  body += tx(406, 146, 'h_t = z_t ⊙ h_{t-1} + (1−z_t) ⊙ h̃_t', { fs: 9.5, fill: C.text, mono: true });
  body += rbox(302, 166, 208, 54, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(406, 184, '0 recurrent GEMMs on h_{t-1}!', { fs: 10, fw: 700, fill: C.green.tx });
  body += tx(406, 202, '2dp ≈ 2d² params (3× smaller)', { fs: 9.5, fill: C.muted, mono: true });
  return svg(W, H, body);
}

function depthRestoresGating() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'What One Linear Layer Loses—and How Stacking Restores State Gating', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 198, C.coral.bg, C.coral.bd);
  body += tx(139, 58, '1-Layer minGRU: Blind to State', { fs: 11, fw: 700, fill: C.coral.tx });
  body += tx(139, 80, 'Task: Output 1st nonzero digit seen', { fs: 10, fill: C.muted });
  body += rbox(30, 96, 218, 56, '#FFFFFF', C.coral.bd, 5, 1.1);
  body += tx(139, 114, 'Stream: 0 → [4] → 0 → [7] → 0', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(139, 134, 'z_t = σ(W_z x_t) sees ONLY x_t = 7', { fs: 9.5, fill: C.coral.tx, mono: true });
  body += tx(139, 172, '• Cannot tell slot is already full', { fs: 10, fill: C.text });
  body += tx(139, 192, '• Second digit [7] overwrites [4]!', { fs: 10, fw: 700, fill: C.coral.tx });

  body += rbox(278, 38, 246, 198, C.green.bg, C.green.bd);
  body += tx(401, 58, '2-Layer minGRU Stack: Restored!', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(401, 80, 'Depth replaces within-layer W_hz h_{t-1}', { fs: 10, fill: C.muted });
  body += rbox(292, 96, 218, 56, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(401, 114, 'Layer 1 state h_t^{(1)} tracks count', { fs: 10, fw: 700, fill: C.blue.tx });
  body += tx(401, 134, 'Gate 2: z_t^{(2)} = σ(W_z h_t^{(1)})', { fs: 9.5, fill: C.green.tx, mono: true });
  body += tx(401, 172, '• Layer 2 locks gate after 1st digit', { fs: 10, fill: C.text });
  body += tx(401, 192, '• Both layers run via parallel scan!', { fs: 10, fw: 700, fill: C.green.tx });
  return svg(W, H, body);
}

function affineCombineOp() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Associative Composition of Affine State Transitions', { fs: 13.5, fw: 700 });

  body += rbox(16, 44, 152, 82, C.blue.bg, C.blue.bd);
  body += tx(92, 66, 'Step 1 Map f_1', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(92, 88, 'f_1(h) = a_1 ⊙ h + b_1', { fs: 10, fill: C.text, mono: true });
  body += tx(92, 108, 'Pair: (a_1, b_1)', { fs: 10, fw: 700, fill: C.blue.tx, mono: true });

  body += arrow(170, 85, 194, 85);

  body += rbox(196, 44, 152, 82, C.purple.bg, C.purple.bd);
  body += tx(272, 66, 'Step 2 Map f_2', { fs: 11, fw: 700, fill: C.purple.tx });
  body += tx(272, 88, 'f_2(h) = a_2 ⊙ h + b_2', { fs: 10, fill: C.text, mono: true });
  body += tx(272, 108, 'Pair: (a_2, b_2)', { fs: 10, fw: 700, fill: C.purple.tx, mono: true });

  body += arrow(350, 85, 374, 85);

  body += rbox(376, 44, 148, 82, C.green.bg, C.green.bd);
  body += tx(450, 66, 'Composed f_2(f_1(h))', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(450, 88, 'A = a_1 ⊙ a_2', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(450, 108, 'B = a_2 ⊙ b_1 + b_2', { fs: 10, fw: 700, fill: C.green.tx, mono: true });

  body += rbox(16, 144, 508, 88, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 164, 'Operator:  (a_1, b_1) * (a_2, b_2) = (a_1 ⊙ a_2,  a_2 ⊙ b_1 + b_2)', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(W / 2, 188, '• Function composition is associative: ((p_1 * p_2) * p_3) = (p_1 * (p_2 * p_3))', { fs: 10, fill: C.text });
  body += tx(W / 2, 208, '• Caution: NOT commutative! Order matters because a_2 scales earlier injection b_1', { fs: 10, fw: 600, fill: C.coral.tx });
  return svg(W, H, body);
}

function complexityCrossover() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Asymptotic Complexity: Nonlinear GRU vs. Linear Scan vs. Attention', { fs: 13.5, fw: 700 });

  const cols = [
    ['Nonlinear GRU', 'Train FLOPs: 12 T d²', 'Seq Depth: O(T)', 'KV/State Mem: O(d)', 'Slowest training (serial)', C.coral],
    ['minGRU (Scan)', 'Train FLOPs: 4 T d²', 'Seq Depth: O(log T)', 'KV/State Mem: O(d)', 'Fast train + O(1) decode!', C.green],
    ['Causal Attention', 'Train FLOPs: 8Td²+2T²d', 'Seq Depth: O(1)', 'KV Cache: O(T d)', 'Quadratic T² when T > 4d', C.amber],
  ];
  cols.forEach(([title, l1, l2, l3, tag, col], i) => {
    const x = 16 + i * 172;
    body += rbox(x, 38, 164, 144, col.bg, col.bd);
    body += tx(x + 82, 58, title, { fs: 11, fw: 700, fill: col.tx });
    body += tx(x + 82, 82, l1, { fs: 9.5, fw: 600, fill: C.text, mono: true });
    body += tx(x + 82, 104, l2, { fs: 9.5, fill: C.text, mono: true });
    body += tx(x + 82, 126, l3, { fs: 9.5, fill: C.text, mono: true });
    body += rbox(x + 10, 144, 144, 24, '#FFFFFF', col.bd, 4, 1.0);
    body += tx(x + 82, 156, tag, { fs: 9, fw: 700, fill: col.tx });
  });

  body += rbox(16, 196, 508, 42, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 217, 'Attention Crossover: causal scores 2 T² d > projections 8 T d²  ⇔  T > 4d', { fs: 10, fw: 700, fill: C.blue.tx });
  return svg(W, H, body);
}

// ============================================================================
// ssm.md (8 diagrams)
// ============================================================================

function timescaleHierarchy() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Multi-Timescale Linguistic Hierarchy vs. Single-Scale Decay', { fs: 13.5, fw: 700 });

  const scales = [
    ['Fast Syntax (~10 tok)', 'Local verb agreement', 'τ_{1/2} ≈ 5–15 steps', C.blue],
    ['Medium Discourse (~200)', 'Paragraph topic & tense', 'τ_{1/2} ≈ 100–300 steps', C.purple],
    ['Slow Identity (~1,000+)', 'Character names & facts', 'τ_{1/2} ≈ 1,000+ steps', C.green],
  ];
  scales.forEach(([title, sub, tau, col], i) => {
    const y = 38 + i * 50;
    body += rbox(16, y, 224, 40, col.bg, col.bd);
    body += tx(128, y + 14, title, { fs: 10.5, fw: 700, fill: col.tx });
    body += tx(128, y + 30, sub, { fs: 9.5, fill: C.text });
    body += rbox(256, y + 6, 260, 28, col.bg, col.bd, 5, 1.3);
    body += tx(386, y + 20, tau, { fs: 10, fw: 700, fill: col.tx, mono: true });
  });

  body += rbox(16, 194, 508, 42, C.coral.bg, C.coral.bd);
  body += tx(W / 2, 215, 'Zero-bias gate a = 0.5 after 1,000 tokens: 2⁻¹⁰⁰⁰ = 0! Need log-spaced state bank.', { fs: 10, fw: 700, fill: C.coral.tx });
  return svg(W, H, body);
}

function continuousSsmBlock() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Continuous-Time State Space Model (ODE Block Diagram)', { fs: 13.5, fw: 700 });

  body += rbox(16, 62, 86, 56, C.slate.bg, C.slate.bd);
  body += tx(59, 84, 'Input x(t)', { fs: 10.5, fw: 700, fill: C.text });
  body += tx(59, 102, 'Scalar / R^d', { fs: 9.5, fill: C.muted });

  body += arrow(104, 90, 130, 90);

  body += rbox(132, 62, 86, 56, C.blue.bg, C.blue.bd);
  body += tx(175, 84, 'Write B', { fs: 10.5, fw: 700, fill: C.blue.tx });
  body += tx(175, 102, 'B ∈ C^{N×1}', { fs: 9.5, fill: C.text, mono: true });

  body += arrow(220, 90, 246, 90);

  body += rbox(248, 48, 144, 84, C.green.bg, C.green.bd);
  body += tx(320, 70, 'Latent State h(t)', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(320, 90, 'h\' = A h + B x', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(320, 110, 'Matrix A ∈ C^{N×N}', { fs: 9.5, fill: C.muted, mono: true });

  body += arrow(394, 90, 418, 90);

  body += rbox(420, 62, 104, 56, C.purple.bg, C.purple.bd);
  body += tx(472, 84, 'Readout C, D', { fs: 10.5, fw: 700, fill: C.purple.tx });
  body += tx(472, 102, 'y = C h + D x', { fs: 9.5, fill: C.text, mono: true });

  body += rbox(16, 152, 508, 80, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 172, 'Why Diagonalize A = diag(λ_1, ..., λ_N) with Re(λ_n) < 0?', { fs: 11.5, fw: 700, fill: C.text });
  body += tx(W / 2, 194, '• Dense A ∈ C^{N×N} costs O(N²)/token; diagonal A decouples into N scalar ODEs, O(N)/token', { fs: 9.5, fill: C.text });
  body += tx(W / 2, 214, '• Re(λ_n) < 0 guarantees bounded energy; Im(λ_n) = ω_n adds oscillatory lag-selective filters', { fs: 9.5, fill: C.blue.tx });
  return svg(W, H, body);
}

function diagonalOdeBank() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Diagonal SSM: N Parallel Leaky Integrators per Feature Channel', { fs: 13.5, fw: 700 });

  body += rbox(16, 86, 92, 74, C.slate.bg, C.slate.bd);
  body += tx(62, 114, 'Channel x_t', { fs: 10.5, fw: 700, fill: C.text });
  body += tx(62, 134, '1 of d dims', { fs: 9.5, fill: C.muted });

  const modes = [
    ['Mode 1 (Fast)', 'h\'_1 = λ_1 h_1 + b_1 x', 'τ_{1/2} ≈ 1.4 steps', C.blue, 38],
    ['Mode 2 (Med)', 'h\'_2 = λ_2 h_2 + b_2 x', 'τ_{1/2} ≈ 25 steps', C.purple, 100],
    ['Mode N (Slow)', 'h\'_N = λ_N h_N + b_N x', 'τ_{1/2} ≈ 693 steps', C.green, 162],
  ];
  modes.forEach(([title, eq, tau, col, y]) => {
    body += arrow(110, 123, 152, y + 25);
    body += rbox(154, y, 224, 50, col.bg, col.bd);
    body += tx(266, y + 18, `${title}:  ${eq}`, { fs: 9.5, fw: 700, fill: col.tx, mono: true });
    body += tx(266, y + 36, tau, { fs: 9.5, fill: C.text });
    body += arrow(380, y + 25, 422, 123 + (y - 100) * 0.37);   // spread the fan-in heads: 100/123/146
  });

  body += rbox(424, 78, 100, 90, C.amber.bg, C.amber.bd);
  body += tx(474, 104, 'Mix Output', { fs: 10.5, fw: 700, fill: C.amber.tx });
  body += tx(474, 124, 'y_t = D x_t +', { fs: 9.5, fill: C.text, mono: true });
  body += tx(474, 142, '∑ c_n h_{t,n}', { fs: 10, fw: 700, fill: C.amber.tx, mono: true });

  body += tx(W / 2, 234, 'Across all d channels, total recurrent state expands to d × N numbers (e.g. d=128, N=16 ⇒ 2,048 floats)', { fs: 9.5, fw: 600, fill: C.muted });
  return svg(W, H, body);
}

function zohStepHold() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Zero-Order Hold (ZOH) Exact Discretization on ((t−1)Δ, tΔ]', { fs: 13.5, fw: 700 });

  body += rbox(16, 40, 236, 118, C.blue.bg, C.blue.bd);
  body += tx(134, 60, 'Piecewise-Constant ZOH Input', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(134, 84, 'x(τ) ≡ x_t   on ((t−1)Δ, tΔ]', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(134, 110, 'Exact variation-of-constants integral:', { fs: 9.5, fill: C.muted });
  body += tx(134, 132, 'h_t = e^{Δλ} h_{t-1} + ∫ e^{sλ} b x_t ds', { fs: 9.5, fill: C.blue.tx, mono: true });

  body += arrow(254, 99, 284, 99);

  body += rbox(286, 40, 238, 118, C.green.bg, C.green.bd);
  body += tx(405, 60, 'Discrete Recurrence Parameters', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(405, 86, 'ā = exp(Δ λ) ∈ (0, 1)', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(405, 110, 'b̄ = ((exp(Δ λ) − 1) / λ) b', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(405, 134, 'First-order limit: b̄ ≈ Δ b', { fs: 9.5, fill: C.muted, mono: true });

  body += rbox(16, 172, 508, 62, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 192, 'Resulting Discrete Linear Recurrence:   h_t = ā h_{t-1} + b̄ x_t', { fs: 11, fw: 700, fill: C.text, mono: true });
  body += tx(W / 2, 214, 'Matches §12.3 linear recurrence! Re(λ) < 0 and Δ > 0 guarantee |ā| = e^{Δ Re(λ)} < 1.', { fs: 9.5, fill: C.blue.tx });
  return svg(W, H, body);
}

function deltaDilationTable() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Step Size Δ as a Continuous Timescale Dilation Gate (λ = −1)', { fs: 13.5, fw: 700 });

  const rows = [
    ['Δ = 1.000', 'ā = 0.368', 'τ_{1/2} = 0.7 steps (fast flush)', C.coral],
    ['Δ = 0.100', 'ā = 0.905', 'τ_{1/2} = 6.9 steps (local syntax)', C.amber],
    ['Δ = 0.010', 'ā = 0.990', 'τ_{1/2} = 69 steps (paragraph)', C.blue],
    ['Δ = 0.001', 'ā = 0.999', 'τ_{1/2} = 693 steps (long memory)', C.green],
  ];
  rows.forEach(([delta, abar, tau, col], i) => {
    const y = 40 + i * 40;
    body += tx(20, y + 15, delta, { fs: 10.5, fw: 700, anchor: 'start', fill: C.text, mono: true });
    body += tx(110, y + 15, abar, { fs: 10.5, anchor: 'start', fill: C.muted, mono: true });
    body += rbox(204, y, 316, 28, col.bg, col.bd, 5, 1.3);
    body += tx(362, y + 15, tau, { fs: 10, fw: 700, fill: col.tx });
  });

  body += rbox(16, 202, 508, 34, C.purple.bg, C.purple.bd);
  body += tx(W / 2, 220, 'Log-uniform initialization log Δ ∈ [ln(0.001), ln(0.1)] spans 10 to 1,000 tokens at Step 0', { fs: 10, fw: 700, fill: C.purple.tx });
  return svg(W, H, body);
}

function fftConvPipeline() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Time-Invariant SSM Dual Mode: O(T log T) FFT Convolution vs. O(1) Step', { fs: 13.5, fw: 700 });

  const steps = [
    ['1. Impulse Kernel', 'K_k = ∑ c_n ā_n^k b̄_n', 'Length T filter', C.blue, 14, 124],
    ['2. Zero-Pad 2T', 'Pad x and K to 2T', 'No wrap-around', C.slate, 150, 120],
    ['3. rFFT Multiply', 'F⁻¹(F(K) ⊙ F(x))', 'O(T log T) parallel', C.purple, 282, 122],
    ['4. Crop [:T] + Dx', 'Causal y_{1:T}', 'Matches scan!', C.green, 416, 110],
  ];
  steps.forEach(([title, l1, l2, col, x, w], i) => {
    body += rbox(x, 44, w, 86, col.bg, col.bd);
    body += tx(x + w / 2, 66, title, { fs: 10, fw: 700, fill: col.tx });
    body += tx(x + w / 2, 86, l1, { fs: 9, fill: C.text, mono: true });
    body += tx(x + w / 2, 106, l2, { fs: 9, fill: C.muted });
    if (i < steps.length - 1) {
      const nx = steps[i + 1][4];
      body += arrow(x + w + 2, 87, nx - 2, 87);
    }
  });

  body += rbox(16, 148, 508, 84, C.amber.bg, C.amber.bd);
  body += tx(W / 2, 168, 'Why Time-Invariant SSMs Enjoy Two Equivalent Execution Modes', { fs: 11.5, fw: 700, fill: C.amber.tx });
  body += tx(W / 2, 190, '• Training (full sequence known): evaluate y = K * x + D x across all T tokens via FFT', { fs: 10, fill: C.text });
  body += tx(W / 2, 210, '• Autoregressive Decoding: switch to h_t = ā ⊙ h_{t-1} + b̄ x_t in O(N) state and O(1) time/tok', { fs: 10, fill: C.text });
  return svg(W, H, body);
}

function realVsComplexBasis() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Real vs. Complex Rates: Fitting a Delayed Spike at Lag k* = 20', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 198, C.coral.bg, C.coral.bd);
  body += tx(139, 58, 'Real Rates (This Chapter\'s S4D)', { fs: 11.5, fw: 700, fill: C.coral.tx });
  body += tx(139, 80, 'λ_n = −n   ⇒   ā_n^k = e^{−n Δ k}', { fs: 10, fill: C.text, mono: true });
  body += rbox(30, 96, 218, 54, '#FFFFFF', C.coral.bd, 5, 1.1);
  body += tx(139, 114, 'All basis curves decay monotonically', { fs: 9.5, fw: 700, fill: C.coral.tx });
  body += tx(139, 134, 'Must cancel 16 nearly parallel curves', { fs: 9.5, fill: C.muted });
  body += tx(139, 170, '• N = 16: spike residual 0.89, bump 0.03', { fs: 10, fill: C.text });
  body += tx(139, 190, '• Readout weights explode: ||c||_{∞} ~ 10¹¹', { fs: 10, fw: 700, fill: C.coral.tx });
  body += tx(139, 212, '• Gradient descent cannot reach such weights', { fs: 10, fill: C.text });

  body += rbox(278, 38, 246, 198, C.green.bg, C.green.bd);
  body += tx(401, 58, 'Complex Rates (S4D-Lin, Exercise)', { fs: 11.5, fw: 700, fill: C.green.tx });
  body += tx(401, 80, 'λ_m = −1/2 ± i π m   (damped cosines)', { fs: 10, fill: C.text, mono: true });
  body += rbox(292, 96, 218, 54, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(401, 114, 'e^{−Δk/2} cos(ω_m Δ k): nearly orthogonal', { fs: 9.5, fw: 700, fill: C.green.tx });
  body += tx(401, 134, 'Constructive interference at k = k*', { fs: 9.5, fill: C.muted });
  body += tx(401, 170, '• Well-conditioned Fourier-like basis', { fs: 10, fill: C.text });
  body += tx(401, 190, '• Bounded weights: ||c||_{∞} ~ O(1)', { fs: 10, fw: 700, fill: C.green.tx });
  body += tx(401, 212, '• Smaller spike residual than real rates', { fs: 10, fill: C.text });
  return svg(W, H, body);
}

function recall2x2Grid() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, '2×2 Ablation on Delayed Recall (T = 1,000): Timescales vs. Architecture', { fs: 13.5, fw: 700 });

  body += tx(206, 42, 'Short Initial Horizon (τ_{1/2} ≤ 7)', { fs: 10.5, fw: 700, fill: C.coral.tx });
  body += tx(420, 42, 'Long Log-Spaced Horizon (τ_{1/2} = 693)', { fs: 10.5, fw: 700, fill: C.green.tx });

  body += rbox(16, 54, 80, 78, C.slate.bg, C.slate.bd);
  body += tx(56, 86, 'minGRU', { fs: 10.5, fw: 700, fill: C.text });
  body += tx(56, 104, '(N = 1)', { fs: 9.5, fill: C.muted, mono: true });

  body += rbox(106, 54, 200, 78, C.coral.bg, C.coral.bd);
  body += tx(206, 76, 'minGRU (b_z = 0, τ_{1/2} = 1)', { fs: 10, fw: 700, fill: C.coral.tx });
  body += tx(206, 98, '0.5⁹⁹⁹ ≈ 10⁻³⁰¹ underflow', { fs: 9.5, fill: C.text, mono: true });
  body += tx(206, 118, 'Stalls at chance (0.12)', { fs: 9.5, fw: 700, fill: C.coral.tx });

  body += rbox(316, 54, 208, 78, C.green.bg, C.green.bd);
  body += tx(420, 76, 'minGRU (log-spaced b_z)', { fs: 10, fw: 700, fill: C.green.tx });
  body += tx(420, 98, 'τ_{1/2} ∈ [7, 693] across d dims', { fs: 9.5, fill: C.text, mono: true });
  body += tx(420, 118, 'Converges to 100% accuracy!', { fs: 9.5, fw: 700, fill: C.green.tx });

  body += rbox(16, 144, 80, 78, C.slate.bg, C.slate.bd);
  body += tx(56, 176, 'S4D-Real', { fs: 10.5, fw: 700, fill: C.text });
  body += tx(56, 194, '(N = 16)', { fs: 9.5, fill: C.muted, mono: true });

  body += rbox(106, 144, 200, 78, C.coral.bg, C.coral.bd);
  body += tx(206, 166, 'S4D (Δ ≥ 0.1, τ_{1/2} ≤ 7)', { fs: 10, fw: 700, fill: C.coral.tx });
  body += tx(206, 188, 'Slowest mode: e⁻¹⁰⁰ ≈ 10⁻⁴⁴', { fs: 9.5, fill: C.text, mono: true });
  body += tx(206, 208, 'Fails to bridge 1,000-step gap', { fs: 9.5, fw: 700, fill: C.coral.tx });

  body += rbox(316, 144, 208, 78, C.blue.bg, C.blue.bd);
  body += tx(420, 166, 'S4D (Δ ∈ [0.001, 0.1])', { fs: 10, fw: 700, fill: C.blue.tx });
  body += tx(420, 188, 'N=16 modes per channel', { fs: 9.5, fill: C.text, mono: true });
  body += tx(420, 208, 'Converges to 100%!', { fs: 9.5, fw: 700, fill: C.blue.tx });
  return svg(W, H, body);
}

// ============================================================================
// mamba.md (6 diagrams)
// ============================================================================

function mambaUnification() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Mamba Unifies minGRU Content Selectivity with S4D State Expansion', { fs: 13.5, fw: 700 });

  body += rbox(16, 42, 224, 86, C.blue.bg, C.blue.bd);
  body += tx(128, 62, '§12.3 minGRU (Selective)', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(128, 84, '+ Input-dependent gate z_t(x_t)', { fs: 10, fill: C.green.tx });
  body += tx(128, 104, '− Only N = 1 scalar per channel', { fs: 10, fill: C.coral.tx });

  body += rbox(300, 42, 224, 86, C.purple.bg, C.purple.bd);
  body += tx(412, 62, '§12.4 S4D (Multi-Timescale)', { fs: 11, fw: 700, fill: C.purple.tx });
  body += tx(412, 84, '+ N = 16 state modes per channel', { fs: 10, fill: C.green.tx });
  body += tx(412, 104, '− Fixed LTI kernel K_k ignores x_t', { fs: 10, fill: C.coral.tx });

  body += arrow(128, 130, 210, 152);
  body += arrow(412, 130, 330, 152);

  body += rbox(46, 154, 448, 80, C.green.bg, C.green.bd);
  body += tx(270, 174, '§12.5 Selective SSM / Mamba (Gu & Dao, 2023)', { fs: 11.5, fw: 700, fill: C.green.tx });
  body += tx(270, 196, 'Token-dependent (Δ_t, B_t, C_t) over H × N expanded state', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(270, 216, 'Drops FFT convolution (time-varying) in favor of parallel associative scan', { fs: 9.5, fill: C.muted });
  return svg(W, H, body);
}

function softplusSigmoidDuality() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Exact Mathematical Duality: ZOH Step Size Δ_t Is a Sigmoid Keep Gate', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 48, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 54, 'Identity (A = −1):  ā_t = exp(−softplus(s_t)) = 1 / (1 + e^{s_t}) = σ(−s_t)', { fs: 10, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(W / 2, 72, 'First-order ZOH write weight: (1 − e^{−Δ_t}) B_t x_t = (1 − z_t) h̃_t  ⇒  1D Selective SSM = minGRU!', { fs: 9.5, fill: C.text });

  body += rbox(16, 98, 246, 138, C.green.bg, C.green.bd);
  body += tx(139, 118, 'Noise / Filler Token (Pause Clock)', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(139, 142, 's_t = −4.6   ⇒   Δ_t = 0.01', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(139, 166, 'Keep gate z_t = σ(+4.6) = 0.990', { fs: 10, fill: C.green.tx, mono: true });
  body += tx(139, 190, 'Write gate 1 − z_t = 0.010', { fs: 10, fill: C.muted, mono: true });
  body += tx(139, 214, 'Freezes memory h_t ≈ h_{t-1} across gap', { fs: 9.5, fw: 700, fill: C.green.tx });

  body += rbox(278, 98, 246, 138, C.amber.bg, C.amber.bd);
  body += tx(401, 118, 'Marked Symbol Token (Latch Fact)', { fs: 11, fw: 700, fill: C.amber.tx });
  body += tx(401, 142, 's_t = +2.0   ⇒   Δ_t = 2.13', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(401, 166, 'Keep gate z_t = σ(−2.0) = 0.119', { fs: 10, fill: C.amber.tx, mono: true });
  body += tx(401, 190, 'Write gate 1 − z_t = 0.881', { fs: 10, fill: C.text, mono: true });
  body += tx(401, 214, 'Opens valve to write B_t x_t into state', { fs: 9.5, fw: 700, fill: C.amber.tx });
  return svg(W, H, body);
}

function selectiveSsmDataflow() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Selective SSM Per-Token Projections and H × N State Update', { fs: 13.5, fw: 700 });

  body += rbox(14, 76, 88, 76, C.slate.bg, C.slate.bd);
  body += tx(58, 102, 'Conv Out', { fs: 10, fw: 700, fill: C.text });
  body += tx(58, 122, 'x_t ∈ R^H', { fs: 10, fw: 700, fill: C.slate.tx, mono: true });

  const branches = [
    ['Δ_t = softplus(W_{Δ} x_t + b_{Δ})', 'Rank-R step size ∈ R_{+}^{H}', C.amber, 36],
    ['B_t = W_B x_t ∈ R^N', 'Token write vector across H', C.blue, 96],
    ['C_t = W_C x_t ∈ R^N', 'Token readout query across H', C.purple, 156],
  ];
  branches.forEach(([eq, sub, col, y]) => {
    body += arrow(104, 114, 130, y + 22);
    body += rbox(132, y, 200, 46, col.bg, col.bd);
    body += tx(232, y + 17, eq, { fs: 9.5, fw: 700, fill: col.tx, mono: true });
    body += tx(232, y + 34, sub, { fs: 9, fill: C.muted });
    body += arrow(334, y + 22, 360, 114 + (y - 96) / 3);   // spread the fan-in heads
  });

  body += rbox(362, 52, 164, 124, C.green.bg, C.green.bd);
  body += tx(444, 74, 'H × N State Scan', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(444, 96, 'a_t = exp(Δ_t λ)', { fs: 9.5, fill: C.text, mono: true });
  body += tx(444, 116, 'b_t = Δ_t B_t x_t', { fs: 9.5, fill: C.text, mono: true });
  body += tx(444, 138, 'h_t = a_t ⊙ h_{t-1} + b_t', { fs: 9.5, fw: 700, fill: C.green.tx, mono: true });
  body += tx(444, 158, 'y_t = h_t C_t + D ⊙ x_t', { fs: 9.5, fw: 700, fill: C.text, mono: true });

  body += tx(W / 2, 228, 'Rates λ_n start at −n, n = 1..N: N timescales; Δ_{t,h} dilates all N modes simultaneously', { fs: 9.5, fw: 600, fill: C.muted });
  return svg(W, H, body);
}

function scan4dMemory() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Pitfall: Autograd Activation Explosion in 4D Expanded Scans', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 152, C.blue.bg, C.blue.bd);
  body += tx(139, 58, '3D minGRU Scan: (B, T, d)', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(139, 82, '1 state scalar per channel', { fs: 10, fill: C.muted });
  body += tx(139, 106, 'Log₂ T doubling rounds in autograd', { fs: 10, fill: C.text });
  body += tx(139, 130, 'Fits easily in GPU HBM at N = 1', { fs: 10, fw: 700, fill: C.blue.tx });
  body += rbox(40, 148, 198, 26, '#FFFFFF', C.blue.bd, 4, 1.1);
  body += tx(139, 161, 'Baseline 1× scan tensor volume', { fs: 9.5, fw: 700, fill: C.blue.tx });

  body += rbox(278, 38, 246, 152, C.coral.bg, C.coral.bd);
  body += tx(401, 58, '4D Mamba Naive Scan: (B, T, H, N)', { fs: 11, fw: 700, fill: C.coral.tx });
  body += tx(401, 82, 'H = 2d, N = 16  ⇒  32× larger tensor!', { fs: 10, fw: 700, fill: C.coral.tx, mono: true });
  body += tx(401, 106, '11 rounds × 2 tensors (a, b) = 22 copies', { fs: 10, fill: C.text });
  body += tx(401, 130, 'B=16, T=2048, H=4096, N=16:', { fs: 9.5, fill: C.muted, mono: true });
  body += rbox(294, 148, 214, 26, '#FFFFFF', C.coral.bd, 4, 1.1);
  body += tx(401, 161, '8.6 GB/tensor ⇒ ≈190 GB per layer!', { fs: 9.5, fw: 700, fill: C.coral.tx });

  body += rbox(16, 202, 508, 38, C.green.bg, C.green.bd);
  body += tx(W / 2, 221, 'Fix: Custom Adjoint Reverse Scan saves only (a, h) = 2 tensors (≈10× less HBM!)', { fs: 10.5, fw: 700, fill: C.green.tx });
  return svg(W, H, body);
}

function adjointReverseScan() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'The Adjoint Scan: Forward Prefix Scan + Reverse Gradient Scan', { fs: 13.5, fw: 700 });

  body += rbox(16, 40, 508, 76, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 58, 'Forward Pass (t = 1 → T):   h_t = a_t ⊙ h_{t-1} + b_t', { fs: 11, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(W / 2, 80, 'Evaluated in O(log₂ T) parallel doubling rounds; discards intermediate rounds!', { fs: 10, fill: C.text });
  body += tx(W / 2, 98, 'Saves ONLY final state trajectory h_{1:T} and decay factors a_{1:T} (2 tensors total)', { fs: 9.5, fw: 700, fill: C.blue.tx });

  body += rbox(16, 130, 508, 102, C.green.bg, C.green.bd);
  body += tx(W / 2, 150, 'Backward Pass (t = T → 1):   g_t = ∂L/∂h_t + a_{t+1} ⊙ g_{t+1}', { fs: 11, fw: 700, fill: C.green.tx, mono: true });
  body += tx(W / 2, 172, 'Time-reversed linear recurrence! Evaluated in O(log₂ T) via flip(a) → parallel_scan → flip', { fs: 9.5, fill: C.text });
  body += tx(W / 2, 194, 'Parameter gradients recovered elementwise:   ∂L/∂b_t = g_t,    ∂L/∂a_t = g_t ⊙ h_{t-1}', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(W / 2, 214, 'Production Mamba fuses discretization + scan into SRAM so (B, T, H, N) never touches HBM', { fs: 9.5, fill: C.muted });
  return svg(W, H, body);
}

function stateContributionFormula() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Exact Decomposition of Final State h_T by Source Position s', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 52, C.purple.bg, C.purple.bd);
  body += tx(W / 2, 54, 'h_T = ∑_{s=1}^{T} exp( ∑_{r=s+1}^{T} ln a_r ) ⊙ b_s', { fs: 11, fw: 700, fill: C.purple.tx, mono: true });
  body += tx(W / 2, 74, 'Contribution norm ||c_s|| shows how much token s survives into summary state h_T', { fs: 10, fill: C.text });

  body += rbox(16, 102, 246, 130, C.amber.bg, C.amber.bd);
  body += tx(139, 122, 'Mamba on Selective Copying', { fs: 11, fw: 700, fill: C.amber.tx });
  body += tx(139, 144, '• 4 sharp spikes at marked symbols', { fs: 10, fill: C.text });
  body += tx(139, 164, '• Width-4 causal conv smears each', { fs: 10, fill: C.text });
  body += tx(139, 182, '  marker across a 4-token window', { fs: 10, fill: C.text });
  body += tx(139, 206, '• Noise tokens suppressed (Δ_t ≈ 0)', { fs: 10, fw: 700, fill: C.green.tx });

  body += rbox(278, 102, 246, 130, C.coral.bg, C.coral.bd);
  body += tx(401, 122, 'S4D on Selective Copying', { fs: 11, fw: 700, fill: C.coral.tx });
  body += tx(401, 144, '• Fixed Δ cannot inspect marker token', { fs: 10, fill: C.text });
  body += tx(401, 164, '• Smoothly decaying recency envelope', { fs: 10, fill: C.text });
  body += tx(401, 182, '  depends only on lag T − s', { fs: 10, fill: C.muted });
  body += tx(401, 206, '• Absorbs filler noise alongside symbols', { fs: 10, fw: 700, fill: C.coral.tx });
  return svg(W, H, body);
}

// ============================================================================
// matrix-state.md (7 diagrams)
// ============================================================================

function vectorVsMatrixState() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Upgrading Recurrent State Geometry: Vector h_t vs. Key-Value Matrix S_t', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 198, C.coral.bg, C.coral.bd);
  body += tx(139, 58, 'Vector State (§§12.1–12.5)', { fs: 11.5, fw: 700, fill: C.coral.tx });
  body += rbox(30, 74, 218, 44, '#FFFFFF', C.coral.bd, 5, 1.1);
  body += tx(139, 92, 'h_t = a_t ⊙ h_{t-1} + b_t', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(139, 108, '1D vector superimposes all facts', { fs: 9.5, fill: C.muted });
  body += tx(139, 142, '• No separate address (key) space', { fs: 10, fill: C.text });
  body += tx(139, 164, '• Readout y_t = C_t h_t mixes slots', { fs: 10, fill: C.text });
  body += tx(139, 186, '• Fails multi-pair associative recall', { fs: 10, fw: 700, fill: C.coral.tx });

  body += rbox(278, 38, 246, 198, C.green.bg, C.green.bd);
  body += tx(401, 58, 'Matrix State (§§12.6–12.7)', { fs: 11.5, fw: 700, fill: C.green.tx });
  body += rbox(292, 74, 218, 44, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(401, 92, 'S_t = S_{t-1} + v_t k_tᵀ', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(401, 108, 'Outer-product store ∈ R^{d_v×d_k}', { fs: 9.5, fill: C.muted });
  body += tx(401, 142, '• Write value v_t at key address k_t', { fs: 10, fill: C.text });
  body += tx(401, 164, '• Query by matrix-vector product S_t q_t', { fs: 10, fill: C.text });
  body += tx(401, 186, '• Exact retrieval for orthogonal keys!', { fs: 10, fw: 700, fill: C.green.tx });
  return svg(W, H, body);
}

function orthogonalVsCrosstalk() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Outer-Product Memory in R²: Orthogonal Exactness vs. Overcomplete Cross-Talk', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 198, C.green.bg, C.green.bd);
  body += tx(139, 58, '2 Orthogonal Keys in d_k = 2', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(139, 80, 'k_1 = (1, 0)ᵀ, v_1 = 3', { fs: 10, fill: C.text, mono: true });
  body += tx(139, 100, 'k_2 = (0, 1)ᵀ, v_2 = 5', { fs: 10, fill: C.text, mono: true });
  body += rbox(32, 116, 214, 54, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(139, 134, 'S_2 = v_1 k_1ᵀ + v_2 k_2ᵀ = [3, 5]', { fs: 10, fw: 700, fill: C.green.tx, mono: true });
  body += tx(139, 154, 'S_2 k_1 = 3,   S_2 k_2 = 5 (0 error!)', { fs: 9.5, fill: C.text, mono: true });
  body += tx(139, 194, 'k_1ᵀ k_2 = 0 ⇒ Zero cross-talk', { fs: 10, fw: 700, fill: C.green.tx });

  body += rbox(278, 38, 246, 198, C.coral.bg, C.coral.bd);
  body += tx(401, 58, '3rd Key Exceeds Dimension (n = 3 > 2)', { fs: 11, fw: 700, fill: C.coral.tx });
  body += tx(401, 80, 'Add k_3 = (1, 1)ᵀ/√2, v_3 = 7', { fs: 10, fill: C.text, mono: true });
  body += tx(401, 100, 'Overlap: k_3ᵀ k_1 = 1/√2 ≈ 0.707', { fs: 10, fill: C.coral.tx, mono: true });
  body += rbox(294, 116, 214, 54, '#FFFFFF', C.coral.bd, 5, 1.1);
  body += tx(401, 134, 'S_3 k_1 = v_1 + (k_3ᵀ k_1) v_3', { fs: 10, fw: 700, fill: C.coral.tx, mono: true });
  body += tx(401, 154, '= 3 + 7/√2 ≈ 7.95  (error +4.95!)', { fs: 9.5, fill: C.text, mono: true });
  body += tx(401, 194, 'Max orthogonal keys in R^{d_k} is d_k', { fs: 10, fw: 700, fill: C.coral.tx });
  return svg(W, H, body);
}

function leftVsRightAssoc() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Linear Attention Associativity: O(d_v d_k) Recurrence vs. O(t) Attention', { fs: 13.5, fw: 700 });

  body += rbox(16, 40, 246, 140, C.green.bg, C.green.bd);
  body += tx(139, 60, 'Left Association (Recurrent View)', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(139, 86, 'y_t = (∑ v_s k_sᵀ) q_t = S_t q_t', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(139, 114, '• Collapses prefix into S_t ∈ R^{d_v×d_k}', { fs: 9.5, fill: C.text });
  body += tx(139, 136, '• O(d_v d_k) state & O(d_v d_k) FLOPs/step', { fs: 9.5, fw: 700, fill: C.green.tx });
  body += tx(139, 158, '• Ideal for O(1) autoregressive decoding', { fs: 9.5, fill: C.muted });

  body += rbox(278, 40, 246, 140, C.blue.bg, C.blue.bd);
  body += tx(401, 60, 'Right Association (Attention View)', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(401, 86, 'y_t = ∑_{s≤t} v_s ( k_sᵀ q_t )', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(401, 114, '• Forms t×t inner-product score matrix', { fs: 9.5, fill: C.text });
  body += tx(401, 136, '• No exp(·) softmax coupling across s!', { fs: 9.5, fw: 700, fill: C.blue.tx });
  body += tx(401, 158, '• Parallel Tensor-Core GEMMs on chunks', { fs: 9.5, fill: C.muted });

  body += rbox(16, 194, 508, 40, C.amber.bg, C.amber.bd);
  body += tx(W / 2, 214, 'Removing softmax exp(·) unlocks associativity: (v_s k_sᵀ) q_t = v_s (k_sᵀ q_t)', { fs: 10.5, fw: 700, fill: C.amber.tx });
  return svg(W, H, body);
}

function retrievalErrorLaw() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Retrieval Signal-to-Noise Law for Random Isotropic Unit Keys', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 508, 54, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 56, 'Readout:  S k_j = ||k_j||² v_j + ∑_{i≠j} (k_iᵀ k_j) v_i = v_j + ε_j', { fs: 10.5, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(W / 2, 76, 'For independent unit vectors in R^{d_k}:   E[(k_iᵀ k_j)²] = 1 / d_k', { fs: 10, fill: C.text, mono: true });

  body += rbox(16, 106, 246, 126, C.amber.bg, C.amber.bd);
  body += tx(139, 126, 'Expected Vector Error E[||ε_j||²]', { fs: 11, fw: 700, fill: C.amber.tx });
  body += tx(139, 152, 'E[||S k_j − v_j||²] = (n − 1) / d_k', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(139, 178, '• Grows linearly with stored pairs n − 1', { fs: 9.5, fill: C.text });
  body += tx(139, 198, '• Shrinks as 1/d_k with key dimension', { fs: 9.5, fill: C.text });

  body += rbox(278, 106, 246, 126, C.green.bg, C.green.bd);
  body += tx(401, 126, '1D Codebook Projection Variance', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(401, 152, 'E[(u_rᵀ ε_j)²] = (n − 1) / (d_k d_v)', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(401, 178, '• Nearest-neighbor decoding projects ε_j', { fs: 9.5, fill: C.text });
  body += tx(401, 198, '• Multi-head splits trade d_k d_v per head', { fs: 9.5, fill: C.text });
  return svg(W, H, body);
}

function correlatedKeyCone() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Pitfall: Key Correlation Creates an Irreducible Cross-Talk Floor', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 198, C.green.bg, C.green.bd);
  body += tx(139, 58, 'Isotropic Keys (ρ = 0)', { fs: 11.5, fw: 700, fill: C.green.tx });
  body += tx(139, 80, 'k_i uniform on unit sphere S^{d_k-1}', { fs: 10, fill: C.text });
  body += rbox(32, 96, 214, 52, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(139, 114, 'E[||ε_j||²] = (n − 1) / d_k', { fs: 10.5, fw: 700, fill: C.green.tx, mono: true });
  body += tx(139, 134, '→ 0 as key dimension d_k → ∞', { fs: 9.5, fill: C.text });
  body += tx(139, 170, '• Centering + L2 norm spreads keys', { fs: 10, fill: C.text });
  body += tx(139, 192, '• Doubling d_k halves retrieval MSE', { fs: 10, fw: 700, fill: C.green.tx });

  body += rbox(278, 38, 246, 198, C.coral.bg, C.coral.bd);
  body += tx(401, 58, 'Correlated Key Cone (ρ > 0)', { fs: 11.5, fw: 700, fill: C.coral.tx });
  body += tx(401, 80, 'k_i = ρ c + √(1 − ρ²) ξ_i  (mean c)', { fs: 9.5, fill: C.text, mono: true });
  body += rbox(294, 96, 214, 52, '#FFFFFF', C.coral.bd, 5, 1.1);
  body += tx(401, 114, 'MSE ≈ (n−1)(ρ⁴ + (1−ρ²)²/d_k)', { fs: 9.5, fw: 700, fill: C.coral.tx, mono: true });
  body += tx(401, 134, 'Floor (n − 1) ρ⁴ NEVER vanishes!', { fs: 9.5, fw: 700, fill: C.coral.tx });
  body += tx(401, 170, '• Unnormalized embeddings cluster on c', { fs: 10, fill: C.text });
  body += tx(401, 192, '• Center keys before L2-normalizing', { fs: 10, fw: 700, fill: C.coral.tx });
  return svg(W, H, body);
}

function slidingWindowDecay() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Scalar Matrix Decay a_t Creates a Bounded Sliding Memory Window', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 142, C.coral.bg, C.coral.bd);
  body += tx(139, 58, 'Undecayed Matrix (a_t = 1)', { fs: 11.5, fw: 700, fill: C.coral.tx });
  body += tx(139, 82, 'S_t = S_{t-1} + v_t k_tᵀ', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(139, 108, '• Interference: (t − 1) / d_k', { fs: 10, fill: C.coral.tx, mono: true });
  body += tx(139, 130, '• Grows without bound as t → ∞', { fs: 10, fill: C.text });
  body += tx(139, 152, '• Old noise swamps all queries', { fs: 9.5, fill: C.muted });

  body += rbox(278, 38, 246, 142, C.green.bg, C.green.bd);
  body += tx(401, 58, 'Decayed Matrix (a_t = a < 1)', { fs: 11.5, fw: 700, fill: C.green.tx });
  body += tx(401, 82, 'S_t = a_t S_{t-1} + v_t k_tᵀ', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(401, 108, '• Effective window: ∑ a^l = 1/(1 − a)', { fs: 9.5, fill: C.green.tx, mono: true });
  body += tx(401, 130, '• Steady MSE ≤ 1 / ((1 − a²) d_k)', { fs: 9.5, fw: 700, fill: C.text, mono: true });
  body += tx(401, 152, '• Bounded error at ANY length T!', { fs: 9.5, fill: C.muted });

  body += rbox(16, 194, 508, 40, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 214, 'Input-dependent a_t ∈ (0, 1) flushes stale context when topic shifts while preserving active keys', { fs: 10, fw: 700, fill: C.blue.tx });
  return svg(W, H, body);
}

function ssdDuality() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'State Space Duality (SSD, Dao & Gu 2024): Primal vs. Dual Form', { fs: 13.5, fw: 700 });

  body += rbox(16, 40, 232, 124, C.blue.bg, C.blue.bd);
  body += tx(132, 60, 'Primal Matrix Recurrence', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(132, 84, 'S_t = a_t S_{t-1} + v_t k_tᵀ', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(132, 106, 'y_t = S_t q_t', { fs: 10.5, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(132, 132, 'O(T d_v d_k) sequential state update', { fs: 9.5, fill: C.muted });

  body += tx(270, 102, '⇔', { fs: 20, fw: 700, fill: C.amber.tx });

  body += rbox(292, 40, 232, 124, C.green.bg, C.green.bd);
  body += tx(408, 60, 'Dual Masked Linear Attention', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(408, 84, 'Y = ( L ⊙ (Q Kᵀ) ) V', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(408, 106, 'L_{ts} = ∏_{r=s+1}^{t} a_r  (t ≥ s)', { fs: 10, fw: 700, fill: C.green.tx, mono: true });
  body += tx(408, 132, 'O(T² d) parallel Tensor-Core GEMM', { fs: 9.5, fill: C.muted });

  body += rbox(16, 178, 508, 56, C.purple.bg, C.purple.bd);
  body += tx(W / 2, 198, 'Chunked Hybrid Execution (Chunk Size m = 64): Best of Both Worlds', { fs: 11, fw: 700, fill: C.purple.tx });
  body += tx(W / 2, 218, 'Uses dual m×m GEMM within chunks and primal S_c update across T/m chunks in O(T m d + T d_v d_k)', { fs: 9.5, fill: C.text });
  return svg(W, H, body);
}

function glaBlock() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Gated Linear Attention (GLA) Multi-Head Sequence Mixer Block', { fs: 13.5, fw: 700 });

  const stages = [
    ['1. QKV + Conv4', 'W_qkv x_t → Conv4', '+ SiLU activation', C.slate, 14, 114],
    ['2. L2 Norm + a_t', 'q̂_t, k̂_t unit norm', 'a_{t,h} = σ(w_aᵀ x_t)', C.blue, 140, 122],
    ['3. Chunked SSD', 'S_t = a_t S + v_t k̂_tᵀ', 'H heads × (d_h×d_h)', C.green, 274, 126],
    ['4. RMSNorm Gate', 'RMS(y) ⊙ SiLU(W_g x)', '→ Output W_o', C.purple, 412, 114],
  ];
  stages.forEach(([title, l1, l2, col, x, w], i) => {
    body += rbox(x, 44, w, 88, col.bg, col.bd);
    body += tx(x + w / 2, 66, title, { fs: 10, fw: 700, fill: col.tx });
    body += tx(x + w / 2, 88, l1, { fs: 9, fill: C.text, mono: true });
    body += tx(x + w / 2, 108, l2, { fs: 9, fill: C.muted, mono: true });
    if (i < stages.length - 1) {
      const nx = stages[i + 1][4];
      body += arrow(x + w + 2, 88, nx - 2, 88);
    }
  });

  body += rbox(16, 150, 508, 82, C.amber.bg, C.amber.bd);
  body += tx(W / 2, 170, 'Why Query/Key L2 Normalization + RMSNorm Output Gate Are Essential', { fs: 11, fw: 700, fill: C.amber.tx });
  body += tx(W / 2, 192, '• Unit-norm k̂_t bounds each outer-product write ||v_t k̂_tᵀ||₂ = ||v_t||₂ without softmax normalization', { fs: 9.5, fill: C.text });
  body += tx(W / 2, 212, '• Per-head RMSNorm stabilizes accumulated S_t q̂_t scale before multiplying by the SiLU output gate', { fs: 9.5, fill: C.text });
  return svg(W, H, body);
}

// ============================================================================
// deltanet.md (8 diagrams)
// ============================================================================

function deltaReadBeforeWrite() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'The Delta Rule: One Online Gradient Step on 1/2 ||S k_t − v_t||²', { fs: 13.5, fw: 700 });

  const steps = [
    ['1. Read Memory', 'v̂_t = S_{t-1} k_t', 'Current prediction', C.blue, 14, 152],
    ['2. Compute Error', 'e_t = v_t − v̂_t', 'Surprise residual', C.amber, 190, 152],
    ['3. Rank-1 Update', 'S_t ← S + β_t e_t k_tᵀ', 'Targeted correction', C.green, 366, 160],
  ];
  steps.forEach(([title, eq, sub, col, x, w], i) => {
    body += rbox(x, 42, w, 86, col.bg, col.bd);
    body += tx(x + w / 2, 64, title, { fs: 10.5, fw: 700, fill: col.tx });
    body += tx(x + w / 2, 86, eq, { fs: 9.5, fw: 700, fill: C.text, mono: true });
    body += tx(x + w / 2, 106, sub, { fs: 9.5, fill: C.muted });
    if (i < 2) body += arrow(x + w + 2, 85, steps[i + 1][4] - 2, 85);
  });

  body += rbox(16, 146, 508, 86, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 166, 'Multiplicative Contraction Along Key Direction k_t:', { fs: 11, fw: 700, fill: C.text });
  body += tx(W / 2, 188, 'S_t = S_{t-1} ( I − β_t k_t k_tᵀ ) + β_t v_t k_tᵀ', { fs: 11, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(W / 2, 210, 'When ||k_t|| = 1 and β_t = 1:  S_t k_t = S_{t-1}(k_t − k_t) + v_t = v_t  (erases stale value in 1 step!)', { fs: 9.5, fill: C.green.tx });
  return svg(W, H, body);
}

function eigenvalueParity() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Spectrum of I − β_t k_t k_tᵀ for Unit Keys (||k_t||₂ = 1)', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 46, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 53, 'Eigenvalues:  λ = 1 on k_t^{⊥} (d_k − 1 dims untouched),   λ_k = 1 − β_t along k_t', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(W / 2, 71, 'Non-expansive stability condition:  |1 − β_t| ≤ 1   ⇔   0 ≤ β_t ≤ 2', { fs: 10, fill: C.blue.tx, mono: true });

  const modes = [
    ['0 < β_t < 1 (EMA)', 'λ_k ∈ (0, 1)', 'Blends old & new value', 'Standard DeltaNet: β_t = σ(·)', C.blue],
    ['β_t = 1 (Projector)', 'λ_k = 0 (idempotent)', 'Exact rank-1 erasure', 'S_t k_t = v_t in one step', C.green],
    ['β_t = 2 (Householder)', 'λ_k = −1 (reflection!)', 'Involutory: H² = I', 'Solves parity / flip-flop!', C.purple],
  ];
  modes.forEach(([title, l1, l2, l3, col], i) => {
    const x = 16 + i * 172;
    body += rbox(x, 96, 164, 140, col.bg, col.bd);
    body += tx(x + 82, 116, title, { fs: 10.5, fw: 700, fill: col.tx });
    body += tx(x + 82, 140, l1, { fs: 10, fw: 700, fill: C.text, mono: true });
    body += tx(x + 82, 164, l2, { fs: 9.5, fill: C.text });
    body += tx(x + 82, 188, l3, { fs: 9, fill: C.muted });
    if (i === 2) {
      body += tx(x + 82, 212, 'Positive a_t > 0 cannot flip!', { fs: 9, fw: 700, fill: C.purple.tx });
    }
  });
  return svg(W, H, body);
}

function signalErosion() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'The Cost of Overwriting: Collateral Signal Erosion Across m Writes', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 52, C.amber.bg, C.amber.bd);
  body += tx(W / 2, 54, 'Each subsequent random unit key k_s (β=1) multiplies stored pair (k_i, v_i) by (I − k_s k_sᵀ)', { fs: 10, fw: 700, fill: C.amber.tx });
  body += tx(W / 2, 74, 'Expected signal after m writes:  E[S_t k_i] = (1 − 1/d_k)^m v_i ≈ exp(−m / d_k) v_i', { fs: 10, fw: 700, fill: C.text, mono: true });

  const cards = [
    ['m = d_k / 2 writes', 'Signal ≈ e⁻⁰·⁵ = 60.7%', 'Strong target recall', C.green],
    ['m = d_k writes', 'Signal ≈ e⁻¹ = 36.8%', '1/e capacity horizon', C.blue],
    ['m = 2 d_k writes', 'Signal ≈ e⁻² = 13.5%', 'Eroded by projections', C.coral],
  ];
  cards.forEach(([title, sig, sub, col], i) => {
    const x = 16 + i * 172;
    body += rbox(x, 102, 164, 84, col.bg, col.bd);
    body += tx(x + 82, 124, title, { fs: 11, fw: 700, fill: col.tx });
    body += tx(x + 82, 148, sig, { fs: 10, fw: 700, fill: C.text, mono: true });
    body += tx(x + 82, 170, sub, { fs: 9.5, fill: C.muted });
  });

  body += rbox(16, 198, 508, 38, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 217, 'Delta rule excels at rebinding ≤ d_k keys; additive sums erode slower when m > d_k', { fs: 10, fw: 600, fill: C.text });
  return svg(W, H, body);
}

function gatedDeltanetModes() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Gated DeltaNet (Yang et al. 2024): Combining a_t Decay with β_t Erasure', { fs: 13.5, fw: 700 });

  body += rbox(16, 36, 508, 44, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 58, 'S_t = a_t S_{t-1} (I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ   (a_t, β_t ∈ (0, 1))', { fs: 10.5, fw: 700, fill: C.blue.tx, mono: true });

  const modes = [
    ['1. Global Topic Flush', 'a_t → 0', 'Wipes entire matrix S_{t-1}', 'At document boundary', C.coral],
    ['2. Surgical Key Update', 'a_t ≈ 1,  β_t → 1', 'Erases ONLY direction k_t', 'Keeps d_k − 1 orthogonal keys', C.green],
    ['3. Gentle Accumulation', 'a_t ≈ 1,  β_t ≈ 0.2', 'Low-rate associative blend', 'Reduces collateral erosion', C.purple],
  ];
  modes.forEach(([title, gate, act, sub, col], i) => {
    const x = 16 + i * 172;
    body += rbox(x, 94, 164, 142, col.bg, col.bd);
    body += tx(x + 82, 114, title, { fs: 10.5, fw: 700, fill: col.tx });
    body += rbox(x + 14, 128, 136, 26, '#FFFFFF', col.bd, 4, 1.0);
    body += tx(x + 82, 141, gate, { fs: 10, fw: 700, fill: col.tx, mono: true });
    body += tx(x + 82, 174, act, { fs: 9.5, fw: 600, fill: C.text });
    body += tx(x + 82, 198, sub, { fs: 9, fill: C.muted });
  });
  return svg(W, H, body);
}

function wyTriangularSolve() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Chunked WY Representation: m × m Unit Lower-Triangular Solve', { fs: 13.5, fw: 700 });

  const steps = [
    ['1. Gram Matrix', 'M = I + diag(β) tril(L⊙KKᵀ)', 'm×m unit lower-tri', C.blue, 14, 158],
    ['2. Triangular Solve', 'M W = diag(β) V', 'M Z = diag(β)(γ ⊙ K)', C.purple, 190, 156],
    ['3. Corrected Value', 'U = W − Z S_0ᵀ', 'Batched GEMM readout', C.green, 364, 162],
  ];
  steps.forEach(([title, l1, l2, col, x, w], i) => {
    body += rbox(x, 42, w, 86, col.bg, col.bd);
    body += tx(x + w / 2, 64, title, { fs: 10.5, fw: 700, fill: col.tx });
    body += tx(x + w / 2, 86, l1, { fs: 9, fw: 700, fill: C.text, mono: true });
    body += tx(x + w / 2, 106, l2, { fs: 9, fill: C.muted, mono: true });
    if (i < 2) body += arrow(x + w + 2, 85, steps[i + 1][4] - 2, 85);
  });

  body += rbox(16, 146, 508, 86, C.amber.bg, C.amber.bd);
  body += tx(W / 2, 166, 'Why Chunked WY Unlocks Tensor-Core Speed for Data-Dependent Transitions', { fs: 11, fw: 700, fill: C.amber.tx });
  body += tx(W / 2, 188, '• Within each length-m chunk (m = 32 or 64), M is inverted across all B × H chunks in parallel', { fs: 9.5, fill: C.text });
  body += tx(W / 2, 208, '• Once U is found, chunk outputs Y and boundary state S_m use the exact same GEMMs as GLA!', { fs: 9.5, fill: C.text });
  return svg(W, H, body);
}

function twoBlockEmulation() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Why 1-Block Benchmarks Isolate the Mixing Law: Depth vs. In-Layer Delta', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 198, C.amber.bg, C.amber.bd);
  body += tx(139, 58, '2-Block Additive GLA Stack', { fs: 11, fw: 700, fill: C.amber.tx });
  body += rbox(30, 74, 218, 48, '#FFFFFF', C.amber.bd, 5, 1.1);
  body += tx(139, 92, 'Block 1 reads v̂_t = S_{t-1} k_t', { fs: 9.5, fw: 700, fill: C.text, mono: true });
  body += tx(139, 108, 'Passes v̂_t up residual stream', { fs: 9, fill: C.muted });
  body += rbox(30, 132, 218, 48, '#FFFFFF', C.amber.bd, 5, 1.1);
  body += tx(139, 150, 'Block 2 writes residual v_t − v̂_t', { fs: 9.5, fw: 700, fill: C.amber.tx, mono: true });
  body += tx(139, 166, 'Spends 2 layers for 1 delta step!', { fs: 9, fill: C.coral.tx });
  body += tx(139, 206, '1-Block GLA, rebound keys: ≈88%', { fs: 10, fw: 700, fill: C.coral.tx });

  body += rbox(278, 38, 246, 198, C.green.bg, C.green.bd);
  body += tx(401, 58, '1-Block GatedDeltaNet', { fs: 11, fw: 700, fill: C.green.tx });
  body += rbox(292, 74, 218, 106, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(401, 98, 'Performs Read-Before-Write', { fs: 10.5, fw: 700, fill: C.green.tx });
  body += tx(401, 120, 'INSIDE every recurrent step t:', { fs: 9.5, fill: C.text });
  body += tx(401, 144, 'S_t = a_t S(I − β_t k kᵀ) + β_t v kᵀ', { fs: 9.5, fw: 700, fill: C.text, mono: true });
  body += tx(401, 166, 'Frees deeper layers for reasoning!', { fs: 9, fill: C.muted });
  body += tx(401, 206, '1-Block GatedDeltaNet: 98–99%', { fs: 10, fw: 700, fill: C.green.tx });
  return svg(W, H, body);
}

function explicitVsImplicitStep() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Explicit SGD Step (DeltaNet) vs. Implicit Proximal Step (Longhorn)', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 198, C.blue.bg, C.blue.bd);
  body += tx(139, 58, 'Explicit Forward Step (DeltaNet)', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(139, 82, 'Linearizes loss at OLD state S_{t-1}:', { fs: 9.5, fill: C.muted });
  body += tx(139, 104, 'S_t = S_{t-1} − β_t (S k_t − v_t) k_tᵀ', { fs: 9.5, fw: 700, fill: C.text, mono: true });
  body += rbox(30, 122, 218, 52, '#FFFFFF', C.blue.bd, 5, 1.1);
  body += tx(139, 140, 'λ_k = 1 − β_t ||k_t||²', { fs: 10, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(139, 158, 'Requires unit-norm ||k_t|| = 1', { fs: 9.5, fill: C.coral.tx });
  body += tx(139, 200, 'Explodes if β_t ||k_t||² > 2!', { fs: 10, fw: 700, fill: C.coral.tx });

  body += rbox(278, 38, 246, 198, C.green.bg, C.green.bd);
  body += tx(401, 58, 'Implicit Proximal Step (Longhorn)', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(401, 82, 'Evaluates loss at NEW state S_t:', { fs: 9.5, fill: C.muted });
  body += tx(401, 104, 'β̃_t = β_t / (1 + β_t ||k_t||²)', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += rbox(292, 122, 218, 52, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(401, 140, 'λ_k = 1 / (1 + β_t ||k_t||²) ∈ (0, 1)', { fs: 9.5, fw: 700, fill: C.green.tx, mono: true });
  body += tx(401, 158, 'Stable for ANY ||k_t|| and β_t > 0!', { fs: 9.5, fill: C.green.tx });
  body += tx(401, 200, 'Sherman-Morrison exact closed form', { fs: 10, fw: 700, fill: C.green.tx });
  return svg(W, H, body);
}

function fastSlowWeights() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Two-Timescale Learning: Inner-Loop Fast Weights vs. Outer-Loop Slow Weights', { fs: 13.5, fw: 700 });

  body += rbox(16, 40, 246, 192, C.green.bg, C.green.bd);
  body += tx(139, 60, 'Inner Loop: Fast Weights S_t', { fs: 11.5, fw: 700, fill: C.green.tx });
  body += tx(139, 82, 'Runs at inference time inside 1 prompt', { fs: 9.5, fill: C.muted });
  body += rbox(30, 98, 218, 52, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(139, 116, 'min_{S} ∑ γ_{t,s} ||S k_s − v_s||²', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(139, 134, 'Updated at every token t = 1..T', { fs: 9.5, fill: C.green.tx });
  body += tx(139, 172, '• GLA: 1 gradient step from S = 0', { fs: 9.5, fill: C.text });
  body += tx(139, 192, '• DeltaNet: Online SGD from S_{t-1}', { fs: 9.5, fill: C.text });
  body += tx(139, 212, '• Longhorn: Implicit proximal step', { fs: 9.5, fill: C.text });

  body += rbox(278, 40, 246, 192, C.purple.bg, C.purple.bd);
  body += tx(401, 60, 'Outer Loop: Slow Weights θ', { fs: 11.5, fw: 700, fill: C.purple.tx });
  body += tx(401, 82, 'Trained by AdamW across millions of docs', { fs: 9.5, fill: C.muted });
  body += rbox(292, 98, 218, 52, '#FFFFFF', C.purple.bd, 5, 1.1);
  body += tx(401, 116, 'θ = {W_q, W_k, W_v, W_a, W_β, W_o}', { fs: 9.5, fw: 700, fill: C.text, mono: true });
  body += tx(401, 134, 'Frozen during inference rollout', { fs: 9.5, fill: C.purple.tx });
  body += tx(401, 172, '• Learns key features k_t that minimize', { fs: 9.5, fill: C.text });
  body += tx(401, 192, '  cross-talk and step sizes β_t that', { fs: 9.5, fill: C.text });
  body += tx(401, 212, '  decide when to overwrite fast memory', { fs: 9.5, fill: C.text });
  return svg(W, H, body);
}

// ============================================================================
// hybrids.md (8 diagrams)
// ============================================================================

function tableLookupCeiling() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'The In-Context Table-Lookup Ceiling of Fixed-Size Recurrent States', { fs: 13.5, fw: 700 });

  body += rbox(16, 40, 174, 88, C.blue.bg, C.blue.bd);
  body += tx(103, 62, 'Prompt Lookup Table', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(103, 84, 'm = 64 (key, val) pairs', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(103, 106, 'Unknown future queries q', { fs: 9.5, fill: C.muted });

  body += arrow(192, 84, 218, 84);

  body += rbox(220, 40, 136, 88, C.coral.bg, C.coral.bd);
  body += tx(288, 62, 'Pure Recurrent S_t', { fs: 11, fw: 700, fill: C.coral.tx });
  body += tx(288, 84, 'Fixed H × d_k × d_v', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(288, 106, 'Saturates when m > d_k!', { fs: 9.5, fw: 700, fill: C.coral.tx });

  body += arrow(358, 84, 384, 84);

  body += rbox(386, 40, 138, 88, C.green.bg, C.green.bd);
  body += tx(455, 62, '3:1 Hybrid Stack', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(455, 84, '3 DeltaNet + 1 Attn', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(455, 106, '100% exact lookup!', { fs: 9.5, fw: 700, fill: C.green.tx });

  body += rbox(16, 146, 508, 86, C.slate.bg, C.slate.bd);
  body += tx(W / 2, 166, 'Why Pre-Query Compression Cannot Match Post-Query Random Access', { fs: 11, fw: 700, fill: C.text });
  body += tx(W / 2, 188, '• Fixed state S_t must compress all m pairs BEFORE knowing which key is queried', { fs: 9.5, fill: C.text });
  body += tx(W / 2, 208, '• One sparse attention layer keeps the token table for exact post-query lookup', { fs: 9.5, fill: C.blue.tx });
  return svg(W, H, body);
}

function pigeonholeVsAlgebra() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Why Linear Algebra Breaks First: Bit Counting vs. Rank Capacity', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 508, 46, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 55, 'Setup: m = 64 keys, M = 4 values, H = 8 heads, d_k = d_v = 4', { fs: 10.5, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(W / 2, 73, 'Why do 256 float32s (8,192 bits) in two blocks recall only about half of 64 pairs?', { fs: 10, fill: C.text });

  const bars = [
    ['Value Table Entropy (m log₂ M)', '64 × 2 = 128 bits', C.green],
    ['Full Table (log₂ C(128,64) + 128)', '≈ 124 + 128 = 252 bits', C.blue],
    ['Raw Float32 State (2 × 128 × 32)', '8,192 bits (>30× larger!)', C.purple],
    ['Linear-Algebraic Rank Limit (d_k = 4)', 'Only 4 orthogonal keys / head!', C.coral],
  ];
  bars.forEach(([lbl, val, col], i) => {
    const y = 98 + i * 36;
    body += tx(20, y + 13, lbl, { fs: 10, fw: 700, anchor: 'start', fill: C.text });
    body += rbox(232, y, 286, 24, col.bg, col.bd, 4, 1.3);
    body += tx(375, y + 13, val, { fs: 9.5, fw: 700, fill: col.tx, mono: true });
  });
  return svg(W, H, body);
}

function welchBoundGeometry() {
  const W = 540, H = 248;
  let body = '';
  body += tx(W / 2, 18, 'Limit 2: Gram Matrix Rank and Welch\'s Overlap Bound (1974)', { fs: 13.5, fw: 700 });

  body += rbox(16, 40, 246, 130, C.blue.bg, C.blue.bd);
  body += tx(139, 60, 'Gram Matrix Rank Deficit', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(139, 84, 'G = ∑_{i=1}^{m} k_i k_iᵀ ∈ R^{d_k×d_k}', { fs: 10.5, fw: 700, fill: C.text, mono: true });
  body += tx(139, 108, 'trace(G) = m,   rank(G) ≤ d_k', { fs: 10, fill: C.blue.tx, mono: true });
  body += tx(139, 132, 'Cauchy-Schwarz on eigenvalues λ_r:', { fs: 9.5, fill: C.muted });
  body += tx(139, 152, '||G||_{F}² = ∑ λ_r² ≥ m² / d_k', { fs: 10, fw: 700, fill: C.text, mono: true });

  body += rbox(278, 40, 246, 130, C.coral.bg, C.coral.bd);
  body += tx(401, 60, 'Welch Lower Bound (m > d_k)', { fs: 11, fw: 700, fill: C.coral.tx });
  body += tx(401, 84, '1/(m(m−1)) ∑_{i≠j} (k_iᵀ k_j)²', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(401, 108, '≥  (m − d_k) / ( d_k (m − 1) )', { fs: 10.5, fw: 700, fill: C.coral.tx, mono: true });
  body += tx(401, 132, 'Max pairwise overlap also bounded:', { fs: 9.5, fill: C.muted });
  body += tx(401, 152, 'max |k_iᵀ k_j| ≥ √((m−d_k)/(d_k(m−1)))', { fs: 9, fw: 700, fill: C.text, mono: true });

  body += rbox(16, 184, 508, 50, C.amber.bg, C.amber.bd);
  body += tx(W / 2, 203, 'Universal Geometric Ceiling for Linear Associative Memories (GLA, DeltaNet)', { fs: 10.5, fw: 700, fill: C.amber.tx });
  body += tx(W / 2, 221, 'Readout y = S q lives in R^{d_k}: no optimizer can push key overlap below Welch\'s bound!', { fs: 9.5, fill: C.text });
  return svg(W, H, body);
}

function welchDk4VsDk16() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Why Trained Embeddings Cannot Rescue d_k = 4 at m = 64 Keys', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 198, C.coral.bg, C.coral.bd);
  body += tx(139, 58, 'Small Key Space: d_k = 4, m = 64', { fs: 11, fw: 700, fill: C.coral.tx });
  body += tx(139, 82, 'Random unit keys: 1/d_k = 0.2500', { fs: 10, fill: C.text, mono: true });
  body += tx(139, 104, 'Welch lower bound: 60/(4×63) = 0.2381', { fs: 10, fw: 700, fill: C.coral.tx, mono: true });
  body += rbox(30, 122, 218, 52, '#FFFFFF', C.coral.bd, 5, 1.1);
  body += tx(139, 140, 'Floor is 95.2% of random overlap!', { fs: 10, fw: 700, fill: C.coral.tx });
  body += tx(139, 158, 'Max overlap ≥ √0.2381 = 0.488', { fs: 9.5, fill: C.text, mono: true });
  body += tx(139, 196, '• Backprop has <5% room to reduce MSE', { fs: 9.5, fill: C.text });
  body += tx(139, 216, '• GG (d_k=4) plateaus at ≈50% recall', { fs: 10, fw: 700, fill: C.coral.tx });

  body += rbox(278, 38, 246, 198, C.green.bg, C.green.bd);
  body += tx(401, 58, 'Expanded Key Space: d_k = 16, m = 64', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(401, 82, 'Random unit keys: 1/d_k = 0.0625', { fs: 10, fill: C.text, mono: true });
  body += tx(401, 104, 'Welch lower bound: 48/(16×63) = 0.0476', { fs: 10, fw: 700, fill: C.green.tx, mono: true });
  body += rbox(292, 122, 218, 52, '#FFFFFF', C.green.bd, 5, 1.1);
  body += tx(401, 140, '5.0× lower mean squared overlap!', { fs: 10, fw: 700, fill: C.green.tx });
  body += tx(401, 158, 'Max overlap floor drops to 0.218', { fs: 9.5, fill: C.text, mono: true });
  body += tx(401, 196, '• Same d=32 model width (H=2, d_k=16)', { fs: 9.5, fill: C.text });
  body += tx(401, 216, '• GG 4x state reaches ≈95% at m = 64!', { fs: 10, fw: 700, fill: C.green.tx });
  return svg(W, H, body);
}

function fourRecallLayouts() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Four 2-Block Architectures on Associative Recall (d = 32)', { fs: 13.5, fw: 700 });

  const archs = [
    ['1. GG (Small)', 'DeltaNet → DeltaNet', 'H=8, d_k=4', '128 floats/layer', 'm=64 acc: ≈50%', C.coral],
    ['2. GG (4x State)', 'DeltaNet → DeltaNet', 'H=2, d_k=16', '512 floats/layer', 'm=64 acc: ≈95%', C.blue],
    ['3. AA (Full Attn)', 'Attn → Attn (no RoPE)', 'H=2, d_k=16', '2 full KV caches', 'm=64 acc: 100%', C.purple],
    ['4. GA (Hybrid)', 'DeltaNet → Attention', '1 state + 1 KV', '50% KV cache!', 'm=64 acc: 100%', C.green],
  ];
  archs.forEach(([title, l1, l2, l3, res, col], i) => {
    const x = 14 + i * 130;
    body += rbox(x, 38, 120, 198, col.bg, col.bd);
    body += tx(x + 60, 58, title, { fs: 10.5, fw: 700, fill: col.tx });
    body += tx(x + 60, 82, l1, { fs: 9, fw: 600, fill: C.text });
    body += tx(x + 60, 104, l2, { fs: 9.5, fill: C.text, mono: true });
    body += tx(x + 60, 126, l3, { fs: 9, fill: C.muted });
    body += rbox(x + 8, 152, 104, 30, '#FFFFFF', col.bd, 4, 1.1);
    body += tx(x + 60, 167, res, { fs: 9, fw: 700, fill: col.tx, mono: true });
    body += tx(x + 60, 206, i === 3 ? 'Best memory/acc!' : i === 0 ? 'Welch limited' : 'High capacity', { fs: 9, fw: 600, fill: col.tx });
  });
  return svg(W, H, body);
}

function conv4KeyValueBinding() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Why Width-4 Short Causal Convolution Is Essential for Matrix States', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 508, 44, C.blue.bg, C.blue.bd);
  body += tx(W / 2, 55, 'In stream [k_1, v_1, k_2, v_2, ..., q], key k_i arrives at t−1 and value v_i at t', { fs: 10, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(W / 2, 72, 'To store (k_i, v_i), outer product v_t k_tᵀ at step t must see BOTH x_{t-1} = k_i and x_t = v_i!', { fs: 9.5, fill: C.text });

  body += rbox(16, 94, 246, 144, C.coral.bg, C.coral.bd);
  body += tx(139, 114, 'Without Short Conv (conv_width = 1)', { fs: 11, fw: 700, fill: C.coral.tx });
  body += tx(139, 138, 'At step t (token x_t = v_i):', { fs: 9.5, fill: C.muted });
  body += tx(139, 158, 'k_t = W_k x_t,   v_t = W_v x_t', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(139, 182, '• k_t depends ONLY on value token v_i!', { fs: 9.5, fill: C.coral.tx });
  body += tx(139, 202, '• Never binds v_i to preceding key k_i', { fs: 9.5, fill: C.text });
  body += tx(139, 222, '• GG recall falls to 0.36–0.92', { fs: 10, fw: 700, fill: C.coral.tx });

  body += rbox(278, 94, 246, 144, C.green.bg, C.green.bd);
  body += tx(401, 114, 'With Width-4 Causal Conv (Default)', { fs: 11, fw: 700, fill: C.green.tx });
  body += tx(401, 138, 'Depthwise conv mixes (x_{t-3}, …, x_{t}):', { fs: 9.5, fill: C.muted });
  body += tx(401, 158, 'k_t = f_k(x_{t-1}),  v_t = f_v(x_t)', { fs: 10, fw: 700, fill: C.text, mono: true });
  body += tx(401, 182, '• Step t writes v_i k_iᵀ into S_t!', { fs: 9.5, fill: C.green.tx });
  body += tx(401, 202, '• Acts as an implicit induction head', { fs: 9.5, fill: C.text });
  body += tx(401, 222, '• GG recall is 0.48–0.99 (+6 to 38 pts)', { fs: 10, fw: 700, fill: C.green.tx });
  return svg(W, H, body);
}

function perplexityVsRetrieval() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Pitfall: Short-Window Perplexity Hides Long-Context Retrieval Deficits', { fs: 13.5, fw: 700 });

  body += rbox(16, 38, 246, 150, C.blue.bg, C.blue.bd);
  body += tx(139, 58, 'WikiText-103 (T = 128), nats/byte', { fs: 11, fw: 700, fill: C.blue.tx });
  body += tx(139, 80, 'Dominated by local n-gram statistics', { fs: 9.5, fill: C.muted });
  body += tx(139, 106, 'GGGG (4× DeltaNet):  1.32–1.33', { fs: 10, fw: 700, fill: C.green.tx, mono: true });
  body += tx(139, 128, 'GGGA (3:1 Hybrid):   1.33–1.34', { fs: 10, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(139, 150, 'AAAA (4× Attention): 1.38     ', { fs: 10, fw: 700, fill: C.coral.tx, mono: true });
  body += tx(139, 172, 'Conv4 + decay give GGGG strong local bias', { fs: 9, fill: C.muted });

  body += rbox(278, 38, 246, 150, C.amber.bg, C.amber.bd);
  body += tx(401, 58, '2-Block Recall (m = 64 Pairs)', { fs: 11, fw: 700, fill: C.amber.tx });
  body += tx(401, 80, 'Requires exact long-range key lookup', { fs: 9.5, fill: C.muted });
  body += tx(401, 106, 'GG (d_k=4):  ≈50% acc (Welch limit!)', { fs: 10, fw: 700, fill: C.coral.tx, mono: true });
  body += tx(401, 128, 'GA (1:1):    100% acc (50% less KV!)', { fs: 10, fw: 700, fill: C.green.tx, mono: true });
  body += tx(401, 150, 'AA (2×):     100% acc (100% KV cost)', { fs: 10, fw: 700, fill: C.blue.tx, mono: true });
  body += tx(401, 172, '1 sparse attention layer removes ceiling', { fs: 9, fill: C.muted });

  body += rbox(16, 200, 508, 38, C.green.bg, C.green.bd);
  body += tx(W / 2, 219, 'Pair perplexity with synthetic retrieval probes: hybrids keep both fluency and recall!', { fs: 10, fw: 700, fill: C.green.tx });
  return svg(W, H, body);
}

function architectureDecisionTree() {
  const W = 540, H = 252;
  let body = '';
  body += tx(W / 2, 18, 'Practitioner\'s Guide: Choosing a Sequence Architecture', { fs: 13.5, fw: 700 });

  const cards = [
    ['1. Unbounded Streaming & Edge', 'Audio, sensor DSP, robotics, RL', 'Strict O(1) state & O(1) latency/step', '→ Mamba / S4D (or GRU for small T)', C.blue],
    ['2. Long-Context LLM / Code / RAG', '32K–1M context + exact citation/copy', 'Fast prefill + bounded KV + exact lookup', '→ 3:1 or 7:1 Hybrid (DeltaNet/Mamba + GQA)', C.green],
    ['3. High-Capacity Pure Linear State', 'Zero KV cache allowed, high recall', 'Must beat Welch bound (m ≤ d_k)', '→ GatedDeltaNet with d_k ≥ 128 + Conv4', C.purple],
    ['4. Short Dense Sequences (T ≤ 2K)', 'Standard classification / short prompts', 'KV cache fits easily in SRAM/HBM', '→ Standard Transformer with FlashAttention', C.amber],
  ];
  cards.forEach(([title, l1, l2, rec, col], idx) => {
    const colIdx = idx % 2;
    const rowIdx = Math.floor(idx / 2);
    const x = 16 + colIdx * 260;
    const y = 36 + rowIdx * 106;
    body += rbox(x, y, 248, 96, col.bg, col.bd);
    body += tx(x + 124, y + 18, title, { fs: 10.5, fw: 700, fill: col.tx });
    body += tx(x + 124, y + 38, l1, { fs: 9.5, fill: C.text });
    body += tx(x + 124, y + 54, l2, { fs: 9.5, fill: C.muted });
    body += rbox(x + 10, y + 66, 228, 22, '#FFFFFF', col.bd, 4, 1.0);
    body += tx(x + 124, y + 77, rec, { fs: 9, fw: 700, fill: col.tx });
  });
  return svg(W, H, body);
}

export const diagrams = {
  'ch12-mrnn-kv-vs-state': kvVsState,
  'ch12-mrnn-two-obstacles': twoObstacles,
  'ch12-mrnn-roadmap': roadmap,
  'ch12-mrnn-stress-tests': stressTests,
  'ch12-mrnn-grammar-decay': grammarDecay,
  'ch12-mrnn-jacobian-chain': jacobianChain,
  'ch12-mrnn-gate-halflife': gateHalflife,
  'ch12-mrnn-silent-storage': silentStorage,
  'ch12-mrnn-lstm-vs-gru': lstmVsGru,
  'ch12-mrnn-zero-bias-underflow': zeroBiasUnderflow,
  'ch12-mrnn-keep-gate-init': keepGateInit,
  'ch12-mrnn-seqlm-backbone': seqlmBackbone,
  'ch12-mrnn-transduction-bottleneck': transductionBottleneck,
  'ch12-mrnn-conditioned-decoder': conditionedDecoder,
  'ch12-mrnn-sutskever-reversal': sutskeverReversal,
  'ch12-mrnn-chrf-pipeline': chrfPipeline,
  'ch12-mrnn-bleu-formula': bleuFormula,
  'ch12-mrnn-reversal-task': reversalTask,
  'ch12-mrnn-bit-budget-bars': bitBudgetBars,
  'ch12-mrnn-two-escape-routes': twoEscapeRoutes,
  'ch12-mrnn-seq-vs-parallel-gpu': seqVsParallelGpu,
  'ch12-mrnn-scalar-ema-impulse': scalarEmaImpulse,
  'ch12-mrnn-affine-precompute': affinePrecompute,
  'ch12-mrnn-gru-to-mingru': gruToMingru,
  'ch12-mrnn-depth-restores-gating': depthRestoresGating,
  'ch12-mrnn-affine-combine-op': affineCombineOp,
  'ch12-mrnn-complexity-crossover': complexityCrossover,
  'ch12-mrnn-timescale-hierarchy': timescaleHierarchy,
  'ch12-mrnn-continuous-ssm-block': continuousSsmBlock,
  'ch12-mrnn-diagonal-ode-bank': diagonalOdeBank,
  'ch12-mrnn-zoh-step-hold': zohStepHold,
  'ch12-mrnn-delta-dilation-table': deltaDilationTable,
  'ch12-mrnn-fft-conv-pipeline': fftConvPipeline,
  'ch12-mrnn-real-vs-complex-basis': realVsComplexBasis,
  'ch12-mrnn-recall-2x2-grid': recall2x2Grid,
  'ch12-mrnn-mamba-unification': mambaUnification,
  'ch12-mrnn-softplus-sigmoid-duality': softplusSigmoidDuality,
  'ch12-mrnn-selective-ssm-dataflow': selectiveSsmDataflow,
  'ch12-mrnn-4d-scan-memory': scan4dMemory,
  'ch12-mrnn-adjoint-reverse-scan': adjointReverseScan,
  'ch12-mrnn-state-contribution-formula': stateContributionFormula,
  'ch12-mrnn-vector-vs-matrix-state': vectorVsMatrixState,
  'ch12-mrnn-orthogonal-vs-crosstalk': orthogonalVsCrosstalk,
  'ch12-mrnn-left-vs-right-assoc': leftVsRightAssoc,
  'ch12-mrnn-retrieval-error-law': retrievalErrorLaw,
  'ch12-mrnn-correlated-key-cone': correlatedKeyCone,
  'ch12-mrnn-sliding-window-decay': slidingWindowDecay,
  'ch12-mrnn-ssd-duality': ssdDuality,
  'ch12-mrnn-gla-block': glaBlock,
  'ch12-mrnn-delta-read-before-write': deltaReadBeforeWrite,
  'ch12-mrnn-eigenvalue-parity': eigenvalueParity,
  'ch12-mrnn-signal-erosion': signalErosion,
  'ch12-mrnn-gated-deltanet-modes': gatedDeltanetModes,
  'ch12-mrnn-wy-triangular-solve': wyTriangularSolve,
  'ch12-mrnn-two-block-emulation': twoBlockEmulation,
  'ch12-mrnn-explicit-vs-implicit-step': explicitVsImplicitStep,
  'ch12-mrnn-fast-slow-weights': fastSlowWeights,
  'ch12-mrnn-table-lookup-ceiling': tableLookupCeiling,
  'ch12-mrnn-pigeonhole-vs-algebra': pigeonholeVsAlgebra,
  'ch12-mrnn-welch-bound-geometry': welchBoundGeometry,
  'ch12-mrnn-welch-dk4-vs-dk16': welchDk4VsDk16,
  'ch12-mrnn-four-recall-layouts': fourRecallLayouts,
  'ch12-mrnn-conv4-key-value-binding': conv4KeyValueBinding,
  'ch12-mrnn-perplexity-vs-retrieval': perplexityVsRetrieval,
  'ch12-mrnn-architecture-decision-tree': architectureDecisionTree,
};
