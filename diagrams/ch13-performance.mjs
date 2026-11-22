// Slide diagrams for chapter_computational-performance (§13 Computational Performance)
import { C, TOKENS, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// Fit font-size so text width never exceeds maxW (with safety margin for bold/mono/caps)
function fitText(x, y, text, maxW, o = {}) {
  const baseFs = o.fs || 13;
  const fw = o.fw || 600;
  const mono = !!o.mono;
  const factor = mono ? 0.65 : (fw >= 700 ? 0.60 : 0.56);
  const est = text.length * factor;
  const fs = est * baseFs > maxW ? Math.max(9.5, +((maxW / est)).toFixed(1)) : baseFs;
  return tx(x, y, text, { ...o, fs, fw });
}

// Reusable card helper: guarantees text stays strictly inside box borders
function card(x, y, w, h, fill, stroke, title, lines, titleFill = C.ink, monoLines = false) {
  let o = box(x, y, w, h, fill, stroke, 8, 2);
  const padW = w - 20;
  const n = lines.length;
  const topPad = n <= 1 ? 22 : (n === 2 ? 20 : 18);
  const step = n <= 1 ? 24 : (n === 2 ? 22 : (n === 3 ? 20 : 18.5));
  o += fitText(x + w / 2, y + topPad, title, padW, { fs: 14, fw: 700, fill: titleFill });
  lines.forEach((ln, i) => {
    const isObj = typeof ln === 'object';
    const text = isObj ? ln.t : ln;
    const mono = isObj ? ln.mono : monoLines;
    const col = isObj && ln.fill ? ln.fill : C.ink;
    const fs = isObj && ln.fs ? ln.fs : 12.5;
    const fw = isObj && ln.fw ? ln.fw : 600;
    o += fitText(x + w / 2, y + topPad + 21 + i * step, text, padW, { mono, fs, fw, fill: col });
  });
  return o;
}

// Reusable full-width footer box
function footerBox(x, y, w, h, fill, stroke, lines) {
  let o = box(x, y, w, h, fill, stroke, 8, 1.8);
  const padW = w - 24;
  const n = lines.length;
  const topPad = n === 1 ? h / 2 : (n === 2 ? 19 : 17);
  const step = n === 2 ? 22 : 19;
  lines.forEach((ln, i) => {
    const isObj = typeof ln === 'object';
    const text = isObj ? ln.t : ln;
    const mono = isObj ? !!ln.mono : false;
    const col = isObj && ln.fill ? ln.fill : C.ink;
    const fs = isObj && ln.fs ? ln.fs : 13;
    const fw = isObj && ln.fw ? ln.fw : 600;
    o += fitText(x + w / 2, y + topPad + i * step, text, padW, { mono, fs, fw, fill: col });
  });
  return o;
}

const header = (W, title) => fitText(W / 2, 18, title, W - 32, { fs: 15, fw: 700, fill: C.ink });

// ── §13.0 index.md ──────────────────────────────────────────────────────

function fourBottlenecks() {
  const W = 560, H = 286;
  let o = header(W, 'Four Physical Limits on a Deep Learning Training Step');
  o += card(20, 36, 250, 92, C.lblue, C.blue, '1. Arithmetic Peak (P)', [
    { t: '165.2 TFLOP/s (bf16)', mono: true, fill: D.blue.dark, fw: 700 },
    '8192² GEMM: I ≈ 2,700 FLOP/B',
  ], D.blue.dark);
  o += card(290, 36, 250, 92, C.lamber, C.amber, '2. Memory Bandwidth (β)', [
    { t: '1.01 TB/s GDDR6X DRAM', mono: true, fill: D.orange.dark, fw: 700 },
    'x + 1: I = 0.25 → 0.25 TFLOP/s',
  ], D.orange.dark);
  o += card(20, 144, 250, 92, D.red.tint, D.red.base, '3. Host Dispatch (h)', [
    { t: '10–100 µs per Python op', mono: true, fill: D.red.dark, fw: 700 },
    'GPU idles between tiny kernels',
  ], D.red.dark);
  o += card(290, 144, 250, 92, D.purple.tint, C.purple, '4. Memory Capacity (C)', [
    { t: '16N = 15.1 GiB / 24 GiB', mono: true, fill: D.purple.dark, fw: 700 },
    '1B GPT state before activations',
  ], D.purple.dark);
  o += footerBox(20, 248, 520, 28, C.lgreen, C.green, [
    { t: 'Method: Count FLOPs & Bytes → Predict Regime → Measure → Explain Gap', fill: D.green.dark, fw: 700 },
  ]);
  return svg(W, H, o);
}

function threeFloors() {
  const W = 560, H = 280;
  let o = header(W, 'Three-Floor Time Bound & Roofline Ridge Point');
  o += card(18, 38, 168, 128, D.red.tint, D.red.base, 'Floor 1: Host Launch', [
    { t: 't ≥ h', mono: true, fs: 14.5, fw: 700, fill: D.red.dark },
    '13–100 µs CPU enqueue',
    'Binds small tensors',
    'Fix: JIT / CUDA Graphs',
  ], D.red.dark);
  o += card(196, 38, 168, 128, C.lamber, C.amber, 'Floor 2: Bandwidth', [
    { t: 't ≥ bytes / β', mono: true, fs: 14.5, fw: 700, fill: D.orange.dark },
    'I < I* = 164 FLOP/B',
    'Binds norms, elementwise',
    'Fix: Operator fusion',
  ], D.orange.dark);
  o += card(374, 38, 168, 128, C.lblue, C.blue, 'Floor 3: Compute', [
    { t: 't ≥ FLOPs / P', mono: true, fs: 14.5, fw: 700, fill: D.blue.dark },
    'I > I* = 164 FLOP/B',
    'Binds large matmuls',
    'Fix: bf16 / fp8 / k GPUs',
  ], D.blue.dark);
  o += footerBox(18, 184, 524, 82, C.lgreen, C.green, [
    { t: 'Single-GPU Bound:  t ≳ max(h,  FLOPs / P,  bytes / β)', mono: true, fill: D.green.dark, fw: 700 },
    { t: 'Roofline Throughput ≤ min(P, I·β)   with Ridge I* = P / β', mono: true, fill: C.ink },
    { t: 'Multi-GPU Step adds Collective Tax:  + c·α + 2(k−1)/k · S · β_link', mono: true, fill: D.purple.dark },
  ]);
  return svg(W, H, o);
}

function hostConstants(fw) {
  const W = 560, H = 284;
  let o = header(W, 'Benchmark Workstation: Five Calibrated Hardware Constants');
  o += card(20, 36, 250, 74, C.lblue, C.blue, 'Peak Compute P (Tensor Cores)', [
    { t: '165.2 TF/s (bf16) · 82.6 (tf32)', mono: true, fill: D.blue.dark, fw: 700 },
  ], D.blue.dark);
  o += card(290, 36, 250, 74, C.lamber, C.amber, 'DRAM Bandwidth β (24 GiB)', [
    { t: '1.01 TB/s rated (~0.9 TB/s)', mono: true, fill: D.orange.dark, fw: 700 },
  ], D.orange.dark);
  o += card(20, 122, 250, 74, C.lgreen, C.green, 'Ridge Point I* = P / β', [
    { t: '164 FLOP/B (bf16) · 82 (tf32)', mono: true, fill: D.green.dark, fw: 700 },
  ], D.green.dark);
  o += card(290, 122, 250, 74, D.red.tint, D.red.base, 'Host Launch Latency h', [
    { t: fw === 'jax' ? '50–100 µs per un-jitted op' : '~13 µs per op', mono: true, fill: D.red.dark, fw: 700 },
  ], D.red.dark);
  o += card(20, 208, 520, 64, D.purple.tint, C.purple, 'Inter-GPU Collective Time β_link (4× RTX 4090 via Host PCIe 4.0)', [
    { t: '0.42 ns/byte (≈ 2.4 GB/s allreduce bus BW, no NVLink/P2P)', mono: true, fill: D.purple.dark, fw: 700 },
  ], D.purple.dark);
  return svg(W, H, o);
}

function roadmap() {
  const W = 560, H = 286;
  let o = header(W, 'Chapter 13 Progression: From One Kernel to Four GPUs');
  o += card(20, 36, 520, 68, C.lblue, C.blue, 'Stage 1: Single-Kernel Bounds & Silicon (§§13.1–13.2)', [
    '§13.1 Roofline, ridge I*, sync timing  ·  §13.2 Memory hierarchy, SMs, sweep C/β',
  ], D.blue.dark);
  o += arrow(280, 106, 280, 122, C.blue);
  o += card(20, 124, 520, 68, C.lamber, C.amber, 'Stage 2: Single-GPU Compilers & Memory (§§13.3–13.4)', [
    '§13.3 Tracing, 9× fusion, CUDA graphs  ·  §13.4 16N Adam state, bf16, remat, accum',
  ], D.orange.dark);
  o += arrow(280, 194, 280, 210, C.amber);
  o += card(20, 212, 520, 64, C.lgreen, C.green, 'Stage 3: Multi-GPU Scaling, Sharding & Case Study (§§13.5–13.8)', [
    '§§13.5–13.6 Ring allreduce & DDP  ·  §13.7 ZeRO/FSDP 16N/k  ·  §13.8 19M GPT 6×',
  ], D.green.dark);
  return svg(W, H, o);
}

// ── §13.1 performance-model.md ──────────────────────────────────────────

function matmulUtilization() {
  const W = 560, H = 276;
  let o = header(W, 'Same Chip, Same Primitive: Why Size Changes Utilization 50×');
  o += card(20, 38, 250, 168, D.red.tint, D.red.base, 'Small Matmul: 256 × 256 (bf16)', [
    { t: 'FLOPs: 3.4 × 10⁷', mono: true, fill: C.ink },
    { t: 'GPU busy: ~2.0 µs', mono: true, fill: D.green.dark },
    { t: 'Python launch h: ~13 µs', mono: true, fill: D.red.dark, fw: 700 },
    'Achieved: < 3 TFLOP/s (< 2% P)',
    'Regime: OVERHEAD-BOUND',
  ], D.red.dark);
  o += card(290, 38, 250, 168, C.lgreen, C.green, 'Large Matmul: 8192 × 8192 (bf16)', [
    { t: 'FLOPs: 1.10 × 10¹²', mono: true, fill: C.ink },
    { t: 'GPU busy: ~6.8 ms', mono: true, fill: D.green.dark, fw: 700 },
    { t: 'Python launch h: ~13 µs (<0.2%)', mono: true, fill: C.muted },
    'Achieved: ~162 TFLOP/s (~98% P)',
    'Regime: COMPUTE-BOUND',
  ], D.green.dark);
  o += footerBox(20, 220, 520, 44, C.lblue, C.blue, [
    { t: 'A faster GPU or 2nd card only helps if device work outlasts launch & memory!', fill: D.blue.dark, fw: 700 },
  ]);
  return svg(W, H, o);
}

function arithmeticIntensity() {
  const W = 560, H = 278;
  let o = header(W, 'Arithmetic Intensity I = FLOPs / Bytes Moved');
  o += card(20, 42, 186, 124, C.lamber, C.amber, 'Off-Chip GPU DRAM', [
    { t: '24 GiB GDDR6X', mono: true, fill: D.orange.dark, fw: 700 },
    { t: 'β = 1.008 TB/s', mono: true, fill: C.ink },
    '1 byte takes ~1 ps',
  ], D.orange.dark);

  o += arrow(212, 86, 346, 86, C.amber);
  o += fitText(279, 68, 'Reads & Writes', 120, { fs: 13, fw: 700, fill: D.orange.dark });
  o += arrow(346, 118, 212, 118, C.amber);
  o += fitText(279, 138, '1.008 TB/s pipe', 120, { mono: true, fs: 12.5, fw: 600, fill: C.ink });

  o += card(354, 42, 186, 124, C.lblue, C.blue, 'On-Chip Tensor Cores', [
    { t: '128 SMs + SRAM', mono: true, fill: D.blue.dark, fw: 700 },
    { t: 'P = 165.2 TFLOP/s', mono: true, fill: C.ink },
    '164 FLOPs per 1 ps!',
  ], D.blue.dark);

  o += card(20, 182, 250, 82, D.red.tint, D.red.base, 'Low Intensity: y = x + 1 (bf16)', [
    { t: '1 FLOP / (2B read + 2B write)', mono: true, fill: C.ink },
    { t: 'I = 0.25 FLOP/B ≪ 164 (Starved)', mono: true, fill: D.red.dark, fw: 700 },
  ], D.red.dark);
  o += card(290, 182, 250, 82, C.lgreen, C.green, 'High Intensity: 4096² GEMM (bf16)', [
    { t: '2·4096³ FLOPs / (3·4096²·2 B)', mono: true, fill: C.ink },
    { t: 'I = 1,365 FLOP/B ≫ 164 (Full)', mono: true, fill: D.green.dark, fw: 700 },
  ], D.green.dark);
  return svg(W, H, o);
}

function linearReuse() {
  const W = 560, H = 280;
  let o = header(W, 'Weight Reuse Across Batch Rows in Y = XW');
  o += card(20, 44, 146, 112, C.lblue, C.blue, 'Batch X', [
    { t: 'shape (B, D)', mono: true, fill: D.blue.dark, fw: 700 },
    'B token rows',
    'Reads b·B·D bytes',
  ], D.blue.dark);
  o += tx(180, 100, '×', { fs: 22, fw: 700, fill: C.ink });
  o += card(196, 44, 158, 112, C.lamber, C.amber, 'Weights W', [
    { t: 'shape (D, F)', mono: true, fill: D.orange.dark, fw: 700 },
    'Streamed 1× from DRAM',
    'Reused across B rows!',
  ], D.orange.dark);
  o += tx(370, 100, '=', { fs: 22, fw: 700, fill: C.ink });
  o += card(386, 44, 154, 112, C.lgreen, C.green, 'Output Y', [
    { t: 'shape (B, F)', mono: true, fill: D.green.dark, fw: 700 },
    '2·B·D·F FLOPs',
    'Writes b·B·F bytes',
  ], D.green.dark);

  o += footerBox(20, 174, 520, 92, D.purple.tint, C.purple, [
    { t: 'I = 2BDF / [b(BD + DF + BF)]  ≈  2B / b   (when B ≪ D, F)', mono: true, fill: D.purple.dark, fw: 700 },
    { t: 'Smallest dimension of (B, D, F) bottlenecks arithmetic intensity', fill: C.ink },
    { t: 'In bf16 (b = 2):  I ≈ B  →  reaches ridge I* = 164 at B ≈ 164 tokens!', mono: true, fill: D.blue.dark, fw: 700 },
  ]);
  return svg(W, H, o);
}

function gptIntensityLadder() {
  const W = 560, H = 288;
  let o = header(W, '1B GPT Linear Layer (D = F = 2048, bf16): 1,000× Intensity Span');
  const rows = [
    { y: 36,  title: '1. Elementwise (x + 1)', sub: '1 FLOP / 4 B moved  →  I = 0.25 FLOP/B  (0.25 TF/s)', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { y: 94,  title: '2. Single-Token Decode (B = 1)', sub: '8.4M FLOPs / 8.4 MB weights  →  I ≈ 1.0 FLOP/B  (1.0 TF/s)', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { y: 152, title: '3. Batched Decoding (B = 64)', sub: '537M FLOPs / 8.9 MB total  →  I ≈ 60 FLOP/B  (60 TF/s)', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { y: 210, title: '4. Training Step (B = 8,192 tokens)', sub: '68.7G FLOPs / 75.5 MB  →  I ≈ 910 FLOP/B > I*=164  (165 TF/s!)', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  rows.forEach((r) => {
    o += box(20, r.y, 520, 50, r.fill, r.st, 7, 2);
    o += fitText(280, r.y + 17, r.title, 496, { fs: 14, fw: 700, fill: r.dk });
    o += fitText(280, r.y + 37, r.sub, 496, { mono: true, fs: 12.5, fw: 600, fill: C.ink });
  });
  return svg(W, H, o);
}

function overheadKnee(fw) {
  const jax = fw === 'jax';
  const W = 560, H = 278;
  let o = header(W, 'The Overhead-to-Compute Crossover Knee n* = (hP / 2)^(1/3)');
  o += card(20, 40, 250, 146, D.red.tint, D.red.base, 'Small Matmuls (n < n*)', [
    { t: '2n³ / P  <  h', mono: true, fs: 14, fw: 700, fill: D.red.dark },
    'Time per call stays flat at ~h',
    'Apparent TFLOP/s scales as n³',
    jax ? 'Single-op dispatch: h ≈ 70 µs' : 'Per-op dispatch: h ≈ 13 µs',
  ], D.red.dark);
  o += card(290, 40, 250, 146, C.lgreen, C.green, 'Large Matmuls (n > n*)', [
    { t: '2n³ / P  >  h', mono: true, fs: 14, fw: 700, fill: D.green.dark },
    'Device execution outlasts launch',
    'Time grows as 2n³/P at roof P',
    jax ? 'Knee: n* ≈ 1,800' : 'Knee: n* ≈ 1,000',
  ], D.green.dark);
  o += footerBox(20, 200, 520, 66, C.lblue, C.blue, [
    { t: 'Faster GPUs Require Wider Matrices:  n* ∝ P^(1/3)', fill: D.blue.dark, fw: 700 },
    { t: '14× higher P (RTX 4090 → B200) shifts knee n* up by 14^(1/3) ≈ 2.4×!', mono: true, fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ── §13.3 compilation.md ────────────────────────────────────────────────

function eagerPenalties() {
  const W = 560, H = 280;
  let o = header(W, '8-Op gelu_ish(x) on 61 MiB Tensor: Eager vs. Compiled');
  o += card(20, 40, 250, 160, D.red.tint, D.red.base, 'Eager Execution (8 Kernels)', [
    { t: '8 Python launches (8 × h)', mono: true, fill: D.red.dark, fw: 700 },
    '6 unary ops: 12 array transfers',
    '2 binary ops: 6 array transfers',
    { t: 'Total: 18 × 61 MiB = 1.07 GiB', mono: true, fill: D.red.dark, fw: 700 },
    'Time: ~1.2 ms at 0.9 TB/s',
  ], D.red.dark);
  o += card(290, 40, 250, 160, C.lgreen, C.green, 'Compiled & Fused (1 Kernel)', [
    { t: '1 Kernel Launch (or Replay)', mono: true, fill: D.green.dark, fw: 700 },
    'Reads x once from DRAM (61 MiB)',
    '7 intermediates stay in registers!',
    { t: 'Total: 2 × 61 MiB = 122 MiB', mono: true, fill: D.green.dark, fw: 700 },
    'Time: ~0.13 ms (9× faster!)',
  ], D.green.dark);
  o += footerBox(20, 214, 520, 52, C.lblue, C.blue, [
    { t: 'Operator Fusion cures the bandwidth tax (18 → 2 DRAM transfers)', fill: D.blue.dark, fw: 700 },
    { t: 'CUDA Graph Replay cures the host launch tax (N launches → 1 launch)', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function branchStrategies(fw) {
  const jax = fw === 'jax';
  const W = 560, H = 282;
  let o = header(W, 'How Graph Tracers Handle Data-Dependent Branches (if x.sum() > 0)');
  o += card(16, 40, 170, 166, D.red.tint, D.red.base, '1. Freeze Path', [
    { t: jax ? 'trace one example' : 'torch.jit.trace', mono: true, fill: D.red.dark, fw: 700 },
    'Records only branch',
    'taken by 1st input',
    { t: 'SILENT BUG on', fill: D.red.dark, fw: 700 },
    { t: 'new data!', fill: D.red.dark, fw: 700 },
  ], D.red.dark);
  o += card(195, 40, 170, 166, C.lgreen, C.green, '2. Refuse & Primitives', [
    { t: jax ? 'jax.jit' : 'fullgraph=True', mono: true, fill: D.green.dark, fw: 700 },
    'Raises trace error;',
    'rewrite with:',
    { t: jax ? 'lax.select' : 'torch.where', mono: true, fill: D.green.dark },
    { t: jax ? 'or lax.cond' : 'or torch.cond', mono: true, fill: D.green.dark },
  ], D.green.dark);
  o += card(374, 40, 170, 166, C.lamber, C.amber, '3. Graph Break', [
    { t: jax ? 'split at the branch' : 'torch.compile default', mono: true, fill: D.orange.dark, fw: 700 },
    'Runs Graph 1 →',
    'Syncs scalar to CPU',
    '→ Python branches',
    '→ Launches Graph 2',
  ], D.orange.dark);
  o += footerBox(16, 220, 528, 48, C.lblue, C.blue, [
    { t: 'Shape branches (x.shape[0] > 4) unroll at trace time: 0 breaks', fill: D.blue.dark, fw: 700 },
  ]);
  return svg(W, H, o);
}

function shapeBucketing() {
  const W = 560, H = 280;
  let o = header(W, 'Ragged Sequence Lengths: Recompilation Thrashing vs. Bucketing');
  o += card(20, 38, 520, 104, D.red.tint, D.red.base, '(a) Unpadded Ragged Batches: Lengths 100 → 128 → 75', [
    { t: 'Step 1 (L=100): Compile #1 (10s)  ·  Step 2 (L=128): Compile #2 (10s)', mono: true, fill: D.red.dark },
    { t: 'Step 3 (L=75): Compile #3 (10s)  →  3 multi-second compilation stalls!', mono: true, fill: D.red.dark, fw: 700 },
  ], D.red.dark);
  o += arrow(280, 144, 280, 162, C.green);
  o += card(20, 164, 520, 104, C.lgreen, C.green, '(b) Padded Shape Buckets (Multiples of 64 → All Padded to L = 128)', [
    { t: 'L=100 + 28 pad → 128   ·   L=128 → 128   ·   L=75 + 53 pad → 128', mono: true, fill: D.green.dark },
    { t: '1 Compiled Kernel Cached & Reused Across All Steps (0 recompiles!)', mono: true, fill: D.green.dark, fw: 700 },
  ], D.green.dark);
  return svg(W, H, o);
}

function ffnFusionRatio() {
  const W = 560, H = 278;
  let o = header(W, '1B GPT FFN Block (d=2048, 4d=8192, M=8192 tokens in bf16)');
  o += card(20, 40, 250, 142, C.lblue, C.blue, 'Surrounding FFN Matmuls', [
    { t: 'Up + Down: 16·d²·M FLOPs', mono: true, fill: D.blue.dark, fw: 700 },
    '5.5 × 10¹¹ FLOPs at 165 TF/s',
    { t: 'Compute Time ≈ 3.3 ms', mono: true, fill: D.blue.dark, fw: 700 },
    'Compute-bound on Tensor Cores',
  ], D.blue.dark);
  o += card(290, 40, 250, 142, C.lamber, C.amber, '8-Op Activation on 4d Width', [
    { t: 'Eager (18x): 2.25 GiB → 2.4 ms', mono: true, fill: D.red.dark, fw: 700 },
    '72% as long as the 2 matmuls!',
    { t: 'Fused (2x): 256 MiB → 0.27 ms', mono: true, fill: D.green.dark, fw: 700 },
    'Shrunk to < 8% of matmul time!',
  ], D.orange.dark);
  o += footerBox(20, 198, 520, 66, D.purple.tint, C.purple, [
    { t: 'Time Ratio (Eager Activation / FFN Matmuls) = 9 · I* / d ≈ 0.72', mono: true, fill: D.purple.dark, fw: 700 },
    { t: 'Independent of token count M! Higher ridge I* makes fusion even more vital.', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function fusionScope() {
  const W = 560, H = 282;
  let o = header(W, 'What Graph Compilers Fuse vs. Hand-Tiled Kernel Algorithms');
  o += card(16, 40, 170, 168, C.lgreen, C.green, '1. Auto-Fused', [
    { t: 'Inductor / XLA', mono: true, fill: D.green.dark, fw: 700 },
    'Elementwise chains',
    '(stay in registers)',
    'Row reductions',
    '(Softmax/RMSNorm)',
  ], D.green.dark);
  o += card(195, 40, 170, 168, C.lblue, C.blue, '2. Library GEMMs', [
    { t: 'cuBLAS / CUTLASS', mono: true, fill: D.blue.dark, fw: 700 },
    'Dense X @ W tiles',
    'Already near peak P',
    'Compiler delegates',
    '+ fuses epilogues',
  ], D.blue.dark);
  o += card(374, 40, 170, 168, D.purple.tint, C.purple, '3. Hand-Tiled', [
    { t: 'FlashAttention', mono: true, fill: D.purple.dark, fw: 700 },
    'Online softmax',
    'rescaling across',
    'Q, K, V SRAM tiles',
    '(never writes T×T!)',
  ], D.purple.dark);
  o += footerBox(16, 222, 528, 46, C.lamber, C.amber, [
    { t: 'Compilers eliminate local DRAM round trips; algorithmic tiling requires Triton / CUDA.', fill: D.orange.dark, fw: 700 },
  ]);
  return svg(W, H, o);
}

function cudaGraphReplay() {
  const W = 560, H = 280;
  let o = header(W, '60-Layer Thin Stack: Per-Kernel Python Launch vs. CUDA Graph Replay');
  o += card(20, 38, 520, 102, D.red.tint, D.red.base, '(a) Default Launch: CPU Dispatches 60 Kernels One by One', [
    { t: 'CPU: [Launch 1 (13µs)] → [Launch 2 (13µs)] → … → [Launch 60] ≈ 0.8 ms', mono: true, fill: D.red.dark },
    { t: 'GPU: [2µs] ..idle gap.. [2µs] ..idle gap.. [2µs] (GPU busy only ~15%!)', mono: true, fill: C.ink },
  ], D.red.dark);
  o += arrow(280, 142, 280, 160, C.green);
  o += card(20, 162, 520, 104, C.lgreen, C.green, '(b) CUDA Graph Replay: One Host Launch per Call', [
    { t: 'CPU: [1 cudaGraphLaunch Call] (static pointers & shapes baked in)', mono: true, fill: D.green.dark, fw: 700 },
    { t: 'GPU: [K1][K2][K3]…[K60] back-to-back on device ≈ 0.3 ms', mono: true, fill: D.green.dark, fw: 700 },
  ], D.green.dark);
  return svg(W, H, o);
}

// ── §13.2 hardware.md ───────────────────────────────────────────────────

function decodeBottleneck() {
  const W = 560, H = 276;
  let o = header(W, 'Why a 165-TFLOP/s GPU Decodes a 1B GPT at Only 450 Tokens/s');
  o += card(20, 40, 250, 146, C.lblue, C.blue, 'Tensor Core Math per Token', [
    { t: '2N = 2 × 10⁹ FLOPs', mono: true, fill: D.blue.dark, fw: 700 },
    'At P = 165 TFLOP/s (bf16):',
    { t: 't_math = 12 µs', mono: true, fs: 14.5, fill: D.blue.dark, fw: 700 },
    'ALUs idle > 99% of the step!',
  ], D.blue.dark);
  o += card(290, 40, 250, 146, D.red.tint, D.red.base, 'GDDR6X Weight Read per Token', [
    { t: '2N = 2.0 GB of bf16 weights', mono: true, fill: D.red.dark, fw: 700 },
    'At achieved β ≈ 0.9 TB/s:',
    { t: 't_mem = 2.2 ms (183× longer!)', mono: true, fs: 13.5, fill: D.red.dark, fw: 700 },
    'Caps 1 stream at ~450 tok/s',
  ], D.red.dark);
  o += footerBox(20, 202, 520, 60, C.lamber, C.amber, [
    { t: 'Three Physical Causes of the 183× Gap:', fill: D.orange.dark, fw: 700 },
    { t: '1. Off-chip boundary  ·  2. Perimeter shoreline O(L) vs. area O(L²)  ·  3. 500× DRAM energy', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function hockneyTransfer() {
  const W = 560, H = 276;
  let o = header(W, 'Hockney Block-Transfer Law & Half-Bandwidth Size n_{1/2} = α·β');
  o += card(20, 40, 250, 144, D.red.tint, D.red.base, 'Random Transfers (n ≪ n_{1/2})', [
    { t: 't(n) = α + n/β ≈ α', mono: true, fill: D.red.dark, fw: 700 },
    'Dominated by start-up latency α',
    '4 KB NVMe read (α=70µs):',
    { t: 'β_eff ≈ 0.06 GB/s (< 0.5% peak)', mono: true, fill: D.red.dark },
  ], D.red.dark);
  o += card(290, 40, 250, 144, C.lgreen, C.green, 'Contiguous Shards (n ≥ n_{1/2})', [
    { t: 'n_{1/2} = α · β  (reaches ½ β)', mono: true, fill: D.green.dark, fw: 700 },
    'NVMe (70µs × 14 GB/s): n_{1/2} ≈ 1 MB',
    'Multi-MB contiguous shards:',
    { t: 'Streams near full 14 GB/s!', mono: true, fill: D.green.dark },
  ], D.green.dark);
  o += footerBox(20, 200, 520, 62, C.lblue, C.blue, [
    { t: 'Effective Bandwidth:  β_eff(n) = n / (α + n/β) = β / (1 + αβ / n)', mono: true, fill: D.blue.dark, fw: 700 },
    { t: 'Coalesce GPU DRAM reads (32-B sectors) and shuffle disk data in ≥ 1 MB shards.', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function generationalRidge() {
  const W = 560, H = 284;
  let o = header(W, 'Data-Center GPU Generations: Tensor Cores Outgrow HBM Bandwidth');
  const gens = [
    { x: 16,  w: 124, name: 'V100 (2017)', p: 'P = 125 TF/s', b: 'β = 0.90 TB/s', r: 'I* = 139', col: C.lgray, st: C.gray, dk: C.ink },
    { x: 150, w: 124, name: 'A100 (2020)', p: 'P = 312 TF/s', b: 'β = 2.04 TB/s', r: 'I* = 153', col: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 284, w: 124, name: 'H100 (2022)', p: 'P = 989 TF/s', b: 'β = 3.35 TB/s', r: 'I* = 295', col: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 418, w: 126, name: 'B200 (2024)', p: 'P = 2250 TF/s', b: 'β = 8.00 TB/s', r: 'I* = 281', col: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  gens.forEach((g) => {
    o += card(g.x, 40, g.w, 144, g.col, g.st, g.name, [
      { t: g.p, mono: true, fs: 12 },
      { t: g.b, mono: true, fs: 12 },
      { t: g.r, mono: true, fs: 13, fw: 700, fill: g.dk },
    ], g.dk);
  });
  o += footerBox(16, 200, 528, 70, D.purple.tint, C.purple, [
    { t: '2017 → 2024:  16-bit Compute P grew 18×   vs.   HBM Bandwidth β grew 8.9×', mono: true, fill: D.purple.dark, fw: 700 },
    { t: 'Ridge I* doubled (139 → ~290 FLOP/B): ops with I=200 became bandwidth-bound!', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function smLatencyHiding() {
  const W = 560, H = 280;
  let o = header(W, "Streaming Multiprocessors & Little's Law Latency Hiding");
  o += card(20, 40, 246, 150, C.lblue, C.blue, 'Inside 1 of 128 SMs (RTX 4090)', [
    'Large On-Chip Register File',
    'Warp 0: Computing on Tensor Cores',
    'Warp 1: Stalled on 250 ns DRAM read',
    'Warp 2..15: Ready / In-flight loads',
    { t: 'Zero-cycle warp context switch!', fill: D.blue.dark, fw: 700 },
  ], D.blue.dark);
  o += card(290, 40, 250, 150, C.lamber, C.amber, "Little's Law In-Flight Budget", [
    { t: 'Bytes = β × Latency', mono: true, fs: 13.5, fill: D.orange.dark, fw: 700 },
    { t: '1.0 TB/s × 250 ns = 250 KB', mono: true, fill: C.ink },
    'Across 128 SMs: ~2 KB / SM',
    '≥ 16 active warps/SM (4 B/thread)',
    'to keep GDDR6X pipe 100% full',
  ], D.orange.dark);
  o += footerBox(20, 206, 520, 60, C.lgreen, C.green, [
    { t: 'Why small grids underperform: too few thread blocks/warps fail to keep 250 KB in flight,', fill: C.ink },
    { t: 'making the kernel latency-bound well below the 1.0 TB/s bandwidth roof!', fill: D.green.dark, fw: 700 },
  ]);
  return svg(W, H, o);
}

function tensorCoreMath() {
  const W = 560, H = 280;
  let o = header(W, 'RTX 4090 Throughput Scaling Across Hardware Units & Formats');
  o += footerBox(20, 36, 520, 52, C.lgray, C.gray, [
    { t: '128 SMs × 128 FP32 ALUs = 16,384 units × 2 FLOP/cycle × 2.52 GHz', mono: true, fill: C.ink, fw: 700 },
    { t: 'Scalar fp32 Peak = 82.6 TFLOP/s (Tensor Cores multiply matrix tiles per instruction)', fill: C.muted },
  ]);

  o += card(16, 102, 170, 162, C.lblue, C.blue, 'tf32 Tensor Core', [
    { t: '1× = 82.6 TF/s', mono: true, fs: 13.5, fill: D.blue.dark, fw: 700 },
    'Stored as 4-B fp32',
    '19-bit internal math',
    '8-bit exp + 10-bit mant',
    'Drop-in fp32 speedup',
  ], D.blue.dark);
  o += card(195, 102, 170, 162, C.lgreen, C.green, 'bf16 / fp16 Core', [
    { t: '2× = 165.2 TF/s', mono: true, fs: 13.5, fill: D.green.dark, fw: 700 },
    'Stored as 2-B (½ DRAM)',
    'fp32 accumulator!',
    'bf16 keeps 8-bit exp',
    '(no loss scaling!)',
  ], D.green.dark);
  o += card(374, 102, 170, 162, D.purple.tint, C.purple, 'fp8 (E4M3 / E5M2)', [
    { t: '4× = 330.4 TF/s', mono: true, fs: 13.5, fill: D.purple.dark, fw: 700 },
    'Stored as 1-B (¼ DRAM)',
    'Needs per-tensor or',
    'per-block (fp4) scale',
    'factors for outliers',
  ], D.purple.dark);
  return svg(W, H, o);
}

function sweepTimeInvariant() {
  const W = 560, H = 284;
  let o = header(W, 'The ~20 ms Memory Sweep Invariant (C / β) Across GPUs');
  const cards = [
    { x: 16,  w: 124, name: 'RTX 4090', c: 'C = 24 GB', b: 'β = 1.01 TB/s', s: '23.8 ms', r: '~42 tok/s', col: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 150, w: 124, name: 'RTX 5090', c: 'C = 32 GB', b: 'β = 1.79 TB/s', s: '17.9 ms', r: '~56 tok/s', col: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 284, w: 124, name: 'H100 SXM', c: 'C = 80 GB', b: 'β = 3.35 TB/s', s: '23.9 ms', r: '~42 tok/s', col: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 418, w: 126, name: 'B200 SXM', c: 'C = 180 GB', b: 'β = 8.00 TB/s', s: '22.5 ms', r: '~44 tok/s', col: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  cards.forEach((c) => {
    o += card(c.x, 38, c.w, 152, c.col, c.st, c.name, [
      { t: c.c, mono: true, fs: 12 },
      { t: c.b, mono: true, fs: 12 },
      { t: `C/β = ${c.s}`, mono: true, fs: 12.5, fw: 700, fill: c.dk },
      { t: c.r, mono: true, fs: 12.5, fw: 700, fill: C.ink },
    ], c.dk);
  });
  o += footerBox(16, 204, 528, 66, D.purple.tint, C.purple, [
    { t: 'Any model filling its card decodes 1 stream at 1 / (C/β) ≈ 40–55 tokens/s!', fill: D.purple.dark, fw: 700 },
    { t: 'Batching B sequences shares weight reads across B tokens: I_weights ≈ B up to ridge I*.', fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ── §13.4 memory-precision.md ───────────────────────────────────────────

function memorySqueeze24GiB() {
  const W = 560, H = 278;
  let o = header(W, '1B-Parameter GPT on a 24 GiB RTX 4090: Fixed State vs. Activations');
  o += card(20, 40, 316, 148, D.red.tint, D.red.base, 'Fixed Parameter State: 16N = 15.1 GiB (63%)', [
    { t: 'fp32 Weights (4N):    3.77 GiB', mono: true, fill: C.ink },
    { t: 'fp32 Gradients (4N):  3.77 GiB', mono: true, fill: C.ink },
    { t: 'fp32 Adam m & v (8N): 7.55 GiB', mono: true, fill: D.red.dark, fw: 700 },
    'Allocated before reading a single token!',
  ], D.red.dark);
  o += card(348, 40, 192, 148, C.lgreen, C.green, 'Headroom: 8.9 GiB', [
    { t: '~2.8–4.3 MiB / tok', mono: true, fill: D.green.dark, fw: 700 },
    'across 18 blocks',
    'Fits only 2–3 seqs',
    'of 1,024 tokens!',
  ], D.green.dark);
  o += footerBox(20, 204, 520, 62, C.lblue, C.blue, [
    { t: 'Step Budget:  M(n) ≈ 16N + n · (L·a + a_out + Δ) ≤ 24 GiB', mono: true, fill: D.blue.dark, fw: 700 },
    { t: 'Reaching 524K-token batches requires Checkpointing + Accumulation + Multi-GPU Sharding.', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function vjpSaveRules() {
  const W = 560, H = 280;
  let o = header(W, 'Three Chain-Rule Identities Determine What Forward Must Save');
  o += card(16, 40, 170, 156, C.lblue, C.blue, '1. Linear Y = XW', [
    { t: 'Saves input X', mono: true, fill: D.blue.dark, fw: 700 },
    'Because weight grad:',
    { t: '∇_W L = Xᵀ (∇_Y L)', mono: true, fill: C.ink },
    'QKV & FFN save inputs',
  ], D.blue.dark);
  o += card(195, 40, 170, 156, C.lamber, C.amber, '2. Unary y = f(x)', [
    { t: 'Saves x (or y)', mono: true, fill: D.orange.dark, fw: 700 },
    'Because VJP multiplies',
    { t: 'by slope f\'(x)', mono: true, fill: C.ink },
    'RMSNorm & SiLU gate',
  ], D.orange.dark);
  o += card(374, 40, 170, 156, D.purple.tint, C.purple, '3. Product z = p ⊙ q', [
    { t: 'Saves BOTH p & q!', mono: true, fill: D.purple.dark, fw: 700 },
    'Because:',
    { t: '∇_p = ∇_z ⊙ q', mono: true, fill: C.ink },
    { t: '∇_q = ∇_z ⊙ p', mono: true, fill: C.ink },
  ], D.purple.dark);
  o += footerBox(16, 210, 528, 56, D.red.tint, D.red.base, [
    { t: 'SwiGLU Consequence: SiLU(g) ⊙ u + Down-Proj saves 4 vectors of width (8/3)d:', fill: D.red.dark, fw: 700 },
    { t: 'g, SiLU(g), u, and SiLU(g)⊙u = 10.7d values/token (> half the entire block!)', mono: true, fill: C.ink },
  ]);
  return svg(W, H, o);
}

function blockActivationTally() {
  const W = 560, H = 282;
  let o = header(W, 'Per-Token Activation Tally of 1 Pre-Norm SwiGLU Block (d = 2048)');
  o += card(20, 40, 164, 144, C.lblue, C.blue, 'Norms & Inputs', [
    { t: '4d values/tok', mono: true, fill: D.blue.dark, fw: 700 },
    '2 RMSNorm inputs (2d)',
    '2 normed outputs (2d)',
    { t: '32 KiB (fp32)', mono: true, fill: C.ink },
  ], D.blue.dark);
  o += card(198, 40, 164, 144, C.lgreen, C.green, 'Fused Attention', [
    { t: '4d–5d values/tok', mono: true, fill: D.green.dark, fw: 700 },
    'Q, K, V (3d) + Out (d)',
    'No T×T matrix saved!',
    { t: '32–40 KiB (fp32)', mono: true, fill: C.ink },
  ], D.green.dark);
  o += card(376, 40, 164, 144, C.lamber, C.amber, 'SwiGLU FFN', [
    { t: '10.7d values/tok', mono: true, fill: D.orange.dark, fw: 700 },
    '4 vectors of (8/3)d',
    'gate, SiLU, up, prod',
    { t: '85 KiB (fp32)', mono: true, fill: C.ink },
  ], D.orange.dark);
  o += footerBox(20, 198, 520, 70, D.purple.tint, C.purple, [
    { t: '1 Fused Block Total: a ≈ 18.7d–19.7d ≈ 20d values/tok (149–160 KiB in fp32)', mono: true, fill: D.purple.dark, fw: 700 },
    { t: 'Output Head & Loss (V=50,257): a_out = 212 KiB saved + Δ = 393 KiB (8V) backward peak!', mono: true, fill: D.red.dark },
  ]);
  return svg(W, H, o);
}

function mixedPrecisionFlow() {
  const W = 560, H = 280;
  let o = header(W, 'Mixed-Precision Training Recipe: What Runs in bf16 vs. fp32');
  o += card(20, 40, 156, 148, C.lblue, C.blue, 'fp32 Master State', [
    { t: '16N bytes (4 B)', mono: true, fill: D.blue.dark, fw: 700 },
    'Master Weights (4N)',
    'Master Grads (4N)',
    'Adam Moments (8N)',
    'Preserves η·g updates',
  ], D.blue.dark);
  o += arrow(180, 114, 200, 114, C.blue);
  o += card(204, 40, 156, 148, C.lgreen, C.green, 'bf16 Matmuls', [
    { t: '+2N weight copy', mono: true, fill: D.green.dark, fw: 700 },
    'Linear X @ W (2 B)',
    'FlashAttention (2 B)',
    '165.2 TFLOP/s (2×)',
    'Block acts × 0.58',
  ], D.green.dark);
  o += arrow(364, 114, 384, 114, C.green);
  o += card(388, 40, 152, 148, C.lamber, C.amber, 'fp32 Reductions', [
    { t: 'High Precision', mono: true, fill: D.orange.dark, fw: 700 },
    'Residual Stream',
    'RMSNorm Variance',
    'Softmax & CE Loss',
    'Prevents underflow',
  ], D.orange.dark);
  o += footerBox(20, 204, 520, 62, D.purple.tint, C.purple, [
    { t: 'Why bf16 Needs No Loss Scaling: Same 8-Bit Exponent Range as fp32 (~10^{±38})', fill: D.purple.dark, fw: 700 },
    { t: 'Net effect: 1.5–1.75× faster step; capacity grows ~15–25% (+2N weights offset act savings).', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function activationCheckpointing() {
  const W = 560, H = 280;
  let o = header(W, 'Per-Block Activation Checkpointing (Rematerialization)');
  o += card(20, 38, 520, 100, D.red.tint, D.red.base, '(a) Standard Backprop: Saves All L = 18 Blocks Internals (L · 20d)', [
    { t: '[Blk 1: 160 KiB] + [Blk 2: 160 KiB] + … + [Blk 18: 160 KiB] = 2.8 MiB/token', mono: true, fill: D.red.dark, fw: 700 },
    '8.9 GiB headroom fits only ~2,800 tokens (2–3 sequences of 1,024)',
  ], D.red.dark);
  o += arrow(280, 140, 280, 158, C.green);
  o += card(20, 160, 520, 108, C.lgreen, C.green, '(b) Checkpointed: Saves Only Block Inputs (L · 4d) + 1 Recomputed Block', [
    { t: '18 × 8 KiB inputs (144 KiB/tok) + 1 live block (160 KiB) → > 3× token capacity!', mono: true, fill: D.green.dark, fw: 700 },
    'Fits 10,000–15,000 tokens! Cost: 1 extra forward pass (+2N FLOPs/tok, ≤ +33% step time)',
  ], D.green.dark);
  return svg(W, H, o);
}

function gradAccumulation() {
  const W = 560, H = 280;
  let o = header(W, 'Gradient Accumulation Over m Micro-Batches & Traffic Overhead');
  o += card(20, 40, 250, 146, C.lblue, C.blue, 'Micro-Batch Loop (i = 1..m)', [
    { t: 'loss_i = gpt_loss(X_i) / m', mono: true, fill: D.blue.dark, fw: 700 },
    'Frees activations after each bwd!',
    'Peak activation memory = M(B_micro)',
    'Optimizer steps once after m passes',
  ], D.blue.dark);
  o += card(290, 40, 250, 146, C.lamber, C.amber, 'Accumulator Traffic Overhead', [
    { t: 'G_acc ← G_acc + g_i (12N bytes)', mono: true, fill: D.orange.dark, fw: 700 },
    'Reads G_acc (4N) + g_i (4N), writes 4N',
    'Vs. micro-batch compute 6Nn FLOPs:',
    { t: 'Overhead = (12N/β)/(6Nn/P) = 2I*/n', mono: true, fill: D.red.dark, fw: 700 },
  ], D.orange.dark);
  o += footerBox(20, 202, 520, 64, C.lgreen, C.green, [
    { t: 'At n = 1,024 tokens in tf32 (I* = 82): 2I*/n ≈ 16% overhead (drops to 4% at n = 4,096)', mono: true, fill: D.green.dark, fw: 700 },
    { t: 'Rule: Always pick the largest micro-batch n that fits in GPU memory without extra remat.', fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ── §13.5 multiple-gpus.md ──────────────────────────────────────────────

function secondGpuTradeoff() {
  const W = 560, H = 278;
  let o = header(W, 'When Does Adding a Second GPU Actually Shorten a Step?');
  o += card(20, 38, 520, 88, C.lblue, C.blue, '1 GPU Step Time (Global Batch B)', [
    { t: '[ Device Compute t_dev(B) (or Host Dispatch h_step) ]  (0 communication)', mono: true, fill: D.blue.dark, fw: 700 },
  ], D.blue.dark);
  o += card(20, 138, 520, 126, C.lamber, C.amber, '2 GPU Step Time (Shard B/2 per GPU + Taxes)', [
    { t: '[ Half-Batch Compute t_dev(B/2) ] + [ Host k·h ] + [ Allreduce α + S·β_link ]', mono: true, fill: D.orange.dark, fw: 700 },
    'GDDR6X On-Card Memory: 1,000 GB/s  vs.  PCIe 4.0 Link: ≤ 32 GB/s nominal (30× slower!)',
    { t: '2nd GPU wins iff:  t_compute(B) − t_compute(B/2)  >  c·α + S·β_link + Δh', mono: true, fill: D.green.dark, fw: 700 },
  ], D.orange.dark);
  return svg(W, H, o);
}

function shardGradAveraging() {
  const W = 560, H = 276;
  let o = header(W, 'Exact Full-Batch Gradient From k Equal Data-Parallel Shards');
  o += card(20, 42, 160, 120, C.lblue, C.blue, 'GPU 0 (Shard B/k)', [
    { t: 'g₀ = (k/B) Σ ∇ℓ_i', mono: true, fill: D.blue.dark, fw: 700 },
    'Local mean loss grad',
  ], D.blue.dark);
  o += card(20, 174, 160, 86, C.lblue, C.blue, 'GPU k−1 (Shard B/k)', [
    { t: 'g_{k-1} = (k/B) Σ ∇ℓ_i', mono: true, fill: D.blue.dark, fw: 700 },
  ], D.blue.dark);

  o += arrow(184, 102, 224, 136, C.blue);
  o += arrow(184, 216, 224, 164, C.blue);

  o += card(228, 96, 136, 108, C.lamber, C.amber, 'Allreduce (SUM)', [
    { t: 'G = Σ g_j', mono: true, fs: 13.5, fill: D.orange.dark, fw: 700 },
    'Sum of k means',
    '(k× too large!)',
  ], D.orange.dark);

  o += arrow(368, 150, 396, 150, C.amber);

  o += card(400, 74, 140, 152, C.lgreen, C.green, 'Divide by k', [
    { t: 'g = G / k', mono: true, fs: 14, fill: D.green.dark, fw: 700 },
    '= (1/B) Σ ∇ℓ_i',
    'Exact global grad!',
    'θ ← θ − (η/k) G',
  ], D.green.dark);
  return svg(W, H, o);
}

function starVsRing() {
  const W = 560, H = 280;
  let o = header(W, 'Star Allreduce vs. Ring Allreduce (k GPUs, Gradient Size S)');
  o += card(20, 40, 250, 156, D.red.tint, D.red.base, 'Star Topology (GPU 0 Hub)', [
    { t: 'Hub Traffic: (k − 1) · S', mono: true, fill: D.red.dark, fw: 700 },
    'GPU 0 gathers (k−1) copies of S,',
    'adds them, scatters (k−1) copies',
    'Grows linearly O(kS) with GPUs!',
    { t: '1B GPT (S=4GB, k=8): 28 GB hub!', mono: true, fill: D.red.dark, fw: 700 },
  ], D.red.dark);
  o += card(290, 40, 250, 156, C.lgreen, C.green, 'Logical Ring (Reduce-Scatter + AG)', [
    { t: 'Per-GPU Traffic: 2(k − 1)/k · S', mono: true, fill: D.green.dark, fw: 700 },
    'Phase 1: Reduce-Scatter (k−1 hops)',
    'Phase 2: All-Gather (k−1 hops)',
    'Strictly < 2S for any k → ∞!',
    { t: '1B GPT (S=4GB, k=8): 7 GB/GPU!', mono: true, fill: D.green.dark, fw: 700 },
  ], D.green.dark);
  o += footerBox(20, 210, 520, 56, C.lblue, C.blue, [
    { t: 'Bandwidth-Optimal vs. Latency-Optimal:', fill: D.blue.dark, fw: 700 },
    { t: 'Ring moves < 2S bytes in 2(k−1) hops; for tiny messages on large k, NCCL uses 2·log₂(k) trees.', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function strongVsWeakScaling() {
  const W = 560, H = 280;
  let o = header(W, 'Multi-GPU Scaling Conventions: Strong Scaling vs. Weak Scaling');
  o += card(20, 40, 250, 156, C.lblue, C.blue, 'Strong Scaling (Fixed Global B)', [
    { t: 'Per-GPU shard: b = B / k ↓', mono: true, fill: D.blue.dark, fw: 700 },
    'Ideal step time: t(1) / k',
    'Same optimization trajectory',
    'Harder to scale at large k:',
    'compute shrinks while comm S stays',
  ], D.blue.dark);
  o += card(290, 40, 250, 156, C.lgreen, C.green, 'Weak Scaling (Fixed Per-GPU b)', [
    { t: 'Global batch: B = k · b ↑', mono: true, fill: D.green.dark, fw: 700 },
    'Ideal step time: t(1) (constant)',
    'Ideal throughput: k × examples/s',
    'Per-GPU compute t_dev(b) stays fixed',
    'to amortize allreduce S·β_link!',
  ], D.green.dark);
  o += footerBox(20, 210, 520, 56, C.lamber, C.amber, [
    { t: 'Weak-Scaling Speedup Condition (k = 2): t_dev(b) > 2·h_step and t_dev(b) > α + S·β_link', mono: true, fill: D.orange.dark, fw: 700 },
    { t: 'LeNet on 2 GPUs: b = 256 slows down (host-bound); b = 32,768 reaches 1.9× speedup!', fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ── §13.6 multi-gpu-practice.md ─────────────────────────────────────────

function lenetVsResnet18() {
  const W = 560, H = 276;
  let o = header(W, 'Why We Move From LeNet (§13.5) to ResNet-18 at 64×64 (§13.6)');
  o += card(20, 40, 250, 152, D.red.tint, D.red.base, 'Toy LeNet (28×28 Fashion-MNIST)', [
    { t: '61.7K params (S = 247 KB)', mono: true, fill: D.red.dark, fw: 700 },
    '< 2 FLOPs per byte of gradient',
    'Device busy < 0.3 ms at B = 256',
    'Dominated by Python launch h_step',
    'Cannot test backward comm overlap',
  ], D.red.dark);
  o += card(290, 40, 250, 152, C.lgreen, C.green, 'ResNet-18 (64×64 Fashion-MNIST)', [
    { t: '11.2M params (S = 44.7 MB)', mono: true, fill: D.green.dark, fw: 700 },
    '2.2G MACs/img (~50 FLOP/B of grad)',
    'Device busy ~100 ms at b = 256',
    'Host launch overhead is < 1%',
    'Tests 25–30 MiB bucket overlap!',
  ], D.green.dark);
  o += footerBox(20, 206, 520, 56, C.lblue, C.blue, [
    { t: 'At ~100 ms of compute vs. ~18–28 ms of PCIe allreduce, ResNet-18 cleanly isolates', fill: C.ink },
    { t: 'how much communication is hidden by bucketed backward-pass overlap on 1, 2, and 4 GPUs.', fill: D.blue.dark, fw: 700 },
  ]);
  return svg(W, H, o);
}

function spmdModels(fw) {
  const W = 560, H = 280;
  const jax = fw === 'jax';
  let o = header(W, jax ? 'Compiler SPMD: One Program Drives k GPUs' : 'Multi-Process SPMD: One Process per GPU');
  o += jax
    ? card(20, 40, 520, 160, D.purple.tint, C.purple, 'Compiler-Partitioned SPMD (XLA)', [
      { t: 'nnx.jit + NamedSharding(mesh)', mono: true, fill: D.purple.dark, fw: 700 },
      '1 Python process writes the 1-GPU step',
      'Annotate P() weights, P("data") batch',
      'GSPMD/Shardy partitions the HLO graph',
      'Auto-inserts & overlaps NCCL ops',
    ], D.purple.dark)
    : card(20, 40, 520, 160, C.lblue, C.blue, 'Multi-Process SPMD (torchrun + DDP)', [
      { t: 'torchrun --nproc-per-node=k', mono: true, fill: D.blue.dark, fw: 700 },
      'Spawns k independent OS processes',
      '1 Python thread per GPU (rank 0..k−1)',
      'Arbitrary dynamic Python per rank',
      'DDP autograd hooks call NCCL',
    ], D.blue.dark);
  o += footerBox(20, 214, 520, 52, C.lgreen, C.green, [
    { t: 'Removes the §13.5 single-thread k·h_step bottleneck', fill: D.green.dark, fw: 700 },
  ]);
  return svg(W, H, o);
}

function torchrunDdp() {
  const W = 560, H = 278;
  let o = header(W, 'PyTorch Multi-Process DDP Execution Flow (torchrun)');
  o += card(20, 40, 160, 152, C.lblue, C.blue, 'Rank 0 Process', [
    { t: 'LOCAL_RANK = 0', mono: true, fill: D.blue.dark, fw: 700 },
    'Slice [s·k·b : +b]',
    'Forward & Backward',
    'Fires DDP hooks',
  ], D.blue.dark);
  o += card(200, 40, 160, 152, C.lblue, C.blue, 'Rank r Process', [
    { t: 'LOCAL_RANK = r', mono: true, fill: D.blue.dark, fw: 700 },
    'Slice [(s·k+r)b : +b]',
    'Forward & Backward',
    'Fires DDP hooks',
  ], D.blue.dark);
  o += card(380, 40, 160, 152, C.lgreen, C.green, 'NCCL Ring / Tree', [
    { t: 'Async Allreduce', mono: true, fill: D.green.dark, fw: 700 },
    '1 MiB 1st bucket',
    '25 MiB later buckets',
    'Identical opt.step()',
  ], D.green.dark);
  o += arrow(182, 116, 198, 116, C.blue);
  o += arrow(362, 116, 378, 116, C.green);
  o += footerBox(20, 206, 520, 58, C.lamber, C.amber, [
    { t: 'At DDP(model) init: Rank 0 broadcasts initial weights & buffers to all ranks.', fill: D.orange.dark, fw: 700 },
    { t: 'During backward(): autograd hooks launch bucket allreduces in reverse parameter order.', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function jaxMeshSharding() {
  const W = 560, H = 278;
  let o = header(W, 'Declarative Data Parallelism in JAX: Inputs → Partitioner → HLO');
  o += card(20, 40, 164, 152, C.lblue, C.blue, '1. Sharding Specs', [
    { t: 'model, opt: P()', mono: true, fill: D.blue.dark, fw: 700 },
    '(replicated on mesh)',
    { t: 'batch idx: P("data")', mono: true, fill: D.purple.dark, fw: 700 },
    '(split across k GPUs)',
  ], D.blue.dark);
  o += arrow(186, 116, 198, 116, C.blue);
  o += card(200, 40, 160, 152, D.purple.tint, C.purple, '2. Single-GPU Code', [
    { t: '@nnx.jit', mono: true, fill: D.purple.dark, fw: 700 },
    { t: 'def train_step(...):', mono: true },
    'loss, g = value_and_grad',
    'opt.update(model, g)',
  ], D.purple.dark);
  o += arrow(362, 116, 374, 116, C.purple);
  o += card(376, 40, 164, 152, C.lgreen, C.green, '3. Compiled XLA HLO', [
    { t: 'GSPMD / Shardy', mono: true, fill: D.green.dark, fw: 700 },
    'Combines grads ≤30MiB',
    'all-reduce-start / done',
    'Global BatchNorm stats',
  ], D.green.dark);
  o += footerBox(20, 206, 520, 58, C.lamber, C.amber, [
    { t: 'Zero manual collective calls: differentiating a mean loss over P("data") wrt P() weights', fill: C.ink },
    { t: 'automatically synthesizes asynchronous gradient allreduces across the "data" axis!', fill: D.orange.dark, fw: 700 },
  ]);
  return svg(W, H, o);
}

function bucketTradeoff() {
  const W = 560, H = 282;
  let o = header(W, 'Gradient Bucketing Trade-Off & the Ideal Bucket Overlap Model');
  o += card(16, 40, 170, 158, D.red.tint, D.red.base, 'Tiny Buckets (c ≫ 1)', [
    { t: '1 call per tensor', mono: true, fill: D.red.dark, fw: 700 },
    'Starts immediately,',
    'but pays latency c·α',
    'dozens of times and',
    'fragments link BW',
  ], D.red.dark);
  o += card(195, 40, 170, 158, C.lgreen, C.green, '25–30 MiB Buckets', [
    { t: '2–3 buckets for 45 MB', mono: true, fill: D.green.dark, fw: 700 },
    'Reverse layer order:',
    'Buckets 1..c−1 hide',
    'behind backward();',
    { t: 'Only S_last exposed!', fill: D.green.dark, fw: 700 },
  ], D.green.dark);
  o += card(374, 40, 170, 158, C.lamber, C.amber, '1 Giant Bucket (c = 1)', [
    { t: 'Flat S = 45 MB buffer', mono: true, fill: D.orange.dark, fw: 700 },
    'Pays α once, but waits',
    'for Layer 1 at very',
    'end of backward()',
    { t: 'Full S exposed!', fill: D.orange.dark, fw: 700 },
  ], D.orange.dark);
  o += footerBox(16, 212, 528, 56, C.lblue, C.blue, [
    { t: 'Exposed, ideal overlap: 2(k−1)/k·S_last·β_link  vs.  none: 2(k−1)/k·S·β_link', mono: true, fill: D.blue.dark, fw: 700 },
    { t: 'Requires: (1) fast links (bucket finishes before next fills) & (2) zero SM contention.', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function dataShardBugs() {
  const W = 560, H = 282;
  let o = header(W, 'Two Silent Multi-GPU Data-Sharding Bugs vs. Correct Partitioning');
  o += card(16, 40, 170, 158, D.red.tint, D.red.base, 'Bug 1: Cloned Shards', [
    { t: 'Forgot rank offset!', mono: true, fill: D.red.dark, fw: 700 },
    'GPU 0: [ex 0, 1, 2, 3]',
    'GPU 1: [ex 0, 1, 2, 3]',
    'Throughput looks 2×,',
    'but wastes k−1 GPUs!',
  ], D.red.dark);
  o += card(195, 40, 170, 158, C.lamber, C.amber, 'Bug 2: Unseeded RNG', [
    { t: 'Different seed/rank', mono: true, fill: D.orange.dark, fw: 700 },
    'GPU 0: [ex 4, 1, 7, 2]',
    'GPU 1: [ex 2, 9, 4, 5]',
    'Overlapping samples',
    'within the same step!',
  ], D.orange.dark);
  o += card(374, 40, 170, 158, C.lgreen, C.green, 'Correct Disjoint Split', [
    { t: 'Shared seed + rank r', mono: true, fill: D.green.dark, fw: 700 },
    'GPU 0: [ex 4, 1, 7, 2]',
    'GPU 1: [ex 8, 0, 3, 6]',
    'Disjoint partition of',
    'global batch B = k·b!',
  ], D.green.dark);
  o += footerBox(16, 212, 528, 56, C.lblue, C.blue, [
    { t: 'Defensive Check: Assert on step 1 that minibatch checksums differ across ranks under 1 shared seed!', fill: D.blue.dark, fw: 700 },
  ]);
  return svg(W, H, o);
}

function pcieVsNvlinkDdp() {
  const W = 560, H = 278;
  let o = header(W, 'ResNet-18 DDP Scaling (S = 45 MB, k = 4 GPUs, 100 ms Compute)');
  o += card(20, 40, 250, 156, C.lamber, C.amber, 'Host-Staged PCIe 4.0 (Our 4× 4090)', [
    { t: 'Bus BW ≈ 2.4 GB/s (0.42 ns/B)', mono: true, fill: D.orange.dark, fw: 700 },
    'Allreduce time (k=4): ~28 ms',
    'NCCL SM copy kernels contend with',
    'backward convs (+10 ms slowdown)',
    { t: 'Weak-Scaling: 3.1–3.3× (77–83%)', mono: true, fill: D.orange.dark, fw: 700 },
  ], D.orange.dark);
  o += card(290, 40, 250, 156, C.lgreen, C.green, 'Data-Center NVLink 4 / 5 (H100/B200)', [
    { t: 'Link BW: 450–900 GB/s (200× faster)', mono: true, fill: D.green.dark, fw: 700 },
    'Allreduce time (k=4): < 0.2 ms!',
    'Dedicated NVLink engines (no host',
    'DRAM staging or SM contention)',
    { t: 'Weak-Scaling: ~4.0× (> 99% eff!)', mono: true, fill: D.green.dark, fw: 700 },
  ], D.green.dark);
  o += footerBox(20, 210, 520, 54, D.purple.tint, C.purple, [
    { t: 'Even on NVLink, DDP still replicates 16N state on every card → motivates FSDP (§13.7)!', fill: D.purple.dark, fw: 700 },
  ]);
  return svg(W, H, o);
}

function ddpSummary(fw) {
  const jax = fw === 'jax';
  const W = 560, H = 280;
  let o = header(W, 'Production Data Parallelism: End-to-End Architecture');
  o += card(20, 38, 250, 106, C.lblue, C.blue, '1. Parallel Dispatch (SPMD)', [
    jax ? '1 jitted program on Auto mesh' : 'k processes via torchrun',
    jax ? 'XLA inserts the collectives' : '1 Python process per GPU',
    'No single-thread k·h_step bottleneck',
  ], D.blue.dark);
  o += card(290, 38, 250, 106, C.lgreen, C.green, '2. Reverse-Order Bucketing', [
    'Packs grads in 25–30 MiB buckets',
    'Launches async allreduce in backward()',
    'Hides comm when link outpaces bwd',
  ], D.green.dark);
  o += card(20, 158, 250, 106, C.lamber, C.amber, '3. BatchNorm Semantics Check', [
    jax ? 'Auto sharding: global batch stats' : 'DDP: per-shard stats by default',
    jax ? 'Matches 1 GPU on the full batch' : 'SyncBatchNorm: global stats',
    'Verify with 1-step fp32 update check',
  ], D.orange.dark);
  o += card(290, 158, 250, 106, D.purple.tint, C.purple, '4. Link & SM Contention Audit', [
    'Compare bare β_link vs. step profile',
    'Host-staged PCIe NCCL uses GPU SMs',
    'Grad accumulation amortizes slow links',
  ], D.purple.dark);
  return svg(W, H, o);
}

// ── §13.7 sharding.md ───────────────────────────────────────────────────

function stateLifetime() {
  const W = 560, H = 280;
  let o = header(W, 'When Does a GPU Actually Touch Each Component of 16N State?');
  o += card(20, 38, 520, 68, C.lblue, C.blue, '1. Weights W_ℓ (4 B/param = 7.15 GiB on 1.92B GPT)', [
    'Needed ONLY while block ℓ executes its forward or backward matrix products!',
  ], D.blue.dark);
  o += card(20, 118, 520, 68, C.lamber, C.amber, '2. Gradients ∇W_ℓ (4 B/param = 7.15 GiB on 1.92B GPT)', [
    'Produced when block ℓ finishes backward(); only 1/k shard is needed by GPU j for Adam!',
  ], D.orange.dark);
  o += card(20, 198, 520, 68, D.purple.tint, C.purple, '3. Adam 1st & 2nd Moments m, v (8 B/param = 14.3 GiB on 1.92B GPT)', [
    'Touched ONLY inside the elementwise optimizer update at the very end of the step!',
  ], D.purple.dark);
  return svg(W, H, o);
}

function zeroStages() {
  const W = 560, H = 284;
  let o = header(W, 'ZeRO Stages 1–3 on 36-Block 1.92B GPT (k = 2 GPUs, 24 GiB Cap)');
  const stages = [
    { x: 16,  w: 124, name: 'DDP (Full)', b: '16 B/param', m: '28.6 GiB', s: 'OOM (>24G)!', col: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: 150, w: 124, name: 'ZeRO-1 (Mom)', b: '8 + 8/k B', m: '21.4 GiB', s: 'Tight (<3G)', col: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: 284, w: 124, name: 'ZeRO-2 (+Grad)', b: '4 + 12/k B', m: '17.9 GiB', s: '6.1G free', col: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: 418, w: 126, name: 'ZeRO-3 / FSDP', b: '16 / k B', m: '14.3 GiB', s: '9.7G free!', col: C.lgreen, st: C.green, dk: D.green.dark },
  ];
  stages.forEach((s) => {
    o += card(s.x, 40, s.w, 152, s.col, s.st, s.name, [
      { t: s.b, mono: true, fs: 12.5, fw: 700 },
      { t: s.m, mono: true, fs: 13.5, fw: 700, fill: s.dk },
      { t: s.s, fs: 12.5, fw: 700, fill: C.ink },
    ], s.dk);
  });
  o += footerBox(16, 206, 528, 64, D.purple.tint, C.purple, [
    { t: 'ZeRO-1 & ZeRO-2 move the exact same 2(k−1)/k · S bytes as DDP!', fill: D.purple.dark, fw: 700 },
    { t: 'ZeRO-3 (FSDP) shards weights too (16/k B/param) at 1.5× comm volume (3(k−1)/k · S).', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function fourCollectives() {
  const W = 560, H = 284;
  let o = header(W, 'Four Collective Primitives Behind Parallel & Sharded Training');
  o += card(20, 38, 250, 110, C.lblue, C.blue, '1. Allreduce (DDP Grads)', [
    { t: '[a₀, a₁]+[b₀, b₁] → [a₀+b₀, a₁+b₁]', mono: true, fill: D.blue.dark, fw: 700 },
    'Every GPU ends with the full sum',
    { t: 'Traffic: 2(k−1)/k · S (= RS + AG)', mono: true },
  ], D.blue.dark);
  o += card(290, 38, 250, 110, C.lgreen, C.green, '2. Reduce-Scatter (ZeRO/FSDP)', [
    { t: '[a₀, a₁]+[b₀, b₁] → GPU0:[a₀+b₀]', mono: true, fill: D.green.dark, fw: 700 },
    'Each GPU gets 1/k shard of the sum',
    { t: 'Traffic: (k−1)/k · S (½ Allreduce)', mono: true },
  ], D.green.dark);
  o += card(20, 160, 250, 110, C.lamber, C.amber, '3. All-Gather (ZeRO/FSDP Weights)', [
    { t: 'GPU0:[w₀], GPU1:[w₁] → [w₀, w₁]', mono: true, fill: D.orange.dark, fw: 700 },
    'Assembles full weights from 1/k shards',
    { t: 'Traffic: (k−1)/k · S (½ Allreduce)', mono: true },
  ], D.orange.dark);
  o += card(290, 160, 250, 110, D.purple.tint, C.purple, '4. All-to-All (MoE Routing)', [
    { t: 'GPU0:[t₀₀,t₀₁] → GPU0:[t₀₀,t₁₀]', mono: true, fill: D.purple.dark, fw: 700 },
    'Transposes distinct slices across GPUs',
    { t: 'Traffic: (k−1)/k · S (MoE experts)', mono: true },
  ], D.purple.dark);
  return svg(W, H, o);
}

function zero1Pipeline() {
  const W = 560, H = 278;
  let o = header(W, 'ZeRO Stage 1: Inserting Sharded Adam Between RS and AG');
  o += card(16, 42, 168, 146, C.lblue, C.blue, '1. Reduce-Scatter', [
    { t: 'Gradients ∇W', mono: true, fill: D.blue.dark, fw: 700 },
    'Sums grads across k',
    'GPUs; GPU j receives',
    'only its 1/k shard g_j',
    { t: 'Sent: (k−1)/k · S', mono: true, fill: D.blue.dark },
  ], D.blue.dark);
  o += arrow(186, 115, 196, 115, C.blue);
  o += card(198, 42, 164, 146, C.lgreen, C.green, '2. Sharded Adam', [
    { t: 'O(N / k) State & Math', mono: true, fill: D.green.dark, fw: 700 },
    'GPU j updates only',
    'its owned 1/k weights',
    'using m_j, v_j (8N/k B)',
    { t: 'Sent: 0 bytes!', mono: true, fill: D.green.dark },
  ], D.green.dark);
  o += arrow(364, 115, 374, 115, C.green);
  o += card(376, 42, 168, 146, C.lamber, C.amber, '3. All-Gather', [
    { t: 'Updated Weights W', mono: true, fill: D.orange.dark, fw: 700 },
    'Broadcasts each 1/k',
    'updated weight shard',
    'to all k GPUs',
    { t: 'Sent: (k−1)/k · S', mono: true, fill: D.orange.dark },
  ], D.orange.dark);
  o += footerBox(16, 204, 528, 60, D.purple.tint, C.purple, [
    { t: 'Total Traffic = (k−1)/k · S + (k−1)/k · S = 2(k−1)/k · S (Identical to DDP Allreduce!)', mono: true, fill: D.purple.dark, fw: 700 },
    { t: 'Cuts Adam moment memory from 8N → 8N/k and optimizer compute by k× for free!', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function fsdpCommScaling() {
  const W = 560, H = 280;
  let o = header(W, 'FSDP Communication Tax: 3 Collective Phases = 1.5× DDP Traffic');
  o += card(16, 40, 168, 136, C.lblue, C.blue, 'Phase 1: Forward AG', [
    { t: 'All-Gather Weights', mono: true, fill: D.blue.dark, fw: 700 },
    'Assembles block ℓ,',
    'then frees non-owned',
    { t: '(k−1)/k · S bytes', mono: true, fill: D.blue.dark },
  ], D.blue.dark);
  o += card(196, 40, 168, 136, C.lamber, C.amber, 'Phase 2: Backward AG', [
    { t: 'All-Gather Weights', mono: true, fill: D.orange.dark, fw: 700 },
    'Re-gathers block ℓ',
    'for backward VJPs!',
    { t: '(k−1)/k · S bytes', mono: true, fill: D.orange.dark },
  ], D.orange.dark);
  o += card(376, 40, 168, 136, C.lgreen, C.green, 'Phase 3: Backward RS', [
    { t: 'Reduce-Scatter Grads', mono: true, fill: D.green.dark, fw: 700 },
    'Sums & shards ∇W_ℓ',
    'to owner GPU j',
    { t: '(k−1)/k · S bytes', mono: true, fill: D.green.dark },
  ], D.green.dark);
  o += footerBox(16, 192, 528, 74, D.purple.tint, C.purple, [
    { t: 't_comm(FSDP) = 3(k−1)/k · S · β_link  (grows from 1.5 S at k=2 → 3 S as k → ∞)', mono: true, fill: D.purple.dark, fw: 700 },
    { t: 'Memory per GPU drops as 16N / k, while bytes sent per GPU grow with k!', fill: D.red.dark, fw: 700 },
    { t: 'Tip: bf16 collective comm halves S_comm → 1.5(k−1)/k · S (25% less than fp32 DDP!).', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function fsdpPrefetch() {
  const W = 560, H = 280;
  let o = header(W, 'FSDP Block Prefetching: Host-Staged PCIe vs. NVLink 5');
  o += card(20, 38, 520, 102, D.red.tint, D.red.base, '(a) PCIe 4.0 via Host DRAM (2.2 GB/s): Link-Bound (g ≫ c)', [
    { t: '1.92B GPT (1.5 S = 11.5 GB): Comm L·g ≈ 5.0 s  vs.  Compute L·c ≈ 0.40 s', mono: true, fill: D.red.dark, fw: 700 },
    'GPU 0 spends > 90% of the 5.5 s step waiting on NCCL collectives!',
  ], D.red.dark);
  o += arrow(280, 142, 280, 160, C.green);
  o += card(20, 162, 520, 106, C.lgreen, C.green, '(b) NVLink 5 (900 GB/s) + Next-Block Prefetching: Compute-Bound (g ≤ c)', [
    { t: 'Comm Stream: [AG Blk 1][AG Blk 2][AG Blk 3]  (11.5 GB takes only ~13 ms!)', mono: true, fill: D.green.dark },
    { t: 'Comp Stream:          [Comp Blk 1][Comp Blk 2] → Forward = g + (L−1)max(g,c) + c', mono: true, fill: D.green.dark, fw: 700 },
  ], D.green.dark);
  return svg(W, H, o);
}

function ddpVsFsdpDecision() {
  const W = 560, H = 280;
  let o = header(W, 'When to Replicate (DDP + Accumulation) vs. Shard (ZeRO / FSDP)');
  o += card(20, 40, 250, 152, C.lgreen, C.green, '16N Fits on 1 GPU (e.g. 1B GPT)', [
    { t: 'Use DDP + Grad Accumulation', fill: D.green.dark, fw: 700 },
    'Over m micro-batches, weights stay',
    'resident on card; GPUs communicate',
    { t: 'ONLY ONCE per global step!', fill: D.green.dark, fw: 700 },
    { t: 'Total comm: 2(k−1)/k · S', mono: true, fill: C.ink },
  ], D.green.dark);
  o += card(290, 40, 250, 152, C.lamber, C.amber, '16N Exceeds 1 GPU (e.g. 1.9B GPT)', [
    { t: 'Escalate ZeRO-1/2 → ZeRO-3 (FSDP)', fill: D.orange.dark, fw: 700 },
    'ZeRO-1/2 keep weights & comm once;',
    'FSDP frees weights per block, so it',
    { t: 're-gathers 2× per micro-batch!', fill: D.red.dark, fw: 700 },
    { t: 'FSDP comm: (2m + 1)(k−1)/k · S', mono: true, fill: C.ink },
  ], D.orange.dark);
  o += footerBox(20, 206, 520, 60, C.lblue, C.blue, [
    { t: 'Escalation Rule on Slow PCIe Links:', fill: D.blue.dark, fw: 700 },
    { t: 'DDP + Accum (16N fits) → ZeRO-1/2 (8N or 4N fits) → ZeRO-3 / FSDP (when 4N > VRAM).', fill: C.ink },
  ]);
  return svg(W, H, o);
}

// ── §13.8 fast-transformer.md ───────────────────────────────────────────

function gptGapOverview() {
  const W = 560, H = 278;
  let o = header(W, '19M-Param Width-512 GPT (8,192 Tokens/Step = 0.93 TFLOP)');
  o += card(20, 40, 164, 152, D.red.tint, D.red.base, 'Eager fp32 Baseline', [
    { t: '~25.0 ms / step', mono: true, fill: D.red.dark, fw: 700 },
    '~330K tokens/s',
    '~23% bf16 MFU',
    '15.6 ms tf32 GEMMs',
    '+ 8.8 ms small kernels',
  ], D.red.dark);
  o += arrow(186, 116, 196, 116, C.amber);
  o += card(198, 40, 164, 152, C.lamber, C.amber, 'tf32 Math Floor', [
    { t: '11.3 ms ideal', mono: true, fill: D.orange.dark, fw: 700 },
    'At P = 82.6 TFLOP/s',
    'Half the 25 ms gap is',
    'number format alone;',
    'half is memory traffic!',
  ], D.orange.dark);
  o += arrow(364, 116, 374, 116, C.green);
  o += card(376, 40, 164, 152, C.lgreen, C.green, 'bf16 Math Floor', [
    { t: '5.7 ms ideal', mono: true, fill: D.green.dark, fw: 700 },
    'At P = 165.2 TFLOP/s',
    'Optimized step reaches',
    '~11.5 ms (~50% MFU)',
    '+ 3× on 4 GPUs!',
  ], D.green.dark);
  o += footerBox(20, 206, 520, 58, C.lblue, C.blue, [
    { t: 'Case-Study Discipline: Profile Current Step → Bound Next Gain → Verify Logits → Measure', fill: D.blue.dark, fw: 700 },
    { t: 'Every rung is applied cumulatively on the 270 MB GPU-resident WikiText-103 byte corpus.', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function width512Roofline() {
  const W = 560, H = 278;
  let o = header(W, 'Why Width d = 512 Places Block Projections Cleanly Above I*');
  o += card(20, 40, 250, 148, C.lblue, C.blue, 'tf32 Projections (b = 4 B, n = 8192)', [
    { t: 'I = 125–185 FLOP/byte', mono: true, fill: D.blue.dark, fw: 700 },
    'RTX 4090 tf32 Ridge: I* = 82',
    'Cleanly compute-bound (I > I*)!',
    'At d = 256: I ≈ 63–94 (straddles I*)',
  ], D.blue.dark);
  o += card(290, 40, 250, 148, C.lgreen, C.green, 'bf16 Projections (b = 2 B, n = 8192)', [
    { t: 'I = 250–370 FLOP/byte', mono: true, fill: D.green.dark, fw: 700 },
    'RTX 4090 bf16 Ridge: I* = 164',
    'Cleanly compute-bound (I > I*)!',
    'Doubling P directly halves GEMM time',
  ], D.green.dark);
  o += footerBox(20, 204, 520, 60, D.purple.tint, C.purple, [
    { t: 'Projection Intensity:  I = 2·n·d·d\' / [b·(n·d + d·d\' + n·d\')]', mono: true, fill: D.purple.dark, fw: 700 },
    { t: 'Because n = 8,192 ≫ d = 512, layer width d is the smallest dimension governing I.', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function amdahlBounds() {
  const W = 560, H = 284;
  let o = header(W, 'Using the Current Step Profile to Bound the Next Speedup');
  o += card(20, 38, 250, 108, C.lblue, C.blue, '1. Host Overhead Removal', [
    { t: 'Target: Idle Gap (t_step − t_busy)', mono: true, fill: D.blue.dark, fw: 700 },
    { t: 'Max Speedup ≤ t_step / t_busy', mono: true, fw: 700 },
    'Baseline busy 98% → ≤ 1.02× now',
  ], D.blue.dark);
  o += card(290, 38, 250, 108, C.lamber, C.amber, '2. Operator Fusion (Compile)', [
    { t: 'Target: Non-Matmul Time t_other', mono: true, fill: D.orange.dark, fw: 700 },
    { t: 'Max Speedup ≤ t_step / t_products', mono: true, fw: 700 },
    'Products take 64% → ≤ 1.56× bound',
  ], D.orange.dark);
  o += card(20, 158, 250, 112, C.lgreen, C.green, '3. Reduced Precision (tf32 → bf16)', [
    { t: 'Target: Matmul Share f ≈ 0.68', mono: true, fill: D.green.dark, fw: 700 },
    { t: 'Speedup ∈ [1/(1 − f/2), 2.0×]', mono: true, fw: 700 },
    '1.52× (GEMMs) to 2.0× (+2B acts)',
  ], D.green.dark);
  o += card(290, 158, 250, 112, D.purple.tint, C.purple, '4. Batch 64→512 & Checkpointing', [
    { t: 'Batch 8×: amortizes 0.75ms fixed', mono: true, fill: D.purple.dark, fw: 700 },
    'Predicts ~1.05× (+ enables 4-GPU DP)',
    'Remat branch: ≥ 6/8 = 0.75× retention',
  ], D.purple.dark);
  return svg(W, H, o);
}

function multigpuBatchCrossover() {
  const W = 560, H = 280;
  let o = header(W, 'Multi-GPU Scaling of 19M GPT (S = 76 MB Grads, β_link = 0.42 ns/B)');
  o += card(20, 40, 250, 156, D.red.tint, D.red.base, 'Short Step: fp32, Batch 64 / GPU', [
    { t: '1-GPU Step: t(1) ≈ 22 ms', mono: true, fill: D.red.dark, fw: 700 },
    '2-GPU Allreduce: 31 ms (> 22 ms!)',
    '4-GPU Allreduce: 48 ms (> 22 ms!)',
    { t: '2 GPUs: 0.85× (SLOWDOWN!)', mono: true, fill: D.red.dark, fw: 700 },
    { t: '4 GPUs: ~1.35× (Comm-starved)', mono: true, fill: C.ink },
  ], D.red.dark);
  o += card(290, 40, 250, 156, C.lgreen, C.green, 'Long Step: bf16, Batch 512 / GPU', [
    { t: '1-GPU Step: t(1) ≈ 95 ms', mono: true, fill: D.green.dark, fw: 700 },
    '2-GPU Allreduce: 31 ms (≪ 95 ms)',
    '4-GPU Allreduce: 48 ms (< ½ of 95 ms)',
    { t: '2 GPUs: ~1.6× (beats 1.5× floor)', mono: true, fill: D.green.dark, fw: 700 },
    { t: '4 GPUs: ~3.0× (2.1M tok/s = 6×!)', mono: true, fill: D.green.dark, fw: 700 },
  ], D.green.dark);
  o += footerBox(20, 210, 520, 56, C.lblue, C.blue, [
    { t: 'No-Overlap Speedup Floor:  Speedup(k) ≥ k · t(1) / [t(1) + t_allreduce(k)]', mono: true, fill: D.blue.dark, fw: 700 },
    { t: 'Margin above the floor measures backward-pass bucket overlap on our PCIe workstation.', fill: C.ink },
  ]);
  return svg(W, H, o);
}

function criticalBatchNoise() {
  const W = 560, H = 278;
  let o = header(W, 'Token Throughput vs. Time-to-Target-Loss: E(b) = E_min(1 + b / b_noise)');
  o += card(20, 40, 250, 150, C.lgreen, C.green, 'Small Batch Regime (b ≪ b_noise)', [
    { t: 'E(b) ≈ E_min (constant tokens)', mono: true, fill: D.green.dark, fw: 700 },
    'Doubling batch size b halves steps',
    'needed to reach target validation loss',
    'Higher tok/s translates 1:1 into',
    'shorter time-to-target-loss!',
  ], D.green.dark);
  o += card(290, 40, 250, 150, C.lamber, C.amber, 'Large Batch Regime (b ≫ b_noise)', [
    { t: 'E(b) ∝ b / b_noise (more tokens!)', mono: true, fill: D.orange.dark, fw: 700 },
    'Gradient noise is already averaged out',
    'Larger batches make same step progress',
    '200 steps @ B=512 (13M tok, loss 2.08)',
    'lags 1000 steps @ B=64 (8M tok, 1.35)',
  ], D.orange.dark);
  o += footerBox(20, 206, 520, 58, C.lblue, C.blue, [
    { t: 'Batch-Preserving Ops (compile, bf16, FlashAttention, strong DP) Always Cut Time-to-Target;', fill: D.blue.dark, fw: 700 },
    { t: 'Batch-Enlarging Ops (B=512, weak DP) Cut Time-to-Target iff b_noise > (512 − 64ρ)/(ρ − 1).', mono: true, fill: C.ink },
  ]);
  return svg(W, H, o);
}

export const diagrams = {
  // index.md
  'ch13-perf-four-bottlenecks':         fourBottlenecks,
  'ch13-perf-three-floors':             threeFloors,
  'ch13-perf-host-constants-pytorch':   () => hostConstants('pytorch'),
  'ch13-perf-host-constants-jax':       () => hostConstants('jax'),
  'ch13-perf-roadmap':                  roadmap,
  // performance-model.md
  'ch13-perf-matmul-utilization':       matmulUtilization,
  'ch13-perf-arithmetic-intensity':     arithmeticIntensity,
  'ch13-perf-linear-reuse':             linearReuse,
  'ch13-perf-gpt-intensity-ladder':     gptIntensityLadder,
  'ch13-perf-overhead-knee-pytorch':    () => overheadKnee('pytorch'),
  'ch13-perf-overhead-knee-jax':        () => overheadKnee('jax'),
  // compilation.md
  'ch13-perf-eager-penalties':          eagerPenalties,
  'ch13-perf-branch-strategies-pytorch': () => branchStrategies('pytorch'),
  'ch13-perf-branch-strategies-jax':    () => branchStrategies('jax'),
  'ch13-perf-shape-bucketing':          shapeBucketing,
  'ch13-perf-ffn-fusion-ratio':         ffnFusionRatio,
  'ch13-perf-fusion-scope':             fusionScope,
  'ch13-perf-cuda-graph-replay':        cudaGraphReplay,
  // hardware.md
  'ch13-perf-decode-bottleneck':        decodeBottleneck,
  'ch13-perf-hockney-transfer':         hockneyTransfer,
  'ch13-perf-generational-ridge':       generationalRidge,
  'ch13-perf-sm-latency-hiding':        smLatencyHiding,
  'ch13-perf-tensor-core-math':         tensorCoreMath,
  'ch13-perf-sweep-time-invariant':     sweepTimeInvariant,
  // memory-precision.md
  'ch13-perf-24gib-squeeze':            memorySqueeze24GiB,
  'ch13-perf-vjp-save-rules':           vjpSaveRules,
  'ch13-perf-block-activation-tally':   blockActivationTally,
  'ch13-perf-mixed-precision-flow':     mixedPrecisionFlow,
  'ch13-perf-activation-checkpointing': activationCheckpointing,
  'ch13-perf-grad-accumulation':        gradAccumulation,
  // multiple-gpus.md
  'ch13-perf-second-gpu-tradeoff':      secondGpuTradeoff,
  'ch13-perf-shard-grad-averaging':     shardGradAveraging,
  'ch13-perf-star-vs-ring':             starVsRing,
  'ch13-perf-strong-vs-weak-scaling':   strongVsWeakScaling,
  // multi-gpu-practice.md
  'ch13-perf-lenet-vs-resnet18':        lenetVsResnet18,
  'ch13-perf-spmd-models-pytorch':      () => spmdModels('pytorch'),
  'ch13-perf-spmd-models-jax':          () => spmdModels('jax'),
  'ch13-perf-torchrun-ddp':             torchrunDdp,
  'ch13-perf-jax-mesh-sharding':        jaxMeshSharding,
  'ch13-perf-bucket-tradeoff':          bucketTradeoff,
  'ch13-perf-data-shard-bugs':          dataShardBugs,
  'ch13-perf-pcie-vs-nvlink-ddp':       pcieVsNvlinkDdp,
  'ch13-perf-ddp-summary-pytorch':      () => ddpSummary('pytorch'),
  'ch13-perf-ddp-summary-jax':          () => ddpSummary('jax'),
  // sharding.md
  'ch13-perf-state-lifetime':           stateLifetime,
  'ch13-perf-zero-stages':              zeroStages,
  'ch13-perf-four-collectives':         fourCollectives,
  'ch13-perf-zero1-pipeline':           zero1Pipeline,
  'ch13-perf-fsdp-comm-scaling':        fsdpCommScaling,
  'ch13-perf-fsdp-prefetch':            fsdpPrefetch,
  'ch13-perf-ddp-vs-fsdp-decision':     ddpVsFsdpDecision,
  // fast-transformer.md
  'ch13-perf-gpt-gap-overview':         gptGapOverview,
  'ch13-perf-width512-roofline':        width512Roofline,
  'ch13-perf-amdahl-bounds':            amdahlBounds,
  'ch13-perf-multigpu-batch-crossover': multigpuBatchCrossover,
  'ch13-perf-critical-batch-noise':     criticalBatchNoise,
};
