// Diagrams for chapter_preliminaries/index.md (§1 Chapter Opener).
//
// Three high-level conceptual diagrams anchoring the chapter opener slides:
//   prelim-pillars          Two software pillars (Contiguous Tensor Kernels +
//                           Reverse-Mode Autograd DAGs) resting on Statistical
//                           Uncertainty Quantification.
//   prelim-ml-loop          Closed 6-stage machine learning iteration loop
//                           mapping directly to §§1.1–1.6.
//   prelim-running-example  The chapter's running benchmark: minibatch image
//                           classification O = XW + b and reverse-mode VJP.

import { C, TOKENS, grid, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;
const N = TOKENS.neutrals;

const box = (x, y, w, h, fill, stroke, rx = 8, sw = 2, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// 1. Two Software Pillars + Statistical Foundation
function pillars() {
  const W = 560, H = 340;
  let o = '';
  // Top roof bar: Deep Learning Training & Evaluation
  o += box(24, 14, 512, 46, D.purple.tint, C.purple, 8, 2);
  o += tx(280, 37, 'End-to-End Deep Learning (Preliminaries)', { fs: 17, fw: 700, fill: D.purple.dark });

  // Left Pillar: Contiguous Tensor Kernels (§§1.1-1.3)
  const lx = 24, rx = 292, py = 74, pw = 244, ph = 166;
  o += box(lx, py, pw, ph, C.lblue, C.blue, 10, 2);
  o += tx(lx + pw / 2, py + 24, 'Pillar 1: Tensor Kernels', { fs: 16, fw: 700, fill: D.blue.dark });
  o += tx(lx + pw / 2, py + 44, 'Contiguous GPU Memory', { fs: 13.5, fw: 600, fill: C.muted });
  // Mini contiguous buffer + SIMD lanes inside left pillar
  const s = 30, g = 4, gx = lx + 28, gy = py + 62;
  o += grid([['x₀', 'x₁', 'x₂', 'x₃', 'x₄']], gx, gy, s, g, {
    fs: 13,
    fill: () => ['#FFFFFF', C.blue, false, C.ink],
  });
  for (let i = 0; i < 5; i++) {
    const cx = gx + i * (s + g) + s / 2;
    o += arrow(cx, gy + s + 4, cx, gy + s + 28, C.blue);
  }
  o += box(lx + 14, gy + s + 32, pw - 28, 34, '#FFFFFF', C.blue, 6, 1.8);
  o += tx(lx + pw / 2, gy + s + 49, 'O = XW + b (SIMD)', { mono: true, fs: 13.5, fw: 700, fill: D.blue.dark });

  // Right Pillar: Reverse-Mode Autograd (§§1.4-1.5)
  o += box(rx, py, pw, ph, D.red.tint, D.red.base, 10, 2);
  o += tx(rx + pw / 2, py + 24, 'Pillar 2: Reverse Autograd', { fs: 16, fw: 700, fill: D.red.dark });
  o += tx(rx + pw / 2, py + 44, 'Dynamic Compute DAG', { fs: 13.5, fw: 600, fill: C.muted });
  // Mini forward/backward DAG inside right pillar
  const ny = py + 104, nxs = [rx + 44, rx + 122, rx + 200], nlabs = ['w', 'O', 'L'];
  o += arrow(nxs[0] + 22, ny - 8, nxs[1] - 22, ny - 8, C.blue);
  o += arrow(nxs[1] + 22, ny - 8, nxs[2] - 22, ny - 8, C.blue);
  o += tx(rx + pw / 2, ny - 26, 'forward: values', { fs: 13, fw: 700, fill: D.blue.dark });
  o += arrow(nxs[2] - 22, ny + 10, nxs[1] + 22, ny + 10, D.red.base);
  o += arrow(nxs[1] - 22, ny + 10, nxs[0] + 22, ny + 10, D.red.base);
  o += tx(rx + pw / 2, ny + 32, 'backward: 1 VJP sweep → ∇L', { fs: 13, fw: 700, fill: D.red.dark });
  nxs.forEach((cx, i) => {
    o += box(cx - 20, ny - 18, 40, 36, '#FFFFFF', i === 2 ? D.red.base : C.gray, 6, 2);
    o += tx(cx, ny, nlabs[i], { mono: true, fs: 15, fw: 700, fill: C.ink });
  });

  // Foundation Bar: Statistical Uncertainty (§1.6)
  const fy = 254;
  o += box(24, fy, 512, 72, C.lgreen, C.green, 10, 2);
  o += tx(280, fy + 24, 'Foundation: Probability & Statistical Inference', { fs: 16, fw: 700, fill: D.green.dark });
  o += tx(280, fy + 50, 'Unbiased minibatch E[g] = ∇L  ·  Standard error σ / √n  ·  Bayes P(H | D)', { fs: 13, fw: 600, fill: C.ink });

  return svg(W, H, o);
}

// 2. Closed 6-Stage ML Iteration Loop
function mlLoop() {
  const W = 580, H = 330;
  let o = '';
  // 6 boxes arranged in a clockwise 3x2 loop:
  // Top row (left -> right): 1. Ingest (§1.2) -> 2. Batch (§1.1) -> 3. Forward (§1.3)
  // Bottom row (right -> left): 4. Loss/Slope (§1.4) -> 5. Backprop (§1.5) -> 6. Eval/Update (§1.6)
  const bw = 156, bh = 86;
  const xs = [18, 212, 406];
  const ty = 28, by = 198;

  const stages = [
    { x: xs[0], y: ty, sec: 'Pandas', title: '1. Clean & Encode', sub: 'CSV → float32 X, y', fill: C.lgray, st: C.gray, dk: C.ink },
    { x: xs[1], y: ty, sec: 'Tensors', title: '2. Store & Batch', sub: 'X: (256, 784) GPU', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: xs[2], y: ty, sec: 'LinAlg', title: '3. Forward Pass', sub: 'O = XW + b', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { x: xs[2], y: by, sec: 'Calculus', title: '4. Loss & Slopes', sub: 'L(O, y) & ∂L/∂O', fill: C.lamber, st: C.amber, dk: D.orange.dark },
    { x: xs[1], y: by, sec: 'Autograd', title: '5. Reverse VJP', sub: 'reverse sweep → ∇W', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { x: xs[0], y: by, sec: 'Prob & Stats', title: '6. Update & Eval', sub: 'W ← W − η∇W', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  // Top arrows (1 -> 2 -> 3)
  o += arrow(xs[0] + bw + 4, ty + bh / 2, xs[1] - 4, ty + bh / 2, C.blue);
  o += arrow(xs[1] + bw + 4, ty + bh / 2, xs[2] - 4, ty + bh / 2, C.blue);
  // Right down arrow (3 -> 4)
  o += arrow(xs[2] + bw / 2, ty + bh + 4, xs[2] + bw / 2, by - 4, D.orange.base);
  // Bottom arrows (4 -> 5 -> 6)
  o += arrow(xs[2] - 4, by + bh / 2, xs[1] + bw + 4, by + bh / 2, D.red.base);
  o += arrow(xs[1] - 4, by + bh / 2, xs[0] + bw + 4, by + bh / 2, D.red.base);
  // Left up arrow closing the loop (6 -> 2/3 next step)
  o += arrow(xs[0] + bw / 2, by - 4, xs[0] + bw / 2, ty + bh + 4, C.green);
  o += tx(xs[0] + bw / 2 + 10, (ty + bh + by) / 2, 'next batch', { fs: 13.5, fw: 700, fill: D.green.dark, anchor: 'start' });

  // Center loop badge
  o += box(196, 136, 188, 38, D.purple.tint, C.purple, 19, 1.5);
  o += tx(290, 155, '1 Training Iteration', { fs: 14.5, fw: 700, fill: D.purple.dark });

  stages.forEach((s) => {
    o += box(s.x, s.y, bw, bh, s.fill, s.st, 9, 2);
    o += tx(s.x + bw / 2, s.y + 20, s.sec, { mono: true, fs: 13, fw: 700, fill: s.dk });
    o += tx(s.x + bw / 2, s.y + 44, s.title, { fs: 15.5, fw: 700, fill: C.ink });
    o += tx(s.x + bw / 2, s.y + 68, s.sub, { mono: true, fs: 12.5, fw: 600, fill: C.muted });
  });

  return svg(W, H, o);
}

// 3. Running Benchmark: Minibatch Image Classification Dataflow
function runningExample() {
  const W = 560, H = 285;
  let o = '';
  const cy = 118;

  // Front 28x28 image block on left (single clean block with depth shadow badge below)
  o += box(24, cy - 42, 76, 68, C.lblue, C.blue, 6, 2);
  o += tx(62, cy - 16, '28×28', { mono: true, fs: 14, fw: 700, fill: D.blue.dark });
  o += tx(62, cy + 6, '× 256', { mono: true, fs: 13, fw: 600, fill: C.ink });
  o += tx(62, cy + 48, 'B = 256 images', { fs: 14, fw: 700, fill: C.ink });
  o += tx(62, cy + 68, '(256, 1, 28, 28)', { mono: true, fs: 13, fill: C.muted });

  // Arrow: reshape(256, -1)
  o += arrow(112, cy - 8, 158, cy - 8, C.purple);
  o += tx(135, cy - 26, 'reshape', { mono: true, fs: 13, fw: 700, fill: D.purple.dark });

  // Matrix X (256 x 784)
  o += box(164, cy - 52, 92, 88, C.lblue, C.blue, 6, 2);
  o += tx(210, cy - 16, 'X', { fs: 20, fw: 700, fill: D.blue.dark });
  o += tx(210, cy + 8, '256 × 784', { mono: true, fs: 13.5, fw: 600, fill: C.ink });
  o += tx(210, cy + 54, 'inputs', { fs: 14, fill: C.muted });

  // · W (784 x 10)
  o += tx(272, cy - 8, '·', { fs: 28, fw: 700, fill: C.ink });
  o += box(288, cy - 64, 58, 112, N.panel, C.gray, 6, 2);
  o += tx(317, cy - 16, 'W', { fs: 20, fw: 700, fill: C.ink });
  o += tx(317, cy + 8, '784×10', { mono: true, fs: 13, fw: 600, fill: C.ink });
  o += tx(317, cy + 64, '7,840 params', { fs: 13.5, fill: C.muted });

  // + b (10,)
  o += tx(362, cy - 8, '+', { fs: 22, fw: 700, fill: C.ink });
  o += box(378, cy - 20, 44, 24, C.lamber, C.amber, 5, 2);
  o += tx(400, cy - 8, 'b', { fs: 16, fw: 700, fill: D.orange.dark });
  o += tx(400, cy + 22, '(1, 10)', { mono: true, fs: 13, fill: D.orange.dark });

  // = O (256 x 10)
  o += tx(438, cy - 8, '=', { fs: 22, fw: 700, fill: C.ink });
  o += box(454, cy - 52, 84, 88, C.lgreen, C.green, 6, 2);
  o += tx(496, cy - 16, 'O', { fs: 20, fw: 700, fill: D.green.dark });
  o += tx(496, cy + 8, '256 × 10', { mono: true, fs: 13.5, fw: 600, fill: C.ink });
  o += tx(496, cy + 54, 'class logits', { fs: 14, fill: C.muted });

  // Reverse-mode gradient bar below
  const by = 236;
  o += arrow(496, by, 317, by, D.red.base);
  o += tx(406, by - 16, '1 reverse sweep: ∇_W L = Xᵀ (∂L/∂O)', { mono: true, fs: 13.5, fw: 700, fill: D.red.dark });
  o += tx(406, by + 20, 'updates all 7,850 parameters simultaneously', { fs: 14, fw: 600, fill: D.red.dark });

  return svg(W, H, o);
}

// 4. Study Roadmap: Chapter 1 Crash Course vs. MDL Appendix Deep Dives
function studyRoadmap() {
  const W = 560, H = 310;
  let o = '';
  o += tx(280, 22, 'Preliminaries Crash Course → Math Appendix Deep Dives', { fs: 16, fw: 700, fill: C.ink });

  const rows = [
    { ch1: 'Tensors & LinAlg', sub1: 'Shapes, strides, GEMM & norms', mdl: 'Appendix · Linear Algebra', sub2: 'SVD, spectra, conditioning', fill: C.lblue, st: C.blue, dk: D.blue.dark },
    { ch1: 'Calculus & Autograd', sub1: 'Gradients, chain rule & DAGs', mdl: 'Appendix · Calculus', sub2: 'Jacobians, VJPs, dual numbers', fill: D.red.tint, st: D.red.base, dk: D.red.dark },
    { ch1: 'Probability & Statistics', sub1: 'Bayes rule, expectations & 1/√n', mdl: 'Appendix · Probability & Stats', sub2: 'MLE, ELBO, concentration', fill: C.lgreen, st: C.green, dk: D.green.dark },
  ];

  rows.forEach((r, i) => {
    const y = 48 + i * 86;
    o += box(12, y, 240, 70, r.fill, r.st, 8, 2);
    o += tx(132, y + 24, r.ch1, { fs: 13.5, fw: 700, fill: r.dk });
    o += tx(132, y + 48, r.sub1, { fs: 12.5, fw: 600, fill: C.ink });

    o += arrow(258, y + 35, 302, y + 35, r.st);
    o += tx(280, y + 18, 'proofs', { mono: true, fs: 11.5, fw: 700, fill: r.dk });

    o += box(308, y, 240, 70, '#FFFFFF', r.st, 8, 2);
    o += tx(428, y + 24, r.mdl, { fs: 13.5, fw: 700, fill: r.dk });
    o += tx(428, y + 48, r.sub2, { fs: 12.5, fw: 600, fill: C.muted });
  });

  return svg(W, H, o);
}

export const diagrams = {
  'prelim-pillars':         pillars,
  'prelim-ml-loop':         mlLoop,
  'prelim-running-example': runningExample,
  'prelim-study-roadmap':   studyRoadmap,
};
