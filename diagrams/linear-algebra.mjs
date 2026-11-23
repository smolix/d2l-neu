// Diagrams for chapter_preliminaries/linear-algebra.md.
//
// Structural figures: axis reductions with keepdims, the matrix-vector
// product, and the shapes of the running classifier O = XW + b. Values
// match the executed notebook cells so the figures and the In/Out cards
// agree. The geometric figures of the section (dot product, norms, linear
// maps, eigenvectors, cosines) are matplotlib: tools/figstyle/remake_ch02.py
// and tools/gen_mdl_prelim_figures.py.
//
// Stable ids use the `linear-algebra-<concept>` prefix (the full id is the
// filename). Never rename.

import { C, FS, TOKENS, grid, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;   // tint / base / dark trios (dark = colored text)

// plain stroked line (no arrowhead)
const line = (x1, y1, x2, y2, color, w = 1.5, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
// Text vertically centered on y. An alphabetic baseline 0.35 em below the
// center renders the same in browsers and in rsvg (the PDF path), where
// dominant-baseline="central" puts digits above the center of their cells.
const t = (x, y, s, o = {}) =>
  tx(x, y + 0.35 * (o.fs || 15), s, { ...o, base: 'alphabetic' });
// grid of cells (engine `grid`) with values centered by `t`
const cells = (data, x0, y0, s, g, fill, fs = 15) => {
  let o = grid(data, x0, y0, s, g, { fill, showVal: false });
  data.forEach((row, r) => row.forEach((v, c) => {
    o += t(x0 + c * (s + g) + s / 2, y0 + r * (s + g) + s / 2, v,
           { mono: true, fs, fill: fill(r, c, v)[3] || C.ink });
  }));
  return o;
};
// a plain rectangle (block of a schematic matrix)
const rect = (x, y, w, h, fill, stroke, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="4" fill="${fill}" stroke="${stroke}" stroke-width="2"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
// sans text with a true subscript, e.g. sub('o', 'ij')
const sub = (x, y, base, s, o = {}) =>
  `<text x="${x}" y="${y + 0.35 * (o.fs || 17)}" font-family="${FS}" font-size="${o.fs || 17}" font-weight="${o.fw || 700}" `
  + `fill="${o.fill || C.ink}" text-anchor="${o.anchor || 'middle'}" dominant-baseline="alphabetic">`
  + `${base}<tspan baseline-shift="sub" font-size="${Math.round(0.72 * (o.fs || 17))}">${s}</tspan></text>`;

// ── reductions: an axis disappears; keepdims keeps it as length 1 ────
// Left: A (2×3) summed over axis 0 (one sum per column, shape (3,)) and
// over axis 1 (one sum per row, shape (2,)).  Right: with keepdims=True
// the row sums keep shape (2, 1), so broadcasting stretches them across
// the 3 columns of A (dashed copies).  Values match the notebook cells.
function reduceAxes() {
  const s = 42, g = 6; let o = '';
  const A = [[0, 1, 2], [3, 4, 5]];        // 2×3, as in the cells
  const Aw = 3 * s + 2 * g, Ah = 2 * s + g;
  const blue = () => [C.lblue, C.blue, false, C.ink];
  const grn = () => [C.lgreen, C.green, false, C.ink];
  const amb = () => [C.lamber, C.amber, false, C.ink];
  const lab = (x, y, s, fill, anchor = 'middle') =>
    t(x, y, s, { mono: true, fs: 15, fw: 600, fill, anchor });
  // left: A and its two reductions
  const ax = 20, ay = 34;
  o += lab(ax + Aw / 2, ay - 18, 'A: shape (2, 3)', C.ink);
  o += cells(A, ax, ay, s, g, blue);
  const by = ay + Ah + 44;                  // axis=0 → below
  for (let c = 0; c < 3; c++) {
    const cx = ax + c * (s + g) + s / 2;
    o += arrow(cx, ay + Ah + 5, cx, by - 5, C.green);
  }
  o += cells([[3, 5, 7]], ax, by, s, g, grn);
  o += lab(ax + Aw / 2, by + s + 20, 'sum(axis=0)', D.green.dark);
  o += lab(ax + Aw / 2, by + s + 40, 'shape (3,)', D.green.dark);
  const rx = ax + Aw + 48;                  // axis=1 → right
  for (let r = 0; r < 2; r++) {
    const cy = ay + r * (s + g) + s / 2;
    o += arrow(ax + Aw + 5, cy, rx - 5, cy, C.amber);
  }
  o += cells([[3], [12]], rx, ay, s, g, amb);
  o += lab(rx + s / 2, ay + Ah + 20, 'sum(axis=1)', D.orange.dark);
  o += lab(rx + s / 2, ay + Ah + 40, 'shape (2,)', D.orange.dark);
  // divider
  const dx = rx + s + 70;
  o += line(dx, ay - 26, dx, by + s + 44, TOKENS.neutrals.hairline, 1.5);
  // right: keepdims=True keeps a (2, 1) column; broadcasting stretches it
  // (dashed copies) across the 3 columns, so A can be divided by it
  const title = 'sum over axis 1, keepdims';
  const hw = 0.3 * 15 * title.length;       // half width of the mono title
  const kx = dx + hw - Aw / 2 + 18;         // keep the title clear of the divider
  o += lab(kx + Aw / 2, ay - 18, title, D.orange.dark);
  o += cells([[3], [12]], kx, ay, s, g, amb);
  o += cells([[3, 3], [12, 12]], kx + s + g, ay, s, g,
             () => ['#FFFFFF', C.amber, true, C.muted]);
  o += lab(kx + Aw / 2, ay + Ah + 20, 'sum_A: shape (2, 1)', D.orange.dark);
  const ry = ay + Ah + 92;                  // the result A / sum_A
  o += arrow(kx + Aw / 2, ay + Ah + 36, kx + Aw / 2, ry - 6, C.ink);
  o += t(kx + Aw / 2 + 10, (ay + Ah + 30 + ry) / 2, 'A / sum_A', { mono: true, fs: 15, fw: 600, fill: C.ink, anchor: 'start' });
  const R = [['0', '.33', '.67'], ['.25', '.33', '.42']];
  o += cells(R, kx, ry, s, g, blue, 14.5);
  o += t(kx + Aw / 2, ry + Ah + 20, 'each row sums to 1', { fs: 15, fill: C.ink });
  return svg(kx + Aw / 2 + hw + 16, ry + Ah + 36, o);
}

// ── matrix–vector product: each output = one row · x ────────────────
// Values match the cell: A = [[0, 1, 2], [3, 4, 5]], x = (0, 1, 2).
// Row 1 of A and output 1 are highlighted; the caption states the rule.
function matvec() {
  const s = 44, g = 6; let o = '';
  const A = [[0, 1, 2], [3, 4, 5]];        // 2×3
  const ax = 20, ay = 72;
  const Aw = 3 * s + 2 * g, Ah = 2 * s + g;
  const off = ['#F0F2F5', '#C9CFD6', false, '#565D66'];
  o += t(ax + Aw / 2, ay - 20, 'A: (2, 3)', { mono: true, fs: 15, fw: 600, fill: C.ink });
  o += cells(A, ax, ay, s, g, (r) => r === 0 ? [C.lblue, C.blue, false, C.ink] : off);
  const cx = ax + Aw + 18;
  o += t(cx, ay + Ah / 2, '·', { fs: 30, fw: 700, fill: C.ink });
  const xx = cx + 18, xy = ay - (s + g) / 2;               // x: 3×1, vertically centered
  o += t(xx + s / 2, xy - 20, 'x: (3,)', { mono: true, fs: 15, fw: 600, fill: C.ink });
  o += cells([[0], [1], [2]], xx, xy, s, g, () => [C.lamber, C.amber, false, C.ink]);
  const ex = xx + s + 20;
  o += t(ex, ay + Ah / 2, '=', { fs: 26, fw: 700, fill: C.ink });
  const yx = ex + 20;
  o += t(yx + s / 2, ay - 20, 'Ax: (2,)', { mono: true, fs: 15, fw: 600, fill: C.ink });
  o += cells([[5], [14]], yx, ay, s, g, (r) => r === 0 ? [C.lgreen, C.green, false, C.ink] : off);
  const xBot = xy + 3 * s + 2 * g;          // x is the tallest column (3 rows)
  return svg(yx + s + 20, xBot + 16, o);
}

// ── the running classifier: O = XW + b on a batch (schematic) ───────
// X (256 × 784) holds one image per row, W (784 × 10) one column of
// weights per category, b (10) the offsets, O (256 × 10) the scores.
// Row i of X and column j of W (gold) meet in the score o_ij.  Block
// sizes are schematic: 784 is drawn shorter than to scale.
function matmul() {
  let o = '';
  const top = 34;                            // top of W, the tallest block
  const Wh = 196, cw = 40;                   // height of W; width of 10 columns
  const Xw = 196, Xh = 112;                  // X: 784 columns, 256 rows
  const mid = top + Wh / 2;
  const Xy = mid - Xh / 2;
  const xX = 30, xW = xX + Xw + 38, xb = xW + cw + 42, xO = xb + cw + 42;
  const gold = D.gold;
  const ri = Xy + 0.38 * Xh, rh = 14;        // highlighted row i
  const cj = 0.55 * cw, cwj = 9;             // highlighted column j (offset, width)
  // X with row i
  o += rect(xX, Xy, Xw, Xh, D.blue.tint, D.blue.base);
  o += rect(xX, ri, Xw, rh, gold.tint, gold.base);
  o += t(xX - 12, ri + rh / 2, 'i', { fs: 17, fw: 700, fill: C.ink, anchor: 'end' });
  // · W with column j
  o += t(xW - 19, mid, '·', { fs: 30, fw: 700, fill: C.ink });
  o += rect(xW, top, cw, Wh, TOKENS.neutrals.panel, C.gray);
  o += rect(xW + cj - cwj / 2, top, cwj, Wh, gold.tint, gold.base);
  o += t(xW + cj, top - 14, 'j', { fs: 17, fw: 700, fill: C.ink });
  // + b with entry j
  o += t(xb - 21, mid, '+', { fs: 26, fw: 700, fill: C.ink });
  o += rect(xb, mid - 8, cw, 16, TOKENS.neutrals.panel, C.gray);
  o += rect(xb + cj - cwj / 2, mid - 8, cwj, 16, gold.tint, gold.base);
  // = O with entry (i, j)
  o += t(xO - 21, mid, '=', { fs: 26, fw: 700, fill: C.ink });
  o += rect(xO, Xy, cw, Xh, D.green.tint, D.green.base);
  o += rect(xO + cj - cwj / 2, ri, cwj, rh, gold.base, gold.dark);
  o += sub(xO + cw + 10, ri + rh / 2, 'o', 'ij', { fs: 17, anchor: 'start' });
  // names and shapes below each block
  const ly = top + Wh + 22;
  const name = (x, s) => t(x, ly, s, { fs: 18, fw: 700, fill: C.ink });
  const shape = (x, s) => t(x, ly + 22, s, { mono: true, fs: 15, fill: C.ink });
  const role = (x, s) => t(x, ly + 43, s, { fs: 14.5, fill: C.muted });
  o += name(xX + Xw / 2, 'X') + shape(xX + Xw / 2, '256 × 784') + role(xX + Xw / 2, 'images, one per row');
  o += name(xW + cw / 2, 'W') + shape(xW + cw / 2, '784 × 10') + role(xW + cw / 2, 'weights');
  o += name(xb + cw / 2, 'b') + shape(xb + cw / 2, '10') + role(xb + cw / 2, 'offsets');
  o += name(xO + cw / 2, 'O') + shape(xO + cw / 2, '256 × 10') + role(xO + cw / 2, 'scores');
  return svg(xO + cw + 44, ly + 56, o);
}

// ── matrix transpose A -> A^T (row colors become column colors) ───────
function transpose() {
  const W = 520, H = 246; let o = '';
  const s = 38, g = 4;
  const A = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]];
  const AT = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]];
  const rowCols = [
    [C.lblue, C.blue, false, C.ink],
    [C.lgreen, C.green, false, C.ink],
    [C.lamber, C.amber, false, C.ink],
  ];
  o += t(104, 24, 'A: shape (3, 4)', { mono: true, fs: 15, fw: 700, fill: C.ink });
  o += cells(A, 22, 44, s, g, (r) => rowCols[r]);
  o += t(104, 184, 'rows 0, 1, 2', { fs: 13.5, fw: 600, fill: C.muted });

  o += arrow(202, 104, 256, 104, C.purple);
  o += t(229, 86, 'Aᵀ', { mono: true, fs: 15, fw: 700, fill: D.purple.dark });
  o += t(229, 126, '(Aᵀ)ᵢⱼ = Aⱼᵢ', { fs: 13.5, fw: 600, fill: D.purple.dark });

  o += t(330, 24, 'Aᵀ: shape (4, 3)', { mono: true, fs: 15, fw: 700, fill: C.ink });
  o += cells(AT, 270, 44, s, g, (r, c) => rowCols[c]);
  o += t(330, 226, 'become columns 0, 1, 2', { fs: 13.5, fw: 600, fill: C.muted });

  o += t(454, 84, 'Symmetric:', { fs: 14.5, fw: 700, fill: D.blue.dark });
  o += t(454, 106, 'S = Sᵀ', { mono: true, fs: 15, fw: 700, fill: C.ink });
  o += t(454, 128, 'Sᵢⱼ = Sⱼᵢ', { fs: 14, fill: C.muted });
  return svg(W, H, o);
}

// ── Hadamard elementwise product (A ⊙ B) vs Matrix product (A B) ──────
function hadamard() {
  const W = 540, H = 268; let o = '';
  const s = 36, g = 4;
  // Top: Hadamard gating A ⊙ B (same shapes -> elementwise scale/mask)
  o += t(16, 20, '(a) Hadamard Product A ⊙ B (A * B): element-by-element gating', { fs: 15.5, fw: 700, fill: D.blue.dark, anchor: 'start' });
  const A = [[2, 4, 6], [8, 10, 12]];
  const G = [['1', '0', '.5'], ['0', '1', '.5']];
  const R = [[2, 0, 3], [0, 10, 6]];
  const hl = (r, c) => (r === 0 && c === 2);
  o += cells(A, 20, 42, s, g, (r, c) => hl(r, c) ? [D.gold.tint, D.gold.base, false, C.ink] : [C.lblue, C.blue, false, C.ink]);
  o += t(148, 80, '⊙', { fs: 24, fw: 700, fill: C.ink });
  o += cells(G, 168, 42, s, g, (r, c) => hl(r, c) ? [D.gold.tint, D.gold.base, false, C.ink] : [C.lamber, C.amber, false, C.ink], 13.5);
  o += t(296, 80, '=', { fs: 24, fw: 700, fill: C.ink });
  o += cells(R, 316, 42, s, g, (r, c) => hl(r, c) ? [D.gold.tint, D.gold.base, false, C.ink] : [C.lgreen, C.green, false, C.ink]);
  o += t(482, 70, 'shape (m, n)', { mono: true, fs: 13.5, fw: 700, fill: D.green.dark });
  o += t(482, 92, '6 · 0.5 = 3', { mono: true, fs: 13.5, fw: 700, fill: D.gold.dark });

  o += line(16, 136, 524, 136, TOKENS.neutrals.hairline, 1.5);

  // Bottom: Matrix product A @ B (row-by-column contraction)
  o += t(16, 160, '(b) Matrix Product A B (A @ B): row i contracts with column j', { fs: 15.5, fw: 700, fill: D.purple.dark, anchor: 'start' });
  o += rect(20, 182, 116, 62, C.lblue, C.blue);
  o += rect(20, 196, 116, 16, D.gold.tint, D.gold.base);
  o += t(78, 256, '(m, k)', { mono: true, fs: 13.5, fill: C.ink });

  o += t(152, 213, '·', { fs: 28, fw: 700, fill: C.ink });

  o += rect(168, 176, 76, 74, TOKENS.neutrals.panel, C.gray);
  o += rect(202, 176, 16, 74, D.gold.tint, D.gold.base);
  o += t(206, 260, '(k, n)', { mono: true, fs: 13.5, fill: C.ink });

  o += t(262, 213, '=', { fs: 24, fw: 700, fill: C.ink });

  o += rect(280, 182, 76, 62, C.lgreen, C.green);
  o += rect(314, 196, 16, 16, D.gold.base, D.gold.dark);
  o += t(318, 256, '(m, n)', { mono: true, fs: 13.5, fill: C.ink });

  o += t(448, 204, 'inner dim k sums out:', { fs: 13.5, fw: 600, fill: C.ink });
  o += t(448, 226, '(A B)ᵢⱼ = ∑ₖ aᵢₖ bₖⱼ', { mono: true, fs: 13.5, fw: 700, fill: D.purple.dark });
  return svg(W, H, o);
}

// ── arithmetic intensity: FLOPs per element moved from GPU HBM ────────
function arithmeticIntensity() {
  const W = 540, H = 264; let o = '';
  o += t(16, 22, 'Arithmetic Intensity for n = 4,096 (FLOPs per element moved)', { fs: 15.5, fw: 700, fill: C.ink, anchor: 'start' });

  const rows = [
    { op: 'Hadamard A ⊙ B', mem: '3n² els', flops: 'n² FLOPs', ratio: '0.33 FLOP/el', w: 18, fill: D.red.tint, st: D.red.base, dk: D.red.dark, tag: 'HBM bandwidth bound' },
    { op: 'MatVec A x',     mem: 'n² els',  flops: '2n² FLOPs', ratio: '2.0 FLOP/el',  w: 44, fill: C.lamber, st: C.amber, dk: D.orange.dark, tag: 'memory-bound (batch 1)' },
    { op: 'MatMul A B',     mem: '3n² els', flops: '2n³ FLOPs', ratio: '2,730 FLOP/el', w: 230, fill: C.lgreen, st: C.green, dk: D.green.dark, tag: 'Tensor Core compute bound!' },
  ];

  rows.forEach((r, i) => {
    const y = 52 + i * 68;
    o += t(14, y + 12, r.op, { mono: true, fs: 14.5, fw: 700, fill: C.ink, anchor: 'start' });
    o += t(14, y + 32, `${r.flops} / ${r.mem}`, { mono: true, fs: 12.5, fill: C.muted, anchor: 'start' });
    o += rect(170, y, r.w, 28, r.fill, r.st);
    o += t(170 + r.w + 8, y + 14, r.ratio, { mono: true, fs: 14, fw: 700, fill: r.dk, anchor: 'start' });
    o += t(170, y + 42, r.tag, { fs: 13, fw: 600, fill: r.dk, anchor: 'start' });
  });
  return svg(W, H, o);
}

// ── spectral dynamics: lambda^k across k = 1..50 layers ───────────────
function spectralDepth() {
  // lambda^k for lambda = 1.1, 1, 0.9 on a log axis: 1.1^50 = 117.4 and
  // 0.9^50 = 0.0052 sit five decades apart, which a linear axis cannot show.
  let o = '';
  const x0 = 72, x1 = 420, y0 = 224, yTop = 40;
  const X = (k) => x0 + (k / 50) * (x1 - x0);
  const Y = (v) => y0 - ((Math.log10(v) + 3) / 6) * (y0 - yTop);
  o += line(x0, y0, x1, y0, C.ink, 1.8);
  o += line(x0, y0, x0, yTop, C.ink, 1.8);
  [0, 10, 20, 30, 40, 50].forEach(k => {
    o += line(X(k), y0, X(k), y0 + 5, C.ink, 1.5);
    o += t(X(k), y0 + 18, `${k}`, { mono: true, fs: 14, fill: C.ink });
  });
  o += t((x0 + x1) / 2, y0 + 42, 'depth k', { fs: 15, fw: 600, fill: C.ink });
  ['0.001', '0.01', '0.1', '1', '10', '100', '1000'].forEach((lab, i) => {
    const y = Y(Math.pow(10, i - 3));
    o += line(x0 - 5, y, x0, y, C.ink, 1.5);
    o += t(x0 - 9, y, lab, { mono: true, fs: 14, fill: C.ink, anchor: 'end' });
  });
  o += t(x0, yTop - 18, 'λᵏ', { fs: 16, fw: 600, fill: C.ink });
  const curve = (lam, color) => {
    let d = '';
    for (let k = 0; k <= 50; k++) d += (k ? 'L' : 'M') + X(k).toFixed(1) + ' ' + Y(Math.pow(lam, k)).toFixed(1);
    return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.8"/>`;
  };
  o += curve(1.1, D.red.base) + line(X(0), Y(1), X(50), Y(1), C.green, 2.2, true) + curve(0.9, C.amber);
  o += t(x1 + 10, Y(Math.pow(1.1, 50)), 'λ = 1.1', { fs: 15, fw: 700, fill: D.red.dark, anchor: 'start' });
  o += t(x1 + 10, Y(1), 'λ = 1', { fs: 15, fw: 700, fill: D.green.dark, anchor: 'start' });
  o += t(x1 + 10, Y(Math.pow(0.9, 50)), 'λ = 0.9', { fs: 15, fw: 700, fill: D.orange.dark, anchor: 'start' });
  return svg(x1 + 84, y0 + 56, o);
}

// ── outliers (l2^2 vs l1 curve) and sparsity (l2 circle vs l1 diamond) ─
function normOutliersSparsity() {
  const W = 540, H = 264; let o = '';
  // Panel (a): Outlier sensitivity r^2 vs |r|
  o += t(135, 20, '(a) Outlier Penalty: ℓ₂² vs. ℓ₁', { fs: 15, fw: 700, fill: C.ink });
  const ox = 135, oy = 216, sx = 46, sy = 38;
  o += line(32, oy, 238, oy, C.ink, 1.6);
  o += line(ox, oy + 10, ox, 40, C.ink, 1.6);
  o += t(238, oy + 16, 'r', { fs: 14, fw: 700, fill: C.ink });
  // Parabola r^2
  let d2 = '';
  for (let i = -40; i <= 40; i++) {
    const r = (i / 40) * 2.1;
    const x = ox + r * sx, y = oy - (r * r) * sy;
    d2 += (i === -40 ? 'M' : 'L') + x.toFixed(1) + ' ' + y.toFixed(1);
  }
  o += `<path d="${d2}" fill="none" stroke="${C.blue}" stroke-width="2.6"/>`;
  // V-shape |r|
  const xL = ox - 2.1 * sx, xR = ox + 2.1 * sx, yV = oy - 2.1 * sy;
  o += `<path d="M${xL.toFixed(1)} ${yV.toFixed(1)} L${ox} ${oy} L${xR.toFixed(1)} ${yV.toFixed(1)}" fill="none" stroke="${C.amber}" stroke-width="2.6"/>`;
  o += t(ox + 8, 56, 'ℓ₂² = r²', { fs: 13, fw: 700, fill: D.blue.dark, anchor: 'start' });
  o += t(ox + 8, 73, '(slope 2r)', { fs: 12, fw: 600, fill: D.blue.dark, anchor: 'start' });
  o += t(ox + 102, 176, 'ℓ₁ = |r|', { fs: 13, fw: 700, fill: D.orange.dark, anchor: 'end' });
  o += t(ox + 104, 193, '(slope ±1)', { fs: 12, fw: 600, fill: D.orange.dark, anchor: 'end' });

  // Panel (b): Sparsity geometry (l2 disk vs l1 diamond touching loss contour at corner w1 = 0)
  o += t(405, 20, '(b) Sparsity: ℓ₁ Corner w₁ = 0', { fs: 15, fw: 700, fill: C.ink });
  const cx = 396, cy = 154, R = 60;
  o += line(cx - 88, cy, cx + 104, cy, C.ink, 1.6);
  o += line(cx, cy + 80, cx, cy - 102, C.ink, 1.6);
  o += t(cx + 106, cy + 14, 'w₁', { fs: 13.5, fw: 700, fill: C.ink });
  o += t(cx - 16, cy - 96, 'w₂', { fs: 13.5, fw: 700, fill: C.ink });
  // l2 circle (dashed blue)
  o += `<circle cx="${cx}" cy="${cy}" r="${R}" fill="none" stroke="${C.blue}" stroke-width="2" stroke-dasharray="5 4"/>`;
  // l1 diamond (amber tint)
  o += `<path d="M${cx} ${cy - R} L${cx + R} ${cy} L${cx} ${cy + R} L${cx - R} ${cy} Z" fill="${C.lamber}" fill-opacity="0.7" stroke="${C.amber}" stroke-width="2.4"/>`;
  // Loss contours touching top corner (cx, cy - R)
  o += `<ellipse cx="${cx + 34}" cy="${cy - R - 20}" rx="42" ry="25" transform="rotate(-18 ${cx + 34} ${cy - R - 20})" fill="none" stroke="${C.gray}" stroke-width="1.6"/>`;
  o += `<ellipse cx="${cx + 34}" cy="${cy - R - 20}" rx="21" ry="12.5" transform="rotate(-18 ${cx + 34} ${cy - R - 20})" fill="none" stroke="${C.gray}" stroke-width="1.6"/>`;
  o += `<circle cx="${cx}" cy="${cy - R}" r="6" fill="${C.green}"/>`;
  o += t(cx - 10, cy - R - 10, 'w₁ = 0!', { fs: 14, fw: 700, fill: D.green.dark, anchor: 'end' });
  o += t(cx + 56, cy + 56, '‖w‖₁ ≤ 1', { fs: 13.5, fw: 700, fill: D.orange.dark, anchor: 'start' });
  o += t(cx - 54, cy + 56, '‖w‖₂ ≤ 1', { fs: 13.5, fw: 700, fill: D.blue.dark, anchor: 'end' });
  return svg(W, H, o);
}

export const diagrams = {
  'linear-algebra-reduce-axes': reduceAxes,
  'linear-algebra-matvec':      matvec,
  'linear-algebra-matmul':      matmul,
  'linear-algebra-transpose':              transpose,
  'linear-algebra-hadamard':               hadamard,
  'linear-algebra-arithmetic-intensity':   arithmeticIntensity,
  'linear-algebra-spectral-depth':         spectralDepth,
  'linear-algebra-norm-outliers-sparsity': normOutliersSparsity,
};
