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
  const title = 'A.sum(axis=1, keepdims=True)';
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

export const diagrams = {
  'linear-algebra-reduce-axes': reduceAxes,
  'linear-algebra-matvec':      matvec,
  'linear-algebra-matmul':      matmul,
};
