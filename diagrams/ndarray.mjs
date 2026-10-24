// Diagrams for chapter_preliminaries/ndarray (Data Manipulation).
//
// One function per diagram. Each returns a complete <svg> string via the
// engine's `svg()` wrapper. Export them keyed by STABLE diagram id using
// the `<chapter-file>-<concept>` convention — the id becomes the SVG
// filename (img/auto/<id>.svg) and the slide/figure reference. Never
// rename an id once authored (same rule as code-cell ids).
//
// Sizing rule (figure-style guide §5–6): every canvas is at most 720 units
// wide, so it displays at design size in the book column; labels are
// 15–17 px, cell values 14–15 px, nothing below 14.5 px.  Labels name
// things (code, shapes); sentences belong in the caption or slide text.
// Structural text is INK; accent text uses the accent's `dark` shade.

import { C, TOKENS, grid, tx, arrow, chip, svg } from './engine.mjs';

const D = TOKENS.accents;                 // tint / base / dark trios
const N = TOKENS.neutrals;
const BLUE = [C.lblue, C.blue, false, C.ink];
const GREEN = [C.lgreen, C.green, false, C.ink];
const AMBER = [C.lamber, C.amber, false, C.ink];
const GHOST = (stroke) => ['#FFFFFF', stroke, true, C.muted];

// a monospace label (code or shape), 15 px
const code = (x, y, t, fill = C.ink, o = {}) =>
  tx(x, y, t, { mono: true, fs: o.fs || 15, fw: o.fw || 600, fill, anchor: o.anchor || 'middle' });
// a sans label, 17 px
const name = (x, y, t, fill = C.ink, o = {}) =>
  tx(x, y, t, { fs: o.fs || 17, fw: o.fw || 700, fill, anchor: o.anchor || 'middle' });
// a horizontal dashed separator
const hline = (x1, x2, y) =>
  `<line x1="${x1}" y1="${y}" x2="${x2}" y2="${y}" stroke="${N.hairline}" stroke-width="1.5" stroke-dasharray="4 4"/>`;
// a rounded, tinted rectangle (a whole matrix drawn as one block)
const rect = (x, y, w, h, fill, stroke, dash = false) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${fill}" stroke="${stroke}" stroke-width="2"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
// a memory buffer: a box with a mono title and a sans subtitle, accent rail
// on the left; `faded` dims it (memory that no name refers to any more)
function buffer(x, y, title, sub, accent, faded = false, w = 210) {
  const h = 58, op = faded ? 0.55 : 1;
  return `<g opacity="${op}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="#FFFFFF" stroke="${accent}" stroke-width="2"/>`
    + `<rect x="${x}" y="${y}" width="6" height="${h}" rx="3" fill="${accent}"/>`
    + tx(x + w / 2 + 3, y + 20, title, { mono: true, fs: 15, fw: 700, fill: C.ink })
    + tx(x + w / 2 + 3, y + 40, sub, { fs: 15, fw: 500, fill: C.ink }) + `</g>`;
}

// ── tensors with 0, 1, 2, 3 axes and their shapes ────────────────────
// Used as fig_ndarray_shapes in the book.  Names follow the book's
// terminology: a tensor with k axes has order k.
function rank() {
  const s = 34, g = 4, base = 168; let o = '';
  const gw = 4 * s + 3 * g, mh = 3 * s + 2 * g, off = 12;
  const M = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]];
  const label = (cx, nm, shape) =>
    name(cx, base + 26, nm) + code(cx, base + 50, shape);
  let x = 24;                                   // scalar
  o += grid([['5']], x, base - s, s, g, { fs: 14 });
  o += label(x + s / 2, 'scalar', '()');
  x = 112;                                      // vector
  o += grid([[0, 1, 2, 3]], x, base - s, s, g, { fs: 14 });
  o += label(x + gw / 2, 'vector', '(4,)');
  x = 314;                                      // matrix
  o += grid(M, x, base - mh, s, g, { fs: 14 });
  o += label(x + gw / 2, 'matrix', '(3, 4)');
  x = 516;                                      // order-3 tensor: 2 stacked 3×4
  o += rect(x + off, base - mh - off, gw, mh, C.lblue, C.blue);   // the second 3×4 slice, behind
  o += grid(M, x, base - mh, s, g, { fs: 14 });
  o += label(x + (gw + off) / 2, 'order-3 tensor', '(2, 3, 4)');
  return svg(x + gw + off + 24, base + 66, o);
}

// ── reshape: 12-vector → 3×4, row-major fill (color-tinted thirds) ──
function reshape() {
  const s = 40, g = 4; let o = '';
  const tint = (r, c, v) => [[C.lblue, C.blue], [C.lgreen, C.green], [C.lamber, C.amber]][Math.floor(v / 4)].concat([false, C.ink]);
  const vw = 12 * s + 11 * g, W = vw + 40, vx = 20, vy = 36;
  o += code(vx + vw / 2, vy - 18, 'x: shape (12,)');
  o += grid([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], vx, vy, s, g, { fill: tint });
  const ay1 = vy + s + 10, ay2 = vy + s + 58;
  o += arrow(W / 2, ay1, W / 2, ay2, C.purple, false);
  o += code(W / 2 + 14, (ay1 + ay2) / 2, 'reshape(3, 4)', D.purple.dark, { anchor: 'start' });
  const mw = 4 * s + 3 * g, mh = 3 * s + 2 * g, mx = (W - mw) / 2, my = ay2 + 30;
  o += code(mx + mw / 2, my - 18, 'X: shape (3, 4)');
  o += grid([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], mx, my, s, g, { fill: tint });
  return svg(W, my + mh + 16, o);
}

// ── indexing (read): an integer index drops its axis, a slice keeps it ─
function indexRead() {
  const s = 40, g = 5, y = 44; let o = '';
  const d = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], gw = 4 * s + 3 * g, gh = 3 * s + 2 * g;
  let x = 24;
  o += code(x + gw / 2, y - 22, 'X[-1]', D.orange.dark, { fs: 17, fw: 700 });
  o += grid(d, x, y, s, g, { fill: (r) => r === 2 ? AMBER : BLUE });
  o += code(x + gw / 2, y + gh + 24, 'shape (4,)');
  x = 24 + gw + 72;
  o += code(x + gw / 2, y - 22, 'X[1:3]', D.green.dark, { fs: 17, fw: 700 });
  o += grid(d, x, y, s, g, { fill: (r) => (r === 1 || r === 2) ? GREEN : BLUE });
  o += code(x + gw / 2, y + gh + 24, 'shape (2, 4)');
  return svg(x + gw + 24, y + gh + 44, o);
}

// ── indexing (write): X[1, 2] = 17, then X[:2, :] = 12 ────────────────
function indexWrite(t1 = 'X[1, 2] = 17', t2 = 'X[:2, :] = 12') {
  const s = 40, g = 5, y = 44; let o = '';
  const gw = 4 * s + 3 * g, gh = 3 * s + 2 * g;
  let x = 24;
  o += code(x + gw / 2, y - 22, t1, D.orange.dark, { fs: 15, fw: 700 });
  o += grid([[0, 1, 2, 3], [4, 5, 17, 7], [8, 9, 10, 11]], x, y, s, g,
            { fill: (r, c) => (r === 1 && c === 2) ? AMBER : BLUE });
  const ay = y + gh / 2;
  o += arrow(x + gw + 24, ay, x + gw + 72, ay, C.purple, false);
  x = x + gw + 96;
  o += code(x + gw / 2, y - 22, t2, D.green.dark, { fs: 15, fw: 700 });
  o += grid([[12, 12, 12, 12], [12, 12, 12, 12], [8, 9, 10, 11]], x, y, s, g,
            { fill: (r) => r < 2 ? GREEN : BLUE });
  return svg(x + gw + 24, y + gh + 20, o);
}

// ── concatenation of two 3×4 matrices along axis 0 and along axis 1 ───
// Each matrix is one block (24 units per row/column) so the letters sit
// on a clean tint, not on cell borders.
function concat() {
  const u = 24, w = 4 * u, h = 3 * u, gap = 4; let o = '';
  const X = (x, y) => rect(x, y, w, h, C.lblue, C.blue) + name(x + w / 2, y + h / 2, 'X', D.blue.dark, { fs: 20 });
  const Y = (x, y) => rect(x, y, w, h, C.lamber, C.amber) + name(x + w / 2, y + h / 2, 'Y', D.orange.dark, { fs: 20 });
  const top = 42;
  // axis 0: X above Y → (6, 4)
  let x = 24;
  o += code(x + w / 2, top - 22, 'axis 0', D.purple.dark, { fs: 16, fw: 700 });
  o += X(x, top) + Y(x, top + h + gap);
  o += code(x + w / 2, top + 2 * h + gap + 24, '(6, 4)');
  // axis 1: X beside Y → (3, 8)
  x = 24 + w + 80;
  const cy = top + (h + gap) / 2;          // centered against the tall stack
  o += code(x + w + gap / 2, top - 22, 'axis 1', D.purple.dark, { fs: 16, fw: 700 });
  o += X(x, cy) + Y(x + w + gap, cy);
  o += code(x + w + gap / 2, cy + h + 24, '(3, 8)');
  return svg(x + 2 * w + gap + 24, top + 2 * h + gap + 44, o);
}

// ── broadcasting: a (3, 1) + b (1, 2) → (3, 2), ghost-stretched axes ──
// One row: a → stretched a, + , b → stretched b, = , result.  Stretched
// copies are dashed with muted values: they are virtual.
function broadcasting() {
  const s = 40, g = 6; let o = '';
  const col = 2 * s + g, row3 = 3 * s + 2 * g, top = 44, mid = top + row3 / 2;
  const gA = GHOST(C.blue), gB = GHOST(C.amber);
  const op = (x, t) => tx(x, mid, t, { fs: 30, fw: 700, fill: C.ink });
  // 17 px labels and values: the figure also sits in a slide column at ~0.9x
  const lab = (x, t) => code(x, top - 24, t, C.ink, { fs: 17 });
  let x = 40;                                    // a: (3, 1)
  o += lab(x + s / 2, 'a: (3, 1)');
  o += grid([[0], [1], [2]], x, top, s, g, { fs: 17, fill: () => BLUE });
  o += arrow(x + s + 8, mid, x + s + 44, mid, C.ink, false);
  x += s + 54;                                   // a stretched to (3, 2)
  o += lab(x + col / 2, '(3, 2)');
  o += grid([[0, 0], [1, 1], [2, 2]], x, top, s, g, { fs: 17, fill: (r, c) => c === 0 ? BLUE : gA });
  x += col + 30; o += op(x, '+'); x += 30;
  o += lab(x + col / 2, 'b: (1, 2)');   // b: (1, 2)
  o += grid([[0, 1]], x, mid - s / 2, s, g, { fs: 17, fill: () => AMBER });
  o += arrow(x + col + 8, mid, x + col + 44, mid, C.ink, false);
  x += col + 54;                                 // b stretched to (3, 2)
  o += lab(x + col / 2, '(3, 2)');
  o += grid([[0, 1], [0, 1], [0, 1]], x, top, s, g, { fs: 17, fill: (r) => r === 0 ? AMBER : gB });
  x += col + 30; o += op(x, '='); x += 30;
  o += lab(x + col / 2, 'a + b: (3, 2)');
  o += grid([[0, 1], [1, 2], [2, 3]], x, top, s, g, { fs: 17, fill: () => GREEN });
  return svg(x + col + 40, top + row3 + 16, o);
}

// ── memory: Y = Y + X allocates; an in-place write reuses memory ─────
// Two panels stacked vertically so the figure fits a slide column.  The
// lower panel depends on the library: slice assignment (PyTorch, MXNet),
// Variable.assign (TensorFlow), or a jit-compiled update that donates
// its input (JAX).  The upper panel is the same for all four.
function allocationPanel(o) {
  o += code(210, 22, 'Y = Y + X', D.red.dark, { fs: 17, fw: 700 });
  o += chip(56, 124, 'Y');
  o += buffer(170, 52, 'buffer 1', 'old Y', C.gray, true);
  o += buffer(170, 140, 'buffer 2', 'Y + X', C.green);
  o += arrow(80, 124, 168, 81, C.gray, true);
  o += arrow(80, 124, 168, 169, C.purple, false);
  o += hline(30, 400, 226);
  return o;
}

function inPlacePanel(title, nm, sub) {
  let o = code(210, 262, title, D.green.dark, { fs: 17, fw: 700 });
  o += chip(56, 330, nm);
  o += buffer(170, 301, `buffer of ${nm}`, sub, C.green);
  o += arrow(80, 330, 168, 330, C.purple, false);
  return o;
}

function savingMemory() {
  let o = allocationPanel('');
  o += inPlacePanel('Z[:] = X + Y', 'Z', 'overwritten: X + Y');
  return svg(410, 372, o);
}

function savingMemoryTf() {
  let o = allocationPanel('');
  o += inPlacePanel('Z.assign(X + Y)', 'Z', 'overwritten: X + Y');
  return svg(410, 372, o);
}

function savingMemoryJax() {
  // JAX has no in-place update: X += Y rebinds X to a new array, and the
  // old array's memory is freed once no name refers to it.
  let o = code(210, 22, 'X += Y', D.red.dark, { fs: 17, fw: 700 });
  o += chip(56, 124, 'X');
  o += buffer(170, 52, 'buffer 1', 'old X', C.gray, true);
  o += buffer(170, 140, 'buffer 2', 'X + Y', C.green);
  o += arrow(80, 124, 168, 81, C.gray, true);
  o += arrow(80, 124, 168, 169, C.purple, false);
  return svg(410, 212, o);
}

// ── PyTorch: X, A = X.numpy(), B = torch.from_numpy(A) share a buffer ─
// Portrait, so that it fills a slide column at a legible size: the three
// names stacked on the left, their arrows converging on one memory block.
function numpyShare() {
  const s = 34, g = 4, n = 5, bw = 196, bh = 58, gap = 22; let o = '';
  const heads = [
    ['X', 'torch.Tensor', C.blue],
    ['A = X.numpy()', 'numpy.ndarray', C.amber],
    ['B = from_numpy(A)', 'torch.Tensor', C.blue],
  ];
  const x0 = 16, y0 = 16, H = 3 * bh + 2 * gap, mh = n * s + (n - 1) * g;
  const mx = x0 + bw + 84, my = y0 + (H - mh) / 2, mid = my + mh / 2;
  heads.forEach(([t, sub, acc], i) => {
    const y = y0 + i * (bh + gap);
    o += buffer(x0, y, t, sub, acc, false, bw);
    o += arrow(x0 + bw + 8, y + bh / 2, mx - 8, mid + (i - 1) * (s + g), C.purple, false);
  });
  o += grid(Array.from({ length: n }, () => ['']), mx, my, s, g,
            { showVal: false, fill: () => [N.panel, N.faint, false] });
  o += name(mx + s / 2, my + mh + 24, 'memory', C.ink, { fs: 16, fw: 600 });
  return svg(mx + s + 24, my + mh + 42, o);
}

// Keyed by stable diagram id. The id is the SVG filename and the
// reference used from slide divs / figures.
export const diagrams = {
  'ndarray-rank-ladder':   rank,
  'ndarray-reshape':       reshape,
  'ndarray-index-read':    indexRead,
  'ndarray-index-write':   () => indexWrite(),
  'ndarray-index-write-jax': () => indexWrite('X.at[1, 2].set(17)', 'X_new.at[:2, :].set(12)'),
  'ndarray-index-write-tensorflow': () => indexWrite('X_var[1, 2].assign(17)', 'X_var[:2, :].assign(12)'),
  'ndarray-concat':        concat,
  'ndarray-broadcasting':  broadcasting,
  'ndarray-saving-memory': savingMemory,
  'ndarray-saving-memory-tensorflow': savingMemoryTf,
  'ndarray-saving-memory-jax': savingMemoryJax,
  'ndarray-numpy-share':   numpyShare,
};
