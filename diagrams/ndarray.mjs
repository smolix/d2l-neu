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

// ── Python object pointers vs contiguous tensor buffer ────────────────
function pointersVsContiguous() {
  const W = 540, H = 324; let o = '';
  // Top panel: Python list of boxed floats
  o += name(20, 20, '(a) Python list: 8-byte pointers chasing scattered 24-byte heap boxes', D.red.dark, { anchor: 'start', fs: 15 });
  const ps = 44, pg = 6, px = 24, py = 36;
  for (let i = 0; i < 4; i++) {
    const x = px + i * (ps + pg);
    o += rect(x, py, ps, 30, D.red.tint, D.red.base);
    o += code(x + ps / 2, py + 15, `p${i}`, D.red.dark, { fs: 13.5 });
  }
  o += tx(240, 51, 'cache misses + 24 B/float + GIL loop', { fs: 13.5, fw: 600, fill: D.red.dark, anchor: 'start' });
  // Place scattered heap boxes below p0..p3 so every arrow runs strictly in the y=68..104 gap
  const boxes = [
    [20,  106, '1.0', px + 0 * (ps + pg) + ps / 2],
    [152, 114, '2.0', px + 2 * (ps + pg) + ps / 2],
    [284, 104, '3.0', px + 1 * (ps + pg) + ps / 2],
    [412, 112, '4.0', px + 3 * (ps + pg) + ps / 2],
  ];
  boxes.forEach(([bx, by, val, sx]) => {
    o += arrow(sx, py + 31, bx + 54, by - 2, D.red.base, true);
    o += rect(bx, by, 108, 32, '#FFFFFF', C.gray);
    o += code(bx + 54, by + 16, `PyFloat(${val})`, C.ink, { fs: 13 });
  });

  o += hline(20, 520, 160);

  // Bottom panel: Contiguous tensor buffer feeding SIMD / GPU lanes
  o += name(20, 184, '(b) Tensor buffer: contiguous 4-byte float32s → SIMD / GPU', D.green.dark, { anchor: 'start', fs: 15.5 });
  const cs = 54, cg = 3, cx0 = 24, cy0 = 204;
  const vals = ['1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0'];
  for (let i = 0; i < vals.length; i++) {
    const x = cx0 + i * (cs + cg);
    o += rect(x, cy0, cs, 38, C.lblue, C.blue);
    o += code(x + cs / 2, cy0 + 19, vals[i], C.ink, { fs: 14 });
    o += arrow(x + cs / 2, cy0 + 40, x + cs / 2, cy0 + 66, C.green);
  }
  const kw = vals.length * cs + (vals.length - 1) * cg;
  o += rect(cx0, cy0 + 68, kw, 34, C.lgreen, C.green);
  o += tx(cx0 + kw / 2, cy0 + 85, 'Parallel GPU / SIMD Vector Lanes (coalesced burst read)', { fs: 14.5, fw: 700, fill: D.green.dark });
  return svg(W, H, o);
}

// ── Metadata header + 1D storage buffer (O(1) view vs O(n) copy) ──────
function headerBuffer() {
  const W = 540, H = 312; let o = '';
  // Two metadata headers sharing one 1D storage buffer
  o += rect(20, 14, 228, 78, C.lblue, C.blue);
  o += name(134, 34, 'Header X (original)', D.blue.dark, { fs: 15.5 });
  o += code(134, 56, 'shape:(3,4) strides:(4,1)', C.ink, { fs: 13 });
  o += code(134, 76, 'offset:0  dtype:float32', C.muted, { fs: 12.5 });

  o += rect(292, 14, 228, 78, D.purple.tint, C.purple);
  o += name(406, 34, 'View Xᵀ — O(1) zero-copy', D.purple.dark, { fs: 15.5 });
  o += code(406, 56, 'shape:(4,3) strides:(1,4)', C.ink, { fs: 13 });
  o += code(406, 76, 'offset:0  (same pointer!)', D.purple.dark, { fs: 12.5 });

  // Arrows converging on the single 1D storage buffer
  const by = 148, s = 38, g = 3, bx0 = 24;
  o += arrow(134, 94, 174, by - 24, C.blue);
  o += arrow(406, 94, 366, by - 24, C.purple);

  o += code(270, by - 12, 'Shared 1D Physical Storage Buffer in RAM / HBM', C.ink, { fs: 13.5 });
  const row = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]];
  o += grid(row, bx0, by, s, g, { fs: 13.5, fill: () => BLUE });

  // Contrast: O(n) copy allocates a brand-new buffer
  const cy = 236;
  o += arrow(190, by + s + 6, 190, cy - 6, C.amber, true);
  o += code(202, (by + s + cy) / 2, 'copy → O(n) new buffer', D.orange.dark, { anchor: 'start', fs: 13.5 });
  o += grid(row, bx0, cy, s, g, { fs: 13.5, fill: () => AMBER });
  o += code(270, cy + s + 18, 'New 1D Storage Buffer (independent bytes)', D.orange.dark, { fs: 13.5 });
  return svg(W, H, o);
}

// ── Bit layout of float32, bfloat16, float16, int8/uint8 ──────────────
function dtypes() {
  const W = 540, H = 290; let o = '';
  const x0 = 104, bw = 9.5, h = 34;
  const drawBits = (y, label, mem, sBits, eBits, mBits, eNote, eCol) => {
    o += code(x0 - 10, y + h / 2, label, C.ink, { anchor: 'end', fs: 15, fw: 700 });
    const sw = sBits * bw, ew = eBits * bw, mw = mBits * bw;
    // sign
    o += rect(x0, y, sw, h, N.panel, C.gray);
    // exponent
    o += rect(x0 + sw, y, ew, h, eCol[0], eCol[1]);
    o += code(x0 + sw + ew / 2, y + h / 2, `${eBits}b`, eCol[2], { fs: 12.5 });
    // mantissa
    if (mw > 0) {
      o += rect(x0 + sw + ew, y, mw, h, C.lgreen, C.green);
      o += code(x0 + sw + ew + mw / 2, y + h / 2, `${mBits}b mant`, D.green.dark, { fs: 12.5 });
    }
    const totW = sw + ew + mw;
    o += tx(x0 + totW + 10, y + h / 2 - 7, mem, { mono: true, fs: 13.5, fw: 700, fill: C.ink, anchor: 'start' });
    o += tx(x0 + totW + 10, y + h / 2 + 10, eNote, { fs: 13, fw: 600, fill: eCol[2], anchor: 'start' });
  };

  o += tx(x0, 18, 'sign + exp + mantissa', { fs: 13, fw: 600, fill: C.muted, anchor: 'start' });
  drawBits(32,  'float32',  '4 B', 1, 8, 23, 'full FP32 range', [C.lblue, C.blue, D.blue.dark]);
  drawBits(94,  'bfloat16', '2 B', 1, 8, 7,  'same range as FP32!', [C.lblue, C.blue, D.blue.dark]);
  drawBits(156, 'float16',  '2 B', 1, 5, 10, 'narrow range (overflows)', [C.lamber, C.amber, D.orange.dark]);

  // uint8 row
  const y8 = 218;
  o += code(x0 - 10, y8 + h / 2, 'uint8', C.ink, { anchor: 'end', fs: 15, fw: 700 });
  o += rect(x0, y8, 8 * bw, h, D.purple.tint, C.purple);
  o += code(x0 + 4 * bw, y8 + h / 2, '8b int', D.purple.dark, { fs: 12.5 });
  o += tx(x0 + 8 * bw + 10, y8 + h / 2, '1 B  (0..255 raw pixels, 4× smaller)', { fs: 13.5, fw: 600, fill: D.purple.dark, anchor: 'start' });

  o += tx(270, 274, '1B params: float32 = 4 GB    vs.   bfloat16 / float16 = 2 GB', { mono: true, fs: 14, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// ── Automatic dimension inference (-1) and singleton axes (None) ──────
function singletonAxes() {
  const W = 540, H = 294; let o = '';
  // Top panel: 4D batch (256, 1, 28, 28) -> reshape(256, -1) -> (256, 784)
  o += name(16, 22, '(a) Flatten per-example features with -1', D.blue.dark, { anchor: 'start', fs: 16 });
  o += rect(28, 42, 76, 66, C.lblue, C.blue);
  o += code(66, 64, '28×28', D.blue.dark, { fs: 13.5 });
  o += code(66, 86, '× 256', C.ink, { fs: 13 });
  o += code(66, 124, '(256, 1, 28, 28)', C.ink, { fs: 13.5 });

  o += arrow(118, 75, 252, 75, C.purple);
  o += code(185, 56, 'reshape(256, -1)', D.purple.dark, { fs: 13, fw: 700 });
  o += tx(185, 94, 'infers 784 = 1·28·28', { fs: 13, fill: C.muted });

  o += rect(260, 42, 260, 66, C.lgreen, C.green);
  o += code(390, 68, 'X_flat: shape (256, 784)', D.green.dark, { fs: 15, fw: 700 });
  o += tx(390, 92, '256 rows × 784 pixels', { fs: 13.5, fill: C.ink });

  o += hline(16, 524, 146);

  // Bottom panel: x (3,) -> x[:, None] (3, 1) vs x[None, :] (1, 3)
  o += name(16, 170, '(b) Insert length-1 axes with None / expand_dims', D.orange.dark, { anchor: 'start', fs: 16 });
  const s = 34, g = 4;
  o += code(78, 198, 'x: (3,)', C.ink, { fs: 14 });
  o += grid([[0, 1, 2]], 24, 212, s, g, { fs: 14, fill: () => BLUE });

  // Arrow 1 to column vector (3, 1)
  o += arrow(142, 224, 218, 224, C.purple);
  o += code(180, 206, 'x[:, None]', D.purple.dark, { fs: 12.5 });
  o += grid([[0], [1], [2]], 228, 182, 28, 3, { fs: 13, fill: () => AMBER });
  o += code(242, 282, '(3, 1) col', D.orange.dark, { fs: 13.5 });

  // Arrow 2 from right of column to row matrix (1, 3)
  o += code(434, 198, 'x[None, :] → (1, 3) row', D.green.dark, { fs: 13.5 });
  o += grid([[0, 1, 2]], 386, 214, 30, 3, { fs: 13.5, fill: () => GREEN });

  return svg(W, H, o);
}

// ── Integer gather (cross-entropy) vs boolean mask / where ────────────
function gatherMask() {
  const W = 540, H = 310; let o = '';
  const s = 34, g = 4;
  const M = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]];
  // Panel (a): Integer gather picks [row i, label_i]
  o += name(16, 20, '(a) Integer gather: columns [3, 0, 2], one per row', D.orange.dark, { anchor: 'start', fs: 15.5 });
  const picks = [3, 0, 2];
  o += grid(M, 24, 38, s, g, {
    fs: 14,
    fill: (r, c) => picks[r] === c ? AMBER : BLUE,
  });
  o += arrow(186, 92, 254, 92, C.amber);
  o += code(220, 74, 'gather', D.orange.dark, { fs: 13.5 });
  o += grid([[3, 4, 10]], 264, 75, s, g, { fs: 14, fill: () => AMBER });
  o += code(318, 128, 'shape (3,) — 1 logit per example', D.orange.dark, { fs: 13.5 });

  o += hline(16, 524, 158);

  // Panel (b): Boolean mask X > 5: X[mask] (1D) vs where(mask, X, 0) (2D)
  o += name(16, 182, '(b) Boolean mask (X > 5): 1D filter vs. 2D where', D.green.dark, { anchor: 'start', fs: 15.5 });
  o += code(116, 206, 'mask X > 5 → 1D shape (6,)', D.purple.dark, { fs: 13.5 });
  o += grid([[6, 7, 8, 9, 10, 11]], 20, 222, 30, 3, { fs: 13, fill: () => GREEN });
  o += tx(116, 272, 'drops 2D grid coordinates', { fs: 13, fill: C.muted });

  const W2 = [[0, 0, 0, 0], [0, 0, 6, 7], [8, 9, 10, 11]];
  o += code(406, 200, 'where(X > 5, X, 0)', D.green.dark, { fs: 13.5 });
  o += grid(W2, 344, 212, 28, 3, {
    fs: 12.5,
    fill: (r, c, v) => v > 0 ? GREEN : [N.panel, N.hairline, false, C.muted],
  });
  return svg(W, H, o);
}

// ── The silent (B, 1) - (B,) -> (B, B) outer-difference bug ──────────
function broadcastBug() {
  const W = 540, H = 278; let o = '';
  const s = 28, g = 3;
  // Left panel: Silent bug (3, 1) - (3,) -> (3, 3) outer matrix!
  o += name(142, 20, 'Silent Bug: (3, 1) − (3,) → (3, 3)!', D.red.dark, { fs: 15.5 });
  o += code(30, 46, 'ŷ:(3,1)', D.blue.dark, { fs: 12.5 });
  o += grid([[10], [20], [30]], 16, 60, s, g, { fs: 12.5, fill: () => BLUE });
  o += tx(54, 104, '−', { fs: 22, fw: 700, fill: C.ink });
  o += grid([[9, 19, 29]], 66, 90, s, g, { fs: 12.5, fill: () => AMBER });
  o += code(111, 134, 'y:(3,)→(1,3)', D.orange.dark, { fs: 12 });

  o += tx(166, 104, '=', { fs: 20, fw: 700, fill: D.red.dark });
  const outer = [[1, -9, -19], [11, 1, -9], [21, 11, 1]];
  o += code(222, 46, '3×3 outer diffs!', D.red.dark, { fs: 12 });
  o += grid(outer, 180, 60, s, g, {
    fs: 12,
    fill: (r, c) => r === c ? [D.red.tint, D.red.base, false, D.red.dark] : [D.red.tint, D.red.base, true, D.red.dark],
  });

  // Vertical divider
  o += `<line x1="286" y1="16" x2="286" y2="182" stroke="${N.hairline}" stroke-width="1.5"/>`;

  // Right panel: Fix (3, 1) - (3, 1) -> (3, 1) aligned column
  o += name(414, 20, 'Fix: reshape y to (3, 1)', D.green.dark, { fs: 15.5 });
  o += code(324, 46, 'ŷ:(3,1)', D.blue.dark, { fs: 12.5 });
  o += grid([[10], [20], [30]], 308, 60, 32, 4, { fs: 13, fill: () => BLUE });
  o += tx(358, 112, '−', { fs: 24, fw: 700, fill: C.ink });
  o += code(396, 46, 'y:(3,1)', D.orange.dark, { fs: 12.5 });
  o += grid([[9], [19], [29]], 380, 60, 32, 4, { fs: 13, fill: () => AMBER });
  o += tx(430, 112, '=', { fs: 22, fw: 700, fill: C.ink });
  o += code(472, 46, 'err:(3,1)', D.green.dark, { fs: 12.5 });
  o += grid([[1], [1], [1]], 456, 60, 32, 4, { fs: 13, fill: () => GREEN });

  o += tx(270, 206, 'Rule 1 right-aligns (B,) as (1, B), stretching against (B, 1) into B×B!', { fs: 14, fw: 700, fill: D.red.dark });
  o += tx(270, 232, 'Always assert y_hat.shape == y.shape before computing a loss.', { fs: 14, fw: 600, fill: D.green.dark });
  return svg(W, H, o);
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
  'ndarray-pointers-vs-contiguous': pointersVsContiguous,
  'ndarray-header-buffer':          headerBuffer,
  'ndarray-dtypes':                 dtypes,
  'ndarray-singleton-axes':         singletonAxes,
  'ndarray-gather-mask':            gatherMask,
  'ndarray-broadcast-bug':          broadcastBug,
};
