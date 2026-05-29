// Diagrams for chapter_preliminaries/ndarray (§2.1 Data Manipulation).
//
// One function per diagram. Each returns a complete <svg> string via the
// engine's `svg()` wrapper. Export them keyed by STABLE diagram id using
// the `<chapter-file>-<concept>` convention — the id becomes the SVG
// filename (img/auto/<id>.svg) and the slide/figure reference. Never
// rename an id once authored (same rule as code-cell ids).
//
// These eight are the north-star set. To add a diagram for another
// chapter, copy this file to <file>.mjs, write the function(s), and add
// the module to registry.mjs.

import { C, grid, tx, arrow, block, chip, svg } from './engine.mjs';

// ── rank ladder: scalar → vector → matrix → 3-D tensor ──────────────
function rank() {
  const s = 46, g = 4, base = 212; let o = '';
  let x = 66; o += grid([['5']], x, base - s, s, g);
  o += tx(x + s / 2, base + 28, 'scalar', { fw: 700, fs: 16 });
  o += tx(x + s / 2, base + 49, 'shape ( )', { mono: true, fs: 12.5, fill: C.muted });
  x = 196; const vw = 4 * s + 3 * g; o += grid([[0, 1, 2, 3]], x, base - s, s, g);
  o += tx(x + vw / 2, base + 28, 'vector', { fw: 700, fs: 16 });
  o += tx(x + vw / 2, base + 49, '(4,)', { mono: true, fs: 12.5, fill: C.muted });
  x = 470; const mw = 4 * s + 3 * g, mh = 3 * s + 2 * g;
  o += grid([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], x, base - mh, s, g);
  o += tx(x + mw / 2, base + 28, 'matrix', { fw: 700, fs: 16 });
  o += tx(x + mw / 2, base + 49, '(3, 4)', { mono: true, fs: 12.5, fill: C.muted });
  x = 800; const off = 15;
  o += grid([['', '', '', ''], ['', '', '', ''], ['', '', '', '']], x + off, base - mh - off, s, g, { showVal: false, fill: () => ['#EAF4FE', '#90CAF9', false] });
  o += grid([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], x, base - mh, s, g);
  o += tx(x + mw / 2, base + 28, '3-D tensor', { fw: 700, fs: 16 });
  o += tx(x + mw / 2, base + 49, '(2, 3, 4)', { mono: true, fs: 12.5, fill: C.muted });
  return svg(1060, 272, o);
}

// ── reshape: 12-vector → 3×4, row-major fill (color-tinted thirds) ──
function reshape() {
  const s = 44, g = 4, W = 900; let o = '';
  const tint = (r, c, v) => { const k = Math.floor(v / 4); return [[C.lblue, C.blue], [C.lgreen, C.green], [C.lamber, C.amber]][k].concat([false, C.ink]); };
  const vw = 12 * s + 11 * g, vx = (W - vw) / 2, vy = 34;
  o += tx(vx - 14, vy + s / 2, 'x', { anchor: 'end', mono: true, fs: 18, fill: C.muted });
  o += grid([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], vx, vy, s, g, { fill: tint });
  o += arrow(W / 2, vy + s + 12, W / 2, vy + s + 58, C.purple, false);
  o += tx(W / 2 + 14, vy + s + 36, 'reshape(3, 4)', { anchor: 'start', mono: true, fs: 15, fill: C.purple });
  const mw = 4 * s + 3 * g, mx = (W - mw) / 2, my = vy + s + 74;
  o += tx(mx - 14, my + (3 * s + 2 * g) / 2, 'X', { anchor: 'end', mono: true, fs: 18, fill: C.muted });
  o += grid([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], mx, my, s, g, { fill: tint });
  return svg(W, my + 3 * s + 2 * g + 14, o);
}

// ── indexing (read): X[-1] and X[1:3] highlighted on a 3×4 grid ─────
function indexRead() {
  const s = 42, g = 5, W = 880, y = 88; let o = '';
  const d = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], gw = 4 * s + 3 * g, gh = 3 * s + 2 * g;
  let x = 80;
  o += tx(x + gw / 2, y - 24, 'X[-1]', { mono: true, fs: 18, fw: 700, fill: C.amber });
  o += grid(d, x, y, s, g, { fill: (r) => r === 2 ? [C.lamber, C.amber, false, C.ink] : [C.lblue, C.blue, false, C.ink] });
  o += tx(x + gw / 2, y + gh + 26, 'last row', { fs: 13.5, fill: C.muted });
  x = 500;
  o += tx(x + gw / 2, y - 24, 'X[1:3]', { mono: true, fs: 18, fw: 700, fill: C.green });
  o += grid(d, x, y, s, g, { fill: (r) => (r === 1 || r === 2) ? [C.lgreen, C.green, false, C.ink] : [C.lblue, C.blue, false, C.ink] });
  o += tx(x + gw / 2, y + gh + 26, 'rows 1–2  (3 excluded)', { fs: 13.5, fill: C.muted });
  return svg(W, y + gh + 44, o);
}

// ── indexing (write): X[1,2]=17 then X[:2,:]=12 ─────────────────────
function indexWrite() {
  const s = 42, g = 5, W = 880, y = 86; let o = '';
  const gw = 4 * s + 3 * g, gh = 3 * s + 2 * g;
  let x = 110;
  o += tx(x + gw / 2, y - 24, 'X[1, 2] = 17', { mono: true, fs: 15.5, fw: 700, fill: C.amber });
  o += grid([[0, 1, 2, 3], [4, 5, 17, 7], [8, 9, 10, 11]], x, y, s, g, { fill: (r, c) => (r === 1 && c === 2) ? [C.lamber, C.amber, false, C.ink] : [C.lblue, C.blue, false, C.ink] });
  const ay = y + gh / 2;
  o += arrow(x + gw + 16, ay, x + gw + 74, ay, C.purple, false);
  x = x + gw + 106;
  o += tx(x + gw / 2, y - 24, 'X[:2, :] = 12', { mono: true, fs: 15.5, fw: 700, fill: C.green });
  o += grid([[12, 12, 12, 12], [12, 12, 12, 12], [8, 9, 10, 11]], x, y, s, g, { fill: (r) => r < 2 ? [C.lgreen, C.green, false, C.ink] : [C.lblue, C.blue, false, C.ink] });
  return svg(W, y + gh + 30, o);
}

// ── concatenation: dim=0 stacks rows (6×4); dim=1 stacks cols (3×8) ─
function concat() {
  const s = 24, g = 3, W = 1000; let o = '';
  const m = [['', '', '', ''], ['', '', '', ''], ['', '', '', '']], gw = 4 * s + 3 * g, gh = 3 * s + 2 * g;
  const blue = () => [C.lblue, C.blue, false], amber = () => [C.lamber, C.amber, false];
  let ax = 130, ay = 58;
  o += tx(ax + gw / 2, ay - 22, 'dim = 0', { mono: true, fs: 15, fw: 700, fill: C.purple });
  o += grid(m, ax, ay, s, g, { showVal: false, fill: blue }); o += tx(ax + gw / 2, ay + gh / 2, 'X', { fw: 700, fs: 18 });
  let by = ay + gh + g;
  o += grid(m, ax, by, s, g, { showVal: false, fill: amber }); o += tx(ax + gw / 2, by + gh / 2, 'Y', { fw: 700, fs: 18 });
  o += tx(ax + gw / 2, by + gh + 26, '(6, 4)', { mono: true, fs: 13.5, fill: C.muted });
  let cx = 560, cy = 110;
  o += tx(cx + gw, cy - 22, 'dim = 1', { mono: true, fs: 15, fw: 700, fill: C.purple });
  o += grid(m, cx, cy, s, g, { showVal: false, fill: blue }); o += tx(cx + gw / 2, cy + gh / 2, 'X', { fw: 700, fs: 18 });
  let dx = cx + gw + g;
  o += grid(m, dx, cy, s, g, { showVal: false, fill: amber }); o += tx(dx + gw / 2, cy + gh / 2, 'Y', { fw: 700, fs: 18 });
  o += tx(cx + gw, cy + gh + 26, '(3, 8)', { mono: true, fs: 13.5, fill: C.muted });
  return svg(W, 300, o);
}

// ── broadcasting: a(3×1) + b(1×2) → (3×2), ghost-stretched axes ─────
function broadcasting() {
  const s = 40, g = 6, W = 700, H = 358; let o = '';
  const sB = [C.lblue, C.blue, false, C.ink], gB = ['#EBF5FE', '#90CAF9', true, '#90A4AE'];
  const sA = [C.lamber, C.amber, false, C.ink], gA = ['#FFF3E2', '#FFCC80', true, '#B0A48F'];
  const sG = [C.lgreen, C.green, false, C.ink];
  const col = 2 * s + g, row3 = 3 * s + 2 * g;
  let ax = 46, ay = 30;
  o += tx(ax + s / 2, ay - 13, 'a (3×1)', { fs: 13, fill: C.muted });
  o += grid([[0], [1], [2]], ax, ay, s, g, { fill: () => sB });
  let apx = 196;
  o += tx(apx + col / 2, ay - 13, '(3×2)', { fs: 13, fill: C.muted });
  o += grid([[0, 0], [1, 1], [2, 2]], apx, ay, s, g, { fill: (r, c) => c === 0 ? sB : gB });
  o += arrow(ax + s + 8, ay + row3 / 2, apx - 8, ay + row3 / 2, C.gray, false);
  let bx = 46, by = 242;
  o += tx(bx + col / 2, by - 13, 'b (1×2)', { fs: 13, fill: C.muted });
  o += grid([[0, 1]], bx, by, s, g, { fill: () => sA });
  let bpx = 196, bpy = 190;
  o += tx(bpx + col / 2, bpy - 13, '(3×2)', { fs: 13, fill: C.muted });
  o += grid([[0, 1], [0, 1], [0, 1]], bpx, bpy, s, g, { fill: (r) => r === 0 ? sA : gA });
  o += arrow(bx + col + 8, by + s / 2, bpx - 8, by + s / 2, C.gray, false);
  const midY = (ay + row3 + bpy) / 2;
  o += tx(apx + col + 34, midY, '+', { fs: 30, fw: 700, fill: C.muted });
  o += tx(apx + col + 96, midY, '=', { fs: 30, fw: 700, fill: C.muted });
  let rx = apx + col + 128, ry = midY - row3 / 2;
  o += tx(rx + col / 2, ry - 13, 'result (3×2)', { fs: 13, fill: C.green });
  o += grid([[0, 1], [1, 2], [2, 3]], rx, ry, s, g, { fill: () => sG });
  return svg(W, H, o);
}

// ── saving memory: new allocation vs in-place write ─────────────────
// NOTE framework variance: this is the PyTorch/TF/MXNet story. JAX arrays
// are immutable (no in-place); author a jax variant before reusing — see
// HANDOFF.md "Framework variants".
function savingMemory() {
  const W = 960, H = 300; let o = '';
  o += `<line x1="480" y1="46" x2="480" y2="268" stroke="#E0E0E0" stroke-width="1.5" stroke-dasharray="4 4"/>`;
  o += tx(232, 40, 'Y = Y + X', { mono: true, fs: 17, fw: 700, fill: '#C0341D' });
  o += chip(92, 150, 'Y');
  o += block(196, 78, 'addr #1', 'old buffer', true, C.gray);
  o += block(196, 170, 'addr #2', 'Y + X', false, C.green);
  o += arrow(118, 150, 196, 197, C.purple, false);
  o += arrow(118, 150, 196, 104, '#C9CDD2', true);
  o += tx(232, 262, 'id(Y) changed → False', { fs: 14, fw: 700, fill: '#C0341D' });
  o += tx(728, 40, 'Y[:] = X + Y', { mono: true, fs: 17, fw: 700, fill: '#2E7D32' });
  o += chip(584, 150, 'Y');
  o += block(700, 123, 'addr #1', 'overwritten', false, '#2E7D32');
  o += arrow(610, 150, 700, 150, C.purple, false);
  o += tx(770, 262, 'id(Y) same → True', { fs: 14, fw: 700, fill: '#2E7D32' });
  return svg(W, H, o);
}

// ── numpy round-trip: two handles, one shared buffer ────────────────
function numpyShare() {
  const W = 820, H = 246, s = 40, g = 4; let o = '';
  o += block(118, 38, 'torch.Tensor', 'B = from_numpy(A)', false, C.blue);
  o += block(500, 38, 'numpy.ndarray', 'A = X.numpy()', false, C.amber);
  const vals = ['12.', '12.', '12.', '8.', '9.', '10.', '11.', '…'];
  const bw = vals.length * s + (vals.length - 1) * g, bx = (W - bw) / 2, by = 150;
  o += grid([vals], bx, by, s, g, { fill: () => ['#ECEFF1', '#90A4AE', false, '#37474F'], fs: 12.5 });
  o += tx(W / 2, by + s + 24, 'one shared memory buffer', { fs: 14, fw: 700, fill: C.muted });
  o += arrow(207, 94, bx + 58, by - 8, C.blue, false);
  o += arrow(589, 94, bx + bw - 58, by - 8, C.amber, false);
  return svg(W, H, o);
}

// Keyed by stable diagram id. The id is the SVG filename and the
// reference used from slide divs / figures.
export const diagrams = {
  'ndarray-rank-ladder':   rank,
  'ndarray-reshape':       reshape,
  'ndarray-index-read':    indexRead,
  'ndarray-index-write':   indexWrite,
  'ndarray-concat':        concat,
  'ndarray-broadcasting':  broadcasting,
  'ndarray-saving-memory': savingMemory,
  'ndarray-numpy-share':   numpyShare,
};
