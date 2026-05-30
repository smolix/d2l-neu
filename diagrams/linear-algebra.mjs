// Diagrams for chapter_preliminaries/linear-algebra (§2.3).
//
// Mix of structural figures (transpose, axis reduction, matrix products)
// and geometric-intuition figures (a vector as an arrow, the dot product
// as an angle/projection, norms as lengths). Values match the executed
// notebook cells so the figures and the In/Out cards agree.
//
// Stable ids use the `linear-algebra-<concept>` prefix (abbreviated `la-`
// in slide refs is NOT used — full id is the filename). Never rename.

import { C, grid, tx, arrow, svg } from './engine.mjs';

// plain stroked line (no arrowhead)
const line = (x1, y1, x2, y2, color, w = 1.5, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
const dot = (x, y, r, fill) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}"/>`;

// ── a vector as an arrow in the plane (geometric intuition) ─────────
function vectorArrow() {
  const W = 360, H = 300; let o = '';
  const ox = 66, oy = 250, u = 52;          // origin, pixels per unit
  o += arrow(ox - 12, oy, W - 16, oy, C.gray);   // x-axis
  o += arrow(ox, oy + 12, ox, 34, C.gray);       // y-axis
  o += tx(W - 20, oy + 18, 'x₁', { fs: 12.5, fill: C.muted });
  o += tx(ox - 16, 44, 'x₂', { fs: 12.5, fill: C.muted });
  const vx = ox + 3 * u, vy = oy - 2 * u;        // v = (3, 2)
  o += line(vx, vy, vx, oy, C.gray, 1.2, true);  // component guides
  o += line(vx, vy, ox, vy, C.gray, 1.2, true);
  o += tx((ox + vx) / 2, oy + 18, '3', { fs: 13, fw: 700, fill: C.muted });
  o += tx(ox - 16, (oy + vy) / 2, '2', { fs: 13, fw: 700, fill: C.muted });
  o += arrow(ox, oy, vx, vy, C.blue);
  o += dot(ox, oy, 3.5, C.ink);
  o += tx(vx + 8, vy - 8, 'x = (3, 2)', { fs: 14, fw: 700, fill: C.blue, anchor: 'start' });
  o += tx(ox - 13, oy + 17, '0', { fs: 12, fill: C.muted });
  return svg(W, H, o);
}

// ── transpose: rows ↔ columns (3×2 → 2×3), one element tracked ──────
function transpose() {
  const s = 42, g = 5; let o = '';
  const A = [[0, 1], [2, 3], [4, 5]];      // 3×2
  const AT = [[0, 2, 4], [1, 3, 5]];       // 2×3
  const ax = 56, ay = 72;
  const Aw = 2 * s + g, Ah = 3 * s + 2 * g;
  const hot = (r, c, R, Cc) => (r === R && c === Cc);
  o += tx(ax + Aw / 2, ay - 18, 'A  (3×2)', { fs: 14, fw: 700, fill: C.ink });
  o += grid(A, ax, ay, s, g, { fill: (r, c) => hot(r, c, 2, 0) ? [C.lamber, C.amber, false, C.ink] : [C.lblue, C.blue, false, C.ink] });
  const arX = ax + Aw + 22;
  o += arrow(arX, ay + Ah / 2, arX + 58, ay + Ah / 2, C.purple);
  o += tx(arX + 29, ay + Ah / 2 - 13, 'transpose', { fs: 12, fill: C.purple });
  const tx0 = arX + 76, ty0 = ay + (Ah - (2 * s + g)) / 2;
  const Tw = 3 * s + 2 * g;
  o += tx(tx0 + Tw / 2, ty0 - 18, 'Aᵀ  (2×3)', { fs: 14, fw: 700, fill: C.ink });
  o += grid(AT, tx0, ty0, s, g, { fill: (r, c) => hot(r, c, 0, 2) ? [C.lamber, C.amber, false, C.ink] : [C.lblue, C.blue, false, C.ink] });
  o += tx((ax + tx0 + Tw) / 2, ay + Ah + 22, 'A[i, j] = Aᵀ[j, i]', { fs: 13, fw: 600, fill: C.muted });
  return svg(tx0 + Tw + 28, ay + Ah + 40, o);
}

// ── reduction along an axis: collapse rows (axis 0) / cols (axis 1) ──
function reduceAxes() {
  const s = 44, g = 5; let o = '';
  const A = [[0, 1, 2], [3, 4, 5]];        // 2×3
  const ax = 126, ay = 66;
  const Aw = 3 * s + 2 * g, Ah = 2 * s + g;
  o += grid(A, ax, ay, s, g, { fill: () => [C.lblue, C.blue, false, C.ink] });
  o += tx(ax - 16, ay + Ah / 2, 'A', { fs: 16, fw: 700, fill: C.muted, anchor: 'end' });
  // axis=0 → column sums (3,5,7), drawn below
  const by = ay + Ah + 42;
  for (let c = 0; c < 3; c++) { const cx = ax + c * (s + g) + s / 2; o += arrow(cx, ay + Ah + 4, cx, by - 4, C.green); }
  o += grid([[3, 5, 7]], ax, by, s, g, { fill: () => [C.lgreen, C.green, false, C.ink] });
  o += tx(ax + Aw / 2, by + s + 20, 'axis=0 → shape (3,)  ·  column sums', { fs: 12.5, fw: 600, fill: C.green });
  // axis=1 → row sums (3,12), drawn to the right
  const rx = ax + Aw + 46;
  for (let r = 0; r < 2; r++) { const cy = ay + r * (s + g) + s / 2; o += arrow(ax + Aw + 4, cy, rx - 4, cy, C.amber); }
  o += grid([[3], [12]], rx, ay, s, g, { fill: () => [C.lamber, C.amber, false, C.ink] });
  o += tx(rx + s + 12, ay + Ah / 2 - 9, 'axis=1 → (2,)', { fs: 12.5, fw: 600, fill: C.amber, anchor: 'start' });
  o += tx(rx + s + 12, ay + Ah / 2 + 10, 'row sums', { fs: 12, fill: C.muted, anchor: 'start' });
  return svg(rx + 150, by + s + 32, o);
}

// ── dot product as projection / angle (geometric) ───────────────────
function dotProduct() {
  const W = 430, H = 290; let o = '';
  const ox = 70, oy = 232, u = 44;
  o += arrow(ox - 12, oy, W - 18, oy, C.gray);
  o += arrow(ox, oy + 12, ox, 26, C.gray);
  const a = [4, 1], b = [2, 3];
  const ax = ox + a[0] * u, ay = oy - a[1] * u;
  const bx = ox + b[0] * u, by = oy - b[1] * u;
  // projection of a onto b
  const t = (a[0] * b[0] + a[1] * b[1]) / (b[0] * b[0] + b[1] * b[1]);
  const fx = ox + t * b[0] * u, fy = oy - t * b[1] * u;
  o += line(fx, fy, ax, ay, C.gray, 1.3, true);            // perpendicular
  o += line(ox, oy, fx, fy, C.green, 4);                   // projection on b
  o += arrow(ox, oy, bx, by, C.amber);
  o += arrow(ox, oy, ax, ay, C.blue);
  o += dot(ox, oy, 3.5, C.ink);
  o += tx(ax + 8, ay + 2, 'a', { fs: 15, fw: 700, fill: C.blue, anchor: 'start' });
  o += tx(bx - 4, by - 8, 'b', { fs: 15, fw: 700, fill: C.amber, anchor: 'middle' });
  o += tx(ox + 30, oy - 14, 'θ', { fs: 14, fw: 700, fill: C.muted });
  o += tx(W / 2, H - 40, 'a · b = Σᵢ aᵢ bᵢ', { fs: 15, fw: 700, fill: C.ink });
  o += tx(W / 2, H - 18, '= ‖a‖ ‖b‖ cos θ', { fs: 14, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── matrix–vector product: each output = one row · x ────────────────
function matvec() {
  const s = 42, g = 5; let o = '';
  const A = [[0, 1, 2], [3, 4, 5]];        // 2×3
  const ax = 60, ay = 78;
  const Aw = 3 * s + 2 * g, Ah = 2 * s + g;
  o += tx(ax + Aw / 2, ay - 16, 'A  (2×3)', { fs: 13, fw: 700, fill: C.muted });
  o += grid(A, ax, ay, s, g, { fill: (r) => r === 0 ? [C.lblue, C.blue, false, C.ink] : ['#EEF4FB', '#C2D9F1', false, C.ink] });
  let cx = ax + Aw + 16;
  o += tx(cx, ay + Ah / 2, '·', { fs: 26, fw: 700, fill: C.muted });
  const xx = cx + 16, xy = ay - (s + g) / 2;               // x: 3×1, vertically centered
  o += tx(xx + s / 2, xy - 16, 'x (3)', { fs: 13, fw: 700, fill: C.muted });
  o += grid([[0], [1], [2]], xx, xy, s, g, { fill: () => [C.lamber, C.amber, false, C.ink] });
  let ex = xx + s + 16;
  o += tx(ex, ay + Ah / 2, '=', { fs: 24, fw: 700, fill: C.muted });
  const yx = ex + 18;
  o += tx(yx + s / 2, ay - 16, 'y (2)', { fs: 13, fw: 700, fill: C.muted });
  o += grid([[5], [14]], yx, ay, s, g, { fill: (r) => r === 0 ? [C.lgreen, C.green, false, C.ink] : [C.lgray, C.gray, false, C.ink] });
  const Wm = yx + s + 30;
  const xBot = (ay - (s + g) / 2) + 3 * s + 2 * g;   // x is the tallest column (3 rows)
  const capY = xBot + 28;
  o += tx(Wm / 2, capY, 'each yᵢ = (row i of A) · x', { fs: 13, fw: 600, fill: C.muted });
  return svg(Wm, capY + 20, o);
}

// ── matrix–matrix product: Cᵢⱼ = (row i of A) · (col j of B) ────────
function matmul() {
  const s = 34, g = 4; let o = '';
  const A = [[0, 1, 2], [3, 4, 5]];                 // 2×3
  const B = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]; // 3×4
  const Cm = [[3, 3, 3, 3], [12, 12, 12, 12]];      // 2×4
  const Aw = 3 * (s + g) - g, Ah = 2 * (s + g) - g;
  const Bw = 4 * (s + g) - g, Bh = 3 * (s + g) - g;
  const ax = 40, ay = 70 + Bh + 24;                 // A bottom-left
  const bx = ax + Aw + 28, by = 36;                 // B top-right
  const cx = bx, cy = ay;                           // C aligns: cols under B, rows beside A
  // B (top)
  o += tx(bx + Bw / 2, by - 14, 'B  (3×4)', { fs: 12.5, fw: 700, fill: C.muted });
  o += grid(B, bx, by, s, g, { fill: (r, c) => c === 0 ? [C.lblue, C.blue, false, C.ink] : ['#F2F6FB', '#CBDDF0', false, '#6B7280'] });
  // A (left)
  o += tx(ax + Aw / 2, ay - 14, 'A  (2×3)', { fs: 12.5, fw: 700, fill: C.muted });
  o += grid(A, ax, ay, s, g, { fill: (r) => r === 0 ? [C.lblue, C.blue, false, C.ink] : ['#F2F6FB', '#CBDDF0', false, '#6B7280'] });
  // C (bottom-right)
  o += tx(cx + 4 * (s + g) - g - 14, cy + Ah + 16, 'C = AB  (2×4)', { fs: 12.5, fw: 700, fill: C.ink, anchor: 'end' });
  o += grid(Cm, cx, cy, s, g, { fill: (r, c) => (r === 0 && c === 0) ? [C.lamber, C.amber, false, C.ink] : [C.lgreen, C.green, false, C.ink] });
  // True subscripts via tspans (the unicode ⱼ glyph is missing in many
  // fonts, so it rendered un-subscripted); larger than before.
  const sub = `baseline-shift="sub" font-size="11"`;
  o += `<text x="${bx + Bw + 20}" y="${by + 6}" font-family="Source Sans 3, sans-serif" `
    + `font-size="15" font-weight="700" fill="#3A4049" text-anchor="start">`
    + `C<tspan ${sub}>ij</tspan> = row<tspan ${sub}>i</tspan> · col<tspan ${sub}>j</tspan></text>`;
  return svg(bx + Bw + 168, ay + Ah + 34, o);
}

// ── norms as geometric lengths: ℓ₂ (hypotenuse) vs ℓ₁ (path) ─────────
function norms() {
  const W = 350, H = 338; let o = '';
  const ox = 64, oy = 256, u = 46;
  o += arrow(ox - 12, oy, W - 16, oy, C.gray);
  o += arrow(ox, oy + 12, ox, 30, C.gray);
  const vx = ox + 3 * u, vy = oy - 4 * u;          // v = (3, 4)
  o += line(ox, oy, vx, oy, C.amber, 3);           // ℓ₁ path: 3 across …
  o += line(vx, oy, vx, vy, C.amber, 3);           // … then 4 up
  o += tx((ox + vx) / 2, oy + 17, '3', { fs: 13, fw: 700, fill: C.amber });
  o += tx(vx + 12, (oy + vy) / 2, '4', { fs: 13, fw: 700, fill: C.amber, anchor: 'start' });
  o += arrow(ox, oy, vx, vy, C.blue);              // ℓ₂: the vector itself
  o += tx((ox + vx) / 2 - 8, (oy + vy) / 2 - 6, '5', { fs: 14, fw: 700, fill: C.blue, anchor: 'end' });
  o += dot(ox, oy, 3.5, C.ink);
  o += tx(vx + 6, vy - 6, 'v = (3, 4)', { fs: 13, fw: 700, fill: C.ink, anchor: 'start' });
  o += tx(W / 2, H - 42, '‖v‖₂ = √(3² + 4²) = 5   (Euclidean)', { fs: 12.5, fw: 600, fill: C.blue });
  o += tx(W / 2, H - 20, '‖v‖₁ = 3 + 4 = 7   (Manhattan)', { fs: 12.5, fw: 600, fill: C.amber });
  return svg(W, H, o);
}

// ── symmetric matrix: A equals its mirror across the diagonal ───────
function symmetric() {
  const s = 48, g = 5; let o = '';
  const A = [[1, 2, 3], [2, 0, 4], [3, 4, 5]];   // A = Aᵀ (matches the cell)
  const ax = 64, ay = 54, W3 = 3 * s + 2 * g;
  const pal = { '2': [C.lblue, C.blue], '3': [C.lgreen, C.green], '4': [C.lamber, C.amber] };
  const fill = (r, c, v) => (r === c)
    ? [C.lgray, C.gray, false, C.ink]
    : [...pal[String(v)], false, C.ink];
  // dashed axis of symmetry (the main diagonal)
  o += line(ax - 6, ay - 6, ax + W3 + 6, ay + W3 + 6, C.gray, 1.4, true);
  o += grid(A, ax, ay, s, g, { fill });
  o += tx(ax + W3 / 2, ay + W3 + 26, 'A = Aᵀ  ·  mirror across the diagonal', { fs: 13, fw: 600, fill: C.muted });
  return svg(ax + W3 + 28, ay + W3 + 44, o);
}

// ── 4-D tensor = a train (sequence) of 3-D tensors ──────────────────
function tensor4d() {
  const s = 22, g = 3, off = 12; let o = '';
  const face = [['', '', ''], ['', '', '']];     // 2×3 face
  const fw = 3 * s + 2 * g, fh = 2 * s + g;
  const cube = (x, y) => {
    let c = '';
    c += grid(face, x + off, y - off, s, g, { showVal: false, fill: () => ['#EAF4FE', '#A9CCEF', false] });
    // edges from back to front corners
    c += line(x + off, y - off, x, y, C.blue, 1.2);
    c += line(x + off + fw, y - off, x + fw, y, C.blue, 1.2);
    c += line(x + off + fw, y - off + fh, x + fw, y + fh, C.blue, 1.2);
    c += grid(face, x, y, s, g, { showVal: false, fill: () => [C.lblue, C.blue, false] });
    return c;
  };
  const ys = 58, xs = [62, 232, 402];
  o += tx((xs[0] + xs[2] + fw) / 2, 30, 'a 4-D tensor = a sequence of 3-D tensors', { fs: 14, fw: 700, fill: C.ink });
  xs.forEach((x, i) => { o += cube(x, ys); o += tx(x + fw / 2, ys + fh + 24, `[${i}]`, { mono: true, fs: 13, fw: 700, fill: C.muted }); });
  o += tx(xs[2] + fw + 42, ys + fh / 2, '…', { fs: 26, fw: 700, fill: C.muted });
  o += tx((xs[0] + xs[2] + fw) / 2, ys + fh + 50, 'shape  (N, C, H, W)', { mono: true, fs: 13, fill: C.muted });
  return svg(xs[2] + fw + 92, ys + fh + 66, o);
}

export const diagrams = {
  'linear-algebra-symmetric':  symmetric,
  'linear-algebra-tensor4d':   tensor4d,
  'linear-algebra-vector':     vectorArrow,
  'linear-algebra-transpose':  transpose,
  'linear-algebra-reduce-axes': reduceAxes,
  'linear-algebra-dot':        dotProduct,
  'linear-algebra-matvec':     matvec,
  'linear-algebra-matmul':     matmul,
  'linear-algebra-norms':      norms,
};
