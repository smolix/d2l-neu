// Diagrams for chapter_preliminaries/calculus (§2.4).
//
// This chapter is code-light and intuition-heavy, so the figures carry
// most of the geometric load: the limit procedure (Archimedes), the
// derivative as the slope of a tangent, partial derivatives as slopes of
// 1-D slices, the gradient as a field normal to the contours, descent
// following −∇L, and the chain rule as a computational graph traversed
// forward (evaluate) and backward (differentiate → backprop).
//
// Stable ids use the `calculus-<concept>` prefix. Never rename.

import { C, tx, arrow, chip, svg } from './engine.mjs';

const line = (x1, y1, x2, y2, color, w = 1.5, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
const dot = (x, y, r, fill) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}"/>`;
const ellipse = (cx, cy, rx, ry, st, w = 1.4, op = 1) =>
  `<ellipse cx="${cx}" cy="${cy}" rx="${rx}" ry="${ry}" fill="none" stroke="${st}" stroke-width="${w}" opacity="${op}"/>`;
const poly = (pts, fill, st, w = 2, op = 1) =>
  `<polygon points="${pts.map(p => p.map(n => n.toFixed(1)).join(',')).join(' ')}" fill="${fill}" fill-opacity="${op}" stroke="${st}" stroke-width="${w}"/>`;
const plines = (pts, st, w = 2, dash = false) =>
  `<polyline points="${pts.map(p => p.map(n => n.toFixed(1)).join(',')).join(' ')}" fill="none" stroke="${st}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

// ── the limit procedure: inscribe polygons with more vertices ───────
function circleLimit() {
  const W = 560, H = 300; let o = '';
  const R = 78, cy = 138;
  const cxs = [110, 290, 470], ns = [4, 8, 16];
  const verts = (cx, r, n) => Array.from({ length: n }, (_, k) => {
    const a = -Math.PI / 2 + k * 2 * Math.PI / n;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  });
  o += tx(W / 2, 32, 'Inscribe polygons with more and more vertices', { fs: 15, fw: 700, fill: C.ink });
  ns.forEach((n, i) => {
    const cx = cxs[i];
    o += `<circle cx="${cx}" cy="${cy}" r="${R}" fill="none" stroke="${C.gray}" stroke-width="1.6"/>`;
    o += poly(verts(cx, R, n), C.lblue, C.blue, 2, 0.7);
    if (i === 0) {  // one triangular wedge highlighted on the n=4 figure
      const v = verts(cx, R, n);
      o += poly([[cx, cy], v[0], v[1]], C.lamber, C.amber, 1.6, 0.9);
      o += tx((cx + v[0][0] + v[1][0]) / 3, (cy + v[0][1] + v[1][1]) / 3, 'r', { fs: 12, fw: 700, fill: C.amber });
    }
    o += tx(cx, cy + R + 28, `n = ${n}`, { mono: true, fs: 14, fw: 700, fill: C.muted });
    if (i < 2) o += tx((cxs[i] + cxs[i + 1]) / 2, cy + 4, '→', { fs: 24, fw: 700, fill: C.muted });
  });
  o += tx(W / 2, H - 18, 'n triangles, each ≈ ½ · (2πr/n) · r   →   area → πr²', { fs: 13.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── the derivative is the slope of the tangent (secant → tangent) ───
function secantTangent() {
  const W = 470, H = 360; let o = '';
  const ox = 72, oy = 312, sx = 120, sy = 58;
  const f = x => 0.5 * x * x + 0.4;
  const X = x => ox + x * sx, Y = y => oy - y * sy;
  o += arrow(ox - 12, oy, W - 16, oy, C.gray);     // x-axis
  o += arrow(ox, oy + 12, ox, 44, C.gray);         // f-axis
  o += tx(W - 16, oy + 18, 'x', { fs: 13, fill: C.muted });
  o += tx(ox - 12, 50, 'f(x)', { fs: 13, fill: C.muted, anchor: 'end' });
  // curve
  const cpts = []; for (let x = 0.12; x <= 2.62; x += 0.04) cpts.push([X(x), Y(f(x))]);
  o += plines(cpts, C.ink, 2.6);
  const xP = 1, xQ = 2;
  const Px = X(xP), Py = Y(f(xP)), Qx = X(xQ), Qy = Y(f(xQ));
  // run + rise guides (the difference quotient)
  o += line(Px, Py, Qx, Py, C.amber, 2, true);
  o += line(Qx, Py, Qx, Qy, C.green, 2, true);
  o += tx((Px + Qx) / 2, Py + 18, 'h', { fs: 14, fw: 700, fill: C.amber });
  o += tx(Qx + 8, (Py + Qy) / 2, 'f(x+h) − f(x)', { fs: 12.5, fw: 700, fill: C.green, anchor: 'start' });
  // secant through P, Q (extended)
  const ms = (Qy - Py) / (Qx - Px);
  const secAt = px => Py + ms * (px - Px);
  const sL = X(0.4), sR = X(2.42);
  o += line(sL, secAt(sL), sR, secAt(sR), C.blue, 2.4);
  // tangent at P, slope f'(1)=1 in math units
  const mt = -(sy / sx) * 1.0;
  const tanAt = px => Py + mt * (px - Px);
  const tL = X(0.25), tR = X(1.9);
  o += line(tL, tanAt(tL), tR, tanAt(tR), C.purple, 2.4);
  // points + labels
  o += dot(Px, Py, 4, C.ink); o += dot(Qx, Qy, 4, C.ink);
  o += tx(Px - 9, Py + 5, 'P', { fs: 13, fw: 700, fill: C.ink, anchor: 'end' });
  o += tx(Qx + 8, Qy - 6, 'Q', { fs: 13, fw: 700, fill: C.ink, anchor: 'start' });
  o += tx(sR - 2, secAt(sR) - 9, 'secant', { fs: 12.5, fw: 700, fill: C.blue, anchor: 'end' });
  o += tx(tL + 2, tanAt(tL) - 9, 'tangent', { fs: 12.5, fw: 700, fill: C.purple, anchor: 'start' });
  o += tx(W / 2, 28, 'The derivative is the slope of the tangent', { fs: 15, fw: 700, fill: C.ink });
  o += tx(W / 2, H - 14, 'as h → 0, the secant through P, Q becomes the tangent at P', { fs: 12.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── gradient descent: each step moves downhill, along −∇L ───────────
function gradientDescent() {
  const W = 420, H = 330; let o = '';
  const cx = 222, cy = 178;
  const rings = [[152, 98], [118, 76], [86, 55], [55, 35], [27, 17]];
  rings.forEach((r, i) => o += ellipse(cx, cy, r[0], r[1], C.gray, 1.4, 0.5 + 0.1 * i));
  o += dot(cx, cy, 4, C.green);
  o += tx(cx, cy - 12, 'min', { fs: 12, fw: 700, fill: C.green });
  // a bending trajectory: drop into the valley, then slide along it
  const path = [[120, 96], [148, 150], [176, 168], [204, 174], [226, 177]];
  for (let i = 0; i < path.length - 1; i++) o += arrow(path[i][0], path[i][1], path[i + 1][0], path[i + 1][1], C.blue);
  path.forEach(p => o += dot(p[0], p[1], 3.2, C.blue));
  o += tx(path[0][0] - 6, path[0][1] - 8, 'θ₀', { mono: true, fs: 13, fw: 700, fill: C.blue, anchor: 'end' });
  o += tx(W / 2, 26, 'Each step moves downhill, along −∇L', { fs: 14.5, fw: 700, fill: C.ink });
  o += tx(W / 2, H - 14, 'gradient descent on the loss surface', { fs: 12.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── partial derivatives = slopes of 1-D slices of the surface ───────
function partialSlices() {
  const W = 470, H = 360; let o = '';
  const O = [150, 256], e1 = [168, 30], e2 = [128, -60], ey = [0, -150];
  const P3 = (a, b, h) => [
    O[0] + a * e1[0] + b * e2[0] + h * ey[0],
    O[1] + a * e1[1] + b * e2[1] + h * ey[1]];
  const hf = (a, b) => 1.6 * ((a - 0.5) ** 2 + (b - 0.5) ** 2);   // a bowl
  // faint surface mesh
  const N = 5, S = 22;
  for (let i = 0; i < N; i++) {
    const b = i / (N - 1), pa = [], pb = [];
    for (let j = 0; j <= S; j++) { const t = j / S; pa.push(P3(t, b, hf(t, b))); pb.push(P3(b, t, hf(b, t))); }
    o += plines(pa, '#CBD8E2', 1.1); o += plines(pb, '#CBD8E2', 1.1);
  }
  // axes
  o += arrow(O[0], O[1], O[0] + e1[0] + 24, O[1] + e1[1] + 4, C.gray);
  o += arrow(O[0], O[1], O[0] + e2[0] + 16, O[1] + e2[1] - 6, C.gray);
  o += arrow(O[0], O[1], O[0] + ey[0], O[1] + ey[1] - 8, C.gray);
  o += tx(O[0] + e1[0] + 30, O[1] + e1[1] + 12, 'x₁', { fs: 13, fw: 700, fill: C.muted, anchor: 'start' });
  o += tx(O[0] + e2[0] + 18, O[1] + e2[1] - 12, 'x₂', { fs: 13, fw: 700, fill: C.muted, anchor: 'start' });
  o += tx(O[0] - 6, O[1] + ey[1] - 12, 'f', { fs: 13, fw: 700, fill: C.muted, anchor: 'end' });
  // the point P and its two slices
  const aP = 0.74, bP = 0.74;
  const slA = [], slB = [];                       // A: vary x1 (b=bP); B: vary x2 (a=aP)
  for (let j = 0; j <= S; j++) { const t = j / S; slA.push(P3(t, bP, hf(t, bP))); slB.push(P3(aP, t, hf(aP, t))); }
  o += plines(slA, C.amber, 2.6);
  o += plines(slB, C.green, 2.6);
  const P = P3(aP, bP, hf(aP, bP));
  // tangents at P (screen-space derivative of P3 along each slice)
  const d = 3.2 * (aP - 0.5);                     // dh/da = dh/db at P
  const norm = (v, L) => { const m = Math.hypot(v[0], v[1]); return [v[0] / m * L, v[1] / m * L]; };
  const tA = norm([e1[0] + d * ey[0], e1[1] + d * ey[1]], 48);
  const tB = norm([e2[0] + d * ey[0], e2[1] + d * ey[1]], 48);
  o += line(P[0] - tA[0], P[1] - tA[1], P[0] + tA[0], P[1] + tA[1], '#B26A00', 2.6);
  o += line(P[0] - tB[0], P[1] - tB[1], P[0] + tB[0], P[1] + tB[1], '#2E7D32', 2.6);
  o += dot(P[0], P[1], 4.2, C.ink);
  o += tx(P[0] + tA[0] + 6, P[1] + tA[1] + 4, '∂f/∂x₁', { fs: 13, fw: 700, fill: '#B26A00', anchor: 'start' });
  o += tx(P[0] + tB[0] + 6, P[1] + tB[1] - 2, '∂f/∂x₂', { fs: 13, fw: 700, fill: '#2E7D32', anchor: 'start' });
  o += tx(W / 2, 30, 'Each partial = slope of a 1-D slice', { fs: 15, fw: 700, fill: C.ink });
  o += tx(W / 2, H - 14, 'hold the other variable fixed, cut the surface, read the slope', { fs: 12.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── the gradient is normal to the contours, pointing uphill ─────────
function gradientField() {
  const W = 420, H = 330; let o = '';
  const cx = 210, cy = 176, a0 = 152, b0 = 100;
  [1, 0.74, 0.5, 0.28].forEach(s => o += ellipse(cx, cy, a0 * s, b0 * s, C.gray, 1.5));
  o += dot(cx, cy, 3.6, C.green);
  o += tx(cx, cy - 12, 'min', { fs: 11.5, fw: 700, fill: C.green });
  const a = a0 * 0.74, b = b0 * 0.74, len = 34;
  const angs = [-52, 18, 74, 132, 244].map(d => d * Math.PI / 180);
  angs.forEach((t, i) => {
    const x = cx + a * Math.cos(t), y = cy + b * Math.sin(t);
    let nx = Math.cos(t) / a, ny = Math.sin(t) / b; const L = Math.hypot(nx, ny); nx /= L; ny /= L;
    o += arrow(x, y, x + nx * len, y + ny * len, C.blue);
    if (i === 0) o += tx(x + nx * len + 5, y + ny * len, '∇f', { fs: 12.5, fw: 700, fill: C.blue, anchor: 'start' });
  });
  // one −∇f (downhill) arrow for contrast
  const t0 = 200 * Math.PI / 180, x0 = cx + a * Math.cos(t0), y0 = cy + b * Math.sin(t0);
  let nx = Math.cos(t0) / a, ny = Math.sin(t0) / b; const L = Math.hypot(nx, ny); nx /= L; ny /= L;
  o += arrow(x0, y0, x0 - nx * len, y0 - ny * len, C.amber);
  o += tx(x0 - nx * len - 6, y0 - ny * len + 15, '−∇f', { fs: 12.5, fw: 700, fill: C.amber, anchor: 'end' });
  o += tx(W / 2, 26, '∇f ⟂ contours, pointing uphill', { fs: 14.5, fw: 700, fill: C.ink });
  o += tx(W / 2, H - 14, 'steepest ascent — −∇f points downhill', { fs: 12.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── the chain rule as a computational graph (forward / backward) ────
function chainGraph() {
  const W = 440, H = 238; let o = '';
  const y = 112, xs = [82, 220, 358], labels = ['x', 'u', 'y'];
  o += tx(W / 2, 26, 'Chain rule: multiply along the path', { fs: 15, fw: 700, fill: C.ink });
  o += tx(W / 2, 48, '▸ forward: evaluate', { fs: 12, fw: 700, fill: C.blue });
  // forward (blue, above)
  o += arrow(xs[0] + 30, y - 8, xs[1] - 30, y - 8, C.blue);
  o += arrow(xs[1] + 30, y - 8, xs[2] - 30, y - 8, C.blue);
  o += tx((xs[0] + xs[1]) / 2, y - 30, 'u = g(x)', { mono: true, fs: 13, fw: 700, fill: C.blue });
  o += tx((xs[1] + xs[2]) / 2, y - 30, 'y = f(u)', { mono: true, fs: 13, fw: 700, fill: C.blue });
  // backward (amber, below)
  o += arrow(xs[2] - 30, y + 36, xs[1] + 30, y + 36, C.amber);
  o += arrow(xs[1] - 30, y + 36, xs[0] + 30, y + 36, C.amber);
  o += tx((xs[1] + xs[2]) / 2, y + 56, 'dy/du', { mono: true, fs: 13, fw: 700, fill: C.amber });
  o += tx((xs[0] + xs[1]) / 2, y + 56, 'du/dx', { mono: true, fs: 13, fw: 700, fill: C.amber });
  o += tx(W / 2, y + 80, '◂ backward: differentiate', { fs: 12, fw: 700, fill: C.amber });
  xs.forEach((x, i) => o += chip(x, y, labels[i]));
  o += tx(W / 2, H - 12, 'dy/dx = (dy/du)(du/dx)', { fs: 14.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

export const diagrams = {
  'calculus-circle-limit':    circleLimit,
  'calculus-secant-tangent':  secantTangent,
  'calculus-gradient-descent': gradientDescent,
  'calculus-partial-slices':  partialSlices,
  'calculus-gradient-field':  gradientField,
  'calculus-chain-graph':     chainGraph,
};
