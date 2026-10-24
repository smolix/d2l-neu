// Diagrams for chapter_preliminaries/calculus.
//
// The derivative as the slope of a tangent (secant -> tangent, also a book
// figure in img/) and the chain rule as a computational graph traversed
// forward (evaluate) and backward (differentiate), slide-only.
//
// calculus-gradient-field and calculus-partial-slices are drawn in
// matplotlib from the section's running example
// (tools/figstyle/remake_ch02_calculus.py); the earlier calculus-circle-limit
// and calculus-gradient-descent diagrams were retired (the book's polygon
// figure is img/polygon-circle.svg).
//
// Variables are set in italics (figure-style guide 5.1): `mtx` italicizes
// every Latin letter of a label and keeps digits, operators, and brackets
// upright.
//
// Stable ids use the `calculus-<concept>` prefix. Never rename.

import { C, TOKENS, FS, tx, baselineY, arrow, rc, svg } from './engine.mjs';

const ORANGE_DARK = TOKENS.accents.orange.dark;
const GREEN_DARK = TOKENS.accents.green.dark;

const line = (x1, y1, x2, y2, color, w = 1.5, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
const dot = (x, y, r, fill) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}"/>`;
const plines = (pts, st, w = 2, dash = false) =>
  `<polyline points="${pts.map(p => p.map(n => n.toFixed(1)).join(',')).join(' ')}" fill="none" stroke="${st}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
// A label with italic variables: Latin letters italic, everything else upright.
// Centred on y via an alphabetic baseline (engine baselineY; librsvg ignores
// dominant-baseline).
function mtx(x, y, t, o) {
  o = o || {};
  const fs = o.fs || 16;
  const body = String(t).replace(/[A-Za-z]/g, c => `<tspan font-style="italic">${c}</tspan>`);
  return `<text x="${x}" y="${baselineY(y, fs)}" font-family="${FS}" font-size="${fs}" font-weight="${o.fw || 500}" fill="${o.fill || C.ink}" text-anchor="${o.anchor || 'middle'}">${body}</text>`;
}
// Rotated label centred on (x, y); the baseline shift is applied before the
// rotation, so it stays perpendicular to the text.
const rotated = (x, y, deg, text, color) =>
  `<text x="${x}" y="${baselineY(y, 15)}" transform="rotate(${deg} ${x} ${y})" font-family="${FS}" font-size="15" font-weight="700" fill="${color}" text-anchor="middle">${text}</text>`;

// ── the derivative is the slope of the tangent (secant → tangent) ───
function secantTangent() {
  const W = 470, H = 345; let o = '';
  const ox = 72, oy = 312, sx = 120, sy = 58;
  const f = x => 0.5 * x * x + 0.4;
  const X = x => ox + x * sx, Y = y => oy - y * sy;
  o += arrow(ox - 12, oy, W - 16, oy, C.ink);      // x-axis
  o += arrow(ox, oy + 12, ox, 44, C.ink);          // f-axis
  o += mtx(W - 16, oy + 20, 'x', { fs: 17 });
  o += mtx(ox - 12, 50, 'f(x)', { fs: 17, anchor: 'end' });
  // curve
  const cpts = []; for (let x = 0.12; x <= 2.62; x += 0.04) cpts.push([X(x), Y(f(x))]);
  o += plines(cpts, C.ink, 2.6);
  const xP = 1, xQ = 2;
  const Px = X(xP), Py = Y(f(xP)), Qx = X(xQ), Qy = Y(f(xQ));
  // run + rise guides (the difference quotient)
  o += line(Px, Py, Qx, Py, C.amber, 2, true);
  o += line(Qx, Py, Qx, Qy, C.green, 2, true);
  o += mtx((Px + Qx) / 2, Py + 20, 'h', { fs: 17, fill: ORANGE_DARK });
  o += mtx(Qx + 8, (Py + Qy) / 2, 'f(x + h) − f(x)', { fs: 16, fill: GREEN_DARK, anchor: 'start' });
  // secant through P, Q (extended)
  const ms = (Qy - Py) / (Qx - Px);
  const secAt = px => Py + ms * (px - Px);
  const sL = X(0.4), sR = X(2.42);
  o += line(sL, secAt(sL), sR, secAt(sR), C.blue, 2.4);
  // tangent at P, slope f'(1)=1 in math units
  const mt = -(sy / sx) * 1.0;
  const tanAt = px => Py + mt * (px - Px);
  const tL = X(0.25), tR = X(2.6);
  o += line(tL, tanAt(tL), tR, tanAt(tR), C.purple, 2.4);
  // points + labels
  o += dot(Px, Py, 4.5, C.ink); o += dot(Qx, Qy, 4.5, C.ink);
  o += mtx(Px, Py + 20, 'P', { fs: 17 });
  o += mtx(Qx - 10, Qy - 14, 'Q', { fs: 17, anchor: 'end' });
  o += rotated(241, 205, -35.9, 'secant', C.blue);
  o += rotated(384, 151, -25.8, 'tangent', TOKENS.accents.purple.dark);
  return svg(W, H, o);
}

// ── the chain rule as a computational graph (forward / backward) ────
function chainGraph() {
  const W = 440, H = 238; let o = '';
  const y = 112, xs = [82, 220, 358], labels = ['x', 'u', 'z'];
  const node = (cx, cy, t) => rc(cx - 23, cy - 23, 46, TOKENS.accents.slate.tint, C.gray, false)
    + mtx(cx, cy, t, { fs: 22, fill: C.ink });
  o += tx(W / 2, 48, '▸ forward: evaluate', { fs: 14.5, fw: 700, fill: C.blue });
  // forward (blue, above)
  o += arrow(xs[0] + 30, y - 8, xs[1] - 30, y - 8, C.blue);
  o += arrow(xs[1] + 30, y - 8, xs[2] - 30, y - 8, C.blue);
  o += mtx((xs[0] + xs[1]) / 2, y - 30, 'u = g(x)', { fs: 16, fill: TOKENS.accents.blue.dark });
  o += mtx((xs[1] + xs[2]) / 2, y - 30, 'z = f(u)', { fs: 16, fill: TOKENS.accents.blue.dark });
  // backward (red, the gradient role, below)
  o += arrow(xs[2] - 30, y + 36, xs[1] + 30, y + 36, TOKENS.accents.red.base);
  o += arrow(xs[1] - 30, y + 36, xs[0] + 30, y + 36, TOKENS.accents.red.base);
  o += mtx((xs[1] + xs[2]) / 2, y + 56, 'dz/du', { fs: 16, fill: TOKENS.accents.red.dark });
  o += mtx((xs[0] + xs[1]) / 2, y + 56, 'du/dx', { fs: 16, fill: TOKENS.accents.red.dark });
  o += tx(W / 2, y + 80, '◂ backward: differentiate', { fs: 14.5, fw: 700, fill: TOKENS.accents.red.dark });
  xs.forEach((x, i) => o += node(x, y, labels[i]));
  o += mtx(W / 2, H - 12, 'dz/dx = (dz/du)(du/dx)', { fs: 18 });
  return svg(W, H, o);
}

export const diagrams = {
  'calculus-secant-tangent':  secantTangent,
  'calculus-chain-graph':     chainGraph,
};
