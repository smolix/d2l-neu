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
  o += rotated(228, 190, -35.9, 'secant', C.blue);
  o += rotated(396, 184, -25.8, 'tangent', TOKENS.accents.purple.dark);
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


// ── the product rule as the area of a growing rectangle ─────────────
// Sides f(x) and g(x); when x grows by h they grow by about f'(x)h and
// g'(x)h. The area grows by two thin strips and a corner of order h².
function productRule() {
  const W = 470, H = 300; let o = '';
  const x0 = 96, yb = 236;            // bottom-left corner of f(x) g(x)
  const fw = 220, gh = 132;           // the sides f(x) and g(x)
  const dw = 46, dh = 34;             // the growth f'(x)h and g'(x)h
  const rect = (x, y, w, h, fill, st) =>
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="${st}" stroke-width="2"/>`;
  o += rect(x0, yb - gh, fw, gh, C.lblue, C.blue);                    // f g
  o += rect(x0 + fw, yb - gh, dw, gh, C.lamber, C.amber);             // f'h g
  o += rect(x0, yb - gh - dh, fw, dh, C.lgreen, C.green);             // f g'h
  o += rect(x0 + fw, yb - gh - dh, dw, dh, C.lgray, C.gray);          // f'h g'h
  o += mtx(x0 + fw / 2, yb - gh / 2, 'f(x) g(x)', { fs: 17 });
  o += mtx(x0 + fw / 2, yb - gh - dh / 2, 'f(x) · g′(x) h', { fs: 15, fill: GREEN_DARK });
  o += `<g transform="rotate(-90 ${x0 + fw + dw / 2} ${yb - gh / 2})">`
    + mtx(x0 + fw + dw / 2, yb - gh / 2, 'f′(x) h · g(x)', { fs: 15, fill: ORANGE_DARK }) + '</g>';
  o += mtx(x0 + fw + dw + 10, yb - gh - dh / 2, 'f′(x) g′(x) h²', { fs: 15, anchor: 'start' });
  // side lengths
  o += mtx(x0 + fw / 2, yb + 20, 'f(x)', { fs: 16 });
  o += mtx(x0 + fw + dw / 2, yb + 20, 'f′(x) h', { fs: 15, fill: ORANGE_DARK });
  o += mtx(x0 - 10, yb - gh / 2, 'g(x)', { fs: 16, anchor: 'end' });
  o += mtx(x0 - 10, yb - gh - dh / 2, 'g′(x) h', { fs: 15, fill: GREEN_DARK, anchor: 'end' });
  return svg(W, H, o);
}

// ── Worked example: summing parallel paths on a diamond DAG ───────────
// u = x1 * x2, v = x2^2, z = u^2 + v at (x1, x2) = (1, 2)
function parallelDag() {
  const W = 560, H = 290; let o = '';
  const RED = TOKENS.accents.red;
  const BLUE_DK = TOKENS.accents.blue.dark;
  const box = (cx, cy, w, h, fill, st, title, val, grad) =>
    `<rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${st}" stroke-width="2"/>`
    + tx(cx, cy - 17, title, { mono: true, fs: 13.5, fw: 700, fill: C.ink })
    + tx(cx, cy + 2, val, { mono: true, fs: 13, fw: 600, fill: BLUE_DK })
    + tx(cx, cy + 20, grad, { mono: true, fs: 12.5, fw: 700, fill: RED.dark });

  // Node coordinates with wider horizontal separation
  const X1 = [68, 72], X2 = [68, 212];
  const U = [276, 72], V = [276, 212];
  const Z = [488, 142];

  // Forward blue arrows + backward red arrows
  // x1 -> u
  o += arrow(X1[0] + 56, X1[1] - 6, U[0] - 62, U[1] - 6, C.blue);
  o += arrow(U[0] - 62, U[1] + 10, X1[0] + 56, X1[1] + 10, RED.base);
  o += tx(170, 48, '∂u/∂x₁ = 2', { mono: true, fs: 12.5, fw: 700, fill: RED.dark });

  // x2 -> u (diagonal)
  o += arrow(X2[0] + 54, X2[1] - 24, U[0] - 60, U[1] + 24, C.blue);
  o += arrow(U[0] - 52, U[1] + 34, X2[0] + 62, X2[1] - 14, RED.base);
  o += tx(198, 156, '∂u/∂x₂ = 1', { mono: true, fs: 12.5, fw: 700, fill: RED.dark, anchor: 'start' });

  // x2 -> v
  o += arrow(X2[0] + 56, X2[1] - 6, V[0] - 62, V[1] - 6, C.blue);
  o += arrow(V[0] - 62, V[1] + 10, X2[0] + 56, X2[1] + 10, RED.base);
  o += tx(170, 238, '∂v/∂x₂ = 4', { mono: true, fs: 12.5, fw: 700, fill: RED.dark });

  // u -> z
  o += arrow(U[0] + 62, U[1] + 10, Z[0] - 56, Z[1] - 18, C.blue);
  o += arrow(Z[0] - 60, Z[1] - 6, U[0] + 58, U[1] + 22, RED.base);
  o += tx(388, 66, '∂z/∂u = 4', { mono: true, fs: 12.5, fw: 700, fill: RED.dark });

  // v -> z
  o += arrow(V[0] + 62, V[1] - 10, Z[0] - 56, Z[1] + 18, C.blue);
  o += arrow(Z[0] - 60, Z[1] + 6, V[0] + 58, V[1] - 22, RED.base);
  o += tx(388, 218, '∂z/∂v = 1', { mono: true, fs: 12.5, fw: 700, fill: RED.dark });

  // Nodes
  o += box(X1[0], X1[1], 108, 64, TOKENS.accents.slate.tint, C.gray, 'x₁', 'val = 1', '∂z/∂x₁ = 8');
  o += box(X2[0], X2[1], 108, 64, C.lamber, C.amber, 'x₂ (2 paths)', 'val = 2', '4·1 + 1·4 = 8');
  o += box(U[0], U[1], 120, 64, C.lblue, C.blue, 'u = x₁ x₂', 'val = 2', '∂z/∂u = 4');
  o += box(V[0], V[1], 120, 64, C.lblue, C.blue, 'v = x₂²', 'val = 4', '∂z/∂v = 1');
  o += box(Z[0], Z[1], 108, 64, C.lgreen, C.green, 'z = u² + v', 'val = 8', '∂z/∂z = 1');

  o += tx(280, 272, 'Sum over all paths from x₂ to z:  ∂z/∂x₂ = (4 × 1) + (1 × 4) = 8', { mono: true, fs: 14, fw: 700, fill: RED.dark });
  return svg(W, H, o);
}

// ── Two-stage VJP pipeline for L(w) = ||Xw - y||^2 ────────────────────
function vjpPipeline() {
  const W = 540, H = 250; let o = '';
  const RED = TOKENS.accents.red;
  const BLUE_DK = TOKENS.accents.blue.dark;
  const card = (x, y, w, h, fill, st, t1, t2, t3) =>
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${fill}" stroke="${st}" stroke-width="2"/>`
    + tx(x + w / 2, y + 22, t1, { mono: true, fs: 15, fw: 700, fill: C.ink })
    + tx(x + w / 2, y + 46, t2, { mono: true, fs: 13, fw: 600, fill: BLUE_DK })
    + tx(x + w / 2, y + 70, t3, { mono: true, fs: 13, fw: 700, fill: RED.dark });

  const y = 78, bh = 90;
  o += tx(16, 26, '▸ Forward Pass (left → right): evaluate predictions & scalar loss', { fs: 14.5, fw: 700, fill: BLUE_DK, anchor: 'start' });

  // Arrows between w -> r -> L
  o += arrow(150, y + 28, 204, y + 28, C.blue);
  o += tx(177, y + 10, 'Xw − y', { mono: true, fs: 13, fw: 700, fill: BLUE_DK });
  o += arrow(204, y + 64, 150, y + 64, RED.base);
  o += tx(177, y + 84, 'Xᵀ g_r', { mono: true, fs: 13, fw: 700, fill: RED.dark });

  o += arrow(346, y + 28, 400, y + 28, C.blue);
  o += tx(373, y + 10, '‖r‖₂²', { mono: true, fs: 13, fw: 700, fill: BLUE_DK });
  o += arrow(400, y + 64, 346, y + 64, RED.base);
  o += tx(373, y + 84, '2 r · 1', { mono: true, fs: 13, fw: 700, fill: RED.dark });

  // 3 Nodes
  o += card(14,  y, 132, bh, TOKENS.accents.slate.tint, C.gray, 'w ∈ R²', 'w = (1, 0)', '∇_w L = (2, 4)');
  o += card(208, y, 134, bh, C.lblue, C.blue, 'r ∈ R³', 'r = (0, 0, 1)', 'g_r = (0, 0, 2)');
  o += card(404, y, 122, bh, C.lgreen, C.green, 'L ∈ R', 'L = 1', '∂L/∂L = 1');

  o += tx(16, 212, '◂ Backward Pass (right → left): each stage applies its VJP', { fs: 14, fw: 700, fill: RED.dark, anchor: 'start' });
  o += tx(16, 236, 'Never forms giant m×n Jacobians — passes only vectors of shape Rᵐ and Rⁿ!', { fs: 13, fw: 600, fill: C.ink, anchor: 'start' });
  return svg(W, H, o);
}

// ── Smooth surrogate loss vs 0-1 step error & 1 gradient vs d+1 probes ─
function surrogateVsStep() {
  const W = 540, H = 264; let o = '';
  const RED = TOKENS.accents.red;
  const BLUE_DK = TOKENS.accents.blue.dark;
  // Panel (a): 0-1 step error vs smooth surrogate log(1 + exp(-z))
  o += tx(135, 20, '(a) Surrogate Loss vs. 0–1 Error', { fs: 15, fw: 700, fill: C.ink });
  const ox = 135, oy = 216, sx = 36, sy = 44;
  o += line(26, oy, 246, oy, C.ink, 1.6);
  o += line(ox, oy + 8, ox, 40, C.ink, 1.6);
  o += tx(244, oy + 16, 'margin z', { fs: 13, fw: 600, fill: C.ink, anchor: 'end' });
  // 0-1 step function (red: flat slope = 0!)
  o += line(32, oy - sy, ox, oy - sy, RED.base, 2.8);
  o += line(ox, oy - sy, ox, oy, RED.base, 1.6, true);
  o += line(ox, oy, 240, oy, RED.base, 2.8);
  o += tx(ox + 12, oy - sy - 10, '0–1 step: slope = 0', { fs: 12.5, fw: 700, fill: RED.dark, anchor: 'start' });
  // Smooth logistic surrogate log2(1 + exp(-z)) clamped to z >= -2.1 so top stays clear of title
  const pts = [];
  for (let i = -21; i <= 28; i++) {
    const z = i / 10;
    const val = Math.log(1 + Math.exp(-z)) / Math.LN2;
    pts.push([ox + z * sx, oy - val * sy]);
  }
  o += plines(pts, C.blue, 2.8);
  o += tx(ox + 10, 68, 'surrogate L(z):', { fs: 13, fw: 700, fill: BLUE_DK, anchor: 'start' });
  o += tx(ox + 10, 86, 'slope ≠ 0 everywhere', { fs: 12.5, fw: 600, fill: BLUE_DK, anchor: 'start' });

  // Panel (b): d+1 random perturbation probes vs 1 gradient vector -∇L
  o += tx(405, 20, '(b) 1 Gradient vs. d+1 Probes', { fs: 15, fw: 700, fill: C.ink });
  const cx = 434, cy = 104;
  o += `<ellipse cx="${cx}" cy="${cy}" rx="86" ry="56" fill="none" stroke="${C.gray}" stroke-width="1.4"/>`;
  o += `<ellipse cx="${cx}" cy="${cy}" rx="54" ry="35" fill="none" stroke="${C.gray}" stroke-width="1.4"/>`;
  o += `<ellipse cx="${cx}" cy="${cy}" rx="24" ry="15" fill="none" stroke="${C.gray}" stroke-width="1.4"/>`;
  o += dot(cx, cy, 5.5, C.green);
  const wx = 336, wy = 162;
  // Blind probe rays
  for (const ang of [20, 80, 140, 200, 260, 320]) {
    const rad = (ang * Math.PI) / 180;
    const px = wx + 32 * Math.cos(rad), py = wy + 32 * Math.sin(rad);
    o += line(wx, wy, px, py, C.muted, 1.4, true);
    o += dot(px, py, 3, C.muted);
  }
  o += arrow(wx, wy, cx - 14, cy + 10, C.green);
  o += dot(wx, wy, 5, C.ink);
  o += tx(442, 184, '−∇_w L (1 sweep)', { fs: 13, fw: 700, fill: GREEN_DARK });
  o += tx(405, 224, 'finite differences need O(d) passes', { fs: 13, fill: C.muted });
  return svg(W, H, o);
}

export const diagrams = {
  'calculus-secant-tangent':    secantTangent,
  'calculus-chain-graph':       chainGraph,
  'calculus-product-rule':      productRule,
  'calculus-parallel-dag':      parallelDag,
  'calculus-vjp-pipeline':      vjpPipeline,
  'calculus-surrogate-vs-step': surrogateVsStep,
};
