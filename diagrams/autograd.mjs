// Diagrams for chapter_preliminaries/autograd (§2.5).
//
// The computational graph for the worked example y = 2 xᵀx: the forward
// pass builds the graph left→right; reverse-mode autodiff walks it
// right→left, multiplying the local derivative at each node (chain rule)
// to accumulate ∂y/∂x = 4x. Used as the book figure (img/) and, later,
// in the slide deck.
//
// Stable ids use the `autograd-<concept>` prefix. Never rename.

import { C, tx, arrow, chip, svg } from './engine.mjs';

const line = (x1, y1, x2, y2, color, w = 1.6, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const node = (cx, cy, w, h, fill, stroke, label, sub) => {
  let o = `<rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="10" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`;
  o += tx(cx, cy - (sub ? 8 : 0), label, { mono: true, fs: 18, fw: 700, fill: C.ink });
  if (sub) o += tx(cx, cy + 13, sub, { fs: 11.5, fw: 500, fill: C.muted });
  return o;
};

function computeGraph() {
  const W = 600, H = 250; let o = '';
  const cy = 104, h = 58;
  const X = { cx: 78, w: 78 }, A = { cx: 270, w: 110 }, Y = { cx: 488, w: 130 };

  o += tx(W / 2, 30, 'Computational graph for  y = 2 xᵀx', { fs: 15, fw: 700, fill: C.ink });

  // forward edges (dark, top) with the op that produced the next node
  o += arrow(X.cx + X.w / 2, cy - 10, A.cx - A.w / 2, cy - 10, C.ink);
  o += arrow(A.cx + A.w / 2, cy - 10, Y.cx - Y.w / 2, cy - 10, C.ink);
  o += tx((X.cx + A.cx) / 2, cy - 26, 'dot', { mono: true, fs: 12.5, fw: 700, fill: C.ink });
  o += tx((A.cx + Y.cx) / 2, cy - 26, '× 2', { mono: true, fs: 12.5, fw: 700, fill: C.ink });

  // backward edges (blue, bottom) with the local derivative multiplied in
  o += arrow(Y.cx - Y.w / 2, cy + 10, A.cx + A.w / 2, cy + 10, C.blue);
  o += arrow(A.cx - A.w / 2, cy + 10, X.cx + X.w / 2, cy + 10, C.blue);
  o += tx((A.cx + Y.cx) / 2, cy + 28, '∂y/∂a = 2', { fs: 12.5, fw: 700, fill: C.blue });
  o += tx((X.cx + A.cx) / 2, cy + 28, '∂a/∂x = 2x', { fs: 12.5, fw: 700, fill: C.blue });

  // nodes
  o += node(X.cx, cy, X.w, h, '#F3E5F5', C.purple, 'x', 'input');
  o += node(A.cx, cy, A.w, h, C.lblue, C.blue, 'a = xᵀx', 'scalar');
  o += node(Y.cx, cy, Y.w, h, C.lgreen, C.green, 'y = 2a', 'output');

  // forward / backward legend + accumulated result
  o += tx(X.cx, 64, 'forward →', { fs: 11.5, fw: 700, fill: C.ink, anchor: 'start' });
  o += tx(Y.cx + Y.w / 2, 64, '← backward', { fs: 11.5, fw: 700, fill: C.blue, anchor: 'end' });
  o += tx(W / 2, H - 22, 'chain rule:  ∂y/∂x = (∂y/∂a)(∂a/∂x) = 2 · 2x = 4x', { fs: 14, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// ── the autograd loop: mark → forward → backward → read ─────────────
function workflow() {
  const W = 700, H = 236; let o = '';
  const cy = 100, bw = 138, bh = 66;
  const gap = (W - 56 - 4 * bw) / 3;
  const xs = [28, 28 + bw + gap, 28 + 2 * (bw + gap), 28 + 3 * (bw + gap)];
  const steps = [
    ['1', 'mark', 'flag an input'],
    ['2', 'forward', 'records the graph'],
    ['3', 'backward', 'chain rule, reversed'],
    ['4', 'read .grad', 'use it, then reset'],
  ];
  o += tx(W / 2, 28, 'The autograd loop', { fs: 15, fw: 700, fill: C.ink });
  steps.forEach((s, i) => {
    const x = xs[i], cxb = x + bw / 2;
    const accent = i === 2 ? C.blue : C.gray;
    o += `<rect x="${x}" y="${cy - bh / 2}" width="${bw}" height="${bh}" rx="11" fill="#fff" stroke="${accent}" stroke-width="2"/>`;
    o += `<circle cx="${x + 19}" cy="${cy - bh / 2 + 19}" r="11.5" fill="${C.purple}"/>`;
    o += tx(x + 19, cy - bh / 2 + 19, s[0], { fs: 13, fw: 700, fill: '#fff' });
    o += tx(cxb + 9, cy - 6, s[1], { mono: true, fs: 14.5, fw: 700, fill: i === 2 ? C.blue : C.ink });
    o += tx(cxb, cy + 16, s[2], { fs: 11, fw: 500, fill: C.muted });
    if (i < 3) o += arrow(x + bw + 3, cy, xs[i + 1] - 3, cy, C.ink);
  });
  // return loop underneath
  const yb = H - 24, x0 = xs[0] + bw / 2, x1 = xs[3] + bw / 2;
  o += line(x1, cy + bh / 2, x1, yb, C.gray, 1.5);
  o += line(x1, yb, x0, yb, C.gray, 1.5);
  o += arrow(x0, yb, x0, cy + bh / 2 + 2, C.gray);
  o += tx((x0 + x1) / 2, yb - 7, 'repeated every optimization step', { fs: 11.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── detach: the graph above u is severed; gradient flows around it ───
function detachGraph() {
  const Wd = 620, H = 258; let o = '';
  const cy = 92, bh = 52;
  const N = { x: { cx: 70, w: 62 }, sq: { cx: 214, w: 78 }, u: { cx: 372, w: 116 }, z: { cx: 532, w: 116 } };
  o += tx(Wd / 2, 28, 'Detaching freezes a value to a constant', { fs: 15, fw: 700, fill: C.ink });
  // forward (dark, top)
  o += arrow(N.x.cx + N.x.w / 2, cy, N.sq.cx - N.sq.w / 2, cy, C.ink);
  o += tx((N.x.cx + N.sq.cx) / 2, cy - 12, 'square', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  // sq → u edge is the detach: dashed + a red cut
  o += line(N.sq.cx + N.sq.w / 2, cy, N.u.cx - N.u.w / 2, cy, C.gray, 2, true);
  const mx = (N.sq.cx + N.u.cx) / 2;
  o += line(mx - 7, cy - 9, mx + 7, cy + 9, '#E53935', 2.4);
  o += line(mx - 7, cy + 9, mx + 7, cy - 9, '#E53935', 2.4);
  o += tx(mx, cy - 16, 'detach', { mono: true, fs: 11.5, fw: 700, fill: '#E53935' });
  o += arrow(N.u.cx + N.u.w / 2, cy, N.z.cx - N.z.w / 2, cy, C.ink);
  o += tx((N.u.cx + N.z.cx) / 2, cy - 12, '× x', { mono: true, fs: 11.5, fw: 700, fill: C.ink });
  // nodes
  o += node(N.x.cx, cy, N.x.w, bh, '#F3E5F5', C.purple, 'x', 'input');
  o += node(N.sq.cx, cy, N.sq.w, bh, C.lblue, C.blue, 'x²', null);
  // u: dashed border = treated as constant
  o += `<rect x="${N.u.cx - N.u.w / 2}" y="${cy - bh / 2}" width="${N.u.w}" height="${bh}" rx="10" fill="${C.lgray}" stroke="${C.gray}" stroke-width="2" stroke-dasharray="5 4"/>`;
  o += tx(N.u.cx, cy - 8, 'u', { mono: true, fs: 18, fw: 700, fill: C.ink });
  o += tx(N.u.cx, cy + 13, 'constant', { fs: 11, fw: 500, fill: C.muted });
  o += node(N.z.cx, cy, N.z.w, bh, C.lgreen, C.green, 'z = u·x', null);
  // direct edge x → z, drawn as the BACKWARD (blue) gradient path along the bottom
  const yb = cy + 78;
  o += `<path d="M${N.z.cx},${cy + bh / 2} C ${N.z.cx},${yb} ${N.x.cx},${yb} ${N.x.cx},${cy + bh / 2}" fill="none" stroke="${C.blue}" stroke-width="2.4"/>`;
  o += arrow(N.x.cx + 1, cy + bh / 2 + 14, N.x.cx, cy + bh / 2 + 1, C.blue);
  o += tx(Wd / 2, yb + 4, '∂z/∂x = u   (gradient returns only through the direct edge)', { fs: 12.5, fw: 700, fill: C.blue });
  o += tx(Wd / 2, H - 14, 'the path above u is cut, so z = x·x·x but ∂z/∂x = u, not 3x²', { fs: 12, fw: 600, fill: C.muted });
  return svg(Wd, H, o);
}

// ── dynamic graphs: control flow realizes a different graph per input ─
function dynamic() {
  const W = 600, H = 250; let o = '';
  o += tx(W / 2, 28, 'The graph is built at runtime', { fs: 15, fw: 700, fill: C.ink });
  const doublings = (y, n, inLabel, branch, bcolor) => {
    let s = '', x = 44;
    s += `<rect x="${x - 22}" y="${y - 17}" width="60" height="34" rx="8" fill="#F3E5F5" stroke="${C.purple}" stroke-width="2"/>`;
    s += tx(x + 8, y, inLabel, { mono: true, fs: 12.5, fw: 700, fill: C.purple });
    x += 50;
    for (let i = 0; i < n; i++) {
      s += `<rect x="${x}" y="${y - 15}" width="30" height="30" rx="6" fill="${C.lblue}" stroke="${C.blue}" stroke-width="1.8"/>`;
      s += tx(x + 15, y, '×2', { mono: true, fs: 10.5, fw: 700, fill: C.ink });
      if (i < n - 1) s += arrow(x + 30, y, x + 36, y, C.ink);
      x += 36;
    }
    s += arrow(x, y, x + 12, y, C.ink);
    x += 16;
    s += `<rect x="${x}" y="${y - 17}" width="118" height="34" rx="8" fill="#fff" stroke="${bcolor}" stroke-width="2"/>`;
    s += tx(x + 59, y, branch, { mono: true, fs: 11, fw: 700, fill: bcolor });
    x += 118;
    s += arrow(x, y, x + 14, y, C.ink);
    s += `<circle cx="${x + 30}" cy="${y}" r="15" fill="${C.lgreen}" stroke="${C.green}" stroke-width="2"/>`;
    s += tx(x + 30, y, 'c', { mono: true, fs: 14, fw: 700, fill: C.ink });
    return s;
  };
  o += tx(40, 70, 'a = 0.30', { fs: 11.5, fw: 700, fill: C.muted, anchor: 'start' });
  o += doublings(96, 3, 'a', 'b.sum()>0 → b', C.green);
  o += tx(40, 150, 'a = −0.75', { fs: 11.5, fw: 700, fill: C.muted, anchor: 'start' });
  o += doublings(176, 5, 'a', 'else → 100·b', C.amber);
  o += tx(W / 2, H - 14, 'different input → different path → autograd records whatever ran', { fs: 12, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── forward vs reverse mode: which way you sweep the graph ───────────
function fwdVsRev() {
  const W = 560, H = 250; let o = '';
  const xs = [130, 290, 450];
  const labels = ['x', 'h', 'y'];
  o += tx(W / 2, 26, 'Two ways to sweep the same graph', { fs: 15, fw: 700, fill: C.ink });
  // forward (top, dark, →)
  const yt = 86;
  o += tx(W / 2, yt - 34, 'forward mode: one input → all outputs', { fs: 12.5, fw: 700, fill: C.ink });
  o += arrow(xs[0] + 26, yt, xs[1] - 26, yt, C.ink);
  o += arrow(xs[1] + 26, yt, xs[2] - 26, yt, C.ink);
  xs.forEach((x, i) => o += chip(x, yt, labels[i]));
  // reverse (bottom, blue, ←)
  const yb = 174;
  o += arrow(xs[2] - 26, yb, xs[1] + 26, yb, C.blue);
  o += arrow(xs[1] - 26, yb, xs[0] + 26, yb, C.blue);
  xs.forEach((x, i) => o += chip(x, yb, labels[i]));
  o += tx(W / 2, yb + 36, 'reverse mode (backprop): one output → all inputs', { fs: 12.5, fw: 700, fill: C.blue });
  o += tx(W / 2, H - 12, 'a scalar loss over millions of parameters → reverse wins', { fs: 12, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

export const diagrams = {
  'autograd-comp-graph': computeGraph,
  'autograd-workflow':   workflow,
  'autograd-detach':     detachGraph,
  'autograd-dynamic':    dynamic,
  'autograd-fwd-vs-rev': fwdVsRev,
};
