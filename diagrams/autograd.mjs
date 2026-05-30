// Diagrams for chapter_preliminaries/autograd (§2.5).
//
// The computational graph for the worked example y = 2 xᵀx: the forward
// pass builds the graph left→right; reverse-mode autodiff walks it
// right→left, multiplying the local derivative at each node (chain rule)
// to accumulate ∂y/∂x = 4x. Used as the book figure (img/) and, later,
// in the slide deck.
//
// Stable ids use the `autograd-<concept>` prefix. Never rename.

import { C, tx, arrow, svg } from './engine.mjs';

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

export const diagrams = {
  'autograd-comp-graph': computeGraph,
};
