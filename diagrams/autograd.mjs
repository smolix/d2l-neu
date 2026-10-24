// Diagrams for chapter_preliminaries/autograd.
//
// Colors follow the book-wide roles (figure-style guide 4.3, tokens.mjs):
// the forward pass, which computes values, is blue (`stream`); the backward
// pass and gradients are red (`grad`); parameters and other nodes are
// neutral slate (`params`). Structure is ink or gray.
//
//   autograd-comp-graph  the graph recorded for the running example, the
//                        line model's loss L = ||Xw - y||^2 at w = (1, 0):
//                        values forward, gradients backward (book + slides)
//   autograd-detach      the same product z = u x with u attached (two
//                        gradient paths, 3x^2) and detached (one path, x^2)
//                        (book + slides)
//   autograd-fwd-vs-rev  forward mode fills the Jacobian one column per
//                        sweep, reverse mode one row per sweep (book + slides)
//   autograd-dynamic     control flow: two inputs of the section's function
//                        f record two different graphs (book + slides)
//
// Variables are italic, vectors and matrices bold upright (figure-style
// guide 5.1); `rich` handles both, plus sub- and superscripts.
//
// Stable ids use the `autograd-<concept>` prefix. Never rename.

import { C, TOKENS, FS, tx, baselineY, arrow, svg } from './engine.mjs';

const BLUE_DARK = TOKENS.accents.blue.dark;
const RED = TOKENS.accents.red;          // gradients and the backward pass
const SLATE = TOKENS.accents.slate;      // parameters and other nodes
const FAINT = TOKENS.neutrals.faint;
const HAIR = TOKENS.neutrals.hairline;

const line = (x1, y1, x2, y2, color, w = 2, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
const box = (x, y, w, h, fill, stroke, dash = false, rx = 8, sw = 2) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;

const esc = (t) => String(t).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// Math-like label. Latin letters are italic; \b{...} is bold upright (a
// vector or matrix); _{...} and ^{...} are sub- and superscripts (a single
// character may follow _ or ^ without braces). Digits and operators stay
// upright.
function parse(src, style, out) {
  let i = 0;
  const group = () => {
    if (src[i] === '{') {
      let depth = 1, j = i + 1;
      while (j < src.length && depth) { if (src[j] === '{') depth++; else if (src[j] === '}') depth--; j++; }
      const inner = src.slice(i + 1, j - 1); i = j; return inner;
    }
    return src[i++];
  };
  while (i < src.length) {
    if (src.startsWith('\\b{', i)) { i += 2; parse(group(), { ...style, bold: true }, out); continue; }
    if (src.startsWith('\\r{', i)) { i += 2; parse(group(), { ...style, roman: true }, out); continue; }
    const c = src[i];
    if (c === '_' || c === '^') { i++; parse(group(), { ...style, level: c === '_' ? -1 : 1 }, out); continue; }
    const letter = /[A-Za-z]/.test(c);
    out.push({ t: c, italic: letter && !style.bold && !style.roman, bold: !!style.bold, level: style.level || 0 });
    i++;
  }
  return out;
}

function rich(x, y, src, o) {
  o = o || {};
  const fs = o.fs || 17;
  const segs = parse(src, {}, []);
  // merge runs with identical style
  const runs = [];
  for (const s of segs) {
    const last = runs[runs.length - 1];
    if (last && last.italic === s.italic && last.bold === s.bold && last.level === s.level) last.t += s.t;
    else runs.push({ ...s });
  }
  const shift = (lv) => (lv < 0 ? 0.32 * fs : lv > 0 ? -0.42 * fs : 0);
  let prev = 0, body = '';
  for (const r of runs) {
    const dy = shift(r.level) - prev; prev = shift(r.level);
    const attrs = [
      dy ? `dy="${dy.toFixed(1)}"` : '',
      r.italic ? 'font-style="italic"' : '',
      r.bold ? 'font-weight="700"' : '',
      r.level ? `font-size="${(0.7 * fs).toFixed(1)}"` : '',
    ].filter(Boolean).join(' ');
    body += `<tspan ${attrs}>${esc(r.t)}</tspan>`;
  }
  // xml:space keeps the spaces at tspan boundaries (librsvg drops them otherwise)
  return `<text xml:space="preserve" x="${x}" y="${baselineY(y, fs, false)}" font-family="${FS}" font-size="${fs}" font-weight="${o.fw || 500}" fill="${o.fill || C.ink}" text-anchor="${o.anchor || 'middle'}">${body}</text>`;
}

const mono = (x, y, t, o) => tx(x, y, t, { mono: true, fs: 14, fw: 500, ...(o || {}) });

// ── the recorded graph of the line model's loss ─────────────────────
// One node per value, one edge per recorded operation. Each node shows its
// value (ink) and the gradient of L with respect to it (orange).
function computeGraph() {
  const W = 704, H = 190; let o = '';
  const bw = 90, bh = 92, gap = 58, x0 = 8, top = 50;
  const xs = [0, 1, 2, 3, 4].map(k => x0 + k * (bw + gap));
  const nodes = [
    ['\\b{w}', '(1, 0)', '(2, 4)'],
    ['\\b{Xw}', '(1, 1, 1)', '(0, 0, 2)'],
    ['\\b{r}', '(0, 0, 1)', '(0, 0, 2)'],
    ['\\b{r}⊙\\b{r}', '(0, 0, 1)', '(1, 1, 1)'],
    ['L', '1', '1'],
  ];
  const fwdOps = ['\\b{X}·', '−\\b{y}', '(·)^{2}', '\\r{sum}'];
  const bwdOps = ['\\b{X}^{\\r{T}}·', '×1', '⊙2\\b{r}', '\\r{copy}'];
  const yName = top + 20, yVal = top + 47, yGrad = top + 72;
  const yF = top + 33, yB = top + 62;
  // forward (blue, above) and backward (orange, below) edges with labels
  for (let k = 0; k < 4; k++) {
    const a = xs[k] + bw, b = xs[k + 1];
    o += arrow(a + 4, yF, b - 4, yF, C.blue);
    o += rich((a + b) / 2, yF - 17, fwdOps[k], { fs: 15, fill: BLUE_DARK });
    o += arrow(b - 4, yB, a + 4, yB, RED.base);
    o += rich((a + b) / 2, yB + 17, bwdOps[k], { fs: 15, fill: RED.dark });
  }
  nodes.forEach(([name, val, grad], k) => {
    const x = xs[k];
    o += box(x, top, bw, bh, k === 0 ? SLATE.tint : '#fff', C.gray, false, 8, 2);
    o += rich(x + bw / 2, yName, name, { fs: 18 });
    o += line(x + 10, top + 33, x + bw - 10, top + 33, HAIR, 1.5);
    o += mono(x + bw / 2, yVal, val, { fill: C.ink });
    o += mono(x + bw / 2, yGrad, grad, { fill: RED.dark, fw: 700 });
  });
  o += tx(x0, top - 26, '▸ forward pass: values', { fs: 15, fw: 700, fill: BLUE_DARK, anchor: 'start' });
  o += rich(xs[4] + bw, top + bh + 26, '\\r{◂ backward pass: gradients of }L', { fs: 15, fw: 700, fill: RED.dark, anchor: 'end' });
  return svg(W, H, o);
}

// ── detach: the same product with u attached and detached ───────────
// Forward edges are ink; an orange arrow runs backward along every edge that
// carries gradient, labelled with the local derivative it multiplies by.
function chipAt(cx, cy, t, dashed = false) {
  const s = 44;
  return box(cx - s / 2, cy - s / 2, s, s, dashed ? '#fff' : SLATE.tint, C.gray, dashed, 6, 2)
    + rich(cx, cy, t, { fs: 21, fill: C.ink });
}

// a forward edge between two chip centres, trimmed to the chip borders, and
// (optionally) a parallel backward arrow offset to one side with a label
function edge(p, q, opts) {
  const r = 27, dx = q[0] - p[0], dy = q[1] - p[1], L = Math.hypot(dx, dy);
  const ux = dx / L, uy = dy / L, nx = -uy, ny = ux;
  const a = [p[0] + ux * r, p[1] + uy * r], b = [q[0] - ux * r, q[1] - uy * r];
  let o = arrow(a[0], a[1], b[0], b[1], C.blue);
  if (opts && opts.back) {
    const off = opts.side * 13, s = opts.side;
    const a2 = [a[0] + nx * off + ux * 8, a[1] + ny * off + uy * 8];
    const b2 = [b[0] + nx * off - ux * 8, b[1] + ny * off - uy * 8];
    o += arrow(b2[0], b2[1], a2[0], a2[1], RED.base);
    const m = [(a[0] + b[0]) / 2 + nx * (off + s * (opts.lab || 20)), (a[1] + b[1]) / 2 + ny * (off + s * (opts.lab || 20))];
    o += rich(m[0], m[1], opts.back, { fs: 17, fill: RED.dark });
  }
  return o;
}

function detachGraph() {
  const W = 712, H = 282; let o = '';
  // (a) attached: x -> u -> z and x -> z
  {
    const X = [44, 186], U = [170, 70], Z = [296, 186];
    o += tx(10, 18, '(a)', { fs: 16, fw: 700, anchor: 'start' });
    o += edge(X, U, { back: '2x', side: 1, lab: 18 });
    o += edge(U, Z, { back: 'x', side: 1, lab: 16 });
    o += edge(X, Z, { back: 'u', side: 1, lab: 16 });
    o += chipAt(...X, 'x'); o += chipAt(...U, 'u'); o += chipAt(...Z, 'z');
    o += rich(U[0], U[1] - 38, 'u = x^{2}', { fs: 16 });
    o += rich(Z[0], Z[1] + 40, 'z = ux', { fs: 16 });
    o += rich(170, 262, '∂z/∂x = 2x · x + u = 3x^{2}', { fs: 17, fw: 700, fill: RED.dark });
  }
  // (b) detached: c is a copy of u without history; only x -> z carries gradient
  {
    const ox = 380;
    const X = [ox + 44, 186], U = [ox + 112, 70], Cc = [ox + 232, 70], Z = [ox + 296, 186];
    o += tx(ox - 6, 18, '(b)', { fs: 16, fw: 700, anchor: 'start' });
    o += edge(X, U);
    // the detach: value copied, edge cut
    o += line(U[0] + 27, U[1], Cc[0] - 27, U[1], C.gray, 2, true);
    const mx = (U[0] + Cc[0]) / 2;
    o += line(mx - 7, U[1] - 9, mx + 7, U[1] + 9, C.ink, 2.6);
    o += line(mx - 7, U[1] + 9, mx + 7, U[1] - 9, C.ink, 2.6);
    o += tx(mx, U[1] + 24, 'detach', { mono: true, fs: 14, fw: 700, fill: C.ink });
    o += edge(Cc, Z);
    o += edge(X, Z, { back: 'c', side: 1, lab: 16 });
    o += chipAt(...X, 'x'); o += chipAt(...U, 'u'); o += chipAt(...Cc, 'c', true);
    o += chipAt(...Z, 'z');
    o += rich(U[0], U[1] - 38, 'u = x^{2}', { fs: 16 });
    o += rich(Cc[0], Cc[1] - 38, 'c = u\\r{, constant}', { fs: 16 });
    o += rich(Z[0], Z[1] + 40, 'z = cx', { fs: 16 });
    o += rich(ox + 170, 262, '∂z/∂x = c = x^{2}', { fs: 17, fw: 700, fill: RED.dark });
  }
  return svg(W, H, o);
}

// ── forward mode fills columns, reverse mode fills rows ─────────────
function fwdVsRev() {
  const W = 712, H = 268; let o = '';
  const m = 3, n = 5, s = 30, g = 5;
  const gw = n * s + (n - 1) * g, gh = m * s + (m - 1) * g;
  const cell = (x, y, fill, stroke, w = 2, dash = false) => box(x, y, s, s, fill, stroke, dash, 5, w);
  const panel = (ox, oy, mode) => {
    let p = '';
    for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) {
      const x = ox + c * (s + g), y = oy + r * (s + g);
      if (mode === 'fwd') {
        if (c < 2) p += cell(x, y, TOKENS.accents.blue.tint, C.blue, 1.5);
        else if (c === 2) p += cell(x, y, TOKENS.accents.blue.tint, C.blue, 3);
        else p += cell(x, y, '#fff', FAINT, 1.5, true);
      } else {
        if (r < 1) p += cell(x, y, RED.tint, RED.base, 1.5);
        else if (r === 1) p += cell(x, y, RED.tint, RED.base, 3);
        else p += cell(x, y, '#fff', FAINT, 1.5, true);
      }
    }
    // axis annotations: n inputs across, m outputs down
    p += rich(ox + gw / 2, oy + gh + 22, 'n\\r{ inputs}', { fs: 16, fill: C.ink });
    p += rich(ox - 14, oy + gh / 2, 'm\\r{ outputs}', { fs: 16, fill: C.ink, anchor: 'end' });
    return p;
  };
  // (a) forward: one column per sweep
  const ax = 104, ay = 74;
  o += tx(8, 18, '(a)', { fs: 16, fw: 700, anchor: 'start' });
  o += panel(ax, ay, 'fwd');
  const colX = ax + 2 * (s + g) + s / 2;
  o += arrow(colX, ay - 40, colX, ay - 8, C.blue);
  o += rich(colX + 12, ay - 40, '\\b{Je}_{3}', { fs: 17, fill: BLUE_DARK, anchor: 'start' });
  o += tx(ax + gw / 2, ay + gh + 58, 'forward mode: one column per sweep', { fs: 16, fw: 700, fill: BLUE_DARK });
  // (b) reverse: one row per sweep
  const bx = 470, by = 74;
  o += tx(352, 18, '(b)', { fs: 16, fw: 700, anchor: 'start' });
  o += panel(bx, by, 'rev');
  const rowY = by + 1 * (s + g) + s / 2;
  o += arrow(bx + gw + 42, rowY, bx + gw + 8, rowY, RED.base);
  o += rich(bx + gw + 25, rowY - 24, '\\b{J}^{\\r{T}}\\b{e}_{2}', { fs: 17, fill: RED.dark });
  o += tx(bx + gw / 2, by + gh + 58, 'reverse mode: one row per sweep', { fs: 16, fw: 700, fill: RED.dark });
  return svg(W, H, o);
}

// ── control flow: each input records its own graph ──────────────────
function dynamic() {
  const W = 712, H = 184; let o = '';
  const row = (y, aLabel, n, branch, slope) => {
    let s = '', x = 8;
    s += box(x, y - 19, 98, 38, SLATE.tint, C.gray, false, 8, 2);
    s += rich(x + 49, y, aLabel, { fs: 16, fill: C.ink });
    x += 98;
    const doubles = ['×2', '×2', '⋯', '×2'];
    const x1 = x + 16;
    doubles.forEach((d, i) => {
      s += arrow(x + 3, y, x + 15, y, C.blue);
      x += 16;
      if (d === '⋯') { s += tx(x + 17, y, '⋯', { fs: 18, fw: 700, fill: C.ink }); }
      else {
        s += box(x, y - 16, 34, 32, TOKENS.accents.blue.tint, C.blue, false, 6, 2);
        s += tx(x + 17, y, d, { mono: true, fs: 14, fw: 700, fill: C.ink });
      }
      x += 34;
    });
    // brace-like bracket under the doublings
    const x2 = x;
    s += `<path d="M${x1},${y + 24} L${x1},${y + 30} L${x2},${y + 30} L${x2},${y + 24}" fill="none" stroke="${C.gray}" stroke-width="1.5"/>`;
    s += tx((x1 + x2) / 2, y + 44, `${n} doublings`, { fs: 14, fw: 600, fill: C.ink });
    s += arrow(x + 3, y, x + 19, y, C.blue);
    x += 20;
    s += box(x, y - 19, 150, 38, '#fff', C.gray, false, 8, 2);
    s += tx(x + 75, y, branch, { mono: true, fs: 14, fw: 700, fill: C.ink });
    x += 150;
    s += arrow(x + 3, y, x + 19, y, C.blue);
    x += 20;
    s += rich(x + 4, y, slope, { fs: 17, fw: 600, anchor: 'start' });
    return s;
  };
  o += row(26, 'a = 0.3', 12, 'sum > 0: c = b', '\\r{slope} 2^{12} = 4096');
  o += row(122, 'a = −0.75', 11, 'else: c = 100·b', '\\r{slope} 100 · 2^{11} = 204,800');
  return svg(W, H, o);
}

export const diagrams = {
  'autograd-comp-graph': computeGraph,
  'autograd-detach':     detachGraph,
  'autograd-fwd-vs-rev': fwdVsRev,
  'autograd-dynamic':    dynamic,
};
