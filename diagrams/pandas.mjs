// Diagrams for chapter_preliminaries/pandas (§2.2 Data Preprocessing).
//
// The section is mostly tabular data, so the diagrams carry the
// *decisions*: the load→clean→tensor pipeline, the three responses to a
// missing value, one-hot encoding, and why features get standardized.
//
// Stable ids use the `pandas-<concept>` prefix. Never rename.

import { C, grid, tx, arrow, rc, svg } from './engine.mjs';

const line = (x1, y1, x2, y2, color, w = 1.5, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
const dot = (x, y, r, fill) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}"/>`;
// a non-square labelled cell (for text labels grid() can't size)
const cell = (x, y, w, h, fill, st, text, tc, fs = 12) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="${fill}" stroke="${st}" stroke-width="2"/>`
  + tx(x + w / 2, y + h / 2, text, { fs, fw: 600, fill: tc || C.ink });
const box = (cx, cy, w, h, fill, st, label, sub) => {
  let o = `<rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="10" fill="${fill}" stroke="${st}" stroke-width="2"/>`;
  o += tx(cx, cy - (sub ? 7 : 0), label, { fs: 14, fw: 700, fill: C.ink });
  if (sub) o += tx(cx, cy + 13, sub, { fs: 10.5, fw: 500, fill: C.muted });
  return o;
};

// ── the preprocessing pipeline: file → DataFrame → clean → tensor ───
function pipeline() {
  const W = 720, H = 168; let o = ''; const cy = 92, bh = 58;
  o += tx(W / 2, 28, 'Preprocessing: from file to tensor', { fs: 15, fw: 700, fill: C.ink });
  const n = [
    { cx: 70,  w: 96,  fill: C.lgray,  st: C.gray,  l: 'house.csv',      s: 'raw file' },
    { cx: 232, w: 112, fill: C.lblue,  st: C.blue,  l: 'DataFrame',      s: 'rows × columns' },
    { cx: 430, w: 150, fill: C.lamber, st: C.amber, l: 'clean &amp; encode', s: 'impute · one-hot · scale' },
    { cx: 628, w: 104, fill: C.lgreen, st: C.green, l: 'X, y',           s: 'tensors' },
  ];
  const gapmid = (a, b) => (n[a].cx + n[a].w / 2 + n[b].cx - n[b].w / 2) / 2;
  o += arrow(n[0].cx + n[0].w / 2, cy, n[1].cx - n[1].w / 2, cy, C.ink);
  o += tx(gapmid(0, 1), cy - 13, 'read_csv', { mono: true, fs: 10.5, fw: 700, fill: C.ink });
  o += arrow(n[1].cx + n[1].w / 2, cy, n[2].cx - n[2].w / 2, cy, C.ink);
  o += arrow(n[2].cx + n[2].w / 2, cy, n[3].cx - n[3].w / 2, cy, C.ink);
  o += tx(gapmid(2, 3), cy - 13, 'to_numpy', { mono: true, fs: 10.5, fw: 700, fill: C.ink });
  n.forEach(b => o += box(b.cx, cy, b.w, bh, b.fill, b.st, b.l, b.s));
  return svg(W, H, o);
}

// ── three responses to a missing value ──────────────────────────────
function missing() {
  const W = 580, H = 374; let o = ''; const s = 26, g = 4;
  const blue = () => [C.lblue, C.blue, false, C.ink];
  o += tx(W / 2, 26, 'Three ways to handle a missing value', { fs: 15, fw: 700, fill: C.ink });
  // the shared "before" column with one gap
  const tcx = W / 2, ty = 50;
  o += tx(tcx, ty - 11, 'one column, one gap', { fs: 11, fill: C.muted });
  o += grid([['3'], ['?'], ['2'], ['4']], tcx - s / 2, ty, s, g,
    { fill: (r, c, v) => v === '?' ? [C.lamber, C.amber, false, C.amber] : blue() });
  const topBot = ty + 4 * (s + g) - g;
  // three results
  const ry = 198, ax = 74, bx = 268, cx = 446;
  // A — deletion (row dropped)
  o += grid([['3'], ['2'], ['4']], ax, ry, s, g, { fill: blue });
  // B — imputation (filled with mean)
  o += grid([['3'], ['3'], ['2'], ['4']], bx, ry, s, g,
    { fill: (r) => r === 1 ? [C.lgreen, C.green, false, C.green] : blue() });
  // C — indicator (value + missing flag)
  o += grid([['3', '0'], ['?', '1'], ['2', '0'], ['4', '0']], cx, ry, s, g,
    { fill: (r, c, v) => c === 1 ? ['#F3E5F5', C.purple, false, C.purple]
      : (v === '?' ? [C.lamber, C.amber, false, C.amber] : blue()) });
  // fan arrows
  o += arrow(tcx - 30, topBot, ax + s / 2, ry - 6, C.gray);
  o += arrow(tcx, topBot, bx + s / 2, ry - 6, C.gray);
  o += arrow(tcx + 30, topBot, cx + s, ry - 6, C.gray);
  // labels under each result
  o += tx(ax + s / 2, ry + 3 * (s + g) + 14, 'deletion', { fs: 12, fw: 700, fill: C.ink });
  o += tx(ax + s / 2, ry + 3 * (s + g) + 30, 'loses rows', { fs: 10, fill: C.muted });
  o += tx(bx + s / 2, ry + 4 * (s + g) + 14, 'imputation', { fs: 12, fw: 700, fill: C.ink });
  o += tx(bx + s / 2, ry + 4 * (s + g) + 30, 'fill with mean', { fs: 10, fill: C.muted });
  o += tx(cx + s + g / 2, ry + 4 * (s + g) + 14, 'indicator', { fs: 12, fw: 700, fill: C.ink });
  o += tx(cx + s + g / 2, ry + 4 * (s + g) + 30, '+ "was-missing" flag', { fs: 10, fill: C.muted });
  return svg(W, H, o);
}

// ── one-hot encoding a categorical column ───────────────────────────
function onehot() {
  const W = 560, H = 234; let o = ''; const s = 30, g = 5;
  o += tx(W / 2, 26, 'One-hot encoding a categorical column', { fs: 15, fw: 700, fill: C.ink });
  // left: the RoofType column (text cells → drawn manually)
  const lw = 70, lh = s, lx = 54, ly = 74;
  const cats = ['Slate', 'Tile', '—', 'Slate'];
  o += tx(lx + lw / 2, ly - 12, 'RoofType', { fs: 11.5, fw: 700, fill: C.muted });
  cats.forEach((v, i) => {
    const y = ly + i * (lh + g);
    const [f, st, tc] = i === 0 ? [C.lgreen, C.green, C.ink]
      : v === '—' ? [C.lamber, C.amber, C.amber] : [C.lblue, C.blue, C.ink];
    o += cell(lx, y, lw, lh, f, st, v, tc);
  });
  // arrow
  o += arrow(lx + lw + 8, ly + 2 * (lh + g) - g / 2, lx + lw + 56, ly + 2 * (lh + g) - g / 2, C.purple);
  o += tx(lx + lw + 32, ly + 2 * (lh + g) - g / 2 - 12, 'get_dummies', { mono: true, fs: 10, fw: 700, fill: C.purple });
  // right: the 0/1 columns
  const rx = lx + lw + 70, ry = ly;
  const heads = ['Slate', 'Tile', 'nan'];
  heads.forEach((h, c) => o += tx(rx + c * (s + g) + s / 2, ry - 12, h, { fs: 10.5, fw: 700, fill: C.muted }));
  const oh = [['1', '0', '0'], ['0', '1', '0'], ['0', '0', '1'], ['1', '0', '0']];
  o += grid(oh, rx, ry, s, g, {
    fill: (r, c, v) => r === 0 ? [C.lgreen, C.green, false, C.ink]
      : (v === '1' ? [C.lblue, C.blue, false, C.ink] : [C.lgray, '#CBD5DC', false, '#90A4AE'])
  });
  o += tx(W / 2, H - 14, "each category — and “missing” — becomes its own 0/1 column", { fs: 12, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── why standardize: raw scales are incomparable ────────────────────
function scale() {
  const W = 540, H = 226; let o = '';
  o += tx(W / 2, 26, 'Why standardize numerical features', { fs: 15, fw: 700, fill: C.ink });
  // raw axis
  const x0 = 150, x1 = 500, yr = 80;
  const rawX = v => x0 + (v / 2700) * (x1 - x0);
  o += line(x0, yr, x1, yr, C.gray, 1.4);
  o += tx(70, yr, 'raw', { fs: 12, fw: 700, fill: C.ink, anchor: 'start' });
  // NumRooms 2..5 (a tiny sliver near the origin)
  o += line(rawX(2), yr, rawX(5), yr, C.blue, 5);
  o += dot(rawX(3.3), yr, 3, C.blue);
  o += tx(rawX(3.5), yr - 12, 'NumRooms 2–5', { fs: 10, fw: 600, fill: C.blue, anchor: 'start' });
  // Area 850..2600 (dominates)
  o += line(rawX(850), yr + 14, rawX(2600), yr + 14, C.amber, 5);
  o += tx(rawX(1700), yr + 26, 'Area 850–2600', { fs: 10, fw: 600, fill: C.amber });
  o += tx(W / 2, yr + 44, '↓  subtract mean, divide by std', { fs: 11, fw: 600, fill: C.muted });
  // standardized axis (-2..2 centered)
  const cx = W / 2, ys = 168, half = 150;
  const stdX = z => cx + (z / 2) * half;
  o += line(cx - half, ys, cx + half, ys, C.gray, 1.4);
  o += tx(70, ys, 'standardized', { fs: 12, fw: 700, fill: C.ink, anchor: 'start' });
  o += dot(cx, ys, 3, C.ink);
  o += tx(cx, ys + 16, '0', { fs: 10, fill: C.muted });
  o += line(stdX(-1.4), ys - 8, stdX(1.4), ys - 8, C.blue, 5);
  o += line(stdX(-1.5), ys + 8, stdX(1.5), ys + 8, C.amber, 5);
  o += tx(W / 2, H - 12, 'both features now share one scale (≈ zero mean, unit variance)', { fs: 11.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

export const diagrams = {
  'pandas-pipeline': pipeline,
  'pandas-missing':  missing,
  'pandas-onehot':   onehot,
  'pandas-scale':    scale,
};
