// Diagrams for chapter_preliminaries/pandas (Data Preprocessing).
//
// The section is mostly tabular output, so the diagrams carry the
// decisions: the file -> DataFrame -> tensors pipeline (also the book
// figure fig_pandas_pipeline), the three responses to a missing value,
// one-hot encoding, and what standardization does to the two numerical
// columns of the running example (house_tiny.csv).
//
// Sizing follows the chapter's other modules (figure-style guide §5-6):
// canvases at most 720 units wide, labels 15-17 px, nothing below 14.5 px.
// Structural text is INK; colored text uses the accent's dark shade.
// Labels name things; sentences belong in the caption or the slide text.
//
// Stable ids use the `pandas-<concept>` prefix. Never rename.

import { C, TOKENS, grid, tx, arrow, svg } from './engine.mjs';

const D = TOKENS.accents;                     // tint / base / dark trios
const N = TOKENS.neutrals;
const BLUE = [C.lblue, C.blue, false, C.ink];
const GREEN = [C.lgreen, C.green, false, C.ink];
const AMBER = [C.lamber, C.amber, false, D.orange.dark];
const PURPLE = [D.purple.tint, C.purple, false, D.purple.dark];
const GRAY = [C.lgray, N.hairline, false, C.muted];

const code = (x, y, t, fill = C.ink, o = {}) =>
  tx(x, y, t, { mono: true, fs: o.fs || 15, fw: o.fw || 600, fill, anchor: o.anchor || 'middle' });
const name = (x, y, t, fill = C.ink, o = {}) =>
  tx(x, y, t, { fs: o.fs || 17, fw: o.fw || 700, fill, anchor: o.anchor || 'middle' });
const note = (x, y, t, fill = C.ink, o = {}) =>
  tx(x, y, t, { fs: o.fs || 15, fw: o.fw || 500, fill, anchor: o.anchor || 'middle' });
const line = (x1, y1, x2, y2, color, w = 2) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"/>`;
// a rounded rectangle cell with a centered text label (for words, which
// grid()'s square cells cannot hold)
const wcell = (x, y, w, h, [fill, stroke, , tc], t) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${fill}" stroke="${stroke}" stroke-width="2"/>`
  + code(x + w / 2, y + h / 2, t, tc, { fw: 500 });

// ── the preprocessing pipeline: CSV file -> DataFrame -> tensors ───────
// States are boxes; the operations that connect them label the arrows.
function pipeline() {
  const cy = 72, bh = 60; let o = '';
  const n = [
    { w: 96, fill: C.lgray, st: C.gray, l: 'CSV file', s: 'text' },
    { w: 116, fill: C.lblue, st: C.blue, l: 'DataFrame', s: 'gaps, words' },
    { w: 116, fill: C.lblue, st: C.blue, l: 'DataFrame', s: 'numbers only' },
    { w: 96, fill: C.lgreen, st: C.green, l: 'tensors', s: 'X, y' },
  ];
  const gaps = [82, 108, 82];
  let x = 8;
  n.forEach((b, i) => { b.x = x; x += b.w + (gaps[i] || 0); });
  const W = x + 8;
  for (let i = 0; i < 3; i++) {
    const x1 = n[i].x + n[i].w, x2 = n[i + 1].x;
    o += arrow(x1 + 6, cy, x2 - 6, cy, C.ink);
    const mx = (x1 + x2) / 2;
    if (i === 1) {
      o += note(mx, cy - 40, 'impute,', C.ink, { fs: 14.5, fw: 600 });
      o += note(mx, cy - 21, 'encode, scale', C.ink, { fs: 14.5, fw: 600 });
    } else {
      o += code(mx, cy - 18, i === 0 ? 'read_csv' : 'to_numpy', C.ink, { fs: 14.5, fw: 700 });
    }
  }
  n.forEach(b => {
    o += `<rect x="${b.x}" y="${cy - bh / 2}" width="${b.w}" height="${bh}" rx="8" fill="${b.fill}" stroke="${b.st}" stroke-width="2"/>`;
    o += name(b.x + b.w / 2, cy - 11, b.l);
    o += note(b.x + b.w / 2, cy + 13, b.s, C.ink, { fs: 14.5 });
  });
  return svg(W, cy + bh / 2 + 10, o);
}

// ── three responses to a missing value ────────────────────────────────
// The column holds 3, NaN, 2, 4; its mean is 3.
function missing() {
  const s = 40, g = 4, step = s + g; let o = '';
  const W = 540, tcx = W / 2, ty = 36;
  const src = [['3'], ['NaN'], ['2'], ['4']];
  o += code(tcx, ty - 20, 'NumRooms', C.ink, { fw: 700 });
  o += grid(src, tcx - s / 2, ty, s, g, { fs: 14.5, fill: (r, c, v) => v === 'NaN' ? AMBER : BLUE });
  const srcBot = ty + 4 * step - g;
  const ry = srcBot + 70;
  const ax = 40, bx = W / 2 - s / 2, cx = W - 40 - (2 * s + g);
  // fan arrows from the source column to the three results
  o += arrow(tcx - 26, srcBot + 8, ax + s / 2 + 8, ry - 10, C.gray);
  o += arrow(tcx, srcBot + 8, tcx, ry - 10, C.gray);
  o += arrow(tcx + 26, srcBot + 8, cx + s - 8, ry - 10, C.gray);
  // deletion: the row with the gap is gone
  o += grid([['3'], ['2'], ['4']], ax, ry, s, g, { fs: 14.5, fill: () => BLUE });
  // imputation: the gap holds the mean
  o += grid([['3'], ['3'], ['2'], ['4']], bx, ry, s, g, { fs: 14.5,
    fill: (r) => r === 1 ? GREEN : BLUE });
  // indicator: imputed value plus a column that flags the gap
  o += grid([['3', '0'], ['3', '1'], ['2', '0'], ['4', '0']], cx, ry, s, g, { fs: 14.5,
    fill: (r, c) => c === 1 ? PURPLE : (r === 1 ? GREEN : BLUE) });
  // labels on one baseline, below the tallest result
  const lab = (x, t, sub) =>
    name(x, ry + 4 * step + 14, t) + note(x, ry + 4 * step + 36, sub);
  o += lab(ax + s / 2, 'deletion', 'drops the row');
  o += lab(bx + s / 2, 'imputation', 'fills in the mean');
  o += lab(cx + s + g / 2, 'indicator', 'mean and a flag');
  return svg(W, ry + 4 * step + 50, o);
}

// ── one-hot encoding of the first four RoofType entries ───────────────
// Each category keeps one accent: the word on the left and the 1 in its
// column on the right share a color; zeros are gray.
function onehot() {
  const s = 42, g = 10, step = s + g; let o = '';
  const cats = ['Slate', 'Tile', 'NaN', 'Slate'];
  const col = { Slate: BLUE, Tile: GREEN, NaN: AMBER };
  const lx = 8, lw = 80, top = 34;
  o += code(lx + lw / 2, top - 20, 'RoofType', C.ink, { fw: 700 });
  cats.forEach((v, i) => o += wcell(lx, top + i * step, lw, s, col[v], v));
  // arrow with its label above, clear of the cells
  const ax1 = lx + lw + 12, ax2 = ax1 + 112, ay = top + 2 * step - g / 2;
  o += arrow(ax1, ay, ax2, ay, C.purple);
  o += code((ax1 + ax2) / 2, ay - 18, 'get_dummies', D.purple.dark, { fs: 14.5, fw: 700 });
  // the three 0/1 columns
  const rx = ax2 + 12;
  const heads = ['Slate', 'Tile', 'nan'];
  heads.forEach((h, c) => o += code(rx + c * step + s / 2, top - 20, h, C.ink, { fw: 700 }));
  const oh = cats.map(v => heads.map(h => (h === v || (h === 'nan' && v === 'NaN')) ? '1' : '0'));
  o += grid(oh, rx, top, s, g, { fs: 15,
    fill: (r, c, v) => v === '1' ? col[cats[r]] : GRAY });
  return svg(rx + 3 * step - g + 8, top + 4 * step - g + 8, o);
}

// ── standardization puts NumRooms and Area on one scale ───────────────
// Ranges from the section's data: raw NumRooms 2-5, Area 850-2600;
// standardized (after mean imputation) NumRooms -1.42..1.89,
// Area -1.44..1.70.
function scale() {
  const x0 = 130, x1 = 530, W = 556; let o = '';
  const row = (y, label) => code(x0 - 14, y, label, C.ink, { anchor: 'end' });
  // a range as a rounded bar; a range too short to see becomes a dot
  const bar = (xa, xb, y, color) => xb - xa < 10
    ? `<circle cx="${(xa + xb) / 2 + 5}" cy="${y}" r="5" fill="${color}"/>`
    : `<rect x="${xa}" y="${y - 5}" width="${xb - xa}" height="10" rx="5" fill="${color}"/>`;
  const axis = (y, ticks, map) => {
    let a = line(x0, y, x1, y, C.ink, 2);
    ticks.forEach(t => {
      const x = map(t);
      a += line(x, y, x, y + 6, C.ink, 2);
      a += note(x, y + 20, t < 0 ? `−${-t}` : t.toLocaleString('en-US'), C.ink, { fs: 14.5 });
    });
    return a;
  };
  // panel 1: raw units, 0..2700
  const raw = v => x0 + (v / 2700) * (x1 - x0);
  let y = 22;
  o += name(8, y, 'raw', C.ink, { anchor: 'start' });
  o += row(y + 30, 'NumRooms') + bar(raw(2), raw(5), y + 30, C.blue);
  o += row(y + 58, 'Area') + bar(raw(850), raw(2600), y + 58, C.amber);
  o += axis(y + 80, [0, 1000, 2000], raw);
  // panel 2: standardized, -2..2
  const std = z => x0 + ((z + 2) / 4) * (x1 - x0);
  y = 152;
  o += name(8, y, 'standardized', C.ink, { anchor: 'start' });
  o += row(y + 30, 'NumRooms') + bar(std(-1.42), std(1.89), y + 30, C.blue);
  o += row(y + 58, 'Area') + bar(std(-1.44), std(1.70), y + 58, C.amber);
  o += axis(y + 80, [-2, -1, 0, 1, 2], std);
  return svg(W, y + 110, o);
}

export const diagrams = {
  'pandas-pipeline': pipeline,
  'pandas-missing':  missing,
  'pandas-onehot':   onehot,
  'pandas-scale':    scale,
};
