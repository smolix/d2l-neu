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
// The column holds 3, NaN, 2, 4; its mean is 3.  Slide-only, shown in a
// figure column at ~0.9x, so values and labels are 16-18 px.
function missing() {
  // The first four rows of NumRooms; the imputed value is the mean of all
  // seven observed room counts, 23/7 = 3.29 (as in the section's output).
  const w = 60, h = 36, g = 4, step = h + g, fs = 17; let o = '';
  const cell = (x, y, [f, st, , tc], t) =>
    `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${f}" stroke="${st}" stroke-width="2"/>`
    + code(x + w / 2, y + h / 2, t, tc, { fs, fw: 500 });
  const col = (x, y, vals, fill) => vals.map((v, r) => cell(x, y + r * step, fill(r), v)).join('');
  const W = 560, tcx = W / 2, ty = 38;
  o += code(tcx, ty - 22, 'NumRooms', C.ink, { fs, fw: 700 });
  o += col(tcx - w / 2, ty, ['3', 'NaN', '2', '4'], (r) => r === 1 ? AMBER : BLUE);
  const srcBot = ty + 4 * step - g;
  const ry = srcBot + 58;
  const ax = 40, bx = W / 2 - w / 2, cx = W - 40 - (2 * w + g);
  // fan arrows from the source column to the three results
  o += arrow(tcx - 30, srcBot + 8, ax + w / 2 + 8, ry - 10, C.gray);
  o += arrow(tcx, srcBot + 8, tcx, ry - 10, C.gray);
  o += arrow(tcx + 30, srcBot + 8, cx + w - 8, ry - 10, C.gray);
  // deletion: the row with the gap is gone
  o += col(ax, ry, ['3', '2', '4'], () => BLUE);
  // imputation: the gap holds the mean of the observed values
  o += col(bx, ry, ['3', '3.29', '2', '4'], (r) => r === 1 ? GREEN : BLUE);
  // indicator: imputed value plus a column that flags the gap
  o += col(cx, ry, ['3', '3.29', '2', '4'], (r) => r === 1 ? GREEN : BLUE);
  o += col(cx + w + g, ry, ['0', '1', '0', '0'], () => PURPLE);
  // labels on one baseline, below the tallest result
  const lab = (x, t, sub) =>
    name(x, ry + 4 * step + 16, t, C.ink, { fs: 18 })
    + note(x, ry + 4 * step + 40, sub, C.ink, { fs: 16 });
  o += lab(ax + w / 2, 'deletion', 'drops the row');
  o += lab(bx + w / 2, 'imputation', 'fills in the mean');
  o += lab(cx + w + g / 2, 'indicator', 'mean and a flag');
  return svg(W, ry + 4 * step + 54, o);
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

// ── separating input features X from target labels y by column name ───
function xySplit() {
  const W = 540, H = 260; let o = '';
  const heads = ['NumRooms', 'RoofType', 'Area', 'Price'];
  const rows = [
    ['3.0', 'Slate', '1500', '210000'],
    ['NaN', 'Tile',  '2100', '290000'],
    ['2.0', 'NaN',   '850',  '127500'],
  ];
  const cw = 88, ch = 32, g = 4;
  // Left: X = data.drop(columns=['Price'])
  o += code(152, 20, "inputs = data.drop(columns='Price')", D.blue.dark, { fs: 13, fw: 700 });
  for (let c = 0; c < 3; c++) {
    const x = 16 + c * (cw + g);
    o += wcell(x, 38, cw, ch, BLUE, heads[c]);
    for (let r = 0; r < 3; r++) {
      o += wcell(x, 38 + (r + 1) * (ch + g), cw, ch, [ '#FFFFFF', C.blue, false, C.ink ], rows[r][c]);
    }
  }
  o += name(152, 202, 'Feature Matrix X (n × 3)', D.blue.dark, { fs: 16 });
  o += note(152, 224, 'safe against column reordering', C.muted, { fs: 13.5 });

  // Arrow / gap to y = data['Price']
  o += code(444, 20, "targets = data['Price']", D.green.dark, { fs: 13, fw: 700 });
  const tx0 = 400;
  o += wcell(tx0, 38, 96, ch, GREEN, heads[3]);
  for (let r = 0; r < 3; r++) {
    o += wcell(tx0, 38 + (r + 1) * (ch + g), 96, ch, [ '#FFFFFF', C.green, false, C.ink ], rows[r][3]);
  }
  o += name(448, 202, 'Target Vector y (n,)', D.green.dark, { fs: 16 });
  o += note(448, 224, 'extracted before imputation', C.muted, { fs: 13.5 });
  return svg(W, H, o);
}

// ── mean imputation shrinks variance; indicator column preserves signal ─
function imputationShrinkage() {
  const W = 540, H = 278; let o = '';
  const x0 = 84, x1 = 492, mx = (x0 + x1) / 2;
  // Panel (a): the 7 observed NumRooms (2,2,3,3,4,4,5; mean 23/7 = 3.29)
  // vs the same 7 plus the 3 missing rows imputed at the mean
  o += name(16, 20, '(a) Plugging μ ≈ 3.29 into missing rows shrinks spread σ', D.orange.dark, { anchor: 'start', fs: 15.5 });
  const y1 = 68, y2 = 132;
  const mu = 23 / 7, k = 70;
  const vx = (v) => mx + (v - mu) * k;
  const observed = [2, 2, 3, 3, 4, 4, 5];
  const dots = (y) => {
    const seen = {};
    observed.forEach(v => {
      const j = seen[v] = (seen[v] || 0) + 1;
      o += `<circle cx="${vx(v).toFixed(1)}" cy="${y - (j - 1) * 11}" r="5.5" fill="${C.blue}"/>`;
    });
  };
  o += note(16, y1, 'observed', C.muted, { anchor: 'start', fs: 14 });
  o += line(x0, y1, x1, y1, C.ink, 2);
  dots(y1);

  o += note(16, y2, 'imputed', D.orange.dark, { anchor: 'start', fs: 14 });
  o += line(x0, y2, x1, y2, C.ink, 2);
  dots(y2);
  // Stack of 3 imputed dots piled at the mean
  for (let k2 = 0; k2 < 3; k2++) {
    o += `<circle cx="${mx}" cy="${y2 - k2 * 11}" r="5.5" fill="${C.amber}" stroke="${D.orange.dark}" stroke-width="1.5"/>`;
  }
  o += code(mx, y2 + 22, 'μ ≈ 3.29 (3 imputed rows pile here → σ shrinks!)', D.orange.dark, { fs: 13.5 });

  // Panel (b): Adding binary indicator feature NumRooms_isna
  o += `<line x1="16" y1="176" x2="524" y2="176" stroke="${N.hairline}" stroke-width="1.5"/>`;
  o += name(16, 200, '(b) Solution: pair imputed value with binary is_na indicator', D.purple.dark, { anchor: 'start', fs: 15.5 });
  o += wcell(28, 222, 156, 36, GREEN, 'NumRooms = 3.29');
  o += tx(196, 240, '+', { fs: 22, fw: 700, fill: C.ink });
  o += wcell(216, 222, 164, 36, PURPLE, 'NumRooms_isna = 1');
  o += arrow(386, 240, 406, 240, C.purple);
  o += code(412, 240, 'w₁·3.29 + w₂·1', D.purple.dark, { fs: 14.5, fw: 700, anchor: 'start' });
  return svg(W, H, o);
}

// ── mean imputation shrinks the spread (book figure, one panel) ─────────
// The seven observed NumRooms (2, 2, 3, 3, 4, 4, 5; mean 23/7 = 3.29,
// sample sd 1.11) and the ten values after mean imputation (sd 0.91).
function imputationSpread() {
  let o = '';
  const x0 = 120, x1 = 500, k = (x1 - x0) / 4;
  const vx = (v) => x0 + (v - 1.5) * k;
  const mu = 23 / 7, y1 = 64, y2 = 150, r = 6, dy = 13;
  const observed = [2, 2, 3, 3, 4, 4, 5];
  const dots = (y) => {
    const seen = {};
    observed.forEach(v => {
      const j = seen[v] = (seen[v] || 0) + 1;
      o += `<circle cx="${vx(v).toFixed(1)}" cy="${y - (j - 1) * dy}" r="${r}" fill="${C.blue}"/>`;
    });
  };
  o += note(16, y1, 'observed', C.ink, { anchor: 'start' });
  o += line(x0, y1, x1, y1, C.ink, 1.5);
  dots(y1);
  o += note(16, y2, 'imputed', C.ink, { anchor: 'start' });
  o += line(x0, y2, x1, y2, C.ink, 1.5);
  dots(y2);
  for (let j = 0; j < 3; j++)
    o += `<circle cx="${vx(mu).toFixed(1)}" cy="${y2 - j * dy}" r="${r}" fill="${C.amber}" stroke="${D.orange.dark}" stroke-width="1.5"/>`;
  o += tx(vx(mu), y2 - 2 * dy - 24, 'μ', { fs: 17, fw: 600, fill: D.orange.dark });
  [2, 3, 4, 5].forEach(v => {
    o += line(vx(v), y2 + 1, vx(v), y2 + 6, C.ink, 1.5);
    o += code(vx(v), y2 + 20, String(v), C.ink, { fw: 500 });
  });
  o += note(x1 + 14, y1, 'σ = 1.11', C.ink, { anchor: 'start', fw: 600 });
  o += note(x1 + 14, y2, 'σ = 0.91', C.ink, { anchor: 'start', fw: 600 });
  return svg(x1 + 92, y2 + 36, o);
}

// ── categorical pitfalls: fake integer order vs one-hot vs embedding ───
function categoricalPitfalls() {
  const W = 540, H = 280; let o = '';
  // 1. Bad: integer IDs imply fake order
  o += `<rect x="16" y="16" width="508" height="72" rx="8" fill="${D.red.tint}" stroke="${D.red.base}" stroke-width="2"/>`;
  o += name(28, 38, '✗ Integer IDs invent fake metric ordering:', D.red.dark, { anchor: 'start', fs: 15.5 });
  o += code(270, 66, 'Slate = 0     Tile = 1     NaN = 2   ⇒   (Slate + NaN)/2 = Tile?!', D.red.dark, { fs: 13.5 });

  // 2. Good for small K: One-hot orthogonal corners
  o += `<rect x="16" y="104" width="244" height="160" rx="8" fill="${C.lblue}" stroke="${C.blue}" stroke-width="2"/>`;
  o += name(138, 128, '✓ Small K: One-Hot {0, 1}ᴷ', D.blue.dark, { fs: 15.5 });
  [['Slate', '[1, 0, 0]'], ['Tile', '[0, 1, 0]'], ['NaN', '[0, 0, 1]']].forEach(([k, v], i) => {
    o += code(100, 156 + 24 * i, k, C.ink, { fs: 13.5, anchor: 'end' });
    o += code(112, 156 + 24 * i, '→ ' + v, C.ink, { fs: 13.5, anchor: 'start' });
  });
  o += note(138, 240, 'equidistant orthogonal axes', D.blue.dark, { fs: 13.5 });

  // 3. Good for huge K: Learned dense embedding table
  o += `<rect x="280" y="104" width="244" height="160" rx="8" fill="${D.purple.tint}" stroke="${C.purple}" stroke-width="2"/>`;
  o += name(402, 128, '✓ Large K: Embedding Rᴷˣᵈ', D.purple.dark, { fs: 15.5 });
  o += code(402, 156, 'K = 50,000 zip codes / words', C.ink, { fs: 13 });
  o += wcell(312, 174, 180, 34, [ '#FFFFFF', C.purple, false, D.purple.dark ], 'e_id = E[id] ∈ R⁶⁴');
  o += note(402, 240, 'd ≪ K learned dense features', D.purple.dark, { fs: 13.5 });
  return svg(W, H, o);
}

// ── data leakage vs fit-on-train / apply-to-test pipeline ─────────────
function leakageSplit() {
  const W = 560, H = 272; let o = '';
  const stats = (t, c) => [ '#FFFFFF', c, false, C.ink ];
  // (a) statistics fitted on all rows, then the split
  o += `<rect x="14" y="14" width="532" height="98" rx="10" fill="${D.red.tint}" stroke="${D.red.base}" stroke-width="2"/>`;
  o += name(26, 36, '(a) fit before the split', D.red.dark, { anchor: 'start', fs: 15 });
  o += wcell(26, 54, 134, 40, [ '#FFFFFF', D.red.base, false, C.ink ], 'train + test');
  o += arrow(166, 74, 206, 74, D.red.base);
  o += code(186, 60, 'fit', D.red.dark, { fs: 14.5 });
  o += wcell(212, 54, 152, 40, [ '#FFFFFF', D.red.base, false, C.ink ], 'μ_all, σ_all');
  o += arrow(370, 74, 410, 74, D.red.base);
  o += wcell(416, 54, 116, 40, [ '#FFFFFF', D.red.base, false, C.ink ], 'test score');
  // (b) split first, fit on the training split, apply to both
  o += `<rect x="14" y="128" width="532" height="130" rx="10" fill="${C.lgreen}" stroke="${C.green}" stroke-width="2"/>`;
  o += name(26, 150, '(b) fit on the training split only', D.green.dark, { anchor: 'start', fs: 15 });
  o += wcell(26, 168, 108, 36, BLUE, 'train');
  o += wcell(26, 212, 108, 36, AMBER, 'test');
  o += arrow(140, 186, 186, 186, C.blue);
  o += code(163, 172, 'fit', D.blue.dark, { fs: 14.5 });
  o += wcell(192, 168, 154, 36, stats('', C.green), 'μ_train, σ_train');
  o += arrow(352, 186, 396, 186, C.green);
  o += arrow(269, 206, 269, 228, C.green);
  o += arrow(140, 230, 396, 230, C.amber);
  o += wcell(402, 168, 130, 36, [ '#FFFFFF', C.blue, false, C.ink ], 'X_train_std');
  o += wcell(402, 212, 130, 36, [ '#FFFFFF', C.amber, false, C.ink ], 'X_test_std');
  return svg(W, H, o);
}

// ── mixed pandas DataFrame dtypes -> contiguous float32 tensor ────────
function dtypeCast() {
  const W = 560, H = 250; let o = '';
  // Left: Mixed DataFrame columns (float64 + bool -> object trap!)
  o += name(122, 22, 'Mixed DataFrame Columns', C.ink, { fs: 15.5 });
  o += wcell(14, 42, 104, 42, BLUE, 'float64');
  o += wcell(124, 42, 104, 42, AMBER, 'bool (T/F)');
  o += arrow(121, 92, 121, 132, D.red.base, true);
  o += code(129, 114, 'uncast .to_numpy()', D.red.dark, { anchor: 'start', fs: 12.5 });
  o += wcell(14, 138, 214, 44, [ D.red.tint, D.red.base, false, D.red.dark ], 'dtype=object error!');
  o += note(121, 204, '2× RAM if left as float64', D.red.dark, { fs: 13.5 });

  // Arrow across to explicit float32 cast
  o += arrow(236, 63, 344, 63, C.purple);
  o += code(290, 44, 'dtype=float32', D.purple.dark, { fs: 12.5, fw: 700 });

  // Right: Contiguous float32 Tensor X & y
  o += name(448, 22, 'Contiguous GPU Tensors', D.green.dark, { fs: 15.5 });
  o += `<rect x="352" y="42" width="194" height="140" rx="8" fill="${C.lgreen}" stroke="${C.green}" stroke-width="2"/>`;
  o += code(449, 74, 'X: float32 (n, d)', D.green.dark, { fs: 15, fw: 700 });
  o += code(449, 104, 'y: float32 (n, 1)', D.green.dark, { fs: 15, fw: 700 });
  o += note(449, 140, '✓ 4 bytes/val (50% RAM)', C.ink, { fs: 13 });
  o += note(449, 164, '✓ matches GPU weights', C.ink, { fs: 13 });
  return svg(W, H, o);
}

export const diagrams = {
  'pandas-pipeline': pipeline,
  'pandas-missing':  missing,
  'pandas-onehot':   onehot,
  'pandas-scale':    scale,
  'pandas-xy-split':             xySplit,
  'pandas-imputation-shrinkage': imputationShrinkage,
  'pandas-imputation-spread':    imputationSpread,
  'pandas-categorical-pitfalls': categoricalPitfalls,
  'pandas-leakage-split':        leakageSplit,
  'pandas-dtype-cast':           dtypeCast,
};
