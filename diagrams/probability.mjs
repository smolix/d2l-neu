// Diagrams for chapter_preliminaries/probability.
//
// Book figures (also synced to img/): the Venn diagram for the axioms,
// mass vs density, the joint/marginal/conditional table, the three
// (in)dependence graphs, the HIV test in natural frequencies, the
// prior -> posterior update, and the sign of the covariance. Slide-only
// figures (img/auto/): expectation as a balance point, variance as spread,
// and aleatoric vs epistemic uncertainty.
//
// Axes and annotation text are INK; colored text uses the dark variant of
// its accent. No figure repeats its caption.
//
// Stable ids use the `probability-<concept>` prefix. Never rename.

import { C, TOKENS, grid, tx, arrow, svg } from './engine.mjs';

const DK = {
  blue: TOKENS.accents.blue.dark,
  amber: TOKENS.accents.orange.dark,
  green: TOKENS.accents.green.dark,
  purple: TOKENS.accents.purple.dark,
};

const line = (x1, y1, x2, y2, color, w = 1.5, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
const box = (cx, cy, w, h, fill, st, label, sub) => {
  let o = `<rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="9" fill="${fill}" stroke="${st}" stroke-width="2"/>`;
  o += tx(cx, cy - (sub ? 10 : 0), label, { fs: 15, fw: 700, fill: C.ink });
  if (sub) o += tx(cx, cy + 12, sub, { fs: 13, fw: 500, fill: C.ink });
  return o;
};

// ── curve helpers (fn(x) returns the screen y) ──────────────────────
const curve = (fn, x0, x1, n, color, w = 2.5) => {
  let d = '';
  for (let i = 0; i <= n; i++) { const x = x0 + (x1 - x0) * i / n; d += (i ? 'L' : 'M') + x.toFixed(1) + ' ' + fn(x).toFixed(1); }
  return `<path d="${d}" fill="none" stroke="${color}" stroke-width="${w}"/>`;
};
const area = (fn, a, b, n, base, fill, op = 0.22) => {
  let d = `M${a.toFixed(1)} ${base.toFixed(1)}`;
  for (let i = 0; i <= n; i++) { const x = a + (b - a) * i / n; d += `L${x.toFixed(1)} ${fn(x).toFixed(1)}`; }
  return `<path d="${d}L${b.toFixed(1)} ${base.toFixed(1)}Z" fill="${fill}" fill-opacity="${op}" stroke="none"/>`;
};
const gauss = (x, mu, sig) => Math.exp(-((x - mu) * (x - mu)) / (2 * sig * sig));
const dot = (x, y, r, fill) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}"/>`;

// ── Bayes in counts: why a positive test is usually a false alarm ───
function naturalFrequencies() {
  const W = 660, H = 360; let o = '';
  // arrows FIRST: tails start under the (opaque) source boxes and emerge
  // from behind them; tips land exactly on the target box boundary
  o += arrow(300, 89, 196, 129, C.gray);
  o += arrow(360, 89, 464, 129, C.gray);
  o += arrow(168, 179, 168, 225, C.green);
  o += arrow(486, 179, 486, 225, C.amber);
  o += arrow(210, 275, 296, 303, C.gray);
  o += arrow(444, 275, 364, 303, C.gray);
  o += tx(222, 100, '0.15%', { fs: 14, fw: 700, fill: DK.green });
  o += tx(442, 100, '99.85%', { fs: 14, fw: 700, fill: C.ink });
  // boxes on top
  o += box(330, 66, 170, 46, C.lgray, C.gray, '10,000 people', null);
  o += box(168, 154, 176, 50, C.lgreen, C.green, '15 have HIV', 'P(H=1) = 0.0015');
  o += box(486, 154, 176, 50, C.lgray, C.gray, '9,985 healthy', 'P(H=0) = 0.9985');
  o += box(168, 250, 176, 50, C.lgreen, C.green, '15 test positive', 'all detected');
  o += box(486, 250, 176, 50, C.lamber, C.amber, '≈ 100 test positive', '1% false positives');
  o += box(330, 328, 400, 50, C.lblue, C.blue, '≈ 115 positive results', 'P(H=1 | D₁=1) ≈ 15 / 115 ≈ 13%');
  return svg(W, H, o);
}

// ── joint / marginal / conditional as a grid ────────────────────────
function jointGrid() {
  const W = 410, H = 380; let o = ''; const s = 52, g = 5;
  const x0 = 104, y0 = 102;
  const J = [['0.12', '0.18', '0.10'], ['0.20', '0.15', '0.25']];   // rows A=a1,a2 · cols B=b1,b2,b3
  const hot = (r) => r === 1;   // highlight A=a2
  o += tx(x0 + 1.5 * (s + g) - g / 2, y0 - 46, 'joint P(A, B)', { fs: 15, fw: 700, fill: C.ink });
  ['b₁', 'b₂', 'b₃'].forEach((b, i) => {
    o += tx(x0 + i * (s + g) + s / 2, y0 - 16, b, { fs: 15, fw: 600, fill: C.ink });
  });
  o += grid(J, x0, y0, s, g, { fill: (r) => hot(r) ? [C.lamber, C.amber, false, C.ink] : [C.lblue, C.blue, false, C.ink] });
  o += tx(x0 - 14, y0 + s / 2, 'a₁', { fs: 15, fw: 700, fill: C.ink, anchor: 'end' });
  o += tx(x0 - 14, y0 + s + g + s / 2, 'a₂', { fs: 15, fw: 700, fill: C.ink, anchor: 'end' });
  // marginal P(A) = row sums, to the right
  const mx = x0 + 3 * (s + g) + 24;
  o += grid([['0.40'], ['0.60']], mx, y0, s, g, { fill: () => [C.lgreen, C.green, false, C.ink] });
  o += tx(mx + s / 2, y0 - 16, 'P(A)', { fs: 15, fw: 700, fill: DK.green });
  // marginal P(B) = column sums, below
  const my = y0 + 2 * (s + g) + 22;
  o += grid([['0.32', '0.33', '0.35']], x0, my, s, g, { fill: () => [C.lgreen, C.green, false, C.ink] });
  o += tx(x0 - 14, my + s / 2, 'P(B)', { fs: 15, fw: 700, fill: DK.green, anchor: 'end' });
  // conditional: highlighted row renormalized
  const cy = my + s + 24;
  o += grid([['0.33', '0.25', '0.42']], x0, cy, s, g, { fill: () => [C.lamber, C.amber, false, C.ink] });
  o += tx(x0 - 14, cy + s / 2, 'P(B | a₂)', { fs: 15, fw: 700, fill: DK.amber, anchor: 'end' });
  o += tx(x0 + 3 * (s + g) + 6, cy + s / 2, 'row a₂ ÷ 0.60', { fs: 14, fw: 700, fill: DK.amber, anchor: 'start' });
  return svg(W, H, o);
}

// ── (in)dependence graphs: conditioning creates or destroys it ──────
function explainingAway() {
  const W = 660, H = 262; let o = '';
  const node = (x, y, l) =>
    `<circle cx="${x}" cy="${y}" r="23" fill="${C.lblue}" stroke="${C.blue}" stroke-width="2"/>`
    + tx(x, y, l, { fs: 20, fw: 700, fill: C.ink });
  const cap = (cx, t1, t2) => tx(cx, 210, t1, { fs: 18, fw: 700, fill: C.ink }) + tx(cx, 238, t2, { fs: 17, fill: C.ink });
  // 1 — common cause  C → A, C → B
  let cx = 110;
  o += node(cx, 70, 'C'); o += node(cx - 46, 152, 'A'); o += node(cx + 46, 152, 'B');
  o += arrow(cx - 12, 90, cx - 34, 132, C.ink); o += arrow(cx + 12, 90, cx + 34, 132, C.ink);
  o += cap(cx, 'common cause', 'A ⟂ B given C');
  // 2 — chain  A → B → C
  cx = 330;
  o += node(cx - 70, 112, 'A'); o += node(cx, 112, 'B'); o += node(cx + 70, 112, 'C');
  o += arrow(cx - 47, 112, cx - 23, 112, C.ink); o += arrow(cx + 23, 112, cx + 47, 112, C.ink);
  o += cap(cx, 'chain', 'A ⟂ C given B');
  // 3 — common effect  A → C ← B  (explaining away)
  cx = 552;
  o += node(cx - 46, 70, 'A'); o += node(cx + 46, 70, 'B'); o += node(cx, 152, 'C');
  o += arrow(cx - 34, 90, cx - 12, 132, C.ink); o += arrow(cx + 34, 90, cx + 12, 132, C.ink);
  o += cap(cx, 'common effect', 'A ⟂ B, but not given C');
  return svg(W, H, o);
}

// ── prior → posterior: evidence accumulates ─────────────────────────
function bayesUpdate() {
  const W = 560, H = 250; let o = '';
  const base = 200, top = 66, scale = (base - top) / 100;   // % → px
  const bars = [
    { x: 132, pct: 0.15, l: 'prior', s: '0.15%', c: C.gray, cl: C.lgray, d: C.ink },
    { x: 300, pct: 13, l: 'after test 1', s: '13%', c: C.amber, cl: C.lamber, d: DK.amber },
    { x: 466, pct: 83, l: 'after test 2', s: '83%', c: C.green, cl: C.lgreen, d: DK.green },
  ];
  o += line(72, base, 520, base, C.ink, 1.6);
  o += tx(80, top - 6, 'P(H = 1)', { fs: 15, fw: 700, fill: C.ink, anchor: 'start' });
  bars.forEach((b, i) => {
    const h = Math.max(b.pct * scale, 2), y = base - h, bw = 78;
    o += `<rect x="${b.x - bw / 2}" y="${y}" width="${bw}" height="${h}" rx="3" fill="${b.cl}" stroke="${b.c}" stroke-width="2"/>`;
    o += tx(b.x, y - 14, b.s, { fs: 17, fw: 700, fill: b.d });
    o += tx(b.x, base + 22, b.l, { fs: 15, fw: 600, fill: C.ink });
    if (i < bars.length - 1) o += arrow(bars[i].x + bw / 2 + 8, base - 16, bars[i + 1].x - bw / 2 - 8, base - 16, C.gray);
  });
  return svg(W, H, o);
}

// ── Venn diagram for events in a sample space ───────────────────────
function venn() {
  const W = 440, H = 270; let o = '';
  o += `<rect x="40" y="40" width="360" height="200" rx="8" fill="${C.lgray}" fill-opacity="0.35" stroke="${C.gray}" stroke-width="1.6"/>`;
  o += tx(60, 60, 'S', { fs: 18, fw: 700, fill: C.ink });
  const ay = 140, bx = 250, ax = 190;
  // two overlapping circles (translucent so the intersection shows)
  o += `<circle cx="${ax}" cy="${ay}" r="78" fill="${C.blue}" fill-opacity="0.16" stroke="${C.blue}" stroke-width="2.2"/>`;
  o += `<circle cx="${bx}" cy="${ay}" r="78" fill="${C.amber}" fill-opacity="0.16" stroke="${C.amber}" stroke-width="2.2"/>`;
  o += tx(ax - 42, ay, 'A', { fs: 20, fw: 700, fill: DK.blue });
  o += tx(bx + 42, ay, 'B', { fs: 20, fw: 700, fill: DK.amber });
  o += tx((ax + bx) / 2, ay, 'A ∩ B', { fs: 19, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// ── discrete PMF (bars) vs continuous PDF (curve + shaded interval) ──
function density() {
  const W = 700, H = 280; let o = '';
  const base = 244;
  // left: PMF
  o += tx(191, 46, 'P(X = v)', { fs: 19, fw: 700, fill: DK.blue });
  o += line(60, base, 322, base, C.ink, 1.6);
  const pmf = [0.10, 0.24, 0.30, 0.21, 0.15], bx0 = 99, bp = 46, bw = 30, hsc = 480;
  pmf.forEach((p, i) => {
    const cx = bx0 + i * bp, h = p * hsc;
    o += `<rect x="${cx - bw / 2}" y="${base - h}" width="${bw}" height="${h}" rx="3" fill="${C.lblue}" stroke="${C.blue}" stroke-width="2"/>`;
    o += tx(cx, base + 20, `${i + 1}`, { fs: 18, fw: 600, fill: C.ink });
  });
  // right: PDF
  o += tx(526, 46, 'P(a ≤ X ≤ b) = area', { fs: 19, fw: 700, fill: DK.amber });
  o += line(388, base, 664, base, C.ink, 1.6);
  const mu = 526, sig = 46, peak = 170, f = (x) => base - peak * gauss(x, mu, sig), a = 500, b = 584;
  o += area(f, a, b, 40, base, C.amber, 0.30);
  o += curve(f, 396, 656, 80, C.amber, 2.6);
  o += line(a, base, a, base + 6, C.ink, 1.6); o += tx(a, base + 20, 'a', { fs: 19, fw: 700, fill: C.ink });
  o += line(b, base, b, base + 6, C.ink, 1.6); o += tx(b, base + 20, 'b', { fs: 19, fw: 700, fill: C.ink });
  o += tx(614, 132, 'p(x)', { fs: 19, fw: 700, fill: DK.amber, anchor: 'start' });
  return svg(W, H, o);
}

// ── expectation as a probability-weighted average (the investment) ──
function expectation() {
  const W = 600, H = 250; let o = '';
  const base = 172, x0 = 92, sc = 44, X = (r) => x0 + r * sc;
  o += line(x0 - 10, base, X(10) + 18, base, C.ink, 1.6);
  const out = [
    { r: 0,  p: 0.5, c: C.gray,  cl: C.lgray,  d: C.ink,    l: '0' },
    { r: 2,  p: 0.4, c: C.amber, cl: C.lamber, d: DK.amber, l: '2×' },
    { r: 10, p: 0.1, c: C.green, cl: C.lgreen, d: DK.green, l: '10×' },
  ];
  const hsc = 210, bw = 38;
  out.forEach(d => {
    const cx = X(d.r), h = d.p * hsc;
    o += `<rect x="${cx - bw / 2}" y="${base - h}" width="${bw}" height="${h}" rx="3" fill="${d.cl}" stroke="${d.c}" stroke-width="2"/>`;
    o += tx(cx, base - h - 40, d.l, { fs: 19, fw: 700, fill: C.ink });
    o += tx(cx, base - h - 16, `${Math.round(d.p * 100)}%`, { fs: 17, fw: 700, fill: d.d });
  });
  o += tx(X(6.5), base + 20, 'return', { fs: 18, fill: C.ink });
  const mx = X(1.8);
  o += `<path d="M${mx} ${base - 1} L${mx - 11} ${base + 20} L${mx + 11} ${base + 20} Z" fill="${C.blue}"/>`;
  o += tx(mx, base + 42, 'E[X] = 1.8', { fs: 19, fw: 700, fill: DK.blue });
  return svg(W, H, o);
}

// ── variance: two distributions, same mean, different spread ────────
function spread() {
  const W = 600, H = 270; let o = '';
  const base = 236, mu = 300;
  o += line(64, base, 536, base, C.ink, 1.6);
  const fW = (x) => base - 66 * gauss(x, mu, 82), fN = (x) => base - 150 * gauss(x, mu, 34);
  o += curve(fW, 70, 530, 110, C.amber, 2.6);
  o += curve(fN, 70, 530, 110, C.blue, 2.6);
  o += line(mu, base, mu, 72, C.ink, 1.4, true);
  o += tx(mu, 58, 'μ', { fs: 20, fw: 700, fill: C.ink });
  o += line(mu - 82, base, mu - 82, base - 8, C.ink, 1.6); o += tx(mu - 82, base + 21, 'μ − σ', { fs: 18, fw: 700, fill: DK.amber });
  o += line(mu + 82, base, mu + 82, base - 8, C.ink, 1.6); o += tx(mu + 82, base + 21, 'μ + σ', { fs: 18, fw: 700, fill: DK.amber });
  o += `<rect x="392" y="82" width="18" height="16" rx="2" fill="${C.lblue}" stroke="${C.blue}" stroke-width="2"/>`;
  o += tx(418, 90, 'small variance', { fs: 18, fw: 600, fill: C.ink, anchor: 'start' });
  o += `<rect x="392" y="112" width="18" height="16" rx="2" fill="${C.lamber}" stroke="${C.amber}" stroke-width="2"/>`;
  o += tx(418, 120, 'large variance', { fs: 18, fw: 600, fill: C.ink, anchor: 'start' });
  return svg(W, H, o);
}

// ── covariance: sign of the relationship, as scatter clouds ─────────
function covariance() {
  const W = 720, H = 222; let o = '';
  const cy = 86, ax = 66;
  const panel = (cx, pts, color, dark, t1, t2) => {
    let s = line(cx - ax, cy, cx + ax, cy, C.ink, 1.4) + line(cx, cy - ax, cx, cy + ax, C.ink, 1.4);
    pts.forEach(([dx, dy]) => { s += dot(cx + dx, cy - dy, 3.6, color); });
    s += tx(cx, 180, t1, { fs: 16, fw: 700, fill: dark });
    s += tx(cx, 204, t2, { fs: 14, fill: C.ink });
    return s;
  };
  const pos = [[-52, -44], [-38, -30], [-30, -40], [-14, -8], [-4, -18], [6, 2], [16, 14], [28, 8], [36, 30], [48, 40], [54, 48], [-22, -26]];
  const zero = [[-48, 12], [-36, -30], [-20, 28], [-8, -14], [2, 40], [8, -36], [18, 18], [26, -22], [38, 34], [46, -8], [-30, 44], [14, -44]];
  const neg = pos.map(([dx, dy]) => [dx, -dy]);
  o += panel(130, pos,  C.green, DK.green, 'Cov > 0', 'move together');
  o += panel(360, zero, C.gray,  C.ink,    'Cov ≈ 0', 'no linear relation');
  o += panel(590, neg,  C.amber, DK.amber, 'Cov < 0', 'move in opposite ways');
  return svg(W, H, o);
}

// ── aleatoric (irreducible) vs epistemic (shrinks with data) ────────
function uncertainty() {
  const W = 680, H = 250; let o = '';
  // left: aleatoric
  o += tx(176, 28, 'aleatoric: irreducible', { fs: 19, fw: 700, fill: DK.amber });
  const lb = 184;
  o += line(70, lb, 300, lb, C.ink, 1.6);
  [['heads', 124], ['tails', 214]].forEach(([lab, cx]) => {
    o += `<rect x="${cx - 34}" y="${lb - 100}" width="68" height="100" rx="3" fill="${C.lamber}" stroke="${C.amber}" stroke-width="2"/>`;
    o += tx(cx, lb - 116, '0.5', { fs: 18, fw: 700, fill: DK.amber });
    o += tx(cx, lb + 20, lab, { fs: 18, fw: 600, fill: C.ink });
  });
  o += tx(176, 234, 'p known, still 50/50', { fs: 18, fw: 600, fill: C.ink });
  // right: epistemic — converging confidence band around the true p
  o += tx(510, 28, 'epistemic: shrinks with data', { fs: 19, fw: 700, fill: DK.blue });
  const rx0 = 392, rx1 = 628, tY = 120;
  const hw = (x) => 60 * (1 - 0.86 * (x - rx0) / (rx1 - rx0)), up = (x) => tY - hw(x), lo = (x) => tY + hw(x);
  let band = `M${rx0} ${up(rx0).toFixed(1)}`;
  for (let i = 1; i <= 60; i++) { const x = rx0 + (rx1 - rx0) * i / 60; band += `L${x.toFixed(1)} ${up(x).toFixed(1)}`; }
  for (let i = 60; i >= 0; i--) { const x = rx0 + (rx1 - rx0) * i / 60; band += `L${x.toFixed(1)} ${lo(x).toFixed(1)}`; }
  o += `<path d="${band}Z" fill="${C.lblue}" fill-opacity="0.75" stroke="${C.blue}" stroke-width="1.6"/>`;
  o += line(rx0, tY, rx1, tY, C.ink, 1.4, true);
  o += tx(rx1 + 12, tY, 'p', { fs: 20, fw: 700, fill: C.ink, anchor: 'start' });
  o += arrow(rx0, 196, rx1, 196, C.ink);
  o += tx(510, 222, 'number of tosses n', { fs: 18, fw: 600, fill: C.ink });
  return svg(W, H, o);
}

export const diagrams = {
  'probability-natural-frequencies': naturalFrequencies,
  'probability-joint-grid':          jointGrid,
  'probability-explaining-away':     explainingAway,
  'probability-bayes-update':        bayesUpdate,
  'probability-venn':                venn,
  'probability-density':             density,
  'probability-expectation':         expectation,
  'probability-spread':              spread,
  'probability-covariance':          covariance,
  'probability-uncertainty':         uncertainty,
};
