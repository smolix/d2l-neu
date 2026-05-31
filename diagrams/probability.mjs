// Diagrams for chapter_preliminaries/probability (§2.6).
//
// The section is equation-heavy; these figures carry the intuition —
// Bayes in natural frequencies, the prior→likelihood tree, joint/marginal/
// conditional as a grid, the (in)dependence DAGs (explaining away), the
// prior→posterior update, and a Venn diagram for the event axioms.
//
// Stable ids use the `probability-<concept>` prefix. Never rename.

import { C, grid, tx, arrow, svg } from './engine.mjs';

const line = (x1, y1, x2, y2, color, w = 1.5, dash = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
const box = (cx, cy, w, h, fill, st, label, sub, fs = 14) => {
  let o = `<rect x="${cx - w / 2}" y="${cy - h / 2}" width="${w}" height="${h}" rx="9" fill="${fill}" stroke="${st}" stroke-width="2"/>`;
  o += tx(cx, cy - (sub ? 9 : 0), label, { fs, fw: 700, fill: C.ink });
  if (sub) o += tx(cx, cy + 13, sub, { fs: 11.5, fw: 500, fill: C.muted });
  return o;
};
const circ = (cx, cy, r, fill, st, label) =>
  `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${fill}" stroke="${st}" stroke-width="2"/>`
  + tx(cx, cy, label, { fs: 15, fw: 700, fill: C.ink });

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
  o += tx(W / 2, 28, 'Bayes in counts: a positive test is usually a false alarm', { fs: 15.5, fw: 700, fill: C.ink });
  o += box(330, 66, 162, 46, C.lgray, C.gray, '10,000 people', null);
  // split by true status
  o += box(168, 154, 158, 50, C.lgreen, C.green, '15 have HIV', 'P(H=1) = 0.0015');
  o += box(486, 154, 170, 50, C.lgray, C.gray, '9,985 healthy', 'P(H=0) = 0.9985');
  o += arrow(300, 90, 202, 130, C.gray); o += tx(230, 104, '0.15%', { fs: 11.5, fw: 700, fill: C.green });
  o += arrow(360, 90, 458, 130, C.gray); o += tx(432, 104, '99.85%', { fs: 11.5, fw: 700, fill: C.muted });
  // test-positive counts
  o += box(168, 250, 158, 48, C.lgreen, C.green, '15 test +', '100% detected');
  o += box(486, 250, 170, 48, C.lamber, C.amber, '≈ 100 test +', '1% false positive');
  o += arrow(168, 180, 168, 225, C.green);
  o += arrow(486, 180, 486, 225, C.amber);
  // verdict
  o += box(330, 326, 476, 46, C.lblue, C.blue, '115 test positive · only 15 truly have HIV', 'P(H=1 | D=1) ≈ 15 / 115 ≈ 13%');
  o += arrow(202, 274, 300, 305, C.gray);
  o += arrow(452, 274, 360, 305, C.gray);
  return svg(W, H, o);
}

// ── joint / marginal / conditional as a grid ────────────────────────
function jointGrid() {
  const W = 384, H = 372; let o = ''; const s = 46, g = 5;
  const x0 = 96, y0 = 102;
  o += tx(190, 26, 'Joint, marginal, conditional', { fs: 15, fw: 700, fill: C.ink });
  const J = [['.12', '.18', '.10'], ['.20', '.15', '.25']];   // rows A=a1,a2 · cols B=b1,b2,b3
  const hot = (r) => r === 1;   // highlight A=a2
  // caption + column headers, well clear of the title
  o += tx(x0 + 1.5 * (s + g) - g / 2, y0 - 44, 'joint  P(A, B)', { fs: 12.5, fw: 700, fill: C.ink });
  o += tx(x0 + s / 2, y0 - 16, 'b₁', { fs: 12, fill: C.muted });
  o += tx(x0 + s + g + s / 2, y0 - 16, 'b₂', { fs: 12, fill: C.muted });
  o += tx(x0 + 2 * (s + g) + s / 2, y0 - 16, 'b₃', { fs: 12, fill: C.muted });
  o += grid(J, x0, y0, s, g, { fill: (r) => hot(r) ? [C.lamber, C.amber, false, C.ink] : [C.lblue, C.blue, false, C.ink] });
  o += tx(x0 - 16, y0 + s / 2, 'a₁', { fs: 13, fw: 700, fill: C.muted, anchor: 'end' });
  o += tx(x0 - 16, y0 + s + g + s / 2, 'a₂', { fs: 13, fw: 700, fill: C.muted, anchor: 'end' });
  // marginal P(A) = row sums, to the right
  const mx = x0 + 3 * (s + g) + 24;
  o += grid([['.40'], ['.60']], mx, y0, s, g, { fill: () => [C.lgreen, C.green, false, C.ink] });
  o += tx(mx + s / 2, y0 - 16, 'P(A)', { fs: 11.5, fw: 700, fill: C.green });
  // marginal P(B) = column sums, below
  const my = y0 + 2 * (s + g) + 22;
  o += grid([['.32', '.33', '.35']], x0, my, s, g, { fill: () => [C.lgreen, C.green, false, C.ink] });
  o += tx(x0 - 16, my + s / 2, 'P(B)', { fs: 11.5, fw: 700, fill: C.green, anchor: 'end' });
  // conditional: highlighted row renormalized
  const cy = my + s + 24;
  o += grid([['.33', '.25', '.42']], x0, cy, s, g, { fill: () => [C.lamber, C.amber, false, C.ink] });
  o += tx(x0 - 16, cy + s / 2, 'P(B|a₂)', { fs: 11, fw: 700, fill: C.amber, anchor: 'end' });
  o += tx(x0 + 3 * (s + g) + 6, cy + s / 2, 'row ÷ 0.60', { fs: 11.5, fw: 700, fill: C.amber, anchor: 'start' });
  return svg(W, H, o);
}

// ── (in)dependence DAGs: conditioning creates or destroys it ────────
function explainingAway() {
  const W = 660, H = 272; let o = '';
  o += tx(W / 2, 28, 'Conditioning can create or destroy dependence', { fs: 16, fw: 700, fill: C.ink });
  const node = (x, y, l) =>
    `<circle cx="${x}" cy="${y}" r="20" fill="${C.lblue}" stroke="${C.blue}" stroke-width="2"/>`
    + tx(x, y, l, { fs: 16, fw: 700, fill: C.ink });
  const cap = (cx, t1, t2) => tx(cx, 218, t1, { fs: 13.5, fw: 700, fill: C.ink }) + tx(cx, 240, t2, { fs: 12, fill: C.muted });
  // 1 — common cause  C → A, C → B
  let cx = 110;
  o += node(cx, 80, 'C'); o += node(cx - 46, 160, 'A'); o += node(cx + 46, 160, 'B');
  o += arrow(cx - 10, 96, cx - 40, 144, C.ink); o += arrow(cx + 10, 96, cx + 40, 144, C.ink);
  o += cap(cx, 'common cause', 'A, B dependent — ⟂ given C');
  // 2 — chain  A → B → C
  cx = 330;
  o += node(cx - 64, 120, 'A'); o += node(cx, 120, 'B'); o += node(cx + 64, 120, 'C');
  o += arrow(cx - 44, 120, cx - 20, 120, C.ink); o += arrow(cx + 20, 120, cx + 44, 120, C.ink);
  o += cap(cx, 'chain (Markov)', 'A ⟂ C given B');
  // 3 — collider  A → C ← B  (explaining away)
  cx = 552;
  o += node(cx - 46, 80, 'A'); o += node(cx + 46, 80, 'B'); o += node(cx, 160, 'C');
  o += arrow(cx - 40, 96, cx - 10, 144, C.ink); o += arrow(cx + 40, 96, cx + 10, 144, C.ink);
  o += cap(cx, 'collider (explaining away)', 'A ⟂ B — dependent given C');
  return svg(W, H, o);
}

// ── prior → posterior: evidence accumulates ─────────────────────────
function bayesUpdate() {
  const W = 560, H = 258; let o = '';
  o += tx(W / 2, 28, 'Each positive test updates the belief', { fs: 16, fw: 700, fill: C.ink });
  const base = 200, top = 66, scale = (base - top) / 100;   // % → px
  const bars = [
    { x: 132, pct: 0.15, l: 'prior', s: '0.15%', c: C.gray, cl: C.lgray },
    { x: 300, pct: 13, l: 'after test 1', s: '13%', c: C.amber, cl: C.lamber },
    { x: 466, pct: 83, l: 'after test 2', s: '83%', c: C.green, cl: C.lgreen },
  ];
  o += line(72, base, 520, base, C.gray, 1.6);
  // y-axis label, inside the plot at the top-left
  o += tx(80, top - 6, 'P(HIV)', { fs: 13.5, fw: 700, fill: C.muted, anchor: 'start' });
  bars.forEach((b, i) => {
    const h = Math.max(b.pct * scale, 2), y = base - h, bw = 78;
    o += `<rect x="${b.x - bw / 2}" y="${y}" width="${bw}" height="${h}" rx="3" fill="${b.cl}" stroke="${b.c}" stroke-width="2"/>`;
    o += tx(b.x, y - 14, b.s, { fs: 15.5, fw: 700, fill: b.c });
    o += tx(b.x, base + 22, b.l, { fs: 13, fw: 600, fill: C.ink });
    if (i < bars.length - 1) o += arrow(bars[i].x + bw / 2 + 8, base - 16, bars[i + 1].x - bw / 2 - 8, base - 16, C.gray);
  });
  o += tx(W / 2, H - 14, 'two independent tests turn a 0.15% prior into 83% confidence', { fs: 13, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── Venn diagram for events in a sample space ───────────────────────
function venn() {
  const W = 440, H = 300; let o = '';
  o += tx(W / 2, 26, 'Events in a sample space', { fs: 15, fw: 700, fill: C.ink });
  // sample space S
  o += `<rect x="40" y="52" width="360" height="200" rx="8" fill="${C.lgray}" fill-opacity="0.35" stroke="${C.gray}" stroke-width="1.6"/>`;
  o += tx(58, 70, 'S', { fs: 14, fw: 700, fill: C.muted });
  const ay = 152, bx = 250, ax = 190;
  // two overlapping circles (translucent so the intersection shows)
  o += `<circle cx="${ax}" cy="${ay}" r="78" fill="${C.blue}" fill-opacity="0.16" stroke="${C.blue}" stroke-width="2.2"/>`;
  o += `<circle cx="${bx}" cy="${ay}" r="78" fill="${C.amber}" fill-opacity="0.16" stroke="${C.amber}" stroke-width="2.2"/>`;
  o += tx(ax - 40, ay, 'A', { fs: 18, fw: 700, fill: C.blue });
  o += tx(bx + 40, ay, 'B', { fs: 18, fw: 700, fill: C.amber });
  o += tx((ax + bx) / 2, ay, 'A ∩ B', { fs: 12.5, fw: 700, fill: C.ink });
  o += tx(W / 2, H - 16, 'P(A ∪ B) = P(A) + P(B) − P(A ∩ B)', { fs: 13, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// ── discrete PMF (bars) vs continuous PDF (curve + shaded interval) ──
function density() {
  const W = 700, H = 300; let o = '';
  o += tx(W / 2, 26, 'Discrete mass vs. continuous density', { fs: 15.5, fw: 700, fill: C.ink });
  const base = 244;
  // left: PMF
  o += tx(180, 58, 'discrete:  P(X = v) is a bar', { fs: 12.5, fw: 700, fill: C.blue });
  o += line(60, base, 322, base, C.gray, 1.6);
  const pmf = [0.10, 0.24, 0.30, 0.21, 0.15], bx0 = 92, bp = 46, bw = 30, hsc = 380;
  pmf.forEach((p, i) => {
    const cx = bx0 + i * bp, h = p * hsc;
    o += `<rect x="${cx - bw / 2}" y="${base - h}" width="${bw}" height="${h}" rx="3" fill="${C.lblue}" stroke="${C.blue}" stroke-width="2"/>`;
    o += tx(cx, base + 16, `${i + 1}`, { fs: 11, fw: 600, fill: C.muted });
  });
  // right: PDF
  o += tx(525, 58, 'continuous:  P(a ≤ X ≤ b) = area', { fs: 12.5, fw: 700, fill: C.amber });
  o += line(388, base, 664, base, C.gray, 1.6);
  const mu = 526, sig = 46, peak = 152, f = (x) => base - peak * gauss(x, mu, sig), a = 500, b = 584;
  o += area(f, a, b, 40, base, C.amber, 0.30);
  o += curve(f, 396, 656, 80, C.amber, 2.6);
  o += line(a, base, a, base + 6, C.amber, 1.6); o += tx(a, base + 18, 'a', { fs: 11, fw: 700, fill: C.amber });
  o += line(b, base, b, base + 6, C.amber, 1.6); o += tx(b, base + 18, 'b', { fs: 11, fw: 700, fill: C.amber });
  o += tx(W / 2, 282, 'an exact value has zero probability — you integrate over an interval', { fs: 11, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── expectation as a probability-weighted average (the investment) ──
function expectation() {
  const W = 600, H = 260; let o = '';
  const base = 172, x0 = 92, sc = 44, X = (r) => x0 + r * sc;
  o += line(x0 - 10, base, X(10) + 18, base, C.gray, 1.6);
  const out = [
    { r: 0,  p: 0.5, c: C.gray,  cl: C.lgray,  l: '0' },
    { r: 2,  p: 0.4, c: C.amber, cl: C.lamber, l: '2×' },
    { r: 10, p: 0.1, c: C.green, cl: C.lgreen, l: '10×' },
  ];
  const hsc = 210, bw = 38;
  out.forEach(d => {
    const cx = X(d.r), h = d.p * hsc;
    o += `<rect x="${cx - bw / 2}" y="${base - h}" width="${bw}" height="${h}" rx="3" fill="${d.cl}" stroke="${d.c}" stroke-width="2"/>`;
    o += tx(cx, base - h - 30, d.l, { fs: 13, fw: 700, fill: C.ink });
    o += tx(cx, base - h - 13, `${Math.round(d.p * 100)}%`, { fs: 11.5, fw: 700, fill: d.c });
  });
  o += tx(X(6), base + 16, 'return', { fs: 11, fill: C.muted });
  const mx = X(1.8);
  o += `<path d="M${mx} ${base - 1} L${mx - 11} ${base + 20} L${mx + 11} ${base + 20} Z" fill="${C.blue}"/>`;
  o += tx(mx, base + 38, 'E[X] = 1.8×', { fs: 13, fw: 700, fill: C.blue });
  o += tx(W / 2, H - 12, '0.5·0  +  0.4·2  +  0.1·10  =  1.8   (the balance point)', { fs: 12.5, fw: 700, fill: C.ink });
  return svg(W, H, o);
}

// ── variance: two distributions, same mean, different spread ────────
function spread() {
  const W = 600, H = 290; let o = '';
  o += tx(W / 2, 26, 'Variance: same mean, different spread', { fs: 15.5, fw: 700, fill: C.ink });
  const base = 236, mu = 300;
  o += line(64, base, 536, base, C.gray, 1.6);
  const fW = (x) => base - 66 * gauss(x, mu, 82), fN = (x) => base - 150 * gauss(x, mu, 34);
  o += curve(fW, 70, 530, 110, C.amber, 2.6);
  o += curve(fN, 70, 530, 110, C.blue, 2.6);
  o += line(mu, base, mu, 72, C.ink, 1.4, true);
  o += tx(mu, 62, 'μ', { fs: 13, fw: 700, fill: C.ink });
  o += line(mu - 82, base, mu - 82, base - 8, C.amber, 1.6); o += tx(mu - 82, base + 16, 'μ−σ', { fs: 10.5, fw: 700, fill: C.amber });
  o += line(mu + 82, base, mu + 82, base - 8, C.amber, 1.6); o += tx(mu + 82, base + 16, 'μ+σ', { fs: 10.5, fw: 700, fill: C.amber });
  o += `<rect x="408" y="85" width="16" height="14" rx="2" fill="${C.lblue}" stroke="${C.blue}" stroke-width="2"/>`;
  o += tx(432, 94, 'low variance', { fs: 11, fw: 600, fill: C.ink, anchor: 'start' });
  o += `<rect x="408" y="109" width="16" height="14" rx="2" fill="${C.lamber}" stroke="${C.amber}" stroke-width="2"/>`;
  o += tx(432, 118, 'high variance', { fs: 11, fw: 600, fill: C.ink, anchor: 'start' });
  o += tx(W / 2, H - 12, 'σ (standard deviation) = typical distance from the mean', { fs: 11.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── covariance: sign of the relationship, as scatter clouds ─────────
function covariance() {
  const W = 720, H = 220; let o = '';
  const cy = 90, ax = 64;
  const panel = (cx, pts, color, t1, t2) => {
    let s = `<line x1="${cx - ax}" y1="${cy}" x2="${cx + ax}" y2="${cy}" stroke="${C.gray}" stroke-width="1.2"/>`
          + `<line x1="${cx}" y1="${cy - ax}" x2="${cx}" y2="${cy + ax}" stroke="${C.gray}" stroke-width="1.2"/>`;
    pts.forEach(([dx, dy]) => { s += dot(cx + dx, cy - dy, 3.4, color); });
    s += tx(cx, 180, t1, { fs: 12.5, fw: 700, fill: color });
    s += tx(cx, 200, t2, { fs: 11, fill: C.muted });
    return s;
  };
  const pos = [[-52, -44], [-38, -30], [-30, -40], [-14, -8], [-4, -18], [6, 2], [16, 14], [28, 8], [36, 30], [48, 40], [54, 48], [-22, -26]];
  const zero = [[-48, 12], [-36, -30], [-20, 28], [-8, -14], [2, 40], [8, -36], [18, 18], [26, -22], [38, 34], [46, -8], [-30, 44], [14, -44]];
  const neg = pos.map(([dx, dy]) => [dx, -dy]);
  o += panel(140, pos,  C.green, 'Cov > 0', 'move together');
  o += panel(360, zero, C.gray,  'Cov ≈ 0', 'unrelated');
  o += panel(580, neg,  C.amber, 'Cov &lt; 0', 'move oppositely');
  return svg(W, H, o);
}

// ── Markov's inequality: a pictorial, distribution-free tail bound ──
function markov() {
  const W = 640, H = 306; let o = '';
  o += tx(W / 2, 24, "Markov's inequality: a distribution-free tail bound", { fs: 15, fw: 700, fill: C.ink });
  const base = 224, x0 = 80, x1 = 580, k = 112;
  o += line(x0 - 4, base, x1 + 10, base, C.gray, 1.6);
  const g = (x) => { const t = (x - x0) / k; return base - 380 * t * Math.exp(-t); };
  const muX = 300, aX = 408;
  o += area(g, aX, x1, 50, base, C.amber, 0.35);
  o += curve(g, x0, x1, 110, C.purple, 2.6);
  o += line(muX, base, muX, g(muX) - 6, C.ink, 1.5, true); o += tx(muX, g(muX) - 16, 'E[X]', { fs: 11.5, fw: 700, fill: C.ink });
  o += line(aX, base, aX, 92, C.amber, 1.8); o += tx(aX, 82, 'a', { fs: 13, fw: 700, fill: C.amber });
  o += tx(478, 150, 'P(X ≥ a)', { fs: 11.5, fw: 700, fill: C.amber });
  o += arrow(478, 160, 456, base - 14, C.amber);
  o += `<rect x="${W / 2 - 160}" y="242" width="320" height="40" rx="9" fill="${C.lblue}" stroke="${C.blue}" stroke-width="2"/>`;
  o += tx(W / 2, 265, 'P(X ≥ a)  ≤  E[X] / a', { fs: 16, fw: 700, fill: C.ink });
  o += tx(W / 2, H - 8, 'distribution-free: Chebyshev applies it to (X−μ)²; Hoeffding &amp; Bernstein sharpen it', { fs: 10.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

// ── aleatoric (irreducible) vs epistemic (shrinks with data) ────────
function uncertainty() {
  const W = 680, H = 254; let o = '';
  // left: aleatoric
  o += tx(176, 34, 'Aleatoric — irreducible', { fs: 13, fw: 700, fill: C.amber });
  const lb = 184;
  o += line(70, lb, 300, lb, C.gray, 1.6);
  [['heads', 124], ['tails', 214]].forEach(([lab, cx]) => {
    o += `<rect x="${cx - 34}" y="${lb - 100}" width="68" height="100" rx="3" fill="${C.lamber}" stroke="${C.amber}" stroke-width="2"/>`;
    o += tx(cx, lb - 112, '0.5', { fs: 13, fw: 700, fill: C.amber });
    o += tx(cx, lb + 16, lab, { fs: 11.5, fw: 600, fill: C.ink });
  });
  o += tx(176, 224, 'infinite data → still 50/50', { fs: 11, fw: 600, fill: C.muted });
  // right: epistemic — converging confidence band
  o += tx(508, 34, 'Epistemic — reducible', { fs: 13, fw: 700, fill: C.blue });
  const rx0 = 392, rx1 = 628, tY = 122;
  const hw = (x) => 60 * (1 - 0.86 * (x - rx0) / (rx1 - rx0)), up = (x) => tY - hw(x), lo = (x) => tY + hw(x);
  let band = `M${rx0} ${up(rx0).toFixed(1)}`;
  for (let i = 1; i <= 60; i++) { const x = rx0 + (rx1 - rx0) * i / 60; band += `L${x.toFixed(1)} ${up(x).toFixed(1)}`; }
  for (let i = 60; i >= 0; i--) { const x = rx0 + (rx1 - rx0) * i / 60; band += `L${x.toFixed(1)} ${lo(x).toFixed(1)}`; }
  o += `<path d="${band}Z" fill="${C.lblue}" fill-opacity="0.75" stroke="${C.blue}" stroke-width="1.6"/>`;
  o += line(rx0, tY, rx1, tY, C.ink, 1.4, true); o += tx(rx0 + 6, tY - 8, 'true p', { fs: 10.5, fw: 700, fill: C.ink, anchor: 'start' });
  o += arrow(rx0, 194, rx1, 194, C.gray); o += tx(510, 212, 'samples n →', { fs: 11, fw: 600, fill: C.muted });
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
  'probability-markov':              markov,
  'probability-uncertainty':         uncertainty,
};
