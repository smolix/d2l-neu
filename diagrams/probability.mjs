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

export const diagrams = {
  'probability-natural-frequencies': naturalFrequencies,
  'probability-joint-grid':          jointGrid,
  'probability-explaining-away':     explainingAway,
  'probability-bayes-update':        bayesUpdate,
  'probability-venn':                venn,
};
