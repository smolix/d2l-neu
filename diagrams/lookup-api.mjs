// Diagram for chapter_preliminaries/lookup-api (§2.7, "Documentation").
//
// The "discovery loop": a small, repeatable routine for figuring out an
// unfamiliar API — discover names, inspect a signature, read the docs or
// source, verify with a quick run, and loop if you're still stuck.
//
// Stable ids use the `lookup-api-<concept>` prefix. Never rename.

import { C, tx, arrow, svg } from './engine.mjs';

function discoveryLoop() {
  const W = 748, H = 200; let o = '';
  o += tx(W / 2, 26, 'Figuring out an unfamiliar API: a repeatable loop', { fs: 15.5, fw: 700, fill: C.ink });
  const steps = [
    { x: 98,  t: 'Discover', s: 'dir() · Tab',     c: C.blue,   cl: C.lblue },
    { x: 282, t: 'Inspect',  s: 'help() · ?',       c: C.amber,  cl: C.lamber },
    { x: 466, t: 'Read',     s: 'docs · source ??', c: C.purple, cl: '#F3E5F5' },
    { x: 650, t: 'Verify',   s: 'run a quick test', c: C.green,  cl: C.lgreen },
  ];
  const bw = 148, bh = 60, cy = 98;
  steps.forEach((st, i) => {
    o += `<rect x="${st.x - bw / 2}" y="${cy - bh / 2}" width="${bw}" height="${bh}" rx="10" fill="${st.cl}" stroke="${st.c}" stroke-width="2"/>`;
    o += tx(st.x, cy - 9, st.t, { fs: 15, fw: 700, fill: C.ink });
    o += tx(st.x, cy + 13, st.s, { fs: 11.5, fw: 600, fill: C.muted });
    if (i < steps.length - 1) o += arrow(st.x + bw / 2 + 6, cy, steps[i + 1].x - bw / 2 - 6, cy, C.gray);
  });
  // loop-back: Verify → Discover, routed under the row
  const yb = cy + bh / 2 + 30;   // 158
  o += `<path d="M ${steps[3].x} ${cy + bh / 2} L ${steps[3].x} ${yb} L ${steps[0].x} ${yb}" fill="none" stroke="${C.gray}" stroke-width="2"/>`;
  o += arrow(steps[0].x, yb, steps[0].x, cy + bh / 2 + 2, C.gray);
  o += tx(W / 2, yb + 16, 'still stuck? refine the query and loop', { fs: 11.5, fw: 600, fill: C.muted });
  return svg(W, H, o);
}

export const diagrams = {
  'lookup-api-discovery-loop': discoveryLoop,
};
