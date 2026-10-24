// Diagram for chapter_preliminaries/lookup-api (Documentation).
//
// The four-step procedure for an unfamiliar function: discover names,
// inspect a signature and docstring, read the documentation or the source,
// verify with a small example, and go back to reading when the result is
// unexpected. Book figure fig_lookup_loop and slide figure.
//
// Sizing follows the chapter's other modules: at most 720 units wide,
// titles 17 px, labels 15 px, text in INK.
//
// Stable ids use the `lookup-api-<concept>` prefix. Never rename.

import { C, TOKENS, tx, arrow, svg } from './engine.mjs';

function discoveryLoop() {
  const bw = 140, bh = 62, gap = 40, x0 = 8, cy = 39; let o = '';
  // Steps are neutral; only Verify, the step that decides, carries an accent.
  const N = TOKENS.neutrals;
  const steps = [
    { t: 'Discover', s: 'dir · Tab', c: C.gray, cl: N.panel },
    { t: 'Inspect', s: 'help · ?', c: C.gray, cl: N.panel },
    { t: 'Read', s: 'docs · ??', c: C.gray, cl: N.panel },
    { t: 'Verify', s: 'small example', c: C.green, cl: C.lgreen },
  ];
  steps.forEach((st, i) => { st.x = x0 + i * (bw + gap) + bw / 2; });
  // the way back from Verify to Read, routed under the row; drawn first so
  // its ends tuck under the opaque boxes
  const yb = cy + bh / 2 + 30, from = steps[3].x, to = steps[2].x;
  o += `<path d="M ${from} ${cy} L ${from} ${yb} L ${to} ${yb} L ${to} ${cy + bh / 2 + 12}" fill="none" stroke="${C.gray}" stroke-width="2.5"/>`;
  o += arrow(to, cy + bh / 2 + 14, to, cy + bh / 2, C.gray);
  o += tx((from + to) / 2, yb + 18, 'unexpected result', { fs: 15, fw: 500, fill: C.ink });
  steps.forEach((st, i) => {
    o += `<rect x="${st.x - bw / 2}" y="${cy - bh / 2}" width="${bw}" height="${bh}" rx="8" fill="${st.cl}" stroke="${st.c}" stroke-width="2"/>`;
    o += tx(st.x, cy - 11, st.t, { fs: 17, fw: 700, fill: C.ink });
    o += tx(st.x, cy + 13, st.s, { fs: 15, fw: 500, fill: C.ink });
    if (i < steps.length - 1)
      o += arrow(st.x + bw / 2 + 6, cy, steps[i + 1].x - bw / 2 - 6, cy, C.ink);
  });
  return svg(x0 + 4 * bw + 3 * gap + 8, yb + 34, o);
}

export const diagrams = {
  'lookup-api-discovery-loop': discoveryLoop,
};
