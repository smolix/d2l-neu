// d2l slide-diagram engine — pure, DOM-free SVG string builders.
//
// Everything here returns a plain string. There is no `document`, no
// browser API: `node` alone can produce the SVG (see render.mjs). The
// same functions are what the north-star deck embeds inline at runtime,
// so a diagram looks identical whether inlined in a slide or written to
// a static file in img/auto/.
//
// COLOR TOKENS come from the unified figure token layer
// (tools/figstyle/tokens.py -> `python3 -m figstyle.export` -> tokens.mjs).
// Do not hardcode hexes here or in the figure modules — import C/TOKENS.

import { C, TOKENS } from './tokens.mjs';
export { C, TOKENS };

// Font stacks. Both end in a generic family so a static SVG still reads
// correctly even where "Source Sans 3" / "JetBrains Mono" aren't loaded
// (see docs/slides-northstar-design.md "Fonts in static SVG" for the fidelity caveat).
//
// The family names MUST be quoted. "Source Sans 3" contains the token "3",
// which is not a valid CSS identifier (identifiers cannot start with a
// digit); unquoted, the whole font-family declaration is invalid and
// renderers silently fall back to their default *serif* face — so figures
// embedded as <img> (web font blocked) and the rsvg→PDF path both came out
// in Times. Quoting keeps the declaration valid so the generic `sans-serif`
// fallback applies. JetBrains Mono has no such token, which is why the
// monospace cells always rendered correctly.
export const FM = "'JetBrains Mono', monospace";
export const FS = "'Source Sans 3', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif";

// A rounded square cell.
export function rc(x, y, s, f, st, dash) {
  return `<rect x="${x}" y="${y}" width="${s}" height="${s}" rx="6" fill="${f}" stroke="${st}" stroke-width="2"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
}

// XML-escape label text (a bare & / < in a label breaks strict parsers
// like librsvg, even though browsers tolerate it when inlined in HTML).
const esc = (t) => String(t)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

// A centered text label. opts: {mono, fs, fw, fill, anchor, base}
export function tx(x, y, t, o) {
  o = o || {};
  return `<text x="${x}" y="${y}" font-family="${o.mono ? FM : FS}" font-size="${o.fs || 15}" font-weight="${o.fw || 500}" fill="${o.fill || C.ink}" text-anchor="${o.anchor || 'middle'}" dominant-baseline="${o.base || 'central'}">${esc(t)}</text>`;
}

// A grid of cells. `data` is a 2D array of values (use '' / null for blank).
// opt.fill(r,c,v) -> [fillColor, strokeColor, dashBool, textColor]
// opt.showVal=false draws empty cells; opt.fs sets value font size.
export function grid(data, x0, y0, s, gap, opt) {
  opt = opt || {};
  let o = '';
  for (let r = 0; r < data.length; r++) for (let c = 0; c < data[r].length; c++) {
    const v = data[r][c];
    const fc = opt.fill ? opt.fill(r, c, v) : [C.lblue, C.blue, false, C.ink];
    const x = x0 + c * (s + gap), y = y0 + r * (s + gap);
    o += rc(x, y, s, fc[0], fc[1], fc[2]);
    if (opt.showVal !== false && v != null && v !== '')
      o += tx(x + s / 2, y + s / 2, v, { mono: true, fs: opt.fs || 15, fill: fc[3] || C.ink });
  }
  return o;
}

// An arrow from (x1,y1) to (x2,y2). Arrowhead is drawn as a filled path,
// so no <marker>/<defs> id juggling across multiple inlined SVGs.
export function arrow(x1, y1, x2, y2, color, dash) {
  // The SHAFT stops at the head's base — drawing it through to the tip
  // makes the head look stubby and the line poke past it (Alex, ch2 review).
  const ang = Math.atan2(y2 - y1, x2 - x1), h = 10.5;
  const bx = x2 - 0.88 * h * Math.cos(ang), by = y2 - 0.88 * h * Math.sin(ang);
  const a1 = ang + Math.PI - 0.42, a2 = ang + Math.PI + 0.42;
  const hx1 = x2 + h * Math.cos(a1), hy1 = y2 + h * Math.sin(a1);
  const hx2 = x2 + h * Math.cos(a2), hy2 = y2 + h * Math.sin(a2);
  return `<line x1="${x1}" y1="${y1}" x2="${bx.toFixed(2)}" y2="${by.toFixed(2)}" stroke="${color}" stroke-width="2.5"${dash ? ' stroke-dasharray="5 4"' : ''}/>`
       + `<path d="M${x2},${y2} L${hx1.toFixed(2)},${hy1.toFixed(2)} L${hx2.toFixed(2)},${hy2.toFixed(2)} Z" fill="${color}"/>`;
}

// A labelled box (variable address / handle). title + sub stacked, with a
// colored accent rail on the left. faded=true dims it (e.g. an orphaned buffer).
export function block(x, y, title, sub, faded, accent) {
  const w = 178, h = 54, op = faded ? 0.42 : 1;
  accent = accent || C.blue;
  return `<g opacity="${op}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="#fff" stroke="${accent}" stroke-width="2"/>`
    + `<rect x="${x}" y="${y}" width="6" height="${h}" rx="3" fill="${accent}"/>`
    + tx(x + w / 2 + 3, y + 19, title, { mono: true, fs: 13.5, fw: 700, fill: C.ink })
    + tx(x + w / 2 + 3, y + 38, sub, { mono: true, fs: 12, fw: 500, fill: C.muted }) + `</g>`;
}

// A small variable-name chip (rounded square with a monospace letter).
export function chip(cx, cy, t) {
  const s = 46;
  return rc(cx - s / 2, cy - s / 2, s, TOKENS.accents.purple.tint, C.purple, false)
    + tx(cx, cy, t, { mono: true, fs: 20, fw: 700, fill: TOKENS.accents.purple.dark });
}

// Wrap inner markup in an <svg> with a viewBox. `class="dgm-svg"` is what
// the deck styles with width:100%. render.mjs rewrites this opening tag
// for standalone files (adds xmlns + width/height).
export const svg = (w, h, inner) => `<svg viewBox="0 0 ${w} ${h}" class="dgm-svg">${inner}</svg>`;
