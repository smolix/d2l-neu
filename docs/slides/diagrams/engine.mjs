// d2l slide-diagram engine — pure, DOM-free SVG string builders.
//
// Everything here returns a plain string. There is no `document`, no
// browser API: `node` alone can produce the SVG (see render.mjs). The
// same functions are what the north-star deck embeds inline at runtime,
// so a diagram looks identical whether inlined in a slide or written to
// a static file in img/auto/.
//
// COLOR TOKENS must stay in sync with _d2l-slides.scss / _d2l-theme.scss.
// If the book palette changes, change it here too (or, better, generate
// this object from the scss tokens — see HANDOFF.md "Theming").

export const C = {
  blue:'#2196F3', lblue:'#E3F2FD', amber:'#FB8C00', lamber:'#FFE7C7',
  green:'#43A047', lgreen:'#D7EFD8', purple:'#7D12BA', gray:'#90A4AE',
  lgray:'#ECEFF1', ink:'#15181C', muted:'#6B7280'
};

// Font stacks. Both end in a generic family so a static SVG still reads
// correctly even where "Source Sans 3" / "JetBrains Mono" aren't loaded
// (see HANDOFF.md "Fonts in static SVG" for the fidelity caveat).
export const FM = 'JetBrains Mono, monospace';
export const FS = 'Source Sans 3, sans-serif';

// A rounded square cell.
export function rc(x, y, s, f, st, dash) {
  return `<rect x="${x}" y="${y}" width="${s}" height="${s}" rx="6" fill="${f}" stroke="${st}" stroke-width="2"${dash ? ' stroke-dasharray="5 4"' : ''}/>`;
}

// A centered text label. opts: {mono, fs, fw, fill, anchor, base}
export function tx(x, y, t, o) {
  o = o || {};
  return `<text x="${x}" y="${y}" font-family="${o.mono ? FM : FS}" font-size="${o.fs || 15}" font-weight="${o.fw || 600}" fill="${o.fill || C.ink}" text-anchor="${o.anchor || 'middle'}" dominant-baseline="${o.base || 'central'}">${t}</text>`;
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
  const ang = Math.atan2(y2 - y1, x2 - x1), h = 8;
  const a1 = ang + Math.PI - 0.42, a2 = ang + Math.PI + 0.42;
  const hx1 = x2 + h * Math.cos(a1), hy1 = y2 + h * Math.sin(a1);
  const hx2 = x2 + h * Math.cos(a2), hy2 = y2 + h * Math.sin(a2);
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="2.3"${dash ? ' stroke-dasharray="5 4"' : ''}/>`
       + `<path d="M${x2},${y2} L${hx1},${hy1} L${hx2},${hy2} Z" fill="${color}"/>`;
}

// A labelled box (variable address / handle). title + sub stacked, with a
// colored accent rail on the left. faded=true dims it (e.g. an orphaned buffer).
export function block(x, y, title, sub, faded, accent) {
  const w = 178, h = 54, op = faded ? 0.42 : 1;
  accent = accent || C.blue;
  return `<g opacity="${op}"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="#fff" stroke="${accent}" stroke-width="2"/>`
    + `<rect x="${x}" y="${y}" width="6" height="${h}" rx="3" fill="${accent}"/>`
    + tx(x + w / 2 + 3, y + 19, title, { mono: true, fs: 13.5, fw: 700, fill: '#37474F' })
    + tx(x + w / 2 + 3, y + 38, sub, { mono: true, fs: 11.5, fw: 500, fill: '#78909C' }) + `</g>`;
}

// A small variable-name chip (rounded square with a monospace letter).
export function chip(cx, cy, t) {
  const s = 46;
  return rc(cx - s / 2, cy - s / 2, s, '#F3E5F5', C.purple, false)
    + tx(cx, cy, t, { mono: true, fs: 20, fw: 700, fill: C.purple });
}

// Wrap inner markup in an <svg> with a viewBox. `class="dgm-svg"` is what
// the deck styles with width:100%. render.mjs rewrites this opening tag
// for standalone files (adds xmlns + width/height).
export const svg = (w, h, inner) => `<svg viewBox="0 0 ${w} ${h}" class="dgm-svg">${inner}</svg>`;
