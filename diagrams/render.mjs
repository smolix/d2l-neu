// Render diagrams to standalone SVG files. No browser required.
//
//   node render.mjs                       # render all → ./out
//   node render.mjs --out ../../img/auto  # render all → repo img/auto
//   node render.mjs --out DIR id1 id2     # render only the listed ids
//   node render.mjs --list                # print known ids and exit
//
// In the d2l-neu repo, point --out at <repo>/img/auto. The slide build
// (or a future book-figure pass) references files as img/auto/<id>.svg.

import { diagrams } from './registry.mjs';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const args = process.argv.slice(2);
if (args.includes('--list')) {
  console.log(Object.keys(diagrams).join('\n'));
  process.exit(0);
}
let out = './out';
const ids = [];
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--out') out = args[++i];
  else ids.push(args[i]);
}
const targets = ids.length ? ids : Object.keys(diagrams);

// Turn the deck's inline <svg ... class="dgm-svg"> into a standalone file:
// add the XML namespace + explicit width/height (predictable sizing as an
// <img> and for rsvg-convert → PDF). A <style> @import helps font fidelity
// when the SVG is INLINED into an HTML page or opened directly in a
// browser; it is ignored for <img>-embedded SVG and by rsvg (those fall
// back to the generic families already in each text's font-family).
const FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');";

function standalone(inner) {
  const m = inner.match(/^<svg viewBox="0 0 ([\d.]+) ([\d.]+)"[^>]*>/);
  if (!m) throw new Error('diagram did not start with a viewBox <svg>');
  const [W, H] = [m[1], m[2]];
  const body = inner.replace(/^<svg[^>]*>/, '').replace(/<\/svg>\s*$/, '');
  return `<?xml version="1.0" encoding="UTF-8"?>\n`
    + `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`
    // CDATA so the `&` in the font-import URL stays valid XML — otherwise
    // the file is unparseable as a standalone SVG (e.g. an <img> book
    // figure, or rsvg-convert). Inlined-into-slides decks strip <style>.
    + `<style><![CDATA[${FONT_IMPORT}]]></style>`
    + body + `</svg>\n`;
}

mkdirSync(out, { recursive: true });
let n = 0;
for (const id of targets) {
  const fn = diagrams[id];
  if (!fn) { console.error(`!! unknown diagram id: ${id}`); process.exitCode = 1; continue; }
  const file = join(out, `${id}.svg`);
  const data = standalone(fn());
  writeFileSync(file, data);
  console.log(`  ${file}  (${data.length} bytes)`);
  n++;
}
console.log(`\nrendered ${n} diagram(s) → ${out}`);
