#!/usr/bin/env node
// Build, visually audit (for text-graph intersections, box-border overlaps,
// text-on-text collisions, and viewBox clipping), and outline all diagrams
// exported by one or more diagram modules (e.g. diagrams/ch02-linear-regression.mjs).
//
// Usage (from anywhere; the repo root is this script's parent directory,
// or $D2L_REPO):
//   node tools/build_module_diagrams.mjs diagrams/ch02-linear-regression.mjs
//
// Needs the `playwright` (or `playwright-core`) npm package with a Chromium
// it has installed.  Without a global one it uses the playwright-core driver
// that ships inside the Python playwright wheel in .venv-build (or the
// package directory named by $PLAYWRIGHT_CORE).  Python: $D2L_PYTHON, else
// .venv-pytorch/bin/python (fontTools + PIL), else python3.

import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);

const REPO = process.env.D2L_REPO
  || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const AUTO_DIR = path.join(REPO, 'img', 'auto');
const PY = process.env.D2L_PYTHON
  || [path.join(REPO, '.venv-pytorch', 'bin', 'python')].find((p) => fs.existsSync(p))
  || 'python3';

function loadChromium() {
  const tries = ['playwright', 'playwright-core'];
  if (process.env.PLAYWRIGHT_CORE) tries.unshift(process.env.PLAYWRIGHT_CORE);
  const lib = path.join(REPO, '.venv-build', 'lib');
  if (fs.existsSync(lib)) {
    for (const py of fs.readdirSync(lib).sort()) {
      tries.push(path.join(lib, py, 'site-packages', 'playwright', 'driver', 'package'));
    }
  }
  for (const t of tries) {
    try { return require(t).chromium; } catch (e) { /* next */ }
  }
  throw new Error(`no playwright found (tried ${tries.join(', ')})`);
}
const chromium = loadChromium();

const FONT_IMPORT = "@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Source+Sans+3:wght@400;500;600;700&display=swap');";

function standalone(inner) {
  const m = inner.match(/^<svg viewBox="0 0 ([\d.]+) ([\d.]+)"[^>]*>/);
  if (!m) throw new Error('diagram did not start with a viewBox <svg>');
  const [W, H] = [m[1], m[2]];
  const body = inner.replace(/^<svg[^>]*>/, '').replace(/<\/svg>\s*$/, '');
  return `<?xml version="1.0" encoding="UTF-8"?>\n`
    + `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">`
    + `<style><![CDATA[${FONT_IMPORT}]]></style>`
    + body + `</svg>\n`;
}

// Flags: --ids=a,b,c renders (checks, writes) only those ids of the modules;
// --warn reports visual defects but still writes the files.  The check runs
// on live text in this host's Chromium with whatever fonts it finds, so its
// verdict on an untouched diagram can differ between hosts; use --warn to
// regenerate a module after an outliner-only change.
const argv = process.argv.slice(2);
const onlyIds = new Set(argv.filter((a) => a.startsWith('--ids='))
  .flatMap((a) => a.slice(6).split(',')).filter(Boolean));
const warnOnly = argv.includes('--warn');
const modPaths = argv.filter((a) => !a.startsWith('--'));
if (modPaths.length === 0) {
  console.error('Usage: node tools/build_module_diagrams.mjs [--ids=a,b] [--warn] <diagrams/module.mjs> [...]');
  process.exit(2);
}

const allEntries = [];
for (const mp of modPaths) {
  const abs = path.resolve(REPO, mp);
  const mod = await import(pathToFileURL(abs).href + `?t=${Date.now()}`);
  if (!mod.diagrams || typeof mod.diagrams !== 'object') {
    throw new Error(`${mp} does not export a 'diagrams' object`);
  }
  for (const [id, fn] of Object.entries(mod.diagrams)) {
    if (onlyIds.size && !onlyIds.has(id)) continue;
    allEntries.push([id, fn]);
  }
}

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });

let totalDefects = 0;
const renderedMap = new Map();

for (const [id, fn] of allEntries) {
  const svgStr = standalone(fn());
  renderedMap.set(id, svgStr);

  const html = `<!doctype html><html><body style="margin:0;padding:20px;background:#fff;">${svgStr}</body></html>`;
  await page.setContent(html, { waitUntil: 'load' });

  const defects = await page.evaluate(() => {
    const svg = document.querySelector('svg');
    if (!svg) return ['no-svg'];
    const svgRect = svg.getBoundingClientRect();
    const issues = [];

    const allEls = Array.from(svg.querySelectorAll('*'));
    const domOrder = new Map(allEls.map((el, i) => [el, i]));

    const texts = Array.from(svg.querySelectorAll('text')).map((el) => {
      const r = el.getBoundingClientRect();
      const bbox = el.getBBox();
      const ctm = el.getScreenCTM();
      const invCtm = ctm ? ctm.inverse() : null;
      return {
        el,
        order: domOrder.get(el) || 0,
        text: (el.textContent || '').trim(),
        left: r.left + 1.0,
        right: r.right - 1.0,
        top: r.top + 2.0,
        bottom: r.bottom - 2.0,
        raw: r,
        bbox,
        invCtm,
      };
    }).filter(t => t.text.length > 0 && t.raw.width > 1 && t.raw.height > 1);

    // 1. Check text outside SVG viewBox
    for (const t of texts) {
      if (
        t.raw.left < svgRect.left - 1.5 ||
        t.raw.right > svgRect.right + 1.5 ||
        t.raw.top < svgRect.top - 1.5 ||
        t.raw.bottom > svgRect.bottom + 1.5
      ) {
        issues.push(`text-outside-viewbox("${t.text}")`);
      }
    }

    // 2. Check text-vs-text overlap
    for (let i = 0; i < texts.length; i++) {
      for (let j = i + 1; j < texts.length; j++) {
        const a = texts[i], b = texts[j];
        if (a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top) {
          issues.push(`text-overlap("${a.text}" <-> "${b.text}")`);
        }
      }
    }

    // 3. Check <rect> border edges cutting through <text>
    const rects = Array.from(svg.querySelectorAll('rect')).map(el => ({
      el,
      order: domOrder.get(el) || 0,
      stroke: el.getAttribute('stroke'),
      fill: el.getAttribute('fill'),
      r: el.getBoundingClientRect(),
    }));

    for (const rc of rects) {
      if (!rc.stroke || rc.stroke === 'none') continue;
      const r = rc.r;
      if (r.width < 4 || r.height < 4) continue;
      for (const t of texts) {
        // Ignore a rect that is painted BEFORE another opaque rect that fully covers this text
        const coveredByLaterRect = rects.some(other =>
          other.order > rc.order &&
          other.order < t.order &&
          other.fill && other.fill !== 'none' &&
          other.r.left <= t.raw.left + 1 &&
          other.r.right >= t.raw.right - 1 &&
          other.r.top <= t.raw.top + 1 &&
          other.r.bottom >= t.raw.bottom - 1
        );
        if (coveredByLaterRect) continue;

        const vOverlap = t.bottom > r.top + 1 && t.top < r.bottom - 1;
        if (vOverlap) {
          if (r.left > t.left && r.left < t.right) issues.push(`rect-left-edge-cuts-text("${t.text}")`);
          if (r.right > t.left && r.right < t.right) issues.push(`rect-right-edge-cuts-text("${t.text}")`);
        }
        const hOverlap = t.right > r.left + 1 && t.left < r.right - 1;
        if (hOverlap) {
          if (r.top > t.top && r.top < t.bottom) issues.push(`rect-top-edge-cuts-text("${t.text}")`);
          if (r.bottom > t.top && r.bottom < t.bottom) issues.push(`rect-bottom-edge-cuts-text("${t.text}")`);
        }
      }
    }

    // 4. Check curves/lines/arrows/circles/ellipses intersecting <text>
    const geomEls = Array.from(svg.querySelectorAll('line, path, polyline, circle, ellipse'));
    for (const g of geomEls) {
      if (typeof g.getTotalLength !== 'function') continue;
      let len = 0;
      try { len = g.getTotalLength(); } catch (e) { continue; }
      if (!len || len <= 0) continue;
      const ctm = g.getScreenCTM();
      if (!ctm) continue;
      const gOrder = domOrder.get(g) || 0;
      const steps = Math.max(10, Math.ceil(len / 2.5));
      const hitTexts = new Set();
      for (let s = 0; s <= steps; s++) {
        const pt = g.getPointAtLength((s / steps) * len).matrixTransform(ctm);
        for (const t of texts) {
          let inside = false;
          if (t.invCtm) {
            const loc = pt.matrixTransform(t.invCtm);
            inside = (
              loc.x >= t.bbox.x + 1.0 &&
              loc.x <= t.bbox.x + t.bbox.width - 1.0 &&
              loc.y >= t.bbox.y + 2.0 &&
              loc.y <= t.bbox.y + t.bbox.height - 2.0
            );
          } else {
            inside = (pt.x >= t.left && pt.x <= t.right && pt.y >= t.top && pt.y <= t.bottom);
          }
          if (inside) {
            // Check if an opaque rect between g and t covers this point
            const occluded = rects.some(rc =>
              rc.order > gOrder &&
              rc.order < t.order &&
              rc.fill && rc.fill !== 'none' &&
              pt.x >= rc.r.left && pt.x <= rc.r.right &&
              pt.y >= rc.r.top && pt.y <= rc.r.bottom
            );
            if (!occluded) hitTexts.add(t.text);
          }
        }
      }
      for (const ht of hitTexts) {
        issues.push(`${g.tagName}-intersects-text("${ht}")`);
      }
    }

    return Array.from(new Set(issues));
  });

  if (defects.length > 0) {
    totalDefects += defects.length;
    console.log(`DEFECTS in ${id}.svg (${defects.length}):`);
    for (const d of defects) console.log(`  - ${d}`);
  }
}

await browser.close();

if (totalDefects > 0 && warnOnly) {
  console.error(`\nWARNING: ${totalDefects} visual defect(s) across ${allEntries.length} diagram(s); writing anyway (--warn).`);
} else if (totalDefects > 0) {
  console.error(`\nFAILED: ${totalDefects} visual defect(s) across ${allEntries.length} diagram(s). Fix coordinates before outlining.`);
  process.exit(1);
}

// Write to img/auto/<id>.svg and run outline_svg.py on ONLY these files.
// A file the legacy remake_ch02.py pipeline tight-cropped (its committed
// viewBox differs from the fresh render's) is cropped again, so a re-run
// reproduces the committed tree byte for byte; a book copy img/<id>.svg,
// where one exists, is refreshed from the result.
const viewBox = (s) => (s.match(/<svg\b[^>]*\sviewBox="([^"]*)"/) || [])[1];
fs.mkdirSync(AUTO_DIR, { recursive: true });
const writtenPaths = [];
const cropPaths = [];
for (const [id, svgStr] of renderedMap.entries()) {
  const outPath = path.join(AUTO_DIR, `${id}.svg`);
  if (fs.existsSync(outPath)
      && viewBox(fs.readFileSync(outPath, 'utf8')) !== viewBox(svgStr)) {
    cropPaths.push(outPath);
  }
  fs.writeFileSync(outPath, svgStr);
  writtenPaths.push(outPath);
}

const runPy = (script, files) => execFileSync(
  PY, [path.join(REPO, 'tools', 'figstyle', script), ...files],
  { cwd: REPO, stdio: 'inherit' });
if (writtenPaths.length > 0) runPy('outline_svg.py', writtenPaths);
if (cropPaths.length > 0) runPy('tightcrop.py', cropPaths);
let copied = 0;
for (const p of writtenPaths) {
  const book = path.join(REPO, 'img', path.basename(p));
  if (fs.existsSync(book)) { fs.copyFileSync(p, book); copied++; }
}

console.log(`\nSUCCESS: Rendered, verified (${totalDefects} visual defects), and outlined ${writtenPaths.length} diagram(s)`
  + ` (${cropPaths.length} re-cropped, ${copied} book copies refreshed).`);
