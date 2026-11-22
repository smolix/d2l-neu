#!/usr/bin/env node
// Automated SVG Visual Defect Checker using Playwright's real SVGGeometryElement engine.
//
// Renders live-text SVGs from diagrams/registry.mjs (before outline_svg.py converts
// <text> to <path>) in headless Chromium and checks every diagram for 4 classes of
// visual defects:
//   1. graph/line/arrow/curve/circle/ellipse intersecting a <text> bounding box
//   2. <rect> border edge intersecting/clipping a <text> bounding box
//   3. <text> overlapping another <text> bounding box
//   4. any element extending outside the SVG viewBox

const { chromium } = require("playwright");
const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const REPO = "/Users/smola/Repositories/github/d2l-neu";
const NODE_BIN = process.execPath;
const TMP_SVG_DIR = "/tmp/d2l_svg_audit_live";

const filterIds = process.argv.slice(2);

(async () => {
  fs.mkdirSync(TMP_SVG_DIR, { recursive: true });
  const renderArgs = [path.join(REPO, "diagrams", "render.mjs"), "--out", TMP_SVG_DIR, ...filterIds];
  execFileSync(NODE_BIN, renderArgs, { cwd: REPO, stdio: "ignore" });

  const svgFiles = (filterIds.length > 0
    ? filterIds.map(id => `${id}.svg`)
    : fs.readdirSync(TMP_SVG_DIR).filter(f => f.endsWith(".svg"))
  ).sort();

  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });

  let totalDefects = 0;
  for (const file of svgFiles) {
    const svgPath = path.join(TMP_SVG_DIR, file);
    if (!fs.existsSync(svgPath)) continue;
    const svgContent = fs.readFileSync(svgPath, "utf8");
    const html = `<!doctype html><html><body style="margin:0;padding:20px;background:#fff;">${svgContent}</body></html>`;
    await page.setContent(html, { waitUntil: "load" });

    const defects = await page.evaluate(() => {
      const svg = document.querySelector("svg");
      if (!svg) return ["no-svg"];
      const svgRect = svg.getBoundingClientRect();
      const issues = [];

      const texts = Array.from(svg.querySelectorAll("text")).map((el, idx) => {
        const r = el.getBoundingClientRect();
        // Shrink by 1.5px vertically and 1px horizontally so normal baseline proximity doesn't false-trigger
        return {
          idx,
          text: (el.textContent || "").trim(),
          left: r.left + 1.0,
          right: r.right - 1.0,
          top: r.top + 2.0,
          bottom: r.bottom - 2.0,
          raw: r,
        };
      }).filter(t => t.text.length > 0 && t.raw.width > 1 && t.raw.height > 1);

      // 1. Check text outside SVG viewBox
      for (const t of texts) {
        if (
          t.raw.left < svgRect.left - 1 ||
          t.raw.right > svgRect.right + 1 ||
          t.raw.top < svgRect.top - 1 ||
          t.raw.bottom > svgRect.bottom + 1
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
      const rects = Array.from(svg.querySelectorAll("rect"));
      for (const rc of rects) {
        const st = rc.getAttribute("stroke");
        if (!st || st === "none") continue;
        const r = rc.getBoundingClientRect();
        if (r.width < 4 || r.height < 4) continue;
        for (const t of texts) {
          // Does the vertical left or right edge of rect pass through text horizontally while overlapping vertically?
          const vOverlap = t.bottom > r.top + 1 && t.top < r.bottom - 1;
          if (vOverlap) {
            if (r.left > t.left && r.left < t.right) {
              issues.push(`rect-left-edge-cuts-text("${t.text}")`);
            }
            if (r.right > t.left && r.right < t.right) {
              issues.push(`rect-right-edge-cuts-text("${t.text}")`);
            }
          }
          const hOverlap = t.right > r.left + 1 && t.left < r.right - 1;
          if (hOverlap) {
            if (r.top > t.top && r.top < t.bottom) {
              issues.push(`rect-top-edge-cuts-text("${t.text}")`);
            }
            if (r.bottom > t.top && r.bottom < t.bottom) {
              issues.push(`rect-bottom-edge-cuts-text("${t.text}")`);
            }
          }
        }
      }

      // 4. Check curves/lines/arrows/circles/ellipses intersecting <text>
      const geomEls = Array.from(svg.querySelectorAll("line, path, polyline, circle, ellipse"));
      for (const g of geomEls) {
        if (typeof g.getTotalLength !== "function") continue;
        let len = 0;
        try { len = g.getTotalLength(); } catch (e) { continue; }
        if (!len || len <= 0) continue;
        const ctm = g.getScreenCTM();
        if (!ctm) continue;
        const steps = Math.max(10, Math.ceil(len / 2.5));
        const hitTexts = new Set();
        for (let s = 0; s <= steps; s++) {
          const pt = g.getPointAtLength((s / steps) * len).matrixTransform(ctm);
          for (const t of texts) {
            if (pt.x >= t.left && pt.x <= t.right && pt.y >= t.top && pt.y <= t.bottom) {
              hitTexts.add(t.text);
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
      console.log(`DEFECTS in ${file} (${defects.length}):`);
      for (const d of defects) console.log(`  - ${d}`);
    }
  }

  await browser.close();
  console.log(`\nAudited ${svgFiles.length} SVGs: ${totalDefects} total visual defect(s).`);
  process.exit(totalDefects > 0 ? 1 : 0);
})();
