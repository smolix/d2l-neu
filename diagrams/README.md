# Diagram engine

Pure, DOM-free SVG builders. A diagram is a JS function returning an `<svg>`
string; `node render.mjs` writes it to a standalone `.svg`. No browser.

```
engine.mjs     helpers + color tokens (C)
ndarray.mjs    the §1.1 diagrams (Data Manipulation) — read these as worked examples
registry.mjs   id → fn map; add a chapter module here
render.mjs     CLI: write img/auto/<id>.svg
out/           rendered golden SVGs
contact-sheet.html   eyeball all of them
```

## Coordinate model

Each diagram works in its own `viewBox` pixel space (e.g. `0 0 700 358`) and is
scaled by the slide/page. Place elements with absolute x/y. `y` grows downward.
Keep a diagram's aspect wide-ish (it sits in a slide column or full width).

## Helpers (`engine.mjs`)

| Helper | Returns |
|---|---|
| `rc(x,y,s,fill,stroke,dash?)` | a rounded `s×s` square cell |
| `tx(x,y,text,opts?)` | text centred vertically on `y`. `opts`: `{mono,fs,fw,fill,anchor,base}` (see *Text placement*) |
| `baselineY(y,fs,mono?,base?)` | the alphabetic-baseline y that `tx` uses; call it in any hand-written `<text>` |
| `grid(data,x0,y0,s,gap,opt?)` | a 2-D grid of cells from a 2-D array |
| `arrow(x1,y1,x2,y2,color,dash?)` | line + filled arrowhead (no `<marker>`) |
| `block(x,y,title,sub,faded?,accent?)` | labelled box w/ accent rail |
| `chip(cx,cy,text)` | small variable-name chip (centered on cx,cy) |
| `svg(w,h,inner)` | wrap inner markup in `<svg viewBox=…>` |
| `C` | color tokens — **use these, never raw hex** |

`grid`'s `opt.fill(r,c,v)` returns `[fillColor, strokeColor, dashBool,
textColor]`, so you can highlight cells by row/col/value. `opt.showVal=false`
draws empty cells (shape-only). `opt.fs` sets the value font size.

`C` keys: `blue lblue amber lamber green lgreen purple gray lgray ink muted`
(light variants are the `l*` ones). These mirror `_d2l-slides.scss`; keep in
sync.

## Text placement

Never emit `dominant-baseline`: librsvg (`rsvg-convert`, the PDF pipeline)
ignores it, so `central` text sat ~0.35 em too high in the PDF. `tx` writes
an alphabetic baseline (the SVG default, honoured everywhere) placed by
`baselineY`: `base` names what sits at `y` — `'central'` (default: the
centre of capitals and digits, half the cap height above the baseline;
0.33 em for Source Sans 3, 0.365 em for JetBrains Mono, `CAP_HEIGHT` in
`engine.mjs`), `'middle'` (centre of lowercase), `'hanging'` (top of
capitals), or `'alphabetic'` (`y` is the baseline; no shift). A module that
builds its own `<text>` (tspans for italics, sub/superscripts) passes its
`y` through `baselineY(y, fs, mono)` instead of setting a baseline
attribute.

## Pipeline: render → outline → crop → copy

`render.mjs` writes live `<text>` in 'Source Sans 3' / 'JetBrains Mono'.
The committed SVGs carry no live text: `tools/figstyle/outline_svg.py`
converts every `<text>` to glyph outlines from the bundled fonts in
`static/fonts/` (JetBrains Mono, which is not bundled, is drawn in
Inconsolata scaled to the same cap height), keeps each string as an
`aria-label` on its group, and drops the Google-Fonts `@import`. So the
figure looks the same in every browser, offline, and in the PDF. Then
`tools/figstyle/tightcrop.py` trims the canvas to the outlined glyphs, and
book-referenced ids (an `img/<id>.svg` exists) are copied to `img/`. Each
step is byte-idempotent; re-running any of them on a finished file is a
no-op.

One id (after editing its module), from the repo root:

```bash
.venv-pytorch/bin/python tools/figstyle/remake_ch02.py --js <id> [<id> ...]
```

which runs, per id:

```bash
node diagrams/render.mjs --out img/auto <id>
.venv-pytorch/bin/python tools/figstyle/outline_svg.py img/auto/<id>.svg
.venv-pytorch/bin/python tools/figstyle/tightcrop.py img/auto/<id>.svg
[ -f img/<id>.svg ] && cp img/auto/<id>.svg img/<id>.svg   # book copy, if referenced
```

(`.venv-pytorch` has fontTools and PIL; system `python3` usually does not.)
All ids: `remake_ch02.py --js` with no ids. A NEW book figure needs one
manual `cp img/auto/<id>.svg img/` first; after that the copy stays in sync.
Run `tightcrop.py` only on this pipeline's files: it is not meant for the
matplotlib SVGs that share `img/auto/`. Then render the PNG and look:
`rsvg-convert -z 2 img/auto/<id>.svg -o /tmp/<id>.png`.

`outline_svg.py` reports a character that no bundled font has and draws it
from matplotlib's DejaVu/STIX fonts (pinned with matplotlib); a character no
font has is an error. It refuses text features it cannot reproduce
(`textPath`, per-glyph `rotate`, `textLength`, vertical writing).

## Add a diagram

1. Open the chapter module (or copy `ndarray.mjs` → `<file>.mjs`).
2. Write a function returning `svg(W, H, parts)`. Build `parts` by
   concatenating helper calls. Lay out with explicit coordinates; reuse
   `grid`/`arrow`/`block`/`chip`.
3. Export it under a **stable id**: `'<file>-<concept>'`. The id is the
   filename and the slide reference — **never rename it**.
4. If new module: import + spread it in `registry.mjs`.
5. Render it through the pipeline above
   (`remake_ch02.py --js <id>`) and look at the PNG.

### Skeleton

```js
import { C, grid, tx, arrow, svg } from './engine.mjs';

function myThing() {
  const s = 44, g = 4; let o = '';
  o += grid([[0,1,2,3]], 40, 40, s, g);            // a row of cells
  o += tx(40 + 2*(s+g), 110, 'caption', { fs: 14, fill: C.muted });
  o += arrow(120, 70, 200, 70, C.purple);
  return svg(420, 140, o);
}

export const diagrams = { 'myfile-mything': myThing };
```

## Framework variants

When a concept differs by framework (e.g. JAX arrays are immutable, so the
in-place memory story changes), expose variants — either separate ids
(`ndarray-saving-memory`, `ndarray-saving-memory-jax`) or a function that
branches on a framework argument. Reference the right one per framework in the
slide (`@fig:<id>@jax`, or a `.only/.except`-scoped slide). See docs/slides-northstar-design.md §6.

## Value-bound vs structural

Most diagrams here are **structural** — they draw shapes/relationships and never
change when a notebook re-runs. A few could be **value-bound** (drawing actual
numbers). For those, read the values from the executed notebook by cell id at
render time so a re-run regenerates them. See docs/slides-northstar-design.md §9.
