# Diagram engine

Pure, DOM-free SVG builders. A diagram is a JS function returning an `<svg>`
string; `node render.mjs` writes it to a standalone `.svg`. No browser.

```
engine.mjs     helpers + color tokens (C)
ndarray.mjs    the 8 §2.1 diagrams — read these as worked examples
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
| `tx(x,y,text,opts?)` | centered text. `opts`: `{mono,fs,fw,fill,anchor,base}` |
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

## Add a diagram

1. Open the chapter module (or copy `ndarray.mjs` → `<file>.mjs`).
2. Write a function returning `svg(W, H, parts)`. Build `parts` by
   concatenating helper calls. Lay out with explicit coordinates; reuse
   `grid`/`arrow`/`block`/`chip`.
3. Export it under a **stable id**: `'<file>-<concept>'`. The id is the
   filename and the slide reference — **never rename it**.
4. If new module: import + spread it in `registry.mjs`.
5. Render: `node render.mjs --out <repo>/img/auto <id>` and check it in
   `contact-sheet.html` (add an `<img>` row) or open the `.svg`.

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
slide (`@fig:<id>@jax`, or a `.only/.except`-scoped slide). See HANDOFF.md §6.

## Value-bound vs structural

Most diagrams here are **structural** — they draw shapes/relationships and never
change when a notebook re-runs. A few could be **value-bound** (drawing actual
numbers). For those, read the values from the executed notebook by cell id at
render time so a re-run regenerates them. See HANDOFF.md §9.
