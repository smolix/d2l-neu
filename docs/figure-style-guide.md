# d2l figure style guide

Status: **proposal 2026-08-03** — unifies the corpus's six divergent figure
generations around one token system. Extends (and does not contradict) the
approved architecture-diagram guide `docs/convnet-rewrite/figure-style.md`;
that guide's visual grammar is incorporated unchanged in §7.1.

Scope: **generated/illustrative figures** — the committed SVGs in `img/`
drawn by `tools/gen_*` scripts and the `diagrams/` JS pipeline. Notebook
execution outputs (`d2l.plot` results in `outputs/`) are out of scope, though
§6 applies to them aspirationally.

Implementation: every value named here lives in **`tools/figstyle/tokens.py`**
(exported to `tools/figstyle/tokens.json` and `diagrams/tokens.mjs` by
`python3 -m figstyle.export`). Do not copy hex codes into generator scripts;
import them.

---

## 1. Why: what the audit found

A 2026-08 audit of the 440 SVGs in `img/` found six coexisting style
generations produced by four unrelated toolchains, with:

- **6 different "d2l blue"s** (`#66BFFF`, `#519BF7`, `#6AC0FC`, `#1f77b4`,
  `#0B6BB2`, `#2196F3`) and 3 different "black"s;
- **4 mutually exclusive typography systems** (anonymous Cairo glyphs;
  DejaVu Sans; DejaVu + Computer Modern; Source Sans 3 + Inconsolata);
- stroke widths from 0.8 to 3 with no convention, 5 unrelated pastel pairs,
  arrow spaghetti (32 arrowheads in `mlp.svg` saying what one label could),
  and canvases up to 4× larger than their content
  (`mdl-transformers-block-anatomy.svg`).

The fix is not another style — it is **one token layer** that all pipelines
read, plus the rules below for spending those tokens.

## 2. Design principles

Distilled from the three exemplar systems we analyzed (Distill's momentum and
GNN articles, measured from their live DOM; Anthropic research figures,
pixel-sampled against their published CSS tokens; Raschka's 93-diagram LLM
architecture gallery) and from what already works in this repo:

1. **Structure is neutral; color is meaning.** Axes, grids, containers,
   connectors, and construction lines are grays (`INK`/`MUTED`/`FAINT`/
   `HAIRLINE`). Saturated color is spent only on elements that carry the
   figure's point. If everything is colored, nothing is.
2. **One semantic palette across the whole book.** A color means the same
   thing in chapter 3 and chapter 11 (§4.3). Readers learn the code once.
   This is the single mechanism that makes 400 figures feel like one system
   (it is what unifies Raschka's 93 diagrams).
3. **Color comes in trios.** Every accent is `tint` (area wash), `base`
   (stroke/arrow/line), `dark` (colored text). A colored region is a light
   wash with a readable outline — never a saturated slab with white text.
4. **Emphasis by weight before hue.** The important path gets `SW_HEAVY` (or
   a tint under-halo), not a new color. Hover-style emphasis in Distill is
   stroke-width 3 vs 1, same hue.
5. **Solid vs. dashed is semantics, not decoration.** Solid = real structure
   and data flow. Dashed (`DASH_SOFT`) = optional, reference, or candidate
   structure. Dotted (`DOT_LEADER`) = annotation leaders only, never flow.
6. **Two arrowheads, two jobs.** Data flow: small filled triangle. Annotation
   leader: light open chevron ("a pointer, not a stop sign" — Distill uses
   open carets for all annotation arrows).
7. **Figures are typeset in the site's own fonts.** Source Sans 3 for labels,
   Inconsolata for tokens/code/shapes, Computer Modern math for formulas
   (matches MathJax in the prose). Labels are quiet: small, often `MUTED`,
   bold only for short emphasis spans — never bold whole sentences.
8. **Flat.** No gradients, no drop shadows, no 3-D bevels. Rounded corners in
   exactly two tiers (§5.3).
9. **Whitespace does the grouping.** Prefer spacing over boxes; prefer boxes
   over lines; never draw a separator you can replace with a gap.
10. **The canvas is tight.** An SVG's viewBox hugs its content (+`PAD_CANVAS`).
    Whitespace around a figure belongs to the page, not the file.

## 3. Anti-patterns (all present in the current corpus — do not readd)

- matplotlib defaults leaking through: tab10 colors, DejaVu Sans anywhere.
- Pure `#000000` text or strokes (use `INK = #15181C`).
- Dropping semantics when restyling: arrowheads that carry direction, edges
  that carry dependencies, and the figure's *point* (e.g. non-monotonicity
  in `functionclasses.svg`) must survive a remake exactly. In dense graphs,
  keep the heads but render edges in `FAINT` so nodes stay the figure.
  Before redrawing any figure, state in one sentence what it must prove.
- Saturated fills behind black text (fails contrast, looks 2005).
- Inventing a pastel because the palette "doesn't have one for this" — pick
  the semantic role, or use `SLATE`.
- Labels touching lines, arrowheads, borders, or other labels (standing repo
  rule; unchanged).
- Bespoke figure widths per figure; snap to §6's canonical widths.
- Panoramic canvases: long arrows between adjacent concepts, or a figure
  wider than the text column for single-row content — the page scales it
  down and the type shrinks with it.
- Non-deterministic SVG output (timestamps, random ids) — breaks the
  byte-idempotent regeneration workflow.

## 4. Color

### 4.1 Neutrals

| Token | Hex | Use |
|---|---|---|
| `INK` | `#15181C` | primary label text, structural strokes (= site `$ink`) |
| `MUTED` | `#565D66` | secondary labels, units, axis ticks (site `$ink-3` darkened one step for mobile legibility) |
| `FAINT` | `#8F98A1` | construction lines, grids, ghosted structure |
| `HAIRLINE` | `#D9DEE4` | separators, neutral panel borders |
| `PANEL` | `#F0F2F5` | generic grouping-panel wash |
| `PAPER` | `#FFFFFF` | canvas — identical to the page background |
| `NOVELTY_FILL` | `#3B3B3B` | architecture "novelty box" (white text) |
| `CONTAINER_FILL` | `#E4E4E4` | outermost network container (architecture) |
| `INSET_FILL` | `#ECECEC` | dashed zoom-in insets (architecture) |

### 4.2 Accent trios

Each accent is a trio; the enforcement script
`tools/figstyle/check_contrast.py` verifies: `dark` ≥ 4.5:1 on white **and**
on its own tint (WCAG AA text); `base` ≥ 4.3:1 on white (raised from 3:1 —
figures render at 50–70% scale on phones, so strokes need text-grade
contrast); `INK` ≥ 4.5:1 on every tint; adjacent cycle colors ≥ 20 ΔE. Run
it after any edit.

| Accent | tint | base | dark |
|---|---|---|---|
| blue | `#D6EDFC` | `#1565C0` | `#094F86` |
| orange | `#FBE8D3` | `#B45309` | `#7F3B06` |
| green | `#DDF0DD` | `#2E7D3E` | `#1F5C2C` |
| purple | `#EDE9F8` | `#6C51B4` | `#503B8C` |
| red | `#FBE7E4` | `#C03B2F` | `#93261C` |
| teal | `#DEF1EF` | `#1F7D74` | `#145A53` |
| gold | `#FBF2D4` | `#9A7212` | `#6E5108` |
| slate | `#F0F2F5` | `#565D66` | `#15181C` |

Provenance: orange base is the approved gallery amber `#B45309` verbatim,
and its tint `#FBE8D3` likewise; blue tint is one step lighter than the
gallery's `#CDE8FA` (contrast on tint), and blue base/dark sit one step
darker than the site link blue / gallery accent after the 2026-08-03
mobile-contrast review; green tint is the `#ddf0dd` already used by the
transformer-anatomy figures.

Deviation log:
- 2026-08-03 (Alex's review, two rounds): all `base` colors darkened to
  ≥ 4.3:1, MUTED and FAINT darkened, type scale raised twice (labels 13 →
  15 → 17), stroke ladder raised twice (default line 1.6 → 2.0 → 2.5) —
  readability on mobile takes precedence over the earlier, lighter values.

Categorical series order (matplotlib `prop_cycle`):
blue, orange, green, purple, red, teal, gold, gray.

### 4.3 Semantic roles — one meaning, one color, everywhere

| Role | Accent | Covers |
|---|---|---|
| `stream` | blue | residual stream, main data flow, activations, encoder/decoder cores |
| `attention` | orange | attention blocks, scores, heads, QK products |
| `ffn` | green | feed-forward/MLP blocks, elementwise transforms, outputs |
| `norm` | slate (neutral) | normalization and bookkeeping ops |
| `embed` | purple | embeddings, vocab tables, memory, KV-cache |
| `state` | teal | recurrent/hidden state carried across time |
| `grad` | red | gradients, losses, backward pass, errors |
| `highlight` | gold | "look here" wash behind the element under discussion |

These continue conventions the corpus already teaches (blue stream / orange
attention / green FFN in the transformer figures). A figure normally uses
**at most two roles** plus neutrals; three only when the subject genuinely
has three interacting parts. Architecture-gallery figures keep their stricter
"one accent per figure" rule (§7.1).

## 5. Typography, strokes, shapes

### 5.1 Fonts

- **Labels**: Source Sans 3 (the site's body font; bundled at
  `static/fonts/`). Italic = math-ish variables in diagrams (`x`, `hₜ`).
- **Tokens / code / tensor shapes / literal data**: Inconsolata. Monospace
  is a semantic signal ("this is the model's actual data"), as in
  Anthropic's figures — not a styling choice.
- **Formulas** (matplotlib figures): Computer Modern mathtext
  (`mathtext.fontset: cm`) — matches the MathJax math in surrounding prose.
- Text in committed SVGs is **outlined to paths** (matplotlib
  `svg.fonttype: path`; the composer outlines via fontTools). Rendering is
  then identical in every browser, in librsvg (`rsvg-convert`, the PDF
  pipeline), and in print, with no font installation anywhere. Add a
  `<desc>` for accessibility.

### 5.2 Type scale (px in SVG user units, = CSS px at display size)

| Token | Size | Use |
|---|---|---|
| `FS_TITLE` | 19 | per-panel titles "(a) post-LN" — sparingly |
| `FS_LABEL` | 17 | **default**: block labels, node names, axis labels |
| `FS_SMALL` | 14.5 | secondary annotations, edge labels, dims (`d = 512`) |
| `FS_TINY` | 13 | ALL-CAPS letterspaced (+0.08 em) group/stage labels, ticks |
| `FS_MIN` | 12 | hard floor — nothing renders smaller |

Bold: short emphasis spans and callout numbers only. ALL-CAPS + tracking is
the group-label voice (quiet, structural); it replaces bold group titles.

### 5.3 Strokes, radii, dashes, arrows

| Token | Value | Use |
|---|---|---|
| `SW_HAIR` | 1.5 | grids, construction, panel borders, leaders |
| `SW_BOX` | 2.0 | block/pill outlines (up from the gallery's 1.2, 2026-08-03) |
| `SW_LINE` | 2.5 | **default** edges, arrows, schematic lines |
| `SW_HEAVY` | 4.0 | the one emphasized path per figure |
| `SW_HALO` | 10 | tint-colored under-halo beneath an emphasized line |
| `R_BLOCK` | 6 | leaf blocks (ops, layers) |
| `R_PANEL` | 12 | grouping panels/containers (rounder = outer tier) |
| pill | h/2 | fully-rounded small ops (⊕, ⊗, tags) |
| `DASH_SOFT` | `4 3` | dashed: optional/reference structure |
| `DOT_LEADER` | `1.5 2.5` | dotted: annotation leaders only |
| `ARROW_L/W` | 9.5 × 7.8 | filled data-flow head (scales √ with stroke) |
| `CHEVRON_L` | 7.5 | open annotation chevron |

## 6. Layout, proportions, sizing

- **Canonical widths** (user units): `W_NARROW 420`, `W_MEDIUM 560`
  (default), `W_COLUMN 720`, `W_WIDE 960` (multi-panel; displayed at column
  width). Snap to one; do not hand-tune per figure. Emit **unitless**
  `width`/`height` equal to the viewBox (1 unit = 1 CSS px) — no `pt`
  suffixes (a `pt` width displays 33% larger than designed).
- **Chapter includes**: `![caption](../img/name.svg){#fig-name}` with no
  `width=` when the figure was designed at display size; if `width=` is
  given it must equal the SVG's intrinsic width. This is what keeps apparent
  label sizes consistent book-wide (the current corpus mixes 320–960px
  displays of similar figures, so identical labels render 9px in one figure
  and 16px in the next).
- **Size the canvas to the content, not the content to the canvas.** A
  connector between adjacent blocks needs ~40 px to read as an arrow;
  anything longer is dead air unless it spans a semantic distance (e.g.
  encoder → decoder). Once spacing minima are met, a figure that could be
  drawn smaller without violating them is too big — apparent type size on
  the page equals design size only while the intrinsic width stays under
  the text column, so compactness IS readability (Alex, 2026-08-03: the
  chapter-1 remakes were rejected for oversized canvases). Wire arrows from
  blocks' returned rects, not guessed coordinates.
- **Spacing grid**: `GAP = 8`; pad blocks `12×7`, panels `14`, canvas `6`.
  Blocks in one figure share width where aligned (comparisons share pill
  sizes, rhythm, and baselines — same scene, same scale, always).
- **Aspect**: whole figures wider than tall (test: ≤ 0.75 of display width
  in height); vertical architecture stacks are the exception.
- **Reading direction**: time and sequences flow left→right; network stacks
  flow bottom→top (architecture) — pick per figure kind and never mix.
- **Density**: one idea per panel; if a figure needs a paragraph to walk
  through, split it into (a)/(b) panels with shared scale. Legends are
  placed inline next to the thing they name, not in a legend box.

## 7. Figure kinds and their grammars

### 7.1 Architecture / dataflow diagrams

The approved gallery grammar (`docs/convnet-rewrite/figure-style.md` §2)
applies verbatim: vertical spine bottom→top; white pills `SW_BOX` outline;
nested containers (max depth 2) with `CONTAINER_FILL` outer / accent-tint
inner + repeat brace; **one accent per figure**; one `NOVELTY_FILL` box;
rectilinear skips into ⊕; dashed insets with dotted leaders; bold callouts
with accent numbers; gray shape annotations; monospace input anchors; no
titles. Only the token *sources* change (import from `figstyle.tokens`).

### 7.2 Concept/block diagrams (transformers, RNNs, systems `bg-*`)

Like 7.1 but horizontal or free-form, and **semantic roles (§4.3) replace
the one-accent rule**: blocks are `tint` fills with `base` outlines and
`INK` labels; the residual stream / main path is the heavy blue line;
elementwise ops are pills on the path; panel labels are `FS_TINY` caps.
Branch arrows may take the color of the block they serve (the wiring that
belongs to attention is orange). Build with `figstyle.svg.Figure`.

### 7.3 Mathematical illustrations (`mdl-*`)

matplotlib via `figstyle.mpl.use_style()`. Coordinate scaffolding
(`axis_cross`) is `MUTED`, grids `HAIRLINE`/`FAINT`, data vectors/curves in
cycle order with `dark` text labels; construction lines dashed `FAINT`;
the result/emphasis object gets `SW_HEAVY` or a `halo_line`. CM math for
all symbol labels. Never title panels the caption already names.

### 7.4 Mechanics grids (kernels, padding, im2col)

Unchanged family (white grids, light-blue shaded cells, thin strokes) —
now with `BLUE.tint` shading, `INK` strokes, `FS_SMALL` labels.

### 7.5 Data plots in figures

`figstyle.mpl.use_style()` covers them: no top/right spines, no legend
frame, `MUTED` ticks, cycle palette, `FS_TINY` ticks / `FS_LABEL` axes.

## 8. Pipelines and tools

| Tool | Role |
|---|---|
| `tools/figstyle/tokens.py` | **single source of truth** for every value above |
| `tools/figstyle/check_contrast.py` | palette invariants (run after edits) |
| `tools/figstyle/svg.py` | deterministic SVG composer: auto-sized semantic blocks, panels, data-flow arrows, leaders, pills, rich text (italic vars, sub/superscripts, mono) — text outlined via fontTools against the bundled TTFs; tight canvas; byte-idempotent |
| `tools/figstyle/textmetrics.py` | exact text measurement + outlining (no matplotlib needed) |
| `tools/figstyle/mpl.py` | `use_style()` rcParams theme + the primitives the old house style lacked (`box`, `halo_line`, `leader`, `panel_label`, refined `arrow`/`axis_cross`/`clean_axes`); legacy names `BLUE/ORANGE/GREEN/GRAY/LIGHT` re-exported onto the new palette for drop-in migration |
| `tools/figstyle/export.py` | writes `tokens.json` + `diagrams/tokens.mjs` (the JS engine imports `C` from there — ends the palette drift) |
| `tools/figstyle/demo_figures.py` | the four §9 pilots → `img/figstyle-demo/` |

Byte-idempotence is preserved everywhere: fixed `svg.hashsalt`, `Date: None`,
fixed float formatting, no generated ids. Workflow rule (unchanged, standing):
**render and look** — `rsvg-convert -z 2` after every edit; contact sheets
for batches; never reason about label collisions from code.

## 9. Pilots and migration

Pilots in `img/figstyle-demo/` (run `python3 tools/figstyle/demo_figures.py`
and `demo_figures2.py`): composer remakes `mlp`, `seq2seq-attention`,
`mdl-transformers-block-anatomy`, `qkv`, `attention`, `forward`, `book-org`,
`seq2seq-state`, `functionclasses`; matplotlib-theme remakes
`mdl-la-vector-add`, `mdl-la-angle`, `mdl-la-projection`, `mdl-la-span`,
`mdl-la-eig-ellipse`; plus `specimen.svg` (token reference card).

Suggested adoption order (each step is `gmake figures`-incremental and
independently shippable):

1. **Approve tokens** (this doc + pilots).
2. `tools/gen_mdl_figures.py`: adopt `figstyle.mpl.use_style()` and map its
   `BLUE/ORANGE/GREEN/GRAY/LIGHT` constants to the unified palette → every
   `gen_mdl_*` figure restyles in one regeneration.
3. `tools/arch_diagrams.py`: import its constants from `figstyle.tokens`
   (values are near-identical by construction; diffs are whitespace-level).
4. `diagrams/engine.mjs`: `import { C } from './tokens.mjs'`, delete its
   local palette.
5. Remake the worst legacy Gen A figures with the composer as chapters get
   touched (the pilot remakes show the pattern); delete orphans per the
   existing legacy policy.
6. New figures: composer for diagrams, `figstyle.mpl` for math — never raw
   rcParams, never a literal hex.
