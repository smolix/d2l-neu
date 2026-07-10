# Architecture-diagram style guide ("gallery style")

Status: **approved direction, pending pilot review** (see §5). Applies to the
convnet chapter rewrite (chapters 7–8) and, going forward, to all *network
architecture / dataflow* diagrams in the book.

## 1. Where this style came from and where it applies

Alex reviewed the diagram style of Sebastian Raschka's LLM Architecture
Gallery (https://sebastianraschka.com/llm-architecture-gallery/) and wants our
architecture diagrams modeled on it. This guide is our own adaptation of that
style, not a copy: same visual grammar, adjusted for a printed book (captions
instead of titles), for convnets (spatial resolution matters), and for our
SVG→HTML/PDF pipeline.

Scope boundaries:

- **Gallery style (this guide):** network architecture and dataflow diagrams —
  blocks, stages, skip connections, whole-model overviews, module insets.
- **Mechanics diagrams** (grids showing how a kernel slides, padding, dilation,
  im2col): these stay in the existing chapter family look (plain white grids,
  light-blue shaded cells, thin black strokes, black labels — see
  `img/correlation.svg`, `img/conv-pad.svg`). New mechanics figures must match
  that family, drawn with the same generator library (§4) using its grid
  primitives.
- **Data plots** (loss curves, accuracy ablations): `d2l.plot` in notebook
  code, as everywhere else in the book.
- **Photographs and reproduced paper figures** (`filters.png`,
  `regnet-fig.png`, Waldo, Field 1987): keep as-is, never redraw.
- The matplotlib "house style" (`tools/gen_mdl_figures.py`) remains the style
  for *mathematical* illustrations (math appendix, ch6 `bg-*` figures). Do not
  use it for architecture diagrams anymore.

## 2. Visual grammar

Before drawing anything, download two references and look at them —
`https://sebastianraschka.com/llm-architecture-gallery/images/architectures/thumbnails/gpt-2-xl.webp`
and `.../llama-3-8b.webp` (do not commit them; they are copyrighted). The
grammar:

1. **Vertical spine, bottom→top.** Data enters at the bottom, predictions exit
   at the top. One main path. Solid black arrows (~1.2 pt) with small solid
   triangular heads.
2. **Ops are pills.** Each layer/op is a rounded rectangle ("pill"): white
   fill, thin black border (~1.2 pt), corner radius ≈ 40–50% of pill height,
   centered black sans-serif label. Label = op name + the hyperparameters that
   matter pedagogically: `3×3 Conv, 64, s=2`, `7×7 DWConv`, `LayerNorm`,
   `Linear (4096)`. Keep labels under ~28 characters; details go in callouts.
3. **Hierarchy via nested containers.** Outermost: light-gray rounded
   rectangle with generous padding = the whole network. Inside it, an
   **accent-tinted** rounded panel = the repeated unit (the block), marked at
   its lower-left with a repeat multiplier in accent color plus a curly brace:
   `8 ×  }`. Maximum nesting depth: 2 containers (network → block). Anything
   deeper becomes an inset (rule 7).
4. **One accent color per figure.** Default accent: d2l blue
   (`#66BFFF` panel tint at ~35% opacity, `#0B6BB2`-class saturated tone for
   text/numbers; the pilot fixes exact values). The accent is used for exactly
   three things: the repeated-block panel tint, key numbers inside callouts,
   and the highlighted keyword in the novelty box. Everything else is
   grayscale. A second accent is allowed only when a figure *contrasts two
   designs* (e.g. ResNet block vs. ConvNeXt block side by side).
5. **The novelty box.** The one op that is *new in this architecture* gets a
   near-black (`#3B3B3B`) fill with white text, its defining keyword in
   accent color (e.g. `Masked multi-head attention` with "multi-head" in
   accent; for us: `7×7 Depthwise Conv` with "Depthwise" in accent). Exactly
   one novelty box per figure. If you can't decide which op is the novelty,
   the figure is trying to say too much — split it.
6. **Residual / elementwise ops.** Skip connections are rectilinear black
   lines routed outside the pill column into a small ⊕ circle (white fill,
   black border, black plus) on the spine. Elementwise product: ⊗. Never
   diagonal skip lines.
7. **Insets.** To zoom into a pill's internals (inverted bottleneck, SE
   module, Inception branch structure), draw a dashed rounded box to the side
   with a light-gray fill, containing its own mini-spine, connected to the
   parent pill by a **dotted leader line**. One or two insets max.
8. **Callouts.** Quantitative facts live in the margins as bold black
   annotations with the numbers in accent color, connected by dotted leaders
   to what they describe: "Effective receptive field of **51×51**", "Only
   **1/8.6** of the multiplications of a dense 3×3 conv". Prefer a callout
   over cramming numbers into pill labels. 2–5 callouts per figure.
9. **Convnet-specific: resolution/channel annotations.** Where the spatial
   resolution changes, annotate the spine arrow with small gray italic
   `(c, r×r)` shape labels (the convention of the old `anynet.svg`). Strided
   ops state their stride in the pill label. Stage containers are labeled
   "Stage 1…4" in gray small caps at the container's top-left.
10. **No figure titles.** The book caption carries the title (house rule: no
    redundant supercaptions). Raschka's big bold title is replaced by the
    `![caption](...)` text. Per-panel labels in multi-panel figures are
    allowed when functional.
11. **Input anchor.** Where helpful, anchor the bottom of the spine with the
    input in monospace gray: `224×224×3 image`.
12. **Comparisons.** Side-by-side variants (block with/without projection
    shortcut; ResNet vs. ConvNeXt block) share pill sizes, vertical rhythm,
    and baselines. Same scene → same scale, always.

Typography: one sans-serif family throughout (the generator pins it and
renders text as paths, §4). Pill labels ~11–12 pt equivalent, callouts
~12–13 pt bold, container/stage labels ~10 pt. Nothing below 9 pt. The
text-collision rules from the repo figure checklist apply unchanged: no
label may touch a line, arrowhead, border, or another label.

## 3. Legacy figures

The old d2l architecture SVGs (`img/alexnet.svg`, `vgg.svg`, `nin.svg`,
`inception*.svg`, `resnet*.svg`, `resnext-block.svg`, `densenet*.svg`,
`anynet.svg`, `lenet.svg`, …) are pdf2svg exports with text as glyph paths —
there are no editable sources in the repo. They were *not* retired during the
ch6 rewrite (ch6 simply stopped referencing its share; several are now
orphans). Policy for this rewrite:

- Every architecture figure in a **rewritten or new section** is produced
  fresh in gallery style. Do not mix legacy and gallery diagrams in one
  section.
- Legacy SVGs that lose their last reference get deleted in the final cleanup
  commit (grep `chapter_*/*.md` for `img/<name>` before deleting; also check
  slide decks).
- `lenet.svg`/`lenet-vert.svg` (ch7, untouched section) may stay legacy for
  now; replacing them is optional polish at the end if time permits.

## 4. Implementation

- **Library:** new `tools/arch_diagrams.py` — a small primitive library:
  `pill(label, kind=...)`, `novelty(label, keyword)`, `container(children,
  tint)`, `repeat(container, n)`, `spine(arrows)`, `skip(from, to, op='+')`,
  `inset(parent_pill, children)`, `callout(text, accent_parts, target)`,
  `shape_note(arrow, c, r)`, plus `grid(...)` primitives for mechanics
  diagrams. Build it on matplotlib (FancyBboxPatch etc.) **reusing the
  byte-idempotent `save()` infrastructure** from `tools/gen_mdl_figures.py`
  (fixed `svg.hashsalt`, nulled date, fonts rendered as paths so macOS and the
  GPU box produce identical bytes and the PDF pipeline needs no fonts). The
  Raschka look is a product of exact colors, paddings, and corner radii — pin
  them once as module constants and never override per figure.
- **Generators:** `tools/gen_arch_convnets_figures.py` (ch7) and
  `tools/gen_arch_convmodern_figures.py` (ch8) → `img/arch-<id>.svg`.
  Byte-idempotent: a second run must produce identical bytes.
- **Render-and-review loop is mandatory** (the standing rule): after every
  edit, `rsvg-convert -z 2` the SVG to PNG and *look at it*; never reason
  about label positions from code. For batches, build a contact sheet
  (`montage`) and scan it.

## 5. Pilot gate — do this before mass production

Claude has struggled to match wished-for figure styles before. Therefore:

1. Implement `tools/arch_diagrams.py` and produce exactly **two pilot
   figures**: `arch-resnet-vs-convnext-block.svg` (the marquee two-accent
   comparison, exercising pills, novelty box, skip/⊕, insets, callouts) and
   `arch-inception-block.svg` (multi-branch layout).
2. Put them on a contact sheet next to two Raschka references and two legacy
   d2l figures, commit, push, and **ask Alex to review the style match before
   generating the remaining figures.** Iterate on the pilots until approved.
3. Only then produce the full figure manifests in the ch7/ch8 specs.
