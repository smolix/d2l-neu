# Authoring slides

Slides live alongside book content in the same `chapter_*/<file>.md`
source. The book and the slide deck are different views of the same
file: book prose stays in the body of the file, slide content lives
in a `<!-- slides -->` section near the bottom, and code cells are
referenced from slides by stable `#<id>` anchors so they appear in
both places without duplication.

This doc covers:
- the slide syntax,
- cell-ID placeholders,
- output injection,
- the VS Code workflow,
- building, previewing, and troubleshooting.

For the design rationale, see `PLAN-slides-and-notebook-editing.md`.
For the source-directive reference, see `docs/syntax.md`.

> **Teaching first — intuition and diagrams before walls of words or code.**
> This is the governing rule for every deck. Each slide teaches *one* idea
> and leads with a picture of it: a diagram beats a paragraph, one
> illustrative line of code beats a dump, and the output you show is the
> point (not a log). If a slide is mostly prose or mostly code, find the
> figure and cut the rest — the deck teaches the *idea*, the notebook holds
> the full code. The reference decks (§2.1 `ndarray`, §2.3 `linear-algebra`,
> §2.4 `calculus`; §2.5 `autograd` next) and the diagram-led *Quality rules*
> below are how this gets applied.

---

## Quick start

```markdown
# My Chapter

Some prose for the book.

```{.python .input #intro-import}
import torch
```

More prose. Code cell at `intro-import` is shown in the book.

## A section

```{.python .input #intro-vec-loop}
c = [a[i] + b[i] for i in range(n)]
```

```{.python .input #intro-vec-add}
c = a + b
```

<!-- slides -->

# My Chapter

::: {.slide title="Vectorize"}
A naïve Python loop pays interpreter overhead per element:

@intro-vec-loop

. . .

A vectorized call hands the work to a C kernel:

@intro-vec-add
:::
```

The book renderer (`d2l_preprocess.py`) strips the `<!-- slides -->`
block before generating the chapter `.qmd`, so readers see only the
top half. The slide builder (`gen_slides.py`) reads only the bottom
half plus the code cells it references.

---

## The slide div

```markdown
::: {.slide title="…" layout="…" transition="…"}
slide content
:::
```

Attributes:

| Attribute              | Purpose                                                |
|------------------------|--------------------------------------------------------|
| `title="…"`            | Optional. Renders as a `## Title` slide heading.       |
| `only="fw[,fw]"`       | Render this slide **only** for the listed framework decks. For sections whose *framing* (not just code) differs per framework. |
| `except="fw[,fw]"`     | Render for all decks **except** the listed frameworks. |
| `layout="…"`           | Legacy section class (`2col`/`figure`/`code` → `.slide-<L>`). Prefer body-level `::: {.cols}` (see *Two-column layout* below). |
| `transition="…"`       | Reveal.js transition. `fade`, `slide`, `convex`, `concave`, `zoom`, `none`. |
| `data-background-color="#…"`   | Slide background color.                        |
| `data-background-image="…"`    | Slide background image.                        |
| `output-lines="N"`     | Override the per-deck text-output cap (default 12) for cells injected into this slide. |

`only=` / `except=` let one shared `<!-- slides -->` block emit *different
slides per framework* where the concept itself diverges (e.g. JAX arrays
are immutable, so its "writing" slide is genuinely different code **and**
prose, not just a code swap). Most cells differ only in code/output and
need no scoping — the `#@tab` mechanism handles those automatically.

Slides without a `title=` attribute render with a `---` separator and
no heading — useful for continuation slides under the same section.

The first slide in a file inherits the file's H1 as the deck title;
no `title=` needed.

### Sub-slides (vertical)

```markdown
::: {.slide title="Profiling"}
@profile-overview

::: {.subslide title="Memory layout"}
@profile-memory
:::

::: {.subslide title="Cache misses"}
@profile-cache
:::
:::
```

Sub-slides become vertical slides under the parent in Reveal.js.
Press <kbd>↓</kbd> in the deck to navigate into them.

### Fragments

Quarto's reveal.js syntax passes through verbatim:

```markdown
::: {.slide title="…"}
First point.

. . .

Second point appears after one click.

::: {.fragment .fade-up}
Third point fades up.
:::

[inline highlight]{.fragment .highlight-blue}
:::
```

Snippets (`frag` in VS Code) cover the common cases.

### Visual vocabulary (the north-star building blocks)

The §2.1 (`ndarray`) and §2.3 (`linear-algebra`) decks established a small
set of authored classes, all styled in `_d2l-slides.scss`. Compose them
in the `<!-- slides -->` block:

| Markup | Renders as | Use |
|--------|-----------|-----|
| `[Section]{.kicker}` | small uppercase eyebrow label | one per content slide, under the title |
| `::: {.cover}` … `:::` | centered title-slide treatment | the first slide (deck H1 supplies the title) |
| `::: {.divider}` + `[01]{.dnum}` `[Title]{.dtitle}` `[sub]{.dsub}` | centered section divider | between section groups |
| `::: {.cols}` / `::: {.col}` | flex two-column row | code/prose beside a figure |
| `::: {.d2l-note}` / `{.d2l-note .rule}` / `{.d2l-note .warn}` | callout (blue / purple / amber) | a rule or warning, sparingly |

**Two-column layout.** Use nested divs, *not* `layout="2col"`:

```markdown
::: {.cols .vc}
::: {.col}
Prose / `@code-cell` on the left.
:::

::: {.col .fig}
@fig:my-diagram
:::
:::
```

- `.cols` is a flex row; add `.vc` to vertically center the columns.
- A plain `.col` flex-grows; the **figure** column takes a fixed width:
  `.fig` (≈44 %, content-heavy slides), `.fig .big` (≈54 %, figure-led),
  or `.narrow` (≈36 %, a short text/scalar column).
- **Don't** use Quarto's reserved `.callout` (it becomes a Quarto callout
  component) or `.columns`/`.column` (Quarto's own grid) — hence the
  `.d2l-note` and `.cols`/`.col` names.

Add or tweak classes in `_d2l-slides.scss` (pulled in via the deck's
`theme:` entry); never inline CSS in a deck.

---

## Code-cell placeholders

A line that is just a placeholder inside a slide div is replaced by the
slide builder. The forms:

| Placeholder | Emits |
|-------------|-------|
| `@<id>` | the code cell **and** its injected output |
| `@<id>@<fw>` | force a specific framework's variant (regardless of the deck) |
| `@!<id>` | **output only** — no code echo (cover/teaser figures) |
| `@-<id>` | **code only** — no output (verbose setup cells whose output would overflow; emitted without a label so `inject_outputs.py` skips it) |
| `@fig:<id>` | inline a committed **diagram** SVG (`img/auto/<id>.svg`) — see *Diagrams* below |
| `@fig:<id>@<fw>` | prefer a framework variant (`img/auto/<id>-<fw>.svg`), else fall back to `<id>.svg` |

```markdown
::: {.slide}
First we set up our data:

@vec-init

Then we benchmark:

@vec-loop
:::
```

### Resolution rules

`@<id>` resolves to the variant whose `#@tab` matches the deck's
framework, falling back to the variant tagged `#@tab all` (or
untagged) if no framework match exists. So a base ID like
`linreg-train-step` may have four variants in source — one per
framework — and each per-framework deck picks its own.

`@<id>@<fw>` forces a specific framework's variant regardless of the
deck. Use this when you want the same code in every framework's deck
(e.g., a PyTorch reference snippet shown in the JAX deck).

If a deck has no matching variant for `@<id>`, the slide builder
emits a warning and skips that placeholder. The deck still renders.

### Cell IDs

Every Python code fence in the corpus has a `#<id>` attribute,
assigned by `tools/add_cell_ids.py`. IDs are derived from the
section heading the cell sits under, with a numeric suffix for
sections that have more than one cell:

```
chapter_linear-regression/linear-regression.md, "Vectorization for
Speed" section, second code cell  →  `#linear-regression-vectorization-for-speed-2`
```

IDs are **immutable**. Once assigned to a cell, the ID stays even if
the surrounding section is renamed or the cell is moved. Behaves like
`:label:` — pick once, reference forever. The linter warns if a
section rename leaves the ID looking stale, but never errors.

To rename an ID by hand: edit the `#<id>` in the source fence info
string, then update every `@<id>` reference. The linter
(`tools/lint_source.py`) catches dangling references.

To add a new code cell, write the fence with no `#<id>` and run
`tools/add_cell_ids.py`. It assigns one and is idempotent (running
again is a no-op for existing IDs).

### `@d2l.add_to_class` and `#@save`

Library-bound cells stay library-bound in the book. In slides, the
`@d2l.add_to_class(...)` decorator is **stripped at emit time** so
the deck shows just the function body; if the rare `eval=true`
fallback fires, the decorator does not re-execute and double-register
the method. `#@save` markers and residual `%%tab` lines are
stripped too.

`tab.selected()` branches are flattened per the deck's framework, so
a multi-framework cell shows only the relevant code in any given
deck.

---

## Diagrams

Structural and geometric figures live in `diagrams/` as DOM-free SVG
builders, are rendered to `img/auto/<id>.svg`, and committed. Reference
one from a slide with `@fig:<id>`.

### The engine

```
diagrams/engine.mjs    helpers (grid, tx, arrow, block, chip, svg) + color tokens C
diagrams/<chapter>.mjs one function per diagram, keyed by a stable `<chapter>-<concept>` id
diagrams/registry.mjs  imports + spreads each chapter module
diagrams/render.mjs     node CLI → standalone img/auto/<id>.svg
```

```bash
node diagrams/render.mjs --out img/auto                 # all
node diagrams/render.mjs --out img/auto ndarray-reshape # one
node diagrams/render.mjs --list                         # known ids
```

Add a chapter: create `diagrams/<file>.mjs` exporting `diagrams`, import
it in `registry.mjs`, render, and **commit the SVGs** (reviewable diffs,
no reader build step). Ids are global and **immutable** (same rule as
cell ids). `gen_slides.py` inlines the SVG into the deck via a pandoc
`{=html}` raw block, so it inherits the page fonts.

### Authoring diagrams — what we learned

- **Match the cell.** Draw the actual shapes/values the executed cell
  shows (read `outputs/<fw>/<chapter>/<file>.json`), so figure and the
  In/Out card agree (e.g. `transpose` uses the cell's real 3×2 matrix).
- **Portrait, not wide.** A wide-short figure is tiny in a column.
  Design column-friendly aspects — stack panels vertically (the
  `saving-memory` and `concat` figures became portrait this way). A
  `max-height` on `.dgm-svg` caps tall figures.
- **Geometric intuition pays off.** Arrows, angles, projections, lengths
  (e.g. `linear-algebra-dot`, `linear-algebra-norms`, `calculus-secant-tangent`,
  `calculus-gradient-field`) carry meaning text can't. Use them generously.
- **Process & flow diagrams too.** Not only static shapes — a *mechanism*
  (forward build + reverse sweep in `autograd-comp-graph`, the descent path
  in `calculus-gradient-descent`) is itself a teaching figure. Annotate the
  two directions / steps directly on the graph.
- **Subscripts:** the unicode subscript `j` (ⱼ) is missing in many fonts
  and renders full-size — use SVG `<tspan baseline-shift="sub">` instead.
- **Colors:** use the `C` tokens only; keep them in sync with the scss.
- **Standalone SVGs must be valid XML.** `render.mjs` wraps the font-import
  `<style>` in `<![CDATA[…]]>` so the committed SVG parses as an `<img>`
  (book figures) and via rsvg→PDF. The *slide* path inlines the SVG and
  strips `<style>`, so it never hit this — but a diagram reused as a book
  figure (e.g. `autograd-comp-graph` is `fig_autograd_graph` in the
  chapter) needs the valid-XML form.

---

## Output injection

Slides have `eval: false` in their YAML, so Quarto never executes
Python at slide-render time. Outputs come from the per-framework
notebooks under `_notebooks/<fw>/`:

```
chapter_*/*.md → executed _notebooks/<fw>/<chapter>/<file>.ipynb
                                       │
                                       └── cell.metadata.id
                                               │
                                               ▼
_slides/<fw>/<chapter>/<file>.qmd  ←  inject_outputs.py slides
        ↑                                      │
        └── #| label: <id>          ──────────┘
```

The slide builder runs `inject_outputs.py slides` automatically when
`--render` is passed and `_notebooks/<fw>/` exists. To inject
manually:

```bash
python tools/inject_outputs.py slides --framework pytorch
```

Output is wrapped between `<!-- d2l:output -->` / `<!-- /d2l:output -->`
sentinels in the slide `.qmd`, so re-injection is idempotent.

By default, text output is capped at 12 lines per cell (vs 40 for
HTML/PDF). Override per slide with `output-lines="N"`; useful for
training-loop logs that you want to display in full.

---

## VS Code workflow

### One-time setup

1. **Register the per-framework kernels** so notebook open auto-picks
   the right interpreter:
   ```bash
   make kernels
   ```
   This runs `ipykernel install --user` for each framework venv,
   creating `d2l-pytorch`, `d2l-tensorflow`, `d2l-jax`, `d2l-mxnet`.

2. **Install the workspace extension** (TypeScript source under
   `.vscode-extension/`):
   ```bash
   cd .vscode-extension
   npm install
   npm run compile
   npm run package
   code --install-extension d2l-tools-0.1.0.vsix
   ```

3. **Open the workspace.** VS Code will offer the recommended
   extensions (Quarto, Python, Jupyter, YAML) on first open.

### Day-to-day

The extension contributes these commands (Cmd-Shift-P palette):

| Command                         | Default keybind  | What it does |
|---------------------------------|------------------|--------------|
| `D2L: Edit Framework View`      | `Cmd+E Cmd+J`    | Opens the per-framework `.ipynb` for the active `.md`. Regens the notebook if stale. |
| `D2L: Switch Framework`         | `Cmd+E Cmd+S`    | Closes the current notebook, regens for the picked framework, opens it. |
| `D2L: Watch Slides`             | `Cmd+E Cmd+W`    | Spawns `tools/watch_slides.py` for the current `.md`. Browser opens at `localhost:4444`; saves trigger live reload. |
| `D2L: Reveal Source for Cell`   |                  | From a notebook cell, jumps to the matching `#<id>` in source. |
| `D2L: Lint Source`              |                  | Runs `tools/lint_source.py` on the current `.md`. |
| `D2L: Open Slide Preview`       |                  | One-shot render + browser open (no watcher). |
| `D2L: Toggle Sync Daemon`       |                  | Pause/resume notebook→source sync-back. |

Keybindings live in `.vscode/keybindings.json` (workspace-level only;
copy to user keybindings if you want them globally).

### Sync daemon

When `d2l.syncDaemon.enabled` is true (default), saving a per-fw
`.ipynb` invokes `tools/sync_back.py` after a 500 ms debounce. The
daemon round-trips:

- code cells via `cell.metadata.id` → matching `#<id>` fence in source,
- markdown cells with `<!-- d2l:prose id=… fw=<fw> -->` headers →
  matching `:begin_tab:`<fw>` block in source.

Status bar:

- `✓ d2l` — daemon idle, latest save synced cleanly.
- `⟳ d2l: syncing…` — sync_back running.
- `⚠ d2l: conflict` — source `.md` was edited externally between
  notebook open and save; click to view the diff.

### Snippets

Type these prefixes in a `.md` file (auto-trigger via the YAML
language association):

| Prefix         | Expands to                                              |
|----------------|---------------------------------------------------------|
| `slide`        | `::: {.slide title="…"}` … `:::`                        |
| `subslide`     | `::: {.subslide title="…"}` … `:::`                     |
| `frag`         | `::: {.fragment .fade-up}` … `:::` (with quick-pick)    |
| `pyin`         | Python input fence with placeholder `#<id>`             |
| `pyin-tab`     | Same plus `#@tab <fw>` directive                        |
| `pytab`        | `:begin_tab:`<fw>` … `:end_tab:`                        |
| `fwtag`        | `#@tab <fw>` (with framework quick-pick)                |
| `sec`          | Section heading + `:label:`sec_…``                      |
| `numref`       | `:numref:` reference                                    |
| `eqref`        | `:eqref:` reference                                     |
| `cite`         | `:cite:` reference                                      |
| `@`            | `@<cell-id>` placeholder                                |

### Lint on save

Saving a `chapter_*/*.md` triggers `lint_source.py` automatically.
Issues appear in the Problems pane:

- Unbalanced `::: {.slide}` / `:::` divs.
- `@<id>` placeholders that don't match any code fence.
- Duplicate `#<id>` (with overlapping `#@tab` framework sets).
- Unknown framework names in `#@tab`, `:begin_tab:`, or `@<id>@<fw>`.
- Unknown `:directive:` lines.

The same lint runs in pre-commit (`.pre-commit-config.yaml`) and on
demand via `D2L: Lint Source`.

---

## Building and previewing

### Single-deck preview (live reload)

```bash
make watch-slides FW=jax FILE=chapter_linear-regression/linear-regression.md
```

…or via VS Code: open the source `.md`, press `Cmd+E Cmd+W`, pick
`jax`. Browser tab opens automatically; saves trigger reload in
~500 ms.

### One-shot render

```bash
# All slides, all frameworks, parallel-safe
make -j4 slides

# One framework
make slides-pytorch

# Subset by file
make slides-pytorch SLIDES_FILTER="chapter_linear-regression/linear-regression.md"
```

`make slides` runs `gen_slides.py` for each framework, injects
notebook outputs (when present), and renders to HTML via
`quarto render --to revealjs` in a thread pool of 8 workers. CPU-only
— no GPU contention, parallel-safe across frameworks.

### Outputs

Rendered decks live at:

```
_slides/<fw>/<chapter>/<file>.html
_slides/<fw>/<chapter>/<file>_files/   # libraries
_slides/<fw>/img/outputs/              # injected output images
```

Open one in a browser to view. For batch sharing, copy the whole
`_slides/<fw>/` tree.

### Executing notebooks locally (Apple Silicon)

Slide *rendering* is CPU-only and needs no framework. But the outputs a
deck shows come from **executed** notebooks (the committed `outputs/`
store), and execution now also works on a Mac: all four frameworks have
arm64 CPU builds wired into UV (`pyproject.toml` darwin extras + the macOS
mxnet wheel from the smolix/mxnet release). So one laptop can run a section
end-to-end:

```bash
make venv-pytorch venv-jax venv-tensorflow venv-mxnet   # one-time, per fw
# execute one notebook per framework (CPU; force the slots so no GPU is assumed):
make -B _notebooks/pytorch/chapter_x/foo.executed NUM_GPUS=0 GPU_SLOTS=0 CPU_SLOTS=2
# … repeat for jax/tensorflow/mxnet, then bless into the committed store:
make capture-outputs FILES=chapter_x/foo.md
```

After capture, `make slides` (or the book render) picks the new outputs up
by cell id. This means a deck built from freshly-edited cells no longer
needs the GPU box just to refresh outputs — author, execute, capture, and
build all on the Mac. (`tools/check_runtime_deps.py` has a darwin branch:
no CUDA/`ldd`, just an import smoke-test.)

---

## Migration & deploy (north-star rollout)

The switch from the old decks to north-star is **gradual and
source-driven**: a deck goes live in its north-star form only when one
exists; every other deck stays as the old deck, frozen, until upgraded.

- **What counts as north-star is auto-detected.** `tools/northstar_slides.py`
  flags a deck as north-star iff its `<!-- slides -->` block uses the
  vocabulary (`.cover`/`.divider`/`.kicker`/`@fig:`/`.cols`/`.d2l-note`).
  Rewrite a block → the deck becomes eligible automatically; no allowlist.
  `python tools/northstar_slides.py . ` prints the migration count.
- **The landing page badges them.** `build_slides_index.py` marks
  north-star decks "new" on `slides/index.html`.
- **Deploy substitutes only the north-star decks.**
  `tools/stage_northstar_slides.sh` overlays just those decks (+ the
  refreshed index, the new content-hashed theme CSS, and any plot assets
  they reference) onto `_book/slides/`, leaving the legacy decks untouched;
  `tools/upload_northstar_r2.sh` then pushes *only* that staged set to the
  `staging-d2l` R2 bucket (region `auto`). Legacy decks on the bucket are
  never overwritten.
- **One-off legacy fix:** `tools/patch_slides_navlink_r2.sh` patches the
  in-deck navbar "Slides" link on already-live legacy decks in place
  (byte-precise, idempotent) — the source overlay is fixed, but frozen
  decks need the in-place touch.

The in-deck chrome (`_d2l-slides-overlay.html`) uses paths relative to a
deck at `slides/<fw>/<chapter>/<file>.html` (3 dirs under root): the slides
landing is `../../index.html`, book home `../../../index.html`. Keep that
depth correct when editing the overlay.

---

## Authoring patterns

### Quality rules (learned building §2.1, §2.3, §2.4)

`ndarray.md`, `linear-algebra.md`, and `calculus.md` are the reference
decks; `docs/slides/north-star.html` is the visual bar. When authoring or
regenerating a deck:

1. **Teach the idea — diagram first.** Lead each slide with the picture of
   its one idea (a shape, a region, a tangent, a computational graph),
   then a minimal cell and a one-line caption. A slide that is mostly text
   or mostly code is wrong. Calculus is the model: 6 geometric diagrams
   carry the chapter; code is incidental.
2. **Build fresh to the bar.** A pre-existing `<!-- slides -->` block is a
   *source of ideas*, not a target — the north-star rewrite is markedly
   better than what it replaced. Cover, dividers, kickers, In/Out cards,
   2-col diagram pairings, callouts.
3. **Curate; one idea per slide.** Follow the notebook's teaching order
   but drop cells that don't teach (e.g. an `axis=[0,1]` cell redundant
   with the per-axis slide). Trim noisy output.
4. **Check per-framework *framing*, not just code.** Inspect all four
   `outputs/<fw>/…json` and the `#@tab` source. Where a concept itself
   differs (JAX immutability; TF `Variable`/`tf.function`; NumPy
   shared-vs-copy; **PyTorch gradient accumulation vs reset elsewhere**)
   write `only=`/`except=` scoped slides — and a framework diagram variant
   if needed. Where only code/output differ, one shared slide suffices.
   (Linear algebra and calculus needed *zero* scoped slides; ndarray
   needed several; autograd will too.)
5. **`. . .` fragments work only at slide top level — never inside a
   `::: {.col}`** (they render as a literal "..."). In a two-column
   slide, stack the cells or show one; put progressive reveals on
   full-width slides.
6. **Fit 720 px.** Verify with the overflow sweep (below) — no slide's
   `scrollHeight` should exceed the deck height. If a slide is too tall:
   shorten the intro, make a verbose setup cell `@-` (code-only), split
   it into two slides (`only=`-scoped continuations), or widen the
   content column. Don't rely on per-slide scrollbars.
7. **Mind column width.** Cells with matrix/verbose output and long code
   need a *wide* content column; column **prose** is shrunk so it doesn't
   wrap to orphaned words. Wide-short diagrams should be redesigned
   portrait rather than squeezed.
8. **Verify across all four frameworks**, then hand to a human and
   iterate.

Overflow / scroll sweep (run in the rendered deck's console, or via
Playwright):

```js
const H = Reveal.getConfig().height;
[...document.querySelectorAll('.reveal .slides section.slide, .title-slide')]
  .filter(s => s.scrollHeight > H + 4)
  .map(s => s.querySelector('h2')?.textContent);   // → [] when clean
```

### Convert an existing slide deck to the new format

Inline `[**…**]` / `(**…**)` / `[~~…~~]` / `(~~…~~)` markers were
removed in favor of slide divs. If you find old markers in source,
run:

```bash
python tools/migrate_slide_markers.py
```

This converts them to a `<!-- slides -->` section with one `.slide`
div per slide group. The script is idempotent — files that already
have a `<!-- slides -->` section are left alone.

### Add a new slide

1. Open `chapter_x/foo.md`.
2. Scroll to the `<!-- slides -->` section (or add one near the
   bottom if missing).
3. Use the `slide` snippet to insert a new `.slide` div.
4. Add prose. Use `@<id>` placeholders to reference existing code
   cells, or write `pyin` snippets and let `add_cell_ids.py` assign
   IDs.
5. `Cmd+E Cmd+W` to preview.

### Reuse a code cell across decks

Same `@<id>` works in every framework's deck — the resolver picks
the right framework variant. To force a specific framework:
`@cell-id@pytorch`.

### Slide-only content

Any prose inside a `.slide` div that isn't echoed in the body is
slide-only by design — the book renderer strips the whole
`<!-- slides -->` section. There's no marker syntax for
"slide-only paragraph" anymore; just put it in a slide div.

### Hidden setup cells

Slides don't execute Python. If a deck depends on a setup cell that
isn't otherwise interesting to show, just don't reference it from a
slide div — the cell stays in source for the book and the executed
notebook, but doesn't appear in slides.

---

## Troubleshooting

### A callout renders without its `.warn` / `.rule` color

You used `::: {.callout}` — Quarto reserves that class and converts the
div into its own callout component, dropping your modifiers. Use
`::: {.d2l-note}` / `{.d2l-note .warn}` / `{.d2l-note .rule}` instead.

### A diagram renders as scrambled text / stray tags

The SVG was parsed as Markdown (so `[…]` in labels became links and `'`
a smart quote). `@fig:` already wraps the SVG in a `{=html}` raw block to
prevent this — only an issue if you hand-place a raw `<svg>` in a deck;
wrap it in a ` ```{=html} ` block.

### A `. . .` shows up as a literal "..."

It's inside a `::: {.col}` — fragment pauses work only at the slide's top
level. Move it out of the column, or restructure the slide.

### Slides don't rescale with the window

Reveal scales the slide canvas to fit; the **navbar** is scaled to match
by `_d2l-slides-overlay.html`. Don't set `scrollable: true` (per-slide
scroll containers fight the fit-to-window scaling), and don't pin a
`height`/`min-height` on `.reveal .slides` or a section.

### "@cell-id has no variant for `<fw>`"

The placeholder references a cell that doesn't exist for this deck's
framework. Either:
- The cell is `#@tab pytorch`-only and you're building the JAX deck.
  Use `@cell-id@pytorch` to force the PyTorch variant, or accept the
  warning and let the deck render without the cell.
- The cell has been removed; update or delete the placeholder.

### "placeholder @cell-id references unknown cell ID"

Linter warning. Either you typo'd the ID or the cell hasn't been
written yet. Run `tools/add_cell_ids.py` if a new fence is missing
its ID.

### Cell IDs in citation warnings during HTML build

```
[WARNING] Citeproc: citation gd-learning-rate-2 not found
```

Caused by Quarto's chunk-label processing. These are **warnings**,
not errors — the book still builds. Cell IDs without a `fig-`,
`tbl-`, `sec-` prefix are seen by Citeproc as unresolved citation
keys. Safe to ignore.

### "Unexpected end of JSON input" / "Address already in use"

Race conditions when running `make -j4 slides` (parallel renders
contending on Quarto's per-project cites cache, or ipykernel port
binding). The slide builder retries once on these transient errors.
A `RETRY OK` log line marks recoveries. If a deck still fails after
the retry, run that one alone:

```bash
make slides-jax SLIDES_FILTER="chapter_x/foo.md"
```

### Slide preview won't reload

`watch_slides.py` requires the `watchdog` Python package:

```bash
.venv-pytorch/bin/pip install watchdog
```

Or use any framework venv that's already synced.

### VS Code picks the wrong kernel

Make sure `make kernels` was run after the venvs were synced. Open
the notebook, click the kernel selector in the top-right, and pick
`d2l (<framework>)`. The next time it'll auto-select via
`metadata.kernelspec.name`.

---

## File locations

```
chapter_*/<file>.md                Source: book content + <!-- slides --> section
_d2l-slides.scss                   Slide styling/layout SCSS (root of repo)
_d2l-slides-overlay.html           Navbar + presenter button + nav-scale hook
docs/slides/north-star.html        The visual exemplar (the bar)
diagrams/<chapter>.mjs             DOM-free SVG diagram builders
diagrams/{engine,registry,render}.mjs   helpers · id→fn map · render CLI
img/auto/<id>.svg                  Rendered diagrams (committed)
_slides/<fw>/<chapter>/<file>.qmd  Generated slide source (gitignored)
_slides/<fw>/<chapter>/<file>.html Rendered deck (gitignored)
_slides/<fw>/img/outputs/          Injected output images (gitignored)

tools/add_cell_ids.py              Assigns/maintains #<id> on code fences
tools/migrate_slide_markers.py     One-shot: inline markers → slide divs
tools/strip_tab_all.py             One-shot: removes #@tab all / %%tab all
tools/gen_slides.py                Generates and renders slide .qmd (@fig/@-/@!, only=/except=)
tools/inject_outputs.py            Injects notebook outputs (slides mode)
tools/audit_slides.py              Teachability/overflow audit (@fig/@-/@! aware)
tools/watch_slides.py              Live preview daemon
tools/lint_source.py               Source linter
tools/sync_back.py                 Notebook → source sync
tools/northstar_slides.py          Auto-detects which decks are north-star (migration gate)
tools/stage_northstar_slides.sh    Overlays only north-star decks → _book/slides/
tools/upload_northstar_r2.sh       Surgical R2 upload of the staged north-star set
tools/patch_slides_navlink_r2.sh   One-off: fix in-deck Slides nav link on live legacy decks
tools/build_slides_index.py        Slides landing page + manifest (badges north-star decks)

.vscode/                           Workspace settings, keybindings, snippets
.vscode-extension/                 d2l-tools VS Code extension source
```
