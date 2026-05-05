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
| `layout="…"`           | One of `2col`, `figure`, `code`. Adds a `.slide-<L>` class for SCSS rules in `_d2l-slides.scss`. |
| `transition="…"`       | Reveal.js transition. `fade`, `slide`, `convex`, `concave`, `zoom`, `none`. |
| `data-background-color="#…"`   | Slide background color.                        |
| `data-background-image="…"`    | Slide background image.                        |
| `output-lines="N"`     | Override the per-deck text-output cap (default 12) for cells injected into this slide. |

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

### Layouts

```markdown
::: {.slide layout="2col" title="Two columns"}
Prose on the left.

@some-figure
:::
```

Out of the box:

- `layout="2col"` — CSS grid, prose left / code-or-figure right.
- `layout="figure"` — centered figure, large.
- `layout="code"` — code dominant, smaller font, scrollable output cap
  at 30vh.

The CSS lives in `_d2l-slides.scss` at the repo root. Add or tweak
classes there if you need new layouts; the slide deck pulls the file
in via the YAML `theme:` entry.

---

## Code-cell placeholders

A line that is just `@<id>` (or `@<id>@<framework>`) inside a slide
div is a placeholder. The slide builder replaces it with the matching
code cell from the source.

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

---

## Authoring patterns

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
_d2l-slides.scss                   Slide layout SCSS (root of repo)
_slides/<fw>/<chapter>/<file>.qmd  Generated slide source (gitignored)
_slides/<fw>/<chapter>/<file>.html Rendered deck (gitignored)
_slides/<fw>/img/outputs/          Injected output images (gitignored)

tools/add_cell_ids.py              Assigns/maintains #<id> on code fences
tools/migrate_slide_markers.py     One-shot: inline markers → slide divs
tools/strip_tab_all.py             One-shot: removes #@tab all / %%tab all
tools/gen_slides.py                Generates and renders slide .qmd
tools/inject_outputs.py            Injects notebook outputs (slides mode)
tools/watch_slides.py              Live preview daemon
tools/lint_source.py               Source linter
tools/sync_back.py                 Notebook → source sync

.vscode/                           Workspace settings, keybindings, snippets
.vscode-extension/                 d2l-tools VS Code extension source
```
