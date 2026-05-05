# PLAN: Slide overhaul + per-framework notebook editing

> Status: design. Implemented in a single sequence by Claude.
> Owner: smola@boson.ai. Drafted 2026-04-26, revised 2026-04-28.

This plan describes the **end state** of two related refactors that share
infrastructure:

1. **Slides** — replace inline markers with fenced divs, drop slide
   re-execution by injecting cached notebook outputs, support transitions
   and richer layouts.
2. **Authoring** — make per-framework `.ipynb` files an editable surface
   that round-trips back to the source `.md`, all inside VS Code.

Both depend on a single new primitive: **stable, immutable IDs on every
code cell and prose paragraph**. Once IDs exist, output injection,
slide layout placeholders, notebook → source sync-back, framework
switching, and the watch/preview loops all key on the same anchor.

The plan targets one cohesive design. There is no phased rollout, no
backward-compat for inline markers, no deprecation period. Migration
scripts run once over the corpus to bring it to the new state.

---

## Motivation

### Slides today

- Inline markers (`[**…**]`, `(**…**)`, `[~~…~~]`, `(~~…~~)`) force slide
  prose and book prose into the same sentence.
- Slides re-execute every code cell under the GPU scheduler. ~40 min wall
  per build, requires `kill_stale_kernels()`, `MULTI_GPU_NOTEBOOKS`
  classification, transient-failure retries, per-cell timeouts.
- Notebook outputs and slide outputs drift (different RNG seeds, kernel
  versions, XLA fusion).
- Slides cannot be built in parallel across frameworks (GPU contention).

### Authoring today

- Source `.md` interleaves four frameworks. Editing the JAX path means
  visually filtering past `#@tab pytorch`, `:begin_tab:`mxnet`, etc.
- 197 occurrences of `#@tab all` are noise (the default case).
- No way to run a single cell, see outputs, then sync back; you edit
  source, regen notebook, execute the whole notebook, repeat.
- No live preview for slides or chapters.

## Goals

- Slide prose can diverge from book prose without duplicating code.
- Slide rendering is CPU-only, parallel-safe across frameworks, <5 min.
- Notebook outputs are produced once and reused everywhere (HTML, PDF,
  slides) by injection.
- Per-framework notebook editing in VS Code, with auto sync-back to the
  source `.md` and zero re-execution of unchanged cells.
- "All frameworks" is the default; framework-specific is the exception.
- Sub-second feedback for prose / slide-marker edits.

## Non-goals

- Replacing the multi-framework `.md` source-of-truth model.
- Changing the rendered HTML book output for readers.
- Migrating to a new build system or template engine.
- Supporting JupyterLab as a primary editor (VS Code only).

## Architectural principles

1. **One source of truth.** `chapter_*/**/*.md` is canonical. Every
   generated artifact (`.qmd`, `.ipynb`, slides, PDFs, `d2l/*.py`) is
   derivable from it.
2. **IDs everywhere, position nowhere.** Every code cell and every prose
   paragraph carries a stable ID written *into the source*. No tool
   relies on cell or paragraph *position* to match content across
   artifacts.
3. **IDs are immutable.** Once an ID is assigned, it stays — even if its
   surrounding section is renamed or the cell is moved. IDs behave like
   `:label:` equation/section labels: pick once, reference forever.
4. **Outputs flow one direction.** Notebooks execute → outputs cached in
   `.ipynb` → injected into HTML/PDF/slide `.qmd` files → rendered.
   Nothing else executes Python.
5. **`eval: false` by default for derived artifacts.** Slides, PDFs, and
   the HTML book all render with `eval: false`; outputs come from
   injection.
6. **Default-shared, opt-in framework-specific.** Untagged code and prose
   apply to all frameworks. `#@tab pytorch`, `:begin_tab:`pytorch` are
   exceptions.
7. **Bidirectional sync via projection.** A per-framework `.ipynb` is a
   projection of the source `.md` for one framework. Edits to the
   projection are projected back through the same ID anchors.
8. **VS Code is the IDE.** All UX integrates via a workspace extension
   that hosts the sync daemon, watchers, and conflict UI.

---

## The ID model

This is the central primitive. Everything else assumes it works.

### Code cell IDs

**Form on a code fence:**

```markdown
```{.python .input #linreg-vector-addition}
c = a + b
```
```

**Generation rules** (one-shot, by `tools/add_cell_ids.py`):

- Default ID: `<chapter-slug>-<section-slug>-<seq>` where `<seq>` is the
  cell's 1-based position within its section. Example:
  `linreg-vector-addition-1`. The `-<seq>` suffix is omitted when there
  is exactly one code cell in the section.
- Slug = section heading lowercased, non-alphanumerics → `-`, trimmed.
- If two cells in different sections happen to slugify to the same
  string, append a 4-char hash of the cell's normalized source:
  `linreg-vector-addition-1-a3f1`. The collision is logged.
- Per-framework variants of the same base ID share a base ID; they are
  differentiated by their `#@tab` directive (see "Per-framework variants"
  below). The `<seq>` is computed across variants, not within each
  framework.

**Stability rules:**

- Once written into a `.md` file, an ID is immutable. Renaming the
  containing section does **not** change existing IDs.
- Reordering cells does **not** change their IDs.
- Editing cell content does **not** change its ID.
- Authors may rename an ID by hand; the linter warns once if the ID no
  longer matches its derivation but does not error.

`add_cell_ids.py` is idempotent: it skips fences that already have an
ID. New cells added by authors get fresh IDs the next time the script
runs (or via the VS Code "assign id" code action).

### Prose paragraph IDs

Prose paragraphs carry IDs in the source via an HTML comment immediately
before the paragraph:

```markdown
<!-- d2l:p linreg-overview-1 -->
A naïve Python loop pays interpreter overhead per element. By contrast,
a vectorized call hands the work to a C kernel.
```

- Generation: `<chapter-slug>-<section-slug>-<seq>` (no `p` infix).
- Same immutability rules as code cells.
- HTML comments are invisible in rendered output and pass through Quarto
  unchanged. They survive `quarto convert` to `.ipynb`, where
  `gen_notebooks.py` re-formats them as the more verbose
  `<!-- d2l:prose id=<id> fw=<fw-list> -->` notebook header (so the
  framework context is explicit in the projected view).
- `:begin_tab:` blocks each get their own prose ID, since they are
  separately syncable.
- Trivially-short paragraphs (single-sentence transitions, headings,
  blank-line separators) are skipped — sync-back resolves them
  positionally, anchored by surrounding ID-bearing prose and code cells.

### Per-framework variants

Two cells in the same source file may share the same base ID if they
are framework-specific:

```markdown
```{.python .input #linreg-train-step}
#@tab pytorch
def train_step(model, batch):
    ...
```

```{.python .input #linreg-train-step}
#@tab jax
def train_step(state, batch):
    ...
```
```

The base ID `linreg-train-step` is the contract. Each variant is
selected by its `#@tab` directive. The linter enforces:

- All variants of a base ID have a `#@tab` set.
- The variants' `#@tab` sets are non-overlapping (no two variants both
  apply to PyTorch).

A reference to `@linreg-train-step` from a slide div resolves to the
variant whose `#@tab` matches the slide's framework, falling back to a
variant tagged `#@tab all` (or untagged) if no framework match exists.

A reference to `@linreg-train-step@pytorch` forces the PyTorch variant
regardless of the deck's framework. The `@<fw>` suffix is
**placeholder-side syntax only** — it never appears in source `.md`
fence info strings.

---

## Source syntax (post-refactor)

### Code fences

```markdown
# All-frameworks (default — no #@tab line)
```{.python .input #linreg-vector-addition}
c = a + b
```

# Framework-specific
```{.python .input #linreg-train-step}
#@tab pytorch
def train_step(model, batch):
    ...
```

# Library-bound (extracted into d2l package)
```{.python .input #linreg-train-step-method}
#@tab pytorch
@d2l.add_to_class(d2l.Module)
def configure_optimizers(self):  #@save
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```
```

### Prose tabs

```markdown
:begin_tab:`pytorch`
<!-- d2l:p linreg-pytorch-note-1 -->
PyTorch-specific paragraph.
:end_tab:
```

The `<!-- d2l:p -->` line is optional inside a `:begin_tab:` block — the
block itself is a sync-back anchor. The comment is added by
`add_cell_ids.py` for explicitness and to support multi-paragraph tabs.

### Slide divs

```markdown
::: {.slide title="Why vectorize?" layout="2col"}
A naïve Python loop pays interpreter overhead per element:

@vec-loop

. . .

A vectorized call hands the work to a C kernel:

@vec-add

::: {.fragment .fade-up}
~3000× faster on this size.
:::
:::

::: {.slide title="Profiling deep-dive" transition="fade"}
@profile-flame

::: {.subslide title="Memory layout"}
Strides matter.
:::
:::
```

Rules:

- `::: {.slide title="…"}` opens a slide. Title is optional.
- `@<id>` placeholder on its own line resolves to the code cell with
  that ID, in the framework appropriate for the deck. Cells appear
  inline at the placeholder location.
- `@<id>@<fw>` forces a specific framework's variant.
- `::: {.subslide title="…"}` nested inside a `.slide` emits a vertical
  sub-slide.
- `transition="…"` attribute maps to `data-transition`.
- `layout="2col"` (or `"figure"`, `"code"`) attaches a layout class for
  the SCSS rules in `_d2l-slides.scss`.
- Quarto reveal.js syntax (`. . .`, `::: {.fragment}`,
  `[text]{.fragment}`, `data-fragment-index=N`,
  `data-background-color="#…"`) passes through verbatim.
- `output-lines="N"` overrides the default 12-line cap on injected text
  output for that slide.

### Cross-references (unchanged)

```markdown
:label:`sec_linear-regression`
:numref:`fig_example`
:eqref:`eq_loss`
:cite:`Author.2020`
:citet:`Author.2020`
```

### What was removed

- `#@tab all` and `%%tab all` directives are stripped from the corpus by
  `tools/strip_tab_all.py`. Untagged = all frameworks.
- Inline slide markers (`[**…**]`, `(**…**)`, `[~~…~~]`, `(~~…~~)`) are
  converted to `::: {.slide}` divs by `tools/migrate_slide_markers.py`.
  The parser code path for inline markers is deleted from `gen_slides.py`.

---

## Build pipeline (post-refactor)

```
chapter_*/*.md  (source of truth, with #<id> on every code fence
                 and <!-- d2l:p <id> --> on every prose paragraph)
   │
   ├── d2l_preprocess.py ──→ chapter_*/*.qmd ──→ quarto render html ──→ _book/
   │                                                  ↑
   │                                                  └── inject_outputs.py html
   │
   ├── gen_pdf.py ──→ _pdf/<fw>/*.qmd ──→ inject_outputs.py pdf ──→ quarto pdf
   │
   ├── gen_notebooks.py ──→ _notebooks/<fw>/*.ipynb (with cell.metadata.id)
   │                              │
   │                              └─→ run_notebooks.py (executes; outputs cached in-place)
   │                                       │
   │                                       └─→ outputs in .ipynb, ready for injection
   │
   └── gen_slides.py ──→ _slides/<fw>/*.qmd ──→ inject_outputs.py slides ──→ quarto revealjs
                                                       │
                                                       └─ matches by cell.metadata.id
```

**Key changes from today:**

- Slides have `eval: false` by default. No GPU. `make -j4 slides` is
  parallel-safe across frameworks.
- The GPU scheduler in `gen_slides.py` (lines ~370–515) is deleted.
  Replaced with a `ThreadPoolExecutor(max_workers=8)` running
  `quarto render --to revealjs` with no GPU env vars.
- Output injection works for slides as well as html and pdf; matching is
  ID-based with no hash fallback (every cell has an ID post-migration).
- HTML book and PDFs are unchanged for readers.

If a slide references a code cell that has no matching notebook cell
(rare: slide-only code), the placeholder resolver emits that single
cell with `eval: true` and a lint warning. The deck still builds.

---

## Component-by-component

### `tools/d2l_preprocess.py`

- `parse_blocks()` extracts `#<id>` from code-fence info strings
  (`{.python .input #foo}`) into `CodeBlock.cell_id`.
- `extract_tab()` returns `'all'` when no tag is found (was `None`).
  Call sites that distinguished `None` from `'all'` are unified to
  `'all'`. `is_boilerplate()` is unaffected (it's a string-content
  check).
- Emits `#<id>` into the generated `.qmd` so Quarto carries it through.
- HTML book builds the book without slide divs: `::: {.slide}` and
  `::: {.subslide}` divs are stripped during preprocessing (book-only
  output drops slide-only content).

### `tools/gen_notebooks.py`

- Sets `cell.metadata.id` on every code cell from the source `#<id>`.
- Sets `cell.metadata.kernelspec.name = "d2l-<fw>"` so VS Code
  auto-selects the right interpreter.
- Emits a synthesized header on every prose markdown cell:
  `<!-- d2l:prose id=<id> fw=<fw-list> -->`. Source `<!-- d2l:p <id> -->`
  comments inside `:begin_tab:` blocks become `fw=<fw>` headers; outside
  `:begin_tab:` they become `fw=all`.
- Emits a hidden d2l-setup cell at the top of every notebook
  (`from d2l import <fw> as d2l`, etc.) so cells that reference
  `d2l.train_step` etc. without redefining them don't fail at execute
  time.

### `tools/gen_slides.py` (rewritten)

For each `.md` source and each framework:

1. Parse `.md` into blocks (reuses `parse_blocks()` and the new ID
   metadata).
2. Build an `id → fence` index, with framework variants tracked as
   `(base_id, fw) → fence`.
3. Walk the source for `::: {.slide …}` divs.
4. For each placeholder line `@<id>` or `@<id>@<fw>`:
   - Resolve to the appropriate framework variant.
   - Strip `#@save` markers and `%%tab` lines (already done today).
   - **Strip `@d2l.add_to_class(...)` decorators.** Slides do not add to
     the library; the decorator is a teaching detail that, if executed
     by the rare `eval: true` fallback, would re-register the method
     and risk double-application. Match `^@d2l\.add_to_class\([^)]*\)\s*$`
     and drop the line. Function body and signature remain visible.
   - Emit the cell as a `.qmd` Python fence.
5. Emit `eval: false, echo: true` in the YAML header. Theme `simple`,
   plus `_d2l-slides.scss` via
   `theme: [simple, ../../../_d2l-slides.scss]` (path relative to the
   slide `.qmd` at depth 3).
6. Quarto's reveal.js syntax (`. . .`, fragments, transitions) passes
   through verbatim.

The script no longer renders. Rendering is a separate Makefile step
running `quarto render --to revealjs` in a thread pool.

Result: ~250 lines (down from ~500). No GPU code, no
`MULTI_GPU_NOTEBOOKS` classification, no `kill_stale_kernels()`, no
transient-failure retry loop.

### `tools/inject_outputs.py`

Three modes: `html`, `pdf`, `slides`. Common matching logic:

1. **ID match**: `qmd_cell.metadata.id` against
   `_notebooks/<fw>/<...>.ipynb` cells indexed by `metadata.id`.
2. **Per-fw resolution**:
   - `html`: each framework tab gets the matching cell from that
     framework's notebook.
   - `pdf`: deck's framework only.
   - `slides`: deck's framework only.
3. **No hash fallback.** Post-migration, every cell has an ID.
   A missed match is a lint condition; the cell falls through to
   `eval: true` so the build still produces something, with a warning.

Per-mode parameters:

| Mode | `MAX_TEXT_LINES` | Image dir |
|------|------------------|-----------|
| html | 40 | `img/outputs/<chapter>/` |
| pdf | 40 | `_pdf/<fw>/img/outputs/<chapter>/` |
| slides | 12 (per-slide override via `output-lines="N"`) | `_slides/<fw>/img/outputs/<chapter>/` |

Sentinels remain `<!-- d2l:output -->` / `<!-- /d2l:output -->`
(idempotent re-injection).

### `tools/sync_back.py`

Inputs:
- `--notebook _notebooks/<fw>/<chapter>/<file>.ipynb`
- `--source chapter_<chapter>/<file>.md`

Algorithm:

1. Index source `.md` by `#<id>` for code fences and by `<id>` for prose
   paragraphs.
2. Walk notebook cells in order.
   - **Code cell:** look up `cell.metadata.id`.
     - Found → replace the source fence body. Preserve the source's
       `#<id>`, `#@tab` line, `#@save` markers, and `@d2l.add_to_class`
       decorator (sync-back never strips these — they're source-only
       constructs that the notebook view didn't have to begin with).
     - Not found → new cell. Insert into source with a generated ID.
       Insertion point determined from neighbor cells' IDs.
   - **Markdown cell:** parse `<!-- d2l:prose id=… fw=… -->`.
     - `fw=all` → replace the matching shared paragraph in source.
     - `fw=<single>` → replace the matching `:begin_tab:` body.
     - Multi-fw → replace each matching `:begin_tab:` body.
3. Detect deletions: source IDs not in the notebook. Surface to the user
   via the VS Code conflict UI (never a CLI prompt).
4. Atomic write of source `.md` (write to tmp + rename).

State file `.d2l-sync-state/<chapter>/<file>.json` tracks last-synced
mtimes per artifact and per-cell content hashes for conflict detection.

**Conflict resolution (single mechanism):** when source `.md` mtime >
last-synced source mtime AND the source has changed in cells the
notebook also touched, the VS Code extension surfaces a 3-way diff
(source / notebook / common ancestor from state file). User picks per
cell: source wins / notebook wins / merge. No CLI prompts.

### `tools/lint_source.py`

Single-pass lint over `.md` files. GCC-style output
(`path:line:col: severity: message`) for VS Code's problem-matcher.

Checks:

- Unbalanced markers: `:begin_tab:` / `:end_tab:`,
  `::: {.slide}` / `:::`, `::: {.subslide}` / `:::`,
  `::: {.fragment}` / `:::`.
- `@<id>` and `@<id>@<fw>` placeholders reference an existing code
  fence ID.
- Code fence IDs are unique per file, OR are framework-specific variants
  whose `#@tab` sets are non-overlapping.
- Framework names (`pytorch | tensorflow | jax | mxnet`) are valid in
  `#@tab`, `%%tab`, `tab.interact_select`, prose tabs, and placeholder
  `@<fw>` suffixes.
- Cross-references (`:numref:`, `:eqref:`, `:cite:`, `:ref:`) point to
  existing labels (with `--corpus` for cross-file).
- `:label:` / `:eqlabel:` IDs are globally unique across the corpus
  (with `--corpus`).
- Unknown directives: anything `:foo:` not in the known set.
- Cell ID derivation drift: warn (not error) when an ID no longer
  matches its derivation from section slug + sequence.

Performance target: <100 ms per file, <5 s for the 191-file corpus.

### `tools/watch_slides.py`

Long-running watcher invoked from the VS Code extension or the CLI.

1. Ensure `_notebooks/<fw>/<chapter>/<file>.ipynb` exists (warn if not —
   outputs will be missing).
2. Run `gen_slides.py` for the file.
3. Run `inject_outputs.py slides`.
4. Spawn `quarto preview <slide.qmd> --to revealjs --no-watch-inputs
   --port 4444` as a subprocess.
5. `watchdog.observers.Observer` on the source `.md`. On modify
   (debounced): re-run gen_slides + inject_outputs. Quarto preview
   re-detects the `.qmd` change and reloads.
6. On Ctrl-C / extension teardown: kill the quarto preview subprocess.

Target: <500 ms reload on prose / slide-marker edit (no notebook
re-execution).

Atomic writes throughout — `.qmd` is written to `tmp` and renamed so
quarto's preview never reloads a half-written file.

### `tools/run_cells.py` (per-cell notebook output cache)

Wraps `nbconvert --execute` with cell-level caching keyed by
`cell.metadata.id` + content hash + upstream cells' content hashes.
Cache stored at `_notebooks/<fw>/.cache/<id>.json` (gitignored).

On `make run-notebooks-<fw>`:

- For each cell, compute key. Cache hit → restore outputs without
  execution.
- Cache miss → find the earliest miss in the notebook; execute from
  there to the end (kernels are stateful; downstream cells may depend
  on upstream re-execution). This matches Quarto's `freeze: auto`
  semantics.
- Image outputs cached as files referenced by JSON (not inline base64).
- Cache size bounded by `--max-cache-mb`, LRU eviction.
- `make clean-cache` always works.

Lowest-priority component. Builds correctly without it; the cache is a
cycle-time optimization. `run_notebooks.py` invokes `run_cells.py` if
present, falls back to direct nbconvert otherwise.

### `_d2l-slides.scss`

Layout / sizing rules for slide layouts:

- `.slide-2col` → CSS grid, prose left, code/figure right; image
  `max-height: 70vh; max-width: 50vw`.
- `.slide-figure` → centered figure, `max-height: 80vh`, smaller caption.
- `.slide-code` → code dominant; `pre` at `font-size: 0.8em`,
  `line-height: 1.2`; output box capped at `max-height: 30vh; overflow: auto`.
- Default (no `layout=`): existing reveal.js simple theme behavior.
- Output-line cap behavior: when `.cell-output-display` exceeds
  `--d2l-output-max-lines` (CSS variable, default 12), it scrolls.

~80 lines of SCSS. Kept separate from `_d2l-theme.scss` (which is for
the HTML book) so the two can diverge.

### VS Code extension (`.vscode-extension/`)

A single TypeScript extension shipped via `.vsix` in repo, recommended
in `.vscode/extensions.json`.

The extension hosts the **sync daemon in-process** — file watchers,
debounce logic, conflict UI, and command palette commands all live here.
There is no separate daemon process.

**Commands:**

- `D2L: Edit Framework View` — quick-pick framework, opens corresponding
  `.ipynb` in the same window.
- `D2L: Switch Framework` — closes current notebook, regens for the new
  fw, opens.
- `D2L: Watch Slides` — quick-pick framework, starts slide preview for
  the current `.md`.
- `D2L: Reveal Source for Cell` — from a notebook cell, jump to the
  matching `#<id>` in source `.md`.
- `D2L: Lint Source` — runs `lint_source.py` on the current file.
- `D2L: Open Slide Preview` — one-shot slide render + browser open.

**Status bar:**

- `synced ✓` / `syncing…` / `conflict ⚠`. Click to reveal sync log or
  open conflict UI.

**CodeLens:**

- "Reveal source" link above each notebook code cell with
  `metadata.id`.

**Conflict UI:**

- 3-way diff (source / notebook / common ancestor from state file).
- Per-cell pick: source wins / notebook wins / merge in editor.

**Problem provider:**

- Hooks into `lint_source.py` output. Issues appear in the Problems
  pane on save.

**Activation:**

- On workspace open: spawn watchers, register commands, bind keys
  (`Cmd+E Cmd+J`, `Cmd+E Cmd+S`, `Cmd+E Cmd+W`).
- On deactivate: kill `quarto preview` subprocess if running.

### `.vscode/` configuration

```
.vscode/extensions.json     Recommend the d2l extension + Quarto/Python/Jupyter/YAML
.vscode/settings.json       File exclusions; Quarto association for chapter_*/*.md;
                            kernelspec UX (askForKernelRestart=false, output scrolling)
.vscode/keybindings.json    Cmd+E Cmd+J / Cmd+E Cmd+S / Cmd+E Cmd+W
.vscode/d2l.code-snippets   slide / subslide / frag / pyin / pytab / fwtag
```

`make kernels` registers `d2l-pytorch`, `d2l-tensorflow`, `d2l-jax`,
`d2l-mxnet` ipykernels from the per-framework venvs so VS Code can
auto-select interpreters.

---

## Authoring workflow

**Pure prose / slide-marker edit:**

1. Edit `chapter_linear-regression/linear-regression.md`.
2. Save.
3. (If `D2L: Watch Slides` is running) browser tab reloads in <500 ms.

**Code edit, single framework:**

1. Open the source `.md`.
2. `Cmd+E Cmd+J` → pick `jax`. JAX notebook opens with `d2l-jax` kernel.
3. Edit a cell, run it (`Shift+Enter`), see output, save (`Cmd+S`).
4. Extension's daemon syncs back to source `.md`. Cached output stays
   in the notebook; injected into HTML/slides on next render.

**Verify across frameworks:**

1. `Cmd+E Cmd+S` → pick `pytorch`. PyTorch notebook regenerates from
   the just-updated source. The edited cell's `execution_count` is
   cleared (outputs are framework-specific).
2. Run the cell. Save.

**Build for readers:**

1. `make slides` (parallel, <5 min, no GPU contention).
2. `make html`, `make pdfs` (unchanged).
3. Notebooks already executed during editing; no separate
   `run-notebooks` step for casual changes. `make run-all-notebooks`
   for clean rebuilds or CI.

---

## Migration scripts (one-shot)

Run once over the corpus to bring it to the new state. Each writes back
to source `.md` files atomically and is idempotent.

1. **`tools/add_cell_ids.py`** — assigns `#<id>` to every code fence
   and inserts `<!-- d2l:p <id> -->` comments before every prose
   paragraph and inside every `:begin_tab:` block. Section-slug-derived
   IDs with hash on collision.
2. **`tools/strip_tab_all.py`** — removes `#@tab all` and `%%tab all`
   lines. ~338 occurrences.
3. **`tools/migrate_slide_markers.py`** — converts inline markers
   (`[**…**]`, `(**…**)`, `[~~…~~]`, `(~~…~~)`) into
   `::: {.slide}` divs with `@<id>` placeholders for code cells that
   appear within the slide group. Heading-level `[**…**]` becomes
   `::: {.slide title="<heading>"}`. Continuations `(**…**)` join the
   previous slide. `[~~…~~]`-only content stays slide-only (becomes a
   slide div with no corresponding book content).

Run order: 1, 2, 3. Verify with `make all-quick` byte-for-byte against a
pre-migration build (allowing for added cell metadata and the new
`<!-- d2l:p -->` comments, which Quarto strips from rendered output).

---

## Files added

```
tools/add_cell_ids.py           Migration: assign #<id> to fences, <!-- d2l:p -->
tools/strip_tab_all.py          Migration: remove #@tab all / %%tab all
tools/migrate_slide_markers.py  Migration: inline markers → slide divs
tools/sync_back.py              Notebook → source sync
tools/watch_slides.py           Slide preview daemon
tools/lint_source.py            Source linter
tools/run_cells.py              Per-cell notebook output cache (lowest priority)

_d2l-slides.scss                Slide layout SCSS

.vscode/extensions.json         Recommend extensions
.vscode/settings.json           File exclusions, kernel UX
.vscode/keybindings.json        Edit / switch / watch shortcuts
.vscode/d2l.code-snippets       slide / subslide / frag / pyin / pytab / fwtag

.vscode-extension/              TypeScript extension source
  package.json                  Commands, activation, status bar contributions
  tsconfig.json
  src/extension.ts              Activation, watchers
  src/sync.ts                   Sync orchestration
  src/conflict.ts               3-way diff UI
  src/codelens.ts               Reveal source for cell
  src/lint.ts                   Problem provider

.pre-commit-config.yaml         Run lint_source on changed .md
.d2l-sync-state/                Runtime state (gitignored)
```

## Files modified

```
tools/d2l_preprocess.py         #<id> in fence info string; extract_tab → 'all';
                                strip slide divs from book HTML preprocessing
tools/gen_slides.py             Slide div parser; @<id> resolution; @d2l.add_to_class strip;
                                eval: false; GPU scheduler deleted (~150 lines net)
tools/gen_notebooks.py          cell.metadata.id; prose-cell headers; kernelspec.name;
                                hidden d2l setup cell
tools/inject_outputs.py         Slides mode (~+100 lines); ID-based matching
tools/build_lib.py              Audit for tab is None branches; coerce to 'all'
tools/run_notebooks.py          Optionally wraps run_cells.py for cache
docs/syntax.md                  Slide divs, #<id>, "all" by default, prose IDs
docs/architecture.md            Updated dataflow, ID model, authoring workflow
Makefile                        slides target -j4 safe; new kernels, watch-slides,
                                slide-preview, clean-cache targets
_quarto.yml                     Register _d2l-slides.scss as project resource
```

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| ID collisions on section slug | Append 4-char content hash on collision. Migration script logs collisions. |
| Author renames a section, ID looks stale | IDs are immutable. Linter warns "id no longer matches section slug" but doesn't error. |
| Sync-back data loss on crash | Atomic write (write to tmp + rename). State file survives across runs. Extension logs every operation. |
| Source ↔ notebook concurrent edit | 3-way diff in extension; user resolves per cell. State file tracks common ancestor. |
| `@d2l.add_to_class` re-adds in slide eval-true fallback | Decorator stripped at slide-emit time. Never re-executes. |
| Slide injection misses an output | Lint warns on missing ID; per-cell `eval=true` fallback so the deck still builds. |
| `eval: false` slides hide a kernel-side bug | CI runs `make run-all-notebooks` for execution coverage; same as today. |
| Inline marker migration loses content | Migration script is idempotent and dry-run-able; verified against pre-migration `make all-quick` output. |
| Kernel mismatch (wrong venv selected) | `kernelspec.name = "d2l-<fw>"` set explicitly. `make kernels` registers all four. Documented in CLAUDE.md. |
| Per-cell cache produces stale outputs | Hash includes upstream cells. `make clean-cache` always works. |
| New cell inserted by sync-back lacks an ID | sync-back generates a derivation-style ID at insertion; idempotent. |
| `<!-- d2l:p -->` comments visible in raw `.md` | Acceptable — same noise level as `:label:` markers; HTML/PDF/slide rendering strips them. |

---

## Build sequence

The implementation has internal dependencies. This is the order — not
phases, just dependency edges. A single change set can encompass the
whole sequence; intermediate states only need to compile, not ship.

1. **`add_cell_ids.py` + source-side ID parsing.** Write the migration
   script. Run on the corpus. Update `d2l_preprocess.py` to parse
   `#<id>` from fence info strings and `<!-- d2l:p -->` from prose.
   Update `gen_notebooks.py` to set `cell.metadata.id`. Verify
   `make all-quick` is byte-identical (modulo metadata and prose ID
   comments).
2. **`strip_tab_all.py` + tools audit.** Run the script. Make
   `extract_tab()` default to `'all'`. Audit and unify `tab is None`
   branches. Verify `make all-quick`.
3. **Slide div parser + `migrate_slide_markers.py`.** Rewrite
   `gen_slides.py`'s slide-extraction logic from inline markers to
   `::: {.slide}` divs with `@<id>` and `@<id>@<fw>` placeholders.
   Implement `@d2l.add_to_class` decorator stripping. Run the migration
   script.
4. **`gen_slides.py` GPU scheduler deletion.** Replace with a thread
   pool running `quarto render --to revealjs`. Update Makefile `slides`
   recipe to drop GPU flags and add `-j4` safety.
5. **`_d2l-slides.scss` + `_quarto.yml` wiring.** Add the SCSS file;
   register as a project resource; reference from emitted slide `.qmd`
   YAML headers.
6. **`inject_outputs.py slides` mode + ID-based matching everywhere.**
   Add the slides mode. Update html and pdf modes to drop hash
   fallback. Verify `make slides` finishes in <5 min and outputs match
   notebook outputs byte-for-byte.
7. **`watch_slides.py` + `lint_source.py`.** Standalone CLIs. Wire
   `lint_source.py` into pre-commit.
8. **`gen_notebooks.py` enhancements.** Prose-cell headers,
   `kernelspec.name`, hidden d2l setup cell.
9. **`sync_back.py`.** Round-trip code cells (by ID) and prose
   paragraphs (by ID and `:begin_tab:` boundaries). Test cases:
   single-cell edit, multi-cell edit, deletion, insertion, shared-cell
   edit, prose-tab edit.
10. **`make kernels` + `.vscode/` configuration.** Register kernels.
    Add settings, keybindings, snippets.
11. **VS Code extension.** Daemon-hosted file watching, sync
    orchestration, status bar, CodeLens, conflict UI, problem
    provider, command palette commands. Build and ship `.vsix` in repo.
12. **`run_cells.py` per-cell cache.** Lowest priority. Defer until
    items 1–11 are verified working end-to-end.

---

## Resolved design questions

- **Notebooks committed to git?** No. Source `.md` is the only source of
  truth; notebooks are derived artifacts.
- **Daemon process or extension-hosted?** Extension-hosted.
- **`MAX_TEXT_LINES` for slides?** Default 12; per-slide
  `output-lines="N"` attribute overrides.
- **`_d2l-slides.scss` separate from `_d2l-theme.scss`?** Separate.
- **Inline marker deprecation period?** None. Migrated and removed in
  one change.
- **Cell ID stability?** Immutable once written. Behaves like
  `:label:` references.
- **Cross-framework cell references in slides?** `@<id>` auto-resolves
  to the deck's framework. `@<id>@<fw>` forces a specific variant.
- **`@d2l.add_to_class` in slides?** Decorator stripped at emit time;
  function body remains visible. Slides never add to the library.
- **Prose paragraph IDs in source?** Yes — `<!-- d2l:p <id> -->` HTML
  comments. Same immutability rules as code-cell IDs.

---

## What this plan does NOT do

- Does not change the multi-framework HTML book layout for readers.
- Does not change the PDF rendering pipeline structurally (still
  single-framework per PDF, still uses `inject_outputs.py pdf`; only
  the matching logic changes from hash to ID).
- Does not migrate to Jupytext, Marimo, or any other notebook
  alternative.
- Does not introduce a database, frontend framework, or hosted service.
  Everything stays file-based and works offline.
- Does not commit notebook outputs to git.
- Does not change the GPU-parallel notebook *execution* model
  (`run_notebooks.py` is unchanged structurally; slides stop using GPU,
  notebooks still need it).
