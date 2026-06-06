# Build System: Decoupled Notebook Execution & Site Rendering

**Status:** **landed** — this decoupled model is the *current* reality, not a
proposal. The `outputs/` store, `tools/capture_outputs.py`,
`tools/audit_outputs.py`, the `inject_outputs.py` repoint, the Make rewiring
(§6), and the Git-LFS setup (§7) are all in the tree and in use: `outputs/` is
committed (≈1400 manifests + LFS assets), `make html` renders from it and gates
on `tools/audit_outputs.py --verify-fresh`, and `bootstrap.sh` exists. The
execution-*coupled* flow described in `architecture.md` has been **superseded**
by this model (that doc's component inventory is still accurate; its "render
reads `_notebooks/`" claim is historical). See **§16 Implementation status** for
the per-piece state. Read this before changing anything in the
notebook → output → render path.

---

## 0. Mental model (read this first)

The book is produced by **two pipelines that share one narrow, committed seam.**

```
  EXECUTION  (expensive, GPU box, 4 framework venvs, ~110 GPU-min)
     make run-notebooks-<fw>          source .md ─► .ipynb ─► EXECUTE ─► _notebooks/<fw>/ (scratch, gitignored)
                                                                  │
     make capture-outputs   ◄── distills + extracts ─────────────┘
                                                                  ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  THE SEAM:  outputs/  — the committed "golden outputs" store               │
  │    • text outputs       → inline in per-notebook JSON manifests (plain git)│
  │    • image/binary output → asset files (Git LFS)                           │
  │    • per-cell code-provenance fingerprints (freshness)                     │
  └──────────────────────────────────────────────────────────────────────────┘
                                                                  │
  RENDER  (cheap, ANY machine, CPU-only, only .venv-build/Quarto) │
     make html / slides / pdf         source .md + outputs/  ─────┴─► _book/ _slides/ _pdf/
```

The two facts that make this possible already hold in the repo:

1. **Quarto never executes code.** `_quarto.yml` sets `execute.enabled: false`.
   Rendering is a pure text transform; outputs are injected as *static*
   markdown (`::: {.cell-output}` divs, `![](…)` images) by
   `tools/inject_outputs.py`.
2. **Execution and render already talk only through stable cell IDs.**
   `inject_outputs.py` indexes a notebook by `cell.metadata.id`
   (`index_ipynb_by_id`), not by position, and already fingerprints outputs to
   dedup across frameworks (`_output_fingerprint`).

So the *only* thing that ties rendering to execution today is that
`inject_outputs` reads outputs out of the on-disk `_notebooks/` tree. We persist
that tree's *content* (distilled) into git and repoint the readers. Nothing else
about Quarto, numbering, or cross-refs changes.

**The payoff:**

- Render the entire book / slides / PDFs on a laptop with **no CUDA, no
  torch/tf/jax/mxnet, no GPU** — only `.venv-build`.
- Iterate on layout, SCSS, the diagram engine, and the slide redesign with **zero
  notebook execution**.
- Re-run only the notebooks whose **code actually changed**, bless just those, and
  leave the rest of the book on its committed outputs — the book is always
  fully renderable and internally consistent in between.

---

## 1. Why decouple

Today every site target (`html`, `slides`, `pdf`) hard-depends on a locally
populated `_notebooks/<fw>/` tree, which means a full site rebuild requires the
four mutually-exclusive framework venvs, a CUDA stack, and ~110 GPU-minutes of
execution — *even when the change is a CSS tweak or a new slide diagram.* That
makes the high-value, low-risk work (layout, slides, visual quality) pay the
price of the high-cost, high-variance work (notebook execution). Decoupling lets
each run on its own cadence.

A second motivation: **partial re-execution**. Most edits touch a handful of
notebooks. We want to re-run and re-publish exactly those, with a mechanism that
keeps every index correct under partial execution (see §6).

---

## 2. The committed outputs store (`outputs/`)

### 2.1 Location and layout

The store is committed at repo root under `outputs/` (no leading underscore,
to distinguish it from the gitignored `_notebooks/`, `_book/`, etc.):

```
outputs/
  <fw>/                                   pytorch | tensorflow | jax | mxnet
    <chapter>/
      <stem>.json                         per-notebook MANIFEST  (plain git)
      <stem>/                             per-notebook ASSET dir (Git LFS)
        <cellid>-<n>.png
        <cellid>-<n>.svg
        ...
```

One manifest per `(source-file × framework)` — the same granularity as
execution. Re-running one notebook rewrites exactly one manifest and the asset
files under one directory: clean, reviewable, merge-conflict-free.

### 2.2 The split: text in git, binary in LFS

This is the core storage decision (the "split pattern").

| Output kind | Where it lives | Why |
|---|---|---|
| **Text** (`stream`, `text/plain`, `text/html`, `text/markdown`, error tracebacks) | **Inline** in the `.json` manifest, plain git | Reviewable diffs ("loss 0.34 → 0.31" shows as a one-line change); you *want* this history. |
| **Images / binary** (`image/png`, `image/svg+xml`, `image/jpeg`, anything else) | **Asset file** under `outputs/<fw>/<chapter>/<stem>/`, tracked by **Git LFS** | Training plots churn every run (random init, shuffling, dropout, cuDNN autotune). Every blessed re-run adds image bytes permanently; LFS keeps them out of the packfile so clones stay lean and old versions are prunable. |

Rationale for putting **all** text inline (not just "small" text): text is
reviewable and compresses well; the churn problem that motivates LFS is specific
to binary blobs. A safety valve caps pathological text dumps (see §2.3,
`max_inline_bytes`) — over the cap, the text is head/tail-truncated inline with
an explicit `…[truncated N bytes]…` marker (the full book already trims noisy
output; nothing reader-facing is lost). SVG is treated as binary (an asset) even
though it is technically text, because matplotlib SVGs are large and churn like
PNGs; *authored* diagram SVGs are a different thing and live in `img/auto/`
(see §13).

> **Asset paths are stable, never content-addressed.** An asset is named
> `<cellid>-<n>.<ext>` and **overwritten in place** on re-capture. Content-
> addressed (hash) filenames would spawn a new blob and orphan the old one on
> every stochastic re-run, forcing a GC sweep. Stable paths give a bounded
> working tree and no garbage. (The content *hash* is still recorded in the
> manifest — for freshness and cross-framework dedup — just not used as the
> filename.)

### 2.3 Manifest schema

`outputs/<fw>/<chapter>/<stem>.json`:

```jsonc
{
  "schema": 1,
  "source": "chapter_preliminaries/ndarray.md",
  "framework": "pytorch",
  "provenance": {
    // Notebook-level invalidation keys (compared whole; see §3).
    "framework_version": "torch==2.11.0+cu128",   // or the mxnet wheel tag, etc.
    "d2l_lib_fingerprint": "sha256:…",            // hash of just the d2l/_blocks/<fw>/*.py
                                                  //   this notebook uses (from its .d file)
    "max_inline_bytes": 4096
  },
  // NOTE: the manifest is a *pure function of the executed notebook + repo state*.
  // No wall-clock or HEAD-commit fields live here — otherwise re-capturing an
  // unchanged notebook would churn git. The capture commit is recoverable from
  // `git log` of the manifest itself.
  "cells": {
    // keyed by STABLE cell id (tools/add_cell_ids.py)
    "ndarray-getting-started-5": {
      "code_fingerprint": "sha256:…",   // hash of normalized code of cells 0..i; see §3.1
      "kind": "text",                   // "text" | "asset" | "mixed"
      "outputs": [
        { "type": "stream", "name": "stdout",
          "text": "tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])" }
      ]
    },
    "ndarray-broadcasting-3": {
      "code_fingerprint": "sha256:…",
      "kind": "asset",
      "outputs": [
        { "type": "display_data", "mime": "image/png",
          "asset": "ndarray/ndarray-broadcasting-3-0.png",   // path under outputs/<fw>/<chapter>/
          "bytes": 48213, "sha256": "…" }
      ]
    }
  }
}
```

- Cell order is irrelevant to correctness (lookup is by id), but the writer emits
  cells in notebook order so diffs read top-to-bottom.
- A cell with **no** output has an entry with `"outputs": []` (so the audit can
  tell "ran, produced nothing" from "missing").
- The manifest is **always machine-generated** by `capture-outputs`. It is never
  hand-edited. (See §3.5.)

---

## 3. Freshness model (the heart of the design)

Because stochastic outputs change every run, freshness **cannot** be defined as
"does the committed output match a fresh run?" — that would flag every training
notebook as stale forever. Freshness is defined on the **provenance of the code
that produced the output**, which is deterministic and randomness-immune.

### 3.1 The two freshness signals

Freshness is decided by two complementary, content-derived signals — kept
*separate* so the per-cell hash stays a pure function of code (deterministic, no
churn) while coarse environment changes invalidate at notebook granularity.

**(a) Per-cell `code_fingerprint`** = `sha256` of the **normalized source of that
cell and every preceding code cell** in the *generated* notebook (prefix-inclusive
— a cell's output depends on the state earlier cells built, so editing cell 3
marks cells 3..N stale, not just cell 3). "Normalized" = `rstrip` each line, join
with `\n`. The cell id lives in `metadata.id`, not the source, so it never enters
the hash. This signal pinpoints *which* cells drifted.

**(b) Notebook-level provenance** (stored once per manifest, compared whole):

- **`d2l_lib_fingerprint`** — hash of exactly the `d2l/_blocks/<fw>/*.py` block
  files this notebook uses, read straight from its existing `.d` file
  (`tools/scan_d2l_usage.py` already emits one per notebook → per-symbol block
  paths). A `#@save` change therefore marks *precisely* the notebooks that use
  the changed symbol stale — replacing CLAUDE.md's "when in doubt rerun all
  frameworks."
- **`framework_version`** — the installed framework version, keyed on the
  PEP 440 **public** version only (`tools/capture_outputs.py:public_version`
  strips the local build segment, e.g. `torch==2.11.0+cu128` → `torch==2.11.0`).
  The local segment names the *platform wheel*, not the version: torch 2.11.0 is
  torch 2.11.0 whether it's the Linux CUDA-12.8 build or the macOS arm64 CPU/MPS
  build, and the same code produces the same results either way. Keying on the
  full wheel string would hard-stale the *entire* store the instant you audit
  from a different machine — defeating portability. A genuine version change
  (`2.11.0 → 2.12.0`, or a MXNet-wheel rebuild) still invalidates. The audit
  also normalizes legacy `+cu128` manifests on read, so old captures match.

### 3.2 Stale ⟺ code or provenance drift

A notebook is **stale** iff either provenance key differs from the current
environment, **or** any cell's recomputed `code_fingerprint` differs from the
stored one. (A provenance mismatch marks the whole notebook; a code-fingerprint
mismatch marks that cell and — being prefix-inclusive — its successors.) This is:

- **Randomness-immune** — a training plot that *would* look different on a re-run
  is **not** stale, because its code didn't change; the committed plot remains a
  valid representative sample.
- **Precise** — `tools/audit_outputs.py` reports the exact stale cells/notebooks,
  i.e. the minimal re-execution set.

### 3.3 Gate severity: inline text hard, assets soft

A *stale* output means different things for the two storage classes:

- **Inline text (`kind: "text"`) → HARD gate.** A stale text output is
  *definitively wrong* relative to the code shown beside it (a reader sees
  `arange(12)` but an old printout), and it's cheap to keep current. The render
  **fails** on a stale inline output with a precise message.
- **Asset (`kind: "asset"`) → SOFT gate.** A stale plot is an old-but-valid
  sample; the render **warns** but proceeds. You refresh it on a deliberate
  bless.

### 3.3a Host-capability-aware gate: render anywhere

The HARD gate of §3.3 only makes sense where the author could *act* on it. A
stale **GPU** notebook on a GPU-less Mac, or a stale **multi-GPU** notebook on a
single-GPU box, cannot be re-executed there — hard-failing the render would
block a machine from building the book over a notebook it could never fix,
defeating the portable-store promise (a CPU box should render the whole book and
re-run only what its hardware supports).

So `verify-fresh` is **capability-tiered**. Each notebook's resource class
(`cpu` / `gpu` / `multi-gpu`, from `tools/scan_notebook_manifests.py:source_execution_class`,
which keys off `runtime_env.MULTI_GPU_NOTEBOOKS` + GPU keyword scan) implies a
GPU floor — `cpu`=0, `gpu`=1, `multi-gpu`=2. `tools/audit_outputs.py` counts the
host's GPUs (`nvidia-smi -L`; 0 if absent) and, for each HARD-stale notebook:

- host GPUs **<** the class floor → the host *can't* run it → **deferred to a
  WARNING**, rendered from the committed store;
- host GPUs **≥** the floor → the host *can* run it → still **hard-fails**.

This tiers cleanly:

| Host | Renders | Hard-fails on stale… |
|------|---------|----------------------|
| CPU only (0 GPU, e.g. Apple Silicon) | everything | nothing (cpu notebooks are runnable → still block) |
| Single-GPU | everything | `cpu` + `gpu` stale (defers only `multi-gpu`) |
| Multi-GPU (≥2) | everything | all (the strict canonical gate) |

The canonical green build runs on the multi-GPU box, where the gate is strict
and refreshes whatever drifted; CPU/single-GPU authors still get a loud warning
listing the deferred notebooks and render from the store meanwhile.

### 3.4 The d2l `#@save` ripple, made precise

CLAUDE.md currently says a library rebuild "can affect notebooks outside the
edited file and outside the edited framework — rerun all plausibly affected
notebooks; when unsure, rerun all frameworks." Under this model that guesswork
disappears: the `d2l_lib_fingerprint` in each manifest covers exactly the symbols
that notebook uses, so `audit_outputs.py` names the exact blast radius.

### 3.5 "Regenerated when a new render occurs" (the requirement, made concrete)

Inline outputs are a **regenerated snapshot, not a frozen cache:**

- **Never hand-authored.** Every `make capture-outputs` rewrites all inline
  outputs wholesale from the freshly executed notebook.
- **Never rendered stale.** Every render target depends on
  `verify-outputs-fresh`, which hard-fails if any *inline* output's
  `code_fingerprint` has drifted from current source:

  ```
  outputs/pytorch/chapter_x/foo.json  cell `foo-train-7`:
    inline output stale — source changed since capture.
    Fix: make _notebooks/pytorch/chapter_x/foo.executed && make capture-outputs FILES=chapter_x/foo.md
  ```

  So no render can ship a stale inline output: it is regenerated (via the forced
  re-run + capture) before any render that would otherwise present a drifted
  value.

- **Optional auto-regeneration.** `make render-fresh` = audit → re-execute *only*
  the stale notebooks → capture → render. This needs the relevant framework
  venv/GPU (it executes), so it is *not* the default; the default render stays
  CPU-only and simply refuses to proceed on stale inline outputs. Choose
  `render-fresh` when you're on the GPU box and want a one-shot "make it
  current and render."

This is the contract that lets us keep deterministic small outputs inline safely:
they are kept correct by construction.

---

## 4. Index & referential integrity under partial execution

The worry with partial re-runs is "do the indices stay correct?" There are two
distinct kinds of index; only one is even touched by execution.

### 4.1 Execution-independent: numbering & cross-references

Chapter/section/equation/figure numbers, `:numref:`/`:eqref:`/`:cite:`, and the
`fix_crossref_numbers.py` renumber pass are computed **at render time from the
full source set** (all `.md` are always in git) and the static
`CHAPTER_NUMBERING` map — **not from execution.** Re-running zero notebooks or
all notebooks yields *identical* numbering. Partial execution cannot perturb
these. (Generated plots are cell *outputs*, not `:numref:`-labeled figures —
labeled figures are authored SVGs in `img/` — so output churn doesn't touch
figure numbering either.)

**The map is the single source of truth for numbers.** `CHAPTER_NUMBERING` (in
`tools/d2l_preprocess.py`) maps each source `.md` to its logical number: `[N]` =
chapter *N* (a chapter-group's `index.md`), `[N, k]` = section *N.k*, `None` =
unnumbered (front matter, references). `fix_crossref_numbers.py` **imports this
same dict**, so preprocessing and the post-render renumber pass never disagree.
Two consequences worth knowing:

- **A file *absent* from the map renders unnumbered.** The preprocessor treats a
  missing key as `None` and prepends `---\nnumber-sections: false\n---` front
  matter, so Quarto emits the page with no chapter/section number. To give a new
  chapter real numbers you **must** add it to `CHAPTER_NUMBERING` — listing it in
  `_quarto.yml` alone is not enough. (This was exactly why the
  Mathematics-for-Deep-Learning chapters first rendered without numbers.)
- **Part titles in `_quarto.yml` are cosmetic.** `build_chapter_map` flattens the
  `chapters:` lists and ignores the `part:` strings, so whether *N* groups live
  under one part or *N* parts has **zero** effect on numbering — only the map and
  the flat file order do. Part grouping is purely the left-nav/sidebar structure.
  (A part whose `part:` is a bare title string keeps its `index.md` as a numbered
  chapter; pointing `part:` at a `.qmd` file instead makes that page an *unnumbered
  divider*, which breaks any `:numref:` that targets it — so we use title strings.)

Current tail of the map: the **Mathematics for Deep Learning** part is chapters
**22–27** (Linear Algebra 22, Calculus 23, Optimization 24, Probability &
Statistical Learning 25, Information Theory 26, Dynamics 27) — each group's
`index.md` is the chapter, its siblings the `N.k` sections — and the **Tools for
Deep Learning** appendix is chapter **28**. The legacy
`chapter_appendix-mathematics-for-deep-learning/` part (formerly chapter 22) has
been retired; inserting or retiring a group means renumbering the map's tail and
the matching `_quarto.yml` order together.

### 4.2 The one index that matters: cell-id → output

Injection is keyed by **stable cell IDs**. `tools/add_cell_ids.py` is idempotent
and **never renumbers** an id, even on rename or move. The store is keyed by
`(source-file, framework, cell-id)`, so re-running notebook *X* updates only
*X*'s entries and cannot perturb anyone else's. **Cell-id stability is the load-
bearing invariant** — never break it.

### 4.3 The integrity gate

`tools/audit_outputs.py` enforces referential integrity (an extension of the
existing `audit_slides.py` slide↔cell check) so partial blessing can't silently
break a reference:

- Every `@id` / injected cell a page or slide references **must** have a live
  store entry. (Dangling reference → fail.)
- Every store entry **must** map to a live source cell. (Orphaned output → warn /
  prune.)
- Every executed notebook cell with an id **should** be in the store. (Missing →
  warn: capture skipped it.)

Run it in CI so a dangling id fails loudly instead of rendering a blank.

---

## 5. Lifecycle & workflows

### 5.1 Execute → Capture → Commit (on the GPU box)

```bash
# 1. Execute (unchanged): writes outputs into _notebooks/<fw>/ (scratch).
make run-notebooks-pytorch                       # or one stamp:
make -B _notebooks/pytorch/chapter_x/foo.executed

# 2. Capture/bless: distill _notebooks/ → committed store + LFS assets.
make capture-outputs                             # all that changed since last capture
make capture-outputs FILES=chapter_x/foo.md      # just one source file (all fw)

# 3. Review and commit deliberately.
git status outputs/                              # text diffs are meaningful
git add outputs/ && git commit -m "outputs: refresh chapter_x/foo (pytorch)"
git push        # use gh/HTTPS — see CLAUDE.md memory; LFS pushes blobs here
```

`capture-outputs` only reads the notebooks that are newer than their manifest (or
those named by `FILES=`), so a partial execution produces a partial capture
produces a partial commit. That is the whole point.

### 5.2 Render (on any machine, CPU-only)

```bash
uv sync --extra build          # only Quarto; no framework, no CUDA
make html                      # reads outputs/ ; fails fast if inline outputs stale
make -j4 slides
make -j4 pdfs
```

A fresh clone with only `.venv-build` renders the whole book. (LFS assets are
fetched on clone; `GIT_LFS_SKIP_SMUDGE=1` gives a text-only checkout if you only
need to edit prose/slides and don't care about plot images yet.)

This holds on *any* host, including one that has framework venvs and so can
detect staleness: per §3.3a the gate is capability-tiered, so a CPU or
single-GPU box renders the whole book — deferring (with a warning) only the
stale notebooks it lacks the GPUs to re-execute, which the multi-GPU box
refreshes on the canonical build. To re-run a CPU notebook locally without a
full rebuild: `make -B _notebooks/<fw>/<chapter>/<file>.executed` then
`make capture-outputs FILES=<chapter>/<file>.md`.

### 5.3 Partial / selective refresh (convenience)

```bash
make audit-outputs             # report stale notebooks (the to-do list)
make refresh-stale             # audit → re-execute ONLY stale notebooks → capture
```

### 5.4 The to-do model

Staleness (what *could* be refreshed) and blessing (the explicit run-+-commit
act) are separate. `audit-outputs` is the to-do list; you may bless a subset.
Un-blessed stale notebooks keep rendering their committed outputs and stay on the
list until you get to them (inline-stale ones block render and must be done
first; asset-stale ones only warn). The book never breaks in between.

---

## 6. Make interface

### 6.1 Dependency rewiring (the actual decoupling)

| Target | Before | After |
|---|---|---|
| `html`, `slides-<fw>`, `pdf-<fw>` | depend (transitively) on `_notebooks/` / `.executed` | depend on **`outputs/` + `verify-outputs-fresh`**; **never** trigger execution |
| `inject_outputs.py {html,pdf,slides}` | read `_notebooks/<fw>/*.ipynb` | read `outputs/<fw>/…` (manifest + asset paths) |

Editing a slide block, SCSS, a diagram, or page layout therefore never makes
Make want to execute a notebook.

### 6.2 New / changed targets

| Target | Role |
|---|---|
| `capture-outputs [FILES=…]` | Distill `_notebooks/` → `outputs/`; extract assets; recompute fingerprints. The bless step. |
| `audit-outputs` | Report stale notebooks/cells + integrity (dangling/orphaned ids). Non-zero exit on integrity failure. |
| `verify-outputs-fresh` | Render prerequisite: hard-fail on any stale **inline** output; warn on stale assets. |
| `refresh-stale` | `audit-outputs` → re-execute only stale notebooks → `capture-outputs`. (Needs framework venvs.) |
| `render-fresh` | `refresh-stale` then `html`/`slides`/`pdf`. (Needs framework venvs.) |

`run-notebooks-*`, the `.executed` stamps, `.generated`, `notebooks-*`, `lib`,
and `.d` dep files are **unchanged** — execution works exactly as today; capture
is a new consumer of its results.

### 6.3 Gotcha: don't hand-delete `.preprocess.stamp` under GNU Make 3.81

The `.d` auto-dependencies are `-include`d, so every `make` first remakes those
files (the `scan_d2l_usage` lines) and **re-execs itself**. Under the macOS
system **GNU Make 3.81**, if you manually `rm .preprocess.stamp`, that re-exec
can leave `make html` deciding `_book/index.html` is already up to date and skip
preprocessing **and** rendering entirely — it prints only the trailing
`Output:`/`Log:` echoes and exits 0, fooling you into thinking it rebuilt. Force
the stamp explicitly first (`make .preprocess.stamp`) or touch a real
prerequisite (`touch _quarto.yml`); then `make html` renders normally. Better
still, use `make -B <target>` when you deliberately want a forced rebuild rather
than deleting stamps by hand. (GNU Make ≥ 4.3 resolves the missing stamp
correctly; 3.81 also emits harmless `overriding commands for target '&'` warnings
because it predates the `&:` grouped-target syntax the Makefile uses.)

Note also that the preprocess step is a **single pass** — `d2l_preprocess.py`
with no `--files` regenerates every `.qmd` for the files in `CHAPTER_NUMBERING`.
(The former second pass that explicitly re-ran the `chapter_mdl-*` files was a
workaround for their absence from the map; now that they are in it, the main pass
covers them and the workaround has been removed.)

### 6.4 Aggregate / full-build targets (the "rebuild everything" path)

The surgical flows in §11 are the day-to-day path; they deliberately avoid full
rebuilds. But a *whole-book* rebuild is still real — initial population, the
canonical green run on the multi-GPU box, or a "clean and rebuild the world"
request. Those run through the aggregate targets:

| Target | Expands to | Executes notebooks? |
|---|---|---|
| `make all` | `lib` → `notebooks` → `run-all-notebooks` → `rebuild-book-artifacts`. The full pipeline; ~3 h on the 4×4090 box. | **Yes** (all 4 fw) |
| `make all-quick` | `lib` → `notebooks` → `rebuild-book-artifacts`. Regenerate + render from the **committed** `outputs/`; no execution. | No |
| `make rebuild-book-artifacts` | `slides` → `html` → `-j4 pdfs` → `check-all-artifacts`. Renders everything from `outputs/`. | No |
| `make check-all-artifacts` | Asserts `_book/index.html`, `_slides/index.html`, `_book/slides/index.html` exist. | No |
| `make clean` | Wipe build products (`_book _pdf _notebooks _slides`, generated `.qmd`, `d2l/.built`, stamps). **Keeps** `./data/`, `logs/`, `.upload-manifest-*.txt`. | — |
| `make veryclean` | `clean` **plus** wipe `./data/` (datasets), `logs/`, upload manifests — forces dataset re-download. | — |

Things that bite if you don't know them:

- **`make all` does NOT bless.** Its body is `lib; notebooks; run-all-notebooks;
  rebuild-book-artifacts` — there is **no `capture-outputs`**. Execution writes
  into `_notebooks/` (scratch), but the render reads the **committed `outputs/`
  store** (`inject_outputs.resolve_id_outputs` prefers the store whenever a
  manifest exists; §8). So the freshly-executed results do **not** appear in the
  rendered site until you run `make capture-outputs` and re-render. `make all`'s
  job is to (a) re-execute and surface failures and (b) let the freshness gate
  verify the committed store still matches source — *not* to publish new outputs.
  The "execute → **capture** → render" publish path is flow A in §11.1.
- **`make all` tolerates notebook failures.** The execution queues run under
  `make -k`; `all` records the exit code but still runs `rebuild-book-artifacts`,
  so the site/slides/PDFs are produced from available outputs and `make all`
  *then* exits non-zero. A non-zero `make all` therefore does **not** mean "no
  artifacts." Per-notebook tracebacks land in
  `logs/nb-errors/<fw>/<chapter>/<file>.log`; `tools/notebook_run_summary.py`
  gives the pass/fail roll-up.
- **"Clean everything" usually means `clean`, not `veryclean`.** Datasets in
  `./data/` are *inputs*, not build products — re-downloading them cannot change
  any output, only cost time/network. Reach for `veryclean` only to re-exercise
  the download path itself.
- **A local-wheel-tag bump does not auto-stale.** `framework_version` keys on the
  PEP 440 *public* version (§3.1), so e.g. bumping the MXNet wheel
  `2.0.0+cu13.bw.20260529` → `…20260529.3` (public version still `2.0.0`) does
  **not** mark the store stale — the gate passes and the committed outputs render
  unchanged. Only a public-version change (`2.0.0 → 2.1.0`) or a per-cell code
  change invalidates. Re-capture after a same-public-version wheel rebuild is a
  deliberate choice, not something the gate forces.

### 6.5 Updating a framework wheel pin (MXNet)

The custom MXNet wheel is pinned by URL in `pyproject.toml`. To move to a newer
published build:

```bash
python3 tools/update_mxnet_wheel.py --source github   # pin the latest GitHub *release* asset
uv lock && make venv-mxnet                            # relock + install into .venv-mxnet
```

Caveats (each learned the hard way):

- **`--source github` is not the default.** Bare
  `python3 tools/update_mxnet_wheel.py` (and `make update-mxnet-wheel`) defaults
  to `--source local`, which scans `../mxnet/dist/` for a `file://` wheel — for
  iterating on a local build that outpaces GitHub. Pass `--source github` to
  track the published release.
- **The tool's rewrite strips platform markers.** Its regex re-emits only
  `; python_version == '3.12'`. If `pyproject.toml` carries *multiple*
  platform-specific mxnet pins (a linux `x86_64` line **and** a macOS `arm64`
  line), running it drops `and sys_platform == '…' and platform_machine == '…'`
  from the first match — which then resolves the linux wheel on macOS too and
  breaks the lock there. With multiple pins, **bump the URL by hand** (a surgical
  edit preserving the markers) rather than running the tool.
- **The `uv sync` preflight ignores `https://` pins.** `tools/preflight_mxnet_pin.py`
  only auto-repairs a `file://` pin whose wheel went missing; a GitHub URL pin
  returns 0 immediately, so it won't clobber your bump.
- After moving the wheel, see §6.4 on staleness: a same-public-version rebuild is
  *not* auto-flagged, so re-run + re-capture MXNet only if you want the new
  wheel's outputs in the store. Runtime gotchas specific to the wheel live in
  `docs/mxnet-runtime-diagnostics.md`.

### 6.6 PDF build (XeLaTeX via Quarto) — pitfalls and how it's kept green

`make pdfs` (and `make all`) renders one `_pdf/<fw>/` Quarto book per framework
→ `quarto render --to pdf` → XeLaTeX → `_pdf/<fw>/_pdf/Dive-into-Deep-Learning-<fw>.pdf`,
then copies it into `_book/pdf/` for the site. The render is **CPU-only** and
reads the committed `outputs/` store like every other artifact (§6.1). Four
pitfalls bit us; each now has a guard, so a clean `make all` builds all four PDFs
(~46 MB each) with no manual steps:

- **`static/d2l-preamble.tex` is a tracked SOURCE file.** It is the hand-written
  LaTeX preamble (`gen_pdf.py` injects it via `include-in-header`) and **must**
  exist for the PDF to build. The repo-wide `*.tex` ignore rule (for *generated*
  intermediates) used to swallow it, so it was never committed and PDFs only
  built on machines that happened to have it locally. `.gitignore` now carries
  `!static/d2l-preamble.tex`. If you add another hand-written `.tex`, add a
  matching negation.
- **`mathtools` must be in the preamble.** pandoc's LaTeX writer normalizes
  `\left(\begin{smallmatrix}…\end{smallmatrix}\right)` into `\begin{psmallmatrix}`,
  which `amsmath` alone does not define → *"Environment psmallmatrix undefined."*
  The preamble loads `mathtools`. (Authoring tip: write parenthesized small
  matrices in source as `\left(\begin{smallmatrix}…\right)` — portable to HTML
  MathJax — and let pandoc emit the mathtools form for PDF.)
- **A `$` immediately followed by a digit does not close inline math** (pandoc's
  "don't make `$5 … $10` math" rule). `$\mathcal F=\{$1-Lipschitz$\}…$` therefore
  mis-parsed and leaked `\mathcal` into text mode → *"\symcal allowed only in
  math mode."* Keep math + adjacent digits inside a single span, using
  `\text{1-Lipschitz}` for the words. (`grep -nE '\$[0-9]'` over a math-heavy
  chapter is a quick smell test, though most hits are legitimate *opening*
  `$1\times 1$`.)
- **Stale PDFs in `img/outputs/` abort the parallel render.** Quarto's
  `convert_svg` (built-in `main.lua`) has a "skip conversion if the `.pdf`
  already exists" optimization, then `assert(read(pdf) ~= nil)`. A leftover
  corrupt/empty scratch PDF there makes the assert fail (`main.lua:7348`), killing
  the whole PDF render — and `img/outputs/` is gitignored scratch that survives
  across builds. Both `make clean` *and* `rebuild-book-artifacts` (just before
  `make -j4 pdfs`) now `find img/outputs -name '*.pdf' -delete`, so `make all` is
  self-sufficient. For a bare standalone `make pdfs` on a dirty tree, run
  `make clean` first if you suspect stale scratch.

Not fixed here (cosmetic, non-fatal): a few `:numref:`fig_mdl-dyn-*`` references
in the still-incomplete Dynamics chapter render as "?" — the figures, SVGs, and
labels exist, but Quarto fails to register the crossref (math-heavy captions are
the suspect). This warns but does **not** stop the PDF build, and it is present in
the HTML too.

---

## 7. Git & LFS setup (one-time)

`git-lfs` is **installed** on the build host (`git-lfs/3.4.1`). The per-clone
hooks still need to be set up once in each working copy:

```bash
git lfs install                # once per clone (installs the smudge/clean hooks)
```

`.gitattributes` (new):

```gitattributes
# Notebook-output image/binary assets → LFS. Manifests (*.json) stay plain git.
outputs/**/*.png  filter=lfs diff=lfs merge=lfs -text
outputs/**/*.jpg  filter=lfs diff=lfs merge=lfs -text
outputs/**/*.jpeg filter=lfs diff=lfs merge=lfs -text
outputs/**/*.svg  filter=lfs diff=lfs merge=lfs -text
outputs/**/*.gif  filter=lfs diff=lfs merge=lfs -text
```

`.gitignore` change: `outputs/` is **committed** — make sure no broad rule
ignores it (the existing `_notebooks/ _book/ _pdf/ _slides/ img/outputs/` rules
already do not touch it). `img/outputs/` stays gitignored: in the new model it is
a *render-time* scratch copy that `inject_outputs` may still materialize from the
committed store; the committed truth is `outputs/`.

**CI / clone implications:** LFS must be available wherever git operations run
against these blobs. The R2 upload step (`tools/upload_r2.sh`) consumes the
rendered `_book/`, so it is unaffected by LFS. Use `gh`/HTTPS for push (the
yubikey-free path recorded in CLAUDE.md memory); LFS rides the same HTTPS remote.

---

## 8. Tooling contracts

| Script | Status | Contract |
|---|---|---|
| `tools/capture_outputs.py` | **NEW** | `_notebooks/<fw>/<chapter>/<stem>.ipynb` → `outputs/<fw>/<chapter>/<stem>.json` + asset files. Idempotent: identical inputs → byte-identical manifest (stable key order, no timestamps in the hashed payload). Writes assets to stable per-cell paths (overwrite in place); records `bytes`+`sha256` per asset. Computes `code_fingerprint` per §3.1. |
| `tools/audit_outputs.py` | **NEW** | Freshness (§3.2) + integrity (§4.3). Exit non-zero on integrity failure; `--stale` lists the minimal re-execution set; `--json` for tooling. |
| `tools/inject_outputs.py` | **CHANGED** | Output source flips from `_notebooks/` to `outputs/` via a new `index_store_by_id()` that returns the **same shape** as `index_ipynb_by_id()` — `{cell_id: [nbformat-output-dict]}` — by reconstructing nbformat dicts from the manifest (inline text → `stream`/`text/plain` dicts; assets → re-read the file, re-encode to the dict's `data`). Everything downstream (`format_cell_output`, dedup, markup) is **unchanged**, so injected output is byte-identical to the `_notebooks/` path. Auto-detects: uses `outputs/` when a manifest exists, else falls back to `_notebooks/`. |
| `tools/add_cell_ids.py` | unchanged | Still the authority for stable ids — the invariant capture/audit rely on. |
| `tools/scan_notebook_manifests.py` | unchanged | Execution queues. Capture reuses its file enumeration. |
| `tools/scan_d2l_usage.py` | unchanged | Emits the per-notebook `.d` files capture reads for `d2l_lib_fingerprint`. |

Determinism requirement for `capture_outputs.py`: the manifest must be a pure
function of the executed notebook + repo state (sorted keys, fixed float
formatting, **no wall-clock or HEAD-commit fields**). Otherwise re-capturing an
unchanged notebook would churn git for no reason. The integration test for this
is "capture twice → `git diff` empty."

---

## 9. CI gates

1. `make audit-outputs` — integrity must pass (no dangling/orphaned ids).
2. `make verify-outputs-fresh` — no stale **inline** outputs on `main`
   (stale assets are allowed as a warning; they're valid old samples).
3. `make html && make -j4 slides` — renders clean from the committed store with
   **no framework venv** (proves the decoupling actually holds — if someone
   reintroduces an execution dependency, this CPU-only job fails).

---

## 10. Migration (one-time, from the current green build)

The current `_notebooks/` tree is green (129/129 MXNet, all frameworks). Bless it
into the store once, then flip the readers.

```bash
# 0. Prereqs (git-lfs/3.4.1 already installed; just wire the clone hooks)
git lfs install
git checkout -b decouple-outputs

# 1. Land .gitattributes (§7) BEFORE adding any asset, so blobs go to LFS, not git.
git add .gitattributes && git commit -m "lfs: track outputs/ image assets"

# 2. Build the new tools (capture_outputs.py, audit_outputs.py) and repoint
#    inject_outputs.py per §8.

# 3. Bless the current executed tree wholesale.
make capture-outputs                      # full sweep of _notebooks/ → outputs/
make audit-outputs                        # expect: integrity clean, 0 stale
git add outputs/ && git commit -m "outputs: initial capture of current green build"

# 4. Rewire site targets to depend on outputs/ + verify-outputs-fresh (§6.1);
#    drop the implicit _notebooks/ dependency.

# 5. Prove the decoupling: in a clean worktree with ONLY .venv-build,
#    make html && make -j4 slides  must succeed.

# 6. PR via gh; squash-merge.
```

After migration, `_notebooks/` remains a gitignored scratch dir; deleting it
costs nothing (a render no longer needs it). Re-executing is only needed to
*refresh* outputs, never to *render* them.

---

## 11. Operational runbook

### 11.0 First checkout on a new machine (document work, CPU-only)

To work on the book/slides on a fresh server or laptop — no GPU, no framework
venvs, no notebook execution — you only need Quarto (`.venv-build`), Node (for
diagrams), and the committed `outputs/` store (LFS):

```bash
git lfs install                       # once per machine, BEFORE cloning if possible
git clone <repo-url> d2l-neu && cd d2l-neu
./bootstrap.sh                        # DOCUMENT mode: uv + git-lfs + node + .venv-build + make lib
#   add --pdf to also install TeX Live + rsvg for `make pdfs`
make html        # or: make -j4 slides            # renders from the store, CPU-only
```

`bootstrap.sh` is idempotent and has three modes: `--doc` (default, render-only),
`--pdf` (adds the TeX stack), `--full` (both, plus a note that notebook execution
needs the GPU framework venvs). It installs `git-lfs` and runs `git lfs pull` so
`outputs/` materializes as real images, not pointer stubs — the one step a
pre-store checkout would miss. If you cloned before installing git-lfs, run
`git lfs install && git lfs pull` to smudge the pointers.

You do **not** need `_notebooks/`, any `.venv-<fw>`, CUDA, or a GPU for any of
the document flows (C/D below). You need those only to *change* a notebook's
outputs (flows A/B), which is a separate machine/role.

### 11.1 The four canonical flows

The split that defines this build: flows **A–B** (notebook code changed) run on
the GPU box and produce `outputs/` commits; flows **C–D** (presentation changed)
are **CPU-only**, touch no `outputs/`, and commit only source/config.

#### A. Build from scratch → commit  *(GPU box, all four venvs — rare, ~110 GPU-min)*

```bash
# ── one-time prereqs (git-lfs already installed; hooks already wired here) ──
git lfs install                                    # idempotent
git add .gitattributes && git commit -m "lfs: track outputs/ image assets"   # BEFORE any asset

# ── execution (GPU, expensive) ──
make lib                                            # d2l/*.py + d2l/_blocks/ from #@save
make notebooks                                      # generate .ipynb for all fw
make run-all-notebooks                              # EXECUTE → _notebooks/ (scratch)

# ── capture / bless → committed store ──
make capture-outputs                                # distill _notebooks/ → outputs/
make audit-outputs                                  # expect: integrity clean, 0 stale

# ── review & commit ──
git status outputs/                                 # *.json text diffs; *.png = LFS pointers
git add outputs/ && git commit -m "outputs: full capture of green build"
git push                                            # gh/HTTPS; LFS blobs ride the same remote

# ── render (could be a different, CPU-only machine) & publish ──
make html && make -j4 slides && make -j4 pdfs
tools/upload_r2.sh
```

#### B. Rebuild just one notebook  *(GPU box, only that framework's venv — cheap)*

```bash
$EDITOR chapter_convolutional-neural-networks/lenet.md          # edit SOURCE .md, never .qmd/.ipynb

make notebooks-pytorch                                          # regen that fw's .ipynb
make -B _notebooks/pytorch/chapter_convolutional-neural-networks/lenet.executed   # execute just this one
make capture-outputs FILES=chapter_convolutional-neural-networks/lenet.md         # bless just this file

git status outputs/pytorch/chapter_convolutional-neural-networks/
#   M lenet.json                    ← text outputs, reviewable diff
#   M lenet/lenet-train-7-0.png     ← LFS pointer (plot changed)
git add outputs/ && git commit -m "outputs: refresh lenet (pytorch)"
```

`make audit-outputs` first if a `#@save` symbol was involved — it names the exact
notebooks that use it (via their `.d` files). Nothing you didn't touch re-runs.

#### C. Edit slides → rebuild  *(any machine, `.venv-build` only — no GPU, no execution)*

```bash
$EDITOR chapter_convolutional-neural-networks/lenet.md   # <!-- slides --> divs, captions, @id refs
$EDITOR _d2l-slides.scss                                 # visual changes here, never per-deck
node diagrams/render.mjs --out img/auto lenet-feature-maps   # if a diagram changed

make -B slides-pytorch SLIDES_FILTER=chapter_convolutional-neural-networks/lenet.md   # one deck
# or: make -j4 slides                                    # all decks, all fw, parallel
tools/audit_slides.py                                    # slide↔cell integrity

git status                                               # only source/styling/diagrams change
#   M chapter_.../lenet.md   M _d2l-slides.scss   ?? img/auto/lenet-feature-maps.svg
git add -A && git commit -m "slides: lenet redesign + feature-map diagram"
```

Code cells + outputs come from the committed `outputs/` store by cell id — zero
notebook execution. `verify-outputs-fresh` (a render prereq) only complains if
you changed *code* without re-blessing, which you didn't.

#### D. Page layout / theme → render  *(any machine, `.venv-build` only — CPU-only)*

```bash
$EDITOR _quarto.yml          # page layout, sidebar, TOC
$EDITOR _d2l-theme.scss      # book colors / typography
$EDITOR _d2l-tabs.html       # tab-sync / sidebar JS

make html                    # full re-render from source + committed outputs/ (runs fix_crossref_numbers.py)

git status                   # only config/theme change; outputs/ untouched
git add -A && git commit -m "layout: <change>"
make html && tools/upload_r2.sh   # publish when happy
```

Numbering and cross-refs recompute here from the full source set (execution-
independent), so a layout change never needs a notebook to run.

### 11.2 Quick reference (scenario → commands)

| You changed… | Do this | Needs GPU/venv? |
|---|---|---|
| Prose only, one file | `make html` (or `make -B slides-<fw> SLIDES_FILTER=<f>.md`) | No |
| Slide block / SCSS / a diagram / layout | `make -j4 slides` (and `node diagrams/render.mjs` if a diagram) | No |
| One notebook's **code** | `make -B _notebooks/<fw>/<ch>/<f>.executed` → `make capture-outputs FILES=<ch>/<f>.md` → review → commit | Yes (that fw) |
| A `#@save` library symbol | `make lib` → `make audit-outputs` (names the blast radius) → re-execute those → `make capture-outputs` | Yes (affected fw) |
| Rebuilt a framework wheel (e.g. MXNet) | bump version → `make audit-outputs` (whole fw shows stale via `framework_version`) → `make run-notebooks-<fw>` → `make capture-outputs` | Yes |
| New chapter / new cells | `make notebooks-<fw>` → execute → `make capture-outputs` → `make audit-outputs` (confirms ids resolve) | Yes |
| Just want to publish current outputs to R2 | `make html && make -j4 slides && make -j4 pdfs` → `tools/upload_r2.sh` | No |

---

## 12. Tie-in: the slides redesign

The slide-quality work in `docs/slides/HANDOFF.md` is exactly the workload this
decoupling unblocks. Its diagrams already follow the same philosophy — authored
SVGs committed to `img/auto/<id>.svg`, reviewable, no build dependency. With the
outputs store in place:

- Captions, `::: {.slide}` blocks, `_d2l-slides.scss`, diagram engine, and the
  `@fig:<id>` inlining can all be iterated and rendered **CPU-only**, with the
  executed code/outputs supplied by the committed store.
- A code edit + re-run + capture refreshes the deck's outputs by cell id with **no
  slide-source change** (HANDOFF §9), exactly as today — except now it doesn't
  require a populated `_notebooks/` on the rendering machine.
- Distinguish the two SVG kinds: **`img/auto/`** = authored diagrams (committed,
  plain git, hand-curated); **`outputs/<fw>/…/*.svg`** = matplotlib output
  figures (LFS, machine-regenerated). Don't cross them.

---

## 13. Failure modes & gotchas

- **Cell id churn breaks everything.** If `add_cell_ids.py` ever renumbered an id,
  every store entry for that cell would orphan and every reference would dangle.
  It is idempotent and must stay so. Never hand-edit ids.
- **Non-deterministic capture churns git.** If `capture_outputs.py` emits
  timestamps or unsorted keys, re-capturing unchanged notebooks produces spurious
  diffs. Keep it a pure function of the notebook (§8).
- **Forgetting LFS before the first asset add** commits image blobs into the
  packfile permanently. Land `.gitattributes` first (§10 step 1).
- **A render that "needs torch"** means someone reintroduced an execution
  dependency. The CPU-only CI render (§9.3) is the tripwire.
- **Stale inline output blocks render by design.** That's not a bug; it's the
  §3.5 guarantee. Re-run + capture the named notebook, or use `render-fresh`.
- **`GIT_LFS_SKIP_SMUDGE=1`** gives a fast text-only checkout (manifests, no
  plot images) for prose/slide-structure work; remember plot images will be
  missing until you `git lfs pull`.
- **Don't edit a source `.md` while a build is running.** `_notebooks/%/.generated`
  depends on `$(SRC_MDS)` (every `chapter_*/*.md` + `index.md`), and notebook
  *generation is per-framework, not per-file*. So touching any source `.md`
  mid-build bumps its mtime, invalidates the `.generated` stamp, and makes an
  in-flight `run-notebooks-<fw>` queue **regenerate that framework's entire
  `.ipynb` set** — which can re-execute notebooks the regen freshened. The edit
  is *not* picked up by the queues that already finished, so it also won't fix
  what you intended. Let the build finish, then do the per-notebook refresh
  (§5.1): `make -B _notebooks/<fw>/<ch>/<f>.executed` → `make capture-outputs
  FILES=<ch>/<f>.md`. (Editing `docs/*.md`, SCSS, or `_quarto.yml` mid-build is
  fine — they are not in `$(SRC_MDS)` and don't feed `.generated`.)

---

## 14. Summary of guarantees

1. **Render is CPU-only** — needs only `.venv-build` + source + `outputs/`.
2. **Partial execution is safe** — re-run/bless any subset; cell-id keying +
   source-derived numbering keep every index correct.
3. **Freshness is randomness-immune** — stale ⟺ *code* drift, never output
   inequality.
4. **No render ships a stale inline output** — hard freshness gate; assets warn.
5. **History stays lean** — text in git (reviewable), churning binary in LFS.

---

## 15. Cross-references

- `docs/architecture.md` — component inventory and the *current* (coupled) flow
  this supersedes.
- `docs/slides/HANDOFF.md` — the slide-quality redesign this unblocks.
- `CLAUDE.md` — repo rules (source-of-truth `.md`, `make` targets, gh/HTTPS).

---

## 16. Implementation status

All of this has **landed** — the table below is now a verification checklist, not
a plan. (Verified 2026-06-06.)

| Piece | State |
|---|---|
| `execute.enabled: false`, cell-id-keyed injection, `_output_fingerprint`, stable ids, `.d` deps | **Landed** — the seam this design builds on. |
| `outputs/` store, manifest schema, split (§2) | **Landed** — `outputs/` committed (≈1400 manifests; `git ls-files outputs/`); images in LFS. |
| `tools/capture_outputs.py`, `tools/audit_outputs.py` (§8) | **Landed** — both present and used by the targets below. |
| `inject_outputs.py` repoint to `outputs/` (§8) | **Landed** — `resolve_id_outputs()` prefers the store, falls back to `_notebooks/`. |
| Make rewiring + new targets (§6) | **Landed** — `capture-outputs`, `audit-outputs`, `verify-outputs-fresh`, `refresh-stale`, `render-fresh` all exist; `_book/index.html` runs `audit_outputs.py --verify-fresh`. |
| Git LFS + `.gitattributes` (§7) | **Landed** — `git-lfs/3.4.1` installed; `.gitattributes` tracks `outputs/**/*.{png,jpg,jpeg,svg,gif}`. Per-clone `git lfs install` still required once. |
| Initial bless + reader flip (§10) | **Done** — `bootstrap.sh` exists; render reads `outputs/`. |
| Host-capability-aware gate (§3.3a) | **Landed** — `audit_outputs.py --verify-fresh` tiers by `nvidia-smi` GPU count. |

The repo no longer behaves as `architecture.md`'s coupled flow describes; this
document is the operative model. If you find a target or tool that contradicts
what's written here, the code is right and this doc has drifted — fix the doc.
