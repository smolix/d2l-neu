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
     make notebook-zips               source .md + outputs/  ─────┴─► _book/notebooks/d2l-<fw>.zip
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

## 0.1 Building on macOS (and any CPU-only host)

The whole RENDER half of the diagram above runs on an Apple-silicon (or Intel)
Mac with **no GPU and no framework venvs** — only `.venv-build` (Quarto). Two
host facts trip people up; handle both up front:

1. **Use GNU Make ≥ 4.3 — i.e. `gmake`, not the system `make`.** macOS ships
   `/usr/bin/make` = **GNU Make 3.81**, which silently mis-parses the build's
   grouped-target rules (`a b &: prereqs`, a 4.3 feature) as a target literally
   named `&` and builds incorrectly. The Makefile now **fails fast** with this
   message rather than mis-building:

   ```
   *** GNU Make >= 4.3 required, but this is 3.81 … install gmake
       (`sudo port install gmake`) and run `gmake` instead of `make`.
   ```

   Install it once (`sudo port install gmake`, MacPorts) and use `gmake`
   everywhere. `bootstrap.sh` checks for it too.

2. **No GPU is fine — the build detects it and degrades.**
   `tools/detect_resources.py` (run `gmake detect`) probes the host with no
   hard dependency on Linux: GPUs via `nvidia-smi` (absent → 0 GPUs, CPU-only),
   cores via `os.cpu_count()`, and RAM via `sysctl`+`vm_stat` (there is no
   `/proc` on macOS). Slot counts derive from that, so the same Makefile sizes
   itself to a laptop or a 4×GPU server unchanged.

**What needs a GPU, and what doesn't:**

| Task | Command | GPU? |
|------|---------|------|
| Render HTML / PDF / slides from `outputs/` | `gmake html` / `pdfs` / `slides` | **No** |
| Render everything-but-execute | `gmake all-quick` | **No** |
| Re-execute a **CPU** notebook + re-capture | `gmake -B _notebooks/<fw>/<ch>/<f>.executed && gmake capture-outputs FILES=…` | **No** |
| Edit/run a CPU notebook in VS Code | (kernel `d2l-<fw>`) | **No** |
| Execute a **CUDA / multi-GPU** notebook | `gmake run-notebooks-<fw>` / `gmake all` | **Yes** |

On a GPU-less host the freshness gate is **capability-aware** (§3.3a): it
renders the whole book and only *defers* (warns about, never fails on) stale
notebooks it lacks the GPUs to re-execute. So you can always `gmake html` here;
you only *re-run* what your hardware supports. First-time setup on a fresh Mac:
`./bootstrap.sh` then `gmake html` (see §11.0).

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

### 3.4a Build-only library sources (`LIB_ONLY_FILES`)

`build_lib.py` scans exactly the files in `CHAPTER_NUMBERING` — not a
directory glob — plus an explicit `LIB_ONLY_FILES` list (defined next to the
file-list construction in `tools/build_lib.py`). A build-only source is a
`chapter_*/…​.md` file that carries `#@save` blocks for the library but is not
part of the rendered book: no `_quarto.yml` entry, no `CHAPTER_NUMBERING`
entry, no outputs, never executed as a notebook. Entries are appended after
the numbered chapters, so a build-only file can never shadow a rendered
chapter's definition under the last-writer-wins collision rule. Current sole
member: `chapter_natural-language-processing-pretraining/legacy-attention-lib.md`,
which quarantines the frozen 2017 `TransformerEncoderBlock` (all four
frameworks) plus the tensorflow/mxnet variants of the attention primitives
whose PyTorch/JAX homes moved to `chapter_attention/` — BERT (ch. 17) builds
on these until the Language-Models part is modernized, at which point the file
is deleted and its entry removed.

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

  Both phases run in **parallel**, not serially:
  - `refresh-stale` **force-re-executes exactly the audit-stale set**, then
    re-captures it in one pass. It asks the audit for the stale
    `(framework, file)` `.executed` stamps (`audit_outputs.py --stale-stamps`)
    and `rm`s them, then feeds the stale *source* set to the unified scheduler
    (`notebook_scheduler.py --files <stale>`) for full GPU/CPU-slot parallel
    dispatch. Removing the stamps is load-bearing — see the trap below. (The
    earlier implementation looped `make -B …executed` one notebook at a time
    with the GPU pool idle, then captured per file.)
  - `render-fresh` then rebuilds slides and PDFs with the same RAM-aware fleet as
    `rebuild-book-artifacts` (`$(MAKE) -j$(RENDER_JOBS)` under `RENDER_V8_ENV`),
    rather than a bare serial `make slides`.

  > **The freshness-disagreement trap (why `refresh-stale` removes stamps).**
  > The audit judges staleness by **code provenance** (fingerprints: framework
  > version, `d2l_lib_fingerprint`, per-cell `code_fingerprint`), comparing the
  > *committed store* against the *current notebook*. The scheduler's own skip
  > check (`notebook_scheduler._stale`) judges by **mtime** (is the `.executed`
  > stamp newer than its `.ipynb` / `.d` deps?). These disagree exactly when a
  > notebook's **source is unchanged but a `#@save` symbol it imports was edited
  > elsewhere**: `make notebooks` regenerates a byte-identical `.ipynb`, the
  > stamp stays newer than it, so the scheduler thinks it is fresh and **skips**
  > it — yet its outputs were produced against the *old* library and are stale.
  > A plain `--files` dispatch skips it, and the capture pass then blesses its
  > pre-edit outputs under the *new* fingerprint: **falsely green**. `rm`-ing the
  > audit's `--stale-stamps` first makes a missing stamp force both `_stale()`
  > and Make to actually re-run the notebook, so only genuinely-fresh outputs are
  > ever captured. (This is the bug that once shipped stale word-level BLEU into
  > the attention chapter after `MTFraEng` moved to a shared byte-level BPE.)

  > **The capture-side guard (the deeper, bless-boundary defense).** `refresh-stale`
  > removing stamps fixes the *scheduler* path, but the trap can still be entered
  > by hand — anyone who runs `make capture-outputs FILES=…` (the fast path just
  > below) on a lib-stale-but-un-re-executed notebook would bless old-lib outputs
  > under the new fingerprint. So the same disagreement is caught a second time at
  > the point where it would do damage: **`capture_outputs.py` refuses to bless
  > outputs it cannot prove were produced under the current lib.**
  >
  > The mechanism is an **execution-provenance sidecar**. Every successful notebook
  > execution (`tools/run_notebooks.write_execution_provenance`, the single
  > chokepoint through `execute_notebook`, so it covers the scheduler, direct
  > `make …executed`, and best-of-N paths) drops
  > `_notebooks/<fw>/<ch>/<stem>.provenance.json` recording the
  > `framework_version` + `d2l_lib_fingerprint` + per-cell `code_fingerprint`
  > **at execution time** (gitignored scratch, wiped by `make clean`). At capture
  > time `capture_outputs.py` recomputes the *current* fingerprints and compares:
  >
  > | sidecar vs current | meaning | capture does |
  > |---|---|---|
  > | present, **agree** | proven executed under the current lib/source | bless |
  > | present, **disagree** | old-lib outputs (or source regenerated w/o re-run) | **REFUSE** (non-zero exit, store untouched) |
  > | absent, store already lib-stale | can't prove a re-run | **WARN**, still bless (back-compat) |
  > | absent, nothing suspicious | pre-sidecar tree, nothing to flag | bless |
  >
  > A genuine re-execution rewrites the sidecar (provenance is a pure function of
  > source+lib, so it is identical across best-of-N attempts), so a matching
  > sidecar is *positive proof* of freshness — the guard is a strict no-op on a
  > freshly-executed tree and has **zero false positives** on the normal capture
  > flow. It refuses only the exact trap: a sidecar written under an old lib with
  > no re-execution since. `--force` overrides (escape hatch; the refusal is
  > normally correct — re-execute instead). `make test-trap`
  > (`tools/test_refresh_stale_trap.py`) is a fast, GPU-free regression test that
  > reproduces the trap on a tmp fixture and asserts all three defenses hold: the
  > audit flags it, stamp-removal forces a re-run, and the guard refuses the stale
  > bless while still accepting a genuine re-capture.

  Because `refresh-stale` now always re-executes the stale set, the *fast*
  recovery for the "I just ran `make all` after editing sources, so the
  `_notebooks/` outputs are already fresh and I only need to bless them" case is a
  **direct capture**, which never re-executes:

  ```
  make capture-outputs FILES="chapter_x/foo.md chapter_y/bar.md"
  ```

  This shortcut is safe precisely because the capture-side guard above backs it:
  if those `_notebooks/` outputs turn out to be lib-stale (not actually
  re-executed under the current lib), the guard refuses rather than silently
  blessing them. Reserve `refresh-stale` / `render-fresh` for "re-execute the
  audit-stale set from scratch to be safe, then render." A no-op `refresh-stale`
  on a fresh store prints `Nothing stale — store is fresh.` and exits without
  touching anything.

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
  `chapters:` lists, so whether *N* groups live under one part or *N* parts has
  **zero** effect on numbering — only the map and flat file order do. A bare
  `part:` title contributes no page. Quarto renders a file-backed part as an
  unnumbered divider but does not allocate it a Pandoc chapter position, so the
  postprocessor deliberately excludes that page from its positional map. The
  Preface uses this form and remains `None` in `CHAPTER_NUMBERING` for source
  preprocessing.

Current tail of the map (2026-07-17 layout): **Attic** = GP **20**, HPO **21**,
RecSys **22**; the **Mathematics for Deep Learning** part is chapters **23–28**
(Linear Algebra 23, Calculus 24, Optimization 25, Probability & Statistical
Learning 26, Information Theory 27, Dynamics 28) — each group's `index.md` is
the chapter, its siblings the `N.k` sections — and the **Tools for Deep
Learning** appendix is chapter **29**. The legacy
`chapter_appendix-mathematics-for-deep-learning/` part has been retired;
inserting or retiring a group means renumbering the map's tail and the matching
`_quarto.yml` order together.

**Dict order is load-bearing for the PDF.** `gen_pdf.py` emits the PDF book's
chapter list from `PDF_CHAPTER_FILES` — i.e. `CHAPTER_NUMBERING` **dict order**
— and `fix_latex.py` pairs the rendered tex's `\chapter` commands against that
same list *positionally*. HTML instead follows `_quarto.yml` order. If the two
orders diverge, the HTML stays correct but the PDF silently misnumbers and
misorders chapters (this happened in the 2026-07-16 restructure: the dict was
renumbered but not reordered, so the PDF placed State Space Models before
Attention and lettered GANs as appendix "T"; fixed 2026-07-17). Keep
`CHAPTER_NUMBERING` dict order identical to `_quarto.yml` chapter order —
there is a matching guard in `gen_pdf.py`. Related constant: `fix_latex.py`
Phase 6 starts `\appendix` lettering at logical chapter **23** (the math part;
the Attic stays numbered so PDF matches HTML), keyed on
`\setcounter{chapter}{22}` — renumbering the map's tail means revisiting that
boundary.

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
| `refresh-stale` | `audit-outputs` → re-execute only the stale set **in parallel** (`notebook_scheduler.py --files`; unforced, so already-executed-but-uncaptured notebooks skip straight to capture) → one `capture-outputs`. (Needs framework venvs.) |
| `render-fresh` | `refresh-stale` then **parallel** `slides`/`pdf` (`-j$(RENDER_JOBS)` under `RENDER_V8_ENV`) + `html`. (Needs framework venvs.) |

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
| `make rebuild-book-artifacts` | `slides` → `html` → `notebook-zips` → `-j4 pdfs` → `check-all-artifacts`. Renders everything from `outputs/`. | No |
| `make notebook-env-locks` | Refresh the committed `pylock.cpu.toml` and `pylock.gpu.toml` files under `notebook_envs/` from their `.in` inputs. MXNet also gets an Apple Silicon CPU lock. Requires package-index/network access; run only when changing reader dependencies. | No |
| `make notebook-zips` | One runnable `d2l-<fw>.zip` per framework → `_book/notebooks/`. Each deterministic archive combines generated notebooks, committed outputs, referenced figures, the matching `d2l` source, `pyproject.toml`, and pinned CPU/GPU uv locks (`tools/build_notebook_zips.py`). Linked from the navbar **Notebooks** menu (`/notebooks/d2l-<fw>.zip`). | No |
| `make hosted-env-locks` | Regenerate `hosted/hosted-lock-*.json` and `hosted/constraints-*.txt` from `uv.lock`. | No |
| `make check-hosted-runtime-contracts` | Validate hosted package versions, imports, critical APIs, and one optimizer/state update in each local framework venv. | No notebooks |
| `make check-hosted-notebooks` | Stage all hosted variants, run runtime contracts and focused tests, and verify deterministic output. | No notebooks |
| `make check-hosted-docker` | Slowly test the generated setup layer and a real optimizer update for PyTorch, TensorFlow, and JAX on CPU and GPU in current official Colab images. | Six small contracts; no notebooks |
| `make check-all-artifacts` | Asserts `_book/index.html`, `_slides/index.html`, `_book/slides/index.html` exist, per-fw PDFs, slide coverage, and per-fw notebook zips. | No |
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

### 6.4.1 Hosted notebook environment contracts

The public Colab/Kaggle notebooks download a revision-pinned `d2l` helper, so
their framework runtime must match that revision too. Package presence is not a
compatibility test: a provider may already contain an older Flax, or a newer
generated Protobuf module with an older runtime. Hosted environments therefore
have a committed, generated contract under `hosted/`:

- `tools/export_hosted_env.py` reads `uv.lock` and emits one JSON profile and
  pip constraints file for PyTorch, TensorFlow, and JAX. Core packages,
  cross-framework ABI packages (`protobuf`, `ml-dtypes`), and page-specific
  optional dependencies all get their reference versions from the lock. Each
  core record also declares its hosted policy. Colab's coherent
  PyTorch/TensorFlow/protobuf stack is preserved and validated behaviorally;
  the JAX/Flax/Optax/Orbax NNX stack and notebook-specific optional packages are
  exact pins. Extras such as `jax[cuda12]`, `tensorflow-probability[tf]`, and
  `syne-tune[gpsearchers]` remain installation requirements while constraints
  use their base distribution names.
- `tools/build_hosted_notebooks.py` consumes those JSON files when creating the
  setup cell. It embeds the environment fingerprint in notebook metadata,
  preserves compatible provider-managed packages, installs exact mismatches in
  one transaction, and validates those exact pins before downloading the
  revision-pinned helper. A missing preserved package falls back to the lock
  version. JAX selects its CPU wheel by default and its CUDA 12 extra only when
  a GPU is available.
- `tools/check_hosted_runtime.py` is the small compatibility canary. It checks
  exact local versions by default. Its provider-compatible mode checks exact
  hosted pins plus external API symbols, exercises the TensorFlow Metadata
  Protobuf path, and performs a real optimizer update on the selected CPU or
  GPU. The JAX contract additionally tests `nnx.view`, shared parameter state,
  BatchNorm, Dropout, and a jitted update.

The normal update sequence is:

```bash
# after changing pyproject.toml / uv.lock
make hosted-env-locks

# fast compatibility tests; no Docker and no notebook execution
make check-hosted-runtime-contracts

# full generation/publication gate
make check-hosted-notebooks
```

`hosted-notebooks` fails when the generated profiles are stale, so a direct
publication cannot silently combine a new lock with old setup cells. Dockerized
Colab-image canaries remain an occasional, explicit check; these local contracts
are the fast first gate.

#### Colab Docker compatibility matrix

Run the slow provider check after changing a framework stack, hosted setup
logic, or when Colab updates its runtime:

```bash
make check-hosted-docker
```

`tools/run_hosted_docker.py` runs six serial cases: PyTorch, TensorFlow, and JAX
on both CPU and GPU. Each case starts from the current official Colab image,
executes the same generated setup cell published in notebooks, imports the
revision-pinned downloaded `d2l` helper, and performs an optimizer update on the
requested device. It also exercises the TensorFlow Metadata generated-Protobuf
path. Full notebook execution is deliberately outside this canary.

The harness defaults to at most 8 CPUs, 24 GiB RAM, 2,048 PIDs, one explicitly
selected GPU, and serial execution. Overrides are rejected when they exceed the
host's CPU count, 50% of currently available RAM, or the smaller of 4,096 PIDs
and 75% of `RLIMIT_NPROC`. Framework thread pools, JAX preallocation, TensorFlow
memory growth, shared memory, and pip's download cache are also bounded.
Each case compares `pip check` before and after setup: conflicts already present
in the provider image are reported as its baseline, while newly introduced
conflicts fail the case. A zero process exit is not sufficient: the completed
case log is also scanned, and CUDA/XLA error diagnostics, Python tracebacks, and
version/API exceptions fail the case.

Useful narrower runs are:

```bash
make check-hosted-docker-cpu
make check-hosted-docker-gpu HOSTED_DOCKER_ARGS='--framework jax --gpu 1'
make check-hosted-docker HOSTED_DOCKER_ARGS='--framework pytorch --device gpu'
```

The official CPU and GPU images are large and may share no layers. On a Docker
volume that cannot hold both, process CPU and GPU serially and explicitly allow
the harness to delete the other cached provider image before pulling:

```bash
make check-hosted-docker HOSTED_DOCKER_ARGS='--prune-other-image'
```

Images are otherwise retained, containers are always removed, and each run
writes per-case setup scripts, output logs, the resolved immutable image digest,
resource limits, timings, and `summary.json` under
`logs/hosted-docker/<timestamp>/`. The default `:latest` tags intentionally
detect provider drift; pass `--cpu-image` or `--gpu-image` to reproduce a
specific image. Linux hosts need Docker plus NVIDIA Container Toolkit for GPU
cases. `--network auto` uses bridge networking when its DNS works and otherwise
falls back to host networking on Linux.

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

### 6.7 Resource-aware scheduling (GPU / CPU / RAM autodetect)

`tools/detect_resources.py` determines host resources at the start of every
notebook build and sizes the slot pools:

- **GPU:** 1 slot per `GPU_MIB_PER_SLOT` (7.5 GiB) of **each** GPU's VRAM,
  reported **per physical GPU** so heterogeneous GPUs just contribute different
  counts — `GPU_SLOTS_PER` e.g. `3,3,3,3` for 4×24 GiB (sum 12), or `3,3,3,4`
  for 3×24 GiB + 1×32 GiB (sum 13). `GPU_VRAM_PER` carries each card's MiB.
- **CPU:** 1 slot per `CPU_PER_LIGHT` (8) cores, **min 1** (a 6-core box still
  gets 1) → `CPU_SLOTS`.

It prints the plan with `--report` at the top of `run-all-notebooks`; the
Makefile passes the knobs to the scheduler via `SCHED_ENV` (all `?=` overridable).

**Unified scheduler (`tools/notebook_scheduler.py`).** Full reference:
**`docs/notebook-scheduler.md`**. `run-all-notebooks` (and `run-notebooks-<fw>`)
put every stale notebook on **one queue** and serve it as resources free up.
It's deliberately a plain queue, not a phased plan:

- **Queue order = framework-grouped.** All of pytorch's notebooks, then all of
  jax's, then tensorflow's, then mxnet's (relpath order within each). This is the
  only ordering; there is **no barrier** between frameworks and **no per-relpath
  mutex**. The framework grouping separates a notebook's framework variants by
  ~one framework's worth of dispatches (~130), so with a ~12-20 slot pool the
  same notebook never runs in two frameworks at once — the cross-framework
  contention (kaggle-cifar10/-dog `data/.../train_valid_test/` reorg race, lib
  rebuild) is avoided by **ordering**, statistically, not by a lock.
- **Served as slots free.** Each dispatch round scans the whole queue and starts
  every item whose required slots are free *right now* (first-fit, skipping —
  not blocking on — items that don't fit yet). GPU and CPU pools are independent,
  so **CPU notebooks just run through as CPU slots free, even across the
  framework boundary while earlier frameworks' GPU notebooks are still going** —
  it's a queue, requests served on resource availability, not a per-framework
  gate.
- **Resource pools.** GPU slots tracked **per physical GPU** (`gpu_free[i]`,
  capacity from `GPU_SLOTS_PER`); CPU slots each pinned to a core group
  (affinity). Per-notebook requirement (`runtime_env.notebook_resource`):
  default **1 GPU or 1 CPU slot**; overridable to **2 slots on one GPU**
  (memory-heavy, `HEAVY_GPU_NOTEBOOKS`), **2×1** (one slot on each of two GPUs,
  data-parallel, `MULTI_GPU_NOTEBOOKS`), or **2×2** (two on each of two GPUs,
  `TWO_GPU_SLOTS_PER`). A 2-GPU item is placed on the two emptiest GPUs that each
  have the slots free; 2-GPU and 1-GPU and CPU items are all **mixed** — no
  separate phase or gating.

Each dispatch shells **`make <stamp>`** with the chosen device(s) in
`D2L_ASSIGNED_CUDA` (`""`=CPU, `"2"`=one GPU, `"0,1"`=two GPUs) +
`D2L_ASSIGNED_CPU_CORES`, reusing the per-framework `EXEC_RULE` env verbatim (no
duplication); `run_one_notebook._run_once` runs on the assignment and **skips its
own flock** (the self-locking path + `serialize_dataset_prep` remain the fallback
for a direct `make _notebooks/<fw>/<x>.executed`). For **jax** the scheduler also
sets `XLA_PYTHON_CLIENT_MEM_FRACTION = slots×7.5 GiB / card-VRAM` (jax
preallocates; others allocate on demand). Two race guards, both because each
notebook is a *separate* make process:

- **No concurrent lib rebuild:** the dispatch passes `-o .preprocess.stamp -o
  d2l/.built -o d2l/<fw>.py -o _notebooks/<fw>/.generated` (make `--old-file`) so
  an inner make can never rebuild the shared d2l library / preprocess / notebook
  set (built once upfront). Unguarded, concurrent inner makes raced to rewrite
  `d2l/*.py` mid-run and corrupted it — one race even wiped the hand-maintained
  preamble (`DATA_URL`/`DATA_HUB`/`download` gone → `partially initialized
  module 'd2l.<fw>'`).
- **Warm `.pyc` before dispatch:** the scheduler pre-compiles `d2l` bytecode once
  so the first import burst finds a valid `d2l/__pycache__/*.pyc` and doesn't race
  on writing it (all venvs are the same CPython → one compile warms the cache).

Net effect: a full `make clean` + re-execute of all 524 notebooks runs in
~100 min (`run-all-notebooks`), 524/524 with 0 failures.

**Gotcha — async ProgressBoard + MXNet (cross-thread CUDA).** The async board's
drawing thread must **not** issue framework GPU ops. MXNet's engine + the custom
CUDA-13 wheel corrupts the CUDA context when `NDArray.asnumpy()` is called from a
foreign thread (manifests as `Xid 154` / `uvm global fatal error` → `CUDA error
999`, **node-reboot required**, and the engine also stops reclaiming memory so a
notebook balloons to fill the card). The MXNet tab of `Module.plot`
(`oo-design.md`) therefore resolves the device→host metric transfer on the
**main** thread and enqueues a host scalar; the board thread does only
matplotlib. PyTorch/JAX/TF keep the deferred thunk (they tolerate cross-thread
host transfers). If you add a framework whose plotting path moves a GPU read off
the main thread, check it survives a long multi-notebook GPU run.

### 6.8 HTML render — why it's a single `quarto render` (not parallel)

`make html` renders the whole book with **one** `quarto render --to html`. That
is deliberate, and we measured why parallelizing it is counter-productive here:

- A **single full-project render amortizes**: quarto scans the crossref DB once,
  then renders pages at **~2.4 s/page**. Cold, all 209 pages: **558 s (~9.3 min)**,
  0 errors, and it regenerates `search.json` itself.
- A **subset / per-file render does NOT amortize** — each invocation re-does the
  whole-project crossref scan: `quarto render <1 file>` ≈ 40 s; `<10 files>`
  ≈ 39 s/file; a `project: render:` subset ≈ 53 s/file. So splitting the book
  across processes makes each page **15-20× more expensive**, and you'd need
  >17 isolated workers just to break even with the single render.
- Concurrent renders sharing one project dir also **flake** under load
  (transient `Error running filter …/main.lua`, or a render falling back to
  standalone mode and writing next to the source) — the community workaround is
  one isolated working copy per worker, which only pays off when *execution*
  dominates. Here the render is pandoc/crossref-bound, which the single render
  already amortizes. (An earlier `tools/render_html_parallel.sh` pool hit a ~56 %
  cold flake rate and was removed.)

Recipe order: `quarto render --to html` → `fix_crossref_numbers.py` (rewrites
logical numbering in the HTML *and* `search.json`) → `add_cfasync.py`. The render
is single-threaded but it's only ~9 min of a ~2.5 h full build — see §6.9 for the
artifact-phase breakdown.

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
| `tools/capture_outputs.py` | **NEW** | `_notebooks/<fw>/<chapter>/<stem>.ipynb` → `outputs/<fw>/<chapter>/<stem>.json` + asset files. Idempotent: identical inputs → byte-identical manifest (stable key order, no timestamps in the hashed payload). Writes assets to stable per-cell paths (overwrite in place); records `bytes`+`sha256` per asset. Computes `code_fingerprint` per §3.1. **Freshness guard (§3.3):** a pre-write pass refuses (non-zero exit, nothing written) to bless any notebook whose execution-provenance sidecar disagrees with the current lib/source; `--force` overrides. |
| `tools/audit_outputs.py` | **NEW** | Freshness (§3.2) + integrity (§4.3). Exit non-zero on integrity failure; `--stale` lists the minimal re-execution set; `--stale-stamps` lists the `.executed` stamps to `rm` to force re-execution (§3.3); `--json` for tooling. |
| `tools/run_notebooks.py` | **CHANGED** | Executes notebooks; on every successful run `write_execution_provenance()` records the source+lib fingerprints the run used into `_notebooks/<fw>/<ch>/<stem>.provenance.json` (gitignored, `make clean`-wiped) — the signal the capture guard reads (§3.3). Single chokepoint (`execute_notebook`) covers scheduler / direct-make / best-of-N paths. |
| `tools/test_refresh_stale_trap.py` | **NEW** | `make test-trap`. Fast, GPU-free e2e regression for the §3.3 trap: builds a tmp fixture, moves a lib fingerprint under an unchanged notebook, and asserts the audit flags it, `refresh-stale`'s stamp-removal forces a re-run, and the capture guard refuses the stale bless (and accepts a genuine re-capture). Never touches the real store. |
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
make html && make -j4 slides && make -j4 pdfs && make notebook-zips
tools/upload_r2.sh          # add --delete to purge bucket objects gone locally
```

Note `make notebook-zips`: `quarto render` (`make html`) wipes `_book/`, which
also removes the downloadable `notebooks/d2l-<fw>.zip` bundles. The upload
script rebuilds missing zips from the store when `_notebooks/` is present, but
building them explicitly keeps the publish step deterministic. `--delete`
diffs a *live bucket listing* against `_book/` (not the upload manifest), so
it removes orphans from any earlier upload or layout change; a run without
`--delete` loses nothing for a later `--delete` run.

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
| Just want to publish current outputs to R2 | `make html && make -j4 slides && make -j4 pdfs && make notebook-zips` → `tools/upload_r2.sh` | No |

---

## 12. Tie-in: the slides redesign

The slide-quality work in `docs/slides-northstar-design.md` is exactly the workload this
decoupling unblocks. Its diagrams already follow the same philosophy — authored
SVGs committed to `img/auto/<id>.svg`, reviewable, no build dependency. With the
outputs store in place:

- Captions, `::: {.slide}` blocks, `_d2l-slides.scss`, diagram engine, and the
  `@fig:<id>` inlining can all be iterated and rendered **CPU-only**, with the
  executed code/outputs supplied by the committed store.
- A code edit + re-run + capture refreshes the deck's outputs by cell id with **no
  slide-source change** (docs/slides-northstar-design.md §9), exactly as today — except now it doesn't
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
- `docs/slides-northstar-design.md` — the slide-quality redesign this unblocks.
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
