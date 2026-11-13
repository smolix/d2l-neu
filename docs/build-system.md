# Build System: Decoupled Notebook Execution & Site Rendering

**What this is:** the reference for how notebooks are executed, how their
outputs are stored and checked, and how the book is rendered and published.
**When to read it:** when a recipe in [`build-guide.md`](build-guide.md) is not
enough, when a build fails in a way the guide does not cover, or before you
change anything in the notebook → output → render path. **New here? Read
[`build-guide.md`](build-guide.md) first:** it has the mental model, a
glossary and step-by-step recipes.

## How to use this document

Look up the section you need; the document is not meant to be read in order.
Passages marked *Rationale* explain why a design was chosen and can be skipped
when you only need to know what to do. Dated incidents ("fixed 2026-10-27")
record why a guard exists; they are kept so the guard is not removed by
mistake. Migration plans and implementation status moved to
[`build-history.md`](build-history.md) on 2026-11-12. Section numbers are
stable: code comments and other docs cite them.

| § | Topic | Read it when |
|---|---|---|
| 0, 0.1 | Mental model; building on macOS / CPU-only hosts | first contact |
| 1 | Why execution and rendering are separate (*Rationale*) | curious |
| 2 | The committed `outputs/` store: layout, text vs LFS, manifest schema | reading or changing the store |
| 3–3.3a | Freshness: fingerprints, staleness, gate severity, host tiers | the gate fails or a notebook looks stale |
| 3.4, 3.4a | `#@save` library dependencies, shards, `LIB_ONLY_FILES` | you changed library code |
| 3.5 | Capture, the freshness-disagreement trap, the capture guard | capture refused, or `refresh-stale` |
| 3.6 | Output hygiene (ANSI, progress bars, line width) | odd characters in outputs |
| 3.7 | Locked notebooks (the owner of all lock details) | expensive notebooks |
| 4 | Numbering, `CHAPTER_NUMBERING`, cell IDs, integrity | adding or moving chapters |
| 5 | Lifecycle overview (recipes live in the guide) | — |
| 6–6.3 | Make targets, the staging override, forcing rebuilds | which target does what |
| 6.4–6.4.2 | Aggregate targets; hosted/Colab environment contracts; execution venvs (`.venv-lm`) | full builds, Colab, venvs |
| 6.5 | Updating the MXNet wheel pin | wheel bumps |
| 6.6–6.6b | PDF pitfalls, part PDFs, page size | PDF builds |
| 6.7 | Resource detection and the scheduler (summary) | notebook execution |
| 6.8 | Why the HTML render is one `quarto render` | render failures, speed |
| 6.9 | Makefile layout, deploy targets, R2 buckets, data uploads (`make upload-data`), timings | publishing, uploading a `DATA_HUB` file, Makefile edits |
| 6.10 | The hidden standard-imports cell | first cell of every notebook |
| 7–9 | Git LFS; tool contracts; checks before merging | tooling work |
| 11 | Runbook: first checkout, quick reference | — |
| 12 | Slides and the store | slide work |
| 13 | Failure modes and gotchas, by area | something broke |
| 14, 15 | Guarantees; cross-references | — |
| 10, 16 | Moved to [`build-history.md`](build-history.md) | — |

The decoupled model has been in use since June 2026: `outputs/` is committed
(676 manifests on 2026-11-12, images in Git LFS), and `make html` renders from
it after `tools/audit_outputs.py --verify-fresh`. If a target or tool
contradicts this document, the code is right; fix the document.

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

Two facts make this work:

1. **Quarto never executes code.** `_quarto.yml` sets `execute.enabled: false`.
   Rendering is a pure text transform; outputs are injected as *static*
   markdown (`::: {.cell-output}` divs, `![](…)` images) by
   `tools/inject_outputs.py`.
2. **Execution and render talk only through stable cell IDs.**
   `inject_outputs.py` looks outputs up by `cell.metadata.id`, not by
   position, and fingerprints outputs to dedup across frameworks
   (`_output_fingerprint`).

`inject_outputs.py` reads the committed store (`index_store_by_id`) whenever a
manifest exists and falls back to `_notebooks/` only for a notebook that was
never captured. `make all` executes into `_notebooks/` but does **not**
capture, so its results reach the book only after `make capture-outputs`
(§6.4).

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

*Rationale.* Before the store existed (until June 2026), every site target
(`html`, `slides`, `pdf`) depended on a locally populated `_notebooks/<fw>/`
tree, so a full site rebuild required the four mutually exclusive framework
venvs, a CUDA stack and about 110 GPU-minutes of execution, even for a CSS
tweak or a new slide diagram. Layout, slide and visual work paid the price of
notebook execution. Separating the two lets each run on its own cadence.

A second motivation is **partial re-execution**. Most edits touch a handful of
notebooks; we re-run and re-publish exactly those, and every index stays
correct under partial execution (§4).

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

*Rationale* for putting **all** text inline (not just "small" text): text is
reviewable and compresses well; the churn problem that motivates LFS is specific
to binary blobs. `capture_outputs.py --max-inline-bytes N` can cap a
pathological text dump (over the cap, the text is head/tail-truncated inline
with an explicit `…[truncated N bytes]…` marker), but the default is `0`, no
cap, and every committed manifest records `"max_inline_bytes": 0`
(2026-11-12). SVG is treated as binary (an asset) even
though it is technically text, because matplotlib SVGs are large and churn like
PNGs; *authored* diagram SVGs are a different thing and live in `img/auto/`
(see §12).

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
    "framework_version": "torch==2.11.0",         // public version only (§3.1)
    "d2l_lib_fingerprint": "sha256:…",            // hash of just the d2l/_blocks/<fw>/*.py
                                                  //   this notebook uses (from its .d file)
    "max_inline_bytes": 0                         // 0 = no cap (the default)
    // locked notebooks only: "execution": {finished, runtime_s} (§3.7)
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
  files this notebook uses, directly or through other library symbols, read
  straight from its `.d` file (`tools/scan_d2l_usage.py` emits one per notebook
  → per-symbol block paths, closed transitively, §3.4). A `#@save` change
  therefore marks exactly the notebooks that reach the changed symbol stale.
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
  The version is read from the venv that executes the notebook: `.venv-<fw>`,
  or `.venv-lm` for `chapter_lm-*` (§6.4.2), falling back to `.venv-pytorch`
  on a host without `.venv-lm` (both pin the same torch, so the key is equal).

A notebook executed in a separate venv (§6.4.2) also records `venv` and
`packages` (today `{"transformers": "transformers==5.22.2"}`) in its
provenance. These are **not** freshness keys: a Transformers bump does not
stale the Language Models part. `audit_outputs.py` prints a warning ("captured
under other venv package versions"), and the author re-runs those notebooks
by hand.

### 3.2 Stale ⟺ code or provenance drift

A notebook is **stale** iff either provenance key differs from the current
environment, **or** any cell's recomputed `code_fingerprint` differs from the
stored one, **or** it has code cells and no manifest at all (never captured,
§3.4). (A provenance mismatch marks the whole notebook; a code-fingerprint
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

Only `make html` runs the gate (`audit_outputs.py --verify-fresh`, the first
step of the `_book/index.html` recipe). The slide and PDF targets have no gate
of their own. Besides stale inline outputs the gate fails on orphaned cell IDs
(§4.3), on locked notebooks with code drift (§3.7) and on images that are still
Git-LFS pointers (`git lfs pull`).

The gate compares the store against the *generated* notebooks in
`_notebooks/`. Where a notebook has not been generated (a fresh clone), it
checks only the no-manifest rule (§3.4) and lists the manifest under "no
generated notebook"; run `make notebooks` (needs only `.venv-build`) to let it
see edited code.

The freshness-disagreement trap and the capture guard, which several tool
comments cite as "§3.3", are described in §3.5.

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

A library rebuild can affect notebooks outside the edited file and outside the
edited framework. The `d2l_lib_fingerprint` in each manifest covers the library
code a notebook can reach (with the exceptions listed below), and
`audit_outputs.py --stale` names the notebooks a library edit invalidates.
Recipe: [build-guide.md §4.4](build-guide.md#44-change-a-save-library-function).

**Shards.** `tools/build_lib.py` writes, next to each `d2l/<fw>.py`, one shard
file `d2l/_blocks/<fw>/<name>.py` per module-level name a `#@save` block
defines: every *unindented* `class`, `def` (also `async def`) or assignment
(`_block_shard_names`). A helper `def` nested inside a function body is not a
library symbol and names no shard. A block that defines two top-level symbols
is written to both shards; if two blocks define the same name, that shard
holds both. `build_lib.py` merges `@d2l.add_to_class` methods into their
class's block first, so `Trainer.py` holds `fit_epoch`, `prepare_batch`,
`clip_gradients` and the ch. 5 device overrides. The merge needs the
decorator as the block's first line. A standalone `#@save` line followed by a
`def` or `class` collects every following line that is indented, blank, or
starts with `def`, `class` or `@` (`save_blocks`), so a decorated
`@d2l.add_to_class(X)  #@save` method later in the same cell becomes part of
that block and is emitted as a module-level function instead of a method.
Put such methods in a cell of their own, or right after another decorated
method (an inline marker stops at the next `@` line). A shard is rewritten only
when its content changes. Four kinds of library code that name no symbol get
shards too:

- `_preamble.py`: the hand-written header of `d2l/<fw>.py` (everything above
  the `#####   WARNING` line: the TF logger level, the MXNet `mnp` alias,
  `nn_Module`, ...) plus every `#@save` block that defines no module-level
  name (the preface imports, `import regex`, the utils.md imports). Every
  notebook of the framework depends on it, since `import d2l` runs it; it has
  no outgoing edges in the closure.
- One shard per `DATA_HUB` key: `DATA_HUB.<key>.py` holds the statement
  `d2l.DATA_HUB['<key>'] = (...)` and nothing else
  (`build_lib.registry_entries`, `KEYED_REGISTRIES`). A block of several
  entries is split by statement; a named block that also registers keys
  (`GPT2_URL = ...` plus the three `gpt2-*` entries) keeps its own shard and
  each entry gets a key shard as well, whose bare `GPT2_URL` reference pulls
  the named shard in. A key that is not a plain file name
  (`[A-Za-z0-9][A-Za-z0-9._-]*`) is sanitised and suffixed with a hash of
  the original (`registry_shard_name`). Other registries
  (`d2l.HF_REVISIONS[...] = ...`) keep one shard per container, named after
  it; so does a `DATA_HUB` entry whose key is computed.
- One shard per config.ini alias (`reshape.py` holds `reshape =
  torch.reshape`), so `d2l.reshape` users depend on it.
- One shard per hand-written module next to the libraries, named after the
  module: `d2l/nnx_resnet.py` → `nnx_resnet.py`, matched by the notebook text
  `from d2l.nnx_resnet import ...`.

The name-less `#@save` blocks in `_preamble.py` (after the header) and the
registry entries come from many chapters, so they are **sorted within
their shard** rather than kept in chapter order. Moving or renumbering a
chapter therefore leaves every shard byte-identical, and no notebook goes
stale. Before this (2026-11), moving the NLP chapters to the Attic reordered
`DATA_HUB.py` and marked 179 notebooks stale with no code change.
`d2l/<fw>.py` itself still emits blocks in chapter order; the shards only
feed the fingerprints. `tools/test_freshness_deps.py` checks the invariant.

**The dependency list is transitive.** `tools/scan_d2l_usage.py` collects the
textual `d2l.<name>` references in a notebook's code cells, keeps those with a
shard, and then closes the set over the library's own references: a shard
references another symbol either as `d2l.X` or as a bare module global
(`_trainer_train_step(...)` inside the JAX `Trainer.fit_epoch`).
`shard_references` finds both with `ast` and `symtable`, so a parameter or
local variable that happens to share a symbol's name does not count. The
closure goes on to a fixed point and is written to the notebook's `.d` file;
`capture_outputs.d2l_lib_fingerprint` hashes exactly those shards (path and
bytes), and Make uses the same list as the `.executed` stamp's prerequisites.
Examples: `d2l.Trainer` pulls in `ProgressBoard`, `Averager`, `as_metrics`,
`HyperParameters`, `num_gpus`, `gpu`, `set_log_ticks`, `use_svg_display` (and
in JAX `_trainer_train_step`, `_trainer_validation_step`); `d2l.resnet18`
pulls in `Residual`; `d2l.hpo_objective_lenet` pulls in `HPOTrainer`,
`Trainer`, `LeNet`, `FashionMNIST` and their closures.

**Registry keys are followed from string literals.** Every one-line string
literal in a notebook's code cells, and in every shard on the closure, is a
candidate key: if `DATA_HUB.<literal>.py` exists, it joins the list
(`scan_d2l_usage.registry_key_refs`). So `d2l.download('airfoil')` depends on
the `airfoil` entry only; `d2l.load_gpt2()` reaches the `gpt2-*` entries
through the literals in its own shard; and `d2l.TokenEmbedding('glove.6b.100d')`
depends on that key although `TokenEmbedding` downloads a variable, because the
notebook names the key. `download` itself reads `DATA_HUB` by a variable and
reaches no entry, so a notebook that downloads a URL
(`d2l.download(d2l.DATA_URL + 'timemachine.txt')`) depends on no key at all.
Registering a new key therefore adds one shard and stales nothing; editing an
entry stales exactly the notebooks that name its key. The rule misses a key
built at run time (`f'glove.6b.{d}d'`): write registered keys out whole.
(Until 2026-11-12 every entry sat in one `DATA_HUB.py` that `download`
reached, so a new key staled every notebook that downloads anything. The
switch itself changed the fingerprint of the 191 notebooks that had depended
on `DATA_HUB.py`: 57 PyTorch, 49 JAX, 27 TensorFlow, 58 MXNet; their code and
entries were unchanged.)

What the fingerprint still does not see:

- `d2l/__init__.py` and package versions other than the framework's own
  (matplotlib, numpy, pandas, ...): an edit or upgrade there flags nothing.
- Data files under `data/` and remote datasets: a changed download changes
  outputs without changing any shard.

What the closure does not follow: a method a notebook calls on an object it got
from elsewhere is covered only through the class's own shard (the notebook
names `d2l.Classifier`, whose shard names its base `d2l.Module`, and so on).
A library function that receives another symbol as an argument (a model
class passed to `Trainer.fit`) depends on it only through the notebook that
passes it, which is the notebook that is flagged anyway.

The `.d` files are regenerated whenever a source `.md` or any shard changes
(`make/notebooks.mk`, `$(DEP_FILES)`); the scan takes about a second.

The dependency list became transitive, and shards got their current names, in
October 2026 (#94); [build-history.md](build-history.md#library-dependency-tracking-before-94)
records what the old scheme missed.

**A notebook without a manifest is stale.** A source that produces a
notebook for a framework (`file_frameworks`, the generator's own rule) and has
an executable `{.python .input}` cell for it, but has no
`outputs/<fw>/…json` (a new section, or one that gained its first code cells),
was never captured. `audit_outputs.py` reports it as HARD-stale ("no
manifest"): `--stale` and `--stale-stamps` list it, and `--verify-fresh` fails
on it under the host-capability tiers of §3.3a. The rule reads the sources,
not `_notebooks/`, so it also works on a fresh clone where no notebook was
generated. (The other staleness rules compare against the generated notebooks
and stay silent where those are absent; the human report lists such
manifests.)

Tests: `tools/test_freshness_deps.py` (run by `make test-trap`) covers the
shard naming, the closure, the preamble/alias/module shards, the per-key
`DATA_HUB` shards (a new key changes no other shard and no fingerprint; an
edited entry stales exactly its users) and the no-manifest rule on tmp
fixtures. `tools/test_build_lib_async.py` (run by `make test-lm`) checks that
`async def` blocks are collected whole, sharded, merged by `add_to_class` and
labelled like `def` blocks.

### 3.4a Build-only library sources (`LIB_ONLY_FILES`)

`build_lib.py` scans exactly the files in `CHAPTER_NUMBERING` — not a
directory glob — plus an explicit `LIB_ONLY_FILES` list (defined next to the
file-list construction in `tools/build_lib.py`). A build-only source is a
`chapter_*/…​.md` file that carries `#@save` blocks for the library but is not
part of the rendered book: no `_quarto.yml` entry, no `CHAPTER_NUMBERING`
entry, no outputs, never executed as a notebook. Entries are appended after
the numbered chapters, so a build-only file can never shadow a rendered
chapter's definition under the last-writer-wins collision rule (except the
files in `LIB_ONLY_EARLY_FILES`, which are prepended because a later chapter
uses one of their symbols at import time). Members:

- `chapter_natural-language-processing-pretraining/legacy-attention-lib.md`
  quarantines the frozen 2017 `TransformerEncoderBlock` (all four
  frameworks) plus the tensorflow/mxnet variants of the attention primitives
  whose PyTorch/JAX homes moved to `chapter_attention/` — BERT builds on these
  until the Language-Models part is modernized, at which point the file is
  deleted and its entry removed.
- `chapter_computational-performance/legacy-multigpu-lib.md` (early) carries
  the tensorflow/mxnet `split_batch`, `resnet18` and `evaluate_accuracy_gpus`
  that the Image Models and Language Models parts still call.
- `chapter_appendix-tools-for-deep-learning/hyperparameters-lib.md` (early)
  declares the empty class `HyperParameters`, the import-time base of
  `Module`, `DataModule`, `Trainer` and `SGD` (`sec_oo-design`). The book
  shows only its method, `save_hyperparameters` (`utils.md`, merged into the
  class by `add_to_class`), not a stub (review finding OO-2, 2026-11).
- `chapter_optimization/legacy-timer-lib.md` carries the tensorflow/mxnet
  `Timer`, which `sgd.md` held in an untagged cell until 2026-11; that cell
  is now `%%tab pytorch, jax` and times with `time.perf_counter`, while the
  legacy copies keep `time.time` so that their shards stay unchanged.
- `chapter_lm-pretrained/eval-suites-lib.md` holds the library code that
  `sec_lm_eval_suites` (18.5) explains in one sentence and does not show: the
  task registry and suite manifests (`EVAL_TASKS`, `EVAL_SUITES`), the row
  adapters, the dataset loader and the result cache. PyTorch only.
- `chapter_lm-pretrained/api-lib.md` holds the API-client scaffolding of
  `sec_lm_api` (18.7): `ChatResult`, the `ChatAPI` class body (the section
  shows its `async def achat`, merged in by `add_to_class`),
  `openrouter_api`, `LocalChatServer`, `FakeChatServer`. PyTorch only.

Both Language Models files are appended (nothing uses them at import time).
Their blocks follow the part's import rule: third-party modules
(`transformers`, `openai`, `httpx`, `asyncio`) are imported inside the
functions, so `import d2l.torch` works without them and the `_preamble` shard
does not change.

Each file repeats the `:label:` of the symbols' original home, so the
"Defined in" docstring suffix, and with it every shard under `d2l/_blocks/`,
stays byte-identical after a move.

### 3.5 "Regenerated when a new render occurs" (the requirement, made concrete)

Inline outputs are a **regenerated snapshot, not a frozen cache:**

- **Never hand-authored.** Every `make capture-outputs` rewrites all inline
  outputs wholesale from the freshly executed notebook.
- **Never rendered stale in HTML.** `make html` first runs
  `audit_outputs.py --verify-fresh`, which hard-fails if a notebook the host
  can re-run has a stale *inline* output (§3.3, §3.3a). The report lists each
  notebook as

  ```
  [HARD] pytorch     chapter_x/foo.md  — 2 cell(s) with changed code
  ...
  FAIL (verify-fresh): 1 notebook(s) have stale INLINE outputs you can re-run on
  this host. Re-run + capture them, or use `make render-fresh`.
  ```

  and the fix is a re-run plus capture
  ([build-guide.md §4.3](build-guide.md#43-change-a-code-cell-and-refresh-its-outputs)).
  Slides and PDFs have no gate of their own. `rebuild-book-artifacts` renders
  the slides before `html`, so with a stale store `all` and `all-quick`
  produce slides and then stop at the gate.

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
    dispatch. Removing the stamps is required — see the trap below.
    `refresh-stale` then captures the stale set **whether or not every
    notebook passed**; check the scheduler's `0 failed` line before
    committing (build-guide.md §4.4).
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

  **Capture takes every generated notebook it is pointed at.**
  `capture_outputs.py` selects each `_notebooks/<fw>/chapter_*/*.ipynb` that
  matches `FILES` (all of them without `FILES`) and does not look at
  `.executed` stamps. A notebook that was generated but never executed in this
  tree has no sidecar, so the guard cannot object (only notebooks
  materialized from the store for a lock are excluded, §3.7), and its manifest is
  overwritten with whatever the `.ipynb` holds (outputs carried over by
  `gen_notebooks.py` for unchanged cells, nothing for changed or new ones).
  So capture all only after a complete green run, and when you re-ran only
  some frameworks of a file, name them:
  `python3 tools/capture_outputs.py --frameworks pytorch chapter_x/foo.md`.

  Because `refresh-stale` always re-executes the stale set, the *fast*
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

### 3.6 Output hygiene: no terminal decoration in the store

Captured output is plain text rendered into **two** media: HTML, and a LaTeX
`verbatim` block in the PDF. Neither can render anything a library wrote for a
*live terminal*. Three classes of noise, in decreasing order of how badly they
break a page:

| class | what it looks like | where it is fixed |
|---|---|---|
| **ANSI** | `[38;2;79;201;177mState[0m`, `[1m` | execution env / `sitecustomize` |
| **control** | an animated progress bar recorded frame-by-frame: `\r`, `\b`, `━━━━` | **the call site** |
| **width** | a line past the page's ~100-column verbatim budget | the call site, or a display-width setting |

**The rule: fix it at execution, never at capture.** `capture_outputs.py` records
what the kernel produced; if a progress bar is in the notebook, the bar is real
output and hiding it at capture would be a lie. So all of this lives upstream of
capture, in two places:

1. **`QUIET_ENV` in `tools/runtime_env.py`** — environment switches applied by
   `setup_framework_env()`, inherited by `jupyter nbconvert` and its kernel.
   `NO_COLOR=1`, `TERM=dumb`, `TTY_COMPATIBLE=0`, `TQDM_DISABLE=1`
   (tqdm's `__init__` is `@envwrap("TQDM_")`-decorated, so this *is* its
   `disable=` parameter), `HF_HUB_DISABLE_PROGRESS_BARS=1`,
   `TF_CPP_MIN_LOG_LEVEL=3` (TF *and* jaxlib's CUDA plugin — level 3 is what
   suppresses XLA's benign `cuda_timer.cc … Delay kernel timed out` ERROR line),
   `MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0` (the multi-kilobyte oneDNN
   "input storage types = [default, default, …]" dump, once per batch),
   `MXNET_CUDNN_AUTOTUNE_DEFAULT=0` (whose notice names itself; note it also
   turns cuDNN algorithm autotuning *off*, so MXNet convolutions may run a few
   percent slower — the only knob here with a performance side effect).

   > **Never set `FORCE_COLOR=0`.** Per force-color.org — and rich's
   > implementation — `FORCE_COLOR` is truthy whenever it is *present*, whatever
   > its value, so `0` forces colour **on**. `apply_output_hygiene()` unsets it.

2. **`tools/nbquiet/sitecustomize.py`** — put on `PYTHONPATH` by
   `apply_output_hygiene()`, so CPython's `site` imports it at startup in every
   descendant interpreter. It handles the residue that no environment variable
   can reach, via post-import hooks:

   * **ipykernel re-forces colour.** `ZMQInteractiveShell.init_environment()`
     unconditionally sets `TERM=xterm-color`, `FORCE_COLOR=1`, `CLICOLOR=1`,
     `CLICOLOR_FORCE=1` in `os.environ` at kernel startup, *after* the build has
     set its own. This is why `NO_COLOR` alone cannot fix notebook output. The
     hook wraps it and re-applies the book's policy.
   * **flax.nnx** picks colour from "stdout is a tty *or* IPython is
     importable" and ignores `NO_COLOR`; its palette is rebound to its own
     `NO_COLOR` palette.
   * **rich** (behind `keras.Model.summary()`) returns `TRUECOLOR`
     unconditionally when it detects a kernel; `_is_jupyter` is forced False so
     it renders the same tables as plain text.
   * **warnings** are reformatted to `Category: message`, wrapped to 80 columns,
     with the absolute path dropped — a `/home/<user>/.venv-jax/lib/…` or
     `/tmp/ipykernel_65430/…` prefix is both too wide and machine-specific.
     Only three third-party deprecation messages are *filtered*; warnings a
     chapter demonstrates deliberately stay.
   * **SGR escapes on the kernel's stdout/stderr are stripped** as a last line
     of defence, for libraries that hardcode colour with no switch at all
     (`torch.serialization` emboldens part of its `weights_only` refusal).
     Cursor-motion and erase sequences are deliberately *not* stripped: those
     are progress bars, and a progress bar is a call-site bug.

   `D2L_NO_QUIET=1` disables the whole mechanism when debugging a library's own
   output.

**Progress bars are a call-site fix.** Keras `fit`/`evaluate`/`predict` default
to `verbose=1` (the animated bar). Pass **`verbose=2`** — one clean line per
epoch, every number preserved — not `verbose=0`, which would delete information
the reader wants.

**The gate.**

```bash
python3 tools/audit_outputs.py --check-noise     # exit 1 on any finding
python3 tools/audit_outputs.py                   # warns, never blocks a render
```

`--check-noise` fails on any stored output carrying an ANSI escape, a
progress-bar control character, or a line wider than `--max-line-width`
(default 100). The default follows from the page: since 2026-11-11 the PDF's
text block is 5.55 in = 399.6pt wide (§6.6b), and `verbatim` is `\small` (9pt
in the 10pt book) in Inconsolata at `Scale=0.9`, i.e. 0.5em ≈ 4.05pt per
character → about 99 columns. (The A4/11pt layout before it gave the same
~100.) Past that, `fvextra`'s `breakanywhere` wraps mid-token; nothing is lost
but it reads badly.

The plain `audit_outputs.py` report (which `make html` runs via
`--verify-fresh`) prints the same findings as a **warning** so regressions
surface on every render without blocking one.

`tools/output_noise_allow.txt` allowlists output whose noise *is* the lesson —
a cell that deliberately demonstrates a library's error message, a docstring the
chapter is about. Format: `<framework> <source> <cell-id> [kind,...]  # why`,
with `*` accepted for framework and cell-id. It is not a parking lot for
output that should have been fixed at execution time.

### 3.7 Locked notebooks

Some notebooks are too expensive to re-execute on every build: paid API calls
(OpenRouter), training or inference on an A100, 8×A100 or several machines.
Once the author is happy with their captured outputs, **locking** the source
file makes every build reuse those outputs until a re-run is explicitly
requested. The first users are the Language Models chapters (PyTorch only).

**Directive.** One line anywhere in the source `.md`, outside code fences, in
the style of the `<!-- split: -->` and `<!-- slides -->` markers:

```
<!-- execution: locked; requires: openrouter -->
<!-- execution: locked; requires: 8xa100; frameworks: pytorch,jax -->
```

- The first field is `locked`. `requires:` is a free-form cost class token
  (`openrouter`, `a100`, `8xa100`, `multinode`, ...) used for reporting; it is
  optional (shown as `-`). `frameworks:` locks only some of the frameworks the
  file has; the default is all of them.
- Whitespace-tolerant; one directive per file. A comment that starts as the
  directive (`<!-- execution:`, any case) but does not parse (an unknown field
  or framework, a second directive, a missing `-->`) is an error in
  `tools/lint_source.py`, in the MANIFEST.mk scan, in the scheduler and in the
  audit. A comment that only resembles it (`<!-- executon: locked -->`,
  `<!-- execution locked -->`) is a lint **warning**, never a build failure;
  unrelated comments such as `<!-- executive summary: ... -->` are ignored.
- `tools/locks.py` is the only parser. The scheduler, `run_one_notebook.py`,
  `audit_outputs.py`, `capture_outputs.py`, `scan_notebook_manifests.py`
  (`LOCKED_<fw>` in `_notebooks/<fw>/MANIFEST.mk`, read by
  `notebook_run_plan.py`), `notebook_run_summary.py`, the linter and
  `make locks` all import it.
- The line never renders: `d2l_preprocess.strip_slide_divs`, which the HTML,
  PDF and notebook generators all call first, drops it, and the slide generator
  reads only slide divs. The lock stays in the source after a re-run.

**Behaviour.**

| Situation | Unlocked notebook | Locked notebook |
|---|---|---|
| `make all`, `make clean && make all`, `run-all-notebooks`, `run-notebooks-<fw>` | executed when its stamp is stale | **never executed**: the stamp is satisfied by *materializing* the notebook from `outputs/`; the scheduler prints `LOCKED (skipped)` and counts it apart from ok/failed |
| direct `make _notebooks/<fw>/<x>.executed` (also the legacy `run-notebooks-{cpu,gpu,multigpu}-<fw>`) | executed | materialized by `run_one_notebook.py` before any GPU slot or venv setup |
| `refresh-stale`, `render-fresh` | re-executed if the audit says stale | never on the audit's `--stale` / `--stale-stamps` list; `--stale` names a locked notebook with code drift on stderr |
| `make rerun-locked FILES=<chapter/file.md> [FRAMEWORKS=pytorch]` | refused ("not locked") | **executed** through the scheduler (exactly the named notebooks) |
| `make capture-outputs` (all) | captured | skipped with a note; store untouched |
| `make capture-outputs FILES=...` | captured | skipped with a note, the file's unlocked frameworks and the other files are captured; exit 2 (nothing written) only when **every** requested pair is skipped |
| `make capture-outputs FILES=... UNLOCK=1` | captured | captured only if its execution sidecar covers **exactly** the current code cells (a real re-run); lib/version drift since the run warns, and the manifest keeps the run's own lib/version provenance; a re-run that misses a changed or appended cell is refused (exit 2) |
| any capture of a notebook **materialized** from the store | skipped (it was locked when materialized; the scheduler re-executes it) | skipped: nothing to capture |
| the lock is removed | — | the next scheduler run re-executes every notebook that was materialized while locked |
| prose, figure, reference, exercise or slide edits | fine (no code fingerprint changes) | fine |
| d2l lib fingerprint or framework version drift | stale (re-run) | **WARNING** only: "locked, outputs from older lib/version" |
| a code cell changed, added or removed | stale (re-run) | **ERROR**: `--verify-fresh` fails on any host; the notebook is not materialized and not stamped |
| `tools/run_hosted_full.py` (hosted sweep) | executed | skipped; `--include-locked` runs it |
| locked but never captured (no manifest) | stale ("no manifest") | **ERROR**: `--verify-fresh` fails; materialization fails with a re-run hint |

**Materialization** (`locks.materialize`) has three outcomes:

1. **Keep an uncaptured re-run.** When the `<stem>.provenance.json` sidecar
   covers exactly the notebook's current code cells with matching
   fingerprints, every code cell ran, and the notebook is not marked (below),
   the notebook is a real execution of the current code (`rerun-locked` done,
   capture not yet). It is kept as it is and stamped, **even if the d2l lib or
   framework version moved since**: lib drift is a warning for a locked
   notebook, so a lib change between the paid re-run and the capture does not
   discard it.
2. **Materialize.** Otherwise, if the stored cells match the current code,
   the stored outputs are injected by cell id (the same reconstruction
   `notebook-zips` uses, `inject_outputs.index_store_by_id`), the code cells
   are numbered 1..N like a top-to-bottom run, the notebook is **marked**
   (`metadata.d2l.materialized_from_store = true`, indexed by an empty
   `<stem>.materialized` file next to it for a cheap scheduler check), any
   sidecar is removed, and the `.executed` stamp is touched. Slides, the
   notebook zips (which strip the mark), the Colab publish (which rewrites
   `metadata.d2l`) and the HTML/PDF render work unchanged.
3. **Refuse.** If the store's cells do not match the current code (a code
   cell changed, appeared or disappeared since capture) or there is no
   manifest, nothing is injected and nothing is stamped (an old stamp is
   removed); the notebook is only marked. The scheduler reports it as
   `FAILED: … (LOCKED: not materialized)` and the gate reports the drift.

The mark means "this notebook's outputs did not come from executing this
code". Capture never blesses a marked notebook. `gen_notebooks.py` carries the
mark through a regeneration (it carries the outputs forward too), and
`run_notebooks.execute_notebook` clears it (and the `.materialized` file)
before a real execution. When the lock is removed, the scheduler queues every
notebook that still has a `.materialized` file, removes its stamp and
executes it, so unlocking can never turn store outputs into a "clean" capture
of different code. In the scheduler path materialization is cheap and needs no
GPU or framework venv (the scheduler does it in-process before dispatch); a
direct `make <stamp>` still has the `EXEC_RULE`'s order-only venv
prerequisites.

**Re-run and unlock workflow.**

```bash
make rerun-locked FILES=chapter_lm-pretrained/api.md FRAMEWORKS=pytorch   # paid run
make capture-outputs FILES=chapter_lm-pretrained/api.md UNLOCK=1          # bless it
make locks                                                                # check: clean
git add outputs/pytorch/chapter_lm-pretrained/ && git commit ...
```

`rerun-locked` regenerates notebooks, then calls `notebook_scheduler.py
--rerun-locked --files ...`: it refuses unlocked files and an empty `FILES`,
removes the named notebooks' stamps and provenance sidecars, and sets
`D2L_RERUN_LOCKED=1` so that `run_one_notebook.py` executes instead of
materializing. The re-run writes a fresh sidecar, which is the scheduler
provenance capture requires. A locked re-run's per-notebook time limit is
`LOCKED_TIMEOUT` (default 21600 s, 6 h), unless `NOTEBOOK_TIMEOUTS`
(`tools/runtime_env.py`) has an entry for it. Capture right after the re-run:
until then the outputs live only in `_notebooks/` (gitignored, wiped by `make
clean`). A failed re-run leaves no stamp and no sidecar, so the next build
restores the stored outputs. If a code cell is edited or appended between the
re-run and the capture, the sidecar no longer covers the code: capture
refuses and the re-run must be repeated. For a locked notebook, capture also
copies the sidecar's
`execution` record (`finished` UTC time and `runtime_s`) into
`provenance.execution`. This is the store's one wall-clock field, written only
on an explicit re-run, so it never churns. `make locks` reports it as the
capture date and runtime (falling back to the manifest's last commit date).

To edit code in a locked file: change it, re-run, capture, in one go. Until the
capture, `make html` fails with the locked-notebook error; for a staging
preview, `make html ALLOW_STALE=1` (§6.2a) renders anyway.

**The gate** (`audit_outputs.py --verify-fresh`, run by `make html`). A locked
notebook is never "stale", so it never enters the host-capability tiers
(§3.3a), which keep working unchanged for unlocked notebooks. Its record
carries `locked_status`:

- `clean`: outputs match the current code, lib and framework version;
- `lib-drift`: the `d2l_lib_fingerprint` or the framework's public version
  moved (an unknown version, no venv on the host, does not count). WARNING;
  the page renders the stored outputs;
- `code-drift`: a cell's prefix-inclusive `code_fingerprint` changed, an
  id'd code cell was added, or a captured one disappeared. ERROR on every host:
  the page would show code beside outputs of other code;
- `no-manifest`: locked but never captured. ERROR;
- `unchecked`: no generated notebook to compare against. Like every other
  notebook, the code check needs `make notebooks`; `make all`/`all-quick`
  generate them first, but a standalone `make html` on a fresh clone does
  not, so it cannot see a code edit in a locked file. (Computing the
  fingerprints from the source alone would need the full `gen_notebooks` +
  `quarto convert` path per file, so it is not done.)

Prose edits never trigger anything because only code cells enter a
fingerprint (§3.1). `make locks` lists every locked (file, framework) with its
`requires`, capture date, runtime and status, and exits non-zero when any is
an ERROR. `refresh-stale` (and so `render-fresh`) now fails when its capture
step fails. Tests: `tools/test_notebook_locks.py` (run by `make test-trap`).

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

Current tail of the map (2026-11-11 layout): **Language Models** = chapters
**18–31** (`chapter_lm-*`), **Image Models** = **32**; the **Mathematics for
Deep Learning** part is chapters **33–38** (Linear Algebra 33, Calculus 34,
Optimization 35, Probability & Statistical Learning 36, Information Theory 37,
Dynamics 38) — each group's `index.md` is the chapter, its siblings the `N.k`
sections — the **Tools for Deep Learning** appendix is chapter **39**, and the
**Attic** closes the book: GP **40**, HPO **41**, RecSys **42**, NLP
Pretraining **43**, NLP Applications **44** (it moved behind Tools on
2026-11-11). The PDFs number these chapters exactly as the HTML does: since
2026-11-11 `fix_latex.py` emits no `\appendix` and letters nothing (the math
part used to print as appendices A–F, Tools as G); each part keeps its
unnumbered part page as the divider. The legacy
`chapter_appendix-mathematics-for-deep-learning/` part has been retired;
inserting or retiring a group means renumbering the map's tail and the matching
`_quarto.yml` order together. Moving a whole part (as the Attic move did)
reorders the emitted `d2l/<fw>.py` library but not its freshness shards
(`d2l/_blocks/`, sorted per shard), so it stales no notebook; check with
`tools/test_freshness_deps.py` and a before/after diff of `d2l/_blocks/`.
Hand-numbered figures that show chapter numbers (`img/front-book-org.svg`,
drawn by `tools/gen_mdl_front_figures.py`) must be regenerated by hand.

**Dict order matters for the PDF.** `gen_pdf.py` emits the PDF book's
chapter list from `PDF_CHAPTER_FILES` — i.e. `CHAPTER_NUMBERING` **dict order**
— and `fix_latex.py` pairs the rendered tex's `\chapter` commands against that
same list *positionally*. HTML instead follows `_quarto.yml` order. If the two
orders diverge, the HTML stays correct but the PDF silently misnumbers and
misorders chapters (this happened in the 2026-07-16 restructure: the dict was
renumbered but not reordered, so the PDF placed State Space Models before
Attention and lettered GANs as appendix "T"; fixed 2026-07-17). Keep
`CHAPTER_NUMBERING` dict order identical to `_quarto.yml` chapter order —
there is a matching guard in `gen_pdf.py`. `fix_latex.py` Phase 2 writes
`\setcounter{chapter}{N-1}` from the map before every chapter, so the PDF's
chapter numbers are the map's; `tools/test_fix_latex.py` checks this on a
synthetic book (no `\appendix`, 33/39/40 for math, Tools and GP).

### 4.2 The one index that matters: cell-id → output

Injection is keyed by **stable cell IDs**. `tools/add_cell_ids.py` is idempotent
and **never renumbers** an id, even on rename or move. The store is keyed by
`(source-file, framework, cell-id)`, so re-running notebook *X* updates only
*X*'s entries and cannot perturb anyone else's. **Cell-id stability is the
invariant everything else depends on** — never break it.

### 4.3 The integrity gate

`tools/audit_outputs.py` checks the store against the generated notebooks, so
partial blessing cannot silently break a reference:

- A manifest entry whose cell id is gone from the notebook is **ORPHANED**: an
  integrity error. `make audit-outputs` exits non-zero and the `make html` gate
  fails. Re-run and capture the notebook (a cell was deleted or a file split).
- An id'd notebook cell with no manifest entry is **MISSING**: a warning
  ("run capture"); the page shows the code without output.
- A source file with code cells for a framework but no manifest at all is
  hard-stale ("no manifest", §3.4).

The audit does not check slide `@<id>` placeholders; `gen_slides.py` resolves
them when it renders a deck, and `tools/audit_slides.py` reviews slide
blocks.

---

## 5. Lifecycle & workflows

Step-by-step recipes live in [`build-guide.md`](build-guide.md) §4; this
section states the model they follow.

### 5.1 Execute → Capture → Commit (on the GPU box)

1. **Execute** into `_notebooks/<fw>/` (scratch): `make run-all-notebooks`,
   `make run-notebooks-<fw>`, one stamp (`rm` it, then
   `make _notebooks/<fw>/<chapter>/<file>.executed`), or
   `python3 tools/notebook_scheduler.py --files ...`. Every successful run
   writes the execution sidecar (§3.5).
2. **Capture** into `outputs/`: `make capture-outputs [FILES=...]`. Capture
   rewrites the manifest of every generated notebook it selects, with no
   mtime or stamp check (§3.5); a re-capture of an unchanged notebook is
   byte-identical, so `git status outputs/` shows only real changes.
3. **Review and commit** `outputs/` (text diffs are readable; images are LFS
   pointers). Push over HTTPS with `gh`; LFS uploads the images there.

A partial execution produces a partial capture and a partial commit; that is
the point of the design.

### 5.2 Render (on any machine, CPU-only)

`make html`, `make slides`, `make pdfs` and `make notebook-zips` read only the
source and `outputs/`, and need only `.venv-build` (plus TeX for PDFs). A
fresh clone renders the whole book. `GIT_LFS_SKIP_SMUDGE=1` gives a text-only
checkout for prose or slide-structure work; plot images are missing until
`git lfs pull`, and the gate then fails on the pointers. Per §3.3a, a CPU or
single-GPU host renders the whole book and only defers the stale notebooks it
lacks the GPUs to re-run.

### 5.3 Partial / selective refresh

`make audit-outputs` is the to-do list; `make refresh-stale` re-executes
exactly the audit-stale set and captures it (§3.5); `make render-fresh` adds
the slide, HTML and PDF renders.

### 5.4 The to-do model

Staleness (what *could* be refreshed) and blessing (the explicit
run-capture-commit act) are separate. You may bless a subset. Un-blessed stale
notebooks keep rendering their committed outputs and stay on the list:
inline-stale ones block `make html` on hosts that can re-run them, asset-stale
ones only warn. The book stays renderable in between.

---

## 6. Make interface

### 6.1 What render targets depend on

| Target | Reads | Gate | Triggers execution? |
|---|---|---|---|
| `html` (`_book/index.html`) | sources, the whole `outputs/` store (every framework tab) | `audit_outputs.py --verify-fresh` | never |
| `pdf-<fw>` | sources, `outputs/<fw>/` | none | never |
| `slides-<fw>` (`_slides/<fw>/.built`) | sources, `outputs/<fw>/` | none | never |
| `notebook-zips` | generated notebooks + `outputs/` | none | never |

`inject_outputs.py {html,pdf,slides}` reads `outputs/<fw>/…` (manifest + asset
files) and falls back to `_notebooks/` only for a notebook without a manifest.
Editing a slide block, SCSS, a diagram or page layout therefore never makes
Make want to execute a notebook. The manifests are Make prerequisites
(`OUTPUT_MANIFESTS` in `make/render.mk`), so a changed manifest re-renders the
HTML and only its own framework's PDF and slides.

### 6.2 Store targets (`make/store.mk`)

| Target | Role |
|---|---|
| `capture-outputs [FILES=…] [UNLOCK=1]` | Distill `_notebooks/` → `outputs/`; extract assets; recompute fingerprints. The bless step. Captures every generated notebook matching `FILES` (§3.5). |
| `audit-outputs` | Report stale notebooks/cells, locks and integrity (orphaned/missing ids). Non-zero exit on integrity failure. |
| `verify-outputs-fresh` | The gate `make html` runs, as a standalone target. |
| `refresh-stale` | `make notebooks`, then delete the audit's stale stamps, execute exactly that set **in parallel** (`notebook_scheduler.py --files`), then one `capture-outputs FILES=<stale set>`, also after failures; exits non-zero if the run or the capture failed. Needs framework venvs and GPUs. |
| `render-fresh` | `refresh-stale`, then slides and PDFs at `-j$(RENDER_JOBS)` under `RENDER_V8_ENV`, `html`, `pdf-parts`. |
| `test-trap` | CPU-only regression tests: `test_refresh_stale_trap.py`, `test_freshness_deps.py`, `test_notebook_locks.py`, `test_notebook_venvs.py`. |
| `test-lm` | CPU unit tests of the Language Models library code: `tools/test_build_lib_async.py` and every `tools/test_lm_*.py`, by `unittest` in `.venv-lm` (§6.4.2). Log: `logs/test-lm-<TS>.log`. |
| `locks` | List locked notebooks (§3.7). |
| `rerun-locked FILES=… [FRAMEWORKS=…]` | Execute exactly the named **locked** notebooks (§3.7). |

Execution itself (`run-notebooks-*`, `run-all-notebooks`, the `.executed`
stamps, `notebooks-*`, `lib`, the `.d` dependency files) lives in
`make/notebooks.mk` and `make/lib.mk`; capture is a consumer of its results.

### 6.2a Staging override: `make html ALLOW_STALE=1`

`make html` runs `audit_outputs.py --verify-fresh` first and stops on stale
outputs (§3.3, §3.3a). For a **staging or preview** render while stale
outputs are known and a re-execution is already planned (2026-11: about 440
notebooks stale after library changes in ch. 2–4, to be re-executed after the
Basics pass), `make html ALLOW_STALE=1` runs the same gate but turns its
failure into a warning and renders anyway. The default (`ALLOW_STALE` unset)
stays strict, and no other target sets it: `all`, `all-quick`, `deploy` and
`rebuild-book-artifacts` still stop on stale outputs. A stale cell renders
its old output beside new code, so an `ALLOW_STALE` build may go to a staging
bucket (`make upload-r2 R2_BUCKET=…`), never to the release site. The PDF and
slide targets have no freshness gate of their own.

### 6.3 Forcing a rebuild: delete the stamp, or `make -B`

To redo one step, delete its stamp and make it again:

```bash
rm -f _notebooks/pytorch/chapter_x/foo.executed && make _notebooks/pytorch/chapter_x/foo.executed
```

`make -B <target>` also works, but GNU Make's `--always-make` remakes **every**
prerequisite it visits, order-only ones included (checked 2026-11-12 with a
two-rule Makefile and with `make -n -B` on a stamp). For a notebook stamp that
means `uv sync` of `.venv-<fw>` and `.venv-build`, the mxnet-pin preflight,
`make lib`, the runtime-deps patch step and the notebook's regeneration before
it executes. That is slower and touches venvs that other worktrees may share.
For `make -B slides-<fw> SLIDES_FILTER=…` it adds one `uv sync` of
`.venv-build`, which is harmless.

Deleting the stamp is also the only way to force a re-run after cells were
merely moved or deleted: `gen_notebooks.py` counts a change only when a cell's
id+source changed, and otherwise re-touches the existing stamp
(notebook-scheduler.md §5, `--force-all`).

The preprocess step is a single pass: `d2l_preprocess.py` with no `--files`
regenerates every `.qmd` for the files in `CHAPTER_NUMBERING`.

### 6.4 Aggregate / full-build targets (the "rebuild everything" path)

The single-notebook recipes in [build-guide.md](build-guide.md) §4 are the
day-to-day path; they avoid full rebuilds. But a *whole-book* rebuild is still real — initial population, the
canonical green run on the multi-GPU box, or a "clean and rebuild the world"
request. Those run through the aggregate targets:

| Target | Expands to | Executes notebooks? |
|---|---|---|
| `make all` | `lib` → `notebooks` → `run-all-notebooks` → `rebuild-book-artifacts`. The full pipeline; ~3 h on the 4×4090 box. | **Yes** (all 4 fw) |
| `make all-quick` | `lib` → `notebooks` → `rebuild-book-artifacts`. Regenerate + render from the **committed** `outputs/`; no execution. | No |
| `make rebuild-book-artifacts` | `slides` (parallel, per-deck-incremental) → **`html`** → **`pdfs`** (the PDFs at `-j$(RENDER_JOBS)`, all at once) → **`pdf-parts`** (§6.6a). html and pdfs are serialized: a concurrent PDF render breaks the html render's startup tree walk (§6.8) → `notebook-zips` → `check-all-artifacts`. Renders everything from `outputs/`. | No |
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
  The "execute → **capture** → render" path is
  [build-guide.md §4.7](build-guide.md#47-full-rebuild-on-the-gpu-box).
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

The Language Models part (`chapter_lm-*`, §6.4.2) is PyTorch-only and needs
the `lm` extra on top of the PyTorch core. The PyTorch profile lists those
packages among its optional packages (`LM_PACKAGES` in
`tools/export_hosted_env.py`, versions taken from the lock's `lm` fork, so
`huggingface-hub` 2.x and not the 1.23.0 the `jax` extra holds) and names them
under `venv_packages.lm`. Every hosted `chapter_lm-*` notebook installs all of
them in its setup cell (`transformers[serving]`, `huggingface-hub`,
`accelerate`, `openai` exact; `scikit-learn` kept if the provider has it),
whatever its own cells import, because the d2l helpers import them lazily;
such a page never becomes a NumPy variant. `check-hosted-runtime-contracts`
checks the PyTorch profile in `.venv-pytorch` and again in `.venv-lm`.

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

### 6.4.2 Execution venvs: one per framework, plus `.venv-lm`

Notebooks execute in `.venv-<fw>` (`uv sync --extra <fw> --extra run`, made by
`make venv-<fw>`; the `uv` conflict set in `pyproject.toml` keeps every
framework extra in its own resolution). The one exception is the Language
Models part: notebooks whose source lies under `chapter_lm-*` execute in
**`.venv-lm`** (extra `lm`: torch/torchvision on the *same* pins as the
`pytorch` extra, plus `transformers[serving]==5.22.2`, `huggingface-hub`,
`accelerate`, `openai==3.49.0`, `safetensors`, `scikit-learn`), so the rest of
the book never installs those packages. `make venv-lm` builds it with the d2l
package installed editable from the checkout it runs in.

Everything else about such a notebook stays `pytorch`: its stamp
(`_notebooks/pytorch/chapter_lm-*/…executed`), its manifest
(`outputs/pytorch/…`), `EXTRA_ENV_pytorch`, `run_one_notebook.py pytorch` and
its freshness key (`torch==2.11.0`, §3.1). Only the interpreter changes. The
routing lives in three places that a test keeps in step
(`tools/test_notebook_venvs.py`, run by `make test-trap`):

| Where | What |
|---|---|
| `tools/runtime_env.py` | `NOTEBOOK_VENVS = {"pytorch": (("chapter_lm-", "lm"),)}`, `notebook_venv(fw, rel)`; `VENV_PROVENANCE_PACKAGES` (what provenance records); `VENV_ENV` (environment defaults, below) |
| `make/notebooks.mk` | `EXEC_RULE(fw, venv, prefix)`; `$(eval $(call EXEC_RULE,pytorch,lm,chapter_lm-))` is a pattern rule with a shorter stem than the generic one, so GNU make picks it for exactly these stamps |
| `make/venvs.mk` | `KERNEL_VENVS` (`make kernels` registers `d2l-lm` too; `make kernels-lm` alone) |

Consumers: `tools/gen_notebooks.py` writes the `d2l-lm` kernelspec into the
generated notebooks (VS Code picks the right interpreter);
`tools/capture_outputs.py` reads the framework version from `.venv-lm` and
records the venv and its Transformers version (§3.1);
`tools/notebook_scheduler.py` is unchanged in what it schedules (a dry run
lists the same items) and tags each `chapter_lm-*` line with `[.venv-lm]`;
hosted setup cells install the `lm` packages (§6.4.1); the PyTorch notebook
ZIP carries `pylock.lm-{cpu,gpu}.toml` beside the PyTorch locks
(`notebook_envs/pytorch/lm-{cpu,gpu}.in`, `make notebook-env-locks`).

**The Hugging Face cache.** The part's notebooks read tens of GB of
checkpoints and datasets through the Hugging Face cache. `VENV_ENV` in
`tools/runtime_env.py` gives every `.venv-lm` notebook
`HF_HOME=~/.cache/hf-lm` unless the launching shell already sets `HF_HOME`
(`apply_notebook_env`, used by every execution path: the scheduler,
`make _notebooks/...executed` and `run_notebooks.py`). On the build box the
default `~/.cache/huggingface` is a symlink to a slow network home, so a
launch that forgot `HF_HOME` would download everything again there. Scripts
outside the notebook runner (`tools/lm/*.py`, `make test-lm`, a VS Code
kernel) do not get the default: export `HF_HOME` yourself. Run the part
online: `d2l.load_qwen3` asks for `model.safetensors.index.json`, which a
single-file checkpoint lacks, and under `HF_HUB_OFFLINE=1`
`hf_hub_download` (huggingface_hub 2.4.0) raises `LocalEntryNotFoundError`
for it even when the Hub's "file absent" marker is cached (checked
2026-11-13 for Qwen3-0.6B and Qwen3-1.7B-Base).

Two version rules come with it:

- **The torch pins move together.** Bumping torch means bumping it in both the
  `pytorch` and the `lm` extra (the test fails otherwise), then re-capturing.
- **An `lm` package bump does not stale anything.** The audit warns that the
  `chapter_lm-*` manifests were captured under another Transformers version;
  re-run those notebooks by hand (`make refresh-stale` will not pick them).

Scheduling needs no venv-specific code: the chapter's notebooks are ordinary
`pytorch` entries in `tools/runtime_env.py` (ch. 18: `hf-models.ipynb` and
`api.ipynb` in `FULL_CARD_NOTEBOOKS`, the other five in `HEAVY_GPU_NOTEBOOKS`
at 2 slots). They take no `NOTEBOOK_TIMEOUTS` entry: a normal build stays
near 15 min per notebook, and a `D2L_RERUN=1` regeneration run must finish
within 30 min (ruling 2026-11-12), inside the default 3,600 s. Their library
code is tested on CPU by `make test-lm` (§6.2).

**Regeneration runs (`D2L_RERUN`).** A `chapter_lm-*` notebook never regenerates
an expensive artifact in a normal build. Its producing cells read
`RERUN = os.environ.get('D2L_RERUN', '0') == '1'` (18.6 calls the flag
`RETRAIN`, same switch): without it they download the registered, versioned
file (`DATA_HUB['lm-…-v1']`) and run a short live check whose agreement they
print, so the captured outputs and the registered artifact agree by
construction. With it they write unversioned files to `../data/lm/rerun/`,
print their sha1 and agreement, and change no source; the author then uploads
the file (§6.9) and edits the registry cell (a new content is a new `-v2` key).
The scheduler copies `os.environ` into each job (`_make_env`), so

```bash
D2L_RERUN=1 python3 tools/notebook_scheduler.py --frameworks pytorch \
    --files chapter_lm-pretrained/scoring.md
```

reaches the kernel. A regeneration run must finish within 30 min measured;
past that, the section applies its declared cuts in their declared order,
chosen from wall clock alone. 18.4 (`scoring.md`) reads its cut level from
`D2L_RERUN_CUT=k` (the first k of its three cuts), and
`tools/oneshot/time_scoring_rerun.py` picks k before the run from the times
of one batch per job, so the cut is fixed before any accuracy is printed.
18.4 also regenerates single artifacts: `D2L_RERUN=arc-generations` (any
comma-separated subset of `bpb`, `gsm8k-samples`, `arc-generations`,
`arc-scores`; `1` means all four) runs only those producing blocks, so a
change to one artifact need not pay for the other three or their cuts
(`time_scoring_rerun.py --only arc-generations` probes the same subset). The
settings each artifact was made with are recorded in it, and the analysis
cells print them (the GSM8K budget, the number of reasoning items), since the
source's constants show the uncut values. The notebook's kernel gets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`NOTEBOOK_ENV`): without
it the 2026-11-13 regeneration runs reserved 20.3-20.7 GB on the card for a
10.8 GiB tensor peak, past the 2-slot budget; its `peak()` prints both
numbers. Measured with the setting (2026-11-13, `D2L_RERUN=arc-generations`,
the job behind that peak): 10.8 GiB allocated, 12.7 GiB reserved, 13.5 GB in
`nvidia-smi`, inside the 2-slot budget; the run reproduced
`lm-arc-generations-v2` bit for bit. A normal run peaks at about 4.9 GB.
Regenerate, upload, then run the notebook once
more without `D2L_RERUN` and capture that run.

**API logs (`D2L_API_LOG_DIR`).** The models-behind-an-API section (18.7,
locked, §3.7) writes the caches and call logs of development, smoke-test and
closed-model calls to `$D2L_API_LOG_DIR` (default
`~/.local/share/d2l/lm-api`, append-only), outside the repository and outside
`data/`, so `make veryclean` cannot delete the record of paid calls and none of
it can be committed by accident. Only the capture-run caches that the notebook
replays are uploaded as `DATA_HUB` artifacts. `OPENROUTER_API_KEY` is read from
the environment and must never appear in a committed file or a captured output.

Adding the `lm` extra moved the shared lock in two places outside it: the `jax`
extra pins `huggingface-hub==1.23.0` (what `.venv-jax` ran; Transformers 5.22
needs ≥ 1.31, and without the pin `uv` moved jax to the 2.x API too), and the
`jax` and `build` resolutions moved `idna` 3.11 → 3.20 and
`typing-extensions` 4.15.0 → 4.16.0 (`jax` also `hf-xet` 1.5.1 → 1.8.0); the
next `make venv-jax` / `make venv-build` picks these up. The `pytorch`,
`tensorflow` and `mxnet` resolutions are unchanged.

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

Two crossref tripwires (2026-07-20). (1) **PDF-excluded pages**: files in
`PDF_EXCLUDED_FILES` (`d2l_preprocess.py` — the website-only `utils.md` /
`d2l.md` API appendices) render in HTML but not the PDF, so `:numref:` refs
to their labels would emit "Unable to resolve crossref" + dead `??` links;
`gen_pdf.py` (`_pdf_excluded_ref_replacements`) now rewrites such refs to
plain "the online appendix …" text automatically. (2) **`subsec_X` /
`sec_X` collisions**: `convert_label_id` maps *both* prefixes to `sec-X`,
so a `subsec_ssm` label collides with `sec_ssm` ("Duplicate identifier" in
pandoc, and one anchor silently wins). Keep subsection label stems distinct
from every section label stem (the `subsec_ssm` case was renamed
`subsec_state-space`).

`make pdfs` (and `make all`) builds one `_pdf/<fw>/` Quarto book per framework
via **`tools/build_one_pdf.sh`**: `quarto render --to latex` (NOT `--to pdf` — we
skip Quarto's own ~2 discarded XeLaTeX passes) → `tools/fix_latex.py`
(hierarchical chapter/section numbering) → **XeLaTeX ×3** →
`_pdf/<fw>/_pdf/Dive-into-Deep-Learning-<fw>.pdf`, then copies it into `_book/pdf/`
for the site. `build_one_pdf.sh` is the extracted form of the old inline
`PDF_RULE` recipe (which had to quadruple-escape `$$$$` inside a `define`/`$(eval)`).
`make pdfs` alone builds the four books one after another; `rebuild-book-artifacts`
and `render-fresh` build them `-j$(RENDER_JOBS)` at a time with the Quarto heap
capped by `RENDER_V8_ENV`, always after `html` has finished (§6.8). The render is
**CPU-only** and reads the committed `outputs/` store like every other artifact
(§6.1). Four pitfalls bit us; each now has a guard, so a clean `make all` builds
all four PDFs (~46–51 MB each) with no manual steps:

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
- **Preamble changes must be deferred with `\AtBeginDocument`, and the staged
  copy must be refreshed.** Two independent traps, both of which made a correct
  preamble line silently do nothing (2026-07-27):
  1. `gen_pdf.py` injects `static/d2l-preamble.tex` via `include-in-header`, so
     it lands *before* Quarto's own preamble block. That block loads
     **`tcolorbox`** (callouts + the `Shaded` code box), which
     `\RequirePackage{verbatim}` — and `verbatim.sty` re-defines the `verbatim`
     environment unconditionally. Any `\DefineVerbatimEnvironment{verbatim}{…}`
     written at header level is therefore overwritten before `\begin{document}`.
     Wide cell output kept running off the page edge (**text lost**, not
     wrapped) for exactly this reason. Wrap such redefinitions in
     `\AtBeginDocument{…}`, which runs after every package.
  2. `gen_pdf.py` used to copy `static/` into `_pdf/<fw>/` only
     `if not static_dst.exists()`, so an edited preamble never reached an
     already-generated tree. It now re-copies every run, and
     `static/d2l-preamble.tex` is a prerequisite of `_pdf/%/.generated`.
- **The book is `\raggedbottom`.** `book.cls` leaves `\flushbottom` in force for
  `twoside`; on code-dense pages that inflates every gap between code block,
  output block and figure to centimetres of white space just to square off the
  bottom margin. The preamble sets `\raggedbottom` (inside the same
  `\AtBeginDocument`), so pages end short instead. Facing pages may end at
  slightly different heights — the accepted trade.
- **Chinese characters need the CJK fallback font (2026-11-12).** Source Serif 4,
  Source Sans 3 and Inconsolata have no CJK glyphs, and XeTeX drops a missing
  glyph with only a `Missing character` line in the log. The Language Models
  part prints Chinese in code cells and outputs, so `static/d2l-preamble.tex`
  gives the CJK code points (U+3000–303F, 3400–4DBF, 4E00–9FFF, F900–FAFF,
  FF00–FFEF) an inter-character class of their own and switches each run of
  them to `static/fonts/NotoSansSC-Regular-subset.otf` (no xeCJK, which some
  TeX Live hosts lack and which changes Latin spacing; Latin output is
  pixel-identical with and without the block). The font is a subset (GB 2312
  hanzi, CJK punctuation, fullwidth forms, plus every CJK character found in
  the chapter sources and `outputs/`), built by
  `tools/make_cjk_font_subset.py` from Noto Sans SC (OFL,
  `static/fonts/NotoSansSC-LICENSE.txt`). A captured output with a rarer
  character still shows as `Missing character: … NotoSansSC` in the PDF log:
  re-run the script (it rescans `outputs/`) and commit the new font.
- **Stale PDFs in `img/outputs/` abort the parallel render.** Quarto's
  `convert_svg` (built-in `main.lua`) has a "skip conversion if the `.pdf`
  already exists" optimization, then `assert(read(pdf) ~= nil)`. A leftover
  corrupt/empty scratch PDF there makes the assert fail (`main.lua:7348`), killing
  the whole PDF render — and `img/outputs/` is gitignored scratch that survives
  across builds. Both `make clean` *and* `rebuild-book-artifacts` (just before
  `make -j4 pdfs`) now `find img/outputs -name '*.pdf' -delete`, so `make all` is
  self-sufficient. For a bare standalone `make pdfs` on a dirty tree, run
  `make clean` first if you suspect stale scratch.
- **Concurrent PDF renders must not share output images (2026-10-23).** A
  chapter without a given framework's tab (the PyTorch+JAX Advanced part in the
  TF/MXNet books) falls back to another framework's outputs, so several of the
  parallel PDF renders injected the *same* `*-pytorch-1.svg`. With one shared
  `img/outputs/`, one session's half-written `.pdf` tripped the other's
  `convert_svg` assert (same `main.lua:7348` failure as a stale PDF, but on a
  freshly cleaned tree). `build_one_pdf.sh` now injects into a per-framework
  `_pdf/<fw>/img/outputs/`, so the four renders share no files; the stale-PDF
  cleanup covers both trees.
- **A full `/tmp` truncates the book silently (2026-08-02).** Quarto stages
  each render session under `$TMPDIR` (default `/tmp`), which on the GPU box
  is a dedicated 9.3 GiB partition. When it filled mid-render, quarto died
  partway through writing the `.tex` (`No space left on device`, Deno stack
  trace), `build_one_pdf.sh` compiled the truncated `.tex` it left behind, and
  plausible-looking half-book PDFs (411 and 1,189 pages) passed the verifier's
  byte-size floor. Three guards now close this chain: (1) `build_one_pdf.sh`
  hard-fails on a nonzero `quarto render --to latex` exit and on a missing
  fresh output PDF (it also deletes the prior run's PDF first, so a stale file
  cannot satisfy the check); (2) `pdf-preflight` requires ≥ 2 GiB free on
  `$TMPDIR` before the render fleet starts (`D2L_SKIP_TMP_CHECK=1` overrides);
  (3) `check_book_artifacts.py` verifies a page-count floor per PDF
  (`MIN_PDF_PAGES = 1500`; healthy books run 1,900–2,300 pages) via `pdfinfo`,
  degrading to the byte check if poppler-utils is absent. The disk pressure
  itself came from outside the build: the Codex CLI leaks a ~378 MB git
  snapshot dir (`/tmp/tmp.XXXXXXXXXX` containing `index` + `objects/`) per
  review session — every repo build path was tested clean of `/tmp` leaks.
  If the preflight trips, clear stale `/tmp/tmp.*` dirs first.

Known and not fixed (observed 2026-07, not re-checked since the Dynamics
chapter was completed): a few `:numref:`fig_mdl-dyn-*`` references rendered as
"?" — the figures, SVGs and labels exist, but Quarto failed to register the
crossref (math-heavy captions are the suspect). This warns but does **not**
stop the PDF build, and it showed in the HTML too.

### 6.6a Per-part PDFs (`make pdf-parts`)

`make pdf-parts` builds one PyTorch PDF per part — **Basics**, **Advanced**,
**Language Models**, **Mathematics for Deep Learning** — as
`_pdf/parts/Dive-into-Deep-Learning-<slug>-pytorch.pdf` (slugs `basics`,
`advanced`, `language-models`, `mathematics-for-deep-learning`) and copies
them to `_book/pdf/parts/`, from where `tools/upload_r2.sh` uploads them with
the rest of `_book/` (no upload change was needed). `rebuild-book-artifacts`
(so `all-quick`, `all`, `deploy`) and `render-fresh` run it right after
`pdfs`. Each PDF holds the part's cover, a table of contents of that part
only, the part's chapters (part page first) and the references that part
cites. Pages are 7 × 10 in, the covers' trim size.

**Design: cut the parts out of the finished full book.** There is no second
render pipeline. `tools/build_part_pdfs.py` reads the full PyTorch book's
`_pdf/pytorch/Dive-into-Deep-Learning.tex` (after `fix_latex.py`) and its
`.aux`, so `pdf-parts` depends on `pdf-pytorch`:

1. **Slicing.** `fix_latex.py` writes a `%D2L-SRC <file.md>` comment line
   before each source file's top-level heading. A part's slice runs from its
   part page's marker to the marker of the first file not in that part
   (membership from the `part:` block in `_quarto.yml`); the script refuses to
   build if the members are not contiguous in the `.tex`.
2. **Numbering.** `fix_latex.py` already puts `\setcounter{chapter}{N-1}`
   before every chapter, and `book.cls` resets section, equation, figure and
   table counters per chapter, so every number inside a slice equals the full
   book's and the HTML's: Advanced opens with chapter 9, the math part with
   chapter 33. The script checks this after the compile: the slice's chapter
   counters must equal `CHAPTER_NUMBERING` (and it must hold no `\appendix`),
   and every label the part defines must have the same number in the part's
   `.aux` as in the full book's (the unnumbered part page's label excepted),
   or the build fails.
3. **References out of the part** (`\ref{X}` / `\hyperref[X]{text}` with `X`
   not defined in the slice) are replaced by plain text before compiling: the
   number from the full book's `.aux`, or the link text. They read as in the
   full book ("Chapter 9", "Section A.3") without a link, and never as `??`.
4. **Bibliography.** The book uses pandoc citeproc, so the bibliography is a
   static `CSLReferences` list of `\bibitem[\citeproctext]{ref-KEY}` entries
   and citations are `\citeproc{ref-KEY}{…}`. The script keeps the entries
   whose keys the slice cites, in the full list's order. (No biblatex
   refsection is involved.)
5. **Front matter and page.** The preamble is the full book's, with only
   the title changed and `pdfpages` added. Page and type are those of every
   PDF (§6.6b): 7 × 10 in at 10pt, set once in `gen_pdf.py`; the script
   refuses a full-book `.tex` built with another page. The body is `\frontmatter`, the cover via `\includepdf`, a blank
   back of the cover, `\tableofcontents` (LaTeX builds it from this
   document, so it lists only the part), `\mainmatter` (arabic page 1 =
   part page), the slice, the filtered references (omitted when the part
   cites nothing, as the skeleton Language Models part does). Links that
   point outside the part, such as the exercise tags' links to the
   preface's "Code" subsection, become their plain link text.
6. **Covers.** The print-ready `d2l-part-<Name>-cover-trim-7x10in.pdf`
   (vector text, 450 ppi art) is read from
   `static/cover/d2l-part-covers-and-spines-pdf.zip` at build time; the
   `d2l-part-<Name>-7x10in-300dpi*.png` files (one is named "… (1).png") are
   the fallback.
7. **Compile and checks.** XeLaTeX ×3 in `_pdf/pytorch/` (where `chapter_*/`,
   `img/` and `static/fonts/` live) with `-jobname part-<slug>`, one part at a
   time. The build fails on a renumbered label, an undefined reference or
   citation in the log, or a `??` in the `pdftotext` output.

Adding a part means adding its slug, part page and cover stem to `PARTS` in
`build_part_pdfs.py` and its slug to `PDF_PARTS` in `make/render.mk`.

### 6.6b One page for every PDF: 7 × 10 in, 10pt

Since 2026-11-11 every PDF — the four full framework books and the part
books — uses the part covers' trim size, 7 × 10 in, at 10pt. The single
source is `gen_pdf.py`: `PDF_FONTSIZE = '10pt'` and `PDF_GEOMETRY`
(`paperwidth=7in, paperheight=10in, inner=0.8in, outer=0.65in, top=0.8in,
bottom=0.85in, twoside`, a 5.55 × 8.35 in text block), written into each
`_pdf/<fw>/_quarto.yml`; `build_part_pdfs.py` inherits the resulting preamble
(`tools/test_gen_pdf.py` pins both). The full books used A4 at 11pt with
1 in margins (6.27 in text block). At 10pt the 5.55 in block holds about the
same line as the old one (5.55 × 11/10 ≈ 6.1 in at 11pt), so wide display
equations, tables and code fit about as before; at 11pt on the 7 × 10 page
the math part alone had 187 overfull boxes over 10pt instead of 25. Code and
cell output wrap via fvextra either way (§6.6). The `pdf:` block in the HTML
`_quarto.yml` is not used by any build (the PDFs render from `gen_pdf.py`'s
generated project) and still says A4.

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
sets `XLA_PYTHON_CLIENT_MEM_FRACTION = slots×7.5 GiB / card-VRAM`, or 0.94 on a
card the job holds entirely (whole-box and whole-card reservations;
`docs/notebook-scheduler.md` §3.3) (jax preallocates; others allocate on demand). Two race guards, both because each
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

Net effect (measured 2026-06, when the book had 524 notebooks; it has 676 on
2026-11-12): a full `make clean` + re-execute ran in about 100 min
(`run-all-notebooks`) with 0 failures.

**Gotcha — no framework GPU ops off the main thread (MXNet).** MXNet's engine +
the custom CUDA-13 wheel corrupts the CUDA context when `NDArray.asnumpy()` is
called from a foreign thread (manifests as `Xid 154` / `uvm global fatal error`
→ `CUDA error 999`, **node-reboot required**, and the engine also stops
reclaiming memory so a notebook balloons to fill the card). This hit the old
asynchronous `ProgressBoard`, whose drawing thread converted metrics. Since the
2026-10 trainer redesign the board has no thread: training and validation
steps return device values, `Averager` sums them on the device, and the
`Trainer` converts the sums with `float()` once per epoch on the calling
(main) thread before drawing. Keep it that way: if a helper ever moves a GPU
read or a plot off the main thread again, check that it survives a long
multi-notebook MXNet GPU run.

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
logical numbering in the HTML *and* `search.json`, and the Part→Chapter
breadcrumb in **both** the mobile and desktop breadcrumb blocks Quarto emits) →
`add_cfasync.py` → `tools/integrate_slides.sh` (folds `_slides/` into
`_book/slides/`). The render is single-threaded but it's only ~9 min of a
~2.5 h full build — see §6.9 for the artifact-phase breakdown.

**`_book/slides/` staging is NOT owned by the html recipe alone** (fixed
2026-07-24). `integrate_slides.sh` runs inside the html recipe, so it only ran
when html itself was out of date — but `_book/slides/` is a product of the
*slides* render, and the two go stale independently. Render html, then render
slides (or add a chapter whose `.built` stamp is stale while `_book/index.html`
is current), and Make correctly skips html while `_book/slides/` silently keeps
the **previous** render's deck set. That is how a `make deploy` shipped a
`_book/` missing the seven new RL decks in every framework — caught only by
`check-all-artifacts` (§6.9), which is exactly the silent-deck-drop class it
exists to catch. `rebuild-book-artifacts` therefore calls
`tools/integrate_slides.sh` **unconditionally** after the html and pdf renders. It is
safe there (slides and html are finished, so nothing races the
`rm -rf`/`rsync`) and idempotent (one extra rsync when html already staged).
Note the gate's expected-deck set is *store-driven*: a PyTorch-only chapter like
RL generates decks for all four frameworks but is only asserted for pytorch, so
a missing deck in a framework with no `outputs/` tree for that chapter will not
trip it.

**html and pdfs never render at the same time (2026-10-27).**
`rebuild-book-artifacts` used to run `make html` concurrently with the PDF fleet.
At startup `quarto render` collects project resource files by globbing the
**whole project tree** (`projectResourceFiles` in `quarto.js` expands
`**/*.quarto_ipynb`, excluding only the output dir `_book/`), so it walks into
`_pdf/<fw>/`, where every running PDF render creates and deletes its
`_pdf/<fw>/_pdf/book-latex/` scratch dir. A directory listed and then removed
before its `stat` aborts the html render with
`ERROR: NotFound: … stat '…/_pdf/jax/_pdf/book-latex'`
(`logs/fixround-allquick.log`). `_quarto.yml` has no option that excludes a
directory from this walk, so the recipe serializes: `html`, then `pdfs` at
`-j$(RENDER_JOBS)`. The same walk covers `_slides/` and `_notebooks/`; do not
run `make html` while slides render or notebooks execute either.

**A failed html render is not marked built.** The html recipe runs its steps
(inject → manifests → `quarto render` → `fix_crossref_numbers.py` →
`add_cfasync.py` → `integrate_slides.sh`) in one `{ …; } | tee` block. Before
2026-10-27 that block returned the exit status of its *last* step, so a failed
quarto render was followed by the post-processing steps, which succeeded, and
`integrate_slides.sh` touched `_book/index.html`; make then treated the failed
render as current, and `rebuild-book-artifacts` reported success. The block now
starts with `set -e` (pipefail carries the failure through `tee`), and on
failure the recipe deletes `_book/index.html` so the next `make html` renders
again. The recipe tests `$?` after the pipeline instead of appending
`|| { rm …; }`: bash ignores `set -e` in every command left of a `||`, including
the commands inside the block.

### 6.9 Makefile layout, extracted scripts & artifact-phase timing

**`make/*.mk` includes (2026-07-22 cleanup).** The top-level `Makefile` is a thin
orchestrator (variables, `.PHONY`, `help`, the aggregate goals `all` / `all-quick`
/ `deploy`, `clean`) that `include`s ordered rule modules. `resources.mk` **must**
come first — it defines the `_detected`/detection helpers every other module
consumes.

| Include | Owns |
|---|---|
| `make/resources.mk` | host resource detection + render/slide-fleet sizing (`_detected`, `RENDER_JOBS`, `SLIDE_WORKERS`, …) |
| `make/venvs.mk` | UV framework venvs and `.venv-lm` (§6.4.2), mxnet preflight, Quarto build venv, `kernels` |
| `make/lib.mk` | `d2l/.built` (rebuilt from `#@save` blocks) |
| `make/figures.mk` | committed illustrative SVGs (incremental; see below) |
| `make/store.mk` | committed outputs store — `capture-outputs`, `audit-outputs`, `verify-outputs-fresh`, `refresh-stale`, `render-fresh`, `test-trap`, `test-lm`, `locks`, `rerun-locked` |
| `make/notebooks.mk` | Notebook generation and execution: `MANIFEST.mk`, `.d` files, `EXEC_RULE`, `run-notebooks-<fw>`, `run-all-notebooks` (which call `notebook_scheduler.py`) |
| `make/hosted.mk` | hosted/Colab notebook staging + downloadable zips |
| `make/render.mk` | `html`, `pdf-<fw>`, `pdfs`, `pdf-parts`, `pdf-preflight`, `slides-<fw>`, `slides` |
| `make/universities.mk` | landing-page adoption grid (independent of the book) |
| `make/deploy.mk` | `publish-colab`, `upload-r2*`, `upload-data`, `deploy` |

**Extracted recipe scripts.** Two large inline-shell recipe bodies now live in
`tools/`: `tools/build_one_pdf.sh <fw>` (the PDF pipeline, §6.6) and
`tools/integrate_slides.sh` (the html slide-integration block). This removed the
quadruple-`$$$$` escaping that `define`/`$(eval)` templates force, and makes both
independently runnable and lintable.

**Deploy targets (`make/deploy.mk`).** `make deploy` = `all-quick` → then
`publish-colab` **and** `upload-r2-delete`, both gated on the build passing but
independent of each other (a Colab failure does not block the R2 upload).
`make publish-colab` is the *verified* Colab path (`tools/publish_colab_notebooks.sh`
— orphan-ness, per-fw counts, inline-`img/`, raw-URL checks); `make upload-r2`
[`-delete`|`-full`|`-dry-run`] wraps `tools/upload_r2.sh` behind an `_r2-preflight`
guard (`_book/` built, `.env` creds, `aws` present).

**Targeting another bucket.** The upload defaults to `staging-d2l` (the public
staging site) but takes any bucket via `R2_BUCKET`, which `deploy.mk` exports so
it reaches the recipe: `make upload-r2-full R2_BUCKET=temporary-d2l`, or
`R2_BUCKET=… tools/upload_r2.sh` directly. Upload state is per-bucket
(`.upload-manifest-<bucket>.txt`), so one bucket's manifest never stands in for
another's and a fresh bucket starts with a full sync.

**Credentials are per-bucket, because R2 tokens are.** An R2 API token is scoped
to its bucket unless created account-wide: the `staging-d2l` keypair is denied on
`temporary-d2l` and vice versa (verified against both live buckets), so one
ambient `AWS_ACCESS_KEY_ID` cannot serve both. `.env` therefore holds one set per
bucket — the staging pair unsuffixed, plus a suffixed set per additional bucket —
and `upload_r2.sh` maps bucket → suffix in one `case` table near the top
(`R2_CRED_SUFFIX` overrides it for a bucket not yet listed). Two traps worth
knowing:

- The suffixed `.env` names are **misnomers kept for compatibility**:
  `R2_ACCOUNT_ID_<SUFFIX>` holds the S3 *Access Key ID* and
  `CLOUDFLARE_TOKEN_<SUFFIX>` the *Secret Access Key* — the pair Cloudflare
  issues for a bucket-scoped token, neither an account ID nor a Cloudflare API
  token. `R2_ENDPOINT_<SUFFIX>` overrides the endpoint built from
  `R2_ACCOUNT_ID`.
- The script **HeadBuckets before doing any work**, so a missing or out-of-scope
  token fails in about a second with that hint. Previously it surfaced only after
  staging slides, rebuilding notebook zips and hashing ~4k files, as a wall of
  per-file `AccessDenied` from the 32 parallel uploaders.

Do not `aws s3 ls` a bucket while an upload runs against it — R2 answers
`ServiceUnavailable: Reduce your concurrent request rate` when a listing competes
with the 32 uploaders. Verify after the run, not during.

**Data artifacts (`make upload-data`).** Files that a `DATA_HUB` entry pins
(the Language Models part's `lm-...-v1` artifacts) live in the R2 bucket
`d2l-data`, served at `https://d2l-data.smola.org/<key>`, not in the site
buckets above and not under the legacy `d2l.DATA_URL` (S3). Registry cells
name the `lm/` folder by the PyTorch-tagged constant `d2l.LM_DATA_URL =
'https://d2l-data.smola.org/lm/'` (`chapter_lm-pretrained/hf-models.md`); its
shard `LM_DATA_URL.py` is reached only through the key shards that use it, so
the constant stales nothing else. Upload with

    make upload-data FILES="data/lm/rerun/lm-x-v1.json" PREFIX=lm/ [DRY_RUN=1]

(`tools/upload_data_r2.py --prefix lm/ FILE... [--dry-run] [--force]`). The
object key is `PREFIX` + the file's base name. For each file the script prints
the key, content type, size, sha1, public URL and a ready `DATA_HUB` line; the
log goes to `logs/upload-data-<TS>.log`. Rules:

- **Versioned names; no overwrite.** A key that already holds the same bytes
  (its ETag equals the file's MD5) is reported as uploaded; a key with other
  bytes is refused (exit 1) unless `FORCE=1`. Readers' caches and the sha1 in
  every `DATA_HUB` entry assume a published file never changes, so publish
  `-v<N+1>` instead of forcing.
- **After the upload** the script re-reads the object's size and ETag; the
  sha1 is also stored as object metadata (`x-amz-meta-sha1`).
- `DRY_RUN=1` uploads nothing; it still checks which keys exist when the
  credentials file is present.
- `tools/upload_data_r2.py --delete KEY...` removes keys; outside a `_test/`
  folder it also needs `--force`. Use `lm/_test/` for trial uploads.

Credentials: `~/.config/d2l/r2-data.env` (or the path in `D2L_R2_DATA_ENV`)
holds one line `CF_API_TOKEN=...`, a Cloudflare API token scoped to the
bucket. The S3 pair is derived as Cloudflare documents for R2: access key id
= the token's `id` from `GET /client/v4/user/tokens/verify`, secret = the
sha256 hex digest of the token. The script hands both to the AWS CLI through
its environment only (with `AWS_CONFIG_FILE`/`AWS_SHARED_CREDENTIALS_FILE`
pointed at `/dev/null`, so no profile interferes) and never prints them;
errors name the HTTP status, not the request. Endpoint
`https://1712830f746ac9278e7db6de736dc430.r2.cloudflarestorage.com`, region
`auto`.

Downloads: `d2l.download` (which uses `requests`) and `curl` fetch these URLs;
Python's `urllib` with its default `Python-urllib/3.x` user agent gets
**403** from Cloudflare (checked 2026-11-12), so any other downloader must
send its own `User-Agent`.

`d2l.download` checks the sha1 only when the file is already in the cache; a
fresh fetch is written without a check. After an upload, confirm a key by
downloading it into an empty folder and calling `d2l.download` a second time:
a second call that prints no `Downloading` line found the cached bytes equal
to the pinned sha1.

Fixed inputs that are not notebook outputs come from authoring scripts that
write to `data/lm/authoring/` and print the sha1: `tools/lm/make_port_fixtures.py`
(18.2: `prompts` on the CPU, `logit-ref` on a GPU, `a1-corrupt DIR` for
assignment A1's hidden state dicts, which are never uploaded),
`tools/oneshot/make_lm_eval_arc_fixture.py` (18.4; runs lm-evaluation-harness
0.4.13 in a throwaway `uv` env outside the repository, never in a book venv),
`tools/oneshot/make_lm_heldout_texts.py` (18.4) and
`tools/oneshot/build_lm_recall_pairs.py` (18.6). Book text is read with
`git show` at a commit pinned in each script (`tools/lm/book_text.py`), so a
rerun reproduces the bytes.

**`make figures` is incremental + manual.** It covers *every* chapter generator —
including the ch6/7/8/9 ones (`gen_bg_*`, `gen_arch_*`, `gen_opt_figures`) the old
`gen_mdl_*_figures.py` glob silently skipped — via a `.figstamps/<name>` per
generator keyed on the generator script + the shared `gen_mdl_figures.py` style.
Only a changed generator re-runs; it is never part of `all`/`all-quick`.

**Logging.** Every substantive recipe tees to `logs/<target>-<TS>.log` (pipefail
is global, so the upstream exit status is preserved). Purely informational guards
(`detect`, `pdf-preflight`, `.preflight.mxnet-pin`) stay terminal-only.

Deleted scripts and archived one-shot tools are listed in
[build-history.md](build-history.md#makefile-cleanup-2026-07-22).

**Artifact-phase timing** (from-scratch, post-notebook-execution, on the 4×4090
box, rendering from the committed store): `clean` ~4 s · **build (`all-quick`)
~14.7 min** [lib ~3 s · notebook regen ~65 s · slides ~60 s cold (0 s when
unchanged — per-deck hash) · `html` ~12.3 min · then `pdfs` ~6 min (all four
concurrently) · notebook-zips ~9 s] · colab publish ~45 s · R2 upload ~12 s.
These timings predate serializing html and pdfs (2026-10-27, §6.8), which adds
the ~6 min PDF render to the total (then ≈ 22 min). HTML is the bottleneck
(single `quarto render`, §6.8); everything else is ≤6 min.

**Incrementality (2026-07-22).** The committed store
(`outputs/<fw>/<chapter>/<stem>.json`) is a tracked prerequisite of the HTML book
(the whole store — HTML carries every framework tab) and of each PDF (that
framework's subtree only); `rebuild-book-artifacts` no longer force-`rm`s HTML +
the PDFs. So an **unchanged `make all-quick` is a ~26 s no-op** (audit-lfs +
slides' zero-deck recheck + notebook-zips + check), and a single notebook-output
change re-renders HTML (whole-book) and rebuilds **only the affected framework's
PDF**. HTML stays whole-book by design (§6.8); slides keep the cheap force-recheck
since `gen_slides.py` is already per-deck incremental.

### 6.10 Hidden standard imports cell (render hook)

Every notebook's first cell is the framework's standard preamble
(`tools/preamble.py:CANONICAL`, taught once at `sec_preamble` on the
installation page; convention in `docs/code-guide.md` §2.1). The rendered
editions hide it; the notebooks keep it:

| Path | Hook | Effect |
|---|---|---|
| HTML (`d2l_preprocess.convert_file`) | `hide_canonical_preambles(blocks)` after `group_code_tabs` | drops each tab whose code equals `CANONICAL[fw]`; drops the tabset when no tab is left (an untagged canonical cell too) |
| PDF (`gen_pdf.convert_file_pdf`) | same function | the framework's canonical cell is not emitted |
| Slides (`gen_slides._resolve_body`) | `preamble.is_canonical(code, fw)` | an `@id` placeholder resolving to the preamble emits nothing |
| Notebooks (`gen_notebooks.py`), hence Colab and the ZIPs | none | the cell stays, with its id |

The match is exact (leading/trailing blank lines aside), so a cell that differs
in any byte still renders: unique preambles stay visible. The hook drops cells
from the `.qmd` instead of marking them `#| include: false`: non-primary tabs are
display-only ```` ```python ```` fences, where Quarto cell options do not apply.
Dropping is safe for the store: `inject_outputs.py` walks the `.qmd`'s cells, so
an output captured for the hidden cell (e.g. an import-time warning) is simply
not placed, and freshness (§3) reads the notebooks, which still hold the cell.
`tools/preamble.py` is a prerequisite of `.preprocess.stamp`,
`_pdf/%/.generated` and `_slides/%/.built`, not of the notebooks: changing the
canonical text re-renders the book, while the notebooks go stale only when
`tools/normalize_imports.py --apply` rewrites their source.

---

## 7. Git & LFS setup (one-time)

`git-lfs` is **installed** on the build host (`git-lfs/3.4.1`). The per-clone
hooks still need to be set up once in each working copy:

```bash
git lfs install                # once per clone (installs the smudge/clean hooks)
```

`.gitattributes`:

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
already do not touch it). `img/outputs/` stays gitignored: it is
a *render-time* scratch copy that `inject_outputs` may still materialize from the
committed store; the committed truth is `outputs/`.

**CI / clone implications:** LFS must be available wherever git operations run
against these blobs. The R2 upload step (`tools/upload_r2.sh`) consumes the
rendered `_book/`, so it is unaffected by LFS. Use `gh`/HTTPS for push (the
yubikey-free path recorded in CLAUDE.md memory); LFS rides the same HTTPS remote.

---

## 8. Tooling contracts

| Script | Contract |
|---|---|
| `tools/capture_outputs.py` | `_notebooks/<fw>/<chapter>/<stem>.ipynb` → `outputs/<fw>/<chapter>/<stem>.json` + asset files. Idempotent: identical inputs → byte-identical manifest (stable key order, no timestamps in the hashed payload). Writes assets to stable per-cell paths (overwrite in place); records `bytes`+`sha256` per asset. Computes `code_fingerprint` per §3.1. **Freshness guard (§3.5):** a pre-write pass refuses (non-zero exit, nothing written) to bless any notebook whose execution-provenance sidecar disagrees with the current lib/source; `--force` overrides. Locked notebooks: §3.7. |
| `tools/audit_outputs.py` | Freshness (§3.2) + integrity (§4.3). Exit non-zero on integrity failure; `--verify-fresh` is the `make html` gate; `--stale` lists the minimal re-execution set; `--stale-stamps` lists the `.executed` stamps to `rm` to force re-execution (§3.5); `--check-noise` (§3.6); `--check-lfs`; `--json` for tooling. |
| `tools/run_notebooks.py` | Executes notebooks; on every successful run `write_execution_provenance()` records the source+lib fingerprints the run used into `_notebooks/<fw>/<ch>/<stem>.provenance.json` (gitignored, `make clean`-wiped) — the signal the capture guard reads (§3.5). Single chokepoint (`execute_notebook`) covers scheduler / direct-make / best-of-N paths. |
| `tools/run_one_notebook.py` | The per-notebook recipe of `EXEC_RULE`: picks the device (scheduler assignment, or its own flock slots for a direct `make`), applies the per-notebook timeout (`NOTEBOOK_TIMEOUTS`, default 3600 s) and best-of-N retries (`BEST_OF_N`), writes `logs/nb-errors/<fw>/<rel>.log` on failure, materializes locked notebooks. |
| `tools/notebook_scheduler.py` | The execution queue; [notebook-scheduler.md](notebook-scheduler.md). |
| `tools/test_refresh_stale_trap.py` | `make test-trap`. Fast, GPU-free e2e regression for the §3.5 trap: builds a tmp fixture, moves a lib fingerprint under an unchanged notebook, and asserts the audit flags it, `refresh-stale`'s stamp-removal forces a re-run, and the capture guard refuses the stale bless (and accepts a genuine re-capture). Never touches the real store. |
| `tools/test_freshness_deps.py` | `make test-trap`: shard naming, the transitive `.d` closure, the preamble/alias/module shards, per-key `DATA_HUB` shards and literal-key following, the no-manifest rule (§3.4). |
| `tools/test_build_lib_async.py` | `make test-lm`: `async def` in `#@save` blocks (standalone and inline markers, `add_to_class`, shards, docstring labels, dedup) (§3.4). |
| `tools/upload_data_r2.py` | `make upload-data`: versioned `DATA_HUB` artifacts → R2 bucket `d2l-data`; never overwrites a key with other bytes without `--force`; prints URL and sha1, never the credentials (§6.9). |
| `tools/scan_d2l_usage.py` (keys) | Also follows string literals that name a registered `DATA_HUB` key to its `DATA_HUB.<key>.py` shard (§3.4). |
| `tools/locks.py`, `tools/test_notebook_locks.py` | Notebook locks (§3.7): the directive parser, materialization from the store and `make locks`; the GPU-free fixture test (run by `make test-trap`). |
| `tools/test_notebook_venvs.py` | `make test-trap`: the separate `.venv-lm` (§6.4.2): pyproject pins, routing in `runtime_env.py` and `make/notebooks.mk`, provenance (`venv`, `packages`, warn-only drift), the `d2l-lm` kernelspec. |
| `tools/inject_outputs.py` | Reads the store through `index_store_by_id()`, which returns the **same shape** as `index_ipynb_by_id()` — `{cell_id: [nbformat-output-dict]}` — by reconstructing nbformat dicts from the manifest (inline text → `stream`/`text/plain` dicts; assets → re-read the file, re-encode to the dict's `data`). Uses `outputs/` when a manifest exists, else falls back to `_notebooks/`. |
| `tools/inject_outputs.py` (stream coalescing, 2026-07-27) | `format_cell_output()` first runs `coalesce_stream_outputs()`, merging runs of **adjacent same-name** `stream` outputs into one. A cell that prints once per loop iteration emits one stream message per flush, so Jupyter stores N stream outputs (each already `\n`-terminated); rendering each as its own fenced block put a paragraph break between consecutive printed lines — a blank line in HTML, a stretched inter-block gap in the PDF. The store is *faithful* (the split is in the `.ipynb`); the fix is render-side only, so no re-capture is needed. Non-stream outputs and stderr↔stdout transitions still separate blocks. Side effect: the `MAX_TEXT_LINES` cap applies to the merged block — 10 output groups book-wide elide to head-20 / `...` / tail-20 in the PDF. Also: `is_wide_table()` tags column-aligned output wider than the column with `.d2l-output-wide` (**html mode only** — 18 blocks: profiler tables, Syne Tune trial status, …). `_d2l-style.css` wraps output by default and scrolls those instead, since wrapping destroys the alignment that carries their meaning. The PDF has no scroll, so it wraps them. |
| `tools/add_cell_ids.py` | The authority for stable ids — the invariant capture/audit rely on. Idempotent; `--check` reports without writing. |
| `tools/scan_notebook_manifests.py` | Writes `_notebooks/<fw>/MANIFEST.mk`: which (framework, source) pairs produce a notebook, split CPU / GPU / multi-GPU, plus `LOCKED_<fw>`. Capture reuses its file enumeration. |
| `tools/scan_d2l_usage.py` | Emits the per-notebook `.d` files capture reads for `d2l_lib_fingerprint`. |

Determinism requirement for `capture_outputs.py`: the manifest must be a pure
function of the executed notebook + repo state (sorted keys, fixed float
formatting, **no wall-clock or HEAD-commit fields**). Otherwise re-capturing an
unchanged notebook would churn git for no reason. The integration test for this
is "capture twice → `git diff` empty."

---

## 9. Checks before merging

There is no CI (2026-11-12); run these by hand before merging a change to the
build or to many sources:

1. `make test-trap` — CPU-only regression tests (freshness trap, library
   dependencies, locks).
2. `python3 tools/lint_source.py --corpus` — source lint plus cross-file label
   uniqueness.
3. `make audit-outputs` — integrity must pass (no orphaned ids); no hard-stale
   notebooks on `main`.
4. `make html` (and `make slides`) in a tree with only `.venv-build` — proves
   rendering still needs no framework venv. A render that "needs torch" means
   an execution dependency crept back in.

---

## 10. Migration (moved)

The one-time migration from the execution-coupled build to the store (June
2026) is recorded in [build-history.md](build-history.md#migration-to-the-outputs-store-2026-06).

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

The flows that used to be spelled out here are recipes in
[build-guide.md](build-guide.md) §4 since 2026-11-12:

| Flow | Machine | Recipe |
|---|---|---|
| A. Build from scratch → capture → commit | GPU box, all four venvs | [§4.7](build-guide.md#47-full-rebuild-on-the-gpu-box) |
| B. Rebuild one notebook | that framework's venv (GPU if it is a GPU notebook) | [§4.3](build-guide.md#43-change-a-code-cell-and-refresh-its-outputs) |
| C. Edit slides → render | any, `.venv-build` only | [§4.2](build-guide.md#42-change-prose-only), §12 |
| D. Page layout / theme → render | any, `.venv-build` only | [§4.1](build-guide.md#41-render-the-book-on-a-laptop) |

Flows A and B change `outputs/`; C and D never do. Layout files: `_quarto.yml`
(layout, sidebar, TOC), `_d2l-theme.scss` (colours, typography),
`_d2l-tabs.html` (tab sync, sidebar JS), `_d2l-slides.scss` (all slide
styling; never per deck). Slide diagrams: `node diagrams/render.mjs --out
img/auto <name>`.

### 11.2 Quick reference (scenario → commands)

| You changed… | Do this | Needs GPU/venv? |
|---|---|---|
| Prose only, one file | `make html` (or `make slides-<fw> SLIDES_FILTER=<f>.md`) | No |
| Slide block / SCSS / a diagram / layout | `make -j4 slides` or `make html` (and `node diagrams/render.mjs` if a diagram) | No |
| One notebook's **code** | `rm` its stamp → `make _notebooks/<fw>/<ch>/<f>.executed` → capture → review → commit ([guide §4.3](build-guide.md#43-change-a-code-cell-and-refresh-its-outputs)) | that fw |
| A `#@save` library symbol | `make lib notebooks` → `make audit-outputs` (names the blast radius) → `make refresh-stale` ([guide §4.4](build-guide.md#44-change-a-save-library-function)) | affected fw |
| A framework version (public version, e.g. torch 2.11 → 2.12) | bump the pin in `pyproject.toml`, `uv lock`, `make venv-<fw>` → `make audit-outputs` (the whole framework is stale) → `make run-notebooks-<fw>` → `make capture-outputs` | Yes |
| A wheel rebuild with the same public version (e.g. a new MXNet `+cu13…` build) | nothing is stale (§6.4); re-run + capture only if you want its outputs | optional |
| New chapter / new cells | [guide §4.5](build-guide.md#45-add-split-or-move-a-section) | Yes |
| Publish current outputs | `make deploy`, or `make all-quick` then `make upload-r2` ([guide §4.11](build-guide.md#411-publish)) | No |

---

## 12. Tie-in: the slides redesign

*Rationale.* The slide-quality work in `docs/slides-northstar-design.md` is the
workload this decoupling was built for. Its diagrams follow the same idea —
authored SVGs committed to `img/auto/<id>.svg`, reviewable, no build
dependency. With the outputs store:

- Captions, `::: {.slide}` blocks, `_d2l-slides.scss`, diagram engine, and the
  `@fig:<id>` inlining can all be iterated and rendered **CPU-only**, with the
  executed code/outputs supplied by the committed store.
- A code edit + re-run + capture refreshes the deck's outputs by cell id with **no
  slide-source change** (docs/slides-northstar-design.md §9), and the rendering
  machine needs no populated `_notebooks/`.
- Distinguish the two SVG kinds: **`img/auto/`** = authored diagrams (committed,
  plain git, hand-curated); **`outputs/<fw>/…/*.svg`** = matplotlib output
  figures (LFS, machine-regenerated). Don't cross them.
- **Slide-only figures survive via an explicit copy step.** Quarto's HTML render
  populates `_book/img/` only with images referenced from *page* bodies. A
  figure referenced solely from a `<!-- slides -->` section (legal — e.g.
  `lstm.md`'s deep-RNN/bi-RNN schematics after the ch. 12 slimming) would be
  silently missing from `_book/img/` and get purged from the bucket by
  `upload_r2.sh --delete`. The `make html` slides-integration block therefore
  scans the staged decks' `../../../img/` refs and copies any missing asset
  from `img/` into `_book/img/` (added 2026-07-19 after exactly this breakage).

---

## 13. Failure modes & gotchas

Grouped by where they bite. PDF-specific pitfalls are in §6.6, HTML-render
ones in §6.8, scheduler ones in [notebook-scheduler.md](notebook-scheduler.md),
MXNet runtime ones in `docs/mxnet-runtime-diagnostics.md`.

| Symptom | Bullet |
|---|---|
| `RuntimeError: can't start new thread`, `Resource temporarily unavailable` | 13.1 thread limit |
| A run never finishes after the CPU notebooks are done | 13.1 unplaceable notebooks |
| Notebooks re-execute that you did not touch | 13.1 editing sources mid-build |
| A manifest lost its outputs after capture | 13.2 capture selects every generated notebook |
| `REFUSING to capture` | §3.5 |
| `make html` fails with stale outputs | §3.3, [build-guide.md §4.12](build-guide.md#412-diagnose-a-stale-or-failing-freshness-gate) |
| `NotFound: … stat '<chapter>/<page>_files'` | 13.3 orphaned Quarto processes |
| PDFs or zips vanished from the site | 13.4 standalone `make html` |
| An updated image still shows the old version | 13.4 CDN cache |

### 13.1 Execution

- **A full-concurrency run can exhaust the per-user thread limit.** The
  default shell gets `ulimit -u` soft 4096 / hard 8192, and every
  JAX/TensorFlow/MXNet job keeps its own thread pools. At ~20-way concurrency
  (`make all`, `run-all-notebooks`) the scheduler can die with
  `RuntimeError: can't start new thread` and make with `/bin/bash: Resource
  temporarily unavailable`; the run summary then tallies the never-started
  notebooks as FAILED. Run `ulimit -u 8000` in the same shell before the build
  (8000, not the 8192 cap, leaves headroom for other work on the account).
  The scheduler pins GPU jobs to a core group as well as CPU jobs, because
  XLA/oneDNN size their pools from the *visible* core count, not
  `OMP_NUM_THREADS`; this cut a JAX notebook from ~350 threads to a few dozen,
  but the limit still applies. Diagnose by counting threads, not processes:
  `ps -eLf -u $USER | wc -l` against the limit. Recovery is cheap: re-run
  without `make clean` and only the un-run tail executes.
- **The scheduler waits for ever on a notebook the host cannot place.** It
  skips an item whose slots are not free and retries when a job finishes; it
  never drops an item that can *never* fit. On a host with no GPU (empty
  `GPU_SLOTS_PER`) a GPU notebook, and on a one-GPU host a 2-GPU notebook,
  therefore leaves `run-all-notebooks`, `run-notebooks-<fw>` and
  `refresh-stale` waiting once everything else is done (from reading
  `Scheduler.run` and `_reserve`, 2026-11-12; not reproduced). On such hosts
  re-run single notebooks by stamp (build-guide.md §4.3).
- **Don't edit a source `.md` while a build is running.** `_notebooks/%/.generated`
  depends on `$(SRC_MDS)` (every `chapter_*/*.md` + `index.md`), and notebook
  *generation is per-framework, not per-file*. So touching any source `.md`
  mid-build bumps its mtime, invalidates the `.generated` stamp, and makes an
  in-flight `run-notebooks-<fw>` queue **regenerate that framework's entire
  `.ipynb` set** — which can re-execute notebooks the regen freshened. The edit
  is *not* picked up by the queues that already finished, so it also won't fix
  what you intended. Let the build finish, then refresh the notebook
  (build-guide.md §4.3). (Editing `docs/*.md`, SCSS, or `_quarto.yml`
  mid-build is fine — they are not in `$(SRC_MDS)` and don't feed
  `.generated`.)
- **`make -B` on a stamp re-syncs venvs.** See §6.3; delete the stamp instead.
- **Never run framework GPU operations off the main thread (MXNet).** §6.7.

### 13.2 Capture and the store

- **Cell id churn breaks everything.** If `add_cell_ids.py` ever renumbered an id,
  every store entry for that cell would orphan and every reference would dangle.
  It is idempotent and must stay so. Never hand-edit ids.
- **Capture selects every generated notebook.** `make capture-outputs`
  without `FILES` captures every `.ipynb` under `_notebooks/`, and with
  `FILES` every framework of the named files, executed or not (§3.5). Capture
  everything only after a complete green run; otherwise name files and
  frameworks, and read `git diff --stat outputs/` before committing.
- **Non-deterministic capture churns git.** If `capture_outputs.py` emits
  timestamps or unsorted keys, re-capturing unchanged notebooks produces spurious
  diffs. Keep it a pure function of the notebook (§8).
- **Forgetting LFS before the first asset add** commits image blobs into the
  packfile permanently. Keep `.gitattributes` (§7) in place before adding
  any new asset type.
- **Stale inline output blocks render by design.** That's not a bug; it's the
  §3.5 guarantee. Re-run + capture the named notebook, or use `render-fresh`.
- **`GIT_LFS_SKIP_SMUDGE=1`** gives a fast text-only checkout (manifests, no
  plot images) for prose/slide-structure work; remember plot images will be
  missing until you `git lfs pull`, and `make html`'s gate fails on the
  pointers.

### 13.3 Rendering

- **A render that "needs torch"** means someone reintroduced an execution
  dependency. Rendering in a tree with only `.venv-build` (§9 check 4) is the
  tripwire.
- **Stopping a render must stop Quarto's engine processes too.** Killing `make`,
  the `quarto` wrapper or the shell around it leaves the `deno` (and `pandoc`)
  children running. An orphaned HTML render keeps working in the same project
  and deletes the temporary `<page>_files/` folders of the *next* render, which
  then fails on a different page each time with `NotFound: … stat
  '<chapter>/<page>_files'`. After stopping a build, check
  `ps -eo pid,args | grep -E 'quarto|deno|pandoc'` and kill what is left before
  starting another render.
- **Never run `make html` next to a PDF render, a slide render or a notebook
  run in the same tree** (§6.8).

### 13.4 Publishing

- **A standalone `make html` empties `_book/` — including the PDFs and notebook
  ZIPs.** `make all-quick` puts `_book/pdf/*.pdf` and `_book/notebooks/*.zip` there
  (the PDF rule copies each PDF in; `notebook-zips` writes the ZIPs), but a later
  `make html` re-renders into a fresh `_book/` without them, and
  `check-all-artifacts` checks `_pdf/<fw>/` rather than `_book/pdf/`. An
  `upload-r2-delete` from that tree then **deletes the PDFs and ZIPs from the
  bucket** as orphans. Before uploading after a standalone `make html`, run
  `make notebook-zips` and copy the already-built PDFs back
  (`cp _pdf/<fw>/_pdf/Dive-into-Deep-Learning-<fw>.pdf _book/pdf/` for each `<fw>`,
  and `make pdf-parts` for `_book/pdf/parts/`),
  then `tools/upload_r2.sh --dry-run --delete` and read the orphan list.
- **Images replaced under the same file name are served stale for hours.** The
  site's CDN caches images (`cache-control: max-age=28800`, 8 h) under their URL,
  and the R2 key used for uploads cannot purge it. The landing page therefore
  references its logos and map with a content hash (`?v=…`,
  `tools/render_logo_grid.py`); other assets keep their plain URLs, so give a
  changed asset a new file name, or expect up to 8 h before readers see it.
- **Do not list a bucket while an upload runs against it** (§6.9).

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

- `docs/build-guide.md` — the learning path and task recipes.
- `docs/notebook-scheduler.md` — how notebooks are queued and placed on GPUs.
- `docs/architecture.md` — component inventory: what each tool and file is.
- `docs/build-history.md` — migration, implementation status and superseded designs.
- `docs/slides-northstar-design.md` — the slide-quality redesign this unblocks.
- `docs/mxnet-runtime-diagnostics.md` — MXNet wheel and runtime problems.
- `CLAUDE.md` — repo rules (source-of-truth `.md`, `make` targets, gh/HTTPS).

---

## 16. Implementation status (moved)

Every piece of this design has landed. The per-piece checklist (verified
2026-06-06, locks added 2026-11) is in
[build-history.md](build-history.md#implementation-status).
