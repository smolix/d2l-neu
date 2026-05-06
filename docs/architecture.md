# D2L-Neu Architecture

Quarto-based rebuild of *Dive into Deep Learning* (d2l.ai), replacing the
custom d2l-book/Sphinx pipeline. Source content originates from the d2l-en
repository; the `.md` files now live in-tree under `chapter_*/`.

## Content scale

| Artifact        | Count |
|-----------------|-------|
| Source .md      | 191 files across 27 chapter directories |
| HTML pages      | 192 (multi-framework tabs) |
| PDFs            | 4 (one per framework, ~34 MB each) |
| Notebooks       | 413 (PT 130, TF 90, JAX 65, MX 128) |
| Slides          | 261 (PT 75, TF 56, JAX 55, MX 75) |
| d2l library     | 11,229 lines (torch 3441, mxnet 3321, tf 2429, jax 2026) |
| Bibliography    | 487 entries in d2l.bib |

Frameworks: **PyTorch**, **TensorFlow**, **JAX**, **MXNet**.

---

## Repository layout

```
chapter_*/                  Source .md + generated .qmd (27 dirs, 191 files each)
tools/                      Build scripts (5,235 lines total)
  d2l_preprocess.py         .md → .qmd conversion (1,075 lines)
  build_lib.py              #@save extraction → d2l package (705 lines)
  inject_outputs.py         Notebook output → HTML/PDF/slides injection
  gen_slides.py             Slide generation + CPU-only revealjs rendering
  gen_pdf.py                Single-framework PDF sources (355 lines)
  run_notebooks.py          GPU-parallel notebook execution (355 lines)
  fix_crossref_numbers.py   Post-render HTML chapter/section numbering (332 lines)
  gen_notebooks.py          Per-framework .ipynb generation (248 lines)
  fix_latex.py              LaTeX hierarchy restructuring (214 lines)
  runtime_env.py            Env vars, memory, CPU affinity, kernel cleanup (191 lines)
  gen_api_doc.py            Per-framework tabbed API reference (307 lines)
d2l/                        Generated Python package (preamble + auto-generated code)
  torch.py, tensorflow.py, jax.py, mxnet.py, __init__.py
config.ini                  Per-framework library aliases and metadata
Makefile                    Primary build entry point (277 lines)
build.sh                    Legacy bash wrapper (203 lines)
_quarto.yml                 Quarto book config (307 lines)
pyproject.toml              UV package manager, framework extras (93 lines)
_d2l-theme.scss             Material Design blue theme (149 lines)
_d2l-tabs.html              Tab sync + sidebar collapse JS (207 lines)
_d2l-style.css              Scrollable output, hidden tab headers (18 lines)
_svg-to-pdf.lua             Lua filter: SVG → PDF for LaTeX
static/                     Fonts (Source Serif 4, Source Sans 3, Inconsolata) + LaTeX preamble
data/                       Shared dataset cache (git-ignored, survives `make clean`)
img/                        SVGs (copied from d2l-en)
d2l.bib                     Bibliography
```

**Build artifacts** (git-ignored):

```
_book/                      HTML output (192 pages)
_pdf/<fw>/                  Per-framework PDF sources + compiled PDF
_notebooks/<fw>/            Per-framework Jupyter notebooks (executed in-place)
_slides/<fw>/               Per-framework Reveal.js slide decks
logs/                       Timestamped build logs (<target>-YYYYMMDD-HHMMSS.log)
.venv-<fw>/                 Isolated UV environments per framework
```

---

## Source content format

Source `.md` files use d2l-book directives. Key patterns:

**Prose tabs** — framework-specific prose:
```markdown
:begin_tab:`pytorch`
PyTorch-specific explanation.
:end_tab:
```

**Code tabs** — framework-specific code blocks:
````markdown
```{.python .input}
#@tab pytorch
import torch
x = torch.arange(12)
```
````

**Saved code** — extracted into d2l library by build_lib.py:
````markdown
```{.python .input}
#@tab pytorch
def my_function(x):  #@save
    return x + 1
```
````

**Conditional branches** — inline framework switching:
```python
if tab.selected('pytorch'):
    import torch
elif tab.selected('tensorflow'):
    import tensorflow
```

**Cross-references and labels**:
```markdown
# Section Title
:label:`sec_my_section`

As shown in :numref:`sec_my_section` and :numref:`fig_my_figure`...
:eqref:`eq_loss`
:cite:`Author.2020`
```

**Slide divs** — control what enters slide decks:
```markdown
<!-- slides -->

::: {.slide title="Why vectorize?" layout="2col"}
@vec-loop

@!vec-add
:::
```

Slides are authored in `<!-- slides -->` sections as `::: {.slide ...}`
divs. `@<id>` placeholders include a source code cell by its stable
cell ID, and `@!<id>` injects the executed output without echoing code.
Use `@<id>@<fw>` or `@!<id>@<fw>` to force a framework-specific variant.

---

## Build pipeline

Entry point: `make <target>` (preferred) or `./build.sh <target>`.

### Full pipeline (`make all`)

```
 1. lib                     build_lib.py → d2l/*.py
 2. notebooks               gen_notebooks.py → _notebooks/<fw>/*.ipynb
 3. run-all-notebooks        run_notebooks.py (PT → TF → JAX → MX, sequential)
 4. slides                  gen_slides.py --render → _slides/<fw>/*.html
 5. html (rebuild)           d2l_preprocess.py → quarto render → fix_crossref_numbers.py
 6. pdfs (rebuild, -j4)      gen_pdf.py → inject_outputs.py → quarto → fix_latex.py → xelatex ×2
```

Steps 5–6 rebuild HTML/PDFs with executed notebook outputs injected.

### Individual stages

#### HTML (multi-framework)

```
chapter_*/*.md → d2l_preprocess.py --primary pytorch → chapter_*/*.qmd
                → quarto render --to html → _book/
                → fix_crossref_numbers.py → _book/ (patched)
```

The preprocessor converts d2l directives to Quarto equivalents, groups
framework code into tabsets, auto-numbers equations, and translates
cross-references. The primary framework (PyTorch) gets executable code
cells; others are display-only.

`fix_crossref_numbers.py` remaps Quarto's file-position chapter numbers
to logical chapter numbers (defined in `CHAPTER_NUMBERING`, a 191-entry
static map in d2l_preprocess.py). It patches `header-section-number`
spans, `data-number` attributes, sidebar entries, and equation/figure
references across all 192 HTML files.

#### PDF (single-framework, parallel-safe)

```
gen_pdf.py → _pdf/<fw>/*.qmd (single-framework, stripped prose)
inject_outputs.py → inject notebook cell outputs into .qmd
rsvg-convert → _pdf/<fw>/img/*.svg → .pdf
quarto render --to pdf → .tex
fix_latex.py → patched .tex (chapter hierarchy, frontmatter, appendix)
xelatex ×2 → Dive-into-Deep-Learning-<fw>.pdf
```

`fix_latex.py` restructures the flat LaTeX output into proper book form:
frontmatter with roman numerals, numbered chapters, sections demoted
within file-level chapters, appendix with letter numbering.

All 4 PDFs can build in parallel (`make -j4 pdfs`).

#### Notebooks (per-framework)

```
gen_notebooks.py → _notebooks/<fw>/chapter_*/*.qmd → quarto convert → .ipynb
```

Per file: extracts the target framework's code, flattens
`tab.selected()` branches, strips boilerplate. Converts to .ipynb via
`quarto convert` (10 threads). Creates symlinks for `img/` and `data/`.

Not every source file produces a notebook for every framework — only
files that contain code for that framework.

#### Notebook execution (GPU-parallel)

```
run_notebooks.py <fw> --parallel 8 --num-gpus 4 --continue-on-error
```

Execution writes outputs into `_notebooks/<fw>/*.ipynb` in place. These
outputs are later injected into HTML, PDF, and slide `.qmd` files.

`run_notebooks.py` can narrow execution with `--files` or `--glob`.
Through Make, use `NB_FILES="chapter/name/file.ipynb"` or `FILES=...`
with notebook paths relative to `_notebooks/<fw>/`. Make still tracks a single
`_notebooks/<fw>/.executed` stamp per framework, so force a rerun when
the stamp is already fresh:

```bash
make -B run-notebooks-pytorch NB_FILES=chapter_linear-regression/linear-regression.ipynb
```

If a source edit changes any `#@save` code, rebuild `d2l/` first and
treat downstream notebook results as potentially stale until rerun.

#### Slides (CPU-only rendering)

```
gen_slides.py <source> _slides --frameworks <fw> --render --workers 8
```

Extracts `::: {.slide}` divs from source `.md`, resolves `@<id>`
placeholders to framework-specific source cells, generates per-framework
Reveal.js `.qmd`, injects outputs from executed notebooks when present,
then renders with `quarto render --to revealjs`.

Slide rendering is CPU-only. It is safe to run multiple frameworks in
parallel, for example `make -j4 slides`. To regenerate and render a
single deck for one framework:

```bash
make -B slides-pytorch SLIDES_FILTER=chapter_linear-regression/linear-regression.md
```

The `-B` matters for the same reason as notebook execution: Make uses
one `_slides/<fw>/.built` stamp per framework.

#### d2l library

```
build_lib.py <source> d2l/
```

Scans all source `.md` files for `#@save` code blocks, extracts
framework-specific code, merges `@d2l.add_to_class()` methods into their
class bodies, deduplicates (later definitions win), drops blocks with
unresolved `d2l.*` references, and writes to `d2l/<fw>.py`.

Each `d2l/<fw>.py` has a hand-written **preamble** (before the `#####
WARNING #####` marker) that build_lib.py preserves. The preamble handles
framework imports and initialization. The auto-generated section below
the marker is overwritten on every build.

**Preamble highlights:**
- `d2l/jax.py`: Initializes JAX first (`jax.devices()`), then imports TF
  and enables `memory_growth` so TF doesn't pre-allocate GPU memory.
- `d2l/tensorflow.py`: Sets `TF_CPP_MIN_LOG_LEVEL=2`, enables
  `memory_growth` on all GPUs.
- `d2l/torch.py`, `d2l/mxnet.py`: Straightforward framework imports.

Aliases (simple, fluent, custom) are appended from `config.ini`.

#### Output injection

```
inject_outputs.py html
inject_outputs.py pdf --framework <fw> --pdf-dir <dir>
inject_outputs.py slides --framework <fw> --slides-dir _slides
```

After notebooks are executed, this tool injects their cell outputs
(text, images, tables) back into HTML, PDF, or slide source `.qmd`
files. This is what makes rendered artifacts show actual computation
results. Matching is by stable notebook cell ID.

---

## Incremental build map

The Makefile is the source of truth. It optimizes by target stamps, not
by per-notebook dependency tracking.

| Change | Minimum useful command | Why |
|--------|------------------------|-----|
| Prose only in one source file | `make html` or `make -B slides-<fw> SLIDES_FILTER=<file.md>` | HTML and slides read source text directly. |
| One notebook source cell | `make notebooks-<fw>` then `make -B run-notebooks-<fw> NB_FILES=<file.ipynb>` | Notebook generation is per-framework; execution can be filtered. |
| One slide deck | `make -B slides-<fw> SLIDES_FILTER=<file.md>` | Slide generation/rendering accepts source `.md` paths. |
| One framework PDF | `make pdf-<fw>` | PDFs are isolated under `_pdf/<fw>/`. |
| `#@save` library code | `make lib` then rerun affected notebooks; when in doubt rerun all frameworks | `d2l/*.py` is imported by many notebooks, so a package rebuild can affect results far beyond the edited file. |
| `pyproject.toml` or `uv.lock` | `make venv-<fw>` or target that depends on it | `.venv-<fw>/.synced` is keyed on these files. |
| Shared tools in `tools/*.py` | Rebuild the stage that uses the tool | Most generated artifacts depend on the relevant script. |

Useful single-file examples:

```bash
# Generate notebooks for one framework, then execute one notebook.
make notebooks-pytorch
make -B run-notebooks-pytorch NB_FILES=chapter_linear-regression/linear-regression.ipynb

# Render one slide deck for one framework.
make -B slides-jax SLIDES_FILTER=chapter_attention-mechanisms-and-transformers/transformer.md

# Build one framework's PDF after notebook outputs exist.
make pdf-tensorflow
```

There is no Make target today that generates only one notebook file.
`gen_notebooks.py` generates all notebooks for the requested framework,
then `run_notebooks.py --files` can execute a subset. Likewise, `html`
and `pdf-<fw>` render full books, not single pages.

---

## Notebook execution model

`run_notebooks.py` uses GPU-aware scheduling from `runtime_env.py`.
Slides no longer use this path; they render with CPU-only Quarto
subprocesses.

### Worker layout (default: --parallel 8 --num-gpus 4)

```
GPU 0: worker 0, worker 4    ← CUDA_VISIBLE_DEVICES=0
GPU 1: worker 1, worker 5    ← CUDA_VISIBLE_DEVICES=1
GPU 2: worker 2, worker 6    ← CUDA_VISIBLE_DEVICES=2
GPU 3: worker 3, worker 7    ← CUDA_VISIBLE_DEVICES=3
CPU:   worker 8..11           ← CUDA_VISIBLE_DEVICES="" (no GPU)
```

- **GPU workers**: `ThreadPoolExecutor(max_workers=8)`, GPU IDs from a
  queue (2 tokens per GPU). Each subprocess gets
  `CUDA_VISIBLE_DEVICES=<gpu>`.
- **CPU workers**: `ThreadPoolExecutor(max_workers=4)`, run with
  `CUDA_VISIBLE_DEVICES=""` plus `CPU_ONLY_ENV` (JAX_PLATFORMS=cpu,
  TF_CPP_MIN_LOG_LEVEL=3).
- **Multi-GPU notebooks**: Run serially after the parallel phase with
  all GPUs visible. Known list in `MULTI_GPU_NOTEBOOKS`.

### Classification

Notebooks are classified by scanning code cells for GPU keywords:
`gpu(`, `cuda`, `GPU`, `num_gpus`, `try_gpu`, `Trainer(`, `d2l.train`, etc.

### CPU affinity

Each worker is pinned to a subset of host CPUs via
`os.sched_setaffinity()` (preexec_fn). GPU workers get up to 32 CPUs,
CPU workers get 16. Cores are strided across the host to minimize
contention between adjacent workers.

### Execution method

- **Notebooks**: `jupyter nbconvert --execute --inplace` as a subprocess.
  Per-cell timeout 3600s.

### Stale kernel cleanup

After each notebook run, `kill_stale_kernels(venv_dir)` scans `/proc` for
surviving `ipykernel_launcher` processes from the framework's venv and
SIGKILL-s them. MXNet kernels in particular deadlock during Python
interpreter shutdown (GIL vs MXNet engine threads) and survive their
parent nbconvert process, leaking GPU memory.

---

## Framework environment (runtime_env.py)

`setup_framework_env(framework)` sets per-framework env vars before
execution:

| Variable | pytorch | tensorflow | jax | mxnet |
|----------|---------|-----------|-----|-------|
| OMP_NUM_THREADS | 4 | 4 | 4 | 4 |
| MKL_NUM_THREADS | 4 | — | — | — |
| OPENBLAS_NUM_THREADS | 4 | 4 | 4 | 4 |
| TF_NUM_INTRAOP_THREADS | — | 4 | 4 | — |
| TF_NUM_INTEROP_THREADS | — | 2 | 2 | — |
| XLA_PYTHON_CLIENT_MEM_FRACTION | — | — | .40 | — |
| TF_CPP_MIN_LOG_LEVEL | — | 2 | — | — |
| TF_XLA_FLAGS | — | --tf_xla_auto_jit=2 | — | — |
| MXNET_CUDNN_LIB_CHECKING | — | — | — | 0 |
| MXNET_CPU_WORKER_NTHREADS | — | — | — | 4 |
| MXNET_GPU_WORKER_NTHREADS | — | — | — | 2 |

Additionally: LD_LIBRARY_PATH is extended with `nvidia/*/lib` paths from
the venv's pip-installed nvidia packages, and the RLIMIT_NPROC soft limit
is raised to the hard limit (XLA spawns many threads).

### JAX + TF memory coexistence

JAX notebooks use TF for data loading (via `tensorflow_datasets`).
Two mechanisms prevent TF from fighting JAX for GPU memory:

1. **d2l/jax.py preamble**: Calls
   `tf.config.experimental.set_memory_growth(gpu, True)` after JAX
   initializes, so TF allocates on-demand instead of pre-allocating.
2. **XLA_PYTHON_CLIENT_MEM_FRACTION=.40**: JAX pre-allocates only 40% of
   GPU memory (down from the 90% default), leaving room for a second
   worker on the same GPU plus TF overhead.

---

## Makefile variables

```makefile
SOURCE              ?= .          # Source directory
NUM_GPUS            ?= 4          # GPUs available
PARALLEL_pytorch    ?= 8          # 2 per GPU
PARALLEL_tensorflow ?= 8
PARALLEL_jax        ?= 8
PARALLEL_mxnet      ?= 8
SLIDES_FILTER       ?=            # Optional glob to select specific slides
NB_FILES            ?=            # Optional file list for notebook execution
FILES               ?=            # Default for NB_FILES and SLIDES_FILTER
```

### Key targets

| Target | What it does | Parallel-safe? |
|--------|-------------|----------------|
| `html` | Preprocess + quarto render + fix numbering | yes |
| `pdf-<fw>` | Generate + render + fix one PDF | yes |
| `pdfs` | All 4 PDFs (use `make -j4 pdfs`) | yes |
| `notebooks` | Generate .ipynb (no execution) | yes |
| `run-notebooks-<fw>` | Execute one framework's notebooks | **no** (GPU) |
| `run-all-notebooks` | Execute PT → TF → JAX → MX sequentially | **no** (GPU) |
| `slides-<fw>` | Generate + render one framework's slides | yes (CPU-only) |
| `slides` | All 4 frameworks' slides; use `make -j4 slides` | yes (CPU-only) |
| `lib` | Build d2l Python package | yes |
| `all` | Full pipeline (lib → notebooks → run → slides → html → pdfs) | **no** |
| `all-quick` | html + pdfs + notebooks + slides + lib (no execution) | partially |
| `clean` | Remove _book/ _pdf/ _notebooks/ _slides/ (keeps data/) | — |
| `veryclean` | Also delete data/ | — |

All build output is logged to `logs/<target>-YYYYMMDD-HHMMSS.log`.

---

## UV environments

```bash
uv sync --extra pytorch --extra run   # PyTorch + CUDA 12.4 + Jupyter
uv sync --extra jax --extra run       # JAX + TF + Jupyter
uv sync --extra tensorflow --extra run
uv sync --extra mxnet --extra run
```

- PyTorch and JAX extras are **mutually exclusive** (conflicting CUDA
  packages) — each gets its own venv (`.venv-pytorch`, `.venv-jax`, etc.).
- torch/torchvision pinned to `pytorch-cu124` index.
- uv.lock covers Python 3.10–3.14, linux x86_64 only.
- The `run` extra adds jupyter, nbconvert, ipykernel.

---

## Theme and styling

**Colors** (_d2l-theme.scss): Material Design blue palette.
Primary `#2196F3`, dark `#1976D2`, light `#BBDEFB`, accent `#FF5722`.

**Fonts** (static/fonts/): Source Serif 4 (body), Source Sans 3 (headings),
Inconsolata (code).

**Tab sync** (_d2l-tabs.html): Custom JS syncs all `panel-tabset
group="framework"` tabs, persists selection to localStorage. Works on
`file://` URLs (Quarto's native tab sync requires ES modules over HTTP).

**Sidebar** (_d2l-tabs.html): Part headers are clickable toggles
(JS-driven). Sections default collapsed; expand state persists to
localStorage. The section containing the current page auto-expands.

**Code blocks** (_d2l-theme.scss): Left blue border, light gray
background, 0.9rem font.

---

## Hardware assumptions

The build system assumes a machine with:
- 4× NVIDIA GPUs (RTX 4090 with 24 GB each, or similar)
- Sufficient CPU cores for 8 GPU workers + 4 CPU workers with affinity
- ~50 GB disk for all build artifacts
- Linux x86_64

The full pipeline (`make all`) takes approximately:
- Notebook execution: ~110 min total (PT ~21, TF ~24, JAX ~11, MX ~54)
- Slides: a few minutes with CPU parallel rendering
- HTML + PDFs: ~15 min
- Total wall time: ~3 hours
