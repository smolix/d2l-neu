# D2L-Neu Architecture

Quarto-based rebuild of *Dive into Deep Learning* (d2l.ai), replacing the
custom d2l-book/Sphinx pipeline. Source content originates from the d2l-en
repository; the `.md` files now live in-tree under `chapter_*/`.

> **Note — build model in transition.** This document describes the *current*
> pipeline, in which site rendering (HTML/slides/PDF) reads executed notebook
> outputs from the on-disk `_notebooks/` tree. That coupling is being replaced
> by the **decoupled build** specified in **`build-system.md`**: a committed,
> cell-ID-keyed notebook-output store (text in git, image assets in Git LFS) so
> rendering runs CPU-only, with no notebook execution and no framework venvs, and
> notebooks re-execute only when their *code* changes. Read `build-system.md` for
> the capture/bless workflow, the randomness-immune freshness model, and how
> partial re-execution keeps every index correct.

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
  run_one_notebook.py       Per-notebook execution worker + resource locks
  run_notebooks.py          Legacy/manual batch notebook runner
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
 3. run-all-notebooks        Make CPU/GPU queues → run_one_notebook.py
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

#### Notebook execution (resource-scheduled)

```
make run-notebooks-<fw>
make run-all-notebooks
```

Execution writes outputs into `_notebooks/<fw>/*.ipynb` in place. These
outputs are later injected into HTML, PDF, and slide `.qmd` files.

Make tracks one `.executed` stamp per notebook. To rerun one notebook,
target its stamp directly:

```bash
make -B _notebooks/pytorch/chapter_linear-regression/linear-regression.executed
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

The Makefile is the source of truth. Notebook execution is tracked by
per-notebook `.executed` stamps plus generated `.d` files for `d2l.<symbol>`
dependencies.

| Change | Minimum useful command | Why |
|--------|------------------------|-----|
| Prose only in one source file | `make html` or `make -B slides-<fw> SLIDES_FILTER=<file.md>` | HTML and slides read source text directly. |
| One notebook source cell | `make notebooks-<fw>` then `make -B _notebooks/<fw>/<chapter>/<file>.executed` | Notebook generation is per-framework; execution is per notebook. |
| One slide deck | `make -B slides-<fw> SLIDES_FILTER=<file.md>` | Slide generation/rendering accepts source `.md` paths. |
| One framework PDF | `make pdf-<fw>` | PDFs are isolated under `_pdf/<fw>/`. |
| `#@save` library code | `make lib` then rerun affected notebooks; when in doubt rerun all frameworks | `d2l/*.py` is imported by many notebooks, so a package rebuild can affect results far beyond the edited file. |
| `pyproject.toml` or `uv.lock` | `make venv-<fw>` or target that depends on it | `.venv-<fw>/.synced` is keyed on these files. |
| Shared tools in `tools/*.py` | Rebuild the stage that uses the tool | Most generated artifacts depend on the relevant script. |

Useful single-file examples:

```bash
# Generate notebooks for one framework, then execute one notebook.
make notebooks-pytorch
make -B _notebooks/pytorch/chapter_linear-regression/linear-regression.executed

# Render one slide deck for one framework.
make -B slides-jax SLIDES_FILTER=chapter_attention-mechanisms-and-transformers/transformer.md

# Build one framework's PDF after notebook outputs exist.
make pdf-tensorflow
```

There is no Make target today that generates only one notebook file.
`gen_notebooks.py` generates all notebooks for the requested framework.
Likewise, `html` and `pdf-<fw>` render full books, not single pages.

---

## Notebook execution model

The Makefile is the primary scheduler. `tools/scan_notebook_manifests.py`
emits per-framework stamp lists split into:

- `EXECUTED_CPU_<fw>`
- `EXECUTED_GPU_<fw>`
- `EXECUTED_MULTI_GPU_<fw>`

`run-notebooks-<fw>` starts two sub-makes concurrently: one with
`-j$(CPU_SLOTS)` over the CPU list, and one with `-j$(GPU_SLOTS)` over the
single-GPU list. The multi-GPU list then runs with `-j1` after the
single-GPU queue drains. `run-all-notebooks` does the same thing globally
across all frameworks, so CPU-only notebooks from one framework can run
while GPU notebooks from another framework are still using the GPUs.

Each stamp recipe runs `tools/run_one_notebook.py`, which still uses
flock-based CPU/GPU slot locks as a backstop for direct stamp builds or
accidental external parallelism. CPU notebooks run with
`CUDA_VISIBLE_DEVICES=""` plus `CPU_ONLY_ENV`; GPU notebooks get a single
`CUDA_VISIBLE_DEVICES=<gpu>`; multi-GPU notebooks acquire all GPU slots.

`tools/run_notebooks.py` remains as a legacy/manual batch runner, but it is
not the Makefile's primary execution path. Successful direct runs also touch
the corresponding `.executed` stamp so the audit report can distinguish a
current manual rerun from a stale generated notebook.

### Classification

Notebooks are classified by scanning source text for GPU keywords:
`gpu(`, `cuda`, `GPU`, `num_gpus`, `try_gpu`, `Trainer(`, `d2l.train`, etc.
Files listed in `MULTI_GPU_NOTEBOOKS` are routed to the multi-GPU queue.

### CPU affinity

The Make scheduler controls CPU load with `CPU_SLOTS`. CPU-affinity helpers
remain in `runtime_env.py` and are used by the legacy batch runner; the
per-notebook Make path does not currently pin individual notebook processes.

### Execution method

- **Notebooks**: `jupyter nbconvert --execute --inplace` as a subprocess.
  Per-cell timeout 3600s.

### Stale kernel cleanup

`runtime_env.py` provides `kill_stale_kernels(venv_dir)` for the legacy batch
runner. The per-notebook Make path relies on normal nbconvert/kernel shutdown
and the resource locks around each process.

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

The custom MXNet wheel also has native OpenCV 4.6 runtime dependencies that
are not expressible in Python package metadata. On Ubuntu 24.04 the required
packages are `libopencv-core406t64`, `libopencv-imgproc406t64`, and
`libopencv-imgcodecs406t64`. The Makefile runs `tools/check_runtime_deps.py
mxnet` before MXNet notebook execution so a missing system library fails early
with the install hint.

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
GPU_SLOTS           ?= 8          # Concurrent single-GPU notebook slots
CPU_SLOTS           ?= 4          # Concurrent CPU-only notebook slots
SLIDES_FILTER       ?=            # Optional glob to select specific slides
FILES               ?=            # Default for SLIDES_FILTER
```

### Key targets

| Target | What it does | Parallel-safe? |
|--------|-------------|----------------|
| `html` | Preprocess + quarto render + fix numbering | yes |
| `pdf-<fw>` | Generate + render + fix one PDF | yes |
| `pdfs` | All 4 PDFs (use `make -j4 pdfs`) | yes |
| `notebooks` | Generate .ipynb (no execution) | yes |
| `run-notebooks-<fw>` | Execute one framework's notebooks with CPU/GPU queues | yes, internally scheduled |
| `run-all-notebooks` | Execute all frameworks with global CPU/GPU queues | yes, internally scheduled |
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
uv sync --extra pytorch --extra run   # PyTorch + CUDA 12.8 + Jupyter
uv sync --extra jax --extra run       # JAX + TF + Jupyter
uv sync --extra tensorflow --extra run
uv sync --extra mxnet --extra run
```

- PyTorch, JAX, TensorFlow, and MXNet extras are **mutually exclusive**
  (conflicting CUDA package majors) — each gets its own venv
  (`.venv-pytorch`, `.venv-jax`, `.venv-tensorflow`, `.venv-mxnet`).
- torch/torchvision pinned to the `pytorch-cu128` index (CUDA 12.8
  wheels; includes sm_100/sm_120 SASS for Blackwell).
- JAX uses `jax[cuda12]` (native CUDA 12 wheels).
- TensorFlow and MXNet venvs install their own
  `nvidia-*-cu12` / `nvidia-*-cu11` runtime libs; the Makefile prepends
  those paths to `LD_LIBRARY_PATH` before launching notebooks.
- uv.lock covers Python 3.12+, linux x86_64 only.
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
- 1–4× NVIDIA GPUs (the Makefile autodetects count and per-GPU memory
  from `nvidia-smi`; rule of thumb is ~11 GB per notebook job, so a
  24 GB GPU runs 2 in parallel, a 16 GB GPU runs 1, etc.)
- NVIDIA driver **590.48+** (CUDA 13.x driver line) recommended.
  Lower drivers work for everything except Blackwell (sm_100/sm_120):
  the cu128 and cu12 wheels load on driver 525+ via CUDA forward
  compatibility, but Blackwell SASS in the `pytorch-cu128` index needs
  driver 570+ and the CUDA 13 driver line (590+) for the full stack.
- Sufficient CPU cores for the parallel workers + affinity
- ~50 GB disk for all build artifacts
- Linux x86_64

### Per-framework GPU coverage

| Framework  | Wheel arch coverage | Notes |
|------------|--------------------|-------|
| pytorch    | `pytorch-cu128` (torch 2.7+): SASS through sm_120 | Blackwell-native. |
| jax        | `jax[cuda12]>=0.10` native CUDA 12 | XLA JIT targets the detected device. |
| tensorflow | SASS through sm_89 + `compute_90` PTX (TF 2.21) | First op on Blackwell JIT-compiles from PTX (slower cold start, then cached). |
| mxnet      | Custom MXNet 2.0 CUDA 13 wheel, Blackwell-oriented | The current wheel imports on Ada (`sm_89`) but fails common GPU kernels with `cudaErrorNoKernelImageForDevice`; see `docs/mxnet-runtime-diagnostics.md`. Rebuild the wheel with `sm_89` and a PTX fallback before treating MXNet GPU notebooks as supported on RTX 4090 hosts. |

The full pipeline (`make all`) takes approximately:
- Notebook execution: ~110 min total (PT ~21, TF ~24, JAX ~11, MX ~54)
  on 4× 24 GB GPUs; scales linearly with effective GPU-job parallelism.
- Slides: a few minutes with CPU parallel rendering
- HTML + PDFs: ~15 min
- Total wall time: ~3 hours
