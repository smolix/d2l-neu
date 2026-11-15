# D2L-Neu Project Context

Rebuild of "Dive into Deep Learning" (d2l.ai) using Quarto.

**Building anything? Read `docs/build-guide.md` first**: mental model,
glossary and step-by-step recipes (render, refresh a notebook, library
changes, new sections, full rebuild, locks, PDFs, publishing, a failing
freshness gate). Reference: `docs/build-system.md` (store, freshness, targets,
pitfalls), `docs/notebook-scheduler.md` (execution), `docs/architecture.md`
(what each tool is).

## Getting started

`make` means **GNU Make ≥ 4.3**; on macOS install it (`sudo port install
gmake`) and type `gmake`. Rendering needs no GPU and no framework venv.

```bash
./bootstrap.sh                          # once: git-lfs, .venv-build (Quarto), LFS images
make html                               # render the book from the committed outputs/
python3 -m http.server 8000 -d _book    # preview at http://localhost:8000/
```

| You want to… | Command | Needs | Guide |
|---|---|---|---|
| Render the HTML book | `make html` | any host | §4.1 |
| Render everything, no execution | `make all-quick` | any host (+TeX) | §3 |
| PDFs / part PDFs / slides | `make pdfs` / `make pdf-parts` / `make slides` | any host (+TeX) | §4.9 |
| Refresh one notebook's outputs | `rm -f _notebooks/<fw>/<ch>/<f>.executed && make _notebooks/<fw>/<ch>/<f>.executed`, then `python3 tools/capture_outputs.py --frameworks <fw> <ch>/<f>.md` | that venv; GPU for GPU notebooks | §4.3 |
| After a `#@save` change | `make lib notebooks && make audit-outputs`, then `make refresh-stale` | GPU box | §4.4 |
| Why does `make html` say stale? | `make audit-outputs` | any host | §4.12 |
| Full pipeline | `ulimit -u 8000; make all`, then `make capture-outputs` if `0 failed` | GPU box, 2.5–3 h | §4.7 |
| Re-run a locked notebook | `make rerun-locked FILES=...`, then `make capture-outputs FILES=... UNLOCK=1` | its hardware | §4.8 |
| Publish | `make deploy` | credentials | §4.11 |
| Resource plan | `make detect` | any host | — |

Venvs: one per framework (`.venv-pytorch`, `-tensorflow`, `-jax`, `-mxnet`,
via `make venv-<fw>`) plus `.venv-build`. **torch is pinned to 2.11.0 /
torchvision 0.26.0**, the version the store was captured under; moving to 2.12
means bumping `pyproject.toml` **and** re-capturing the store. VS Code:
`make kernels` once, then the `.vscode-extension/` "Edit Framework View"
command (see `.vscode-extension/README.md`).

## Rules

Each rule's reason and details are in `docs/build-guide.md` §5.

- Source `.md` files are the source of truth. **Never edit `.qmd` files** or
  anything under `_notebooks/`; they are regenerated.
- Use `make` targets (they log to `logs/`), not manual venv invocations.
- Notebook execution goes through the scheduler (`make run-all-notebooks`,
  `make run-notebooks-<fw>`, `make refresh-stale`, or
  `tools/notebook_scheduler.py --files`). **Never add an outer `make -j`**;
  the scheduler owns concurrency (`docs/notebook-scheduler.md`).
- **`make all` does not capture.** The book shows new outputs only after
  `make capture-outputs` and a commit of `outputs/`.
- **Capture only after a run that ended `0 failed`**, and only the frameworks
  you executed: capture copies whatever generated notebook it is given.
- Never wrap `make html` in `-j`, and never run it next to PDF or slide
  renders or notebook runs in the same tree (`docs/build-system.md` §6.8).
- Raise `ulimit -u 8000` before a full-concurrency run
  (`docs/build-system.md` §13).
- Never edit or reuse a cell ID; never edit source `.md` files during a
  notebook run.
- **Locked notebooks** (`<!-- execution: locked; ... -->`) are never executed
  by a build; only `make rerun-locked` runs them (`docs/build-system.md` §3.7).
- `make clean` keeps `data/`; `make veryclean` also deletes it.
- A `#@save` change can stale notebooks in other chapters and frameworks; the
  audit names them (`docs/build-system.md` §3.4).

## Source conventions (post-slides-refactor)

- Every Python code fence carries a stable `#<id>`. IDs are assigned
  by `tools/add_cell_ids.py` (idempotent) and **never change** once
  written, even on section rename or cell move.
- Untagged Python cells = "all frameworks". `#@tab all` and `%%tab all`
  are dropped from the corpus (`tools/strip_tab_all.py`).
- Slides live in `<!-- slides -->` sections at end-of-file as
  `::: {.slide ...}` divs with `@<id>` (or `@<id>@<fw>`) placeholders
  pointing to code cells. Inline `[**…**]` markers no longer exist.
- Slide rendering uses `eval: false`; outputs come from
  `tools/inject_outputs.py slides`, matched by cell ID against the committed
  `outputs/<fw>/...` store. Slides need no GPU.

## Content authoring (how to write a chapter)

Conventions for writing/revising chapter `.md` content. Model the density and
shape on the **preliminaries** and **linear-neural-networks** chapters.

- **Framework coverage policy (2026-07-17, per Alex):** the Basics part
  (ch. 1–8) and the math appendix carry all four frameworks; the **Advanced
  part (ch. 9–17) carries PyTorch and JAX only**. New or rewritten Advanced
  content must not add TensorFlow/MXNet tabs, and existing TF/MXNet material
  there is slated for removal as chapters are touched. Per-chapter framework
  subsets are already supported by the pipeline (the RL chapter ships
  PyTorch-only: coverage follows per-cell tabs, and `outputs/` simply has no
  tf/mxnet trees for it).
- **Code teaches; it does not draw.** Every notebook code cell must *compute or
  demonstrate* something the prose discusses (fit a model, decompose a matrix,
  verify an identity, classify data). Code whose only purpose is to draw an
  illustrative figure does NOT belong in the notebook — it is a wall of
  matplotlib plumbing that teaches nothing. Keep teaching code compact and
  elegant.
- **Illustrative figures are pre-generated, never drawn inline.** Conceptual /
  schematic / illustrative figures are produced by committed matplotlib
  generators → `img/mdl-<chapter>-<id>.svg` and included as
  `![caption](../img/<id>.svg)` + `:label:`fig_…`` with **no drawing code** in
  the notebook — exactly like the slide SVGs. The shared house style lives in
  `tools/gen_mdl_figures.py` (also the Linear Algebra figures); each other
  chapter has `tools/gen_mdl_<chapter>_figures.py` that imports it. To change a
  figure, edit its generator and run `make figures` (byte-idempotent); commit
  the `.svg`. Use the **`mdl-figure`** skill to add one and **`figure-style-audit`**
  to check a chapter. (Data plots that *teach* a computed result, e.g. a loss
  curve, may stay as a short `d2l.plot(...)` cell.)
- **One figure style per chapter.** Use a single consistent look (the
  generator's shared style). Do not mix hand-drawn SVGs, inline matplotlib, and
  generated plots in the same chapter — reproduce stragglers in the chosen style.
- **Figure house style — the checklist every generated figure must pass.**
  These rules are distilled from repeated review feedback; apply them to *every*
  panel of *every* figure:
  1. **Black coordinate axes and text labels.** Pass `color="black"` to
     `axis_cross(...)`; recolor any grey `axhline`/`axvline` that acts as a
     coordinate axis, and any grey *annotation text*, to black. This is about
     axes and labels only — leave data curves, contours, and light reference
     shapes (unit circles, faint grids) in their palette colours.
  2. **Legible, consistent fonts.** In-figure math/labels ~13–16 pt; never leave
     text tiny relative to the rest of the figure; match sizes within a figure.
  3. **No text↔line intersections.** A label must not sit on a curve, arrowhead,
     grid line, axis, or another label. Offset it *outward* from the thing it
     names — radially past an ellipse-axis tip (the eigenvectors are the axes, so
     radial is the true outward normal), or into the empty wedge where the shape
     has narrowed; put a white `bbox` behind labels that fall over scatter/bars/
     grids. This is the #1 recurring defect and you **cannot** get it right by
     reading the code — see the render-and-review loop below.
  4. **No dead whitespace.** Tighten `lim` so the content roughly fills the panel.
  5. **Balanced pairs.** Paired/rowed panels must render at the same height. Same
     scene → identical `lim`. Different scenes (e.g. a matrix mapping a square to
     a tall sheared region) → keep equal aspect and either the same window *size*
     (both square) or set `gridspec` `width_ratios` to match the `lim` spans so
     the *scale* is shared. If the image genuinely won't fit at shared scale,
     zoom the panel out to show it whole (not-to-scale is fine — the caption can
     say so) rather than cropping off part of it.
  6. **No redundant supercaptions.** Drop `ax.set_title`/`suptitle` when the book
     caption already carries it (the "(a) …/(b) …" kind). *Keep* titles that are
     functional and absent from the caption: a left→right process sequence
     ("(1) input → (2) after Σ"), or per-panel quantitative labels ("rank 5,
     err 0.08", "κ ≈ 2.7").
  7. **3-D frames.** Keep matplotlib's default box + shaded panes; only blacken
     the axis lines (hide the numeric ticks if noisy). Do not replace the box
     with bare quiver arrows.
- **ALWAYS render and visually inspect a figure before declaring it done — never
  guess at label positions.** After editing a generator, regenerate and *look*:

  ```bash
  .venv-pytorch/bin/python tools/gen_mdl_<chapter>_figures.py   # writes img/mdl-<prefix>-*.svg
  /opt/local/bin/rsvg-convert -z 2 img/mdl-<prefix>-<id>.svg -o /tmp/f.png   # then open /tmp/f.png
  ```

  Fix, regenerate, re-inspect until clean; when overcorrecting a position, look
  again rather than assuming. For a whole chapter, tile the PNGs into grids
  (`/opt/local/bin/montage *.png -tile 3x -geometry 460x340 -label '%f' grid.png`)
  and scan them. Generators are **byte-idempotent** — the shared `save()` fixes
  `svg.hashsalt` and nulls the SVG date — so never add timestamps or unseeded
  `random`; a second run must produce identical SVGs.
- **One imports cell.** Import each library exactly once, in a single
  per-framework imports cell near the top; never re-import inside later cells.
  Prefer d2l helpers (`d2l.plot`, `d2l.plt`, `d2l.set_figsize`, …) over raw
  matplotlib.
- **Hierarchical structure.** Group content into **3–5 top-level `##`
  sections**, each with `###` (and occasional `####`) subsections — not a long
  flat run of `##` headings.
- **Proofs, intuition-first.** Foundations chapters favour short, *elegant*
  proofs (`**Proposition.**` / `**Proof.**` … `$\blacksquare$`) over assertion,
  led by a picture or intuition.
- **Experimental results in prose: quote only the precision that survives
  re-execution.** Notebook training runs are re-captured regularly, and
  seed-level accuracy fluctuates by a few tenths of a point. Prose, tables,
  and slides must therefore never quote per-seed last-digit values
  ("90.21 ± 0.34"), per-framework orderings that flip within noise, or exact
  timing ratios. Round to the stability the store actually has (≈½ point for
  10-epoch Fashion-MNIST runs), prefer ranges and qualitative statements
  ("both about 92%", "roughly 2× faster"), and never build a *conclusion* on
  a difference smaller than the run-to-run noise. Single seeded runs per
  framework, not multi-seed sweeps — the compute belongs to readers.
- **References use labels, never hardcoded numbers.** Always `:numref:`/
  `:ref:` against a `:label:`; chapter renumbering is routine (it has
  happened twice) and hardcoded "Chapter 7"/"§5.4" strings silently rot.
  Slide-cover kickers (`[Dive into Deep Learning · §N.M]{.kicker}`) are
  rewritten automatically by `gen_slides.py` from `CHAPTER_NUMBERING`.
- **Gotchas.** Multi-line display equations: put `$$` alone on its own opening
  and closing lines with `:eqlabel:` on the next line, or the label fails to
  attach (`tools/lint_source.py` reports this as an error). Image captions may hold one level of balanced brackets (`$E[R]$`,
  `$[a,b]$`), but never an unbalanced `]` or nested `[[…]]` — the figure's
  `:label:` then detaches and prints as literal `{#fig-…}` text; write matrices
  in captions with `\begin{smallmatrix}…\end{smallmatrix}`.
  **A closing `$` must not be followed by a digit** — pandoc's "don't make
  `$5 … $10` math" rule means `$x$2` won't close, mangling the math and breaking
  the PDF (`\symcal allowed only in math mode`); keep the digit inside the span
  or use `\text{…}` (e.g. `$\mathcal F=\{\text{1-Lipschitz}\}$`). For a
  parenthesized small matrix write `\left(\begin{smallmatrix}…\right)` (renders
  in HTML; pandoc emits the `psmallmatrix` the PDF preamble's `mathtools`
  defines). These are PDF-build tripwires — see `docs/build-system.md` §6.6.

## Landing-page university adopters

The "Adopted at universities worldwide" grid on the landing page is data-driven.

- **Single source of truth:** `tools/universities.json` — one entry per
  institution with `slug`, `name`, `country`, `logo` (filename in
  `static/landing/universities/`, or `null` if no logo yet), and an
  `evidence` list of `{url, year, course, instructor}` (course and
  instructor optional) pointing at a course/syllabus page that cites d2l.ai or
  the Zhang/Lipton/Li/Smola textbook. An entry with `"evidence": []` and an
  `evidence_note` is a real adopter whose link died; it is shown unlinked.
- **Evidence of record is in this repo:** `docs/university_courses_instructors.json`
  (and the `.md` rendering `docs/university-courses-and-instructors.md`) hold the
  ledger (course, instructor, snippet, source per adoption). The old external
  `/home/smola/d2l/data/uni_evidence/` directory no longer exists. To add a new
  adoption, add it to the ledger and to `tools/universities.json`.

Pipeline (chained by `make universities`):

  1. Legacy, only if `UNI_EVIDENCE_DIR` exists (default
     `/home/smola/d2l/data/uni_evidence`): `_consolidate.py` builds a TSV and
     `tools/build_universities_json.py` merges it with the on-disk logo set;
     both are skipped with a note when the directory is absent. That merge
     preserves `logo` filenames already present.
  3. `tools/download_logos.py` — fetches missing logos via Wikipedia
     REST `page/summary`. Saves `universities.json` after every download
     so it resumes safely after rate limits or interruptions. A skip
     list lives in `tools/_logo_skip.txt` for cases the API can't help
     with — those need a manually-placed logo file in
     `static/landing/universities/<slug>.svg|png` and a re-run.
  4. `tools/render_logo_grid.py` — replaces the block between
     `<!-- @universities-begin -->` / `<!-- @universities-end -->`
     in `index.md`. Each entry becomes
     `<a href="course_url"><img></a>` when its top evidence URL is
     non-weak (the d2l.ai homepage and paywalled aggregators like
     Studocu / CourseHero / Scribd are filtered out as link targets).
     `index.qmd` is regenerated from `index.md` by `d2l_preprocess.py`.

To add a single adoption, edit `tools/universities.json` directly, drop a logo into
`static/landing/universities/`, and run `tools/render_logo_grid.py`.

`make universities-rebuild` deletes `tools/universities.json` and rebuilds it
from the legacy evidence directory; it now refuses to run (and deletes nothing)
when that directory is missing.

## VS Code workflow

- `make kernels` registers `d2l-pytorch / d2l-tensorflow / d2l-jax /
  d2l-mxnet` ipykernels. Notebook metadata.kernelspec.name points
  here so VS Code auto-selects the right interpreter.
- Workspace extension at `.vscode-extension/`: hosts the sync daemon
  (round-trips notebook saves to source via `tools/sync_back.py`),
  command palette, status bar.
- `tools/lint_source.py` runs on save (problems pane) and pre-commit.
- `tools/watch_slides.py --fw <fw> --file <chapter/file.md>` drives
  the live slide preview.

## Environment

- Python 3.12+, XeLaTeX + rsvg-convert (PDF only). Quarto is declared as
  a `build` extra (`uv sync --extra build` populates `.venv-build/bin/quarto`).
- NVIDIA driver **590.48+** (CUDA 13.x driver line; backward-compatible
  with the CUDA 11.7/12.x pip toolkits the framework venvs install).
- GPU count and GPU slot counts are auto-detected from
  `nvidia-smi` (one scheduler slot per 7.5 GiB of each card: 24 GB GPU → 3,
  16 GB GPU → 2; `make detect`). Override with `GPU_SLOTS_PER=2,2,2,2
  CPU_SLOTS=...` on the make command line.
- The custom MXNet wheel has native OpenCV 4.6 runtime dependencies. On
  Ubuntu 24.04 install `libopencv-core406t64 libopencv-imgproc406t64
  libopencv-imgcodecs406t64`; `tools/check_runtime_deps.py mxnet` verifies
  these before MXNet notebooks execute.
- UV venvs per framework: `.venv-pytorch`, `.venv-jax`, etc., plus
  `.venv-build` for Quarto.
- PyTorch and JAX extras are mutually exclusive (conflicting CUDA packages)
- Per-framework GPU support (driver 590 / CUDA 13.1 floor):
  - **pytorch**: native via the `pytorch-cu128` index (torch 2.7+);
    SASS for sm_100/sm_120 included.
  - **jax**: native via `jax[cuda12]>=0.10`; XLA JIT targets the
    detected device.
  - **tensorflow**: TF 2.21 ships SASS through `sm_89` plus
    `compute_90` PTX; on Blackwell the first use of each op
    JIT-compiles from PTX (slower cold start, then cached).
  - **mxnet**: custom MXNet 2.0 CUDA 13 wheel, pinned in `pyproject.toml` to
    `2.0.0+cu133.bw.zombie.1` (GitHub release `mxnet-2.0.0.zombie.1`; it
    replaced `…20260529.3`). This build **ships `sm_89`
    kernels and runs on the RTX 4090** — the earlier
    `cudaErrorNoKernelImageForDevice` (error 209) is **resolved**; all MXNet GPU
    notebooks execute. One caveat: MXNet image-dataset notebooks spawn many
    DataLoader threads, so `tools/detect_resources.py` caps concurrent MXNet
    jobs (`MXNET_GPU_SLOTS`) from the open-file and process ulimits (8 on the
    4×4090 box at `ulimit -u` 4096; override with `MXNET_GPU_SLOTS=N`) —
    otherwise a starved worker misreports a missing-OpenCV error. See `docs/mxnet-runtime-diagnostics.md`
    (2026-06-06) and `docs/build-system.md` §6.5/§6.6.

## Current status

- **torch pinned to 2.11.0 / torchvision 0.26.0** (the committed store's
  capture version). Freshness provenance keys on the *public* version (build
  tag stripped) and the `verify-fresh` gate is host-capability-aware, so the
  store is a portable baseline: a CPU/Apple-Silicon machine renders the whole
  book and re-runs only the CPU notebooks it touches. See `docs/build-system.md`
  §3.1/§3.3a.
- **Part structure (2026-07-17 restructure, per Alex):** front matter
  (introduction is unnumbered) | **Basics** = ch. 1–8 (preliminaries through
  Sequence Models) | **Advanced** = 9 Optimization, 10 Attention, 11
  Transformers, 12 State Space Models (formerly "Gated and Linear
  Recurrence"), 13 Computational Performance (rebuilt 2026-07-20:
  roofline/regimes method, hardware, compilation, memory/precision, from-
  scratch + practical multi-GPU, transformer case study; PyTorch+JAX only),
  14 RL, 15 Deep RL (the RL chapter was split in two, 2026-07), 16 GANs
  (rebuilt 2026-08: closed-form objectives, relativistic/R3GAN, gradient
  penalties, six sections; PyTorch+JAX only), 17 Diffusion Models
  (placeholder) | **Language Models** = 18–31 (14 skeleton chapters
  `chapter_lm-*`, generated 2026-11 from the plan in
  `docs/llm/language_models_part_plan.md`: each file holds its title, label
  and the plan entry as a hidden author brief) | **Image Models** (formerly
  "Computer Vision") = 32 | math appendix 33–38 | Tools 39 | **Attic** =
  40 GP, 41 HPO, 42 RecSys (GP leads the Attic: it is the
  Bayesian-optimization background HPO builds on). The two NLP
  Pretraining/Applications chapters were deleted on 2026-11-14 (PR #188):
  ch. 19 (`chapter_lm-encoders`) replaces them, and their `seq2seq.md` moved
  to ch. 12 as §12.2 with its labels (kept mainly for history; the ch. 12
  pass will likely cut most of it). The Attic moved to the very end, after Tools, on
  2026-11-11; the PDFs number every chapter as the HTML does (no appendix
  lettering), on one 7×10 in, 10pt page (`gen_pdf.py`), and `make pdf-parts`
  cuts per-part PyTorch PDFs from the full book (`docs/build-system.md`
  §6.6a/b). Title shortenings ("Computation",
  "Modern Convnets", "Sequence Models", singular "Multilayer Perceptron")
  are deliberate: inside a single "Basics" part the long framework-y titles
  were redundant, and ch. 12's rename marks the book's pivot to the
  linear-recurrence/SSM view. The preface still describes the old three-part
  plan and needs a
  full rewrite (deferred, per Alex). **Invariant:** `CHAPTER_NUMBERING`
  dict order must equal `_quarto.yml` chapter order — `gen_pdf.py` emits
  the PDF book in dict order and `fix_latex.py` pairs tex `\chapter`s
  against it positionally; a mismatch silently misnumbers the PDF (see
  `docs/build-system.md` §4.1).
- The **"Mathematics for Deep Learning"** part (branch
  `math-for-deep-learning-appendix`) is now the **default math appendix**: a
  single part of 6 numbered chapters under `chapter_mdl-*` — **33** Linear
  Algebra, **34** Calculus, **35** Optimization, **36** Probability & Statistical
  Learning, **37** Information Theory, **38** Dynamics (Tools for Deep Learning is
  now **39**) — with `mdl-`-prefixed labels. All six chapters are now fully
  written; the appendix gained two new sections (`mdl-adaptive-stochastic-methods`
  = §35.4, `mdl-concentration-generalization` = §36.7) and the probability
  chapter was reordered (naive Bayes moved to the end, §36.9). The branch `math-appendix-polish` carries a
  full review-driven quality pass over all six chapters with pytorch-only
  recaptured outputs; the tf/jax/mxnet stores for the edited files still need
  re-execution on the GPU box.
  The legacy `chapter_appendix-mathematics-for-deep-learning/` has been
  **retired** — removed from `_quarto.yml` + `CHAPTER_NUMBERING` and its source
  + committed outputs deleted. Numbering lives in `CHAPTER_NUMBERING`
  (`tools/d2l_preprocess.py`); a file absent from it renders unnumbered — see
  `docs/build-system.md` §4.1.
- **Fresh all-framework green run (2026-06-06)** on the 4×RTX 4090 box with the
  `…20260529.3` MXNet wheel: `make clean && make all` executes all four
  frameworks with **0 notebook failures** (the previously-failing MXNet GPU/image
  and error-209 notebooks all pass), the freshness gate is clean, and **all 4
  PDFs build** (~46 MB each). `multiple-gpus*` refreshed in that run.
- **PDFs build green** via `make pdfs` (and `make all`). Several
  pandoc/LaTeX/Quarto pitfalls were fixed to get here — see `docs/build-system.md`
  §6.6 (mathtools preamble, the `$`-before-digit math rule, stale `img/outputs`
  PDFs, and the now-tracked `static/d2l-preamble.tex`).
- ~192 HTML pages (plus the math part), 4 PDFs, ~261 slide decks, d2l library —
  zero errors.
- Full `make all` takes ~3 hours. No CI yet: building is the manual `make all`
  (or `make clean && make all` for a pristine run); it now completes green with
  no manual fix-ups.

### d2l-en source fixes applied (in ../d2l-en, not committed there)
- matplotlib: removed deprecated `use_line_collection` from stem() calls
- gymnasium: `import gym` → `import gymnasium as gym`, removed `env.seed()`
- syne_tune: updated PythonBackend import path
- pandas 3.0: `select_dtypes(include='number')` instead of `dtypes!='object'`
- wikitext-2: updated DATA_HUB URL and hash in bert-dataset.md
