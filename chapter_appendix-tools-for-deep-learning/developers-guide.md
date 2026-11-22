# Contributor Guide
:label:`sec_developers_guide`

By now you have run many of this book's notebooks, and you may have found a
sentence that confused you, a transposed subscript in an equation, or a cell
that fails with a newer library. A fix should appear wherever that text or
code appears: on the website, in the PDF, in the notebook of every
framework, and on the slide that quotes the same cell. How does a one-line
correction reach all of these places, and how do we know that it broke
nothing?

The book is maintained the way open-source software is. Every page comes
from one Markdown file under version control, a build system derives
everything else from these files, and a change is ready for review once
the notebooks it touches execute and the site renders. We describe the build,
the Git workflow for proposing a change, the optional use of coding agents,
and the checks a change must pass. The same concerns arise in any machine
learning repository whose documents must run.

## The Build

### One Source, Many Artifacts

Each section is stored in a single Markdown file; this one is
`chapter_appendix-tools-for-deep-learning/developers-guide.md`. The build
derives everything else from these sources:

```text
chapter_*/*.md              source of truth (edit these)
├── chapter_*/*.qmd         generated Quarto input → HTML site and PDFs
├── _notebooks/<fw>/...     generated Jupyter notebooks, one set per framework
├── _slides/<fw>/...        generated lecture slides
└── d2l/*.py                generated d2l library, from the #@save blocks
outputs/<fw>/...            executed outputs, captured from _notebooks/, committed
```

This design has three practical consequences:

* **Never edit generated files.** A `.qmd` or `.ipynb` edit is destroyed
  by the next build. Edit the `.md` source and regenerate.
* **Framework tabs live in one file.** A code block tagged `%%tab pytorch`
  appears only in the PyTorch view; an untagged Python block appears in
  every notebook of the file. The file yields one notebook for each
  framework that at least one of its blocks is tagged for, and a file
  without tagged blocks yields a PyTorch notebook only. One source file
  thus yields up to four notebooks that must all execute. Chapters of the
  Basics part and the mathematics appendix carry all four frameworks;
  chapters of the Advanced part carry PyTorch and JAX only.
* **Cell identity is stable.** Every Python block carries a permanent ID
  (assigned by `tools/add_cell_ids.py`) so that slides, executed outputs,
  and framework views can refer to a cell across renames and moves.

The sources are Markdown rather than notebooks because of version control.
A `.ipynb` file is JSON that mixes code, outputs, execution counts, and
editor metadata, so a one-word edit followed by a re-run changes many
lines besides that word; a Markdown diff shows only the changed prose and
code. Executed outputs are stored
separately and are re-captured only when the code that produced them, the
library code it uses, or the framework version changes.

The `d2l` library you have imported throughout the book is generated the
same way. `make lib` extracts every block marked `#@save` into the
per-framework modules `d2l/torch.py`, `d2l/tensorflow.py`, `d2l/jax.py`,
and `d2l/mxnet.py`. A fix made where a function is taught reaches every
later chapter after a rebuild, which is why those files carry a banner
that says not to edit them directly. The Colab and Kaggle notebooks of
:numref:`sec_hosted_notebooks` are generated too, onto a separate branch,
`notebooks`, that is rebuilt from the sources and never edited by hand.

### Repository Layout

The top level of the repository holds the sources, the build entry
points, and the documentation that explains them:

```text
d2l-neu/
├── chapter_*/            source Markdown, one directory per chapter
├── img/                  figures; illustrative SVGs come from tools/gen_*_figures.py
├── d2l/                  the generated d2l library
├── outputs/              committed store of executed notebook outputs
├── tools/                build scripts (preprocessing, notebooks, slides, audits)
├── docs/                 build-system documentation, writing and code guides
├── Makefile, make/       entry point: make html, make lib, make capture-outputs ...
├── bootstrap.sh          one-time setup of a fresh clone
└── CLAUDE.md             build summary, written for coding agents
```

On a fresh clone, `./bootstrap.sh` installs Git LFS, which stores the
images of the committed outputs, fetches those images, and creates
`.venv-build`, the environment that holds Quarto. After that, `make html`
renders the whole site from the committed outputs on a laptop without a
GPU; executing notebooks is a separate step. The Makefile needs GNU Make
4.3 or newer, which macOS provides only as an extra install (`gmake` from
MacPorts, for example). `make help` lists the targets, and
`docs/build-system.md` explains them.

## Git Basics

Every change to the book travels as a Git commit on a branch of your own
copy of the repository. If you already use Git daily, skip ahead. First,
make that copy: click **Fork** on the
[repository page](https://github.com/smolix/d2l-neu), then clone your fork and create a branch for your change:

```bash
git clone https://github.com/<your-username>/d2l-neu.git
cd d2l-neu
git switch -c fix-typo-in-attention    # a branch per change
```

Edit the source `.md` file, then record and publish the change:

```bash
git status                  # what changed?
git diff                    # show the exact edits
git add chapter_attention/attention-scoring.md
git commit -m "Fix transposed subscript in attention equation"
git push -u origin fix-typo-in-attention
```

GitHub will offer to open a *pull request*: your branch, proposed for
merging, with a diff the maintainers can review and comment on. (The
[`gh` command-line tool](https://cli.github.com/) does the same with
`gh pr create`.) Small, focused pull requests are easier to review than
changes spanning several chapters.

## Coding Agents

A coding agent is a language model that reads the repository, edits
files, and runs commands. It helps with bounded tasks: finding where a
function is defined, propagating a mechanical change across framework
variants, or running the checks and reading their failures. Its use is
optional, and its edits need the same review as anyone else's. Three
properties of this repository make such help practical:

* **The build is documented.** `CLAUDE.md` and `docs/` tell an agent how
  to regenerate, execute, and render the artifacts. The same instructions
  serve human contributors.
* **Verification is executable.** An edit works when the notebook executes
  top to bottom in every framework it has, the audit of the committed
  outputs passes, and the page renders. An agent can run the loop of
  editing, regenerating, executing, and inspecting failures, but the
  contributor reviews both the final diff and the evidence from the checks.
* **The tedious parts are mechanical.** Keeping framework implementations
  in sync, checking that every `:numref:` resolves, and regenerating
  figures are multi-file tasks that an agent can carry out and a person can
  review.

A useful request states the change and how to verify it, for example: "the
contour plot of the momentum section mislabels its axes; correct the source
and rerun the affected notebooks." Review for two failure modes. An agent
may edit a generated file, as newcomers do, unless the project
documentation warns against it (ours does). And an agent asked to make a
failing check pass may weaken the check or the code instead of fixing the
cause: a notebook that runs again may no longer teach the right thing.

## Submitting Changes
:label:`sec_how_to_contribute`

Contributions range from one-line corrections to new code, and the checks
grow with the change. A one-line fix does not even need a local checkout:
open the source file on GitHub and press the edit (pencil) button. GitHub
offers to fork the repository, and after the edit it leads you through the
commit to a pull request, all in the browser.

For code and content changes, clone, branch, and edit the source `.md`.
Check the edited file with `tools/lint_source.py`, which catches
unbalanced tab and slide markers, duplicate cell IDs, and equation labels
that would not attach; with `--corpus` it also checks that every
cross-reference points to an existing label. Then
re-execute every notebook that the edit touches, copy its outputs into the
committed store, and render the site:

```bash
make venv-pytorch                                     # once per framework
make -B _notebooks/pytorch/<chapter>/<file>.executed  # repeat per framework
make capture-outputs FILES=<chapter>/<file>.md
make html
```

If you edit code that carries `#@save`, rebuild the library with `make lib`
before you execute any notebook. Notebooks of other sections that use the edited symbol then become
stale too, and `make audit-outputs` lists them. A laptop without a GPU
cannot re-execute GPU notebooks; `make html` warns about such a notebook
instead of stopping, and the pull request should name the notebooks that
still need a GPU run. New prose follows `docs/writing-guide.md`, and new
code follows `docs/code-guide.md`.

If you prefer editing notebooks to raw Markdown, the VS Code extension in
`.vscode-extension/` (its README explains how to build and install it)
opens the generated notebook of the current section with the right kernel
preselected, switches between its framework views, previews the slide
deck live, and re-executes and captures a notebook with one command. It
writes edits made in the notebook back into the Markdown source, so a
generated file never becomes the thing you edited. `make kernels`
registers the Jupyter kernels it uses.

Whatever the route, a change is ready when it executes cleanly from a
fresh state in every framework it touches, and when the source, never a
generated file, is what changed.

## Summary

* One Markdown source generates the website, PDFs, per-framework
  notebooks, executed outputs, slides, and the `d2l` library; generated
  files are never edited directly.
* Markdown sources serve version control: readable diffs and reviewable
  pull requests keep a book of more than 600 notebooks maintainable.
* The contribution loop is fork → branch → edit source → verify (lint,
  execute notebooks, capture outputs, render) → small pull request.
* Coding agents can automate parts of this workflow because the build is
  documented and verification is executable; contributors review and take
  responsibility for submitted changes.

## Exercises

1. **Framework tabs.** A source file contains exactly five code blocks, in
   this order: (1) a block tagged `%%tab pytorch` that runs `import torch`
   and `x = torch.arange(6.0)`; (2) a block tagged `%%tab jax` that runs
   `import jax.numpy as jnp` and `x = jnp.arange(6.0)`; (3) an untagged
   block `y = x * 2`; (4) a block tagged `%%tab pytorch` that runs
   `x[0] = 10.0`; (5) an untagged block `print(y.sum(), x.sum())`.
    1. Which notebooks does the build generate from this file, which
       blocks does each of them contain, and what does each print?
    1. A contributor deletes the tag of block (4) so that every framework
       runs it. Which notebook now fails, and with which error? Replace
       block (4) by tagged blocks so that both notebooks print the same
       values.
    1. You edit block (3) in the generated PyTorch notebook, and the VS
       Code extension writes the edit back into the source file. Which
       notebooks does the edit change, and which of them must you run
       before you submit it?
   :exlabel:`ex_developers-guide-framework-tabs`

1. **Stale outputs.** The committed store keeps the outputs of a notebook
   in `outputs/<fw>/<chapter>/<stem>.json`, filed under the ID of the cell
   that produced them. With each output it stores a fingerprint: a hash of
   the code of that cell and of every earlier code cell of the notebook,
   with trailing whitespace stripped from every line. For the whole
   notebook it also stores the installed framework version without its
   local build tag (`torch==2.11.0`, not `torch==2.11.0+cu128`) and a hash
   of the library blocks of every symbol `d2l.X` that the code of the
   section mentions, together with the blocks that those symbols use in
   turn. An output is stale when one of these values no longer
   matches the current source and environment. The audit
   `tools/audit_outputs.py`, which `make html` runs before it renders the
   site, reports a stale image as a warning and stops the build at a stale
   text output, unless the notebook needs more GPUs than the host has.
    1. A section has ten code cells, and every framework variant of it
       calls `d2l.Trainer`. Which outputs, in which frameworks, become
       stale after each of the following edits: (i) a typo fixed in a
       paragraph; (ii) a typo fixed in a comment of the third code cell,
       which has no tab; (iii) trailing spaces removed from every line;
       (iv) the fifth and the sixth code cell swapped; (v) the docstring of
       a `Trainer` method changed in an untagged `#@save` cell of another
       section; (vi) PyTorch upgraded from 2.11.0 to 2.12.0; (vii) PyTorch
       2.11.0 installed from the CPU build for macOS instead of the CUDA
       build for Linux?
    1. Which of the stale marks of (a) flag outputs that cannot have
       changed? Which real change of an output does (vii) let pass? Why do
       the rules accept both kinds of error?
    1. Why does the build not declare an output stale when a fresh run
       prints something different from the stored output?
   :exlabel:`ex_developers-guide-stale-outputs`

1. [code] **Duplicate definitions.** `make lib` collects the `#@save`
   blocks of all sections in book order. When several blocks define the
   same function or method, the library keeps only one of them. The method
   `Trainer.prepare_batch` is defined in the untagged cell
   `#linear-regression-scratch-training-1` of :numref:`sec_oo-design`,
   which every framework runs, and again, with device support, in the
   framework variants of the cell
   `#gpus-devices-memory-the-trainer-now-with-devices-1`
   of :numref:`sec_use_gpu`.
    1. Add a comment line to the body of the untagged definition and of
       the PyTorch variant of the device-aware one, run `make lib`, and
       search `d2l/torch.py` for both comments. Which comment arrives,
       and which rule does this reveal?
    1. Which definition does `d2l.Trainer.prepare_batch` hold when the
       PyTorch notebook of :numref:`sec_oo-design` starts, and which
       one after its cell `#linear-regression-scratch-training-1` has run?
    1. Why does the build keep the last definition rather than the first?
    1. A contributor changes the device-aware PyTorch `prepare_batch` so
       that it also converts every batch to 16-bit floats. The PyTorch
       notebooks of which sections can print different results: those
       before :numref:`sec_use_gpu` in book order, those after it, or both?
   :exlabel:`ex_developers-guide-duplicate-definitions`

1. [extended] **Dataflow staleness.** ●● The fingerprint of exercise
   :exref:`ex_developers-guide-stale-outputs` marks every cell from the
   first edited cell on as stale. A dataflow rule marks only the cells
   that read, directly or through other cells, a name that the edited cell
   binds. The detector of exercise
   :exref:`ex_interactive-development-use-before-define` of
   :numref:`sec_interactive_development` already extracts the names that
   each cell binds and reads.
    1. Generate the PyTorch notebooks with `make notebooks-pytorch` and
       extract, for every code cell, the names that it binds and reads.
       For every edit of a single cell, compute the cells that the dataflow
       rule marks stale and the cells that must run to recompute them,
       which include every earlier cell that a stale cell depends on.
       Report per chapter the mean fraction of cells marked stale and the
       mean fraction that must run, under both rules.
    1. Find cells in the book through which an edit changes later outputs
       without binding a name that the later cells read, for instance
       through the state of a random number generator. Which other
       channels of this kind do the notebooks of the book use?
    1. :citet:`Macke.Gong.Lee.ea.2021` record such dependencies while the
       notebook runs instead of reading them from the source. Which of the
       channels of (b) can a run-time tracer observe, and which stay
       invisible to it?
    1. Would you replace the prefix rule of the build by your rule? Base
       the answer on the savings of (a) and the failures of (b).
   :exlabel:`ex_developers-guide-dataflow-staleness`

1. [code] **Noisy bisection.** ●● `git bisect` finds the commit that
   introduced a fault by binary search: it checks out a commit halfway
   between a known good and a known bad commit, and a test of that commit
   halves the range. A test that trains a model is noisy. Suppose that a
   notebook's check "validation accuracy at least 0.9" passes with
   probability $p = 0.9$ on every commit before the first bad commit $k$
   and with probability $q = 0.3$ on $k$ and every later commit,
   independently across runs. Commit 0 is known good, commit 64 is known
   bad, and $k$ is uniform on $\{1, \ldots, 64\}$.
    1. With one run per test, what is the probability that `git bisect`
       names $k$?
    1. Repeat each test $m$ times and call the commit good when the number
       of passes reaches a threshold. Find the smallest $m$, with its
       threshold, for which the probability of naming a wrong commit is at
       most 5%. How many runs does the search take?
    1. The probabilistic bisection algorithm keeps a posterior distribution
       over the location of the fault, tests at its median, and updates the
       posterior with Bayes' rule after every run
       :cite:`Frazier.Henderson.Waeber.2019`. Simulate it with a posterior
       over $k$ and the likelihoods given by $p$ and $q$, until the
       posterior puts 0.95 on one commit. Why can testing at the median
       stop making progress on a discrete set of commits? Change the choice
       of the test commit to fix this, and compare the mean number of runs
       and the error rate with (b).
    1. Treat one run as a channel from $\{\text{good}, \text{bad}\}$ to
       $\{\text{pass}, \text{fail}\}$ and compute its capacity
       $C = \max_{p(x)} I(X; Y)$, the largest mutual information between
       input and output over all input distributions
       (:numref:`sec_mdl-mutual-information`). Each run reveals at most
       $C$ bits about $k$, even when the tests are chosen adaptively.
       Fano's inequality states that an estimate $\hat k$ computed from the
       runs, with error probability $P_e = P(\hat k \neq k)$, satisfies
       $H_b(P_e) + P_e \log_2 63 \geq H(k \mid \text{runs})$, where $H_b$ is
       the binary entropy. Use it to bound from below the mean number of
       runs of any strategy that names $k$ with probability at least 0.95.
       How close does (c) come?
   :exlabel:`ex_developers-guide-noisy-bisection`

[Discussions](https://d2l.discourse.group/t/36848)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §39.8]{.kicker}

Contributor Guide: Single-Source Markdown Build, Framework Tabs, and Git Workflow<br>**Markdown source of truth · multi-framework tabs · #@save library generation · output fingerprinting · lint & audit · Git PR workflow**
:::
:::

::: {.slide title="Why D2L Uses Markdown as the Single Source of Truth"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why raw `.ipynb` files fail in Git:** Nested JSON mixes prose, code, base64 PNGs, and execution counters—a 1-word fix + cell run modifies 500+ lines and causes constant merge conflicts.
- **Single-source Markdown (`chapter_*/*.md`):** Pure human-readable Markdown gives 1-line Git diffs, while executed cell outputs live separately in `outputs/<fw>/*.json`.
:::

::: {.col .fig .big}
@fig:ch39-tools-ipynb-vs-markdown
:::
:::
:::

::: {.slide title="One Source File, Five Generated Artifacts"}
[The Build Pipeline]{.kicker}

::: {.cols .vc}
::: {.col}
- **Golden rule:** Edit **only** `chapter_*/*.md` (never `_notebooks/`, `*.qmd`, `_slides/`, or `d2l/*.py`).
- **Deterministic fan-out:** The build splices `outputs/<fw>/*.json` and derives the HTML website, PDFs, per-framework `.ipynb` notebooks, Reveal.js slide decks, and the `d2l` package automatically.
:::

::: {.col .fig .big}
@fig:ch39-tools-single-source-fanout
:::
:::
:::

::: {.slide title="Multi-Framework Tabs and Notebook Generation"}
[Cross-Framework Parity]{.kicker}

::: {.cols .vc}
::: {.col}
- **Tagged (`%%tab <fw>`) vs. untagged cells:** Tagged cells route only to their named framework(s); **untagged** Python cells are shared and execute in **every** framework notebook generated by that file!
- **Shared-cell discipline:** Editing an untagged cell requires re-executing and capturing outputs for all frameworks present in the file (`pytorch` and `jax` in Advanced chapters).
:::

::: {.col .fig .big}
@fig:ch39-tools-framework-tab-routing
:::
:::
:::

::: {.slide title="Permanent Cell IDs and the #@save Library Pipeline"}
[Code Identity & Reuse]{.kicker}

::: {.cols .vc}
::: {.col}
- **Permanent Cell IDs (`#sec-slug-k`):** Assigned by `tools/add_cell_ids.py`, keying committed outputs in `outputs/<fw>/` and slide references (`@cell-id`) across prose edits.
- **`#@save` library extractor (`make lib`):** Scans chapters in book order into `d2l/*.py` under a **last-definition-wins** rule (e.g., §{sec_use_gpu} GPU `Trainer` overrides §{sec_oo-design} CPU `Trainer` in `d2l`, while §{sec_oo-design} runs its own in-notebook definition).
:::

::: {.col .fig .big}
@fig:ch39-tools-cell-id-and-save
:::
:::
:::

::: {.slide title="Repository Layout and Fresh-Clone Bootstrap"}
[Developer Setup]{.kicker}

::: {.cols .vc}
::: {.col}
- **One-time `./bootstrap.sh`:** Pulls Git LFS plot assets and creates `.venv-build` with Quarto and Python build tools.
- **No-GPU local HTML build (`gmake html`):** Splices committed outputs from `outputs/<fw>/*.json` into `.qmd` files and renders the interactive website and slides in seconds on any laptop (requires GNU Make $\ge 4.3$).
:::

::: {.col .fig .big}
@fig:ch39-tools-bootstrap-pipeline
:::
:::
:::

::: {.slide title="How Output Fingerprinting Detects Stale Notebooks" only="pytorch"}
[Reproducibility Audit]{.kicker}

::: {.cols .vc}
::: {.col}
- **Three hashes per output JSON:** (1) Per-cell **prefix code hash** (cells $1\dots k$, trailing spaces stripped), (2) **public framework version** (local build tag stripped), and (3) **transitive `d2l.X` symbol hash**.
- **Surgical invalidation:** Prose edits cause 0 stale cells; editing cell $k$ marks cells $k\dots n$ stale; editing a `#@save` helper marks only downstream notebooks that call it.
:::

::: {.col .fig .big}
@fig:ch39-tools-output-fingerprint@pytorch
:::
:::
:::

::: {.slide title="How Output Fingerprinting Detects Stale Notebooks" only="jax"}
[Reproducibility Audit]{.kicker}

::: {.cols .vc}
::: {.col}
- **Three hashes per output JSON:** (1) Per-cell **prefix code hash** (cells $1\dots k$, trailing spaces stripped), (2) **public framework version** (local build tag stripped), and (3) **transitive `d2l.X` symbol hash**.
- **Surgical invalidation:** Prose edits cause 0 stale cells; editing cell $k$ marks cells $k\dots n$ stale; editing a `#@save` helper marks only downstream notebooks that call it.
:::

::: {.col .fig .big}
@fig:ch39-tools-output-fingerprint@jax
:::
:::
:::

::: {.slide title="How Output Fingerprinting Detects Stale Notebooks" only="tensorflow"}
[Reproducibility Audit]{.kicker}

::: {.cols .vc}
::: {.col}
- **Three hashes per output JSON:** (1) Per-cell **prefix code hash** (cells $1\dots k$, trailing spaces stripped), (2) **public framework version** (local build tag stripped), and (3) **transitive `d2l.X` symbol hash**.
- **Surgical invalidation:** Prose edits cause 0 stale cells; editing cell $k$ marks cells $k\dots n$ stale; editing a `#@save` helper marks only downstream notebooks that call it.
:::

::: {.col .fig .big}
@fig:ch39-tools-output-fingerprint@tensorflow
:::
:::
:::

::: {.slide title="How Output Fingerprinting Detects Stale Notebooks" only="mxnet"}
[Reproducibility Audit]{.kicker}

::: {.cols .vc}
::: {.col}
- **Three hashes per output JSON:** (1) Per-cell **prefix code hash** (cells $1\dots k$, trailing spaces stripped), (2) **public framework version** (local build tag stripped), and (3) **transitive `d2l.X` symbol hash**.
- **Surgical invalidation:** Prose edits cause 0 stale cells; editing cell $k$ marks cells $k\dots n$ stale; editing a `#@save` helper marks only downstream notebooks that call it.
:::

::: {.col .fig .big}
@fig:ch39-tools-output-fingerprint@mxnet
:::
:::
:::

::: {.slide title="Why Prefix Staleness Beats Static Dataflow Tracing"}
[Hidden Side-Effect Channels]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why static AST variable dataflow is unsound:** Adding one random draw in cell 3 shifts the PRNG stream for cell 8 without sharing any variable name, and in-place weight/config mutations alter later cells invisibly.
- **Prefix rule + `audit_outputs.py` gate:** Because a fresh kernel must replay cells $1\dots k-1$ anyway, invalidating $k\dots n$ guarantees 100% soundness (hard error on stale text; warning on stale images).
:::

::: {.col .fig .big}
@fig:ch39-tools-prefix-vs-ast-staleness
:::
:::
:::

::: {.slide title="The Git Contribution Workflow: Fork, Branch, Edit, PR"}
[Proposing a Change]{.kicker}

::: {.cols .vc}
::: {.col}
- **Fork & dedicated branch:** Use the GitHub pencil icon for 1-line prose fixes, or clone your fork and create one branch per logical change (`git switch -c fix-topic`).
- **Verify diff & open PR:** Edit `chapter_*/*.md`, check `git diff` to confirm no generated files were hand-edited, push to your fork, and run `gh pr create`.
:::

::: {.col .fig .big}
@fig:ch39-tools-git-pr-flow
:::
:::
:::

::: {.slide title="Linting, Executing, Capturing, and Rendering"}
[Verification Loop]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Lint & Library (`< 100 ms`):** Run `python3 tools/lint_source.py --corpus <file>.md` and `gmake lib` (if `#@save` changed).
- **2. Execute, Capture & Audit:** Run `gmake -B _notebooks/<fw>/<ch>/<f>.executed` and `gmake capture-outputs FILES=<ch>/<f>.md` for every framework tab, then verify with `gmake audit-outputs && gmake html`.
:::

::: {.col .fig .big}
@fig:ch39-tools-four-step-verify
:::
:::
:::

::: {.slide title="Interactive Authoring with the D2L VS Code Extension"}
[Editor Integration]{.kicker}

If you prefer interactive cell execution over editing raw Markdown fences, the repository includes a dedicated VS Code extension in `.vscode-extension/`:

::: {.cols}
::: {.col}
- **Two-way source synchronization:**
  - Opens the generated `.ipynb` notebook of the active `.md` section with the matching framework kernel preselected (`make kernels`).
  - Edits made to code or prose inside the notebook view are **written directly back into the `chapter_*/*.md` source file**, so a generated `.ipynb` never becomes an orphaned edit!
:::

::: {.col}
- **Integrated multi-framework & slide workflow:**
  - Switch live between PyTorch, JAX, TensorFlow, and MXNet tab views.
  - Preview the section's Reveal.js slide deck side-by-side.
  - Trigger full top-to-bottom notebook execution and output capture into `outputs/<fw>/` with a single command.
:::
:::
:::

::: {.slide title="Collaborating Effectively with Coding Agents"}
[AI-Assisted Engineering]{.kicker}

Three architectural properties of `d2l-neu` make coding agents unusually effective for bounded, multi-file engineering tasks:

::: {.cols}
::: {.col}
- **1. Explicit build & style contracts:** `CLAUDE.md`, `docs/build-system.md`, `docs/writing-guide.md`, `docs/code-guide.md`, and `docs/slide-guide.md` document every rule in machine-readable form.
- **2. Executable verification oracle:** An agent can run `lint_source.py`, execute every framework notebook, run `audit_outputs.py`, and inspect tracebacks autonomously.
- **3. Mechanical cross-framework propagation:** Syncing a change across PyTorch and JAX tabs or regenerating SVG figures via `tools/gen_*_figures.py` is fast to automate and easy for a human to review in `git diff`.
:::

::: {.col}
- **Two agent failure modes every human reviewer must guard against:**
  1. **Editing generated files:** Verify `git status` shows changes in `chapter_*/*.md` (plus captured `outputs/`), never ad-hoc patches inside `_notebooks/` or `d2l/*.py`.
  2. **Reward hacking / weakening checks:** When a test or notebook fails, an agent may loosen a threshold, silence an exception, or simplify the teaching example rather than fixing the root cause—always review the pedagogical diff!
:::
:::
:::

::: {.slide title="Debugging Regressions Across Noisy Training Runs"}
[Information-Theoretic Bisection]{.kicker}

::: {.cols .vc}
::: {.col}
$$N_{\text{runs}} \ge \frac{\log_2 64 - H_b(P_e) - P_e \log_2 63}{C}$$

- **Why 1-run `git bisect` fails:** Across $64 = 2^6$ commits, a noisy test ($p = 0.90$ pass on good, $q = 0.30$ on bad) steers all 6 binary steps correctly only $\sim 26\%$ of the time!
- **Bayesian bisection & Fano's bound:** Updating a posterior $P(K = k \mid \text{runs})$ after each test locates the bad commit at $\ge 95\%$ confidence in $\sim 22\text{--}25$ runs, approaching the Shannon capacity limit $C$.
:::

::: {.col .fig}
@fig:ch39-tools-noisy-bisection
:::
:::
:::

::: {.slide title="Concept Check: Tabs, Staleness, and Library Order"}
[Check Your Understanding]{.kicker}

Test your grasp of the D2L build system on these three questions:

1. A source `.md` file has 5 cells: (1) `%%tab pytorch` defining `x`, (2) `%%tab jax` defining `x`, (3) an untagged cell `y = x * 2`, (4) `%%tab pytorch` modifying `x`, and (5) an untagged `print(y.sum())`. If you edit cell (3), which notebooks change and must be re-executed?
2. You upgrade your local PyTorch environment from `torch==2.11.0+cu124` on Linux to `torch==2.11.0` CPU on macOS. Does `tools/audit_outputs.py` mark the committed PyTorch outputs stale? What if you upgrade to `torch==2.12.0`?
3. `Trainer.prepare_batch` is defined with `#@save` in §{sec_oo-design} (CPU-only) and redefined with `#@save` in §{sec_use_gpu} (moving batches to GPU). Which version is in `d2l/torch.py`, and which version runs at the end of the §{sec_oo-design} notebook?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **Both the PyTorch and JAX notebooks!** Because cell (3) is untagged, it belongs to every framework generated from the file (`pytorch` and `jax`), and editing it invalidates the prefix fingerprint of cells (3), (4), and (5) in both frameworks.
2. Switching `2.11.0+cu124` $\to$ `2.11.0` CPU **does not mark outputs stale** because the fingerprint strips local build tags (`+cu124`) so CPU laptops can build the site without false alarms. Upgrading to **`2.12.0` marks all PyTorch outputs stale**.
3. `d2l/torch.py` contains the **§{sec_use_gpu} GPU-aware definition** (last definition in book order wins). When §{sec_oo-design} starts, `d2l.Trainer.prepare_batch` holds the §{sec_use_gpu} version until §{sec_oo-design}'s own `@d2l.add_to_class` cell executes and overwrites it in memory with the CPU version for the remainder of §{sec_oo-design}.
:::
:::

::: {.slide title="Summary: Contributing to Dive into Deep Learning"}
[Key Takeaways]{.kicker}

| Build Mechanism | Design Principle | Contributor Action |
| :--- | :--- | :--- |
| **Single source (`chapter_*/*.md`)** | Generates `.qmd`, `.ipynb`, `_slides/`, and `d2l/*.py` | Edit only the `.md` source (or via the VS Code sync extension); never hand-edit generated files |
| **Multi-framework tabs** | `%%tab <fw>` scopes code; untagged cells run in all frameworks | When editing a shared cell, re-execute and capture outputs for *every* framework tab in that file |
| **`#@save` library extraction** | `make lib` collects `#@save` blocks in book order (last wins) | Run `make lib` before re-executing notebooks whenever you edit a `#@save` block |
| **Prefix output fingerprinting** | Hashes cell prefix, `major.minor.patch` fw version & `d2l.X` closure | Run `make capture-outputs` and verify `make audit-outputs` reports zero stale text cells |
| **Source linter (`lint_source.py`)** | Fast ($<100\text{ ms}$) check on div/tab balance, cell IDs & xrefs | Run `python3 tools/lint_source.py --corpus <file>.md` before every commit |
| **Git & coding-agent review** | Small branch-per-change PRs with executable verification | Review `git diff` to ensure agents edited sources rather than generated files or weakened checks |
:::
