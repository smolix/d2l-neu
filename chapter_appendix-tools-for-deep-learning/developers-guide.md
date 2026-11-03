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
[repository page](https://github.com/smolix/d2l-neu), then

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

1. **Stable cell IDs.** The audit of exercise
   :exref:`ex_developers-guide-stale-outputs` matches the stored outputs of
   a notebook to its cells by ID. A stored ID that no cell of the notebook
   carries any more is an error that stops `make html`, while a cell
   without a stored output only causes a warning. Slide sources cite cells
   by ID as well; the slides of :numref:`sec_resnet`, for example, cite
   `@resnet-residual-blocks-2`.
    1. In :numref:`sec_use_gpu`, the cell that adds `__init__` and
       `prepare_batch` to `d2l.Trainer` for PyTorch has the ID
       `gpus-devices-memory-the-trainer-now-with-devices-1`, although it
       stands under the heading "Device-Aware Trainer". Why does the ID
       name a different heading?
    1. Suppose that the build derived every ID from the current heading. An
       author renames a heading and changes no code. What does the audit
       report, and what must the author run before `make html` passes
       again?
    1. Suppose instead that the build numbered the cells by position, and
       an author inserts a code cell. Does the audit notice that the stored
       outputs now sit under the IDs of other cells? Which other users of
       the IDs receive no warning, and what do they show afterward?
   :exlabel:`ex_developers-guide-cell-ids`

1. [code] **Duplicate definitions.** `make lib` collects the `#@save`
   blocks of all sections in book order. When several blocks define the
   same function or method, the library keeps only one of them. The method
   `Trainer.prepare_batch` is defined in the untagged cell
   `#linear-regression-scratch-training-1` of :numref:`sec_linear_scratch`,
   which every framework runs, and again, with device support, in the
   framework variants of the cell
   `#gpus-devices-memory-the-trainer-now-with-devices-1`
   of :numref:`sec_use_gpu`.
    1. Add a comment line to the body of the untagged definition and of
       the PyTorch variant of the device-aware one, run `make lib`, and
       search `d2l/torch.py` for both comments. Which comment arrives,
       and which rule does this reveal?
    1. Which definition does `d2l.Trainer.prepare_batch` hold when the
       PyTorch notebook of :numref:`sec_linear_scratch` starts, and which
       one after its cell `#linear-regression-scratch-training-1` has run?
    1. Why does the build keep the last definition rather than the first?
    1. A contributor changes the device-aware PyTorch `prepare_batch` so
       that it also converts every batch to 16-bit floats. The PyTorch
       notebooks of which sections can print different results: those
       before :numref:`sec_use_gpu` in book order, those after it, or both?
   :exlabel:`ex_developers-guide-duplicate-definitions`

1. **Agent patches.** A contributor changes the format string of a `print`
   call in the PyTorch variant of the last code cell of a section, and
   `make html` stops because the committed text output of that cell is
   stale. Asked to make the build pass, a coding agent proposes five
   patches: (i) set the cell's `code_fingerprint` in
   `outputs/pytorch/<chapter>/<stem>.json` to the fingerprint of the new
   code; (ii) delete the cell's entry from that file, which the audit
   reports only as a warning about a missing output; (iii) edit the
   generated `.qmd` file of the section so that the page shows the new
   output; (iv) re-execute the notebook with
   `make -B _notebooks/pytorch/<chapter>/<stem>.executed` and then run
   `make capture-outputs FILES=<chapter>/<stem>.md`; (v) change the
   Makefile so that a stale text output only prints a warning.
    1. For each patch, does it remove the cause, hide it, or leave it in
       place? What does a reader of the page see after each?
    1. After which patches does `make html` pass? What would you look for
       in the pull request to confirm that the committed output came from
       the committed code?
    1. Write the task statement that you would have given the agent: the
       change it may make, the files it must not edit, and the checks
       whose results it must report.
   :exlabel:`ex_developers-guide-agent-patches`

1. **Execution stamps.** ● Make and the notebook scheduler
   `tools/notebook_scheduler.py` re-execute a notebook when its stamp
   `_notebooks/<fw>/<chapter>/<stem>.executed` is older than the generated
   `.ipynb` or than one of the library block files that its `.d` file
   lists. Besides `d2l/torch.py`, `make lib` writes one block file per
   `#@save` symbol to `d2l/_blocks/<fw>/`, and the `.d` file of a section
   lists the block files of every symbol `d2l.X` that its code mentions
   and of the symbols that those blocks use in turn.
   `make notebooks-pytorch` regenerates every PyTorch notebook and writes
   each `.ipynb` anew. When the code cells of a regenerated notebook are
   unchanged, `tools/gen_notebooks.py` then sets the modification times of
   the `.ipynb` and of its stamp to the current time.
    1. Why does the script advance the stamp, and not only the `.ipynb`?
       What would a prose edit in one section cost otherwise?
    1. An author edits a `#@save` method in one section and runs
       `make run-notebooks-pytorch`, which rebuilds the library,
       regenerates the notebooks, and then executes those that the
       modification times call out of date. Notebook B lists the edited
       block in its `.d` file, but its own source is unchanged. Order the
       modification times of the block file, of the `.ipynb` of B, and of
       the stamp of B. Is B executed? What does the audit of exercise
       :exref:`ex_developers-guide-stale-outputs` report afterward?
    1. Propose a stamp that keeps prose edits free of re-execution and
       still executes B. What must it record that a modification time
       cannot?
    1. The repository guards against the failure of (b) twice:
       `make refresh-stale` deletes the stamps of the notebooks that the
       audit reports stale before it executes them, and
       `make capture-outputs` refuses outputs whose fingerprints, recorded
       when the notebook ran, differ from the current ones. Which route to
       committing stale outputs does each guard close?
   :exlabel:`ex_developers-guide-execution-stamps`

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
