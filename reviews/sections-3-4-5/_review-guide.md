# Review guide — D2L sections 3, 4, 5 (Linear Regression, Linear Classification, MLPs)

*Internal guide for reviewer agents. Read this fully before reviewing your file.*

## Mission

We are bringing the **Linear Neural Networks for Regression** (ch. 3),
**Linear Neural Networks for Classification** (ch. 4), and **Multilayer
Perceptrons** (ch. 5) chapters of this book to a standard where they can serve
as the assigned textbook in the very best programs — Stanford, MIT, CMU, UC
Berkeley. The bar is not "good enough"; it is **best-in-class, and ideally
better than any existing treatment of this material**. Your job is to review
**one file** against that bar and produce a precise, actionable report.

The **gold standard already in this repo** is the *Preliminaries* chapter
(`chapter_preliminaries/`, especially `calculus.md`, `linear-algebra.md`,
`probability.md`) and the *Mathematics for Deep Learning* appendix
(`chapter_mdl-*`). Open one or two of these to calibrate. What makes them
excellent — and what your file should meet or exceed:

- **Intuition first, then formalism.** A concrete hook or picture opens the
  topic (e.g. Archimedes inscribing polygons to motivate limits) before any
  definition. Every formal object is given a geometric or operational meaning.
- **Tight logical spine.** 3–5 top-level `##` sections, each with `###`
  subsections — never a long flat run of headings. Each section earns its place.
- **Prose that breathes.** No walls of text, no walls of math, no walls of
  code. Display equations are motivated and immediately interpreted.
- **Figures that carry ideas**, pre-generated and clean, referenced by
  `:numref:`. Schematic figures are *drawn by committed generators*, never by
  matplotlib code sitting in a teaching notebook.
- **Code that teaches.** Every code cell *computes or demonstrates* something
  the prose is discussing. Compact and elegant.
- **A "Discussion"/"Summary" that connects forward and outward**, plus
  cross-references (`:numref:`) and citations (`:citet:`) to deeper treatments.
- **Exercises that build** — from mechanical checks to genuinely thought-
  provoking extensions.

> The user's explicit instruction: **exceed** this gold standard if you can.

## Scope discipline (read this before recommending anything to "add")

**These three chapters are foundational sections inside a much larger book.** The
book already has dedicated chapters for the advanced material these topics touch.
Your recommendations must **stay within each section's role** and the chapter's
place in the book's arc. The default way to honour "currency" and "exceed the
bar" is to **sharpen the core teaching and forward-point** (`:numref:` /
`:citet:`) to the chapter that owns a topic — **not** to import that chapter's
content here. A best-in-class foundations chapter is *lean and exact*, not a
survey. Before you write "add X," check the scope map below: if X lives
downstream, recommend a one-line mention + forward pointer, not a new subsection.

Concretely:
- **Do** propose: clearer derivations, better/again figures, tighter prose,
  fixing dated/incorrect claims, a missing *in-scope* idea, a sharper exercise,
  removing a dead framework, a one-sentence pointer to where a topic is developed.
- **Don't** propose: a full treatment of optimizers, normalization layers,
  transformer-era activations, deep learning-theory machinery, or modern
  architectures *inside these sections* — those have their own chapters (below).
  Flag them as forward-pointers at most.

## Book scope map — what later chapters already own (forward-point, don't import)

| Topic that may tempt you to "add" it | Where it lives in the book | What's OK here |
|---|---|---|
| SGD variants, momentum, Adam/**AdamW**, LR schedules, convexity | **Optimization** part (`chapter_optimization/*`) and `chapter_mdl-optimization` | Define minibatch SGD; mention weight-decay≠L2-for-Adam in **one line** + `:numref:` to Optimization. |
| Entropy, cross-entropy, KL divergence (information theory) | `chapter_mdl-information-theory` (already cross-referenced by the ch.4 cover) | State the cross-entropy/MLE link; forward-point for the information-theoretic depth. |
| Batch norm / Layer norm / RMSNorm | `chapter_convolutional-modern/batch-norm.md` (+ later) | In init/stability, a pointer is fine; don't teach normalization here. |
| Building blocks: custom layers, parameter mgmt, init API, GPU | **Builders' Guide** (`chapter_builders-guide/*`) | MLP impl uses basics; don't pre-empt the module/parameter machinery. |
| Linear algebra, calculus, probability depth | **Preliminaries** (ch.2) + **Math for DL** appendix (`chapter_mdl-*`) | Use results; forward-point for proofs/depth. |
| CNNs, RNNs, attention, transformers, modern activations (GELU/SwiGLU), modern architectures (MLP-Mixer, KAN) | Their own parts later in the book | At most a forward-pointer/footnote; **not** in-scope content. |
| Deep generalization: double descent, implicit bias, why classical bounds fail | **This is in scope** — it is the explicit subject of `generalization-deep.md` (ch.5) | Develop it *here*, honestly and at foundational depth. |
| VC/Rademacher/PAC learning theory | In scope but *light* — `generalization-classification.md` (ch.4) | Introductory-but-rigorous; forward-point external refs for full proofs. |

When in doubt about whether something is in scope, say so explicitly in your
report and recommend the conservative (forward-pointer) option.

## Source-of-truth rules (critical)

- The **`.md` file is the source of truth**. Review it. Propose all changes at
  the `.md` level.
- **Never** propose edits to `.qmd` files — they are generated and overwritten.
- You are **writing a report**, not editing the book. Do not modify any chapter
  files. Only write your report to the path you are given.

## Inputs available to you

1. **The source `.md`** — your primary object of review. Read it in full.
2. **The rendered page** on the served site:
   `https://d2l.smola.org/<chapter_dir>/<file-stem>.html` (e.g.
   `https://d2l.smola.org/chapter_linear-regression/linear-regression.html`).
   Fetch it to see how figures, equations, and code render and to read the
   section numbering. (WebFetch returns a text description, not pixels — use it
   for structure/figure presence, not fine visual critique.)
3. **Executed outputs (committed store)** — the notebooks have been run; their
   outputs live in
   `outputs/<fw>/<chapter_dir>/<file-stem>.json` for each framework
   `fw ∈ {pytorch, jax, tensorflow, mxnet}`. The manifest's `cells` dict is
   keyed by cell ID; each cell has `kind` (`text`/`asset`) and `outputs`. Text
   outputs (`{"type":"stream","text":...}` and display text) show you the
   **actual printed numbers, losses, accuracies, warnings**. `asset` outputs name
   the generated plot SVG under `outputs/<fw>/<chapter_dir>/<file-stem>/*.svg`.
   **Use the text outputs** to judge whether each example produces a clear,
   sensible, teaching-quality result (does training converge? are the numbers
   reasonable? are there stray warnings?).
4. **Research digest** for your chapter's topic area — *only if a path is given
   in your task* (one exists for the MLP chapter at
   `reviews/sections-3-4-5/_research/mlp.md`; regression/classification reviewers
   rely on their own expertise + the scope map). Where a digest exists, treat its
   "add" suggestions as **candidates to filter through the scope map** — several
   modern items it lists (e.g. SwiGLU, KANs, MLP-Mixer) are forward-pointers at
   most, not in-scope additions. Use it mainly for pedagogy and resource curation.
5. **Reference exemplars** — `chapter_preliminaries/calculus.md` and the
   `chapter_mdl-*` files. Skim one to calibrate the bar.

## Code-tab structure (so you can read the code correctly)

Code cells look like ```` ```{.python .input #cell-id} ```` optionally followed
by `%%tab <fw>` (or `%%tab mxnet, pytorch`). An **untagged** Python cell applies
to **all four frameworks**. A cell ID repeated across `%%tab` blocks means those
are the per-framework variants of the *same* logical cell. The
`tab.interact_select(...)` line at the very top is boilerplate. `#@save` marks
code saved into the `d2l` library for reuse. Trailing `<!-- slides -->` sections
are **out of scope — ignore slides entirely.**

## The three review dimensions

### 1. Coverage — what to add, what to remove, what to reorder

- Is anything **missing** that a top-tier treatment of this topic must have?
  (Derivations, a probabilistic/geometric viewpoint, a key caveat, a modern
  connection, a missing baseline.) Ground your "add" suggestions in what the
  research digest says CS229 / ESL-ISL / Bishop / Murphy / Goodfellow / CMU
  11-785 / etc. do.
- Is anything **redundant, dated, or off-topic** that should be cut or tightened?
  (Overlap with sibling files, dwelling on trivia, dead ends.)
- Is the **ordering** right within the file, and relative to sibling files in
  the chapter? Flag content that belongs in a different file.
- Is the **depth** calibrated for a top program — neither hand-wavy nor
  drowning in formalism?

#### 1a. Currency — what has changed since this was written (~2021)

**These notebooks were written around 2021.** The three topics are classics, but
the field has moved, and a best-in-class edition must reflect that. Treat this as
a first-class part of coverage. Explicitly hunt for:

- **Dated tooling/APIs and dead frameworks.** Most importantly, **Apache MXNet
  was retired/archived by the ASF (2023)** — flag every place the text presents
  MXNet as a live, co-equal option, and say whether the MXNet tab should be
  de-emphasized or dropped. Flag dated PyTorch (pre-2.x idioms), old Flax
  `linen` patterns vs. modern JAX, and pre-Keras-3 TensorFlow.
- **Stale claims and framings.** Statements that were true in 2021 but are now
  incomplete or wrong, and "state-of-the-art"/"recent" references that have aged.
- **Modern results that belong here**, even as a paragraph or forward pointer.
  Examples by topic (use judgement; the digest has more):
  *regression/generalization* — benign overfitting & ridgeless least squares
  (Bartlett et al. 2020; Hastie et al. 2022), double descent (Belkin 2019;
  Nakkiran 2021), scaling laws, **AdamW / decoupled weight decay** (L2 ≠ weight
  decay for adaptive optimizers — Loshchilov & Hutter 2019);
  *classification* — modern calibration (Minderer et al. 2021), linear probing
  of pretrained features, Rademacher complexity as the modern complement to VC;
  *MLPs* — He/Kaiming init for ReLU, **GELU/SwiGLU** and gated activations,
  **LayerNorm/RMSNorm** vs BatchNorm, grokking (Power et al. 2022), the MLP
  renaissance (MLP-Mixer 2021), KANs (2024), and the well-established result that
  **gradient-boosted trees still beat deep nets on tabular data** (Grinsztajn et
  al. 2022) — directly relevant to the Kaggle house-price capstone.

For each, say *what* is dated/missing, *why it matters in 2026*, and *how* to fix
it — but **apply the scope map**: prefer a corrected sentence, a citation, or a
forward pointer over a new subsection, unless the modern result is genuinely
in-scope for this section (e.g. double descent belongs in `generalization-deep`).
Be surgical — these are foundations, not a survey; the goal is a treatment that
reads as written in 2026, not retrofitted, **and stays in its lane.**

### 2. Teaching quality — figures, structure, clarity, concision

- **Structure**: Does it follow the 3–5 `##` / nested `###` shape? Is the spine
  logical? Propose a better outline if not.
- **Figures**: Inventory every figure (`![...](...)` + `:numref:`). For each:
  is it pulling its weight, clear, well-captioned? Where is a figure **missing**
  that would unlock an idea (name it and say what it should depict)? **Flag any
  figure-drawing matplotlib code living in a teaching cell** — illustrative/
  schematic figures must be pre-generated (`img/<id>.svg` via a committed
  generator), not drawn inline. (Data plots that *teach a computed result*, e.g.
  a loss curve from `d2l.plot`, are fine.)
- **Prose**: Walls of text/math? Passages that are confusing, hand-wavy, or
  needlessly verbose? Is each equation motivated and then interpreted? Quote the
  worst offenders by line number and propose tighter rewrites for the highest-
  value cases.
- **Exercises**: Do they build from mechanical to thought-provoking? Are they
  worthy of a top course's problem set? What should be added/cut?

### 3. Code & examples quality — all four frameworks

- **Does every code cell teach?** Flag walls of code, boilerplate that obscures
  the idea, and any cell whose only purpose is to draw an illustrative figure.
- **Per-framework review.** Read the `pytorch`, `jax`, `tensorflow`, and `mxnet`
  variants of each multi-tab cell. For each framework assess:
  - **Idiomatic & modern?** Does it use current, recommended APIs and style for
    that framework, or dated/deprecated patterns? (Note: MXNet is legacy/least-
    maintained upstream — still flag issues, but weight PyTorch and JAX, the
    primary frameworks, most heavily.)
  - **Unnecessary divergence.** Do the four implementations differ more than the
    frameworks actually require? Gratuitous divergence is a teaching cost.
  - **Correctness / best practice.** Numerical pitfalls (e.g. `log` of softmax
    vs `log_softmax`), device/dtype handling, seeding/reproducibility, in-place
    ops, vectorization, off-by-one, silent broadcasting bugs.
  - **Actual outputs.** Cross-check the executed outputs (manifests) — does the
    cell produce a sensible, clear result? Any warnings/errors? Stale numbers?
- **d2l conventions.** One per-framework **imports cell** near the top (no
  re-imports later); `#@save` hygiene; stable `#cell-id`s; prefer `d2l` helpers
  (`d2l.plot`, `d2l.plt`, …) over raw matplotlib.

## Report format (write exactly this structure)

```
# Review — <chapter_dir>/<file>.md  (§<num> "<Title>")

**Role in the chapter:** <1–2 sentences: what job this file does.>
**Verdict:** <2–4 sentences. Where it stands vs the best-textbook bar; the
single highest-value change.>
**Grade:** <A/B/C… on the "assignable at a top program as-is" scale, with a one-
line justification.>

**Top priorities (ranked):**
1. [P0/P1] …
2. …
3. …

## 1. Coverage
### Add
### Remove / trim
### Reorder / restructure

## 2. Teaching quality
### Structure & flow
### Figures
### Prose & clarity
### Exercises

## 3. Code & examples
### Does the code teach?
### PyTorch
### JAX
### TensorFlow
### MXNet
### Cross-framework consistency & d2l conventions

## 4. Implementation spec (the executable part — downstream agents act on THIS)

Every P0/P1 finding in sections 1–3 MUST appear here as a self-contained task an
implementation agent can execute given only your report + the file. Use stable
IDs `<FILE-KEY>-N` (e.g. `WD-1`, `MLP-3`). One block per change:

### <ID> — <short title>  ·  [P0|P1|P2] · [S|M|L] · [mechanical|authored|judgment]
- **Type:** coverage | teaching | figure | code | currency | prose
- **Where:** `<file path>` — verbatim anchor (quote the exact text or cell-id to locate; line numbers are hints only, they drift).
- **Change:** the precise edit. `mechanical` → give exact `old → new` strings the agent can apply blindly. `authored` → paste the full drafted prose / figure spec (what it depicts) / code. `judgment` → a precise instruction + constraints, agent decides specifics.
- **Touches:** other files / build steps (e.g. `tools/gen_mdl_<ch>_figures.py` + `make figures` + `img/<id>.svg`), or "none".
- **Done when:** a concrete acceptance check (e.g. "all four concise ‖w‖² outputs ∈ [0.0015, 0.0021]"; "renders in HTML and PDF, `make html` clean"; "exercise has a worked solution in the slot").
- **Depends on:** other change IDs / cross-file prerequisites, or "none".

## 5. Keep — what is already excellent (do not lose this)
- …
```

**Severity scale:** `P0` = blocks "assignable at a top program" (errors, serious
gaps, confusing core explanations); `P1` = important quality lift; `P2` = polish.
**Effort:** S (≤30 min), M (a few hours), L (substantial / new figures or
sections).
**Tag:** `mechanical` = exact old→new given, an agent applies it blindly (typos,
LaTeX, citation keys, renames, a one-line pointer); `authored` = you supply the
full drafted prose/figure-spec/code and an agent integrates it; `judgment` = you
give a precise instruction and an agent makes the specific calls.

> **Why this rigor:** these reports are the input to a downstream phase where
> **agents edit the actual book pages from your spec**. A competent agent given
> *only* your Section 4 entry and the file must be able to make the change
> correctly, and a verifier must be able to confirm it from "Done when". Vague or
> unanchored entries cannot be executed — they will be dropped. Precision here is
> the whole point of the report.

## Standards for your report

- **Be specific and grounded.** Cite line numbers, cell IDs, and exact quotes.
  Vague advice ("add more intuition") is useless; show the sentence and propose
  the replacement.
- **Propose, don't just diagnose.** For the highest-value issues, draft the
  actual rewrite, the figure to draw (what it depicts), or the new exercise.
- **Reference the field.** When you say "add X," point to where a leading
  course/text does it (use the digest).
- **Be honest about what's already great** — section 5 of your report protects
  excellent material from well-meaning churn.

## What to return to the orchestrator

After writing your report file, return a **compact summary (≤220 words)**:
the grade; the count of P0/P1/P2 changes and the IDs + one-line titles of all
**P0/P1** changes (so the orchestrator can build a dispatch manifest); any
cross-file issue the overview must reconcile (overlap/ordering with sibling
files, or a book-wide decision like the MXNet tab); and any factual/code error
you found. This summary feeds the cross-cutting overview.
