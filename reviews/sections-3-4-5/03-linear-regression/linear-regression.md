# Review — chapter_linear-regression/linear-regression.md  (§3.1 "Linear Regression")

**Role in the chapter:** The conceptual anchor of the whole regression chapter. It introduces the
linear model, squared loss, the analytic (normal-equations) solution, minibatch SGD, vectorization,
the Gaussian-noise → MLE justification of squared loss, and the "single-layer neural network"
framing. It is *math-and-concepts only* — the implementations live in `synthetic-regression-data.md`,
`oo-design.md`, `linear-regression-scratch.md`, and `linear-regression-concise.md`. It is referenced
by `weight-decay.md` (the loss recap at its line 128, and the Bayesian-prior exercise at its line 608).

**Verdict:** This is a strong, well-paced foundations section that already sits close to the
best-textbook bar: the prose breathes, the MLE derivation is clean and correct, the scope discipline
is good (it correctly defers SGD variants, generalization, and the implementation to their owners).
It falls short of the gold-standard *Preliminaries* chapters in two specific ways: (1) it opens with a
list of application bullets rather than a concrete hook + picture (the calculus/Archimedes standard),
and (2) the **vectorization timing cell makes a quantitative claim ("order of magnitude", and "3 orders
of magnitude" in the slides) that the committed outputs contradict for the primary framework** — PyTorch
measures only ~58× on this size, while the prose/slide say 10× / 1000×. The single highest-value change
is to fix that timing narrative so it teaches the real lesson (Python-loop overhead, size-dependent)
instead of an unreproducible number. The biological-neuron figure is a dated hand-traced Illustrator
asset and should be reconsidered. A *lean* one-paragraph geometry-of-least-squares forward pointer is a
strong, in-scope enrichment (the picture already exists in the Math-for-DL appendix — point to it, do
not redraw it here).

**Grade:** **B+.** Assignable at a top program with light edits; the timing-claim mismatch and the
list-style opening are the gap between this and an unqualified "A / best-in-class."

**Top priorities (ranked):**
1. [P0] **LR-1** — Fix the vectorization timing narrative: the "order of magnitude"/"3 orders of
   magnitude" claims do not match the committed PyTorch outputs (~58×). Make the lesson size- and
   overhead-based, not a hard multiplier.
2. [P1] **LR-2** — Replace the application-list opening with a concrete hook (house-price scatter +
   one-line picture) to meet the intuition-first bar of the Preliminaries chapters.
3. [P1] **LR-3** — Add a *lean* one-paragraph "geometry of least squares" pointer (projection onto the
   column space) forward-linking the existing `fig_mdl-la-hyperplane` / projection result; do **not**
   draw a new figure.
4. [P1] **LR-4** — Decide the fate of the dated biological-neuron figure `fig_Neuron` (hand-traced
   Illustrator SVG) and its `### Biology` subsection: keep-but-reframe, or trim.
5. [P1] **LR-5** — Currency/MXNet: de-emphasize MXNet (archived by the ASF, 2023) in the chapter's
   framework framing (book-wide decision; flagged here, owned by the overview).
6. [P2] **LR-6** — Tighten two prose passages (the `inference`/`prediction` aside; the SGD redundancy
   paragraph) and add one forward-pointing sentence on adaptive optimizers + AdamW.

---

## 1. Coverage

### Add

- **Geometry of least squares (one paragraph, forward-pointing — LR-3).** The watch-point asks whether a
  projection view is a good in-scope enrichment. It is — but the picture and proof *already exist* in the
  Math-for-DL appendix (`chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md`:
  `### Projection and Orthogonality`, with `fig_mdl-la-projection`, `fig_mdl-la-hyperplane`, and the
  orthogonal-projection proposition `eq_mdl-projection`). The lean, scope-respecting move is to add **one
  paragraph** after the normal-equations block (after line 276) interpreting $\mathbf{X}\mathbf{w}^*$ as the
  orthogonal projection of $\mathbf{y}$ onto the column space $\operatorname{col}(\mathbf{X})$, with the
  residual $\mathbf{y}-\mathbf{X}\mathbf{w}^*$ orthogonal to every column — which is *exactly* the normal
  equations $\mathbf{X}^\top(\mathbf{X}\mathbf{w}-\mathbf{y})=0$ restated geometrically. This is the view in
  Strang/VMLS (Boyd–Vandenberghe, already in the chapter's `index.md` reading list) and ESL §3.2. Keep it to
  one paragraph; forward-point, do **not** add a new SVG. Drafted in §4 (LR-3).

- **One sentence on adaptive optimizers (forward pointer — part of LR-6).** §3.1 introduces minibatch SGD
  as *the* recipe. Per the scope map, optimizer depth is OUT (it belongs to the Optimization part), but a
  best-in-2026 reading deserves a single forward-pointing clause acknowledging that adaptive methods
  (Adam/AdamW) are what is used in practice and that "weight decay ≠ $\ell_2$ for adaptive optimizers"
  (Loshchilov–Hutter 2019) — with a `:numref:` to the Optimization part. One clause + citation, not a
  subsection. Note the chapter `index.md` already cites Hoerl–Kennard for ridge; the AdamW pointer belongs
  in `weight-decay.md` more than here, so keep it to a *single* clause here and let `weight-decay.md` own it.

- **Nothing else.** Resist adding benign-overfitting/double-descent/scaling-laws here — those are owned by
  `generalization.md` and (in the broader book) `generalization-deep.md`. The existing
  generalization paragraph (lines 384–399) + the "We return to these topics" pointer is the correct,
  scope-respecting amount. Do not expand it.

### Remove / trim

- **The `inference`/`prediction` terminology aside (lines 407–416)** is a ~10-line digression about
  statisticians vs. deep-learning practitioners that interrupts the spine. It is *correct and mildly
  interesting* but over-long for its value at this point in the book. Trim to 2–3 sentences (LR-6).
- **The SGD motivation paragraph (lines 302–324)** is slightly repetitive: it makes the "full-batch is
  slow" and "single-example is cache-inefficient" points at length, then the matrix-vector vs.
  vector-vector point overlaps with the *Vectorization for Speed* section that follows. Tighten by one
  sentence and let the vectorization section carry the cache/throughput argument (LR-6, low priority).
- **`### Biology` subsection + `fig_Neuron` (lines 685–733).** See LR-4 — this is the watch-point's "dated
  cartoon" question. Recommendation below.

### Reorder / restructure

- The within-file ordering is sound and matches the rendered numbering (3.1.1 Basics → 3.1.2 Vectorization
  → 3.1.3 Normal Distribution & Squared Loss → 3.1.4 NN framing → Summary). **One structural nit:** the
  *Vectorization for Speed* section (3.1.2) sits *between* the SGD subsection and the probabilistic
  motivation, breaking the conceptual arc (model → loss → optimization → *why this loss*). It reads as an
  interlude. This is defensible (it motivates minibatching, which was just introduced) and I would **not**
  force a move — but flag it so the overview can decide whether all four regression files want vectorization
  consolidated. Leave as-is unless the overview says otherwise.

---

## 2. Teaching quality

### Structure & flow

Good 5-section spine with nested subsections; matches the gold-standard shape (3–5 `##`, nested `###`).
The one gap vs. the Preliminaries bar is the **opening**: `calculus.md` opens with Archimedes inscribing
polygons + `fig_circle_area` *before* any definition; `linear-regression.md` opens with four lines of
application bullets ("predicting prices… length of stay… demand…") and then a verbal house-price setup with
no picture. The house-price example is a fine hook — it just needs to be *shown*, not only told. See LR-2.

### Figures

Three figures, all committed and rendering (confirmed on the served page: Fig. 3.1.1–3.1.3):

| Figure | Label | Source asset | Verdict |
|---|---|---|---|
| 3.1.1 Fitting a linear regression model to 1-D data | `fig_fit_linreg` | `img/fit-linreg.svg` (16 KB, clean schematic) | **Keep.** Pulls its weight; clean house style. |
| 3.1.2 Linear regression as a single-layer NN | `fig_single_neuron` | `img/singleneuron.svg` (23 KB, clean schematic) | **Keep.** Earns its place for the NN framing. |
| 3.1.3 The real neuron | `fig_Neuron` | `img/neuron.svg` (12 KB, **Adobe Illustrator export**, hand-traced anatomical drawing) | **Reconsider (LR-4).** |

- **`fig_Neuron` is the dated asset.** Its SVG header is `Generator: Adobe Illustrator 12.0.0` — it is a
  hand-traced anatomical cartoon, stylistically inconsistent with the chapter's two clean schematic SVGs
  and with the book's "one figure style per chapter / generator-drawn schematics" convention. The CLAUDE.md
  house rule is that illustrative/schematic figures are generator-produced `img/mdl-<ch>-<id>.svg`. This is
  neither. Three honest options, in order of my preference (LR-4):
  1. **Keep the subsection, redraw the figure in house style** as a clean schematic neuron (dendrites →
     soma/Σ → axon) that visually *rhymes* with `fig_single_neuron` so the McCulloch–Pitts analogy is
     immediate. This is the best teaching outcome but is effort-L (a new generator figure).
  2. **Keep both figure and subsection as-is.** Defensible: the biological-neuron history is genuinely part
     of why these are called "neural" networks, the caption is properly sourced (SEER/NCI), and the
     `Russell.Norvig.2016` "airplanes vs. birds" framing (lines 726–733) is excellent and self-aware about
     not over-claiming the biology. The cost is a one-off non-house-style figure.
  3. **Trim the `### Biology` subsection to a short paragraph and drop `fig_Neuron`.** Leanest, but loses a
     nice piece of intellectual history.
  My recommendation: **option 2 short-term (it is not wrong, just stylistically off), with option 1 as the
  aspirational fix** if the chapter is being brought fully to the generator-figure standard. This is a
  *judgment* call for the overview/author — I do not think the figure is so dated as to be a P0.
- **No figure-drawing matplotlib in any teaching cell.** Confirmed: the only plotting cell
  (`linear-regression-the-normal-distribution-and-squared-loss-2`) plots *computed* normal densities, which
  is a legitimate teaching plot, not a schematic. Good — compliant with the house rule.
- **Missing figure (optional, ties to LR-2):** a small house-price scatter with the fitted line would make
  the opening concrete (see LR-2). `fig_fit_linreg` already does the abstract "fit to 1-D data" job, so this
  is optional, not required.

### Prose & clarity

Generally excellent — this reads like a textbook, not a tutorial. Specific spots:

- **Lines 212–215 (the MSE-1/2 note):** *Keep — this is a genuinely useful, correct, modern note* that the
  built-in `nn.MSELoss` / `MeanSquaredError` omit the $\tfrac12$ and the gradient is therefore 2× larger, so
  halve the LR if swapping. Verified consistent with the sibling files: `linear-regression-concise.md` lines
  220/226 explicitly note `MSELoss` omits the $1/2$ factor, and `linear-regression-scratch.md` uses
  `**2 / 2`. This is exactly the kind of practitioner caveat that separates best-in-class from generic.
- **Lines 407–416 (`inference` aside):** over-long; trim (LR-6).
- **Lines 484–488 (vectorization conclusion):** "Vectorizing code often yields order-of-magnitude speedups"
  understates and the slide's "3 orders of magnitude" overstates — see LR-1; the prose lesson should be
  *Python-interpreter overhead*, which is real and reproducible, rather than a single multiplier.
- **Line 119** ("Strictly speaking, … *affine transformation* …"): good, precise, keep.

### Exercises

The exercise set is genuinely strong and already meets a top-course bar — it builds from mechanical
(Ex. 1: minimize $\sum(x_i-b)^2$) to conceptual (Ex. 2: affine = linear on $(\mathbf{x},1)$; Ex. 6: why two
stacked linear layers collapse) to genuinely thought-provoking applied modeling (Ex. 7: log-prices,
Black–Scholes; Ex. 8: Poisson regression for counts). Ex. 5 (exponential-noise → $\ell_1$ loss, and "what
goes wrong near the stationary point" — the non-smoothness of $|\cdot|$) is exactly the kind of question a
CS229/CMU problem set would ask. **Keep all of them.** Two small lifts (LR-6/LR-7):
- Ex. 4 (rank-deficient $\mathbf{X}^\top\mathbf{X}$) is the natural place to *forward-point to weight decay /
  ridge*: add a final sub-part "(e) Relate the fix in (b) to the $\ell_2$ penalty of
  :numref:`sec_weight_decay`." This stitches §3.1 to `weight-decay.md` and previews ridge as the cure for
  non-invertibility — a connection ESL makes explicitly.
- Optional new exercise: derive the closed form $b^* = \bar y - \mathbf{w}^{*\top}\bar{\mathbf{x}}$ (the
  fitted line passes through the centroid) — a clean, satisfying result that connects to Ex. 1.

---

## 3. Code & examples

Code is light here (this is a concepts file): an imports cell, the three-cell vectorization timing demo,
and the normal-density `normal()` + plot. No model is trained in this file. Overall the code is clean.

### Does the code teach?

- **Vectorization demo (cells `…-1/-2/-3`): teaches, but the *narration* is broken (LR-1).** The demo
  itself is good pedagogy — Python loop vs. one `+`. But the committed outputs show the headline claim is
  not reproducible across frameworks at $n=1000$:

  | fw | loop (cell -2) | `+` (cell -3) | ratio |
  |---|---|---|---|
  | pytorch | `0.01228 sec` | `0.00021 sec` | **~58×** |
  | jax | `1.01515 sec` | `0.06525 sec` | ~16× |
  | tensorflow | `0.76635 sec` | `0.00037 sec` | ~2000× |
  | mxnet | `0.59433 sec` | `0.00129 sec` | ~460× |

  The prose says "**order of magnitude**" (≈10×, understated for TF/MXNet) and the **slide** (line 897) says
  "**Roughly 3 orders of magnitude faster**" (≈1000×, true for TF only, ~17× off for PyTorch — the *primary*
  framework). PyTorch's per-element Python-level tensor indexing is unusually slow, so the loop is *cheaper*
  than TF's `Variable.assign`, compressing the ratio. The fix is to (a) teach the real, reproducible lesson —
  **Python interpreter / per-op dispatch overhead dominates the loop; the vectorized call hands the work to a
  single C/kernel call** — and (b) drop the hard "order(s) of magnitude" multiplier or hedge it explicitly as
  size- and framework-dependent. Drafted in LR-1. (The JAX loop taking ~1 s is also worth a one-line
  comment: each `c.at[i].set(...)` triggers dispatch/trace overhead — already partly explained by the JAX
  immutability comment at lines 466–468, good.)
- **`normal()` + density plot: teaches.** Plots three computed Gaussians — legitimate. Keep.

### PyTorch

- Imports cell (lines 42–50): idiomatic. `import numpy as np` alongside `torch` is fine (NumPy is used for
  the density `x = np.arange(...)`). No dated pre-2.x idioms. **Modern.**
- `normal()` uses `np.exp` on a NumPy array — clean. Note the function is defined per-framework but is
  *NumPy-identical* for pytorch/tensorflow/mxnet (see consistency note). **OK.**
- Loop cell uses `c[i] = a[i] + b[i]` on a torch tensor — correct, and the point of the demo is precisely
  that this is slow. **OK** (it is intentionally bad code shown as a foil).

### JAX

- Imports (lines 62–69): `from jax import numpy as jnp` + `math` + `time`; modern. Note jax does **not**
  `import numpy as np`, so its density cell uses `jnp.arange` (line 590) while pytorch/tf/mxnet use NumPy —
  a minor, defensible divergence (JAX users would use `jnp`). **OK.**
- Loop cell: the immutable-update comment (lines 466–468) and `c.at[i].set(...)` are correct modern JAX. The
  ~1 s runtime is expected (per-iteration dispatch); fine as a foil. **Modern.**

### TensorFlow

- Imports (lines 52–60): `import tensorflow as tf` + `numpy as np`; pre-Keras-3 era but nothing here touches
  Keras, so no staleness. **OK for 2021; still valid.**
- Loop cell uses `tf.Variable(...)` + `c[i].assign(...)` — correct TF idiom for mutable element assignment.
  This is what makes the TF loop measurement ~2000× (assign is heavy), which is the source of the slide's
  "3 orders of magnitude." **OK** as code; the *claim built on it* is the LR-1 problem.

### MXNet

- Imports (lines 33–40): `from mxnet import np`. **Currency flag (LR-5):** Apache MXNet was retired to the
  ASF Attic in 2023. The chapter (and book) present it as a co-equal fourth tab. Per the review guide this
  should be explicitly flagged for a book-wide de-emphasis/drop decision. In *this file* there is nothing
  MXNet-specific that is wrong (the code runs; `mxnet==2.0.0` in the manifest), so there is no per-file code
  fix — the action is the book-wide tab decision, owned by the overview. The density cell's
  `x.asnumpy()` calls (lines 558) are the one MXNet-only wart (needed because `mxnet.np` arrays aren't
  directly matplotlib-friendly) — harmless, but another small sign of the legacy tab's cost.

### Cross-framework consistency & d2l conventions

- **One imports cell per framework near the top.** Confirmed (cell `linear-regression`, lines 33–69). No
  re-imports later. **Compliant.**
- **`normal()` is defined four times, but three are byte-identical** (pytorch/tensorflow/mxnet all use
  `np.exp`; only jax differs with `jnp.exp`). This is *gratuitous divergence* (the review guide flags this):
  the pytorch/tf/mxnet variants could collapse to a single untagged cell that imports work for all three
  NumPy-backed tabs, leaving only the jax variant separate. **Minor**, optional cleanup (LR-8) — but it is
  exactly the "differ more than the frameworks require" cost the guide warns about. Note: this is a
  source-`.md` `%%tab` restructuring; verify it doesn't break the slide placeholder
  `@linear-regression-the-normal-distribution-and-squared-loss-1` (line 911) before applying.
- **Stable cell IDs present** on every code fence (`#linear-regression`, `#…-vectorization-for-speed-1/2/3`,
  `#…-the-normal-distribution-and-squared-loss-1/2`). **Compliant.**
- Uses `d2l.plot`, `d2l.ones`, `d2l.zeros` helpers rather than raw framework calls where possible.
  **Compliant.**

---

## 4. Implementation spec (downstream agents act on this)

### LR-1 — Fix the vectorization timing narrative (claim ≠ committed outputs)  ·  [P0] · [S] · [authored]
- **Type:** code / currency / prose
- **Where:** `chapter_linear-regression/linear-regression.md`.
  (a) Prose conclusion, lines 484–488, verbatim anchor:
  `The second method is dramatically faster than the first.\nVectorizing code often yields order-of-magnitude speedups.`
  (b) Slide, line 897, verbatim anchor:
  `Roughly **3 orders of magnitude faster** on this size — Python's\ninterpreter overhead is the killer; the C kernel barely breaks a\nsweat.`
- **Change:** The committed outputs give ratios of ~58× (pytorch), ~16× (jax), ~2000× (tf), ~460× (mxnet) at
  $n=1000$ — so neither "order of magnitude" nor "3 orders of magnitude" is right across frameworks. Make the
  lesson about *why* (per-op Python dispatch overhead) and hedge the magnitude.
  - Replace the prose (a) with:
    `The second method is dramatically faster. The reason is not that addition itself got cheaper but that we replaced *n* round-trips through the Python interpreter—one per element, each dispatching a separate tensor operation—with a single call into a compiled linear-algebra kernel. The speedup therefore grows with the vector length and varies widely across frameworks and hardware (here, anywhere from roughly tenfold to a thousandfold), but the qualitative lesson is universal: push inner loops down into vectorized library calls rather than writing them in Python.`
  - Replace the slide tail (b) with:
    `Often **one to three orders of magnitude** faster, growing with vector\nlength — Python's per-element dispatch overhead is the killer; the\ncompiled kernel does the whole add in one call.`
- **Touches:** none (source `.md` only; no re-execution needed — the numbers already live in the manifests).
- **Done when:** neither the prose nor the slide asserts a single fixed multiplier; both attribute the
  speedup to Python per-op/dispatch overhead and note size/framework dependence; `make html` clean; slide
  still renders with the two timing cells beneath it.
- **Depends on:** none.

### LR-2 — Intuition-first opening (hook + picture) to meet the Preliminaries bar  ·  [P1] · [M] · [authored]
- **Type:** teaching / prose (optionally figure)
- **Where:** `chapter_linear-regression/linear-regression.md`, lines 9–31 (the opening, from
  `*Regression* problems pop up whenever…` through `…are called *features* (or *covariates*).`).
- **Change:** Keep the terminology paragraph (lines 23–31, which defines training set / example / label /
  feature — that is good and must stay). Replace **only** the application-list lead-in (lines 9–22) with a
  concrete, single-example hook that mirrors the calculus/Archimedes standard. Drafted replacement for lines
  9–22:
  `Suppose you are about to buy a house and want to know what a fair price is. You collect recent sales in the neighborhood, and for each one you note its area, its age, and the price it fetched. Plotting price against area, the points scatter around a rising line: bigger houses cost more, not exactly but on average. *Linear regression* is the tool that draws that line—and, with more than one feature, the corresponding plane or hyperplane—and turns it into a prediction for a house you have not seen.\n\nRegression problems arise whenever we want to predict a numerical value: the price of a home or a stock, a patient's length of stay in hospital, the demand for a product next quarter. Not every prediction is of this kind—later we turn to *classification*, where the target is a category rather than a number—but regression is the natural place to begin, and the running example we return to throughout this chapter is predicting house prices from area and age.`
  Then the existing terminology paragraph (lines 23–31) follows unchanged.
  *Optional figure (judgment, can be deferred):* a small house-price scatter with a fitted line, in house
  style, would make the hook literal. `fig_fit_linreg` (Fig. 3.1.1) already shows an abstract 1-D fit later,
  so this is optional; if added, it is a generator SVG (`tools/gen_*` → `img/...svg`), not inline matplotlib.
- **Touches:** none for the prose. If the optional scatter figure is added: a figure generator + `make
  figures` + committed `img/<id>.svg` (see the `mdl-figure` skill / house convention).
- **Done when:** the section opens with a concrete house-price scenario before any list of applications; the
  training-set/example/label/feature definitions are preserved; `make html` clean.
- **Depends on:** none.

### LR-3 — Lean geometry-of-least-squares forward pointer (no new figure)  ·  [P1] · [S] · [authored]
- **Type:** coverage / teaching
- **Where:** `chapter_linear-regression/linear-regression.md`, immediately after the normal-equations
  uniqueness sentence ending at line 276 (`…are linearly independent :cite:`Golub.Van-Loan.1996`.`), before
  the blank lines preceding `### Minibatch Stochastic Gradient Descent`.
- **Change:** Insert one paragraph (authored verbatim):
  `This solution has a clean geometric reading. The vector of predictions $\mathbf{X}\mathbf{w}$ ranges over the *column space* of $\mathbf{X}$—all linear combinations of the feature columns—as $\mathbf{w}$ varies. Minimizing $\|\mathbf{y}-\mathbf{X}\mathbf{w}\|^2$ therefore asks for the point in that subspace closest to the observed labels $\mathbf{y}$, which is exactly the *orthogonal projection* of $\mathbf{y}$ onto the column space. The residual $\mathbf{y}-\mathbf{X}\mathbf{w}^*$ is what is left over, and it must be perpendicular to every feature column—precisely the statement $\mathbf{X}^\top(\mathbf{X}\mathbf{w}^*-\mathbf{y})=\mathbf{0}$ we just derived. The same orthogonal-projection idea, for projecting onto a single direction, is developed with a picture in :numref:`sec_mdl-geometry-linear-algebraic-ops`.`
- **Touches:** none (the target label `sec_mdl-geometry-linear-algebraic-ops` already exists and is
  confirmed; no new figure — explicitly reuse the appendix's existing projection material).
- **Done when:** the paragraph renders after the normal-equations block; the `:numref:` resolves (no
  "??" in HTML/PDF); no new SVG is added; `make html` clean. Keep it to a single paragraph — if it grows past
  ~6 sentences it has overstepped scope.
- **Depends on:** none.

### LR-4 — Decide fate of the dated biological-neuron figure & `### Biology`  ·  [P1] · [S–L] · [judgment]
- **Type:** figure / teaching
- **Where:** `chapter_linear-regression/linear-regression.md`, `### Biology` subsection lines 685–733 and the
  figure block lines 701–702:
  `![The real neuron (source: "Anatomy and Physiology" by the US National Cancer Institute's Surveillance, Epidemiology and End Results (SEER) Program).](../img/neuron.svg)\n:label:`fig_Neuron``
  Asset: `img/neuron.svg` (Adobe Illustrator export, hand-traced — not house style).
- **Change (agent/author picks one, constrained):**
  - **Preferred (keep + redraw):** keep the subsection prose (it is good — the McCulloch–Pitts history and
    the `Russell.Norvig.2016` airplanes/birds framing at lines 726–733 must be preserved), and replace
    `img/neuron.svg` with a clean house-style schematic neuron drawn by a committed generator, visually
    rhyming with `fig_single_neuron` (dendrites = inputs $x_i$, soma = weighted sum $\sum x_i w_i + b$, axon
    = output, optional $\sigma(\cdot)$). Keep the `:label:` and `:numref:` stable.
  - **Acceptable (keep as-is):** leave figure and prose unchanged; the figure is stylistically off but not
    wrong and is properly attributed. Lowest effort.
  - **Acceptable (trim):** reduce `### Biology` to a 3–4 sentence paragraph and drop `fig_Neuron`.
  - **Do NOT** keep a half-measure (e.g., new caption on the old traced asset).
- **Touches:** if redrawing: figure generator (e.g. `tools/gen_mdl_*` per the `mdl-figure` skill) + `make
  figures` + committed `img/<id>.svg`; run the `figure-style-audit` skill afterward. If trimming: none.
- **Done when:** the chapter's figure style is internally consistent OR an explicit decision-to-keep is
  recorded; the McCulloch–Pitts history and Russell–Norvig framing survive in some form; `make html` clean;
  `fig_Neuron`/`:numref:` references resolve (or are removed if the figure is dropped).
- **Depends on:** none. (This is the watch-point's "dated cartoon" question; I lean toward *keep + redraw*
  but mark it judgment because it is effort-L and partly an editorial call.)

### LR-5 — De-emphasize MXNet (archived 2023) in the framework framing  ·  [P1] · [S] · [judgment]
- **Type:** currency
- **Where:** Book-wide; in this file the MXNet presence is the `%%tab mxnet` imports cell (lines 33–40), the
  mxnet variants of the timing/normal cells, and the `:begin_tab:`mxnet`` Discussions link (lines 788–790).
- **Change:** No per-file code edit is correct in isolation — MXNet here is not *wrong* (it executes;
  `mxnet==2.0.0`). The action is to register, for the cross-cutting overview, that MXNet was retired to the
  ASF Attic in 2023 and should be de-emphasized or dropped as a co-equal tab across §§3–5 (and the book). If
  the book-wide decision is "drop MXNet," this file's mxnet cells and `:begin_tab:`mxnet`` block are removed
  in that pass; if "de-emphasize," a one-line note belongs at the chapter cover, not here.
- **Touches:** book-wide tab policy; do not edit this file alone.
- **Done when:** the overview records a single MXNet decision applied consistently across §§3–5; this file
  conforms to it.
- **Depends on:** the cross-cutting overview's MXNet decision.

### LR-6 — Tighten two prose passages + one adaptive-optimizer clause  ·  [P2] · [S] · [authored]
- **Type:** prose / currency
- **Where & Change:** `chapter_linear-regression/linear-regression.md`:
  - (a) `inference`/`prediction` aside, lines 407–416, verbatim anchor
    `Deep learning practitioners have taken to calling the prediction phase *inference*` …through… `we will stick to *prediction* whenever possible.` Replace the whole block with:
    `Deep learning practitioners often call the prediction phase *inference*. This is a mild misnomer: in statistics, *inference* usually means estimating parameters, not scoring new points, so the overloaded term can confuse. We will say *prediction* throughout.`
  - (b) After the minibatch-size guidance sentence ending at line 337 (`…is a good start.`), append one
    forward-pointing clause:
    `Plain SGD is the simplest member of a large family; in practice the adaptive variants of :numref:`chap_optimization` (notably Adam and its decoupled-weight-decay form AdamW) are the default, and we revisit them there.`
    (Use the actual Optimization-part label; if `chap_optimization` is not the exact key, the agent
    substitutes the correct `:numref:` target — see the Optimization part's index `:label:`.)
- **Touches:** none, except the agent must confirm the Optimization-part label for (b).
- **Done when:** the `inference` aside is ≤3 sentences; the adaptive-optimizer clause renders with a
  resolving `:numref:`; `make html` clean.
- **Depends on:** none.

### LR-7 — Stitch Exercise 4 to weight decay / ridge  ·  [P2] · [S] · [authored]
- **Type:** teaching (exercise)
- **Where:** `chapter_linear-regression/linear-regression.md`, Exercise 4 (lines 768–772), which ends with
  the sub-part `1. What happens with stochastic gradient descent when $\mathbf{X}^\top \mathbf{X}$ does not have full rank?`
- **Change:** Append one sub-part immediately after that line:
  `    1. The standard remedy for a (near-)singular $\mathbf{X}^\top \mathbf{X}$ is to add $\lambda \mathbf{I}$ before inverting. Relate this to the $\ell_2$ penalty introduced in :numref:`sec_weight_decay`, and show that the resulting estimator $\mathbf{w}^* = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ is always well defined for $\lambda>0$.`
- **Touches:** none (the label `sec_weight_decay` is the weight-decay file's section label; agent confirms
  the exact key — `weight-decay.md` is referenced by `sec_linear_regression` reciprocally, so the label
  exists).
- **Done when:** Exercise 4 has the new ridge sub-part; the `:numref:` resolves; `make html` clean.
- **Depends on:** none.

### LR-8 — Collapse the three identical `normal()` definitions  ·  [P2] · [S] · [mechanical/judgment]
- **Type:** code / cross-framework consistency
- **Where:** `chapter_linear-regression/linear-regression.md`, the four `#linear-regression-the-normal-distribution-and-squared-loss-1` cells (lines 521–547). The pytorch (522–526), tensorflow (529–533), and mxnet (543–547) variants are **byte-identical** (`np.exp`); only jax (536–540) differs (`jnp.exp`).
- **Change:** Replace the three identical `%%tab pytorch` / `%%tab tensorflow` / `%%tab mxnet` blocks with a
  single `%%tab pytorch, tensorflow, mxnet` block containing the one shared definition, leaving the `%%tab
  jax` block as-is. (Do not merge jax — it must use `jnp`.) This removes gratuitous divergence per the d2l
  convention.
- **Touches:** **Verify slide placeholder** `@linear-regression-the-normal-distribution-and-squared-loss-1`
  (line 911) and the cell-ID injection (`tools/inject_outputs.py`) still resolve after the merge — the cell
  ID is unchanged, so this should be safe, but `make slides` + `make html` must be re-checked.
- **Done when:** one shared `normal()` cell covers pytorch/tensorflow/mxnet, the jax cell is unchanged, all
  four density plots still render identically (outputs unchanged), and the slide referencing this cell ID
  still builds. `make html` + `make slides` clean.
- **Depends on:** none. (Marked judgment because it touches the slide/injection machinery; if that adds risk,
  it is fine to leave the divergence — it is cosmetic.)

---

## 5. Keep — what is already excellent (do not lose this)

- **The Gaussian-noise → MLE derivation (lines 491–648).** Clean, correct, well-motivated, and self-contained:
  it sets up the likelihood, factorizes via independence, takes the log, drops the $\sigma$-independent term,
  and lands precisely on squared loss. This is exactly the CS229/Bishop derivation and it reads beautifully.
  It is the load-bearing prerequisite for `weight-decay.md`'s Bayesian-prior exercise — **do not touch the
  math.** (The two-stage "functional motivation first, then probabilistic" framing at lines 494–502 is a
  particularly nice pedagogical move.)
- **The MSE-$\tfrac12$ / built-in-loss caveat (lines 212–215).** Correct, practitioner-grade, and consistent
  with the concise/scratch siblings. A best-in-class touch.
- **The analytic-solution humility (lines 280–285):** "you should not get used to such good fortune… would
  exclude almost all exciting aspects of deep learning." Sets expectations perfectly. Keep.
- **The whole exercise set (lines 760–786).** Genuinely top-course quality — exponential-noise → $\ell_1$,
  Poisson counts, log-prices/Black–Scholes, the two-stacked-linear-layers collapse. Keep all; only *add* the
  ridge sub-part (LR-7).
- **The Russell–Norvig "airplanes vs. birds" framing of the biology (lines 724–733).** Self-aware, correct,
  and a graceful way to invoke biological inspiration without over-claiming it. Preserve this prose even if
  the figure changes (LR-4).
- **The scope discipline.** The file correctly defers generalization (one paragraph + "we return to these
  topics"), the implementation (to the scratch/concise siblings), and optimizer depth — it does not
  over-reach. This is the right instinct; keep it.
- **The single-neuron NN framing + `fig_single_neuron` (3.1.4 down to line 683).** The bridge from "weighted
  sum + bias" to "single-layer fully connected net" is exactly the right pivot into the rest of the book.
  Keep.
