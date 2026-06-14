# Review — chapter_linear-regression/generalization.md  (§3.6 "Generalization")

**Role in the chapter:** The conceptual hinge of the regression chapter: it
defines training vs. generalization error, the IID assumption, model complexity,
under/overfitting, the train/validation/test discipline (incl. test-set reuse),
model selection, and $K$-fold cross-validation. It sets up §3.7 *Weight Decay*
(the first practical regularizer) and is the back-reference target for the two
deeper generalization chapters (§4.6 classification, §5.5 deep).

**Verdict:** The prose is genuinely excellent — intuition-first (the Ellie/Irene
hook is one of the best in the book), correct, and admirably honest about
test-set reuse and the limits of training error. But measured against the
best-textbook bar it has three real gaps: (1) it is *entirely conceptual* — the
classic **polynomial under/just-right/over fitting demo that the upstream d2l
text runs as code has been deleted**, leaving only a single static schematic, so
the chapter never lets the reader *see* the U-curve emerge; (2) **bias–variance
is never named**, even though the full decomposition lives in §25 and that file
already cross-references *back* to this section — a one-way broken link; (3) it
does **not forward-point** to the deep/overparametrized story (§5.5) or VC theory
(§4.6), so the "currency" foreshadow the field now demands is absent. The
single highest-value change is **restoring a runnable polynomial-fitting demo**
(GEN-1): it converts the chapter's central claim from assertion to demonstration
and is what a Stanford/CMU treatment would never omit.

**Grade:** **B.** Assignable today and pedagogically strong in prose, but the
missing demonstration, the absent bias–variance vocabulary, and the missing
forward pointers keep it from the A bar a top program's *primary* generalization
reading should hit.

**Top priorities (ranked):**
1. [P0] **GEN-1** — Restore a runnable polynomial-fit demo (under/just-right/over) so the U-curve is *shown*, not just asserted.
2. [P1] **GEN-2** — Fix the dangling "As the above bound already indicates" (line 341): no bound was ever stated.
3. [P1] **GEN-3** — Name **bias–variance** and forward-point to the decomposition in §25 (`sec_mdl-statistics`); currently a one-way broken link.
4. [P1] **GEN-4** — Add the honest overparametrized/double-descent foreshadow + forward pointers to §5.5 and §4.6 (the explicit currency touch the brief asks for).
5. [P2] **GEN-5** — Move the **Dataset Size** subsection out from under "Underfitting or Overfitting?" (it is orphaned there and is what the dangling "bound" sentence belonged near).
6. [P2] **GEN-6** — Tighten the cross-validation subsection (add the standard pitfalls: variance/cost trade-off, leakage, why the estimate is biased) and add `:numref:` forward pointers from the Summary.
7. [P2] **GEN-7** — Replace the bare paperswithcode benchmark-reuse links with a proper citation (Recht et al.) and tighten two verbose passages.

---

## 1. Coverage

### Add

**(P0) A runnable polynomial under/just-right/over-fit experiment — GEN-1.**
This is the chapter's defining gap. The upstream d2l.ai version of this section
*executes* a polynomial-regression demo: it generates noisy data from a degree-3
target, then fits degree-1 (underfit), degree-3 (just right), and a high-degree
polynomial (overfit), and plots train vs. test loss for each. **That code has
been removed here** — confirmed: the source has **0 code fences, 0 `%%tab`
blocks, and no executed-output manifests exist for any framework**
(`outputs/<fw>/chapter_linear-regression/generalization*` is absent for all of
pytorch/jax/tensorflow/mxnet). What remains is one static schematic
(`fig_capacity_vs_error`) plus the prose at lines 298–337 promising the reader
they will "compare the relationship between polynomial degree … and both
underfitting and overfitting" — a promise the page never cashes. A best-in-class
treatment (CS229's bias–variance lecture, ESL §7, Bishop §1.1 polynomial demo,
Murphy §1) *shows* the three regimes. This is squarely **in scope** (it is the
core teaching of *this* file, not downstream material) and per the brief's
watch-point (c) it is exactly the demo I was asked to cross-check — and it does
not exist. See GEN-1 in §4 for the full drafted cell set.

**(P1) Name the bias–variance trade-off and forward-point — GEN-3.** The phrase
"bias–variance" appears **nowhere** in this file (confirmed by grep). Yet
`chapter_mdl-probability-statistics/mdl-statistics.md` (§25) derives the full
decomposition and explicitly says, three times, that its U-curve *is* the
under/over-fitting picture "from `sec_generalization_basics`," even attributing
the double-descent caveat to "why `sec_generalization_basics` cautions that
larger architectures … generalize *better*." **But this file issues no such
caution and contains no pointer to §25** — the cross-link is one-way and
currently dangling from the reader's perspective. Per the scope map, the *full*
decomposition stays in the appendix (don't import); this file should **name the
concept once** where it discusses model complexity / the U-curve and forward-
point. Drafted in GEN-3.

**(P1) Honest overparametrized-regime foreshadow + forward pointers — GEN-4.**
The brief calls this out explicitly: "A one-line, honest foreshadow that 'in the
overparametrized regime this classical picture is incomplete (see §5.5)' is a
good in-scope currency touch." Right now the file *gestures* at it — lines
246–264 admit deep nets "are too powerful to allow us to conclude much from
training error," and the Summary parenthetical mentions "larger architectures
with more parameters generalizing better" (lines 444–446) — but it **never names
double descent, never says the classical U-curve is incomplete, and never points
to §5.5 (`sec_generalization_deep`) or §4.6 (`chap_classification_generalization`)**.
Both citations needed (`nakkiran2021deep`, `Belkin.Hsu.Ma.ea.2019`) are **already
in `d2l.bib`** (verified). This is the cleanest currency win in the file and is
in scope as a *pointer*, not an import. Drafted in GEN-4.

**(P2) Cross-validation pitfalls — GEN-6.** The CV subsection (lines 421–433) is
correct but thin: it defines $K$-fold and stops. A top problem-set-grade
treatment names the standard pitfalls, which the exercises (4, 5) already hint
at but the text never states: (i) cost is $K\times$ training; (ii) the variance/
bias trade-off in choosing $K$ (LOO = low bias, high variance, expensive); (iii)
**data leakage** — preprocessing (scaling, feature selection) must happen
*inside* each fold, not before splitting, or the validation estimate is
optimistic. Drafted in GEN-6.

### Remove / trim

- **Line 341 "As the above bound already indicates" — GEN-2 (this is a removal/fix, not an add).** There is **no bound anywhere above it** (confirmed: no inequality, `\leq`, concentration result, or stated bound appears in the file). This is a leftover from an earlier draft (the upstream text had a Model-Complexity discussion that gestured at sample-complexity bounds). As written it sends the reader hunting for something that isn't there. Fix in GEN-2.
- **No dead frameworks to flag.** Because the file has **no code**, there is no MXNet tab, no dated PyTorch/JAX/TF idiom, nothing to de-emphasize. The MXNet-archival concern does not bite here — *except* that if GEN-1 restores the demo, it should be authored to the chapter's current convention (PyTorch primary; see GEN-1's framework note). Worth surfacing to the cross-cutting overview: this file is a clean case where re-adding code must *not* re-introduce a co-equal MXNet tab.
- **Verbose passages (P2, GEN-7).** Lines 87–102 (the overfitting/regularization paragraph) and 60–85 (dataset-scale paragraph) are slightly padded; minor tightening only, not a priority.

### Reorder / restructure

- **Move "Dataset Size" out from under "Underfitting or Overfitting?" — GEN-5.** The `### Dataset Size` subsection (lines 339–364) currently sits as a child of `## Underfitting or Overfitting?`, but it is really a *model-complexity / sample-complexity* point (it even opens with the dangling "bound" line). It reads as orphaned. Best home: promote it to a `###` under a renamed complexity section, or fold it into Model Complexity (§3.6.1.1). See GEN-5.
- **Top-level shape is otherwise good.** Four `##` sections (Training/Generalization Error → Underfitting/Overfitting → Model Selection → Summary) is within the 3–5 target and the spine is logical. The only structural smell is GEN-5.

---

## 2. Teaching quality

### Structure & flow

Strong. The Ellie/Irene memorization-vs-pattern hook (lines 4–34) is intuition-
first done right and is among the book's best openers — it should be **kept
verbatim** (see §5). The progression "what is generalization → train vs. test
error → complexity → under/overfit → model selection → CV" is the canonical and
correct order. The one structural defect is the orphaned Dataset Size subsection
(GEN-5).

### Figures

Single figure, and it under-delivers:

| Fig | Label | Caption | Assessment |
|---|---|---|---|
| 3.6.1 | `fig_capacity_vs_error` | "Influence of model complexity on underfitting and overfitting." | `../img/capacity-vs-error.svg`, **39 KB, no committed generator** (grep of `tools/` finds none — it is a legacy hand-drawn d2l SVG). It is *shared* with §5.5 (`generalization-deep.md` also `:numref:`s it). It carries the U-curve idea adequately, but it is the *only* visual and it is static. |

Two figure actions:
- **(P0, part of GEN-1)** The restored polynomial demo will produce **computed
  loss curves** (train vs. test vs. degree) — these are *data plots that teach*
  (the guide explicitly permits `d2l.plot` loss curves) and are the missing
  visual payoff. Not a schematic; no generator needed.
- **(P2, optional, noted not drafted)** `capacity-vs-error.svg` has no generator,
  so it cannot be restyled via `make figures`. Since it is shared with §5.5,
  bringing it under a committed generator is a chapter-spanning house-style item
  better owned by the overview than by this single-file report. Flag only.

No figure-drawing matplotlib lives in a teaching cell here (there are no cells at
all), so the "no inline schematic drawing" rule is trivially satisfied.

### Prose & clarity

Generally excellent and quotable. Specific notes:

- **Lines 165–180** (fixed classifier on test set = mean estimation; training
  error is biased) is a genuinely sharp, correct point that many texts fumble —
  **keep**.
- **Lines 208–221** (Popper / falsifiability framing of complexity) is elegant
  and earns its place — **keep**.
- **Line 341** dangling "bound" — see GEN-2.
- **Lines 397–419** (the "murky business … no true test sets … reported accuracy
  is really validation accuracy") is unusually honest and is a teaching strength
  — **keep**, but pair with the Recht citation (GEN-7) so the benchmark-reuse
  claim is sourced rather than resting on two bare paperswithcode URLs.
- **Lines 91–92**: "While it is no substitute for a proper introduction to
  statistical learning theory" — good, but this is the natural spot to *also*
  forward-point to §4.6 (the book's own intro to VC theory) rather than only to
  external `Vapnik98`/`boucheron2005theory`. Folded into GEN-4.

### Exercises

Good set, genuinely thought-provoking (esp. Ex. 6 on VC dimension ignoring
function magnitude, and Ex. 7 the "argue for more data by *removing* data"
learning-curve question). Gaps:
- The exercises lean on **VC dimension** (Ex. 6) and the **biased CV estimate**
  (Ex. 5) that the *body text never introduces* — Ex. 6 mentions VC dimension for
  the first time in the whole file. Either seed a one-line mention + §4.6 pointer
  (GEN-4 does this) or the exercise floats free.
- **Missing:** an exercise tied to the (restored) polynomial demo — e.g. "Vary
  the training-set size and re-plot the degree-vs-error curves; at what degree
  does overfitting set in for n=10 vs n=100?" This is the highest-value new
  exercise and is drafted in GEN-1 (so the demo and its exercise land together).
- Ex. 1 ("When can you solve polynomial regression exactly?") is a touch terse;
  fine to keep.

---

## 3. Code & examples

### Does the code teach?

**There is no code.** Confirmed: 0 code fences, 0 `%%tab` blocks, no
executed-output manifests for any framework. This is the file's central coverage
gap, not a virtue — the chapter *asserts* the under/just-right/over story and
shows a static cartoon, but never demonstrates it. See GEN-1. Everything below is
therefore about the code that **should** exist.

### PyTorch

To be authored (GEN-1). The chapter's current convention is **PyTorch-primary**;
the demo is pure NumPy/torch tensor math plus `d2l.plot`, so a single untagged
all-framework cell set is achievable for the data generation, with at most a thin
per-framework fit. Drafted in GEN-1 as framework-agnostic where possible.

### JAX

Same (GEN-1). The least-squares fit (`numpy.polyfit` / `lstsq`) is framework-
neutral; no JAX-specific idiom is required. If a framework split is used, keep it
minimal.

### TensorFlow

Same (GEN-1). No TF-specific code is needed for a least-squares polynomial fit;
avoid gratuitous divergence.

### MXNet

**Do not add an MXNet tab.** MXNet was archived by the ASF (2023). Since the demo
is being re-introduced from scratch, author it for the live frameworks only
(PyTorch primary; JAX/TF as the chapter's other tabs warrant). Re-adding a
co-equal MXNet variant here would re-introduce dead-framework surface area the
rest of the review effort is trying to retire — flag to the overview.

### Cross-framework consistency & d2l conventions

N/A today (no code). For GEN-1: one imports cell, stable `#cell-id`s, use
`d2l.plot` for the loss curves (not raw matplotlib), and prefer a single untagged
cell for the framework-neutral parts to minimize divergence.

---

## 4. Implementation spec (downstream agents act on THIS)

### GEN-1 — Restore the runnable polynomial under/just-right/over-fit demo  ·  [P0] · [L] · [authored]
- **Type:** coverage / code / teaching
- **Where:** `chapter_linear-regression/generalization.md`, inside
  `### Polynomial Curve Fitting` (`:label:`subsec_polynomial-curve-fitting``,
  ~lines 298–337). Insert the code cells **after** the schematic figure block
  (after line 336 `:label:`fig_capacity_vs_error``) and before `### Dataset Size`.
- **Change:** Add a compact, mostly-framework-neutral demonstration that
  generates noisy data from a degree-3 target, fits polynomials of degree
  {1, 3, high (e.g. 19)} by least squares, and plots **train vs. test loss vs.
  degree** so the U-curve is *computed*. Author to the chapter's PyTorch-primary
  convention; keep the data-gen and fitting in a single untagged all-framework
  cell where possible (the math is `numpy`/`lstsq`; no framework-specific API is
  required). Drafted skeleton (an agent adapts to current d2l helpers + assigns
  stable `#`-ids via `tools/add_cell_ids.py`):

  ```
  Imports cell (one, near the demo): numpy as np, math, and `from d2l import torch as d2l`.

  Cell A — synthesise data from a known degree-3 polynomial with Gaussian noise:
    max_degree, n_train, n_test = 20, 100, 100
    true_w = np.zeros(max_degree); true_w[:4] = [5, 1.2, -3.4, 5.6]
    features = np.random.randn(n_train + n_test, 1)
    np.random.shuffle(features)
    poly = np.power(features, np.arange(max_degree).reshape(1, -1))
    poly /= gamma(np.arange(max_degree) + 1)        # scale x^i by 1/i! for conditioning
    labels = poly @ true_w + 0.1 * np.random.randn(n_train + n_test)

  Cell B — a small train/eval helper that fits the first `d` polynomial columns
    by least squares (np.linalg.lstsq or a 1-layer linear fit) and returns
    train/test MSE.

  Cell C — underfit: fit degree 1 (columns 0:1) → report train/test loss (both high).
  Cell D — just right: fit degree 3 (columns 0:4) → report (both low, close).
  Cell E — overfit: fit full degree 19 (columns 0:20) with only n_train=100 →
    train loss ≈ 0, test loss large.
  Cell F — d2l.plot of train vs. test loss across degrees 1..19 (the U-curve),
    with a one-sentence prose interpretation tying it back to fig_capacity_vs_error.
  ```

  After the curves, add **one new exercise** (append to the Exercises list):
  *"Re-run the polynomial demo with `n_train` = 10, 40, 100. At what polynomial
  degree does the test loss start to rise in each case? Relate your answer to the
  'more complex models require more data' rule of thumb."*
- **Touches:** `tools/add_cell_ids.py` (assigns the `#`-ids on save);
  notebook execution for each live framework to produce committed outputs
  (`make -B _notebooks/<fw>/chapter_linear-regression/generalization.executed`
  then `make capture-outputs FILES=chapter_linear-regression/generalization.md`);
  re-render with `make html`. **No new schematic SVG** (the loss curves are
  computed `d2l.plot` outputs). Do **not** add an MXNet tab.
- **Done when:** (1) `outputs/pytorch/chapter_linear-regression/generalization.json`
  exists with the demo cells' outputs; (2) the degree-1 fit shows *both* train and
  test loss high, degree-3 shows both low and close, degree-19 shows train loss
  near zero with clearly larger test loss; (3) the degree-vs-loss plot renders a
  visible U / divergence in HTML and PDF; (4) `make html` clean.
- **Depends on:** none. (GEN-5 is best done in the same pass since it touches the
  adjacent Dataset Size subsection, but is independent.)

### GEN-2 — Fix the dangling "above bound" reference  ·  [P1] · [S] · [mechanical]
- **Type:** prose / correctness
- **Where:** `chapter_linear-regression/generalization.md`, line ~341, opening of
  `### Dataset Size`. Verbatim anchor: `As the above bound already indicates,`
- **Change:** No bound is stated anywhere earlier in the file. Replace the dangling clause.
  - `old`: `As the above bound already indicates,\nanother big consideration\nto bear in mind is dataset size.`
  - `new`: `Beyond model complexity,\nanother big consideration\nto bear in mind is dataset size.`
  (If GEN-5 promotes this subsection under Model Complexity, the agent may instead
  open with `As the discussion of model complexity suggests,` — either resolves the
  dangling pointer.)
- **Touches:** none.
- **Done when:** the word "bound" no longer appears without an antecedent; `make html` clean.
- **Depends on:** none (compatible with GEN-5).

### GEN-3 — Name bias–variance and forward-point to §25  ·  [P1] · [S] · [authored]
- **Type:** coverage / currency
- **Where:** `chapter_linear-regression/generalization.md`, in `### Model Complexity`,
  immediately after the sentence ending `...complex matter.` (line ~224, before
  "Often, models with more parameters…"). Verbatim anchor:
  `Now what precisely constitutes an appropriate\nnotion of model complexity is a complex matter.`
- **Change:** Insert (do **not** derive the decomposition here — it lives in §25):

  > The classical way to make this trade-off precise is the *bias–variance
  > decomposition*: a model too simple to capture the signal makes a systematic
  > error (high *bias*, i.e. underfitting), while a model flexible enough to chase
  > the noise in a particular training set varies wildly from one dataset to the
  > next (high *variance*, i.e. overfitting). Their sum traces the familiar U-shaped
  > curve in :numref:`fig_capacity_vs_error`. We derive this decomposition formally
  > in :numref:`sec_mdl-statistics`.

- **Touches:** none (`sec_mdl-statistics`, `fig_capacity_vs_error` both exist —
  verified). Makes the existing §25→§3.6 cross-link bidirectional.
- **Done when:** "bias–variance" appears in the file; `:numref:`sec_mdl-statistics``
  resolves; `make html` clean.
- **Depends on:** none.

### GEN-4 — Honest overparametrized-regime foreshadow + §5.5/§4.6 pointers  ·  [P1] · [M] · [authored]
- **Type:** currency / coverage
- **Where:** `chapter_linear-regression/generalization.md`. Two insertions:
  - (a) End of `### Model Complexity`, after the holdout/validation paragraph
    (after line ~264, `...is called the *validation error*.`).
  - (b) The statistical-learning-theory caveat at lines 91–92 (add the in-book
    pointer alongside the external refs).
- **Change:**
  - (a) Insert the honest foreshadow:

    > This classical "more capacity ⇒ more overfitting" picture is, however,
    > *incomplete* for the heavily overparametrized models at the heart of modern
    > deep learning. Once a model is large enough to *interpolate* its training data
    > (drive training error to zero), pushing capacity even higher often makes test
    > error *fall again* rather than rise — the *double descent* phenomenon
    > :cite:`Belkin.Hsu.Ma.ea.2019,nakkiran2021deep`. We take up this modern story,
    > and why the classical complexity intuition breaks down, in
    > :numref:`sec_generalization_deep`.

  - (b) Mechanical edit at line ~92:
    - `old`: `While it is no substitute for a proper introduction\nto statistical learning theory (see :citet:`Vapnik98,boucheron2005theory`),`
    - `new`: `While it is no substitute for a proper introduction\nto statistical learning theory (:numref:`chap_classification_generalization` gives a first, rigorous taste; see also :citet:`Vapnik98,boucheron2005theory`),`
- **Touches:** none — all four targets verified present:
  `Belkin.Hsu.Ma.ea.2019` and `nakkiran2021deep` in `d2l.bib`;
  `sec_generalization_deep` and `chap_classification_generalization` are valid labels.
- **Done when:** "double descent" appears once with both citations; the §5.5 and
  §4.6 `:numref:`s resolve; `make html`/`make pdfs` clean (watch the `$`-before-digit
  rule — none of the inserted text triggers it).
- **Depends on:** none. (Reads naturally after GEN-3; do GEN-3 then GEN-4.)

### GEN-5 — Re-home the "Dataset Size" subsection  ·  [P2] · [S] · [judgment]
- **Type:** teaching / structure
- **Where:** `chapter_linear-regression/generalization.md`, `### Dataset Size`
  (lines ~339–364), currently a child of `## Underfitting or Overfitting?`.
- **Change:** Move this subsection so it sits under model-complexity reasoning,
  not under under/over-fitting. Preferred: relocate it to a `###` immediately
  following `### Model Complexity` (so it reads as the "sample side" of the
  complexity/data balance). Keep the text; only the position and the GEN-2 opener
  change. Agent decides exact placement to preserve flow.
- **Touches:** none.
- **Done when:** "Dataset Size" no longer renders as a sub-point of
  "Underfitting or Overfitting?"; section numbering re-renders sensibly; `make html` clean.
- **Depends on:** none (coordinate with GEN-2's opener wording).

### GEN-6 — Flesh out cross-validation with the standard pitfalls  ·  [P2] · [M] · [authored]
- **Type:** coverage / teaching
- **Where:** `chapter_linear-regression/generalization.md`, `### Cross-Validation`
  (lines ~421–433), after the existing definition paragraph.
- **Change:** Append:

  > Three caveats matter in practice. First, $K$-fold cross-validation is
  > *expensive*: it trains the model $K$ times, so it is most attractive precisely
  > when data — not compute — is the binding constraint. Second, the choice of $K$
  > trades bias against variance: large $K$ (in the limit, *leave-one-out*) trains
  > on nearly all the data and so is nearly unbiased, but the $K$ fitted models are
  > highly correlated, making the averaged estimate high-variance and costly;
  > $K=5$ or $10$ is the usual compromise. Third, and most easily overlooked, any
  > data-dependent preprocessing — feature scaling, feature selection,
  > hyperparameter choices — must be redone *inside each fold*, using only that
  > fold's training portion. Fitting a scaler or selecting features on the full
  > dataset before splitting leaks information from the held-out fold and yields an
  > optimistic, untrustworthy estimate.

  Also add two forward `:numref:`s in the **Summary** rule-of-thumb list (rule 1
  about validation/CV → `:numref:`sec_generalization_deep``; the IID rule 5 is
  fine as-is).
- **Touches:** none.
- **Done when:** the CV subsection states the cost, the $K$ bias/variance
  trade-off, and the leakage pitfall; exercises 4 and 5 now have textual support;
  `make html` clean.
- **Depends on:** none.

### GEN-7 — Source the benchmark-reuse claim; minor prose tightening  ·  [P2] · [S] · [mechanical]
- **Type:** prose / currency
- **Where:** `chapter_linear-regression/generalization.md`, lines ~403–408 (the
  "recycling benchmark data for decades" sentence with two bare paperswithcode URLs).
- **Change:** Keep the paperswithcode links as live SOTA illustrations but add a
  citation for the empirical benchmark-overfitting result. Insert after
  `...development of algorithms,` (line ~405):
  - `new clause`: `as documented when researchers rebuilt fresh test sets for long-standing benchmarks :cite:`recht2019imagenet`,`
  **Note (blocking sub-step):** `recht2019imagenet` is **not** in `d2l.bib` — the
  agent must add the BibTeX entry for Recht, Roelofs, Schmidt & Shankar, *"Do
  ImageNet Classifiers Generalize to ImageNet?"*, ICML 2019, before this `:cite:`
  resolves. (The related `Zhang…Recht…2017` "understanding deep learning requires
  rethinking generalization" key already exists if a substitute is preferred, but
  Recht et al. 2019 is the on-point reference for *benchmark reuse*.)
- **Touches:** `d2l.bib` (add one entry).
- **Done when:** the benchmark-reuse sentence carries a resolving citation;
  `make html` clean (no unresolved-citation warning).
- **Depends on:** none.

---

## 5. Keep — what is already excellent (do not lose this)

- **The Ellie/Irene opener (lines 4–34).** A model intuition-first hook —
  memorization vs. pattern-extraction made concrete before a single symbol. One
  of the best openers in the book; do not touch.
- **The "leap from particular observations to general statements" framing (lines
  49–57)** situating generalization as the central problem of statistics and
  science — keep.
- **The IID-assumption motivation (lines 110–126)**, including the honest "absent
  any such assumption we would be dead in the water" and the $P=Q$ vs. $P\neq Q$
  setup — crisp and correct.
- **The fixed-classifier-on-test-set = mean-estimation insight vs. the biased
  training error (lines 165–180).** A subtle, correct point most texts get wrong;
  preserve exactly.
- **The Popper/falsifiability framing of model complexity (lines 208–221)** —
  elegant and rare; keep.
- **The kernel-methods caveat (lines 229–232)** — "infinite parameters yet
  controlled complexity" correctly pre-empts the naive "complexity = parameter
  count" reading; keep (and GEN-3's bias–variance text is consistent with it).
- **The honest test/validation muddiness passage (lines 397–419)** — "what we
  report is really validation accuracy, not true test accuracy." Unusually candid
  and pedagogically valuable; keep (GEN-7 only *adds* a citation).
- **Exercises 6 and 7** (VC dimension ignoring magnitude; argue for more data by
  *removing* data) — genuinely thought-provoking, top-course quality; keep.
- **The 5 rules-of-thumb Summary (lines 447–453)**, especially rule 5 on the IID
  predicate — keep (GEN-6 only adds forward `:numref:`s).
