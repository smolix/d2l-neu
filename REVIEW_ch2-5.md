# Review: Chapters 2–5 (Preliminaries through Multilayer Perceptrons)

*Companion to `REVIEW_math_appendix.md`. Compiled 2026-07-02 on branch `math-appendix-polish`
(post-appendix-polish state: §24.2 and §25.5 exist, ch. 25 reordered, double descent consolidated,
Rademacher/Hoeffding hand-offs repointed). Method: one line-by-line review per chapter, every
`file:line` verified against source, all worked arithmetic recomputed; two external-calibration
research passes (UDL/Prince, Bishop & Bishop 2024, ISL/ESL, CS231n, Nielsen, Karpathy, Colah,
3Blue1Brown, Blitzstein, Seeing Theory, Google MLCC, WILDS, and the primary papers). Review only —
nothing below is implemented. Chapter numbers: ch. 2 = `chapter_preliminaries`, ch. 3 =
`chapter_linear-regression`, ch. 4 = `chapter_linear-classification`, ch. 5 =
`chapter_multilayer-perceptrons`.*

## 1. Executive summary

Chapters 2–5 are structurally sound and in several places already best-in-class: the backprop
worked numeric example (every number verified, dead-ReLU made concrete), the softmax derivation
chain across §4.1→§4.4→§4.5 (show the failure, derive the fix, quantify the fp32 thresholds), the
distribution-shift treatment (still ahead of every intro-tier peer), the pandas decision-centric
rewrite, the HIV Bayes worked example, and the OO-design section's async-plotting discussion. The
spiral with the polished math appendix works where it was wired: every early→appendix `:numref:`
resolves, and the bias–variance, Hoeffding, and double-descent hand-offs added on this branch are
verified current.

The gap to "best in existence" is not rigor — external calibration validates the book's
derivation-heavy identity everywhere it was checked — it is **pictures, demonstrations, and
one-directional hand-offs**. Concretely: (i) the slide decks are systematically *richer than the
prose* — at least a dozen load-bearing derivations and facts (the linear-regression gradient,
Markov's inequality, loss convexity, the MAP derivation, the 0.25¹⁰ vanishing number, the expected
MLP accuracy) exist only in slides; (ii) the prose is figure-starved while ~15 finished house-style
SVGs sit slide-only in `img/auto/` — linear-algebra.md and calculus.md have *zero* prose figures;
(iii) five sections assert quantitative claims with no code in a book whose edge is executable
math; (iv) the appendix points back at ch. 2–5 but the chapters rarely point forward — ~20 missing
`:numref:`s at specific assertion sites (autograd.md has none at all); (v) **14 confirmed errors**
(§3), all cheap to fix, including two false mathematical claims (one with an exercise assigning a
proof of the false statement) and a Kaggle ensembling cell whose comment describes the opposite of
what the code computes.

Per-chapter verdicts, one line each — details in §5:

- **Ch. 2 Preliminaries** — strong on-ramp; engineering files near best-in-class; needs figure
  promotions, Markov→Chebyshev in prose, an eigenvalue first-look, and its appendix pointers.
- **Ch. 3 Linear Regression** — excellent bones and the best exercise sets; must promote the
  slide-only gradient derivation, add the quantitative shrinkage story, and wire four hand-offs.
- **Ch. 4 Linear Classification** — already ahead of CS231n/Bishop on rigor; the one genuine
  coverage gap is evaluation beyond accuracy (confusion-matrix thread); two code-free sections need
  their demonstration cells.
- **Ch. 5 Multilayer Perceptrons** — three best-in-class assets; the single biggest gap in all four
  chapters is here (universal approximation with zero intuition + a dangling promise); two hard
  math/factual errors to fix.

Estimated total effort: ~35 S items (<30 min each), ~20 M items (<2 h), 4–5 L items — roughly two
focused weeks, or one week if the L figure-restyling items are deferred.

## 2. Cross-cutting assessment

**A. The slides outrank the prose.** A systematic asymmetry, worth a policy decision before
fixing piecemeal: prose readers (HTML *and* PDF) get a strictly weaker treatment than slide
viewers. Confirmed instances — each is a promotion candidate, not a deletion:

| Slide-only content | Where | Why it is load-bearing |
|---|---|---|
| ∂ℓ/∂w = (ŷ−y)x gradient derivation | linear-regression-scratch.md:868–889 | readers code `sgd` without seeing what `param.grad` contains |
| Markov's inequality + Markov→Chebyshev | probability.md:1573–1581 | the one-line proof that would let prose derive Chebyshev |
| Loss convexity ("every local min is global") | linear-regression.md:884–886 | the fact that makes one critical point sufficient |
| "Match the loss to the noise model" recipe | linear-regression.md:1025–1027 | the named generalization of the chapter's headline derivation |
| MAP = MLE + prior derivation (with figure) | weight-decay.md:898–928 | prose asserts, defers wholly to exercise 6 |
| Floating-point cancellation in numerical derivatives | calculus.md:580–583 | the best motivation for autograd |
| 0.25¹⁰ ≈ 10⁻⁶ sigmoid-vanishing arithmetic | numerical-stability-and-init.md:552 | quantifies the section's thesis |
| Spectral-radius compounding argument | numerical-stability-and-init.md:528–543 | slides more rigorous than prose (:81) |
| Expected accuracy ≈ 0.87 for the MLP | mlp-implementation.md:493–495 | prose never says what the reader should see |
| Inclusion–exclusion | probability.md:1322–1325 | prose Venn caption references a rule never stated |
| "Refit with chosen hyperparameters" step | kaggle-house-price.md:1016–1017 | slides claim a step the section never performs (see §3) |

**B. Figure starvation beside a full pantry.** The cheapest large win in ch. 2–3: ~15 finished
house-style SVGs exist for slides but never appear in prose (the `probability-venn.svg` /
`autograd-comp-graph.svg` promotion pattern is already established). Ch. 2 alone has 11 verified
promotion targets (§5, ch. 2 list). Ch. 4 needs three genuinely new figures (decision regions,
temperature sweep, density ratio — a chapter about a geometric object contains no picture of a
decision boundary). Ch. 5 needs two (UAT hinges, grokking). Secondary: ch. 3–5 each mix legacy
hand-drawn SVGs with house-style `mdl-*` figures against the one-style-per-chapter rule
(`capacity-vs-error.svg`, `softmaxreg.svg`, `mlp.svg`, `dropout2.svg` are the offenders).

**C. Sections that assert without running.** Five sections carry quantitative claims with no code:
probability.md after the coin demo (~600 lines, zero cells — simulate the HIV posterior, demonstrate
1/√n), generalization-classification.md (simulate √n concentration and best-of-k adaptive
overfitting), environment-and-distribution-shift.md's correction half (2-D covariate-shift demo),
backprop.md ("you can confirm every number with autograd" — no cell does), and generalization.md's
bias–variance (named, never computed — a ~12-line resampling cell decomposes the section's own
U-curve). Each proposed cell demonstrates a claim the prose currently only asserts — squarely the
house "code teaches" rule.

**D. Hand-offs are one-directional.** The appendix cites ch. 2–5 accurately; the chapters rarely
cite forward at the assertion site. ~20 missing `:numref:`s, all with grep-verified targets
(per-chapter lists in §5): 11 in ch. 2 (autograd.md has zero appendix pointers despite
`sec_mdl-matrix-calculus-autodiff` covering exactly its material), 4 in ch. 3 (SGD convergence,
duality/λ, decoupled decay, double descent), 2 in ch. 4 (log-partition convexity, Rademacher), 3 in
ch. 5 (eigenvalues/spectral radius, reverse-mode AD ×2). One appendix-side edit is implied: the
"Spectral Radius, Stability, and Deep Networks" section (mdl-eigendecomposition.md:1051) needs a
subsection label to be targetable.

**E. Load-bearing facts that live only in exercises.** Var[p̂] = p(1−p)/n (justifies every 1/√n
claim in the book), the K-fold bias question (posed as exercise, answered nowhere in the book —
verified), the MAP constants (exercise 6, with the prose stating an off-by-a-factor version, §3),
and the He "halving lemma" (exercise 3 assigns a proof of a false statement, §3). Fine pedagogy per
instance; in aggregate the prose-only reader never sees several facts the text relies on.

**F. Structure.** Ch. 2 carries the heading-rule violations: linear-algebra.md (11 flat `##` vs the
3–5 house rule), autograd.md (7), pandas.md (7) — in each case the slide decks' dividers already
found the right grouping, so the restructure is heading surgery only. Ch. 3–5 conform (one
inversion in softmax-regression.md).

**G. External calibration verdict.** The book's rigor is validated — keep the MLE derivations, the
Xavier derivation, the shift corrections, the stable-softmax chain. The competition's advantages
are (i) geometric devices (3b1b's columns-as-transformed-basis, unit-square Bayes, eigen-axis
shrinkage), (ii) named recipes (UDL's distribution→NLL menu, the broadcastable checklist),
(iii) verification cells (Karpathy's finite-difference-vs-autograd check), and (iv) a handful of
2026-baseline topics: UAT intuition device, confusion matrix/precision-recall, temperature figure,
three flavors of double descent, grokking, SAM-as-operationalized-flat-minima, the ESL shrinkage
story. Several digest headline claims were checked against source and **rebutted** — recorded in
§4 so the next pass doesn't relitigate them.

## 3. Priority fix list

### Confirmed errors (all re-verified against source; fix before anything else)

1. **numerical-stability-and-init.md:377–383** — "Var[ReLU(z)] = ½Var[z]" is **false** (for
   z∼N(0,σ²), Var[ReLU(z)] = (½ − 1/2π)σ² ≈ 0.34σ²). The true identity is second-moment:
   E[ReLU(z)²] = ½E[z²] — which is also what the He derivation actually consumes (the :387
   conclusion is correct; only the lemma is mislabeled). **Exercise 3 (:448) assigns a proof of the
   false version.** Restate both in second moments.
2. **kaggle-house-price.md:515–526** — the submission cell computes `mean(exp(preds))` (arithmetic
   mean of prices) while its comment describes log-space averaging (geometric mean) as if that were
   what the code does; the RMSLE-consistent ensemble is precisely the log-space mean the comment
   gestures at. Fix code to `exp(mean(preds))` + one sentence why.
3. **numerical-stability-and-init.md:401–402** — "`kaiming_normal_`, in fact the default for
   `nn.Linear`" is factually wrong (`reset_parameters` uses `kaiming_uniform_(a=√5)`).
4. **linear-regression-scratch.md:395–398** — "the synthetic regression dataset does not provide a
   validation dataset" — contradicted by `num_val=1000` and this very section's validation-curve
   discussion at :641–646.
5. **linear-regression.md:786** — p(ε) = ½e^{−|ε|} called "the exponential distribution"; it is the
   **Laplace** distribution (the appendix's own `sec_mdl-distributions` names it correctly — the
   book currently contradicts itself).
6. **softmax-regression.md:179–184** — the probit description "y = o + ε" with one-hot y is
   dimensionally garbled and is not the probit model. Correct form: y = argmax_i(o_i + ε_i);
   Gaussian noise gives probit, Gumbel noise gives exactly softmax (a fix that upgrades a wrong
   aside into the Gumbel-max foreshadowing of "softmax = smoothed argmax").
7. **generalization-deep.md:243–245** — "1-nearest neighbor … is consistent (eventually converging
   to the optimal predictor)" — false in general; Cover–Hart gives R\* ≤ R ≤ 2R\*(1−R\*), optimal
   only when Bayes risk is 0. (k-NN with k→∞, k/n→0 is what is consistent.)
8. **image-classification-dataset.md:11** — "today even simple linear models exceed 95%" on MNIST
   is wrong (linear ≈ 91–93%; simple *nonlinear* models exceed 95%).
9. **mlp.md:326–329** — UAT quoted in Hornik's *bounded, non-constant* form, then claimed to cover
   ReLU (unbounded). Cite Leshno et al. 1993 (non-polynomial) or weaken the wording. Related:
   :349 cites VGG for depth-trades-width — an empirical architecture paper, not a representational
   result (Montúfar 2014 / Telgarsky 2016).
10. **generalization-classification.md:143–147** — conflates *test error* with the *precision of
    its estimate* ("to reduce our test error by a factor of one hundred, collect ten thousand times
    as large a test set" — data shrinks the uncertainty, never the error).
11. **probability.md:1063–1066** — covariance magnitude equated with strength of association
    (scale-dependent; correlation ρ never defined). One-sentence fix defining ρ.
12. **environment-and-distribution-shift.md:523–525** — "sufficiently accurate ⇒ confusion matrix
    invertible" overstates (diagonal dominance, not accuracy, gives invertibility).
13. **kaggle-house-price.md:155–157** — the Kaggle *test* set called "validation data", colliding
    with the K-fold validation folds 150 lines later; plus the prose/slide contradiction on
    fold-ensembling vs refit-on-all-data (slides claim a refit step the section never performs).
14. **index.md (ch. 4):45–46** — internal link-audit residue leaked into reader-facing prose
    ("(page bot-blocked, noted)", "(paywalled, noted)").

Smaller correctness polish (fix in passing): softmax-regression-concise.md:155–158 float32 exp
reaches exactly 0 near −104, not −88 (subnormals — the section's brand is precision);
softmax-regression.md slides say entropy in "bits" where prose uses nats; scratch slide :811 gives
symmetry-breaking as the reason for Gaussian init of a *single linear layer* (wrong reason —
exercise 1 asks exactly this); calculus.md:170 "n ≠ 0" is the wrong caveat on the power rule;
autograd.md:402 "Faster:" overclaims (`y.sum().backward()` is equivalent, not faster);
mlp-implementation slide "both versions compute the same function" (different inits);
dropout.md:82's 2ⁿ-ensemble claim needs the "exact only for a single linear layer" sharpening;
linear-regression.md:232 "average (or equivalently, sum)" — equivalent only up to rescaling η.

### Highest-value additions (top 10 across all four chapters, ranked)

1. **UAT intuition sketch + `mdl-mlp-uat-hinges.svg` + region-counting demo cell** (mlp.md §5.1)
   — the single biggest gap in ch. 2–5; also resolves the dangling :350 "subsequent chapters"
   promise in place. [M+M]
2. **Confusion-matrix thread in ch. 4** — concept + "limits of accuracy" (imbalance counterexample
   cell, precision/recall at formula depth) in §4.3; computed on Fashion-MNIST in §4.4 (turns the
   asserted "shirts ≈ pullovers" into an observation); recalled, not redefined, in §4.7's
   label-shift derivation. [M+M+S]
3. **Gradient derivation into §3.4 prose** — the chapter's must-fix slide asymmetry; four lines +
   the "gradient is the error-weighted input" slogan. [M]
4. **Ridge closed form + per-direction shrinkage d_j²/(d_j²+λ) + df(λ) + demo cell on the existing
   20×200 data** (weight-decay.md) — the quantitative "why shrinkage helps" story, absent
   book-wide (verified). [M]
5. **Eigenvalue first-look stub in linear-algebra.md** (~25 lines: Av = λv, `eigvals` demo on the
   section's own symmetric matrix, spectral-norm connection) + fix numerical-stability-and-init.md:81
   (eigenvalues → singular values for rectangular Jacobians; pointer to `sec_mdl-eigendecompositions`)
   — closes the one dangling ch2↔ch5 dependency. [L+S]
6. **Markov→Chebyshev in probability.md prose** (~15 lines: state, prove in one line, derive
   Chebyshev, hand off to `sec_mdl-concentration-generalization`) + the 1/√n demonstration cell
   (Var[p̂] derivation + log–log std-vs-n plot) + the 10⁶-patient HIV simulation cell. [M+M+M]
7. **Simulation cells for the code-free sections of ch. 4** — √n-concentration + best-of-k
   adaptive-overfitting (§4.6), 2-D covariate-shift correction demo (§4.7). [M+L]
8. **Backprop upgrades** (§5.3): gradients-add-at-forks call-out at the first "+", the 6-line
   autograd verification cell the prose already promises, forward pointer to
   `sec_mdl-matrix-calculus-autodiff`, micrograd-style capstone exercise. [S+S+S+M]
9. **Loss-menu recipe named in prose** (§3.1, UDL-style distribution→NLL table + MAE/outlier demo
   cell) + **computed bias²/variance decomposition cell** and **K-fold why-K paragraph** (§3.6). [M+M+S]
10. **Ch. 2 figure-promotion pass** — 11 finished SVGs from `img/auto/` into prose + the one new
    cosine-similarity panel; ch. 4's three new geometry figures; ch. 5's grokking schematic. [S each; figures M]

Also agreed but lower rank: three-flavors double-descent paragraph + SAM sentence + Cover–Hart fix
(§5.5), inverted-dropout naming at the definition + variance-shift cite (§5.6), WILDS
axis paragraph + OOD callout (§4.7), temperature figure (§4.1), Gumbel-max exercise (§4.1),
directional-derivative one-liner + cancellation cell (§2.4), broadcasting failure-mode cell (§2.1),
depth-sweep variance cell (§5.4 — promotes exercise 4 to a teaching cell), log-space ensembling +
fold-ensemble honesty paragraph (§5.7), micrograd/3b1b/Blitzstein additions to the ch. 2/5 resource
lists.

### Cut / compress (complete list — the chapters are not bloated)

- calculus.md:192–270 — relocate/demote the matplotlib plumbing ("Visualization Utilities") to the
  end of the section; cut its prose by half. The one real "code that only draws" offender.
- linear-regression-concise.md:62–74 — paragraph duplicates the section intro nearly clause for clause.
- dropout.md:180–184 (duplicates exercise 5) and the 11-line JAX key comment (trim to 4).
- softmax-regression.md: merge duplicate exercises 1/7.1; trim the survival-modeling digression by a third.
- probability.md:584–610 height riff (~25 lines → ~10); ndarray.md:97–137 triplicated `arange`
  tab prose; linear-algebra.md:1087–1117 throat-clearing.
- softmax-regression-scratch.md:52–61 axis-sum refresher → one pointer sentence.
- environment-and-distribution-shift.md:492–498 confusion-matrix definition → recall-pointer once
  the ch. 4 thread lands; "More Anecdotes" heading.

### New d2l.bib entries required by the proposals

Leshno et al. 1993; Montúfar et al. 2014 and/or Telgarsky 2016; Cover & Hart 1967; Foret et al.
2021 (SAM); Li et al. 2018 (dropout–BN variance shift). Verified already present: Hornik.1991
(needs `:citet:` not plain text), Gal.Ghahramani.2016, Power et al. 2022, Nakkiran et al.,
Guo et al., Quiñonero-Candela.

### Digest claims checked and rebutted (do not relitigate)

- "Backprop framing is dated / shrink the symbolic derivation" — **rebutted**: graphs, the
  add/multiply gate pattern, and reverse-mode cost framing are already in backprop.md; the numeric
  example beats CS231n's. Augment (forks rule, verify cell, capstone exercise), don't replace.
- "Calibration/temperature absent from ch. 4" — **stale**: already in §4.1 prose + exercises and
  §4.3 exercise 4; needs only the temperature figure and optionally a reliability-diagram exercise.
- "No BatchNorm bridge in ch. 5 init" — **rebutted**: numerical-stability-and-init.md:421–425
  already has it, with correct pointers.
- "Dropout oversized; cut one implementation" — **rebutted**: the scratch+concise pair is the
  book's structural identity; the real trims are two small paragraphs.
- "Bias–variance never decomposed" / "add double descent to ch. 3" — **stale for this branch**:
  `sec_mdl-statistics` proves the decomposition and generalization.md:237 points there;
  double descent is deliberately consolidated in §25.5 (ch. 3 gets a second pointer, no treatment).
- "Add UDL's regularizer taxonomy to ch. 3" — **rebutted**: the other mechanisms don't exist yet at
  that point in the book.
- "Locally weighted regression" — **rebutted**: a nonparametric detour serving neither the DL arc
  nor the spiral.
- In-chapter double-descent code (§5.5) — **rebutted**: appendix §25.5 owns the 25-line
  reproduction; an epoch-wise exercise is the right vehicle.

## 4. Placement and sequencing

Suggested execution order if/when this review is executed (mirrors the appendix-polish workflow —
one chapter per work package, errors first within each):

1. **Errors-only pass across all four chapters** (items 1–14 above + the polish list) — one compact
   package, no structural risk, immediately committable.
2. **Ch. 3** (gradient promotion, loss menu, shrinkage story, K-fold, bias²/variance cell, 4
   hand-offs) — highest density of M-items with no new-file mechanics.
3. **Ch. 4** (confusion-matrix thread, beyond-accuracy, simulation cells, WILDS paragraph, 3
   figures via a new `gen_mdl_classification_figures.py` additions).
4. **Ch. 5** (UAT sketch + figure + demo cell, backprop upgrades, depth-sweep cell, DD flavors +
   grokking figure, dropout naming, Kaggle fixes).
5. **Ch. 2** (figure promotions, Markov/Chebyshev, HIV + 1/√n cells, eigenvalue stub, heading
   restructures, 11 pointers) — largest but most mechanical package.
6. **Slide reconciliation pass** at the end (the promoted content changes what slides should
   emphasize; also fixes the slide-side errata noted in §5: bits/nats, Jacobian label,
   symmetry-breaking reason, §-kicker on the ch. 4 generalization deck, kaggle refit claim).

Appendix-side micro-edits implied: add a `subsec_` label to "Spectral Radius, Stability, and Deep
Networks" (mdl-eigendecomposition.md:1051); optionally have `sec_mdl-svd-low-rank`'s Tikhonov
"revisit" line point at the new weight-decay shrinkage passage once it exists.

Framework note: all proposed new cells are CPU-trivial (NumPy/small tensors); per the standing
workflow, capture pytorch on this machine and defer tf/jax/mxnet re-execution to the GPU box.

## 5. Chapter-by-chapter reviews

## Chapter 2 · Preliminaries (`chapter_preliminaries/`)

### Chapter verdict

Chapter 2 is already a strong on-ramp — the engineering sections (ndarray, pandas, lookup-api) are close to best-in-class, the HIV worked example in probability is genuinely superior to Blitzstein's equivalent for this audience, and autograd's verify-everything cell discipline is exactly d2l's executable edge. It is not yet "best in existence" for three reasons. **(1) The prose is figure-starved while a full set of polished SVGs sits unused in `img/auto/`**: linear-algebra.md and calculus.md contain *zero* figures in prose, yet the slide decks reference ~20 finished illustrations (dot product, matvec, secant→tangent, gradient field, broadcasting, density-vs-mass, Markov...) that HTML/PDF readers never see — promoting even half of these is the single cheapest large win. **(2) Load-bearing content hides in slides and exercises**: Markov's inequality, the floating-point-cancellation caveat about numerical derivatives, and inclusion–exclusion exist only in slide decks; the variance-of-the-coin-estimator (the fact that *justifies* the asserted 1/√n rate) exists only as an exercise. **(3) Hand-offs to the polished math appendix are thin and asymmetric**: linear-algebra, calculus, and probability each carry exactly one chapter-level pointer in their Discussion, but the specific assertion-sites (steepest ascent, Cauchy–Schwarz/cosine, matrix-calc identities, spectral norm, Chebyshev, forward/reverse cost) lack targeted `:numref:`s, and autograd.md has none at all. Secondary: linear-algebra.md runs 11 flat `##` sections against the 3–5 house rule, and eigenvalues remain undefined anywhere in ch2–5 despite ch5 using them.

### Section: index.md (§2.0)

**Strengths.** The seven "survival skills" framing (index.md:4-20) is charming and accurate. The Resources block (index.md:38-60) is well-curated for the engineering half: McKinney, VanderPlas, the NumPy Nature paper, official doc hubs.

**Issues.**
1. index.md:42-47 [weak] — The book list covers data-wrangling and general math (Deisenroth) but has *nothing* for probability or visual linear algebra, the two topics self-studiers most often need scaffolding for. Given the stated competition (3Blue1Brown, Blitzstein), their absence is conspicuous.
2. index.md:59 [polish] — Only PyTorch's docs are listed under documentation; the other three frameworks' doc hubs appear only in lookup-api.md:22-27. Either say "see :numref: for the other frameworks" or add them.

**Proposed improvements.**
1. [S] Add to "Courses and video lectures": 3Blue1Brown *Essence of Linear Algebra* and *Essence of Calculus* (free, visual, the perfect complement to this chapter's code-first treatment), and Blitzstein & Hwang *Introduction to Probability* (free PDF + lectures) under Books.
2. [S] One-line cross-reference for the non-PyTorch doc hubs.

### Section: ndarray.md (§2.1)

**Strengths.** Excellent pacing and motivation throughout; the added connective tissue is first-rate: the float32 rationale (ndarray.md:163-174), reshape-shares-storage forward hook (:263-266), symmetry-breaking motivation for random init (:319-322), the seeds/reproducibility paragraph with JAX's explicit-key stance (:351-357), and the precise right-aligned broadcasting rule (:660-664). The saving-memory section explains *why* (parameter updates, aliasing) not just *how* (:728-740). Exercises 3–6 (:909-912) have real predictive-then-verify structure. JAX asymmetries (immutability, `.at[]`, buffer donation) are handled honestly per tab.

**Issues.**
1. ndarray.md:395-401 [polish] — "Finally, we can access whole ranges… Finally, when only one index…": two consecutive "Finally" sentences.
2. ndarray.md:710 vs :717/:723 [polish] — Prose introduces `Y = X + Y`, then the very next sentences and the code cell use `Y = Y + X`. Pick one.
3. ndarray.md:159 [polish] — JAX tab creates `x = jnp.arange(12)` (int32) while pytorch/tf explicitly request float32; the dtype paragraph at :163-174 ("that is why several cells in this section request `dtype=...`") is then illustrated by only two of four tabs. Add `dtype=jnp.float32` for parity.
4. ndarray.md:899 [polish] — This file closes with `## Summary`; every sibling closes with `## Discussion`. Standardize.
5. ndarray.md:640-704 [weak] — Broadcasting, the #1 silent-bug generator for beginners, is prose-only in the chapter body while a finished diagram (`img/auto/ndarray-broadcasting.svg`) exists for slides. Same for reshape (`ndarray-reshape.svg`).

**Proposed improvements.**
1. [S] Promote `ndarray-broadcasting.svg` and `ndarray-reshape.svg` from `img/auto/` into the prose (the `probability-venn.svg` copy-to-`img/` pattern is already established), with captions and `:label:`s.
2. [S] Fix issues 1–4 (wording, dtype, heading).
3. [M] Add one 3-line cell after :704 demonstrating the *failure* mode: try `(3,2) + (2,3)`, catch/show the error, and tie it to the alignment rule — currently that teaching moment lives only in exercise 5 (:911). This converts an assertion ("raises an error rather than guessing", :664) into a demonstration.

### Section: pandas.md (§2.2)

**Strengths.** The strongest rewrite in the chapter relative to upstream d2l. The decision-centric framing (:11-24), "look before you transform" (:71-95), measuring missingness before choosing a strategy (:117-139), the three-response taxonomy with costs (:121-133), the indicator-strategy connection to `dummy_na` (:161-163), the standardization + leakage warning (:196-202), and the honest "real preprocessing gets hairy" discussion (:259-275) form a genuinely useful mental model, not a pandas cookbook. Exercise 4 (:282) on leakage and zero variance is excellent.

**Issues.**
1. pandas.md:170 [polish] — "RoofType has become three columns (`Slate`, `Tile`, and `nan`)": `get_dummies` names them `RoofType_Slate`, `RoofType_Tile`, `RoofType_nan`. State the actual names; readers grep output for them.
2. pandas.md:26-204 [polish] — Seven flat `##` sections for ~250 prose lines. The slide deck already found the right 3-act grouping (Load & Look / Clean & Encode / Scale & Convert, pandas.md:333-451).
3. pandas.md:95 [polish] — "Keep that in mind; it returns below" — good foreshadowing, but the payoff (:179-187) never explicitly recalls it ("as promised" would close the loop).
4. pandas.md:189-193 [weak] — The code standardizes with whole-dataset statistics and the *text* flags it as leakage to fix "in a later chapter" (:196-202), but the promised later chapter is unnamed. `:numref:`sec_kaggle_house`` is already cited at :256 for more data-processing; cite it (or the model-selection section) here at the leakage warning, where the reader actually cares.

**Proposed improvements.**
1. [S] Fix the dummy column names; add the forward `:numref:` at the leakage paragraph.
2. [M] Restructure to 3 top-level `##` (Load & Inspect / Clean & Encode / Scale & Convert) with the current sections demoted to `###`, mirroring the slides — brings the file inside the house 3–5 rule with no content changes.
3. [S] Promote `img/auto/pandas-pipeline.svg` (raw file → tensor pipeline) into the intro — it is exactly the "path is paved with decisions" picture the opening paragraph paints in words.

### Section: linear-algebra.md (§2.3)

**Strengths.** Clean definitions, correct notation discipline (order vs dimensionality vs rank disambiguation, :203-213, is better than most textbooks), correct arithmetic everywhere I checked (dot product, matmul dims :903-907, cubic-vs-quadratic cost claim :1130-1131). The exercise set is the best in the chapter: the matrix-chain association/memory questions (:1150-1151) and the stack-then-slice tensor question (:1152) are genuinely instructive. The Discussion's single pointer to `chap_mdl-linear-algebra` (:1112) exists.

**Issues.**
1. linear-algebra.md:43-1087 [weak] — **11 flat `##` sections** (Scalars, Vectors, Matrices, Tensors, Basic Properties, Reduction, Non-Reduction Sum, Dot Products, Matrix–Vector, Matrix–Matrix, Norms) against the 3–5 house rule; confirmed worst structural offender in the chapter. The slide deck's four dividers (Objects / Arithmetic & Reduction / Products / Norms, :1202-1490) are the obvious grouping.
2. linear-algebra.md:721-723 [weak] — "After normalizing two vectors to have unit length, the dot products express the cosine of the angle between them." Asserted with no formula, no name, no forward pointer; Cauchy–Schwarz (which makes cos θ = x·y/(‖x‖‖y‖) legitimate, i.e. bounds the ratio in [−1,1]) is proved at `chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md:219` but never referenced. Also "the dot products express" is a grammatical slip. Confirms and extends the prior finding.
3. linear-algebra.md:948-958 [weak] — Norm axioms stated; never once verified for ℓ1/ℓ2 in code or exercise. d2l's edge is exactly a 4-line cell checking homogeneity and the triangle inequality on random vectors — currently missing, and no exercise asks either.
4. linear-algebra.md:1032-1039 [weak] — Spectral norm is named and motivated ("how much longer could Xv be relative to v") then abandoned with no pointer. Its actual development (largest singular value) lives in `sec_mdl-svd-low-rank`. Confirms prior finding.
5. **Eigenvalues never defined** [weak] — nothing in this file (nor anywhere in ch2–5) defines Av = λv, yet chapter_multilayer-perceptrons/numerical-stability-and-init.md:81 reasons about "a wide variety of eigenvalues" and its exercise :450 asks for eigenvalue bounds. Verified gap.
6. linear-algebra.md:768-775 [weak] — "we can represent rotations as multiplications by certain square matrices" — asserted, never shown, no figure. One rotation-matrix cell would demonstrate it in 3 lines.
7. linear-algebra.md:113-135 [polish] — Vectors are introduced purely as data records (loan applicants, patients); the geometric picture (arrow, length, direction) never appears in prose — yet `img/auto/linear-algebra-vector.svg`, `-dot.svg`, `-matvec.svg`, `-matmul.svg`, `-norms.svg`, `-transpose.svg`, `-reduce-axes.svg`, `-tensor4d.svg` all exist as finished slide figures. Prose figure count for this 1100-line math section: **zero**.
8. linear-algebra.md:719 [polish] — `$\left(\sum_{i=1}^{n} {w_i} = 1\right)$` — parentheses around an equation mid-sentence read oddly; drop `\left(...\right)`.
9. linear-algebra.md:1136 [polish] — Summary says "common matrix norms include the *spectral* and Frobenius norms" — the section never actually defined the spectral norm, only gestured at it; the summary overstates coverage.

**Proposed improvements.**
1. [M] Restructure to 4 `##`: "The Objects" (Scalars/Vectors/Matrices/Tensors as `###`), "Arithmetic and Reductions" (Basic Properties/Reduction/Non-Reduction as `###`), "Products" (Dot/Matrix–Vector/Matrix–Matrix as `###`), "Norms". Pure heading surgery; no cell IDs move.
2. [S] Promote 4 slide figures into prose: `linear-algebra-dot.svg` (at :660), `linear-algebra-matvec.svg` (at :747), `linear-algebra-matmul.svg` (at :898), `linear-algebra-norms.svg` (ℓ1/ℓ2 unit balls, at :995).
3. [M] At :723, add the general cosine formula plus two sentences: "Why is this ratio always in [−1,1]? That is the Cauchy–Schwarz inequality, proved in :numref:`sec_mdl-geometry-linear-algebraic-ops`" — and a 3-line cell computing the angle between two vectors and verifying `|dot| <= norm*norm` on random draws. Turns the section's one hand-wave into a demonstration + hand-off.
4. [M] After :1030 (ℓp norms), add a 5-line cell verifying the three axioms numerically for ℓ2 on random `u, v` (`norm(a*u) == abs(a)*norm(u)`, `norm(u+v) <= norm(u)+norm(v)`), with one sentence noting the triangle inequality for ℓ2 *is* Cauchy–Schwarz in disguise (pointer as in #3).
5. [L] Add a ~25-line "Eigenvalues: a first look" `###` under Norms (or a short `##` if restructure #1 lands): define Av = λv for square A, show `torch.linalg.eigvals` on the symmetric 3×3 already defined at :313, state (no proof) that for symmetric matrices the spectral norm equals max|λ| and that repeated multiplication amplifies/damps by λ — with pointers to :numref:`sec_mdl-eigendecompositions` and forward mention that ch5's stability analysis uses exactly this. This closes the ch5 hand-off gap at on-ramp depth.
6. [S] Fix :1136 summary wording; add spectral-norm pointer at :1038 to `sec_mdl-svd-low-rank`.
7. [S] Rotation demo: 3-line cell at :775 multiplying [[cosθ,−sinθ],[sinθ,cosθ]] into a vector, or drop the rotation claim.

### Section: calculus.md (§2.4)

**Strengths.** The Archimedes opening (:9-28) with its committed SVG is a model intro. Structure is house-compliant (4 content `##`). The numerical-limit cell (:157-160) is the right first computation. The chain-rule-to-matrix-product passage (:380-393) — deriving that the multivariate chain rule *is* a matvec, hence "why linear algebra is integral" — is the intellectual high point and is done well. The Discussion's forward pointer to `chap_mdl-calculus` (:424-430) exists, satisfying the spiral for derivative rules at chapter level. Exercise 11 (:455-460), a single gradient step with η sweeps, is a perfect bridge to optimization.

**Issues.**
1. calculus.md:330-338 [weak] — Gradient = steepest ascent asserted ("Geometrically, the gradient has a crucial interpretation") with no argument and **no pointer**, although the appendix proves it via Cauchy–Schwarz (`mdl-multivariable-calculus.md:234`, Proposition "steepest ascent and descent"). Worse, prose never defines the directional derivative, so "direction in which the function grows fastest" is not even interpretable precisely. Confirms prior finding; the missing pointer is the minimal fix.
2. calculus.md:343-348 [weak] — Matrix-calc identities tabulated with no derivation and no pointer (`subsec_mdl-matrix-identities` at mdl-matrix-calculus-autodiff.md:270 derives them). Additionally, ∇ₓAx = Aᵀ silently extends "gradient" to vector-valued functions — a matrix-valued object — one paragraph after the gradient was defined only for scalar f (:313-323). One sentence establishing the convention (or restricting the bullet list to the scalar-valued x^TAx and ‖x‖² identities) is needed; as written a careful reader is stuck.
3. calculus.md:580-583 (slides) [weak] — **Slide-only load-bearing content**: "Push h far smaller and floating-point cancellation eventually corrupts the quotient — a reason autograd computes derivatives analytically" appears *only* in the slide deck. This is the best available motivation for the next section and prose readers never see it. The prose cell (:158) stops at h=1e-5, just before the phenomenon appears.
4. calculus.md:192-270 [weak] — "Visualization Utilities" devotes ~80 lines (a full quarter of the prose) to matplotlib plumbing (`set_axes`, the shape-munging `plot`) between the derivative and the gradient. It must exist somewhere for `#@save`/`make lib`, but per the house "code teaches" rule it is the weakest stretch of the chapter — the `plot` body (:244-270) teaches nothing about calculus and interrupts the arc at its most delicate hand-off (1-D → multivariate).
5. calculus.md:170 [polish] — "d/dx xⁿ = n xⁿ⁻¹ for n ≠ 0": the rule holds for n = 0 too; the caveat that matters (x > 0 for non-integer n) is not the one stated.
6. calculus.md:758 (slides) [polish] — Slide labels A ∈ ℝⁿˣᵐ "(the Jacobian)"; with A_{ij} = ∂u_j/∂x_i (prose :383) it is the *transpose* of the conventional Jacobian. Prose avoids the name; slide misuses it. Align both by naming A explicitly "the transpose of the Jacobian ∂u/∂x" with a pointer to `subsec_mdl-jacobian`.
7. calculus.md:79-190 [polish] — No secant→tangent figure in prose while `img/auto/calculus-secant-tangent.svg` exists; no gradient-field/level-set figure at :330 while `calculus-gradient-field.svg` and `calculus-partial-slices.svg` exist. The steepest-ascent claim, in particular, begs for the level-set picture.

**Proposed improvements.**
1. [S] Add `:numref:`sec_mdl-multivariable_calculus`` at :338 ("proved via Cauchy–Schwarz in …") and `:numref:`sec_mdl-matrix-calculus-autodiff`` (or the `subsec_mdl-matrix-identities` anchor) after :348.
2. [M] Extend cell `#calculus-derivatives-and-differentiation-2` discussion: add one prose paragraph + extend the h-sweep to 1e-13 in a second cell (or widen the range in-place is barred — IDs are stable, so add a *new* cell) showing the quotient degrade, importing the slide's cancellation point into prose. This is a new-cell addition, no renumbering.
3. [M] Fix issue 2: one sentence before :343 — "For vector-valued u = Ax we collect the partials into a matrix; by convention we write ∇ₓ(Ax) = Aᵀ (the transpose of the Jacobian, developed in :numref:`sec_mdl-matrix-calculus-autodiff`)" — and fix the slide's Jacobian label to match.
4. [M] Demote "Visualization Utilities" to a `###` at the *end* of the section titled "Plotting Utilities for This Book" (after Chain Rule, before Discussion), moving the tangent-line plot cell up to close "Derivatives and Differentiation" — the reader meets the payoff plot first, the plumbing later. (Cells move but IDs are stable; slides reference by ID so nothing breaks.)
5. [S] Promote `calculus-secant-tangent.svg` (at :104) and `calculus-gradient-field.svg` (at :330) into prose.
6. [S] Add a two-line directional-derivative definition before :330 — "the rate of change along a unit vector u is ∇f·u; over all unit u this is maximized when u aligns with ∇f" — which makes the steepest-ascent claim well-posed *and* quietly uses the §2.3 cosine fact (nice intra-chapter spiral).

### Section: autograd.md (§2.5)

**Strengths.** Best code discipline in the chapter: every concept lands with a verification cell (`x.grad == 4*x` :247-265, detach `== u` and `== 2x` :446-520, control flow `== d/a` :703-721, second derivative values in comments :756-781). The detach mini-derivation (:430-444, u = x² constant ⇒ ∂z/∂x = u vs 3x² undetached) is correct and exactly intuition-first. The history paragraph (:32-48, Wengert/Linnainmaa/Speelpenning/Griewank) is scholarly without dragging. The new committed figure `autograd-comp-graph.svg` (:178) with its forward/backward caption is precisely the promote-slide-figure pattern the rest of the chapter should copy. Framework asymmetries (PyTorch accumulation :275-289, JAX functional stance :546-550) handled candidly.

**Issues.**
1. autograd.md — **Zero pointers to the math appendix anywhere in the file** (verified by grep), despite `sec_mdl-matrix-calculus-autodiff` containing the full development of exactly this material (`subsec_mdl-forward-mode` :520, `subsec_mdl-reverse-mode` :684). The only forward pointer is to `sec_backprop` (:184). [weak]
2. autograd.md:737-740 [weak] — MXNet higher-order gradients "beyond the scope of this introduction" with no pointer and no MXNet cell in `#autograd-higher-order-derivatives-1` (pytorch/tf/jax only). Confirms prior finding; a pointer to `sec_mdl-matrix-calculus-autodiff` (which is framework-agnostic math) is the honest fix.
3. autograd.md:783-798 [weak] — "Forward versus Reverse Mode" is the only code-free section in the file. The cost claim ("reverse is cheap when many inputs, few outputs") is asserted; neither a one-line counting argument (one reverse sweep per output vs one forward sweep per input) nor a demo appears; "The exercises explore this trade-off" (:798) defers to exercises 6–7, which do it by hand on a scalar — so the *quantitative* claim rests nowhere.
4. autograd.md:73-798 [polish] — 7 content `##` vs the 3–5 rule. Natural grouping already exists in the slides' dividers (:887-1031): Mechanics / Working with Gradients / Dynamic Graphs (+Beyond).
5. autograd.md:402 [polish] — Comment `# Faster: y.sum().backward()` overclaims; `y.sum().backward()` is equivalent (one extra sum node), not meaningfully faster than `backward(gradient=ones)`. "Equivalently:" would be accurate.
6. autograd.md:374-376 [polish] — "for reasons that will become clear later, this argument … is named `gradient`" — the promised clarification never arrives anywhere in the book path named here; either point to :numref:`sec_backprop` explicitly or cut the tease.

**Proposed improvements.**
1. [S] Add appendix pointers: at :185 alongside `sec_backprop`, and in "Forward versus Reverse Mode" (:798) → ":numref:`sec_mdl-matrix-calculus-autodiff` derives both modes and their costs"; at :739 for MXNet higher-order.
2. [M] Add the counting argument to :792-795 in three sentences: a network with n inputs and m outputs needs m reverse sweeps or n forward sweeps to fill the Jacobian; a scalar loss (m=1) over millions of parameters makes reverse mode the only sane choice; forward mode wins for Hessian-vector products and per-input sensitivities. Optionally [M] a JAX-tab-only cell timing `jacrev` vs `jacfwd` on f: ℝ¹⁰⁰⁰→ℝ³ — the single best use of d2l's executable edge available in this file.
3. [M] Regroup to 3 `##` (Mechanics: A Simple Function + Non-Scalar Backward; Controlling the Graph: Detaching + Turning Off Tracking; Beyond the Basics: Control Flow + Higher-Order + Forward-vs-Reverse), demoting current sections to `###`. Heading-only surgery.
4. [S] Fix the "Faster" comment and the ":374 reasons will become clear" dangler.

### Section: probability.md (§2.6)

**Strengths.** The strongest math section: axioms → P(∅)=0 derivation (:511-524), Bayes derived not asserted (:711-718), variance identity expanded (:1012-1016), vᵀΣv = Var[vᵀx] (:1054-1062) — all correct (I re-verified every number in the HIV example: 0.011485, 0.1306, 0.0003/0.98, 0.00176955, 0.8307 all check; investment expectation 1.8, utility 0.7, variance 8.36 all check). The five committed figures (venn :526, joint-grid :697, explaining-away :818, natural-frequencies :867, bayes-update :915) are the model for the rest of the chapter; the natural-frequencies rendering of the base-rate fallacy is better pedagogy than most probability texts. The convergence plot (:363-449) earns its space. Exercises ramp beautifully from examples to the correlated-tests problem (:1172-1175) and the Markowitz portfolio (:1176-1180).

**Issues.**
1. probability.md:1573-1581 (slides) [weak] — **Markov's inequality exists only in the slide deck**, including its punchline "apply it to (X−μ)² to get Chebyshev" — the one-line proof that would let prose *derive* rather than cite Chebyshev. Prose readers get Chebyshev bare (:1138-1145) with a Wikipedia link instead of the book's own appendix. Confirms prior finding.
2. probability.md:1138-1145 [weak] — Chebyshev is stated in the *Discussion* — structurally an afterthought — and carries no pointer to `sec_mdl-concentration-generalization`, whose entire opening (mdl-concentration-generalization.md:65-89, "From Chebyshev to Chernoff") is the designed continuation.
3. probability.md:348-358 [weak] — The 1/√n rate is asserted with a citation; its two-line justification (Var[p̂] = p(1−p)/n) exists only as exercise 3 (:1165-1168). A load-bearing quantitative fact of the whole book (echoed again at :1104-1116) rests on an exercise.
4. probability.md:465-1069 [weak] — After the coin-toss opening, the remaining ~600 prose lines contain **zero code cells**. The HIV example, expectations, variance, and covariance are all pen-and-paper — in the book whose distinctive edge is executable math. The single highest-value addition in the chapter: simulate the HIV posterior.
5. probability.md:1063-1066 [weak] — "a larger positive value means that they are more strongly correlated": covariance magnitude is scale-dependent; strength of association is the *correlation* ρ = Σᵢⱼ/(σᵢσⱼ), never defined. As written this is the one overclaim in an otherwise careful file.
6. probability.md:526-527 [polish] — The Venn figure's caption invokes "the *inclusion–exclusion* rule" but P(A∪B) = P(A)+P(B)−P(A∩B) never appears in prose (it exists in slides :1322-1325 and implicitly as exercise 5 :1170). Caption references a rule the text never states.
7. probability.md:960 [polish] — `E[X] = ∫ x dp(x)` vs `∫ f(x) p(x) dx` five lines later (:965): two different measure notations for the same object with no comment; use `p(x)dx` in both.
8. probability.md:118-465 [polish] — 6 content `##` is fine, but "A More Formal Treatment" and "Random Variables" are both short and could nest under one "The Formal Language" `##` (as the slides do, :1299-1305), bringing the file to 5.

**Proposed improvements.**
1. [M] Import Markov into prose: in the Discussion (or better, at the end of Expectations), state Markov for nonnegative X with its 3-line proof (E[X] ≥ a·P(X≥a)), derive Chebyshev by applying it to (X−μ)², then hand off: "sharper bounds — Chernoff, Hoeffding — and what they say about generalization are developed in :numref:`sec_mdl-concentration-generalization`." ~15 lines, fixes issues 1, 2, and the odd placement at once.
2. [M] After :358, add two prose sentences deriving Var[p̂] = p(1−p)/n (uses only the variance identity the section proves later — or defer the derivation and state it), plus a new code cell: simulate 1000 batches at n ∈ {10, 100, 1000, 10000}, plot std of p̂ vs n on log–log and observe slope −1/2. Turns the asserted 1/√n into a demonstrated law; exercise 3 then becomes the analytic follow-up.
3. [M] New cell after the HIV analysis (:909): 6 lines sampling 10⁶ patients (H ~ Bernoulli(0.0015), D₁,D₂ from the conditional tables), then computing `mean(H[D1&D2])` and comparing to 0.8307. The frequency-simulation *is* the natural-frequencies figure made executable — the single best marriage of the section's math and d2l's edge.
4. [S] Fix the covariance overclaim: define ρ in one sentence (or say "more strongly *associated*, once rescaled by the standard deviations — the *correlation*") at :1063-1066.
5. [S] State inclusion–exclusion in prose right before the Venn figure (:524) — it is a 2-line consequence of additivity and makes the caption honest; exercise 5 still works as practice.
6. [S] Notation fix at :960; heading merge per issue 8 if desired [S].

### Section: lookup-api.md (§2.7)

**Strengths.** The reconception around the discover → inspect → read → verify loop (:29-34, with committed figure) elevates a filler section into an actual skill. The coding-assistant paragraph (:162-171) is exactly the right 2026 stance: use it, then verify with the same loop. Exercises (:175-187) operationalize the loop, including running an assistant's answer through it. Section is short, house-compliant, done.

**Issues.**
1. lookup-api.md:27 [polish] — MXNet docs links pin version 1.9.1 while the repo ships a custom MXNet 2.0 wheel and the book uses the `np`/`npx` API; the 1.9.1 pages document the legacy `mx.nd` world more prominently. At minimum note the version caveat.
2. lookup-api.md:6 [polish] — `# Documentation` carries no `:label:`, the only section in the chapter without one; index.md prose (:19-20) refers to it descriptively because it cannot `:numref:` it.

**Proposed improvements.**
1. [S] Add `:label:`sec_lookup_api`` and fix/annotate the MXNet doc link.

### Chapter-level coverage

**Add.**
- *Eigenvalue stub* in linear-algebra.md (see §2.3 proposal 5) — the one genuine coverage hole, given ch5's dependence and the appendix hand-off map. [L]
- *Markov→Chebyshev* in probability prose (§2.6 proposal 1). [M]
- *Directional derivative* one-liner in calculus (§2.4 proposal 6). [S]
- *Numerical-derivative cancellation* in calculus prose (§2.4 proposal 2). [M]
- Nothing else: named distributions, CDFs, integrals, SVD, convexity are all correctly deferred to the appendix — resist adding them.

**Cut/compress.**
- calculus.md:192-270 Visualization Utilities — keep the `#@save` code (required for `make lib`) but demote/relocate and cut its prose by half (§2.4 proposal 4).
- probability.md:584-610 the height-measurement riff on continuous variables runs ~25 lines for one idea; compress to ~10.
- linear-algebra.md:1087-1117 Discussion's "we reserve the right to introduce more mathematics later" paragraph is throat-clearing; 3 sentences suffice ahead of the (good) appendix pointer.
- ndarray.md tf/mxnet/pytorch triplicated `arange` prose (:97-137) — three near-identical `:begin_tab:` blocks differing in one word; collapse to a shared paragraph + one-word tab notes.

**New figures (chapter-wide list).** The cheapest wins are *promotions* of already-committed slide SVGs from `img/auto/` into prose (pattern established by `probability-venn.svg` and `autograd-comp-graph.svg`); all exist today:
1. `ndarray-broadcasting.svg` → ndarray.md :660 (3×1 and 1×2 stretching to 3×2, arrows on the virtual copies).
2. `ndarray-reshape.svg` → ndarray.md :263 (same 12 elements re-wrapped, strides annotated).
3. `pandas-pipeline.svg` → pandas.md intro (CSV → inspect → impute → one-hot → standardize → tensor, one box per stage).
4. `linear-algebra-dot.svg` → :660 (elementwise-multiply-then-sum, plus projection reading).
5. `linear-algebra-matvec.svg` → :747 (Ax as one dot product per row, row-highlighted).
6. `linear-algebra-matmul.svg` → :898 (cᵢⱼ as row-i×col-j intersection).
7. `linear-algebra-norms.svg` → :995 (ℓ1 diamond vs ℓ2 disk unit balls).
8. `calculus-secant-tangent.svg` → :104 (secant pivoting into tangent as h→0).
9. `calculus-gradient-field.svg` → :330 (level sets with ∇f arrows orthogonal, steepest ascent).
10. `probability-density.svg` → :604 (mass on points vs density with shaded interval integral).
11. `probability-markov.svg` → alongside the proposed Markov prose (threshold a, shaded tail, lever E[X]/a).
Genuinely *new* figure worth generating (mdl-figure skill): **(12)** a cosine-similarity panel for linear-algebra :721 — two unit vectors at angle θ with dot product = cos θ annotated for θ ∈ {0°, 60°, 90°, 150°} — nothing in `img/auto` covers it and it anchors both the §2.3 hand-wave and the §2.4 steepest-ascent argument.

**Hand-off fixes (missing `:numref:` pointers; all targets verified to exist).**
1. calculus.md:338 (steepest ascent) → `sec_mdl-multivariable_calculus` (proof at mdl-multivariable-calculus.md:234). **Missing.**
2. calculus.md:348 (matrix-calc identities) → `sec_mdl-matrix-calculus-autodiff` / anchor `subsec_mdl-matrix-identities`. **Missing.**
3. calculus.md:184 (derivative rules "taken for granted", cf. exercise 1-2) → `sec_mdl-single_variable_calculus` (table at `sec_mdl-derivative_table`). **Missing** (chapter-level pointer at :428 exists but the assertion site should carry it).
4. linear-algebra.md:723 (cosine/Cauchy–Schwarz) → `sec_mdl-geometry-linear-algebraic-ops`. **Missing.**
5. linear-algebra.md:1038 (spectral norm) → `sec_mdl-svd-low-rank` (spectral norm = σ_max; also `subsec_mdl-condition-number`). **Missing.**
6. linear-algebra.md (proposed eigenvalue stub) → `sec_mdl-eigendecompositions`. **Missing** (whole-chapter pointer at :1112 exists and is good).
7. autograd.md:739 (MXNet higher-order "beyond the scope") → `sec_mdl-matrix-calculus-autodiff`. **Missing** — file currently has zero appendix pointers.
8. autograd.md:798 (forward vs reverse cost) → `sec_mdl-matrix-calculus-autodiff` (`subsec_mdl-forward-mode`/`subsec_mdl-reverse-mode`). **Missing.**
9. probability.md:1145 (Chebyshev) → `sec_mdl-concentration-generalization` (its §"From Chebyshev to Chernoff" is the designed continuation). **Missing** — the Wikipedia link should not be the only reference when the book proves it in-house.
10. probability.md:358 (CLT / 1/√n) → `sec_mdl-distributions` (CLT at mdl-distributions.md:656) or `sec_mdl-statistics`. **Missing.**
11. pandas.md:202 (leakage / train–test splits "a later chapter") → name the section (`sec_kaggle_house` already cited at :256, or the model-selection section). **Vague.**
12. Present and correct (no action): linear-algebra.md:1112 → `chap_mdl-linear-algebra`; calculus.md:428 → `chap_mdl-calculus`; probability.md:1149-1155 → `chap_mdl-probability-statistics` + `sec_mdl-distributions` + `sec_mdl-maximum_likelihood` + `chap_mdl-information-theory`.

## Chapter 3 · Linear Regression (`chapter_linear-regression/`)

### Chapter verdict
This is already a strong chapter — the spiral with the math appendix works (normal equations fully
derived with the projection reading, Gaussian-MLE⇒MSE fully derived, bias–variance *named* here and
*proved* in §25.4), the OO-design section earns its keep with a genuinely distinctive
async-plotting/compilation discussion, and the polynomial-overfitting demo now computes the U-curve
instead of merely asserting it. Four things keep it short of "best in existence." First, a cluster of
missing hand-offs: the new appendix sections (§24.2 SGD convergence, §24.4 duality/λ-as-multiplier,
§24.2's decoupled-weight-decay subsection, §25.5 double descent) all point *at* this chapter but the
chapter never points back, so the reader who wants the proof is stranded. Second, load-bearing
content lives only in slides: the hand-derived gradient ∂ℓ/∂w=(ŷ−y)x, the convexity of the loss,
the "match the loss to the noise model" recipe, and the MAP=MLE+prior derivation all appear in slide
decks but not prose — prose readers implement `sgd` without ever seeing the gradient they are coding.
Third, weight decay stops at the picture: no ridge closed form, no per-direction shrinkage
d_j²/(d_j²+λ), no effective-dof — the *quantitative* "why shrinkage helps" story is absent from the
whole book. Fourth, two outright errors (a "does not provide a validation dataset" claim contradicted
by the code, and Laplace mislabeled "exponential"). All are fixable at S/M cost; the chapter's bones
are excellent.

---

### Section: linear-regression.md (§3.1)

**Strengths.** The house-price cold open is concrete and returns as the running example. The analytic
solution is complete and *correct* (∂w‖y−Xw‖² = 2Xᵀ(Xw−y), verified), with the column-space/
projection reading at :278-288 and a live pointer to `sec_mdl-geometry-linear-algebraic-ops`
(verified resolving). The Gaussian-MLE⇒MSE derivation (:616-660) is airtight, including the correct
NLL at :649. The framework-MSE-omits-½ warning (:212-215) and the inference-vs-prediction
terminology aside (:419-422) are exactly the practitioner-grade touches a top course wants. The
exercise set is one of the book's best (median/MAE, rank deficiency + ridge preview, Laplace noise,
log-price, Poisson).

**Issues.**
1. :786 — [error] "the noise model … is the exponential distribution. That is, p(ε)=½exp(−|ε|)".
   That density is the **Laplace** distribution (the exponential is λe^{−λx} on x≥0). Inherited from
   d2l-en, but at this bar it must be fixed; the appendix's own `sec_mdl-distributions` (:502-509)
   names it Laplace, so the book currently contradicts itself.
2. :232 — [weak] "we simply average (or equivalently, sum)". Only equivalent up to rescaling η by n —
   and the book *relies* on the distinction later (scratch :214-216 "we do not need to adjust the
   learning rate against the batch size"; concise exercise 1 asks exactly this). Say "or, up to a
   rescaling of the learning rate, sum".
3. :299-411 — [weak] Minibatch SGD is stated with zero convergence content and **no pointer** to the
   appendix that now owns this material: `sec_mdl-gradient-based-optimization` (descent lemma, noise
   ball) and the new §24.2 `sec_mdl-adaptive-stochastic-methods` (Ghadimi–Lan rate, schedules).
   Verified: neither label is referenced anywhere in `chapter_linear-regression/`.
4. :388-394 — [weak] "converges slowly towards the minimizers … will not find them exactly" is vague
   where the appendix now has the precise statement (noise ball of radius ≈ ησ²/(2λ),
   `mdl-adaptive-stochastic-methods.md:50-51`). One sentence + :numref: upgrades hand-waving to fact.
5. :884-886 (slide) — [weak] "The loss is **convex** in (w,b), so every local minimum is the global
   one" — a load-bearing fact stated *only* in the slide deck. Prose says only "just one critical
   point" under full rank (:253-256). Add one prose sentence: L is convex; full column rank makes it
   strictly convex, hence the unique minimizer.
6. :1025-1027 (slide) — [weak] "The template: match the loss to the noise model" — the UDL-style
   recipe is *named in the slide but not in prose*. Prose :658-660 stops at the Gaussian instance.
7. :225-230 — [weak] Quadratic sensitivity to outliers is asserted ("excessive sensitivity to
   anomalous data") but never demonstrated in code anywhere in the chapter; MAE robustness lives only
   in scratch exercise 7.
8. :531-559 — [polish] `normal()` is defined in four near-identical tabs (only mxnet/jnp differ);
   unavoidable under the tab system, but the pytorch tab computing with NumPy (`np.exp`) rather than
   torch is worth an explanatory clause ("plain NumPy suffices for plotting a density").

**Proposed improvements.**
1. (S) Fix "exponential"→"Laplace" at :786 and cross-link the exercise to `sec_mdl-distributions`.
2. (S) Reword :232 per issue 2.
3. (S) After :372 (or at :394) add the two forward pointers: "why this converges, how η and schedules
   interact with gradient noise → :numref:`sec_mdl-gradient-based-optimization` and
   :numref:`sec_mdl-adaptive-stochastic-methods`"; fold the noise-ball sentence into :388-394.
4. (M) **Name the loss-function recipe in prose** (adapting UDL ch5, per digest — accepted): after
   :660 add a short subsection or paragraph: "choose a noise model p(y|x), minimize its NLL" is the
   general recipe; Gaussian⇒MSE is one row of a menu whose siblings the exercises walk (Laplace⇒MAE
   = ex. 5, Poisson⇒counts = ex. 8, log-price = ex. 7), plus one sentence on heteroscedastic
   regression (predict σ(x) too). A compact 3-column table (output type → distribution → loss) makes
   it scannable. This converts three excellent exercises from orphans into instances of a named idea.
5. (M) Add one short *teaching* code cell after :230: fit the synthetic data twice, once with squared
   loss and once with MAE, after corrupting one label (y₅=10000); print both ŵ. Cashes out the
   outlier claim in ~8 lines and sets up scratch ex. 7. (Digest's MAE demo — accepted, placed here.)
6. (S) Promote the convexity sentence (issue 5) into prose near :253.

---

### Section: oo-design.md (§3.2)

**Strengths.** The motivation ("a small change to the training procedure would force us to touch
every chapter", :9-16) is the right one, and the Lightning provenance is honest. The three-class
figure `mdl-linreg-oo-classes.svg` (house-style, generated by `tools/gen_mdl_linreg_figures.py`) is
reused across four sections — exactly how scaffolding should be taught. The `add_to_class` demo
(:85-103) is minimal and complete. Best of all, the *why-async* discussion (:144-163) — compiled
steps must be pure, devices run ahead of Python, so `draw` queues and a background thread renders —
is a genuinely distinctive piece of teaching that previews a book-wide theme; the per-framework
comments in `Module.plot` (e.g., the MXNet thread-safety note :268-272) reward careful readers. The
rewritten exercises (design trade-offs, a no-`inspect` reimplementation, sync-vs-async) are a real
ramp, not filler. Verdict on the standing question: the OO indirection **earns its keep** — the
section flows, does not stall the narrative, and every magic trick (`add_to_class`,
`save_hyperparameters`, `ProgressBoard`) is either explained or explicitly deferred to `sec_utils`
(label verified).

**Issues.**
1. :208-225 — [polish] `Module.plot` is the gnarliest code in the section (fractional-epoch x-axis
   arithmetic, `every_n` computation) and gets zero prose. One sentence — "plot converts the batch
   index into a fractional epoch so train (many points/epoch) and validation (one point/epoch) share
   an x-axis" — would spare every reader the same 2-minute decode.
2. :129 — [polish] "the optional `every_n` smooths the line by only showing 1/n points … averaged
   from the n neighbor points" — garbled; say "shows one point per n calls, plotting the average of
   the last n values".
3. :107-114 — [polish] The `HyperParameters` stub is `#@save`d with `raise NotImplementedError`, then
   `B` immediately subclasses the *full* `d2l.HyperParameters`. The deferral is stated (:114) but a
   reader may still wonder why a stub is worth saving; half a sentence ("the stub fixes the
   interface; :numref:`sec_utils` fills it in on this same class") closes the loop.

**Proposed improvements.**
1. (S) Add the one-sentence `plot` explanation (issue 1) and fix the `every_n` wording (issue 2).
2. (S) Tighten :107-114 per issue 3.
3. (S) In the Trainer prose (:509), say explicitly that `fit_epoch` is implemented in
   :numref:`sec_linear_scratch` — currently "later chapters", but it lands two sections later; the
   precise pointer reassures readers the abstraction gap closes almost immediately.

---

### Section: synthetic-regression-data.md (§3.3)

**Strengths.** The opening argument (:10-27) — synthetic data removes two of three failure suspects,
so any failure is the algorithm's, "full stop" — is the best motivation of this section in any
edition. The hand-rolled-vs-built-in loader contrast is honestly costed (:250-264: memory,
single-threaded, no prefetch). The JAX `drop_remainder=train` treatment (:303-316, :356-366) is
superb framework-tab teaching: it explains *why* (jit recompilation for the odd last batch), *what it
costs* (8 examples/epoch), and reconciles the visible 31-vs-32 discrepancy. ⌈1000/32⌉=32=31+8
checks out. Exercises 4 (functional PRNG, why key reuse is a bug) and 5 (recovery error vs σ) are
excellent, load-appropriate additions.

**Issues.**
1. :159 — [weak] "Each row in `features` … each row in `labels`" — stale names from old d2l; the
   attributes are `data.X` and `data.y` (used in the very next cell :162). Rename in prose.
2. :116-117, :134-135, :148-149 — [polish] tf/jax/mxnet tabs shadow the constructor argument
   `noise` (`noise = tf.random.normal(...) * noise`) while pytorch uses `eps` (:102). Works, but
   shadowing a saved hyperparameter in a section that just praised `save_hyperparameters`
   introspectability is sloppy; standardize all tabs on `eps` (well, cell IDs stay put — this is a
   body edit only).
3. :19 vs :80 — [polish] intro says "the true weights $\mathbf{w}^*$, the true bias $b^*$" but the
   generating equation :80 writes plain $\mathbf{w}$, $b$. Use starred symbols in the equation to
   match (the slides at :448 already do).
4. :387 — [polish] Exercise 1 duplicates what the JAX tab callout :356-366 now answers in full;
   sharpen it to the *other* frameworks ("what does PyTorch's `drop_last` do, and when would you
   want it?").

**Proposed improvements.**
1. (S) Fix issues 1–3 (pure wording/naming).
2. (S) One sentence after :354 noting the train loader *reshuffles every epoch* (the built-in one
   does; the hand-rolled one reshuffles per call) — scratch exercise 8 asks why reshuffling matters,
   but this section never says it happens.

---

### Section: linear-regression-scratch.md (§3.4)

**Strengths.** The four-step-loop paragraph (:322-337) — zero, forward, backward, update, with the
failure mode of skipping each — is the clearest statement of the training-loop contract I know of at
this level, and the slide deck mirrors it. The noise-floor reading of the loss curve (:641-646,
σ²/2 ≈ 5×10⁻⁵ — arithmetic verified) turns a "line goes down" plot into a quantitative check, and
the no-gap observation tees up §3.6 perfectly. The JAX tab is a model of honest functional-style
teaching (why `EmptyState()`, why two jitted updates, :438-459). The closing paragraph (:713-718)
correctly frames SGD and squared loss as the first members of families developed later.

**Issues.**
1. :395-398 — [error] "Recall that the synthetic regression dataset … **does not provide a
   validation dataset.**" It does: `SyntheticRegressionData` defaults `num_val=1000`
   (synthetic-regression-data.md:96), `val_dataloader` works, and this very section plots the
   validation curve and discusses it at :641-646. The stale sentence (inherited from old d2l)
   directly contradicts the code three cells later. Replace with "our synthetic `DataModule` holds
   out 1000 validation examples; we run the validation loader once per epoch."
2. :868-889 (slides) — [weak] The hand-derived gradient ∂ℓ/∂w=(ŷ−y)x, ∂ℓ/∂b=(ŷ−y) — the single
   most load-bearing equation of the section — exists **only in the slide deck**. Prose readers
   write `param -= self.lr * param.grad` without ever seeing what `param.grad` contains. This is the
   known prose/slide asymmetry and it is real: promote the derivation into prose (it is four lines
   and the chapter already derived the batch form at eq_linreg_batch_update — cross-reference it),
   ideally at the end of "Defining the Loss Function" so the SGD class that follows consumes a known
   quantity. The "gradient is the error-weighted input" slogan from the slide (:888) deserves prose.
3. :628-631 — [polish] `torch.manual_seed(1)` is pytorch-only; the other three tabs are unseeded, so
   their committed outputs are irreproducible-by-rerun in a section whose sibling (§3.3, ex. 4) is
   *about* reproducibility. Either seed all tabs or drop the cell and the slide (:959-965) built on it.
4. :383 — [polish] "(assuming that the number of examples is divisible by the batch size)" — it
   isn't (1000/32), and nothing depends on it; the parenthetical only confuses. Delete or say "up to
   a final partial batch".
5. :811 (slide) — [polish] "Draw `w` from a tiny Gaussian (**to break symmetry**)" — symmetry
   breaking is irrelevant for a single linear layer (any init works; exercise 1 asks exactly this and
   the slide pre-empts it with a wrong reason). Slide-deck fix; flagged because it is a correctness
   point.
6. :169-174 — [polish] Prose promises "we need to transform the true value y into the predicted
   value's shape" but the pytorch/mxnet/tf `loss` never reshapes (`(y_hat - y) ** 2`); only the JAX
   tab does (:191). Exercise 5 then asks why `reshape` is needed. Align the prose with the shown code
   (the DataModule already yields matching (B,1) shapes) or reshape in all tabs.

**Proposed improvements.**
1. (S) Fix the validation-set contradiction (issue 1).
2. (M) Promote the gradient derivation into prose (issue 2) — this is the section's one must-do.
3. (S) Issues 3–4, 6 wording/seeding.
4. (S) After :693, one forward pointer for the "many good minima" claim (:688-692) to
   `sec_generalization_deep` — the claim is otherwise dangling.

---

### Section: linear-regression-concise.md (§3.5)

**Strengths.** The piece-by-piece substitution framing (:55-60: layer replaces (w,b), built-in loss
replaces the math, optimizer replaces the update loop) is exactly right, and the slide deck's
by-hand/built-in table is a nice artifact. Framework quirks are handled with unusual care: the lazy
`.data` initialization comment (:143-145), Gluon's ×2 `L2Loss` correction (:231), and the ½-factor
learning-rate consequence chain back to §3.1's warning. Exercise 6 (timing scratch vs concise) is a
good new empirical exercise.

**Issues.**
1. :62-74 — [weak] This paragraph duplicates the section intro (:9-19) almost clause for clause
   ("You *should* know how to do this" appears at :12 and :70; "defined our model parameters
   explicitly" at both). One of the two must go — keep the blog analogy, cut the rest.
2. :220-223 — [polish] `fn = nn.MSELoss()` constructed inside `loss()` on every call (same for tf
   :238). Harmless here, but it models a mild anti-pattern; construct once in `__init__` or add a
   clause admitting the shortcut.
3. :344-355 — [polish] "we access the weights and bias of the layer that we need" — limp phrasing
   for the one new idea (parameters now live *inside* `net`, so we reach through it; JAX reaches into
   `state` instead). The JAX contrast is shown (:381-389) but never remarked on in prose (the slides
   do, :718-722).

**Proposed improvements.**
1. (S) Deduplicate :62-74.
2. (S) One prose sentence on the where-parameters-live contrast (issue 3) — it is the conceptual
   payload of `get_w_b`.
3. (S) Exercise 2 (Huber) could point back to the new MAE/outlier demo cell proposed for §3.1 —
   "combine the best of both" then has a computed baseline.

---

### Section: generalization.md (§3.6)

**Strengths.** The Ellie/Irene parable remains the best opening in the chapter. The
train-error-as-biased-estimate point (:171-186: a *fixed* classifier on fresh data is mere mean
estimation; the training set is not) is the right statistical heart and is stated cleanly. The
bias–variance paragraph (:231-237) names the decomposition and defers the proof to
`sec_mdl-statistics` — **verified**: the label resolves, and the target (mdl-statistics.md:93-141)
proves the decomposition, plots the U-curve, and points back here; the digest's "never actually
decomposed" claim is **stale for this branch** — the spiral works. The new polynomial demo
(:400-447) is house-perfect: NumPy-only, computes under/right/over (verified numerically: train
2.5×10⁻¹³ vs test 4.9×10¹³ at degree 19) and then *sweeps* the U-curve. The double-descent
paragraph (:279-286) is exactly the right size given the deliberate consolidation into §25.5 —
correctly a teaser with citations, not a treatment.

**Issues.**
1. :196 — [weak] "the generalization gap to grow" — term used before its definition at :299
   ("the *generalization gap* (R − R_emp)"). Move the italicized definition to first use (:188-197
   area) or forward-reference it.
2. :534-546 — [weak] K-fold is procedural only, and the *why* of K is nowhere: exercise 5 asks "Why
   is the K-fold CV error estimate biased?" with no in-book answer (verified: no K-fold content in
   the appendix either). Digest's ISL §5.1.4 borrow — **accepted**: add 3-4 sentences (each fold
   trains on (K−1)/K of the data, so fold models are slightly worse than the final model → the
   estimate is pessimistically biased; LOOCV minimizes that bias but its K models are nearly
   identical, so the average has high variance and costs K fits; K=5 or 10 is the standard
   compromise). This also answers exercises 4 and 5 honestly instead of leaving them load-bearing.
3. :357 — [weak] `fig_capacity_vs_error` uses legacy `capacity-vs-error.svg` while the house-style
   `mdl-prob-bias-variance-u-curve.svg` exists and is already used by *this section's own slides*
   (:687, :790). One-figure-style-per-chapter says swap the prose figure to the house one (or a
   chapter variant labeled with polynomial degree on the x-axis).
4. :171 — [polish] "when we evaluate our **classifier** on the test set" — this is the regression
   chapter; say "model".
5. :145-147 — [polish] "the training error is expressed as a *sum*" — the formula :147 is an
   average (1/n); the intended contrast is sum-vs-integral, but as written it misstates its own
   equation.
6. :153 — [polish] $R[p,f]$ indexes by the density $p$ while the expectation subscript uses $P$;
   pick one.
7. :279-286 — [polish] The double-descent teaser points only to `sec_generalization_deep`; this
   branch's *quantitative* treatment (which reproduces the curve from scratch) is
   `sec_mdl-concentration-generalization` §"Interpolation and Double Descent"
   (mdl-concentration-generalization.md:814). Add it as a second pointer. Digest's "add a double
   descent treatment" — **rebutted** beyond this: the consolidation into §25.5 is deliberate and
   right; two pointers suffice.
8. Coverage — [weak] The section *names* bias and variance but never computes them, even though the
   polynomial demo has everything needed. Bishop Fig 4.7/4.8-style decomposition demo — **accepted,
   adapted**: not a re-derivation (appendix owns that) but a ~12-line cell: resample the degree-d fit
   over ~200 fresh noise draws, estimate bias²(x), var(x) on the test grid, and `d2l.plot` bias²,
   variance, and their sum against degree. The U-curve then *visibly decomposes*, closing the loop
   with the `sec_mdl-statistics` identity it cites.

**Proposed improvements.**
1. (S) Issues 1, 4, 5, 6, 7 — wording and one added :numref:.
2. (S) K-fold "why K" paragraph (issue 2) + include the existing `mdl-mlp-kfold.svg` (already used in
   the slide at :834) as a prose figure in the Cross-Validation subsection, which currently has none.
3. (M) The computed bias²/variance decomposition cell (issue 8).
4. (S) Swap :357 to the house-style U-curve figure (issue 3).

---

### Section: weight-decay.md (§3.7)

**Strengths.** The monomial-explosion motivation (:33-47, formula verified) is a crisp argument for
*continuous* capacity control. The penalty⇔constraint geometry with `mdl-linreg-ridge-geometry.svg`
(:190-201) is house-style and the tangent-vs-corner reading of ridge-vs-lasso is exactly the right
intuition. The (1−ηλ) decay derivation :207-218 is correct. Three modern paragraphs lift this above
the old edition: the SGD-only equivalence + AdamW decoupling (:228-237), the MAP reading (:239-248),
and the effective-λ mismatch subtlety between loss-penalty and optimizer-decay (:566-574) — plus the
Keras `wd/2` convention fix (:537-540). The 20-examples/200-dims experiment remains the cleanest
possible overfit-then-rescue demo.

**Issues.**
1. :190-198 — [weak] The penalty⇔constraint equivalence is asserted with the picture only, while the
   appendix now *derives* λ-as-multiplier and even verifies it numerically **on ridge regression**
   (mdl-constrained-optimization-duality.md:767-773, :1515-1526, code at :824) and name-checks
   `sec_weight_decay` — but this section never points there. The hand-off is one-directional. Add
   ":numref:`sec_mdl-constrained-optimization-duality` derives this equivalence and computes the
   λ↔t correspondence."
2. :228-237 — [weak] Same pattern for AdamW: the appendix has a full subsection
   `subsec_mdl-decoupled-weight-decay` (mdl-adaptive-stochastic-methods.md:505-590) with the
   per-coordinate shrinkage formula and a `decay_race` code demo — precisely the fast.ai two-liner
   mechanism the digest asks for, already written. This paragraph points only to `sec_adam`. Add the
   subsec pointer; digest's "add the fast.ai two-liner here" is thereby **adapted**: pointer, not
   duplication.
3. :246 — [weak] "the regularization constant λ plays the role of the prior precision" — off by a
   factor: with the *averaged* loss L of :131, matching the MAP objective gives penalty coefficient
   λ = σ²λ_prior/n, not λ_prior. Say "proportional to the prior precision (with the exact
   correspondence, involving σ² and n, worked in exercise 6)" — otherwise a careful student who does
   exercise 6 finds the text wrong.
4. :898-928 (slides) — [weak] The MAP derivation chain (−log p(w) = (λ/2)‖w‖²+const; MAP = MLE +
   prior) is fully written **only in the slides**, with the `mdl-prob-map-prior.svg` figure; prose
   asserts and defers wholly to exercise 6. The derivation is three lines — promote it (keeping
   exercise 6 as the "make the constants precise" step per issue 3).
5. Coverage — [weak] The **quantitative** shrinkage story is absent book-wide (verified: no
   d_j²/(d_j²+λ) or effective-dof anywhere in `chapter_mdl-*`; `sec_mdl-svd-low-rank` gestures at
   Tikhonov at :684 "which we revisit" — a revisit that never happens). Also missing here: the ridge
   closed form (XᵀX+λI)⁻¹Xᵀy — currently stated only inside linear-regression.md exercise 4.5, so
   the §3.1 analytic-solution thread silently dies. Digest's ESL borrow — **accepted, sized to one
   paragraph + one cell**: (i) state the closed form (one line, resolving §3.1 ex. 4.5's forward
   reference); (ii) via X=UDVᵀ, ridge multiplies the response along the j-th principal direction by
   d_j²/(d_j²+λ) — low-variance directions are hit hardest, which is *why* shrinkage tames the
   noise-chasing directions the geometry figure can't distinguish; (iii) one sentence on
   df(λ)=Σ d_j²/(d_j²+λ) sliding from p to 0 — the continuous-capacity dial made literal. Then a
   ~10-line *teaching* cell on the existing 20×200 `Data`: SVD of X, print the shrinkage factors for
   λ∈{0,3,30}, and df(λ) — connecting the section's own hyperparameter (λ=3) to an effective
   parameter count ≪ 200. Rebut the digest's fuller ESL program (Fig 3.8 coefficient profiles, full
   SVD identity derivation): too much for this tier; the identity's derivation belongs in
   `sec_mdl-svd-low-rank` if anywhere.
6. :85-90 — [polish] "More commonly called ℓ2 regularization outside of deep learning circles when
   optimized by minibatch stochastic gradient descent, weight decay might be…" — the dangling
   "when optimized by…" clause is trying to whisper the SGD-only equivalence that :228-231 states
   properly; as written it is ungrammatical. Simplify the opening sentence and let :228 carry the
   caveat.
7. :628-629 — [polish] Exercises 1–2 (λ sweep, validation-selected λ) are where the section's own
   U-curve-in-λ lives — exercise-only. Acceptable, but a one-line hint ("expect a U; compare
   :numref:`fig_mdl-bias-variance-u-curve`") ties them to §3.6/§25.4. Digest's UDL Fig 9.2 λ-sweep
   figure — **adapted** to this hint; digest's UDL Fig 9.14 regularizer taxonomy — **rebutted** for
   ch3: the other three mechanisms (augmentation, ensembles, flat minima) don't exist yet at this
   point in the book; a taxonomy belongs with dropout in ch5.

**Proposed improvements.**
1. (S) Add the two appendix pointers (issues 1, 2).
2. (S) Fix the prior-precision claim (issue 3); promote the 3-line MAP derivation (issue 4),
   reusing `mdl-prob-map-prior.svg` in prose.
3. (M) Ridge closed form + per-direction shrinkage + df(λ) paragraph and demo cell (issue 5) —
   the single highest-value content addition in the chapter.
4. (S) Issues 6–7 wording.

---

### Chapter-level coverage

**Add.**
- Named loss-menu recipe + table in §3.1 (M) and the MAE/outlier demo cell (S/M) — digest accepted.
- Gradient derivation in §3.4 prose (M) — the chapter's must-fix asymmetry.
- Ridge closed form + ESL shrinkage + df(λ) paragraph and cell in §3.7 (M) — digest accepted, sized down.
- K-fold "why K" + bias answer in §3.6 (S); computed bias²/variance decomposition cell (M) —
  digest adapted (demo + pointer, no re-derivation; appendix §25.4 already proves it).
- index.md Resources (S): add UDL (Prince) and Bishop & Bishop 2024 — the two texts this review is
  calibrated against are both free online and both missing from an otherwise excellent list.

**Cut/compress.**
- concise :62-74 duplicated motivation paragraph.
- scratch :395-398 false no-validation-set claim (replace, don't merely cut).
- Nothing else: section lengths are well judged (§3.2's 1024 lines are mostly 4× framework tabs of
  the same three classes; the prose itself is lean). Digest's locally-weighted-regression enrichment
  — rebutted: a nonparametric detour that serves neither the DL arc nor the appendix spiral; at most
  a future exercise.

**New figures (chapter-wide list).**
1. §3.1 loss-menu companion (2-panel SVG, `gen_mdl_linreg_figures.py`): left — Gaussian vs Laplace
   noise densities (matched variance, log-y inset to show the tails); right — per-residual penalty
   ½r², |r|, Huber vs residual r. Caption ties row-of-table → curve.
2. §3.6: replace legacy `capacity-vs-error.svg` with house-style `mdl-prob-bias-variance-u-curve.svg`
   (exists; the section's slides already use it) or a linreg variant with "polynomial degree" on x.
3. §3.6 Cross-Validation prose: include existing `mdl-mlp-kfold.svg` (currently slide-only).
4. §3.7 MAP prose: reuse existing `mdl-prob-map-prior.svg` (currently slide-only).
5. (L, optional style unification) Redraw the four legacy SVGs (`fit-linreg`, `singleneuron`,
   `neuron`, `capacity-vs-error`) in the mdl house style via the chapter generator; the chapter
   currently mixes two figure styles, against the one-style-per-chapter rule. `neuron.svg` (public-
   domain SEER art) may reasonably stay as-is; the other three are line drawings the generator can
   reproduce cheaply.
   (Shrinkage factors and bias²/variance are deliberately *computed cells*, not SVGs — code teaches.)

**Hand-off fixes.** (all target labels verified to exist on this branch)
- linear-regression.md :299-411 → add :numref:`sec_mdl-gradient-based-optimization` and
  :numref:`sec_mdl-adaptive-stochastic-methods` (SGD convergence/noise/schedules). Missing.
- weight-decay.md :190-198 → add :numref:`sec_mdl-constrained-optimization-duality` (λ-as-multiplier,
  derived + numerically verified on ridge there). Missing; currently one-directional.
- weight-decay.md :228-237 → add :numref:`subsec_mdl-decoupled-weight-decay` (AdamW mechanism + code
  demo) alongside the existing `sec_adam`. Missing.
- generalization.md :279-286 → add :numref:`sec_mdl-concentration-generalization` as second
  double-descent pointer (its §"Interpolation and Double Descent" reproduces the curve). Missing.
- generalization.md :237 → `sec_mdl-statistics` — **verified correct and current** despite the ch25
  reorder (statistics is §25.4; the label did not move). No fix needed.
- All other :numref: targets in the chapter checked and resolving: `sec_mdl-geometry-linear-algebraic-ops`,
  `subsec_broadcasting`, `subsec_lin-algebra-norms`, `sec_generalization_deep`,
  `chap_classification_generalization`, `sec_batch_norm`, `sec_utils`, `sec_model_construction`,
  `sec_lazy_init`, `sec_adam`, `chap_optimization`. No stale pointers found.

## Chapter 4 · Linear Classification (`chapter_linear-classification/`)

### Chapter verdict

This is a strong chapter, already ahead of CS231n/Bishop on rigor where it counts: the MLE→cross-entropy derivation with the log-partition/exponential-family framing (§4.1), the fully derived stable log-sum-exp loss (§4.5), and the derived covariate/label-shift corrections (§4.7) are best-in-class for an intro. The calibration/temperature thread the external survey flags as missing is in fact already present (§4.1 caution + Boltzmann tie-in + exercises; §4.3 exercise 4) — it just lacks a picture and a demo. The three biggest wins available: **(1) evaluation beyond accuracy** — the chapter trains a 10-class classifier and never shows a confusion matrix (which §4.7 then needs and must define ad hoc); a confusion-matrix cell in §4.4 plus a short "limits of accuracy" passage (imbalance counterexample, precision/recall at formula depth) in §4.3 closes the one genuine coverage gap that belongs *here*. **(2) Code for the two code-free sections** — §4.6 and §4.7 assert quantitative claims (√n concentration, adaptive overfitting, reweighting works) that one compact simulation cell each would demonstrate, per the house "code teaches" rule. **(3) Geometry figures** — a text-heavy chapter about a *geometric* object (linear decision boundaries) contains no picture of a decision region, a temperature sweep, or a density-ratio weight; three or four house-style SVGs would materially lift it. Everything else is polish: a garbled probit description, an overclaimed MNIST number, some legacy phrasing, and two missing appendix pointers.

---

### Section: index.md (§4.0)

**Strengths.** Tight intro; the Resources list is well curated and already anticipates the chapter's calibration and shift threads (Guo et al., Quiñonero-Candela). Pointing "classical ML texts" back to ch. 3's list avoids duplication.

**Issues.**
1. index.md:45–46 [weak] — Internal link-audit residue leaked into reader-facing text: "(open-access edition; page bot-blocked, noted)" and "(paywalled, noted)". "Noted" is a note-to-self, not prose.
2. index.md:31 [polish] — One 60-word sentence; split after "we do not repeat them here."

**Proposed improvements.**
1. (S) Rewrite the two parentheticals: "(an open-access edition is available from MIT Press)" / "(print/paywalled)".

---

### Section: softmax-regression.md (§4.1)

**Strengths.** The core derivation chain is the best I know of at this level: one-hot → linear model → softmax with translation invariance and the σ special case (:217–222, correct), → MLE → cross-entropy → gradient via the log-partition function (:365–381, verified: ∂g = softmax, gradient = ŷ − y), with the exponential-family "prediction minus observation" punchline (:392–394) and Hessian-=-softmax-covariance deferred to exercise 1 — the right spiral split with the appendix, which proves lse convexity in full. The "Why cross-entropy?" box (:418) is exactly the right depth, with the minimized-at-Q=P claim correctly deferred to `sec_mdl-information_theory` (label verified). Calibration + temperature (:294–303) is a modern, well-placed differentiator vs. the 2017-era original. Exercises are unusually good (Bradley–Terry, RealSoftMax, temperature/calibration ramp).

**Issues.**
1. softmax-regression.md:179–184 [weak] — The probit description is garbled: "$\mathbf{y} = \mathbf{o} + \boldsymbol\epsilon$" with one-hot $\mathbf y$ makes no sense dimensionally, and it is not the probit model (which thresholds/argmaxes a noisy latent score). The correct and *more useful* statement: $y = \operatorname{argmax}_i (o_i + \epsilon_i)$ with Gaussian noise gives (multinomial) probit; with Gumbel noise it gives exactly the softmax. That one-line fix turns a wrong aside into a beautiful foreshadowing of "softmax = smoothed argmax."
2. softmax-regression.md:294–303 [polish] — The calibration paragraph interrupts the likelihood setup mid-thought ("These are probabilities by construction, but…" then eight lines of Guo/temperature before returning to $P(\mathbf Y\mid\mathbf X)$). Content is right; placement stalls the derivation.
3. softmax-regression.md:397–398 [weak] — "revisit log-partition convexity in the optimization part" has no `:numref:`. The target exists: `sec_mdl-convexity` (chapter_mdl-optimization/mdl-convexity.md:2), whose "Log-Sum-Exp and the Softmax Covariance" proposition (:758ff) is exactly this. Missing hand-off.
4. softmax-regression.md:471–473 vs :495–499 [polish] — Exercise 1 (second derivative = variance) and exercise 7.1 (convexity via "second derivative is the variance") are near-duplicates.
5. softmax-regression.md:118–123 [polish] — The "we only need one fewer" justification ("the final category has to be the difference between 1 and the sum") reasons about probabilities, not affine scores; the honest argument (translation invariance) arrives only at :222. A forward-reference ("as we will see, only logit differences matter") would keep the first mention honest.
6. softmax-regression.md:69 [polish] — Heading discipline: only two content-level `##` sections ("Classification", "Loss Function"); "The Softmax" and "Vectorization" are buried as `###` under *Classification*, which they are not conceptually part of. House convention wants 3–5 top-level sections.
7. Slides (:703) [polish] — slide says entropy is "average **bits**" while prose (:418) correctly says **nats** (natural log throughout). Inconsistent units.

**Proposed improvements.**
1. (S) Fix probit as in issue 1; optionally add an exercise part: "show that $\operatorname{argmax}_i(o_i + \gamma_i)$ with i.i.d. Gumbel $\gamma_i$ is distributed as $\mathrm{softmax}(\mathbf o)$" (Gumbel-max trick) — high wow-per-line, connects probit, softmax, and temperature.
2. (S) Move the calibration/temperature paragraph out of "Log-Likelihood" into a short titled paragraph ("**Confidence is not calibrated probability.**") at the end of the Loss Function section, where the reader has just seen the loss reward confidence past correctness; keep exercise 8 as its ramp.
3. (S) Add `:numref:`sec_mdl-convexity`` at :397–398.
4. (M) Promote "The Softmax" (and fold Vectorization into it) to a top-level `##`, giving §4.1 the shape *Classification problem / The Softmax Model / Loss Function* — no cell IDs move.
5. (M) **New figure:** decision-region SVG — a 2-D, 3-class softmax regression partitioning the plane into three convex polyhedral cones meeting at a point (argmax of affine functions), with one binary inset showing σ level-sets perpendicular to $\mathbf w$. The chapter never *shows* what a linear classifier's decision looks like; this is the single most valuable missing picture (CS231n leads with it).
6. (S) **New figure:** temperature sweep — one 3-class distribution's softmax bars at $T \in \{0.25, 1, 4\}$ (hard argmax → uniform), placed beside the Boltzmann paragraph (:224–243), which currently name-drops temperature without an image. Directly supports exercise 8.
7. (S) Merge exercise 7.1 into exercise 1 (or re-point 7.1 at proving convexity *from* ex. 1's result), freeing 7 to focus on translation equivariance + stability.

---

### Section: image-classification-dataset.md (§4.2)

**Strengths.** The MNIST→Fashion-MNIST motivation is crisp; channel-first/channel-last is pinned down once and hidden thereafter (:125) — exactly right; the I/O-vs-compute timing cell (:227–232) teaches a real systems lesson; the TF/JAX `drop_remainder` retracing comments (:178–184, :196–197) are quietly excellent. Exercises (throughput vs. batch size, `num_workers`) are practical and well ramped.

**Issues.**
1. image-classification-dataset.md:11 [weak] — "today even simple linear models exceed 95%" on MNIST is wrong: logistic regression/linear classifiers reach ≈91–93% (Xiao et al. 2017 report ~91.7%; LeCun's linear baseline is 88%). Simple *nonlinear* models (k-NN, kernel SVM) exceed 95–98%. The argument survives, the number doesn't.
2. image-classification-dataset.md:127 [polish] — Orphaned fragment: "A single grayscale image, so $c = 1$." dangles after the layout paragraph with no verb-bearing lead-in (the HyMap sentence then changes topic).
3. image-classification-dataset.md:204–223 [polish] — Two adjacent cells each pull `next(iter(data.train_dataloader()))`; the intro "To see how this works, let's load a minibatch" (:218) is stale — a batch was just loaded to check the per-image shape. Merge the narration ("the same batch, now at batch granularity") or print both shapes in one cell.
4. image-classification-dataset.md:225 [polish] — "a single forward and backward pass … typically takes 10 to 100 times longer than the corresponding I/O" is asserted generically right before a *linear* model chapter where compute is tiny; qualify ("for the models of the next chapters") or the reader's own timing may contradict it.

**Proposed improvements.**
1. (S) Fix the MNIST claim: "even simple models exceed 95% (and a linear classifier already tops 90%), so differences between strong and weak models are hard to see."
2. (S) Repair :127 into the preceding paragraph: "Here $c=1$ since the images are grayscale; most photographs have $c=3$ …".
3. (S) De-duplicate the two batch-pulls per issue 3.

---

### Section: classification.md (§4.3)

**Strengths.** The loss-vs-accuracy fork is the pedagogical heart of the chapter and it is superbly done: the two-branch narrative (:135–137), the `mdl-clf-loss-accuracy.svg` figure with *verified* worked numbers (softmax of (1.0, 2.2, 0.3) = (0.21, 0.69, 0.10), CE = −ln 0.689 = 0.372 ✓), the Gmail hard-decision example, and the "disagreement is diagnostic, not a bug" line. Exercises 4 (confidence/temperature) and 5 (top-k) are new, concrete, and well ramped; exercise 3 (Bayes decision rule under a general loss) quietly plants cost-sensitive classification.

**Issues.**
1. classification.md:42 [polish] — Legacy phrasing survives: "We draw an update for every `num_val_batches` batches. This has the benefit of generating the averaged loss and accuracy on the whole validation data" — unclear (what is "drawing an update"?) and the second sentence doesn't follow from the first.
2. classification.md:133–163 [weak, coverage] — The section that owns "how do we evaluate a classifier" stops at accuracy. Per the external survey this is now below baseline (Google MLCC, scikit-learn): no confusion matrix, no precision/recall, no class-imbalance caveat. The confusion matrix is *also* load-bearing later — §4.7 must define it from scratch mid-derivation (environment-and-distribution-shift.md:492–498). This is the right home for the concept; §4.4 is the right home for computing one (model exists there).
3. classification.md:100 [polish] — `configure_optimizers` is installed on `d2l.Module` (not `Classifier`) inside a section titled "The `Classifier` Class"; one clause acknowledges it ("We install the default here, on `d2l.Module` itself") but a reader skimming the code sees `add_to_class(d2l.Module)` under a Classifier heading. Consider a sentence on *why* it belongs on Module (regression models reuse it too).

**Proposed improvements.**
1. (M) Add a third top-level section "## Beyond Accuracy" (~2/3 page): (i) the 99%-accuracy counterexample — 1%-positive screening task, always-predict-negative gets 99%; one 4-line code cell *computing* it (per house style: a cell demonstrating an asserted claim); (ii) precision/recall definitions from the 2×2 confusion table, F1 at one-sentence depth (survey's advice: don't over-invest); (iii) one paragraph naming the confusion matrix for $k$ classes with a forward pointer to its two reappearances (misclassification structure in :numref:`sec_softmax_scratch`, label-shift correction in :numref:`sec_environment-and-distribution-shift`). ROC/AUC: exercise-level only (threshold sweep on a binary sub-task) — full ROC treatment would bloat a scaffolding section; multi-class ROC averaging rightly out of scope.
2. (S) Rewrite the :42 validation-averaging paragraph in plain terms (per-batch values are plotted; their average over equal-size batches equals the epoch average, off only on a short last batch).
3. (S) Add a class-imbalance exercise: reweight the cross-entropy per class (one-line change to `loss`) and relate it to the weighted ERM of :eqref:`eq_weighted-empirical-risk-min` in §4.7 — stitches the chapter together and covers the survey's imbalance gap at the right (exercise) tier.

---

### Section: softmax-regression-scratch.md (§4.4)

**Strengths.** The overflow demo cell (:98–128) is the house style at its best — the prose warns, then the code *shows* NaN vs. the stable builtin on identical input, in all four frameworks. The clip-as-band-aid honesty (:340, :472–479) with the pointer to §4.5's real fix builds a genuine narrative arc. The "What the training curve tells you / linear ceiling ≈83%" summary (:464–470) explains the *result*, not just the mechanics. Exercise 1 (probe softmax at ±100, then fix it) is the perfect ramp into §4.5.

**Issues.**
1. softmax-regression-scratch.md:448–453 [weak] — The error-visualization cell shows *which* images fail but the section never aggregates *how* they fail, despite the summary asserting "shirts and pullovers look nearly identical to a linear model" (:468–469). That claim is exactly a confusion matrix, and the chapter never computes one anywhere.
2. softmax-regression-scratch.md:431–438 [polish] — The full-validation accuracy sweep exists only in the pytorch tab; tf/jax/mxnet readers get the "approximately 83%" claim (:440) with no cell producing it. If per-framework cost is the concern, say so; otherwise mirror the cell.
3. softmax-regression-scratch.md:52–61 [polish] — The axis-sum warm-up cites `subsec_lin-alg-reduction`/`non-reduction` (labels verified) — good — but the cell (:58–61) re-teaches what those sections taught; two sentences + cell could compress to one pointer sentence. Minor pacing drag before the main event.
4. Slides :756–773 [polish] — The "Why a linear model caps out" slide invokes shattering/XOR and `mdl-clf-shattering.svg`, which the *prose* of this section never mentions (it belongs to §4.6). Not strictly slides-only content (it exists in §4.6 prose), but the slide runs ahead of the reader's text; a "coming in §4.6" cue would fix it.

**Proposed improvements.**
1. (M) Add a confusion-matrix cell to Prediction: `torch.zeros(10,10)` accumulation loop (or `index_add_`), rendered with `d2l.show_heatmap`-style plot, one short paragraph reading off the shirt/pullover/coat block and the sandal/sneaker/ankle-boot block. This (i) demonstrates the summary's central claim, (ii) instantiates §4.3's proposed concept, (iii) pre-seats the exact object §4.7's label-shift correction inverts. Highest-value single addition to the chapter.
2. (S) Mirror the accuracy-sweep cell to the other three tabs (it is four lines each).
3. (S) Add an exercise: "compute per-class accuracy from the confusion matrix; which pairs of classes account for most errors, and why would a *linear* model struggle on exactly those?"

---

### Section: softmax-regression-concise.md (§4.5)

**Strengths.** This is the book's authoritative stable-softmax treatment and it earns that status: the ±88 float32 thresholds (:153–158), the max-shift identity, the denominator-in-$[1,q]$ bound (:168–171, verified), and the final fused loss as a function of logits alone (:197–199) are derived, not asserted — and the appendix correctly *cites* this section rather than re-deriving (chapter_mdl-optimization/mdl-numerical-stability-conditioning.md references `subsec_softmax-implementation-revisited`). The "logits, not probabilities; the loss owns the softmax" contract, stated per-framework (:265–274), prevents the #1 real-world bug (double softmax). Exercises 3–4 are exactly the right verification tasks.

**Issues.**
1. softmax-regression-concise.md:155–158 [polish] — "underflows to $0$ once it drops below about $-88$" is slightly off: exp goes *subnormal* below ≈−87.3 but only reaches exactly 0 near −103.97 in float32 (subnormals). The symmetric "±88" is a fine mnemonic; one word ("gradually underflows past about −88, reaching 0 near −104") keeps it exact — this section's whole brand is numerical precision.
2. softmax-regression-concise.md:202–204 [polish] — "log-sum-exp … a smooth upper bound on $\max_k o_k$" is asserted; the proof is one line (LSE ≤ max + log q, LSE ≥ max) and *is already an exercise in §4.1* (RealSoftMax, ex. 6). Add a back-pointer "(you proved this in :numref:`sec_softmax`, exercise 6)" so the spiral closes.
3. softmax-regression-concise.md:287–289 [polish] — "read it off the validation curve" for the 83–84% claim; unlike §4.4 there is no numeric printout in any tab. Acceptable, but one `accuracy` sweep line would make the "same solution" claim checkable.

**Proposed improvements.**
1. (S) Precision fix per issue 1.
2. (S) Back-pointer per issue 2.
3. (S) Optional micro-figure or 3-line cell: plot $\mathrm{lse}(x, 0)$ vs. $\max(x,0)$ on $[-4,4]$ — the "smooth max" picture. (A cell is preferable to an SVG here: it *computes* the bound gap $\le \log 2$, per the code-teaches rule.)

---

### Section: generalization-classification.md (§4.6)

**Strengths.** The narrative arc (test set → reuse → a-priori theory) is the best-motivated learning-theory on-ramp I've seen in a textbook: the 3am-idea vignette (:262–275) makes adaptive overfitting *felt*, and the "test sets are all that we really have" pivot (:340) motivates SLT honestly. All worked arithmetic verified: sd ≤ √(0.25/n) (:175), n = 2500 for ±0.01 at 1σ (:184), 10,000 at 2σ (:188), Hoeffding n ≈ 18,444 → "roughly 18,500" (:216), and the Hoeffding statement (:208) is the correct two-sided bounded-variable form. The Hoeffding hand-off to `sec_mdl-concentration-generalization` (:206) is verified — the appendix proves it (mdl-concentration-generalization.md:123ff) and points back to this section. The new shattering figure with prose tie-in (:484–492) turns the previously naked VC(linear)=d+1 assertion into a proven d=2 case — a real improvement.

**Issues.**
1. generalization-classification.md:143–147 [weak] — "to estimate our test error twice as precisely… To reduce our test error by a factor of one hundred, we must collect ten thousand times as large a test set": the second sentence conflates *test error* with the *precision of its estimate*. Collecting test data never reduces the error, only the uncertainty. Should read "to shrink the uncertainty in our estimate a hundredfold."
2. generalization-classification.md:1–906 [weak] — Zero code in a book whose identity is executable claims. Two quantitative assertions beg one cell each: (i) the √n law + Hoeffding-vs-CLT comparison (simulate Bernoulli(0.1) test sets at n ∈ {100, 1k, 10k}, plot the spread of $\epsilon_\mathcal D$ vs. the two envelopes); (ii) adaptive overfitting (:302–327 asserts, never shows: evaluate k random-guess "classifiers" on one n=1000 test set and plot best-of-k apparent accuracy climbing above chance as k grows — 10 lines, unforgettable).
3. generalization-classification.md:482–483 [weak] — VC(linear on ℝ^d) = d+1 still asserted in general; only d=2 is now argued. The lower bound is a clean 3-line argument (unit vectors + origin, weights read off the labels); the upper bound (Radon) can be cited. Currently a reader can't tell if d+1 is deep or bookkeeping.
4. generalization-classification.md:559–562 [polish, hand-off] — "numerous alternative complexity measures have been proposed" cites only Boucheron et al., while the book's *own* Rademacher treatment (developed in `sec_mdl-concentration-generalization`, verified) goes unmentioned. The chapter cites the appendix for Hoeffding but not for the strictly-more-relevant Rademacher complexity — a missed spiral hand-off.
5. generalization-classification.md:592–594 [polish] — Exercises 3–4 (VC of 5th-order polynomials; axis-aligned rectangles) are classics but the ramp is thin: nothing exercises §1–2 (test-set sizing, reuse) beyond exercise 1–2, and nothing touches the shattering figure.
6. Slides :601 [polish] — Cover kicker reads "Dive into Deep Learning · Linear Classification" while every other deck in the chapter carries its "§4.x" number.

**Proposed improvements.**
1. (S) Fix the estimate-vs-error wording (:143–147).
2. (M) Add the two simulation cells per issue 2 (needs the standard imports/tab scaffold this file currently lacks; pytorch-primary per project workflow). These are data plots that teach computed results — squarely allowed by house style. The adaptive-overfitting one doubles as the demonstration for the whole "Test Set Reuse" section.
3. (S) Add the VC(linear)=d+1 lower-bound as a guided exercise ("shatter $\{0, e_1, …, e_d\}$; hint: choose $w_i$ = desired label of $e_i$, bias = label of 0") and one prose sentence citing Radon's theorem for the matching upper bound.
4. (S) At :559–562 add: "One such measure, *Rademacher complexity*, is developed in :numref:`sec_mdl-concentration-generalization`, where the double-descent phenomenon previewed below is also reproduced from scratch."

---

### Section: environment-and-distribution-shift.md (§4.7)

**Strengths.** Still the best intro-tier treatment of shift anywhere, and the survey agrees: the impossibility argument (:78–91), the causal framing of covariate vs. label shift (x→y vs. y→x, :111–113, :143–148), the derived reweighting identity (:337–356, verified), the discriminator-as-density-ratio derivation (:399–409, algebra verified: odds = p/q, β = exp(h)), the clipping-variance remark (:432–434), and the support-overlap caveat (:436–441) form a complete, honest pipeline. The tank fable rewrite (:259–273) with its "possibly apocryphal… the lesson is exact" framing *is* the survey's requested spurious-correlation vignette. The WILDS citation in the summary (:667–676) and the strengthened exercises (2–5 are now derive/implement tasks) are real upgrades. Kernel mean matching with the MMD cite (:364–367) gives the density-ratio literature exactly one sentence — right.

**Issues.**
1. environment-and-distribution-shift.md:298–558 [weak] — The Correction section derives everything and *runs* nothing: zero code in the section that ends the chapter, with the implementation relegated to exercises 3–4. The survey's peers are no better, but "best in existence" wants the payoff cell: a 2-D synthetic covariate-shift demo (shifted Gaussians; logistic discriminator; weighted vs. unweighted ERM target accuracy) closes the loop the chapter opened with `Classifier`.
2. environment-and-distribution-shift.md:492–498 [weak] — The confusion matrix is defined here, mid-derivation, for the first time in the book (column-normalized, correctly). If §4.3/§4.4 adopt the proposals above, this passage should shrink to "recall the confusion matrix of :numref:`sec_classification`, column-normalized" — as written it forces a new evaluation concept into the reader's head at the moment they're tracking a linear-system argument.
3. environment-and-distribution-shift.md:523–525 [polish] — "If our classifier is sufficiently accurate to begin with, then the confusion matrix C will be invertible" — accuracy alone does not imply invertibility (diagonal dominance does); the hedge "under some mild conditions" (:511) is right, this later sentence overstates it. One clause fixes it ("accurate enough that C is diagonally dominant, hence invertible").
4. environment-and-distribution-shift.md:681 [polish] — Exercise 2 asks to "derive the covariate-shift reweighting identity :eqref:`eq_weighted-empirical-risk-min`" — that label is the weighted *objective*; the identity being derived (:337–342) is unlabeled. Either label the identity or reword.
5. environment-and-distribution-shift.md:287–291 [polish] — "### More Anecdotes" is a limp heading for three bullets; fold into "Nonstationary Distributions" or retitle ("Further Failure Modes").
6. environment-and-distribution-shift.md:667–676 [weak, coverage] — WILDS appears as a citation, but the survey's two genuinely missing modern layers are absent from the prose: (i) the domain-generalization vs. subpopulation-shift axis (worst-group performance), which is *orthogonal* to the chapter's mechanism taxonomy and takes one paragraph; (ii) a one-callout distinction "OOD detection ≠ shift correction" (reject vs. reweight). Test-time adaptation and IRM: rightly omitted at this tier; do not add.

**Proposed improvements.**
1. (L) Add a "Covariate Shift Correction in Code" cell block after :434 (needs imports/tab scaffold; pytorch-primary): make 2-D source/target Gaussians, train the z-classifier, show β = exp(h), compare target accuracy of unweighted vs. weighted ERM, and clip β to show the variance effect. ~30 lines total across 3 cells; converts exercises 3–4 into verification rather than first contact.
2. (S) Two-paragraph modernization at the summary (:667): the WILDS axis (same-domains-different-proportions vs. unseen domains; name Camelyon17 and CivilComments, one each) and the OOD-detection callout. Keep to ~12 lines; the section must stay a taxonomy-plus-corrections chapter, not a robustness survey.
3. (S) Condense :492–498 to a recall-pointer once the confusion matrix lands in §4.3/§4.4.
4. (S) Fix the invertibility overstatement and exercise 2's eqref per issues 3–4.
5. (M) **New figure:** density-ratio reweighting SVG — one panel: source density q(x), target density p(x), and the weight curve β(x)=p/q rising where target outweighs source, with a clipped-β dashed line. Makes :337–441 visual and previews why clipping matters.

---

### Chapter-level coverage

**Add.**
- Confusion matrix as a first-class concept: named in §4.3, computed on Fashion-MNIST in §4.4, recalled (not redefined) in §4.7. The single highest-leverage addition; it also converts §4.4's "shirts vs. pullovers" claim from assertion to observation.
- "Limits of accuracy" in §4.3: imbalance counterexample cell + precision/recall at formula depth + F1 in one sentence. ROC/AUC and reliability diagrams at exercise tier only (threshold-sweep exercise in §4.3; reliability-diagram/binned-confidence exercise in §4.5 using the trained model). This is the full extent of the survey's "metrics" gap that belongs in a *linear classification* chapter — anything more (PR curves, multi-class averaging) should wait for a later evaluation-focused home.
- Class imbalance: one remark + the weighted-cross-entropy exercise linking §4.3 to §4.7's weighted ERM. Focal loss: citation-level at most (survey lists it, but it is a detection-era tool; out of scope here).
- Simulation cells for §4.6 (√n concentration; best-of-k adaptive overfitting) and a correction demo for §4.7 — the chapter's two code-free files each get code that *demonstrates their own claims*.
- WILDS axis paragraph + OOD callout in §4.7; Gumbel-max exercise in §4.1.
- Calibration/temperature: already covered in prose+exercises (§4.1, §4.3) — needs only the temperature figure and optionally the reliability-diagram exercise, not new sections. (Contra the research digest's headline, this gap is already closed in text.)

**Cut/compress.**
- §4.1: merge duplicate exercises 1/7.1; trim the survival-modeling digression (:17–28) by a third — charming but slow for page 1 of the chapter.
- §4.4: compress the axis-sum refresher (:50–61) to one pointer sentence.
- §4.6: the VC bound discussion is already appropriately lean (survey confirms peers give VC one line); do **not** expand VC beyond the d+1 exercise — the Rademacher pointer to the appendix covers the modern road.
- §4.7: "More Anecdotes" heading; the taxonomy subsections (Batch/Online/Bandits/Control/RL, :561–597) could each lose a sentence but earn their place as forward pointers — keep.

**New figures (chapter-wide list).** All via `tools/gen_mdl_classification_figures.py` in house style (mdl-figure skill):
1. `mdl-clf-decision-regions.svg` (§4.1) — 3-class softmax regression partitioning the plane; binary σ-level-set inset. [M]
2. `mdl-clf-temperature.svg` (§4.1) — one distribution at T ∈ {0.25, 1, 4}. [S]
3. `mdl-clf-density-ratio.svg` (§4.7) — p, q, and β=p/q with clipping. [S–M]
4. (optional) redraw `softmaxreg.svg` in house style — it is the chapter's one legacy hand-drawn diagram amid house-style `mdl-clf-*` SVGs (one-style-per-chapter rule). The Getty cat/dog PNGs and the PopVsSoda map are content/external images, acceptable exceptions. [S]
5. §4.6 concentration and §4.5 lse-vs-max visuals deliberately proposed as *code cells*, not SVGs — they teach computed results.

**Hand-off fixes (missing/wrong :numref: pointers).** (All target labels verified by grep.)
1. softmax-regression.md:397–398 — add `:numref:`sec_mdl-convexity`` for log-partition convexity (chapter_mdl-optimization/mdl-convexity.md:2; its lse proposition is at :758ff).
2. generalization-classification.md:559–562 — add `:numref:`sec_mdl-concentration-generalization`` for Rademacher complexity (chapter_mdl-probability-statistics/mdl-concentration-generalization.md:2; it already back-references `chap_classification_generalization`).
3. environment-and-distribution-shift.md:681 — exercise 2's `:eqref:`eq_weighted-empirical-risk-min`` points at the objective, not the identity it asks the reader to derive; label the identity (:337–342) or reword.
4. Verified healthy (no action): `sec_mdl-information_theory` (§4.1 ×3), `sec_mdl-concentration-generalization` for Hoeffding (§4.6:206), `subsec_softmax-implementation-revisited` cited *from* the appendix (correct direction, per the no-re-derivation policy), `subsec_empirical-risk-and-risk`/`eq_empirical-risk-min`/`eq_true-risk` (§4.7:311–317 → chapter_linear-regression/generalization.md:113/148/154), `sec_generalization_deep`, `subsec_generalization-model-selection`, `subsec_normal_distribution_and_squared_loss`, `sec_prob`, `subsec_lin-alg-reduction`.

## Chapter 5 · Multilayer Perceptrons (`chapter_multilayer-perceptrons/`)

### Chapter verdict
This is already a strong chapter, and in three places it is genuinely best-in-class: the backprop
worked numeric example (every number recomputed and verified correct, with the dead-ReLU row-zeroing
made concrete), the "competently trained baseline" lesson in the Kaggle section, and the honest,
modern generalization survey with its clean hand-off to appendix §25.5. The external digest's two
headline pedagogy claims split: "UAT has zero intuition" is **correct and is the chapter's single
biggest gap** (plus a dangling promise at mlp.md:350–351 that nothing in the book redeems); but
"backprop framing is dated" is **largely wrong** — the section already has computation graphs, the
add-broadcasts/multiply-scales gate pattern, and a numeric example better than CS231n's; what it
actually lacks is the gradients-add-at-forks rule (used silently at backprop.md:198), one autograd
verification cell, and a forward pointer to the appendix AD theory. Two real math/factual errors
need fixing: the He "halving lemma" is stated as a variance identity that is false (it is a
second-moment identity), and the claim that `kaiming_normal_` is PyTorch's `nn.Linear` default is
wrong. The Kaggle ensembling cell has a misleading comment and implements the metric-inconsistent
average. With the fixes below (roughly two days of work, mostly S/M), this becomes a legitimately
top-tier MLP/backprop/generalization introduction.

### Section: index.md (§5.0)
**Strengths.** The annotated further-reading list is excellent — curated, current (KAN 2024,
Prince, Bishop & Bishop), honest about paywalls, and each entry says *why* to read it. The chapter
overview correctly previews the arc.

**Issues.**
1. index.md:74–79 [polish] The "Tutorials, notes, and interactive" list omits the two resources the
   backprop section most needs as companions: Karpathy's "Yes you should understand backprop" and
   micrograd/Zero-to-Hero. CS231n optimization-2 is there; these are its natural siblings.

**Proposed improvements.**
1. (S) Add micrograd + "Yes you should understand backprop" entries to the Tutorials list, one line
   each, matching the existing annotation style.

### Section: mlp.md (§5.1)
**Strengths.** The affine-collapse argument is crisp; the XOR subsection is a model of "code
teaches" — the hand-built construction is *verified* (I recomputed all four corners: h-map folds
(0,1),(1,0) onto (1,0); output h₁−2h₂ realizes XOR exactly) and the house-style two-panel figure
carries the fold geometry. UAT caveats (existence ≠ findability ≠ generalization; exponential
width) and the C-language analogy are honest and memorable. Activation survey ends with a
genuinely current GELU/Swish/SwiGLU paragraph. Exercises 5–6 already gesture at region counting.

**Issues.**
1. mlp.md:317–351 [weak] Universal approximation is citation-only with no picture or construction —
   the digest's highest-value claim, and I **accept** it. Every peer (Nielsen, Prince, CS231n)
   builds a visual device. Given this chapter is ReLU-first and XOR already introduced "folding",
   the ReLU hinge/region device fits better than Nielsen's sigmoid bump-pair.
2. mlp.md:350–351 [weak] "We will touch upon more rigorous arguments in subsequent chapters" —
   dangling promise, verified: no chapter or appendix section covers universal approximation or
   depth separation. The fix belongs *here* (intuition sketch + exercise 6), not in a new appendix
   section.
3. mlp.md:326–329 [weak] Internal inconsistency: the theorem is quoted in Hornik's *bounded,
   non-constant* form, but the next sentence claims "the conclusion does not hinge on which of
   ReLU, sigmoid, or tanh we pick" — ReLU is unbounded, so the quoted form does not cover it. The
   non-polynomial version (Leshno et al. 1993) does. Either cite Leshno or weaken to "essentially
   any non-polynomial activation".
4. mlp.md:328 [polish] "(Hornik, 1991)" is plain text; `Hornik.1991` exists in d2l.bib (line 6014).
   Use `:citet:`.
5. mlp.md:349 [weak] The depth-trades-width compactness claim cites `Simonyan.Zisserman.2014`
   (VGG — an empirical architecture paper), not a representational result. The right citations are
   Montúfar et al. 2014 and/or Telgarsky 2016 / Eldan–Shamir 2016 (none currently in d2l.bib).
6. mlp.md:419–420 [polish] "the input may never actually be zero" — in float arithmetic
   pre-activations are exactly 0 routinely (zero-init biases, ReLU-of-ReLU). The measure-zero adage
   is fine; soften the empirical claim.
7. mlp.md:380–658 [polish] Six near-identical plot cells (function + derivative × 3 activations).
   They do teach (each derivative cell demonstrates the framework's autograd idiom), so they pass
   the house bar, but pairing function+gradient in one figure per activation (as
   numerical-stability-and-init.md:113–150 already does for sigmoid) would halve the cell count
   with no loss.

**Proposed improvements.**
1. (M) **UAT intuition sketch** — add ~2 paragraphs after mlp.md:329: each ReLU unit contributes one
   hinge; a width-D one-hidden-layer net on ℝ is piecewise linear with ≤ D+1 pieces, so
   approximating any continuous function is just placing enough joints; composing layers *re-folds*
   existing pieces, roughly doubling region count per layer — which is exactly the
   depth-versus-width gap. Rewrite lines 350–351 to point at this sketch and exercise 6 instead of
   the phantom "subsequent chapters". New figure: see chapter-wide list (mdl-mlp-uat-hinges.svg).
7. (M) **Region-counting demo cell** (pytorch): evaluate a random ReLU MLP on a dense 1-D grid,
   count distinct slopes (`np.unique` of finite differences, tolerance) for width ∈ {2,4,8,16} and
   depth ∈ {1,2,3}; print the counts against the D+1 / doubling predictions. ~10 lines; turns
   exercises 5–6's asserted claims into an observation. This is the chapter's most valuable missing
   code cell after the depth-sweep in §5.4.
3. (S) Fix Hornik cite, add Leshno-or-weakened wording, swap the VGG citation for
   Montúfar/Telgarsky (needs 1–2 new bib entries).
4. (S) One-line "try it" pointer to TensorFlow Playground (XOR/spiral) at the end of the XOR
   subsection — cheap, and index.md already links it.

### Section: mlp-implementation.md (§5.2)
**Strengths.** Tight and correct; the "three questions remain open" summary (312–318) is an
excellent forward map that makes the chapter feel architected. Exercise 9 (init scales, pointing at
§5.4) is exactly the right hook. The `mdl-mlp-arch.svg` figure and the power-of-2 width rationale
(SIMD/tensor cores) are nice touches. Exercise 8 (GELU comparison) is current.

**Issues.**
1. mlp-implementation.md:113–121 vs 239–245 [polish] The scratch model initializes with
   N(0, 0.01²) while the concise `LazyLinear` uses PyTorch's kaiming-uniform default; the slide
   ":both versions compute the *same* function" (566–567) overclaims — same architecture, different
   init and hence different training trajectory. One clause in prose (~line 302) fixes it.
2. mlp-implementation.md:204–209 [polish] The expected outcome (≈0.87 validation accuracy, a real
   but modest gain over softmax regression's ≈0.83) exists **only in the slides** (:493–495). The
   prose never says what the reader should see. Load-bearing number living slides-only — add one
   sentence after the training cell.
3. mlp-implementation.md:59 [polish] "divisible by larger powers of 2" — garbled phrasing; "a
   multiple of a large power of 2" is what's meant.

**Proposed improvements.**
1. (S) Add the expected-accuracy sentence and the init-difference clause.
2. (S) Exercise 2 ("try adding a hidden layer") could ask for the *observation* that plain deeper
   nets with σ=0.01 Gaussian init train *worse* — a perfect cliffhanger for §5.4; one added
   sentence.

### Section: backprop.md (§5.3)
**Strengths.** The worked example is the best thing in the chapter and among the best backprop
teaching artifacts anywhere: I recomputed everything — z=[−1,2], h=[0,2], o=−2, L=2, ∂L/∂o=−2,
∂L/∂W⁽²⁾=[0,−4], ∂L/∂h=[−4,2], ∂L/∂z=[0,2], ∂L/∂W⁽¹⁾=[[0,0],[2,4]] — all correct, and the
(a+b)c warm-up (−3,−3,3) too. The add-broadcasts/multiply-scales gate framing (257–259, reinforced
in slides) *is* the CS231n device; "From the Chain Rule to Autograd" (299–309) names reverse mode
and its cost regime. **I rebut the digest's "dated framing / shrink the symbolic derivation"
claim**: the symbolic pass here is one page, feeds the worked example directly, and the graph
framing is present. Augment, don't replace.

**Issues.**
1. backprop.md:193–198, 229–235 [weak] The **gradients-add-at-forks** rule is used but never
   stated: eq_backprop-J-h's "+" is precisely the two paths J←L←o←W⁽²⁾ and J←s←W⁽²⁾ summing. This
   is the one genuinely missing gate-pattern idea (Karpathy's `+=` bug is the classic student
   error). One call-out sentence at the first "+".
2. backprop.md:292–294 [weak] "You can confirm every number here in a few lines with automatic
   differentiation" — but the section contains **zero code cells**. A 6-line pytorch cell
   (`requires_grad` tensors, forward, `L.backward()`, print the four gradients) would close the
   loop, satisfy "code teaches", and make the section self-verifying. (Needs the tab-select header
   the file currently lacks, since it has no code.)
3. backprop.md:147–151, 307–309 [weak] Hand-off gap: `prod` is waved away ("hides the notational
   overhead") and the autograd paragraph points only *backward* to `sec_autograd`. The appendix
   section `sec_mdl-matrix-calculus-autodiff` (verified: exists, with `subsec_mdl-forward-mode`,
   `subsec_mdl-reverse-mode`, VJP framing, checkpointing) is the rigorous version of exactly this
   material and is never referenced from ch5. Add a forward pointer at :151 ("made precise as
   vector–Jacobian products in :numref:`sec_mdl-matrix-calculus-autodiff`") and/or :309.
4. backprop.md:50–74 [polish] Convention switch from §5.1 unremarked: mlp.md uses row-batches
   **XW** (W ∈ ℝ^{d×h}); this section uses single-example column vectors **Wx** (W⁽¹⁾ ∈ ℝ^{h×d}).
   One sentence at :50 ("for a single example we switch to column vectors, so weight shapes are
   transposed relative to :numref:`sec_mlp`") prevents a classic confusion.
5. backprop.md:357–367 [polish] Exercise ramp is good (bias, memory, second derivatives, model
   parallelism) but there is no capstone. The digest's micrograd exercise is a strong **accept**: a
   scalar `Value` class with 3–4 ops and a topological-sort `backward()`, verified against the
   worked example, is the ideal load-bearing final exercise. The sum-over-paths/why-reverse-wins
   factoring (Colah) fits as a sub-part of exercise 1 or the micrograd exercise rather than prose —
   prose already states the many-parameters/one-loss cost argument at :306–308.

**Proposed improvements.**
1. (S) State gradients-add-at-forks at eq_backprop-J-h; one boxed sentence.
2. (S) Add the autograd verification cell after :294 (pytorch tab at minimum; the numbers are
   framework-independent).
3. (S) Add the forward pointer(s) to `sec_mdl-matrix-calculus-autodiff` and the convention-switch
   sentence.
4. (M) Add the micrograd-style exercise (with a hint sketching `__add__`/`__mul__`/`relu` and the
   topo sort) and a one-line "9 paths vs (α+β+γ)(δ+ε+ζ)" path-counting sub-question.

### Section: numerical-stability-and-init.md (§5.4)
**Strengths.** Motivation is sharp; the symmetry-breaking discussion (with the dropout-breaks-it
remark) is better than peers; the Xavier derivation is clean and the forward/backward dilemma
honestly presented; the "Beyond" paragraph **already contains the BatchNorm/residual bridge the
digest asked for** (421–425, with correct :numref:s to `sec_batch_norm`/`sec_resnet` — digest claim
"no forward pointer at all" is **rebutted**). Exercise 4 is exactly the fast.ai activation-stats
depth sweep. He→Xavier rule-of-thumb box is practical.

**Issues.**
1. numerical-stability-and-init.md:377–383 [error] "Its effect on the variance of a zero-mean,
   symmetric signal is therefore to **halve** it: Var[ReLU(z)] = ½Var[z]" — **false as stated**.
   ReLU(z) is not zero-mean; for z∼N(0,σ²), Var[ReLU(z)] = (½ − 1/2π)σ² ≈ 0.34σ². The true halving
   identity is in **second moments**: E[ReLU(z)²] = ½E[z²], which is also what the next layer's
   variance computation actually consumes (o_i is zero-mean because the *weights* are, so
   Var[o_i] = n·σ²·E[h²] — line 387's conclusion is correct; only the lemma's labeling is wrong).
   Exercise 3 (:448) propagates the same false statement ("show that Var[ReLU(z)] = ½Var[z]") —
   as written, the assigned proof is of a false proposition. Restate both in second moments.
2. numerical-stability-and-init.md:401–402 [error] "PyTorch's `kaiming_normal_`, in fact the
   default for `nn.Linear`" — factually wrong: `nn.Linear.reset_parameters` uses
   `kaiming_uniform_(a=√5)` (effectively U(±1/√fan_in)), not `kaiming_normal_`. Say "shipped as
   named initializers (`kaiming_normal_`, `kaiming_uniform_` — a variant of which is `nn.Linear`'s
   default)".
3. numerical-stability-and-init.md:81 [weak] "the matrices M⁽ˡ⁾ may have a wide variety of
   **eigenvalues**" — two problems, confirming and extending the known hand-off fact: (i)
   eigenvalues are never defined in ch2–5 (first defined in `sec_mdl-eigendecompositions`); (ii)
   layer Jacobians are **rectangular** for varying widths, so eigenvalues are the wrong object —
   singular values / operator norm is the honest quantity. Meanwhile the slides (:528–543) use
   *spectral radius* and the ρ^{L−ℓ} compounding rule — the slides are more rigorous than the
   prose. Fix the prose: "each Jacobian can stretch or shrink vectors by widely varying factors
   (its singular values); their product compounds geometrically — see the spectral-radius account
   in :numref:`sec_mdl-eigendecompositions`." Note: the appendix's tailor-made section "Spectral
   Radius, Stability, and Deep Networks" (mdl-eigendecomposition.md:1051) has **no subsection
   label**, so the pointer must target the section label (or a label should be added there in a
   separate appendix-side edit).
4. numerical-stability-and-init.md:152–161 [weak] The quantitative vanishing argument
   (0.25^10 ≈ 10⁻⁶) lives **only in slides** (:552 here; also mlp.md slides :1017–1020). The prose
   says "the gradients of the overall product may vanish" qualitatively. One sentence carries the
   number into prose.
5. numerical-stability-and-init.md:181–215 [weak] The 100-matrix product demonstrates *exploding*
   only; nothing in code demonstrates *vanishing* or the fix. The best missing cell of the section
   (and arguably the chapter): a **depth sweep** — 50 linear(+ReLU) layers of width 100, plot
   Var[h⁽ˡ⁾] vs ℓ under N(0,1) / Xavier / He on one axes. That is currently exercise 4; promote a
   compact version to a teaching cell (keep the exercise as the ReLU/extension variant). It
   demonstrates the entire section's thesis in one plot and directly justifies He-for-ReLU.
6. numerical-stability-and-init.md:313–320 [polish] The variance computation silently drops the
   cross terms E[w_ij w_ik x_j x_k] (j≠k); they vanish by independence + zero-mean weights. Half a
   sentence, since the chapter elsewhere prides itself on airtight small proofs.
7. numerical-stability-and-init.md:67 [polish] v⁽ˡ⁾ = ∂_{W⁽ˡ⁾}h⁽ˡ⁾ called "the gradient vector" —
   it is a matrix/3-tensor; inherited looseness, one word ("operator") fixes.

**Proposed improvements.**
1. (S) Fix the He lemma and exercise 3 to second moments (issue 1); fix the PyTorch-default claim
   (issue 2).
2. (S) Rewrite :81 per issue 3 with the `sec_mdl-eigendecompositions` pointer; carry 0.25^10 into
   prose.
3. (M) Add the depth-sweep variance cell (issue 5) — this also gives §5.4 a figure-quality payoff
   plot from code (a data plot that teaches, so inline is house-legal).

### Section: generalization-deep.md (§5.5)
**Strengths.** The streamlining already done on this branch works: double descent has one
authoritative in-chapter treatment (own subsection, mechanism paragraph :197–209, house figure)
with clean verified hand-offs to `sec_mdl-concentration-generalization` (:171, :208 both resolve).
The implicit-regularization subsection is current and correct (Soudry max-margin, grokking cited to
Power et al.), the NTK/nonparametric lens is well told, and early stopping carries the
label-noise/realizable nuance most intros miss. Tone ("pour yourself a drink") is a feature.

**Issues.**
1. generalization-deep.md:190–192 [weak] Epoch-wise and sample-wise double descent get one clause
   ("also as we train for more epochs or add more data"). I checked appendix §25.5: it develops
   **model-wise only** (minimum-norm mechanism, 25-line reproduction, benign overfitting) — no
   epoch-wise, sample-wise, or effective-model-complexity anywhere in the book. Digest claim
   **accepted, adapted**: add one short paragraph (not a subsection) naming Nakkiran's three
   flavors, with the sample-wise "more data can hurt near the threshold" called out as the
   genuinely shocking one, and "effective model complexity" as the unifying axis. Keeps the
   spiral: chapter names, appendix proves (model-wise).
2. generalization-deep.md:243–245 [weak] "the 1-nearest neighbor algorithm is consistent
   (eventually converging to the optimal predictor)" — false in general. Cover–Hart: asymptotic
   1-NN risk satisfies R* ≤ R ≤ 2R*(1−R*); it is optimal only when Bayes risk is 0. k-NN with
   k→∞, k/n→0 is what is consistent. Fix: "comes within a factor of two of the optimal error
   (Cover & Hart, 1967), and is optimal in the noiseless case" (needs a bib entry).
3. generalization-deep.md:379–383 [polish] "flat minima, which tend to generalize" — uncited and
   never operationalized. One sentence + SAM cite (Foret et al. 2021 — **not in d2l.bib**, needs
   entry) turns the correlational claim into an actionable algorithm pointer. Digest accepted at
   one-sentence weight, not more.
4. generalization-deep.md:390–395 [polish] Grokking is prose-only; it is the one place in the
   section where a picture does 90% of the work. See chapter-wide figures (mdl-mlp-grokking.svg).
5. generalization-deep.md:437–443 [polish] Exercises cover only early stopping + classical bounds —
   nothing on double descent, implicit bias, or grokking, the section's actual headliners. Add 2:
   (a) reproduce epoch-wise double descent by training the §5.2 MLP far past convergence with 15%
   label noise (ties to early stopping!); (b) starred: grokking on modular addition (point to
   Power et al.'s setup).
6. [polish] Section has no code. **Rebutting the implicit push for an in-chapter double-descent
   demo**: appendix §25.5 owns the 25-line reproduction; duplicating it here would violate the
   spiral. The exercise in (5a) is the right vehicle. No change to prose.

**Proposed improvements.**
1. (S) Three-flavors paragraph (issue 1); Cover–Hart fix (issue 2); SAM sentence (issue 3).
2. (S) Two new exercises (issue 5).
3. (M) Grokking schematic figure (see chapter-wide list).

### Section: dropout.md (§5.6)
**Strengths.** The three-views framing (Bishop/Tikhonov noise, co-adaptation, implicit 2ⁿ
ensemble) is the best short "why" for dropout I know of in a textbook, and the honesty about which
narrative is imposed (:56–58) is characterful. The E[h′]=h derivation with the "unique factor"
remark (:120–124) is exactly right. The currency paragraph (:538–545: BN in CNNs, 0.0–0.1 in
transformers, classifier heads) is more honest than any peer. Exercises are outstanding — MC
dropout with calibration (:554), DropConnect (:556), invent-your-own-noise (:557). The digest's
"name inverted dropout" is **already half-done** (summary :522 and slides); "MC dropout aside" and
"DropConnect sibling" are **already done as exercises** — right weight, rebut moving them to prose.

**Issues.**
1. dropout.md:82–83, 533–535 [weak] The ensemble view says test-time full network "approximates
   averaging the predictions of all 2ⁿ subnetworks" — the hedge "approximates" is present (softer
   than the known-issue billing), but the standard sharpening is still missing: the weight-scaling
   identity is **exact only for a single linear layer** (and yields a *geometric* mean of softmax
   outputs in deeper nets). One sentence at :83 or in the summary.
2. dropout.md:102–124 [weak] "Inverted dropout" is named only in the Summary (:522). Name it at
   the definition where 1/(1−p) is derived, and add the one-line historical contrast: the original
   formulation left activations untouched in training and multiplied by (1−p) at *test* time;
   inverting moves all bookkeeping into training so inference code never changes. That contrast is
   load-bearing (it explains why `Dropout` layers are no-ops in eval) and currently absent.
3. dropout.md:538–545 [polish] The BN paragraph should carry the one-sentence dropout↔BN
   *variance-shift* disharmony (Li et al. 2018, arXiv:1801.05134 — **not in d2l.bib**): BN's
   running statistics accumulated under dropout noise mismatch eval-time variance, so don't
   sandwich dropout before BN. Digest accepted at one-sentence weight.
4. dropout.md:180–184 [polish] The test-time-uncertainty paragraph duplicates exercise 5 (MC
   dropout, with the Gal & Ghahramani cite). Compress to one sentence pointing at the exercise.
5. dropout.md:294–296 vs 396–397 [polish] Prose states the convention "lower dropout probability
   closer to the input layer", then the training run uses 0.5/0.5 — the code never exercises the
   stated rule. Either set (0.2, 0.5) in `hparams` or cut the convention claim; slides repeat it
   (:774–777) so alignment matters.
6. dropout.md:235–251 [polish] The 11-line JAX mutable-default-key comment is correct and
   valuable, but it teaches a framework trap, not dropout; consider trimming to 4 lines +
   pointing at the fuller :begin_tab: jax note at :482–495.
7. Digest "dropout is oversized / cut one implementation": **rebut**. The scratch+concise pair is
   the book's structural identity (identical to §5.2), each is short, and the prose is lean. The
   real compressions are issues 4 and 6 only.

**Proposed improvements.**
1. (S) Name inverted dropout at the definition + original-formulation contrast (issue 2); add the
   exact-only-for-linear sentence (issue 1).
2. (S) Variance-shift sentence + bib entry (issue 3); compress the MC-dropout paragraph (issue 4);
   fix the 0.5/0.5-vs-convention mismatch (issue 5).

### Section: kaggle-house-price.md (§5.7)
**Strengths.** The leakage discussion (:199–214) — statistics from train only, why, and the named
pitfall — is exemplary and correctly mirrored in code (:242–245). The log-RMSE ratio identity
(:287–288) is verified correct. The "a baseline only counts if trained competently" passage
(:424–433, 0.18 underfit vs 0.05 converged) is a distinctive, honest lesson almost no textbook
teaches. The trees-beat-nets caveat with citations (:413–422) sets expectations exactly right. The
small-regularized `KaggleMLP` rationale (:448–459) is sound.

**Issues.**
1. kaggle-house-price.md:515–526 [error] The pytorch submission cell's comment contradicts its
   code: the code computes `mean(exp(preds))` — an **arithmetic mean in price space** — while the
   parenthetical says "(Averaging in log space, then exponentiating, makes this a geometric mean
   of the per-fold price predictions.)", describing the *other* order as if it were what the cell
   does. Worse, since the competition metric is RMSLE, the metric-consistent ensemble is precisely
   the log-space mean (geometric mean of prices) that the comment gestures at and the code does
   not implement. Fix code to `exp(mean(preds))`, and say *why* in one sentence (mean of
   log-predictions minimizes squared log error). This also makes a lovely callback to the Error
   Measure section.
2. kaggle-house-price.md:504–512 [weak] K-fold ensembling honesty (the review's assigned check):
   averaging the K fold models is presented as the natural next step with no acknowledgment that
   (a) each member saw only (K−1)/K of the data, (b) the reported "average validation log mse" is
   an estimate for a *single* model, not for the ensemble being submitted, and (c) the canonical
   alternative is refit-on-all-data with the chosen hyperparameters — which the **slides claim we
   do** ("5. Refit with the chosen hyperparameters", :1016–1017) but the section never does.
   Prose/slide inconsistency + a teachable moment: one short paragraph naming the choice
   ("fold-ensembling: standard Kaggle practice, mild extra variance reduction, but your CV score
   no longer measures the thing you submit; refitting is the cleaner experiment") fixes both.
3. kaggle-house-price.md:155–157 [weak] "the **validation** data contains 1459 examples" — that is
   the Kaggle *test* set (`raw_val`/`data.val`); calling it validation collides head-on with the
   K-fold validation folds introduced 150 lines later. Rename in prose (keep code attribute if
   churn is undesirable, with one clarifying sentence: "we store the test features in `val`
   because the loader treats any unlabeled split uniformly").
4. kaggle-house-price.md:336–344 [polish] `k_fold_data` uses `fold_size = n // k`, so the n mod K
   remainder rows are in *every* training split and *no* validation fold. Harmless here
   (1460 = 5×292 exactly) but the function reads as reusable; add a comment or use index
   partitioning.
5. kaggle-house-price.md:435–445 [polish] Tab asymmetry: pytorch gets the 100-epoch competent
   baseline the prose narrates; mxnet/tf/jax tabs still run 10 epochs at lr 0.01 — those readers
   execute exactly the underfit trap the surrounding prose warns against, and see ~0.18. At
   minimum bump non-pytorch tabs to matching budgets; the `k_fold` signature difference
   (`model_fn` vs `lr`) also means only pytorch can reuse the loop for the MLP.
6. kaggle-house-price.md:249 [polish] `pd.get_dummies` on concatenated train+test defines the
   one-hot *schema* from test rows. Label-free and standard, but after the strong leakage sermon a
   half-sentence distinguishing "schema from test (fine, it's what deployment sees)" vs
   "statistics from test (leakage)" would be sharp.
7. kaggle-house-price.md:517 [polish] The final `model(...)` calls rely on `Trainer.fit` having
   left the model in eval mode (verified: d2l/torch.py:375 ends epochs with `model.eval()`, so
   dropout is correctly off) — but it is implicit and fragile. Add `model.eval()` or a one-line
   comment; it models good practice for a section teaching practice.

**Proposed improvements.**
1. (S) Fix the ensembling cell: log-space averaging + corrected comment + one metric-consistency
   sentence (issue 1); add `model.eval()` (issue 7).
2. (S) Honesty paragraph on fold-ensembling vs refit; align the slide recipe wording (issue 2);
   fix "validation"→"test" terminology (issue 3).
3. (M) Bring the non-pytorch tabs up to the competent-baseline budget and `model_fn` interface
   (issue 5).

### Chapter-level coverage

**Add.**
- UAT intuition sketch + figure + region-count demo cell (§5.1) — the chapter's #1 gap; resolves
  the mlp.md:350 dangling promise in place.
- Gradients-add-at-forks call-out, autograd verification cell, appendix-AD forward pointer,
  micrograd capstone exercise (§5.3).
- Depth-sweep variance cell (§5.4) — the asserted-but-never-demonstrated claim of the chapter.
- Three-flavors double-descent paragraph + SAM sentence + 2 exercises (§5.5).
- Inverted-dropout naming at definition + scale-at-test contrast + variance-shift sentence (§5.6).
- Log-space ensembling + fold-ensemble honesty paragraph (§5.7).
- Not added (rebutted): full CS231n gate trio as prose (add/mul already present; max adds little
  for MLPs); in-chapter double-descent code (appendix §25.5 owns it); cutting dropout's dual
  implementation (house identity); moving MC dropout/DropConnect to prose (right weight as
  exercises); temperature/calibration material (ch4 territory per digest area 1).

**Cut/compress.**
- dropout.md:180–184 (duplicates exercise 5); dropout jax scratch comment block (trim to 4 lines).
- Optionally merge function+derivative activation plots in §5.1 (6 cells → 3 per framework).
- Nothing else — the chapter is not bloated; §5.5's prose-heaviness is earned.

**New figures (chapter-wide list).** All via `tools/gen_mdl_mlp_figures.py` house style
(generators for xor/backprop/double-descent/kfold already exist, so the pattern is established):
1. `img/mdl-mlp-uat-hinges.svg` — 2 panels. Left: three single-ReLU "hinges" a·ReLU(wx+b) on one
   axes (distinct joint locations marked with dots on the x-axis). Right: their sum + a smooth
   target curve f(x), shaded error band, joints again marked; annotation "width D ⇒ ≤ D+1 linear
   pieces". (If the bump-pair device is preferred instead: two steep offset sigmoids ±h summing to
   a rectangular bump, then 5 bumps tiling a curve — but the ReLU variant matches the chapter.)
2. `img/mdl-mlp-grokking.svg` — schematic train/val accuracy vs log₁₀(steps): train hits 100%
   early; val flat near chance for decades of steps, then a sharp late rise; dashed vertical
   "memorization" and "generalization" markers. Companion piece to mdl-mlp-double-descent.svg,
   same axes styling.
3. Restyle stragglers (one-style-per-chapter rule, currently violated): `mlp.svg` (fig_mlp, §5.1)
   and `dropout2.svg` (fig_dropout2, §5.6) are legacy hand-drawn network diagrams sitting next to
   five `mdl-mlp-*` house figures. Reproduce both in the generator style
   (`mdl-mlp-fc.svg`, `mdl-mlp-dropout.svg`), keeping labels/structure identical. (L, low
   priority, but it is the chapter's only figure-convention violation.)
4. (Optional, S) `img/mdl-mlp-vanishing-sweep.svg` is *not* needed if the §5.4 depth-sweep code
   cell is added — that plot teaches a computed result and may stay inline per house rules.

**Hand-off fixes.**
- numerical-stability-and-init.md:81 → point to `sec_mdl-eigendecompositions` (its "Spectral
  Radius, Stability, and Deep Networks" section at mdl-eigendecomposition.md:1051 is the intended
  target but has no subsection label — add `subsec_mdl-spectral-radius` there in an appendix-side
  edit, or reference the section label). Also rewrite eigenvalues→singular values (rectangular
  Jacobians) and reconcile the prose/slide rigor asymmetry.
- backprop.md:151 and :309 → add forward pointers to `sec_mdl-matrix-calculus-autodiff` (verified
  label; currently referenced only from mdl-* chapters — ch5 never points at the book's own
  reverse-mode theory). Keep the existing backward pointer to `sec_autograd` (verified: it does
  cover forward-vs-reverse at chapter_preliminaries/autograd.md:783–796, so :309's claim is
  honest).
- mlp.md:350–351 → replace "subsequent chapters" promise with in-section sketch + exercise 6
  (nothing downstream delivers; verified).
- generalization-deep.md:171/:208 hand-offs to `sec_mdl-concentration-generalization` verified
  working — no change (per instructions, not re-flagged).
- All other :numref: targets in the chapter verified to resolve (sec_batch_norm, sec_resnet,
  sec_lstm, chap_computation, subsec_model-construction-sequential, oo-design-training,
  subsec_generalization-model-selection, fig_capacity_vs_error, chap_classification_generalization,
  sec_attention-pooling, subsec_classification-problem, subsec_softmax_vectorization, …).
- New d2l.bib entries required by proposals: Leshno et al. 1993 (or drop), Montúfar et al. 2014 /
  Telgarsky 2016 (depth separation), Cover & Hart 1967, Foret et al. 2021 (SAM), Li et al. 2018
  (variance shift). Verified present already: Hornik.1991 (just needs :cite:), Gal.Ghahramani.2016,
  Power.Burda.Edwards.ea.2022, Soudry…, nakkiran2021deep, micchelli1984interpolation.

## 6. Reference landscape and calibration (2026)

Condensed from the two research digests (full digests were session artifacts; the durable
conclusions are folded into §2G/§3/§4 and the per-chapter coverage lists above). What "best in
existence" means per area, and where d2l stands:

- **Tensors/arrays** — NumPy broadcasting docs (visual right-aligned shape table), PyTorch
  broadcastable checklist, JAX sharp bits. d2l's prose rule is exact but pictureless; borrow the
  alignment table/diagram, the pairwise-distance "broadcasting does real work" example, and the
  error-first `(3,2)+(2,3)` demonstration. einops: one forward-pointer callout at most.
- **Linear algebra** — 3Blue1Brown (columns = transformed basis vectors; dual-grid pictures),
  MML/Deisenroth ch. 2–3 (norm → inner product → angle → orthogonality → projection arc),
  immersivemath. d2l ch. 2 stops at norms with zero geometry figures; the missing geometric layer
  is its weakest flank vs. the field. Confirmed right to omit: determinants, four subspaces,
  elimination, named decompositions (appendix territory).
- **Calculus/autograd** — Karpathy micrograd (build the engine; graphviz DAG with values+grads;
  finite-difference check), Colah (sum-over-paths, path factoring), JAX cookbook (jacrev/jacfwd).
  d2l's mechanics are strong and verified; it lacks the *why-reverse-mode* counting argument and a
  finite-difference-vs-autograd cell — both one-liners to add.
- **Probability** — Blitzstein (naive definition first, story proofs), Seeing Theory (CLT
  visuals), 3b1b (unit-square Bayes). d2l's HIV example is better than peers; its CLT is one
  sentence and its post-coin prose is code-free — the simulation cells close both.
- **Regression/generalization** — UDL ch. 5 (the distribution→NLL loss menu, Fig 5.11), Bishop
  2024 §4.3/§9.3.2 (plotted bias–variance; double descent as mandatory context), ISL (stated-not-
  proved tier; the k-in-K-fold tradeoff §5.1.4), ESL §3.4.1 (per-direction shrinkage, effective
  dof), fast.ai (AdamW two-liner). d2l has the derivations but not the named recipe, the computed
  decomposition, or the quantitative shrinkage story — all sized and placed in §3/§5.
- **Classification** — CS231n notes (decision-boundary geometry), Google MLCC + scikit-learn
  (metrics baseline: confusion matrix, precision/recall, ROC/PR, reliability diagrams), Guo et al.
  (calibration), WILDS (shift axes). d2l exceeds all of them on derivations and already covers
  calibration in prose; it is below the 2026 metrics baseline only on the confusion-matrix/
  precision-recall thread, and its shift chapter remains best-in-class with two one-paragraph
  modernizations (WILDS axes, OOD callout).
- **MLPs/backprop/init/generalization** — Nielsen ch. 4 (bump construction) and UDL ch. 3
  (hinge/region counting) for UAT; CS231n gate patterns + Karpathy's failure modes for backprop;
  CS231n/fast.ai for init (Xavier derivation still standard; LSUV/μP correctly out of scope);
  Nakkiran (three double-descent flavors, effective model complexity), Power et al. (grokking),
  Foret et al. (SAM) for the 2026 generalization survey; CS231n inverted dropout + Li et al.
  variance shift. d2l's backprop numeric example beats every peer artifact; its UAT passage is the
  one place a peer device is strictly better than what exists.

Cross-cutting: the best 2026 treatments (a) verify claims in code, (b) lead with geometry,
(c) name their reusable recipes, and (d) pair every classical curve with its modern counterpoint
(U-curve ↔ double descent, accuracy ↔ calibration). d2l ch. 2–5 already wins on (a)'s
infrastructure and loses on execution density; (b)–(d) are what §3's top-10 list buys.
