# Math appendix writing pass: outstanding issues (2026-11-10)

State after the writing pass over chapters 38 to 43 (36 sections, six
introductions, six further-reading files). The pass changed prose and inline
mathematics only: no code cell, cell ID, figure include or slide section was
touched, and no label was removed, so the committed `outputs/` store is still
fresh. All 48 files pass the prose-only check and `tools/lint_source.py`;
every `:numref:`/`:eqref:`/`:exref:` to an appendix label resolves across the
book. The book was not rendered.

Related files: `overview.md` (review, author decisions at the top), the
chapter and section reviews, and `writing-pass/<chapter>/<file>.md` (one log
per file: correctness changes with how each was verified, before/after pairs,
repetition removed, labels added, cells for the development machine, claims
added, slide disagreements).

## 1. What was done

- **Correctness.** The findings of the review that could be fixed in prose
  were applied and verified (NumPy or by hand; this laptop has no framework
  environments). The main ones: the step-size threshold on $x^4/4$ is $2/9$
  (40, gradient descent); the edge-of-stability passage agrees with chapter
  9; the Hilbert example blames the right-hand side; Slater's condition is
  defined once (`eq_mdl-opt-slater`) and cited; Fokker-Planck uniqueness is
  stated once and referred to; results in the SDE section are labelled as
  proved, sketched or cited; the adjoint is the total derivative and
  $\partial$ is kept for edge derivatives throughout chapter 39; $P$ is the
  data and $Q$ the model throughout the divergences section; "Heun's errors
  cancel on the crowded grid" was false and is corrected (43 and
  `chapter_diffusion-models/samplers.md`).
- **Ownership.** Gibbs' inequality is proved once in chapter 41 (exponential
  family, `eq_mdl-exp_family_gibbs`) and once in chapter 42; maximum
  likelihood cites it. Chapter 39 teaches the adjoint, the layer rules and
  the algorithm once each. Joint and conditional entropy are defined in the
  first section of chapter 42 (`subsec_mdl-joint-conditional-entropy`).
  Data processing has its own label (`subsec_mdl-f-div-data-processing`).
- **One ending per section, cell introductions varied, results ranked**
  (Proposition blocks kept for results that are cited or matter; skippable
  parts marked).
- **Splits prepared, not made.** Markers `<!-- split: ... -->` in:
  eigendecompositions (1), multivariable calculus (2, see below), integral
  calculus (1), gradient descent (1), random variables (1), maximum
  likelihood (1), divergences (2), ODEs and solvers (1). Each part has its
  own opening and its own group in the Summary.
- **Added for the main text:** the driven linear ODE and its exact
  discretization (`subsec_mdl-ode-driven-linear`, `eq_mdl-ode-zoh`).
- **Main-text pointers fixed** (one sentence each): `backprop.md`,
  `lr-scheduler.md` (the warmup claim its cited experiment refutes),
  `gaussian-paths.md`, `samplers.md`, `relativistic.md`, and the diffusion
  chapter's further reading (a circular pointer).

## 2. Length: the part is not shorter

About 322,200 prose words before and 321,300 after (the checker's count,
exercises included). Each writer cut 400 to 700 words of repetition and spent
most of it on what the review asked for: hypotheses, proof repairs, honest
readings of experiments, ranking sentences, openings for the future sections.
Six files grew: divergences (+257, three topics kept by the author's
decision), score matching (+184), Fokker-Planck (+160), coding (about +90),
neural ODEs (about +90), MI estimation (+86). The cuts the review proposed
(a quarter of several sections) need cells removed, which is development-
machine work (section 4).

## 3. For the author to read and confirm

**New mathematics written by the pass** (each checked by its writer, logged
with the method; worth an author's eye):

- 41 distributions: the de Moivre-Laplace step, now called a sketch.
- 40 constrained: penalty implies constraint for $\ell_1$ in one line; the
  converse is stated as not proved. 40 duality: a paragraph on general
  penalties.
- 38 linear maps: a block-triangular determinant rule (used by the repaired
  proof in eigendecompositions); invertibility now proved in "Inverses".
- 38 SVD: a three-sentence justification inside the Eckart-Young proof.
- 41 uniform convergence: the noise term of benign overfitting as noise
  variance times an average feature ratio per singular direction.
- 42 mutual information: Gaussian mutual information by conditional
  variance; invariance stated without smoothness. 42 MI estimation: two new
  parts of exercise 6 with hand-derived answers.
- 43 ODEs: variation of constants and zero-order hold (checked against RK4).
- 41 Bayesian computation: a formula predicting a log shift of about 2.9
  where the run measures 1.6; the text explains the gap loosely.

**Judgments that may be too strong.** "Three results matter most" in each
introduction (the writers' picks); Gershgorin and PageRank marked skippable;
"this bound is what made the computation feasible for the web"; "a network
is both misspecified and non-identifiable"; VC dimension "mainly of
theoretical interest"; "Most introductions never say this" (uniform
p-values); coordinate descent: "nothing proved predicts the measured
factor"; "Adam ... resembles this sign step"; reflow as the result "practice
leans on most"; "No network is trained in this section" (score matching);
"The width excess is the bias of Euler-Maruyama".

**Claims from memory of a paper or book, not re-read.** Bertoin et al. on
ReLU'(0) by precision; Reddi et al. counterexamples for every
$\beta_1 < \sqrt{\beta_2}$; Pitman-Koopman-Darmois (no citation); Ng and
Jordan (logarithmic against linear in $d$); Roberts and Tweedie for Langevin
convergence; Øksendal chapter references; Bogachev et al. for uniqueness;
Benamou-Brenier in weak form; heavy tails of gradient noise "still
disputed"; "latents rescaled to unit variance". The further-reading files
softened or dropped about twenty attributions; each log lists them.

**Review findings the writers declined**, with reasons, are in the logs
(for example distillation kept in the entropy section because three
language-model chapters cite it; one subsection kept for its label).

## 4. Development machine

Per file, the log section "cells to cut or redesign" has the cell IDs.

- **Make the splits** at the markers (eight files become eighteen), update
  `_quarto.yml` and `CHAPTER_NUMBERING`, re-execute. Multivariable calculus
  needs cells reordered first so that critical points are contiguous (its
  log describes the order). Integral calculus: move the second part's
  density and Monte Carlo material to chapter 41, keep the gradient
  estimators.
- **Confirmation cells** (author's rule): linear maps (9), matrix calculus
  (4), distributions (5), SDEs (6), convexity (4), and those listed in the
  single-variable, multivariable, divergences and SVD logs. Cut or redesign.
- **Checks that cannot fail**: Langevin and reverse-sampler mass on a
  symmetric mixture (use 0.8/0.2 weights); the matrix-calculus random-point
  test (use square shapes); the p-value histogram in statistics.
- **Re-capture**: Bayesian computation's running data set (fix the data
  set); the Fokker-Planck reverse sampler; items in the calculus, statistics
  and naive Bayes logs.
- **Numbers no cell prints**: kept only where recomputed; a cell should
  print them (MI estimation, flow matching, coding, eigendecompositions, SVD).
- **Reorders that move cells**: linear maps (`einsum`, limits), convexity,
  exponential family (maximum entropy last), duality (conjugates), flow
  matching (FMT-4).
- **Code-bound notation**: $X$ target / $Y$ input in random variables; `B`,
  `k` in statistics; the confusion-matrix orientation in naive Bayes
  (transpose of the main text's); $\mathbf A$ for the generic matrix and
  both running examples in linear maps; `B` for the transient-growth matrix
  against the new input matrix $\mathbf B$ in ODEs; old $P$/$Q$ letters in
  the Stein and KSD cell comments.
- **Figures**: `fig_mdl-cal-fwd-vs-rev` (may still draw $\partial$ for
  totals); the residual-network-as-Euler caption changed without a re-render.

## 5. Slides are out of step

Slide sections were not edited. Almost every log ends with the slide
statements that now disagree with the body, among them: the $2/27$
threshold; "1×1 Check Catches Shapes"; Adam's 48 against 296; "Two
failures"; $e^{H}$ against $2^{H}$ for the perplexity floor; fp/bf naming;
the old $P$/$Q$ letters; Heun's cancellation. The re-sync is mechanical.

## 6. Main text: not done

- Add pointers where the main text should cite the appendix: `resnet.md`
  (Euler reading, sentence in the ODE log), `ssm.md` (`subsec_zoh` to
  `subsec_mdl-ode-driven-linear`), `muon.md` (Newton-Schulz analysis in
  `subsec_mdl-orthogonalized-updates`), a main-text link to naive Bayes.
- Chapter 9 against chapter 40: the per-item sorting in the overview
  (decision 2) has been applied on the appendix side only.
- Notation differences with chapter 9 (`adam.md`: placement of
  $\epsilon$, $\mathbf w$ against $\mathbf x$) and between chapters 38 and
  40 (ridge filter symbols), and the clipping point $y$ against $x_0$.
- `mdl-mutual-information.md` cites `sec_lm_word_embeddings`, still a
  skeleton.
- "Resources and Further Reading" (introductions) against "Further Reading"
  (files): purposes are now stated, the headings are unchanged.
- `CLAUDE.md` still calls chapter 17 a placeholder.
- `chapter_builders-guide/_outline/numerics.md` cites a label
  (`sec_mdl-numerics`) that does not exist; it predates this pass.
