# Writing Review of "Mathematics for Deep Learning": Overview

Date: 2026-11-09. Source: branch `basics-writing-pass` at `a6d9c39b` plus the
review commits (the six `chapter_mdl-*` directories are unchanged from
`main`). Scope: chapters 38 to 43, all 36 sections, the six chapter
introductions and the six further-reading files, about 390,000 prose words.
Question asked: the same as for Part 1. How clear, well thought out, well
taught and enjoyable is this part, measured against Cover and Thomas, MacKay,
and The Illustrated Transformer? No source text was changed.

Method: 20 reviewers each read two or three sections in full (prose, proofs,
code, exercises, stored PyTorch outputs, rendered figures), recomputed worked
numbers, and wrote one file per section. Six further readers then wrote the
chapter files from the section files and checked the most serious
cross-section claims in the sources. This overview is written from the
chapter files and the reviewers' reports. Not checked anywhere: the
TensorFlow, JAX and MXNet outputs (most cells are NumPy and differ only in the
imports), the slides, the rendered HTML, and most statements attributed to
papers (checked from memory, flagged in the section files).

## Author decisions so far

1. **Cells that only confirm a proved theorem** (Alex Smola, 2026-11-09):
   "Confirming proved theorems by sampling is not a great thing to do. It's
   nice once or twice but not worth doing excessively, unless the experiment
   teaches something in addition to the theorem." Finding 4 below lists
   where this applies. Where section reviews disagree about such cells (one
   praised them in `42-information-theory/mdl-divergences-distances.md`),
   this decision settles it.
2. **Duplication between the appendix and the main text** (Alex Smola,
   2026-11-09, said of chapter 40 against chapter 9): "some duplication with
   chapter 9 is inevitable, if we don't want the reader to jump between both
   chapters all the time. If it's small, duplicate, if it's a complex
   derivation, put it into the math appendix and if it's only specific to the
   optimizations chapter, it's OK to keep it only there. In other words, the
   math part is additional reading for someone wanting to understand the
   mathematics and dive more deeply into it." So the rule is by size and
   kind, not "one home for everything": small results may appear in both
   places; a long derivation lives in the appendix and the main text states
   the result and points to it; material specific to the main chapter stays
   there alone. The 14-item table in `40-optimization.md` is to be read
   against this rule (see finding 2), and the same test is the natural one
   for chapter 43 against chapter 17 and for the overlaps with Basics. The
   main text must stay readable without the appendix.
3. **Long sections** (Alex Smola, 2026-11-09): "if sections are overly long
   and need splitting, split them". Which ones is under discussion; see
   finding 3.
4. **Closings and cell introductions** (same date): "remove the double
   summary and fix the mechanical question-then-cell formula. it should read
   organic." So the "We can now" recap goes (anything it holds that the
   Summary lacks moves into the Summary), and cells are introduced the way
   the surrounding argument calls for, not by one formula. See finding 6.
5. The Part 1 decisions carry over unless changed: the voice model is chapter
   1 of *Learning with Kernels* (writing guide 10.2), repetition is removed
   (guide 7.7), ranges for random experiments are deliberate.

## Verdict

This part is better than Part 1 was before its writing pass, and its faults
are different. The mathematics is sound: across 36 sections and several
hundred propositions, proofs and derivations, the reviewers found **no false
theorem and no failing proof**. What they found is one wrong explanation
(the quartic step-size ceiling), about twenty statements in the surrounding
prose that are false or need a hypothesis, a few proofs that use a fact
before it is established, and about ten places where a stored output or an
unprinted number disagrees with the text. The sections are built on small,
concrete running examples, most cells are preceded by a prediction, and the
exercises are the best in the book. Several passages meet the bar outright
(listed at the end).

What keeps the part below the bar is **economy and ownership**, not
sentences. The same result is taught two to four times: inside a section,
in neighbouring sections, in another appendix chapter, and again in the main
text, often at the same level and sometimes with different numbers. A third
of the sections are really two or three sections under one title. Nothing is
ranked: a one-line identity gets the same Proposition and Proof block as
Fano's inequality. Every section ends twice (a "We can now" recap, then a
Summary that repeats it), most cells are introduced by the same
question-then-cell formula, and many cells confirm what was just proved.
The part reads as six carefully made chapters that were written without
looking at each other or at the main text.

## Scores

Five dimensions, 1 to 5: explanation / thought / pedagogy / language /
enjoyment. The scale is that of the Part 1 review (5 = at home in Cover and
Thomas or MacKay). Calibration note: 20 reviewers, and about 85% of all
scores are 4, so the table discriminates poorly. The findings carry the
information; read a 3 as "a structural problem" and a 5 as "protect this".

| Section | E | T | P | L | J |
|---|---|---|---|---|---|
| **38 Linear Algebra** | | | | | |
| Geometry and linear-algebraic operations | 4 | 4 | 4 | 4 | 4 |
| Linear maps | 4 | 3 | 3 | 4 | 3 |
| Eigendecompositions | 4 | 4 | 3 | 4 | 4 |
| SVD and low-rank approximation | 4 | 4 | 4 | 4 | 4 |
| **39 Calculus** | | | | | |
| Single-variable calculus | 4 | 4 | 4 | 4 | 3 |
| Multivariable calculus | 4 | 3 | 4 | 4 | 3 |
| Matrix calculus | 4 | 4 | 4 | 4 | 3 |
| Automatic differentiation | 4 | 4 | 4 | 4 | 4 |
| Integral calculus | 4 | 4 | 4 | 4 | 3 |
| **40 Optimization** | | | | | |
| Gradient descent and curvature | 4 | 4 | 4 | 4 | 4 |
| Convexity | 4 | 4 | 3 | 4 | 3 |
| Stochastic gradient methods | 4 | 4 | 4 | 4 | 3 |
| Adaptive methods | 4 | 3 | 4 | 4 | 4 |
| Constrained optimization | 4 | 4 | 4 | 4 | 3 |
| Duality | 5 | 4 | 4 | 4 | 4 |
| Numerical stability and conditioning | 4 | 4 | 5 | 4 | 4 |
| **41 Probability and Statistical Learning** | | | | | |
| Random variables | 4 | 4 | 4 | 4 | 4 |
| Distributions | 4 | 3 | 4 | 4 | 3 |
| The exponential family | 4 | 4 | 4 | 4 | 4 |
| Statistics | 4 | 4 | 4 | 4 | 4 |
| Maximum likelihood | 4 | 4 | 4 | 4 | 3 |
| Bayesian computation | 4 | 4 | 4 | 4 | 4 |
| Concentration inequalities | 4 | 4 | 4 | 4 | 4 |
| Uniform convergence and generalization | 4 | 3 | 4 | 4 | 4 |
| Naive Bayes | 4 | 4 | 4 | 4 | 4 |
| **42 Information Theory** | | | | | |
| Information theory (entropy) | 4 | 3 | 4 | 4 | 3 |
| Coding and compression | 4 | 4 | 5 | 4 | 4 |
| Divergences and distances | 4 | 4 | 3 | 4 | 3 |
| Mutual information | 4 | 4 | 4 | 4 | 3.5 |
| Estimating mutual information | 4 | 4.5 | 4 | 4 | 4 |
| **43 Dynamics** | | | | | |
| ODEs and solvers | 4 | 4 | 4 | 4 | 4 |
| Neural ODEs | 4 | 4 | 4 | 4 | 3 |
| SDEs | 4 | 4 | 4 | 4 | 3 |
| Fokker-Planck and probability flow | 4 | 4 | 5 | 4 | 4 |
| Score matching | 4 | 4 | 3 | 4 | 3 |
| Flow matching | 4 | 4 | 4 | 4 | 4 |

The chapter introductions score 4 with one shared defect (finding 6); the
further-reading files score 3 to 4 (finding 9). Chapter files:
`38-linear-algebra.md`, `39-calculus.md`, `40-optimization.md`,
`41-probability-statistics.md`, `42-information-theory.md`,
`43-dynamics.md`; section files in the directories of the same names.

## The main findings

### 1. Correctness: sound proofs, a short list of wrong statements

Each chapter file has the full list with IDs and the reviewer's confidence.
The items of substance, most serious first. IDs are local to the section
file named.

**A wrong explanation.**

- `40-optimization/mdl-gradient-based-optimization.md` GD-4. Under
  "Backtracking Line Search" the divergence of $\eta = 0.3$ on $x^4/4$ from
  $x_0 = 3$ is explained by "the local stability ceiling is
  $2/27 \approx 0.074$". The step sizes 0.1 and 0.2 exceed that ceiling and
  converge; the threshold from this start is $2/9$. The section's own
  exercise 4(b) says so. Verified independently for this overview. The slide
  repeats it.

**False or overstated statements in prose.**

- `40-optimization/mdl-numerical-stability-conditioning.md` NSC-17: the
  Hilbert example blames rounding of the matrix entries; the deviation comes
  from rounding in the right-hand side, and the three quoted errors come from
  no cell and did not reproduce.
- `40-optimization/mdl-gradient-based-optimization.md` GD-8: the per-mode
  recursion is said to explain why the sharpness stops at the ceiling.
  Chapter 9 and this chapter's further reading say, correctly, that the
  quadratic model does not.
- `41-probability-statistics/mdl-concentration-generalization.md` CON-13: the
  squared norm's exponential tail is given as the norm's.
- `41-probability-statistics/mdl-exponential-family.md` EXP-9: the Summary's
  "prediction minus observation" holds for the canonical link only; probit,
  discussed in the section, is a counterexample. EXP-14: the pseudo-count
  reading of Beta$(a, b)$ disagrees between text, exercise and the
  maximum-likelihood section.
- `41-probability-statistics/mdl-generalization-bounds.md` GEN-16: a bound is
  applied to a squared-loss regression the theorem does not cover. GEN-7:
  theorem in expected complexity, experiments in empirical complexity.
- `43-dynamics/mdl-sdes.md` SDE-11 (data "are normalized to unit variance
  before training a diffusion model"), SDE-14 (Rademacher data "as far from
  the Gaussian as unit-variance data can").
- `39-calculus/mdl-matrix-calculus.md` MC-11: "This 1×1 check catches wrong
  shapes" is backwards. MC-7: an asymptotic statement given as a bound.
- `39-calculus/mdl-multivariable-calculus.md` MVC-12: $\partial f/\partial p$
  is taught as the edge derivative, then the adjoint is defined with the same
  symbol meaning the total; the autodiff section repeats it.
- `42-information-theory/mdl-divergences-distances.md` DD-22 ("each symbol
  with one meaning" is false), DD-28 ($P$ and $Q$ swap roles from Stein's
  identity on), DD-20.
- Further-reading files: quasi-Monte Carlo bound (`39-calculus` FR-4),
  subgradient rate (`39-calculus` FR-2), "every estimator is a lower bound"
  (`42-information-theory` FR-6).

**Proofs that use a fact before it is established, or need a hypothesis.**

- `43-dynamics/mdl-fokker-planck-probability-flow.md` FPK-4, FPK-7, FPK-11:
  the section proves that the SDE's marginal solves the PDE and then uses the
  converse three times without stating uniqueness. One sentence each.
- `38-linear-algebra/mdl-linear-maps.md` MAP-14 (uniqueness of the
  determinant used before it is derived), `mdl-geometry-linear-algebraic-ops.md`
  GEO-8 (projection theorem assumes a basis built afterwards),
  `mdl-eigendecomposition.md` EIG-6.
- `41-probability-statistics/mdl-random-variables.md` RV-1 (near-circular
  proof of $F' = p$), `mdl-distributions.md` DIST-14 (two Gaussian
  propositions proved in a loop), DIST-9 (a sketch called "the de
  Moivre-Laplace theorem").
- `40-optimization`: Slater's condition is defined twice with different
  hypotheses (`mdl-constrained-optimization-duality.md` CON-6,
  `mdl-lagrangian-duality.md` DUA-7); a leading-order derivation is closed
  with a proof mark (`mdl-stochastic-gradient-methods.md` SGM-8).
- `43-dynamics/mdl-flow-matching-transport.md` FMT-5: the hypotheses exclude
  the section's own two-point and two-moons examples.
- "Proposition" labels proved, heuristic and unproved results alike in
  `43-dynamics/mdl-sdes.md` (SDE-8).

**Stored outputs that contradict the prose** (re-capture on the development
machine; no prose change):

- `41-probability-statistics/mdl-bayesian-computation.md` S1: the section's
  running data set is an atypical draw (plug-in 0.096 against predictive
  0.165, a factor 1.7, where the text says "typically three to ten times").
  Fix the data set, leave the samplers unseeded.
- `43-dynamics/mdl-fokker-planck-probability-flow.md` FPK-13: reverse-sampler
  mass 0.487 is a 3.7σ draw.
- Smaller: `39-calculus` IC-24, IC-11, IC-30, SVC-1, AD-5, AD-7;
  `41-probability-statistics` STA-17, NB-8; `43-dynamics` SMD-14.

**Numbers that no cell prints.** About forty across the part, nearly all
verified true by the reviewers (about ten in
`42-information-theory/mdl-mi-estimation.md`, about ten in
`43-dynamics/mdl-flow-matching-transport.md` FMT-11 and FMT-12, others in
`38-linear-algebra` EIG-9 and SVD-8, `40-optimization` NSC-11). Either a
cell prints them or the prose drops them.

### 2. The same result is taught two to four times

This is the largest finding and the one that needs decisions binding more
than one chapter. Each chapter file has a table with a proposed single home.
The cases that cross chapters:

- **Entropy, KL divergence, cross-entropy, Gibbs' inequality, maximum
  entropy.** Proved in chapter 41 twice (exponential family and maximum
  likelihood, same tangent-line argument, confirmed in the sources) and
  again in chapter 42, which says chapter 41 only "used the answer".
- **Backpropagation, the adjoint, the cost-by-order argument, the
  log-sum-exp Hessian.** Each in three sections of chapter 39 (multivariable,
  matrix calculus, autodiff), and backpropagation again in chapters 1 and 4
  of the main text. The October finding "backprop is explained three times"
  is still open.
- **Optimization inside calculus.** The descent lemma, Newton's error
  recursion and subgradients are proved in single-variable calculus and again
  in chapter 40 (open since October).
- **Appendix against chapter 9.** The edge-of-stability experiment (same
  network, same step sizes), Newton's derivation, the noise ball and
  Robbins-Monro conditions, the schedule comparison, the AdaGrad to Adam
  derivation, coupled against decoupled decay, Newton-Schulz and Muon. The
  division "chapter 9 measures, the appendix proves" is sensible and mostly
  kept; `40-optimization.md` lists 14 items taught in both. By author
  decision 2 these are not all defects. Sorted by that rule: *small, fine in
  both* (Robbins-Monro conditions, the statement of the noise ball, coupled
  against decoupled decay, the AdaGrad to Adam update rules; make the two
  versions agree in notation and numbers and let each cite the other);
  *complex derivation, appendix owns it* (Newton's convergence analysis, the
  noise-ball and schedule rate proofs, Newton-Schulz convergence; chapter 9
  states the result and points here); *specific to chapter 9, keep only
  there* (the edge-of-stability experiment on a network, the schedule
  comparison on a trained model, Muon as an optimizer). What remains a
  defect is the same experiment run twice and the same long derivation
  written out twice.
- **Appendix against chapter 17 (diffusion).** Sections 5 and 6 of chapter 43
  are both the proof store for chapter 17 (which cites the chapter 51 times
  across 36 labels) and a second, smaller course on guidance and sampling.
- **Appendix against Basics.** The first part of single-variable calculus and
  of geometry re-teach chapter 1 at the same level; uniform convergence part
  one re-teaches the chapter 3 generalization section with the same proof and
  near-identical numbers; Gaussian likelihood and Gaussian prior repeat
  chapter 2; the perplexity subsections repeat the language-model section.
- **Probability inside calculus.** The second half of integral calculus
  (densities, Monte Carlo, gradient estimators) uses variance and
  independence two chapters before they are defined, and chapter 41 proves
  the same $\sigma^2/n$ result.
- **Inside chapters.** The condition number is defined three times in
  chapter 38; the ridge filter factors, the two-line ill-conditioned system
  and the 6,908-step count recur across chapters 38 and 40 with different
  symbols; the learning-rate sweep of chapter 40 is rerun in the ODE section;
  the Cauchy averaging experiment appears twice in chapter 41 with one
  sentence repeated word for word.

### 3. A third of the sections are two or three sections

Reviewers proposed a split, with the seam named, for: eigendecompositions
(core, then Gershgorin, PageRank, pseudospectra, Lyapunov exponents, random
matrices, unranked), multivariable calculus (critical points are cut in two
by 3,000 words of chain rule and backpropagation), integral calculus,
gradient descent (Newton, BFGS, trust regions), numerical stability
(floating point, then conditioning), random variables (seam at "Several
Variables"), maximum likelihood (EM and the ELBO already carry their own
label and are cited from seven files), divergences and distances (three:
f-divergences, test functions and transport, scores), ODEs and solvers
(three). Two pairs that look like one section told twice are correctly split
and need only their boundary repaired: constrained optimization and duality,
concentration and uniform convergence. Mutual information and its estimation
need the boundary drawn once.

Order problems inside sections: linear maps is ordered by proof logic (nine
forward references; one fact promised, half-proved and settled 740 lines
later); the Gaussian subsection of distributions uses its normalizer and
variance 90 lines before deriving them; joint and conditional entropy are
used in section 2 of chapter 42 and defined in section 4 (open since
October); convexity delays its payoff by 4,400 words.

### 4. Verification cells and experiments that show less than claimed

Following the author's decision above. Three kinds, with different remedies.

**Cells that confirm a formula just derived and cannot fail** (cut, or keep
one per chapter): linear maps (9 of 13 cells re-run hand arithmetic), matrix
calculus (4 of 5), distributions (5 of 14), SDEs (5 of 13), convexity (4 of
7 confirm proved theorems by sampling), single-variable and multivariable
calculus (autograd confirming a hand derivative), two cells in divergences.

**Checks that would pass a broken method** (redesign; these are worse than
redundant):

- `43-dynamics/mdl-score-matching-diffusion-flow.md` SMD-6: Langevin's "half
  the mass in each mode" holds by symmetry of start and target. With 0.8/0.2
  weights the same sampler gives 0.48 against 0.2, which is the mode-weight
  failure chapter 17 leads with. The same weakness applies to the
  reverse-sampler check in the Fokker-Planck section.
- `39-calculus/mdl-matrix-calculus.md` MC-16: the random-point test uses
  non-square shapes, so the silent transpose error the section warns about
  cannot occur.

**Experiments read for more than they show** (reword now, redesign later):

- `40-optimization/mdl-adaptive-stochastic-methods.md` ADA-1, ADA-12, ADA-13:
  Adam's "48 steps" against tuned gradient descent (6,160) and heavy ball
  (296) runs Adam at an unexplained $\eta = 0.1$; its count is 1 step at
  $\eta = 1$ and 4,145 from a farther start. ADA-22: the warmup subsection
  blames the unestimated preconditioner, its own experiment shows otherwise,
  and chapter 9 cites it for the refuted claim.
- `40-optimization/mdl-constrained-optimization-duality.md` CON-8: the "974
  steps" of projected gradient is finite termination caused by the step size
  being the exact Newton step along the constraint, compared with a
  guaranteed 7,660.
- `41-probability-statistics/mdl-generalization-bounds.md` GEN-17, GEN-19:
  "ten times better than the best classical fit" compares with a classical
  regime barely better than predicting zero.
- `43-dynamics/mdl-neural-odes.md` NODE-14: the Hutchinson experiment needs
  $10^5$ probes for a trace of 0.17 and never says why one probe per step
  suffices in training.
- `43-dynamics` SMD-1, FMT-1: nothing is trained in the score-matching
  section (17 features at one noise level, exact scores in every sampler),
  yet the next section says "The previous section trained a network to
  predict the noise."

Cells worth keeping as models are those that show where a bound is loose,
where a hypothesis fails, or how fast an asymptotic statement sets in; the
first half of numerical stability and the coding section do this throughout.

### 5. Nothing is ranked

Proposition and Proof blocks are used for one-line identities and for the
main theorems alike: 14 in mutual information, 10 in divergences, a
one-line duplicated-evidence identity in naive Bayes. Eigendecompositions
gives 4,300 words to five side topics at the weight of the spectral theorem.
The reader cannot tell what to remember. The fix is editorial: name the two
or three results per section that matter, demote the rest to a sentence or an
exercise, and say which parts can be skipped.

### 6. Templates

The same ones as in Part 1 before its writing pass, more uniform here:

- **Every section ends twice.** A "We can now ..." recap paragraph, then a
  Summary that repeats it. All 36 sections.
- **Question, then "The next cell ...".** Nearly every cell (11 of 11 in the
  two sections of one reviewer, 9 of 13 in SDEs). Fine when the answer is
  open; several questions have foregone answers.
- **Chapter routes.** Each introduction's route paragraph is five or six
  consecutive "asks how/which/why" sentences. Draft rewrites are in the
  `index.md` review files.
- "Thus" opening the gloss after an equation (14 times in maximum likelihood,
  11 in the entropy section); "Every model in this book ..." opening four
  files of chapter 41.
- Things named by position ("the opening constant", "the smallest case"),
  which fails a reader who arrives from a reference
  (`39-calculus/mdl-integral-calculus.md` IC-1).

The Part 1 remedy applies: guide sections 10.2, 7.7, 12.

### 7. Connection to the main text

The part is used as an appendix by lookup, and the links are weak in both
directions.

- **Stale or wrong pointers.** `chapter_multilayer-perceptrons/backprop.md`
  line 214 points to the autodiff section for layer rules that live in
  `subsec_mdl-layer-vjp`. `mdl-matrix-calculus-autodiff.md` cites
  `sec_mdl-odes-solvers` for the adjoint method, now in
  `sec_mdl-adjoint-method`. `chapter_diffusion-models/gaussian-paths.md` line
  600 cites `sec_mdl-ddim-guidance` where `subsec_mdl-fm-two-point` is meant.
  `chapter_optimization/lr-scheduler.md` line 544 cites the warmup subsection
  for a claim its experiment refutes.
- **Main text that should point here and does not.** `resnet.md` (the Euler
  reading), `ssm.md` (the driven linear ODE), `muon.md` (Newton-Schulz), no
  main-text section links to naive Bayes, and the GAN chapter reaches data
  processing only through `sec_mdl-tv-pinsker`.
- **Appendix sections with no labelled use.** The bodies of matrix calculus
  and of divergences name applications without a `:numref:`; kernel Stein
  discrepancy, Sinkhorn and α-divergences have no inbound reference at all.
- **Missing for the main text.** The driven linear ODE and its exact
  discretization, which the State Space Models chapter needs
  (`43-dynamics` ODE-S3). The change of variables for densities, promised in
  the linear-maps opening and never delivered. The penalty-constraint
  equivalence is proved for $\ell_2$ only although the main text cites it for
  lasso. Joint and conditional entropy early in chapter 42.
- **Dead-end framing.** Bayesian computation says three times that none of
  its methods scales to networks and never says that the book later uses the
  ELBO, importance weights and Langevin proposals. Mutual information leaves
  its three deep-learning points to an exercise, a late aside, or nowhere.
- **Project notes.** `CLAUDE.md` still calls chapter 17 a placeholder; it is
  ten files and about 83,000 words. Two section reviewers were briefed with
  the stale statement; `43-dynamics.md` corrects for it.

### 8. Notation

Symbols that carry two or three meanings within a section or chapter: bold
$\mathbf{A}$ for the generic matrix and both running examples (chapter 38);
$T$ for sufficient statistic and temperature; $B$, $k$, $c$ in statistics;
"generator" for the convex function $f$ and for the network (the GAN chapter
resolved it the other way); $X$ as target and $Y$ as input in random
variables, against the rest of the book; $\gamma$, $\mathbf{x}$ against
$\lambda$, $\mathbf{w}$ for the ridge filter; the clipping point $y$ against
$x_0$ across the two duality sections; bf16 against bfloat16.

### 9. Further-reading files and exercises

The six `further-reading.md` files open with one identical template
paragraph (including the sentence "Each topic states the problem it
addresses and what it changes relative to the chapter."), clash in name with
the "Resources and Further Reading" lists in the chapter introductions, rank
nothing, and in chapter 43 duplicate chapter 17. They contain several of the
loose statements above.

The exercises are a strength of the part. Three recurring defects: main-line
content lives only in exercises (naive Bayes: 2,400 words of exercises
against a 3,950-word body, holding the text event models and temperature
scaling); some exercises cite cell IDs that may not be visible in the
rendered book (statistics, divergences); a few are broken (numerical
stability exercise 2 part 3 never overflows; coding exercise 8(a) cites the
wrong equation; an `binned_mi` exercise in a notebook that does not define
it).

## What is best and should set the model

- Coding and compression (chapter 42): the closest to MacKay in the part.
- Numerical stability, first half: a prediction before nearly every cell and
  a number that confirms or corrects it.
- Fokker-Planck and probability flow: honest about what is proved, heuristic,
  and cited, on the hardest mathematics of the part.
- Duality's opening; gradient descent parts one to three; the
  scale-invariance and rotated-valley pages of adaptive methods.
- The dart example that carries random variables; Chebyshev through
  Hoeffding in concentration; the Rademacher part of uniform convergence.
- Eckart-Young, the condition number, transient growth, and the measured
  image cosines in chapter 38.
- Dual numbers, implicit differentiation, saddles, the Gaussian integral,
  score against pathwise gradients in chapter 39.
- The predict-then-run pattern wherever the answer is not foregone.

## Recommended order of work

1. **Decisions that bind more than one chapter** (author): one home for
   entropy, KL, Gibbs and maximum entropy (41 or 42); one home for
   backpropagation and the adjoint in chapter 39; which sections to split.
   The chapter 9 list in `40-optimization.md` is settled in principle by
   decision 2 and needs only the per-item sorting confirmed; whether the
   same rule governs chapter 17 against chapter 43 is for the author to
   confirm.
2. **Correctness pass** (prose and inline math only): the items of finding 1,
   the stale pointers of finding 7. Small, and independent of the decisions.
3. **Writing pass** as for Part 1: voice, repetition by the agreed homes,
   closings, routes, ranking of results, skippable parts marked.
4. **Code and experiments** on the development machine: cut or redesign the
   cells of finding 4, re-capture the atypical outputs, print or drop the
   unprinted numbers, add the missing material of finding 7.

Steps 2 and 3 change no code and can be done on a laptop. Splitting sections
moves code between files and belongs with step 4.
