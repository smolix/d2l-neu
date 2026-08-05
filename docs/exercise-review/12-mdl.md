# Exercise Review — MDL group (chapter_mdl-linear-algebra, chapter_mdl-calculus,
chapter_mdl-optimization, chapter_mdl-probability-statistics,
chapter_mdl-information-theory, chapter_mdl-dynamics)

Repo: `/Users/smola/Repositories/github/d2l-neu`. All line numbers verified with
`grep -n` / `Read`. 26 files total (3+4+5+7+3+4).

---

## chapter_mdl-linear-algebra (3 files)

```
file: chapter_mdl-linear-algebra/mdl-geometry-linear-algebraic-ops.md
heading_line: 1697
n_exercises: 14
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:eqref:`eq_mdl-pythagoras` ex10; :eqref:`eq_mdl-det-3x3` ex14; both resolve in-file)
subproblems: nested-list(ex4)
discussions: tabbed(4 tabs) — mxnet /410, pytorch /1084, tensorflow /1085, jax /1085
defects:
  - L1715-1717: exercise 4's three sub-items use an unordered `*` marker indented by a single
    leading space. This pipeline requires 4-space indentation for a sub-item to nest under its
    parent list item; at 1 space these bullets break out of item 4's nesting instead of
    rendering as a sub-list.
clarity: none found — all 14 items are direct compute/prove/verify/show tasks with concrete
  deliverables.
notable: Discussions block is followed by ~515 lines of slide content (>15x the 31-line
  exercises section). Ex1/ex3 embed zero-indented `$$...$$` display-math blocks directly inside
  a list item (consistent with sibling files, treated as existing pipeline convention).

file: chapter_mdl-linear-algebra/mdl-svd-low-rank.md
heading_line: 987
n_exercises: 8
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 3 (:eqref:`eq_mdl-svd-gram` ex2, ex5; :numref:`subsec_mdl-eckart-young` ex8; all resolve)
subproblems: nested-list(ex ~ordered sub-items, correctly 4-space indented, no defect)
discussions: tabbed(4 tabs) — all four point to the identical URL /t/svd (see defect)
defects:
  - L1003: within exercise 3, one continuation line begins at column 1 (0 leading spaces) while
    every other continuation line of the same item is indented 3 spaces — inconsistent
    indentation within one list item's paragraph.
  - L1022: exercise 7 references its supporting cell via raw backtick anchor "the
    `#svd-weight-spectrum` cell" instead of the :eqref:/:numref: convention used elsewhere in
    this same section (ex2, ex5, ex8).
  - L1044,1048,1052,1056: all four framework Discussions tabs point to the identical
    non-numeric URL "https://d2l.discourse.group/t/svd," unlike sibling files in this chapter,
    which use distinct numbered thread IDs per framework.
clarity:
  - ex7: "find the LoRA rank achieving 95% spectral energy... (the `#svd-weight-spectrum`
    cell)." The referenced cell already computes and prints exactly this rank/saving, so it is
    ambiguous whether the reader should read off that output or construct a different weight
    matrix. The follow-up "How does the answer change if the spectrum decays more slowly?" gives
    no parameter to vary and no metric to report.
notable: Exercises section short (8 items) followed by ~357 lines of slide content. No tone
  violations found.

file: chapter_mdl-linear-algebra/mdl-eigendecomposition.md
heading_line: 1390
n_exercises: 11
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 4 (:eqref:`eq_mdl-nonnormal-example` ex1; :eqref:`eq_mdl-normal-power-norm` ex2;
  :eqref:`eq_mdl-power-iter` ex10; :numref:`subsec_mdl-jordan` ex11; all resolve in-file)
subproblems: none
discussions: tabbed(4 tabs) — mxnet /411, pytorch /1086, tensorflow /1087, jax /1087
defects:
  - L1405: the fourth "1." item opens with two spaces after the marker, inconsistent with the
    single space used by every other item in this list.
  - L1456: a doubled blank line appears between the pytorch and tensorflow Discussions tabs,
    inconsistent with the single blank line separating every other tab pair in this block and
    in sibling files.
clarity: none found — ex4's "what is strange about this example compared to the previous one"
  is casual in register but has a concrete, answerable target.
notable: Uses "repeated-1" (bare-list, legacy) numbering for all 11 items, unlike the sequential
  numbering used by both sibling files in this chapter directory — a numbering-style
  inconsistency within the same chapter.
```

---

## chapter_mdl-calculus (4 files)

```
file: chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md
heading_line: 1200
n_exercises: 9
numbering: sequential
names: some(3/9)
name_style: bold-period (ex7 "**Hessian--vector product.**", ex8 "**Gradient of the
  log-absolute-determinant.**", ex9 "**Attention Jacobians.**")
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=4 (eq_mdl-dual-eval ex4, eq_mdl-softmax-jacobian ex6 & ex9, eq_mdl-hvp ex7);
  numref=3 (sec_mdl-random_variables, sec_mdl-flow-matching, sec_attention-scoring-functions, all ex8/ex9)
subproblems: none
discussions: tabbed(4 tabs) — mxnet, pytorch, tensorflow, jax, all bare
  "https://d2l.discourse.group/" (generic, non-thread-specific link, all four identical)
defects:
  - L1271-1285: all four Discussions tabs link to the same generic, non-chapter-specific URL
    ("https://d2l.discourse.group/") rather than a distinct per-framework discussion thread.
clarity: none found. ex8 and ex9 are long/dense but each closes with a concrete numerical
  verification step.
notable: ex9 ("Attention Jacobians") is unusually long/dense relative to the others, layering a
  proof, a second differentiation step, and a numerical check in one item.

file: chapter_mdl-calculus/mdl-single-variable-calculus.md
heading_line: 855
n_exercises: 10
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=2 (eq_mdl-descent, ex5 & ex7); numref=0
subproblems: none
discussions: tabbed(4 tabs) — mxnet /412, pytorch /1088, tensorflow /1089, jax /1089
defects: none found
clarity:
  - ex8: "Relate the one-dimensional update ... to the vector update ... from the introduction."
    references "the introduction" informally (no :numref:) rather than a resolved cross-ref;
    borderline, likely valid content but not verifiable without reading the whole chapter intro.
notable: none

file: chapter_mdl-calculus/mdl-multivariable-calculus.md
heading_line: 936
n_exercises: 8
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=2 (eq_mdl-nabla_use ex3; eq_mdl-lagrange-condition ex8); numref=0
subproblems: none
discussions: tabbed(4 tabs) — mxnet /413, pytorch /1090, tensorflow /1091, jax /1091
defects:
  - L936: no blank line between the "## Exercises" heading and the first list item (item 1
    starts on the very next line), unlike every sibling file in this chapter, which inserts a
    blank line after the heading.
clarity: none found
notable: none

file: chapter_mdl-calculus/mdl-integral-calculus.md
heading_line: 973
n_exercises: 10
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=2 (eq_mdl-score-function-gradient ex2; eq_mdl-parts ex10); numref=2
  (sec_mdl-distributions, twice, both ex10)
subproblems: none
discussions: tabbed(4 tabs) — mxnet /414, pytorch /1092, tensorflow /1093, jax /1093
defects: none found
clarity: none found — ex8's Fubini "paradox" is resolved with a concrete hint; ex10 is
  well-scaffolded (integration by parts -> Gamma function -> factorial).
notable: none
```

---

## chapter_mdl-optimization (5 files)

```
file: chapter_mdl-optimization/mdl-constrained-optimization-duality.md
heading_line: 1228
n_exercises: 8
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: numref=1 (subsec_mdl-jensen, ex2); eqref=2 (eq_mdl-opt-pgd ex4; eq_mdl-opt-waterfilling ex7)
subproblems: none
discussions: other — a prose "## Discussions" section (not framework-tabbed) followed by a
  single bare link, no :begin_tab: tabs at all
defects:
  - L1281-1298: Discussions is a prose paragraph plus one non-tabbed link
    (https://d2l.discourse.group/t/constrained-optimization-duality), not the customary 4-tab
    :begin_tab: pattern used throughout the linear-algebra/calculus/probability-statistics
    chapters — no per-framework discussion links at all.
clarity: none found — ex6/ex7 reference named code cells (`#constrained-svm-dual`,
  `#constrained-water-filling`) which is the chapter's own convention, not a broken reference.
notable: This is the first of five optimization-chapter files that replace the tabbed
  Discussions block with prose + a single link — a chapter-wide pattern (see group summary).

file: chapter_mdl-optimization/mdl-convexity.md
heading_line: 1308
n_exercises: 9
numbering: sequential
names: some(2/9)
name_style: mixed — ex8 "**Implicit bias, verified.**" bold-period, ex9 "**Coordinate descent as
  Gauss--Seidel.**" bold-period
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: numref=2 (sec_mdl-gradient-based-optimization ex6, ex8); eqref=2 (eq_mdl-opt-pl ex7;
  eq_mdl-quadratic-coordinate ex9)
subproblems: inline-letters(ex2: (a)(b)(c)(d); ex8: (a)(b)(c))
discussions: other — prose "## Discussions" + single link, no tabs
defects:
  - L1314-1318: exercise 2 crams four sub-questions as inline "(a) ... (b) ... (c) ... (d) ..."
    in one paragraph rather than a nested list.
  - L1358-1362: exercise 8 crams three sub-tasks as inline "(a) ... (b) ... (c) ..." in one
    paragraph.
  - L1363: a stray blank line appears between exercise 8 and exercise 9 (the only gap in an
    otherwise tight list), converting the list to loose rendering at that point for no apparent
    reason.
clarity: none found — ex5 (quasiconvexity) and ex8/ex9 are all well-scaffolded with clear
  deliverables.
notable: ex8 cross-references "Exercise 9 of :numref:`sec_mdl-gradient-based-optimization`" —
  verified: that file's exercise 9 is indeed the paper-and-pencil implicit-bias derivation this
  exercise's numerical companion completes. Cross-reference is accurate.

file: chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md
heading_line: 992
n_exercises: 8
numbering: sequential
names: all(8/8)
name_style: bold-period (e.g. "**AdaGrad from the metric view.**", "**The Reddi example by
  hand.**", "**SVRG unbiasedness and cost.**")
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: numref=2 (sec_mdl-gradient-based-optimization, ex1 & ex6); eqref=5
  (eq_mdl-opt-adagrad, eq_mdl-opt-bias-correction, eq_mdl-opt-reddi x2, eq_mdl-opt-svrg)
subproblems: inline-letters(ex4: (a)(b))
discussions: other — prose "## Discussions" (two paragraphs) + single link, no tabs
defects:
  - L1032-1033: exercise 4 crams two sub-cases as inline "(a) Adam ... and (b) AdamW ..." in
    one sentence rather than a nested list.
  - L1072: a stray blank line appears immediately before exercise 8 (the last item), the only
    gap in an otherwise tight list — same pattern as convexity.md and
    gradient-based-optimization.md in this chapter.
clarity: none found — every exercise closes with an explicit, checkable deliverable (a
  numeric sweep, a specific quantity to derive, or a named cell to modify).
notable: First file where every exercise carries a bold descriptive name — a much higher rate
  than any other chapter in this group. Exercise 1 cites "Exercise 2 of
  :numref:`sec_mdl-gradient-based-optimization`" — verified correct (that file's ex2 is exactly
  the steepest-descent-under-a-metric derivation referenced here).

file: chapter_mdl-optimization/mdl-gradient-based-optimization.md
heading_line: 1239
n_exercises: 11
numbering: sequential
names: some(3/11)
name_style: mixed — ex9 "*Implicit bias.*" italic-period, ex10 "**BFGS and the secant
  equation.**" bold-period, ex11 "**Trust-region acceptance.**" bold-period
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=11 (descent-lemma, per-mode, heavy-ball, hb-rate, variance, noise-ball x2,
  newton-step, bfgs, trust-ratio); numref=2 (sec_mdl-convexity, ex9 x2)
subproblems: none
discussions: other — prose "## Discussions" + single link, no tabs
defects:
  - L1310: a stray blank line appears immediately before exercise 10, separating the unnamed
    exercises 1-9 from the two bold-named exercises 10-11 — same recurring pattern as
    convexity.md and adaptive-stochastic-methods.md.
  - name_style inconsistency: ex9 uses an italic-period name while ex10/ex11 use bold-period
    names, within the same list.
clarity: none found — all items have explicit deliverables; ex7's open discussion prompt
  ("Discuss when large batches are worth it") is scoped with an explicit hint and comparison
  target, not underspecified.
notable: ex9 cross-references its "numerical companion" as "Exercise 8 of
  :numref:`sec_mdl-convexity`" — verified correct (matches convexity.md's ex8, "Implicit bias,
  verified.", which is exactly that numerical companion).

file: chapter_mdl-optimization/mdl-numerical-stability-conditioning.md
heading_line: 986
n_exercises: 8
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Sterbenz.1974`, ex5, inside the exercises; Discussions prose separately
  cites :citet:`Higham.2002`, not counted here)
crossrefs: eqref=5 (rounding-model, ce-from-logits, welford, condition-bound, ridge-kappa);
  numref=3 (sec_mdl-convexity, sec_mdl-gradient-based-optimization, sec_mdl-constrained-
  optimization-duality)
subproblems: none
discussions: other — prose "## Discussions" (two paragraphs, cites Higham.2002) + single link,
  no tabs
defects: none found
clarity: none found — all 8 items are precise, numerically checkable tasks.
notable: Fifth and last optimization file to use the prose-Discussions convention — all five
  files in this chapter share it uniformly (see group summary).
```

---

## chapter_mdl-probability-statistics (7 files)

```
file: chapter_mdl-probability-statistics/mdl-naive-bayes.md
heading_line: 364
n_exercises: 6
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`Koller.Friedman.2009`, ex3)
crossrefs: eqref=1 (eq_mdl-naive_bayes_log, ex4); numref=1 (sec_softmax, ex4)
subproblems: none
discussions: tabbed(4 tabs) — mxnet /418, pytorch /1100, tensorflow /1101, jax /1101
defects:
  - L364: no blank line between "## Exercises" heading and item 1.
clarity: none found — ex6 even states the expected numeric answer as a self-check.
notable: none

file: chapter_mdl-probability-statistics/mdl-bayesian-computation.md
heading_line: 704
n_exercises: 5
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: missing — no "Discussions" text anywhere in the file (confirmed via
  file-wide grep); the exercises list is followed directly by a blank line and then
  "<!-- slides -->".
defects:
  - L723: Discussions block entirely absent — no heading, no tabs, no link of any kind, unlike
    every other file in this group.
clarity: none found — all 5 items (prior sensitivity, importance-sampling proposal,
  Metropolis tuning, variational family, multimodality) have concrete comparison targets.
notable: Only file in the entire MDL group with zero cross-references (:eqref:/:numref:) and
  zero citations inside its exercises, and the only file with a fully missing Discussions
  section.

file: chapter_mdl-probability-statistics/mdl-concentration-generalization.md
heading_line: 1049
n_exercises: 8
numbering: sequential
names: none
name_style: n/a (ex3 bolds the theorem name "**McDiarmid's inequality**" inline, mid-sentence —
  not the "Name." exercise-title convention, so not counted as a named exercise)
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 2 (:cite:`McDiarmid.1989` ex3; :cite:`Dwork.Feldman.Hardt.ea.2015` ex8)
crossrefs: eqref=4 (eq_mdl-hoeffding x2, eq_mdl-norm-concentration, eq_mdl-finite-class);
  numref=3 (sec_mdl-statistics x2, sec_mdl-numerical-stability-conditioning)
subproblems: inline-numbers(ex3: "(i) ... (ii) ...")
discussions: single-link — no heading, no framework tabs, just a bare
  "[Discussions](.../t/concentration-and-generalization)" link directly after the list
defects:
  - L1067: exercise 3 crams two sub-tasks as inline "(i) Recover Hoeffding's inequality from
    it. (ii) The bootstrap of ... " rather than a nested list.
clarity: none found — all exercises specify concrete numeric targets or comparisons.
notable: A third distinct Discussions convention within this one chapter (bare link, no
  heading, no tabs) — different from both the naive-bayes/random-variables tabbed style and
  the optimization chapter's prose style.

file: chapter_mdl-probability-statistics/mdl-random-variables.md
heading_line: 1070
n_exercises: 7
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=5 (eq_mdl-chebyshev; eq_mdl-var_affine, eq_mdl-var_comp, eq_mdl-exp_linear all
  on one line in ex5; eq_mdl-var_sum in ex6)
subproblems: none
discussions: tabbed(4 tabs) — mxnet /415, pytorch /1094, tensorflow /1095, jax /1095
defects:
  - L1070: no blank line between "## Exercises" heading and item 1.
clarity: none found
notable: none

file: chapter_mdl-probability-statistics/mdl-maximum-likelihood.md
heading_line: 1168
n_exercises: 10
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: numref=2 (subsec_mdl-gaussian-mse ex2; sec_softmax ex4); eqref=4
  (eq_mdl-softmax-jacobian, eq_mdl-cramer-rao, eq_mdl-fisher, eq_mdl-gmm-mstep)
subproblems: none
discussions: tabbed(4 tabs) — mxnet /416, pytorch /1096, tensorflow /1097, jax /1097
defects:
  - L1168: no blank line between "## Exercises" heading and item 1.
clarity: none found
notable: none

file: chapter_mdl-probability-statistics/mdl-distributions.md
heading_line: 1194
n_exercises: 10
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=7 (softmax x2, multinomial_pmf, binomial_pmf, exp_family, beta_posterior,
  mvn_conditional); numref=1 (sec_mdl-maximum_likelihood, ex9)
subproblems: none
discussions: tabbed(4 tabs) — mxnet /417, pytorch /1098, tensorflow /1099, jax /1099
defects: none found
clarity: none found
notable: none

file: chapter_mdl-probability-statistics/mdl-statistics.md
heading_line: 620
n_exercises: 9
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=3 (eq_mdl-bias-variance x2, eq_mdl-gauss_confidence)
subproblems: none
discussions: tabbed(4 tabs) — mxnet /419, pytorch /1102, tensorflow /1103, jax /1103
defects: none found
clarity: none found — ex1's classic bootstrap-failure example is unusually well explained.
notable: Every exercise is written as a single unwrapped long line (no ~80-column hard
  wrapping with 3-space continuation indents), unlike every sibling file in this chapter,
  which hard-wraps prose across multiple indented lines. Purely a source-formatting
  difference; does not affect rendering.
```

---

## chapter_mdl-information-theory (3 files)

```
file: chapter_mdl-information-theory/mdl-information-theory.md
heading_line: 1335
n_exercises: 9
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Gneiting.Raftery.2007`, ex6)
crossrefs: eqref=1 (eq_mdl-gaussian_kl, ex4)
subproblems: nested-list(ex3: three unordered `*` sub-items, 4-space indented — correct, no
  defect)
discussions: tabbed(4 tabs) — mxnet /420, pytorch /1104, tensorflow /1105, jax /1105
defects: none found
clarity: none found — the monkey/typesetter/language-model progression in ex3 is a clean
  worked comparison.
notable: Uses bare-list "repeated-1" numbering for all 9 items.

file: chapter_mdl-information-theory/mdl-mutual-information.md
heading_line: 1461
n_exercises: 10
numbering: repeated-1
names: none
name_style: n/a
tags: some(2/10)
tag_vocab: ["(Data-processing)", "(Synergy)"] — parenthetical topic tags, not the
  square-bracket "[short-code]" convention described in the rubric's anchor examples
difficulty_markers: none
citations: 0
crossrefs: eqref=6 (gaussian_mi, mut_ent_def, ba_bound, infonce_def, infonce_est, fano)
subproblems: none
discussions: tabbed but only 1 of 4 frameworks present — pytorch only, generic bare link
  "https://d2l.discourse.group/" (no thread number)
defects:
  - L1507-1509: Discussions block has only a single `:begin_tab:`pytorch`` tab; mxnet,
    tensorflow, and jax tabs are entirely absent, unlike every other file in this chapter and
    most of the group. The one link present is also the generic discourse-group root URL, not
    a chapter-specific thread.
clarity: none found — ex8's "What does the optimal critic look like?" is a bounded follow-up
  to an already-derived bound, not open-ended.
notable: Parenthetical tags "(Data-processing)" and "(Synergy)" mark exercises 5 and 6 — a tag
  style variant distinct from both the rubric's bracket-tag example and this group's
  bold/italic-name convention.

file: chapter_mdl-information-theory/mdl-divergences-distances.md
heading_line: 1360
n_exercises: 9
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=4 (f-div-def, f-gan-bound, w1-cdf, stein-identity); numref=1
  (sec_mdl-geometry-linear-algebraic-ops, ex6)
subproblems: none
discussions: tabbed(4 tabs) — mxnet, pytorch, tensorflow, jax all link to the identical URL
  "https://d2l.discourse.group/t/divergences"
defects:
  - L1431-1445: all four framework tabs point to the same non-per-framework URL
    (.../t/divergences repeated four times), same anomaly as mdl-svd-low-rank.md in the
    linear-algebra chapter.
clarity: none found — ex6's two-sided Hellinger/TV sandwich is unusually well hinted and
  closed with a numerical verification.
notable: none
```

---

## chapter_mdl-dynamics (4 files)

```
file: chapter_mdl-dynamics/mdl-fokker-planck-probability-flow.md
heading_line: 1117
n_exercises: 8
numbering: sequential
names: all(8/8)
name_style: bold-period (e.g. "**Stationary distribution from scratch.**", "**The factor of
  two.**", "**Break the sampler.**")
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=7 (heat-equation, continuity, ou-marginal, reverse-family, mixture-score,
  tanh-score x2); numref=1 (sec_mdl-continuous-normalizing-flows, ex3)
subproblems: inline-letters(ex8: (a)(b))
discussions: tabbed but only 1 of 4 frameworks — pytorch only, generic bare link
defects:
  - L1160-1164: exercise 8 crams two sub-experiments as inline "(a) ... ; (b) ..." rather than
    a nested list.
  - L1167-1169: only a pytorch Discussions tab is present; mxnet/tensorflow/jax are missing,
    and the link is the generic discourse-group root, not a chapter-specific thread.
clarity: none found — every named exercise closes with an explicit computation or experiment.
notable: Every exercise in this file is bold-named — matches the pattern seen in the
  optimization chapter's later files.

file: chapter_mdl-dynamics/mdl-sdes.md
heading_line: 1044
n_exercises: 8
numbering: sequential
names: all(8/8)
name_style: bold-period (e.g. "**The square-root scaling is forced.**", "**Itô's lemma
  practice.**", "**Variance preservation, exactly.**")
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=6 (ito-lemma, em, vp-sde, ou-solution, ou-kernel, vp-marginal); numref=1
  (sec_mdl-ito-lemma, ex4)
subproblems: inline-letters(ex6: (a)(b)(c)(d))
discussions: tabbed but only 1 of 4 frameworks — pytorch only, generic bare link
defects:
  - L1077-1085: exercise 6 crams four sub-tasks as inline "(a) ... (b) ... (c) ... (d) ..." —
    the most extensive inline-lettering instance found in this group.
  - L1103-1105: only a pytorch Discussions tab is present; mxnet/tensorflow/jax missing;
    generic non-thread-specific link.
clarity: none found — every item ends with an explicit derivation, numeric check, or named
  cell to adapt.
notable: none beyond the discussions/lettering defects above.

file: chapter_mdl-dynamics/mdl-score-matching-diffusion-flow.md
heading_line: 1897
n_exercises: 8
numbering: sequential
names: some(3/8)
name_style: mixed — ex1-5 unnamed; ex6 "*(Langevin stationarity.)*", ex7 "*(CFG as a score
  tilt.)*", ex8 "*(Stochastic interpolants.)*" — a novel italic-parenthetical variant not
  matching either the rubric's "*Name.*" or "**Name.**" anchor examples
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 2 (:cite:`song2019generative` ex5; :citet:`Albergo.Boffi.VandenEijnden.2023` ex8)
crossrefs: eqref=6 (hyvarinen, regression-lemma, rf-path, ncsm-loss, guidance-bayes, cfg);
  numref=1 (sec_mdl-euler-runge-kutta, ex3)
subproblems: inline-letters(ex8: (a)(b), minor — two items in one clause)
discussions: tabbed but only 1 of 4 frameworks — pytorch only, generic bare link
defects:
  - L1950-1951: exercise 8 embeds "(a) rectified flow and (b) a variance-preserving diffusion
    path" inline rather than as a two-item list; minor instance of the inline-lettering pattern.
  - L1954-1956: only a pytorch Discussions tab is present; mxnet/tensorflow/jax missing;
    generic non-thread-specific link.
clarity: none found.
notable: The italic-parenthetical name style, "*(Topic.)*", appears only in this file within
  the entire group — a third distinct "named" convention alongside bold-period and
  italic-period.

file: chapter_mdl-dynamics/mdl-odes-solvers.md
heading_line: 1387
n_exercises: 8
numbering: sequential
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: eqref=7 (integral-form, matrix-exp-series, adjoint, adjoint-grad, instant-cov,
  matrix-exp-eig, hutchinson); numref=1 (sec_mdl-ode-existence-uniqueness, ex6)
subproblems: none
discussions: tabbed(4 tabs) — mxnet, pytorch, tensorflow, jax all present, but all four link
  to the same bare generic URL "https://d2l.discourse.group/" (no thread number)
defects:
  - L1449-1463: all four framework tabs are present (unlike the other three dynamics files)
    but all four point to the identical placeholder URL, not a chapter-specific thread —
    functionally as broken as a missing link.
clarity: none found — ex7's adjoint-vs-backprop comparison and ex8's trace/determinant
  contrast are both well-posed with explicit deliverables.
notable: The only dynamics-chapter file with all four framework tabs present structurally,
  yet the least informative link content (fully generic URL repeated four times).
```

---

## Group-level summary (see final message for the version returned to caller)

Group: 26 files, 229 exercises total (linear-algebra 33, calculus 37, optimization 44,
probability-statistics 55, information-theory 28, dynamics 32).

Dominant style per chapter: linear-algebra/calculus/most of probability-statistics use
sequential numbering, no names, no tags, tabbed(4) Discussions with real thread IDs (the
book's mainline convention). optimization breaks from that convention entirely: all 5 files
replace tabbed Discussions with a prose "## Discussions" section + one bare link, and its
later files (adaptive-stochastic-methods 8/8, gradient-based-optimization 3/11) are the most
heavily bold-named in the group. dynamics is the most heavily named chapter overall
(fokker-planck and sdes: 8/8 bold-named) but 3 of its 4 files keep only a single pytorch
Discussions tab with a placeholder link. information-theory mixes repeated-1 and sequential
numbering and contains the group's only tagged file (mutual-information, nonstandard
parenthetical tags, not square-bracket).

Names: 7/26 files use names (bold-period dominant; 1 italic-period; 1 novel
italic-parenthetical in score-matching-diffusion-flow). Tags: 1/26 files
(mutual-information.md, "(Data-processing)"/"(Synergy)", 2/10) — the rubric's bracket
"[short-code]" tag convention is entirely absent from all 26 files. Citations: 7 total
across 5 files (Koller.Friedman.2009, McDiarmid.1989, Dwork.Feldman.Hardt.ea.2015,
Sterbenz.1974, Gneiting.Raftery.2007, song2019generative, Albergo.Boffi.VandenEijnden.2023).

Worst formatting defects (by prevalence): (1) Discussions is the dominant defect class — five
incompatible conventions coexist: clean tabbed-with-real-threads (~14 files); tabbed but all 4
links identical/generic (matrix-calculus-autodiff, svd-low-rank, divergences-distances,
odes-solvers); single-framework-tab-only with placeholder link (mutual-information,
fokker-planck, sdes, score-matching-diffusion-flow — 4 files, all pytorch-only); prose+single-
link replacing tabs (all 5 optimization files); bare link/no heading/no tabs
(concentration-generalization); and fully missing (bayesian-computation, zero discussion
content of any kind). (2) Inline-lettered sub-items crammed into one paragraph instead of a
nested list: 7 instances (convexity x2, adaptive-stochastic-methods, fokker-planck, sdes,
score-matching-diffusion-flow, concentration-generalization). (3) Missing blank line after the
"## Exercises" heading: multivariable-calculus, naive-bayes, random-variables,
maximum-likelihood. (4) Stray mid-list blank line isolating trailing bold-named exercises,
recurring 3x within optimization (convexity, adaptive-stochastic-methods,
gradient-based-optimization). (5) Sub-list indentation breaks: geometry-linear-algebraic-ops
(1-space bullets), svd-low-rank (inconsistent continuation indent), eigendecomposition
(double-space marker; doubled blank line between Discussion tabs).

Worst clarity offenders: clarity is strong throughout — only two soft flags in 229 exercises:
svd-low-rank ex7 (ambiguous whether to read off an existing code cell's output or build a new
matrix; unscoped follow-up question) and single-variable-calculus ex8 (informal "the
introduction" reference instead of a resolved cross-ref). No tone-guide violations
(promotional adjectives, filler questions, theatrical phrasing) found anywhere in this group.
