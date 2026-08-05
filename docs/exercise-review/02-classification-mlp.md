# Exercise Review — Group 02: classification, MLP, builders-guide

Chapters reviewed: `chapter_linear-classification`, `chapter_multilayer-perceptrons`, `chapter_builders-guide` (d2l-neu).
22 files total. All `.md` source files, read in full from `## Exercises` heading to true EOF.

---

## chapter_builders-guide (8 files)

```
file: chapter_builders-guide/custom-layers.md
heading_line: 1081
n_exercises: 4 distinct exercises, each restated per framework across 4 tabs (pytorch L1083-1103, jax L1105-1128, tensorflow L1130-1150, mxnet L1152-1175) = 16 raw top-level "1." items
numbering: repeated-1 (each tab restarts "1." x4)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: missing (file ends at L1175 with mxnet's `:end_tab:`; no Discussions section at all)
defects:
  - L1175 (EOF): no Discussions block follows the final `:end_tab:` — every other d2l section of this kind normally carries a tabbed Discussions/forum-links block after Exercises.
clarity: none found — all 4 exercises specify a concrete artifact and verification step (e.g. "Verify that the state dict ... grows by the expected entry, and that a state dict saved without the bias no longer loads with strict=True").
notable: Unusually clean/well-specified exercise writing for a bare-list (no names/tags) file — every item pairs a build task with an explicit verification step or comparison question. Tab order (pytorch, jax, tensorflow, mxnet) is canonical.
```

```
file: chapter_builders-guide/gpus-devices-memory.md
heading_line: 1847
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: missing (file ends at L1864, immediately after exercise 4's last sentence, no trailing blank line and no Discussions block)
defects:
  - L1864 (EOF): no Discussions block; file terminates mid-list with the last exercise sentence.
clarity: none found — ex1-3 are concrete measure/predict/explain tasks; ex4 is conditional ("If you have two GPUs...") but gates itself explicitly (`num_gpus() >= 2`) and names the metric (scaling vs. linear).
notable: All four exercises reference concepts verified present earlier in the section ("four-plateau accounting", "checkpointing comparison", "capstone training run") — not dangling references.
```

```
file: chapter_builders-guide/init.md
heading_line: 1110
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 3 (:numref:`sec_repro` L1114, :numref:`sec_numerical_stability` L1121, :numref:`sec_read_write` L1126)
subproblems: none
discussions: missing (file ends at L1130 directly after exercise 4, no Discussions block)
defects:
  - L1130 (EOF): no Discussions block after the exercises list.
clarity: none found — all 4 exercises reference verifiable, section-grounded concepts ("default and scaled treatments," "geometric-growth prediction," "the truncation demo above").
notable: Heaviest crossref density of the eight builders-guide files (3 :numref: in 4 exercises), all resolving to real sections.
```

```
file: chapter_builders-guide/model-construction.md
heading_line: 1641
n_exercises: 4 distinct exercises, each restated per framework across 4 tabs (pytorch L1643-1660, mxnet L1662-1680, jax L1682-1698, tensorflow L1700-1717) = 16 raw top-level "1." items
numbering: repeated-1 (each tab restarts "1." x4)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 4 (:numref:`sec_resnet`, once per tab: L1659, L1679, L1697, L1716)
subproblems: none
discussions: missing (file ends at L1717 with tensorflow tab's `:end_tab:`, no Discussions block)
defects:
  - L1662: Exercises tab order is pytorch → mxnet → jax → tensorflow, breaking the canonical order used everywhere else in this same file (pytorch → jax → tensorflow → mxnet, confirmed at L68/74/83/91 and every other `:begin_tab:` group). Internal-consistency defect unique to this file among the eight.
  - L1717 (EOF): no Discussions block.
clarity: none found — PlainListMLP/ListMLP, ParallelBlock, MLPConfig/build, ResidualBlock are all defined earlier in the file; each exercise pairs a concrete task with a specific question.
notable: Same four-exercise template as custom-layers.md's tab structure, but with the tab-order anomaly noted above.
```

```
file: chapter_builders-guide/numerics.md
heading_line: 1324
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:numref:`sec_parameters` L1326, :numref:`sec_custom_layer` L1342)
subproblems: none
discussions: missing (file ends at L1344 directly after exercise 4, no Discussions block)
defects:
  - L1344 (EOF): no Discussions block.
clarity: none found — ex2 explicitly requires finding a crossover point and explaining it; ex3/ex4 have concrete, checkable deliverables. ex3 (L1334-1337) folds a multi-framework API list into one dense parenthetical (torch.finfo/jnp.finfo/ml_dtypes.finfo) rather than separate tabs — dense but not incorrect.
notable: Departs from custom-layers.md/model-construction.md's per-framework `:begin_tab:` pattern; instead folds framework-specific API names inline within single list items. This tabbed-vs-inline inconsistency for multi-framework content recurs across the chapter.
```

```
file: chapter_builders-guide/parameters-state-memory.md
heading_line: 1659
n_exercises: 4 distinct exercises — ex1-2 shared/untabbed (L1661-1669), ex3-4 restated per framework across 4 tabs (pytorch L1671-1678, jax L1680-1688, tensorflow L1690-1700, mxnet L1702-1711) = 2 shared + 8 tabbed = 10 raw top-level items
numbering: sequential (literal "1.", "2.", then "3.", "4." continuing inside each tab — confirmed each tab's items read "3." and "4." rather than restarting at "1.")
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 3 (:numref:`sec_read_write`, in jax/tensorflow/mxnet's exercise 3 only — L1682, L1692, L1704; pytorch's exercise 3 uses `copy.deepcopy` instead, a legitimate content difference not a defect)
subproblems: none
discussions: missing (file ends at L1711 with mxnet tab's `:end_tab:`, no Discussions block)
defects:
  - L1711 (EOF): no Discussions block.
clarity: none found — "net", "TinyLM", "BatchNorm running mean/variance" all verified defined earlier; ex1/ex2 hypotheticals have clear, checkable deliverables.
notable: Distinctive hybrid structure (2 shared + 2 tabbed) — a third distinct exercise-authoring pattern within this chapter, alongside fully-tabbed and fully-untabbed files. Numbering (sequential literal digits) also differs from the repeated-"1." convention used by 5 of the other 7 files in this chapter.
```

```
file: chapter_builders-guide/reproducibility-inspection.md
heading_line: 1134
n_exercises: 5 distinct exercises — ex1-4 shared/untabbed (L1136-1158), ex5 restated per framework across only 3 tabs (jax L1160-1168, tensorflow L1170-1179, mxnet L1181-1191) — no pytorch variant of ex5 = 4 shared + 3 tabbed = 7 raw top-level items
numbering: sequential (literal "1."-"4." shared, then "5." repeated identically in each of the 3 tabs)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: missing (file ends at L1192 with mxnet tab's `:end_tab:`, no Discussions block)
defects:
  - L1160-1191: Exercise 5's tabbed block covers jax, tensorflow, and mxnet only — no pytorch variant, unlike every other tabbed exercise group in this chapter (custom-layers.md, model-construction.md, parameters-state-memory.md), which cover all 4 frameworks. Asymmetry undocumented in the text.
  - L1192 (EOF): no Discussions block.
clarity:
  - ex 4 (L1154-1158): presented as a shared, framework-agnostic exercise ("Use a forward hook that... zero out the output of a single residual block's body"), but the section's own framework tabs (L603-652) state this capability does NOT hold uniformly: TensorFlow's tab explicitly says post-hoc modification "has no TensorFlow equivalent" (L629-630); MXNet's tab says Gluon's hook contract "is observe-only... unlike PyTorch's it cannot modify the output" (L651-652); JAX's `nnx.capture` only records return values, doesn't modify them (L617-619). MXNet's ex5 (L1187-1189) quietly patches this by telling the reader to "revisit exercise 4" with a workaround, but TensorFlow's ex5 never revisits ex4's inapplicability, and JAX's doesn't either. A JAX or TensorFlow reader attempting ex4 as literally written has no way to know it can't be done as stated.
notable: Most content-level (not just formatting) issue found in this chapter: exercise 4's premise silently fails for 2 of 4 frameworks per the section's own text, patched only partially and only for one framework. Combined with ex5's missing pytorch tab, the most structurally uneven exercise set in builders-guide.
```

```
file: chapter_builders-guide/saving-loading.md
heading_line: 1372
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`sec_numerics` L1384)
subproblems: none
discussions: missing (file ends at L1388 directly after exercise 4, no Discussions block)
defects:
  - L1388 (EOF): no Discussions block.
clarity: none found — safetensors format, `save_checkpoint`, "the regressor," bfloat16/float32 casting all verified defined earlier; each exercise specifies a concrete measurement or comparison.
notable: Ex1 (L1374-1379) folds three frameworks' atomic-write mechanisms into one dense parenthetical clause — grammatically run-on but not ambiguous. Same inline-multi-framework style as numerics.md rather than tabbed style.
```

---

## chapter_linear-classification (7 files)

```
file: chapter_linear-classification/classification.md
heading_line: 230
n_exercises: 7
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 3 (:eqref:`eq_weighted-empirical-risk-min` L237, :numref:`sec_environment-and-distribution-shift` L237, :numref:`sec_softmax_scratch` L238)
subproblems: other(inline roman-numeral parens "(i) ... (ii) ... (iii)" crammed into a single paragraph, not a nested list: ex4 L235, ex5 L236, ex6 L237, ex7 L238)
discussions: tabbed(4 tabs: mxnet L241/pytorch L245/tensorflow L249/jax L253)
defects:
  - L235: ex4's three sub-parts "(i) Compute ... (ii) Explain ... (iii) Construct ..." inline-lettered within one paragraph rather than a clean nested list.
  - L236: ex5's "(i)/(ii)/(iii)" same inline-cramming defect.
  - L237: ex6's "(i)/(ii)/(iii)" same inline-cramming defect.
  - L238: ex7's "(i)/(ii)/(iii)" same inline-cramming defect.
clarity: none found — every sub-part specifies a concrete deliverable (compute/derive/plot/construct) even where crammed inline.
notable: Unusually rigorous/applied content for a bare-numbered section — ROC curves, weighted risk, calibration, top-k accuracy. Two crossrefs point forward within the chapter (ex6→§4.7, ex7→§4.4); ex7 hedges correctly with "once you have trained the model of...". File continues ~300 lines past Discussions (L256-551) with unrelated slide-deck content — a pattern shared by all 7 files in this chapter.
```

```
file: chapter_linear-classification/environment-and-distribution-shift.md
heading_line: 674
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 4 (:eqref:`eq_covariate-shift-identity` L677, :eqref:`eq_weighted-empirical-risk-min` L677, :numref:`subsec_covariate-shift-correction` L678, :eqref:`eq_weighted-empirical-risk-min` L679)
subproblems: none
discussions: single-link (L684) — consistent with zero `begin_tab` blocks anywhere in this file (no framework-specific code in this section)
defects: none found
clarity: none found — ex3/ex4 (L678-679) chain cleanly ("the classifier from the previous exercise"); ex1's (L676) "loan/footwear example at the start of the section" is a real, verifiable example in the section.
notable: Two consecutive blank lines (L682-683) before the Discussions link rather than one — cosmetic, doesn't affect rendering. Slide-deck content resumes after Discussions through EOF (L1094).
```

```
file: chapter_linear-classification/generalization-classification.md
heading_line: 538
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`fig_mdl-clf-shattering` L564)
subproblems: none
discussions: single-link (L571)
defects: none found
clarity: none found — standard prove/derive VC-dimension tasks with concrete deliverables; ex6 (L564-569) preempts a real ambiguity ("some" vs. "every" set) with an explicit hint.
notable: Purely statistical-learning-theory content (Hoeffding-type bound, VC dimension of polynomials/rectangles/linear classifiers, shattering proof) — the most textbook-classic, least "customized" set in the chapter. Slide deck resumes after Discussions through EOF (L976).
```

```
file: chapter_linear-classification/image-classification-dataset.md
heading_line: 310
n_exercises: 3
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet L317/pytorch L321/tensorflow L325/jax L329)
defects: none found
clarity: none found — all three (L312-314) are concrete engineering tasks (throughput benchmark across batch sizes, num_workers ablation, tensor-layout inspection across all four framework implementations) with clear, checkable deliverables.
notable: Shortest exercises section in the chapter (3 items vs. 4-9 elsewhere) and the only one with zero math/citations/crossrefs — matches the section's pure data-loading focus. Slide deck resumes after Discussions through EOF (L531).
```

```
file: chapter_linear-classification/softmax-regression-concise.md
heading_line: 302
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`sec_softmax_scratch` L307)
subproblems: none
discussions: tabbed(4 tabs: mxnet L311/pytorch L315/tensorflow L319/jax L323)
defects: none found
clarity:
  - ex1 (L304-305): lists six numeric formats (FP64, FP32, BFLOAT16, FP16, TF32, INT8) as context, then asks to "compute the smallest and largest argument of the exponential function for which the result does not lead to numerical underflow or overflow" without stating for which format(s) this should be done — ambiguous whether once (implicit default) or separately for each of the six enumerated formats.
notable: Numerically-stable-softmax focus; ex3/ex4 (L307-308) unusually precise/well-scaffolded with an explicit success criterion ("verify ... agree to floating-point precision"). Slide deck resumes after Discussions through EOF (L628).
```

```
file: chapter_linear-classification/softmax-regression-scratch.md
heading_line: 641
n_exercises: 6
numbering: repeated-1 (top-level and nested)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`sec_softmax` L643)
subproblems: nested-list(ex1: L644-646; ex2: L648-651; ex5: L655-656) — verified exact 4-space indentation, renders as clean sub-lists
discussions: tabbed(4 tabs: mxnet L661/pytorch L665/tensorflow L669/jax L673)
defects: none found
clarity:
  - ex3 (L652): "Is it always a good idea to return the most likely label? ... How would you try to address this?" — open discussion/design question with no specified artifact or success criterion.
  - ex4 (L653): "Assume that we want to use softmax regression to predict the next word... What are some problems that might arise from a large vocabulary?" — open-ended identify-the-problems prompt, no indication of expected scope/depth.
notable: Best-formatted nested lists in the chapter (exact 4-space indent for every sub-item). ex6 (L657) legitimately references "the matrix computed above" — the confusion matrix is in fact computed earlier in this section (L463, L562-598) — not a dangling reference. Slide deck resumes after Discussions through EOF (L1021).
```

```
file: chapter_linear-classification/softmax-regression.md
heading_line: 466
n_exercises: 9
numbering: repeated-1 (top-level and nested)
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Bradley.Terry.1952` L481)
crossrefs: 3 (:eqref:`eq_softmax_to_sigmoid` L472, :numref:`sec_mdl-information_theory` L474, :numref:`subsec_softmax_operation` L502)
subproblems: nested-list(ex1 L469-470; ex2 L472-473; ex3 L475-476; ex5 L482-483; ex6 L485-491; ex7 L493-496; ex8 L498-501; ex9 L503-505) — verified 4-space indentation throughout; ex4 (L477, PAM-3) is the only top-level item with no sub-list
discussions: single-link (L507) — zero `begin_tab` blocks (framework-agnostic content)
defects: none found
clarity:
  - ex3 sub-item (L476, verified verbatim): "Can you design a better code?" — tone violation, a "Can you...?" filler question per docs/style-guide.md.
notable: Densest, most mathematically elaborate exercises section in the chapter — 9 top-level items, up to 6 nested sub-parts (ex6, RealSoftMax), the chapter's only citation, and an internal self-reference ("Combine the two parts of exercise 1...", L493) verified consistent with ex1's actual content. Exemplary nested-list formatting despite legacy repeated-"1." numbering. Slide deck resumes after Discussions through EOF (L830).
```

---

## chapter_multilayer-perceptrons (7 files)

```
file: chapter_multilayer-perceptrons/backprop.md
heading_line: 473
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none (verbatim "(*Hint:* ...)" appears twice inside ex6's sub-items — an informational hint, not a difficulty label)
citations: 0
crossrefs: 1 (:eqref:`eq_backprop-J-h` in ex6.b, L486)
subproblems: nested-list(2, 5, 6) — ex2 has 2 sub-items (L477-478), ex5 has 2 (L482-483), ex6 has 4 (L485-488); all correctly 4-space indented
discussions: single-link (L490)
defects: none found in the exercise text itself (markup/math/indentation all clean)
clarity:
  - ex5 sub-item a (L482): "Can you partition it over more than one GPU?" — "Can you...?" filler-question phrasing.
  - ex6 sub-item d (L488): "Show that reverse mode computes $(\alpha+\beta+\gamma)(\delta+\epsilon+\zeta)$..." — α,β,γ,δ,ε,ζ are never defined anywhere in the file; a reader cannot map these six Greek letters onto "a chain of three inputs feeding three outputs feeding one loss" described in the same sentence. Ambiguous/underspecified.
notable: Section (473-490) short, followed by ~365 lines (492-856) of an embedded slide deck that re-teaches the section and explicitly calls out "Capstone (exercise 6)" at L852. This slide-deck-after-Discussions structure recurs in all 7 files in this chapter.
```

```
file: chapter_multilayer-perceptrons/dropout.md
heading_line: 538
n_exercises: 8
numbering: repeated-1
names: some(1/8)
name_style: italic-period ("*Monte Carlo dropout.*", ex5, verified verbatim)
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 2 (:citet:`Gal.Ghahramani.2016` in ex5, L544; :cite:`Wan.Zeiler.Zhang.LeCun.Fung.2013` in ex7, L546)
crossrefs: 0
subproblems: none (all 8 items flat)
discussions: tabbed(4 tabs: mxnet/pytorch/tensorflow/jax), L549-563
defects: none found
clarity:
  - ex8 (L547): "...can you develop a method that matches or outperforms dropout on Fashion-MNIST?" — "Can you...?" filler-question phrasing used for the core task itself.
notable: Slide deck (565-876) directly references exercise numbers ("Exercise 5 keeps dropout on at test time," L871), confirming numbering. Otherwise the best-specified bare-list file among the non-tagged ones — most items name a metric/plot as the deliverable.
```

```
file: chapter_multilayer-perceptrons/generalization-deep.md
heading_line: 382
n_exercises: 7
numbering: repeated-1
names: some(2/7)
name_style: italic-period ("*Epoch-wise double descent.*" ex6, "*Grokking.*" ex7, verified verbatim)
tags: none
tag_vocab: n/a
difficulty_markers: "(*)" — verbatim, prefixed only to ex7 (L390); no other occurrence of this pattern found elsewhere in the group
citations: 1 (:citet:`Power.Burda.Edwards.ea.2022`, ex7, L390)
crossrefs: 2 (:numref:`sec_mlp-implementation` ex6 L389; :numref:`fig_grokking` ex7 L390)
subproblems: none
discussions: single-link (L392)
defects: none found
clarity: none flagged — ex1/ex3/ex4/ex5 are short conceptual questions with clearly bounded answers; ex6/ex7 are long research-flavored exercises with explicit procedures and named plots as deliverables (exempted per rubric).
notable: ex6 and ex7 are among the most rigorously specified exercises found across the whole group (exact dataset, exact procedure, exact plot, explicit follow-up questions). The lone "(*)" marker is an outlier — no other file in the group uses any difficulty marker.
```

```
file: chapter_multilayer-perceptrons/kaggle-house-price.md
heading_line: 670
n_exercises: 7
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet/pytorch/tensorflow/jax), L680-694
defects: none found
clarity:
  - ex2 (L673): "Hint: can you construct a situation where the values are not missing at random?" — "Can you...?" filler-question phrasing used as the hint itself.
notable: ex3/ex4/ex6/ex7 are "improve the score" tasks without a fixed numeric target, but the section's Kaggle leaderboard/log-RMSE metric supplies a natural success criterion, so not flagged as underspecified. ex7 is unusually rigorous — explicitly requires out-of-fold target encoding to avoid leakage.
```

```
file: chapter_multilayer-perceptrons/mlp-implementation.md
heading_line: 321
n_exercises: 9
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none (verbatim "(*Hint:* ...)" in ex8 L335 and ex9 L336 — informational, not a difficulty label)
citations: 0
crossrefs: 2 (:numref:`sec_numerical_stability` in ex2 L324 and ex9 L336)
subproblems: mixed(nested-list for ex5 [3 sub-items, L328-330] and ex7 [2 sub-items, L333-334], both correctly 4-space indented; inline-letters for ex9 [L336])
discussions: tabbed(4 tabs: mxnet/pytorch/tensorflow/jax), L338-352
defects:
  - L336: ex9 crams its three sub-conditions "(a) small Gaussian noise...; (b) the value used in this section...; (c) large Gaussian noise..." inline into a single paragraph instead of a nested list, inconsistent with the clean nested-list sub-items used for the structurally similar ex5 and ex7 in this same file.
clarity: none additional flagged — every item names a concrete deliverable (a plot, a best value, a measured speed, a written comparison).
notable: Internally inconsistent subproblem formatting within one file (nested lists for ex5/ex7 vs. inline lettering for ex9, all "compare N settings" tasks).
```

```
file: chapter_multilayer-perceptrons/mlp.md
heading_line: 696
n_exercises: 9
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Ioffe.Szegedy.2015` in ex8, L716)
crossrefs: 2 (:numref:`fig_mdl-mlp-xor` in ex2, L703; :numref:`sec_batch_norm` in ex8, L716)
subproblems: nested-list(7) — ex7 "Sigmoid and tanh are very similar." has 2 sub-items (L714-715), correctly 4-space indented
discussions: tabbed(4 tabs: mxnet/pytorch/tensorflow/jax), L719-733
defects: none found
clarity: none flagged — pReLU (defined L482-488) and Swish (defined L690-691) are both introduced earlier in the file before ex3/ex4 use them; both :numref: targets resolve to real labels.
notable: ex8 transparently forward-references batch normalization ("covered in :numref:`sec_batch_norm`"), a concept from a later chapter — flagged as a forward-reference by the text itself, not treated as a defect. Cleanest file of the seven: no formatting defects and no clarity issues found.
```

```
file: chapter_multilayer-perceptrons/numerical-stability-and-init.md
heading_line: 546
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`You.Gitman.Ginsburg.2017` in ex6, L553)
crossrefs: 0
subproblems: none (all 6 items flat)
discussions: tabbed(4 tabs: mxnet/pytorch/tensorflow/jax), L556-570
defects:
  - L553: double-space typo — "layerwise adaptive rate scaling  for inspiration" (two spaces between "scaling" and "for").
  - L554-555: two consecutive blank lines between the last exercise and the `:begin_tab:` block, vs. exactly one blank line at the equivalent spot in every other tabbed-discussion file in this chapter — harmless for rendering but inconsistent.
clarity:
  - ex1 (L548): "Can you design other cases where a neural network might exhibit symmetry that needs breaking...?" — "Can you...?" filler-question phrasing.
  - ex2 (L549): "Can we initialize all weight parameters in linear regression or in softmax regression to the same value?" — "Can we...?" filler-question phrasing.
  - ex6 (L553): "If we know that some terms diverge, can we fix this after the fact?" — "Can we...?" filler-question phrasing.
  - ex5 (L552): "Look up analytic bounds on the eigenvalues of the product of two matrices. What does this tell you about ensuring that gradients are well conditioned?" — no bound is named and no expected output format is given (proof? named theorem? paragraph?), noticeably less specified than ex3's fully worked derivation in the same list.
notable: ex3 (L550) is the densest, most rigorously specified single exercise found across the whole group (explicit target formulas, explicit "show that... equals..." checkpoints). Three of six exercises use "Can you/we...?" phrasing — the highest concentration of that tone pattern in the reviewed set.
```

---

## Group-level summary

**Dominant style per chapter.** `builders-guide` (8 files): uniform bare-list, no names/tags, and — the chapter's defining trait — **every single file (8/8) has no Discussions block at all**, section just ends at EOF; 5 files use per-framework `:begin_tab:` tab repetition for exercises, 2 fold multi-framework APIs inline, 1 is a shared+tabbed hybrid. `linear-classification` (7 files): uniform bare repeated-`1.` numbering, no names/tags anywhere; Discussions present in all 7 (4 tabbed, 3 single-link matching framework-agnostic content); the chapter's signature defect is inline roman-numeral subproblems `(i)...(ii)...(iii)` crammed into one paragraph (classification.md, 4 exercises). `multilayer-perceptrons` (7 files): also bare repeated-`1.` numbering with no tags, but 2/7 files break from pure-bare with named+italic exercises (dropout.md 1/8, generalization-deep.md 2/7) and one has the group's only difficulty marker, `(*)`.

**Totals.** 22 files, **~139 top-level raw exercise items** counted per-file (builders-guide alone contains many "1 concept × 4 tabs" repeats, e.g. custom-layers.md/model-construction.md = 16 raw items each for 4 real exercises). Files with names: 3/22 (dropout.md 1/8, generalization-deep.md 2/7); all others none. Files with tags: 0/22 — **tag_vocab is empty across the entire group**, the "named+tagged" new style (per chapter_reinforcement-learning) does not appear anywhere in these three chapters.

**Worst formatting defects.** (1) Systemic: all 8 builders-guide files end abruptly with no Discussions block — likely a dropped section during a rewrite, not incidental. (2) reproducibility-inspection.md ex5's tabbed block has no pytorch variant (jax/tensorflow/mxnet only), asymmetric with every other tabbed file. (3) model-construction.md's exercise tabs use a non-canonical framework order (pytorch→mxnet→jax→tensorflow) vs. the rest of that same file. (4) classification.md: 4 exercises with inline `(i)/(ii)/(iii)` subproblems instead of nested lists. (5) mlp-implementation.md ex9: inline-lettered sub-conditions inconsistent with nested-list ex5/ex7 in the same file.

**Worst clarity offenders.** reproducibility-inspection.md ex4's premise (framework-agnostic forward-hook output modification) is contradicted by the section's own text for TensorFlow (no equivalent) and MXNet (observe-only) — a genuine content bug, not just a style nit. backprop.md ex6.d uses six undefined Greek-letter variables with no mapping to the described scenario. Recurrent tone violation: "Can you/we...?" filler phrasing appears in softmax-regression.md, backprop.md, dropout.md, kaggle-house-price.md, and numerical-stability-and-init.md (3 instances in the last alone) — the most common house-style violation found. softmax-regression-scratch.md ex3/ex4 and numerical-stability-and-init.md ex5 lack stated success criteria/scope for open-ended asks.

Full 22 per-file profiles are in this file above this summary.
