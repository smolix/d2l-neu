# Exercise Review — Group 03: Convolutional Networks

Chapters: `chapter_convolutional-neural-networks`, `chapter_convolutional-modern`
Repo: `/Users/smola/Repositories/github/d2l-neu`
Files found via `grep -rln "^## Exercises"`: 14

---

## chapter_convolutional-neural-networks/why-conv.md

```
file: chapter_convolutional-neural-networks/why-conv.md
heading_line: 316
n_exercises: 6
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Lin.Chen.Yan.2013`, ex 1)
crossrefs: 0
subproblems: nested-list(2) — ex 2 has 3 sub-items (lines 323-325), correct 4-space indent
discussions: single-link (L332, https://d2l.discourse.group/t/64)
defects: none found
clarity:
  - ex 2c (L325): "Can you treat audio using the same tools as computer vision? Hint: use the spectrogram." — "Can you...?" filler-question phrasing (house-style tone violation).
  - ex 4 (L327-328): "Do you think that convolutional layers might also be applicable for text data?" — opinion-phrased ("Do you think") rather than a direct task; second sentence salvages it with a concrete ask.
  - ex 5 (L329): "What happens with convolutions when an object is at the boundary of an image?" — reading/discussion prompt, no artifact or comparison specified.
notable: clean file overall; ex 1 and ex 6 are solid prove/derive tasks; nested sub-list (ex 2) properly indented.
```

## chapter_convolutional-neural-networks/conv-layer.md

```
file: chapter_convolutional-neural-networks/conv-layer.md
heading_line: 663
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: nested-list(1,2) — ex 1: 3 subs (L666-668); ex 2: 4 subs (L670-675); both 4-space indent, correct.
discussions: tabbed(4 tabs: mxnet L680, pytorch L684, tensorflow L688, jax L692)
defects: none found
clarity: none — all four items (incl. "what happens if you transpose X/K", gradient-error question, matrix-multiplication question) have clear, checkable outcomes.
notable: well-formed, no issues.
```

## chapter_convolutional-neural-networks/padding-and-strides.md

```
file: chapter_convolutional-neural-networks/padding-and-strides.md
heading_line: 454
n_exercises: 7
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:numref:`sec_transposed_conv` ex 6; :eqref:`eq_receptive_field` ex 7)
subproblems: none (all 7 items are single-paragraph, no nested lists)
discussions: tabbed(4 tabs: mxnet L466, pytorch L470, tensorflow L474, jax L478)
defects: none found
clarity:
  - ex 2 (L458): "For audio signals, what does a stride of 2 correspond to?" — references audio, but the word "audio" appears nowhere else in the file (verified by grep); the section is entirely about image convolutions, so this invokes context never established.
notable: ex 7 (L463) packs three distinct questions (which pixels are covered, when gridding is a problem, how to choose a dilation schedule) into one dense prose sentence — not lettered/broken out, unusually long/complex vs. the rest of the file's items, but each question is answerable and well-defined so not a hard defect.
```

## chapter_convolutional-neural-networks/channels.md

```
file: chapter_convolutional-neural-networks/channels.md
heading_line: 405
n_exercises: 8
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:numref:`sec_depthwise_separable` ex 7; :numref:`sec_vgg` ex 8)
subproblems: nested-list(1,2,7,8) — ex1: 3 subs, ex2: 4 subs, ex7: 2 subs, ex8: 2 subs; all 4-space indent, correct.
discussions: tabbed(4 tabs: mxnet L434, pytorch L438, tensorflow L442, jax L446)
defects: none found
clarity: none — every item (incl. ex 4's reference to "the final example of this section," which does exist) has a concrete deliverable.
notable: ex 7 and ex 8 are strong, well-scaffolded quantitative exercises (grouped/depthwise-separable cost comparisons).
```

## chapter_convolutional-neural-networks/pooling.md

```
file: chapter_convolutional-neural-networks/pooling.md
heading_line: 404
n_exercises: 8
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: nested-list(3,8) — ex3: 3 subs (L409-411), ex8: 2 subs (L417-418); 4-space indent, correct.
discussions: tabbed(4 tabs: mxnet L421, pytorch L425, tensorflow L429, jax L433)
defects: none found
clarity:
  - ex 5 (L413): "Why do you expect max-pooling and average pooling to work differently?" — open discussion prompt, no comparison metric or expected-answer shape given.
notable: ex 8 (aliasing / blur-pool on explicit numeric signals) is a strong, concrete exercise; good contrast with ex 5.
```

## chapter_convolutional-neural-networks/lenet.md

```
file: chapter_convolutional-neural-networks/lenet.md
heading_line: 371
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: nested-list(1,2) — ex1: 2 subs, ex2: 5 subs; 4-space indent, correct.
discussions: tabbed(4 tabs: mxnet L388, pytorch L392, tensorflow L396, jax L400)
defects: none found
clarity:
  - ex 5 (L385): "What happens to the activations when you feed significantly different images into the network (e.g., cats, cars, or even random noise)?" — no comparison criterion or expected observation specified, matches rubric's "see what happens" pattern.
notable: ex 3 correctly uses "original MNIST" as a distinct dataset from the Fashion-MNIST used earlier in the section — legitimate reference, not a defect.
```

## chapter_convolutional-modern/alexnet.md

```
file: chapter_convolutional-modern/alexnet.md
heading_line: 410
n_exercises: 8
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: nested-list(1,5) — ex1: 3 subs (L413-415), ex5: 2 subs (L420-421); 4-space indent, correct.
discussions: tabbed(4 tabs: mxnet L427, pytorch L431, tensorflow L435, jax L439)
defects: none found
clarity:
  - ex 7 (L423): "...Can you improve things further by preprocessing to take advantage of the invariances inherent in the images?" — "Can you...?" filler-question tone violation.
  - ex 8 (L424): "Can you make AlexNet overfit? Which feature do you need to remove or change to break training?" — "Can you...?" filler tone; first half is a yes/no framing before the real (fine) second question.
notable: ex 6 (L422) is a good example — explicit metrics named (throughput images/s, accuracy, GPU memory).
```

## chapter_convolutional-modern/blocks.md

```
file: chapter_convolutional-modern/blocks.md
heading_line: 788
n_exercises: 11
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 3 (:cite:`Simonyan.Zisserman.2014` ex 3 and ex 4 [repeated]; :cite:`Hu.Shen.Sun.2018` ex 8) — style: :cite: only
crossrefs: 1 (:numref:`sec_depthwise_separable` ex 9)
subproblems: nested-list(1) — 3 subs (L791-793), 4-space indent, correct.
discussions: tabbed(4 tabs: mxnet L806, pytorch L810, tensorflow L814, jax L818)
defects: none found
clarity:
  - ex 4 (L796): "...Can you do so without reducing the accuracy of the network?" — "Can you...?" filler tone.
  - ex 10 (L802): "...Can you design a variant that works on Fashion-MNIST's native resolution of 28×28 pixels?" — "Can you...?" filler tone.
notable: this file covers VGG, NiN, and GoogLeNet together (11 exercises across 3 architectures) — unusually broad scope vs. the single-architecture exercise sets elsewhere in the group. ex 8 and ex 9 (squeeze-and-excitation add-on, depthwise replacement of Inception branches) are strong quantitative exercises.
```

## chapter_convolutional-modern/batch-norm.md

```
file: chapter_convolutional-modern/batch-norm.md
heading_line: 1033
n_exercises: 7
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: nested-list(2,7) — ex2: 2 subs (L1037-1038), ex7: 5 subs (L1045-1049); 4-space indent, correct.
discussions: tabbed(4 tabs: mxnet L1052, pytorch L1056, tensorflow L1060, jax L1064)
defects: none found
clarity:
  - ex 3 (L1039): "Do we need batch normalization in every layer? Experiment with it." — textbook "experiment and see" pattern with no metric, range, or comparison specified.
  - ex 5 (L1042): "Fix the parameters `beta` and `gamma`. Observe and analyze the results." — no specific values, no expected artifact, "analyze" undefined.
  - ex 6 (L1043): "Can you replace dropout by batch normalization? How does the behavior change?" — "Can you...?" filler tone; "how does the behavior change" also unspecified.
  - ex 7 sub-items a,b,c,e (L1045,1046,1047,1049): "Can you apply...", "Can you use a full-rank...", "Can you use other compact matrix variants...", "...that you can use?" — four "Can you...?" filler constructions inside a single exercise's sub-list; sub-items are brainstorm prompts with no defined deliverable ("think of other normalization transforms").
notable: worst tone/clarity offender in the group — 5 distinct "Can you...?" instances across ex 6 and ex 7's sub-items, plus two underspecified "experiment/observe" items (ex 3, ex 5).
```

## chapter_convolutional-modern/resnet.md

```
file: chapter_convolutional-modern/resnet.md
heading_line: 1075
n_exercises: 7
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 4 (:cite:`He.Zhang.Ren.ea.2016` ex 2; :citet:`He.Zhang.Ren.ea.2016*1` ex 4 [verified as a distinct, real bib key in d2l.bib, not a typo]; :cite:`Huang.Liu.Van-Der-Maaten.ea.2017` ex 6; :citet:`pleiss2017memory` ex 7) — styles used: :cite: and :citet: (mixed)
crossrefs: 2 (:numref:`fig_inception` ex 1; :numref:`sec_convnext` ex 4)
subproblems: none (all 7 items single-paragraph)
discussions: tabbed(4 tabs: mxnet L1086, pytorch L1090, tensorflow L1094, jax L1098)
defects: none found
clarity: none — all items are well-formed prove/derive/compare tasks with concrete deliverables.
notable: heaviest and most precise citation use in the group (4 citations, 2 macro styles); no clarity problems despite technical density.
```

## chapter_convolutional-modern/convnext.md

```
file: chapter_convolutional-modern/convnext.md
heading_line: 694
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`liu2022convnet` ex 2)
crossrefs: 2 (:numref:`sec_training_recipes` ex 3 and ex 5)
subproblems: none
discussions: missing — no "Discussions" link or `:begin_tab:` block anywhere after L694 (verified via grep across the whole file; the only `:begin_tab:` blocks in the file are earlier, at L148, L378, L430, L437, unrelated to exercises); text goes directly from the last exercise (L700) to `<!-- slides -->` (L702).
defects:
  - L700-702: Discussions block entirely absent (see above) — anomaly per rubric.
clarity: none — all 5 exercises are exceptionally well-specified (explicit formulas, named ablation values, explicit comparisons against numref'd baselines).
notable: best-quality exercise set in the group by clarity standard (fully quantitative ablations with clear pass/fail criteria); the missing Discussions block is the one blemish.
```

## chapter_convolutional-modern/cnn-design.md

```
file: chapter_convolutional-modern/cnn-design.md
heading_line: 470
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`sec_convnext` ex 4)
subproblems: none
discussions: tabbed(4 tabs: mxnet L478, pytorch L482, tensorflow L486, jax L490)
defects: none found
clarity:
  - ex 1 (L472): "Increase the number of stages to four. Can you design a deeper RegNetX that performs better?" — "Can you...?" filler tone.
notable: short file (4 exercises), otherwise clean.
```

## chapter_convolutional-modern/efficient-convnets.md

```
file: chapter_convolutional-modern/efficient-convnets.md
heading_line: 815
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:eqref:`eq_bn_fold` ex 1; :eqref:`eq_depthwise_sep_ratio` ex 2)
subproblems: none
discussions: missing — no "Discussions" link anywhere in the file; only unrelated `:begin_tab:` blocks at L280, L370, L379 precede the exercises section. Text goes directly from the last exercise (L821) to `<!-- slides -->` (L823).
defects:
  - L821-823: Discussions block entirely absent.
clarity: none — all 5 exercises specify exact verification steps (`allclose` check, measured batch size, explicit α=0.5 test point, etc.).
notable: second file in this chapter (after convnext.md) with an excellent, fully-quantitative exercise set and a missing Discussions block — this looks like a pattern in the newer/rewritten sections of chapter_convolutional-modern rather than an isolated omission.
```

## chapter_convolutional-modern/training-recipes.md

```
file: chapter_convolutional-modern/training-recipes.md
heading_line: 792
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`wightman2021resnet` ex 3)
crossrefs: 2 (:numref:`tab_recipe_results` ex 2; :eqref:`eq_mixup` ex 5)
subproblems: none
discussions: missing — no "Discussions" link anywhere in the file; `:begin_tab:` blocks at L444, L448, L687, L695 are all unrelated (earlier in the chapter body). Text goes directly from the last exercise (L798) to `<!-- slides -->` (L800).
defects:
  - L798-800: Discussions block entirely absent.
clarity: none — every exercise specifies exact procedure and dataset (`FashionMNIST10k`, named ablation ingredients (i)-(iv), decay sweep values 0.9/0.99/0.999).
notable: third consecutive missing-Discussions file in this chapter; otherwise the strongest, most quantitatively precise exercise set in the group alongside convnext.md and efficient-convnets.md.
```

---

# Group-level summary

**Scope:** 14 files, 90 exercises total (chapter_convolutional-neural-networks: 38 across 6 files; chapter_convolutional-modern: 52 across 8 files).

**Dominant style:** 100% bare-list legacy style — every file uses repeated `1.` auto-numbering, and **zero files** in this group have named or tagged exercises (`names: none`, `tags: none`, `tag_vocab: n/a` everywhere). This contrasts with the newer named+tagged style cited in the rubric (chapter_reinforcement-learning). No difficulty markers anywhere.

**Formatting quality:** Uniformly clean. All nested sub-lists use correct 4-space indentation (why-conv, conv-layer, channels, pooling, lenet, alexnet, blocks, batch-norm all have proper nested subproblems). No broken markup, stray characters, or malformed math found in any file.

**Worst formatting defect — missing Discussions blocks:** 3 of 8 files in chapter_convolutional-modern have **no Discussions link at all**: `convnext.md` (L700-702), `efficient-convnets.md` (L821-823), `training-recipes.md` (L798-800). All other 11 files have `tabbed(4)` Discussions (mxnet/pytorch/tensorflow/jax). This looks like a systematic gap in newer/rewritten sections, not a one-off.

**Tags/citations:** citations: 10 total, using `:cite:` and `:citet:` (resnet.md mixes both; verified `He.Zhang.Ren.ea.2016*1` is a real, distinct d2l.bib key, not a typo). crossrefs (`:numref:`/`:eqref:`): 14 total. Heaviest citation use: resnet.md (4), blocks.md (3).

**Worst clarity offender:** `batch-norm.md` — 5 "Can you...?" filler-tone instances (ex 6 and four of ex 7's sub-items) plus two underspecified "experiment and observe" items (ex 3, ex 5) with no metric or expected artifact. Other "Can you...?" tone hits: alexnet.md (ex 7, ex 8), blocks.md (ex 4, ex 10), cnn-design.md (ex 1). padding-and-strides.md ex 2 references "audio signals," a context never established anywhere in that file.

**Best exercises:** convnext.md, efficient-convnets.md, and training-recipes.md (the same three files missing Discussions) have the group's most rigorous, fully-quantitative exercises — explicit formulas, named datasets/hyperparameter sweeps, and stated verification steps.
