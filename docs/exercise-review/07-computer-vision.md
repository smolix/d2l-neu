# Exercise Review: chapter_computer-vision

14 files, all in `/Users/smola/Repositories/github/d2l-neu/chapter_computer-vision`.
All counts and line numbers below were independently verified with grep/awk against
the source `.md` files (top-level exercise counts and numbering style were
cross-checked for all 14 files; the Discussions-tab-omission defects and the
fine-tuning.md/ssd.md indentation and nesting claims were spot-verified directly).

## Per-file profiles

### chapter_computer-vision/anchor.md
```
file: chapter_computer-vision/anchor.md
heading_line: 1397
n_exercises: 5
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Bodla.Singh.Chellappa.ea.2017` in ex 4, :cite: style)
crossrefs: 2 (:numref:`subsec_labeling-anchor-boxes`, :numref:`subsec_predicting-bounding-boxes-nms`, both in ex 3)
subproblems: none
discussions: tabbed(4 tabs: mxnet L1406→t/370, pytorch L1410→t/1603, jax L1414→t/1603, tensorflow L1418→t/1603)
defects: none found in the exercise list or Discussions block
clarity:
  - ex 5 (L1403): "Rather than being hand-crafted, can non-maximum suppression be learned?" is a bare yes/no rhetorical question with no requested deliverable (no "propose/design/implement" ask, no success criterion) — reads as a think-about-it prompt rather than a task.
notable: Exercises section itself is tiny (5 bare items, L1399-1403) but is followed by a large Quarto slide deck (L1421-1623, ~200 lines) that dwarfs the exercise content. mxnet Discussions thread (370) differs from the shared pytorch/jax/tensorflow thread (1603) — normal d2l pattern, not a defect.
```

### chapter_computer-vision/bounding-box.md
```
file: chapter_computer-vision/bounding-box.md
heading_line: 174
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(3 tabs: mxnet L180→t/369, pytorch L184→t/1527, jax L188→t/1527) — tensorflow tab absent (VERIFIED)
defects:
  - L188-190 (after jax `:end_tab:`, before `<!-- slides -->`): Discussions block is missing a `:begin_tab:`tensorflow`` entry, even though this file uses tensorflow code cells in its body (`#@tab tensorflow` at L53) and other files in this chapter (anchor, fcn, fine-tuning, image-augmentation, kaggle-cifar10, kaggle-dog, multiscale-object-detection, neural-style, object-detection-dataset) include a tensorflow Discussions tab.
clarity: none found — both exercises are concrete and well-scoped
notable: Shortest exercises section in the chapter (2 bare items, L176-177). Double blank line at L178-179 between the list and the Discussions block (cosmetic only). Followed by a slide deck (L194-266, ~73 lines) longer than the exercises section itself.
```

### chapter_computer-vision/fcn.md
```
file: chapter_computer-vision/fcn.md
heading_line: 874
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:cite:`Long.Shelhamer.Darrell.2015` in ex 4, :cite: style)
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet L882→t/377, pytorch L886→t/1582, jax L890→t/1582, tensorflow L894→t/1582)
defects: none found in the exercise list or Discussions block
clarity:
  - ex 2 (L877): "Can you further improve the accuracy of the model by tuning the hyperparameters?" — tone violation ("Can you...?" filler question per house style) and underspecified (no metric target, no range/list of hyperparameters, no comparison basis).
notable: Followed by a slide deck (L899-1022, ~124 lines) substantially longer than the exercises section.
```

### chapter_computer-vision/fine-tuning.md
```
file: chapter_computer-vision/fine-tuning.md
heading_line: 793
n_exercises: 4
numbering: sequential (literal 1., 2., 3., 4. — VERIFIED; one of only two files in the chapter using explicit sequential numbers rather than repeated "1.")
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: other(ex 3 and ex 4 each embed four framework-tabbed code blocks — mxnet/pytorch/jax/tensorflow — directly inside the list item rather than as lettered sub-items)
discussions: tabbed(4 tabs: mxnet L860→t/368, pytorch L864→t/1439, jax L868→t/1439, tensorflow L872→t/1439)
defects:
  - L799, L805, L811, L822 (ex 3's four framework code fences) and L830, L837, L844, L852 (ex 4's four framework code fences): the ``` code blocks are flush-left (0-space indent) rather than indented 4 spaces under the parent list item (item 3 at L797, item 4 at L828). VERIFIED directly — this breaks list nesting in this pipeline; the code is not attached to its exercise item.
clarity:
  - ex 4 (L828): "How can we leverage this weight parameter?" is open-ended with no requested deliverable or success criterion.
notable: Only file in the chapter whose exercises embed executable per-framework code cells (ids `#fine-tuning-exercises-1/2`). The same code-cell ids are pulled into the trailing slide deck via `@fine-tuning-exercises-1/2` (L982, L986), so the (mis-indented) exercise code doubles as slide content.
```

### chapter_computer-vision/image-augmentation.md
```
file: chapter_computer-vision/image-augmentation.md
heading_line: 1042
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
discussions: tabbed(4 tabs: mxnet L1049→t/367, pytorch L1053→t/1404, jax L1057→t/1404, tensorflow L1061→t/1404)
defects:
  - L1045: trailing whitespace after "Does it improve test accuracy?".
clarity:
  - ex 3 (L1046): "Refer to the online documentation of the deep learning framework. What other image augmentation methods does it also provide?" is a lookup/reading prompt with no concrete deliverable specified.
notable: Followed by a slide deck (L1066-1210, ~145 lines) much longer than the exercises section.
```

### chapter_computer-vision/kaggle-cifar10.md
```
file: chapter_computer-vision/kaggle-cifar10.md
heading_line: 1060
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet, pytorch, jax, tensorflow)
defects:
  - L1062: double space ("`lr_decay = 0.1`.  See what accuracy") — two spaces between sentences mid-item.
clarity:
  - ex 1: ends with "Can you further improve them?" — a "Can you...?" filler question with no target metric, method, or comparison baseline; the substantive part (run with given hyperparameters, report accuracy/ranking) is clear, but this closing question adds nothing actionable.
notable: Very short exercises section (2 items) for a long (1213-line) file; most of the file is an 11-slide instructor deck following the Discussions block.
```

### chapter_computer-vision/kaggle-dog.md
```
file: chapter_computer-vision/kaggle-dog.md
heading_line: 971
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet, pytorch, jax, tensorflow)
defects: none
clarity:
  - ex 1: "what results can you achieve when you increase `batch_size` ... and `num_epochs`" gives no range or target for the increase (unlike the sibling kaggle-cifar10 exercise, which pins exact hyperparameter values) — underspecified experiment with no stated success criterion.
  - ex 2: three questions stacked in one item ("Do you get better results if you use a deeper pretrained model? How do you tune hyperparameters? Can you further improve the results?"); "deeper pretrained model" is unnamed/unscoped, "tune hyperparameters" has no target parameter or range, and the closing question is a tone-violating filler with no metric or threshold.
notable: Section is short (2 items) relative to the 1095-line file; remainder is an 11-slide deck after Discussions, same pattern as kaggle-cifar10.md.
```

### chapter_computer-vision/multiscale-object-detection.md
```
file: chapter_computer-vision/multiscale-object-detection.md
heading_line: 270
n_exercises: 3
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 2 (:numref:`sec_alexnet` in ex 1, :numref:`subsec_multiscale-anchor-boxes` in ex 2)
subproblems: none
discussions: tabbed(4 tabs: mxnet, pytorch, jax, tensorflow)
defects:
  - L275-276: two consecutive blank lines between the last exercise item and the `:begin_tab:` Discussions block, vs. a single blank line in most other files. Does not change rendering but is inconsistent.
clarity: none — all 3 exercises are standard "why or why not" / "implement and report shape" / "derive the transform" tasks with clear deliverables.
notable: Shortest file in the chapter (361 lines) and among the shortest exercises sections (3 short items); ends with a 6-slide deck after Discussions.
```

### chapter_computer-vision/neural-style.md
```
file: chapter_computer-vision/neural-style.md
heading_line: 904
n_exercises: 4
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 1 (:citet:`10.1145/3544903.3544906` in ex 4, DOI-style bibkey)
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet, pytorch, jax, tensorflow)
defects: none
clarity:
  - ex 1: "How does the output change when you select different content and style layers?" — open-ended experiment with no specific layers to try, no comparison metric, no stated success criterion.
  - ex 3: ends with "Can you create more interesting synthesized images?" — "Can you...?" filler question plus the subjective, unscoped adjective "interesting" with no way to judge success.
  - ex 4: "Can we apply style transfer for text?" is essentially a reading/reflection prompt pointing to an external survey paper rather than a task with a concrete deliverable; the section never discusses text data, models, or a text analog of content/style losses, so it also invokes a domain the section never introduced.
notable: Exercises otherwise are the most substantive of the chapter (ex 2 has a clear comparative question re: content vs. noise tradeoff). Long file (1066 lines) with an 11-slide deck after Discussions.
```

### chapter_computer-vision/object-detection-dataset.md
```
file: chapter_computer-vision/object-detection-dataset.md
heading_line: 403
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet, pytorch, jax, tensorflow)
defects: none
clarity: none — ex 1 asks to visualize other images and compare boxes/objects (concrete artifact); ex 2 asks for a specific conceptual answer, guided by a concrete hint.
notable: Shortest file in the chapter overall (497 lines); ends with a 5-slide deck after Discussions.
```

### chapter_computer-vision/rcnn.md
```
file: chapter_computer-vision/rcnn.md
heading_line: 392
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 2 (:cite: x1 [L394], :citet: x1 [L395])
crossrefs: 0
subproblems: none
discussions: tabbed(4 tabs: mxnet, pytorch, jax, tensorflow) — pytorch/jax/tensorflow share link t/1409, mxnet distinct t/374 (standard pattern)
defects: none found
clarity:
  - ex 1: "Can we frame object detection as a single regression problem...?" opens as a rhetorical yes/no question rather than a stated task (house-style "Can we...?" filler); deliverable is implicit (a reasoned comparison to YOLO) but not spelled out as "argue/compare" — mild tone issue, not blocking since a reference (YOLO cite) anchors it.
notable: Exercises section is extremely short (2 items, 3 lines of content) — followed immediately by a long instructor slide deck (~92 lines, L413-505) covering the R-CNN/Fast R-CNN/Faster R-CNN/Mask R-CNN family, unrelated in content to the two exercises (which are about YOLO-style single-stage regression and SSD comparison).
```

### chapter_computer-vision/semantic-segmentation-and-dataset.md
```
file: chapter_computer-vision/semantic-segmentation-and-dataset.md
heading_line: 774
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`sec_image_augmentation` in ex 2, L777)
subproblems: none
discussions: tabbed(4 tabs: mxnet, pytorch, jax, tensorflow) — pytorch/jax/tensorflow share link t/1480, mxnet distinct t/375
defects:
  - L778-779: two consecutive blank lines between the last exercise and the Discussions block, inconsistent with the single-blank-line convention used elsewhere.
clarity:
  - ex 1 (L776): "How can semantic segmentation be applied in autonomous vehicles and medical image diagnostics? Can you think of other applications?" — pure reading/reflection prompt with no artifact to produce and no success criteria; also a "Can you...?" filler-question tone violation.
notable: Exercises section is very short (2 items); followed by a long slide deck (L798-903) walking through the VOC2012 dataset-loading pipeline, unrelated in content to the two exercises.
```

### chapter_computer-vision/ssd.md
```
file: chapter_computer-vision/ssd.md
heading_line: 1486
n_exercises: 2
numbering: sequential (literal "1." then "2." — VERIFIED; the other of two files in the chapter using sequential numbers rather than repeated "1.")
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 2, both :cite: style (L1578 :cite:`Lin.Goyal.Girshick.ea.2017`; L1638 :cite:`Liu.Anguelov.Erhan.ea.2016`)
crossrefs: 0
subproblems: nested-list(ex 2) — 4 sub-items at L1635-1638, each at correct 4-space indent (VERIFIED — renders as a proper nested list, not a defect)
discussions: tabbed(4 tabs: mxnet, pytorch, jax, tensorflow) at L1642-1656, all present and correctly ordered
defects:
  - L1639-1641: three consecutive blank lines between the end of exercise 2's sub-list and the Discussions block, versus the single-blank-line convention used elsewhere.
clarity:
  - ex 1 (L1488): opens with "Can you improve the single-shot multibox detection by improving the loss function?" — "Can you...?" filler-question framing (tone violation), though the body that follows is well-specified (explicit smooth-L1 formula, per-framework code cells at L1500-1574, focal-loss formula and code cells at L1589-1632), so the actual deliverable is concrete despite the framing.
  - ex 2 (L1634): also opens "Can you further improve the model in the following aspects" (same filler-question issue). Sub-items 1 and 2 (L1635-1636: "resize the input image bigger", "downsample negative anchor boxes") give no magnitude/ratio or success metric — underspecified compared to sub-items 3-4, which are concrete (assign loss weights; use the cited paper's evaluation methods).
notable: Exercise 1's four per-framework code cells are tagged with reusable ids `#ssd-exercises-1/2` and later invoked via `@ssd-exercises-1/2` inside the "Detect bananas" slide (L1843, L1847) — exercise code is repurposed as slide content. The instructor-slide portion after Discussions (L1660-1861, ~200 lines) is substantially longer than the exercises section itself.
```

### chapter_computer-vision/transposed-conv.md
```
file: chapter_computer-vision/transposed-conv.md
heading_line: 437
n_exercises: 2
numbering: repeated-1
names: none
name_style: n/a
tags: none
tag_vocab: n/a
difficulty_markers: none
citations: 0
crossrefs: 1 (:numref:`subsec-connection-to-mat-transposition` in ex 1, L439)
subproblems: none
discussions: tabbed(3 tabs: mxnet, pytorch, jax) — missing the tensorflow tab (VERIFIED)
defects:
  - L442-452: Discussions block has only mxnet/pytorch/jax tabs, no tensorflow tab, unlike most other files in the chapter (4 tabs) and unlike this same file's body, which does contain `#@tab tensorflow` code blocks (confirmed at L48, L116, L347) — so tensorflow is a supported framework for this chapter and its omission from Discussions looks like an unintentional gap.
clarity:
  - ex 2 (L440): "Is it efficient to use matrix multiplications to implement convolutions? Why?" relies on the matrix-multiplication-as-convolution framing established only in ex 1's :numref: reference and a later slide; ex 2 itself carries no explicit reference. Minor — understandable in context, not a blocking ambiguity.
notable: Like rcnn.md and semantic-segmentation-and-dataset.md, exercises are very short (2 items) versus a long instructor slide deck (L456-556, ~100 lines) covering transposed-conv mechanics and matrix-transpose duality — none of it exercise-specific.
```

## Group-level summary

- **Dominant style**: 100% bare-list (legacy d2l), 0/14 files use names or bracketed
  tags. 12/14 use repeated-`1.` auto-numbering; 2/14 (fine-tuning.md, ssd.md) use
  literal sequential numbering (1,2,3,4 / 1,2). No difficulty markers anywhere.
- **Total exercises**: 39 across 14 files (2-5 per file; median 2). Sections are
  uniformly short (3-5 lines of content typical) relative to the large instructor
  slide decks that follow every file.
- **Names/tags**: 0/14 files have named or tagged exercises (contrast with
  chapter_reinforcement-learning's `[tag] *Name.*` style) — tag_vocab is n/a
  chapter-wide.
- **Tag vocabulary**: none exists in this chapter.
- **Worst formatting defects**:
  - fine-tuning.md L799/805/811/822 and L830/837/844/852 — 8 framework code
    fences flush-left instead of 4-space-indented under their parent list item
    (ex 3, ex 4), breaking list nesting.
  - bounding-box.md and transposed-conv.md both omit the tensorflow tab from
    their Discussions block even though tensorflow code cells exist in the
    file body — likely an unintentional gap, not deliberate framework scoping.
  - Recurring minor issue: inconsistent blank-line count (1 vs. 2 vs. 3) before
    the Discussions block across multiscale-object-detection.md,
    semantic-segmentation-and-dataset.md, and ssd.md — cosmetic, doesn't break
    rendering, but inconsistent.
- **Worst clarity offenders**: a chapter-wide tic of "Can you/we ... ?" closing
  filler questions with no metric/target (fcn.md ex2, kaggle-cifar10.md ex1,
  kaggle-dog.md ex2, neural-style.md ex3, semantic-segmentation-and-dataset.md
  ex1, rcnn.md ex1, ssd.md ex1&2) — a clear, repeated house-style tone
  violation. Pure reading/reflection prompts with no deliverable: anchor.md ex5,
  neural-style.md ex4 (references an external text-transfer survey the section
  never covers), semantic-segmentation-and-dataset.md ex1,
  image-augmentation.md ex3. Underspecified open-ended experiments (no range/
  target given): kaggle-dog.md ex1&2, neural-style.md ex1, ssd.md ex2
  sub-items 1-2.
