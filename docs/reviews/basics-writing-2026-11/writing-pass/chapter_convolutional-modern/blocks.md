# Writing pass: `chapter_convolutional-modern/blocks.md` (7.2)

Second section of the chapter (after `alexnet`). Prose only; checker ends
"OK: only prose changed", `lint_source.py` clean.

**Prose words:** 3918 before, 3906 after (-12). The cuts (schedule
restatements, NiN lab-notebook paragraph, VGG-11 number run, stem inventory,
recap paragraph) were largely spent on the rewritten comparison paragraph,
the NiN motivation, and the bottleneck definition.

## Before/after pairs

1. Before: "Three networks of 2013 and 2014 answered these two problems with
   ideas that every later convolutional network uses."
   After: "Two networks of 2013 and 2014 each solved one of these problems,
   and a third combined their answers and added a question of its own. Nobody
   trains these networks any more, but their ideas are in every convolutional
   network since."
2. Before: "VGG places small kernels there."
   After: "Which kernels should it place there? VGG's answer is the smallest
   one that still sees a neighbor."
3. Before: "Three of them see $7 \times 7$ with $27c^2$ weights instead of
   $49c^2$, and they apply three nonlinearities instead of one."
   After: "The stack is not only cheaper. It applies three nonlinearities
   where the single large kernel applies one, so it can compute functions of
   its $7 \times 7$ window that one linear filter followed by a ReLU cannot."
4. Before: "VGG-11, VGG-16, and VGG-19 differ only in `arch`, and
   :citet:`Simonyan.Zisserman.2014` compared six members of the family."
   After: "VGG-11, VGG-16, and VGG-19 differ only in `arch`. The network has
   become a piece of data, and this, more than any layer, is what VGG left
   behind."
5. Before: "VGG-11 performs eight times the arithmetic of AlexNet, so we
   train a narrower member of the family on Fashion-MNIST"
   After: "Eight times AlexNet's arithmetic is more than a demonstration
   should cost, so we train a narrower member of the family on Fashion-MNIST"
6. Before: "Inserting fully connected layers earlier, to let the network
   compute more per position, would destroy the spatial structure and cost
   even more. NiN solves both problems with layers that the reader has met in
   :numref:`subsec_1x1`."
   After: "What does it buy with all those connections? Only the step from a
   feature map to one score per class. NiN removes the head, and it changes
   the body as well. A convolution followed by a ReLU computes, at each
   position, a thresholded linear function of its window. To compute
   something richer from the same window without widening it, we can add
   layers that look at one position only. Luckily, we have already met such
   a layer (:numref:`subsec_1x1`)."
7. Before: "Reducing the channels before an expensive convolution recurs
   throughout this chapter."
   After: "A narrow layer placed before an expensive one in this way is
   called a *bottleneck*. It is the most widely reused idea of this section,
   and we will meet it again in :numref:`subsec_resnext` and
   :numref:`sec_efficient_cnns`."
8. Before: "Third, GoogLeNet matches AlexNet with less than a third of its
   arithmetic, although it sees $96 \times 96$ images instead of
   $224 \times 224$."
   After: "Second, GoogLeNet matches AlexNet with an eighth of its parameters
   and from images with less than a fifth of the pixels. Its small count of
   multiply-adds, however, is due to those smaller images and not to the
   design: the count grows roughly in proportion to the number of pixels, so
   at $224 \times 224$ GoogLeNet would perform about 1.5 billion
   multiply-adds, more than AlexNet."

## Repetition removed

- Training procedure (owned by `alexnet`): the full restatement of the
  warmup/cosine schedule under VGG is now one clause with
  :eqref:`eq_cosine_warmup`; NiN's is "exactly as we trained VGG";
  GoogLeNet's names only the one difference (input size).
- $1 \times 1$ convolution (owned by ch. 6, `subsec_1x1`): the
  re-explanation with $c_\textrm{i} \times c_\textrm{o}$ weight matrix is
  reduced to one "Recall that" sentence.
- "VGG-11 performs eight times AlexNet's arithmetic": was stated in two
  consecutive paragraphs, now once. "Each block halves the resolution, 224 to
  7": was stated three times, now twice (definition, figure paragraph).
- "4% of AlexNet's parameters" for NiN: was stated twice, now once.
- GoogLeNet's ImageNet 2014 win: was in the GoogLeNet opening and again in
  the comparison; kept only in the comparison.
- The paragraph "The three networks contributed the parts ..." recapped each
  network one paragraph before the Summary did the same. Removed; only its
  judgment on GoogLeNet's hand-tuned branches is kept (two sentences).
- Summary: last sentence shortened; it no longer repeats NiN's result.

## Paper retelling shortened

- Simonyan and Zisserman "trained networks of 11 to 19 layers ... and found"
  and "compared six members of the family": removed.
- Winograd algorithms in GPU libraries (`lavin2016fast`): removed (BLK-2).
- Original NiN's channel counts (1,024 on ImageNet, 192 on CIFAR-10):
  removed; the fact that ours is a simplification stays.
- GoogLeNet stem inventory in prose: replaced by "five layers, three
  convolutions and two max-poolings, which shrink the image eightfold and
  emit 192 channels" (BLK-13).
- RepVGG forward reference (`ding2021repvgg`): removed (BLK-15, partly).
- Kept: the last paragraph on pretraining and auxiliary classifiers (the
  review singles it out as the bridge), and the one-sentence ImageNet 2014
  caveat.

## Review findings

Applied: BLK-1, BLK-2, BLK-3, BLK-4 (prose part: three facts kept; no table),
BLK-5, BLK-6, BLK-7, BLK-8 (cut to one sentence where 0.03 is chosen; the
constant-rate paragraph is gone), BLK-9, BLK-10 and BLK-11 (prose: caveat
first, and the text now says that the table does not settle the head
question and points to Exercise 1), BLK-12, BLK-13, BLK-14 (spelling noted
once; class not renamed), BLK-15 (RepVGG clause cut, ResNeXt clause kept as
one sentence; nothing moved between files), "bottleneck" named where
introduced (structural finding 5).

Deferred:
- Structural 1 (train all four at one input size): needs new runs.
- Structural 2 (NiN with the paper's wide last block): needs code and runs.
- Structural 3 (VGG body with two heads in the text): needs a new cell.
- Structural 4 (replace full GoogLeNet by a small Inception network): code.
- Structural 6 (counting function for parameters and multiply-adds): code.
- BLK-4 table of VGG-11 against AlexNet: not added, to keep the section
  from growing; prose reduced instead.
- BLK-14 rename of the class `GoogleNet`: code.
- Three sibling headings called "Training" (BLK-5): headings kept as the
  brief requires.

## Factual claims added or changed

- Added: three nonlinearities let the stack "compute functions of its
  $7 \times 7$ window that one linear filter followed by a ReLU cannot".
- Added: the name *bottleneck* for a narrow layer placed before an expensive
  one. Note that Inception's version only reduces; ResNet's
  reduce-compute-restore block is the fuller form.
- Added: GoogLeNet at $224 \times 224$ "would perform about 1.5 billion
  multiply-adds, more than AlexNet" ($276\textrm{M} \times (224/96)^2
  \approx 1.50$G, the review's figure; not measured by a cell), and "less
  than a fifth of the pixels" ($(96/224)^2 = 18\%$), "an eighth of its
  parameters" (6.0M of 46.8M).
- Changed: the observation "the parameters of a dense head add little
  accuracy here" is replaced by "the table does not settle whether a dense
  head is needed".
- Changed: "GoogLeNet matches AlexNet with less than a third of its
  arithmetic, although ..." is replaced as in pair 8.
- Added: "apart from NiN the networks end within a point or two of each
  other" and NiN's spread "is the widest of this section". Stored final
  accuracies in the four frameworks: VGG 0.915 to 0.919, NiN 0.839 to 0.876,
  GoogLeNet 0.904 to 0.910; AlexNet about 91%. All consistent with the
  unchanged table and with "about 92%", "84–88%", "about 91%".
- Added judgment: NiN's result is "rather modest ... for a network that
  performs nearly as much arithmetic as AlexNet".
- Removed claims: "at 0.1 some runs diverge"; the constant-rate 0.01 runs
  (78% to 87%); VGG-11's 9.2M convolutional parameters, 98% of arithmetic,
  119.6M head, 410 MB; the 470M convolutional multiply-adds of the narrow
  VGG; GoogLeNet's 6.7% top-5 error.
- Changed wording: the name Inception "alludes to the film *Inception* and
  the catchphrase it inspired" (was "its line").
- The stem "emits 192 channels" and reduces "eightfold": from the code.

## Slides that now disagree with the body (not edited)

- Slide "NiN Needs the Warmup and the Decay Most": reports the constant-rate
  0.01 runs (78% to 87%) and divergence at 0.1, which the body no longer
  mentions.
- Slide "GoogLeNet Ends at About 91% ...": "AlexNet's accuracy with an
  eighth of its parameters and less than a third of its arithmetic". The
  body now attributes the arithmetic to the smaller input.
- Scoreboard slide title "The Dense Head Adds Little Accuracy; ...": the
  body now says the table does not settle this.
- VGG slides: table rows with 9.2 M / 119.6 M (93%) and the bullet
  "nonlinearities instead of one" remain; consistent with the body, only
  more detailed.
- The slides never use the word "bottleneck" except in one subtitle
  ("parallel branches and 1×1 bottlenecks"); consistent.
