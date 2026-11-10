# Writing pass: `chapter_convolutional-modern/convnext.md` (7.7)

Prose words (checker): 3550 before, 3521 after. Checker: "OK: only prose
changed". `lint_source.py`: clean. No code cell, label, figure, caption,
table, exercise, or slide was touched. Position in the chapter: seventh of
eight sections, so it owns almost nothing; alexnet, blocks, batch-norm,
resnet, efficient-convnets and training-recipes come first.

## Before and after

1. Opening (CNX-1: question first, transformer sketch second).
   - Before: "In 2021 a network without convolutions overtook them. A
     *vision transformer* ... cuts the image into a grid of patches ... [six
     sentences] ... Did attention cause the gap?"
   - After: "In 2021 a network without convolutions overtook them. The Swin
     Transformer :cite:`liu2021swin`, built around a layer called
     *attention*, reached 81.3% top-1 accuracy on ImageNet, against 76.1% for
     a ResNet-50 of similar cost. Did attention cause the gap? If it did,
     convolutional networks were obsolete." The sketch follows, opened by "A
     sketch of the newcomer is all we need here; :numref:`chap_transformers`
     treats it properly."
2. - Before: "We read the sequence of changes, build the network, and train
     a small ConvNeXt on Fashion-MNIST against a ResNet-18 with the same
     number of parameters."
   - After: "Every row of their table is a full ImageNet training run, so we
     cannot repeat the experiment. We read the table instead, build the
     network it ends in, and train a small ConvNeXt on Fashion-MNIST against
     a ResNet-18 with the same number of parameters."
3. - Before: "Once the spatial convolution is depthwise, the economics
     reverse, as :numref:`sec_efficient_cnns` showed for MobileNetV2: the
     block can keep a thin stream between blocks and expand it inside."
   - After: "Why does the ResNet bottleneck compress its input before the
     spatial convolution and expand it afterwards? Because the cost of a
     dense convolution grows with the square of the width. The cost of a
     depthwise convolution grows only in proportion to the width, so the
     reason to compress is gone, and the block can keep a thin stream between
     blocks and expand it inside, as MobileNetV2 does
     (:numref:`sec_efficient_cnns`)."
4. - Before: "Keeping only the GELU between the two $1 \times 1$
     convolutions adds 0.7 points, more than any other change in this group.
     Keeping only one normalization adds 0.1, ..."
   - After: "... more than any other change in this group. Why should a
     network improve when it loses two of its three nonlinearities? The paper
     offers no reason beyond the analogy with the transformer, and the gain
     is best taken as an empirical finding. Keeping only one normalization
     adds 0.1, ..."
5. - Before: "Of the 5.9 points that separate the 2015 ResNet-50 from
     ConvNeXt, 2.7 come from training and 3.2 from the architecture."
   - After: "Of the 5.9 points that separate the 2015 ResNet-50 from
     ConvNeXt, 2.7 thus come from training and 3.2 from the architecture, at
     least in this order of application."
6. - Before: "We set the stochastic-depth rate of ConvNeXt to zero because
     the ResNet control has no such mechanism."
   - After: "Like recipe B, we leave out stochastic depth, by setting
     ConvNeXt's rate to zero, so that the two networks differ in architecture
     alone."
7. - Before: "The compact ResNet-18 is ahead by about two points in every
     run, and both validation curves level off over the last epochs as the
     learning rate decays to zero. The result is specific to this setting.
     ... It does not match the running time. What the experiment shows is
     that ConvNeXt's design choices do not transfer automatically to a small
     grayscale task; it does not rank convolutional architectures in
     general."
   - After: "The compact ResNet-18 is ahead by about two points in every run.
     On this task, and within 20 epochs, the modernized block does not help.
     As far as arithmetic goes the control is a fair one: ... We did not
     match the running time, which multiply-adds do not determine
     (:numref:`sec_efficient_cnns`). Two cautions are in order. Both
     validation curves flatten over the last epochs, but this reflects the
     cosine schedule taking the learning rate to zero and does not tell us
     that a design tuned for 300 epochs has converged after 20. And one small
     grayscale task does not rank convolutional architectures in general.
     What the experiment does show is that design choices selected on
     ImageNet do not transfer automatically."
8. - Before: "That is a convolution whose kernel size equals its stride"
   - After: "That is nothing more exotic than a convolution whose kernel
     size equals its stride"

Kept verbatim: "Did attention cause the gap? ...", "Which of these
differences carried the improvement?", "comparing recipes as much as
architectures", the mixing sentence, "$49/9 \approx 5$", "here fewer such
layers, one per block, train better than three", "The roadmap removed more
than it added", the parameter-count check, "The experiment does not show that
convolutions are better", "a controlled experiment wearing an architecture's
name".

## Repetition removed

- Recipe effect (owner: training-recipes). The 600-epoch, 80.4% figure and
  the description of the recipe are gone; one clause with a reference stays.
- Depthwise convolution (owner: chapter 6, `sec_depthwise_separable`). The
  definition "applies one $k \times k$ filter to each channel separately" is
  reduced to the reference; the mixing contrast stays.
- Stochastic depth (owner: training-recipes). The mechanism with $p$ and
  $1/(1-p)$ is reduced to "drops the whole branch of a block at random
  during training; we reuse `d2l.drop_path`".
- Layer normalization per position (owner: batch-norm). One sentence instead
  of two; a reference to `subsec_layer-normalization-in-bn` added in the
  roadmap.
- Recipe B settings (owner: training-recipes). Weight decay, base rate,
  smoothing and Mixup values are no longer restated; only what differs
  (length, warmup share) is said.
- Within the file: the downsampling layer was described three times (roadmap,
  figure sentence, "The Full Network"), now once plus a clause; "20 epochs on
  the full training set at $64 \times 64$" was said twice, now once; the four
  disclaimers after the result are two; the summary cites "almost half of its
  gain" instead of repeating 2.7 of 5.9, and no longer re-lists which layer
  does which mixing.
- "Limits and Open Problems" is one paragraph instead of two (CNX-10).

## Review findings

Applied: CNX-1 (question first), CNX-2 (order of application), CNX-3
(argument for the inverted bottleneck stated), CNX-4 (bare equation
reference replaced by `sec_padding`), CNX-5 ("in front of each of its two
parts"; the missing explanation is admitted), CNX-6 in part (a clause before
the figure says that layer scale is defined below), CNX-7 (reason for
switching stochastic depth off corrected), CNX-8 in part (prose only: the
test of transfer is announced before the run, the disclaimers are halved,
"level off" no longer suggests convergence, the running-time remark has a
reason and a reference), CNX-10 (sparse convolutions, the redundancy finding,
GRN by name, $31 \times 31$ and re-parameterization removed from the text;
the three citations stay), structural finding 5.

Deferred:
- CNX-9, structural 2 (mini-roadmap): needs new cells and runs.
- CNX-8, remainder (rebuild the experiment around the null result): needs
  runs.
- Structural 1 (dependence on chapters 10 and 11, or moving the section):
  chapter order is the author's decision. The sketch is now marked as a
  sketch.
- CNX-6, remainder ($\boldsymbol{\gamma}$ collides with batch norm): the
  symbol is in the figure caption and the code; chapter-wide decision.
- CNX-11 (title): the title recurs in the slides and the cell IDs; left for
  one decision.
- CNX-12 ("validation" against "test" accuracy; `ModernRecipe` defined
  twice): chapter-wide wording and code.
- Exercise "Channel redundancy" (cut or keep): exercises not touched. It
  still stands on its own, since it states what GRN is meant to prevent.

## Factual claims added or changed

1. "Every row of their table is a full ImageNet training run, so we cannot
   repeat the experiment." New.
2. "the cost of a dense convolution grows with the square of the width. The
   cost of a depthwise convolution grows only in proportion to the width".
   New; true when input and output widths scale together.
3. Moving the depthwise convolution forward is "the only step backwards on
   the roadmap". New; true of the table and of the 79.9% intermediate value.
4. "The paper offers no reason beyond the analogy with the transformer" for
   the gain from fewer activations. New; from memory of the paper, not
   re-checked against it.
5. "without which training was unstable" replaces "which the authors found
   necessary for stable training" (same content).
6. "The authors report no further gain from kernels beyond $7 \times 7$"
   replaces the sentence naming $9 \times 9$ and $11 \times 11$.
7. AdamW glossed as "Adam with decoupled weight decay
   (:numref:`chap_optimization`)". New.
8. Stochastic depth is off "Like recipe B ... so that the two networks
   differ in architecture alone" (was: "because the ResNet control has no
   such mechanism", untrue of `StochasticResidual`).
9. The flattening of the validation curves is attributed to the schedule,
   with the remark that it "does not tell us that a design tuned for 300
   epochs has converged after 20". The curves themselves were not inspected
   (images); the sentence asserts nothing about them beyond the old one.
10. "We did not match the running time, which multiply-adds do not
    determine (:numref:`sec_efficient_cnns`)."
11. Summary: "almost half of its gain over the 2015 ResNet-50 comes from the
    recipe alone" (2.7 of 5.9).
12. Dropped: "the paper reports the same for larger models and for transfer
    to detection and segmentation"; the 80.4% of the 600-epoch recipe.

Stored outputs, all four frameworks: ConvNeXt 0.915 to 0.918, compact
ResNet-18 0.937 to 0.941, gap 1.9 to 2.5 points. "About two points" and the
table's "≈ 91.5%" and "≈ 94%" hold; nothing changed there.

## Slides now out of step with the body

- "A 600-epoch recipe takes it to 80.4% (Wightman et al., 2021)": the body
  no longer gives this figure (it lives in training-recipes).
- "Stochastic depth off: the control has no counterpart": the body now gives
  the other reason (recipe B omits it; architecture-only comparison).
- The comparison slide restates recipe B's values (0.05, $2 \times 10^{-3}$,
  0.1, 0.2); the body refers to training-recipes for them. Not a
  contradiction.
- The stochastic-depth slide gives the per-example probability $p$; the body
  now leaves the mechanism to training-recipes.
- The "Wide context" slide still names $31 \times 31$ depthwise kernels,
  which the body no longer mentions.
