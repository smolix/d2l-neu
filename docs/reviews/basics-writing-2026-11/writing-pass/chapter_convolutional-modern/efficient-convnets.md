# Writing pass: Efficient Convnets (7.5)

Source: `chapter_convolutional-modern/efficient-convnets.md`. Prose only; no
code cell, label, figure, or slide was changed.

Prose words (checker): 4070 before, 3908 after (-162). Checker: "OK: only
prose changed". `tools/lint_source.py`: clean. The checker's em-dash warning
is the table separator rows (`|---|`), present before the pass; there is no
em dash in the prose. Display equations went from three to one: the cost-ratio
derivation and the compound-scaling formula were removed (neither had a
label).

## Before and after

1. Depthwise-separable cost (owned by chapter 6).
   - Before: "A dense $k \times k$ convolution from $c_\textrm{i}$ to
     $c_\textrm{o}$ channels on an $h \times w$ output performs
     $hw \cdot c_\textrm{i} c_\textrm{o} k^2$ multiply-adds: every output
     channel filters every input channel. A *depthwise-separable* convolution
     (:numref:`sec_depthwise_separable`) splits this work in two." (followed
     by the displayed derivation of the ratio)
   - After: "We already have the tool. The depthwise-separable convolution of
     :numref:`sec_depthwise_separable` replaces a dense $k \times k$
     convolution by a depthwise $k \times k$ convolution, which filters each
     input channel on its own, and a pointwise $1 \times 1$ convolution, which
     mixes the channels at each position. By :eqref:`eq_depthwise_sep_ratio`,
     the pair needs only the fraction $1/c_\textrm{o} + 1/k^2$ of the
     multiply-adds, and of the weights, of the dense layer with
     $c_\textrm{o}$ output channels."

2. Why the inverted bottleneck is wide in the middle.
   - Before: "Once the spatial convolution is depthwise, it is cheap even on
     wide tensors, and the trade reverses."
   - After: "A depthwise convolution is cheap even on wide tensors, and it has
     a reason to want them: it learns one spatial filter per channel, so the
     number of filters in a block equals the width at which the depthwise
     layer runs."

3. The inverted block, with widths.
   - Before: "the residual stream stays thin, a $1 \times 1$ convolution
     expands it (by a factor of 6 in the paper), the depthwise $3 \times 3$
     convolution operates in the wide expanded space, and a $1 \times 1$
     projection returns to the thin stream."
   - After: "The residual stream stays thin, say 24 channels. A $1 \times 1$
     convolution expands it sixfold to 144 channels, the depthwise
     $3 \times 3$ convolution filters in the wide space, and a $1 \times 1$
     projection returns to 24 channels."

4. Parameters at equal budget (EFF-5).
   - Before: "Dense $3 \times 3$ convolutions spend parameters about nine
     times faster per pair of channel counts, so the dense network stops at
     128 channels where the separable one reaches 512."
   - After: "A dense $3 \times 3$ layer between the same two widths has about
     nine times as many weights, so with the same budget the dense network
     stops at 128 channels where the separable one reaches 512."

5. What the experiment does not show.
   - Before: "Matching the arithmetic instead shows what the separable network
     buys. ... Efficient architectures accept the restriction of the separable
     layer and spend the savings on more channels; the exercises measure what
     that buys in accuracy."
   - After: "The comparison that matters on a phone is the one at equal
     arithmetic, and our experiment does not make it. ... Whether the
     separable network then wins on Fashion-MNIST is for the exercises to
     measure."

6. Time against arithmetic.
   - Before: "The time differs by much less than the arithmetic. ... Operation
     counts are a proxy for time; how good a proxy depends on the hardware,
     which is the theme of the rest of this section."
   - After: "The time differs by much less than the arithmetic: a factor of 2
     to 3.5, where the multiply-adds differ by 7.7. ... Operation counts are
     only a proxy for time, and how good a proxy depends on the hardware."

7. Compound scaling, in numbers (EFF-8).
   - Before: "EfficientNet scales all three together, $d = \alpha^\phi$,
     $w = \beta^\phi$, $r = \gamma^\phi$ with
     $\alpha \beta^2 \gamma^2 \approx 2$, relative to a base network. A single
     knob $\phi$ then sets the budget, and each unit of $\phi$ roughly doubles
     the cost."
   - After: "Each step multiplies the depth by 1.2, the width by 1.1, and the
     resolution by 1.15, which about doubles the cost, since
     $1.2 \cdot 1.1^2 \cdot 1.15^2 \approx 1.92$. The number of steps is then
     the only knob. Three steps, for instance, give a network 1.7 times as
     deep and 1.3 times as wide, on images 1.5 times as large, at about 7
     times the cost."

8. RepVGG.
   - Before: "RepVGG :cite:`ding2021repvgg` obtains both."
   - After: "Can we have both? RepVGG :cite:`ding2021repvgg` shows that we
     can, by training one network and deploying another."

## Repetition removed

- **Depthwise-separable convolution and its cost ratio** (owner:
  `chapter_convolutional-neural-networks/channels.md`). The re-derivation and
  its displayed equation are gone; one sentence defines the pair, one cites
  :eqref:`eq_depthwise_sep_ratio`. Kept: "no number of output channels makes
  it more than 9 times cheaper" and the 8.4 at 128 channels, in one clause.
- **The restriction of the separable layer** (same owner, the
  $\mathbf{K}_{o,c} = P_{o,c}\mathbf{D}_c$ paragraph). The two-sentence
  restatement after the table is now "The separable layer is a restricted
  dense layer (:numref:`sec_depthwise_separable`)". The sentence "Efficient
  architectures accept the restriction ... and spend the savings on more
  channels", which chapter 6 has almost verbatim, is removed.
- **He initialization** (owners: chapter 4, then `alexnet.md`). The variance
  $2/n_\textrm{in}$ and the definition of $n_\textrm{in}$ are removed; one
  clause with the reference remains.
- **Cosine schedule and the unstable constant rate** (owners: `alexnet.md`
  for the schedule, `resnet.md` for "with a constant rate ... the validation
  accuracy moves by up to a few points"). Reduced to "again decays to zero
  along a half cosine (:numref:`sec_alexnet`), here without warmup". The
  sentence about one to three points of wobble, a run the notebook does not
  show, is removed.
- **Momentum.** This section is the earliest in the chapter to use it, so it
  keeps one clause of explanation and the reference to `sec_momentum`. The
  derivation $1/(1-0.9) = 10$ and the second statement that momentum
  "lengthens its steps" are removed. `cnn-design.md` repeats the explanation
  and should reduce it to a reference here.
- **MobileNet against VGG-16 on ImageNet.** Stated once, in "MobileNet". The
  restatement of "33 times fewer parameters and 27 times less arithmetic"
  after the table is removed.
- **The ratio $1/c_\textrm{o} + 1/k^2$** now appears once in the body and
  once in the summary (was three times in the body).
- **Paper retelling.** MobileNetV4 is cut. The EfficientNetV2 mechanism,
  which repeated the paragraph on time directly above it, is one clause. "The
  paper's ablations favor the linear projection" and the width-multiplier
  sentence (used only by an exercise that defines the term itself) are cut.

## Review findings

Applied: EFF-2 (width multiplier left to its exercise), EFF-3 prose part (the
missing reason for the wide depthwise layer, three paragraphs, concrete
widths 24, 144, 24), EFF-4 (paragraph split into network and training;
momentum reduced), EFF-5, EFF-7, EFF-8 (formula replaced by numbers, which
also removes the clash of $w$, $\beta$, $\gamma$ with the feature-map width
and the batch-norm parameters), EFF-9 (MobileNetV4 cut, EfficientNetV2 kept),
EFF-10 (example now halves the kernel), EFF-12 ("buys", "the trade reverses",
"spend parameters", "spends 7.7 times" replaced; "spends most of the layer's
time" kept as literal), EFF-13 in the body and the table ("validation
accuracy"). Structural finding 1 is addressed in part: compound scaling has a
worked number and the survey is shorter, but the subsection still has no
code.

Deferred:

- EFF-1: needs a figure for what a phone processor sustains; I had no source
  I could check.
- EFF-3, figure: needs a new two-panel figure.
- EFF-6 and structural finding 2: needs a third training run and table row.
  The prose now says plainly that the experiment does not make the
  equal-arithmetic comparison.
- EFF-11: needs RepVGG's ablation number from the paper, which I could not
  verify here; it would also add paper narration.
- EFF-13 in the exercises: five uses of "test accuracy" remain, since
  exercises were out of scope. The body and table now say "validation
  accuracy".
- Structural finding 3 (no code for the inverted bottleneck), the proposal to
  move "Scaling and Searching" to `cnn-design.md`, and the trimming of
  exercises: code or cross-file changes.

## Factual claims added or changed

- Added: a depthwise convolution "learns one spatial filter per channel, so
  the number of filters in a block equals the width at which the depthwise
  layer runs" (from EFF-3; true for a channel multiplier of 1).
- Added: example widths 24, 144, 24 for the inverted bottleneck (a stage of
  MobileNetV2 at expansion 6; presented with "say").
- Added: three compound-scaling steps give depth 1.7, width 1.3, resolution
  1.5 times the base, at about 7 times the cost ($1.2^3 = 1.73$,
  $1.1^3 = 1.33$, $1.15^3 = 1.52$, $1.92^3 = 7.1$).
- Added: "This answers our opening question rather well" (a judgment on the
  MobileNet ImageNet numbers that follow it).
- Added: "a factor of 2 to 3.5, where the multiply-adds differ by 7.7". The
  stored outputs give forward-time ratios of 2.9 (PyTorch), 2.4 (TensorFlow),
  2.6 (JAX), and 2.4 (MXNet).
- Changed: momentum steps "about $1/(1-0.9) = 10$ times longer" when
  gradients agree became "up to ten times longer".
- Changed: ResNet-18 "was trained with plain SGD" (checked against
  `resnet.md`, `configure_optimizers`).
- Changed: the batch-norm fold example. Was $\sigma_o^2 + \epsilon = 4$,
  $\beta_o = 0.5$, giving $\hat{\mathbf{W}}_o = \mathbf{W}_o$,
  $\hat{b}_o = -0.5$. Now $\sigma_o^2 + \epsilon = 16$, $\beta_o = 1$, giving
  $\hat{\mathbf{W}}_o = \mathbf{W}_o/2$, $\hat{b}_o = 1 - 0.5 = 0.5$.
- Changed: the closing sentence of "Scaling and Searching". Was "an
  optimization against measured latency" for every network named; EfficientNet
  optimized multiply-adds, so it now says "from an optimization".
- Changed: the widths 11, 23, 46, 46 leave the dense network "76,000
  parameters, a seventh of MobileNet's" (recomputed: 75,854). The "49 million
  multiply-adds" stays in the exercise only.
- Removed: "With a constant rate the test accuracy wobbles by one to three
  points"; the width multiplier; MobileNetV4; EfficientNetV2 explained by
  multiply-adds per value read (the body's own paragraph on time says it).
- Stored outputs, all four frameworks, agree with the unchanged numbers:
  542,474 parameters; 50.3 and 384.9 million multiply-adds, ratio 7.66;
  accuracy 0.913 to 0.919 and 0.936 to 0.939; fusion error 3.8 to
  5.7 $\times 10^{-6}$.

## Slides that now differ from the body

- The fold example on the batch-norm slide still uses
  $\sigma_o^2 + \epsilon = 4$, $\beta_o = 0.5$,
  $\hat{\mathbf{W}}_o = \mathbf{W}_o$, $\hat b_o = -0.5$. Correct, but no
  longer the body's example.
- The compound-scaling slides give $d = \alpha^\phi$, $w = \beta^\phi$,
  $r = \gamma^\phi$ and $\alpha\beta^2\gamma^2 \approx 2$; the body no longer
  uses these symbols.
- One slide has a MobileNetV4 bullet; the body no longer mentions it.
- The slides do not say "test accuracy", so the change to "validation
  accuracy" causes no disagreement there.
