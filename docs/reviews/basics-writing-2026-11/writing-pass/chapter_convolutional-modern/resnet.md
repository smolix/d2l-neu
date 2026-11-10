# Writing pass: `chapter_convolutional-modern/resnet.md` (section 7.4)

Prose only. No code cell, label, figure include, caption, or slide was changed.
`check_prose_only.py` ends "OK: only prose changed"; `lint_source.py` exits 0.

Chapter order (from `_quarto.yml`): index, alexnet, blocks, batch-norm,
**resnet**, efficient-convnets, training-recipes, convnext, cnn-design. This
section is fourth, so it owns residual connections, the bottleneck block,
and grouped convolutions, and defers to earlier sections for He
initialization, the warmup and cosine schedule, global average pooling, the
missing bias before batch normalization, and why batch normalization permits
a larger learning rate.

## Words

- Checker count (all prose in the file, exercises included): 4,482 before,
  4,473 after (−9).
- Display equations: 3 delimiters before, 2 after. The definition of
  $L^*_\mathcal{F}$ is gone (RES-1) and the nested DenseNet bracket expression
  became a one-line recursion (RES-13).

The section is flat in length, not shorter. About 330 words of retold papers
and repeated facts were removed, and about 320 were spent on review findings
that ask for content: the two-block worked case for the gradient (RES-4),
the statement of what is known (structural finding 3), the plain-networks-
are-nested correction (RES-2), the exact weight count for ResNeXt (RES-11),
and the limits of the VGG comparison (RES-10).

## Before and after

1. Opening (RES-6).
   - Before: "The networks of this chapter so far became more accurate as they
     became deeper, from the eight layers of AlexNet to the 22 of GoogLeNet"
   - After: "On ImageNet, depth and accuracy had so far grown together, from
     the eight layers of AlexNet to the 22 of GoogLeNet"
2. Paper numbers in the introduction.
   - Before: "A plain network with 34 layers had a higher training error than
     one with 18 layers throughout training, and a top-1 validation error of
     28.5% against 27.9%. With residual blocks, the 34-layer network reached a
     top-1 error of 25.0% where the 18-layer one reached 27.9%"
   - After: "This small change was enough. On ImageNet, where a plain network
     with 34 layers trained worse than one with 18, the residual network with
     34 layers was almost three points more accurate than its 18-layer
     relative"
3. Function classes (RES-1, RES-2).
   - Before: "can compute the identity only for particular values of its
     weights, and the degradation results show that gradient descent does not
     find them."
   - After: "can compute the identity on the nonnegative inputs that it
     receives, but only for particular values of its weights. So the function
     classes of plain networks are nested as well, and plain and residual
     networks both belong in the right panel of the figure. Nesting, then, is
     not what the plain networks lack."
4. Gradient (RES-4).
   - Before: "The shortcut also helps the gradient. The Jacobian of
     $\mathbf{x} + g(\mathbf{x})$ with respect to $\mathbf{x}$ is
     $\mathbf{I} + \partial g / \partial \mathbf{x}$."
   - After: "The shortcut also changes how the gradient travels. Let us stack
     two such blocks, [...] the output contains the input itself, plus two
     corrections."
5. Status of the explanation (new; structural finding 3).
   - After: "How much do these two arguments explain? They make it plausible
     that residual networks are easy to optimize. They do not show why plain
     networks fail. [...] As with batch normalization, the remedy is better
     understood than the disease."
6. ResNet-D (RES-5).
   - Before: ":citet:`He.Zhang.Zhang.ea.2019` therefore average-pool the input
     of the shortcut with a $2 \times 2$ window and stride 2 and then apply a
     $1 \times 1$ convolution with stride 1, so that every position
     contributes; they call the result *ResNet-D*. In the bottleneck blocks of
     ResNet-50 they also move the stride [...]"
   - After: "This is rather crude: with stride 2, the convolution reads only
     one of every four input positions. An exercise repairs it."
7. The comparison with VGG (RES-10).
   - Before: "That is about a point above the quarter-width VGG, the best
     network of :numref:`sec_blocks`, at $96 \times 96$ instead of
     $224 \times 224$ pixels."
   - After: the same sentence with "with a quarter of its parameters", then
     "We should not read more into the comparison than it can bear. The two
     networks differ in architecture, normalization, peak learning rate, and
     input size at once, so the run does not isolate what the residual
     connection contributes."
8. ResNeXt cost (RES-11).
   - Before: "The $1 \times 1$ convolutions cost $O(c b)$ multiply-adds per
     position and the grouped $3 \times 3$ convolution $O(b^2 / g)$."
   - After: "What does "wider at the same cost" amount to? A bottleneck block
     in the first stage of ResNet-50 has $c = 256$, $b = 64$, and a single
     group, hence [...] $69{,}632$ weights. The block of
     :numref:`fig_resnext_block` [...] has [...] $70{,}144$: twice the width
     in the middle for less than 1% more weights."

## Repetition removed

- **He initialization** (owned by `alexnet` and `subsec_he_init`): the
  variance formula and its purpose are gone from the shared prose and from
  the MXNet tab; one clause with the reference remains. "The bias of the
  output layer starts at zero" is now said once in the shared prose instead
  of once there and once in each of four tabs.
- **Warmup and cosine schedule** (owned by `alexnet`): the description of the
  schedule is reduced to its name and the reference.
- **Why batch normalization permits a larger rate** (owned by `batch-norm`):
  the three-line restatement of scale invariance is one clause with the
  reference.
- **No bias before batch normalization** (owned by `batch-norm`): one clause
  with the reference.
- **Global average pooling** (owned by `blocks`): the gloss "which averages
  each channel over all positions" is cut.
- **The two shortcut variants**: described before the code and again after
  it; the second description is now one sentence pointing at the figure.
- **DenseNet alternates dense blocks and transition layers**: said three
  times, now twice (where the transition layer is introduced, and in the
  closing paragraph).
- **Peak learning rate 0.1**: removed from the Summary.

## Review findings

Applied: RES-1 and RES-2 (function-class section reframed; $L^*$ cut; the
figure is kept because the include cannot change, and the prose now says
both kinds of network sit in its right panel), RES-3 (the generic block
zeroes "its last layer"; the ResNet block names $\gamma$ and $\beta$ of the
last batch normalization), RES-4, RES-5, RES-6, RES-7 (prose says once that
`b1` builds the stem and `block` builds a stage), RES-8, RES-9, RES-10,
RES-11, RES-12, RES-13, RES-14, RES-15. Structural finding 3 (what is known)
and the prose half of finding 4 (the block as initialized is not the
identity) are applied. ResNeXt now says that its block returns in
`sec_cnn-design`.

Deferred:

- Structural 1 (paired plain and residual experiment): needs new cells and
  runs. The prose now says that the run does not isolate the residual
  connection and points to the exercise.
- Structural 4, code half (zero-initialize the last $\gamma$): code change.
- Structural 5 (move the DenseNet code to an exercise): removes code cells.
- Structural 6 (retitle the section): heading change that should follow the
  DenseNet decision.
- RES-7, code half (rename `block` to `stage`, `b1` to `stem`): code change,
  and `cnn-design` and the exercises use the names.
- Exercises (move 4 into the text, merge 2 and 8): untouched by the brief.

## Factual claims added or changed

1. "the residual network with 34 layers was almost three points more accurate
   than its 18-layer relative" (He et al., Table 2: 25.03% against 27.88%
   top-1 error). Replaces the four quoted numbers.
2. A plain layer "can compute the identity on the nonnegative inputs that it
   receives", hence plain networks are nested as well.
3. The two-block identity $\mathbf{x}_2 = \mathbf{x}_0 + g_1(\mathbf{x}_0) +
   g_2(\mathbf{x}_1)$ and its Jacobian as a product of two
   $(\mathbf{I} + \partial g)$ factors. Stated for blocks without the final
   ReLU; the next paragraph says what ResNet adds to the path.
4. He et al. "report that the gradients of their plain networks, which use
   batch normalization, kept a healthy size" (their Section 4.1). From memory
   of the paper, not rechecked against the text.
5. The branch of `Residual` is zero when $\gamma$ and $\beta$ of `bn2` are
   zero; $\gamma$ starts at 1, so the block does not start as the identity;
   zero-initializing $\gamma$ is an ingredient of `sec_training_recipes`
   (that section lists it).
6. ResNet-18 has "a quarter of [the] parameters" of the quarter-width VGG
   (11.2M against the 43.1M in the table of `blocks`).
7. The validation loss "stops improving after the first few epochs". Checked
   in the stored plots of all four frameworks: the minimum falls in epochs 3
   to 5 and the final value is 0.26 to 0.28. The other output claims (about
   93%; last three epochs within a point; training loss below 0.01) also hold
   in all four.
8. ResNeXt weights: $2cb$ and $9b^2/g$; 69,632 against 70,144, "less than 1%
   more".
9. The measured speedup of grouped convolutions is smaller than $g$ "since
   multiply-adds count arithmetic and not time", with a reference to
   `sec_efficient_cnns`, which makes that measurement. This replaces "make
   less efficient use of the hardware".
10. DenseNet's parameter efficiency is what "its authors attribute to the
    reuse of features" (was "Because each layer reuses ...").
11. "we do not train one here" (DenseNet).

Removed claims about runs the reader does not see: NiN diverging at a peak of
0.1, and the behaviour at a constant rate of 0.1. "At 0.03 it ends about a
point lower" is kept as the one record of the authors' sweep.

## Slides that now disagree with the body

- "The Shortcut Adds the Identity to Every Block's Jacobian": "Plain stack:
  the gradient passes through every layer and is scaled by each" is the
  vanishing-gradient reading that the body now sets aside.
- "Nested Function Classes ..." and the Recap bullet "Nesting": they present
  nesting as what residual blocks provide; the body now says plain networks
  are nested too and the difference is how easily the identity is reached.
  The slide formula uses $\min_f L$, which the body no longer displays.
- "Batch Normalization Lets ResNet-18 Train at a Peak Rate of 0.1": still
  restates scale invariance and the NiN divergence, which the body dropped.
- "Two Variants ...": mentions ResNet-D by name; the body no longer does.
- Recap: "peak rate 0.1" is no longer in the Summary.
