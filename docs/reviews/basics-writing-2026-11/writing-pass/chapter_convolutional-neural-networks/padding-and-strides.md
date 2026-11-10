# Writing pass: Padding, Stride, and Dilation

Source: `chapter_convolutional-neural-networks/padding-and-strides.md`.
Prose only; no code cell, label, figure include, display equation, exercise,
or slide was changed.

## Size and checks

- Prose words (checker count, exercises included): 3516 before, 3506 after.
- `check_prose_only.py`: "OK: only prose changed", no em-dash warning.
- `tools/lint_source.py`: no findings.

The section did not shrink much because the cuttable material the review
names (two drill cells, the two-axis dilation display) is code or a labeled
equation and stays for now; see "Deferred".

## Before and after

1. Introduction.
   - Before: "Each setting changes the output shape by a formula we derive
     below, and each changes how far a unit in a deep network sees."
   - After: "The formulas for the output shape, which we derive below, are a
     matter of counting window positions. The more interesting question is
     how far a unit in a deep network sees, and here the three settings
     differ a great deal."
2. Introduction (PAD-1).
   - Before: "A third need is for one kernel to see a wide region without the
     weights and the computation of a large kernel."
   - After: "Occasionally we want it both ways, a wide view and an output at
     every pixel, without paying for a large kernel."
3. Zero Padding.
   - Before: "The kernel size decides whether this padding can be split
     evenly."
   - After: "Can this padding always be split evenly between the two sides?
     Only if the kernel size is odd."
4. Zero Padding (toward PAD-3).
   - Before: "An $8\times 8$ input, a $3\times 3$ kernel, and one pixel of
     padding on every side ($p_\textrm{h}=p_\textrm{w}=2$ in total; padding
     arguments in code count the rows or columns on each side, here 1) should
     give an $8\times 8$ output."
   - After: "A word of caution: padding arguments in code count the rows or
     columns on each side, whereas $p$ in our formulas is the total. One
     pixel on every side is `padding=1` in code and
     $p_\textrm{h}=p_\textrm{w}=2$ in the formula. With an $8\times 8$ input
     and a $3\times 3$ kernel, it should give an $8\times 8$ output."
5. Strided Convolution (PAD-6, prose part).
   - Before: "The settings can also differ between the axes. A $3\times 5$
     kernel with padding $(0, 1)$ on each side and stride $(3, 4)$ gives"
   - After: "The settings can also differ between the axes. As a deliberately
     awkward test, a $3\times 5$ kernel with padding $(0, 1)$ on each side
     and stride $(3, 4)$ gives"
6. Stride and the Receptive Field (PAD-8).
   - Before: "Every stride multiplies the reach of all later kernels, so a
     downsampling network's receptive field grows geometrically with depth,
     which is how classifiers see the whole image after a few dozen layers."
   - After: "Every stride multiplies the reach of all later kernels, so the
     receptive field grows geometrically for as long as the network keeps
     striding. After five halvings, common in classifiers for
     $224\times 224$ images, a kernel reaches 32 times further than it would
     at the bottom, and no unit needs hundreds of layers to see the whole
     image."
7. Dilated Convolution (PAD-9, prose part).
   - Before: "For output shapes, a dilated kernel behaves like a dense kernel
     of its effective size. [...] With dilation 1 this is the strided
     formula"
   - After: "For output shapes there is nothing new to learn: a dilated
     kernel behaves like a dense kernel of its effective size. [...] The
     display looks worse than it is. With dilation 1 it is the strided
     formula"
8. Closing.
   - Before: "We can now predict the output shape of a single-channel layer
     under any padding, stride, and dilation, and how far its units see. Real
     inputs, however, have several channels, and a layer should produce
     several feature maps, which is the subject of :numref:`sec_channels`."
   - After: "Throughout this section the input had a single channel and the
     layer a single kernel. Real images have several channels, and a useful
     layer detects more than one pattern. :numref:`sec_channels` removes both
     restrictions."

## Repetition removed

- **Receptive field** (owned by `conv-layer.md`, which defines it and derives
  $r = 1+\sum(k_i-1)$ and the $2L+1$ count). The redefinition and the
  $2L+1$ count are gone; the subsection now opens "We saw in
  :numref:`sec_conv_layer` that at stride 1 each $k\times k$ layer widens the
  receptive field by $k-1$ pixels along each axis, so that it grows only
  linearly with depth. Stride changes this."
- **"Very many layers to see the whole image"** was told in the introduction
  and again, with the number (more than 100 layers), at the start of
  "Strided Convolution". The introduction now says only that units "see only
  a few pixels farther with each layer"; the number stays in the stride
  section.
- **Border cut away** (PAD-4): the last clause of "Boundary Pixels" repeated
  the opening's point; the paragraph now ends on the underuse of the border.
- **Formulaic closing** ("We can now predict ...") replaced; the Summary is
  the only recap.
- **Framework defaults**: four tabs that each said "defaults to padding 0 and
  stride 1" became one shared sentence, plus a one-line JAX tab for the
  exception (`padding='SAME'`).
- **"we speak of dilation $d$; by default it is 1"** folded into the clause
  "$d=1$, the default, is the ordinary convolution".

## Review findings

Applied:

- PAD-1: dilation folded into the second need ("both ways") instead of a
  third need after "two needs".
- PAD-4: paragraph ends at the underuse of the border.
- PAD-7: "$2\times 2=4$" is now "$(3-1)\cdot 2=4$".
- PAD-8: geometric growth qualified ("for as long as the network keeps
  striding"); "a few dozen layers" removed.
- PAD-10: "the dilated stack is the strided stack with nothing discarded".
- PAD-11: reference to `sec_transposed_conv` removed; the sentence ends at
  "learned upsampling".
- Structural note on framework tabs: the four default tabs merged (above).

Already addressed in the file before this pass (left as found):

- PAD-2 (caption explains the stretched cells), PAD-5 (figure redrawn with
  two shades; the text "Each shaded output has the shade of its window. The
  light one ... the dark one ..." is kept verbatim).

Deferred:

- PAD-3 (let $p$ mean padding per side): a notation decision for the whole
  chapter (`pooling.md`, an exercise in `channels.md`, and this file's slides
  all use the total). Mitigated here in prose only: the translation between
  code and formula is now a sentence of its own, and "padding $p$" is said to
  remain the total.
- PAD-6 (cut the two anisotropic drill cells): needs cells removed. The
  second one is now introduced as "a deliberately awkward test".
- PAD-9 (single-axis dilation formula): `eq_conv_output_shape` is a labeled
  display used by exercise 6 with $d_\textrm{h}$, $s_\textrm{h}$,
  $p_\textrm{h}$, and its code check is a cell. Prose around it now says it
  contains nothing new.
- Figure placement of `fig_conv_rf_towers` panel (a): needs a figure split or
  move.
- Exercise notes (rebalance dilation exercises, "statistical benefit" in
  2.2, add a shape-tracing problem): exercises are out of scope.

## Factual claims added or changed

- Added: "After five halvings, common in classifiers for $224\times 224$
  images, a kernel reaches 32 times further than it would at the bottom."
  (Five stride-2 stages take 224 to 7, as in ResNet-style networks;
  $2^5=32$.) Replaces "see the whole image after a few dozen layers".
- Added: shape-preserving padding "is the usual choice".
- Added: the strides 3 and 2 of `img_conv_stride` are "an unusual pair that
  keeps the two axes easy to tell apart". This states a motive for the
  figure; delete the clause if it is not the intended one.
- Changed: defaults are now stated for all libraries at once, "A layer that
  is given neither setting uses stride 1 and no padding", with `nnx.Conv`
  named as the exception in the JAX tab. Same content as the four former
  tabs.
- Removed: the forward reference to transposed convolution.

## Slides against the body

No slide states something the body now contradicts. Two differences of
emphasis:

- Slide "Each Stride Multiplies the Reach of All Later Kernels" says
  "Downsampling makes the receptive field grow geometrically with depth";
  the body now adds "for as long as the network keeps striding".
- Slide "Where this fits" lists "transposed convolution for upsampling
  (computer vision chapter)" under Skips; the body no longer names the
  transposed convolution, only "learned upsampling". ("Computer vision
  chapter" is also the old part name.)
