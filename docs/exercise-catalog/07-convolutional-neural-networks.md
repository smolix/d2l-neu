# Chapter Overview — Convolutional Neural Networks

Six files (`why-conv`, `conv-layer`, `padding-and-strides`, `channels`,
`pooling`, `lenet`), 38 existing exercises, all in bare `1.`-list legacy
style with no names/tags (per the group style review). Best external
sources: **Prince's *Understanding Deep Learning* ch. 10** (Problems
10.1–10.15, freely downloadable) is an outstanding match — it poses the
*exact same* derivations (equivariance proofs, stride/dilation equation
writing, receptive-field arithmetic, multi-channel weight matrices) our
sections already build, letting several of our proposed additions cite a
verified primary source almost verbatim. **CS231n Assignment 2** and
**Michigan EECS 498-007 Assignment 3** both implement conv/pool
forward+backward from raw NumPy and are the standard source for the one
genuine gap in our existing sets: nobody currently asks the reader to
implement pooling's *backward* pass. **CMU 11-785 HW2P1**'s "scanning MLP →
CNN" conversion is a near-perfect code-level echo of `why-conv.md`'s core
argument. **Michael Nielsen ch. 6** contributes three well-matched
Problems/Exercise (conv+pool backprop, omitting the dense head, translation
invariance vs. data augmentation) via direct fetch of the live chapter.
**Goodfellow ch. 9** and the **CS231n conv-notes module** are confirmed
prose-only (no exercises) and used only for topic scoping, as instructed.
Coverage gaps found: pooling has no external tradition at all in Prince's
book (he treats downsampling solely via strided convolution); grouped/
depthwise-separable convolution (`channels.md`) has no exercise tradition
in any of the five suggested sources — our own ex. 7–8 are already the
most rigorous treatment found anywhere. The existing sets are strong
overall (29 of 38 items kept outright); the recurring defect is the
"why do you expect X vs. Y to differ?" / "what happens when...?" pattern
with no comparison metric, present in 3 of the 6 files.

---

## chapter_convolutional-neural-networks/why-conv.md — From Fully Connected Layers to Convolutions

**Topic:** Derive convolution from two inductive biases — translation
equivariance and locality — applied to an unconstrained fully-connected
layer on image-shaped input.
**Current exercises:** 6; disposition: keep 3, rewrite 3, drop 0 — the two
derivation/proof items (Δ=0 → NiN; convolution symmetry) and the
"equivariance as bad bias" item are precise and checkable. The audio
subitem (c), the text item, and the boundary item all use "Can you...?" /
"Do you think...?" filler framing or lack a stated artifact (flagged in
the prior style review); their substance is worth keeping, just not their
current wording.

**External sources found:**
- CMU 11-785 (Introduction to Deep Learning), HW2P1, Spring 2021 — students
  convert a pretrained 3-layer "scanning" MLP (`Flatten→Linear(192,8)→ReLU→
  Linear(8,16)→ReLU→Linear(16,4)`) into an exactly-equivalent CNN via
  weight reshaping, in both a simple and a weight-shared ("distributed")
  variant — this *is* the "Constraining the MLP" derivation, executed in
  code — https://github.com/shuyanwang/11785HW (structure and semester
  confirmed; implementation detail cross-checked against
  https://mantutor.com/product/cmu11485-homework-2-part-1-solved/).
- Michael Nielsen, *Neural Networks and Deep Learning*, ch. 6, Problem
  "The idea of convolutional layers is to behave in an invariant way
  across images... [yet] our network can learn more when all we've done
  is translate the input data. Can you explain why this is actually quite
  reasonable?" — a direct complement to our ex. 3 (equivariance as a
  *bad* bias): here equivariance is a *good* bias, and the puzzle is why
  data augmentation still helps — http://neuralnetworksanddeeplearning.com/chap6.html
  (fetched and verified directly).
- Simon J. D. Prince, *Understanding Deep Learning* (MIT Press, 2023),
  Problem 10.1*: "Show that the operation in equation 10.3 is equivariant
  with respect to translation" — the same proof our section states as a
  derivation rather than an exercise — https://udlbook.github.io/udlbook/
  (problems verified via the linked PDF release, v5.0.3).
- Ian Goodfellow, Yoshua Bengio & Aaron Courville, *Deep Learning*, ch. 9
  — prose only, no exercises, as expected; used solely to confirm our
  "sparse interactions / parameter sharing / equivariance" framing matches
  the standard three-part motivation — https://www.deeplearningbook.org/.
- Stanford CS231n, "Convolutional Networks" notes module — prose only, no
  exercises; its parameter-counting narrative (VGG's 100M/140M FC-layer
  parameter breakdown) is the same rhetorical move as our $10^6\times10^3$
  argument, confirming the framing rather than supplying a new problem —
  https://cs231n.github.io/convolutional-networks/ (fetched and verified).

**Proposed problem set** (7 problems):
1. [conceptual] **Network in Network via $\Delta=0$.** Show that when the
   convolution kernel size is $\Delta = 0$, :eqref:`eq_conv-layer-channels`
   reduces to an MLP applied independently to the channel vector at each
   spatial location, i.e., the Network in Network architecture.
   *Provenance:* original (unchanged from the book's existing ex. 1;
   citation to :cite:`Lin.Chen.Yan.2013` retained).
1. [conceptual] **Locality and equivariance for 1D audio.** Audio is a
   1D sequence.
    1. State one property of audio (e.g., a phoneme's acoustic signature)
       for which imposing locality and translation equivariance is a good
       assumption, and one property (e.g., a fixed-position hotword) for
       which it is not.
    1. Derive the 1D analog of :eqref:`eq_conv-layer` for audio.
    1. Compute the spectrogram of a short recording of your choice,
       treat it as a 2D image (time × frequency), and state which axis
       admits translation equivariance and which does not, justifying
       your answer from how the corresponding physical quantity behaves
       under a shift along that axis.
   *Provenance:* rewrite of the book's existing ex. 2 (removes the
   "Can you...?" framing from part (c), adds a concrete deliverable).
1. [conceptual] **Equivariance as a bad inductive bias.** Give one example
   of a vision task whose correct label changes under translation
   (breaking the value of translation equivariance as a bias), and state
   precisely which of this section's two assumptions the task violates.
   *Provenance:* original (unchanged from the book's existing ex. 3).
1. [conceptual] **Convolutions on text.** Convolutional layers assume that
   shifting the input shifts the output identically. Name two structural
   properties of natural-language text (e.g., long-range subject–verb
   agreement, or a token's meaning depending on absolute sentence
   position such as negation scope) that make this assumption a worse fit
   for text than for images, and for each, state what a 1D convolution
   over word embeddings would fail to capture.
   *Provenance:* rewrite of the book's existing ex. 4 (removes "Do you
   think...?" opinion framing; keeps the concrete follow-up ask).
1. [conceptual] **Convolutions at the image boundary.** For a $\Delta=1$
   kernel applied to an $n \times n$ image with no padding, count how many
   of the $(n-2)^2$ valid output positions use a boundary input pixel (one
   with $i \in \{1, n\}$ or $j \in \{1, n\}$) versus an interior one. State
   how this count changes as $n \to \infty$ and connect the answer to why
   :numref:`sec_padding` is needed.
   *Provenance:* rewrite of the book's existing ex. 5 (replaces the open
   "what happens...?" prompt with a countable deliverable).
1. [short-code] **Constrained MLP, concretely.** Using only basic tensor
   indexing (no `Conv2d`), implement a locally-connected layer: a linear
   map from an $n\times n$ input to an $n\times n$ output where output
   $(i,j)$ depends only on the $\Delta$-neighborhood input patch, with an
   independent weight tensor per output location (i.e., $\mathsf{V}$ from
   :eqref:`eq_conv-layer` before the equivariance constraint). Then tie
   the weights across locations and verify numerically, on a small
   synthetic input and hand-chosen $\mathbf{V}$, that the output now
   matches direct 2D cross-correlation.
   *Provenance:* adapted from CMU 11-785 HW2P1's scanning-MLP-to-CNN
   conversion (overlap medium — same weight-sharing argument, built from
   scratch here rather than converting pretrained weights; cite on
   adoption).
1. [conceptual] **Translation equivariance vs. data augmentation.** A
   translation-equivariant model already treats a pattern the same way at
   every location. Explain why training such a model on translated copies
   of the same images can nonetheless still improve its measured
   accuracy, and identify one concrete way finite image boundaries (not
   the core equivariance assumption) are responsible.
   *Provenance:* adapted from Nielsen, ch. 6, Problem 437600 (overlap
   medium — reframed from "network learns more from augmentation" to a
   direct explain-why task; cite on adoption).

---

## chapter_convolutional-neural-networks/conv-layer.md — Convolutions for Images

**Topic:** Define 2D cross-correlation, implement it directly, use it for
edge detection and kernel learning, and connect it to convolution,
matrix multiplication (im2col), and receptive fields.
**Current exercises:** 4; disposition: keep 4, rewrite 0, drop 0 — the
prior style review found this file "well-formed, no issues"; all four
items (diagonal-edge transpose behavior, manual kernel design, autodiff
error on the custom `Conv2D`, cross-correlation as matrix multiplication)
have concrete, checkable outcomes and are kept unchanged. External
material below is added rather than substituted.

**External sources found:**
- Stanford CS231n, Assignment 2, `ConvolutionalNetworks.ipynb` — implement
  `conv_forward_naive` (checked to $\approx 2\times 10^{-8}$ relative
  error against a reference) and `conv_backward_naive` (checked by
  numerical gradient, tolerance $\sim 10^{-5}$); a separate demo applies
  hand-built grayscale and edge-detection kernels to real images —
  https://cs231n.github.io/assignments2024/assignment2/ (fetched and
  verified; kernel-implementation detail cross-checked against a public
  notebook mirror).
- Michigan EECS 498-007 (Justin Johnson), Assignment 3,
  `convolutional_networks.py` — the same naive forward/backward
  convolution pair, plus a timed comparison against `FastConv` — official
  page https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment3.html
  (existence and Q1/Q2 structure confirmed via the course's own citation
  in this book's Resources list and independently via a third-party
  assignment-solution summary).
- CMU 11-785, HW2P1 — implement `Conv1D` forward and backward "by hand";
  the handout explicitly disallows autodiff for credit — a stricter
  version of our existing ex. 3 (which only asks what error autodiff
  raises) — semester and scope confirmed via
  https://github.com/shuyanwang/11785HW (Spring 2021: "P1: Scanning MLP,
  CNN... Bonus: Mean and Max Pooling").
- Michael Nielsen, ch. 6, Problem 366128: "Backpropagation in a
  convolutional network... Suppose we have a network containing a
  convolutional layer, a max-pooling layer, and a fully-connected output
  layer... How are the equations of backpropagation modified?" — a
  conceptual complement to this section's forward-only treatment —
  http://neuralnetworksanddeeplearning.com/chap6.html (verified).
- Prince, *Understanding Deep Learning*, Problem 10.5: "Draw weight
  matrices in the style of figure 10.4d for (i) the strided convolution...
  (ii) the convolution with kernel size 5... and (iii) the dilated
  convolution" and Problems 10.9–10.10 (receptive field of a 3-layer
  stack at kernel size 3 vs. 7) — direct matches for the "Convolution as
  Matrix Multiplication" and "Feature Map and Receptive Field" subsections
  — https://udlbook.github.io/udlbook/ (verified via PDF release v5.0.3).
- Araujo, Norris & Sim, "Computing Receptive Fields of Convolutional
  Neural Networks," *Distill* (2019) — already cited in this book's
  Resources list; its effective-vs-theoretical receptive field framing
  (also present in-section via :citet:`Luo.Li.Urtasun.ea.2016`) has no
  homework-style exercise anywhere in the five suggested sources, so we
  originate one below — https://distill.pub/2019/computing-receptive-fields/.

**Proposed problem set** (8 problems):
1. [short-code] **Diagonal edges under transpose.** Construct an image
   `X` with a diagonal edge.
    1. Apply this section's kernel `K` to it; describe the result.
    1. Apply it to `X` transposed.
    1. Apply the transposed kernel `K` to the original `X`.
   *Provenance:* original (unchanged from the book's existing ex. 1).
1. [conceptual] **Designing kernels by hand.**
    1. Given a direction $\mathbf{v}=(v_1,v_2)$, derive a kernel that
       detects edges orthogonal to $\mathbf{v}$.
    1. Derive a finite-difference operator for the second derivative;
       state its minimum kernel size and which image structures respond
       most strongly.
    1. Design a blur kernel and state one reason to use it.
    1. State the minimum kernel size for a derivative of order $d$.
   *Provenance:* original (unchanged from the book's existing ex. 2).
1. [conceptual] **Autodiff on the from-scratch `Conv2D`.** Try to compute
   the gradient through the `Conv2D` class defined in this section using
   your framework's automatic differentiation. What error message results,
   and which line of the implementation is responsible?
   *Provenance:* original (unchanged from the book's existing ex. 3).
1. [short-code] **Cross-correlation as a general matrix product.** Extend
   this section's im2col construction (written for a $2\times 2$ kernel)
   to a $3\times 3$ kernel on a $4\times 4$ input, and confirm the matrix
   product reproduces `corr2d`.
   *Provenance:* original (unchanged from the book's existing ex. 4).
1. [conceptual] **Deriving the conv-layer backward pass.** For the
   single-channel `Conv2D` layer of this section, derive $\partial
   \ell/\partial \mathbf{V}$ and $\partial \ell /\partial \mathbf{X}$ for
   a scalar loss $\ell$, in the style of the backpropagation equations
   from earlier chapters. Check your $\partial \ell/\partial\mathbf{V}$
   formula against the kernel updates printed by the "Learning a Kernel"
   experiment above.
   *Provenance:* adapted from Nielsen, ch. 6, Problem 366128 (overlap
   high — same derivation, restricted to one conv layer rather than
   conv+pool+FC; cite on adoption).
1. [short-code] **Verifying the backward pass numerically.** Implement
   the gradient of `corr2d(X, K)` with respect to `K` directly (no
   autodiff), using your answer to the previous problem, and check it
   against a numerical (finite-difference) gradient on a small random `X`
   and `K` to a relative error below $10^{-4}$.
   *Provenance:* adapted from CS231n Assignment 2 / Michigan EECS 498-007
   Assignment 3 `conv_backward_naive` (overlap high — same numerical
   gradient-check pattern; cite on adoption).
1. [short-code] **Filters on a real image.** Load one image from this
   book's own figure directory (or a single Fashion-MNIST example),
   convert it to a single channel, and apply three hand-designed kernels
   from problem 2 (an edge detector, your blur kernel, and a sharpening
   kernel you design as $2\times$identity$-$blur) with `corr2d`. Display
   each result next to the original.
   *Provenance:* adapted from CS231n Assignment 2's grayscale/edge-
   detection image-processing demo (overlap medium — single-channel here
   vs. CS231n's RGB-channel version, since multi-channel inputs are not
   yet introduced; cite on adoption).
1. [extended] **Measuring the effective receptive field.** Build a stack
   of three $3\times 3$ convolutional layers (random weights, stride 1, no
   padding) on a $32\times 32$ single-channel input. Backpropagate a unit
   gradient from one output unit in the last layer to the input, and plot
   $|\partial \text{output}/\partial \text{input}|$ over the $32\times32$
   grid. Compare the region of non-negligible gradient magnitude to the
   $7\times 7$ theoretical receptive field from :eqref:`eq_receptive_field`,
   and describe the shape of the falloff.
   *Provenance:* inspired by Luo et al. (2016) and the Distill article
   "Computing Receptive Fields" (overlap low — no directly assignable
   homework exercise exists for this in the five suggested sources; cite
   both papers on adoption since the section already does).

---

## chapter_convolutional-neural-networks/padding-and-strides.md — Padding and Stride

**Topic:** Control output size and the region of input contributing to
each output element via padding, stride, and dilation.
**Current exercises:** 7; disposition: keep 6, rewrite 0, drop 1 — the
output-shape check, mirror-padding implementation, the stride cost-benefit
pair, the fractional-stride question, and the dilation/gridding-schedule
question are all concrete and kept unchanged (the dilation item is
unusually rich and, per the prior review, "the strongest treatment found"
for this topic). Ex. 2 ("For audio signals, what does a stride of 2
correspond to?") is dropped: the prior style review found the word
"audio" appears nowhere else in this file, and the same underlying idea
(subsampling a signal changes what frequencies survive) is already covered
with a concrete, worked example in `pooling.md` ex. 8's aliasing exercise
— keeping both would be redundant and this file's version has no
established context to anchor it.

**External sources found:**
- Prince, *Understanding Deep Learning*, Problems 10.2–10.4: "Write out
  the equivalent equation for the 1D convolution with a kernel size of
  three and a stride of two..."; "...dilation rate of two..."; "...kernel
  size of seven, a dilation rate of three, and a stride of three" — the
  same equation-instantiation exercise this section's formulas invite,
  not currently asked of the reader — https://udlbook.github.io/udlbook/
  (verified via PDF release v5.0.3).
- Prince, Problem 10.11: three stacked 1D conv layers with different
  kernel/stride/dilation per layer; compute the receptive field at each
  layer — a more incremental complement to this section's existing ex. 7
  (which already asks for a 4-layer dilation-only schedule) — same source.
- Prince, Problem 10.15: draw the sampling matrix that keeps every other
  entry of a 1D input, and show that composing it with the stride-1,
  kernel-3 convolution matrix reproduces the stride-2 convolution — ties
  the stride concept directly to `conv-layer.md`'s matrix-multiplication
  framing — same source.
- Dumoulin & Visin, "A guide to convolution arithmetic for deep learning"
  (2016), already in this book's Resources list — the standard reference
  for padding/stride/dilation, illustrated with animations rather than
  posed as exercises — https://arxiv.org/abs/1603.07285.
- No dedicated padding/stride/dilation problem sets were found in CS231n,
  Michigan EECS 498-007, Nielsen, or Goodfellow: all four treat shape
  arithmetic as an implementation detail inside larger convolution
  exercises rather than as a topic of its own. This section (and Prince's
  book) are the only places that isolate it — a genuine coverage gap
  elsewhere, not a failure of this file.

**Proposed problem set** (8 problems):
1. [conceptual] **Verifying the general output-shape formula.** Given the
   final code example in this section (kernel $(3,5)$, padding $(0,1)$,
   stride $(3,4)$), calculate the output shape by hand and check it
   against the experimental result.
   *Provenance:* original (unchanged from the book's existing ex. 1).
1. [short-code] **Mirror padding.** Implement mirror (reflect) padding,
   where border values are reflected to extend a tensor, and demonstrate
   it on a small tensor next to zero padding.
   *Provenance:* original (unchanged from the book's existing ex. 3).
1. [conceptual] **Computational benefit of stride.** What are the
   computational benefits of a stride larger than 1?
   *Provenance:* original (unchanged from the book's existing ex. 4).
1. [conceptual] **Statistical benefit of stride.** What might be the
   statistical benefits of a stride larger than 1?
   *Provenance:* original (unchanged from the book's existing ex. 5).
1. [conceptual] **Fractional stride.** How would you implement a stride
   of $\frac{1}{2}$? What does it correspond to, and when would it be
   useful? Compare your answer with the transposed convolutions of
   :numref:`sec_transposed_conv`.
   *Provenance:* original (unchanged from the book's existing ex. 6).
1. [conceptual] **Dilation schedule and gridding.** A network stacks four
   $3\times 3$ convolutions with stride 1 and dilations $1,2,4,8$. Use
   :eqref:`eq_receptive_field` (each kernel replaced by its effective
   size) to compute the receptive field of one output element. Which
   pixels inside that field does the output actually depend on? When does
   this *gridding* effect become a problem, and how would you choose a
   dilation schedule that avoids it?
   *Provenance:* original (unchanged from the book's existing ex. 7).
1. [conceptual] **Instantiating the general formula.** Using the combined
   padding/stride/dilation formula derived in this section, predict the
   output shape for a $7\times 7$ kernel, dilation $(3,3)$, stride
   $(3,3)$, and padding $0$ on a $32\times 32$ input. Verify your
   prediction with `comp_conv2d`.
   *Provenance:* adapted from Prince, Problems 10.2 and 10.4 (overlap
   high — same equation-instantiation task, ported to 2D; cite on
   adoption).
1. [conceptual] **Stride as sampling.** For a length-6 1D input, write
   the $3\times 6$ weight matrix (in the style of :numref:`sec_conv_layer`'s
   matrix-multiplication view) for a stride-1, kernel-3 convolution, and
   the $3\times 6$ 0/1 matrix that keeps every other row of a length-6
   vector. Show that composing the sampling matrix with the stride-1
   convolution matrix reproduces the stride-2 convolution derived in this
   section.
   *Provenance:* adapted from Prince, Problem 10.15 (overlap high — same
   construction, connected explicitly to :numref:`sec_conv_layer`'s
   im2col framing; cite on adoption).

---

## chapter_convolutional-neural-networks/channels.md — Multiple Input and Multiple Output Channels

**Topic:** Extend convolution to multiple input/output channels, the
$1\times 1$ convolution, and grouped/depthwise/depthwise-separable
factorizations.
**Current exercises:** 8; disposition: keep 7, rewrite 1, drop 0 — this
is one of the strongest sets in the chapter (ex. 7–8's grouped- and
depthwise-separable-convolution cost comparisons were singled out in the
prior style review as "strong, well-scaffolded"). Ex. 5 ("Express
convolutions as a matrix multiplication, even when the convolution window
is not $1\times 1$") is now redundant with `conv-layer.md`'s newly added
"Convolution as Matrix Multiplication" subsection, which already covers
exactly this for the single-channel case; it is rewritten to extend that
treatment to the multi-channel case instead of repeating it.

**External sources found:**
- Prince, *Understanding Deep Learning*, Problems 10.6–10.7: draw the
  $12\times 6$ and $6\times 12$ weight matrices relating inputs to outputs
  in a multi-channel convolution — the direct multi-channel extension of
  the single-channel matrix view this file's ex. 5 currently lacks —
  https://udlbook.github.io/udlbook/ (verified via PDF release v5.0.3).
- Prince, Problem 10.8: given input shape $c_i\times h\times w$ and two
  stacked multi-channel conv layers, count weights and biases at each
  layer — narrower than, but consistent with, this file's existing ex. 2.
- Prince, Problem 10.14: count weights and biases for a single $5\times 5$,
  3-input/10-output-channel layer — simpler than our existing ex. 2/3, so
  not adopted, but it confirms our set is already at least as rigorous.
- No exercise tradition for grouped, depthwise, or depthwise-separable
  convolutions was found in any of the five suggested sources: CS231n,
  Michigan EECS 498-007, Nielsen, and Goodfellow all predate or omit
  MobileNet/Xception-style channel factorization, and Prince's problems
  stop at dense multi-channel convolution. This file's ex. 7–8 are the
  most rigorous treatment of the topic found anywhere in this search — a
  finding, not a gap in this file.

**Proposed problem set** (8 problems):
1. [conceptual] **Composing two convolutions.** Given kernels of size
   $k_1$ and $k_2$ with no nonlinearity between them:
    1. Prove the composition equals a single convolution.
    1. State the dimensionality of the equivalent single kernel.
    1. Is the converse true — can any convolution be factored into two
       smaller ones?
   *Provenance:* original (unchanged from the book's existing ex. 1).
1. [conceptual] **Cost and memory accounting.** For an input of shape
   $c_i\times h\times w$, a kernel of shape
   $c_o\times c_i\times k_h\times k_w$, padding $(p_h,p_w)$, and stride
   $(s_h,s_w)$: state (i) the forward computational cost, (ii) the
   forward memory footprint, (iii) the backward memory footprint, and
   (iv) the backward computational cost.
   *Provenance:* original (unchanged from the book's existing ex. 2).
1. [conceptual] **Scaling channels and padding.** By what factor does the
   operation count grow if both $c_i$ and $c_o$ double? What happens if
   the padding doubles instead?
   *Provenance:* original (unchanged from the book's existing ex. 3).
1. [short-code] **Verifying the $1\times 1$ equivalence.** Are `Y1` and
   `Y2` in this section's final code example exactly equal? Explain why,
   referencing the reshape used in `corr2d_multi_in_out_1x1`.
   *Provenance:* original (unchanged from the book's existing ex. 4).
1. [short-code] **Multi-channel im2col.** Extend `conv-layer.md`'s
   single-channel im2col construction to this section's multi-input,
   multi-output setting: for the `X`, `K` of this section's first
   example, build the patch matrix (each row now a flattened
   $c_i\times k_h\times k_w$ patch) and the reshaped kernel, and confirm
   their product reproduces `corr2d_multi_in_out`.
   *Provenance:* rewrite of the book's existing ex. 5, extended using the
   multi-channel weight-matrix view of Prince, Problems 10.6–10.7
   (overlap medium; cite on adoption).
1. [conceptual] **Strip-buffered convolution.** To implement a $k\times k$
   convolution, one option scans horizontally, reading a $k$-wide strip
   and producing a 1-wide output strip at a time; the alternative reads a
   $k+\Delta$-wide strip and produces a $\Delta$-wide output strip at
   once. Why is the latter preferable, and is there a limit to how large
   $\Delta$ should be?
   *Provenance:* original (unchanged from the book's existing ex. 6).
1. [conceptual] **Grouped convolution trade-off.** A grouped convolution
   with $g$ groups acts as a block-diagonal matrix on channels.
    1. By what factor does grouping reduce parameters and compute versus
       a dense convolution with the same $c_i$, $c_o$, and kernel size?
    1. What is the downside of $g$ groups, and how could you partially
       fix it without giving up most of the savings?
   *Provenance:* original (unchanged from the book's existing ex. 7).
1. [short-code] **Depthwise-separable VGG block.** Take a block of two
   dense $3\times 3$ convolutions with $c$ input and $c$ output channels
   each (the VGG building block). Replace each with its depthwise-
   separable counterpart.
    1. Compute the parameter count and multiply-count on an $h\times w$
       input for both variants.
    1. Which of the two stages (depthwise or pointwise) dominates the
       separable block's cost, and what does this suggest about where to
       spend additional capacity?
   *Provenance:* original (unchanged from the book's existing ex. 8).

---

## chapter_convolutional-neural-networks/pooling.md — Pooling

**Topic:** Summarize local windows via max- or average-pooling to reduce
spatial resolution and gain local translation tolerance.
**Current exercises:** 8; disposition: keep 5, rewrite 3, drop 0 — ex. 8
(aliasing / blur-pool) was flagged by the prior review as a genuinely
strong, concrete exercise and is kept unchanged. Ex. 5 ("Why do you expect
max-pooling and average pooling to work differently?") has no comparison
metric and is rewritten into a numeric two-patch comparison. Ex. 6 (min-
pooling) and ex. 7 (softmax pooling) are both bounded "why not X"
design-alternative questions and are merged into one entry to make room
for a genuinely missing item: nothing in this file's existing set asks the
reader to implement pooling's *backward* pass, which is the standard
complement to the forward implementation already present (ex. 1–3).

**External sources found:**
- Stanford CS231n, Assignment 2 — `max_pool_forward_naive` and
  `max_pool_backward_naive` (gradient routed only to the argmax location
  within each window, checked by numerical gradient) —
  https://cs231n.github.io/assignments2024/assignment2/ (verified).
- Michigan EECS 498-007, Assignment 3 — the same forward/backward
  max-pooling pair inside `convolutional_networks.py`.
- Michael Nielsen, ch. 6, Problem 366128 (backprop through conv +
  max-pool + FC) — the pooling-gradient half of the same derivation this
  file's forward-only exercises (1–3) leave untested —
  http://neuralnetworksanddeeplearning.com/chap6.html (verified).
- Zhang, "Making Convolutional Networks Shift-Invariant Again" (2019),
  already cited in this section — the direct source for the aliasing /
  blur-pool framing of the existing ex. 8.
- None of Prince's *Understanding Deep Learning* ch. 10 Problems
  (10.1–10.19) address pooling at all — the book downsamples exclusively
  via strided convolution and never introduces max/average pooling as a
  distinct mechanism. This is a genuine, book-level coverage gap in an
  otherwise excellent source, not a shortcoming of this file.

**Proposed problem set** (8 problems):
1. [short-code] **Average pooling via convolution.** Implement average
   pooling as a special case of a convolutional layer.
   *Provenance:* original (unchanged from the book's existing ex. 1).
1. [conceptual] **Max-pooling cannot be a single convolution.** Prove that
   max-pooling cannot be implemented through a convolution alone.
   *Provenance:* original (unchanged from the book's existing ex. 2).
1. [conceptual] **Max-pooling via ReLU.** $\textrm{ReLU}(x)=\max(0,x)$.
    1. Express $\max(a,b)$ using only ReLU operations.
    1. Use this to implement max-pooling via convolutions and ReLU layers.
    1. How many channels and layers does this need for a $2\times 2$
       window? For a $3\times 3$ window?
   *Provenance:* original (unchanged from the book's existing ex. 3).
1. [conceptual] **Cost of pooling.** For an input of shape $c\times h\times
   w$, pooling window $p_h\times p_w$, padding $(p_h,p_w)$, and stride
   $(s_h,s_w)$, state the computational cost of the pooling layer.
   *Provenance:* original (unchanged from the book's existing ex. 4).
1. [short-code] **Max vs. average pooling, concretely.** Construct two
   synthetic $4\times 4$ patches with the same mean value: one with a
   single high-value outlier on a uniform background, one with a smooth
   gradient. Apply $2\times 2$ max- and average-pooling (stride 2) to
   both and report all four outputs. Using these numbers, state one
   scenario where max-pooling is preferable (name it) and one where
   average pooling is preferable (name it).
   *Provenance:* rewrite of the book's existing ex. 5 (replaces the open
   "why do you expect" prompt with a concrete numeric comparison).
1. [conceptual] **Pooling alternatives.**
    1. Do we need a separate minimum-pooling operator, or can it be
       expressed with an operation already in this section?
    1. We could use softmax to combine values in a pooling window instead
       of max or mean. Give one concrete reason this is not popular in
       practice (e.g., in terms of what value it returns for a window
       with one strong activation and many weak ones).
   *Provenance:* rewrite merging the book's existing ex. 6 and ex. 7 into
   one entry (both are bounded design-alternative questions; merging
   frees room in the proposed set for the pooling-backward addition
   below without exceeding 8 problems).
1. [short-code] **Aliasing and blur-pool.** Naive stride-2 downsampling
   keeps every second entry of its input.
    1. Apply it to $(1,0,1,0,1,0)$ and $(0,1,0,1,0,1)$ — the same pattern
       shifted by one position. Compare the two outputs. Why is this
       called *aliasing*, and which input frequencies can a stride-2
       subsampler represent faithfully?
    1. Blur-pool applies a two-tap box filter $(\tfrac12,\tfrac12)$ before
       subsampling. Work out what it computes on the two signals above.
       Which pooling operation from this section does it coincide with?
   *Provenance:* original (unchanged from the book's existing ex. 8;
   citation to :cite:`zhang2019making` retained).
1. [short-code] **Backward pass for max-pooling.** Implement the backward
   pass of 2D max-pooling: route the incoming output gradient to the
   input location that achieved the max in each window (zero elsewhere).
   Verify against your framework's autodiff gradient on a small random
   tensor to a relative error below $10^{-4}$.
   *Provenance:* adapted from CS231n Assignment 2 / Michigan EECS 498-007
   Assignment 3 `max_pool_backward_naive` (overlap high — same routing
   rule and numerical-gradient check; cite on adoption).

---

## chapter_convolutional-neural-networks/lenet.md — Convolutional Neural Networks (LeNet)

**Topic:** Combine convolution, pooling, and dense layers into a complete
image classifier; train LeNet-5 on Fashion-MNIST and compare to modern
practice.
**Current exercises:** 5; disposition: keep 4, rewrite 1, drop 0 — the
modernization, architecture-sweep, original-MNIST, and activation-
visualization items are all concrete and kept unchanged. Ex. 5 ("What
happens to the activations when you feed significantly different images...")
matches the review's flagged "see what happens" pattern with no comparison
criterion and is rewritten with an explicit measurement.

**External sources found:**
- Michigan EECS 498-007, Assignment 3 — `ThreeLayerConvNet`
  ("conv – ReLU – $2\times2$ max-pool – linear – ReLU – linear – softmax")
  is close to what ex. 1's "modernize LeNet" already produces; `DeepConvNet`
  adds Kaiming initialization and an optional batchnorm, and the graded
  deliverables include a speed-training ("one-minute deep conv net") and a
  small-dataset overfitting checkpoint — official page
  https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/assignment3.html;
  deliverable names confirmed via a third-party assignment summary.
- Stanford CS231n, Assignment 2 — the standard "verify your network can
  overfit a small subset of data" sanity check before a full training run,
  applied to a `ThreeLayerConvNet` — https://cs231n.github.io/assignments2024/assignment2/
  (verified).
- Michael Nielsen, ch. 6, Exercise 683491: "What classification accuracy
  do you get if you omit the fully-connected layer, and just use the
  convolutional-pooling layer and softmax layer? Does the inclusion of the
  fully-connected layer help?" — directly tests the claim in this
  section's own modern-vs-LeNet Discussion table that global average
  pooling can replace the dense head —
  http://neuralnetworksanddeeplearning.com/chap6.html (verified).
- LeCun, Bottou, Bengio & Haffner (1998), already cited in this section —
  the original LeNet-5 paper reports an error rate below 1% per digit,
  giving a concrete historical benchmark for ex. 3's "try the original
  MNIST dataset."

**Proposed problem set** (8 problems):
1. [short-code] **Modernize LeNet.** Replace average pooling with
   max-pooling and the sigmoid activations with ReLU; retrain and compare.
   *Provenance:* original (unchanged from the book's existing ex. 1).
1. [short-code] **Architecture sweep.** Beyond max-pooling and ReLU,
   determine whether accuracy improves further by adjusting: (i) the
   convolution window size, (ii) the number of output channels, (iii) the
   number of convolutional layers, (iv) the number of fully connected
   layers, (v) the learning rate, initialization, and epoch count.
   *Provenance:* original (unchanged from the book's existing ex. 2).
1. [short-code] **Original MNIST.** Try your best network from the
   previous two problems on the original MNIST dataset. LeCun et al.
   (1998) report an error rate below 1% per digit for the original
   LeNet-5 on this task — state how your result compares.
   *Provenance:* original (unchanged from the book's existing ex. 3;
   success criterion sharpened using the paper's own reported benchmark,
   already cited in this section).
1. [short-code] **Visualizing activations.** Display the activations of
   the first and second convolutional layers for several different
   Fashion-MNIST inputs (e.g., a sweater and a coat).
   *Provenance:* original (unchanged from the book's existing ex. 4).
1. [short-code] **Activations on out-of-distribution inputs.** Measure
   the maximum activation magnitude of the first and second convolutional
   layers separately for (i) in-distribution Fashion-MNIST test images,
   (ii) out-of-distribution photos (e.g., a cat or a car), and (iii) pure
   random noise. Report whether the OOD and noise magnitudes fall inside
   or clearly outside the range observed for in-distribution inputs.
   *Provenance:* rewrite of the book's existing ex. 5 (replaces "what
   happens...?" with an explicit, comparable measurement).
1. [conceptual] **Where LeNet's parameters live.** Using the parameter-
   count formula from :numref:`sec_channels` ($c_o c_i k_h k_w + c_o$) and
   the layer shapes from :numref:`img_lenet_vert`, compute by hand how
   many parameters live in the two convolutional layers combined versus
   the $400\times120$ dense block. Which dominates, and how would this
   balance shift if global average pooling replaced the flatten and first
   dense layer, as in this section's modern-CNN Discussion table?
   *Provenance:* original (synthesizes this section's own Discussion
   table with the parameter-count formula already derived in
   `channels.md`).
1. [short-code] **Dropping the dense head.** Remove the two hidden
   fully-connected layers (120, 84), connecting the flattened
   convolutional output directly to the 10-way output. Retrain and report
   the accuracy change relative to the original LeNet, and relate the
   result to your prediction in the previous problem.
   *Provenance:* adapted from Nielsen, ch. 6, Exercise 683491 (overlap
   high — same manipulation and comparison; cite on adoption).
1. [short-code] **Overfitting sanity check.** Before trusting a full
   training run, verify that your modernized LeNet (problem 1) can drive
   training accuracy above 99% on a fixed 50-image subset of Fashion-MNIST
   within a small number of epochs. If it cannot, diagnose which
   architectural or optimization choice is responsible.
   *Provenance:* adapted from CS231n Assignment 2's small-dataset
   overfitting sanity check (overlap high — same debugging pattern,
   applied to this section's own model; cite on adoption).
