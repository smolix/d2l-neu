# Writing pass: `chapter_convolutional-neural-networks/conv-layer.md`

Prose words (checker count, exercises included): 3280 before, 3261 after
(-19). One unlabelled display equation removed (the derivative limit).
Checker: "OK: only prose changed". `tools/lint_source.py`: clean.

## Before/after pairs

1. Before: "This section asks what that formula computes on an actual image,
   and whether a network can find good weights for it by itself."
   After: "What does that formula compute on an actual image, and can a
   network find good weights for it by itself?"

2. Before: "The mathematical *convolution* of $\mathbf{X}$ with $\mathbf{K}$
   reads the input at $(i-a, j-b)$ instead of $(i+a, j+b)$, which amounts to
   flipping the kernel horizontally and vertically (:numref:`sec_why-conv`).
   [...] Deep learning libraries compute cross-correlations and call them
   convolutions, and we follow this usage from now on."
   After: "Strictly speaking, `corr2d` does not compute a convolution. The
   mathematical *convolution* flips the kernel horizontally and vertically
   before sliding it (:numref:`sec_why-conv`). The numbers show how little is
   at stake. [...] From now on we say convolution for both, as the libraries
   do."

3. Before: "The number of parameters does not depend on the input."
   After: "Unlike a fully connected layer, this layer is not tied to one
   input size, since its parameters belong to the kernel alone."

4. Before: "This is a *finite difference*. For a smooth function $f(i, j)$ of
   the position, the derivative along the horizontal axis is [display] and
   taking $\epsilon = 1$ shows that $x_{i,j} - x_{i,j+1}$ approximates
   $-\partial_j f(i,j)$. The kernel thus responds to changes of intensity
   along each row."
   After: "This is a *finite difference*: up to sign, it approximates the
   derivative of the brightness along the row. A derivative vanishes where
   nothing changes, which is why the kernel responds only where the
   brightness does."

5. Before: "Designing every kernel by hand is impractical for larger kernels
   and for stacks of layers, where we do not know which patterns the
   intermediate layers should detect. Can gradient descent find the kernel
   from the input `X` and the desired output `Y` alone?"
   After: "We cannot design all of these by hand, least of all for the
   intermediate layers of a stack, where we do not even know which patterns
   to ask for. Can gradient descent find a kernel from the input `X` and the
   desired output `Y` alone? Let us try it on the one kernel whose answer we
   know."

6. Before: "The two loops of `corr2d` visit one output position at a time,
   which is far too slow for real images and networks."
   After: "The two loops of `corr2d` are easy to read and far too slow for
   real images and networks, since they visit one output position at a time."

7. Before: "Larger kernels and more layers enlarge the set of possible
   inputs; they do not guarantee that the network uses all of it equally."
   After: "A unit at the top of a deep stack can see far, but it looks mostly
   at the middle."

8. Before: "We can now compute a convolutional layer, train its kernel, and
   count how far a stack of layers sees. Two limits remain: every layer
   shrinks the image [...]"
   After: "Two defects of the layer are left to repair. Every layer shrinks
   the image [...] :numref:`sec_padding` takes up the first and
   :numref:`sec_channels` the second."

## Repetition removed

- Cross-correlation against convolution (owned by `why-conv.md`): the
  re-explanation of the index flip and the repeated "libraries compute
  cross-correlations and call them convolutions" are reduced to a clause with
  a reference; the numerical instance ($\mathbf{K}'$ gives 19, 25, 37, 43),
  which is new here, stays. Heading kept.
- Feature map (owned by `why-conv.md`, `subsec_why-conv-channels`): the
  definition is reduced to "is a feature map (ref); the edge map `Y` is one".
- "Nobody knows which kernels to design" was told in the opening, at the end
  of "Edge Detection" and at the start of "Learning a Kernel"; the third
  telling now leans on the second ("We cannot design all of these by hand").
- $\mathcal{N}(0, 0.01^2)$ is stated once ("The Layer"); "Learning a Kernel"
  says "drawn as before".
- The formulaic recap in the closing paragraph ("We can now compute ...") is
  gone; the Summary is the only recap.
- The sentence restating the signature of `corr2d` (the code shows it) is cut.

## Review findings

Applied (prose): CONV-1 (corner anchoring also serves kernels without a
center), CONV-2 (limit display replaced by one sentence; notation
$[\mathbf{X}]_{i,j}$ throughout), CONV-4, CONV-6 (hedge paragraph cut to the
memory sentence), CONV-7 (banded matrix tied to panel (c) of
`fig_why_conv_weight_sharing`; the $\mathbf{W}^\top$ backward-pass assertion
and the forward reference to `sec_transposed_conv` removed), CONV-8 (tab
prose trimmed to one sentence; the cell stays), CONV-11 (receptive field =
set of input pixels; relative to an earlier layer as a second sentence).
CONV-5 in part: the four-dimensional input layout now has a sentence of prose.
CONV-9 was already done (figure moved to the introduction); the body cites
`field_visual` by number.

Deferred:
- CONV-3: needs a figure or cell (edge map and a photograph as images).
- CONV-5 (main part): training the reader's own `Conv2D` changes code cells.
- CONV-8 (cutting the `F.unfold` cell): code change.
- CONV-10: showing the effective receptive field needs a new cell; the
  subsection keeps its heading and four sentences.
- Structural "four ideas in one section" / cutting "Convolution as Matrix
  Multiplication" to a paragraph: the check cells would have to go.
- Exercise merges (4 into 6; trimming 2.5 and 5.2): exercises not touched.

## Factual claims added or changed

- Corner anchoring "also serves kernels that have no center, such as the
  $2 \times 2$ kernel [...] and the $1 \times 2$ edge detector" (new).
- "Unlike a fully connected layer, this layer is not tied to one input size"
  (new comparison).
- The built-in layer "expects two more axes, one for the examples of a
  minibatch and one for channels" (new in prose; was a code comment; true of
  all four tabs, whose axis orders differ).
- "A standard remedy is to turn the whole computation into one large matrix
  product, the operation that GPUs execute best" replaces "Fast
  implementations turn [...]" and the paragraph saying libraries may choose
  other algorithms. The caveat about other algorithms is no longer stated.
- The banded matrix $\mathbf{W}$ "is the two-dimensional counterpart of panel
  (c) of `fig_why_conv_weight_sharing`" and "a convolution is a fully
  connected layer whose weight matrix has this pattern" (new link).
- Removed: the gradient with respect to the input is $\mathbf{W}^\top$ times
  the flattened $\partial\ell/\partial\mathbf{Y}$; the pointer to the
  transposed convolution.
- "Admittedly, we made its task easy by supplying the edge map" (new
  judgment).
- "This slow, linear growth is no reason to prefer large kernels" replaces
  "Deep stacks of small kernels can therefore replace single large ones".
- The count $r$ "says nothing about how much each of them does"; "it looks
  mostly at the middle" (rewording of the Luo et al. result).
- Receptive field redefined as the set of input pixels (was "elements of all
  previous layers").
- Summary: "which accelerators execute efficiently" (was "which is how
  accelerators compute it efficiently").

No number describing an output was changed. Checked against all four stores:
initial loss 11.7 to 12.1, loss 0.004 at step 9, weights within 0.012 of
$(1, -1)$.

## Slides that now differ from the body

No slide contradicts the body. Differences of wording only:
- "The Kernel (1, −1) Is a Finite Difference" and the opening slide write
  $x_{i,j} - x_{i,j+1} \approx -\partial_j f(i, j)$; the body now uses
  $[\mathbf{X}]_{i,j}$ and no longer introduces $f$ or $\partial_j$.
- "Cross-Correlation Is One Matrix Product" says "the price is up to
  $k_\textrm{h} k_\textrm{w}$ copies" (economic metaphor the body avoids).
- "F.unfold Extracts the Same Patch Matrix": "Convolution = unfold, matrix
  product, reshape"; the body no longer says this in words.
- The convolution slide's note "Libraries compute cross-correlation and call
  it convolution" is now only a clause in the body ("as the libraries do").
- The slides have no counterpart of the new banded-matrix link.
