# Convolutions for Images

Source: `chapter_convolutional-neural-networks/conv-layer.md`, about 2,600 words of prose, 6 exercises (about 900 words).

## Verdict

The first half of this section is the best teaching in the chapter. It opens on a problem the reader can picture (find the vertical edges of a black band), solves it by hand with the two weights $(1, -1)$, then lets gradient descent find the same two numbers from the image and its edge map. Every cell has a stated question and every output is read, and the numbers in the prose match the stored outputs (loss 12.08 at step 0, 0.004 at step 9, learned kernel $(0.9885, -0.9884)$). The second half is a different section: im2col, a banded-matrix paragraph that hands its content to the exercises and to a section in the Image Models part, receptive fields, effective receptive fields, and neurophysiology. The single most important fix is to decide what the section is about. I would end it after "Learning a Kernel" plus the receptive-field count, cut "Convolution as Matrix Multiplication" to one paragraph or move it to an exercise, and show the edge example as pictures instead of printed tensors.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 3 | 4 | 4 | 4 |

## What works

- The opening places the section in one sentence ("The derivation ended with a formula. This section asks what that formula computes on an actual image, and whether a network can find good weights for it by itself.") and then gives the edge example in words before any notation. Model opening.
- The worked $3\times3$ example with all four sums written out, then checked in code: "Does `corr2d` reproduce the four outputs of :numref:`fig_correlation`?"
- The negative control: applying the same kernel to the transposed image and getting zeros. "The output is zero everywhere: $(1, -1)$ detects only vertical edges." A MacKay-style failure case that motivates learning kernels.
- The reading of the training loss: "Before the first step the loss is close to 12: the small random kernel outputs nearly 0 everywhere, so each of the 12 edge positions misses its target $\pm 1$ by about 1." The reader learns to predict a loss before seeing it. Excellent; preserve.
- The closing lesson of the experiment: "In a deep network nobody supplies the intermediate outputs either, and the kernels of every layer are learned from the loss at the end of the network."
- "three $3 \times 3$ layers match the receptive field of one $7 \times 7$ layer with fewer parameters, $27$ weights instead of $49$ per pair of input and output channels, and with three nonlinearities instead of one."
- The closing paragraph names the two open problems (shrinking, single pattern) that the next two sections solve.

## Structural findings

- **Four ideas in one section.** (1) What cross-correlation computes; (2) a kernel can be learned; (3) the operation is a matrix product; (4) stacked layers see further, with a note on effective receptive fields and the visual cortex. Ideas 1 and 2 are one story and are told well. Idea 4 is needed by the next section. Idea 3 serves none of the others and is the weakest writing in the section (CONV-6, CONV-7).
- **A section about images shows no image.** Input and edge map are printed as $6\times8$ and $6\times7$ tensors of 0s, 1s and $-1$s. A reader can find the two columns, but a picture (the band; the map with one bright and one dark line; the transposed band with an empty map) would make the point at a glance, and a real photograph filtered with $(1,-1)$ and its transpose would make it memorable. The only real image in the chapter with kernels and responses is the reproduction from Field (1987), buried in the neurophysiology paragraph. See CONV-3.
- **The `Conv2D` class is built and then abandoned.** See CONV-5.
- **Neurophysiology sits under the heading "Effective Receptive Fields".** See CONV-9.

## Passage-level findings

**CONV-1** (minor, explanation). Under "The Operation":

> Indices start at 0 here, as in code. The layer of
> :numref:`sec_why-conv` indexed the window by offsets from $-\Delta$ to
> $\Delta$ around its center instead; the two conventions differ only by a
> shift of the output by $\Delta$ positions.

Useful, and worth one more clause, since the corner convention also covers kernels of even size and non-square kernels, which the centered one cannot ($2\times2$ in the figure, $1\times2$ for the edge detector). Say so; otherwise a careful reader wonders what $\Delta$ is for the $2\times2$ kernel in the very next figure.

**CONV-2** (minor, explanation). Under "Edge Detection":

> $$\partial_j f(i,j) = \lim_{\epsilon \to 0} \frac{f(i,j+\epsilon) - f(i,j)}{\epsilon},$$

The displayed limit is a definition the reader knows, introduced to justify a sign-flipped approximation ("approximates $-\partial_j f(i,j)$"). It also switches notation from $[\mathbf{X}]_{i,j}$ to $x_{i,j}$ and to a function $f(i,j)$. Fix: one sentence without a display. "This is a finite difference: up to sign, it approximates the derivative of the brightness along the row, which is why the kernel responds only where the brightness changes."

**CONV-3** (moderate, pedagogy). Under "Edge Detection":

> Cross-correlating `X` with `K` gives the edge map `Y`.

The output is a printed tensor. Add a picture: either a small generated figure (image, kernel, edge map; and the transposed image with its empty map), or one line of code that shows `X` and `Y` side by side as images (`d2l.show_images` exists). Better still, apply $(1,-1)$ and its transpose to one real grayscale photograph in a three-panel figure, which is currently exercise 2.5. The Field (1987) figure could then be dropped or moved to the chapter introduction, where Hubel and Wiesel are mentioned.

**CONV-4** (minor, thought). Subsection "Convolution and Cross-Correlation":

> Deep learning libraries
> compute cross-correlations and call them convolutions, and we follow this
> usage from now on.

`why-conv.md` has a subsection with the same title that ends "Deep learning libraries compute cross-correlation and call it convolution, and this book follows that usage." The point is made twice, 600 words apart. The new content here is the numerical instance ($\mathbf{K}'$ with entries 3, 2, 1, 0 gives the same four outputs), which is good. Fix: keep the instance as two sentences at the end of "Implementation" and drop the subsection heading and the repeated conclusion, or shorten the `why-conv.md` subsection (WHY-11) and keep the full treatment here, where there are numbers.

**CONV-5** (moderate, pedagogy). Under "Learning a Kernel":

> Instead of `Conv2D` we use the built-in
> convolutional layer, which computes the same cross-correlation in one fast
> operation.

The reader has just written `Conv2D` with a trainable `weight` and `bias`. It is used once, for a shape check, and never trained. The switch to the built-in layer then brings in the four-dimensional input layout, which is explained only in a code comment ("Layers take (example, channel, height, width): one example, one channel") and which differs by framework. For a $6\times8$ image and ten steps, speed is not a reason. Fix: train the reader's own `Conv2D((1, 2))` in the experiment (`corr2d` is differentiable as written in the PyTorch, TensorFlow, and JAX tabs; the MXNet tab assigns in place and may need the stacked form the TensorFlow tab uses), then say in one sentence that the built-in layer computes the same thing fast and expects a batch and a channel axis, and show that it agrees. The layout convention then gets a sentence of prose where it first matters, instead of a comment.

**CONV-6** (moderate, thought). Under "Convolution as Matrix Multiplication":

> This rewriting is one way to compute convolutions fast. GPUs and other
> accelerators are built around dense matrix multiplication, and libraries often
> lower convolutions to explicit or implicit matrix products; depending on the
> shapes and the hardware they may also choose other algorithms.

A paragraph of hedges ("one way", "often", "explicit or implicit", "may also choose other algorithms") that leaves the reader with no fact to hold. The preceding paragraph already said what matters: every output is a dot product of the flattened kernel with a flattened patch, so the whole layer is one matrix product. Fix: cut this paragraph to its one concrete sentence ("The explicit form costs memory, since each input element is copied into up to $k_\textrm{h} k_\textrm{w}$ rows.") and go to the code.

**CONV-7** (moderate, explanation). Under "Convolution as Matrix Multiplication", last paragraph:

> The same idea also works in the opposite direction. Instead of unfolding the
> input, we can unroll the kernel into a sparse, banded matrix $\mathbf{W}$
> that multiplies the flattened image; the exercises explore this view.

In six lines this paragraph introduces a second matrix form, asserts what the backward pass is in terms of $\mathbf{W}^\top$, defers both to exercises ("the exercises explore this view", "an exercise below shows"), and points to the transposed convolution in the Image Models part. Nothing is shown. Yet the banded matrix is an object the reader has *already seen*: it is panel (c) of `fig_why_conv_weight_sharing` in the previous section. The text does not make the connection. Fix: either cut the paragraph (the exercise can introduce the banded matrix itself, as it already does), or replace it by two sentences that close the loop: "Unrolling the kernel instead of the input gives a second matrix form: the banded matrix of :numref:`fig_why_conv_weight_sharing`(c), in which every row holds the same few weights. A convolution is a fully connected layer whose weight matrix has this pattern, which is where the derivation of :numref:`sec_why-conv` started." Later exercises (`padding-and-strides.md` exercise 3) refer to "the banded-matrix form of :numref:`subsec_conv_matmul`", so if the paragraph is cut that reference must point to the exercise.

**CONV-8** (minor, pedagogy). PyTorch tab under "Convolution as Matrix Multiplication":

> PyTorch exposes this rearrangement directly:
> `F.unfold` extracts the patch matrix

A one-framework aside with its own cell, in a part of the section that is already a detour. The other three tabs have nothing here. Cut, or move to a code comment in the patch-matrix cell.

**CONV-9** (moderate, thought). Under "Effective Receptive Fields":

> Receptive fields take their name from neurophysiology. Recordings from the
> visual cortex of several animal species

A paragraph on Hubel and Wiesel, Field's filters, and deep-layer correspondences with the visual system, placed under a heading about gradient-measured receptive fields. The chapter introduction has told the Hubel and Wiesel story already. The paragraph ends well ("The analogy motivated the design. It does not show that a particular network works like the brain."), but it does not belong here. Fix: move one sentence on the origin of the name to where "receptive field" is defined ("The term comes from neurophysiology, where it names the part of the visual field to which a neuron responds."), and cut the rest or move it, with the Field figure, to the chapter introduction's history paragraph.

**CONV-10** (minor, thought). Under "Effective Receptive Fields":

> Larger kernels and more layers enlarge the set of possible
> inputs; they do not guarantee that the network uses all of it equally.

The subsection is four sentences on a cited result with no demonstration, and two of the six exercises (4 and 6) are built on it. Either show it (the exercise 4 experiment is ten lines and produces a striking picture: a $7\times7$ support with nearly all the gradient mass in the middle) or reduce it to one sentence at the end of "Stacked Layers" and keep one exercise.

**CONV-11** (minor, explanation). Under "Feature Maps and Receptive Fields":

> some layer, its *receptive field* is the set of elements of all previous
> layers that can affect $x$ during forward propagation.

"Of all previous layers" makes the receptive field a union across layers. Every later use (and `padding-and-strides.md`: "the set of input pixels that can affect its value"; `pooling.md`: "the input region that can influence it") means the region of the *input*. Fix: define it as the set of input pixels that can affect $x$, and say that the same notion applies relative to any earlier layer. Then the three sections agree.

## Exercises

Six problems, about 900 words, on the heavy side for the section.

- Exercise 1 (diagonal edges, transpose identity): good; the second part explains the zero output of the transposed-image cell.
- Exercise 2 (hand-designed kernels): five parts of mixed weight. Part 3 ("What is the minimum size of a kernel to obtain a derivative of order $d$?") is fine; part 5 (apply three kernels to a real image) is the picture the section itself should show (CONV-3). If the text adopts it, shorten the exercise to parts 1, 2, 4.
- Exercise 3 (banded matrix): good, and part 4 (the locally connected layer, with a three-way parameter count) is a valuable link back to the derivation. It carries the content that CONV-7 complains the text only gestures at.
- Exercises 4 and 6 (effective receptive field, and its width against depth): two problems on a four-sentence subsection. Keep 6 as the marked hard problem with 4 as its first part, or keep 4 only.
- Exercise 5 (backward pass): a classic, well specified. The single-precision finite-difference study in part 2 is more about numerics than convolution; one step size would do.

## Still open from October

Closed: the grammar errors, and "Padding and stride are used one section before they are defined" (padding is now defined in the sentence that mentions it; stride is no longer used here). The three $3\times3$ against one $7\times7$ comparison, which October asked to keep, is kept.
