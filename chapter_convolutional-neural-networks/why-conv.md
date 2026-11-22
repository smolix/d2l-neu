# From Fully Connected Layers to Convolutions
:label:`sec_why-conv`

The multilayer perceptrons of :numref:`chap_perceptrons` flatten every image
into an unordered vector of numbers—appropriate for tabular data, but wasteful
for images because flattening discards which pixels sit next to one another.
On a one-megapixel ($1000 \times 1000$) grayscale photograph, even a modest
1,000-unit hidden layer demands $10^6 \times 10^3 = 10^9$ weights (4 GB in
fp32), and a full-resolution $1000 \times 1000$ hidden map demands $10^{12}$
weights! Yet natural images are nothing like arbitrary million-dimensional
vectors: neighboring pixels are strongly correlated (*locality*), and a visual
feature such as a whisker carries the same meaning wherever it appears in the
frame (*translation equivariance*).

What happens if we impose those two geometric principles directly on a dense
layer mapping a 2D pixel grid to a 2D hidden grid? Translation equivariance
alone slashes $10^{12}$ weights down to roughly $4 \times 10^6$, and restricting
each hidden unit to a small local neighborhood cuts them further to a few
hundred. What survives both constraints is a *convolutional layer*.

## Equivariance and Invariance

The children's game "Where's Waldo" (:numref:`img_waldo`) captures the exact
challenge of visual perception. Each busy scene teems with distractions, with
Waldo hiding somewhere unexpected—yet *what Waldo looks like* does not depend on
*where he stands*. We can therefore sweep a single Waldo detector across the
image, scoring every patch by how strongly it matches Waldo using the exact same
rule at every position. Modern object detection
:cite:`Girshick.Donahue.Darrell.ea.2014` and semantic segmentation
:cite:`Long.Shelhamer.Darrell.2015` pipelines build directly on this idea.

![Can you find Waldo (image courtesy of William Murphy (Infomatique))?](../img/waldo-football.jpg)
:width:`400px`
:label:`img_waldo`

This intuition points to two guiding principles for the early layers of a vision
network:

1. They should respond identically to the same local patch regardless of where
   it appears, so shifting the input shifts the output by the same amount.
1. They should attend only to small local neighborhoods without worrying about
   distant parts of the image—the *locality* principle.

Subsequent layers can then compose these local responses into progressively
larger, more abstract patterns, culminating in a whole-image prediction.

To formalize the first principle, let $T_v$ denote the shift operator that
translates an image by an offset $v = (v_1, v_2)$, moving pixel $(i, j)$ to
$(i + v_1, j + v_2)$ so that
$[T_v \mathbf{X}]_{i,j} = [\mathbf{X}]_{i-v_1, j-v_2}$. An image-to-image
mapping $f$ is *translation equivariant* if shifting its input shifts its output
by the exact same vector, whereas an image-to-label mapping is *translation
invariant* if shifting its input leaves its output completely unchanged:

$$
\begin{aligned} f(T_v \mathbf{X}) &= T_v f(\mathbf{X}) && \textrm{(equivariance)},\\ f(T_v \mathbf{X}) &= f(\mathbf{X}) && \textrm{(invariance)}.\end{aligned}
$$
:eqlabel:`eq_equivariance_invariance`

:numref:`fig_why_conv_equivariance` illustrates translation equivariance as a
commuting diagram. Here $f$ matches a small template against every window of the
image—the exact operation we are about to derive. Shifting first and matching
second produces the identical map as matching first and shifting second.

![Translation equivariance. The input holds a $2\times 2$ patch of ones, and $f$ compares every $2\times 2$ window with that patch, so the response is 4 where a window covers the whole patch. Shifting the input by $v = (3, 2)$ and then applying $f$ (bottom row) gives the same map as applying $f$ and then shifting the response (right column).](../img/why-conv-equivariance.svg)
:label:`fig_why_conv_equivariance`

Notice the clean division of labor: *equivariance* preserves *where* a pattern
appears, whereas *invariance* discards location. A Waldo detector should be
equivariant so its score map tracks Waldo across the scene; answering the
yes-or-no question "is Waldo in this picture?" should be invariant, achieved at
the end by aggregating the equivariant map across space (for instance, taking
its maximum).


## Constraints on the Fully Connected Layer

Both constraints become crystal clear in one dimension, where we can draw the
entire weight matrix. Consider an eight-sample signal $x_0, \ldots, x_7$ mapped
by a fully connected layer to eight hidden units,
$h_i = u_i + \sum_k W_{i,k} x_k$. Its $8 \times 8$ weight matrix has 64
independent entries, shown in panel (a) of :numref:`fig_why_conv_weight_sharing`.

![Weights of a linear layer that maps a one-dimensional signal of 8 samples (columns) to 8 hidden units (rows). Equal colors are equal weights; white is zero. (a) Fully connected: every weight is free. (b) Translation equivariant: a weight depends only on the offset between input and output, so the matrix is constant along each diagonal. (c) Equivariant and local: only the offsets $-1$, $0$, and $1$ have nonzero weights.](../img/why-conv-weight-sharing.svg)
:label:`fig_why_conv_weight_sharing`

What does translation equivariance impose on $\mathbf{W}$? Ignoring biases for a
moment (our proof below shows all biases must coincide), feed in a unit impulse
with a single 1 at sample $k$: the layer outputs column $k$ of $\mathbf{W}$.
Shift that impulse to $k + 1$, and the layer outputs column $k + 1$.
Equivariance insists that column $k + 1$ must equal column $k$ shifted down by
one row, i.e., $W_{i+1, k+1} = W_{i, k}$. Consequently, $\mathbf{W}$ must be
constant along every diagonal (a *Toeplitz* matrix), as in panel (b). Each
diagonal corresponds to a single relative offset $a = k - i$, collapsing 64 free
weights down to just 15 (offsets $-7$ through $7$).

Locality prunes almost all of those remaining 15 weights. Restricting each
hidden unit to look only at neighbors within distance 1 zeroes out every offset
outside $\{-1, 0, 1\}$, leaving just 3 shared weights repeated across every row
in panel (c). The layer now sweeps one tiny 3-weight template across the entire
signal. That is the entire derivation in miniature; moving to 2D images simply
adds a second spatial index.

For images, both the input $\mathbf{X}$ and the hidden representation
$\mathbf{H}$ are two-dimensional grids, with input pixel $[\mathbf{X}]_{k,l}$ at
$(k, l)$ and hidden unit $[\mathbf{H}]_{i,j}$ at $(i, j)$. Connecting every
hidden unit to every pixel turns the weights into a fourth-order tensor
$\mathsf{W}$ alongside a bias matrix $\mathbf{U}$:

$$[\mathbf{H}]_{i, j} = [\mathbf{U}]_{i, j} + \sum_k \sum_l [\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}.$$

Because both constraints depend on where a pixel sits *relative* to hidden unit
$(i, j)$, let's re-index by offsets $a = k - i$ and $b = l - j$, defining
$[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$:

$$[\mathbf{H}]_{i, j} = [\mathbf{U}]_{i, j} + \sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_mlp_fourth_order`

Here $a$ and $b$ span both positive and negative offsets. This re-indexing is
purely cosmetic—we are simply reading $\mathsf{W}$ along its diagonals without
reducing parameters yet. On a $1000 \times 1000$ image mapped to a
$1000 \times 1000$ hidden grid, this single layer still demands
$10^6 \times 10^6 = 10^{12}$ weights.

### Translation Equivariance

Just as in one dimension, a layer of the form :eqref:`eq_mlp_fourth_order` is
translation equivariant if and only if its weights and bias are independent of
location $(i, j)$. Checking that shared weights yield equivariance takes one
line; the converse is what makes convolution unavoidable—equivariance leaves no
other linear option.

To handle boundaries cleanly (where a shift pushes pixels off the edge of a
finite frame), we embed the image on an infinite grid padded with zeros outside
its support so no information ever leaves the grid. On finite images,
equivariance holds strictly away from borders, a detail we address in
:numref:`sec_padding`.

**Proposition.** Let the input (with finitely many nonzero pixels) and the
hidden representation live on the infinite grid. The layer
:eqref:`eq_mlp_fourth_order` is translation equivariant for every shift $v$ if
and only if there exist a scalar $u$ and a matrix $\mathbf{K}$ such that
$[\mathbf{U}]_{i,j} = u$ and $[\mathsf{V}]_{i,j,a,b} = [\mathbf{K}]_{a,b}$ for
all $i, j, a, b$.

**Proof.** If neither weights nor bias depend on $(i, j)$, then shifting the
input by $v$ directly shifts the output by $v$:

$$
\begin{aligned}
[f(T_v \mathbf{X})]_{i,j} &= u + \sum_a \sum_b [\mathbf{K}]_{a,b} [\mathbf{X}]_{i - v_1 + a, j - v_2 + b} \\
&= [f(\mathbf{X})]_{i - v_1, j - v_2} = [T_v f(\mathbf{X})]_{i,j}.
\end{aligned}
$$

Conversely, assume the layer is equivariant. The all-zero image is invariant
under every shift and maps to $\mathbf{U}$, forcing $\mathbf{U} = T_v \mathbf{U}$
for all $v$, so the bias must be a constant $u$. Subtracting $u$ from both sides
shows that the linear part alone is also equivariant. Feed in an image with a
single 1 at $(k, l)$ and zeros elsewhere: its response at $(i, j)$ is
$[\mathsf{W}]_{i,j,k,l}$. Shifting that impulse by $v$ moves it to
$(k + v_1, l + v_2)$, and equivariance requires
$[\mathsf{W}]_{i, j, k+v_1, l+v_2} = [\mathsf{W}]_{i-v_1, j-v_2, k, l}$.
Setting $(k, l) = (a, b)$ and $v = (i, j)$ yields
$[\mathsf{V}]_{i,j,a,b} = [\mathsf{W}]_{i,j,i+a,j+b} = [\mathsf{W}]_{0,0,a,b}$,
which is independent of $(i, j)$. $\blacksquare$

In short, probing the layer with a single bright pixel forces every location to
share the exact same weights, collapsing the layer to

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{K}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$

Every hidden unit now applies the same pattern $\mathbf{K}$ centered at its own
coordinates. On a $1000 \times 1000$ zero-padded image, only offsets
$a, b \in [-999, 999]$ ever touch a nonzero pixel, slashing $10^{12}$ weights
down to $1999^2 \approx 4 \times 10^6$.

### Locality

Four million weights for a single filter is still far too many, and most of
those weights link a hidden unit to pixels clear across the image. Enforcing
*locality*—requiring $[\mathbf{H}]_{i,j}$ to depend only on pixels within
distance $\Delta$ along each axis—sets $[\mathbf{K}]_{a, b} = 0$ whenever
$|a| > \Delta$ or $|b| > \Delta$, giving

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{K}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_conv-layer`

Each hidden unit now computes a weighted sum over a
$(2\Delta+1) \times (2\Delta+1)$ local window centered at $(i, j)$, using just
$(2\Delta+1)^2$ shared weights and one bias: 9 weights for $\Delta = 1$ (a
$3 \times 3$ window), or 121 for $\Delta = 5$. Because $\Delta$ is typically
well below 10, a few dozen or hundred weights replace $4 \times 10^6$ while
preserving the full spatial resolution of the hidden map.

Equation :eqref:`eq_conv-layer` defines a *convolutional layer*, the building
block of *convolutional neural networks* (CNNs). The shared matrix $\mathbf{K}$
is called the *convolution kernel* or *filter*. Early pioneers of spatial and
temporal weight sharing included time-delay neural networks for speech
recognition :cite:`Waibel.Hanazawa.Hinton.ea.1989` and the shift-invariant
networks of :citet:`Zhang.ea.1988`.

Crucially, the parameter count shrank from $10^{12}$ to $(2\Delta + 1)^2$ by
restricting the hypothesis class to local, translation-equivariant mappings.
This built-in *inductive bias* (:numref:`sec_generalization_deep`) dramatically
improves sample efficiency whenever features are local and position-independent
(such as edges and textures), though a single layer cannot distinguish identical
patches by absolute position alone (e.g., an eye near the top of a passport
photo versus a nostril lower down). Stacking multiple layers with nonlinearities
allows deeper units to integrate context across progressively wider regions.

## Convolution and Cross-Correlation

Strictly speaking, the name "convolution" in deep learning is a slight misnomer.
In mathematics, the discrete *convolution* of two 1D functions is
$(f * g)(i) = \sum_a f(a) g(i-a)$ (with an integral on $\mathbb{R}^d$),
measuring the overlap between $f$ and a *flipped*, shifted copy of $g$. In two
dimensions,

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$
:eqlabel:`eq_2d-conv-discrete`

Compare this with :eqref:`eq_conv-layer`, which indexes $\mathbf{X}$ at
$(i + a, j + b)$ rather than $(i - a, j - b)$. Substituting $-a$ for $a$ and
$-b$ for $b$ reveals that :eqref:`eq_conv-layer` is the mathematical
convolution of $\mathbf{X}$ with the flipped kernel
$[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$:

$$\sum_{a}\sum_{b} [\mathbf{K}]_{a, b}  [\mathbf{X}]_{i+a, j+b} = \sum_{a}\sum_{b} [\mathbf{K}']_{a, b}  [\mathbf{X}]_{i-a, j-b} = (\mathbf{K}' * \mathbf{X})(i, j).$$

The unflipped operation is properly called *cross-correlation*. Because CNN
kernels are learned from data, the distinction has no effect on expressive
power: a layer that cross-correlates with $\mathbf{K}$ learns the exact same
family of functions as one that convolves with $\mathbf{K}'$. Deep learning
frameworks compute cross-correlation and call it convolution, and we follow
that standard convention throughout this book.


## Channels
:label:`subsec_why-conv-channels`

So far we treated both inputs and hidden maps as single grayscale grids. Real
color photographs have three intensity values per pixel—red, green, and blue—so
an image is a third-order tensor $\mathsf{X}$ of shape
$3 \times \textrm{height} \times \textrm{width}$ with entries
$[\mathsf{X}]_{c,i,j}$. The leading *channel* axis attaches a feature vector to
every spatial pixel, and a kernel sums over both the local spatial window and
all input channels.

Just as importantly, a single kernel produces only a single response map—like
the Waldo score map in :numref:`fig_waldo_mask`—capturing just one visual
pattern. Rich visual representations require detecting many patterns in
parallel (oriented edges, corners, color contrasts, textures), so the hidden
representation $\mathsf{H}$ also becomes a third-order tensor stacked along an
*output channel* axis. Each output channel $o$ has its own learned kernel slice
and bias $u_o$, turning $\mathsf{K}$ into a fourth-order tensor:

$$[\mathsf{H}]_{o,i,j} = u_o + \sum_c \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathsf{K}]_{o, c, a, b} [\mathsf{X}]_{c, i+a, j+b}.$$
:eqlabel:`eq_conv-layer-channels`

Here $c$ indexes input channels and $o$ indexes output channels; we keep this
output-before-input channel convention throughout the chapter. Because each
output channel maps where a particular learned feature occurs across the grid,
hidden channels are also called *feature maps*, and the entire stack feeds into
the next layer. With $c_\textrm{i}$ input channels and $c_\textrm{o}$ output
channels, the layer requires $(2\Delta+1)^2 c_\textrm{i} c_\textrm{o}$ weights
and $c_\textrm{o}$ biases. A $3 \times 3$ layer mapping an RGB image to 64
feature maps therefore has just $9 \cdot 3 \cdot 64 + 64 = 1{,}792$
parameters—completely independent of the image's resolution!

![A Waldo detector: the response of a kernel that matches Waldo peaks at his location (image courtesy of William Murphy (Infomatique)).](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`

Nothing in this derivation is tied to two dimensions: the exact same logic
yields 1D convolutions sliding across audio samples or text tokens, and 3D
convolutions sliding across video frames or volumetric medical scans.

Equation :eqref:`eq_conv-layer-channels` is the foundation for the rest of this
chapter. Because we derived it on an idealized infinite grid, three practical
issues remain: handling finite image borders, expanding how much of the image
deeper units can see, and aggregating spatial feature maps into a single
image-level decision (like whether Waldo appears *anywhere* in the scene).
First, however, let's bring :eqref:`eq_conv-layer` to life in code.

## Summary

Flattening an image into an unstructured vector makes dense layers intractably
large—$10^{12}$ weights to map a megapixel image to an equal-sized hidden grid.
Two natural assumptions about images collapse that count to a few hundred:
*translation equivariance* forces every spatial position to share one kernel
(as proved on the infinite grid), and *locality* restricts that kernel to a
small window. The resulting layer computes cross-correlation (universally called
convolution in deep learning), and stacking multiple input and output channels
lets a single layer detect many visual patterns simultaneously while keeping
parameter counts independent of image size. Those gains stem directly from
inductive bias: a convolutional layer trades away arbitrary position-dependent
mappings for strong generalization on spatially structured data.

## Exercises

1. **Weight sharing on a ring.** Repeat the argument of
   :numref:`fig_why_conv_weight_sharing` for a signal of four samples on a
   periodic grid, where the sample after the last one is the first one
   again, so that a shift loses nothing.
    1. Feed the layer a single 1 at each sample in turn and write down the
       $4 \times 4$ weight matrix of a translation-equivariant layer. How
       many free weights does it have? Compare with the
       $2 \cdot 4 - 1 = 7$ offsets of four samples surrounded by zeros.
    1. Impose locality with offsets $-1$, $0$, and $1$. Which entries of
       the matrix are nonzero, and which of them have no counterpart in
       panel (c) of the figure?
2. **Commutativity of convolution.**
    1. Prove that the convolution of :eqref:`eq_2d-conv-discrete` is
       commutative, i.e., $f * g = g * f$.
    1. Define the cross-correlation
       $(f \star g)(i, j) = \sum_a \sum_b f(i+a, j+b)\, g(a, b)$, the
       operation of :eqref:`eq_conv-layer` with input $f$ and kernel $g$. Express $(g \star f)(i, j)$
       through values of $f \star g$, and decide from this expression
       whether cross-correlation is commutative.
3. **Convolutions for audio.** Audio data is often represented as a
   one-dimensional sequence of samples.
    1. Derive the one-dimensional analog of :eqref:`eq_conv-layer`.
    1. Name one audio task for which a layer that is local and
       translation equivariant in time is a good fit, and one for which
       it is not. Justify each choice.
    1. A spectrogram represents a recording as a two-dimensional array
       whose axes are time and frequency, with frequency on a linear
       scale in hertz. For each axis, state whether translation
       equivariance is a reasonable assumption, and justify your answer
       from what a shift along that axis does to the sound. Then answer
       the same question for a frequency axis on a logarithmic scale.
4. **Equivariance versus invariance.** A convolutional classifier computes
   feature maps $h(\mathsf{X})$ with stacked layers of
   :eqref:`eq_conv-layer-channels` and nonlinearities in between, and then
   a prediction $g(h(\mathsf{X}))$ with a head $g$.
    1. Assume a periodic grid, which wraps around so that the column after
       the last one is the first one again; on it, $h$ is translation
       equivariant by the argument of the proposition. Let $g$ average each channel of $h(\mathsf{X})$ over all locations.
       Using :eqref:`eq_equivariance_invariance`, show that the prediction
       is translation invariant.
    1. An image shows a single digit in one of the nine cells of a
       $3 \times 3$ grid, and the task is to name the cell. Explain why
       the classifier of the first part cannot solve this task. Describe
       a head $g$ that solves it from the same feature maps
       $h(\mathsf{X})$.
5. **Rotation equivariance.** ● The proposition of this section derived
   weight sharing from equivariance to shifts. Ask the same of rotations.
   On the infinite grid of the proposition, let $R$ rotate an image by
   $90^\circ$ about the origin,
   $[R\mathbf{X}]_{i,j} = [\mathbf{X}]_{-j,i}$.
    1. Show that the convolutional layer :eqref:`eq_conv-layer` satisfies
       $f(R\mathbf{X}) = R f(\mathbf{X})$ for every image $\mathbf{X}$ if
       and only if $[\mathbf{K}]_{a,b} = [\mathbf{K}]_{-b,a}$ for all
       offsets $(a, b)$.
    1. For $\Delta = 1$, how many free weights does such a kernel have?
       Show that its response to an image does not change when the image
       is rotated by $90^\circ$ about the pixel being evaluated, so that it
       cannot tell a vertical edge from a horizontal one.
    1. Instead of constraining one kernel, use four output channels whose
       kernels are copies of one unconstrained kernel $\mathbf{K}$, rotated
       by $0^\circ$, $90^\circ$, $180^\circ$, and $270^\circ$. Show that
       rotating the input rotates every output map and permutes the four
       channels cyclically. What does this layer keep that the constrained
       kernel of the first part loses?

[Discussions](https://d2l.discourse.group/t/64)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §6.1]{.kicker}

From Fully Connected Layers to Convolutions<br>**translation equivariance · locality · weight sharing · cross-correlation · channels**
:::
:::

::: {.slide title="Why Dense Layers Fail on High-Resolution Images"}
[The Scaling Bottleneck]{.kicker}

::: {.cols .vc}
::: {.col}
- **All-to-all quadratic explosion:** Mapping a $1000 \times 1000$ image to an equal-sized hidden grid needs $10^{12}$ weights (**4 TB** grayscale, **36 TB** RGB).
- **Even narrow projections fail:** Projecting a 1MP RGB frame to just 300 hidden units demands $9 \times 10^8$ weights (**3.6 GB**).
- **Root cause:** Dense layers assign separate weights to every input–output coordinate pair and ignore 2D spatial proximity.
:::

::: {.col .fig .big}
@fig:ch06-cnn-dense-param-explosion
:::
:::
:::

::: {.slide title="Visual Perception Relies on Two Spatial Symmetries"}
[The "Where's Waldo" Intuition]{.kicker}

Finding a target pattern in a cluttered visual scene illustrates the two geometric principles that govern early vision layers:

::: {.cols .vc}
::: {.col}
- **Position-independent appearance:** What Waldo looks like does not depend on where he stands, so a single local template can sweep across every patch of the scene.
- **Translation equivariance:** Shifting the input image shifts the detector's response map by the exact same offset.
- **Locality:** Early feature detectors need only inspect a small neighborhood around each pixel, leaving deeper layers to aggregate global context.
:::

::: {.col .fig}
![](../img/waldo-football.jpg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Translation Equivariance vs. Translation Invariance"}
[Formalizing Spatial Symmetries]{.kicker}

::: {.cols .vc}
::: {.col}
- **Shift operator $T_v$:** Translates an image by $v = (v_1, v_2)$ so $[T_v \mathbf{X}]_{i,j} = [\mathbf{X}]_{i-v_1, j-v_2}$.
- **Equivariance ($f(T_v \mathbf{X}) = T_v f(\mathbf{X})$):** Feature maps shift with the input, preserving *where* patterns occur for localization.
- **Invariance ($f(T_v \mathbf{X}) = f(\mathbf{X})$):** Whole-image classifiers pool equivariant maps across space so the final label is shift-independent.
:::

::: {.col .fig .big}
@fig:ch06-cnn-equivariance-vs-invariance
:::
:::
:::

::: {.slide title="Equivariance as a Commuting Diagram"}
[Shift Then Filter Equals Filter Then Shift]{.kicker}

When $f$ scores every $2 \times 2$ window against a $2 \times 2$ patch of ones, both paths around the diagram produce the identical output grid:

::: {.cols .vc}
::: {.col .fig}
![](../img/why-conv-equivariance.svg){.nostretch width=100%}
:::

::: {.col}
::: {.d2l-note .rule}
**Commutation invariant:** Shifting the input by $v = (3, 2)$ and then applying the filter $f$ (bottom path) yields the exact same response map as applying $f$ first and shifting the output by $v$ (right path).
:::
:::
:::
:::

::: {.slide title="1D Intuition: From 64 Weights to 15, Then to 3"}
[Toeplitz Structure & Locality in 1D]{.kicker}

On an 8-sample 1D signal mapped to 8 hidden units ($h_i = u_i + \sum_k W_{i,k} x_k$), each geometric constraint prunes the $8 \times 8$ weight matrix $\mathbf{W}$:

![](../img/why-conv-weight-sharing.svg){.nostretch width=46% fig-align="center"}

- **(a) Fully connected (64 weights):** Every entry $W_{i,k}$ is independent.
- **(b) Equivariant / Toeplitz (15 weights):** Shifting a unit impulse at $k$ to $k+1$ shifts the output down one row ($W_{i+1,k+1} = W_{i,k}$), making $\mathbf{W}$ constant along each diagonal $a = k - i \in \{-7,\dots,7\}$.
- **(c) Equivariant + Local (3 weights):** Restricting each unit to neighbors within distance $\Delta = 1$ zeroes out all diagonals with $|a| > 1$, leaving one 3-tap kernel swept across every row.
:::

::: {.slide title="Re-Indexing a 2D Dense Layer by Relative Offsets"}
[From 4D Weight Tensors to Spatial Offsets]{.kicker}

::: {.cols .vc}
::: {.col}
$$[\mathbf{H}]_{i, j} = [\mathbf{U}]_{i, j} + \sum_a \sum_b [\mathsf{V}]_{i, j, a, b} [\mathbf{X}]_{i+a, j+b}$$

- **Change of variables:** Substitute relative offset $(a, b) = (k - i, l - j)$ and define $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$.
- **No reduction yet:** Re-indexing merely reads $\mathsf{W}$ along spatial diagonals ($10^{12}$ free weights until $(i,j)$-dependence is dropped).
:::

::: {.col .fig}
@fig:ch06-cnn-offset-reindexing
:::
:::
:::

::: {.slide title="Proposition: Shared Weights Yield Equivariance"}
[Sufficiency Proof on the Infinite Grid]{.kicker}

::: {.cols .vc}
::: {.col}
$$[f(T_v \mathbf{X})]_{i,j} = u + \sum_{a,b} [\mathbf{K}]_{a,b} [\mathbf{X}]_{i - v_1 + a, j - v_2 + b} = [T_v f(\mathbf{X})]_{i,j}$$

- **Infinite zero-padded grid:** Ensures shifts never clip nonzero pixels at a finite boundary.
- **Sufficiency:** Tying $[\mathbf{U}]_{i,j} = u$ and $[\mathsf{V}]_{i,j,a,b} = [\mathbf{K}]_{a,b}$ makes $f$ commute with every shift $T_v$.
:::

::: {.col .fig}
@fig:ch06-cnn-infinite-grid-shift
:::
:::
:::

::: {.slide title="Converse Proof: Equivariance Forces Weight Sharing"}
[Necessity Proof via Unit Impulse Probing]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Constant bias:** $T_v \mathbf{0} = \mathbf{0} \implies \mathbf{U} = T_v \mathbf{U}$, so $[\mathbf{U}]_{i,j} = u$.
- **2. Impulse probing:** Feeding a single $1$ at $(k,l)$ and shifting by $v = (i,j)$ forces $[\mathsf{V}]_{i,j,a,b} = [\mathsf{W}]_{0,0,a,b} \equiv [\mathbf{K}]_{a,b}$.
- **Equivariance theorem:** Linear layer commutes with shifts $\iff$ it shares one kernel $\mathbf{K}$ and scalar bias $u$ ($10^{12} \to 4 \times 10^6$ weights).
:::

::: {.col .fig .big}
@fig:ch06-cnn-impulse-probing
:::
:::
:::

::: {.slide title="Enforcing Locality Yields the Convolutional Layer"}
[Pruning Distant Offsets to a $(2\Delta+1) \times (2\Delta+1)$ Window]{.kicker}

::: {.cols .vc}
::: {.col}
$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{K}]_{a, b} [\mathbf{X}]_{i+a, j+b}$$

- **Local window $|a|, |b| \le \Delta$:** Zeroes out distant offsets so each unit inspects only a $(2\Delta+1) \times (2\Delta+1)$ neighborhood.
- **11-order-of-magnitude collapse:** For $\Delta = 1$ ($3 \times 3$ kernel), $10^{12}$ weights (**4 TB**) shrink to just **9 shared weights** (**36 B**)!
:::

::: {.col .fig}
@fig:ch06-cnn-locality-window
:::
:::
:::

::: {.slide title="Inductive Bias: Sample Efficiency vs. Expressivity"}
[What We Gained and What We Traded Away]{.kicker}

::: {.cols .vc}
::: {.col}
- **Sample efficiency:** Pooling evidence across all $H \times W$ positions prevents overfitting on local, shift-equivariant visual structures.
- **Single-layer blind spot:** Tied weights cannot tell identical local patches apart by absolute position (e.g., an eye vs. a nostril).
- **Deep recovery:** Stacking layers with nonlinearities expands the receptive field to capture whole-object layout.
:::

::: {.col .fig .big}
@fig:ch06-cnn-inductive-bias
:::
:::
:::

::: {.slide title="Mathematical Convolution vs. Cross-Correlation"}
[Why Deep Learning Libraries Omit the Kernel Flip]{.kicker}

::: {.cols .vc}
::: {.col}
$$\sum_{a,b} [\mathbf{K}]_{a, b} [\mathbf{X}]_{i+a, j+b} = (\mathbf{K}' * \mathbf{X})(i, j)$$

- **Sign convention:** Mathematical convolution uses $(i-a, j-b)$; cross-correlation uses $(i+a, j+b)$.
- **Identical hypothesis class:** Because $[\mathbf{K}']_{a,b} = [\mathbf{K}]_{-a,-b}$ is a 180° flip of learned weights, frameworks implement cross-correlation and call it `Conv2d`.
:::

::: {.col .fig}
@fig:ch06-cnn-conv-vs-xcorr
:::
:::
:::

::: {.slide title="From Grayscale Grids to 3D Channel Tensors"}
[Multi-Channel Inputs and Feature Maps]{.kicker}

Real images carry multiple values per pixel (e.g., Red, Green, Blue), and hidden layers must detect dozens of distinct visual patterns in parallel:

::: {.cols .vc}
::: {.col}
- **Input tensor $\mathsf{X} \in \mathbb{R}^{c_\textrm{i} \times H \times W}$:** Attaches a $c_\textrm{i}$-dimensional vector to each spatial pixel $(i, j)$ ($c_\textrm{i} = 3$ for RGB).
- **Single-channel limitation:** One kernel produces only one score map (like the single Waldo activation map on the right).
- **Output tensor $\mathsf{H} \in \mathbb{R}^{c_\textrm{o} \times H \times W}$:** Stacks $c_\textrm{o}$ parallel **feature maps**, each detecting a different learned pattern (oriented edges, color contrasts, textures).
:::

::: {.col .fig}
![](../img/waldo-mask.jpg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="The 4D Convolutional Kernel and Parameter Count"}
[All-to-All Channel Wiring with Spatial Weight Sharing]{.kicker}

::: {.cols .vc}
::: {.col}
$$[\mathsf{H}]_{o, i, j} = u_o + \sum_{c=1}^{c_\textrm{i}} \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathsf{K}]_{o, c, a, b} [\mathsf{X}]_{c, i+a, j+b}$$

- **4D kernel $\mathsf{K} \in \mathbb{R}^{c_\textrm{o} \times c_\textrm{i} \times k_\textrm{h} \times k_\textrm{w}}$:** Total parameters $= c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w} + c_\textrm{o}$.
- **Resolution independence:** A $3 \times 3$ RGB-to-64 layer uses $1{,}792$ parameters on both $28 \times 28$ and $4000 \times 4000$ images.
:::

::: {.col .fig}
@fig:ch06-cnn-4d-kernel-tensor
:::
:::
:::

::: {.slide title="Convolutions Beyond 2D Images: 1D, 2D, and 3D"}
[The Same Symmetry Principle Across Modalities]{.kicker}

::: {.cols .vc}
::: {.col}
- **1D, 2D, and 3D grids:** Equivariance + locality yields 1D convolutions over time $(c_\textrm{i}, T)$, 2D over images $(c_\textrm{i}, H, W)$, and 3D over video/volumes $(c_\textrm{i}, D, H, W)$.
- **When equivariance breaks:** In a linear-Hz spectrogram, shifting frequency alters harmonic ratios—requiring a log-frequency (mel) axis before applying 2D convolutions.
:::

::: {.col .fig .big}
@fig:ch06-cnn-1d-2d-3d-modalities
:::
:::
:::

::: {.slide title="Concept Check: Deriving and Counting Convolutions"}
[Concept Check]{.kicker}

Work out these three questions from first principles:

1. **28×28 weight collapse:** On a $28 \times 28$ single-channel image mapped to a $28 \times 28$ grid, how many weights does a dense layer have, how many survive translation equivariance (zero-padded outside the frame), and how many remain with a $5 \times 5$ local window ($\Delta = 2$)?
2. **Resolution vs. parameters:** A $3 \times 3$ layer maps $c_\textrm{i}=3$ to $c_\textrm{o}=64$ channels. How many parameters does it have on a $224 \times 224$ image versus a $1024 \times 1024$ image? Compare with the $200{,}704$ first-layer weights of our Fashion-MNIST MLP ($784 \to 256$).
3. **Periodic boundary ring:** On a 1D periodic ring of 4 samples (where index 4 wraps to 0), how many free weights does an equivariant $4 \times 4$ matrix have before imposing locality?

::: {.d2l-note .fragment .fade-up}
**Answers:**
(1) Dense: $784^2 = 614{,}656$; Equivariant (offsets $-27 \dots 27$): $55^2 = 3{,}025$; Local $5 \times 5$: $25$ weights ($+1$ bias).
(2) Both resolutions use $64 \times 3 \times 9 + 64 = 1{,}792$ parameters ($< 0.9\%$ of the MLP's first layer).
(3) Exactly $4$ free weights (a circulant matrix where each row shifts cyclicly by 1, vs. $2(4)-1 = 7$ on a zero-padded line).
:::
:::

::: {.slide title="Summary & Key Takeaways"}
[Section Synthesis]{.kicker}

| Principle / Mechanism | Mathematical Form | Parameter Count ($c_\textrm{i} \to c_\textrm{o}$, grid $H \times W$) | Deep Learning Role |
|---|---|---|---|
| **Dense Layer on 2D Grid** | $[\mathbf{H}]_{i,j} = [\mathbf{U}]_{i,j} + \sum_{a,b} [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i+a,j+b}$ | $c_\textrm{o} c_\textrm{i} H^2 W^2 + c_\textrm{o} HW$ | Arbitrary spatial wiring; intractable at high resolution |
| **Translation Equivariance** | $f(T_v \mathbf{X}) = T_v f(\mathbf{X}) \iff [\mathsf{V}]_{i,j,a,b} = [\mathbf{K}]_{a,b}$ | $c_\textrm{o} c_\textrm{i} (2H-1)(2W-1) + c_\textrm{o}$ | Ties weights across space so feature maps move with the input |
| **Locality ($\|a\|, \|b\| \le \Delta$)** | $[\mathbf{K}]_{a,b} = 0$ outside $(2\Delta+1) \times (2\Delta+1)$ | $c_\textrm{o} c_\textrm{i} (2\Delta+1)^2 + c_\textrm{o}$ | Restricts each unit to local neighborhoods; independent of $H, W$ |
| **Cross-Correlation vs. Conv** | $(\mathbf{X} \star \mathbf{K})_{i,j} = (\mathbf{K}' * \mathbf{X})_{i,j}$ | Identical | Unflipped indexing $(i+a, j+b)$ implemented by all frameworks |

::: {.d2l-note}
**Next (§{sec_conv_layer}):** We implement 2D cross-correlation in code, hand-craft and *learn* an edge-detection kernel via gradient descent, recast convolution as matrix multiplication (*im2col*), and analyze multi-layer receptive fields.
:::
:::
