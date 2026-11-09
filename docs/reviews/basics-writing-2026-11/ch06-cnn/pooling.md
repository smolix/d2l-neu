# Pooling

Source: `chapter_convolutional-neural-networks/pooling.md`, about 2,000 words of prose (245 in framework tabs), 6 exercises (about 520 words).

## Verdict

The section contains the chapter's most honest experiment (a single response moved one pixel at a time through a $2\times2$ pooling layer) and its most honest sentence ("Pooling makes a representation *tolerant* to some small shifts. It does not make it invariant to them."). Around that core the section is uneven. The opening presents pooling as the answer to two obstacles, of which the first (reach) was already solved by stride two sections earlier, in a paragraph that leans on an equation from another page. Over a quarter of the body is API mechanics, interleaved with twelve framework tab notes. And the idea that completes the chapter's argument, global pooling, which is what turns an equivariant feature map into an invariant prediction, appears for the first time in the second paragraph of the Summary. The most important fix is to give global pooling a short subsection of its own, with the same shift experiment, so that the promise made in `why-conv.md` ("a final step that aggregates the equivariant map over all locations") is kept.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 3 | 3 | 4 | 3 |

## What works

- The opening frames the section with a question the reader has not yet been asked: "This section asks how a network gets from many local answers to one global answer."
- "Tolerance to Small Shifts" as a whole. The setup is minimal (one response of value 1), the output is three lines the reader can check by eye, and the reading is exact: "A one-pixel shift of a single response is therefore absorbed when the response stays inside its block, which happens for half of all horizontal positions, and it moves the pooled output by one position otherwise." Most textbooks claim "translation invariance" for pooling; this one measures it and gets one half.
- The distinction drawn from the same output: max-pooling "answers whether the feature is present somewhere in the window"; average pooling "measures how much of the window the feature fills".
- The padding value for a maximum: "For a maximum, the neutral value is $-\infty$: padded positions then never win", checked on `X - 16` with the remark "Zero padding would have produced 0 in the three windows that touch the padding". A small idea, fully worked, with a test that would fail if the idea were wrong.
- The aliasing paragraph: three sentences, one extreme example ($1, 0, 1, 0, \ldots$), the name, and a pointer to the exercise.

## Structural findings

- **Global pooling is missing from the body.** See POOL-8. This is a chapter-level gap: `why-conv.md` sets up equivariance against invariance, this section shows local pooling gives only tolerance, `lenet.md` uses a flatten-and-dense head (not invariant), and the operation that actually produces invariance is named only in a summary.
- **The first "obstacle" is already solved.** See POOL-1.
- **"Padding and Stride" is mostly plumbing.** About 590 words and five cells (fifteen with their framework variants), of which the shape formula is a repeat from `sec_padding`, and the content worth keeping is the $-\infty$ padding argument and the note on average pooling at the border. See POOL-6, POOL-7.
- **History interrupts the definition.** See POOL-3.

## Passage-level findings

**POOL-1** (moderate, thought). Opening:

> The first obstacle is reach. Some unit near the top of the network must have a
> receptive field, the input region that can influence it
> (:numref:`sec_conv_layer`), that covers the whole image.

`sec_padding` has already shown that stride gives geometric reach, with a formula and a figure, and this paragraph relies on that result. Presenting reach as an obstacle that pooling addresses makes the reader ask why they need a new layer. The section's Summary concedes the point ("Later networks often halve the resolution with a convolution of stride 2 instead"). Fix: say what is true and still interesting. "We already know one way to see further: stride (:numref:`subsec_stride_rf`). Pooling is a second way to reduce resolution, with no parameters. What it adds is a deliberate choice of *summary*: the strongest response in a neighborhood, or the mean. That choice matters for the second problem."

**POOL-2** (moderate, explanation). Opening, same paragraph:

> is followed by a $2\times2$ step that halves the resolution, each later kernel
> reaches twice as far in the input (:numref:`subsec_stride_rf`), and each
> $2\times2$ window adds its own extent (:eqref:`eq_receptive_field` with
> $k=s=2$). Six $3\times3$ convolutions with two such steps then see a
> $32\times32$ patch.

A sentence that can be verified only by opening the previous section, substituting into its formula for eight layers, and adding nine terms (I did: $1+2+2+1+4+4+2+8+8 = 32$; it is right). It refers to "a $2\times2$ step" because pooling has not been defined yet. The opening of a section should not need the reader to compute. Fix: keep the contrast in round numbers ("covering a $28\times28$ image takes fourteen $3\times3$ layers at full resolution; with two halvings of the resolution, six are enough") and drop the equation reference, or move the calculation to an exercise (exercise 2 already asks it for $224\times224$).

**POOL-3** (minor, pedagogy). Under "Maximum Pooling and Average Pooling":

> Average pooling dates to the earliest CNNs. It is a form of image
> downsampling that, instead of keeping every second pixel, averages
> neighboring pixels and so smooths out variation from pixel to pixel.

An eight-sentence history paragraph (average pooling, Riesenhuber and Poggio, an earlier use in speech, the Neocognitron "earlier still", the 2010s) sits between the definition of pooling and its first example, and it runs backward in time. The Neocognitron's complex cells are the one item that helps the reader ("responded when one detector fired anywhere in a small neighborhood, so its output tolerated small shifts"), and they would help most in "Tolerance to Small Shifts". Fix: move the figure and worked example up to follow the definition directly; reduce the history to two sentences and place them after the shift experiment.

**POOL-4** (minor, pedagogy). After the average-pooling cell:

> Each window holds the numbers $a, a+1, a+3, a+4$ for some $a$, so its
> mean $a+2$ lies 2 below its maximum $a+4$.

An observation about the test matrix, not about pooling. It reads as filling the "interpret every output" rule. Cut; "Average pooling of the same input returns the four window means, 2, 3, 5, and 6." is enough.

**POOL-5** (minor, explanation). Under "Tolerance to Small Shifts":

> A stride of 2 keeps one value per block, so the output can jump under a
> one-pixel shift, as at the block boundary above.

The experiment prints nested Python lists on one line per position ("max [[1.0, 0.0], [0.0, 0.0]] avg [[0.25, 0.0], [0.0, 0.0]]"). The result is readable but only just. A tiny figure (three $4\times4$ maps with the response in columns 0, 1, 2, and under each the $2\times2$ max-pooled map) would make the half-absorbed, half-passed-on behavior visible at a glance, and would be the natural place to add the fourth panel that POOL-8 asks for (a global maximum that does not move at all).

**POOL-6** (moderate, pedagogy). Under "Padding and Stride":

> The frameworks expect
> four-dimensional input, so
> we put the $4\times4$ matrix of the numbers 0 to 15 into a tensor with one
> example and one channel.

From here to the end of the subsection there are twelve tab notes (layout, default stride, how Keras pads, how Flax pads, how each average pool counts padding) and five code cells, most in several framework versions. The reader of any one tab sees a paragraph of prose, a note, a cell, a note, a cell. The ideas are three: the shape rule is the one for convolutions; the usual stride equals the window; the padding value must be neutral. Fix: state the first two in two sentences with one cell, keep the $-\infty$ argument and its check, and put the framework defaults into code comments. Target: half the current length.

**POOL-7** (minor, pedagogy). Under "Padding and Stride":

> Pooling windows can also have different heights and widths, here a
> $2\times3$ window with stride $(2, 3)$ and one column of padding on the
> left and right.

The cell that follows prints 5, 7, 13, 15, which is identical to the output of the $3\times3$, padding 1, stride 2 cell above it, and the text does not read the output. A reader comparing the two concludes either that nothing changed or that they have misunderstood. Cut the cell and the sentence.

**POOL-8** (major, pedagogy). Summary, second paragraph:

> Pooling then remains in a
> second role, *global average pooling*, which averages each channel over all
> positions of the final feature map :cite:`Lin.Chen.Yan.2013`.

A new concept defined in the Summary, and the one the chapter's argument needs. `why-conv.md` said the invariant answer is obtained "in a final step that aggregates the equivariant map over all locations, for instance by taking its maximum". This section shows that $2\times2$ pooling does not deliver that. The reader is owed the step that does. Fix: add a short subsection "Global Pooling" before "Multiple Channels": define it (window equal to the whole map; one number per channel); rerun the single-response experiment with a global maximum and a global mean and observe that the output is the same for every position; state the consequence (a convolutional encoder followed by global pooling is invariant to shifts, up to border effects, and the classifier head no longer depends on the image size); say that LeNet does not do this and the networks of the next chapter do. About 200 words and one three-line cell. Then the Summary can restate it instead of introducing it.

**POOL-9** (minor, thought). Summary, second paragraph:

> Later networks often halve the resolution with a convolution of stride 2
> instead, which learns how to combine the entries of a window rather than
> fixing the combination to a maximum or a mean.

Also new material in a Summary. It belongs in the body, where POOL-1 suggests the opening should already say it.

## Exercises

A good set, graded, each with a point.

- Exercise 1 (average pooling as a convolution): good; "how the convolution must be structured across channels" quietly requires the depthwise convolution of `sec_channels`. If that material moves to chapter 7 (CHAN-6), rephrase as "what must the kernel be for pairs of different channels?".
- Exercise 2 (receptive field and cost of downsampling): this is the calculation the opening paragraph compresses into one sentence (POOL-2). Good.
- Exercise 3 (minimum pooling; softmax pooling with its two limits): good.
- Exercise 4 (aliasing and blur-pool): very good; the last part lets the reader discover that blur-pool with a two-tap box filter is average pooling.
- Exercise 5 (backward pass of max-pooling): standard and worthwhile.
- Exercise 6 (max-pooling via ReLU): good hard problem. Part 1 ("Prove that max-pooling cannot be implemented by a convolution alone") has a one-line answer (a convolution is linear) and could be folded into part 2.
- If POOL-8 is adopted, add one problem: show that a network of stride-1 convolutions with shape-preserving zero padding followed by global average pooling is *not* exactly shift invariant, and locate the cause (the border). The LeNet exercise on equivariance then has a stepping stone.

## Still open from October

No October finding was specific to this section. The legacy prose is gone.
