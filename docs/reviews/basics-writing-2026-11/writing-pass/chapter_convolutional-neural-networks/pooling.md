# Writing pass: `chapter_convolutional-neural-networks/pooling.md`

Prose words (checker count, body and exercises): **2535 before, 2410 after**
(-125). Checker: "OK: only prose changed". `tools/lint_source.py`: clean.
One unlabeled display equation (the output-height formula, a repeat of
`sec_padding`) became inline mathematics; no code cell, label, figure, or
slide changed.

## Before/after pairs

1. Before: "Two obstacles stand in the way. The first obstacle is reach. Some
   unit near the top of the network must have a receptive field, the input
   region that can influence it (:numref:`sec_conv_layer`), that covers the
   whole image."
   After: "Two things are needed. The first is reach: some unit near the top
   of the network must have a receptive field (:numref:`sec_conv_layer`)
   that covers the whole image."

2. Before: "If every second layer is followed by a $2\times2$ step that
   halves the resolution, each later kernel reaches twice as far in the input
   (:numref:`subsec_stride_rf`), and each $2\times2$ window adds its own
   extent (:eqref:`eq_receptive_field` with $k=s=2$). Six $3\times3$
   convolutions with two such steps then see a $32\times32$ patch."
   After: "At full resolution this takes patience, namely fourteen
   $3\times3$ layers for a $28\times28$ Fashion-MNIST image. We already know
   the remedy, which is to reduce the resolution on the way up
   (:numref:`subsec_stride_rf`). With two halvings, six such layers are
   enough."

3. Before: "A *pooling layer* addresses both obstacles. It replaces each
   small window of its input by one summary value, the maximum or the
   average, and has no parameters."
   After: "A *pooling layer* halves the resolution as a stride does, but it
   also decides what to keep. It replaces each small window of its input by
   one summary value, the maximum or the average."

4. Before: (none)
   After: "How much it helps is something we can measure, and the answer is
   less than the word "invariance", often used of pooling, suggests."

5. Before: "Each output is the largest entry in its window. Small changes to
   the entries that are the largest in no window, here 0, 1, 2, 3, and 6,
   leave the output unchanged."
   After: "Note that five of the nine entries, namely 0, 1, 2, 3, and 6, are
   the largest in no window. Small changes to them leave the output
   unchanged: a maximum takes no notice of the entries that lose."

6. Before: "Pooling with a window of $2\times2$ and stride 2 splits its input
   into disjoint $2\times2$ blocks ..."
   After: "How much of a shift does pooling absorb? Let us try the simplest
   case there is. Pooling with a window of $2\times2$ and stride 2 splits its
   input into disjoint $2\times2$ blocks ..."

7. Before: "Average pooling has no neutral padding value."
   After: "Average pooling is less fortunate: it has no neutral padding
   value."

8. Before: "Pooling windows can also have different heights and widths, here
   a $2\times3$ window with stride $(2, 3)$ and one column of padding on the
   left and right."
   After: "Finally, windows need not be square. A $2\times3$ window with
   stride $(2, 3)$ and one column of padding on the left and right returns
   5, 7, 13, and 15 once more: the windows have changed, but the largest
   entry under each is the same as before."

## Repetition removed

- Receptive field: the definition and the $(2L+1)$ derivation are owned by
  `conv-layer.md` and `padding-and-strides.md`; reduced to a clause with two
  references. The appeal to `eq_receptive_field` is gone.
- Output-shape formula: owned by `sec_padding`; the display and its
  explanation became one inline expression with a reference.
- Channels-last layout: stated in two earlier sections; the two tab notes
  became one shared `tensorflow, jax` note of one sentence.
- "No parameters": was in the opening, the definition, "Padding and Stride",
  and the Summary; the opening no longer says it.
- "Reach twice as far": dropped from the Summary (stated in `sec_padding`).
- History (Neocognitron is told in the chapter introduction): eight sentences
  before the first example became three after the shift experiment.
- Tab notes: 12 became 7. Identical notes are merged into multi-framework
  tabs (default stride; `count_include_pad`; channels-last, twice).
- Summary no longer introduces strided convolution and global average
  pooling; both moved into the body (end of "Tolerance to Small Shifts").

## Review findings

Applied: POOL-1 (reach is no longer an obstacle that pooling solves), POOL-2
(round numbers, no equation reference), POOL-3 (history shortened and moved
after the experiment; the example now follows the definition), POOL-4 (cut),
POOL-6 in part (formula and tab notes trimmed), POOL-7 in part (the prose now
reads the output and says why it repeats), POOL-9 (moved to the body),
POOL-8 in prose only (one paragraph defines global pooling and ties it to
`sec_why-conv`).

Deferred:
- POOL-5: needs a figure.
- POOL-8, the experiment: needs a new cell (global max and mean on the single
  response) and the proposed exercise.
- POOL-6, the rest: moving framework defaults into code comments and removing
  cells changes code.
- POOL-7, cutting the $2\times3$ cell: changes code.
- Exercise notes (fold 6.1 into 6.2, rephrase exercise 1 if depthwise
  convolution moves): exercises were not to be touched.

## Factual claims added or changed

- "the answer is less than the word 'invariance', often used of pooling,
  suggests" (new judgment in the opening).
- "With two halvings, six such layers are enough" replaces "see a
  $32\times32$ patch" (same calculation, $32 \ge 28$).
- "Tolerance to shifts is what this operation was invented for" (new; rests
  on the Neocognitron's complex cells).
- "An early form of max-pooling was used in speech recognition" (was "an
  earlier version had been used"); the order of the Yamaguchi and Riesenhuber
  citations is now chronological.
- New in the body: a window covering the whole feature map gives a maximum,
  or a mean, that "is the same wherever in the map the response lies"; this
  is called global pooling; in the networks of `chap_modern_cnn` downsampling
  is "often left to a convolution with stride 2" and the network "usually
  ends in" global average pooling; "LeNet, in the next section, does
  neither". None of this is shown by a cell.
- The $2\times3$ cell "returns 5, 7, 13, and 15 once more" (checked in the
  stored outputs of all four frameworks).
- Dropped: "Average pooling dates to the earliest CNNs", the description of
  average pooling as smoothing downsampling, and "In the networks of the
  2010s, max-pooling became the usual choice between convolutions".
- Tab notes: the Keras note no longer says where `padding='same'` puts an odd
  row; the TensorFlow layout note no longer says "prefers, and is optimized
  for".

## Slides that now differ from the body

No slide contradicts the body. Three carry material the body no longer has
or frames differently:
- "The Implementation Reproduces the Figure" keeps the $a, a+1, a+3, a+4$
  remark that POOL-4 removed from the body.
- "Fourteen 3×3 Layers ..." and the Recap bullet "Reach" still present reach
  as the problem that pooling solves and quote the $32\times32$ patch; the
  body now credits the remedy to `subsec_stride_rf`.
- "Padding and Stride Follow the Convolution Formula" shows the formula as a
  display; the body has it inline.
