# Writing pass: `chapter_convolutional-neural-networks/channels.md`

Prose words (checker): 2964 before, 2956 after. Checker: "OK: only prose
changed". `tools/lint_source.py`: clean. No code cell, label, figure, or
slide was touched. The depthwise figure text of today (81 versus 36 weights)
is unchanged.

## Before and after

1. Before: "A layer with $c_\textrm{i}$ input and $c_\textrm{o}$ output
   channels has $c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w}$ weights,
   and its cost grows with the product $c_\textrm{i} c_\textrm{o}$. The rest
   of the section reduces that product: the $1\times 1$ convolution mixes
   channels without looking at neighbors, and grouped and depthwise
   convolutions restrict which channels are mixed."
   After: "The count turns out to be governed by one quantity, the product
   $c_\textrm{i} c_\textrm{o}$ of the two channel counts, and the rest of the
   section asks how to make that product smaller."
2. Before: "The sum over channels is what lets a kernel compare colors."
   After: "Why add the channels? Because the sum is what lets a kernel
   compare colors."
3. Before: "A multi-input kernel produces one output grid: a feature map,
   which holds one detector's responses at every position."
   After: "A multi-input kernel, however many channels it reads, produces a
   single feature map (:numref:`subsec_why-conv-channels`): the responses of
   one detector."
4. Before: "Every output channel reads every input channel, each through a
   kernel of its own."
   After: the same sentence, followed by "It is worth remembering this dense
   wiring, since the cost of the layer comes from it."
5. Before: "Training also keeps each layer's output, ... This activation
   memory is largest in the early layers, where the resolution is high and
   the weights are few: ..."
   After: "There is a third cost, and it is easy to overlook. Training keeps
   each layer's output, ... This activation memory is largest in the early
   layers, where the resolution is high and the weights are few."
6. Before: "A $1\times 1$ kernel, $k_\textrm{h} = k_\textrm{w} = 1$, looks at
   a single pixel and cannot detect spatial patterns. With several channels
   it still computes something useful."
   After: "At first sight, a $1\times 1$ kernel, $k_\textrm{h} = k_\textrm{w}
   = 1$, is a convolution in name only: it looks at a single pixel and cannot
   detect spatial patterns. With several channels it still computes something
   useful, as the red-minus-green kernel above did."
7. Before: "The factor $c_\textrm{i} c_\textrm{o}$ arises because every
   output channel reads every input channel. A *grouped convolution* splits
   ..." After: "... reads every input channel. Must it? A *grouped
   convolution* splits ..."
8. Before: "We can now build convolutional layers with any number of input
   and output channels, count their parameters and multiply-adds, and reduce
   the cost of mixing channels with $1\times 1$, grouped, and
   depthwise-separable layers. The output of every such layer still shifts
   with its input pixel by pixel, and ..."
   After: "Dense or factored, all of these layers have one property that
   channels do nothing about: the output still shifts with the input, pixel
   by pixel."

## Repetition removed

- The weight count and "cost grows with $c_\textrm{i} c_\textrm{o}$" were in
  the introduction, the cost subsection, and the Summary. The introduction
  now names the product in one clause; the formula stays in the body and the
  Summary.
- "Reducing the factor is the subject of the rest of the section" was said in
  the introduction, at the end of the cost subsection, and at the start of the
  grouped section. The middle one is gone.
- Feature map: defined in `why-conv.md` and `conv-layer.md`; here now one
  clause with a reference.
- Input layout (channels first or last, with the axis list): taught in
  `conv-layer.md` and `padding-and-strides.md`. The four tab notes under
  "Multiple Output Channels" now give only the kernel order, and the two tab
  notes before the last code cell drop the axis list.
- "It has $c_\textrm{o} c_\textrm{i}$ weights plus $c_\textrm{o}$ biases"
  for the $1\times 1$ layer (the general count with $k = 1$): removed.
- The formulaic closing recap ("We can now build ...") is replaced by the
  hand-off to pooling alone.
- The sentence before the $1\times 1$ check no longer announces the result
  that the sentence after it reports.

## Review findings

Applied: CHAN-1 (roadmap without undefined terms); CHAN-3 (the "feature may
be spread over several channels" remark is cut; the supporting argument the
review suggests holds only without a nonlinearity between layers); CHAN-4
(cost paragraph split into parameters, computation, activation memory;
590,080 replaced by "589,824 weights and 256 biases" so one number is used
throughout); CHAN-5 (ResNet and ResNeXt now carry references to
`sec_resnet` and `subsec_resnext`); CHAN-8 ("The two computations agree up
to rounding error").

Deferred:
- CHAN-2: needs the index names and order changed in `why-conv.md`
  (chapter-wide notation decision). "(there called $d$)" stays.
- CHAN-6: moving grouped and depthwise-separable convolutions to chapter 7
  moves code, figures, and exercises between files.
- CHAN-7: figure; already redrawn today.
- Exercises 2 and 4 (replace, cut): exercises are out of scope.

## Factual claims added or changed

- "589,824 weights and 256 biases" replaces "590,080 parameters" (same
  quantity, split).
- "a convolution in name only" for the $1\times 1$ kernel (a judgment).
- "as the red-minus-green kernel above did": the $1\times 1$ kernel of the
  first subsection is cited as an instance.
- The $1\times 1$ check: "agree up to rounding error" replaces "zero or at
  most about $10^{-6}$". Stored outputs: 0.0 (PyTorch, TensorFlow), 1.5e-6
  (MXNet), 9.7e-8 (JAX).
- TensorFlow and JAX tab notes now cite `sec_padding` for the channels-last
  input layout.
- Removed claim: one feature may be spread over several channels.

## Slides against the body

No disagreement found. The slides give 590,080 parameters in one table and
589,824 weights elsewhere; both remain consistent with the body.
