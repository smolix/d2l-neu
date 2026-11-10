# Writing pass: Parameters, State, and Memory

Source: `chapter_builders-guide/parameters-state-memory.md` (section 5.2, second
of eight after `model-construction`). Prose only; no code cell, label, figure,
exercise, or slide was changed.

## Size

Prose words (all four tabs, checker count): 7,318 before, 6,784 after (-534).
`check_prose_only.py`: "OK: only prose changed". `tools/lint_source.py`: clean.

## Before and after

1. Opening.
   - Before: "So far the framework has handled those tensors for us, and we
     have not asked what else a model holds or how much memory training it
     takes. Both questions matter once a model is large."
   - After: "So far the framework has handled those tensors for us, and we
     never had to ask what else a model holds or how much memory training it
     takes. For a small model there is no need to ask. For a large one both
     questions become pressing."
2. What the section is for (roadmap).
   - Before: "This section follows the tree to its leaves, separates the
     tensors the optimizer trains from those it must leave alone, counts the
     bytes that training stores per parameter, and lets two layers share one
     tensor or excludes part of a model from training."
   - After: "This section is an inventory of that tree. Its first part asks
     what a model holds: the tensors the optimizer trains, and those it must
     leave alone. Its second part asks what they cost, and counts the bytes
     that training stores per parameter. The third part is closer to a
     reference. It collects three things that are routinely done to
     parameters (sharing one between two layers, excluding some from
     training, and averaging them over time), each of which borrows a model
     or a method from a later chapter. It can be skimmed now and consulted
     when those chapters call for it."
3. The rule for kinds of state, formerly once per tab.
   - Before (PyTorch tab): "PyTorch calls persistent tensors without
     gradients *buffers* and registers them with `register_buffer`. The
     rule: make a tensor a *parameter* if the optimizer should update it, a
     *buffer* if checkpoints need it, and a local variable if it lives for
     one call."
   - After (shared): "Such a persistent tensor without a gradient is commonly
     called a *buffer*, and every framework has a way to register one so that
     checkpoints and device moves see it while the optimizer does not. This
     gives a rule that covers every tensor a layer touches: make it a
     *parameter* if the optimizer should update it, a *buffer* if checkpoints
     need it, and a local variable if it lives for one call." (PyTorch tab:
     "PyTorch uses the name *buffer* itself and registers one with
     `register_buffer`.")
4. Mixed precision (PSM-5).
   - Before: "A common recipe stores the weights and the gradients in 16 bits
     for the forward and backward passes, and keeps an fp32 master copy of
     the weights and fp32 moments, so that small updates are not rounded away
     ...: $2 + 2 + 4 + 4 + 4 = 16$ bytes per parameter, the same floor. ...
     A multiplier therefore means little until we know the dtype of each
     tensor."
   - After: "One might hope that computing with 16-bit numbers, the
     *mixed-precision* training of :numref:`sec_numerics`, halves this
     figure. Unfortunately it does not lower the floor at all: an fp32 copy
     of the weights and fp32 moments are kept, so that small updates are not
     rounded away ..., and the 16-bit copies come on top."
5. Counting without allocating.
   - Before: "The count needs only the shape and the dtype of each parameter,
     and both follow from the arguments of the constructors."
   - After: "Luckily, the count needs only the shape and the dtype of each
     parameter, and both follow from the arguments of the constructors."
6. Freezing, first limit (PSM-9), formerly once per tab.
   - Before: "Freezing has two more effects that raise no error. First, it
     does not reclaim optimizer state that already exists."
   - After: "Freezing has two limits, and neither raises an error. The first
     is a matter of housekeeping: a freeze does not reclaim optimizer state
     that already exists. ... The Adam optimizer with which we counted the
     moments of `net` above is such an optimizer, and for 18,634 parameters
     nobody will mind. The last exercise works out what the same oversight
     means for the 8B model."
7. Freezing, second limit (PSM-10).
   - Before (PyTorch tab): "Second, freezing governs parameters only, and
     batch normalization carries state that is not a parameter."
   - After (shared): "The second limit matters more. Freezing stops
     gradients, and some state changes without one."
8. Closing (PSM-12).
   - Before: "We can now find every tensor of a model, tell parameters from
     other state, count what training stores per parameter, share one tensor
     between layers, and exclude parts of a model from training. Every one of
     these tensors started from a value that some rule chose."
   - After: "Every parameter we met in this section, whether tied, frozen, or
     averaged, started from a value that some rule chose."

## Repetition removed

- **Tab prose that was the same in all four tabs** now stands once in the
  shared prose, and each tab keeps one or two sentences true of that
  framework: "A model built from ... is a tree, and its parameters are the
  leaves"; "Reaching parameters one path at a time suits debugging ..."; the
  parameter/buffer/local-variable rule; "Registration is what makes the module
  system aware of a tensor ..."; "On a fresh copy of our residual MLP ...
  leaves 650 of 18,634"; "One Adam step then moves the head and nothing
  else"; "Freezing has two more effects ... First ..."; "Second, ... batch
  normalization"; "the parameter count shows the $|V| \times d$ saving".
- **Tree and paths** (owned by 5.1): reduced to one sentence with
  `:numref:`fig_blocks``; "Read one of the names closely" and "Names are
  paths, so they survive any amount of nesting" are gone, and the tabs keep
  only the reading of their own path.
- **Unregistered tensors are left behind** (owned by 5.1, "Layers in a Plain
  List" and the constant $\alpha$): one shared sentence with a reference; the
  $\alpha$ example is now named as the motivation for buffers.
- **32 GB and 128 GB**: derived in the opening, cited once as "the 128 GB of
  the opening" under "Bytes per Parameter"; the third statement at the end of
  "Counting Without Allocating" now says "the memory figures of the opening".
  The Summary keeps the number.
- **Closing recap** of five capabilities removed; the Summary is the only
  recap.
- **Definition of tokens** now at first use (output layer of the 8B model),
  not repeated under "Tied Parameters".

## Review findings

Applied (prose): PSM-1 (why Adam, and the 8 bytes of minibatch SGD), PSM-2
("specimen" replaced by "our residual MLP" in the body), PSM-5, PSM-7, PSM-8,
PSM-12; the structural finding "three sections in one" is answered in prose
only, by the roadmap telling the reader that the third part is a reference.

Partly applied (prose trimmed, cells stay): PSM-3 (the gradient tab prose is
now one shared question and one or two sentences per tab), PSM-9 (the
demonstration is introduced as housekeeping and points to the exercise),
PSM-10 (the mechanism is stated once in shared prose), PSM-11 (the text now
says the EMA is here only because it is checkpoint state).

Deferred:
- PSM-3, PSM-6, PSM-9, PSM-10, PSM-11 in full: each asks to cut or move code
  cells (the `.grad` cell, the meta device, the stale-optimizer check, the
  BatchNorm cell, the EMA loop).
- PSM-4: renaming `Whitener` changes code. The prose now admits that the name
  is generous.
- "Slow start" in "Access by Path" (six cells): needs cells removed.
- Exercises 1 and 2: exercises were not touched.
- Buffers motivated three times in the chapter: this section keeps the full
  explanation (the review names it the home); 5.1 and 5.4 are other writers'
  files.

## Claims added or changed

- Added: minibatch gradient descent stores 8 bytes per parameter in fp32
  (weight and gradient, no optimizer state).
- Added: "almost all large models are trained with [Adam] or with one of its
  variants" (the review's suggested sentence).
- Changed: mixed precision. The sum $2 + 2 + 4 + 4 + 4$ and the description of
  automatic casting are gone; the text says an fp32 copy of the weights and
  fp32 moments are kept and "the 16-bit copies come on top". It no longer
  says "A multiplier therefore means little ...".
- Added: tokens are "the words and pieces of words ... that the model can
  output".
- Added (PSM-8): intuition for tying: "words that can replace one another in
  a sentence should be similar as inputs and receive similar scores as
  outputs, and tying leaves the model only one notion of similarity to
  learn". Offered as "at least plausible", not as the cited papers' argument.
- Added: "whitening proper would also remove the correlations between the
  features".
- Added judgment: the second limit of freezing (running statistics) "matters
  more" than the first (stale optimizer state), which is "a matter of
  housekeeping".
- Added after the BatchNorm cell, per tab: "The running mean moved" (MXNet,
  PyTorch, JAX print `False`) and "The running mean stayed where it was"
  (TensorFlow prints `True`); checked against the four stored outputs.
- Removed from the TensorFlow tabs: that a `tf.device` scope overrides
  placement and that relocating state means creating it again.

## Slides that now differ from the body

- The slides still call the model "the specimen" (slide title "The Specimen
  Has 18,634 Parameters, 0.28 MiB of Training State" and the summary slide).
- The mixed-precision bullet on the bytes slide ("computes in 16 bits but
  keeps an fp32 master copy and ...") is more detailed than the body now is;
  it does not contradict it.
