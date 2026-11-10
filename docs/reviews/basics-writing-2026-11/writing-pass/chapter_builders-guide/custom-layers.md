# Writing pass: `chapter_builders-guide/custom-layers.md` (5.4)

Prose words (checker, all four tabs, body and exercises): **5262 before, 5150
after** (-112). Checker: "OK: only prose changed". `tools/lint_source.py`:
clean. No code cell, label, figure, exercise, or slide was touched.

Section order used for the earliest-section-owns rule (from `_quarto.yml`):
model-construction, parameters-state-memory, init, **custom-layers**,
numerics, saving-loading, gpus-devices-memory, reproducibility-inspection.
Registration and buffers are therefore owned by 5.1 and 5.2; this section now
cites them.

## Before/after pairs

1. Before: "Sometimes the problem is the derivative instead. Rounding, which
   is needed to store weights as 4-bit integers, has derivative zero almost
   everywhere, so a weight that is rounded in the forward pass receives no
   gradient and never learns."
   After: "Sometimes the trouble lies with the derivative instead. To fit a
   large model into a small memory, its weights are often stored as 4-bit
   integers: the weights of the 8B model of :numref:`sec_model_construction`
   then take 4 GB, a quarter of what 16-bit floating-point numbers need.
   Storing integers means rounding, and rounding has derivative zero almost
   everywhere, so a weight that is rounded in the forward pass receives no
   gradient and never learns."

2. Before: "Recall that a module owns registered parameters and child modules
   and defines a forward computation, and that everything the framework does
   to a whole model, from collecting parameters for the optimizer to saving a
   checkpoint, walks the registered state."
   After: "Everything the framework does to a whole model, from collecting
   parameters for the optimizer to saving a checkpoint, walks the registered
   state, so a layer that registers its state needs no further code to take
   part. The ideas here are few. Most of the section shows the form in which
   each framework expects them."

3. Before: "Feeding data through confirms that it does what it says:"
   After: "The inputs 1 to 5 have mean 3, so the output should run from $-2$
   to $2$:"

4. Before: "Their mean should be zero; since the mean is computed in
   floating-point arithmetic, we see a number of size $10^{-8}$ or smaller
   rather than exactly zero, which is rounding error, not a bug."
   After: "Their mean should be zero. In floating-point arithmetic it comes
   out as exactly zero only by luck; a number of size $10^{-8}$ or smaller is
   rounding error and no cause for alarm."

5. Before: "We wrote RMSNorm as a module, not as a function with a gain tensor
   kept beside it, because registration makes a custom layer behave like a
   built-in one in four ways: [...] We check each once."
   After: "Why did we write RMSNorm as a module and not as a function with a
   gain tensor kept beside it? Because registration makes a custom layer
   behave like a built-in one in four ways: [...] None of this is new.
   :numref:`sec_model_construction` and :numref:`sec_parameters` showed all
   four for the library's layers, so we only run each check once on a layer
   of our own."

6. Before: "Where the library provides the same layer, comparing the outputs
   of the two implementations tests ours."
   After: "The checks so far say that the layer is a good citizen of the
   framework. They do not say that it computes the right thing. Where the
   library provides the same layer, we can find out by comparing the outputs
   of the two implementations."

7. Before: ":numref:`fig_bg_ste` shows the two passes. Automatic
   differentiation cannot produce this rule, because [...]"
   After: "This is deliberately not the true derivative. Its justification is
   that the staircase never strays more than $1/2$ from the identity, as
   :numref:`fig_bg_ste` shows. Automatic differentiation cannot produce such
   a rule, because [...]"

8. Before: "We can now add to a model any layer that tensor operations can
   express, with parameters, buffers, or a backward rule of our own, and have
   it trained, saved, and moved like a built-in one. What we have not asked is
   whether its arithmetic is accurate: [...]"
   After: "What we have not asked of any of these layers is whether their
   arithmetic is accurate. RMSNorm squares its inputs, and [...]"

Sentences with voice that were kept as they stood: the opening on RMSNorm
(2019 to 2024), the $(3, 4)$ worked example, "The gradient is zero, as
calculus says, and useless for training.", the PyTorch warning "the custom
backward pass never runs, and no error says so".

## Repetition removed

- **What a module is** (owned by 5.1): the "Recall that a module owns ..."
  restatement in the opening is reduced to one clause.
- **Registration of parameters** (owned by 5.2): "A layer with something to
  learn must register its parameters" stood in all four tabs, each with its
  own reference; now one shared sentence. The four "First, the gain was
  registered when ..." tab openings, which retold the previous cell, are one
  sentence each.
- **What a buffer is** (owned by 5.2, `subsec_params_buffers`): the shared
  definition is one clause with a reference; each tab now names only the call
  (`gluon.Constant`, `register_buffer`, `add_weight(trainable=False)`, the
  `Buffer` subclass). The per-tab retelling of "stored as plain attributes
  they would be missing, and the loaded layer would keep its own random
  matrix" keeps only its first half; the consequence is stated once in the
  shared prose.
- **"Like a built-in one"**: said in the opening, the Registration intro, the
  Registration closing, and the section closing. The Registration closing is
  one sentence now and the "We can now ..." recap before the Summary is gone.
- **4-bit weights take a quarter of 16-bit**: derived once (opening); the
  custom-gradient subsection now says "Recall the model ...".
- **"The two definitions agree in value and in gradient:"** stood in four
  tabs; now once in shared prose.
- Summary: dropped the clause restating that a custom layer "tested against
  its definition, is the implementation" (already in the body).

## Review findings

Applied (prose): CL-1 (purpose of 4-bit weights, link to the 8B model), CL-2,
CL-3, CL-4 (family of normalization layers, forward reference), CL-7 (cut),
CL-9 (the clipped variant now points to exercise 5, which implements it, and
the later sentence refers to "the clipped variant"), CL-10 (shortcut named as
the common form), CL-11 ("kernels" replaced by "low-level operations for the
GPU"). Structural finding "buffers introduced a third time": reduced as
above.

Partly applied: CL-5. The three cells that print what the reader has seen
cannot be removed in this pass; the prose around them now says that nothing
here is new and is shorter. CL-6: "A simple example" is now "An example", and
the text says candidly that the fit is not shown and that exercise 4 asks for
it.

Deferred:
- CL-5 (remove three of the four registration cells): needs cell removal.
- CL-6 (move the $\sin(3x)$ fit into the text, reload onto a fresh layer):
  new cells and a plot.
- CL-8 (train a rounded weight with and without the estimator): new
  experiment. The text now states that only the rule was checked.
- Exercise notes (drop exercise 5 part 2, shorten attributions): exercises
  are out of scope.

## Factual claims added or changed

- Opening: the weights of the 8B model take 4 GB as 4-bit integers, a quarter
  of the 16-bit size ($8 \times 10^9 \times 4$ bits; the slides already carry
  the same numbers).
- `CenteredLayer` subtracts the mean "over all of its entries, the batch axis
  included" (read off the code; exercise 1 says the same).
- Centered mean: the old sentence said we see a small number "rather than
  exactly zero". The stored TensorFlow output is exactly `0.0` (the others are
  between $6 \times 10^{-10}$ and $7 \times 10^{-9}$), so the text now allows
  both.
- Normalization layers "form a family that we study from
  :numref:`sec_batch_norm` onwards".
- Straight-through estimator: "the staircase never strays more than $1/2$
  from the identity" (true of rounding; added as the justification).
- Stop-gradient form "is the form met most often in practice" (from CL-10; not
  checked against any survey).
- Random features: "Surprisingly, these features need no training" (rewording
  of the existing claim); "We do not show this here; exercise 4 asks you to
  fit $y = \sin(3x)$".
- "Exercise 6 takes up" whether anything learns through the rounding;
  "exercise 5 implements such a rule" (the clamp with a gradient only inside
  the range). Both exercises are referred to by number in prose, so a
  renumbering of the exercises would break them.
- Keras tab, device placement: dropped "an operation runs where its inputs
  are"; nothing new claimed.

## Slides that now differ from the body

- "Where this fits" slide: "**Skips:** writing GPU kernels". The body no
  longer uses the word "kernels" (CL-11).
- No slide contradicts the body. The slides still say "Not one line of
  `RMSNorm` handles these four tasks", which agrees with the shortened
  closing of "Registration".
