# Writing pass: Data Manipulation (`chapter_preliminaries/ndarray.md`)

Prose words (checker count, all four tabs and exercises): **6005 before, 5996
after** (-9). `check_prose_only.py`: "OK: only prose changed", no em-dash
warning. `tools/lint_source.py`: clean. Display equations: 1 before, 0 after
(the elementwise definition is now inline, per NDA-6).

## Before/after pairs

1. Before: "Rescaling images is a typical computation."
   After: "Let us take one humble computation, the rescaling of images, to
   see what is at stake."
2. Before: "Deep learning code computes with whole arrays in this way
   throughout, rather than with loops over their elements."
   After: "This is how deep learning code computes throughout: with whole
   arrays, and hardly ever with loops over their elements."
3. Before: "The weights of a model, by contrast, start at random values. If
   all the weights of a layer started at the same value, all its outputs would
   be equal and would receive equal updates during training, so they could
   never come to differ. Random initial values *break this symmetry*."
   After: "Most parameters of a model, however, must start at small random
   values. The reason has to wait until :numref:`sec_numerical_stability`; for
   now we only need to know how to draw them."
4. Before: "Random numbers differ from one run to the next. For debugging, and
   for results that others can reproduce, we fix the *seed* of the random
   number generator, the value from which it computes its sequence of
   numbers:"
   After: "Random numbers differ from one run to the next, which is a nuisance
   when we hunt for a bug or want others to reproduce a result. The numbers
   are, of course, not random at all: a generator computes them from a
   starting value, the *seed*, and the same seed gives the same numbers."
5. Before: "Until we load a real dataset, random integers from 0 to 255 stand
   in for the pixels, because only the shape and the type of the minibatch
   matter here:"
   After: "Until we load a real dataset, random integers from 0 to 255 stand
   in for the pixels. Such images show static rather than shirts, but only the
   shape and the type of the minibatch matter here:"
6. Before (Writing Elements had no shared prose; it opened with a tab note):
   "An index on the left side of an assignment writes elements in place. We
   display `x`, from which `X` was reshaped, as well:"
   After (shared prose): "So far we have only read elements. Can we also
   change them? Here the libraries part ways, by design. [...] Writing raises
   a question that reading could not. When we reshaped `x` into `X`, did the
   library copy the elements? It need not have [...] For immutable tensors the
   question is idle. For mutable ones we had better know the answer."
7. Before: "Elementwise operations are simple when the two tensors have the
   same shape. Model code, however, constantly combines a minibatch [...]"
   After: "Elementwise operations are simple when the two tensors have the
   same shape. Unfortunately, they rarely do. Model code constantly combines
   a minibatch [...]"
8. Before: "We can now create tensors, select and rearrange their elements,
   and compute with whole tensors at once, including tensors whose shapes
   differ, and we know when such computations allocate memory. The numbers in
   this section were made up."
   After: "The numbers in this section were made up. Real data arrive in
   files, often as tables with missing entries and with text in some of the
   columns, and they are rarely as tidy as the output of `arange`."

Protected and left verbatim: the broadcasting rule paragraph (one clause
added to its first sentence: "and it is worth knowing by heart"), the
"rule also predicts" paragraph, "The numbers in this section were made up."

## Repetition removed

- **Minibatch arithmetic** ($256 \times 28 \times 28 = 200{,}704$): this
  section owns it. It is now computed once, in the opening; "Shape" cites the
  count and says only that the channel contributes a factor of 1 (the second
  full product is gone). The opening minibatch is now tied to the shirt of the
  first sentence ("256 such photographs"), so the garment example is
  introduced once here.
- **Views / shared memory**: was raised in "Reshaping" (paragraph plus four
  tab notes), acted on two subsections later without comment, and met again
  in in-place updates and NumPy conversion with no link. Now defined once, in
  shared prose at the start of "Writing Elements" where the write happens; the
  conversion subsection refers back to it in one clause.
- **Closing recap** ("We can now ...") deleted; the Summary is the only recap.
- **Seed tab notes**: the sentence "`a` and `c` ... agree, whereas `b` ...
  differs" stood in all four tabs; it is now stated once in shared prose.
- **Tab prose trimmed** (guide §9): size notes for MXNet, TensorFlow and JAX
  (the code says it); `expand_dims` alternatives for MXNet and JAX; the
  `tf.function` memory-reuse parenthesis; the MXNet `asnumpy` note shortened;
  "which is why the `arange` cell above asks for `float32`" shortened in three
  tabs.

## Review findings

Applied:

- NDA-1: "(`ndim` in code)" added where *order* is defined.
- NDA-2: symmetry-breaking argument and the coined italic removed; one
  sentence with a forward reference. The "offsets that a layer adds" clause is
  gone from "Initial Values", and from the broadcasting motivation.
- NDA-3: the view paragraph and its tab notes moved from "Reshaping" to
  "Writing Elements" (prose only; no cell moved).
- NDA-4: output read out after the cell, per tab (PyTorch: `x[6]` is now 17,
  a view; MXNet: `x` still holds 6, a copy). Checked against both stored
  outputs.
- NDA-5 (prose part): the text now says where `x`, `y`, `X` are redefined
  (elementwise, broadcasting, concatenation).
- NDA-6: the cell comes first; the definition follows in one inline sentence.
- NDA-7: one parenthesis says that addition does not promote (`uint8` plus
  `uint8` stays `uint8`), preparing the exercise.
- NDA-8: marked as a forward pointer to :numref:`sec_autograd`.
- NDA-9: one tab sentence after the last `id` cell (MXNet/PyTorch: same
  object, holds the sum; TensorFlow/JAX: a new object). Checked against the
  four stored outputs (True, True, False, False).
- Structural: immutability is now named in shared prose, so the PyTorch
  reader meets it before the Summary.
- Exercise 6, JAX tab: broken sentence repaired ("and count").

Deferred:

- NDA-5 (renaming variables to `u`, `v`): needs code changes.
- Dropping the `A[0, 0] = 100` cell (proportion finding): code cell.
- Exercise 2 "token" gloss, and replacing exercise 5 with an indexing or
  masking problem: exercises are out of scope except for errors.
- Reordering the API walk: structure, not this pass.

## Factual claims added or changed

- 7 significant digits "is fewer digits than a pocket calculator shows, and
  it turns out to be enough" (new judgment; calculators show 8 to 12).
- The two additions over NumPy (accelerators, derivatives) "make deep
  learning practical" (new judgment).
- "Most parameters of a model ... must start at small random values", with
  the reason deferred to `sec_numerical_stability` (replaces the symmetry
  argument; that section has "Symmetry among Hidden Units").
- Pseudo-random numbers "are, of course, not random at all: a generator
  computes them from a starting value, the seed".
- "MXNet and PyTorch tensors are mutable ... TensorFlow and JAX tensors are
  immutable: once created, they never change" (was only in tab notes).
- "the sum of two `uint8` tensors is again `uint8`" (true in all four
  libraries; not shown by a cell in this section).
- "`images / 255` is, strictly speaking, not covered by the definition" of
  an elementwise operation (same-shape operands); resolved in Broadcasting.
- "many an error about mismatched shapes goes back to" the integer-index
  versus slice distinction (judgment).
- Removed claim: TensorFlow reuses the memory of intermediates inside
  `tf.function`.
- `ndim` as the code name for order (attribute exists in all four libraries).

## Slides that now disagree with the body (not edited)

- Slide "elementwise" still shows the display equation
  $c_i = f(u_i, v_i) \quad \textrm{for all } i$; the body has it inline.
  Same content, no contradiction.
- Broadcasting slide: "A layer adds its offsets to the outputs of every
  example this way." The body no longer mentions layers or offsets; it uses
  the column-means example only.
- No slide mentions symmetry breaking. No slide defines *view*; if the slides
  show the `X, x` cell, they could carry the new readout.
