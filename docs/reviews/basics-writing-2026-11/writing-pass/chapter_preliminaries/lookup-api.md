# Writing pass: `chapter_preliminaries/lookup-api.md` (Documentation)

Prose only. No code cell, label, figure, exercise, or slide was changed.
`check_prose_only.py` ends "OK: only prose changed"; `tools/lint_source.py`
is clean.

## Words

Prose words (whole file to the slides marker, all four tabs): 3,561 before,
3,548 after (-13). The single-tab view is shorter by more than that in the
MXNet, TensorFlow and JAX tabs (an opening tab note and a closing sentence
removed from each) and about level in PyTorch.

## What changed, in one paragraph

The shared prose used to promise that `ones(2, 3)` might not be a
$2 \times 3$ matrix, which is false on the page for the PyTorch reader. It
now states a fact that is true in every tab and is the actual surprise: the
same call has three different outcomes across the book's four frameworks
(PyTorch returns the matrix, MXNet and JAX raise an error, TensorFlow
silently returns two `int32` ones). The reader's question becomes "which of
these does my framework do, and how could I have known?". The PyTorch tab
now says plainly that PyTorch is the one framework in which the call means
what it appears to mean. The four per-tab sentences about `ones((2, 3))`
became one shared sentence: the tuple is the one spelling all four
frameworks read the same way.

## Before and after

1. Before: "Every project, however, soon needs a function that the book
   never mentions, or uses a familiar one in an unfamiliar way. No book can
   record the details of every function: the main module ..."
   After: "This courtesy cannot last. Every project soon needs a function
   that the book never mentions, or uses a familiar one in an unfamiliar
   way, and no book can keep up: the main module ..."

2. Before: "The call `ones(2, 3)` looks like a request for a $2 \times 3$
   matrix of ones, but whether it is one depends on how the function reads
   its arguments, and only the function's signature says what each argument
   means."
   After: "Take `ones(2, 3)`, which looks like a request for a
   $2 \times 3$ matrix of ones. Written for the four frameworks of this
   book, this one call has three different outcomes. PyTorch returns the
   matrix. MXNet and JAX raise an error. TensorFlow runs without complaint
   and returns a vector of two integer ones. Of the two ways to go wrong,
   the error is the kinder one, since it announces itself. Nothing in the
   call tells us which outcome to expect. The function's signature does."

3. Before: "All four steps work inside a notebook, and at the end of the
   section they explain the result above."
   After: "All four steps work inside a notebook. By the end of the section
   they explain what our own framework does with `ones(2, 3)`, and why."

4. Before: "We set the samplers aside, since the first exercise completes
   the search for a gamma sampler, and inspect a function that we think we
   already know: `ones`, which we have used throughout this chapter."
   After: "The list answers our question, or at least its first twenty
   names do, and among them is an entry for the gamma distribution. How to
   call it is another matter: libraries disagree on whether a gamma
   distribution is described by a rate or by a scale. That search needs a
   little statistics, and we leave it to the first exercise. For the worked
   example we take a function whose result can be checked at a glance, and
   which we think we already know: `ones`."

5. Before: "The first lines of the output show the function's *signature*,
   the list of arguments that it accepts. Three conventions of Python help
   to read it."
   After: "The output is long, and for now only its beginning matters. The
   first lines show the function's *signature*, the list of arguments that
   it accepts. Three conventions of Python are all we need to read one."

6. Before (PyTorch tab): "The prediction holds: a $2 \times 3$ matrix of
   `float32` ones. The last line of the output shows that
   `torch.ones((2, 3))`, with the shape passed as one tuple, returns the
   same matrix. The docstring explains why: ..."
   After: "The prediction holds: a $2 \times 3$ matrix of `float32` ones.
   PyTorch is the one framework of the four in which the call means what it
   appears to mean. The docstring also explains why the tuple works: ..."

7. Before: "More generally, when a result contradicts the prediction, ..."
   After: "With the shape passed as one tuple, the call returns the
   $2 \times 3$ matrix. This is the one spelling that all four frameworks
   read in the same way. When a result contradicts the prediction, ..."

8. Before: "The four steps let us find out what a function does without
   leaving the notebook, and the last step makes the answer reliable,
   whether it came from memory, from the documentation, or from an
   assistant. They complete the tools of this chapter: ..."
   After: "With this procedure the tools of the chapter are complete:
   tensors, tables, linear algebra, derivatives, probability, and a way to
   find what the book leaves out."

## Repetition removed

- The outcome of `ones(2, 3)` was told in an opening tab note, in the
  prediction tab note, and in the verification tab note of each framework.
  The four opening tab notes are gone; the shared prose states the four
  outcomes once.
- "The last line of the output shows the call that we meant: `X.ones((2, 3))`,
  with the shape passed as one tuple, returns a $2 \times 3$ matrix" stood
  in all four tabs. It is now one shared sentence before the cell and one
  after.
- The closing paragraph restated the four steps and the role of the last
  one, which the Summary states again. The closing keeps only the list that
  ends the chapter (DOC-5 asked to keep it).
- No fact here is owned by an earlier section of the chapter; nothing had to
  be reduced to a reference.

## Review findings

Applied:

- DOC-1 (prose part): shared prose is now true in every tab; framework
  surprises sit in the tab notes after the cells.
- DOC-2 (prose part): the section says what it does with the gamma search
  (the list finds the entry; rate versus scale is exercise 1, because it
  needs sample statistics) instead of "we set the samplers aside".
- DOC-3: one parenthesis in the PyTorch tab on the printed `*size, *`.
- DOC-4 (prose part): one sentence telling the reader that only the
  beginning of the long `help` output matters for now.
- DOC-5: closing list kept.

Deferred:

- DOC-1/DOC-2, full fix (carry the gamma sampler, or a per-framework
  surprising call, through all four steps): needs new cells and outputs.
  For PyTorch the investigation still confirms the obvious; the text now
  admits this instead of hiding it.
- DOC-4, code part (print only the first lines, or `inspect.signature`):
  code change.
- Exercise "Checking a coding assistant's answer", part 2: exercises are out
  of scope except for errors.
- Structural note (move the section to the Tools part): not a prose matter.

## Factual claims added or changed

- "This one call has three different outcomes: PyTorch returns the matrix,
  MXNet and JAX raise an error, TensorFlow ... returns a vector of two
  integer ones." Checked against the stored outputs of all four frameworks
  (`torch.Size([2, 3])` `float32`; MXNet and JAX `TypeError: Cannot
  interpret '3' as a data type`; TensorFlow shape `(2,)`, `int32`).
- "With the shape passed as one tuple, the call returns the $2 \times 3$
  matrix ... the one spelling that all four frameworks read in the same
  way." Checked against the last output line of all four stores. "The one
  spelling" is relative to the two spellings the section discusses.
- "The list ... among [its first twenty names] is an entry for the gamma
  distribution." True of all four stored lists (`Gamma` is the twentieth
  name in PyTorch, so this depends on the library's name list not growing
  before it).
- "Libraries disagree on whether a gamma distribution is described by a
  rate or by a scale." General statement (PyTorch and TensorFlow use a rate,
  MXNet and NumPy a scale; JAX's sampler takes neither). Not shown on the
  page; exercise 1 has the reader find it.
- PyTorch tab: "Python itself would reject two stars in one signature.
  After `*size` every argument is keyword-only anyway." Checked: `def
  f(*size, *, out=None)` is a `SyntaxError`.
- PyTorch tab: the keyword spelling `ones(shape=(2, 3))` is one "which NumPy
  accepts". Checked for NumPy. That `torch.ones(shape=(2, 3))` fails was
  already claimed (and is on the slide); no PyTorch environment on this
  machine to re-run it.
- Dropped: "`ones`, which we have used throughout this chapter".

## Slides that now differ from the body

No slide contradicts the body. Two differences of emphasis:

- The PyTorch "problem" slide leads with `torch.ones(shape=(2, 3))` failing.
  The body's opening no longer mentions that call (it appears only in the
  signature tab note); the body's opening problem is now the three outcomes
  across frameworks, which no slide shows.
- The slide cover kicker reads "§2.7" (rewritten automatically by
  `gen_slides.py`).
