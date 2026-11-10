# Writing pass: `chapter_builders-guide/model-construction.md` (5.1)

Prose only. No code cell, label, figure path, or slide changed.
Checker: "OK: only prose changed"; `lint_source.py` clean.

**Prose words (checker count, all four tabs): 6476 before, 6417 after (-59).**
The section is first in the chapter, so it owns every fact it explains; the
reduction is small because the large cuts the review proposes (three of the
four registration cells, half of lazy initialization) remove code cells and
were out of scope.

## Before/after pairs

1. Before: "This section shows how registration works, how it fails, and how
   to write the forward computation of a module yourself."
   After: "This section is a practical one. It shows how registration works
   and how it fails, how to write the forward computation of a module
   ourselves, and how to describe a whole model by a handful of numbers."
2. Before: "A layer that is not registered is skipped by all of them."
   After: "For all three, a layer that is not in the table does not exist."
3. Before: "Now suppose we store the layers in an ordinary Python list."
   After: "Now suppose that we store the layers in an ordinary Python list,
   as anyone would who had not read the previous subsection."
4. Before: "Here the optimizer at least complains: `torch.optim.SGD` refuses
   an empty list of parameters. A model that is only partly registered is
   worse."
   After: "How would we notice? Sometimes the mistake is loud.
   `torch.optim.SGD` refuses an empty list of parameters, so this model
   cannot even be handed to an optimizer. The dangerous case is the model
   that is only partly registered."
5. Before: "The most important example in this book is the *residual
   connection*, which adds the input of a block to its output."
   After: "Let us start with the simplest module whose forward method is not
   a chain. A *residual connection* adds the input of a block to its output
   :cite:`He.Zhang.Ren.ea.2016*1`."
6. Before: "It is missing from `state_dict()` as well, so it is not saved
   with the model, and `.to(device)` does not move it. Some state must travel
   with the model without being trained."
   After: "Unfortunately it is missing from `state_dict()` as well, so it is
   not saved with the model, and `.to(device)` does not move it. A single
   number could have been a Python float, which needs neither. An array
   computed from data could not: it must travel with the model without being
   trained."
7. Before: "A trained model is a set of arrays plus the code that knows what
   to do with them. The arrays alone do not say how to rebuild the model."
   After: "Could we rebuild a model from its saved weights alone? Not quite.
   A trained model is a set of arrays plus the code that knows what to do
   with them."
8. Before: "We can now build a model of any size from modules, find all of
   its parameters, and rebuild it from a few numbers. What we have not yet
   done is look at the parameters themselves: ..."
   After: "Throughout this section the parameters were things to be found
   and counted. We have not yet looked at them: ..."

## Repetition removed

- "The forward method is ordinary Python" was said at the head of "Forward
  Computation", again at the head of "Eager and Compiled Execution", and in
  the summary. The second statement is gone.
- "The block outputs its input plus a correction" / "each block only has to
  learn a correction": merged into one sentence.
- "The factor enters the computation but is not a parameter ... Some state
  must travel with the model without being trained" opened and closed all
  four tabs of "Constants"; each tab now starts with its own API fact.
- "The input width of each layer is the output width of the one before" was
  in the shapes tabs and again in the dry-run paragraph; removed from the
  latter.
- PyTorch: `Trainer` and `weight_init` were explained in "Shapes from Data"
  and again in "Initialization after a Dry Run"; now only in the latter.
- One-sentence tabs removed: "In PyTorch/Gluon the forward method is
  `forward`" (folded into shared prose), "The module below keeps its layers
  in a plain list:" (PyTorch, JAX).
- Formulaic closing ("We can now ...") replaced; summary reduced from six
  facts to three plus one sentence.

## Review findings

Applied (prose): MC-1 (caption now says only what the picture shows; the
Gluon and NNX naming remarks moved to the tabs before the cell that prints
the names; the Keras remark was already there), MC-2, MC-4 (PyTorch and
Gluon paragraphs split in three and ordered loud then silent), MC-5 (the
promise of "three times ... a loop whose length depends on the data" is
gone; the intro now promises two examples and the cost of compilation),
MC-6 (superlative and Jacobian sentence removed), MC-7 (prose only: the text
now admits that a scalar could be a float and says what kind of state does
need a registered place), MC-8, MC-9 (verdict paragraph at the end of "Lazy
Initialization"), MC-10, MC-11.

Deferred:
- MC-3: running the partly registered model needs a new cell. The case is
  still introduced with "Suppose".
- MC-5, code half: giving PyTorch and Gluon the data-dependent loop needs a
  new cell.
- MC-7, code half: replacing the scalar `alpha` by a per-feature vector
  changes cells and outputs.
- Chapter-review cuts for 5.1 (three of the four registration cells): cells.
- Structural: moving "Lazy Initialization" out of the arc; exercise 5 cut
  (exercises untouched).

## Claims added or changed

- "A single number could have been a Python float, which needs neither
  [saving nor moving]" (MXNet, PyTorch, TensorFlow tabs). New; it is what
  the JAX tab's code does.
- "An array computed from data ... must travel with the model without being
  trained" replaces "Some state must travel ...". Narrower, points at the
  `Whitener` example of `sec_parameters`.
- PyTorch dry-run tab: "The `Trainer` has done this before every training
  run since :numref:`sec_linear_concise`, whenever the model still had lazy
  parameters ... `apply_init` ... we have not shown so far." The first half
  restates the old text; "not shown so far" was checked with grep over
  chapters 2 to 4 (only `weight_init` appears there).
- "A model whose parameters are to be counted, initialized by a rule, or
  built from a config should state both widths of every layer, as the models
  in the rest of this section do." New recommendation. I limited it to "this
  section" because `custom-layers`, `saving-loading` and
  `gpus-devices-memory` still use `LazyLinear`.
- "`torch.optim.SGD` refuses an empty list of parameters, so this model
  cannot even be handed to an optimizer": the second clause is new and
  follows from the first.
- Config closing: "Rebuilding a model from a checkpoint needs the weights,
  the config, and the same `build` function and block classes.
  :numref:`sec_read_write` saves the first two together in one file; the code
  we must keep ourselves." The old sentence gave the need for code as the
  reason for saving the config, which does not follow.
- Removed: the Jacobian $\mathbf{I} + \mathbf{J}_f$ remark (its citation
  moved to the definition of the residual connection), and "The most
  important example in this book".
- Summary: "in the worst case nothing tells us" (was unqualified "never
  trained or saved"); true of the partly registered PyTorch model on a CPU,
  while Keras and NNX catch it.

## Slides that now disagree with the body

- Residual-block slide, bullet "Jacobian $\mathbf{I} + \mathbf{J}_f$: the
  gradient at $\mathbf{y}$ ...": the body no longer contains this.
- Nothing else. The slides never used the caption's framework remarks, and
  the lazy-layer, dry-run and count slides agree with the body.
