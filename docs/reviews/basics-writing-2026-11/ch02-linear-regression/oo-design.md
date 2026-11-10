# Object-Oriented Design for Implementation

Source: `chapter_linear-regression/oo-design.md`, about 2,900 words of prose before the exercises (all tabs; roughly 2,300 for the PyTorch reader).

## Verdict

The sentences are clear and the opening is the right one: a six-line training loop, the observation that only three of its lines change between models, and the count of copies the book would otherwise carry. After that the section stops teaching and starts specifying. It documents an API contract (what a step may return, how the trainer weights minibatches, in which order curves are drawn, how each framework attaches a learning-rate schedule) for a reader who has not yet trained a model, and it runs nothing: the only outputs on the page are `Instance attribute "b" is 1` and `self.a = 1 self.b = 2`. Six methods are left as `raise NotImplementedError`. As prose about software it is competent; as the second section of the first modelling chapter it is the weakest part of the chapter and the place where a reader is most likely to stop. The single most important fix is to cut it to what chapter 2 uses and to place it after the reader has seen a bare training loop run (see the chapter file for the proposed order).

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 3 | 3 | 2 | 4 | 2 |

## What works

- **The opening.** The pseudocode loop with its three comments (`# data`, `# model and loss`, `# optimizer`), then "Only the lines with comments depend on the problem; the loop around them never changes", then the argument from maintenance: "a loop written out in each would exist in close to a hundred copies". This is a problem the reader can feel. Keep all of it.
- **The mapping paragraph** after the bullet list ("the `DataModule` supplies `train_data` and `val_data`, the `Module` supplies `net`, `loss_fn`, and the update rule") ties the three classes back to the lines of the loop.
- **Honesty about `add_to_class`**: "This is notebook scaffolding, not a design we recommend: in an ordinary Python module, a class is defined in one place."
- **The epoch-mean example**: "1,000 examples in minibatches of 32 make 31 full minibatches and one of 8 ... each of the last 8 examples would count four times as much". A small worked number for a real pitfall.
- **Exercise 4** ("Reading the training curves") asks why the training curve can lie above the validation curve early in training. It tests understanding of how the two means are computed and is the best exercise of the section.

## Structural findings

- **Nothing runs (major).** The section defines `Module`, `DataModule`, `Trainer` and `ProgressBoard`, and leaves `loss`, `configure_optimizers`, `fit_epoch`, `validate`, `show` and `save_hyperparameters` unimplemented (12 occurrences of `NotImplementedError` across the tabs). The closing admits it: "nothing for it to run: no dataset, no model, and no body for the two loops". A reader cannot check any claim of the section against behaviour, which is the opposite of the method the rest of the chapter teaches. See OO-1.
- **The section is written for chapters 3 to 13 (major).** Dictionary-valued steps and accuracy, `num_gpus`, `gradient_clip_val`, `evaluate`, learning-rate schedules, asynchronous GPUs, parameter saving in four frameworks: none is used in this chapter, which trains three parameters on a CPU with one loss and a constant learning rate. Each should arrive in the section that first needs it. See OO-5, OO-7, OO-8, OO-10.
- **Mechanism before need.** "Utilities" comes first: `add_to_class` and `HyperParameters` are explained before the reader has seen a class that needs them. They would be better as two short remarks at first use.
- **Framework tabs carry internals.** The JAX tab under "Optimizers and Schedules" is 20 lines about an optimizer cache keyed on `jax.tree.structure(model)`; the TensorFlow tab under "Models" explains a Keras 3 attribute workaround. These are maintainer notes (writing guide, section 4.5) and belong in code comments.

## Passage-level findings

**OO-1** (major, pedagogy). Closing paragraph before the summary.
Quote: "nothing for it to run: no dataset, no model, and no body for the two loops."
See the first structural finding. Fix: give the section something to run. The cheapest version: after the opening pseudocode, write the real loop for one framework in eight lines on a toy problem the reader already has (the three-point line model of the previous section, or twenty points on $y = 2x$), run it, print the loss. Then introduce the three classes as a rearrangement of code that works, with every method given a body. The stronger version is the reordering described in the chapter file.

**OO-2** (moderate, explanation). Under "Utilities".
Quote: "Here we only declare the method. Its implementation in :numref:`sec_utils`"
The reader sees `class HyperParameters` whose only method raises `NotImplementedError`, and in the next cell a class that inherits from `d2l.HyperParameters` and works. Two classes of the same name, one of them a stub shown only to be replaced. Fix: do not display the stub. Say "The `d2l` library provides a base class `HyperParameters` whose method `save_hyperparameters` stores the arguments of the constructor that calls it", then show class `B`.

**OO-3** (moderate, explanation). Under "Models".
Quote: "`Module` inherits from `d2l.nn_Module`, the library's name for the base"
An indirection (`nn_Module`) that exists so that one cell serves two frameworks. The PyTorch reader would understand `nn.Module` at once and has to decode this instead. Fix: per-tab class headers (`class Module(nn.Module, d2l.HyperParameters)`), or one clause in each tab.

**OO-4** (moderate, pedagogy). Under "Models".
Quote: "`self.loss(self(*batch[:-1]), batch[-1])`: calling the model on the inputs"
Three paragraphs describe the code before the code appears, partly by quoting it. The reader has to imagine a class and then read it. Fix: show the class first, then two sentences: what `training_step` computes, and that steps return a value and have no side effects.

**OO-5** (moderate, pedagogy). Under "Models".
Quote: "must contain the key `'loss'`, the quantity that is differentiated. The"
A paragraph on steps that return dictionaries of metrics, with a rule (means, not sums) justified by how the trainer, which the reader has not seen, averages them. No model in this chapter returns a dictionary. Fix: here a step returns the loss. Introduce dictionaries, `as_metrics` and the means rule in the classification chapter, where accuracy is the second metric.

**OO-6** (moderate, language). Under "The Training Loop and the History".
Quote: "`show()` draws it once more after training; the book build skips the"
"the book build skips the intermediate redraws" is a note about the authors' build system (writing guide, section 4.5). Delete.

**OO-7** (moderate, thought). Under "Epoch Means".
Quote: "machinery. Lightning, whose design we otherwise follow, lets a step log its"
A design defence addressed to readers who know Lightning's `self.log`, followed by a list of four frameworks' functions for saving parameters (`state_dict`, `save_weights`, `nnx.state`, `save_parameters`). The reader of chapter 2 knows none of these. Fix: one sentence ("A model does not refer to the trainer, so any loop can train or evaluate it") and no list.

**OO-8** (moderate, pedagogy). Under "Epoch Means".
Quote: "The second reason is speed. A GPU runs *asynchronously*: Python places"
A paragraph on asynchronous execution and host synchronization, in a chapter that never uses a GPU. It is repeated in `linear-regression-scratch.md` under "Epochs and Averages". Fix: remove from both; the Computation chapter has a section on it (`subsec_async_dispatch`) and can explain why the trainer sums on the device.

**OO-9** (minor, language). Under "The Training Loop and the History".
Quote: "in alphabetical order, so that a curve keeps its color and line style on the"
An implementation detail of plotting (the order in which metrics are recorded fixes the colour of a curve) explained in the main text, and explained again for `Averager` in the scratch section. A code comment suffices.

**OO-10** (major, pedagogy). Under "Optimizers and Schedules".
Quote: "`configure_optimizers` may therefore return a pair `(optimizer, scheduler)`."
No section of this chapter uses a learning-rate schedule. The subsection exists to install a hook, and its JAX tab ("most recent transformation in `optimizer_cache`. The key is the structure of") spends twenty lines on compilation caching. Fix: move the whole subsection to the section that first uses a schedule. If the hook must be defined here for the library build, define it in a cell with a one-line comment and no prose.

**OO-11** (minor, language). Under "Evaluation".
Quote: "`{'loss': ...}` rather than `'val_loss'`."
Key-naming trivia for a method (`evaluate`) that this chapter does not call. Move with the method to its first use.

## Exercises

Seven problems. Exercises 5, 6 and 7 each begin "Once :numref:`sec_linear_scratch` supplies ..." and cannot be attempted until two sections later; they belong at the end of the scratch section. Exercise 7 also needs a GPU and a model with $10^4$ features. Exercises 1 and 2 test Python mechanics (`setattr` on a class versus an instance, `inspect`); they are well made but teach no machine learning, and one of the two is enough. Exercise 3 (optimizer on the model or on the trainer) is a good design question. Exercise 4 is the keeper. After moving three and merging two, the section would fall below the house minimum of five; that is a sign the section has little to exercise, not a reason to add problems.

## Still open from October

"The `oo-design` section interrupts regression with tracing, graph breaks, and asynchronous devices." Tracing and graph breaks are gone from the PyTorch reader's path; asynchronous devices remain (OO-8), and the interruption itself remains. "Cut: the `ProgressBoard` scheduling discussion" is partly done; the schedule hook is still here (OO-10).
