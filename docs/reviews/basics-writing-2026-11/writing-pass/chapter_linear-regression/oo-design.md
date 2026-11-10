# Writing pass: `chapter_linear-regression/oo-design.md`

Prose only. No code cell, label, figure, exercise, or slide was changed.
`check_prose_only.py` ends "OK: only prose changed"; `tools/lint_source.py`
is clean.

## Words

Prose words (checker, whole file including exercises and slides): 3,749
before, 3,510 after (-239). All of the change is in the body before the
exercises, which is roughly 9% shorter although it gained a paragraph of
about 110 words that says what the section is and what may be skimmed.

## Before and after

1. Candour at the start (new paragraph after the figure; it replaces "This
   section writes the three base classes, and the next two sections fill
   them in for linear regression: a dataset, then a model together with the
   two loops inside the trainer.")

   > After: We should say plainly what kind of section this is. No model is
   > trained in it. We build the bench on which every later model is
   > trained, and a bench is not much to look at: the only outputs below
   > come from two toy classes, and several methods are declared here and
   > written only in :numref:`sec_linear_scratch`, where the first model is
   > trained. A reader impatient to see a loss go down may read this section
   > quickly and return to it whenever a later section calls something
   > unfamiliar. Three short passages should not be skipped: ...

2. > Before: Python looks up a method on the class at the moment it is
   > called, so `do` is available to the instance `a` that existed before
   > it. Inside `do`, `self` is that instance, and the method reads its
   > attribute `b` like a method written inside the class.
   >
   > After: Does the instance `a`, created before `do` existed, get the new
   > method? It does, since Python looks up a method on the class at the
   > moment of the call.

3. > Before: Here we only declare the method. Its implementation in
   > :numref:`sec_utils` uses Python's `inspect` module to read the
   > arguments of the method that called it, and the `d2l` library holds
   > the complete class.
   >
   > After: The cell above only declares the method. Its body uses Python's
   > `inspect` module to read the arguments of the method that called it,
   > which teaches nothing about training, so we keep it in
   > :numref:`sec_utils`. The `d2l` library holds the complete class, and
   > it is this class that the next cell inherits from.

4. > Before: Both steps *return* their result and do nothing else.
   >
   > After: Both steps *return* their result and do nothing else: they do
   > not print, plot, or keep a record. This restriction is the key decision
   > of the design, and we give the reasons for it under "Epoch Means"
   > below.

5. > Before: A step returns either the mean loss over the minibatch or a
   > dictionary of such minibatch means, for instance the loss and the
   > accuracy of a classifier in :numref:`chap_classification`. A dictionary
   > returned by `training_step` must contain the key `'loss'`, the quantity
   > that is differentiated. The values must be means, not sums: ...
   >
   > After: In this chapter a step returns one number, the mean loss over
   > the minibatch. From :numref:`sec_classification` on, a step may instead
   > return a dictionary of such means, with the loss under the key `'loss'`
   > and, say, the accuracy next to it.

6. > Before: the arguments `num_gpus` and `gradient_clip_val` serve later
   > chapters, which train on GPUs and clip large gradients.
   >
   > After: (The other two arguments can be ignored until they are needed:
   > `num_gpus` in :numref:`sec_use_gpu`, and `gradient_clip_val`, which
   > limits the size of large gradients, in :numref:`sec_rnn-scratch`.)

7. > Before: `record` appends each mean to `trainer.history`, a dictionary
   > of lists whose keys combine the part of the data with the name of the
   > quantity ... `fit` itself returns nothing.
   >
   > After: What is left when training ends? `fit` itself returns nothing.
   > Everything is in `trainer.history`, a dictionary of lists to which
   > `record` appends each mean.

8. > Before: We now have a training loop that every model in the book can
   > use, but nothing for it to run: no dataset, no model, and no body for
   > the two loops.
   >
   > After: This completes the bench. As we said, it has nothing to run yet:
   > no dataset, no model, and no body for the two loops.

## Repetition removed

- **Means, not sums.** Was argued under "Models" by appeal to a trainer the
  reader had not seen, and the weighting was explained again under "Epoch
  Means". Now stated once, under "Epoch Means", after the 1,000/32 example.
- **Host synchronization on a GPU.** `oo-design` is the earliest section of
  the chapter that explains it, so it keeps the explanation, cut from ten
  sentences to five and marked as not mattering before GPUs. The sentence
  about the device idling is gone. The scratch section already cites it in
  one clause.
- **Order of metrics in `record`.** One parenthetical sentence instead of a
  mid-paragraph explanation; the reason (stable curve colors) is kept.
- **Dictionary metrics / `as_metrics`.** Two paragraphs became one.
- **"`Module` is a subclass of ..."** was said in the shared prose and in
  each tab. The shared prose now names `d2l.nn_Module` once and each tab
  says which class it is.
- **Redrawing the board** was described under the loop and again under "The
  Progress Board"; the first is removed.
- **Summary** no longer repeats `evaluate` and "sums them on the device".
- The opening roadmap sentence and the closing paragraph said the same
  thing; the roadmap sentence is folded into the candid paragraph.

## Review findings

Applied (in prose):

- OO-1, partly: the section cannot be given something to run without code,
  but it now says so at the start and tells the reader what to skim.
- OO-2, partly: the stub cell stays (code), the prose explains why the body
  is elsewhere and that the next cell uses the library's class.
- OO-3: one clause per tab says what `d2l.nn_Module` is.
- OO-4: the description of the steps moved after the class; one short
  paragraph precedes it.
- OO-5: dictionaries reduced to two sentences with a pointer to
  `sec_classification`.
- OO-6: "the book build skips the intermediate redraws" deleted.
- OO-7: Lightning's `self.log` and the list of four parameter-saving
  functions deleted.
- OO-8: asynchronous execution cut by half, with pointers.
- OO-9: metric order reduced to one parenthesis.
- OO-10: "Optimizers and Schedules" shared prose says the hook is for later
  and points to `sec_scheduler`; PyTorch tab one sentence, JAX tab from 20
  lines to 8, TensorFlow and MXNet one sentence each.
- OO-11: key-naming detail of `evaluate` deleted; the subsection is three
  sentences with a pointer to `sec_softmax_scratch`.
- Tab prose under "Models" trimmed (TensorFlow, JAX).

Deferred:

- OO-1 (a bare loop that runs), the reordering of the chapter, removing the
  `HyperParameters` stub cell, per-tab class headers, moving `evaluate`,
  the schedule hook and `as_metrics` to their first use: all need code to
  change or move.
- "Mechanism before need" (Utilities first): would reorder code cells.
- Exercises 5 to 7 belong after the scratch section, and one of exercises 1
  and 2 would do: exercises were out of scope.

## Claims added or changed

- "the only outputs below come from two toy classes" (true of the stored
  outputs: cells for classes `A` and `B`).
- Pointers to first use, each checked with `git grep`: dictionary metrics
  in `sec_classification` (the `Classifier` steps return `'loss'` and
  `'acc'`); `num_gpus` in `sec_use_gpu`; `gradient_clip_val` in
  `sec_rnn-scratch`; `trainer.evaluate` first called in
  `sec_softmax_scratch`; schedules "are the subject of" `sec_scheduler`.
  Note that `num_gpus=1` appears in passing in `model-construction.md`
  before `sec_use_gpu`, which is where it is explained.
- "this chapter does without them [schedules]": no section of the chapter
  uses a schedule (grep).
- "`prepare_model` stores the model" replaces "readies the model (later it
  moves the model to a GPU)".
- "[the speed argument] will not matter before we train on a GPU".
- "`record` sorts the metrics, the loss first" (the code sorts with the key
  `(name != 'loss', name)`).
- JAX tab: the views "differ only for layers such as dropout, which behave
  differently during training" (the cell sets `deterministic` and
  `use_running_average`); the detailed description of the cache key and the
  caveat on `configure_optimizers` are gone.
- Removed claims: the comparison with Lightning's `self.log`; the list of
  parameter-saving functions; the sentence that the device idles while
  Python waits; the unprefixed keys returned by `evaluate`.

## Slides that now say more than the body

No slide contradicts the body, but several now carry material that the
body no longer has:

- "By the End of This Deck You Can": "explain why the trainer reads losses
  once per epoch" is now supported by five sentences only.
- The slide comparing with Lightning (`self.log('train_loss', loss)`) and
  the bullet "It saves with its framework's tools: `state_dict`,
  `save_weights`, ..." have no counterpart in the body.
- The slide bullet "The loss comes first, then the other metrics
  alphabetically" is more detailed than the body's parenthesis.
- The schedule slides ("`scheduler.step()` after every update", "The last
  `tx` is cached by `jax.tree.structure(model)` and the method") describe
  details the body now leaves to the code and to the scratch section.
- The slide "A Summed Loss Would Enter the Epoch Mean 32× Too Large" is
  filed under the kicker "Module"; the body now makes this point under
  "Epoch Means".

## Consistency with neighbouring sections

- `synthetic-regression-data.md` opens "The classes of
  :numref:`sec_oo-design` can train a model as soon as they are given
  data". This section says, twice, that the two inner loops have no body
  yet. One of the two should give way; I would change the neighbour.
- `linear-regression-scratch.md` cites this section for the cost of
  conversions, `as_metrics`, the schedule hook, and the JAX views; all four
  are still here.
