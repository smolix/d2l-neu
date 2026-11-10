# Writing pass: `chapter_multilayer-perceptrons/dropout.md` (section 4.6)

Prose only. No code cell, label, figure, or slide was changed.
`check_prose_only.py` ends "OK: only prose changed"; `lint_source.py` is clean.

## Words

- Checker count (all prose in the file): 4,023 before, 3,825 after (−198).
- Body prose to the Summary inclusive, without exercises and slides: about
  3,270 before, 3,070 after (−6%).

The review asked for about a third less. That is not reachable in a prose-only
pass: the three training runs, the masked-accuracy cells and the nine tab notes
stay, and the brief for this pass asked for the tie to be argued and the three
accounts to be kept as accounts, which costs about 150 new words. What was cut:
the additive-noise derivation, the learning-rate digression, most of "Use in
Practice", the recap closing. A third would need one of: dropping the concise
training run, moving "Interpretations" to further reading, or replacing the
experiment (DO-1).

## Before and after

1. Opening.
   - Before: "Neither constrains how the hidden units divide the work among
     themselves."
   - After: "Neither has anything to say about how the hidden units divide the
     work among themselves."
2. Roadmap.
   - Before: "We train that MLP with and without dropout on Fashion-MNIST and
     measure what dropout changes in the trained network. Finally, we compare
     the explanations that have been offered for its effect."
   - After: "The outcome is less flattering than this motivation suggests, and
     it will make us careful with the explanations that have been offered for
     dropout's effect."
3. Additive noise (derivation with a display equation, about 130 words).
   - Before: "For a linear model this is exactly weight decay. Let the input
     $\mathbf{x}$ be perturbed to ... Bishop's analysis does not apply to such
     large, multiplicative noise, so it motivates dropout without explaining
     it."
   - After: "For a linear model this turns out to be nothing new. Add to the
     input a noise vector $\boldsymbol{\epsilon}$ whose entries are independent
     with mean zero and variance $s^2$, and the expected squared error grows by
     $s^2\|\mathbf{w}\|^2$, the variance of
     $\mathbf{w}^\top\boldsymbol{\epsilon}$: training on noisy inputs is weight
     decay. ... Noise of that size lies outside Bishop's analysis, so we take
     from it the motivation and no more."
4. Test time.
   - Before: "By the rescaling, each hidden activation then equals the expected
     value of its dropped version. This statement concerns one unit."
   - After: "Does it compute the average of the random networks seen in
     training? For a single unit it does, since each hidden activation equals
     the expected value of its dropped version."
5. Training curve.
   - Before: "For the first few epochs it lies above the validation loss. It
     ends somewhat below it, at about 0.32, while the validation loss levels
     off near 0.35."
   - After: "For the first few epochs the training loss therefore lies above
     the validation loss, which looks like a bug and is not. It ends a little
     below it."
6. The result (DO-1).
   - Before: "On Fashion-MNIST, with this network and 20 epochs of training,
     dropout does not make the MLP more accurate. With the masks switched on,
     however, the two networks behave very differently."
   - After: "This is a tie. ... Should this surprise us? Not really. A
     regularizer can only remove overfitting that is there, and this network
     has little to offer."
7. The masked accuracies (DO-2).
   - Before: "Its units have learned to work in the combinations present during
     training and fail when some of their partners are missing: they are
     co-adapted. Dropout removes this dependence among the units, and here it
     does so without changing the accuracy of the full network."
   - After: "It is tempting to read this as co-adaptation made visible: units
     that fail when their partners are missing. The measurement does not carry
     that much. ... What we have shown is that dropout makes a network robust
     to dropout."
8. Interpretations.
   - Before: "Three explanations have been offered; each accounts for part of
     the mechanism, and none is complete."
   - After: "Why, then, does every framework ship a dropout layer? Because
     elsewhere the method has helped ... Our tie cannot tell us why. Three
     accounts are standard, and we present them as accounts: each describes
     part of the mechanism, and none is complete."

## Repetition removed

- Noise as weight decay was derived twice (additive noise, then dropout noise).
  The additive case is now one sentence in "Simplicity and Robustness"; "Noise
  Injection" keeps the one derivation and refers back in a clause.
- The epoch-to-epoch wobble at a constant learning rate, with decay, warmup and
  `sec_scheduler`: owned by `sec_mlp-implementation`, which already explains
  it. Reduced to one clause with a reference; the schedule remedy is gone.
- He initialization (`subsec_he_init`): one clause in "The Model"; the four
  concise tab notes no longer each repeat "normal weights with variance
  $2/n_\textrm{in}$" (stated once in the shared prose).
- The rescaling was explained before and after :eqref:`eq_dropout`; once now.
- Loss values 0.32 and 0.35 (stated, then repeated in the baseline paragraph):
  only the baseline's 0.25 remains.
- The 8-to-12-point loss was stated in "Accuracy and Robustness" and again in
  "Co-adaptation"; once now.
- The closing paragraph recapped the three regularizers ("Dropout completes the
  regularizers of this chapter"); removed. The Summary is the only recap, and
  the section ends on the open question.

## Review findings

Applied:
- DO-1 (prose part): the result is called a tie; a short paragraph argues why
  a tie is unsurprising; the interpretations are introduced as accounts of an
  effect reported elsewhere.
- DO-2: the co-adaptation inference is withdrawn in both places.
- DO-3: learning-rate digression reduced to a clause.
- DO-4: additive-noise derivation reduced to a sentence.
- DO-5: "Use in Practice" cut from about 160 to about 100 words; the list of
  later CNN techniques, stochastic depth, `sec_resnet`,
  `sec_training_recipes` and the transformer detail are gone. One sentence on
  the mismatch before normalization layers stays because exercise 2 refers to
  it.
- DO-6: loss values 0.32 and 0.35 replaced by the ordering of the curves.
- DO-7: the "regularizers of this chapter" sentence is gone.
- DO-8: exercise 2 counts hidden layers as $K - 1$ (notation error; the only
  change to the exercises).
- DO-9: the original scheme is described first, then inverted dropout is named.

Deferred:
- DO-1 (experiment): a setting in which the baseline overfits, for example the
  noisy-label subset of `subsec_early_stopping`. Needs a new run. The closing
  now points to the exercise that does it.
- Structural: three 20-epoch runs before the comparison, and whether the
  concise run is needed. Code.
- Exercise 3 as the in-text experiment: follows from DO-1.

## Claims added or changed

- "This is a tie" and "which of them is ahead changes from run to run" (the
  latter kept from the old text). Note that in all four stored runs dropout is
  ahead, by 0.6 to 2.1 points (PyTorch 0.861/0.845, TensorFlow 0.878/0.857,
  JAX 0.873/0.867, MXNet 0.873/0.856). That is within the epoch-to-epoch
  movement, but four out of four is not what "changes from run to run"
  suggests. I could not rerun; if the authors' repeated runs do not show the
  order flipping, this sentence and "a tie" need softening to "a point or so,
  no more than a single epoch's movement".
- New: the baseline "has little [overfitting] to offer": "the gap between its
  training loss and its validation loss is modest even without dropout, and
  the validation loss, for all its jumps, was not yet rising when we stopped".
  Checked against the stored baseline plot of all four frameworks (training
  loss about 0.25, validation loss about 0.35 to 0.40, no upward trend).
- New: the baseline's final training loss is "lower than the training loss of
  the run with dropout" (0.25 against about 0.32 in all four plots).
- New: masking half a layer and doubling the rest "inflates the variance of
  every pre-activation in the next layer, whether or not any unit depends on a
  particular partner" (follows from the variance formula of the section).
- New: "What we have shown is that dropout makes a network robust to dropout."
- Changed: co-adaptation is "plausible, and our experiment neither supports
  nor refutes it" (was: the measurement "confirm[s] the effect that dropout
  was designed to have").
- New: many more parameters than examples and many passes over each example
  "is the setting in which we should expect [dropout] to matter" (the review's
  suggested sentence, worded as an expectation).
- Changed: later architectures "use it at small probabilities, around 0.1"
  (was specific to transformers' attention weights and layer outputs).
- Changed: the dropout-before-batch-norm statement now says only that
  normalizing layers "are sensitive to this mismatch" (was "this mismatch
  hurts accuracy").
- Changed: the training loss "ends a little below" the validation loss (was
  0.32 against 0.35).
- Removed claims: later CNNs rely on augmentation, batch normalization and
  weight decay; stochastic depth; both versions of dropout "give the next
  layer inputs of the same expected size".

## Slides that now disagree with the body

- "Without Dropout, Removing Units Costs 8 to 12 Points", note: "Units trained
  without dropout work only in the combinations they were trained in: they
  are co-adapted. Dropout removes this dependence". The body now says the
  measurement shows robustness to the masks and cannot show co-adaptation.
- "Three Explanations, None Complete", table row co-adaptation: "measured, yet
  same accuracy here". The body says it was not measured.
- "Same Accuracy with and without Dropout": the bullet "Fix: decay the
  learning rate toward the end (and warm it up at the start)" has no
  counterpart in the body any more, and the slide lacks the body's reason for
  the tie (little overfitting to remove).
- "Dropout Today: Sparingly, at About 0.1, or Not at All": the bullets on
  later CNNs and on transformers' attention weights go beyond what the body
  now says.
- "Training: 20 Epochs" and the recap are consistent. The slide "For a Linear
  Model, Input Noise Is Weight Decay" carries the derivation that the body now
  states in one sentence; consistent, but the slide is now the only place the
  algebra appears.
