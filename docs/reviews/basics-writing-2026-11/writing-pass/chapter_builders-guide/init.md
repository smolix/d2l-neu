# Writing pass: `chapter_builders-guide/init.md` (5.3 Initialization)

Prose words (checker): **5006 before, 4806 after** (-200). Checker: "OK: only
prose changed". `tools/lint_source.py`: clean. No code cell, label, figure,
exercise, or slide was touched.

## Before/after pairs

1. Opening.
   - Before: "A rule for each layer does not settle how to initialize a whole
     network. In a residual network the activations grow even though every
     layer follows this rule."
   - After: "One might hope that a rule for each layer settles the matter. It
     does not. In a residual network the activations grow even though every
     layer follows the rule."
2. Roadmap.
   - Before: "It ends with initializers beyond the defaults, truncated normals
     and arbitrary distributions, and with editing parameters directly."
   - After: "Truncated normals, distributions of our own, and editing
     parameters by hand come last. They are conveniences, and a reader in a
     hurry may skim them."
3. Library Defaults.
   - Before: "The default is not He initialization, so in general it does not
     keep the scale of a deep stack of ReLU layers
     (:numref:`sec_numerical_stability`). We check the default on a layer with
     512 inputs and 256 outputs."
   - After: "A framework initializes a layer when we do not, and none of the
     four frameworks of this book defaults to He initialization
     (:numref:`sec_numerical_stability`). A statement like this is easy to
     check, so let us check it."
4. PyTorch tab, `apply` (INI-5).
   - Before: "The parameter must remain the tensor object that the module
     registered; replacing it would leave `net.parameters()` and any existing
     optimizer pointing at the old one."
   - After: "This matters. The parameter must remain the tensor object that
     the module registered. Were we to replace it, an existing optimizer would
     go on updating a tensor that the model no longer uses."
5. Variance growth (INI-6).
   - Before: "The first layer and the ReLU pass on $m_k$, by the argument for
     He initialization (:numref:`subsec_he_init`), and the output projection
     multiplies it by ..."
   - After: "The first layer doubles it and the ReLU halves it again
     (:numref:`subsec_he_init`), so together they pass on $m_k$. ... So far
     this is routine. The step that needs an argument is the sum."
6. Variance growth, last sentence.
   - Before: "Every layer was initialized by the rule for one layer, and the
     network as a whole still explodes, because the rule ignores the skip
     connection, which adds the input back in every block."
   - After: "This resolves the puzzle of the introduction. Every layer was
     initialized by the rule for one layer, and the network as a whole
     explodes all the same, because that rule knows nothing of the skip
     connection, which adds the input back in every block."
7. Measurement.
   - Before: "Initial activations this large make optimization impractical."
   - After: "We have not trained this stack, and with initial activations in
     the tens of thousands there is little reason to try."
8. Custom distribution (INI-11, prose part).
   - Before: "As an example with no library counterpart, suppose we want
     weights distributed as"
   - After: "The example is artificial on purpose, so that no library can have
     it. ... We know of no use for this distribution. Its merit is to show
     that any rule we can write in two lines can serve as an initializer."

## Repetition removed

- **He/Xavier theory (owned by ch. 4, `sec_numerical_stability`).** The
  opening keeps only the rule $g/\sqrt{n_\textrm{in}}$ and the gain, which the
  residual derivation needs (ch. 4 does not use the word "gain"); the Xavier
  sentence and the "stack of ten ReLU layers" consequence are gone.
- **Per-framework defaults (told in ch. 4 "Default Initialization").** The
  four tabs under "Library Defaults" now give the rule in a sentence plus what
  is needed to read the cell; the construction/build/laziness narration is
  one clause with `:numref:`sec_lazy_init``.
- **"Xavier first, constant 42 last, poor for training by symmetry"** was
  written out in all four tabs; it is now one shared paragraph, and each tab
  holds one sentence about its own API.
- **Symmetry argument** in "Starting a Block at Zero": one clause with the
  reference.
- **Lazy-layer ordering** (owned by 5.1, `sec_lazy_init`): the four closing
  tabs are shortened; "offsetting a whole matrix or setting a single entry"
  moved from four tabs into the shared sentence.
- **Closing paragraph** "We can now apply any initializer ..." (a recap) is
  replaced by two sentences on what initialization does not do; the Summary is
  the only recap.
- "GPT-2 initializes its residual branches this way: it scales ..." said the
  same thing twice; now once.

## Review findings

Applied (prose): INI-2 (the $b/\sqrt{3}$ fact now sits in the sentence about
the predicted standard deviation), INI-3 (partly: the four situations are
ranked as two about scale and two practical, not cut, because the tabs and a
slide use all four), INI-5, INI-6, INI-7 (as a candid statement that the
forward pass cannot rank the two repairs; no recommendation, since no training
run supports one), INI-8 ("treatments" to "choices of gain" in the body),
INI-10 ("therefore" and the unexplained "dynamic range" sentence removed),
INI-12 (the `no_grad` explanation moved to the PyTorch tab above the cell it
describes), INI-13 (summary).

Applied in part: INI-1. The shared prose now says a network with one or two
hidden layers tolerates the default and a deep one may not; the PyTorch tab
says the default has one sixth of He's variance (as ch. 4 does). The factor
$1/6$ per layer on the mean square is left to exercise 1, which asks for it.

Deferred:
- Structural "Order" (residual analysis last): moves code cells.
- Structural "No training": needs a new experiment.
- INI-4, INI-11 (replace the 0.01 / 42 / $\pm[5,10]$ toys): code changes. The
  prose now says openly that the values are chosen to be recognizable and
  that the custom distribution has no use.
- INI-9: the run-to-run range is kept by author decision; only the sentence
  around it was reworded.
- Terminology "branch" against "body", and $g$ against $\alpha$ and
  $\mathbf{g}$: needs one decision for the chapter. The opening now ties the
  two words once: "its *branch* (the body $f$ of
  :numref:`sec_model_construction`)".

## Factual claims added or changed

- PyTorch default: "one sixth of the variance that He initialization
  prescribes" ($1/(3n_\textrm{in})$ against $2/n_\textrm{in}$; also stated in
  ch. 4).
- "A network with one or two hidden layers tolerates such a default; a deep
  one may not." Judgment, not measured here.
- PyTorch `trunc_normal_`: "With `std=0.02` the defaults would cut at 100
  standard deviations, which is to say not at all" ($2/0.02 = 100$).
- PyTorch `apply`: replacing a parameter object leaves "an existing optimizer
  ... updating a tensor that the model no longer uses".
- Keras shared initializer object: added "and nothing warns us" (the slide
  says the same).
- "The first layer doubles it and the ReLU halves it again" (He, gain
  $\sqrt{2}$).
- "The rule is in practical use" before the GPT-2 sentence.
- New closing of the measurement: both repairs keep the activations at scale
  and differ in how training starts (every branch active against none).
- Removed: "Initial activations this large make optimization impractical";
  "Large weights also use up dynamic range once a model is stored in a
  low-precision format (:numref:`sec_numerics`)"; the causal "therefore" for
  BERT and the vision transformer.
- Checked against the stored outputs of all four frameworks: unscaled 4.3e4 to
  9.1e4 at $N = 32$, scaled 1.48 to 1.7, zero 0.99 to 1, truncated standard
  deviation 0.0176, largest plain normal draw 0.098 to 0.100. No number
  describing an output was changed.

## Slides that now disagree with the body (not edited)

- "Override the Default in Four Situations", rule box: "A model whose
  behavior should not depend on the framework sets its initialization
  explicitly", and Recap: "set the initialization explicitly". The summary now
  says "a deep model without normalization layers needs an explicit rule"
  (INI-13).
- "Truncated Normals Bound the Largest Weight" has no conflict, but the body
  no longer gives a motive for BERT and the vision transformer; the slide
  never did.
- Exercises (not edited) still say "the three treatments", "the unscaled
  treatment", "the scaled treatment"; the body now says "choices of gain".
