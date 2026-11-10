# Writing pass: `chapter_multilayer-perceptrons/kaggle-house-price.md` (§4.7)

Prose words (checker count, whole file): 3912 before, 3914 after (+2). The
cuts (re-taught standardization, one-hot encoding and $K$-fold
cross-validation, the roadmap sentence, two tab notes, summary numbers) pay
for a chapter closing of about 85 words and for the two additions the stored
outputs required. Checker: "OK: only prose changed". `lint_source.py`: clean.

## Before / after

1. Before: "An error of \$50,000 is large for the first and small for the
   second, so what matters is the relative error, not the error in dollars."
   After: "An error of \$50,000 is more than the first house is worth and
   less than 7% of the second, so what matters is the relative error, not the
   error in dollars."
2. Before: "We compute these statistics from the training rows alone and apply
   the same transformation to all other rows."
   After: ":numref:`sec_pandas` computed such statistics from the whole table
   and warned that the shortcut should not be copied. Here we keep the
   promise."
3. Before: "For features whose absence means something, such as the year in
   which a garage was built, it is not, and exercise ... examines the bias it
   introduces."
   After: "For the year in which a garage was built it is a poor one, since
   there a gap means that the house has no garage."
4. Before: "Fitting it once on all labeled houses would leak the statistics of
   each validation fold into its features."
   After: "Fitting it once on all labeled houses would be simpler, and it
   would leak the statistics of each validation fold into its features."
5. Before: "The mean over the five folds varies far less, by one or two
   hundredths from one random split to the next, occasionally more. Models
   evaluated on the same folds share much of this variation, ..."
   After: "As a rule the mean over the five folds varies far less, by one or
   two hundredths from one random split to the next. That is still as much as
   the models differ. Luckily, models evaluated on the same folds share much
   of this variation, ..."
6. Before: "At its minimum, the MLP beats the linear model on the same folds in
   every run, by one or two hundredths, which lowers the typical error from
   about 15% to 13% or 14% of the price."
   After: "At its minimum, the MLP beats the linear model on the same folds in
   every run. The margin is small, though: one or two hundredths as a rule,
   which lowers the typical error from about 15% to 13% or 14% of the price.
   For all its 22,000 parameters, the MLP finds little in this table that a
   weighted sum had missed."
7. Before: "The cross-validated RMSE estimates this score but does not
   guarantee it: ..."
   After: "Do not expect the score to match the cross-validated RMSE. ...,
   which makes the estimate a little too favorable."
8. Before (end of section and chapter): "Neural networks have the advantage
   where the inputs have structure that a network can exploit, such as the
   pixels of an image or the words of a sentence, which the following
   chapters turn to."
   After: "This is a sobering place for a chapter on MLPs to end, and an
   instructive one. We asked in :numref:`chap_perceptrons` for a model in
   which the evidence from one input depends on the others, and a hidden
   layer provides it. A fully connected layer, however, knows nothing of how
   its inputs are arranged: shuffle the pixels of every image in the same
   way, and an MLP learns as well as before. Neural networks have the
   advantage where the inputs have structure that a layer can exploit, ..."

## Repetition removed

- Standardization (owner `sec_pandas`): definition as a term and "mean 0 and
  variance 1" sentence removed; now "standardize ... as in sec_pandas" with
  the formula kept in one clause because $\mu$, $s$ come from training rows.
- Mean imputation (owner `sec_pandas`): no longer introduced as a new term;
  one clause with reference. The 0.82 variance stays as one clause.
- One-hot encoding (owner `sec_pandas`): the worked "a house in zone RL has
  ..." explanation and "This is the *one-hot encoding*" removed; the
  `MSZoning` columns stay as the concrete case. The reference now points to
  `sec_pandas` (features) in place of `subsec_classification-problem`
  (labels).
- Fit-on-training-rows rule (stated in `sec_pandas`, which points forward to
  this section): shortened, tied to that warning; *leakage* is still named
  here because no earlier section uses the noun.
- $K$-fold cross-validation (owner `sec_generalization_basics`): the
  procedure is no longer re-described; one sentence with reference.
- Roadmap sentence of the introduction ("We first describe ... then ...")
  removed.
- `KaggleHouse` test-table behaviour was told twice (module paragraph and
  submission paragraph); the second telling lost its last clause.
- Summary: the RMSE numbers and the target-centering clause are no longer
  repeated; the body states them once.
- Two tab notes on `make_model` folded into one shared sentence that is true
  in every tab.

## Review findings

Applied: KH-2 (dropped the unexplained "one-hot features compete with the
bias" clause; kept the simple reason and the exercise pointer), KH-3 (one
parenthesis: $K$ counts folds here), KH-4 (general advice on identifiers
deleted), KH-5 (`k_fold` paragraph leads with what is printed; the caveat on
the running training loss moved to where the training RMSE is first read),
KH-7 (45-word sentence split). Structural: one sentence acknowledging
dropout as the alternative regularizer; a chapter closing.

Deferred: KH-1 (needs a plot: new cell and output). KH-6 (needs a cell that
prints per-fold differences). Chapter-wide choice of a letter other than $K$
for layers (needs one decision for the chapter).

## Factual claims added or changed

- **Linear model in the JAX store.** The stored JAX run has linear validation
  RMSE 0.174 with one fold at 0.284 (PyTorch 0.146, TensorFlow 0.144, MXNet
  0.148; worst folds 0.19 to 0.20). The old prose ("about 0.15", worst fold
  "about 0.2", MLP ahead "by one or two hundredths", unregularized MLP
  "worse") was false in the JAX tab. New text: "Now and then the linear
  model does much worse on a single fold, near 0.28 in one of our runs, which
  takes the mean to about 0.17"; the margin is "one or two hundredths as a
  rule"; the unregularized MLP "predicts no better than the linear model,
  and in most runs worse". The review did not flag this (it checked the
  PyTorch store only). A re-capture of the JAX notebook may make the
  0.28 sentence unnecessary.
- Best fold of the linear model: "about 0.12" changed to "0.11 or 0.12"
  (stored: 0.114, 0.106, 0.116, 0.120).
- "\$50,000 is more than the first house is worth and less than 7% of the
  second" (34,900; 50,000/755,000 = 6.6%).
- "For the year in which a garage was built ... a gap means that the house
  has no garage" (was implicit; the slides already say it).
- "Only the JAX models need [the `num_inputs` argument]" (checked against
  the `make_linear` and `make_mlp` cells of all four tabs).
- The cross-validated estimate of the selected model is "a little too
  favorable" (direction of the selection bias; the old text said only "does
  not guarantee").
- "shuffle the pixels of every image in the same way, and an MLP learns as
  well as before" (permutation invariance of a fully connected first layer
  under i.i.d. initialization; stated, not demonstrated in this chapter).
- "For all its 22,000 parameters, the MLP finds little in this table that a
  weighted sum had missed" (a judgment on the 0.01 margin).
- Removed claims: the one-hot groups "compete with the bias ..., which slows
  SGD"; `d2l.download` "keeps a copy on disk and checks it against a SHA-1
  hash"; the advice that identifier columns "should be checked".

## Slides that now disagree with the body (not edited)

- "Linear Regression: Validation RMSE About 0.15" slide and the recap
  ("Linear: validation RMSE about 0.15"; "the best fold about 0.12, the
  worst about 0.2"): do not mention the bad fold of the JAX run; body says
  "0.11 or 0.12".
- MLP slide: "validates worse (0.16–0.18 vs about 0.15)" and "beats the
  linear model in every run, by one or two hundredths": body now says "no
  better ..., and in most runs worse" and "one or two hundredths as a rule".
- Nothing in the slides relies on the removed "compete with the bias" clause.
