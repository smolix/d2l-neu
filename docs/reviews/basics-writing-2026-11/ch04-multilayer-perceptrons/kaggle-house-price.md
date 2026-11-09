# Predicting House Prices on Kaggle

Source: `chapter_multilayer-perceptrons/kaggle-house-price.md`. About 2,900 words of prose, 1,080 words of exercises, fourteen code cells, no figure and no plot.

## Verdict

A well-made capstone, and a model of how to write an applied section. It knows what it is about ("Most of the work lies in preparing the data and in estimating the error; the models take a few lines each"), every preprocessing step is motivated by a column of the actual table, leakage is prevented by how the code is structured and not just warned against, and the result is reported with its uncertainty. It also ends with a scope statement that many textbooks would not print: on tables like this, gradient-boosted trees usually beat MLPs. Against the bar, the main weakness is that the section never lets the reader *see* anything: no histogram of prices, no plot of the validation error against weight decay, although the prose describes a U-shaped curve. Everything is printed lines. The most important fix is one plot.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 4 | 4 | 4 |

## What works

- **The opening.** "Fashion-MNIST arrives clean: every example is an image of the same size, every pixel lies between 0 and 1, and every label comes from the same ten classes. Most data that people want to predict from looks different." It starts from what the reader has and says what is new, and the last sentence of the paragraph states the real difficulty of small data: an error estimate "off by more than the differences between the models we want to compare".
- **Concrete columns carry each idea.** `PoolQC` missing for 1,453 houses "which have no pool" (missing can mean absent). `MSSubClass` codes 20, 60, 90: "The differences between these codes measure nothing" (a number is not always a quantity). Code 150 occurs only in the test table (unseen categories). \$34,900 against \$755,000 (why relative error). Each is one or two sentences and each is memorable.
- **The error measure is explained, not just defined.** "If a prediction $\hat{y}$ satisfies $|\log \hat{y} - \log y| \leq \delta$ ..." then "An RMSE of 0.1 thus corresponds to typical errors of about 10% of the price." The reader can interpret every later number.
- **Leakage handled in the design.** `preprocess(train, *others)` fits on its first argument only, and `k_fold_data` builds one data module per round "so that the preprocessing of each round is fitted on its own training folds". The code makes the right thing the easy thing.
- **Methodological sentences worth keeping verbatim**: "A baseline is only useful if it is trained as carefully as the models it is compared with, so it gets the same budget." And: "Models evaluated on the same folds share much of this variation, so the difference between two of them is measured more precisely than either error alone."
- **Honest reporting.** Fold range printed beside the mean (`folds 0.114 to 0.198`); the improvement stated as "one or two hundredths"; the caveat that the selected model's cross-validated error is optimistic because weight decay was chosen on the same folds.
- **The closing paragraph** on trees and on where neural networks have the advantage is the right note on which to end a chapter on fully connected networks and leads into the chapters on structure.

## Structural findings

- **Nothing is plotted.** See KH-1. Fourteen code cells, zero figures. The price distribution, the reason for the log, and the U-curve over weight decay are all described in words.
- **The MLP's gain is small, and the section says so.** Linear 0.146, best MLP 0.136 in the stored run. That is honest and the text frames it well. At chapter level it is the third real-data experiment with a marginal gain for the chapter's model (see the chapter file).
- **Placement.** As the last section of the chapter it works: it uses the MLP, He initialization, weight decay and cross-validation. It does not use dropout, the section just before it, except in an exercise. One sentence acknowledging that (weight decay was chosen as the regularizer here; exercise 8 tries dropout) would close the loop.
- **The chapter has no closing.** The last paragraph of this section is the last paragraph of the chapter. It works as one, but it is about tabular data. A reader finishing chapter 4 gets no statement of what the chapter established and what chapter 5 does about it.
- Length and pace are right. I would cut nothing of substance.

## Passage-level findings

- **KH-1** (moderate, pedagogy: describe a curve, show no curve)
  - Anchor (An MLP with Weight Decay): "well as the linear model does. This is the U-shaped curve of"
  - The reader is told that validation RMSE traces a U over weight decay and that training RMSE rises. The evidence is five printed lines to scan. The stored values (0.169, 0.141, 0.136, 0.142, 0.166 for validation; 0.094 to 0.163 for training) would make a clear plot with the linear baseline as a horizontal line, and `d2l.plot` does it in three lines.
  - Fix: have `k_fold` return training and validation means, collect them, and plot both against weight decay on a logarithmic axis with the linear model's 0.146 as a reference line. Then the prose can shrink to what the plot shows. Similarly, one histogram of the prices next to one of the log prices in "The Error Measure" would show why the logarithm is natural, though that is optional.

- **KH-2** (moderate, explanation: an unexplained claim)
  - Anchor (after the feature count): "offset, and the one-hot features, each group of which sums to 1, compete"
  - "The one-hot features ... compete with the bias in doing so, which slows SGD" is a real effect (each group of indicators is collinear with the bias) stated in a subordinate clause and handed to an exercise. A first-time reader cannot tell what "compete" means.
  - Fix: either drop the clause and keep the simple reason (predictions start near 0, targets near 12, the first updates would be spent on the offset), or explain in one more sentence: "Every group of indicator features sums to 1, so the model can produce a constant offset through 44 groups of weights as well as through the bias, and the gradient spreads the offset over all of them."

- **KH-3** (minor, notation)
  - Anchor (K-Fold Cross-Validation): "ones. *$K$-fold cross-validation* (:numref:`sec_generalization_basics`)"
  - $K$ is the number of layers in the rest of the chapter (`mlp.md`, `backprop.md`, the initialization section) and the number of folds here, and the number of ensemble members in the last exercise. The name "$K$-fold" is standard and should stay; the layers could be counted with a different letter chapter-wide, or this section can say "(here $K$ counts folds, not layers)".

- **KH-4** (minor, scope)
  - Anchor (The Data): "so we do not use it as a feature. Identifiers are not always harmless: an"
  - A sentence and a half of general advice on identifiers that the section does not act on ("such a column should be checked before it is dropped or kept") and does not check.
  - Fix: delete, or do the check in one clause ("here `Id` is unrelated to the price").

- **KH-5** (minor, plumbing in prose)
  - Anchor (K-Fold Cross-Validation): "epoch from the trainer's history. The training loss there is the mean of the"
  - The paragraph before `k_fold` explains the trainer's history, the difference between a running training loss and a separate pass, the meaning of `make_model`, and why the feature count varies between 335 and 345, followed by two tab notes on whether layers infer their input size. The reader wants to know what the function prints.
  - Fix: lead with the output ("`k_fold` trains a fresh model per round and prints the average training and validation RMSE and the range over folds"), and move the caveat about the running training loss to the place where the training RMSE is first read, where it matters.

- **KH-6** (minor, thought)
  - Anchor (An MLP with Weight Decay): "the linear model on the same folds in every run, by one or two hundredths,"
  - "In every run" is a statement about repeated executions that the reader sees one of. It is the kind of claim the paired design supports, and the text has explained why. It would be stronger with the paired differences shown: per-fold difference between the linear model and the selected MLP, five numbers.
  - Fix: optional. Print the per-fold differences once; then "in every run" becomes "on four or five of the five folds", which the reader can verify.

- **KH-7** (minor, language)
  - Anchor (Preprocessing, Numerical Features): "small for another, which slows minibatch SGD, and weight decay, which"
  - A 45-word sentence with two "which" clauses and two separate reasons for standardizing. Split after "slows minibatch SGD".

## Exercises

Nine problems, the most practically useful set in the chapter. "Target leakage" (add a noisy copy of the label and watch cross-validation fail to object) and "Target encoding" part (b) (out-of-fold encoding) teach things practitioners get wrong for years. "Geometric averaging" is a neat two-part inequality. "Fold ensembling versus retraining" picks up the alternative mentioned in the text. "Gradient-boosted trees" lets the reader test the closing claim.

- Nine is at the top of the band and all are good; if one must go, "Numbers that name categories" overlaps most with the text.
- "Improving the score" part (c) is the multiple-comparisons point again, well placed after the reader has tuned many configurations.

## Still open from October

Nothing was raised for this file in October.
