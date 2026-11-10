# Writing pass log: Naive Bayes

Source: `chapter_mdl-probability-statistics/mdl-naive-bayes.md`. Prose only;
no code cell, cell ID, figure include, label or slide changed. Not split, no
markers. No labels added.

## Words

Prose words (checker count, body plus exercises): 7446 before, 7431 after
(-15). The checker ends "OK: only prose changed". It prints "WARN: em dash in
prose", which is the separator row `|---|---|---|` of the spam-filter table
(present before the pass); there is no em dash in the prose.
`tools/lint_source.py` is clean. The body did not shrink by more because the
pass adds the orientation caution (NB-12), the torso explanation (NB-13) and
the pointers to exercises that the assignment asked for; the cuts pay for
them.

Numeric checks were done with `python3` and NumPy on the local Fashion-MNIST
files (the model of the section refitted: accuracy 0.7059, the stored value),
the rest by hand.

## Correctness changes

- **NB-4** (parameter count). Said "the same 7,850 parameters" as softmax
  regression (body twice, Summary once). Naive Bayes has 7,849 free numbers by
  the text's own count. Now "a linear classifier of the same family as softmax
  regression, with arrays of the same shapes, $784\times10$ weights and $10$
  offsets"; "hyperplanes of the same family" in the comparison and in the
  Summary. "7,850" now occurs once, in the first paragraph, for softmax
  regression. Checked by hand.
- **NB-5** (double precision). Said $10^{-236}$ "comes close to the floor of
  double precision". It is 72 orders of magnitude above the smallest normal
  double. Now: "A double survives this example, but not the real scores
  computed below, some of which are near $10^{-1000}$." The stored output
  prints the smallest exponent as $-1002$.
- **NB-9** (what double precision loses). Said "44% of the numerators fall
  below the smallest normal number, where digits are lost, or underflow to
  zero", which suggests that doubles cannot classify. Now: nearly all of the
  44% belong to losing classes; "the largest numerator stays an ordinary
  double for all but a handful of images. On 784 pixels a double would still
  classify, with no margin for a longer input." Recomputed: 3 of 10,000
  images have their best numerator below the smallest normal double, 2 have
  all ten numerators equal to zero, and the raw product in float64 has
  accuracy 0.7058 against 0.7059. The cell still prints only the 44%; see
  deferred.
- **NB-8** (stored output). Said "a handful of pixels for most classes and 57
  for sneakers". The output is `[0 13 1 0 3 3 0 57 0 1]`. Now "none for half
  the classes, 13 for trousers, and 57 for sneakers". Deterministic cell.
- **NB-11** (mechanism asserted, not shown). Said the naive Bayes errors come
  from weighting each pixel as independent evidence "whereas the direct fit
  learns to discount pixels". The run shows only that the family of
  hyperplanes is not the limit. Now: "the loss lies in how counting sets the
  weights. The run does not say what is wrong with them; we return to this
  under calibration", and under "Calibration" the over-counting of large
  redundant regions is called "a plausible reason for the twelve points lost
  ... but our run did not test it". The closing paragraph points to the
  exercise whose part (d) is the test.
- **NB-13** (torso weights). The old text said the offsets "absorb that
  shift" and left the reader to conclude that an inked torso argues for coat.
  Now: a weight is the score of an inked pixel relative to a blank one; a
  torso pixel is inked in about 99% of the pullovers and more than 99.5% of
  the coats, so finding it inked says about $-0.01$ nats; the weight of $-1$
  is what a rare blank torso pixel gives in favour of pullover. Recomputed
  over the 140 pixels with weight below $-0.7$: median $\hat\theta$ 0.989
  (pullover) and 0.996 (coat), median inked contribution $-0.007$ nats, median
  blank contribution $+1.04$, median weight $-1.05$.
- **NB-17**. "hundreds of such groups" (not measured) is now "many such
  groups".
- **NB-7** (figure). "The line fitted directly" suggested a fit to the 120
  points. Now: "the line that a direct fit of $p(y\mid\mathbf x)$ approaches
  with enough data ... no classifier of any kind does better on these two
  classes." Checked: the generator draws the boundary with normal
  $\boldsymbol\Sigma^{-1}\Delta\boldsymbol\mu$, the Bayes rule for two
  Gaussians with a common covariance; logistic regression is well specified
  there. Population errors recomputed: 19.2% and 7.2%.
- **NB-1**. "every possible black-and-white picture" is now "every possible
  picture. Even if we record each pixel only as inked or blank, ...".
- **NB-15**, hypothesis: "binary feature" in the old proposition was never
  used; the demoted calculation speaks of "a feature".

No finding disagreed with. NB-12 was applied differently from the review's
proposal; see below.

## NB-12: confusion matrix orientation (clash logged)

`classification.md` (`eq_confusion-matrix`) and `softmax-regression-scratch.md`
define $N_{ij}$ as the examples of class $j$ predicted as $i$: rows predicted,
columns true. The cell here (`mdl-naive-bayes-calibration-2`) fills
`conf[Y_test, pred]`: rows true, columns predicted, and the next two cells
index it that way (`conf.sum(axis=1)`, `conf[2, 4]`). The review asks to use
the main-text orientation, which needs a change to three cells. In this pass
the prose no longer re-defines the term; it cites `sec_classification` and
says that the cell "prints the transpose of the matrix $\mathbf N$ of
:eqref:`eq_confusion-matrix`, whose rows are the predicted classes. Each row
shows where the 1,000 test images of one garment went." The list of largest
errors is introduced as "true class first", and the per-class accuracy as
"diagonal entry divided by its row sum". The clash itself remains until the
cells are changed (deferred). The comparison the review asked for is added:
same two groups as in `sec_softmax_scratch`, but shirts recognized "40% to 70%
of the time" there (quoted from that section's prose) and "one time in six"
here (stored: 0.17).

## Before and after (verbatim)

1. Before: "A classifier needs a rule that maps an image to a label. This part
   derives such a rule from a model of how each class produces its images, and
   works it by hand on a two-word spam filter." After: cut (the subsection
   below opens on the same sentence).
2. Before: "This part estimates them, protects them against zero counts, and
   rewrites the rule as a sum that a computer can evaluate and that is linear
   in the pixels." After: "Where do these numbers come from, and how does a
   computer evaluate a product of 785 of them?"
3. Before: "What is the test accuracy of the counted model?" After: "Softmax
   regression reached about 83% with scores of this form. How much does
   counting lose?"
4. Before: "The next cell lists the six largest off-diagonal counts and the
   accuracy of each class." After: "A hundred counts are easier to read from
   their largest entries. We list the six largest off-diagonal counts, true
   class first, and the accuracy of each class, its diagonal entry divided by
   its row sum."
5. Before: "A reader may object that the proposition protects the ranking only
   with equal priors. That is right." After: "The calculation protects the
   ranking only for equal priors, and the restriction is real."
6. Before: "The next cell compares the amount of ink and the number of
   ink/background switches per row in the training images and in 1,000 samples
   per class." After: "We count the ink/background switches per row, and the
   amount of ink, in the training images and in 1,000 samples per class."
7. Before: "This is the same family of classifiers as softmax regression, with
   the same 7,850 parameters for Fashion-MNIST". After: "This is the first of
   the two results of the section worth remembering: the counted model is a
   linear classifier of the same family as softmax regression, with arrays of
   the same shapes".

## Repetition

"Correlated evidence is counted many times" (NB-S1, NB-2) is owned by
"Calibration". Removed from: the introduction (which now ends on "no other
model shows as clearly what a false independence assumption costs"), the end
of "The Naive Assumption" (now "How false it is, and what the error costs, we
measure on Fashion-MNIST below"; the Domingos and Pazzani citation moved to
"Calibration"), the end of "Checking the Assumption" (forward pointer cut;
the observation "nearly one feature recorded twice" stays, it is the
measurement), "Confusions and Weights" (see NB-11), and the closing paragraph.
It is stated in "Calibration" and once in the Summary.

Also removed: "7,850" three times; the second explanation of why the $x_i^2$
terms cancel (NB-6: the Gaussian paragraph now precedes the figure paragraph,
no cell moved); the re-definition of the confusion matrix; the coin example of
Laplace smoothing (9/13 against 10/15), which the exponential-family section
owns and the Beta paragraph cites; the sentence that restated the naive Bayes
rule in words; the announcement paragraph at the head of "Fashion-MNIST by
Counting"; the Summary's list of the chapter's tools (the introduction has
it).

Kept because this file owns them: the spam filter and all its numbers, the
pairwise-independence caveat, the duplicated-evidence calculation and its two
examples.

## One ending

The "We can now fit a classifier by counting ..." paragraph is gone (NB-S5).
The section ends on the open question it contained: when counting beats the
direct fit (Ng and Jordan, with the exercise). The Summary is the only recap.

## Demoted

**Proposition (duplicated evidence multiplies the log-odds)** with its Proof
is now a calculation in running text with the same displayed formula (NB-15).
The claimed confidence $\bar r^k/(1+\bar r^k)$ is now stated as odds
$\bar r^k$ against $\bar r$. It had no label. No Proposition block remains in
the section; the text names two results to remember (the affine score, and
"the ranking survives and the confidence does not").

## Exercises and main-line content (assignment)

Exercises untouched. The body now says where the two topics are: the text
paragraph moved from "Test Accuracy" to the end of "Linear Decision
Boundaries" (NB-S2), says a document can be modeled by which words occur or
how often, that the score is linear in both cases, and that exercise
`ex_mdl-naive-bayes-two-event-models` develops them; a paragraph in
"Calibration" names temperature scaling (cites `sec_classification`), says it
leaves the ranking alone and that $T=k$ repairs $k$ exact copies with equal
priors, and points to `ex_mdl-naive-bayes-temperature-scaling`.

## Capstone framing (NB-S4, NB-10)

New second-to-last introduction paragraph: "This section is the chapter's
case study. After the bounds of :numref:`sec_mdl-generalization-bounds` the
mathematics is light". The 784 frequencies get their number: standard error
at most $0.5/\sqrt{6000}\approx0.0065$. "Error Bars" says "This is the coin
with which the chapter opened, flipped once per test image".

## Inbound links from the main text (cannot be made here)

No main-text section cites `sec_mdl-naive_bayes`. Candidates:

- `chapter_linear-classification/softmax-regression-concise.md`, at the
  reliability check: the generative counterpart of the same family, 71% and
  99% claimed confidence.
- `chapter_linear-classification/classification.md`, at temperature scaling
  or calibration: a classifier whose overconfidence has a known cause.
- `chapter_linear-classification/softmax-regression-scratch.md`, at the
  confusion matrix: same two groups of errors (and the orientation clash).
- `chapter_linear-classification/environment-and-distribution-shift.md`, at
  label shift: a generative classifier corrects the prior without retraining
  (exercise 8 here).
- `chapter_lm-encoders/bert.md` plans a reference for word counts and will
  find pixels in the body.

## Cells to cut or redesign on the development machine

No cell here only confirms a proved result. Redesign:

- `naive-bayes-imports` (NB-18): imports `torch`, `nn`, `F`, `optax` and so on
  for a NumPy section; `rng` is created three times.
- `mdl-naive-bayes-test-accuracy-1` (NB-9): print the fraction of images whose
  best numerator underflows float64 (0.0003) and the raw float64 accuracy,
  next to the 44%.
- `mdl-naive-bayes-calibration-2`, `-confusions-and-weights-1`, `-2`
  (NB-12): transpose `conf` to the main-text orientation, then replace the
  orientation caution by a clause.

## Deferred

- NB-S3 / exercise imbalance: 2,400 words of exercises against the body; the
  event models and temperature scaling are still only named in the body. The
  exercise changes of the review (1(d), 7(a), split 12, move 10(d)) are not
  errors and were left. Note that the text still gives the answer to 1(d) and
  to part of 7(a).
- NB-S2: a subsection on the text event models needs new material.
- NB-S4 (ii): training accuracy next to test accuracy. I computed 0.717 on
  the training set, but no cell prints it, so it was not added.
- NB-14: accuracy with half the pixels hidden needs a cell.
- NB-17: the effective duplication (exercise 10(c)) needs a cell.
- NB-11: the experiment that would test the mechanism is exercise 12(d).
- Title and position of the section (review, placement): unchanged.

## Claims added or changed

- Standard error of a frequency from 6,000 images is at most 0.0065 (by hand).
- Doubles: nearly all sub-normal numerators belong to losing classes; a double
  would still classify on 784 pixels (recomputed, above).
- Torso pixels: 99%, more than 99.5%, $-0.01$ nats (recomputed, above).
- The direct fit in the figure approaches the optimal line, and 7% is the
  best any classifier can do there (by hand).
- Softmax regression recognized shirts 40% to 70% of the time (from the prose
  of `sec_softmax_scratch`, not re-run).
- Temperature scaling leaves the ranking unchanged; $T=k$ restores the
  posterior for $k$ copies and equal priors (by hand; exercise 7(c)).
- Ng and Jordan: counted parameters settle after a number of examples
  logarithmic in $d$, the direct fit can need a number linear in $d$ (as the
  exercise already states it; not checked against the paper).
- "The score is linear in both cases" for the two text event models (by hand).
- Numbers kept that no cell prints: 19% and 7% (recomputed), $10^{236}$ (by
  hand), all spam-filter numbers (review recomputed; unchanged).

## Slides that now disagree with the body

- "Same family and same 7,850 numbers as softmax regression", the slide title
  "Same 7,850 Numbers, Fitted Directly: Higher or Lower?" and "a linear
  classifier with 7,850 parameters": the body no longer claims equal counts.
- "Naive Bayes weights each pixel as independent evidence" on the comparison
  slide gives the mechanism the body now calls untested.
- A slide carries the kicker "Proposition"; the body has a calculation.
- The slide cover says "§27.9" (rewritten automatically).
