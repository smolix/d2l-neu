# Naive Bayes

Source: `chapter_mdl-probability-statistics/mdl-naive-bayes.md` (read to the
slides marker, line 1165). About 3,950 words of prose in the body and 2,400
in the exercises; 13 shared code cells plus per-framework import and loader
cells. Stored PyTorch outputs checked cell by cell; both figures rendered;
the worked numbers recomputed, the data-dependent ones against the local
Fashion-MNIST files.

## Verdict

This is one of the best-taught sections I would expect to find in the part,
and it is not mathematics. It is a case study in the manner of the main
text: one question (can a classifier be fitted by counting, and what does it
cost?), one hand-worked toy (the two-word spam filter) that returns three
times, a prediction before every cell, and stored outputs that support
every number the prose quotes. I found no mathematical error in the body or
in the exercises I recomputed. Against the bar it falls short in three
ways. The central lesson, that correlated evidence is counted many times,
is told six times in almost the same words. Two explanations assert a
mechanism the run does not show (why softmax regression gains twelve
points; what double precision does to the raw product). And the exercises
are 60% as long as the body and hold several of the section's best ideas.
The single most important thing to decide is editorial, not textual: say
plainly that this section is the chapter's worked case study, and make that
true of the whole chapter, or move it. As it stands it applies the tools of
sections 3 to 5 and none from the three sections that immediately precede
it.

## Scores

| Dimension | Score |
|---|---|
| Clarity of explanation | 4 |
| Clarity of thought | 4 |
| Pedagogy | 4 |
| Language | 4 |
| Ease and enjoyment | 4 |

## Does it earn a place in a mathematics appendix, and does it belong last?

**It earns a place, on one condition.** Every other section of the chapter
teaches a tool; this one teaches a model. Judged as mathematics for deep
learning it is out of scope: the mathematical content is Bayes' rule, a
product, a logarithm, and one identity ($k$ copies multiply the log-odds by
$k$). Judged as the chapter's capstone it does something the other sections
cannot: it shows maximum likelihood, the Beta posterior mean, the binomial
standard error, the bootstrap and a reliability table doing work on one real
dataset in 4,000 words. A reader who has pushed through
`mdl-generalization-bounds` deserves that. The chapter introduction already
says so ("puts these tools to work on one classifier"), but the section
itself does not say it until the end of its fourth paragraph, and its
title, "Naive Bayes", promises a method. Keep it, and frame it as the
case study it is.

**The condition** is that "the chapter's tools" means the chapter. The body
uses nothing from `mdl-bayesian-computation`,
`mdl-concentration-generalization` or `mdl-generalization-bounds`; Hoeffding
appears only in exercise 12(a). Two cheap additions would close this, both
already half-written: (i) after "Each of these 784 numbers is a frequency
that 6,000 training images per class can estimate well", give the number,
either the standard error $\le 0.5/\sqrt{6000}\approx0.0065$ or the
Hoeffding-plus-union bound of exercise 12(a), which says about 2,200 images
per class already put every estimate within 0.05 with probability 95%;
(ii) report the training accuracy next to the test accuracy and say in one
sentence why a counted model with 7,850 numbers and 60,000 images has
almost no generalization gap, with a reference to
`sec_mdl-generalization-bounds`.

**Last is the right position for a case study, and the wrong one for
difficulty.** By dependency it could follow `mdl-maximum-likelihood`
directly. Placed last it comes after the two hardest sections of the
chapter, and the level drops sharply: the reader goes from Rademacher
complexity to "Training is counting". That is acceptable for a capstone and
odd for a section presented as one more topic. If it stays last, the
opening should mark the change of gear in one sentence. If the authors
would rather the chapter end on its hardest result, the alternative home is
the main text, after `sec_softmax_concise`, as the generative counterpart
of the first classifier; but then it loses the Beta posterior and the
bootstrap, which are what tie it to this chapter. I recommend keeping it
last.

**Lookup use.** No section of the main text refers to it. The only inbound
reference outside the part is a planned one in `chapter_lm-encoders/bert.md`
("naive Bayes ... as the classical baseline on word counts"). A reader sent
here for word counts finds pixels in the body and the text model in
exercise 5. See NB-S2.

**Overlap with the main text** is small and well managed. The section
assumes `sec_softmax`, `sec_softmax_concise`, `sec_fashion_mnist` and
`sec_classification` and says so; it does not re-teach softmax regression.
Two overlaps need a clause each: the confusion matrix (NB-12) and the two
groups of errors, which `softmax-regression-scratch.md` already found.

## What works

- **The opening.** "It found 7,850 numbers ... by gradient descent ...
  Gradient descent is not the only way to obtain such numbers." The reader
  starts from what they did in the main text and gets a question, "Can this
  be done, and what does it cost?", that the section answers in numbers (71%
  against 83%, confidence 99%). The October-era complaint that the opening
  was method-first no longer applies.
- **The spam filter as a recurring toy.** Ten mails, a table of four counts,
  and it comes back exactly as promised ("The filter will return three
  times: when a count is zero, when we take logarithms, and when a feature
  is counted twice"). All its numbers check: 0.90, odds 9, smoothed 0.32 and
  0.80, $\log 9=2.20$, duplicated "free" 40.5 and 0.976. This is the kind of
  small example MacKay would use, and it should be protected in revision.
- **Sentences with voice.** "Training is counting, and prediction is
  multiplying." "One feature vetoes the class." "The hem either reaches row
  26 or it does not, so the two pixels are nearly one feature recorded
  twice." "the digits after the decimal point of '70.59%' are noise."
- **Objections taken seriously.** "A reader may object that Bayes' rule has
  saved nothing ... The objection is correct" leads straight to the chain
  rule and to why the assumption is needed. The second objection, that the
  ranking survives only with equal priors, is answered with a
  counterexample (prior odds $1/5$, two copies flip the decision).
- **The pairwise-independence caveat** (three bits with $x_3=x_1\oplus x_2$)
  is short, exact, and prevents the natural misreading of the hem-pixel
  check.
- **Code that teaches.** The templates-versus-samples pair makes the
  assumption visible, and the "switches per row" statistic puts a number on
  it (pullovers 4.8 against 3.1, sandals 8.2 against 3.0). The hem-pixel
  cell tests the factorization directly (0.30 predicted, 0.54 observed). The
  NumPy softmax regression on the same binary pixels is the right control:
  same family, same features, different fit.
- **The observation** "Every bin is overconfident, so the expected
  calibration error equals the mean confidence minus the accuracy" is a
  small piece of thinking the reader can reuse.
- **`fig_mdl-naive-genvdisc`** carries its paragraph: the axis-aligned
  ellipses against tilted clouds show the cost of the assumption at a
  glance. I recomputed its population errors from the generator's
  parameters: 19.2% and 7.2%.

## Structural findings

**NB-S1. The lesson is told six times.** That correlated pixels are counted
as independent evidence, which inflates confidence while the ranking stays
useful, appears in: the fourth paragraph of the introduction ("correlated
evidence is counted many times"); the end of "The Naive Assumption"
("counts the same evidence several times. We will see that this pushes");
the end of "Checking the Assumption" ("This duplication is the reason for
the overconfidence we will measure"); "Confusions and Weights" ("each pixel
is weighted as if it were independent evidence"); "Calibration", where it
is finally measured and derived; the closing paragraph; and the Summary.
Guide section 7.7 applies. The introduction should pose the question
without the answer (the reader will enjoy finding that a 71% classifier
claims 99%), the forward pointers in the second and third places should
shrink to a clause or go, and the result should be stated once where it is
measured and once in the Summary.

**NB-S2. Text is called the standard application and gets one orphan
paragraph.** "Text was the first application of naive Bayes and is still
its standard one" sits at the end of "Test Accuracy", between float
underflow and error bars, where it has no job. The multinomial and
Bernoulli event models, the form in which most readers will meet naive
Bayes again and the form the planned BERT section wants as a baseline,
exist only as exercise 5. That exercise is good (I recomputed it: on "goal
match poll poll" the multinomial model says politics with odds 0.625 and
the Bernoulli model says sports with odds 6, so the two models disagree).
Promote its setup to a short subsection after "Linear Decision Boundaries"
(the multinomial score is also linear, in word counts), move the orphan
paragraph there, and keep the calculation as the exercise.

**NB-S3. Exercises outweigh the body.** Twelve exercises with about forty
parts and 2,400 words follow 3,950 words of body. Temperature scaling, the
fix for the section's headline defect, is only an exercise; so is the
Ng and Jordan result that answers the section's own closing question. The
closing paragraph ends on two pointers to exercises, which reads as "the
interesting part is homework". Bring one of the two into the body in a few
sentences with its number (temperature scaling is the cheaper: one fitted
scalar, accuracy unchanged, and the reader has met it in
`sec_classification` and as an exercise of `sec_softmax_concise`), and
prune the exercises as listed below.

**NB-S4. The capstone uses the first half of the chapter only.** See the
section on placement above for the two additions.

**NB-S5. Closing paragraph and Summary are two summaries.** The paragraph
beginning "We can now fit a classifier by counting" restates the method and
the two numbers; the Summary restates them again 15 lines later (71%, 83%,
twelve points, 99%). Keep the two open problems from the closing paragraph
and let the Summary carry the recap.

## Passage-level findings

**NB-1** · minor · thought · Introduction
> "probability to every possible black-and-white picture. There are"

The reader has not been told that the pixels will be binarized; the
softmax model of the previous paragraph used grey values. "Black-and-white"
has to be read as "binary", which the text explains 80 lines later. Add
"if we record each pixel only as inked or blank" here, where the count is
first used.

**NB-2** · moderate · thought · Introduction
> "still ranked usefully, but the model's confidence is inflated, because"

The answer to the section's question is given before the reader has seen
the model (NB-S1). Fix: end the paragraph on the properties that motivate
the model (one pass, no optimizer, missing features) and on the question
"what does a false independence assumption cost?", and let the cost arrive
in "Calibration".

**NB-3** · minor · language · "Bayes' Rule for Classification" and "Learning by Counting"
> "This part derives" / "This part estimates them, protects them against zero counts, and"

Two of the four top-level sections open with a three-line announcement of
their contents in the same construction. The fourth ("An accuracy of 71%
raises three questions") shows the better way: open on the question. For
the first two, one sentence of need is enough, for example "The rule needs
one probability per class and feature; where do they come from?"

**NB-4** · minor · correctness (low stakes, sure) · "The Naive Assumption" and "Linear Decision Boundaries"
> "Both models need $K-1=9$" and "classifiers as softmax regression, with the same 7,850 parameters for"

The two counts do not agree. By the first sentence naive Bayes has
$7{,}840+9=7{,}849$ free numbers: the ten offsets $b_y$ are fixed by the
weights and the prior, and a common shift of all $b_y$ is not available
because the prior sums to one. Softmax regression has 7,850 stored numbers
of which 785 are redundant. The claim that matters, that the two models
realize the same decision rules and posteriors, is true (a common shift of
the offsets changes neither). Say "the same array shapes, $784\times10$
weights and $10$ offsets" or "the same family of posteriors", and leave the
exact equality of parameter counts out. "7,850" occurs four times in the
prose; once in the introduction and once here is enough (guide 7.7).

**NB-5** · minor · correctness (sure) · "Scores in Log Space"
> "in single precision and comes close to the floor of double precision"

$10^{-236}$ is 72 orders of magnitude above the smallest normal double
($2.2\cdot10^{-308}$) and 88 above the smallest positive one the sentence
has just quoted. It does not come close. The real numerators, shown later,
reach $10^{-1002}$ and do underflow. Say: "underflows to $0$ in single
precision. Doubles survive this example, but not the real scores we compute
below, some of which are near $10^{-1000}$."

**NB-6** · moderate · explanation · "Linear Decision Boundaries"
> "naive generative fit models each class by a Gaussian with independent"

The figure paragraph uses Gaussian naive Bayes one paragraph before it is
defined ("Gaussian naive Bayes models a continuous feature"), and the
cancellation of the $x_i^2$ terms is explained twice, once before the
figure and once after. A reader of a section on binary pixels is suddenly
looking at continuous clouds with no transition. Swap the order: first the
paragraph "The linear form depends on the kind of feature", then the
figure as its illustration, with the cancellation stated once.

**NB-7** · minor · explanation · "Linear Decision Boundaries"
> "runs along the tilt and errs with probability 7%."

"The line fitted directly to $p(y\mid\mathbf x)$" suggests a logistic
regression fitted to the 120 points drawn. The generator plots the
population-optimal boundary, normal $\boldsymbol\Sigma^{-1}\Delta\boldsymbol\mu$,
which is the Bayes rule for these two Gaussians. That is worth saying
because it is the stronger statement: 7% is the best any classifier can do
here, a linear one attains it, and the naive fit, also linear, gives 19%.
Add "the best line, which is also the best classifier of any kind for
these two classes".

**NB-8** · minor · thought · "Templates and Samples"
> "The zero counts are rare here: a handful of pixels for most classes and 57"

The output is `[0 13 1 0 3 3 0 57 0 1]`: none for five classes, one to
three for four, 13 for trousers, 57 for sneakers. "A handful for most" is
not what the reader sees. Write "none for half the classes, 13 for
trousers, and 57 for sneakers".

**NB-9** · moderate · thought (verified numerically) · "Test Accuracy"
> "In double precision 44% of the numerators fall below"

The sentence is literally true and misleads about its consequence. A
numerator that underflows matters to the decision only if it is the
largest of the ten. I ran the raw product in float64 on the stored model:
its accuracy is 0.7058 against 0.7059 in log space, and only 2 of 10,000
images have all ten numerators equal to zero. So doubles lose the losing
classes, not the decision. The cell should report the fraction of images
whose best numerator underflows in each precision (0.96 in float32, 0.0002
in float64), and the prose should draw the honest conclusion: single
precision cannot classify at all, double precision still can on 784
pixels, and what is lost in both is the ratio between classes, which the
calibration subsection needs, and any margin for a longer input. This is a
prose-and-print change to one cell; if cell changes are deferred, change
the prose to say that the 44% are almost all losing classes.

**NB-10** · minor · pedagogy · "Error Bars"
> "The bootstrap interval runs from about 0.70 to 0.715, the binomial"

Good passage; one missed link. The chapter opened with "A test accuracy is
the same coin, flipped once per test question" (`index.md`). This is the
place where that sentence is cashed, and a clause saying so would tie the
capstone to the chapter's first paragraph.

**NB-11** · moderate · thought · "Confusions and Weights"
> "whereas the direct fit learns to discount pixels that only repeat what"

The run shows that the family of hyperplanes is not the limit, and the text
says so well ("The family of hyperplanes is therefore not the limit"). It
does not show why the direct fit is better. The mechanism is asserted here
and asserted again in "Calibration" ("the direct fit, which can learn to
discount redundant pixels, is less exposed to it"). The experiment that
would show it exists as exercise 12(d): on synthetic data for which the
assumption holds, the gap should close. Either mark the sentence as the
explanation the next subsection will support and cite exercise 12(d) as the
test, or state only what the run shows. The Basics decision applies: the
prose may be changed to say what the output shows.

**NB-12** · moderate · pedagogy · "Confusions and Weights"
> "It counts the test images by true class (row) and predicted class"

`softmax-regression-scratch.md` defines the confusion matrix with the
opposite orientation ("its entry $N_{ij}$ is the number of validation
examples of class $j$ that the model predicts as class $i$"), and its
reading of the result names the same two groups, the four tops and the
three shoes. A reader who compares the two printed matrices will misread
one of them. Use the main-text orientation, replace the re-definition by a
clause with a reference to `eq_confusion-matrix`, and say what is new
compared with softmax regression: the groups are the same, but shirts fall
to 17% and coats to 48%.

**NB-13** · moderate · explanation (verified numerically) · "Confusions and Weights"
> "reliably in coats, and the log-odds scale turns that small difference into"

This is the hardest paragraph in the section and the picture works against
it. The reader has just been told that "an inked pixel moves the decision
toward pullover by its entry", then sees a torso filled with weights near
$-1$ and must conclude that an inked torso argues strongly for coat. The
text then says the offsets "absorb that shift", which is correct but asks
the reader to do the cancellation in their head. The numbers make it plain:
in the torso $\hat\theta$ is about 0.993 for pullovers and 0.997 for coats;
an inked pixel there contributes $\log(0.993/0.997)\approx-0.004$ nats to
the score difference, and a blank one $\log(0.007/0.003)\approx+0.9$. The
weight $-1$ is the difference between those two, so it measures what a rare
blank pixel says, not what the usual inked one says. Draft: "The weight is
the score of an inked pixel relative to a blank one. In the torso almost
every pullover and coat is inked, so inking tells us almost nothing
($-0.004$ nats); the weight of $-1$ is what the model would gain, in favour
of pullover, from the rare image whose torso pixel is blank." Then the
sentence "The decision is made along the sides" follows.

**NB-14** · minor · thought · "Confusions and Weights"
> "Softmax regression has no such operation and must fill in a value"

Right, and the best argument in the section for generative models; it gets
five lines and no number. Exercise 11 measures it. One sentence with the
accuracy at half the pixels hidden would let the body make a claim it
currently only promises. Deferred if it needs a cell.

**NB-15** · moderate · pedagogy · "Calibration"
> "**Proposition (duplicated evidence multiplies the log-odds).** Consider two"

The statement is a one-line identity ($k$ equal factors give $k\log r$),
set as a proposition with a nine-line proof that restates it. The house
style keeps the proposition and proof format for results that need an
argument. Here the format slows the reader at the point where the section
should be quickest, and "binary feature" is a hypothesis the proof never
uses. The two examples after it (the 80% clue reaching 0.999 at five
copies; the duplicated "free") and the counterexample with prior odds $1/5$
are what teach. Demote the proposition to a displayed calculation of three
lines introduced by the spam filter, keep the examples, and keep the
equal-priors caveat. If a proposition is wanted, the partial-copy formula of
exercise 7(b), $r=4(0.8+0.2q)/(0.2+0.8q)$, is the more informative
statement: it shows the over-count growing continuously with the
correlation, which is the case the pixels are in.

**NB-16** · minor · language · "The Generative Route" and "Calibration"
> "A reader may object that the proposition protects the ranking only with"

The device is used twice with the same opening words and the same reply
("The objection is correct" / "That is right"). It is effective the first
time. Vary the second: "The proposition protects the ranking only for
equal priors, and the restriction is real."

**NB-17** · minor · thought · "Calibration"
> "pixel counted twice, and hundreds of such groups add up to the median score"

"Hundreds of such groups" is not measured anywhere; one pair of pixels
was. Exercise 10(c) estimates the effective duplication from the variance
of the score gap. Write "many such groups" and point to that exercise, or
quote its number if temperature scaling is promoted into the body (NB-S3).

**NB-18** · minor · pedagogy · code, imports and tabs
> "After the images are loaded, the code is plain NumPy, the same in every"

The sentence is followed by four import cells that load `torch`, `nn`,
`F`, `mxnet.autograd`, `gluon`, `optax`, `nnx` and `math`, none of which the
section uses. A reader who opens the PyTorch tab sees a deep learning
import block above a section with no tensors. Import `numpy` and `d2l`
only. Also `rng = np.random.default_rng()` is created three times; create
it once. The four loader cells differ only in how the arrays are read,
which is the right use of tabs.

## Exercises

Mathematically sound where I checked: 1(a) (all four scores tie), 2(a,b),
3(a) ($|x|>\sqrt{(8/3)\log 2}\approx1.36$), 3(c) (score difference
$2\rho x_1x_2/(1-\rho^2)$; Bayes error $\tfrac12-\tfrac1\pi\arcsin\rho$,
0.468 and 0.144), 4(a) (confirmed on the data: with $\alpha=0$ five class
scores are NaN for every image and `argmax` returns trouser 10,000 times),
5 (the two event models disagree on the test document), 7(a,b), 8, 9(a),
10(c), 12(a) ($m\ge2{,}209$). The grading marks and the chain from
duplicated features to score gaps to temperature are well designed.

What to change:

- **1(d)** repeats the three-bit example whose answer the text already gives
  ("The pattern $(1,1,1)$ therefore has probability $0$, while the product
  of the three marginals assigns it $\tfrac18$"). Cut the part or cut the
  answer from the text.
- **7(a)**: the text already lists the claimed confidence for five copies as
  0.999. Ask for 99.99% instead, or drop the part.
- **12** is a small research project in five parts that depends on exercises
  2, 9 and a theorem of the cited paper. Split it: (a) and (b) are a good
  Hoeffding exercise on their own and the only place the section uses
  `sec_mdl-concentration-generalization`; (c) and (d) are the learning-curve
  experiment; (e) can go.
- **10(d)** depends on the test set built in 8(b) and on 10(a); it is the
  third level of dependency. Move it into exercise 8 as its last part.
- Exercise 5 should lose its setup to the body (NB-S2); exercise 10(a,b)
  likewise if temperature scaling is promoted.
- With these changes the set drops to about ten exercises and 1,600 words.

## Still open from October

`writing-review-2026-10.md` has only a table row for this chapter ("each
hedged"); I found no hedging problem in this section now.
`standalone-motivation-mathematics.md` asked for an opening built on the
exponential table and the factorization; that is done and can be closed.

## For the chapter review

- This is the chapter's only case study and its easiest section. It assumes
  the coin of `sec_mdl-distributions` and `sec_mdl-exponential-family` (Beta
  posterior mean), `sec_mdl-maximum_likelihood` (frequency as the MLE),
  `sec_mdl-statistics` (standard error, bootstrap), and from the main text
  `sec_softmax`, `sec_softmax_concise` and `sec_classification`. It uses
  nothing from the three sections before it except in exercise 12(a).
- Keep it last, name it as the capstone in its first paragraph, and add the
  two short links to the late sections (a Hoeffding or standard-error number
  for the 7,840 frequencies; the train/test gap). Otherwise the chapter's
  order reads as tools, harder tools, hardest tools, then an unrelated easy
  topic.
- The section sets up two entries of `further-reading.md` ("Classifiers
  from Counts"). If it moves, they move with it.
- Pattern worth checking in the other sections: a result announced in the
  introduction, again by forward pointers ("We will see that"), measured,
  then recapped in a closing paragraph and in the Summary. Here that makes
  six tellings of one idea.
- Pattern worth checking: exercises that hold main-line content and run to
  half the length of the body.
- Method worth copying into the other sections: a hand-worked toy with
  exact numbers that returns each time a new idea arrives, and a question
  before every cell.
