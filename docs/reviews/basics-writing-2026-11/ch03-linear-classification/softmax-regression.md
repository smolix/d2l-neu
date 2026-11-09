# Softmax Regression

Source: `chapter_linear-classification/softmax-regression.md`. About 4,400 words of body prose (at the 4,500 ceiling of the writing guide), 1,050 words of exercises, four figures, no code.

## Verdict

This is the strongest exposition in the chapter and in places it meets the bar. One small example (a 2x2 image, three classes, logits (1.0, 2.2, 0.3)) is carried from the scores through the softmax, the loss, the gradient and the temperature figure, with every number checkable by hand. That is the Cover and Thomas method and it works. The derivation of cross-entropy from the categorical likelihood, the gradient "prediction minus label", and the side-by-side table with linear regression are as clear as I have seen them written. The section's fault is that it does not stop. After the loss is derived and explained, a third part ("Accuracy and Confidence") adds the margin, a bound on the error count, separable data, implicit bias, calibration and temperature scaling. Those 1,100 words teach a second idea, are compressed to the point of being hard to follow in one paragraph, and are then retold in `classification.md` and `softmax-regression-concise.md`. The single most important fix is to cut "Calibration and Temperature" down to what this section needs and let one later section own calibration.

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 4 |

The first two thirds would score 5 for explanation. The last third pulls it down.

## What works

- The opening argument against regressing on class codes, especially "charge 81 times more for mistaking a T-shirt (code 1) for an ankle boot". A number makes the objection concrete.
- The need for probabilities is motivated by decisions (spam filter, doctor) before any formula.
- "Class Labels" to "The Softmax": each object is introduced in words, then as a symbol, then with a number. The requirement on the softmax is stated before the function ("$q$ nonnegative numbers that sum to 1, with a larger score giving a larger probability").
- The worked numbers: exponentials (2.72, 9.03, 1.35), sum 13.09, probabilities (0.21, 0.69, 0.10); the log-odds check "$e^{1.2} \approx 3.3$ times as probable as the cat ($0.69 / 0.21$)"; the loss 0.37 for the chicken and 2.27 for the dog; the gradient (0.21, -0.31, 0.10). All correct, and each follows the formula it illustrates.
- "The entries sum to zero, as they must, since adding a constant to all logits leaves the loss unchanged." A check against intuition in Strang's manner.
- The table comparing linear and softmax regression row by row. It says exactly what is new and what is not.
- The honest-probabilities example: a blurry photo that is a shirt with probability 0.6, where predicting (0.6, 0.4) costs 0.67 nats and (0.9, 0.1) costs 0.98. This is the best single explanation in the section of why the loss is the right one.
- The closing paragraph names two open questions and each has a later section that answers it.

## Structural findings

- **Too many ideas for one section.** The section's idea is "one linear score per class, softmax, and the loss that maximum likelihood gives". Decision regions, the information-theoretic reading, the margin, the error bound, separability, calibration and temperature are seven further topics. The first three support the idea. The last four are the start of a different section ("what does a trained classifier's confidence mean"), and the chapter already has two places for that (`classification.md`, "Calibration"; `softmax-regression-concise.md`, "Calibration"). See SMR-8.
- **Two motivating examples.** The opening uses ten clothing categories, the body switches to cat, chicken, dog on four pixels, and the entropy example goes back to a shirt. The switch is announced ("a problem small enough to write out in full") and is acceptable, but the clothing store could itself be shrunk to three classes (shirt, coat, sandal, as in the chapter introduction) so that one example serves the whole chapter and the blurry shirt of the entropy example is at home.
- **Forward references.** The section points forward four times (`subsec_softmax-implementation-revisited`, `subsec_implicit_regularization` in chapter 4, `sec_training_recipes` in chapter 7, `chap_perceptrons`). The guide allows one. The pointers to chapters 4 and 7 can go with no loss (SMR-8, SMR-12).
- **The appendix is used correctly.** Three pointers (`subsec_mdl-glm`, `sec_mdl-information_theory`, `subsec_mdl-softmax-origins`) add depth and none carries a step of the argument. The October finding about five tangents is resolved: Gumbel is now an exercise, probit a clause in it, Boltzmann one sentence.

## Passage-level findings

**SMR-1** (moderate, explanation). Anchor: "Exponentiating each score makes it positive and keeps the order, and", under "The Softmax".
The text states what the map must do and then presents exponentiation as the way to do it. A first-time reader asks why the exponential, since other positive increasing functions, such as $\log(1+e^o)$, are also positive and order-preserving. The reason arrives implicitly five paragraphs later. Add one sentence here: with the exponential, adding a constant to a score multiplies its probability by a constant, so differences of scores become ratios of probabilities, and (as the section on the gradient will show) the loss gets the simplest possible gradient. Then the paragraph "Only the differences between logits matter" reads as the payoff of the choice.

**SMR-2** (moderate, language and thought). Anchor: "same invariance makes the softmax safe to compute. In single precision,", under "The Softmax".
The paragraph beginning "Only the differences between logits matter" carries four ideas: shift invariance, log-ratios, the redundant degree of freedom, and floating-point overflow at 88 with a forward reference. The overflow remark is premature (the reader has not computed anything yet) and the same fact is told again in the opening of `softmax-regression-scratch.md`, in its "Overflow" subsection, in the opening of `softmax-regression-concise.md` and in its "Overflow and Underflow" subsection. Cut the last two sentences of this paragraph. End on "so that every class is treated alike."

**SMR-3** (minor, explanation). Anchor: "separates, such as the two diagonals of a square, with one class at each", under "Decision Regions".
The exclusive-or configuration is described in a phrase that does not parse on first reading ("the two diagonals of a square, with one class at each pair of opposite corners"). Rewrite: "Four points at the corners of a square, with each class occupying two opposite corners, cannot be separated by any line." A third panel in `fig_mdl-clf-decision-regions` showing these four points would also give the reader the picture that chapter 4 opens with.

**SMR-4** (minor, pedagogy). Anchor: "class. This is the assumption we made at the start, and two common kinds of", under "Decision Regions".
Multi-label and ordinal problems are a scope note. Attached to the end of "Decision Regions" they read as a consequence of the geometry, which they are not. Move the paragraph to the end of "The Model" as its own short closing, or to the section opening where the assumption is first made. Each of the two citations is used once and never again; one sentence each is the right length, which it has.

**SMR-5** (minor, language). Anchor: "gives probability $0.103$ (0.10 after rounding), the loss would be", under "Maximum Likelihood".
The parenthesis repairs a rounding mismatch that the reader would not have noticed. Write "to which the model gives probability 0.10, the loss would be $-\log 0.10 \approx 2.3$". The second decimal is not needed for the point (a wrong label costs six times as much).

**SMR-6** (minor, explanation). Anchor: "the covariance matrix of the predicted distribution, which is positive", under "The Gradient".
Convexity is established in one sentence that relies on an exercise and on the reader knowing why a covariance matrix is positive semidefinite. For this reader, show the two-class case in a clause: the second derivative of $\log(1 + e^{-m})$ is $\sigma(m)(1 - \sigma(m)) > 0$, the variance of a coin with bias $\sigma(m)$. Then state that the general case is the same fact for $q$ classes and point to the exercise. The consequence ("every local minimum is a global one") deserves one more clause saying why the reader should care: gradient descent cannot get stuck, which will stop being true in the next chapter.

**SMR-7** (minor, thought). Anchor: "the exponential family of distributions, and for every member of this family", under "The Gradient".
The exponential family is named, used for one claim, and never appears again in the chapter. The table that follows makes the point without it. Reduce to "This is no coincidence (:numref:`subsec_mdl-glm`)" or drop the sentence.

**SMR-8** (major, explanation and structure). Anchor: "that :numref:`subsec_implicit_regularization` returns to. Large gaps between logits mean predicted probabilities close to 0 and 1. A", under "Calibration and Temperature".
This paragraph is where a first-time reader is lost. In nine sentences it introduces: separable training data, the absence of a minimizer, unbounded weight growth, convergence of the weight direction to "the separator with the largest margin", implicit regularization (with a pointer to chapter 4), the definition of calibration, and the overconfidence of deep networks. Three further problems:

1. "Margin" changes meaning without notice. Two paragraphs earlier it was defined as $m = o_y - \max_{j \neq y} o_j$, a property of one example. "The separator with the largest margin" is a geometric distance of a hyperplane from the data, which has not been defined.
2. The premise does not hold for the model of this chapter. Softmax regression ends training on Fashion-MNIST with a loss near 0.45 and a validation accuracy of about 83%, so it is nowhere near separating its training data, and the experiment in `softmax-regression-concise.md` finds the model slightly underconfident. The reader is prepared for a failure that the chapter never shows.
3. The source line itself shows the paragraph was patched: a sentence was inserted mid-line.

Fix: split and shrink. Keep two sentences on separable data (scaling the weights lowers every loss, so the loss has no minimum and the weights grow; exercise). Delete the max-margin and implicit-regularization sentence, which belongs in chapter 4 where it is developed. Then define calibration in its own paragraph. Better still, move the definition of calibration, the temperature formula, `fig_mdl-clf-temperature` and temperature scaling to the one later section that measures calibration on the trained model, and end this section after "The Error Rate" with the observation that the loss keeps rewarding confidence in decisions that are already right.

**SMR-9** (moderate, thought). Anchor: "average training loss of 0.07 nats misclassifies at most", under "The Error Rate".
The bound (number of errors is at most total loss divided by $\log 2$) is correct and neatly derived, but the example loss of 0.07 is chosen so that the bound looks informative (10%). The chapter's own model ends with a training loss near 0.45, for which the bound allows 65% errors while the model makes about 15%. A reader who does that arithmetic two sections later will feel misled. Say what the bound is for: it shows that pushing the loss toward zero forces the errors toward zero, and it is loose in practice. One sentence: "The bound is loose (a model with loss 0.45 may err on up to 65% of its examples and in practice errs on far fewer), but it shows that driving the loss down cannot leave the error rate high."

**SMR-10** (minor, explanation). Anchor: "(:numref:`fig_mdl-clf-margin-losses`), and therefore of the parameters.", under "The Error Rate".
"A step function of the margin, and therefore of the parameters" skips a step: the margin is continuous in the parameters, so the error is piecewise constant in them. The next two sentences say this correctly. Delete "and therefore of the parameters".

**SMR-11** (minor, language). Anchor: "occupation of energy states at temperature $T$. The appendix derives the", under "Calibration and Temperature".
The sentence about three derivations in the appendix interrupts the explanation of temperature between the origin of the name and the fact that matters ("Dividing by $T$ keeps the order of the logits"). Move it to the end of the paragraph or to the chapter's resource list.

**SMR-12** (minor, pedagogy). Anchor: "smoothing* (:numref:`sec_training_recipes`) uses this to soften one-hot", under "Entropy and Cross-Entropy".
Label smoothing is named with a pointer to chapter 7 and not used. The preceding sentence (the loss accepts labels that are distributions, such as fractions of annotators) makes the point. Cut the label-smoothing sentence.

**SMR-13** (minor, language). Anchor: "likelihood of the observed labels gives the cross-entropy loss", under "Summary".
The summary is one 190-word paragraph whose third sentence runs to about 60 words. It lists everything mentioned, which the guide's template asks summaries not to do. Four sentences suffice: the model, the loss and where it comes from, the gradient, and what the loss does and does not say about accuracy.

## Exercises

Eight problems, all with substance, none filler. The Bradley-Terry problem (independence of irrelevant alternatives through a "none" option), the gradient-descent invariant in the softmax-and-sigmoid problem, and the squared-error comparison are excellent: each makes the reader discover a property the text only states. Two changes:

- **Order and ramp.** The set opens with the Hessian, the most abstract problem. There is no problem a reader can do in two minutes to check that they can compute a softmax and a loss. Add one warm-up (for logits $(2, 0, -1)$ and each possible label, compute the probabilities, the loss and the gradient by hand; then predict what subtracting 5 from every logit changes) and move the Hessian after "Softmax and sigmoid".
- **Notation.** In "Temperature", $\bar{o}$ is the mean logit; in `softmax-regression-concise.md` $\bar{o}$ is the largest logit. Rename one.

## Still open from October

The five tangents are gone from the main line. What remains of them is SMR-7 (exponential family, one sentence) and SMR-11 (the appendix sentence). The transposed $\mathbf{W}$ is now explained ("Storing the features along the rows of $\mathbf{W}$ matches the minibatch form below"). The label encoding is stated once for formulas and once for code. Resolved.
