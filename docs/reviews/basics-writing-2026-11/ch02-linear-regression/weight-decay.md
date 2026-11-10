# Weight Decay

Source: `chapter_linear-regression/weight-decay.md`, about 4,500 words of prose before the exercises (all tabs; about 4,200 for the PyTorch reader).

## Verdict

An ambitious section with some of the chapter's best passages and its most serious problem of substance. The best: the opening example of two near-duplicate features, where the weights $(1, 1)$ and $(501, -499)$ fit equally well and differ by 1,000 on a new example; the reading of the update as multiplication by $1 - \eta\lambda$; the shrinkage factor $\mu_j/(\mu_j + \lambda)$ derived in ten clean lines; and the two-feature experiment in which ridge returns $(1, 1)$ where least squares returns $(-0.84, 2.85)$. The problem: the section's main experiment (200 features, 20 examples) does not show weight decay doing what the section says it is for. By the text's own analysis the penalty helps there only by erasing the random initialization, a model started from zero does as well without it, and (which the text does not say) the regularized model is no better than predicting the same constant for every input. The prose spends some 500 words explaining what the experiment fails to show. The most important fix is to make the duplicated-features example the main experiment, and to redesign or drop the high-dimensional one.

Stored outputs agree with the prose throughout (validation losses 0.0195 and 0.0104, norms 0.132 and 0.032, eigenvalues 5.3 to 16.5, df 15.1, parts 0.042/0.125 and 0.032/0.004, concise 0.0207).

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 3 | 3 | 4 | 3 |

## What works

- **The opening example.** "The weights $(1, 1)$ fit the training data. The weights $(501, -499)$ fit them almost as well ... On a new example whose two measurements differ by 2, the predictions of the two weight vectors differ by 1,000." The problem is felt before any penalty appears, and the numbers return under the Cauchy-Schwarz bound (norms 1.4 and 707).
- **The bound and its limit.** After $|\mathbf{w}^\top \mathbf{x} - \mathbf{w}^\top \mathbf{x}'| \leq \|\mathbf{w}\|\, \|\mathbf{x} - \mathbf{x}'\|$ comes "A model with small weights cannot amplify a small change of its input", and then the boundary: "This reading relies on the model being linear, so that its weights are the slopes of the function it computes." The closing returns to it as the open question. Meaning, formula, boundary.
- **The name explained by arithmetic**: "each step keeps 97% of the weights, and 120 steps keep $0.97^{120} \approx 2.6\%$".
- **Why the bias is not penalized**: "adding a constant to every label should move the fit by that constant at no cost". An objection anticipated and answered in one sentence.
- **$\lambda = \sigma^2/(n\tau^2)$ read aloud**: grows with the noise, grows as the prior narrows, falls as $1/n$.
- **Shrinkage factors.** The derivation in the eigenbasis is short, and "A direction counts as one parameter if it is kept, as none if it is suppressed, and as a fraction in between" turns the effective number of parameters into something a reader can say.
- **"Two Duplicated Features."** Twenty lines of NumPy, two training sets, two test sets; least squares returns $(-0.84, 2.85)$ and $(-1.58, 3.57)$, ridge $(1.00, 1.00)$ and $(0.99, 0.99)$; the error on disagreeing inputs is predicted by a formula ($0.01\,(w_1 - 1)^2 + 0.01$, which gives 0.044 against the printed 0.042). This is the experiment the section should be built on.
- **Exercises 3, 5 and 6** (see below).

## Structural findings

- **The main experiment does not support the section (major).** See WD-1.
- **Four ideas in one section.** The penalty and the decay step; the $\ell_1$ penalty and the geometry of constraint regions; the Gaussian prior; the spectral analysis. At 4,500 words it is at the house ceiling, and 1,300 words of theory pass before the first experiment. The $\ell_1$ subsection and its two-panel figure serve a method the section then sets aside ("The rest of this section uses the $\ell_2$ penalty"). It is a candidate for a compact paragraph plus exercise 1, or for the chapter's further reading.
- **The opening example is abandoned for 450 lines.** The duplicated features are introduced in the second paragraph, referred to under the bound and under $\ell_1$, and tested only in the third subsection of the third part. Fix: run the "Two Duplicated Features" experiment directly after the penalty is defined, by the closed form or by SGD, and derive the shrinkage factors afterwards as its explanation.
- **A third round of factor-of-2 conversions.** See WD-7.

## Passage-level findings

**WD-1** (major, thought and pedagogy). "Weight Decay from Scratch" and "The Directions Our Data Determine".
Quote: "helped by removing the random initialization from the directions that 20"
The experiment trains 200 weights on 20 examples with true weights of 0.01 each. The text's own analysis concludes that weight decay "helped by removing the random initialization", that "Shrinking the determined part does not help here", and that "Started from $\mathbf{w} = \mathbf{0}$, the unpenalized model does about as well". The summary ends on the same note. So the headline demonstration of regularization shows an artifact of initialization that a different starting point removes for free. There is a further point the text omits. By the expected-loss formula of the scratch section, a model that ignores all 200 features and predicts $b^* = 0.05$ has loss $\frac{1}{2}(\|\mathbf{w}^*\|^2 + \sigma^2) = \frac{1}{2}(0.02 + 0.0001) \approx 0.01005$. The regularized model reaches 0.0104 in the stored run. Within the sampling error of a 1,000-example validation set (about 0.0005) the "repaired" model is indistinguishable from the constant predictor, and by the criterion the previous section gave ("A model is useful only if its mean squared error falls clearly below $\textrm{Var}(y)$") it is not useful. The text's lower bound of 0.009 shows why: only a tenth of $\|\mathbf{w}^*\|^2$ lies in directions the data can see. The opening's promise to "watch it repair a model with 200 features trained on 20 examples" is therefore not kept. The writing guide (section 9) is explicit: "fix the experiment or remove it. Do not keep it and explain what it fails to show."
Fix, in order of preference. (a) Make the duplicated-features problem the main experiment, trained once by the closed form and once by SGD with `WeightDecayScratch`, so that the from-scratch code, the decay step and the shrinkage analysis all refer to a case where the penalty does what the section claims. (b) If a high-dimensional experiment is wanted, design one in which the penalized model clearly beats both the unpenalized model started from zero and the mean predictor: for example features with a decaying spectrum (some directions poorly determined, as in the twins), label noise large enough to matter, and more examples than 20, and draw the mean-predictor loss as a horizontal line on the plot. (c) Keep the honest analysis of the present experiment as an exercise; exercise 8 ("Starting point") already contains most of it.

**WD-2** (moderate, pedagogy). Under "A Gaussian Prior on the Weights" and "Training with and without the Penalty".
Quote: "The penalty also has a probabilistic reading, which says how large"
The text says the prior "says how large $\lambda$ should be" and derives $\lambda = \sigma^2/(n\tau^2)$. The experiment then uses $\lambda = 3$ without comment. With the experiment's own values ($\sigma = 0.01$, $n = 20$, weights of size 0.01) the formula gives $\lambda = 0.05$, sixty times smaller; only exercises 3 and 5(c) reveal this. A reader who connects the two subsections is left with a contradiction. Fix: one or two sentences at the experiment saying what the formula suggests and why the run uses a larger value (at $\lambda = 0.05$ and 120 steps the decay factor $(1 - \eta\lambda)^{120} \approx 0.94$ would leave the initialization almost untouched). Under fix (a) of WD-1 the question can be asked of the twins data instead.

**WD-3** (moderate, language). First paragraph.
Quote: "$x_1^2 x_2$ whose exponents add up to $q$, is $\binom{k-1+q}{k-1}$. For $k = 20$ there are 20 features of"
The section opens with a definition of monomials, a binomial coefficient and two symbols ($k$, $q$) that are never used again. The three numbers that follow make the point alone. Draft: "Choosing the degree is a choice of features, and with many inputs it is a coarse one: 20 input variables have 20 features of degree 1, 210 of degree 2, and 1,540 of degree 3."

**WD-4** (moderate, explanation). Second paragraph.
Quote: "weight vectors by how well they fit the noise in the training labels, and"
The sentence that explains why least squares returns large opposite weights is the hardest in the opening: "since the small differences $x_1 - x_2$ are noise as well, it can return large weights of opposite sign". It compresses the whole mechanism (a tiny denominator) into a subordinate clause. Draft: "The data say almost nothing about how to split the weight between the two features, because the two differ so little. Least squares settles the split by fitting the noise in the labels, and to change the predictions through a difference $x_1 - x_2$ of a thousandth it needs weights in the hundreds."

**WD-5** (minor, explanation). Under "The $\ell_1$ Penalty and Sparsity".
Quote: "along the directions that the data determine poorly. The constrained"
"directions that the data determine poorly" is used here as if known; it is explained 250 lines later under "Shrinkage Factors". Either gloss it ("directions along which changing the weights barely changes the training predictions") or drop the clause.

**WD-6** (moderate, explanation). Under "The Directions Our Data Determine", first paragraph.
Quote: "use the raw inputs, not centered ones, because the gradient with respect to"
Before the reader sees the eigenvalues, the text defends a choice (raw versus centered inputs) and reports what the other choice would have given ("lower $\textrm{df}(3)$ a little, to about 14"). This is a reviewer's caveat, placed where it delays the result. The preceding subsection also said the analysis applies "after the features and the labels are centered", so the reader is now told the opposite was done. Fix: cut; if needed, one clause after the result.

**WD-7** (moderate, thought). Under "Concise Implementation".
Quote: "with the halved learning rate the weight decay must double, to 6, for each"
The reader has learned $\lambda = 3$ and now sees `wd=6, lr=0.005`, then a validation loss "near 0.02, which corresponds to 0.01 on the scale of our loss". This is the third section in which the factor $\frac{1}{2}$ between the book's loss and the framework's has to be converted, and here it makes a three-line API demonstration look like a different experiment. Fix at the source (see the chapter file); failing that, have `WeightDecay.loss` return half of the framework loss so that `wd=3, lr=0.01` reproduces the scratch run with no conversion paragraph.

**WD-8** (moderate, language). Last two paragraphs of "The Directions Our Data Determine".
Quote: "its dot product with $\mathbf{x}$ to the prediction, an error with variance"
Two long paragraphs carrying about twenty numbers (0.13, 0.11 to 0.15, 0.134, 0.018, 0.009, 0.026, 0.003 to 0.004, 4 to 19, 0.02, 0.018, 0.009, 0.01, ...). Each step is right, and the agreement of $\frac{1}{2} \cdot 0.018 = 0.009$ with the gap between the two validation losses is satisfying, but the reader has to keep a ledger. Fix: a four-row table (part of the weights; norm at $\lambda = 0$; norm at $\lambda = 3$; contribution to the validation loss) and three sentences. Under fix (a) or (c) of WD-1 most of this moves to an exercise.

**WD-9** (minor, explanation). Notation.
Quote: "one per dimension, with real eigenvalues $\mu_1, \ldots, \mu_d$"
$\mu$ was the mean of the normal distribution in the first section, and `sigma=0.01` (initialization) and $\sigma = 0.01$ (noise) both enter the arithmetic of this section (see SCR-7). Neither is fatal; a reviser choosing symbols should know.

**WD-10** (minor, language). Summary.
Quote: "initialization from the 180 directions that 20 examples cannot see;"
Nine sentences, the last of which ends "started from zero, the unpenalized model does about as well". A summary that closes by withdrawing the section's experiment leaves the reader unsure what was shown. Once WD-1 is fixed, close on the duplicated features and the effective number of parameters.

## Exercises

Eight problems, strong and well connected to the text. Exercise 2 (shrink-then-step against penalize-then-step under a rescaled gradient, adapted from Prince) prepares decoupled weight decay without naming an optimizer the reader has not met. Exercise 3 asks the reader to compute the prior that $\lambda = 3$ implies and to notice that it is inconsistent with the data; it is the honest answer to WD-2 and should be promoted into the text. Exercise 4 (ridge as least squares on augmented data, adapted from Hastie et al.) and exercise 5 (the $\lambda$ sweep against df) are good. Exercise 6 (why early stopping cannot do here what weight decay does) is subtle and worth keeping only if the 200-feature experiment stays. Exercise 7 (duplicated features: sweep $\lambda$; gradient descent from zero for $10^2$, $10^4$, $10^6$ steps) is excellent and shows that the twins problem can carry the section. Exercise 1 defines soft-thresholding in passing; fine.

## Still open from October

"AdamW appears before Adam": resolved; the section now gives one forward pointer and an exercise. "The SVD and effective-degrees-of-freedom material relies on the appendix": resolved in the sense that the eigenvector derivation is now self-contained; the October recommendation to cut effective degrees of freedom was not followed, and I think rightly, since df is what makes "continuous control of capacity" precise. The constrained-form equivalence under the $\ell_1$ subsection still rests on an appendix pointer.
