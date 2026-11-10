# Generalization

Source: `chapter_linear-regression/generalization.md`, about 4,100 words of prose before the exercises (no per-framework prose; the experiment is NumPy only).

## Verdict

The best section of the chapter, and the one closest to the bar. One small experiment, a noisy cubic fitted by polynomials of degree 1 to 19 on 20 points, carries the whole argument: three panels show underfitting and overfitting, a sweep shows the two error curves, eight lines of code compute bias and variance separately, a loop over training-set sizes shows what more data cures, and cross-validation then selects the degree without the test set. Each step answers a question the previous one raised ("From degree 3 on, every model class contains the true cubic. Why, then, does test error rise again?"). The prose is written to survive re-execution ("Every run draws new data, so the numbers change from run to run, but the pattern does not"), and it uses no training scaffolding at all. The flaws are local. The most important one is terminological: the chapter has plotted "validation" losses for four sections, and this section introduces held-out data as a "test set" and reaches "validation set" only 300 lines later.

Stored outputs agree with the prose, with one mismatch of scale noted in GEN-4.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 4 | 5 | 4 | 4 |

## What works

- **The opening contrast.** "Given 20 measurements at distinct inputs, a polynomial of degree 19 passes through every one of them exactly, yet nothing guarantees that it predicts a 21st measurement well." The section's experiment is announced in its first paragraph as a fact the reader can picture.
- **The lookup-table extreme** under "Training Error and Generalization Error": a model that stores the training set "has zero training error, yet it has nothing to say about an input it has not seen". An extreme case fencing a claim, as the guide asks.
- **"The weights show why."** The largest weight of each fit is printed next to its errors, and the text reads it: the degree-19 fit uses huge weights "whose huge positive and negative terms cancel at the 20 training inputs and almost nowhere else". This one observation motivates the whole next section.
- **Bias and variance, computed.** The derivation names its three terms, shows why the cross terms vanish, warns that "This bias, an error of the fitted function, is unrelated to the bias $b$ of a linear model", and then measures both terms with an elegant trick: because least squares is linear in the labels, "The average of the fits over many draws of the noise is therefore the fit to the noiseless labels". Code that teaches.
- **"Underfitting is error from bias, and overfitting is error from variance."** A sentence a student will remember.
- **Popper** under "Model Complexity": a class that can fit any labels learns nothing from fitting these. Brief, apt, and it does not carry the argument.
- **"That was a demonstration, not a method."** The opening of "Model Selection" admits that the experiment so far cheated by reading the test error, and so motivates validation.
- **The cross-validation post-mortem** explains why the cross-validation curve rises earlier than the test curve (16 points instead of 20; validation points beyond the range of the fitted points) instead of merely showing the plot.

## Structural findings

- **Three ideas, held together by one experiment.** Risk and its estimate; bias, variance and capacity; model selection. At 4,100 words it is within the ceiling and I would not split it, because the shared experiment is what makes it good. If anything is trimmed, it should be asides (GEN-3, GEN-7), not the experiment.
- **The scaffolding is absent and nothing is lost.** The section uses neither `Module` nor `Trainer`. That it is the clearest section in the chapter is relevant to the chapter-level question about the software framework.
- **The imports cell arrives mid-section**, after the prose of "Polynomial Curve Fitting", and loads a deep learning framework that the text says the experiment does not need. Harmless to the reader of the rendered page, but an editing agent should move it to the top with the others.

## Passage-level findings

**GEN-1** (minor, explanation). Under "Training Error and Generalization Error".
Quote: "$$R_\textrm{emp}[\mathbf{X}, \mathbf{y}, f] = \frac{1}{n} \sum_{i=1}^n l(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)})),$$"
The notation is heavier than the idea: a three-argument functional, a three-argument loss, and in the next display a double integral over $p(\mathbf{x}, y)$ that is never used. The section writes $R$ and $R_\textrm{emp}$ afterwards. Fix: $R_\textrm{emp}(f) = \frac{1}{n}\sum_i l(f(\mathbf{x}^{(i)}), y^{(i)})$ and $R(f) = \mathbb{E}[l(f(\mathbf{x}), y)]$, with the expectation explained in words as it already is.

**GEN-2** (moderate, explanation). Under "Training Error and Generalization Error" and "Validation Sets".
Quote: "report results on also serves to compare models, so unless stated otherwise,"
Since the software design section the chapter has spoken of "validation data" and plotted `val_loss`. This section's first held-out set is "a *test set*", used for 300 lines, and the distinction between test and validation arrives only under "Model Selection", with the quoted sentence then explaining that the book's reported numbers are validation numbers. A reader arriving from the previous sections wonders whether the earlier "validation loss" was a test loss. Fix: in the first subsection call the examples "held out" and add one sentence: "The validation loss that our trainer has plotted since the scratch section is such an estimate." Keep the test and validation distinction where it is.

**GEN-3** (moderate, thought). Under "Polynomial Curve Fitting".
Quote: "variance that the model explains, is the *coefficient of determination*"
A paragraph that introduces the variance of the labels as a baseline and $R^2$, between the description of the data and the code that draws them. Neither is used again in this section or the next. Either cut it, or, better, use it: the baseline "predict the mean label" is exactly what the weight decay experiment of the next section lacks (see WD-1 there). If the paragraph stays, the printout of the three fits should include the baseline $\textrm{Var}(y)$ so the reader sees degree 1 against it.

**GEN-4** (moderate, correctness). After the printout of the three fits.
Quote: "noise level and a test error more than a thousand times larger than that of degree"
The stored output reads `degree 19: train 4.8e-15, test 2.3e+14, largest |weight| 5.2e+09`. The prose says "more than a thousand times larger" and "weights of a million or more". Both statements are true and both understate what the reader sees by ten orders of magnitude or more. A reader who meets a test error of $10^{14}$ beside prose that speaks of a factor of a thousand will suspect a bug. At this size the number also owes something to the conditioning of the monomial basis, which the text does not mention. Fix: say what happens. Draft: "Degree 19 has a training error of zero up to rounding and a test error that is astronomically large, $10^{10}$ or more in most runs: between the training points the polynomial swings to values in the thousands or millions. (Powers of $x$ are also a numerically poor basis at this degree, which makes the swings worse; exercise ... returns to this.)" The three-panel plot already clips the swings at its `ylim`, which is the right choice.

**GEN-5** (minor, pedagogy). The three-panel plot.
Quote: "xlabel=f'x (degree {d})', ylim=(-7, 11), fmts=('--', '-', 'o'),"
The degree is smuggled into the axis label, and at a figure size of 9 by 2.8 inches the tick labels and legend render very small beside the other plots of the chapter. These three panels are the picture the section rests on. Fix: per-panel titles ("degree 1", "degree 3", "degree 19", which are functional titles) and a larger figure.

**GEN-6** (moderate, thought). End of "Dataset Size".
Quote: "grow with the amount of data, not ahead of it."
The rule is stated flatly one subsection after the text reported that large networks fit random labels yet generalize, and three chapters before the book trains models with far more parameters than examples. The reader is given a rule and its counterexample in the wrong order and without a link. Fix: scope it. "For models like these, whose capacity grows with the number of weights, complexity should grow with the amount of data, not ahead of it. Deep networks will complicate this rule (:numref:`sec_generalization_deep`)."

**GEN-7** (moderate, explanation). Under "Bias and Variance", end of the derivation paragraph.
Quote: "The expectation here holds the training inputs fixed and redraws only the"
Three sentences on fixed versus random designs ("so on average the variance is somewhat larger"). The caveat is correct, but a first-time reader has just absorbed the decomposition and cannot use the refinement; nothing later depends on it except exercise 4(c), which raises it itself. Fix: cut from the text; the exercise already asks "why does the formula of (b), which holds at the training inputs, not apply exactly to the test inputs?"

**GEN-8** (minor, explanation). After the bias and variance plot.
Quote: ":exref:`ex_generalization-optimism` shows that this holds exactly on average"
"on average over the training inputs" can be read as an average over random draws of the inputs. The exercise shows it for the average over the $n$ fixed training inputs. Reword: "exactly, when the variance is averaged over the $n$ training inputs instead of the test inputs".

**GEN-9** (minor, language). Under "Validation Sets".
Quote: "complexity is tightly controlled :cite:`ong2005learning`. Nor can we select on"
"it can be severe even for models whose complexity is tightly controlled" asserts something the reader cannot picture and only a citation supports. Either give the example in a clause or drop the clause.

**GEN-10** (minor, explanation). Notation across the chapter.
Quote: "degree $d$ predicts"
$d$ is the number of features in the first section and in the weight decay section ($d = 200$), and the polynomial degree here, where the number of weights is $d + 1$; the exercises of this section use $p$ for the number of columns. Fix: use another letter for the degree (the weight decay opening uses $q$ for exactly this), and $p = q + 1$ consistently for the number of weights.

## Exercises

Eight problems. Exercise 4 (optimism of the training error, $\sigma^2(n-p)/n$ against $\sigma^2(n+p)/n$) is first-rate: a short derivation with a hint, then a comparison with the section's own experiment. Exercise 6 (how often cross-validation picks degree 3, and what a wrong pick costs) backs the claim "about four runs out of five" in the text and teaches that a selection method has an error rate. Exercises 7 and 8 (bias of $K$-fold under $R(m) = R_\infty + c/m$; the leave-one-out shortcut by Sherman and Morrison) are classical and well posed. Exercise 5(a) keeps the old "your manager asks" framing but is now a concrete learning-curve task. Exercises 1 and 2 are discussion questions; they are fine as warm-ups. If GEN-4 is adopted, a short exercise comparing the monomial basis with Legendre or Chebyshev polynomials at degree 19 would be a natural addition. Nothing to cut.

## Still open from October

"`generalization.md:203–209` says the same thing three times": resolved. Double descent, which the October review asked to keep once in chapter 4, is no longer in this section.
