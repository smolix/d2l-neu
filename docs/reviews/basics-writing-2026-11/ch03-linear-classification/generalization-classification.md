# Generalization in Classification

Source: `chapter_linear-classification/generalization-classification.md`. About 3,650 words of body prose, 910 words of exercises, one figure, three computed cells (NumPy only, identical in all tabs).

## Verdict

The best-argued section of the chapter, and its first two parts are at the level of the model texts. The section starts from the number the reader has just measured (83%), says in plain words why it cannot be read at face value, and turns that into three questions. The first two are answered completely, with arithmetic the reader can redo and two small simulations that each show one thing. The treatment of test set reuse is outstanding: 10,000 coin-flipping "classifiers" reach 56% by selection alone, and a majority vote over the lucky ones reaches 70% on labels it "never saw". The third part, statistical learning theory, is as short and as honest as this material can be made (finite classes, why that fails for linear models, shattering with a picture, the VC bound stated and then shown to be loose for the chapter's own model and empty for deep networks). It is pitched a step above the rest of the chapter, and it has one slip that matters: the theory is stated for binary classifiers and then applied to ten-class Fashion-MNIST without comment. The most important fix is to connect the theory to the model the reader trained: state the binary restriction, and put the observed gap between training and test error next to the bound.

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 5 |

## What works

- The opening: "We would like to read that number as the accuracy the model will have on clothing images it has never seen, and two things stand in the way." The three numbered questions are in the reader's words, and the paragraph after them says in advance which answers will be sharp and which loose, and why the reader should still care.
- The coin: each test example is "like a coin that comes up heads with probability $\epsilon(f)$". Everything in "The Test Set" follows from that one image.
- The arithmetic: standard error 0.0036 for the chapter's model, radius 0.007, $n \ge 9{,}604$ for one percentage point in the worst case, 18,445 with Hoeffding. All correct, each drawn from the formula just given.
- "An accuracy reported as 84.37% has more digits than such a test set supports." Brief, concrete, true, and the reader will remember it when reading papers.
- "More test data does not change the error of the classifier, only our uncertainty about it."
- The selection experiment and its reading: "No classifier learned anything; the climb comes from selection alone."
- The adaptive-overfitting experiment. Four lines of code, a result of 0.707 against a best single guesser of 0.551 in the stored run, and the explanation: "Each test score revealed a little about which guesses agree with the labels, and the vote added up this information over a thousand scores."
- The bridge into theory: "the learner is a modeler who evaluated every classifier it could choose on the same data and kept the best: selection among many classifiers, taken to the extreme." The third part is thereby the second part with $k = |\mathcal{F}|$, which is exactly Strang's "keep the form of the simple solution".
- The memorizer as a recurring counterexample: introduced in the opening, used to show that a class property must enter any bound, shown to have infinite VC dimension, and excluded by the bound "as it must".
- The counting example for lines in the plane ($n^2 - n + 2$ labelings: 8, 14 of 16, 92 of 1,024), which makes Sauer's lemma a generalization of something seen.
- The candour about the bound: "The guarantee is loose", with $\sqrt{785/60{,}000} \approx 0.11$ set against the $\pm 0.01$ of a test set.

## Structural findings

- **Pitch of the third part.** Parts one and two need only the coin and the union bound. Part three introduces hypothesis class, uniform convergence, shattering, VC dimension, convex hulls, Sauer's lemma and a bound with an unnamed constant in about 1,300 words. A reader at chapter 3 can follow it, because each notion is tied to the memorizer or to lines in the plane, but it is the densest stretch of the chapter and its practical conclusion is negative. Two measures would make the pitch right without cutting the idea: mark "Statistical Learning Theory" as a part that can be skipped on first reading (the opening already half says so), and remove the general-dimension proof sketch (GEN-4).
- **No contact with the trained model.** The section opens with the chapter's 83% and uses 0.15 as its running error rate, but the theory part never returns to the model. The reader is told that a linear classifier's training error is close to its population error once $n \gg d$, and is not shown the two numbers. See GEN-2.
- **The first simulation shows the uninteresting case.** See GEN-3.
- **Dependence on other chapters** is handled well: the risk and the validation discipline of `sec_generalization_basics` are restated in a sentence each, and the three appendix pointers add proofs without carrying steps.

## Passage-level findings

**GEN-1** (moderate, correctness). Anchor: "This answers the third question. A linear classifier on $d$ features has VC", under "The VC Bound".
The hypothesis class was introduced as binary classifiers $\operatorname{sign}(\mathbf{w}^\top\mathbf{x} + b)$ with labels $\pm 1$, and the VC dimension is defined for binary labelings. The chapter's model has ten classes and 7,850 parameters. The next paragraph applies the result "With $d = 784$ pixels as features, as in Fashion-MNIST" as if the ten-class model were covered. Say what is being bounded: "For a binary problem on Fashion-MNIST images, such as telling shirts from coats, $d = 784$ ...", and add a clause that the multiclass analogue grows with the number of parameters $dq$ and tells the same story. Without this, the careful reader sees a theorem about one object applied to another.

**GEN-2** (moderate, pedagogy). Anchor: "The guarantee is loose. With $d = 784$ pixels as features, as in", under "The VC Bound".
"Loose" is asserted by comparing the bound with the precision of a test set. The more telling comparison is with what actually happened: softmax regression's training error and test error on Fashion-MNIST differ by a point or two (the training accuracy must first be reported in `softmax-regression-scratch.md`, see SCR-4 there). "The bound allows a gap of $0.11c$; the observed gap is about 0.02" is the sentence that tells the reader how to hold a worst-case bound.

**GEN-3** (moderate, pedagogy). Anchor: "We check both radii by simulation, for a classifier with population error", under "Simulated Test Sets".
At $\epsilon = 0.15$ and $n \ge 100$ the normal approximation is excellent, so the experiment confirms a formula and the plot is three parallel lines. The paragraph before it has just named the case where the approximation fails ("when $\epsilon(f) = 0.001$ and $n = 1{,}000$, so that we expect a single error"), and that case is left to exercise 2. Add the small-error case to the same cell (for instance $\epsilon = 0.01$ as a second set of curves, or the coverage of the plug-in interval), so that the simulation shows why a finite-sample guarantee is worth having. An experiment in which the simple method simply works does not motivate the conservative one.

**GEN-4** (moderate, explanation). Anchor: "be split into two groups whose *convex hulls*, the sets of their weighted", under "The VC Dimension".
The paragraph proves the general result ($d + 1$) by sketch: a lower bound by construction, then an upper bound by a partition into two groups with intersecting convex hulls (Radon's theorem, unnamed), with convex hull defined in an apposition. The reader has just seen the planar case argued fully from a figure; the general case adds a definition and an unproved partition claim in one 45-word sentence. State the result, give the lower-bound construction (which is two clauses and instructive: the bias sets the label of the origin, each weight the label of one unit vector), and send the upper bound to the exercise that already proves it. October asked for Radon's theorem to be cut; it survives in this sentence.

**GEN-5** (minor, explanation). Anchor: "labelings instead of over the classifiers. The term $\log|\mathcal{F}|$ in", under "The VC Bound".
"is then replaced by roughly ... and a finer analysis removes the logarithm" followed by "there is a constant $c$". The reader cannot tell which statements were derived and which are being reported. One clause fixes it: "We state the result without proof (:numref:`sec_mdl-generalization-bounds`)". The pointer currently comes a page later, in "Limits of Worst-Case Bounds".

**GEN-6** (minor, correctness). Anchor: "On its own, each test error lies within the Hoeffding radius with probability", under "Selection Among Many Classifiers".
"with probability 0.95" should be "with probability at least 0.95"; Hoeffding gives a bound, and the simulation above has just shown that the true coverage is higher.

**GEN-7** (minor, thought). Anchor: "we call its half-width", under "Confidence Intervals".
The standard caveat about confidence intervals is handled correctly ("contains the population error for 95% of test sets"). But the sentence also coins "radius" and substitutes the estimate for the unknown error in one breath. Split it: first the interval, then "in practice we do not know $\epsilon(f)$ and use $\epsilon_\mathcal{D}(f)$ in its place", then the number for Fashion-MNIST.

**GEN-8** (minor, language). Anchor: "the purest case: a test set of", under "Selection Among Many Classifiers".
The paragraph setting up the coin-flip experiment is nine lines and ends with a derivation of the reference curve that is harder than the experiment ("Each of the two directions of the deviation ... has probability at most $\exp(-2nt^2)$, so the best of $k$ accuracies exceeds $0.5 + t$ with probability at most $k\exp(-2nt^2)$, which equals 1 on the curve"). Move the derivation after the plot, as the explanation of why the simulated curve stays below the reference curve.

## Exercises

Nine problems. The first four are among the best in the chapter. "Coverage at small error rates" ends with a question that exposes the plug-in interval ("What does this plug-in interval report for a test set with no errors?"). "Leaking a test set" makes adaptive overfitting exact with a counting argument. "A validation split for Fashion-MNIST" has the reader repair the chapter's own shortcut and check the result against the radius they learned to compute.

- Five of nine problems are on VC dimension, and two of them are marked hard. For a section whose practical message is in its first two parts, that is unbalanced. Keep "VC dimension of linear classifiers" (it completes GEN-4), "Collinear points" (a good check of the definition), and one of "polynomials" or "rectangles". Move "Composing hypothesis classes" to the appendix section on generalization bounds.
- Add one easy problem for part one: a paper reports 91.2% and 91.6% for two models on a test set of 10,000 images; is the difference meaningful, and what if both models were evaluated on the same images?

## Still open from October

"The VC bound appears before the VC dimension is defined" is fixed: uniform convergence, then the dimension, then the bound. "Cut Radon's theorem" is half done (GEN-4).
