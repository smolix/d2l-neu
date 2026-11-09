# Measuring Classifiers

Source: `chapter_linear-classification/classification.md`. About 2,500 words of body prose, 990 words of exercises, one figure, one small computed cell plus the `Classifier` code.

## Verdict

This section has the sharpest example in the chapter (a screening test that accuracy ranks below a classifier that never looks at its input) and a clean derivation of the cost-sensitive threshold. It is also the section with the weakest structure. It teaches two unrelated things, a base class for the book's software and a set of evaluation measures, and it sits before the reader has trained any classifier, so every measure is illustrated on invented numbers. Its first third repeats the argument of the previous section about steps and margins. Its last part, on calibration, explains two notions of calibration in the wrong order and then demonstrates a reliability diagram with a made-up table, two sections before the chapter computes a real one that shows the opposite behaviour. The single most important fix is structural: keep the `Classifier` class here as a short section and move "Beyond Accuracy" to after the model is trained, where its confusion matrix and reliability diagram can be the real ones.

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 3 | 3 | 4 | 3 |

## What works

- The screening example under "Class Imbalance". The numbers are human-scale (100,000 people, 1,000 sick), the table is filled in before any term is named, and the conclusion is one flat sentence: "Accuracy ranks the useless classifier above the useful one." The follow-up, "the 1,000 sick people make up only 1% of the accuracy, however well or badly a classifier treats them", says why. This is MacKay-grade.
- "Recall reads the first column of the table, and precision reads its first row." A definition tied to the picture the reader is looking at.
- The threshold derivation under "Asymmetric Costs": expected cost of each decision, one inequality, and then the number that surprises, "a patient with a 2% chance of being sick should be flagged".
- The link back to the HIV test of the preliminaries, restated in one clause as the guide asks.
- The closing paragraph, which names what is still unknown (how far a number measured on a finite test set can be trusted).

## Structural findings

- **Two sections in one.** "Loss and Accuracy" and "The `Classifier` Class" (about 900 words with tabs) are software for the rest of the book. "Beyond Accuracy" (about 1,500 words) is evaluation methodology. The section opening has to announce both and its second paragraph changes subject twice. Neither half needs the other.
- **Placement.** "Beyond Accuracy" comes before the first training run of the chapter. The reader has now read about 8,600 words since the chapter began without seeing a classifier trained. Confusion matrices and reliability diagrams are then shown for hypothetical classifiers, and shown again for the real one in `softmax-regression-scratch.md` and `softmax-regression-concise.md`. If "Beyond Accuracy" followed the concise implementation, it could use the Fashion-MNIST confusion matrix and the real reliability table, the screening example would still open it, and the chapter's second half would read as one arc: what to measure, how much a measurement is worth, what happens when the data change.
- **Repetition of the previous section.** The opening retells the 0.51 to 0.99 example with the same numbers. "Loss and Accuracy" retells the margin, the step function, the zero gradient and the $\log 2$ bound from "The Error Rate" in `softmax-regression.md`. See CLS-2.
- **The default optimizer** is installed on `Module`, the class of the previous chapter, in a section about measuring classifiers. See CLS-3.

## Passage-level findings

**CLS-1** (major, pedagogy and structure). Anchor: "its loss. Accuracy has blind spots of its own. When one class is rare, or", section opening.
The second opening paragraph runs: every classifier reports two numbers; most train with the same optimizer; we write a base class; accuracy has blind spots; the last part shows them. Five subjects in six sentences, because the section has two jobs. Split as described under structural findings. If the section must stay whole, give "Beyond Accuracy" its own opening paragraph that begins from the reader ("We now have a number, the accuracy. Is a classifier with higher accuracy the better one?") and cut the preview from the section opening.

**CLS-2** (moderate, thought). Anchor: "showed, whether an example is classified correctly changes in a step when", under "Loss and Accuracy".
Third telling of the step argument in ten pages (`softmax-regression.md` "The Error Rate"; this section's opening; here), with the margin redefined and the $\log 2$ bound restated. The guide's rule to restate a fact from an earlier section in one sentence is being applied as "re-derive in a paragraph". Replace the paragraph by one sentence: "We report the accuracy but cannot train on it, since it changes only in jumps as the parameters change (:numref:`sec_softmax`)." `fig_mdl-clf-loss-accuracy` then carries the point.

**CLS-3** (moderate, pedagogy). Anchor: "We install the default on `Module` rather than on `Classifier`, so that", under "The Default Optimizer".
The reader learns that the linear regression models of the previous chapter had no default optimizer and that one is now being added to their base class from a later chapter. This is a fact about how the library was assembled, not about classifiers. Define `configure_optimizers` on `Module` where `Module` is defined (`sec_oo-design`) and delete this subsection, or reduce it to one sentence saying that classifiers inherit minibatch SGD from `Module`. The MXNet tab paragraph that follows explains `collect_params` for a model the reader has not seen yet.

**CLS-4** (moderate, explanation). Anchor: "With $\hat{y}^{(m)}$ the predicted class of example $m$, the class with the", under "Asymmetric Costs".
Notation clash. Throughout `softmax-regression.md`, $\hat{\mathbf{y}}$ is the vector of predicted probabilities and $\hat{y}_j$ one of them. Here $\hat{y}^{(m)}$ is a class index, and in `environment-and-distribution-shift.md` $\hat{y}$ is again a predicted class. The example index also changes from $i$ to $m$ because $i$ and $j$ are needed for classes. Introduce a distinct symbol for the predicted class once (for instance $\hat{c}$, or $\hat{k} = \operatorname{argmax}_j \hat{y}_j$) and use it in all three sections.

**CLS-5** (moderate, thought). Anchor: "agrees: scikit-learn's `confusion_matrix` stores the transpose, with the", under "Asymmetric Costs".
The book defines the confusion matrix with predictions in the rows, then warns that scikit-learn stores the transpose, and a TensorFlow tab adds that TensorFlow does too. The reader's most likely tools use the other convention, and the text spends two warnings on the difference. Either adopt the common orientation (true class in rows) or say in one clause why this one is chosen: with true classes in columns, the normalized matrix $\mathbf{C}$ maps class proportions to prediction frequencies, $\mathbf{C}\mathbf{p} = \boldsymbol{\mu}$, which the last section of the chapter uses. A stated reason turns an annoyance into a design choice.

**CLS-6** (moderate, language). Anchor: "counts the examples of class $j$ predicted as class $i$: rows are predicted", under "Asymmetric Costs".
This paragraph, under a heading about costs, defines the $q$-class confusion matrix, its orientation, the diagonal, the cost weights $\kappa_{ij}$, the column-normalized matrix $\mathbf{C}$, its reading as $P(\hat{y} = i \mid y = j)$, and per-class recall: ten sentences, seven objects, no example. The definition of $\mathbf{C}$ is then repeated almost word for word in `softmax-regression-scratch.md` ("Dividing each column by its sum gives the matrix $\mathbf{C}$ ...") and again in `environment-and-distribution-shift.md`. Give the multiclass confusion matrix its own short subsection with a three-class numerical example, and define $\mathbf{C}$ once, where it is first computed.

**CLS-7** (minor, language). Anchor: "mean, the *F1 score* $2PR/(P + R)$, which is high only when both are.", under "Asymmetric Costs".
$P$ and $R$ are not defined, and F1 is not used again in the section or its exercises. Either write "the harmonic mean of precision and recall, called the F1 score" without the formula, or drop it.

**CLS-8** (major, explanation). Anchor: "A check of the confidence covers only that one probability. The threshold", under "Calibration".
The first paragraph of "Calibration" is the hardest in the section. It moves from the threshold rule, to a group of patients with $p = 0.02$ of whom 0.2% are sick, to the definition of calibration quoted from `softmax-regression.md`, to the definition of confidence, to the distinction between calibration of the top probability and calibration of $p(\textrm{sick})$, and ends "we check it the same way, binning the examples by $p(\textrm{sick})$". But the way of checking (binning, the reliability diagram) is described only in the next paragraph. The reader meets a refinement of a procedure before the procedure. Reorder:

> The rule :eqref:`eq_cost-threshold` acts on the probability $p$ itself, so its decisions are only as good as the probabilities. A classifier is *calibrated* if its probabilities match frequencies: of all patients to whom it assigns $p = 0.02$, 2% are sick. If only 0.2% of them are, the rule flags all of them, although none should be flagged.

Then the binning paragraph, stated for $p(\textrm{sick})$ in the binary case. Only then note that for many classes one usually checks the probability of the predicted class, the *confidence*.

**CLS-9** (moderate, pedagogy). Anchor: "such a check for a hypothetical classifier on 10,000 examples:", under "Calibration".
The reliability diagram is described ("plots the accuracy of each bin against its mean confidence") but not drawn, and its table is invented. Two sections later the chapter computes the real table and draws the real diagram for softmax regression, which turns out to be slightly underconfident, the opposite of the invented classifier. The invented table teaches the arithmetic of the expected calibration error and nothing else. Move the definition of the expected calibration error to `softmax-regression-concise.md`, where the formula already appears, and delete the table here. The temperature-scaling paragraph that follows is the second of three explanations of temperature scaling in the chapter and can go with it.

**CLS-10** (minor, pedagogy). Anchor: "Recall separates the two classifiers: 0 for the one that never flags", under "Class Imbalance".
The `report` cell prints three ratios that the prose has already set up from a table the reader can read. It is the only computed cell in the section and it computes $900/5850$. It does no harm, but the numbers could simply be stated; if the cell stays, let it earn its keep by sweeping the prevalence from 0.1% to 50% and printing the precision, which shows the dependence on the base rate that the next paragraph asserts.

## Exercises

Six problems of high quality. "Optimal decisions and rejection" extends the threshold rule to an abstain option; "Loss at equal accuracy" has a nice sting in its first part (why the average probability does not determine the loss); "Calibration and usefulness" separates the two notions with a classifier that is calibrated and useless; the ROC and AUC pair introduces the one standard tool the text omits. Changes:

- "Class-weighted loss" refers forward to the weighted empirical risk of the chapter's last section and writes examples as $\mathbf{x}_i, y_i$ with subscripts, where the chapter uses $\mathbf{x}^{(i)}, y^{(i)}$. Make it self-contained and fix the notation.
- There is no easy problem. Add one: given a 3x3 confusion matrix of counts, compute accuracy, per-class recall and precision, and say which class a classifier that maximizes accuracy would sacrifice.

## Still open from October

October's "keep" items (the always-healthy classifier, 0.51 versus 0.99) are kept. The second is now told twice (here and in `softmax-regression.md`), which is once too many.
