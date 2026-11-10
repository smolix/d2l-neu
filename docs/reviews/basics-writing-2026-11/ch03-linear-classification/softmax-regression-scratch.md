# Softmax Regression from Scratch

Source: `chapter_linear-classification/softmax-regression-scratch.md`. About 2,500 words of body prose (with tabs), 650 words of exercises, 16 code cells in the PyTorch tab.

## Verdict

A good implementation section, and the first place in the chapter where something happens. Its opening is a model of how to start from the reader: it recalls the regression chapter, asks "What changes when the model predicts a class instead of a number?", and answers "three places". Each code cell is preceded by what it will compute and followed by what it showed, with the hand-computed numbers of the theory section reappearing in the output (0.2076, 0.6893, 0.1031). The error analysis at the end (misclassified images, then a confusion matrix built from one-hot products) is the best use of code in the chapter. Two things hold it below the bar. It never shows what the model learned: the weights, reshaped to images, are ten blurry garments and would be the most memorable picture of the chapter, but they are left to an exercise. And about a fifth of the prose goes to a numerical problem (overflow, underflow, a floor in the loss) that cannot occur in this model and is treated again, at greater length, in the next section.

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 4 | 4 | 4 | 4 |

## What works

- The opening: what carries over is listed by name (`Trainer`, the SGD of `Module`, `FashionMNIST`, `Classifier`), so the reader knows that only three pieces are new.
- The check of `softmax` against the hand computation of `softmax-regression.md`, plus the row of equal logits giving 1/3 each.
- The initial-loss sanity check in "The Parameters": small weights give near-uniform predictions "and a loss close to $\log 10 \approx 2.30$". A reader can use this to debug their own code.
- "Flattening discards which pixels are neighbors. The model cannot use that information; the convolutional networks of :numref:`chap_cnn` do." One limit, named in two sentences, with a pointer that is not a prerequisite.
- The confusion matrix as $\mathbf{P}^\top \mathbf{Y}$: a real idea (count pairs with one matrix product), derived in one displayed line and verified by the count 10,000.
- The reading of the matrix: "A shirt resembles each of the other three in some of these details, so it collects errors from all of them." The stored heatmap supports the claims (shirts near 0.55 on the diagonal, the tops trading errors, shoes forming a second block).
- The closing paragraph, which states both limits (a fragile implementation, linear boundaries) and assigns each to a later section.

## Structural findings

- **No picture of the learned model.** Softmax regression on images is a set of ten templates, and looking at them explains at once why it works for trousers and fails for shirts. Exercise 5 asks the reader to "Reshape each column of the weight matrix $\mathbf{W}$ into a $28 \times 28$ image". This belongs in the main text, after the training run, in three lines of code and one paragraph. It is the natural answer to the section's own closing question of why errors lie "among garments of similar shape".
- **The numerical-stability thread is spread over two sections.** Here: a preview in the opening, an "Overflow" subsection with a demonstration, a floor of $10^{-12}$ in the loss with a paragraph on why it is a stopgap, a mention in the closing and in the summary. Next section: the same facts again, then the fix. Since "its inputs are pixel values between 0 and 1 and its weights stay small", none of it affects this model. See SCR-1 and SCR-2.
- **Training accuracy is never reported.** The section compares training and validation loss but only validation accuracy. The generalization section two sections later reasons about the gap between training error and population error, and has no number from the chapter's own model to refer to. See SCR-4.

## Passage-level findings

**SCR-1** (moderate, thought). Anchor: "single precision, $\exp(o)$ overflows to infinity once $o$ exceeds about 88,", section opening.
The opening's second paragraph tells the result of the "Overflow" subsection in advance (the threshold, the example $(1000, 0, 0)$, `NaN`), and the subsection then tells it again with the derivation of 88.7. The same threshold also appears in `softmax-regression.md` and three times in `softmax-regression-concise.md`. Reduce the opening to one sentence: "Writing the softmax directly from its formula also exposes a numerical trap, which we demonstrate here and remove in the next section."

**SCR-2** (moderate, pedagogy). Anchor: "of such an example is at most $-\log 10^{-12} \approx 27.6$. The floor is a", under "The Cross-Entropy Loss".
The loss is given a floor, the floor is explained, declared "a stopgap", shown to kill the gradient, and promised to be removed. The floor never triggers in this section, and `cross_entropy` with the floor is saved into the `d2l` library. The reader is taught a workaround in order to be told it is wrong. Simpler: write the loss as the plain mean of $-\log \hat{y}_y$, note in one sentence that $\log 0$ would be infinite for a probability that underflows, and let the next section own both problems and the fix. If the floor must stay for the library's sake, keep it in the code with its comment and cut the prose to that one sentence.

**SCR-3** (minor, pedagogy). Anchor: "Summing over axis 0 gives a row of shape $(1, 3)$, summing over axis 1 a", under "Normalizing Each Row".
The cell re-teaches `keepdims` and axis sums from the preliminaries on a 2x3 matrix before the softmax is written. The three-step description in the paragraph above already says what is needed, and the `softmax` function is two lines. Cut the demonstration cell and this sentence; keep the reference to `subsec_broadcasting`.

**SCR-4** (moderate, thought). Anchor: "parameters fitted to 60,000 images can memorize little of its training", under "The Training Run".
Two problems. First, the argument is by parameter count ("7,850 parameters fitted to 60,000 images"), which the generalization section later warns is not a rule (its $\sin(\omega x)$ example) and which chapter 4 will contradict outright. Second, the evidence offered is two losses; the training accuracy is not shown. Report the training accuracy next to the validation accuracy (one call to `trainer.evaluate` on the training loader, or a `train_acc` curve) and say what is observed: the two differ by a point or two. Then say, without the parameter count, that a linear model cannot fit arbitrary labels of 60,000 images, and that `chap_classification_generalization` makes this precise.

**SCR-5** (minor, correctness). Anchor: "losses end close together, near 0.45 and 0.49. A model with 7,850", under "The Training Run".
The stored PyTorch run ends with a validation loss of 0.475 (`trainer.evaluate` prints `'loss': 0.47530625`), not 0.49. Write "near 0.45 and 0.48", or better "within a few hundredths of each other".

**SCR-6** (moderate, pedagogy). Anchor: "are numbered from 0 to 9 in the order T-shirt, trouser, pullover, dress,", under "The Confusion Matrix".
The heatmap's axes carry the numbers 0 to 9, and the prose then has to translate ("columns 0, 2, 4, and 6", "columns 5, 7, and 9"). The reader moves between the picture, the list of class names and the paragraph. Put the class names on the ticks (or reorder the classes so that tops, shoes and the rest are adjacent, which makes the two blocks of confusion visible as blocks). Printing the three or four largest off-diagonal entries with their names would let the prose quote real numbers.

**SCR-7** (minor, language). Anchor: "its sum gives the matrix $\mathbf{C}$ with $C_{ij} = N_{ij} / \sum_{i'}", under "The Confusion Matrix".
The definition of $\mathbf{C}$ repeats `classification.md` almost word for word, including "the fraction of the examples of class $j$ that are predicted as class $i$". Define it in one place (here, where it is computed) and refer to it from the other.

**SCR-8** (minor, pedagogy). Anchor: "Between 30 and 45 of the 256 images are misclassified, in line with an", under "Misclassified Images".
The eight misclassified images are shown with "true above predicted" titles, and the prose generalizes ("Most mistakes confuse garments of similar outline"). It would be stronger to point at one image: the stored output shows, for example, a coat taken for a pullover and a sandal taken for a sneaker. Since the particular images are stable (the validation loader is not shuffled and the confusions are robust), one concrete sentence is safe.

## Exercises

Six problems, well matched to the section. "Underflow" makes the reader find the zero gradient of the floor by automatic differentiation, which is the right way to learn it. "Large vocabularies" connects the model to language modelling with numbers. "Hyperparameter sweep" asks for an explanation in terms of the number of updates, so it is a sweep with a question.

- **Promote** the second part of "Precision and templates" (weights as images) to the main text and keep the first part as the exercise.
- "ROC of the trained model" depends on two exercises of an earlier section. State the definition of the ROC curve in one line so that it stands alone.
- "Cross-entropy from one-hot labels", part 1 ("confirm that training is unchanged") is a drill; parts 2 and 3 carry the problem.

## Still open from October

Not applicable; October did not comment on this section.
