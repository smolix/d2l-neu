# Concise Implementation

Source: `chapter_linear-classification/softmax-regression-concise.md`. About 2,050 words of body prose (with tabs), 630 words of exercises, 8 code cells in the PyTorch tab.

## Verdict

The section has a real idea, which most "concise implementation" sections lack: the framework's loss takes logits because the loss can be computed stably from logits and not from probabilities. The opening states this as a question the reader can hold ("How can the loss be computed so that it stays finite, and keeps its gradient, for every value of the logits?"). The calibration experiment at the end is honest and well read: the model turns out slightly underconfident, and the text says so. But the section is the most repetitive in the chapter. By the time the reader reaches "Overflow and Underflow" they have been told that $\exp$ overflows near 88 four times, have seen the largest logit subtracted twice, and have already derived the loss as log-sum-exp minus the true logit. The section then never runs its own loss on the logits that broke the previous section, so the fix is asserted and not shown. The most important change is to make this section the only home of the numerical argument, shorten it by half, and end it with the demonstration.

## Scores

| Explanation | Thought | Pedagogy | Language | Enjoyment |
|:-:|:-:|:-:|:-:|:-:|
| 4 | 3 | 3 | 4 | 3 |

## What works

- The opening question, and the sentence that follows the scratch failure to its cause: "Yet the loss needs only that one logarithm".
- "Each framework's layer would otherwise draw its weights from a default distribution of its own, and the four tabs would train different models." A sentence that explains a line of code the reader would otherwise find arbitrary.
- The double-softmax warning after the loss code: "A model whose `forward` ended with a softmax would hand it probabilities, and the loss would apply a second softmax to them." This is the most common bug in practice and it is named at the right moment (and turned into a good exercise).
- "The framework's components change how the model is computed, not the model."
- The reading of the reliability table: the lowest bins are set aside for having too few predictions, the diagram is restricted to bins with at least 100, and the conclusion is stated at the precision the data support ("slightly underconfident", "below 0.03"). The stored output agrees with every claim (mean confidence 0.972 and accuracy 0.975 in the top bin, ECE 0.018).
- "This is a property of the trained model, not of the softmax". A limit on the claim, in the manner of Cover and Thomas.

## Structural findings

- **Repetition.** See CON-1. The section opening (two paragraphs) summarizes the argument, "Overflow and Underflow" gives it again in full, and the summary a third time. Within the chapter, shift invariance is derived in `softmax-regression.md`, used in `softmax-regression-scratch.md`, and derived again here with a displayed equation; the log-sum-exp form of the loss is derived in `softmax-regression.md` ("The Gradient") and again here; the bound $\max \le \mathrm{lse} \le \max + \log q$ is stated there with an exercise and proved here.
- **The fix is not demonstrated.** See CON-2.
- **A digression in the middle of the argument.** The bound on log-sum-exp and the plot of $\log(1 + e^x)$ against $\max(x, 0)$ say that log-sum-exp is a smooth maximum. That is true and was said in `softmax-regression.md`; it has nothing to do with numerical stability. See CON-3.
- **Calibration is the third stop of a thread.** `softmax-regression.md` defines calibration and temperature scaling and predicts overconfidence; `classification.md` defines them again with an invented table; this section defines calibration a third time and measures it. Only this one has data. It should be the one that owns the topic (or the relocated "Beyond Accuracy" should, see the chapter file).

## Passage-level findings

**CON-1** (major, thought). Anchor: "Single-precision numbers span roughly $10^{-38}$ to $10^{38}$, so", under "Overflow and Underflow".
Fifth statement of the overflow threshold in the chapter and the third in this file (opening paragraph two, here, summary). The subsection also re-derives shift invariance with a two-line display that `softmax-regression.md` gave in a sentence. Proposed shape for the whole of "A Stable Cross-Entropy Loss", about 250 words in place of 600:

> The scratch implementation fails in two ways. For logits $(1000, 0, 0)$ the exponential overflows and the softmax returns `NaN`. For logits $(0, -200, 0)$ and true class 2 the probability underflows to 0 and its logarithm is $-\infty$. Both failures come from forming the probability, which the loss does not need. It needs $\log \hat{y}_y = o_y - \log\sum_k \exp(o_k)$ (:numref:`subsec_softmax_and_derivatives`). Subtracting the largest logit $\bar{o}$ inside the sum, and adding it back outside, changes nothing and makes every exponent at most 0: [display of the loss]. The sum now lies between 1 and $q$, so every term is an ordinary number.

Then the framework loss, then CON-2. Keep the detail about subnormal numbers ("reaches exactly 0 near $-104$") for the number-formats exercise.

**CON-2** (moderate, pedagogy). Anchor: "form the probabilities, so neither the exponential of a large number nor the", under "The Framework Loss".
The claim that the framework loss is safe is not tested in the text; "Fused loss on extreme logits" leaves it to the reader. The previous section showed the failure on $(1000, 0, 0)$ in a cell. Show the repair in a cell here: `model.loss` on that row with label 0 and with label 1 returns 0 and 1000. Two numbers the reader can predict from the formula, and the story that began in the scratch section closes.

**CON-3** (moderate, pedagogy). Anchor: "so log-sum-exp is a smooth version of the maximum that exceeds it by at most", under "The Log-Sum-Exp Form".
The bound and the plot interrupt the path from the stable formula to the framework loss. The reader of `softmax-regression.md` already has the statement and an exercise proving it. Remove the bound, the plot cell and the paragraph after it from this section. If the picture is wanted, it fits in `softmax-regression.md` next to `fig_mdl-clf-margin-losses`, where $\log(1 + e^{-m})$ is already drawn.

**CON-4** (moderate, explanation). Anchor: "exponential of minus this loss is the confidence.", under "Calibration".
The confidence is computed as `exp(-model.loss(O, preds, averaged=False))`, the loss of each example against its own prediction. The identity is correct and the trick avoids a softmax, but a reader must stop and work out why the loss of a prediction against itself is what is wanted. The purpose of the cell is a reliability table, not a demonstration of the identity. Use the maximum of the softmax of the logits (the text has just said a softmax is computed "when we want the probabilities themselves") and move the identity to an exercise.

**CON-5** (moderate, thought). Anchor: "slightly underconfident, and the expected calibration error stays below", under "Calibration".
The finding is that this model needs no repair, and if anything a temperature below 1. The chapter has by now explained temperature scaling three times as a cure for overconfidence "usually with $T > 1$" and never applies it. Either (a) add the three-line fit of $T$ on half of the validation set and report that $T$ comes out slightly below 1 and the ECE barely moves, which closes the thread with an observation, or (b) cut the earlier build-up (see SMR-8, CLS-9) so that the reader has not been led to expect overconfidence. The exercise "Temperature scaling" then remains as is.

**CON-6** (moderate, pedagogy). Anchor: "pass has fixed the shapes: it calls the model's method `weight_init` on", PyTorch tab under "The Model".
In the PyTorch tab the "concise" model needs a lazy layer, a first forward pass to fix shapes, a `weight_init` hook that the `Trainer` calls on every submodule, and a paragraph to explain all three. The scratch model needed two lines. `nn.Linear(784, num_outputs)` with an explicit initialization in the constructor would be shorter and need no explanation; lazy layers can be introduced in the chapter on computation where they are the subject. As written, framework plumbing is the first thing the section teaches.

**CON-7** (minor, explanation). Anchor: "Subtracting the largest logit, $\bar{o} = \max_k o_k$, from all logits", under "Overflow and Underflow".
$\bar{o}$ denotes the largest logit here and the mean logit in the "Temperature" exercise of `softmax-regression.md`. A bar usually suggests a mean. Use $o_{\max}$ here.

**CON-8** (minor, language). Anchor: "this combined expression. With the layer and the loss, softmax regression", section opening.
The second opening paragraph (110 words) is a compressed version of the subsection it precedes, formula included. An opening should pose the problem; the formula can wait. Cut the paragraph after "which the previous section could only hide behind a floor that also stops the gradient."

## Exercises

Five problems. "Double softmax" is excellent: the reader computes the smallest loss a double softmax can reach and then sees training stall at that value. "Stable softplus" extends the section's idea to one variable and asks the right question about gradients. "Temperature scaling" completes the calibration experiment.

- "Number formats" (five formats, a table of exponent limits) is arithmetic on bit widths and names a vendor format (TF32) that the book does not otherwise discuss. Reduce to FP16, BFLOAT16 and FP32 and ask one question with a consequence: at which logit does a half-precision softmax overflow, and what does that imply for mixed-precision training.
- "Fused loss on extreme logits", part 1 becomes text if CON-2 is adopted; keep parts 2 and 3.

## Still open from October

Not applicable; October did not comment on this section.
