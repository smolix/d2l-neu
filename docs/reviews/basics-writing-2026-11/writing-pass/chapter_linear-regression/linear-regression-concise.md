# Writing pass: Concise Implementation (`linear-regression-concise.md`)

Prose words (checker): **3010 before, 2652 after** (-358). Checker: "OK: only
prose changed". `tools/lint_source.py`: clean. No code cell, label, figure,
exercise, or slide was changed.

## Before/after pairs

1. Opening.
   - Before: "Writing them out showed what each part computes, and we could check every line against the formulas of :numref:`sec_linear_regression`. That model had two weights and a bias, and the models of later chapters are far larger."
   - After: "This was instructive, and it was feasible only because the model had two weights and a bias. The models of later chapters are far larger."
2. Opening, last sentences.
   - Before: "One difference needs care: the framework loss is twice ours, which doubles the step that a given learning rate takes."
   - After: "Little here is new, then, and that is the point of the exercise. Only the scale of the loss will need some care."
3. The loss (the one place where the learning rate is now stated).
   - Before: "Our loss and the framework loss have the same minimizer, but the framework loss and its gradient are twice ours. This factor of 2 returns when we choose the learning rate."
   - After: "The two losses have the same minimizer, so what we fit does not change. The steps do. A learning rate has a meaning only together with the scale of its loss (:numref:`subsec_linear-regression-minibatch-sgd`), and the framework loss is twice ours. To repeat the steps that the scratch model took at $\eta = 0.03$, this model must therefore be trained at $\eta = 0.015$."
4. Training, setup.
   - Before: "From the same initial parameters and on the same minibatches, the two models compute the same updates. With 0.03 the concise model would take the steps of the scratch model at 0.06. On this problem training would still converge, but it would no longer repeat the computation of the scratch model."
   - After: "From the same initial parameters and on the same minibatches, the two models would compute the same updates. We do not go to the trouble of arranging this here (the exercises do). The examples, their order, and the initial weights are all drawn afresh, so we should expect the behavior of the scratch run but not its digits."
5. After the plot.
   - Before: "This is the behavior of the scratch model with every loss doubled. In :numref:`sec_linear_scratch` each step of the scratch model at $\eta = 0.03$ multiplied the error of the parameters by $1 - \eta = 0.97$ on average, which compounds to a factor of about 7 per epoch in the loss. The framework gradient is twice as large and the learning rate half as large, so each step multiplies the error by the same factor, $1 - 2\eta = 0.97$."
   - After: "We have seen this plot before. The slope is the one that :numref:`sec_linear_scratch` predicted and measured, as it must be if the steps are the same. Only the height differs, since every loss is now doubled."
6. After the parameter printout.
   - Before: "Errors of this size are expected: even the exact least-squares solution misses each parameter by about $\sigma/\sqrt{n} = 0.01/\sqrt{1000} \approx 3 \times 10^{-4}$, the standard error of an average of $n = 1{,}000$ noisy terms. And as in the scratch model, the bias, which starts 4.2 below $b^*$, is still about $4.2 \times 0.97^{320} \approx 2.5 \times 10^{-4}$ short of it after the 320 steps of ten epochs."
   - After: "These are the errors of the scratch model, and they have the same sources (:numref:`sec_linear_scratch`), chiefly the noise in the 1,000 training labels and what ten epochs leave of the initial error. Neither depends on who wrote the layer."
7. Closing.
   - Before: "We can now assemble a model from framework parts and train it with the loop of the previous sections. The only adjustment was the learning rate, which we scaled with the loss. The rest of the book builds its models the same way, ..."
   - After: "The rest of the book builds its models this way, ... One habit of this chapter cannot be kept, however. So far we have judged a trained model by its distance from parameters that we chose ourselves."
8. Weight shapes (four tab notes merged into shared prose).
   - Before (per tab): "`nn.Linear` stores its weight as a $1 \times d$ row, so the comparison below reshapes it to the shape of $\mathbf{w}^*$."
   - After (shared, once): "A layer stores its weight as a matrix (a $1 \times d$ row in Gluon and PyTorch, a $d \times 1$ column called the `kernel` in Keras and Flax), so the comparison below first reshapes it to the shape of $\mathbf{w}^*$."

## Repetition removed

- **Factor of 2 and the learning rate 0.015 (CON-1).** Was in the opening,
  the loss subsection, Training, after the plot, the closing, and the
  summary. Now: stated once, under "The Loss Function", as a consequence of
  the rule owned by section 2.1 (one clause with a reference; the general
  $cL$, $c\eta$ argument is no longer re-derived here). The opening only says
  that the scale of the loss needs care; the summary keeps one general clause
  with no numbers. After the plot, "every loss is now doubled" remains,
  because it explains the height of the curve and the floor $\sigma^2$, which
  is a different consequence.
- **Convergence rate 0.97 per step, factor 7 per epoch**: derivation removed;
  one reference to the scratch section.
- **Standard error $\sigma/\sqrt{n}$ and the bias remainder
  $4.2 \times 0.97^{320}$ (CON-2)**: removed; one sentence with a reference.
- **Summary (CON-6)**: no longer re-reports parameter errors, estimation
  error, and the noise floor.
- **Data description**: shortened to the generating parameters and sizes.
- **Tab prose**: the $32 \times 32$ broadcasting sentence appeared in three
  tabs and is now stated once in shared prose; "provides the other common
  update rules with the same interface" appeared in three tabs, now once in
  shared prose; the four weight-shape tab notes became one shared sentence.

## Review findings

Applied: CON-1, CON-2 (prose half), CON-5, CON-6; CON-3 prose half (the
sentence about which trainer calls `weight_init` is shortened and moved next
to the method's definition, so the training tab no longer discusses a trainer
three chapters ahead).

Deferred:
- CON-2, second half (train both models from the same start and print the
  largest difference): needs a new cell. The prose now says candidly that the
  section does not arrange this and that the exercises do.
- CON-3, code half (call `weight_init` in the cell instead of two duplicate
  `nn.init` lines): code change.
- CON-4 (draw both validation curves on one axis): new cell.
- CON-1, source fix (remove the $\frac{1}{2}$ mismatch chapter-wide): code
  and a chapter decision.
- Exercise comments (exercise 2 busywork, exercise 4 preamble): exercises
  are out of scope.

## Factual claims added or changed

- "The examples, their order, and the initial weights are all drawn afresh":
  new. Checked against `synthetic-regression-data.md` (the constructor and
  the loader draw from a new seed on each run). The opening's older phrase
  "the same synthetic data" is therefore loose (same generator, new draw);
  left as it was.
- "The slope is the one that :numref:`sec_linear_scratch` predicted and
  measured": replaces the re-derivation; same content.
- JAX loss tab: "raises an error" reworded to "refuses labels and predictions
  that differ in shape, so the error is no longer silent"; same claim.
- PyTorch layer tab: `weight_init` "comes into use with the trainer of
  :numref:`sec_use_gpu`, which calls it after the first forward pass": the
  claim of the old training tab, moved.
- "Each framework offers the other common update rules behind the same
  interface": generalizes the three tab sentences to all four (true of Gluon,
  where the rule is chosen by name).
- No number describing an output was changed. Stored outputs, all four
  frameworks: validation loss 0.95 to 1.10 $\times 10^{-4}$ (within 15% of
  $\sigma^2$); parameter errors 1.1 to 7.5 $\times 10^{-4}$.

## Slides against the body (slides not edited)

No slide contradicts the body. Three now carry more than the body does:
- "Halving η to 0.015 Reproduces the Scratch Steps" derives
  $1 - 2\eta = 0.97$; the body now refers to the scratch section for the rate.
- The loss slides, the learning-rate slide, the recap table, and two check
  slides state the factor of 2 five times between them, the repetition the
  body has just lost.
- "Ten Epochs Recover w* and b*" says "the concise model takes the scratch
  model's steps"; the body now says this would hold from the same start and
  minibatches, which this run does not arrange.

Also untouched: exercise 1 opens by restating that the framework loss is
twice the scratch loss.
