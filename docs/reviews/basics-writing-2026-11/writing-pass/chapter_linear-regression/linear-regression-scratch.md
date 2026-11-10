# Writing pass: Implementation from Scratch (section 2.4)

Source: `chapter_linear-regression/linear-regression-scratch.md`. Prose only; no code
cell, label, figure, exercise, or slide was changed.

Prose words: 4,652 before, 4,613 after (checker: "OK: only prose changed"; lint clean).

The second half ("Checking the Implementation") was left almost as it stood. Most of the
change is in the first half: the reader is now told which lines are the algorithm.

## Before and after

1. Opening.
   - Before: "The previous sections derived linear regression on paper, built classes for models, data, and training, and generated data with a known answer. We now turn the derivation into a program that trains the model. A training program can be wrong without failing."
   - After: "So far linear regression exists on paper, next to a set of classes that have yet to train a model. We now turn the derivation into a program. The trouble with training programs is that they can be wrong without failing."
2. The bug (one sentence added after "learning the wrong weights").
   - After: "Nothing on the screen looks amiss."
3. The Update.
   - Before: "Linear regression could be solved in closed form, with the normal equations of :numref:`sec_linear_regression`. We train it with minibatch stochastic gradient descent instead, because that is how every later model in the book is trained, and here its result can be checked against the known parameters."
   - After: "Strictly speaking, we need no optimizer at all: linear regression can be solved in closed form, with the normal equations of :numref:`sec_linear_regression`. We train it with minibatch stochastic gradient descent nevertheless, because that is how every later model in the book is trained, and because here, for once, its result can be checked against known parameters."
4. One Training Step (added after the list of the four operations).
   - After: "These four operations are the whole algorithm. Whatever else the training loop below contains is there to measure, to report, or to serve a later chapter."
5. Epochs and Averages.
   - Before: "Reading a number from an accelerator makes Python wait until the device has finished all the work queued for it (:numref:`sec_oo-design`), so the steps return values that stay on the device. `Averager` adds up each metric on the device and converts the sums into Python numbers once per epoch, in `mean`. It weights each minibatch by its size, so the shorter last minibatch counts correctly, and it lists the loss first and the other metrics in alphabetical order, the order in which `record` stores them."
   - After: "It weights each minibatch by its size, so the shorter last minibatch counts correctly. It also keeps the sums on the device and converts them into Python numbers only once per epoch, in `mean`, because every such conversion makes Python wait for an accelerator (:numref:`sec_oo-design`). With three parameters on a CPU the second precaution saves nothing, and `Averager` may be read as a slightly elaborate way of taking a mean."
6. Training and Validation Passes (new shared paragraph; replaces per-tab explanations).
   - After: "A word of warning before the code. This is the loop that trains every model in the book, and so it is longer than linear regression needs. The algorithm is the step of the previous subsection: forward pass and loss, backward pass, update, with a clearing of gradients where the framework keeps them. All other lines prepare the minibatch, average its metrics, or are hooks for later chapters. None of them changes the result of training here, and they can safely be ignored until they are used."
7. The Expected Loss.
   - Before: "Before training, we predict what a correct implementation must produce."
   - After: "It is tempting to run the trainer now and see what comes out. But then we could not tell a correct loss curve from a merely plausible one. So before training, we predict what a correct implementation must produce."
8. Recovered Parameters.
   - Before: "Because we generated the data ourselves, we can also compare the learned parameters with the true ones."
   - After: "A loss at the floor is good news, but it is indirect evidence. Because we generated the data ourselves, we can look at the parameters themselves and compare them with the true ones."
9. Closing.
   - Before: "We have written every component of training by hand and checked the result against an answer known in advance. The loss reaches the noise floor at the predicted rate, and the parameters agree with the true ones to the precision that 1,000 noisy examples allow. Much of this code is generic, and the next section replaces ... The check relied on knowing the true parameters, which real data never provide."
   - After: "All of these checks relied on knowing the true parameters, a luxury that real data never provide. How to judge a model by data it has not seen is the subject of :numref:`sec_generalization_basics`."

## Repetition removed

- Opening recap of the three previous sections: one clause.
- Host synchronization (owned by `oo-design.md`, "Epoch Means"): one clause with a reference.
- "Loss first, other metrics alphabetical" (owned by `oo-design.md`, `record`): removed.
- `as_metrics` (owned by `oo-design.md`): one clause with a reference; the reason is no longer retold.
- Standard error: the arithmetic $0.01/\sqrt{1000}$ removed; value and name kept with the reference to `subsec_synthetic-regression-data-recovery`.
- Noise ball: definition ("a small cloud around the minimizer") removed; name and reference kept.
- Learning rate and summed loss: the counterfactual ("with a summed loss it would have to shrink") removed; the reference to section 2.1 stays.
- "Each parameter moves against its own partial derivative, scaled by $\eta$": removed (restated the equation above it).
- "The factor $1 - \eta$ exceeds 1 in magnitude once $\eta > 2$, so learning rates above about 2 diverge": removed after the sweep (said before the sweep).
- `d2l.HyperParameters` sentence, told in three optimizer tabs: stated once in the shared prose.
- "Gradient clipping ... serves later chapters", told in three tabs: the shared warning paragraph carries the point; each tab names its hook in a clause.
- Closing recap paragraph removed; the summary is the only recap, cut from nine sentences to five.
- The factor 1/2 in the loss was already a clause citing section 2.1; unchanged.

## Review findings

Applied (prose):

- SCR-1, in prose only: the reader is told three times, briefly, which lines are the algorithm (opening, "One Training Step", and a warning before `fit_epoch`); each tab then names its own algorithm lines and hooks.
- SCR-4: synchronization story and ordering rule reduced as above.
- SCR-6: "negative in about three runs out of four" became "comes out negative more often than not".
- SCR-7: a parenthesis says that `sigma` is unrelated to the noise level $\sigma$.
- SCR-8: summary shortened.
- Tab prose trimmed (guide section 9): MXNet and TensorFlow loop tabs, JAX loop tab (the paragraph on `jit` recompilation and the reuse of the optax transformation is gone), the three optimizer tabs.

Deferred:

- SCR-1 proper (a bare loop that runs) and SCR-2 (a cell for one step): need new cells.
- SCR-3 ($L_{\mathcal{B}}$ for the minibatch loss): a symbol shared with the slides and other sections; needs one decision for the chapter.
- SCR-5 (a printed table of predicted against measured excess loss): needs a new cell. The paragraph keeps its numbers.
- Structural: moving `prepare_batch`, `Averager`, and `fit_epoch` to the `Trainer` section moves code between files.
- Exercise suggestions (a sentence for Planck's law; the three exercises deferred from `oo-design.md`): exercises not touched.

## Factual claims added or changed

- "a set of classes that have yet to train a model" (true of `oo-design.md` and the synthetic data section as they stand).
- "Nothing on the screen looks amiss" (of the broadcasting bug: no error, and the loss falls).
- `sigma` "has nothing to do with the noise level $\sigma$ of the data. That both are 0.01 is a coincidence."
- The rest of the `SGD` class "gives it the interface of the framework's own optimizers, so that the two can be exchanged in the next section."
- "With three parameters on a CPU the second precaution saves nothing."
- Of the non-algorithm lines in `fit_epoch`: "None of them changes the result of training here" (clipping is off at `gradient_clip_val=0`, no scheduler is configured, `prepare_batch` is the identity).
- PyTorch tab: `model.train()` and `model.eval()` matter "only for layers that we have not met yet".
- "The minibatch gradient equals this gradient only on average, so the numbers that follow are estimates."
- Bias error: "comes out negative more often than not" replaces "in about three runs out of four". In the stored outputs it is negative in PyTorch, JAX, and MXNet and positive in TensorFlow.
- "the limit of 2 is a property of our data, not of gradient descent".
- Summary: autograd reproduces the gradient "up to rounding"; the parameter error is one "that the loss curve cannot show and a comparison of parameters can".

No number that describes an output was changed. Stored outputs of all four frameworks were read and agree with the prose (largest gradient difference 0 or 5e-07 in PyTorch, MXNet, and TensorFlow; my extraction did not find that line in the JAX store; final validation losses 2.1 to 2.5, about 5e-05, about 5e-05, `nan`).

## Slides against the body

No slide contradicts the revised body. Two differences of emphasis:

- The `Averager` slide and the divider "one sync per epoch" present the once-per-epoch read as a main point; the body now calls it a precaution that saves nothing here.
- The slides never say that `fit_epoch` carries hooks for later chapters (the PyTorch slide shows a loop labelled "simplified"); the body now says so explicitly.
