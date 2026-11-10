# Implementation

Source: `chapter_multilayer-perceptrons/mlp-implementation.md`. About 2,500 words of prose, of which about 540 sit in 15 per-framework tab notes; 760 words of exercises; one generated figure; two stored plots.

## Verdict

A competent implementation section with two things that are better than competent (the checks before training, and the explanation of the slow start) and one structural failure: it opens by asking whether the hidden layer fixes the specific errors softmax regression made, and never answers. The reader gets one aggregate number, 84% against 87%, four times. Around that, the prose is fragmented by framework plumbing: more than a fifth of the body is tab notes, and a full subsection is spent on the derivative of the ReLU at zero, a point the previous section had just called one that "rarely matters". Most important fix: answer the opening question in the text with a per-class comparison, and pay for it by cutting plumbing.

## Scores

| Clarity of explanation | Clarity of thought | Pedagogy | Language | Ease and enjoyment |
|---|---|---|---|---|
| 4 | 3 | 3 | 3 | 3 |

## What works

- **Checks before Training.** Three predictions stated before the cell runs (shapes, loss near $\log 10 \approx 2.30$, half the units active), each with its reason, then the output (`loss 2.305, fraction of active units 0.49`), then what a failure of each would mean. This is the habit of a working practitioner, taught by example, and the best passage in the section. It should become a pattern for other implementation sections in the book.
- **The slow start explained.** "every gradient that reaches $\mathbf{W}^{(1)}$ is multiplied by entries of $\mathbf{W}^{(2)}$, whose standard deviation is 0.01 as well, so at first the hidden layer learns slowly." The plot shows the MLP trailing softmax regression after the first epoch, and the text says why. The reader meets the initialization problem as an observation before it becomes a topic.
- "The hidden layer removes about a fifth of the errors of softmax regression." A better way to read 84% against 87% than the three-point difference.
- **The closing paragraph's three open questions** (who computed the gradients, why 0.01 slows training, why 3.4 parameters per image do not memorize) hand off to the next three sections one for one. This is the chapter's spine made visible.
- Controlled comparison: same data, loss, optimizer, learning rate, epochs and initialization scale for all three models, and the text says so ("so any change in accuracy comes from the hidden layer").
- The table of parameters, multiply-adds and accuracy is the right size.

## Structural findings

- **The opening question is dropped.** See IMPL-1. The section has the trained models in memory and the confusion-matrix code exists in `sec_softmax_scratch`. Exercise 2 asks the reader to do exactly what the text promised.
- **The experiment handicaps itself and then reports the handicap as a property of the model.** Weights of standard deviation 0.01 are kept "as for softmax regression", which makes the MLP start slowly, which is why the run is tripled to 30 epochs, at the end of which validation accuracy "keeps rising". The text is honest about each step, and the slow start is a useful lead-in to the initialization section. But the cumulative message of the chapter's first experiment with a deep model is: 26 times the parameters, three times the epochs, three points. A reader may fairly ask whether hidden layers are worth it. Options: (a) keep the design but say plainly, where the 30 epochs are introduced, that the slow start is caused by the initialization we chose for comparability and that `sec_numerical_stability` removes it; (b) add one line to the comparison showing the same MLP with the scale that section will derive, so the reader sees the hidden layer at its normal strength. I would do (a) at least.
- **Plumbing outweighs teaching in the first half.** "The Parameters" has one paragraph on the model, one on initialization, then a paragraph on parameter registration and four tab notes repeating it per framework. "The Forward Pass" opens with the ReLU-at-zero digression and four more tab notes before the reader sees the forward pass. By the time the two-line `forward` appears, the reader has read about `configure_optimizers`, Keras initializer objects, `nnx.Rngs` and the gradient convention of `maximum` in four frameworks.
- **Scratch then concise, each trained.** Standard for the book and justified here because the comparison plot shows the two curves coincide. Fine.
- A lone `import time` cell sits apart from the imports cell. Minor, but the reader sees a one-line cell with no prose.

## Passage-level findings

- **IMPL-1** (major, thought and pedagogy)
  - Anchor (section opening): "compute. Does an MLP with one hidden layer make fewer of these errors, and"
  - The opening recalls that softmax regression confused "garments of similar outline, such as shirts, pullovers, and coats", argues that telling them apart "depends on how pixels combine, which is what a hidden layer can compute", and asks whether the MLP makes fewer of these errors. The body never looks at errors by class. The answer given is overall accuracy. The question that would show the reader *what* the hidden layer learned is delegated to exercise 2.
  - Fix: after the accuracy comparison, add one short cell that prints or plots per-class recall for softmax regression and the MLP (the diagonal of the confusion matrix of `sec_softmax_scratch`), and three sentences reading it: which classes gained, which confusions remain. If the gain turns out to be spread evenly, say so; that is also an answer. Then reduce exercise 2 to its part (c). If this is not done, remove the question from the opening.

- **IMPL-2** (moderate, pedagogy: proportion)
  - Anchor (The Forward Pass): "otherwise. In this form automatic differentiation gives the derivative 0 at"
  - The subsection on the forward pass begins with why `relu` is written as `X * (X > 0)`: to get derivative 0 at exactly $z = 0$. Four tab notes follow on what `maximum` would give instead (1, 1/2, 1, 1/2). The previous section said of this very choice that it "rarely matters because a pre-activation is seldom exactly zero". A first-time reader is led into a corner case of four autograd implementations before seeing the model.
  - Fix: one sentence ("We write the ReLU ourselves; multiplying by the mask `X > 0` also fixes its derivative at 0 to 0, as the built-in function does") and delete the four tab notes, or move them into a [code] exercise ("what derivative does your framework assign to `maximum(x, 0)` at 0?").

- **IMPL-3** (moderate, language: fragmentation)
  - Anchor (The Parameters): "the learning rate `lr`, finds registered parameters by itself, so the model"
  - The paragraph and its four tab notes explain registration, the default optimizer, `configure_optimizers`, `collect_params`, `add_weight`, `nnx.Param` and `rngs`. The tab notes say the same thing four times with different identifiers. The one item with teaching value, that a shared Keras initializer repeats its values, is already a comment in the TensorFlow code.
  - Fix: compress to two sentences in running prose ("We register the four arrays as parameters, as in `sec_softmax_scratch`, so the optimizer finds them.") and let the code and its comments carry the framework detail. Apply the same to the tab notes under "Concise Implementation".

- **IMPL-4** (moderate, thought)
  - Anchor (Training): "instead of 10. The comparison at the end of this section shows why the MLP"
  - "Why the MLP needs the longer run" suggests a property of MLPs. The reason, given 150 lines later, is the initialization scale chosen here.
  - Fix: say it where the choice is made: "With weights this small the hidden layer starts slowly, as we will see, so we train for 30 epochs instead of 10."

- **IMPL-5** (minor, explanation)
  - Anchor (Training): "neighbors, since minibatch SGD at a constant learning rate keeps moving the"
  - The plot shows single-epoch drops of five to eight points (one curve falls from about 0.83 to 0.79, another from 0.865 to 0.81). "Keeps moving the parameters" does not explain drops of that size to a first-time reader, and the same phenomenon is explained again, at more length, in `dropout.md`.
  - Fix: explain once, here: the validation accuracy is measured at whatever point the last minibatch step left the parameters, and with a step size of 0.1 one unlucky minibatch can move the decision boundary for a whole class. Then `dropout.md` can refer back (see DO-3).

- **IMPL-6** (minor, language: repetition)
  - Anchor (closing paragraph): "On Fashion-MNIST, trained for 30 epochs, its hidden layer raises the"
  - "About 84%" and "about 87%" appear four and five times in this file (comparison text, table, closing paragraph, summary, plus the opening of later sections), "26 times" three times, "203,530" four times. The closing paragraph and the Summary say the same thing twenty lines apart.
  - Fix: let the closing paragraph carry only the three open questions and let the Summary carry the numbers. This double ending is a pattern across the chapter (see the chapter file).

- **IMPL-7** (minor, explanation: prose fits one setting)
  - Anchor (Parameters, Computation, and Time): "On a GPU the two times are about the same, although the MLP does 26 times"
  - The stored output shows 2.9 s for both. A reader on a laptop CPU will see a different picture and the text gives no hint what to expect.
  - Fix: add a clause for the CPU case, or phrase the claim so it holds on both ("On a GPU, and for a model this small even on a CPU, most of an epoch goes to ...") after checking it.

## Exercises

Five problems, all worth keeping. "Single hidden unit" is the standout: part (a) is real analysis of what a one-dimensional hidden representation can separate, part (b) is the dying ReLU derived by the reader, part (c) checks it by experiment. "A hidden layer without activation" closes the loop with the collapse argument of `mlp.md`. "Initialization scale and depth" is a good bridge to `sec_numerical_stability`, and asks for a prediction before running, which is good practice.

- Exercise 2 ("Which errors the hidden layer removes") should move into the text (IMPL-1). Keep its part (c), which asks which differences exceed run-to-run variation.
- "Width" part (c) is a sweep, but it has a question (does the best learning rate depend on width), so it passes.

## Still open from October

Nothing specific was raised for this file in October.
