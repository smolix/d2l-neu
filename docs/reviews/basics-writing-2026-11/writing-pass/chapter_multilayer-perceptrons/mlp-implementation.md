# Writing pass: `chapter_multilayer-perceptrons/mlp-implementation.md` (section 4.2)

Prose words (checker count, whole file): 3,159 before, 2,846 after (−313).
Checker: "OK: only prose changed". `tools/lint_source.py`: clean.
Tab notes outside the Discussions links: 15 before, 7 after, each one or two
sentences. "Checks before Training" is untouched.

## Before/after pairs

1. Opening question (IMPL-1).
   - Before: "Does an MLP with one hidden layer make fewer of these errors, and by how much?"
   - After: "How many of the errors does one hidden layer remove, and what does it cost? [...] One limitation should be stated at the outset. This section counts errors but does not sort them: which garments the hidden layer learns to tell apart is left to the exercises."
2. Opening recap.
   - Before: "In :numref:`sec_mlp` a hidden layer let a network combine its inputs, and for the four points of XOR we chose its weights by hand. We now let training choose the weights on real images, and ask whether the hidden layer fixes the errors that softmax regression made."
   - After: "In :numref:`sec_mlp` we chose the weights of a hidden layer by hand, which is feasible for the four points of XOR and for little else. Let us now leave the choice to training, on real images."
3. Initialization scale (structural finding, option (a)).
   - Before: (nothing)
   - After: "The scale 0.01 itself is merely inherited from softmax regression, which keeps the comparison between the two models clean. It is not a good scale for a network with a hidden layer, and the first epochs of training will show it."
4. ReLU at zero (IMPL-2; four tab notes deleted).
   - Before: "In this form automatic differentiation gives the derivative 0 at $z = 0$, the value that the built-in ReLU uses (:numref:`subsec_activation-functions`)." followed by four tab notes on `maximum`.
   - After: "(Written as a product with this mask, it also gets the derivative 0 at $z = 0$ from automatic differentiation, as the built-in ReLU does. A maximum with 0 may get 1 or $1/2$, depending on the framework, and :numref:`subsec_activation-functions` explained why we need not care.)"
5. Thirty epochs (IMPL-4).
   - Before: "The comparison at the end of this section shows why the MLP needs the longer run."
   - After: "The longer run is a consequence of our initialization: with weights this small the hidden layer starts slowly, as the comparison at the end of this section shows."
6. Noisy validation curve (IMPL-5).
   - Before: "From one epoch to the next the accuracy moves by about a point, and a single epoch can fall several points below its neighbors, since minibatch SGD at a constant learning rate keeps moving the parameters."
   - After: "The climb is far from smooth. From one epoch to the next the accuracy moves by about a point, and a single epoch can fall several points below its neighbors. Nothing is broken: minibatch SGD at a constant learning rate never settles, and we measure the validation accuracy wherever the last step of the epoch happened to leave the parameters."
7. The result.
   - Before: "softmax regression levels off at about 84%, while the MLP climbs to about 87%. The hidden layer removes about a fifth of the errors of softmax regression."
   - After: "softmax regression levels off at about 84%, while the MLP climbs to about 87%. Three points sound modest. Read as errors, they are more: the hidden layer removes about a fifth of the errors of softmax regression. Which fifth? It is tempting to assume that these are the shirts taken for pullovers and coats, but a total accuracy cannot tell us, and we have not looked. An exercise below compares the two models class by class."
8. Time.
   - Before: "We compare the measured times per epoch: [...] On a GPU the two times are about the same, although the MLP does 26 times the arithmetic."
   - After: "Does it multiply the training time as well? [...] Surprisingly, on a GPU it does not: the two times are about the same."

Also new: "Nobody who builds networks for a living registers four arrays by hand."
(opening of Concise Implementation), "That is the whole model." (after the
forward pass), "Had we stopped there, we could hardly have told the two
models apart." (ten-epoch comparison), "Our initialization is to blame."

## Repetition removed

- Result (84% against 87%): was in the comparison text, the table, the closing
  paragraph, and the summary. The closing paragraph now carries only the three
  open questions (IMPL-6); the result stands in the comparison text, the
  table, and the summary.
- 203,530: prose mentions 4 to 1 (plus the table). The derivation
  $785 \cdot 256 + 257 \cdot 10$ is owned by `mlp.md` and is now cited, not
  repeated; 203,264 left the opening (it is derived at the table); the closing
  says "some two hundred thousand parameters".
- "26 times": 4 to 3 in the body (opening, at the table, summary).
- Standard deviation 0.01: prose mentions 7 to 4 (the choice, the slow-start
  explanation, the open question, and the judgment sentence).
- Loss on logits (owned by `sec_softmax_concise`): three clauses reduced to
  one with a reference.
- "Automatic differentiation provides the gradients": was in the roadmap, the
  forward-pass paragraph, and the closing; removed from the forward pass.
- Tab notes: registration notes (4) reduced to one MXNet sentence (`data()`)
  and one JAX sentence (`rngs`); `maximum` notes (4) deleted; second MXNet
  `data()` note deleted; concise notes cut to one or two sentences; second
  JAX note folded into the first.
- Concise section: "Compared with `SoftmaxRegression`, the model adds the
  hidden layer and the ReLU" and the restated initialization values removed.

## Review findings

Applied: IMPL-1 (prose route: question changed to one the outputs answer,
limitation stated in the opening and again at the result, pointer to the
exercise), IMPL-2, IMPL-3, IMPL-4, IMPL-5, IMPL-6, and option (a) of the
structural finding on the self-imposed handicap.

Deferred:
- IMPL-1, full fix (per-class recall cell, shrinking exercise 2): needs a new
  cell and outputs.
- IMPL-7 (CPU timing): no CPU measurement to check against; the text still
  says "on a GPU". All four stored outputs are GPU runs.
- Lone `import time` cell: a code change.
- Option (b) of the handicap finding (MLP with He-scale initialization in the
  comparison): a new experiment.
- Exercise 2 has no `:exlabel:`, so the text says "an exercise below" rather
  than referring to it by number. Adding a label is a one-line follow-up.

## Factual claims added or changed

- "It is not a good scale for a network with a hidden layer": a judgment,
  supported by the slow start shown in this section and by
  `sec_numerical_stability`.
- "A maximum with 0 may get 1 or $1/2$, depending on the framework":
  condenses the deleted tab notes (MXNet 1, TensorFlow 1, PyTorch 1/2,
  JAX 1/2); not re-verified by execution.
- "we measure the validation accuracy wherever the last step of the epoch
  happened to leave the parameters" and "never settles": replaces "keeps
  moving the parameters". The review's stronger suggestion (one unlucky
  minibatch moves the boundary of a whole class) was not adopted because it
  is not shown.
- "Had we stopped there, we could hardly have told the two models apart":
  follows from "ahead by about a point on average, within its variation from
  epoch to epoch", which was already in the text.
- "Surprisingly, on a GPU it does not": stored times per epoch, MLP against
  softmax regression: PyTorch 2.9/2.9 s, TensorFlow 0.8/0.8, JAX 0.6/0.5,
  MXNet 2.7/2.6.
- PyTorch tab: "the lazy layers have no weights before the first forward
  pass, so the `Trainer` initializes them afterwards through the method
  `weight_init`" (same content as before, shorter).
- JAX tab: "`forward` flattens the images before it calls the sequence"
  (from the code).
- Numbers describing outputs (83%, 84%, 87%, a fifth, 2.30, one half) are
  unchanged and agree with all four stored outputs.

## Slides that now differ from the body

No contradictions. Differences of emphasis only:
- The slides give the count as a derivation ($785 \cdot 256 + 257 \cdot 10$)
  and repeat 203,530 several times; the body now cites it from `sec_mlp`.
- The slides do not say that 0.01 is a poor scale chosen for comparability,
  nor that the section leaves open which errors were removed.
- The opening slide already asks the answerable question ("more accurate,
  and by how much?"), so it agrees with the new opening.
