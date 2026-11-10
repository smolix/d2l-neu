# Writing pass: `chapter_multilayer-perceptrons/mlp.md` (section 4.1)

Prose words: 4127 before, 4127 after (checker: "OK: only prose changed";
`lint_source.py` clean). The checker's em-dash warning is the table rule
`|:---:|` of the XOR truth table and was present before the pass.

This is the best section of the chapter, so the pass is deliberately small:
about fifteen sentences changed. The XOR thread, the fold, the tent, the
"20 units against 1,023" comparison, "Limits of the Theorem", the summary,
all figures and captions (including the corrected hinge caption) are
untouched. Cuts (duplicated opening, GELU catalogue, formulaic closing, two
within-file repeats) paid for the three additions the review asked for
(MLP-4, MLP-10, the scope sentence on interactions).

## Before/after pairs

1. Opening (MLP-1).
   - Before: "The softmax regression of :numref:`chap_classification`
     classifies a Fashion-MNIST image by adding up evidence from its pixels,
     with one weight for each pixel and class. With 7,850 parameters it
     reaches about 83% validation accuracy. Its limit lies in how it adds the
     evidence: each input moves a class score by the same amount whatever the
     other inputs are. Yet in an image, whether a bright pixel belongs to a
     sleeve, a strap, or the background depends on the pixels around it. Can
     a model learn to combine its inputs in this way, and what does it need
     beyond a weighted sum?"
   - After: "The softmax regression of :numref:`chap_classification` adds up
     evidence pixel by pixel, so each input moves a class score by a fixed
     amount, independently of all the others. What exactly does this rule
     out, and what must a model have beyond a weighted sum to escape it?"
2. - Before: "The smallest problem of this kind has four points."
   - After: "It is instructive to start with the smallest problem on which a
     weighted sum fails. It has four points."
3. - Before: "Images resist such repairs. Whether a bright pixel is evidence
     for a shirt or for a bag depends on its neighbors, and writing down
     features that capture all such dependencies is impractical."
   - After: "Interactions are the more serious limit, and they occupy the
     rest of this section. Images resist repairs by hand: whether a bright
     pixel is evidence for a shirt or for a bag depends on its neighbors, and
     writing down features that capture all such dependencies is hopeless."
4. - Before: "A line can classify three of the four points correctly: the
     line $x_1 + x_2 = 3/2$, for example, misclassifies only $(0, 0)$."
   - After: "Three out of four is the best a line can do: the line
     $x_1 + x_2 = 3/2$, for example, misclassifies only $(0, 0)$."
5. - Before: "The name comes from the perceptron, an early linear classifier,
     of which an MLP stacks several layers."
   - After: "The name comes from the perceptron, an early classifier that
     thresholds a weighted sum. Each hidden unit resembles one, with $\phi$
     in place of the threshold, and an MLP arranges such units in layers."
6. - Before: "Every weight into unit $j$ is scaled by the same factor
     $\phi'(z_j)$. Where it is close to zero, none of these weights moves,
     whatever the loss."
   - After: "The gradient of every weight into unit $j$ contains the same
     factor $\phi'(z_j)$. Where this factor is close to zero, none of these
     weights moves, however wrong the prediction."
7. - Before: "Each composition with $g$ doubles the number of pieces
     (...), and it takes one more hidden layer of two units, since the output
     map of one layer and the affine map of the next merge into a single
     affine map."
   - After: "Each composition with $g$ doubles the number of pieces (...).
     How many units does it take? Each copy of $g$ is a hidden layer of two
     ReLU units followed by a weighted sum, and the next copy begins with an
     affine map. Two affine maps in a row are one, as we saw for networks
     without activation functions, so each composition adds one hidden layer
     of two units and nothing else."
8. Closing.
   - Before: "We can now build models that represent nonlinear functions:
     hidden layers of affine maps and activation functions, followed by the
     linear classifier of softmax regression. Whether training finds good
     weights, and whether they predict well on new data, are questions for
     experiment. The next section trains an MLP on Fashion-MNIST and compares
     it with softmax regression."
   - After: "We have not trained a single network in this section: every
     weight was chosen by hand. Whether training finds good weights, and
     whether they predict well on new data, are questions for experiment.
     The next section runs the first one, on Fashion-MNIST."

Other small changes: "The proof has a picture." became "The proof can also be
seen geometrically." (MLP-2); "Two hidden ReLU units suffice for XOR." became
"How many hidden units does XOR need? Two ReLU units suffice."; "one that
softmax regression computes" became "the very one that softmax regression
computes"; "The hidden layer adds parameters." became "The hidden layer adds
parameters, and rather a lot of them."; the universal-approximation paragraph
was split so that the theorem is named at the head of its own paragraph
(MLP-9).

## Repetition removed

- Opening no longer repeats the chapter introduction (83%, 7,850, sleeve,
  strap and background, the same question). One clause of connection, then
  this section's question. The shirt-or-bag sentence stays in "Monotonicity
  and Interactions". The 7,850 now appears once in the body, in the parameter
  count, where the argument uses it (this section is the chapter's first use
  of 203,530 and therefore its owner; unchanged).
- "The next section trains an MLP on Fashion-MNIST" was in the opening and
  the closing; now only the closing.
- "Both layers are trained together" was told three times (opening, The
  Model, end of Solving XOR); the third is reduced to "training has to choose
  them instead".
- ReLU described twice ("keeps positive values and replaces negative ones by
  zero" at its definition, "passes positive pre-activations unchanged and
  sets negative ones to zero" in the ReLU subsection); the second is cut.
- "$h$ units give at most $h + 1$ pieces" stated three times in "Depth" and
  before; the third is "by the count of pieces above".
- Closing paragraph no longer recaps the model; the Summary is the only
  recap and is unchanged.

## Review findings

Applied: MLP-1, MLP-2 (prose variant), MLP-3, MLP-4, MLP-5, MLP-6 (see
below), MLP-9, MLP-10, MLP-11 (two of the four body uses of "whatever"
removed; one body use and exercise uses remain). Structural finding
"monotonicity is introduced and not used": one scope sentence added.

Already addressed in the file, skipped: MLP-8 (hinge caption corrected
earlier today; content kept).

Deferred:
- "No learned feature is ever shown" and the forward pointer to it: needs a
  new figure or cell in `mlp-implementation.md` (IMPL-1).
- Move "Activation Functions" after "Expressive Power": would reorder code
  cells.
- MLP-2, figure variant (diagonals and midpoint in `mdl-mlp-xor.svg`): figure
  change.
- MLP-7 (the ten-sigmoid chain also opens the initialization section): this
  is the earliest section, so it owns the fact; kept. The later section
  should reduce it.
- Exercise "One hidden layer suffices", preamble describing a construction
  the section never shows: exercises were not to be touched except for
  errors.
- Notation ($m$, $s$, $K$ reused across the chapter): chapter-wide decision.

## Factual claims added or changed

- Perceptron: now "an early classifier that thresholds a weighted sum. Each
  hidden unit resembles one, with $\phi$ in place of the threshold" (was "an
  early linear classifier, of which an MLP stacks several layers").
- "The transposes appear only because $\mathbf{x}$ and $\mathbf{h}$ are
  columns, while examples are rows of $\mathbf{X}$." (new)
- "Three out of four is the best a line can do" (was "can classify three of
  the four"; the stronger form follows from the proposition).
- "Interactions are the more serious limit" (a judgment, new).
- "none of these weights moves, however wrong the prediction" (was "whatever
  the loss").
- SiLU: now "A close relative, the *SiLU* $z\,\sigma(z)$, replaces $\Phi$ by
  the sigmoid". Removed: the tanh approximation of the GELU and its
  $5 \cdot 10^{-4}$ error, SwiGLU, and the forward reference to
  `sec_transformer-block`. The SiLU clause is kept, against the review's
  advice to cut it, because
  `chapter_deep-reinforcement-learning/model-based.md` cites this section for
  the SiLU.
- "Neither the single input nor the ReLU is essential" and "The general
  proof is harder than our picture, and we skip it." (new, candour about the
  theorem).
- "We have not trained a single network in this section: every weight was
  chosen by hand." (new; true of the XOR network and the tent).

## Slides section versus body

No contradictions. Two differences of content: the slides still open with
"7,850 parameters, about 83% validation accuracy" and "whatever the other
inputs are" (twice), which the body's opening no longer says; nothing in the
slides mentions the tanh approximation, SiLU or SwiGLU, so the cut leaves
nothing dangling there. The JAX code comment "the default is a tanh
approximation" now stands without prose support, which is sufficient.
