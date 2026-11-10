# Writing pass: `chapter_multilayer-perceptrons/backprop.md` (section 4.3)

Prose only. No code cell, label, figure include, heading, exercise, or slide
was changed. Checker: "OK: only prose changed"; `tools/lint_source.py` clean.

## Length

Prose words (checker count, includes exercises): **3567 before, 3854 after
(+287)**. The section got longer, against the general rule of the brief. The
cause is the assignment for this file: three things the review found missing
were added (the answer to the opening question, the hidden-layer gradient
read as back-propagated error, the recursion over $K$ layers, together about
330 words and one unlabeled display), while the scalar-graph figure and the
three setups had to stay. About 250 words of existing prose were cut or
merged to make room. If the length must come down, the cheapest cuts are the
$K$-layer paragraph (about 115 words; exercise 3(a) asks for it) and the
last sentence of "Stored Values" on recomputation (33 words).

## Before / after pairs

1. Before: "In :numref:`sec_mlp-implementation` we trained an MLP by calling
   `backward` and using the gradients it returned."
   After: "In :numref:`sec_mlp-implementation` we trained an MLP without
   writing a line of calculus: the framework supplied the gradients."
2. Before: "The framework records the operations of the forward pass as a
   computational graph and then applies the chain rule backward through the
   graph, one operation at a time. The backward pass is thus built from local
   rules."
   After: ":numref:`sec_autograd` gave the answer in general terms: the
   framework records the forward pass as a computational graph and applies
   the chain rule backward through it, one local rule per operation, each
   reading values that the forward pass saved. There the backward pass
   through a network remained a black box, which this section opens. Little
   of the machinery is new. What is new is the setting: the parameters are
   matrices, a nonlinearity sits between the layers, and the gradient has to
   pass *through* one layer to reach the one below."
3. Before: "Before the backward pass can run, two questions must be answered
   for every node of the graph. Which values from the forward pass does its
   local derivative need? And when is its gradient ready to be passed on? We
   answer both for the smallest graph that contains an addition and a
   multiplication,"
   After: "The graph that we traced in :numref:`sec_autograd` had vectors at
   its nodes. As a reminder of how a backward pass works, here are the two
   most common local rules in their barest form, on three numbers:" (the two
   questions now follow the figure: "Two things in this small computation
   will matter for a network.")
4. Before: "Consider first a matrix product $\mathbf{A} = \mathbf{X}\mathbf{W}$"
   After: "Let us start with the matrix product, the only one of the three
   rules that takes some care. In :numref:`sec_autograd` a matrix–vector
   product passed the gradient back to its vector through $\mathbf{X}^\top$.
   Now both factors are matrices, and we need the gradient with respect to
   each. Consider $\mathbf{A} = \mathbf{X}\mathbf{W}$"
5. Before: (nothing; the five-line display was read for order and stored
   values only.)
   After: "The display also has a meaning, and its third line is where the
   method gets its name. [...] A hidden unit has no label of its own. What it
   learns from is this message: how wrong the outputs that it feeds are, and
   in which direction."
6. Before: "A unit that is inactive for every example of the training set
   never receives a gradient and stays inactive; this is how a ReLU unit
   dies."
   After: "A unit that is inactive for every training example receives no
   gradient from any of them: this is the dead unit of :numref:`sec_mlp`,
   seen in its gradient."
7. Before: "The worked example used a single example, the squared loss, and
   no weight decay."
   After: "The worked example was chosen to be easy: a single example, the
   squared loss, no weight decay."
8. Before: "We can now compute the gradient of an MLP by hand, check it
   against automatic differentiation, and say what it costs in memory and
   time."
   After: "This settles the question we started with. Finite differences
   would need one forward pass per parameter, some two hundred thousand per
   training step. The backward pass computes $\partial J / \partial
   \mathbf{O}$ and $\partial J / \partial \mathbf{Z}$ once, and the
   derivative of the loss with respect to every parameter is read off these
   two matrices by a single product or sum: the work that the derivatives
   have in common is done once."

## Repetition removed

- **What `sec_autograd` already taught** (graph, local rules, saved values)
  is no longer re-introduced in the opening: one sentence with the reference,
  then what is new here.
- **The scalar graph** is presented as a reminder; its prose is about 100
  words shorter, and the narration of each local derivative is folded into
  one sentence before the display.
- **"Training needs more memory than prediction"**: no longer posed as a new
  question in the opening and re-derived in "Stored Values". Now one clause
  citing `sec_autograd`, followed by the count, which is what this section
  adds. The sentence "This is why training a large network runs out of memory
  long before prediction does" is gone.
- **Forward pass, backward pass, update alternate** (BP-9 paragraph): deleted.
- **Sum over paths**: one clause citing the detached product $z = ux$ of
  `sec_autograd`; the generic restatement "In general, a value that feeds
  several nodes collects the sum" removed from the body (it remains in the
  Summary). The `=` versus `+=` sentence is kept.
- **Dead ReLU unit**: owned by `mlp.md`; reduced to a clause with reference.
- **Matrix-product rule read out in words** ("multiplies the transposed
  input by the incoming gradient, and ..."): removed; the shape check and the
  outer-product reading remain.
- **Closing paragraph**: the formulaic recap ("We can now compute ...") is
  replaced by the answer to the opening question and one sentence of
  hand-off. The Summary is the only recap.
- **203,530** now appears twice in the body (opening, memory comparison)
  instead of twice plus the recap.

## Review findings

Applied (prose): BP-1 (partly: recall of the autograd graph, scalar graph
shortened and called a reminder), BP-2, BP-3, BP-4 (reason for weight decay
stated), BP-5, BP-7 (equation and prose only), BP-9, BP-10.

Already fixed before this pass, left as found: BP-6 (the sentence "following
the arrows of :numref:`fig_forward` backward from $J$, right to left along
the bottom row" is verbatim), BP-8 (the $-0$ remark in the tensorflow/jax tab
note is untouched).

Deferred:
- BP-1, second half: changing the scalar example to one with a reused value
  needs a new figure.
- BP-4, alternative: dropping weight decay from the derivation changes code
  cells and the figure.
- BP-7, the six-line loop: needs a new code cell. Exercise 3(a) still asks
  the reader to "generalize" the backward pass to $K$ layers, which the text
  now states; rephrase it when exercises are next edited.
- BP-11: one gradient notation for this section and the next is a
  chapter-wide decision.
- "Three different setups" (structural): needs code changes.

## Factual claims added or changed

- Opening: in `sec_autograd` the backward pass through a network "remained
  a black box" (that section's own closing words).
- "In :numref:`sec_autograd` a matrix–vector product passed the gradient
  back to its vector through $\mathbf{X}^\top$." (autograd, "The Backward
  Pass".)
- Weight decay is included because training usually includes it and because
  it makes each weight matrix feed two nodes (author intent as inferred by
  the review; please confirm).
- "$\mathbf{G}\mathbf{W}^\top$ is what hands the gradient on to the layer
  below."
- Reading of line 3 of :eqref:`eq_backprop-mlp`: hidden unit $j$ receives the
  errors of the $q$ outputs, each multiplied by the weight from unit $j$ to
  that output, i.e. $[\partial J/\partial \mathbf{H}]_{ij} = \sum_k
  [\partial J/\partial \mathbf{O}]_{ik} w^{(2)}_{jk}$.
- New display, the step for $K$ layers:
  $\partial J/\partial \mathbf{Z}^{(k)} = (\partial J/\partial
  \mathbf{Z}^{(k+1)} \mathbf{W}^{(k+1)\top}) \odot \phi'(\mathbf{Z}^{(k)})$
  for the hidden layers, and $\partial J/\partial \mathbf{W}^{(k)} =
  \mathbf{H}^{(k-1)\top} \partial J/\partial \mathbf{Z}^{(k)} + \lambda
  \mathbf{W}^{(k)}$ for all layers, with $\mathbf{H}^{(0)} = \mathbf{X}$ and
  $\mathbf{O} = \mathbf{Z}^{(K)}$. It assumes weight decay on every layer.
  The notation ($\partial J/\partial \mathbf{Z}^{(k)}$, minibatch matrices)
  differs from the per-example $\nabla$ notation of the next section
  (BP-11, NS-3).
- "Finite differences would need one forward pass per parameter, some two
  hundred thousand per training step" (203,530 rounded).
- "the derivative of the loss with respect to every parameter is read off
  these two matrices by a single product or sum" ($L$, not $J$: the penalty
  adds $\lambda\mathbf{W}$ separately).
- "Even for this small network, then, the values saved for one minibatch
  outweigh the model" (1.1 MB against 0.8 MB, numbers unchanged).
- The dead unit is attributed to `sec_mlp`, which defines it.
- No number describing an output was changed. Stored outputs checked: all
  four frameworks print relative differences of 1e-16 to 5e-16 (below
  $10^{-14}$ as stated); $-0$ appears only in the tensorflow and jax outputs,
  as the tab note says.

## Slides

No slide statement contradicts the revised body. Differences of coverage
only:
- The slides do not have the new material: the reading of $\partial
  J/\partial \mathbf{H}$ as error sent backward, the $K$-layer step, or the
  answer to the opening question (the first slide poses it with the finite
  differences comparison, the recap does not return to it).
- Slide "Every Node Needs Stored Values and a Finished Gradient" poses "Two
  questions for each node"; the body now presents the same two points after
  the example as "Two things in this small computation".
- The slides present the scalar graph as new material, the body as a
  reminder of `sec_autograd`.
