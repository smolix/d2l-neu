# Writing pass: `chapter_preliminaries/autograd.md` (section 1.5)

Prose only. No code cell, label, figure include, caption, exercise, or slide
was changed; the checker ends "OK: only prose changed" and `lint_source.py`
is clean.

## Words

Prose words (all tabs, to the slides marker): 5099 before, 5088 after.

The hand was kept light: the section's line of thought (hand gradient, then
the framework, then the same step, then "the operations that ran") is
untouched, and so are its quotable sentences ("no derivative appears anywhere
in the code", "Detaching changes the graph, not the values", "Automatic
differentiation differentiates the operations that ran, not the function that
the program represents", "For a loss, the comparison is lopsided", the
closing hand-over to probability).

## Before and after

1. Opening, answer to the opening question.
   - Before: "Deep learning frameworks do this work for us."
   - After: "Luckily, deep learning frameworks do this work for us, and the
     idea behind it is simple."

2. After the first backward pass.
   - Before: "The result is $(2, 4)$, the gradient
     $2\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$ that we derived by
     hand in :numref:`sec_calculus`. There we multiplied by two factors,
     $2\mathbf{r}$ and $\mathbf{X}^\top$."
   - After: "The result is $(2, 4)$, as it should be. How did the framework
     arrive at it without being told the formula
     $2\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$? By hand we
     multiplied by two factors, $2\mathbf{r}$ and $\mathbf{X}^\top$."

3. Gradient accumulation, PyTorch note (now after the cell).
   - Before: "PyTorch adds the new gradient to the one already stored in
     `w.grad`, so `w.grad` now holds twice the gradient. [...] A forgotten
     reset is a common bug."
   - After: "The answer is $(4, 8)$, twice the gradient: PyTorch added the
     new gradient to the one already stored in `w.grad`. [...] A forgotten
     reset is a common bug, and no error message reveals it."

4. Vector-valued functions.
   - Before: "Let $\mathbf{u}$ be a vector of $m$ values computed from $n$
     inputs $\mathbf{x}$. Its derivative is the $m \times n$ Jacobian
     $\mathbf{J}$ [...] The backward pass does not compute $\mathbf{J}$. It
     starts from a vector $\mathbf{v}$ with one entry per output and returns"
   - After: "What should the backward pass return then? It can differentiate
     only one number, so we have to say which. The choice that frameworks
     offer is a weighted sum $\mathbf{v}^\top \mathbf{u} = \sum_j v_j u_j$ of
     the outputs $\mathbf{u}$, with a vector of weights $\mathbf{v}$ that we
     supply."

5. Detaching values.
   - Before: "A common case of the first kind is a target: a value that the
     output of a model should move toward, but that the model itself
     computes. Reinforcement learning trains models this way, with targets
     built from the model's own predictions."
   - After (opening): "To *detach* a value is to keep the value but cut its
     connection to the parameters, so that the backward pass treats it as a
     constant." After (end of subsection): "The typical use is a target that
     the model computes itself. Q-learning (:numref:`sec_qlearning`), for
     instance, moves a prediction toward a target built from the model's own
     predictions. The target must then act as a constant, or else an update
     could lower the loss by moving the target toward the prediction, which
     improves nothing."

6. Control flow.
   - Before: "The function `f` below doubles its input until the norm of the
     result reaches 1000, [...]"
   - After: "Let us try this on a much smaller program with the same two
     features, a loop whose length and a branch whose choice both depend on
     the input. The function `f` below doubles its input until [...]"

7. Rows and columns of the Jacobian.
   - Before: "The errors are $m = 3$ outputs of $n = 2$ inputs, and their
     Jacobian is $\mathbf{X}$, so reverse mode needs three sweeps and forward
     mode two."
   - After: "The answer is no secret, since the Jacobian is $\mathbf{X}$.
     What interests us is the count: the errors are $m = 3$ outputs of
     $n = 2$ inputs, so reverse mode needs three sweeps and forward mode
     two."

8. Closing.
   - Before: "We can now compute the gradient of any loss that our code
     computes, control which parts of the computation the gradient flows
     through, and take a gradient step. We have treated the backward pass
     through a network as a black box; :numref:`sec_backprop` opens it for a
     network with one hidden layer. A gradient tells us how to lower the loss
     on the data at hand."
   - After: "We have not yet looked inside the backward pass of a network;
     :numref:`sec_backprop` does so for a network with one hidden layer. Note
     also what a gradient cannot tell us. It shows how to lower the loss on
     the data at hand."

## Repetition removed

- **Recap of the calculus section** (opening, second paragraph): the
  derivation, the Jacobian, and the cost and precision of finite differences
  (owned by `calculus.md`) reduced to two sentences naming the model and its
  gradient $(2, 4)$.
- **The cost claim** ("a small multiple of the cost of computing the loss,
  whatever the number of parameters"): was in the opening, in "Rows and
  Columns", in "The Cost of a Gradient", and in the Summary. Now stated as a
  promise in the opening and as the result in "The Cost of a Gradient". "Rows
  and Columns" says only that the two kinds of sweep cost about the same, a
  few evaluations of the function (the argument needs it); the Summary no
  longer repeats "whatever the number of parameters".
- **Callback to the line model's backward-pass cost** in "The Cost of a
  Gradient" (a restatement of the calculus closing): removed.
- **"We can now ..." recap** before the Summary: removed; the Summary is the
  only recap.
- The $(2, 4)$ formula is no longer re-attributed to the calculus section
  after the backward pass (the sentence before the cell already does it).

## Review findings

Applied:

- **AG-1** opening recap cut to two sentences.
- **AG-2** dropped "which is `None` until the first backward pass" (no cell
  shows it).
- **AG-3** vector-valued functions reordered: practical statement and cell
  first, the identity $\mathbf{J}^\top\mathbf{v}$ after the output. The JAX
  tab note before the cell now says "the gradient of the weighted sum"
  instead of $\mathbf{J}^\top\mathbf{v}$, which is defined only afterwards.
- **AG-4** detach now leads with the definition and the toy example; the
  target motivation is three sentences at the end, pointing to
  :numref:`sec_qlearning`.
- **AG-5** "leaf tensor" defined in a clause.
- **AG-6** connecting sentence before `f`.
- **AG-9** Hessian-vector product removed from the memory sentence (left to
  the last exercise, which defines the Hessian itself).
- **Structural, tab note before its surprise**: the MXNet, PyTorch and
  TensorFlow notes on gradient accumulation now follow the cell. The JAX note
  has no cell and stays where it was.
- **Structural, two levels**: "Higher-Order Derivatives" opens with "The rest
  of this section can be skipped on a first reading. The reader in a hurry
  needs one result from it: a single backward pass yields the whole gradient
  of a loss.", and the roadmap in the opening says the end of the section
  goes further than the next chapters need.
- **Structural, repetition of the cost claim**: see above.
- **AG-7** in part: the "some methods ... some optimizers" motivation is
  replaced by one plain sentence (curvature, definition of the Hessian).

Deferred:

- **AG-7** (reduce higher-order derivatives to two sentences) and **AG-8**
  (cut the Jacobian-filling cell): both need a code cell removed.
- **Vocabulary, drop "sweep" for "pass"**: "sweep" is in code comments,
  printed output, a figure caption, two exercises and the slides.
- **Shared prose "Both values match the formulas"** has no output in the
  MXNet tab (no cell there): needs a cell or a chapter decision.
- Moving the "Cost of second derivatives" exercise and the exercise
  numbering: exercises are out of scope.

## Factual claims added or changed

- "A forgotten reset is a common bug, and no error message reveals it."
  (New clause: PyTorch raises nothing when gradients accumulate.)
- PyTorch accumulation note now names the stored output: "The answer is
  $(4, 8)$" (stored output `tensor([4., 8.])`).
- "It [the backward pass] can differentiate only one number", and "The choice
  that frameworks offer is a weighted sum $\mathbf{v}^\top\mathbf{u}$".
- Leaf tensor: "a tensor such as `w` that we marked ourselves and that is not
  the result of a recorded operation".
- Detach motivation now cites Q-learning (:numref:`sec_qlearning`) rather
  than "reinforcement learning"; that section holds its target fixed (the
  semi-gradient step). The text says the target "must act as a constant" and
  does not claim that section calls `detach`.
- "The rest of this section can be skipped on a first reading" and "goes
  further than the next chapters need": I found no use of second derivatives
  in code, `jvp`/`vjp`, or forward mode in chapters 2 to 8 (the only hit is a
  pen-and-paper Hessian exercise in softmax regression).
- "The two kinds of sweep cost about the same, a few evaluations of the
  function" replaces "Either sweep costs a small multiple of one evaluation
  of the function".
- "which is no better than finite differences" replaces "as many evaluations
  of the loss as finite differences" (same count; the new wording ignores
  that forward mode is exact where finite differences are not).
- Removed: `w.grad` is `None` before the first backward pass; the
  Hessian-vector product remark; the mention that Hessian-vector products are
  used by some optimizers; the MXNet remark that filling the Jacobian by
  columns would need forward mode (the note before the cell already says
  MXNet has only reverse mode).

## Slides against the body

No statement in the slides contradicts the revised body. The detach slide
already motivates with "a target that the model computes must stay fixed
while the prediction moves", which matches. The slides still say a sweep
costs "a small multiple of one" evaluation, which agrees in substance with
"a few evaluations".
