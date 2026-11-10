# Writing pass: `chapter_preliminaries/calculus.md`

Prose words (checker): 4978 before, 4944 after (-34). Checker: "OK: only prose
changed". `tools/lint_source.py`: clean. This section was changed least of
all, as instructed: 20 passages touched, the running example and every number
in it untouched.

## Before/after pairs

1. Finite differences, opening.
   - Before: "It looks like a practical way to compute derivatives, since its
     error, $3h$ in our example, shrinks with $h$. That is true in exact
     arithmetic but not on a computer."
   - After: "It looks like a practical way to compute derivatives. Its error,
     $3h$ in our example, shrinks with $h$, so why not simply choose a tiny
     $h$? In exact arithmetic this would work. On a computer it does not."

2. Rules of differentiation, opening.
   - Before: "Computing limits is tedious, and for most functions we never
     need to: a few derivatives of basic functions, combined with a few rules,
     cover every function built from them."
   - After: "Computing limits is tedious. Luckily, we almost never need to: a
     few derivatives of basic functions, combined with a few rules, cover
     every function built from them."

3. Linear approximation.
   - Before: "The approximation predicted $1 - 2 \cdot 0.2 = 0.6$. The gap of
     $0.12$ is the term $3h^2$ of $L(1+h) = 1 + 2h + 3h^2$, with $h = -0.2$,
     which the tangent ignores."
   - After: "The approximation predicted $1 - 2 \cdot 0.2 = 0.6$. Where does
     the gap of $0.12$ come from? It is the term $3h^2$ of
     $L(1+h) = 1 + 2h + 3h^2$, with $h = -0.2$, which the tangent ignores."

4. Partial derivatives, part opening.
   - Before: "Here we extend the derivative to functions of many variables,
     and find that the extension answers a new question: among all the
     directions in which the parameters could move, which one lowers the loss
     fastest?"
   - After: "Let us therefore extend the derivative to functions of many
     variables. The extension turns out to answer a question that a single
     parameter could not even pose: among all the directions in which the
     parameters could move, which one lowers the loss fastest?"

5. The gradient, candour about the argument.
   - Before: "The argument evaluates each partial derivative at a point that
     has already moved a little, so the approximation needs partial
     derivatives that change continuously; ..."
   - After: "Our argument is not quite a proof. It evaluates each partial
     derivative at a point that has already moved a little, so the
     approximation needs partial derivatives that change continuously; ..."

6. Steepest descent.
   - Before: "Steepest descent is a local property, however."
   - After: "Note, however, that "steepest" is a local statement, and it says
     nothing about where the minimum lies."

7. Product rule.
   - Before: "The product rule has a picture: ..."
   - After: "The product rule is the only one that may look arbitrary, and it
     has a picture: ..."

8. Closing paragraph.
   - Before: "We can now say how the loss changes when the parameters move, in
     which direction it falls fastest, and how the chain rule computes that
     direction stage by stage. What remains is the labor. Deriving the
     Jacobian of every layer by hand, for networks with dozens of layers and
     millions of parameters, is slow and error-prone. :numref:`sec_autograd`
     shows how the framework records the forward pass and runs the backward
     pass for us, and why the gradient then costs a small multiple of the cost
     of computing the loss, whatever the number of parameters."
   - After: "What remains is the labor of deriving the Jacobian of every layer
     by hand, which is slow and error-prone for networks with dozens of layers
     and millions of parameters. It is also entirely mechanical: nothing in
     the backward pass above required insight, only care.
     :numref:`sec_autograd` hands this labor to the framework, which records
     the forward pass and runs the backward pass for us, and explains why the
     gradient of a deep network, like that of the line model, costs only a
     small multiple of computing its loss."

## Repetition removed

- **Closing recap.** The "We can now say ..." sentence restated the body and
  the Summary; removed. The Summary is the only recap.
- **Cost claim ("a small multiple of the cost, whatever the number of
  parameters").** Ownership by the earliest-section rule: this section
  *demonstrates* the cost on the line model (two matrix-vector products
  against one loss evaluation per parameter) and keeps that. The general
  claim is *explained* only in autograd ("Forward and Reverse Mode"), so
  autograd owns it. Here it is one forward-pointing clause in the closing,
  tied to the line model, without the "whatever the number of parameters"
  refrain.
- **Cosine formula and Cauchy-Schwarz** in "Steepest Descent": these are
  explained (and Cauchy-Schwarz proved) in the linear algebra section, which
  is earlier. Reduced from four sentences to one clause with
  `:eqref:`eq_dot_cos``. The formula itself is still displayed inline, since
  the next equation uses it.
- **Plotting utilities prose** cut from 9 lines to 6: the descriptions of
  `legend_loc='outside'` and `set_log_ticks` repeated the docstrings and are
  gone; `#@save` is one clause pointing at the preface, which explains it.

## Review findings

Applied:

- CAL-3: added "since the loss is about $1$ here" to the $\epsilon/h$ bound.
- CAL-4: the plot paragraph no longer says "both error curves form a V". It
  reads the plot from the right, says each curve leaves the $3h$ line where
  rounding takes over, and explains the flat part of the float32 curve.
- CAL-5: the three domain conditions are now "wherever the function is
  defined (the logarithm, for one, requires $x > 0$)".
- CAL-7: "Later chapters refine the step (:numref:`chap_optimization`), but
  every refinement starts from the gradient." (momentum, per-parameter step
  sizes, curvature no longer named).
- CAL-8 (prose part): the subsection now opens from the line model ("without
  writing out three squares"); $\mathbf{A} = \mathbf{I}$ is gone (the squared
  norm is written as $\sum_i x_i^2$); the Frobenius-norm identity and the
  sentence defining the gradient with respect to a matrix are cut (unused
  here; no other file refers to them in this section).
- CAL-10: "What remains is the labor." folded into the next sentence.
- CAL-11: the unused notation $\partial_{x_i} f$ is cut.
- Two badly wrapped lines (product rule, backpropagation) rewrapped.

Deferred:

- CAL-1 (unused imports): code.
- CAL-2 (move the Archimedes paragraph after the secant figure): would
  reorder two figure includes; the text already says the figure shows only
  the inscribed polygons.
- CAL-6 (rectangle figure for the product rule): needs a figure; the prose
  argument is kept.
- CAL-9 (move "Plotting Utilities" out of the chapter): code move between
  files. Prose around it shortened and made candid ("matplotlib plumbing ...
  can be skipped").
- Structural: moving "Finite Differences" after `numerical_gradient`: reorders
  code cells.
- Exercises (optional mark on 1.2, a tangent-prediction exercise): exercises
  left alone.

## Factual claims added or changed

- "In float32 this happens for every $h$ below about $10^{-7}$, which is why
  the left half of that curve is flat." Checked in NumPy float32 with the
  section's own `h` grid: the error is exactly 2 for all $h \le 5.6 \times
  10^{-8}$, that is 8.75 of the 15 decades plotted. Not checked against the
  four stored plot images pixel by pixel; the review describes the same
  plateau in the stored PyTorch plot.
- "Each curve leaves this line where rounding takes over, and from there on a
  smaller $h$ makes matters worse." (float64 minimum at $10^{-8}$, float32
  near $10^{-4}$ in the same check; the trend is upward but noisy, which the
  next sentence says.)
- "a question that a single parameter could not even pose" (with one
  parameter there are only two directions).
- "The product rule is the only one that may look arbitrary" (a judgment).
- "nothing in the backward pass above required insight, only care" (a
  judgment about the hand derivation of the line model).
- "as the preface explains" for `#@save` (the preface has no label, so this
  is a plain mention, not a `:numref:`).
- Removed claims: the Frobenius-norm gradient $2\mathbf{W}$; that the cosine
  formula defines the angle in more than three dimensions and that
  Cauchy-Schwarz bounds the cosine (both remain in linear algebra).

## Slides that now disagree with the body

None found. The finite-difference slide speaks of a "left branch" with
rounding error below $\epsilon/h$, which is still what the body says; it does
not use the word "V". The slides do not use $\mathbf{I}$, the Frobenius
identity, $\partial_{x_i} f$, or the cost refrain.
