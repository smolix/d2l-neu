# Writing pass: `chapter_recurrent-neural-networks/bptt.md`

Section 6 of 7 in chapter 8 (sequence, text-sequence, language-model, rnn,
rnn-implementation, **bptt**, decoding). Prose only; no cell, label, figure,
display equation, exercise, or slide was changed.

## Words

Prose words (checker): 2,956 before, 2,983 after (+27). The section did not
get shorter. About 190 words of repetition were removed, and the two
additions the review asked for (BPTT-4, what the $\tanh$ changes; BPTT-5,
the forward/backward bridge) used them up. Checker: "OK: only prose
changed"; `tools/lint_source.py`: clean.

## Before and after

1. Before: "We first keep every quantity scalar, which exposes the dependency
   structure. The vector case replaces each derivative by a Jacobian."
   After: "Let us start with scalars. They show the whole dependency
   structure, and the vector case merely replaces each derivative by a
   Jacobian."
2. Before: "... and apply the chain rule. This is called *backpropagation
   through time* :cite:`Werbos.1990`."
   After: "... and apply the backpropagation of :numref:`sec_backprop` to it.
   This is called *backpropagation through time* :cite:`Werbos.1990`. As an
   algorithm it is nothing new; what is new is the shape of the result."
3. Before: "A matrix power is governed by the eigenvalues of the matrix. ...
   Along an eigenvector with eigenvalue $\lambda$, the $k^\textrm{th}$ power
   multiplies by $\lambda^k$, which shrinks geometrically if $|\lambda| < 1$
   and grows geometrically if $|\lambda| > 1$."
   After: "How large is the $k^\textrm{th}$ power of a matrix? We have met
   this question twice, in :numref:`subsec_lin-alg-eigenvalues` and for the
   layers of a deep network in :numref:`sec_numerical_stability`: the power
   multiplies each eigenvector by $\lambda^k$, and the largest $|\lambda|$
   soon dominates."
4. Before: "We can check this numerically. A symmetric matrix has real
   eigenvalues and orthogonal eigenvectors, so its largest singular value
   equals $\rho$ and the operator norm of its $k^\textrm{th}$ power is exactly
   $\rho^k$."
   After: "For a symmetric matrix the statement is exact: the norm of its
   $k^\textrm{th}$ power is $\rho^k$ for every $k$. The following cell is
   therefore a check of the arithmetic more than an experiment."
5. Before: "A gradient that has been shrunk to 1.5% or amplified 45-fold
   carries little reliable information about a dependency 40 steps back."
   After: "A contribution shrunk to 1.5% is lost among the nearer ones; one
   amplified 45-fold swamps them."
6. Before: "A nonlinearity such as the $\tanh$ of :numref:`sec_rnn-scratch`
   multiplies each step by its derivative, a number between 0 and 1 that
   depends on the state. The exercises explore what this changes."
   After: "The model of :numref:`sec_rnn-scratch` is not this tidy: its
   $\mathbf{W}_\textrm{hh}$ is not symmetric, and it has a $\tanh$. Does the
   analysis survive? In outline, yes. ... The $\tanh$ multiplies the gradient
   of each unit by its derivative $1 - h^2$, a number between 0 and 1, so it
   never lengthens a gradient that passes through it. It pushes toward
   vanishing, most strongly when units saturate, and it does not rule out
   explosion when $\mathbf{W}_\textrm{hh}$ is large and the states are near
   zero. We measure neither effect here; the last two exercises do."
7. Before: "The two failure modes need different remedies. For explosion, the
   immediate danger is a single update so large that it wrecks the
   parameters."
   After: "The two failure modes need different remedies, and explosion is
   the easier one. Its immediate danger is a single update so large that it
   wrecks the parameters."
8. Before: "We can now say where the gradient of a recurrent network comes
   from, why it vanishes or explodes, and what truncation removes from it.
   Clipping handles explosion. Vanishing gradients remain. ..."
   After: "Simple as it is, truncation at a fixed length remains the common
   choice. For a plain recurrence it also costs less than the hard limit
   suggests: unless the spectral radius sits close to 1, the signal from a
   few dozen steps back has faded geometrically before truncation removes it.
   Which of the two, the window or the fading, limits our trained model is
   the subject of the last exercise."

## Repetition removed

- **Clipping formula** (owned by `rnn-implementation.md`,
  :eqref:`eq_grad_clipping`): the inline formula is gone; one clause with the
  equation reference says what clipping does.
- **Carry and detach** (owned by `rnn-implementation.md`, "Training"): the
  five-sentence paragraph is now one sentence that points there. The italic
  definition of *detaches* went with it.
- **Powers of a matrix** (owned by `linear-algebra.md` and
  `numerical-stability-and-init.md`): the per-eigenvector argument is reduced
  to one clause with both references. Kept here: the definition of the
  spectral radius, $\rho^k$, and one sentence on what is new (every step
  contributes the same matrix).
- **Backpropagation** (owned by `sec_backprop`): cited in the roadmap; the
  second telling of "unrolling gives a feedforward network" is shortened.
- **$0.9^{40}$ and $1.1^{40}$**: derived in the opening only; after the cell
  the two values are cited without the powers.
- **Closing paragraph** ("We can now say ..."): replaced. It restated what
  the body and the Summary both say. "Truncation biases toward short
  dependencies" is now said once.
- Two smaller cuts: "Training stays stable while later updates move the
  recurrence away from the explosive regime" (a hope, not shown) and "no
  rescaling changes their ratio" (said in the sentence before).

## Review findings

Applied: BPTT-4 (what the $\tanh$ and a nonsymmetric matrix change, in
text), BPTT-5 (bridge between the forward scalar and the backward vector
derivation), BPTT-6, BPTT-7, BPTT-8 (random truncation cut), BPTT-9, IMP-11
(the half of it that lies in this file). BPTT-1 was already fixed in the
file.

Deferred:

- BPTT-2 (rename $b_t$, $g$): changes display equations and the slides,
  which still write $a_t = b_t + c_t a_{t-1}$; needs one decision for both.
- BPTT-3 (replace the tautological experiment): needs a new cell and
  outputs. The prose now says openly that the cell checks arithmetic.
- BPTT-8, second half: the citation `Tallec.Ollivier.2017` should move to the
  chapter's further reading, which is another file. It is now cited only in
  this file's slides.
- Four import cells for one numpy cell: code.
- Exercise 3 wording: exercises not touched.
- Order of `bptt` and `rnn-implementation`: chapter decision.

## Claims added or changed

- "As an algorithm it is nothing new" (of BPTT against backpropagation).
- "Unlike the layers of a deep network, the steps of a recurrence all
  contribute the same matrix" (true of the linear recurrence analysed here).
- The $\tanh$ derivative is written as $1 - h^2$, and: it "never lengthens a
  gradient that passes through it"; it "pushes toward vanishing, most
  strongly when units saturate"; it "does not rule out explosion when
  $\mathbf{W}_\textrm{hh}$ is large and the states are near zero". I did
  **not** adopt the review's wording "can therefore only shrink the product":
  for nonsymmetric matrices a diagonal factor between the matrices can
  remove a cancellation, so the product with the $\tanh$ factors is not
  always smaller than the plain power.
- "A contribution shrunk to 1.5% is lost among the nearer ones; one amplified
  45-fold swamps them" replaces "carries little reliable information".
- "biased, and biased toward short dependencies" now carries the citation
  `Jaeger.2002` (the second half previously stood without one).
- Linear recurrences: "eigenvalues have magnitude at most 1" (was "inside
  the unit circle").
- New closing claim: for a plain recurrence whose spectral radius is not
  close to 1, truncation "costs less than the hard limit suggests", because
  the dropped terms had already faded. It is conditional, and the text says
  the trained model was not measured.
- "about 16 steps on average" is kept; "never more than 31 steps" is now
  implied by "the loss at the last position reaches back 31 steps".
- Removed: "Training stays stable while later updates move the recurrence
  away from the explosive regime."

## Slides that now disagree with the body

- "Clipping Bounds the Step but Cannot Fix Vanishing" still shows the
  clipping formula, which the body now only references.
- The slide on linear recurrences says the eigenvalues "lie inside the unit
  circle"; the body says "magnitude at most 1".
- "Truncation Cuts Each Loss Off at the Start of Its Segment" still has the
  bullet "Random truncation lengths, reweighted: unbiased, but higher
  variance", which the body no longer mentions.
- No slide mentions what the $\tanh$ changes, which the body now does.
