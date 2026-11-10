# Writing pass log: Matrix Calculus

File: `chapter_mdl-calculus/mdl-matrix-calculus.md`. Date: 2026-11-09. Prose
only; no code cell, cell ID, figure include or slide was changed. Not split.

## Length

Checker (`check_mdl_prose.py`, body plus exercises, label names counted as
words): 6241 before, 6285 after (+44). The pass added 23 label references and
3 `:exlabel:` lines, which the counter reads as about 85 words. Counted
without label names and without the exercises, the body went from 4644 to
4619 words. The cuts (index derivation, repeated motivations, recap,
fused-primitive sentences, third telling of the log-sum-exp Hessian) are
about 450 words; the additions that the review asked for (MC-7, MC-10, MC-11,
MC-12, MC-14, MC-16, the minibatch bias, the ranking paragraph) are about 400.

## Correctness changes

- **MC-11.** Was: "This $1\times1$ check catches wrong shapes, but it cannot
  catch a wrong transpose". Now: the check "catches a wrong coefficient or a
  missing term. It is blind to transposes and to the order of factors,
  because $1\times1$ matrices commute and equal their own transposes. The
  shape rule catches most of those errors, but unfortunately not when the
  factor is square ... There only a numerical check helps." The
  counterexample now ends "passes the $1\times1$ check, has the right shape,
  and is wrong". Verified: with $A=\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$,
  $x=(1,2)$: $(A+A^\top)x=(2,1)$, $2Ax=(4,0)$.
- **MC-7.** Was: "$\mathbf W^K$ ... grows or shrinks like
  $|\lambda_{\max}|^K$" (read as a bound), placed under "The Order of
  Multiplication". Now, at the end of "The Jacobian of a Network": "for large
  $K$ the norm of the product grows or shrinks like $|\lambda_{\max}|^K$ ...
  The rate is only the long-run one. When the eigenvectors in $\mathbf V$ are
  far from orthogonal, the product can grow for many steps before it decays
  (`subsec_mdl-transient-growth`), and a product of different factors is
  governed by singular values (`subsec_mdl-jacobian-products`)." Verified
  numerically with $W=\left(\begin{smallmatrix}0.9&5\\0&0.8\end{smallmatrix}\right)$:
  $\|W^K\|_2$ is 5.1, 13.2, 12.1, 0.26 at $K=1,5,10,50$ against $0.9^K$; the
  bound itself is `eq_mdl-nonnormal-power-bound` in the linear algebra
  chapter, which I cite and do not restate. The paragraph was moved across no
  code cell boundary that changes cell order (it now follows the network
  cell's output sentence).
- **MC-10.** "Four rules compute $df$" listed the move rule, which computes
  nothing, and omitted the entrywise rule used later. Now the four are
  linearity, product, transpose, and $d\varphi(\mathbf X)=\varphi'(\mathbf
  X)\odot d\mathbf X$; the move rule is introduced separately as the
  rearranging step, with its entrywise companion $\langle\mathbf A,\mathbf
  B\odot\mathbf C\rangle=\langle\mathbf A\odot\mathbf B,\mathbf C\rangle$
  (both sides are $\sum_{ij}A_{ij}B_{ij}C_{ij}$).
- **MC-15.** "Every rule also uses forward values" became "The rules for the
  product and the activation also use forward values ... The bias rule uses
  none."
- **MC-16 (prose half).** The text after the random-point cell now says what
  the test can detect: only the quadratic-form line (random nonsymmetric
  $3\times3$ $\mathbf A$) could print `False` for a transpose error; the
  matrix product is tested with $3\times4$ and $4\times2$ factors, for which
  a misplaced transpose raises a shape error. Checked against the cell: the
  wrong products `Ybar @ W` and `X @ Ybar` do not conform. "for any random
  draw" was dropped.
- **MC-12 / chapter notation.** The adjoint is no longer introduced as new.
  Now: "$\bar{\mathbf Y}=\nabla_{\mathbf Y}\ell$ for the adjoint of
  `eq_mdl-adjoint`, now of a matrix: the total derivative of the loss with
  respect to $\mathbf Y$, arranged in the shape of $\mathbf Y$." The phrase
  "the row $\partial\ell/\partial\mathbf y$ times the Jacobian" (a total
  derivative written with $\partial$) was removed. Added the reason for the
  name: the backward rule is the transpose of the layer's linearization, and
  the transpose of a linear map is its adjoint.
- **MC-13.** "the left-to-right order of the previous subsection" now cites
  `eq_mdl-jacobian-product`.
- **MC-17.** "Two checks against intuition" became one check ($\mathbf
  J\mathbf 1=\mathbf 0$) and one fact stated as a fact (the Jacobian is
  symmetric because it is a Hessian).
- No finding disputed. One correction to the review: MC-9 says the Kronecker
  delta is never used again in the body; the softmax Jacobian uses
  $\delta_{ij}$, so I define it there in a clause.

## Six before/after pairs

1. Before: "Does the error of the linear model fall faster than
   $\|\boldsymbol\delta\|$ itself? We shrink one perturbation tenfold at a
   time and print the error divided by $\|\boldsymbol\delta\|$."
   After: "The definition says that the error of the linear model, divided by
   $\|\boldsymbol\delta\|$, tends to zero. It does not say how fast. We shrink
   one perturbation tenfold at a time and print that ratio."
2. Before: "Does a framework's automatic differentiation agree with the
   product $\mathbf w_2^\top\operatorname{diag}(\mathbf m)\mathbf W_1$? We
   define the network as a function of the input and of its parameters, then
   compare the two."
   After: "We now define the network in code, for reuse later in the section,
   and print the product ... beside the gradient that a framework's automatic
   differentiation returns."
3. Before: "Do the backward rules reproduce what automatic differentiation
   computes for the parameters? We reuse `g` from above and differentiate
   with respect to $\mathbf W_1$, $\mathbf b$ and $\mathbf w_2$."
   After: "A framework differentiating `g` with respect to $\mathbf W_1$,
   $\mathbf b$ and $\mathbf w_2$ returns the same three arrays."
4. Before: "Each hand formula matches the framework entry for entry. The
   inactive unit shows up three times"
   After: "What deserves a look is where the zeros are. The inactive unit
   shows up three times"
5. Before: "All three lines print `True`, for any random draw."
   After: "All three lines print `True`, but only the second could have
   printed `False` for a transpose error."
6. Before: "The step that is not routine is the second equality:
   $d\mathbf x^\top\mathbf A\mathbf x$ is a $1\times1$ matrix, so it equals
   its own transpose"
   After: "The second equality transposes a $1\times1$ matrix:
   $d\mathbf x^\top\mathbf A\mathbf x$ is a number, so it equals its own
   transpose"

Also: the opening of "The Best Linear Approximation" now says why the network
is set aside (MC-2): "Near $\mathbf x=(1,1)$ the hidden layer of our network
is affine, so its linear approximation is exact there and would leave no
error to study. Let us therefore start with a curved map".

## Repetition

Removed or reduced to a clause with a reference:

- "a layer maps a vector to a vector" (MC-1): the two restatements under "The
  Jacobian" are gone; the opening keeps it.
- Tables and the silent transpose (MC-8): kept in the opening only.
- Counterexample $2xy/(x^2+y^2)$: three sentences with
  `subsec_mdl-gradient` (owner: multivariable).
- Adjoint: cites `eq_mdl-adjoint` (owner: multivariable).
- Log-sum-exp Hessian: the derivation cites `eq_mdl-lse-hessian`; the
  $2\times2$ worked case and the "linear along $\mathbf 1$" argument were
  removed. The display `eq_mdl-softmax-jacobian` stays (label, and the
  $n$-class form is needed for $\mathbf p-\mathbf y$). Kept and extended: the
  variance reading, now with the covariance of a one-hot vector.
- Gradient of $\mathbf x^\top\mathbf A\mathbf x$ by indices (MC-9): one
  sentence; Einstein summation no longer introduced here.
- Fused softmax primitive (MC-18a): one clause with
  `subsec_softmax-implementation-revisited` and `subsec_mdl-stable-softmax`.
- Paragraph after the last cell (MC-18b) and the "We can now" recap (MC-19):
  deleted. The body ends on the hand-over to automatic differentiation.
- NP-completeness: one clause with the citation.
- Determinant remark (MC-3): removed; exercise 2 covers it.
- The factorization-loss derivation in the body: reduced to a pointer to the
  exercise that asks for it.
- $1\times1$ check repeated after least squares and after the matrix product:
  removed.

Kept because this file owns them: vector-Jacobian products and the layer
rules (`subsec_mdl-layer-vjp`), the weight gradient as an outer product, the
count by order of multiplication, $\mathbf W^K$.

Main-text references added in the body: `sec_backprop` (three places: opening,
index route, layer rules), `subsec_softmax_and_derivatives` (opening and
$\mathbf p-\mathbf y$), `subsec_softmax-implementation-revisited`,
`sec_numerical_stability`, `sec_bptt`. Appendix references added:
`sec_mdl-multivariable_calculus`, `subsec_mdl-gradient`,
`sec_mdl-tangent-fails`, `eq_mdl-adjoint`, `eq_mdl-lse-hessian`,
`subsec_mdl-transient-growth`, `subsec_mdl-stable-softmax`,
`sec_mdl-matrix-calculus-autodiff` (modes; memory cost). All resolve.

## Ranking; Propositions; labels

- New paragraph at the end of the opening names the three results to remember
  (`eq_mdl-jacobian-product`, the shape rule, `eq_mdl-matmul-vjp`), says the
  chain-rule proof can be skipped on a first reading and the differential
  method cannot.
- No Proposition demoted: the section has two (chain rule, backward rule of a
  layer) and both are cited later.
- Labels added (exercise labels, MC-5): `ex_mdl-matrix-calculus-relu-jacobian`
  (exercise 1), `ex_mdl-matrix-calculus-chain-order` (exercise 3),
  `ex_mdl-matrix-calculus-matmul-rules` (exercise 4). The body now uses
  `:exref:` in place of "the first exercise" and "see the exercises".

## Split markers

None; the section is not split.

## Cells to cut or redesign on the development machine

- `mdl-matrix-calculus-autodiff-a-few-key-identities-derived-not-tabulated-1`
  (MC-16): redesign. Use square, nonsymmetric $\mathbf X$, $\mathbf W$; print
  the wrong guesses too (`2 A x`, the transposed matmul rule) so that a line
  prints `False`. The prose now states the cell's limits and should be
  rewritten when the cell is.
- `mdl-matrix-calculus-the-jacobian-of-a-network` (MC-16): confirmation only.
  It must stay in some form because it defines `g`, `W1`, `m` for the next
  cell; could be merged with the next.
- `mdl-matrix-calculus-backward-rules-of-a-layer-1` (MC-16): confirmation
  only; its value is the three zeros of the inactive unit. Candidate to merge
  with the previous cell.
- `matrix-calculus-autodiff-softmax-jacobian` (MC-16): four confirmations.
  Candidate to cut down to the fused-loss gradient, or to replace by
  something that can surprise (the eigenvalues, or the gradient for a target
  that does not sum to one).

## Other deferred findings

- Exercises 4(a), 4(b), 5(c) (review: trim the part the text already does;
  the multiplication count is asked twice): left, since the brief limits
  exercise edits to errors. Note that the body now points to exercise 4 for
  the factorization loss.
- MC-14, letter clash: the output width is now $q$ (as in the main text) in
  the body. Exercise 1 still uses $m$ for an output dimension and the mask is
  $\mathbf m$; the Jacobian definition keeps $m$ for the number of outputs.
- The MIT problem numbers in the attributions: not checked.
- `sec_backprop` in the main text still points to
  `sec_mdl-matrix-calculus-autodiff` for the layer rules; it should point to
  `subsec_mdl-layer-vjp` (main-text file, not edited).

## Claims added or changed (all checked)

- The hidden layer of the 2-2-1 network is affine near $(1,1)$
  (pre-activations $(3,-2)$, mask fixed).
- $\|\mathbf W^K\|$: long-run rate only; transient growth for non-normal
  $\mathbf W$; singular values for products of different factors (see MC-7).
- $d\varphi(\mathbf X)=\varphi'(\mathbf X)\odot d\mathbf X$ and
  $\langle\mathbf A,\mathbf B\odot\mathbf C\rangle=\langle\mathbf
  A\odot\mathbf B,\mathbf C\rangle$.
- The $1\times1$ check detects coefficients and missing terms, not transposes
  or order; the shape rule fails for square factors.
- "The transpose of a linear map is also called its adjoint."
- $\mathbf X^\top\bar{\mathbf Y}=\sum_i\mathbf x_i\bar{\mathbf y}_i^\top$
  (rows of $\mathbf X$, $\bar{\mathbf Y}$ as columns).
- Minibatch bias: $\mathbf Z=\mathbf X\mathbf W+\mathbf 1\mathbf b^\top$,
  $\bar{\mathbf b}=\bar{\mathbf Z}^\top\mathbf 1$ (checked numerically; agrees
  with exercise 5 and with `sec_backprop`).
- The softmax Jacobian is the covariance matrix of a one-hot vector drawn with
  probabilities $\mathbf p$.
- What the random-point cell can and cannot detect (see MC-16).
- "the layout of the main text" for $\mathbf Y=\mathbf X\mathbf W$ (checked
  in `backprop.md`, `eq_backprop-matmul`), and that `sec_backprop` derives the
  rule by indices.
- Numbers kept that no cell prints, recomputed: singular values 1.706 and
  0.329; eigenvalues 0, 0.079, 0.378; $\mathbf p=(0.254,0.057,0.690)$;
  $0.9^{50}\approx0.005$, $1.1^{50}\approx117$. Not recomputed by me (taken
  from the review, which did): the gaps 0.061 and 0.015 and the count
  $2.001\times10^9$ (the latter is arithmetic in the text).

## Slides that now disagree with the body

- Slide title "The 1×1 Check Catches Shapes, Not Transposes": the body now
  says the check catches coefficients and missing terms, not shapes (MC-11).
- Slide "Equal Factors Give W^K": "grows or shrinks like
  $|\lambda_{\max}|^K$" without "for large $K$".
- Backward-rule slide: $\mathbf W\in\mathbb R^{d\times m}$, "forward $ndm$,
  backward $2ndm$"; the body uses $q$.
- Cover kicker "§25.3" is stale numbering (rewritten by `gen_slides.py`).
