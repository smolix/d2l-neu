# Writing pass log: Automatic Differentiation

Source: `chapter_mdl-calculus/mdl-matrix-calculus-autodiff.md`. Review:
`docs/reviews/mdl-writing-2026-11/39-calculus/mdl-matrix-calculus-autodiff.md`.
Not split. No code cell, cell ID, figure path or slide changed.

## Words

Prose words (checker count, body and exercises): 6902 before, 6824 after
(-78). The cuts ("Two Orders" recount, "Parameter Gradients" derivation and
example, the third Hessian derivation, the "Two checks" paragraph, the "We
can now" recap, confirmations after cells) were largely offset by additions
the review asked for (AD-1, AD-5, AD-8, AD-13), the ranking sentence in the
opening, and the citations of owners; I did not measure the two separately. Display equations: 17 before, 15 after.

Checker: `OK: only prose changed`, no em-dash warning. `lint_source.py`: clean.

## Correctness changes

- **AD-7.** Said: "The backward sweep visits the nodes in the order
  $y,r,t,u,v$." Now: "in an order such as $y,r,t,u,v$, in which every node
  comes after all the nodes that use it (`backprop` happens to take $v$
  before $u$, which is just as good)". Verified by tracing the depth-first
  search by hand: `order = [u, v, t, r, y]`, reversed $y,r,t,v,u$. (No Python
  venv on this machine, so not run; the reviewer ran it.)
- **AD-5.** Added a paragraph after the measured count that reconciles 2
  (measured), 3 (opening), and the Baur and Strassen bound. Recounted by hand
  for $f=(\sum x_j^2)^2$, $n=200$: forward pass 201 multiplications and 200
  additions; backward sweep 201 product closures (2 multiplications and 2
  accumulating additions each) and 200 addition closures (2 additions each),
  so 402 multiplications and 802 additions, 1204 arithmetic operations.
  Multiplications: (201+402)/201 = 3. All operations: (401+1204)/401 = 4.0.
  The reviewer's proposed sentence ("in arithmetic the sweep costs about
  twice the forward pass") is true only for multiplications; the text now
  says both. "Parameter Gradients" now says "counted in multiplications".
- **Baur and Strassen constant (review: could not check).** Said "at most
  five times as many operations". Now "at most a fixed multiple ... between
  three and five depending on which operations are counted". From memory:
  3 for multiplications and divisions (the original theorem), 4 to 5 in
  later statements that count all operations. **Not checked against the
  paper**; the author should look once.
- **MVC-12 collision (adjoint written as a partial derivative).** Said:
  "The *adjoint* $\bar v=\partial y/\partial v$ is the rate ...". Now: "The
  adjoint $\bar v$ of `eq_mdl-adjoint` is the total derivative of the output
  $y$ with respect to $v$, summed over all paths". The caption of
  `fig_mdl-cal-fwd-vs-rev` changed the same way ($\bar v=\partial\ell/
  \partial v$ became words; $\dot v=(\partial v/\partial x)\dot x$ became
  $\dot v$). Caption text only, the include path is unchanged. The figure
  itself (SVG) may still print the old formulas: check on the development
  machine.
- **AD-12** (partial product called a row): gone with AD-10.
- **AD-14(a).** "Everything a training loop needs is built by composing
  them" became "Curvature along a direction, derivatives of solutions, and
  full Jacobians when they are needed are all built by composing them."
- **AD-6.** Wengert sentence now matches `sec_autograd` (list introduced
  for forward mode; reverse mode Linnainmaa).
- **Stale pointer.** Adjoint method: `sec_mdl-odes-solvers` became
  `sec_mdl-adjoint-method` (label is in `mdl-neural-odes.md`).
- **Summary.** Added "provided the solver has converged" to the implicit
  differentiation clause (the body shows the formula is wrong before
  convergence).
- **Exercises.** Mixed mode: reference changed to `eq_mdl-jacobian-product`.
  Forward gradients: prose title replaced by
  `:citet:`Baydin.Pearlmutter.Syme.ea.2022``; "with `d2l.plot`" removed (the
  section does not import `d2l`).

No finding disagreed with. Stored PyTorch outputs compared with the prose
again: all quoted numbers match.

Numbers quoted that no cell prints, kept because recomputed: 402, 201, 1204
(above); $M=401,85,50,40,104,401$; $\mathbf H\mathbf v$ numbers (the cell
prints them); $3\times10^6$ and $2\times10^9$ (cited from the matrix calculus
section, which derives them).

## Six before/after pairs

1. Before: "Does one run on $1.3+\varepsilon$ return both $f(1.3)$ and
   $f'(1.3)$ for $f(x)=\sin(x^2)+e^x$? We compare the $\varepsilon$-part with
   the derivative $f'(x)=2x\cos(x^2)+e^x$ computed by hand."
   After: "A first run, on $f(x)=\sin(x^2)+e^x$ at $1.3+\varepsilon$, with
   the hand derivative $f'(x)=2x\cos(x^2)+e^x$ printed beside the
   $\varepsilon$-part:"
2. Before: "Each pass returns the value $2.371539$ and one partial
   derivative, and the three partials are the softmax entries ..."
   After: "It needs three. Each pass returns the value $2.371539$ again and
   a single softmax entry, $0.253716$, $0.056612$, or $0.689672$."
3. Before: "Does the tape reproduce the table? We record the expression with
   `Var` and run one backward sweep."
   After: "Recorded with `Var` and swept once, the same expression gives:"
4. Before: "We compute the gradient in both modes and compare the operation
   counts."
   After: "Forward mode will need $200$ passes of $401$ operations. It is
   worth guessing the count for reverse mode before reading the output."
5. Before: "Does :eqref:`eq_mdl-implicit-grad` agree with a finite difference
   of the solver's output on a nonlinear $2\times2$ system?"
   After: "We try :eqref:`eq_mdl-implicit-grad` on a nonlinear $2\times2$
   system, ..., solved by twenty Newton steps, and compare it with a central
   difference of the solver's output."
6. Before: "We can now write both modes of automatic differentiation in a few
   lines, trace a tape by hand, ..." (recap paragraph).
   After: deleted; the body ends "One problem stays open. Automatic
   differentiation evaluates the product $\mathbf J_K\cdots\mathbf J_1$ to
   rounding error, but it cannot change what the product is ... An exact
   gradient that is $10^{-20}$ is of little use."

Questions kept (outcome not obvious): "How many runs does forward mode need
for this gradient?", "Can one pass return the whole matrix?", "Can we compute
$\mathbf H\mathbf v$ without $\mathbf H$?", the unconverged solver.

## Repetition

Removed or reduced to a citation:

- AD-3: the product of Jacobians is now cited from `eq_mdl-jacobian-product`
  with the count $3\times10^6$ against $2\times10^9$ and
  `fig_mdl-cal-jacobian-chain-order`; the second count ($2\times10^6$ against
  $10^9$) is gone.
- AD-10: "Parameter Gradients" is now the rule cited from
  `subsec_mdl-layer-vjp`, the sentence "a weight matters in proportion ...",
  and the cost paragraph with the 30 ms. The perturbation derivation, the
  minibatch remark and the numerical example are gone.
- AD-15: third derivation of the log-sum-exp Hessian and the "Two checks"
  paragraph are gone; `eq_mdl-softmax-jacobian` is cited. The covariance
  reading is kept as one parenthetical clause (it is new; its owner is the
  matrix calculus section, which should state it where the variance
  argument is made).
- AD-2: adjoint and tangent no longer defined a second time; $\dot v$ is
  introduced with `Dual`, the adjoint cites `eq_mdl-adjoint`.
- $\mathbf W^K$ in the closing: one clause with a reference to
  `sec_mdl-matrix-calculus` and no $|\lambda_{\max}|^K$.
- AD-9: tracer described once in shared prose in the section's own terms;
  the four tabs reduced to the API fact.
- AD-14: prose around the framework JVP/VJP cell reduced, with a reference
  to `subsec_autograd_fwd_rev` for the calls.
- AD-4: sentence deleted. AD-11: gone with AD-10.
- Opening: the paragraph retelling "chain rule read as matrix algebra" is
  now one sentence.
- Summary: outer-product clause removed.

Kept because this file owns them: the opening cost of probing (12 days
against 30 ms) and its payoff; seeds, JVP and VJP as programs
(`eq_mdl-ad-two-orders`); the cost rule in sweeps; both engines; the measured
count; checkpointing; implicit differentiation.

## Ranking; Proposition blocks; labels

- The section has one Proposition (dual numbers); kept. The opening now says
  which two results matter most (the dual-number proposition and the cost
  rule) and that "Composing Jacobian Products" can wait for a second reading.
- No labels added, none removed. Because labels may not be removed, three
  displays stay only to carry a label: `eq_mdl-ad-jacobian-product` (now
  unreferenced anywhere; delete display and label together on the
  development machine), `eq_mdl-ad-param-grad` (unreferenced), and
  `eq_mdl-ad-lse-hessian` (referenced after the cell). The heading
  "Parameter Gradients" stays for `subsec_mdl-param-gradients`; the review
  suggests merging it into "Cost of the Two Modes".

## Split markers

None (not to be split).

## Cells to cut or redesign on the development machine

- `mdl-matrix-calculus-autodiff-jacobian-vector-and-vector-jacobian-products`
  (AD-14): prints the plane-map Jacobian a third time and repeats the calls
  of `sec_autograd`. Cut, or cut the one-pass `Dual` printout instead.
- `matrix-calculus-autodiff-tape-check` (AD-16): the framework confirms
  $16,-16$; prose reduced to one sentence.
- `matrix-calculus-autodiff-dual-eval` (AD-16): confirmation of a proved
  proposition; fine once, prose reduced.
- `matrix-calculus-autodiff-tape` (AD-8): give `Var` an `exp` and a `log`
  and run log-sum-exp on the tape (one sweep, three softmax entries). The
  prose now says why the example changes to a polynomial.

## Other deferred findings

- Structural 3: two sections titled "Automatic Differentiation". Title not
  changed (slides cover and table of contents; author's choice).
- Exercises "A tensor tape" (a) and (d) repeat the previous section (trim);
  an exercise adding `exp`/`log` to `Var`. Not errors, left.
- AD-13 second half: `sec_backprop` should point to
  `subsec_mdl-checkpointing` (main-text file).
- `sec_backprop` cites this section for the layer rules; they are in
  `subsec_mdl-layer-vjp` (main-text file).
- TensorFlow, JAX, MXNet stores not compared.

## Claims added or changed

- A program with loops and branches executes one sequence of operations per
  input, which the dual run differentiates (AD-1; cites
  `subsec_autograd_control_flow`).
- Baur and Strassen: "between three and five depending on which operations
  are counted" (unverified against the source, see above).
- The backward sweep of the measured example does 402 multiplications
  against 201, and 1204 operations when additions are counted; factors 3
  and 4 (recomputed).
- A tracer forms the forward-mode rule of each operation and reverses the
  sequence with each linear map transposed (JAX: `linearize` then
  transpose).
- "Frameworks offer the same trade for the blocks of a deep network: a
  checkpointed block keeps only its input and is run a second time when the
  backward sweep reaches it" (`torch.utils.checkpoint`, `jax.checkpoint`).
- Closing: the product "can shrink or grow geometrically" over many steps
  (weaker than the old "like $|\lambda_{\max}|^K$", in line with MC-7); "An
  exact gradient that is $10^{-20}$ is of little use" (a judgment; cut if
  too strong); remedies "begin in `sec_numerical_stability`".
- "Where a framework has both modes, their composition is one line" (the
  MXNet tab explains its exception).

## Notation

$\partial$ is now used only for derivatives of a function of its listed
inputs and for edge derivatives; $\bar v$ is defined in words as the total
derivative. Remaining: "$\partial y/\partial u=2(uv+u)(v+1)$" in the hand
trace, now introduced as differentiating $y$ "as a function of $u$ and $v$".

## Slides that now disagree with the body

- "Adjoint $\bar v=\partial y/\partial v$" (tape slide): the old notation.
- "Sweep $y,r,t,u,v$": stated as the order, not as one valid order.
- Slide "One Outer Product per Layer Gives Every Weight Gradient": the body
  now cites this rule from the previous section instead of deriving it.
- Any slide that repeats "at most five times" or the $2\times10^6$ against
  $10^9$ count (none found by grep).
