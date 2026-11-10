# Writing pass log: `chapter_mdl-dynamics/mdl-neural-odes.md`

Prose words (checker count, exercises included): 6689 before, 6797 after
(+108, 1.6%). The section did not get shorter. The cuts (two of three
statements of the gains, three of four statements of the discrete-adjoint
conclusion, the recap, the tab-bridging sentence, two roadmaps, the "in
words" restatement of the scalar adjoint) removed about 420 words; the
additions the assignment and review asked for (NODE-14, NODE-S2, NODE-10,
the tie to `eq_mdl-adjoint`, ranking, NODE-6, NODE-S4, NODE-17) added about
530. Checker: "OK: only prose changed", no em-dash warning. Lint: one
warning, in the slides region (list item starting with a cross-reference),
present before this pass and not touched.

Numeric checks were done with `python3` (NumPy only) in scratch files, or by
hand where stated. No PyTorch run was possible.

## Correctness changes

| Finding | Before | Now | Verification |
|---|---|---|---|
| NODE-1 | "The previous section ended with two observations ... a residual block is one forward-Euler step of size $h=1$" | Opens on the question; cites `sec_mdl-euler-runge-kutta` and `sec_resnet` | The previous section does not write the block; labels exist |
| NODE-3 | "At $d=3072$ that is the cost we set out to avoid." | "The trace has already replaced an $O(d^3)$ determinant by $d$ passes. A random projection replaces the $d$ passes by one, and returns an unbiased but noisy estimate" | By hand: the avoided cost was the determinant ($\tfrac23 d^3\approx1.9\times10^{10}$) |
| NODE-4 | Caption: "steps of size $h=1$ on the exact flow of the field"; text cites the figure for blocks with different weights | Caption: blocks "share one field"; steps "beside the exact flow of the same field, from which they drift"; text says the figure "draws the case in which all blocks share one field" | Generator docstring (`tools/gen_mdl_dynamics_figures.py`: Euler polygon overlaid on RK4 flow, every block labelled $\mathbf f_\theta$). Figure not re-rendered by me; the reviewer rendered it. Only the caption text changed, not the include path |
| NODE-7 | "Can ten shared-weight residual updates learn a prescribed deformation"; "$400$ full-batch iterations ... for the last $100$" | Task called "a stretch and a shift"; the affine field $\dot{\mathbf x}=\mathrm{diag}(\ln1.2,\ln0.4)\mathbf x+\mathbf c$ fits it without error; "$401$ ... lowered to $0.005$ from iteration $300$" | By hand: $\mathbf x(1)=e^{\mathbf D}\mathbf x_0+\mathbf D^{-1}(e^{\mathbf D}-\mathbf I)\mathbf c$, $e^{\mathbf D}=\mathrm{diag}(1.2,0.4)$, shift $(1.5,0)$ needs $c_1=1.5\ln1.2/0.2$, $c_2=0$. Loop is `range(401)`, milestone 300 |
| NODE-10 | Identification $\mathbf a=\partial\mathcal L/\partial\mathbf x(s)$ in a closing parenthesis | Own sentences before the proof mark: $\delta\boldsymbol\theta=\mathbf 0$, nudge at $s$, pairing constant on $[s,T]$ | From the displayed derivative of the pairing with $\delta\boldsymbol\theta=\mathbf 0$. The proposition now says $\mathbf f(\mathbf x,\boldsymbol\theta,t)$ is $\mathbf f_{\boldsymbol\theta}$ with parameters as an argument |
| NODE-12 | Scalar drift example, then a three-sentence parenthesis on why a lone scalar escapes | Example set from the start in a state with coordinates of size one and the fast direction off the coordinate axes | $10^{-16}e^{50T}$ = $2.7\times10^{-11}$, $7.2\times10^{-6}$, $5.2\times10^{5}$ (python3). "Not along a coordinate axis" agrees with exercise 4b (diagonal matrix reconstructs accurately) |
| NODE-13 | "This is Jacobi's formula"; "as Jacobi's formula says they must"; exercise 9 "..., Jacobi's formula" | "This identity"; "as :eqref:`eq_mdl-ode-jacobi` says they must"; exercise 9 keeps the name only for the derivative formula | Terminology. The label `eq_mdl-ode-jacobi` is unchanged |
| NODE-14 | Noise stated, no resolution | See "claims added" below | python3, below |
| NODE-15 | "and to better than $10^{-8}$ for any draw of the weights"; "for any draw" | "agree to about $10^{-10}$, the accuracy of the solver and of the finite differences" | 150 draws (python3): median $5\times10^{-11}$, max $3\times10^{-10}$; stored $6.6\times10^{-11}$ |
| Exercise 1c | "this error settles near $0.03$" | "at a few hundredths" | Reviewer's wording; not re-run (needs training) |

No finding disagreed with. NODE-20 (step-index conventions $n$, $N$, $k$,
$l$; `H`, `B` in code) not changed: shared with the previous section and
with code.

## Six before/after pairs

1. Before: "Can ten shared-weight residual updates learn a prescribed
   deformation of the plane?" After: "The task is deliberately easy: carry
   $64$ points of the unit circle to the matching points of the ellipse ...
   a stretch and a shift. ... The question is whether training finds a flow
   or only a ten-step program."
2. Before: "Is what we trained a continuous flow or a ten-step program? Two
   tests answer this." After: "Two tests tell a flow from a ten-step
   program. ... Before running the cell, guess what this does to the
   endpoint error."
3. Before: "Does a hand-written backward sweep through the unrolled Euler
   program give the exact gradient of that program? The next cell compares
   it with central finite differences" After: "The cell writes the backward
   sweep :eqref:`eq_mdl-ode-discrete-adjoint` by hand and tests it against
   central finite differences on the same program"
4. Before: "How far is that gradient from the gradient of the exact flow?
   The next cell evaluates ..." After: "... so the sweep is coded correctly.
   The distance that matters is the one to the gradient of the exact flow."
5. Before: "Can we estimate the trace from vector--Jacobian products alone,
   without ever forming $\mathbf J$?" After: "The proposition says that the
   average over probes converges to the trace. It does not say how fast, and
   that is what we measure."
6. Before: "The step that is not routine is to apply uniqueness backward,
   from the meeting time. Thus no Neural ODE ..." After: "The proof has one
   idea, to apply uniqueness backward from the meeting time. As a
   consequence, no Neural ODE ..."

## Repetition

Removed: the gains listed in "From Residual Blocks" ("Why pass to the limit
at all?"), now once in the opening together with the costs (NODE-S1; the
review preferred the subsection as home, I kept the opening because the
costs belong beside them); the $2\times10^{10}$ count in the opening (kept
where the formula is, NODE-2); three restatements of "exact for the program,
first order from the flow" (NODE-11; stated once after the two recursions);
the "We can now" recap (NODE-18); $5\times10^5$ and the spiral's $e^t$ in the
Summary; the tab-bridging sentence (NODE-16); "In words, $a(t)$ answers ...".

Kept because this file owns them: the adjoint method (tied in one clause to
`eq_mdl-adjoint`: "the adjoint $\bar a$ ..., the total derivative of the
output with respect to a node, with one node $\mathbf x(t)$ for every
instant"), discrete against continuous adjoint, the instantaneous change of
variables, Hutchinson.

## Ranking, demotions, labels, splits

No Proposition demoted (order preservation, adjoint equations, change of
variables and Hutchinson are each cited or central; the Lemma is used by
exercise 9). No labels added, none removed. Not split, no markers. The
opening now names the three results to remember, says that
`sec_mdl-continuous-normalizing-flows` can be read alone, and marks order
preservation and "Memory and Drift" as skippable.

## Chapter 17 citations (checked with git grep)

`design-space.md` cites `sec_mdl-continuous-normalizing-flows` (for the rate
$-\nabla\cdot\mathbf u_t$), `eq_mdl-ode-hutchinson` and
`ex_mdl-odes-solvers-hutchinson-variance` (for the one-probe variance). All
three passages are intact; the body still states the Rademacher variance,
and the CNF part now says how a likelihood is evaluated (backward from the
point) with a reference to `sec_diffusion-design-space`. Other inbound
citations (`sec_mdl-adjoint-method`, `subsec_mdl-flow-order`,
`sec_mdl-neural-odes-cnf`) unaffected.

## Cells to cut or redesign on the development machine

- `#odes-solvers-cnf-trace` (NODE-19, overview decision 1): confirms an
  exact identity for a linear field; cannot fail. Prose now says so in a
  clause. Candidate to cut; the nonlinear cell is the test.
- `#odes-solvers-adjoint-check` (NODE-11): a code test of the chain rule.
  Presented now as "the sweep is coded correctly". Could be merged into the
  next cell.
- `#mdl-neural-odes-the-hutchinson-trace-estimator-2` (NODE-14): print the
  single-probe standard deviation; add a contracting field (for instance
  $\mathbf f-\mathbf x$) beside the random one; possibly show one probe per
  step along a trajectory.
- `#odes-solvers-neural-ode-train` (NODE-7): a non-affine target, or a run
  trained with a finer solver to show the step count becoming free.

Other deferred: NODE-20 notation; exercise-label prefixes
(`ex_mdl-odes-solvers-*`, cited by chapter 17); a back-reference from the
flow-matching section and an Euler remark in `resnet.md` (other files).

## Claims added or changed

- A flow of an affine field maps the circle to the ellipse exactly (by
  hand, above).
- "for Euler they differ at first order in $h$" (scalar closed form; gaps
  $1.27$, $0.145$, $0.0148$ recomputed with python3).
- For the step count to be free the field must be fitted to the flow, by a
  finer or higher-order solver or by a solver-free regression as in the
  score-matching and flow-matching sections (NODE-S2; not an experiment).
- Likelihood evaluation runs backward from the point; chapter 17 does this
  (`pf_nll` in `design-space.md`, read).
- Hutchinson (python3, 200 draws of the cell's field): one-probe standard
  deviation $4.0$ to $5.5$ ("about $4$ to $5$", kept; stored
  $0.0154\sqrt{10^5}=4.9$); $|\mathrm{tr}|$ median $0.46$, 10% to 90% range
  $0.09$ to $1.07$ ("a few tenths"); standard error at $10^5$ probes
  $0.013$ to $0.017$ ("about $0.015$"). Dropped: "$2\times10^5$ probes for a
  standard error of $0.01$" (true, $1.6$ to $3.0\times10^5$, but no longer
  needed) and "the relative error grows with $d$".
- "Subtracting $\mathbf x$ from the field ... would move the trace to about
  $-50$ and leave the noise unchanged, a relative error of a tenth": exact
  from the variance formula (off-diagonal entries only) and $4.9/50$.
- One probe suffices in training: unbiased and linear in the
  log-likelihood, so the gradient is unbiased; noise averaged over
  minibatches and iterations. With a fresh probe at each of $N$ steps the
  noise of the trace integral falls like $1/\sqrt N$: simulated with a
  constant Jacobian, $T=1$, 2000 repetitions: sd $1.59$, $0.52$, $0.165$ at
  $N=10$, $100$, $1000$ against the prediction $\sigma/\sqrt N$ = $1.65$,
  $0.52$, $0.165$. Note for the author: FFJORD draws one probe per solve,
  not per step; unbiasedness holds either way, the $1/\sqrt N$ clause is
  conditional ("with a fresh probe at each ... step").

## Slides now disagreeing with the body

- The slide on the spiral writes "Jacobi:" for $\det e^{\mathbf At}$; the
  body no longer uses that name for it.
- The Hutchinson slide and the summary slide present the estimator without
  the noise caveat or the reason one probe suffices.
- The summary slide repeats $5\times10^5$, which the Summary dropped (the
  body still has it).
