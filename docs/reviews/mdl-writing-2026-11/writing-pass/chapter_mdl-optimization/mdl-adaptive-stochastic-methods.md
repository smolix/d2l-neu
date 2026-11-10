# Writing pass log: Adaptive Methods

Source: `chapter_mdl-optimization/mdl-adaptive-stochastic-methods.md`. Prose
only; no cell, cell ID, figure include, label or slide changed. Not split.

## Words

Checker (`check_mdl_prose.py`, body plus exercises): 7,579 before, 7,630
after (+51). Body before "Exercises", whitespace tokens: 6,459 before, 6,478
after; exercises +14. The section did not get shorter. About 600 words of
duplication and recap were removed (below), and about the same amount was
added as repairs the review asked for: the tuning caveats of ADA-1/12/13, the
reframed warmup subsection (ADA-22/23), and the hypothesis sentences of
ADA-3, 9, 11, 18 and 20. Two displays became inline formulas (the unrolled
$\mathbf{v}_t$ and the SGD decay identity), so the display count went from 16
to 14.

## Correctness changes

All numbers below were recomputed with a copy of the cell's `run`/`adam` code
in a scratch file (NumPy, outside the repository).

- **ADA-1, ADA-12 (S1).** Was: Adam "needs 48" (opening) and "six times fewer
  than the tuned heavy ball's 296, which uses no scaling at all". Now: the
  opening says the 48 is at an untuned $\eta = 0.1$, gives 54 at ratio
  $10^5$, and says the count changes severalfold with $\eta$ and the starting
  distance; a paragraph before the table says gradient descent and heavy ball
  are tuned and Adam is not, with 166 steps at $\eta = 0.03$ and 14 at
  $\eta = 0.3$; under the table the 48 is explicitly not compared with 296.
  The scale-invariance subsection adds that the count depends on the start in
  units of $\eta$ (4,145 steps from $(100, 100)$). Verified: 166 / 48 / 14 /
  1 steps at $\eta = 0.03, 0.1, 0.3, 1$; 4,145 from $(100,100)$; 40 / 48 / 54
  at ratios $1, 10^3, 10^5$.
- **ADA-13 (S2).** Was: 7,908 "more than gradient descent", then "Retuning
  ... about 620", "at any step size in that range" with no range. Now: the
  7,908 is tied to the table's $\eta = 0.01$, the range
  $\{0.003, \ldots, 3\}$ is stated, the best is "about 620". Verified:
  11,589 / 7,908 / 4,819 / 2,412 / 1,235 / 622 / 620 for $\beta_1 = 0.9$;
  never, at all seven steps, for $\beta_1 = 0$.
- **ADA-22, ADA-7, ADA-23 (S3).** Was: "The preconditioner must act before it
  has data", and a promise in the bias-correction subsection that warmup
  "returns to these first steps". Now: the subsection opens on the size of
  the first step, says the preconditioner is the tempting suspect and that
  the experiment does not support it, states that the comparison with
  $0.23\,\eta$ is for a noise coordinate (a consistent gradient has
  stationary update $\eta$), concludes momentum, says the equal-scale
  experiment cannot see the preconditioner's effect at $t = 1$, and names
  curvature (ceiling $2/\lambda_{\max}$, `subsec_mdl-quadratic-model`) as the
  mechanism behind long warmups. The bias-correction sentence now says what
  the low bias of $\sqrt{\hat v_1}$ does (first step exactly $\eta$).
  Checked against the stored output (1.00, 0.71, 0.58, 0.33, 0.24, 0.23 on
  both curves).
- **ADA-5.** Robbins-Monro detour removed; the paragraph now uses the
  reviewer's draft.
- **ADA-9.** "the step is much smaller than $\eta$" is now "the iterate moves
  by less than $\eta$ and contracts geometrically, by the factor
  $1 - \eta/\sqrt{\hat v_t}$" (0.9 at $t=1$, 0.67 at $t=48$, recomputed).
- **ADA-11.** "Sign step" is defined once (three cases: $t = 1$;
  $\beta_1 = \beta_2 = 0$; $\beta_1 = 0$ with constant gradient magnitude
  over the memory of $\hat{\mathbf v}$). "Adam's sign-like normalization" in
  the Muon paragraph is now "a per-coordinate normalization".
- **ADA-17.** I could not consult the paper. Removed the two attributions I
  could not confirm (bias correction and constant step; "for every
  $\epsilon > 0$"). Kept, from memory of the paper's Theorems 2 and 3: for
  every fixed $\beta_1 < \sqrt{\beta_2}$ there is an online convex sequence
  and a stochastic convex problem on which Adam does not converge. Please
  check this sentence against the paper.
- **ADA-18.** Added why a period costs $2C - 4$ (Adam at least $C - 2$,
  $x^\star$ exactly $-(C-2)$), and a clause that projection on an interval is
  clipping, with a forward reference.
- **ADA-20.** After the proof: the proposition does not claim existence or
  convergence; and from $1 - \lambda|w_i| = \epsilon/(|g_i| + \epsilon)$ the
  dichotomy holds at $\epsilon = 10^{-8}$, not only in the limit. The example
  now says "the fixed point is" instead of "settles at". Checked the four
  values 2.4, 3.0, 1.5, 1.0 by hand and uniqueness of the AdamW solutions.
- **ADA-21.** The parenthesis reporting an unprinted cosine-decay run is
  removed.
- **ADA-24.** Gauss-Newton is now "a positive semidefinite approximation of
  the Hessian that drops the term containing the second derivatives of the
  network".
- **ADA-25.** "Every method" is now "Every preconditioner"; the independence
  remark is reworded without the unsupported assertion.
- **ADA-27.** "All of these results assume" is now "the positive results of
  this section all assume".
- **Zipf ratio (not a numbered finding).** Body and exercise 2(c) said 0.64;
  $\sum_i\sqrt{p_i}/\sqrt d = 0.6346$ at $d = 10^4$, $\alpha = 1$, and the
  review's own exercise check gives 0.63. Both now say 0.63.
- **ADA-14 / exercise 8(b).** The text keeps the explanation of
  $250\,\eta^2$ (a protected page) and marks the unprinted facts as
  observations of the run; the exercise now asks to compare the printed $f$
  with $250\,\eta^2$ instead of explaining it.
- **Exercise 10(a).** Forward reference to
  `sec_mdl-constrained-optimization-duality` added for KKT.

No finding disagreed with.

Numbers kept that no cell prints, all recomputed: 40/48/54; 166 and 14;
4,145; about 620 and the "never" over seven step sizes; contraction factors
0.9 and 0.67; minimum $6 \times 10^{-5}$ near $t = 2{,}600$ (6.07e-5 at
2,620); final $f = 2.5$; peak 6.6 at $t = 12$ and 3.16; the periodic values
15.12, 1.83, 1.05 and displacements; Zipf 0.63. Not recomputed and kept as it
was: "their $\hat v$ agree to within one percent" (the reviewer confirmed
it).

## Six before/after pairs

1. Before: "An optimizer that gives each coordinate its own step size,
   estimated only from the gradients it has seen, needs 48."
   After: "... needs 48 at a step $\eta = 0.1$ that we did not tune, and 54
   when the ratio is $10^5$. The 48 is not a figure of merit, since it
   changes severalfold with $\eta$ and with the starting distance. What it
   hardly depends on is the curvature ratio."
2. Before: "The next cell runs each method on the aligned and on the rotated
   valley and counts its steps."
   After: "Before reading the counts, note what is tuned. Gradient descent
   and heavy ball run at their optimal settings. Adam's step sizes are not
   tuned: ..."
3. Before: "Does the stiff coordinate receive a thousandfold smaller step,
   and do the two coordinates run the same iteration, as the proposition
   predicts? The next cell runs scaling-only Adam ..."
   After: "Back on the aligned valley, the proposition predicts that the two
   coordinates carry the same iterate and that the stiff one receives a step
   a thousand times smaller. We print the iterate and the effective steps
   ..."
4. Before: "Where do Adam, AMSGrad and SGD end on this sequence? The cell
   runs all three with the same $1/\sqrt{t}$ decay from $x = 0$ ..."
   After: "So Adam should end at $+1$ and the other two at $-1$. We run all
   three with the same $1/\sqrt{t}$ decay from $x = 0$, the middle of the
   interval, so that Adam is not handed its bad endpoint."
5. Before: "The preconditioner must act before it has data."
   After: "Adam's first update has size $\eta$ in every coordinate ... It is
   tempting to blame the preconditioner, which at $t = 1$ rests on one
   gradient. The experiment below does not support that reading."
6. Before: "We can now derive a coordinate's step from its gradient history,
   prove that the result ignores the coordinate's scale, read off the fixed
   points ..."
   After (end of the body): "And the positive results of this section all
   assume curvature aligned with the coordinates, or with the Kronecker
   structure of a layer. The curvature of a deep network is neither diagonal
   nor fixed, so why these methods work as well as they do on deep networks
   is an open question."

## Repetition

Removed or reduced:
- Opening paragraphs 3 and 4 (outcomes of the rotated valley and the
  counterexample, "three further consequences"): replaced by a route and a
  ranking. "48" now occurs in the opening, once in scale invariance, and at
  the table; it is out of the Summary.
- The "We can now ..." recap and the pointer to the next section. The open
  question is the ending. Warmup, which the Summary lacked, is now in it.
- Bias correction (chapter table item 7): motivation paragraph cut, the
  unrolling made inline, the $1\%$ cited to `sec_adam`. The display
  `eq_mdl-opt-bias-correction` stays because exercise 3 cites it. What is
  owned here, the peak 6.6 at $t = 12$, is foregrounded.
- Coupled against decoupled (item 8): the SGD identity is inline with a
  citation of `eq_l2-is-decay`; the "large gradients barely decayed"
  sentence and the Loshchilov-Hutter attribution are gone; the worked rates
  $10^{-5}$ and $10^{-3}$ moved to the reading of the cell that prints them.
- Heavy-ball comparison in the Adam subsection: one sentence.
- "K-FAC, Shampoo and Muon" (item 10, S6): about 60 words shorter; the
  two-sided form and the polar identity kept; Muon as an optimizer is sent to
  `sec_muon` and Newton-Schulz to `subsec_mdl-orthogonalized-updates`. The
  word "ladder" is back in one sentence, since `muon.md` cites
  `subsec_mdl-preconditioning-ladder` for it.
- The explanation of the square root as balancing the regret terms: one
  sentence pointing to exercise 7(c).

Kept because this file owns them: metric and regret views, scale invariance,
the rotated valley, the counterexample read in one period, the decay
experiment, the fixed-point proposition, the Kronecker identity. The three
update equations stay (small, and the propositions are about them), each now
tied once to the main-text label (`eq_adagrad`, `eq_adam-moments`,
`eq_adam-update`, `eq_l2-is-decay`).

The framework tab notes moved from after the decay cell to directly after
`eq_mdl-opt-adamw-decoupled` (review, "Code"); no cell moved.

## Ranking; propositions; labels

The opening names the two results that matter most (scale invariance, AdamW
fixed points) and what can be skipped on a first reading (the regret bounds,
structured preconditioners). No Proposition block was demoted: the three are
the results other text cites. No label added or removed. No split markers.

## For the main text (files I may not edit)

- `chapter_optimization/lr-scheduler.md`, near line 544, says of adaptive
  optimizers: "their per-coordinate step sizes are estimated from the few
  gradients seen so far, and those estimates are unreliable in the first
  steps (:numref:`subsec_mdl-schedules-warmup`)". The cited experiment
  refutes this. Sentence to use instead: "Adaptive optimizers such as Adam
  (:numref:`sec_adam`) have a second, smaller reason to warm up: Adam's
  first update has size $\eta$ in every coordinate, about four times its
  stationary size on a coordinate whose gradient is noise, until the
  momentum average has filled after about ten steps
  (:numref:`subsec_mdl-schedules-warmup`)."
- `chapter_optimization/muon.md`, lines 177 to 230 ("Odd Matrix Polynomials",
  "The Tuned Quintic"): the convergence analysis of Newton-Schulz (odd
  polynomials act on singular values; the cubic's slope 1.5 at zero; the
  basin) is owned by the appendix, in `subsec_mdl-orthogonalized-updates` of
  `chapter_mdl-linear-algebra/mdl-svd-low-rank.md`, not by this file. The
  main text should keep the tuned quintic and the implementation, state the
  odd-polynomial fact in a sentence, and cite
  `subsec_mdl-orthogonalized-updates` for the analysis. Its two citations of
  `subsec_mdl-preconditioning-ladder` (lines 141 and 925) are now accurate
  again: the subsection states the form
  $\mathbf{P}_{\mathrm l}^{-a}\mathbf{G}\mathbf{P}_{\mathrm r}^{-a}$ and
  calls it the ladder from Adam's diagonal to Muon.
- Notation that does not agree with `adam.md` and that I left alone:
  $\epsilon$ is inside the square root in `eq_adagrad` and `eq_rmsprop`
  there and outside here (noted in a clause); the decay equations there use
  $\mathbf{x}$ for the weights, here $\mathbf{w}$ (noted in a clause).

## Cells to cut or redesign on the development machine

- `mdl-adaptive-stochastic-methods-the-running-example-aligned-and-rotated-1`
  (ADA S1, S2, ADA-10): let it print the row 40 / 48 / 54 for ratios
  $1, 10^3, 10^5$, and either tune Adam's $\eta$ or print a small sweep; run
  $\beta_1 = 0.9$ at its best step. The prose caveats can then shrink.
- `mdl-adaptive-stochastic-methods-kronecker-factors` (confirms an identity
  just proved; overview decision 1): the check line can go, the counts need
  no cell. Prose around it is now one clause.
- `mdl-adaptive-stochastic-methods-warmup` (ADA S3): equal scales cannot show
  the preconditioner's effect at $t = 1$; use log-uniform scales as in
  exercise 6(b).
- `adaptive-stochastic-methods-imports`: four tabs loading frameworks no cell
  uses (review, "Code").
- `...aligned-and-rotated-2` and `adaptive-stochastic-methods-reddi` confirm
  predictions but also show something (the 1000:1 effective steps; the start
  at $x = 0$). Kept as they are.

## Other deferred findings

- ADA S4 / ADA-16: full-matrix preconditioner on the rotated valley needs a
  new cell and exercise. Prose now weakens the promise and states that a
  full matrix of outer products rotates with the problem.
- ADA-6: the regret view is still a run-in paragraph (a `####` heading or a
  move to the counterexample would be a restructuring; the opening now marks
  the bound comparison as skippable).
- ADA-15: figure panel letters and $\kappa = 20$; figure change.
- ADA-26: $\boldsymbol\Sigma$ (singular values here, covariance in the
  previous section), $\sigma$ with two meanings, `n`/`m` rebound in cells.
  Shared symbols and code; not changed.
- ADA-23: "The Edge of Stability" in `mdl-gradient-based-optimization.md` has
  no label, so the warmup paragraph cites `subsec_mdl-quadratic-model`, its
  parent. Retarget if that subsection gets a label.
- Exercises: missing difficulty marks; the last exercise has no label; 11(d)
  repeats the Muon identity; exercise 7 uses projection before the next
  section.
- Step index $t$ here against $k$ in the first three sections of the chapter.

## Claims added or changed

- 54 steps at ratio $10^5$; 166 and 14 steps at $\eta = 0.03, 0.3$; 4,145
  steps from $(100, 100)$; best $\eta$ over the seven listed values gives
  about 620; no convergence without momentum at any of them.
- Dividing by $|g_i|$ fixes the distance per step at $\eta$, not the Newton
  step.
- Adam is a sign step in three stated cases.
- A full matrix built from $\mathbf g_\tau\mathbf g_\tau^\top$ rotates with
  the problem (if $\mathbf g \to \mathbf R\mathbf g$ then
  $\mathbf P \to \mathbf R\mathbf P\mathbf R^\top$).
- Regret per period: Adam at least $C - 2$, $x^\star$ exactly $-(C - 2)$.
- Reddi et al.: online sequence and stochastic problem for every fixed
  $\beta_1 < \sqrt{\beta_2}$ (from memory, see ADA-17).
- $1 - \lambda|w_i| = \epsilon/(|g_i| + \epsilon)$ at an AdamW fixed point,
  and the two cases it implies.
- First steps are about four times too large for a noise coordinate, not for
  one with a consistent gradient; long warmups answer to curvature, with the
  stability ceiling $2/\lambda_{\max}$.
- Gauss-Newton as a positive semidefinite approximation of the Hessian.
- Zipf ratio 0.63.

## Slides that now disagree with the body

- "A 1000:1 Valley ..." slide and the summary slide list "heavy ball 296,
  scaling alone 48" as a ranking, without the untuned $\eta$; the body no
  longer makes that comparison.
- "Scaling Alone Solves the Aligned Valley in 48 Steps ..." (title): same
  caveat.
- "Coupled Adam Keeps the MAP Point, AdamW Settles on the Box": the body now
  says "fixed point", not "settles".
- The warmup slide title already states the body's finding.
- Any slide quoting the Zipf ratio as 0.64 should read 0.63 (I did not find
  one).
